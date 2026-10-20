"""Evaluation-suite design analysis on the official EMBER Score: unique worlds, calibration, sample size.

Research tool for docs/evaluation-suite-design.md, docs/difficulty-calibration.md
and docs/suite-design-results.md. It is evaluator-side analysis, not part of
the package. It changes no simulator, scenario, generator, metric or scoring
rule, and it has no score of its own: every score is the official EMBER Score
(``emberbench.scoring.score_episode``, scoring version 1.0), and every world
identity is ``emberbench.environment_world_digest``.

Usage::

    python tools/analyze_suite_design.py audit-public  --out OUT_DIR
    python tools/analyze_suite_design.py study         --out OUT_DIR [--quick] [--workers N]
    python tools/analyze_suite_design.py gap-probe     --out OUT_DIR [--count N]

``audit-public`` profiles the public V1 suite. ``study`` generates a
**development** candidate pool with the existing generator from public
development seeds, runs the frozen calibration panel, and runs the
calibration, allocation, suite-size, seed, model-trial, candidate-pool and
workload studies. ``gap-probe`` checks whether config-only recipes fill the
family x band x fire-model cells that the development pool left empty. Each
command writes a Markdown report (``study`` also writes ``ci-width.svg``) into
``OUT_DIR``. Raw panel runs are cached in ``OUT_DIR``
under a key of every version and recipe, so a rerun with the same code reuses
them; the reported numbers are always recomputed from the cache by the final
code. Keep ``OUT_DIR`` outside the repository; the leak guard rejects JSON
assets outside the public roots.

The development seeds below are public by construction (they are derived
from a fixed public label). They select development worlds only and must
never be used for a scored evaluation.
"""

import argparse
import hashlib
import heapq
import itertools
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from emberbench import (
    ENVIRONMENT_WORLD_VERSION,
    SIMULATOR_VERSION,
    GeneratorConfig,
    Scenario,
    ScenarioFamily,
    agent_policy,
    baseline_factory,
    build_evaluation_manifest,
    build_unique_world_manifest,
    environment_world_digest,
    generate_suite,
    load_suite,
    run_episode,
)
from emberbench.agent import ActionOutput, Agent, AgentFactory, serialize_observation
from emberbench.generator import GENERATOR_VERSION, measure_proxies
from emberbench.grid import Pos, neighbours
from emberbench.observation import Observation
from emberbench.oracle import fire_trajectory
from emberbench.rng import SplitMix64
from emberbench.scoring import (
    OFFICIAL_CONSTANTS,
    SCORING_VERSION,
    EpisodeEvidence,
    EpisodeScore,
    EpisodeStatus,
    _UNREACHABLE,
    _references,
    score_episode,
    score_evidence,
)
from emberbench.simulator import Simulator

ANALYSIS_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"

# Public development-seed domain. Seeds from this domain are published here and
# so can never be scored seeds.
DEVELOPMENT_SEED_DOMAIN = "emberbench/suite-design/development/"

# The frozen calibration panel (calibration panel version 1.0). Four fixed,
# deterministic reference policies with different competences; no submitted
# model is ever a member. ``random-legal-v1`` is the floor: it is reported on
# the public suite but never sets a band. The privileged references (the
# savable count M* and the capacity ceiling) are bounds, never panel members.
CALIBRATION_PANEL_VERSION = "1.0"
CALIBRATION_PANEL: tuple[str, ...] = (
    "nearest-exit-greedy-v1",
    "static-shortest-path-v1",
    "fire-aware-shortest-path-v1",
    "congestion-aware-v1",
)
FLOOR_AGENT = "random-legal-v1"

# Difficulty cutpoints (official EMBER points) on B, the best panel score of a
# world, and the admission rules. Frozen with the calibration panel; derived in
# docs/difficulty-calibration.md sections 4 and 5 from the development pool.
EASY_MIN_BEST = 80.0  # Easy: the best panel member keeps >= 80 points
MODERATE_MIN_BEST = 30.0  # Moderate: 30 <= B < 80
HARD_MIN_BEST = 10.0  # Hard: 10 <= B < 30; Stress: B < 10
SATURATED = 95.0  # every panel member >= 95: no information, never selected
FLOOR = 1.0  # every panel member < 1 point: no information, never selected
CEILING_MIN = 50.0  # capacity ceiling below this: practically unsolvable, never selected
BANDS: tuple[str, ...] = ("easy", "moderate", "hard", "stress")


def development_seed(label: str) -> int:
    """A public 64-bit development seed for ``label``. Never a scored seed."""
    digest = hashlib.sha256((DEVELOPMENT_SEED_DOMAIN + label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# --------------------------------------------------------------------------- unique worlds


@dataclass(frozen=True)
class ScheduledEpisode:
    scenario_id: str
    seed: int
    world: str  # environment_world_digest


@dataclass(frozen=True)
class WorldDedup:
    """A seed schedule reduced to its unique environment worlds (first occurrence wins)."""

    episodes: tuple[ScheduledEpisode, ...]
    unique: tuple[ScheduledEpisode, ...]

    @property
    def duplicate_count(self) -> int:
        return len(self.episodes) - len(self.unique)

    def worlds_per_scenario(self) -> dict[str, int]:
        out: Counter[str] = Counter(e.scenario_id for e in self.unique)
        return dict(sorted(out.items()))


def dedupe_schedule(scenarios: Mapping[str, Scenario], schedule: Sequence[tuple[str, int]]) -> WorldDedup:
    """Map a ``(scenario_id, seed)`` schedule to environment worlds and drop repeated worlds."""
    episodes: list[ScheduledEpisode] = []
    seen: set[str] = set()
    unique: list[ScheduledEpisode] = []
    cache: dict[tuple[str, int], str] = {}
    for scenario_id, seed in schedule:
        key = (scenario_id, seed)
        if key not in cache:
            cache[key] = environment_world_digest(scenarios[scenario_id], seed)
        ep = ScheduledEpisode(scenario_id, seed, cache[key])
        episodes.append(ep)
        if ep.world not in seen:
            seen.add(ep.world)
            unique.append(ep)
    return WorldDedup(tuple(episodes), tuple(unique))


def distinct_world_seeds(scenario: Scenario, candidate_seeds: Iterable[int], count: int) -> tuple[int, ...]:
    """The first ``count`` candidate seeds with pairwise-distinct worlds (one seed for deterministic fire)."""
    chosen: list[int] = []
    seen: set[str] = set()
    for seed in candidate_seeds:
        if scenario.fire.model.value == "deterministic":
            return (seed,)
        world = environment_world_digest(scenario, seed)
        if world not in seen:
            seen.add(world)
            chosen.append(seed)
            if len(chosen) == count:
                break
    return tuple(chosen)


# --------------------------------------------------------------------------- static structure


def _bfs(sources: Iterable[Pos], walls: frozenset[Pos], width: int, height: int) -> dict[Pos, int]:
    dist: dict[Pos, int] = {}
    frontier: list[Pos] = []
    for s in sources:
        if s not in dist:
            dist[s] = 0
            frontier.append(s)
    while frontier:
        nxt: list[Pos] = []
        for cell in frontier:
            for n in neighbours(cell, width, height):
                if n not in walls and n not in dist:
                    dist[n] = dist[cell] + 1
                    nxt.append(n)
        frontier = nxt
    return dist


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def exit_fields(scenario: Scenario) -> dict[str, dict[Pos, int]]:
    """Per exit ID, the static walking distance (walls only) from every reachable cell."""
    return {e.id: _bfs([e.position], scenario.walls, scenario.width, scenario.height) for e in scenario.exits}


def _min_vertex_cut(scenario: Scenario) -> int:
    """Max number of vertex-disjoint person-to-exit paths on the static map (unit cell capacity)."""
    width, height, walls = scenario.width, scenario.height, scenario.walls
    exits = {e.position for e in scenario.exits}
    cells = [Pos(r, c) for r in range(height) for c in range(width) if Pos(r, c) not in walls]
    capacity: dict[tuple[object, object], int] = defaultdict(int)
    adjacency: dict[object, list[object]] = defaultdict(list)

    def edge(u: object, v: object) -> None:
        capacity[(u, v)] += 1
        adjacency[u].append(v)
        adjacency[v].append(u)

    for cell in cells:
        edge((cell, 0), (cell, 1))
        if cell in exits:
            edge((cell, 1), "T")
            continue
        for n in neighbours(cell, width, height):
            if n not in walls:
                edge((cell, 1), (n, 0))
    for p in scenario.people:
        edge("S", (p.position, 0))
    total = 0
    while True:
        parent: dict[object, object] = {"S": "S"}
        frontier: list[object] = ["S"]
        while frontier and "T" not in parent:
            nxt: list[object] = []
            for u in frontier:
                for w in adjacency[u]:
                    if w not in parent and capacity[(u, w)] > 0:
                        parent[w] = u
                        nxt.append(w)
            frontier = nxt
        if "T" not in parent:
            return total
        v: object = "T"
        while v != "S":
            u = parent[v]
            capacity[(u, v)] -= 1
            capacity[(v, u)] += 1
            v = u
        total += 1


@dataclass(frozen=True)
class ScenarioProfile:
    """Seed-free structural measurements of one scenario. Public-safe for public scenarios."""

    scenario_id: str
    scenario_hash: str
    width: int
    height: int
    free_cells: int
    population: int
    exit_count: int
    ignition_count: int
    fire_model: str
    spread_interval: int
    spread_probability: float | None
    max_steps: int
    obstacle_density: float
    crowd_density: float
    passage_width: int
    route_redundancy: int
    route_length: int | None
    ignition_person_distance: int | None
    ignition_exit_distance: int | None
    min_cut: int
    bottleneck_severity: float  # population / min_cut: drain steps through the narrowest cut
    max_nearest_exit_load: int
    split_pressure: float | None  # busiest nearest-exit queue / (1 + median detour to the next exit)
    deceptive_share: float  # people whose straight-line-nearest exit is >= 2 steps worse by path
    exit_gap_median: float | None  # median (second - first) path distance to exits
    detour_ratio: float  # mean path / Manhattan distance to the path-nearest exit
    ignition_centrality: float  # 0 = an ignition at the grid centre, 1 = on the border


def profile_scenario(scenario: Scenario) -> ScenarioProfile:
    proxies = measure_proxies(scenario)
    fields_ = exit_fields(scenario)
    exit_ids = sorted(fields_)
    pos_of = {e.id: e.position for e in scenario.exits}
    people = sorted(scenario.people, key=lambda p: p.id)
    nearest: dict[str, str] = {}
    gaps: list[int] = []
    detours: list[float] = []
    deceptive = 0
    for p in people:
        dists = sorted((fields_[e].get(p.position, math.inf), e) for e in exit_ids)
        d0, e0 = dists[0]
        nearest[p.id] = e0
        if len(dists) > 1 and math.isfinite(dists[1][0]) and math.isfinite(d0):
            gaps.append(int(dists[1][0] - d0))
        if math.isfinite(d0):
            man = _manhattan(p.position, pos_of[e0])
            detours.append(d0 / man if man else 1.0)
            man_best = min(_manhattan(p.position, pos_of[e]) for e in exit_ids)
            man_nearest = [e for e in exit_ids if _manhattan(p.position, pos_of[e]) == man_best]
            if min(fields_[e].get(p.position, math.inf) for e in man_nearest) - d0 >= 2:
                deceptive += 1
    load = Counter(nearest.values())
    q_max = max(load.values()) if load else 0
    split: float | None = None
    if len(exit_ids) > 1 and load:
        busiest = min(e for e, n in load.items() if n == q_max)
        extra = []
        for p in people:
            if nearest[p.id] != busiest:
                continue
            d = sorted(fields_[e].get(p.position, math.inf) for e in exit_ids)
            if math.isfinite(d[1]):
                extra.append(d[1] - d[0])
        split = q_max / (1 + statistics.median(extra)) if extra else None
    cut = _min_vertex_cut(scenario)
    h, w = scenario.height, scenario.width
    centrality = min(
        max(abs(i.row - (h - 1) / 2) / ((h - 1) / 2), abs(i.col - (w - 1) / 2) / ((w - 1) / 2)) for i in scenario.ignitions
    )
    free = sum(1 for r in range(h) for c in range(w) if Pos(r, c) not in scenario.walls)
    return ScenarioProfile(
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario.content_hash(),
        width=w,
        height=h,
        free_cells=free,
        population=len(people),
        exit_count=len(scenario.exits),
        ignition_count=len(scenario.ignitions),
        fire_model=scenario.fire.model.value,
        spread_interval=scenario.fire.spread_interval,
        spread_probability=scenario.fire.spread_probability,
        max_steps=scenario.termination.max_steps,
        obstacle_density=proxies.obstacle_density,
        crowd_density=proxies.crowd_density,
        passage_width=proxies.passage_width,
        route_redundancy=proxies.route_redundancy,
        route_length=proxies.route_length,
        ignition_person_distance=proxies.ignition_person_distance,
        ignition_exit_distance=proxies.ignition_exit_distance,
        min_cut=cut,
        bottleneck_severity=len(people) / cut if cut else math.inf,
        max_nearest_exit_load=q_max,
        split_pressure=split,
        deceptive_share=deceptive / len(people) if people else 0.0,
        exit_gap_median=float(statistics.median(gaps)) if gaps else None,
        detour_ratio=statistics.fmean(detours) if detours else math.inf,
        ignition_centrality=centrality,
    )


# --------------------------------------------------------------------------- privileged world measurements


def _static_plan_survives(scenario: Scenario, start: Pos, field_: Mapping[Pos, int], trajectory: Sequence[frozenset[Pos]]) -> bool:
    """Can one person, alone, follow SOME static shortest route to the given exit without burning?"""
    exits = set(scenario.exit_positions)
    if start in trajectory[0] or start not in field_:
        return False
    frontier = {start}
    for t in range(scenario.termination.max_steps):
        now, after = trajectory[t], trajectory[t + 1]
        reached: set[Pos] = set()
        for cell in frontier:
            for nxt in neighbours(cell, scenario.width, scenario.height):
                if field_.get(nxt) != field_[cell] - 1 or nxt in scenario.walls or nxt in now:
                    continue
                if nxt in exits:
                    return True
                if nxt not in after:
                    reached.add(nxt)
        if not reached:
            return False
        frontier = reached
    return False


def _glover(intervals: Sequence[tuple[int, int]], capacity: Sequence[int]) -> int:
    """Most unit jobs that fit: job ``i`` takes one slot in steps ``[lo_i, hi_i]``, step ``s`` has ``capacity[s]`` slots.

    Earliest-deadline-first over the steps (Glover's greedy). It is optimal for
    this convex bipartite matching.
    """
    jobs = sorted(intervals)
    heap: list[int] = []
    k = placed = 0
    for step in range(1, len(capacity)):
        while k < len(jobs) and jobs[k][0] <= step:
            heapq.heappush(heap, jobs[k][1])
            k += 1
        while heap and heap[0] < step:
            heapq.heappop(heap)
        take = capacity[step]
        while take and heap:
            heapq.heappop(heap)
            placed += 1
            take -= 1
    return placed


def capacity_savable(scenario: Scenario, seed: int) -> tuple[int, int]:
    """``(M*, M_cap)``: the official savable count and a capacity-aware upper bound on evacuations. PRIVILEGED.

    ``M*`` (the official ``max_evacuated``) counts people who can escape ALONE.
    It ignores that people share exits and cuts. ``M_cap <= M*`` adds the two
    capacity facts behind the official Time reference ``Q_cut``: at most one
    person leaves through an open exit cell per step, and at most ``k_s``
    people stand on the cut ``N(C)`` of a component ``C`` of the cells with
    wall distance ``> d`` in step ``s``. Each savable person needs one exit
    slot inside its single-person evacuation window and one cut slot inside its
    single-person window on every cut it must cross; the windows are relaxed
    to intervals, and Glover's greedy gives the exact maximum of every relaxed
    problem. Every relaxation only enlarges the feasible set, so ``M_cap`` is an
    upper bound on the evacuees of every policy. ``M* - M_cap`` savable people
    are lost by every policy: the official score charges them as avoidable.
    This is the deterministic input of the practical-unsolvability filter.
    """
    refs = _references(scenario, seed)
    unreachable = _UNREACHABLE
    horizon = scenario.termination.max_steps
    width, height, walls = scenario.width, scenario.height, scenario.walls
    cells = sorted(refs.index, key=refs.index.__getitem__)
    n = len(cells)
    adj = [tuple(refs.index[q] for q in (c, *neighbours(c, width, height)) if q not in walls) for c in cells]
    exits = frozenset(scenario.exit_positions)
    is_exit = [c in exits for c in cells]
    exit_idx = [i for i in range(n) if is_exit[i]]
    burning = [frozenset(refs.index[c] for c in b if c in refs.index) for b in refs.trajectory]
    earliest = refs.earliest
    wall = [-1] * n
    frontier = list(exit_idx)
    for i in frontier:
        wall[i] = 0
    while frontier:
        nxt: list[int] = []
        for i in frontier:
            for j in adj[i]:
                if wall[j] < 0:
                    wall[j] = wall[i] + 1
                    nxt.append(j)
        frontier = nxt
    # Cells from which a person evacuates in step t + 1: next to an exit that does not burn at the start of the step.
    exit_adjacent = [frozenset(j for x in exit_idx if x not in burning[t] for j in adj[x] if not is_exit[j]) for t in range(horizon)]
    starts = [refs.index[p.position] for p in sorted(scenario.people, key=lambda p: p.id)]
    savable = [s for s in starts if earliest[0][s] != unreachable]
    if not savable:
        return 0, 0

    exit_windows: list[tuple[int, int]] = []
    first_at: list[dict[int, int]] = []  # per person: d -> first snapshot with wall distance <= d
    last_at: list[dict[int, int]] = []  # per person: d -> last snapshot on a cell with wall distance == d
    for s0 in savable:
        reach = {s0}
        latest = 0
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        low = wall[s0] + 1
        for t in range(horizon):
            levels = {wall[i] for i in reach}
            for d in levels:
                last[d] = t
            m = min(levels)
            for d in range(m, low):
                first[d] = t
            low = min(low, m)
            if reach & exit_adjacent[t]:
                latest = t + 1
            now, after, row = burning[t], burning[t + 1], earliest[t + 1]
            reach = {j for i in reach for j in adj[i] if j not in now and j not in after and not is_exit[j] and row[j] != unreachable}
            if not reach:
                break
        exit_windows.append((earliest[0][s0], latest))
        first_at.append(first)
        last_at.append(last)

    exit_capacity = [0] + [sum(1 for x in exit_idx if x not in burning[s - 1]) for s in range(1, horizon + 1)]
    best = min(len(savable), _glover(exit_windows, exit_capacity))
    for d in range(1, max(wall, default=0)):
        comp: dict[int, int] = {}
        cuts: list[set[int]] = []
        for i in range(n):
            if wall[i] <= d or i in comp:
                continue
            k = len(cuts)
            comp[i] = k
            stack, cut = [i], set()
            while stack:
                a = stack.pop()
                for b in adj[a]:
                    if wall[b] > d and b not in comp:
                        comp[b] = k
                        stack.append(b)
                    elif wall[b] == d:
                        cut.add(b)
            cuts.append(cut)
        parent = list(range(len(cuts)))

        def root(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        owner: dict[int, int] = {}
        for k, cut in enumerate(cuts):
            for x in cut:
                if x in owner:
                    parent[root(k)] = root(owner[x])
                else:
                    owner[x] = k
        groups: dict[int, tuple[set[int], list[int]]] = {}
        for k, cut in enumerate(cuts):
            groups.setdefault(root(k), (set(), []))[0].update(cut)
        total = 0
        for p, s0 in enumerate(savable):
            if s0 in comp:
                groups[root(comp[s0])][1].append(p)
            else:
                total += 1
        for cut, members in groups.values():
            if not members:
                continue
            capacity = [0] + [sum(1 for c in cut if c not in burning[s - 1] and c not in burning[s]) for s in range(1, horizon + 1)]
            windows = [(first_at[p][d], last_at[p][d]) for p in members if d in first_at[p] and d in last_at[p]]
            total += _glover(windows, capacity)
        best = min(best, total)
    return len(savable), best


def capacity_ceiling(population: int, savable: int, capacity: int) -> float | None:
    """The best official score with ``savable - capacity`` avoidable stranded people and perfect quality.

    Computed by the official ``score_evidence`` on synthetic evidence (no
    second implementation of the formula). ``None`` when nobody is savable.
    """
    if savable == 0:
        return None
    lost = savable - capacity
    ev = EpisodeEvidence(
        scenario_id="capacity-ceiling",
        scenario_hash="0" * 64,
        seed=0,
        record_digest="0" * 64,
        population=population,
        max_evacuated=savable,
        evacuated=capacity,
        casualties=0,
        stranded=population - capacity,
        avoidable_casualty_ids=(),
        avoidable_stranded_ids=tuple(f"a{i:05d}" for i in range(lost)),
        unavoidable_casualty_ids=(),
        unavoidable_stranded_ids=tuple(f"u{i:05d}" for i in range(population - savable)),
        evacuee_time=capacity,
        evacuee_time_bound=capacity,
        evacuee_time_queue_bound=capacity,
        evacuee_time_cut_bound=capacity,
        movement_attempts=0,
        blocked_occupied=0,
        blocked_conflict=0,
        blocked_cycle=0,
        path_cells_moved=capacity,
        path_moves_bound=capacity,
        path_static_bound=capacity,
        evacuee_danger_states=0,
        evacuee_danger_bound=0,
        invalidation_events=0,
        unrecoverable_invalidations=0,
        replanning_remaining=0,
        replanning_remaining_bound=0,
    )
    return float(score_evidence(0, ev, OFFICIAL_CONSTANTS).score)


@dataclass(frozen=True)
class FireProfile:
    """Fire-dependent, privileged measurements of one world, used only for family tags."""

    threatened_share: float  # people whose nearest exit burns before they can use it
    invalidation_share: float  # people whose every static shortest route is cut by fire
    recoverable_invalidation_share: float  # ... and who can still escape another way
    final_burn_share: float


def fire_profile(scenario: Scenario, seed: int, savable_ids: frozenset[str]) -> FireProfile:
    trajectory = fire_trajectory(scenario, seed)
    fields_ = exit_fields(scenario)
    exit_ids = sorted(fields_)
    pos_of = {e.id: e.position for e in scenario.exits}
    burn_step = {e: next((t for t, b in enumerate(trajectory) if pos_of[e] in b), None) for e in exit_ids}
    people = sorted(scenario.people, key=lambda p: p.id)
    nearest = {p.id: min((fields_[e].get(p.position, math.inf), e) for e in exit_ids) for p in people}
    load = Counter(e for _, e in nearest.values())
    threatened = invalidated = recoverable = 0
    for p in people:
        d, e = nearest[p.id]
        b = burn_step[e]
        if b is not None and math.isfinite(d) and b <= max(d, load[e]):
            threatened += 1
        if not (math.isfinite(d) and _static_plan_survives(scenario, p.position, fields_[e], trajectory)):
            invalidated += 1
            if p.id in savable_ids:
                recoverable += 1
    free = sum(1 for r in range(scenario.height) for c in range(scenario.width) if Pos(r, c) not in scenario.walls)
    n = len(people)
    return FireProfile(threatened / n, invalidated / n, recoverable / n, len(trajectory[-1]) / free)


# --------------------------------------------------------------------------- official scoring of panel runs


@dataclass(frozen=True)
class PanelRun:
    """One reference-policy episode, scored by the official EMBER Score."""

    agent: str
    score: float
    status: str
    safety: float | None
    quality: float | None
    evacuated: int
    avoidable_losses: int
    severe: tuple[str, ...]
    steps: int
    seconds: float  # simulator run + official scoring, evaluator CPU


def official_run(scenario: Scenario, seed: int, agent: str, factory: AgentFactory) -> tuple[PanelRun, EpisodeScore]:
    t0 = time.perf_counter()
    record = run_episode(scenario, agent_policy(factory()), seed=seed)
    scored = score_episode(scenario, record)
    seconds = time.perf_counter() - t0
    ev = scored.evidence
    assert ev is not None
    run = PanelRun(
        agent=agent,
        score=float(scored.score),
        status=scored.status.value,
        safety=None if scored.safety is None else float(scored.safety),
        quality=None if scored.quality is None else float(scored.quality),
        evacuated=ev.evacuated,
        avoidable_losses=ev.avoidable_casualties + ev.avoidable_stranded,
        severe=scored.severe,
        steps=len(record.steps),
        seconds=seconds,
    )
    return run, scored


class NoisyAgent:
    """Analysis-only surrogate of a nondeterministic decision model.

    With probability ``epsilon`` per person and step it replaces the base
    agent's action by a uniform draw from the allowed actions. ``trial`` seeds
    its private stream, so trial ``k`` is reproducible. It is not a baseline.
    """

    def __init__(self, base: Agent, epsilon: float, trial: int) -> None:
        self._base = base
        self._epsilon = epsilon
        self._rng = SplitMix64(development_seed(f"noisy-agent/{epsilon}/{trial}"))

    def act(self, observation: Observation) -> ActionOutput:
        actions = dict(self._base.act(observation))
        allowed = observation.allowed_actions
        for pid in sorted(actions):
            if self._rng.random() < self._epsilon:
                actions[pid] = allowed[self._rng.next_u64() % len(allowed)]
        return actions


def _noisy_factory(base: AgentFactory, epsilon: float, trial: int) -> AgentFactory:
    def make() -> Agent:
        return NoisyAgent(base(), epsilon, trial)

    return make


def observation_chars(scenario: Scenario, seed: int) -> int:
    """Characters of the serialized step-0 observation: a token-count proxy for API cost."""
    return len(serialize_observation(Simulator(scenario, seed=seed).observe()))


@dataclass(frozen=True)
class WorldRecord:
    """Everything the study needs about one unique world. Evaluator-side; development worlds only."""

    seed: int
    world: str
    savable: int  # M*, the official max_evacuated
    capacity_savable: int  # M_cap
    capacity_ceiling: float | None
    fire: FireProfile
    runs: tuple[PanelRun, ...]

    def score(self, agent: str) -> float:
        return next(r.score for r in self.runs if r.agent == agent)

    def panel(self) -> dict[str, float]:
        return {r.agent: r.score for r in self.runs if r.agent in CALIBRATION_PANEL}


@dataclass(frozen=True)
class ScenarioRecord:
    recipe: str
    scenario: Scenario
    profile: ScenarioProfile
    worlds: tuple[WorldRecord, ...]
    obs_chars: int
    seconds: float


def evaluate_scenario(recipe: str, scenario: Scenario, seeds: Sequence[int], agents: Sequence[str]) -> ScenarioRecord:
    """Profile a scenario and score every agent on each of its unique worlds (a worker task)."""
    t0 = time.perf_counter()
    factories = {a: baseline_factory(a) for a in agents}
    worlds = []
    for seed in seeds:
        runs = []
        savable_ids: frozenset[str] = frozenset()
        for a in agents:
            run, scored = official_run(scenario, seed, a, factories[a])
            runs.append(run)
            ev = scored.evidence
            assert ev is not None
            lost = set(ev.unavoidable_casualty_ids) | set(ev.unavoidable_stranded_ids)
            savable_ids = frozenset(p.id for p in scenario.people if p.id not in lost)
        m_star, m_cap = capacity_savable(scenario, seed)
        assert m_star == len(savable_ids)
        worlds.append(
            WorldRecord(
                seed=seed,
                world=environment_world_digest(scenario, seed),
                savable=m_star,
                capacity_savable=m_cap,
                capacity_ceiling=capacity_ceiling(len(scenario.people), m_star, m_cap),
                fire=fire_profile(scenario, seed, savable_ids),
                runs=tuple(runs),
            )
        )
    return ScenarioRecord(recipe, scenario, profile_scenario(scenario), tuple(worlds), observation_chars(scenario, seeds[0]), time.perf_counter() - t0)


def evaluate_trials(scenario: Scenario, seed: int, epsilon: float, trials: int) -> list[float]:
    """Official scores of K trials of the noisy surrogate on ONE frozen world (a worker task)."""
    base = baseline_factory("congestion-aware-v1")
    return [float(score_episode(scenario, run_episode(scenario, agent_policy(_noisy_factory(base, epsilon, k)()), seed=seed)).score) for k in range(trials)]


# --------------------------------------------------------------------------- record serialization (the cache)


def _record_to_json(r: ScenarioRecord) -> dict[str, Any]:
    return {
        "recipe": r.recipe,
        "scenario": r.scenario.to_dict(),
        "profile": asdict(r.profile),
        "worlds": [asdict(w) for w in r.worlds],
        "obs_chars": r.obs_chars,
        "seconds": r.seconds,
    }


def _record_from_json(d: Mapping[str, Any]) -> ScenarioRecord:
    worlds = tuple(
        WorldRecord(
            seed=w["seed"],
            world=w["world"],
            savable=w["savable"],
            capacity_savable=w["capacity_savable"],
            capacity_ceiling=w["capacity_ceiling"],
            fire=FireProfile(**w["fire"]),
            runs=tuple(PanelRun(**{**r, "severe": tuple(r["severe"])}) for r in w["runs"]),
        )
        for w in d["worlds"]
    )
    return ScenarioRecord(d["recipe"], Scenario.from_dict(d["scenario"]), ScenarioProfile(**d["profile"]), worlds, d["obs_chars"], d["seconds"])


def _cache_key(kind: str, payload: Any) -> str:
    versions = {
        "analysis": ANALYSIS_VERSION,
        "scoring": SCORING_VERSION,
        "simulator": SIMULATOR_VERSION,
        "generator": GENERATOR_VERSION,
        "panel": CALIBRATION_PANEL_VERSION,
    }
    text = json.dumps({"kind": kind, "versions": versions, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parallel(fn: Callable[..., Any], tasks: Sequence[tuple[Any, ...]], workers: int, log: Callable[[str], None], label: str) -> list[Any]:
    t0 = time.perf_counter()
    if workers <= 1:
        out = [fn(*t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(fn, *t) for t in tasks]
            out = []
            for i, f in enumerate(futures):
                out.append(f.result())
                if (i + 1) % 100 == 0:
                    log(f"  {label}: {i + 1}/{len(tasks)} ({time.perf_counter() - t0:.0f}s)")
    log(f"  {label}: {len(tasks)} tasks in {time.perf_counter() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------- development pool


def _config(
    name: str,
    *,
    count: int,
    grid: tuple[int, int],
    population: tuple[int, int],
    exits: tuple[int, int],
    ignitions: tuple[int, int],
    partitions: tuple[int, int] = (0, 0),
    doors: tuple[int, int] = (1, 2),
    door_width: tuple[int, int] = (1, 2),
    scatter: tuple[float, float] = (0.0, 0.05),
    model: str = "deterministic",
    interval: tuple[int, int] = (2, 3),
    probabilities: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_steps: int = 80,
    ignition_exit: tuple[int, int] | None = None,
) -> tuple[str, GeneratorConfig]:
    w, h = grid
    fire: dict[str, Any] = {
        "model": model,
        "spread_interval": {"min": interval[0], "max": interval[1]},
        "exits_flammable": True,
    }
    if model == "stochastic":
        fire["spread_probability_choices"] = list(probabilities)
    ign_exit = ignition_exit or (3, w * h)
    data = {
        "config_version": "1.0",
        "scenario_count": count,
        "grid": {"width": w, "height": h},
        "population": {"min": population[0], "max": population[1]},
        "exit_count": {"min": exits[0], "max": exits[1]},
        "ignition_count": {"min": ignitions[0], "max": ignitions[1]},
        "layout": {
            "partition_count": {"min": partitions[0], "max": partitions[1]},
            "doors_per_partition": {"min": doors[0], "max": doors[1]},
            "door_width": {"min": door_width[0], "max": door_width[1]},
            "scatter_density": {"min": scatter[0], "max": scatter[1]},
        },
        "fire": fire,
        "movement": {"enter_fire": "casualty"},
        "max_steps": max_steps,
        "max_attempts": 4000,
        "constraints": {
            "route_length": {"min": 1, "max": max_steps},
            "obstacle_density": {"min": 0.0, "max": 0.6},
            "crowd_density": {"min": 0.0, "max": 0.9},
            "exit_separation": {"min": 4, "max": w + h},
            "passage_width": {"min": 1, "max": max(w, h)},
            "route_redundancy": {"min": 1, "max": min(4, exits[1])},
            "ignition_person_distance": {"min": 3, "max": w * h},
            "ignition_exit_distance": {"min": ign_exit[0], "max": ign_exit[1]},
        },
    }
    return name, GeneratorConfig.from_json(json.dumps(data))


def development_recipes(scale: int) -> list[tuple[str, GeneratorConfig]]:
    """Public development recipes. Each aims at some capability axes; family labels come from measurement, not from the recipe."""
    n = scale
    return [
        _config("open-sparse", count=n, grid=(14, 10), population=(3, 8), exits=(1, 1), ignitions=(1, 1), interval=(2, 4), max_steps=60),
        _config("dense-single", count=n, grid=(14, 10), population=(20, 40), exits=(1, 1), ignitions=(1, 1), interval=(3, 8), max_steps=90),
        _config(
            "partitioned", count=n, grid=(18, 12), population=(6, 16), exits=(1, 2), ignitions=(1, 1),
            partitions=(1, 2), doors=(1, 1), door_width=(1, 1), interval=(2, 3), max_steps=80,
        ),
        _config("multi-exit", count=n, grid=(20, 12), population=(6, 20), exits=(2, 3), ignitions=(1, 1), partitions=(0, 1), interval=(2, 3)),
        _config("multi-ignition", count=n, grid=(18, 12), population=(6, 16), exits=(2, 3), ignitions=(2, 4), interval=(3, 4)),
        _config(
            "maze", count=n, grid=(16, 12), population=(4, 10), exits=(2, 3), ignitions=(1, 1),
            partitions=(2, 4), doors=(1, 1), door_width=(1, 1), scatter=(0.1, 0.25), interval=(3, 4), max_steps=100,
        ),
        _config(
            "exit-threat", count=n, grid=(20, 12), population=(6, 16), exits=(2, 3), ignitions=(1, 1),
            partitions=(0, 1), interval=(2, 4), ignition_exit=(3, 6),
        ),
        _config(
            "crowd", count=max(4, n // 3), grid=(30, 18), population=(100, 120), exits=(2, 4), ignitions=(1, 1),
            interval=(4, 8), max_steps=200,
        ),
        _config(
            "stochastic-open", count=n, grid=(18, 12), population=(5, 14), exits=(1, 3), ignitions=(1, 2),
            model="stochastic", interval=(1, 2), max_steps=80,
        ),
        _config(
            "stochastic-partitioned", count=n, grid=(18, 12), population=(5, 14), exits=(1, 3), ignitions=(1, 2),
            partitions=(1, 2), doors=(1, 2), door_width=(1, 1), model="stochastic", interval=(1, 2), max_steps=80,
        ),
    ]


def build_pool(scale: int, stochastic_worlds: int, workers: int, out: Path, log: Callable[[str], None]) -> list[ScenarioRecord]:
    """The development pool, from the cache when it matches every version and recipe."""
    recipes = development_recipes(scale)
    key = _cache_key("pool", {"scale": scale, "worlds": stochastic_worlds, "recipes": [(n, c.to_dict()) for n, c in recipes]})
    cache = out / f"pool-{key}.json"
    if cache.exists():
        log(f"  pool: cached {cache.name}")
        return [_record_from_json(d) for d in json.loads(cache.read_text(encoding="utf-8"))]
    tasks: list[tuple[Any, ...]] = []
    for name, config in recipes:
        generated = generate_suite(config, development_seed("generator/" + name), batch_id=f"dev-{name}")
        for s in generated.scenarios:
            candidates = [development_seed(f"evaluation/{name}/{s.scenario_id}/{i}") for i in range(4 * stochastic_worlds)]
            tasks.append((name, s, distinct_world_seeds(s, candidates, stochastic_worlds), CALIBRATION_PANEL))
    log(f"  pool: {len(tasks)} scenarios generated")
    # Heaviest first, so the slow crowd scenarios do not finish last.
    tasks.sort(key=lambda t: -len(t[1].people) * len(t[2]))
    records = _parallel(evaluate_scenario, tasks, workers, log, "pool")
    records.sort(key=lambda r: (r.recipe, r.scenario.scenario_id))
    out.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([_record_to_json(r) for r in records]), encoding="utf-8")
    return records


def surrogate_scores(pool: Sequence[ScenarioRecord], epsilons: Sequence[float], workers: int, out: Path, log: Callable[[str], None]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Official scores of the non-panel comparators (noisy surrogates, trial 0) on every applicable pool world.

    They are not calibration-panel members, so their ranking stability is not
    circular with the difficulty bands. Keyed by ``(recipe, scenario_id, seed)``.
    """
    key = _cache_key("surrogates", {"epsilons": list(epsilons), "worlds": [(r.recipe, r.scenario.scenario_id, [w.seed for w in r.worlds]) for r in pool]})
    cache = out / f"surrogates-{key}.json"
    if cache.exists():
        log(f"  surrogates: cached {cache.name}")
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {(k[0], k[1], int(k[2])): v for k, v in raw}
    tasks: list[tuple[Any, ...]] = [(r.recipe, r.scenario, w.seed, tuple(epsilons)) for r in pool for w in r.worlds if w.savable]
    tasks.sort(key=lambda t: -len(t[1].people))
    results = _parallel(_surrogate_task, tasks, workers, log, "surrogates")
    table = {(t[0], t[1].scenario_id, t[2]): res for t, res in zip(tasks, results, strict=True)}
    cache.write_text(json.dumps([[list(k), v] for k, v in table.items()]), encoding="utf-8")
    return table


def _surrogate_task(recipe: str, scenario: Scenario, seed: int, epsilons: Sequence[float]) -> dict[str, float]:
    return {surrogate_name(e): evaluate_trials(scenario, seed, e, 1)[0] for e in epsilons}


def surrogate_name(epsilon: float) -> str:
    return f"noisy-congestion-{epsilon:g}"


# --------------------------------------------------------------------------- difficulty, admission and families


@dataclass(frozen=True)
class Calibration:
    """The frozen calibration of one scenario: admission decision and difficulty band."""

    panel: Mapping[str, float]  # panel member -> mean official score over the scenario's applicable worlds
    best: float
    worst: float
    spread: float
    ceiling: float | None  # mean capacity ceiling over applicable worlds
    capacity_limited: bool  # some world has M_cap < M*
    admission: str  # ADMITTED or the exclusion reason
    band: str | None


ADMITTED = "admitted"
EXCLUSIONS: tuple[str, ...] = ("not_applicable", "capacity_artifact", "saturated", "floor", "no_separation")


def difficulty_band(best: float) -> str:
    """Easy / Moderate / Hard / Stress from ``B``, the best calibration-panel score (official points)."""
    if best >= EASY_MIN_BEST:
        return "easy"
    if best >= MODERATE_MIN_BEST:
        return "moderate"
    if best >= HARD_MIN_BEST:
        return "hard"
    return "stress"


def calibrate(worlds: Sequence[WorldRecord]) -> Calibration:
    """The deterministic admission and band rule (docs/difficulty-calibration.md section 5).

    In order: a scenario without a savable person in any world is not
    applicable; a mean capacity ceiling below ``CEILING_MIN`` is a capacity
    artifact (difficulty that comes from the relaxed savable count, not from
    the policy); every panel member >= ``SATURATED`` is saturated; every panel
    member < ``FLOOR`` is at the floor; a panel spread below the official
    practical difference (1 point) carries no ranking information. Every other
    scenario is admitted and banded by the best panel score.
    """
    applicable = [w for w in worlds if w.savable > 0]
    if not applicable:
        return Calibration({}, 0.0, 0.0, 0.0, None, False, "not_applicable", None)
    panel = {a: statistics.fmean(w.score(a) for w in applicable) for a in CALIBRATION_PANEL}
    best, worst = max(panel.values()), min(panel.values())
    ceilings = [w.capacity_ceiling for w in applicable if w.capacity_ceiling is not None]
    ceiling = statistics.fmean(ceilings) if ceilings else None
    limited = any(w.capacity_savable < w.savable for w in applicable)
    spread = best - worst
    if ceiling is not None and ceiling < CEILING_MIN:
        admission = "capacity_artifact"
    elif worst >= SATURATED:
        admission = "saturated"
    elif best < FLOOR:
        admission = "floor"
    elif spread < float(OFFICIAL_CONSTANTS.practical_difference):
        admission = "no_separation"
    else:
        admission = ADMITTED
    band = difficulty_band(best) if admission == ADMITTED else None
    return Calibration(panel, best, worst, spread, ceiling, limited, admission, band)


def family_tags(profile: ScenarioProfile, fires: Sequence[FireProfile]) -> frozenset[ScenarioFamily]:
    """Every V1 family whose challenge a scenario measurably contains (fixed, published thresholds).

    Structural tags come from :class:`ScenarioProfile`; the fire-dependent tags
    (threatened exit, route invalidation) use the mean over the scenario's worlds.
    """
    F = ScenarioFamily
    tags: set[ScenarioFamily] = set()

    def mean(xs: Sequence[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    if profile.exit_count == 1:
        tags.add(F.SINGLE_EXIT_HIGH_DENSITY if profile.crowd_density >= 0.15 else F.SINGLE_EXIT_LOW_DENSITY)
    else:
        if profile.exit_gap_median is not None and profile.exit_gap_median >= 4:
            tags.add(F.MULTIPLE_EXITS_UNEQUAL_DISTANCE)
        if profile.split_pressure is not None and profile.split_pressure > 1 and profile.population >= 20:
            tags.add(F.SPLIT_FLOW_ACROSS_EXITS)
    if profile.population >= 100:
        tags.add(F.CROWD_100)
    if profile.ignition_count >= 2:
        tags.add(F.MULTIPLE_IGNITIONS)
    if profile.passage_width <= 1 and profile.bottleneck_severity >= 4:
        tags.add(F.NARROW_CORRIDOR_BOTTLENECK)
    if profile.obstacle_density >= 0.25 or profile.detour_ratio >= 1.5:
        tags.add(F.MAZE_OBSTACLE_ROUTING)
    if profile.exit_count > 1 and profile.deceptive_share >= 0.25:
        tags.add(F.DECEPTIVE_NEAREST_EXIT)
    if profile.ignition_centrality <= 0.34:
        tags.add(F.CENTRAL_IGNITION)
    if profile.exit_count > 1 and mean([f.threatened_share for f in fires]) >= 0.2:
        tags.add(F.THREATENED_NEAREST_EXIT)
    if mean([f.recoverable_invalidation_share for f in fires]) >= 0.2:
        tags.add(F.ROUTE_INVALIDATION_REPLANNING)
    if not tags and profile.exit_count > 1:
        tags.add(F.MULTIPLE_EXITS_UNEQUAL_DISTANCE)
    return frozenset(tags)


# The final V1 primary families (docs/evaluation-suite-design.md section 4). The two other V1 family
# values stay valid secondary tags but are not primary strata.
FINAL_FAMILIES: tuple[ScenarioFamily, ...] = (
    ScenarioFamily.SINGLE_EXIT_LOW_DENSITY,
    ScenarioFamily.SINGLE_EXIT_HIGH_DENSITY,
    ScenarioFamily.NARROW_CORRIDOR_BOTTLENECK,
    ScenarioFamily.SPLIT_FLOW_ACROSS_EXITS,
    ScenarioFamily.CROWD_100,
    ScenarioFamily.MAZE_OBSTACLE_ROUTING,
    ScenarioFamily.DECEPTIVE_NEAREST_EXIT,
    ScenarioFamily.THREATENED_NEAREST_EXIT,
    ScenarioFamily.MULTIPLE_IGNITIONS,
    ScenarioFamily.ROUTE_INVALIDATION_REPLANNING,
)
SECONDARY_ONLY: tuple[ScenarioFamily, ...] = tuple(f for f in ScenarioFamily if f not in FINAL_FAMILIES)


@dataclass(frozen=True)
class Candidate:
    scenario_id: str
    scenario_hash: str
    tags: frozenset[ScenarioFamily]
    band: str
    stochastic: bool


@dataclass(frozen=True)
class Selection:
    chosen: tuple[tuple[ScenarioFamily, Candidate], ...]  # (primary family, candidate), in selection order
    shortfall: tuple[tuple[ScenarioFamily, str, int], ...]  # (family, band, missing count)


def _selection_rank(selection_key: str, scenario_hash: str) -> str:
    return hashlib.sha256(f"emberbench/suite-selection/{selection_key}/{scenario_hash}".encode("utf-8")).hexdigest()


def select_suite(candidates: Sequence[Candidate], quotas: Mapping[ScenarioFamily, Mapping[str, int]], selection_key: str) -> Selection:
    """Deterministic family x band stratified selection. Independent of candidate order; no model input.

    Families are filled scarcest first (fewest eligible candidates; ties by
    name). Each candidate is used once and gets the family it was selected
    for as its primary family. Inside a cell, candidates are taken in the
    order of a keyed hash of their content hash, so the evaluator fixes the key
    privately and nobody can steer which scenario is taken. A cell that cannot
    be filled is reported in ``shortfall``; it is never filled from another
    band silently.
    """
    pool = sorted(candidates, key=lambda c: (_selection_rank(selection_key, c.scenario_hash), c.scenario_hash))
    if len({c.scenario_hash for c in pool}) != len(pool):
        raise ValueError("duplicate scenario content hash in the candidate pool")
    eligible = {f: sum(1 for c in pool if f in c.tags) for f in quotas}
    order = sorted(quotas, key=lambda f: (eligible[f], f.value))
    used: set[str] = set()
    chosen: list[tuple[ScenarioFamily, Candidate]] = []
    shortfall: list[tuple[ScenarioFamily, str, int]] = []
    for fam in order:
        for band in BANDS:
            need = quotas[fam].get(band, 0)
            for c in pool:
                if need == 0:
                    break
                if c.scenario_hash in used or fam not in c.tags or c.band != band:
                    continue
                used.add(c.scenario_hash)
                chosen.append((fam, c))
                need -= 1
            if need:
                shortfall.append((fam, band, need))
    return Selection(tuple(chosen), tuple(shortfall))


def primary_family(tags: frozenset[ScenarioFamily], counts: Mapping[ScenarioFamily, int]) -> ScenarioFamily | None:
    """For the study only: the scarcest final-family tag (by pool count, ties by name)."""
    final = [f for f in tags if f in FINAL_FAMILIES]
    return min(final, key=lambda f: (counts.get(f, 0), f.value)) if final else None


# --------------------------------------------------------------------------- statistics helpers


def quantile7(values: Sequence[float], q: float) -> float:
    """Type-7 (linear) sample quantile, as in docs/statistics.md."""
    xs = sorted(values)
    if not xs:
        raise ValueError("empty")
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def width95(values: Sequence[float]) -> float:
    return quantile7(values, 0.975) - quantile7(values, 0.025)


def family_balanced(scores_by_family: Mapping[Any, Sequence[float]]) -> float:
    return statistics.fmean(statistics.fmean(v) for v in scores_by_family.values() if v)


def stratified_se(scores_by_family: Mapping[Any, Sequence[float]]) -> float:
    """CLT standard error of the family-balanced mean (strata = families)."""
    fams = [v for v in scores_by_family.values() if v]
    var = sum((statistics.variance(v) / len(v) if len(v) > 1 else 0.0) for v in fams)
    return math.sqrt(var) / len(fams)


def cvar10(values: Sequence[float]) -> float:
    """Mean of the lowest ceil(n / 10) values: the official CVaR10 lower-tail view."""
    xs = sorted(values)
    return statistics.fmean(xs[: max(1, math.ceil(len(xs) / 10))])


def one_way_components(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """(between-group variance, within-group variance) by ANOVA moments."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return (math.nan, math.nan)
    within = statistics.fmean(statistics.variance(g) for g in groups)
    m = statistics.fmean(len(g) for g in groups)
    between_of_means = statistics.variance([statistics.fmean(g) for g in groups])
    return (max(0.0, between_of_means - within / m), within)


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    pos = {x: i for i, x in enumerate(order_b)}
    n = len(order_a)
    concordant = sum(1 for i in range(n) for j in range(i + 1, n) if pos[order_a[i]] < pos[order_a[j]])
    pairs = n * (n - 1) // 2
    return (2 * concordant - pairs) / pairs if pairs else 1.0


class DrawStream:
    """Deterministic integer draws for the resampling studies (SplitMix64 over a public label)."""

    def __init__(self, label: str) -> None:
        self._rng = SplitMix64(development_seed("draws/" + label))

    def below(self, n: int) -> int:
        return self._rng.next_u64() % n


# --------------------------------------------------------------------------- study units and simulated suites


@dataclass(frozen=True)
class Unit:
    """One admitted pool scenario as a sampling unit."""

    key: str  # recipe/scenario_id
    recipe: str
    family: ScenarioFamily
    tags: frozenset[ScenarioFamily]
    band: str
    stochastic: bool
    scores: Mapping[str, tuple[float, ...]]  # comparator -> one official score per applicable world
    steps: tuple[int, ...]  # congestion-aware decision steps per world
    seconds: tuple[float, ...]  # congestion-aware run + official scoring seconds per world
    max_steps: int
    obs_chars: int
    population: int


@dataclass(frozen=True)
class Calibrated:
    record: ScenarioRecord
    calibration: Calibration
    tags: frozenset[ScenarioFamily]


def calibrate_pool(pool: Sequence[ScenarioRecord]) -> list[Calibrated]:
    return [Calibrated(r, calibrate(r.worlds), family_tags(r.profile, [w.fire for w in r.worlds])) for r in pool]


def build_units(calibrated: Sequence[Calibrated], surrogates: Mapping[tuple[str, str, int], Mapping[str, float]]) -> list[Unit]:
    admitted = [c for c in calibrated if c.calibration.admission == ADMITTED]
    counts: Counter[ScenarioFamily] = Counter(f for c in admitted for f in c.tags if f in FINAL_FAMILIES)
    units = []
    for c in admitted:
        fam = primary_family(c.tags, counts)
        if fam is None or c.calibration.band is None:
            continue
        r = c.record
        worlds = [w for w in r.worlds if w.savable > 0]
        scores: dict[str, tuple[float, ...]] = {a: tuple(w.score(a) for w in worlds) for a in CALIBRATION_PANEL}
        for name in sorted({k for w in worlds for k in surrogates.get((r.recipe, r.scenario.scenario_id, w.seed), {})}):
            scores[name] = tuple(surrogates[(r.recipe, r.scenario.scenario_id, w.seed)][name] for w in worlds)
        ca = [next(x for x in w.runs if x.agent == "congestion-aware-v1") for w in worlds]
        units.append(
            Unit(
                key=f"{r.recipe}/{r.scenario.scenario_id}",
                recipe=r.recipe,
                family=fam,
                tags=c.tags,
                band=c.calibration.band,
                stochastic=r.profile.fire_model == "stochastic",
                scores=scores,
                steps=tuple(x.steps for x in ca),
                seconds=tuple(x.seconds for x in ca),
                max_steps=r.profile.max_steps,
                obs_chars=r.obs_chars,
                population=r.profile.population,
            )
        )
    return units


@dataclass(frozen=True)
class Design:
    per_family: int
    bands: tuple[tuple[str, int], ...]  # (band, scenarios per family), sums to per_family
    stochastic_per_family: int
    stochastic_worlds: int  # unique fire worlds per stochastic scenario

    @property
    def worlds_per_family(self) -> int:
        return self.per_family + self.stochastic_per_family * (self.stochastic_worlds - 1)

    def label(self) -> str:
        bands = "/".join(str(n) for _, n in self.bands)
        return f"{self.per_family}/fam [{bands}], {self.stochastic_per_family} stoch x {self.stochastic_worlds}"


def allocate(per_family: int, shares: Sequence[tuple[str, float]]) -> tuple[tuple[str, int], ...]:
    """Largest-remainder rounding of band shares to ``per_family`` scenarios (ties in band order)."""
    raw = [(b, s * per_family) for b, s in shares]
    base = {b: math.floor(x) for b, x in raw}
    left = per_family - sum(base.values())
    for b, _ in sorted(raw, key=lambda bx: (-(bx[1] - math.floor(bx[1])), BANDS.index(bx[0])))[:left]:
        base[b] += 1
    return tuple((b, base[b]) for b in BANDS)


def _slots(design: Design) -> list[tuple[str, bool]]:
    """The (band, stochastic) slots of one family: stochastic slots spread evenly over the band-ordered slots."""
    bands = [b for b, n in design.bands for _ in range(n)]
    m, s = len(bands), design.stochastic_per_family
    return [(b, (i + 1) * s // m > i * s // m) for i, b in enumerate(bands)]


class SuiteSampler:
    """Draws simulated suites of a design from the admitted pool units (with replacement inside each cell).

    A slot is filled from its (family, band, fire model) cell; when that cell
    is empty, from (family, band, other fire model); then from (any family,
    band, fire model); then from (any family, band). Every fallback is counted
    and reported: it marks a family x band x fire-model cell the current
    generator does not reach.
    """

    def __init__(self, units: Sequence[Unit], families: Sequence[ScenarioFamily]) -> None:
        self.families = tuple(families)
        self.cells: dict[tuple[Any, ...], list[Unit]] = defaultdict(list)
        for u in units:
            for key in ((u.family, u.band, u.stochastic), (u.family, u.band), (u.band, u.stochastic), (u.band,)):
                self.cells[key].append(u)

    def pick(self, family: ScenarioFamily, band: str, stochastic: bool, stream: DrawStream) -> tuple[Unit, int]:
        for level, key in enumerate(((family, band, stochastic), (family, band), (band, stochastic), (band,))):
            cell = self.cells.get(key)
            if cell:
                return cell[stream.below(len(cell))], level
        raise ValueError(f"no admitted unit in band {band}")

    def draw(self, design: Design, stream: DrawStream) -> tuple[dict[ScenarioFamily, list[tuple[Unit, int]]], int]:
        suite: dict[ScenarioFamily, list[tuple[Unit, int]]] = {}
        fallbacks = 0
        for f in self.families:
            chosen = []
            for band, stoch in _slots(design):
                u, level = self.pick(f, band, stoch, stream)
                fallbacks += level > 0
                chosen.append((u, min(design.stochastic_worlds, len(u.steps)) if u.stochastic else 1))
            suite[f] = chosen
        return suite, fallbacks


@dataclass(frozen=True)
class DesignResult:
    design: Design
    scenarios: int
    worlds: float  # mean unique worlds per drawn suite
    order: tuple[str, ...]  # comparators by expected score, ascending
    expected: Mapping[str, float]
    width: Mapping[str, float]  # 95% suite-sampling interval width of the family-balanced score
    clt_width: float  # mean predicted within-suite 95% CI width (stratified CLT), over comparators
    pairs: tuple[tuple[str, str], ...]  # adjacent comparators with an expected gap >= the practical difference
    paired_width: Mapping[tuple[str, str], float]
    reversal: Mapping[tuple[str, str], float]
    power: Mapping[tuple[str, str], float]  # P(the paired 95% CI excludes 0 in the right direction)
    exact: float  # P(every practically different adjacent pair in the right order)
    tau: float  # mean Kendall tau to the expected order (all comparators)
    unresolved_gap: float  # largest expected gap (points) of any comparator pair that reverses in > 5% of suites
    family_width: float  # mean over families of the 95% width of one family's mean (best comparator)
    cvar_width: float  # mean over comparators of the 95% width of CVaR10 over unique worlds
    floor_share: float  # share of comparator x world scores below 1 point
    fallback: float  # share of slots filled by a fallback cell
    calls: float  # mean decision calls per model run (congestion-aware steps)
    worst_calls: float  # max_steps summed: a model that never finishes
    cpu_seconds: float  # evaluator run + scoring seconds per model run
    obs_chars: float  # mean step-0 observation characters per world


def evaluate_design(sampler: SuiteSampler, design: Design, comparators: Sequence[str], draws: int, label: str) -> DesignResult:
    stream = DrawStream(f"design/{label}/{design.label()}")
    est: dict[str, list[float]] = defaultdict(list)
    clt: dict[str, list[float]] = defaultdict(list)
    tails: dict[str, list[float]] = defaultdict(list)
    fam_means: dict[ScenarioFamily, list[float]] = defaultdict(list)
    diffs: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    worlds_n: list[int] = []
    calls: list[float] = []
    worst: list[float] = []
    cpu: list[float] = []
    chars: list[float] = []
    floor_hits = floor_total = fallbacks = slots = 0
    all_pairs = [(a, b) for a in comparators for b in comparators if a < b]
    for _ in range(draws):
        suite, fb = sampler.draw(design, stream)
        fallbacks += fb
        slots += sum(len(v) for v in suite.values())
        per_unit: dict[str, dict[ScenarioFamily, list[float]]] = {a: defaultdict(list) for a in comparators}
        world_scores: dict[str, list[float]] = defaultdict(list)
        n = 0
        c = w_ = s_ = ch = 0.0
        for f, items in suite.items():
            for u, k in items:
                n += k
                c += sum(u.steps[:k])
                w_ += u.max_steps * k
                s_ += sum(u.seconds[:k])
                ch += u.obs_chars * k
                for a in comparators:
                    xs = u.scores[a][:k]
                    per_unit[a][f].append(statistics.fmean(xs))
                    world_scores[a].extend(xs)
                    floor_hits += sum(1 for x in xs if x < 1.0)
                    floor_total += len(xs)
        worlds_n.append(n)
        calls.append(c)
        worst.append(w_)
        cpu.append(s_)
        chars.append(ch / n)
        point = {a: family_balanced(per_unit[a]) for a in comparators}
        for a in comparators:
            est[a].append(point[a])
            clt[a].append(2 * 1.96 * stratified_se(per_unit[a]))
            tails[a].append(cvar10(world_scores[a]))
        best_now = max(comparators, key=lambda a: point[a])
        for f, fam_scores in per_unit[best_now].items():
            fam_means[f].append(statistics.fmean(fam_scores))
        for lo, hi in all_pairs:
            d = {f: [y - x for x, y in zip(per_unit[lo][f], per_unit[hi][f], strict=True)] for f in per_unit[lo]}
            diffs[(lo, hi)].append((family_balanced(d), stratified_se(d)))
    expected = {a: statistics.fmean(est[a]) for a in comparators}
    order = tuple(sorted(comparators, key=lambda a: expected[a]))
    pairs = tuple((lo, hi) for lo, hi in zip(order, order[1:]) if expected[hi] - expected[lo] >= float(OFFICIAL_CONSTANTS.practical_difference))

    def oriented(lo: str, hi: str) -> list[tuple[float, float]]:
        return diffs[(lo, hi)] if lo < hi else [(-m, s) for m, s in diffs[(hi, lo)]]

    paired_width = {p: width95([m for m, _ in oriented(*p)]) for p in pairs}
    reversal = {p: sum(1 for m, _ in oriented(*p) if m <= 0) / draws for p in pairs}
    power = {p: sum(1 for m, s in oriented(*p) if m - 1.96 * s > 0) / draws for p in pairs}
    exact = 0
    for i in range(draws):
        exact += all(oriented(lo, hi)[i][0] > 0 for lo, hi in pairs)
    gaps = [(expected[hi] - expected[lo], sum(1 for m, _ in oriented(lo, hi) if m <= 0) / draws) for lo, hi in itertools.combinations(order, 2)]
    unresolved_gap = max((g for g, rev in gaps if rev > 0.05), default=0.0)
    taus = []
    for i in range(draws):
        now = sorted(comparators, key=lambda a: est[a][i])
        taus.append(kendall_tau(now, order))
    return DesignResult(
        design=design,
        scenarios=design.per_family * len(sampler.families),
        worlds=statistics.fmean(worlds_n),
        order=order,
        expected=expected,
        width={a: width95(est[a]) for a in comparators},
        clt_width=statistics.fmean(statistics.fmean(clt[a]) for a in comparators),
        pairs=pairs,
        paired_width=paired_width,
        reversal=reversal,
        power=power,
        exact=exact / draws,
        tau=statistics.fmean(taus),
        unresolved_gap=unresolved_gap,
        family_width=statistics.fmean(width95(v) for v in fam_means.values() if len(v) >= 20),
        cvar_width=statistics.fmean(width95(tails[a]) for a in comparators),
        floor_share=floor_hits / floor_total,
        fallback=fallbacks / slots,
        calls=statistics.fmean(calls),
        worst_calls=statistics.fmean(worst),
        cpu_seconds=statistics.fmean(cpu),
        obs_chars=statistics.fmean(chars),
    )


# --------------------------------------------------------------------------- report helpers


def _fmt(x: float | int | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        if math.isnan(x):
            return "n/a"
        return f"{x:.{digits}f}"
    return str(x)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], digits: int = 2) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c, digits) if not isinstance(c, str) else c for c in r) + " |")
    return "\n".join(out)


def _short(agent: str) -> str:
    return agent.replace("-v1", "").replace("noisy-congestion-", "noisy ε=")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- audit of the public suite


def audit_public(out: Path, seeds: int, workers: int, log: Callable[[str], None]) -> str:
    suite = load_suite(PUBLIC_SUITE_DIR)
    by_id = {s.scenario_id: s for s in suite.scenarios}
    seed_list = [development_seed(f"public-audit/{i}") for i in range(seeds)]
    nominal = build_evaluation_manifest(suite, seed_list)
    dedup = dedupe_schedule(by_id, [(e.scenario_id, e.seed) for e in nominal.episodes])
    unique_manifest = build_unique_world_manifest(suite, seed_list, stochastic_worlds=3)
    family_of = {e.scenario_id: e.primary_family for e in suite.manifest.entries}
    per = dedup.worlds_per_scenario()
    det = [sid for sid, s in by_id.items() if s.fire.model.value == "deterministic"]
    lines = [
        "# Public V1 suite audit (official EMBER Score)",
        "",
        f"Analysis version {ANALYSIS_VERSION}; scoring version {SCORING_VERSION}; simulator {SIMULATOR_VERSION}; "
        f"suite hash `{suite.manifest.suite_hash()}`. World identity: `environment_world_digest` (version {ENVIRONMENT_WORLD_VERSION}).",
        f"Schedule: `build_evaluation_manifest` with {seeds} public development seeds (the full scenario x seed product).",
        "",
        "## Nominal episodes versus unique environment worlds",
        "",
        _table(
            ["quantity", "value"],
            [
                ["nominal scheduled episodes", len(dedup.episodes)],
                ["unique scenarios", len(by_id)],
                ["unique environment worlds", len(dedup.unique)],
                ["deterministic-fire worlds", sum(per[s] for s in det)],
                ["stochastic-fire worlds", sum(per[s] for s in per if s not in det)],
                ["duplicate episodes (same world as an earlier episode)", dedup.duplicate_count],
                ["duplicate rate", _pct(dedup.duplicate_count / len(dedup.episodes))],
                ["episodes of `build_unique_world_manifest` (3 worlds per stochastic scenario)", len(unique_manifest.episodes)],
            ],
        ),
        "",
        _table(
            ["primary family", "scenarios", "nominal episodes", "unique worlds"],
            [
                [fam.value, sum(1 for s in per if family_of[s] == fam), sum(seeds for s in per if family_of[s] == fam), sum(per[s] for s in per if family_of[s] == fam)]
                for fam in ScenarioFamily
                if any(family_of[s] == fam for s in per)
            ],
        ),
        "",
    ]
    agents = (FLOOR_AGENT, *CALIBRATION_PANEL)
    worlds_of: dict[str, list[int]] = defaultdict(list)
    for e in dedup.unique:
        worlds_of[e.scenario_id].append(e.seed)
    tasks: list[tuple[Any, ...]] = [("public", by_id[sid], tuple(worlds_of[sid]), agents) for sid in by_id]
    records = {r.scenario.scenario_id: r for r in _parallel(evaluate_scenario, tasks, workers, log, "public")}
    rows = []
    for entry in suite.manifest.entries:
        r = records[entry.scenario_id]
        cal = calibrate(r.worlds)
        floor = statistics.fmean(w.score(FLOOR_AGENT) for w in r.worlds if w.savable)
        rows.append(
            [
                entry.scenario_id, len(r.worlds), floor, *[cal.panel.get(a) for a in CALIBRATION_PANEL], cal.best, cal.spread,
                f"{sum(w.savable for w in r.worlds) / len(r.worlds):.1f}/{r.profile.population}",
                f"{sum(w.capacity_savable for w in r.worlds) / len(r.worlds):.1f}", cal.ceiling, cal.admission, cal.band or "—",
            ]
        )
    lines += [
        "## Calibration panel on every unique world (official EMBER Score, mean over the scenario's worlds)",
        "",
        _table(
            ["scenario", "worlds", "random floor", *[_short(a) for a in CALIBRATION_PANEL], "B (best)", "spread", "M* / N", "M_cap", "capacity ceiling", "admission", "band"],
            rows,
            1,
        ),
        "",
    ]
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "public-audit.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


# --------------------------------------------------------------------------- the study

POOL_STOCHASTIC_WORLDS = 6  # unique worlds measured per stochastic pool scenario (for the seed study)
SURROGATE_EPSILONS: tuple[float, ...] = (0.05, 0.2)
STUDY_DRAWS = 400

# The final V1 design (docs/evaluation-suite-design.md section 17). Every number is justified by the study.
FINAL_PER_FAMILY = 16
FINAL_BAND_SHARES: tuple[tuple[str, float], ...] = (("easy", 2 / 16), ("moderate", 4 / 16), ("hard", 6 / 16), ("stress", 4 / 16))
FINAL_STOCHASTIC_PER_FAMILY = 4
FINAL_STOCHASTIC_WORLDS = 3
FINAL_K = 3


def final_design(per_family: int = FINAL_PER_FAMILY) -> Design:
    stochastic = round(FINAL_STOCHASTIC_PER_FAMILY * per_family / FINAL_PER_FAMILY)
    return Design(per_family, allocate(per_family, FINAL_BAND_SHARES), stochastic, FINAL_STOCHASTIC_WORLDS)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    return statistics.correlation(rx, ry) if len(set(rx)) > 1 and len(set(ry)) > 1 else math.nan


def _binomial_tail(n: int, p: float, k: int) -> float:
    """P(Binomial(n, p) >= k)."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    below = 0.0
    term = (1 - p) ** n
    for i in range(k):
        below += term
        term *= (n - i) / (i + 1) * p / (1 - p)
    return max(0.0, 1.0 - below)


def required_draws(p: float, quota: int, confidence: float) -> float:
    """Smallest N with P(Binomial(N, p) >= quota) >= confidence (inf when p == 0)."""
    if quota == 0:
        return 0
    if p <= 0:
        return math.inf
    lo, hi = quota, quota
    while _binomial_tail(hi, p, quota) < confidence:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _binomial_tail(mid, p, quota) >= confidence:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _section_pool(pool: Sequence[ScenarioRecord], cal: Sequence[Calibrated], units: Sequence[Unit]) -> list[str]:
    lines = ["## 1. Development pool", ""]
    rows = []
    for name in sorted({r.recipe for r in pool}):
        rs = [c for c in cal if c.record.recipe == name]
        adm = Counter(c.calibration.admission for c in rs)
        bands = Counter(c.calibration.band for c in rs if c.calibration.band)
        rows.append([name, len(rs), sum(len(c.record.worlds) for c in rs), *[adm.get(x, 0) for x in EXCLUSIONS], adm.get(ADMITTED, 0), *[bands.get(b, 0) for b in BANDS]])
    total = [c.calibration.admission for c in cal]
    lines += [
        f"{len(pool)} generated scenarios, {sum(len(r.worlds) for r in pool)} unique worlds "
        f"({sum(1 for r in pool if r.profile.fire_model == 'stochastic')} stochastic scenarios with up to {POOL_STOCHASTIC_WORLDS} worlds each), "
        f"{len(units)} admitted units. Evaluator CPU for the panel: {sum(r.seconds for r in pool) / 3600:.2f} h.",
        "",
        _table(["recipe", "scenarios", "worlds", *EXCLUSIONS, "admitted", *BANDS], rows),
        "",
        f"Admission over the pool: " + ", ".join(f"{x} {_pct(total.count(x) / len(total))}" for x in (*EXCLUSIONS, ADMITTED)) + ".",
        "",
    ]
    return lines


def _section_distribution(cal: Sequence[Calibrated]) -> list[str]:
    app = [c.calibration for c in cal if c.calibration.admission != "not_applicable"]
    worlds = [w for c in cal for w in c.record.worlds if w.savable]
    world_best = [max(w.panel().values()) for w in worlds]
    world_worst = [min(w.panel().values()) for w in worlds]
    rows = []
    for name, vals in (
        ("B = best panel score (scenario mean)", [c.best for c in app]),
        ("worst panel score (scenario mean)", [c.worst for c in app]),
        ("panel spread B - worst", [c.spread for c in app]),
        *[(f"{_short(a)} (scenario mean)", [c.panel[a] for c in app]) for a in CALIBRATION_PANEL],
    ):
        rows.append([name, *[quantile7(vals, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)], statistics.fmean(vals)])
    hist = Counter(min(9, int(b // 10)) for b in (c.best for c in app))
    return [
        "## 2. Official EMBER Score distribution of the calibration panel",
        "",
        _table(["quantity", "min", "p10", "p25", "median", "p75", "p90", "max", "mean"], rows, 1),
        "",
        "Histogram of B in 10-point bins (0-10, ..., 90-100): " + ", ".join(str(hist.get(i, 0)) for i in range(10)) + ".",
        "",
        f"Per world: every panel member >= {SATURATED:g}: {_pct(sum(1 for x in world_worst if x >= SATURATED) / len(worlds))}; "
        f"every panel member < {FLOOR:g}: {_pct(sum(1 for x in world_best if x < FLOOR) / len(worlds))}; "
        f"best panel member >= {EASY_MIN_BEST:g}: {_pct(sum(1 for x in world_best if x >= EASY_MIN_BEST) / len(worlds))}.",
        "",
    ]


def _section_bands(cal: Sequence[Calibrated], units: Sequence[Unit], comparators: Sequence[str]) -> list[str]:
    adm = [c for c in cal if c.calibration.admission == ADMITTED]
    rows = []
    for b in BANDS:
        us = [u for u in units if u.band == b]
        cs = [c for c in adm if c.calibration.band == b]
        if not us:
            continue
        means = {a: statistics.fmean(statistics.fmean(u.scores[a]) for u in us) for a in comparators}
        distinct = statistics.fmean(1.0 if len({round(c.calibration.panel[a]) for a in CALIBRATION_PANEL}) >= 3 else 0.0 for c in cs)
        zero = statistics.fmean(1.0 if min(c.calibration.panel.values()) < 1 else 0.0 for c in cs)
        rows.append([b, len(us), *[means[a] for a in comparators], statistics.fmean(c.calibration.spread for c in cs), _pct(distinct), _pct(zero)])
    lines = [
        "## 3. Difficulty bands (frozen calibration panel, official score)",
        "",
        f"Bands on B: easy >= {EASY_MIN_BEST:g}, moderate [{MODERATE_MIN_BEST:g}, {EASY_MIN_BEST:g}), hard [{HARD_MIN_BEST:g}, {MODERATE_MIN_BEST:g}), "
        f"stress < {HARD_MIN_BEST:g}. Admission excludes not-applicable, capacity-artifact (ceiling < {CEILING_MIN:g}), saturated (all >= {SATURATED:g}), "
        f"floor (all < {FLOOR:g}) and no-separation (spread < {float(OFFICIAL_CONSTANTS.practical_difference):g} point) scenarios.",
        "",
        _table(["band", "units", *[_short(a) for a in comparators], "panel spread", "panel has >= 3 distinct levels", "some panel member < 1"], rows, 1),
        "",
    ]
    # Separation: mean paired gap and share of units where the order of two comparators flips relative to the band mean.
    sep_rows = []
    for b in BANDS:
        us = [u for u in units if u.band == b]
        if len(us) < 5:
            continue
        means = {a: statistics.fmean(statistics.fmean(u.scores[a]) for u in us) for a in comparators}
        order = sorted(comparators, key=lambda a: means[a])
        gaps = [(lo, hi) for lo, hi in zip(order, order[1:])]
        cells = []
        for lo, hi in gaps:
            d = [statistics.fmean(u.scores[hi]) - statistics.fmean(u.scores[lo]) for u in us]
            sd = statistics.pstdev(d)
            cells.append(f"{_short(hi)} > {_short(lo)}: {statistics.fmean(d):.1f} (d/sd {statistics.fmean(d) / sd if sd else math.inf:.2f})")
        sep_rows.append([b, len(us), "; ".join(cells)])
    lines += ["Adjacent-comparator separation inside each band (mean paired gap in points and standardized effect):", "", _table(["band", "units", "adjacent gaps"], sep_rows), ""]
    return lines


def _section_capacity(cal: Sequence[Calibrated]) -> list[str]:
    app = [c for c in cal if c.calibration.admission != "not_applicable"]
    limited = [c for c in app if c.calibration.capacity_limited]
    worlds = [w for c in app for w in c.record.worlds if w.savable]
    gaps = [w for w in worlds if w.capacity_savable < w.savable]
    rows = []
    for c in sorted(limited, key=lambda c: c.calibration.ceiling or 0.0)[:12]:
        w = min((w for w in c.record.worlds if w.savable), key=lambda w: w.capacity_savable - w.savable)
        rows.append([f"{c.record.recipe}/{c.record.scenario.scenario_id}", c.record.profile.population, w.savable, w.capacity_savable, c.calibration.ceiling, c.calibration.best, c.calibration.admission, difficulty_band(c.calibration.best)])
    would_stress = sum(1 for c in limited if c.calibration.admission == "capacity_artifact" and difficulty_band(c.calibration.best) == "stress")
    return [
        "## 4. Practical unsolvability: the capacity filter",
        "",
        f"Worlds with M_cap < M* (some savable people cannot all leave under exit and cut capacity): {len(gaps)}/{len(worlds)} ({_pct(len(gaps) / len(worlds))}); "
        f"scenarios with such a world: {len(limited)}/{len(app)}. Scenarios excluded as capacity artifacts (mean ceiling < {CEILING_MIN:g}): "
        f"{sum(1 for c in app if c.calibration.admission == 'capacity_artifact')}, of which {would_stress} would otherwise have been banded Stress.",
        "",
        _table(["scenario", "N", "M*", "M_cap", "ceiling", "B", "admission", "band by B alone"], rows, 1) if rows else "No capacity-limited world in the pool.",
        "",
    ]


def _section_proxies(cal: Sequence[Calibrated]) -> list[str]:
    adm = [c for c in cal if c.calibration.admission == ADMITTED]
    gap = [100 - c.calibration.best for c in adm]
    feats: dict[str, list[float]] = defaultdict(list)
    for c in adm:
        p = c.record.profile
        fires = [w.fire for w in c.record.worlds if w.savable]
        feats["population"].append(p.population)
        feats["crowd density"].append(p.crowd_density)
        feats["N / min cut"].append(min(p.bottleneck_severity, 1e6))
        feats["obstacle density"].append(p.obstacle_density)
        feats["route redundancy"].append(p.route_redundancy)
        feats["ignition-person distance"].append(p.ignition_person_distance or 1e6)
        feats["threatened share (privileged)"].append(statistics.fmean(f.threatened_share for f in fires))
        feats["recoverable invalidation (privileged)"].append(statistics.fmean(f.recoverable_invalidation_share for f in fires))
        feats["savable share M*/N (privileged)"].append(statistics.fmean(w.savable / p.population for w in c.record.worlds if w.savable))
    return [
        "## 5. Do structural proxies predict difficulty? (Spearman with 100 - B, admitted scenarios)",
        "",
        _table(["proxy", "rho"], [[k, _spearman(v, gap)] for k, v in feats.items()]),
        "",
    ]


def _section_families(cal: Sequence[Calibrated], units: Sequence[Unit]) -> list[str]:
    adm = [c for c in cal if c.calibration.admission == ADMITTED]
    app = [c for c in cal if c.calibration.admission != "not_applicable"]
    rows = []
    for f in ScenarioFamily:
        tagged = [c for c in app if f in c.tags]
        tadm = [c for c in adm if f in c.tags]
        prim = [u for u in units if u.family == f]
        bands = Counter(c.calibration.band for c in tadm)
        mean_panel = {a: statistics.fmean(c.calibration.panel[a] for c in tadm) for a in CALIBRATION_PANEL} if tadm else {}
        rows.append(
            [
                f.value + ("" if f in FINAL_FAMILIES else " (secondary)"), len(tagged), _pct(len(tagged) / len(app)), len(tadm), len(prim),
                sum(1 for c in tadm if c.record.profile.fire_model == "stochastic"), *[bands.get(b, 0) for b in BANDS],
                *[mean_panel.get(a) for a in CALIBRATION_PANEL],
            ]
        )
    lines = [
        "## 6. Family audit",
        "",
        "Tag availability among applicable pool scenarios (a scenario can carry several tags), admitted scenarios by band, "
        "and the mean official score of each panel member over the admitted tagged scenarios.",
        "",
        _table(["family", "tagged", "share", "admitted", "primary units", "stochastic", *BANDS, *[_short(a) for a in CALIBRATION_PANEL]], rows, 1),
        "",
    ]
    fams = list(ScenarioFamily)
    jac = []
    for f in fams:
        row: list[Any] = [f.value]
        for g in fams:
            both = sum(1 for c in app if f in c.tags and g in c.tags)
            either = sum(1 for c in app if f in c.tags or g in c.tags)
            row.append(both / either if either else 0.0)
        jac.append(row)
    lines += ["Tag overlap (Jaccard index over applicable pool scenarios; 1 = the same scenarios):", "", _table(["family", *[str(i + 1) for i in range(len(fams))]], jac), ""]
    lines += ["Column numbers follow the row order. " + "; ".join(f"{i + 1} = {f.value}" for i, f in enumerate(fams)) + ".", ""]
    # Capability signal: which official quality components bind in each family (mean over admitted tagged worlds, congestion-aware).
    comp_rows = []
    for f in ScenarioFamily:
        runs = [x for c in adm if f in c.tags for w in c.record.worlds if w.savable for x in w.runs if x.agent == "congestion-aware-v1"]
        if not runs:
            continue
        severe = Counter(s for x in runs for s in x.severe)
        comp_rows.append(
            [
                f.value, len(runs), statistics.fmean(x.score for x in runs), statistics.fmean(x.safety or 0.0 for x in runs),
                statistics.fmean(x.quality or 0.0 for x in runs), statistics.fmean(x.avoidable_losses for x in runs),
                *[_pct(severe.get(k, 0) / len(runs)) for k in ("time", "hazard", "path", "replanning")],
            ]
        )
    lines += [
        "What fails in each family (congestion-aware-v1 on admitted tagged worlds): official score, Safety, Quality, avoidable losses "
        "and the share of worlds with each severe quality failure.",
        "",
        _table(["family", "worlds", "score", "safety", "quality", "avoidable losses", "severe time", "severe hazard", "severe path", "severe replanning"], comp_rows),
        "",
    ]
    return lines


def _design_table(results: Sequence[DesignResult], comparators: Sequence[str]) -> str:
    rows = []
    for r in results:
        pairs = ", ".join(f"{_short(lo)}<{_short(hi)} {r.reversal[(lo, hi)]:.3f}/{r.power[(lo, hi)]:.2f}" for lo, hi in r.pairs)
        rows.append(
            [
                r.design.label(), r.scenarios, round(r.worlds), max(r.width.values()), statistics.fmean(r.width.values()), r.clt_width,
                max(r.paired_width.values()) if r.paired_width else math.nan, r.unresolved_gap, r.exact, r.tau, r.family_width, r.cvar_width, _pct(r.floor_share), _pct(r.fallback), pairs,
            ]
        )
    return _table(
        ["design", "scenarios", "unique worlds", "max 95% width", "mean 95% width", "mean CLT CI width", "max paired width", "largest gap with P(reversal) > 5%", "P(exact order)", "Kendall tau",
         "family-mean width", "CVaR10 width", "scores < 1", "fallback slots", "adjacent pairs: P(reversal)/P(paired CI resolves)"],
        rows,
    )


def _section_size(sampler: SuiteSampler, comparators: Sequence[str], draws: int) -> tuple[list[str], dict[str, list[tuple[float, float]]]]:
    results = [evaluate_design(sampler, final_design(m), comparators, draws, "size") for m in (4, 8, 10, 12, 16, 20, 24)]
    curve: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in results:
        for a in comparators:
            curve[a].append((r.worlds, r.width[a]))
    work = [
        [r.design.per_family, r.scenarios, round(r.worlds), r.calls, r.worst_calls, r.cpu_seconds / 60, r.obs_chars, r.calls * r.obs_chars / 4 / 1e6]
        for r in results
    ]
    order = results[-1].order
    lines = [
        "## 7. Suite size (final allocation shares and fire-model mix, scaled)",
        "",
        f"{draws} simulated suites per size, drawn from the admitted units cell by cell. Comparators, by expected score: "
        + ", ".join(f"{_short(a)} {results[-1].expected[a]:.1f}" for a in order) + ".",
        "",
        _design_table(results, comparators),
        "",
        "Workload per model run (one decision call per simulator step; congestion-aware-v1 step counts as the competent-model proxy):",
        "",
        _table(["scenarios / family", "scenarios", "unique worlds", "decision calls", "worst-case calls (max_steps)", "evaluator CPU min", "step-0 obs chars", "input Mtokens (chars/4, step-0 size)"], work, 1),
        "",
    ]
    return lines, curve


ALLOCATIONS: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    ("uniform 4/4/4/4", (("easy", 0.25), ("moderate", 0.25), ("hard", 0.25), ("stress", 0.25))),
    ("previous 2/4/8/6 (scaled)", (("easy", 0.1), ("moderate", 0.2), ("hard", 0.4), ("stress", 0.3))),
    ("final 2/4/6/4", FINAL_BAND_SHARES),
    ("hard-heavy 1/3/8/4", (("easy", 1 / 16), ("moderate", 3 / 16), ("hard", 8 / 16), ("stress", 4 / 16))),
    ("stress-heavy 1/3/4/8", (("easy", 1 / 16), ("moderate", 3 / 16), ("hard", 4 / 16), ("stress", 8 / 16))),
    ("no easy 0/6/6/4", (("easy", 0.0), ("moderate", 6 / 16), ("hard", 6 / 16), ("stress", 4 / 16))),
)


def _section_allocation(sampler: SuiteSampler, comparators: Sequence[str], draws: int) -> list[str]:
    results = []
    for _, shares in ALLOCATIONS:
        d = Design(FINAL_PER_FAMILY, allocate(FINAL_PER_FAMILY, shares), FINAL_STOCHASTIC_PER_FAMILY, FINAL_STOCHASTIC_WORLDS)
        results.append(evaluate_design(sampler, d, comparators, draws, "allocation"))
    rows = [
        [name, r.design.label(), *[r.expected[a] for a in comparators], _pct(r.floor_share), r.exact, r.tau, statistics.fmean(r.power.values()) if r.power else math.nan, statistics.fmean(r.width.values())]
        for (name, _), r in zip(ALLOCATIONS, results, strict=True)
    ]
    return [
        f"## 8. Difficulty allocation ({FINAL_PER_FAMILY} scenarios per family)",
        "",
        _table(["allocation", "design", *[_short(a) for a in comparators], "scores < 1", "P(exact order)", "Kendall tau", "mean P(paired CI resolves)", "mean 95% width"], rows),
        "",
    ]


def _section_mix(sampler: SuiteSampler, units: Sequence[Unit], comparators: Sequence[str], draws: int) -> list[str]:
    stoch = [u for u in units if u.stochastic and len(u.steps) >= 2]
    comp_rows = []
    for a in comparators:
        between, within = one_way_components([list(u.scores[a]) for u in stoch])
        ratio = within / between if between > 0 else math.inf
        comp_rows.append([_short(a), len(stoch), between, within, ratio, within / (between + within) if between + within > 0 else math.nan])
    det_units = [u for u in units if not u.stochastic]
    st_units = [u for u in units if u.stochastic]
    lvl_rows = [
        [name, len(us), *[statistics.fmean(statistics.fmean(u.scores[a]) for u in us) for a in comparators]]
        for name, us in (("deterministic", det_units), ("stochastic", st_units))
        if us
    ]
    lines = [
        "## 9. Deterministic versus stochastic fire, and worlds per stochastic scenario",
        "",
        "Mean official score of admitted units by fire model:",
        "",
        _table(["fire model", "units", *[_short(a) for a in comparators]], lvl_rows, 1),
        "",
        f"Variance components over admitted stochastic units with >= 2 worlds (one-way ANOVA moments; up to {POOL_STOCHASTIC_WORLDS} worlds each):",
        "",
        _table(["comparator", "units", "between-scenario var", "within-scenario (world) var", "within / between", "share of variance within"], comp_rows),
        "",
        "With an episode budget E spent on S stochastic scenarios x W worlds (E = S W), Var = sigma_b^2 / S + sigma_w^2 / E: at a fixed budget, "
        "more scenarios always beat more worlds per scenario. Extra worlds are worth their cost only to measure robustness to fire realisations inside a scenario.",
        "",
    ]
    designs = []
    for s, w in ((0, 1), (4, 1), (4, 2), (4, 3), (4, 5), (8, 1), (8, 3), (16, 1), (16, 3)):
        designs.append(Design(FINAL_PER_FAMILY, allocate(FINAL_PER_FAMILY, FINAL_BAND_SHARES), s, w))
    results = [evaluate_design(sampler, d, comparators, draws, "mix") for d in designs]
    lines += ["Simulated suites with different fire-model mixes:", "", _design_table(results, comparators), ""]
    return lines


def _section_trials(units: Sequence[Unit], workers: int, out: Path, log: Callable[[str], None], limit: int, trials: int) -> list[str]:
    picked = sorted(units, key=lambda u: hashlib.sha256(("trials/" + u.key).encode()).hexdigest())[:limit]
    tasks: list[tuple[Any, ...]] = []
    for u in picked:
        recipe, sid = u.key.split("/", 1)
        tasks.append((u.key, recipe, sid))
    key = _cache_key("trials", {"units": [t[0] for t in tasks], "eps": list(SURROGATE_EPSILONS), "trials": trials})
    cache = out / f"trials-{key}.json"
    results: dict[str, list[list[float]]]
    if cache.exists():
        log(f"  trials: cached {cache.name}")
        results = json.loads(cache.read_text(encoding="utf-8"))
    else:
        scen = _UNIT_SCENARIOS
        run_tasks: list[tuple[Any, ...]] = [(scen[t[0]][0], scen[t[0]][1], e, trials) for e in SURROGATE_EPSILONS for t in tasks]
        flat = _parallel(evaluate_trials, run_tasks, workers, log, "trials")
        results = {}
        for i, e in enumerate(SURROGATE_EPSILONS):
            results[str(e)] = flat[i * len(tasks) : (i + 1) * len(tasks)]
        cache.write_text(json.dumps(results), encoding="utf-8")
    worlds_final = final_design().worlds_per_family * len(FINAL_FAMILIES)
    rows = []
    for e in SURROGATE_EPSILONS:
        per_world = results[str(e)]
        between, within = one_way_components(per_world)
        for k in (1, 2, 3, 5, 8, 10):
            if k > trials:
                continue
            inflation = 1 + (within / k) / between if between > 0 else math.inf
            trial_sd = math.sqrt(within / k / worlds_final)
            gain = math.sqrt(1 + within / between) - math.sqrt(inflation) if between > 0 else math.nan
            rows.append([f"{e:g}", k, between, within, math.sqrt(inflation), trial_sd, gain])
    return [
        "## 10. Model repetition (K trials on the same frozen worlds)",
        "",
        f"Surrogates: congestion-aware-v1 with epsilon random actions, {trials} trials on each of {len(picked)} admitted worlds (first world of each unit). "
        f"Trial-noise SD of the suite mean uses the final suite's {worlds_final} unique worlds.",
        "",
        _table(["epsilon", "K", "between-world var", "within-world (trial) var", "CI width inflation vs K=inf", "trial-noise SD of suite mean (points)", "width saved vs K=1"], rows, 3),
        "",
    ]


_UNIT_SCENARIOS: dict[str, tuple[Scenario, int]] = {}  # unit key -> (scenario, first applicable seed); filled by study()


def _section_pool_size(cal: Sequence[Calibrated], design: Design, draws: int) -> list[str]:
    adm = [c for c in cal if c.calibration.admission == ADMITTED]
    n_pool = len(cal)
    quotas = {f: dict(design.bands) for f in FINAL_FAMILIES}
    cells = len(FINAL_FAMILIES) * sum(1 for _, q in design.bands if q)
    confidence = 1 - 0.05 / cells
    recipes = sorted({c.record.recipe for c in cal})
    per_recipe_n = Counter(c.record.recipe for c in cal)
    fam_rows = []
    cell_rows = []
    mix_need = 0.0
    targeted = 0.0
    infeasible: list[str] = []
    for f in FINAL_FAMILIES:
        tagged = [c for c in adm if f in c.tags]
        fam_rows.append([f.value, len(tagged), _pct(len(tagged) / n_pool), *[_pct(sum(1 for c in tagged if c.calibration.band == b) / n_pool) for b in BANDS]])
        for b, q in design.bands:
            if not q:
                continue
            hits = sum(1 for c in tagged if c.calibration.band == b)
            p = hits / n_pool
            need = required_draws(p, q, confidence)
            best_p, best_r = 0.0, "—"
            for r in recipes:
                pr = sum(1 for c in tagged if c.calibration.band == b and c.record.recipe == r) / per_recipe_n[r]
                if pr > best_p:
                    best_p, best_r = pr, r
            need_t = required_draws(best_p, q, confidence)
            if math.isinf(need) or math.isinf(need_t):
                infeasible.append(f"{f.value}/{b}")
            else:
                mix_need = max(mix_need, need)
                targeted += need_t
            cell_rows.append([f.value, b, q, hits, p, need, best_r, best_p, need_t])
    band_rows = [[b, _pct(sum(1 for c in adm if c.calibration.band == b) / n_pool)] for b in BANDS]
    cell_rows.sort(key=lambda r: (r[4], r[0], BANDS.index(r[1])))
    # Empirical fill: deterministic subsets of the pool, selection with the real stratified selector.
    fill_rows = []
    ordered = sorted(adm, key=lambda c: hashlib.sha256(("pool-size/" + c.record.recipe + c.record.scenario.content_hash()).encode()).hexdigest())
    cands_all = [Candidate(c.record.scenario.scenario_id, c.record.scenario.content_hash(), frozenset(t for t in c.tags if t in FINAL_FAMILIES), c.calibration.band or "", c.record.profile.fire_model == "stochastic") for c in ordered]
    scale_of = {c.record.scenario.content_hash(): i for i, c in enumerate(cal)}
    index_in_pool = sorted(range(n_pool), key=lambda i: hashlib.sha256(f"pool-size/{i}".encode()).hexdigest())
    for size in (n_pool // 8, n_pool // 4, n_pool // 2, n_pool):
        fills, fulls = [], 0
        for rep in range(5):
            chosen_idx = set(index_in_pool[(rep * size) % n_pool :][:size] + index_in_pool[: max(0, (rep * size) % n_pool + size - n_pool)])
            cands = [x for x in cands_all if scale_of[x.scenario_hash] in chosen_idx]
            sel = select_suite(cands, quotas, f"pool-size/{size}/{rep}")
            filled = len(sel.chosen) / (sum(dict(design.bands).values()) * len(FINAL_FAMILIES))
            fills.append(filled)
            fulls += not sel.shortfall
        fill_rows.append([size, statistics.fmean(fills), f"{fulls}/5"])
    total = design.per_family * len(FINAL_FAMILIES)
    return [
        "## 11. Candidate-pool size",
        "",
        f"Acceptance = admitted, tagged with the family and in the band, per generated candidate ({n_pool} candidates, current recipe mix).",
        "",
        _table(["family", "admitted tagged", "acceptance", *[f"{b} acceptance" for b in BANDS]], fam_rows),
        "",
        _table(["band", "acceptance (any family)"], band_rows),
        "",
        f"Per cell of the final quota ({design.label()}), the binomial number of candidates N that fills the cell with probability "
        f">= {confidence:.4f} (a 95% family-wise guarantee over {cells} cells, Bonferroni), rarest cells first. "
        "'targeted' uses the recipe with the highest acceptance for that cell, i.e. generation aimed at the cell.",
        "",
        _table(["family", "band", "quota", "pool hits", "acceptance p", "N (current mix)", "best recipe", "p (best recipe)", "N (targeted)"], cell_rows, 4),
        "",
        f"Cells the current generator never produced (no finite pool fills them): {', '.join(infeasible) or 'none'}. "
        f"Over the other cells: required pool with the current recipe mix {_fmt(mix_need, 0)} candidates ({_fmt(mix_need / total, 1)} x the {total}-scenario suite); "
        f"targeted per-cell generation {_fmt(targeted, 0)} candidates ({_fmt(targeted / total, 1)} x).",
        "",
        "Empirical fill with the real selector on deterministic subsets of the pool (5 subsets per size):",
        "",
        _table(["candidates", "mean share of quota filled", "complete fills"], fill_rows, 3),
        "",
    ]


def _svg(curve: Mapping[str, Sequence[tuple[float, float]]]) -> str:
    """A small dependency-free log-x line chart of 95% suite-sampling width against unique worlds."""
    w, h, left, bottom, top, right = 640, 360, 60, 40, 20, 190
    xs = [x for pts in curve.values() for x, _ in pts]
    ys = [y for pts in curve.values() for _, y in pts]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    y1 = max(ys) * 1.1

    def px(x: float) -> float:
        return left + (math.log10(x) - x0) / (x1 - x0 or 1) * (w - left - right)

    def py(y: float) -> float:
        return h - bottom - y / y1 * (h - top - bottom)

    colors = ["#2a6fdb", "#e07b00", "#1a9e55", "#c43c5c", "#7d5bbe", "#6b6b6b"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="11">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<line x1="{left}" y1="{h - bottom}" x2="{w - right}" y2="{h - bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h - bottom}" stroke="#444"/>',
        f'<text x="{(left + w - right) / 2}" y="{h - 8}" text-anchor="middle">unique worlds (log scale)</text>',
        f'<text x="14" y="{(top + h - bottom) / 2}" transform="rotate(-90 14 {(top + h - bottom) / 2})" text-anchor="middle">95% width, official EMBER points</text>',
    ]
    for x in sorted({round(x) for x in xs}):
        out.append(f'<text x="{px(x):.1f}" y="{h - bottom + 14}" text-anchor="middle">{x}</text>')
    for t in range(0, int(y1) + 1, max(1, int(y1) // 5)):
        out.append(f'<text x="{left - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t}</text>')
        out.append(f'<line x1="{left}" y1="{py(t):.1f}" x2="{w - right}" y2="{py(t):.1f}" stroke="#ddd"/>')
    for i, (agent, pts) in enumerate(sorted(curve.items())):
        c = colors[i % len(colors)]
        path = " ".join(f"{'M' if j == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for j, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2"/>')
        ly = top + 14 + i * 16
        out.append(f'<line x1="{w - right + 10}" y1="{ly - 4}" x2="{w - right + 28}" y2="{ly - 4}" stroke="{c}" stroke-width="2"/>')
        out.append(f'<text x="{w - right + 32}" y="{ly}">{agent}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _section_final(sampler: SuiteSampler, comparators: Sequence[str], draws: int) -> list[str]:
    d = final_design()
    r = evaluate_design(sampler, d, comparators, draws, "final")
    worlds = d.worlds_per_family * len(FINAL_FAMILIES)
    scenarios = d.per_family * len(FINAL_FAMILIES)
    stoch = d.stochastic_per_family * len(FINAL_FAMILIES)
    calls_per_world = r.calls / r.worlds
    worst_per_world = r.worst_calls / r.worlds
    rows: list[list[Any]] = [
        ["primary families", len(FINAL_FAMILIES)],
        ["scenarios per family", d.per_family],
        ["difficulty per family (easy/moderate/hard/stress)", "/".join(str(n) for _, n in d.bands)],
        ["deterministic / stochastic scenarios per family", f"{d.per_family - d.stochastic_per_family} / {d.stochastic_per_family}"],
        ["unique fire worlds per stochastic scenario", d.stochastic_worlds],
        ["total scenarios", scenarios],
        ["total unique worlds (design)", worlds],
        ["unique worlds in simulated suites from the development pool (stochastic slots without a stochastic candidate fall back)", round(r.worlds)],
        ["share of slots filled by a fallback cell", _pct(r.fallback)],
        ["episodes, deterministic model", worlds],
        [f"episodes, nondeterministic model (K = {FINAL_K})", FINAL_K * worlds],
        ["expected decision calls per model run", round(calls_per_world * worlds)],
        [f"expected decision calls, nondeterministic model (K = {FINAL_K})", round(FINAL_K * calls_per_world * worlds)],
        ["worst-case decision calls per model run (every episode to max_steps)", round(worst_per_world * worlds)],
        ["mean step-0 observation size (characters)", round(r.obs_chars)],
        ["approximate input tokens per model run (calls x step-0 chars / 4)", f"{calls_per_world * worlds * r.obs_chars / 4 / 1e6:.1f} M"],
        ["evaluator CPU per model run (simulator + official scoring)", f"{r.cpu_seconds / 60:.0f} min"],
        ["95% suite-sampling width of the family-balanced score (max over comparators)", f"{max(r.width.values()):.2f} points"],
        ["mean within-suite 95% CI width (stratified CLT)", f"{r.clt_width:.2f} points"],
        ["max paired 95% width (adjacent comparators)", f"{max(r.paired_width.values()):.2f} points" if r.paired_width else "—"],
        ["largest expected gap of a comparator pair that reverses in > 5% of suites", f"{r.unresolved_gap:.2f} points"],
        ["P(exact order of adjacent comparators >= 1 point apart)", f"{r.exact:.3f}"],
        ["mean Kendall tau to the expected order", f"{r.tau:.3f}"],
        ["family-mean 95% width (one family, best comparator)", f"{r.family_width:.2f} points"],
        ["CVaR10 95% width", f"{r.cvar_width:.2f} points"],
    ]
    return ["## 12. The final V1 design, measured", "", _table(["quantity", "value"], rows), ""]


def study(out: Path, quick: bool, workers: int, log: Callable[[str], None]) -> str:
    scale = 12 if quick else 120
    draws = 60 if quick else STUDY_DRAWS
    pool = build_pool(scale, POOL_STOCHASTIC_WORLDS, workers, out, log)
    surrogates = surrogate_scores(pool, SURROGATE_EPSILONS, workers, out, log)
    cal = calibrate_pool(pool)
    units = build_units(cal, surrogates)
    for c in cal:
        first = next((w.seed for w in c.record.worlds if w.savable), None)
        if first is not None:
            _UNIT_SCENARIOS[f"{c.record.recipe}/{c.record.scenario.scenario_id}"] = (c.record.scenario, first)
    comparators = (*CALIBRATION_PANEL, *(surrogate_name(e) for e in SURROGATE_EPSILONS))
    sampler = SuiteSampler(units, FINAL_FAMILIES)
    lines = [
        "# Development-pool calibration and suite-design study (official EMBER Score)",
        "",
        f"Analysis version {ANALYSIS_VERSION}; scoring version {SCORING_VERSION}; simulator {SIMULATOR_VERSION}; generator {GENERATOR_VERSION}; "
        f"calibration panel {CALIBRATION_PANEL_VERSION}; recipes at scale {scale}. Every score is the official EMBER Score. Development seeds only.",
        "",
    ]
    lines += _section_pool(pool, cal, units)
    lines += _section_distribution(cal)
    lines += _section_bands(cal, units, comparators)
    lines += _section_capacity(cal)
    lines += _section_proxies(cal)
    lines += _section_families(cal, units)
    size_lines, curve = _section_size(sampler, comparators, draws)
    lines += size_lines
    lines += _section_allocation(sampler, comparators, draws)
    lines += _section_mix(sampler, units, comparators, draws)
    lines += _section_trials(units, workers, out, log, limit=20 if quick else 60, trials=10)
    lines += _section_pool_size(cal, final_design(), draws)
    lines += _section_final(sampler, comparators, draws)
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    (out / "ci-width.svg").write_text(_svg({_short(a): v for a, v in curve.items()}), encoding="utf-8", newline="\n")
    return text


# --------------------------------------------------------------------------- generator coverage probe

# Config-only recipes for the cells the development pool left empty (stochastic fire in the
# single-exit-dense, maze and crowd families; the maze Moderate band). Development seeds only.
GAP_PROBE_RECIPES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("probe-maze-crowded", {"grid": (16, 12), "population": (12, 24), "exits": (2, 3), "ignitions": (1, 2), "partitions": (2, 4),
                            "doors": (1, 1), "door_width": (1, 1), "scatter": (0.1, 0.25), "interval": (1, 2), "max_steps": 100}),
    ("probe-maze-stochastic", {"grid": (16, 12), "population": (6, 16), "exits": (2, 3), "ignitions": (1, 2), "partitions": (2, 4),
                               "doors": (1, 1), "door_width": (1, 1), "scatter": (0.1, 0.25), "model": "stochastic", "interval": (1, 2),
                               "max_steps": 100}),
    ("probe-dense-stochastic", {"grid": (14, 10), "population": (20, 40), "exits": (1, 1), "ignitions": (1, 1), "model": "stochastic",
                                "interval": (1, 3), "max_steps": 90}),
    ("probe-crowd-stochastic", {"grid": (30, 18), "population": (100, 120), "exits": (2, 4), "ignitions": (1, 1), "model": "stochastic",
                                "interval": (2, 4), "max_steps": 200}),
)


def gap_probe(out: Path, count: int, workers: int, log: Callable[[str], None]) -> str:
    """Can the existing generator fill the empty cells with configuration only? Admission and family x band per recipe."""
    tasks: list[tuple[Any, ...]] = []
    for name, kwargs in GAP_PROBE_RECIPES:
        n = max(4, count // 4) if name == "probe-crowd-stochastic" else count
        _, config = _config(name, count=n, **kwargs)
        generated = generate_suite(config, development_seed("probe/" + name), batch_id=f"probe-{name}")
        for s in generated.scenarios:
            seeds = [development_seed(f"probe/{name}/{s.scenario_id}/{i}") for i in range(12)]
            tasks.append((name, s, distinct_world_seeds(s, seeds, FINAL_STOCHASTIC_WORLDS), CALIBRATION_PANEL))
    records = _parallel(evaluate_scenario, tasks, workers, log, "probe")
    lines = [
        "# Generator coverage probe (config-only recipes, official EMBER Score)",
        "",
        f"Analysis version {ANALYSIS_VERSION}; generator {GENERATOR_VERSION}; calibration panel {CALIBRATION_PANEL_VERSION}. Development seeds only.",
        "",
    ]
    rows: list[list[Any]] = []
    for name, _ in GAP_PROBE_RECIPES:
        rs = [r for r in records if r.recipe == name]
        admission: Counter[str] = Counter()
        cells: Counter[tuple[str, str]] = Counter()
        for r in rs:
            cal = calibrate(r.worlds)
            admission[cal.admission] += 1
            if cal.band:
                for f in family_tags(r.profile, [w.fire for w in r.worlds]):
                    if f in FINAL_FAMILIES:
                        cells[(f.value, cal.band)] += 1
        fams = sorted({f for f, _ in cells})
        rows += [[name, len(rs), admission[ADMITTED], admission["capacity_artifact"], f, *[cells[(f, b)] for b in BANDS]] for f in fams]
    lines += [_table(["recipe", "scenarios", "admitted", "capacity artifacts", "family tag", *BANDS], rows), ""]
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap-probe.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=["audit-public", "study", "gap-probe"])
    parser.add_argument("--out", type=Path, required=True, help="output directory (keep it outside the repository)")
    parser.add_argument("--quick", action="store_true", help="small pool, for a smoke run")
    parser.add_argument("--seeds", type=int, default=64, help="development seeds per public scenario (audit-public)")
    parser.add_argument("--count", type=int, default=40, help="scenarios per probe recipe (gap-probe; crowd uses a quarter)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    if args.command == "audit-public":
        print(audit_public(args.out, args.seeds, args.workers, log))
    elif args.command == "gap-probe":
        print(gap_probe(args.out, args.count, args.workers, log))
    else:
        print(study(args.out, args.quick, args.workers, log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
