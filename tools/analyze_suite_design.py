"""Evaluation-suite design analysis: unique worlds, difficulty calibration, sample size.

Research tool for docs/evaluation-suite-design.md and docs/difficulty-calibration.md.
It is evaluator-side analysis, not part of the package, and it changes no
simulator, scenario, generator or metric rule.

Usage::

    python tools/analyze_suite_design.py audit-public  --out OUT_DIR
    python tools/analyze_suite_design.py study         --out OUT_DIR [--quick]

``audit-public`` profiles the public V1 suite. ``study`` generates a
**development** candidate pool with the existing generator from public
development seeds, runs the reference panel, and runs the calibration,
sample-size, seed and model-trial studies. Both write ``report.md`` (and
``study`` writes ``ci-width.svg``) into ``OUT_DIR``. Keep ``OUT_DIR`` outside
the repository; the leak guard rejects JSON assets outside the public roots.

Every score in this tool is the PROVISIONAL safety-only analysis score
(:func:`provisional_score`). It is not the EMBER Score. Rerun the study with
the final score after the scoring contract is merged.

The development seeds below are public by construction (they are derived
from a fixed public label). They select development worlds only and must
never be used for a scored evaluation.
"""

import argparse
import json
import hashlib
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from emberbench import (
    BASELINE_IDS,
    SIMULATOR_VERSION,
    EvaluationManifest,
    GeneratorConfig,
    Scenario,
    ScenarioFamily,
    Suite,
    agent_policy,
    baseline_factory,
    build_evaluation_manifest,
    compute_episode_metrics,
    generate_suite,
    load_suite,
    run_episode,
    verify_manifest,
)
from emberbench.agent import Agent, AgentFactory, ActionOutput, serialize_observation
from emberbench.canonical import sha256_hex
from emberbench.evaluation import EpisodeSpec
from emberbench.generator import measure_proxies
from emberbench.grid import Pos, neighbours
from emberbench.observation import OBSERVATION_VERSION, Observation
from emberbench.oracle import fire_trajectory, privileged_escape_bound
from emberbench.rng import SplitMix64
from emberbench.simulator import Simulator

ANALYSIS_VERSION = "0.1.0"
WORLD_IDENTITY_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"

# Public development-seed domain. Seeds from this domain are published here and
# so can never be scored seeds.
DEVELOPMENT_SEED_DOMAIN = "emberbench/suite-design/development/"

# The fair reference panel used for difficulty. ``random-legal-v1`` is the
# floor (Procgen's R_min analogue) and is reported, but it does not set bands.
REFERENCE_PANEL: tuple[str, ...] = (
    "nearest-exit-greedy-v1",
    "static-shortest-path-v1",
    "fire-aware-shortest-path-v1",
    "congestion-aware-v1",
)
FLOOR_AGENT = "random-legal-v1"

# Difficulty cutpoints on the provisional score (0..100) of the reference panel.
SOLVED = 90.0  # a panel member "solves" a world: no avoidable loss, near-full evacuation
STRESS = 30.0  # below this even the best panel member loses most of the attainable score


def development_seed(label: str) -> int:
    """A public 64-bit development seed for ``label``. Never a scored seed."""
    digest = hashlib.sha256((DEVELOPMENT_SEED_DOMAIN + label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# --------------------------------------------------------------------------- world identity


def fire_trajectory_digest(scenario: Scenario, seed: int) -> str:
    """SHA-256 of the exogenous fire of one episode, WITHOUT the seed.

    Two seeds that give the same burning set at every snapshot ``0..max_steps``
    give the same digest. The seed is deliberately excluded, so equal fire
    worlds are detected as equal. (A digest that hashes the seed in, such as
    the ``fire_world_digest`` proposed in PR #32, is unique per seed and so
    cannot detect duplicates.)
    """
    trajectory = fire_trajectory(scenario, seed)
    return sha256_hex(
        {
            "scenario_hash": scenario.content_hash(),
            "burning": [[[p.row, p.col] for p in sorted(burning)] for burning in trajectory],
        }
    )


def world_id(scenario: Scenario, seed: int) -> str:
    """The canonical, model-independent identity of one environment world.

    In V1 the seed reaches the run only through the fire stream, and the
    observation never contains it. A world is therefore fully determined by
    the simulator and observation contracts, the scenario bytes and the
    exogenous fire trajectory. Every deterministic-fire scenario has exactly
    one world, whatever the seed.
    """
    return sha256_hex(
        {
            "world_identity_version": WORLD_IDENTITY_VERSION,
            "simulator_version": SIMULATOR_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "scenario_hash": scenario.content_hash(),
            "fire_trajectory_digest": fire_trajectory_digest(scenario, seed),
        }
    )


@dataclass(frozen=True)
class ScheduledEpisode:
    scenario_id: str
    seed: int
    world_id: str


@dataclass(frozen=True)
class WorldDedup:
    """A seed schedule reduced to its unique worlds (first occurrence wins)."""

    episodes: tuple[ScheduledEpisode, ...]
    unique: tuple[ScheduledEpisode, ...]

    @property
    def duplicate_count(self) -> int:
        return len(self.episodes) - len(self.unique)

    def worlds_per_scenario(self) -> dict[str, int]:
        out: Counter[str] = Counter(e.scenario_id for e in self.unique)
        return dict(sorted(out.items()))


def dedupe_schedule(scenarios: Mapping[str, Scenario], schedule: Sequence[tuple[str, int]]) -> WorldDedup:
    """Map a ``(scenario_id, seed)`` schedule to world IDs and drop repeated worlds."""
    episodes: list[ScheduledEpisode] = []
    seen: set[str] = set()
    unique: list[ScheduledEpisode] = []
    cache: dict[tuple[str, int], str] = {}
    for scenario_id, seed in schedule:
        key = (scenario_id, seed)
        if key not in cache:
            cache[key] = world_id(scenarios[scenario_id], seed)
        ep = ScheduledEpisode(scenario_id, seed, cache[key])
        episodes.append(ep)
        if ep.world_id not in seen:
            seen.add(ep.world_id)
            unique.append(ep)
    return WorldDedup(tuple(episodes), tuple(unique))


def select_distinct_seeds(scenario: Scenario, candidate_seeds: Iterable[int], count: int) -> tuple[int, ...]:
    """The first ``count`` candidate seeds that give pairwise-distinct fire worlds.

    For a deterministic-fire scenario this returns one seed: further seeds add
    no new world. The evaluator draws ``candidate_seeds`` privately and freezes
    the result before any model runs.
    """
    chosen: list[int] = []
    seen: set[str] = set()
    for seed in candidate_seeds:
        digest = fire_trajectory_digest(scenario, seed)
        if digest in seen:
            continue
        seen.add(digest)
        chosen.append(seed)
        if len(chosen) == count or scenario.fire.model.value == "deterministic":
            break
    return tuple(chosen)


def build_unique_world_manifest(
    suite: Suite,
    candidate_seeds: Callable[[str], Iterable[int]],
    stochastic_worlds: int,
) -> EvaluationManifest:
    """An evaluation manifest with one episode per unique world.

    ``build_evaluation_manifest`` schedules every scenario under every seed,
    so a deterministic scenario gets duplicate episodes and a stochastic one
    may get equal fire worlds. This schedule gives a deterministic scenario
    exactly one episode, and a stochastic one ``stochastic_worlds`` episodes
    with pairwise-distinct fire trajectories (``candidate_seeds(scenario_id)``
    is the evaluator's private, ordered candidate stream). The result passes
    the normal manifest validation and ``verify_manifest``.
    """
    episodes: list[EpisodeSpec] = []
    for entry in suite.manifest.entries:
        scenario = suite.get(entry.scenario_id)
        seeds = select_distinct_seeds(scenario, candidate_seeds(entry.scenario_id), stochastic_worlds)
        if scenario.fire.model.value == "stochastic" and len(seeds) < stochastic_worlds:
            raise ValueError(f"{entry.scenario_id}: only {len(seeds)} distinct fire worlds among the candidate seeds")
        for seed in seeds:
            episodes.append(EpisodeSpec(len(episodes), entry.scenario_id, entry.content_hash, seed))
    template = build_evaluation_manifest(suite, [0])
    manifest = replace(template, episodes=tuple(episodes))
    verify_manifest(manifest, suite)
    return manifest


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
    """Max number of vertex-disjoint person-to-exit paths on the static map (unit cell capacity).

    It is the number of people that can cross the narrowest cut per step, so
    ``population / cut`` is a lower bound on the drain time through it.
    """
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
    exits_flammable: bool
    max_steps: int
    obstacle_density: float
    crowd_density: float
    passage_width: int
    route_redundancy: int
    route_length: int | None
    ignition_person_distance: int | None
    ignition_exit_distance: int | None
    exit_separation: int | None
    min_cut: int
    bottleneck_severity: float  # population / min_cut: drain steps through the narrowest cut
    queue_ratio: float  # bottleneck_severity / route_length: >1 means flow-limited, not distance-limited
    max_nearest_exit_load: int
    split_pressure: float | None  # busiest nearest-exit queue / (1 + median detour to the next exit)
    deceptive_share: float  # people whose straight-line-nearest exit is >= 2 steps worse by path
    exit_gap_median: float | None  # median (second - first) path distance to exits
    detour_ratio: float  # mean path / Manhattan distance to the path-nearest exit
    ignition_centrality: float  # 0 = an ignition at the grid centre, 1 = on the border


def profile_scenario(scenario: Scenario) -> ScenarioProfile:
    proxies = measure_proxies(scenario)
    fields = exit_fields(scenario)
    exit_ids = sorted(fields)
    people = sorted(scenario.people, key=lambda p: p.id)
    nearest: dict[str, str] = {}
    gaps: list[int] = []
    detours: list[float] = []
    deceptive = 0
    for p in people:
        dists = sorted((fields[e].get(p.position, math.inf), e) for e in exit_ids)
        d0, e0 = dists[0]
        nearest[p.id] = e0
        if len(dists) > 1 and math.isfinite(dists[1][0]) and math.isfinite(d0):
            gaps.append(int(dists[1][0] - d0))
        pos_of = {e.id: e.position for e in scenario.exits}
        if math.isfinite(d0):
            man = _manhattan(p.position, pos_of[e0])
            detours.append(d0 / man if man else 1.0)
            man_best = min(_manhattan(p.position, pos_of[e]) for e in exit_ids)
            man_nearest = [e for e in exit_ids if _manhattan(p.position, pos_of[e]) == man_best]
            if min(fields[e].get(p.position, math.inf) for e in man_nearest) - d0 >= 2:
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
            d = sorted(fields[e].get(p.position, math.inf) for e in exit_ids)
            if math.isfinite(d[1]):
                extra.append(d[1] - d[0])
        split = q_max / (1 + statistics.median(extra)) if extra else None
    cut = _min_vertex_cut(scenario)
    severity = len(people) / cut if cut else math.inf
    route_length = proxies.route_length
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
        exits_flammable=scenario.fire.exits_flammable,
        max_steps=scenario.termination.max_steps,
        obstacle_density=proxies.obstacle_density,
        crowd_density=proxies.crowd_density,
        passage_width=proxies.passage_width,
        route_redundancy=proxies.route_redundancy,
        route_length=route_length,
        ignition_person_distance=proxies.ignition_person_distance,
        ignition_exit_distance=proxies.ignition_exit_distance,
        exit_separation=proxies.exit_separation,
        min_cut=cut,
        bottleneck_severity=severity,
        queue_ratio=severity / route_length if route_length else math.inf,
        max_nearest_exit_load=q_max,
        split_pressure=split,
        deceptive_share=deceptive / len(people) if people else 0.0,
        exit_gap_median=float(statistics.median(gaps)) if gaps else None,
        detour_ratio=statistics.fmean(detours) if detours else math.inf,
        ignition_centrality=centrality,
    )


# --------------------------------------------------------------------------- privileged world profile


def _static_plan_survives(scenario: Scenario, start: Pos, field_: Mapping[Pos, int], trajectory: Sequence[frozenset[Pos]]) -> bool:
    """Can one person, alone, follow SOME static shortest route to the given exit without burning?

    Same timing rules as the privileged bound, but every move must decrease
    the static distance by one (no waiting, no detour).
    """
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


@dataclass(frozen=True)
class WorldProfile:
    """Privileged measurements of one world (scenario + seed). Evaluator-side only."""

    scenario_id: str
    seed: int
    world_id: str
    fire_digest: str
    population: int
    escapable: int  # M*: people who can escape alone under the true fire
    min_person_time: int  # privileged relaxed lower bound, summed over people
    final_burn_share: float  # burning share of free cells at the horizon
    exits_lost: int  # exits burning at the horizon
    threatened_share: float  # people whose nearest exit burns before they can use it
    invalidation_share: float  # people whose every static shortest route is cut by fire
    recoverable_invalidation_share: float  # ... and who can still escape another way
    forced_delay_share: float  # escapable people who cannot escape as early as their static distance
    throughput_cap: int  # people the exits can pass before they burn (one per exit per step), an upper bound

    @property
    def bound_is_loose(self) -> bool:
        """True when exit throughput alone proves that ``M*`` people cannot all evacuate."""
        return self.throughput_cap < self.escapable


def profile_world(scenario: Scenario, seed: int) -> WorldProfile:
    trajectory = fire_trajectory(scenario, seed)
    bound = privileged_escape_bound(scenario, seed)
    earliest = dict(bound.earliest_evacuation)
    fields = exit_fields(scenario)
    exit_ids = sorted(fields)
    pos_of = {e.id: e.position for e in scenario.exits}
    burn_step: dict[str, int | None] = {}
    for e in exit_ids:
        burn_step[e] = next((t for t, b in enumerate(trajectory) if pos_of[e] in b), None)
    people = sorted(scenario.people, key=lambda p: p.id)
    nearest = {p.id: min((fields[e].get(p.position, math.inf), e) for e in exit_ids) for p in people}
    load = Counter(e for _, e in nearest.values())
    threatened = invalidated = recoverable = delayed = 0
    for p in people:
        d, e = nearest[p.id]
        b = burn_step[e]
        if b is not None and math.isfinite(d) and b <= max(d, load[e]):
            threatened += 1
        ok = math.isfinite(d) and _static_plan_survives(scenario, p.position, fields[e], trajectory)
        if not ok:
            invalidated += 1
            if earliest[p.id] is not None:
                recoverable += 1
        step = earliest[p.id]
        if step is not None and math.isfinite(d) and step > d:
            delayed += 1
    horizon = scenario.termination.max_steps
    throughput = 0
    for e in exit_ids:
        first = min((fields[e].get(p.position, math.inf) for p in people), default=math.inf)
        if math.isfinite(first):
            burnt = burn_step[e]
            last = horizon if burnt is None else min(horizon, burnt)
            throughput += max(0, last - int(first) + 1)
    free = sum(1 for r in range(scenario.height) for c in range(scenario.width) if Pos(r, c) not in scenario.walls)
    n = len(people)
    escapable = bound.max_evacuated
    return WorldProfile(
        scenario_id=scenario.scenario_id,
        seed=seed,
        world_id=world_id(scenario, seed),
        fire_digest=fire_trajectory_digest(scenario, seed),
        population=n,
        escapable=escapable,
        min_person_time=bound.min_person_time_to_safety_total,
        final_burn_share=len(trajectory[-1]) / free,
        exits_lost=sum(1 for e in exit_ids if burn_step[e] is not None),
        threatened_share=threatened / n,
        invalidation_share=invalidated / n,
        recoverable_invalidation_share=recoverable / n,
        forced_delay_share=delayed / escapable if escapable else 0.0,
        throughput_cap=throughput,
    )


# --------------------------------------------------------------------------- scoring (provisional)


def provisional_score(population: int, escapable: int, evacuated: int, casualties: int, stranded: int) -> float | None:
    """PROVISIONAL safety-only analysis score in [0, 100], or ``None`` if ``escapable == 0``.

    It reproduces only the Safety factor proposed in PR #32 (evacuation,
    casualty and stranding factors and the catastrophic gate) and omits every
    quality component. It exists only for this design study. It is not the
    EMBER Score.
    """
    if escapable == 0:
        return None
    unavoidable = population - escapable
    c = max(0, casualties - unavoidable)
    s = max(0, stranded - max(0, unavoidable - casualties))
    if (c + s) / escapable >= 0.5:
        return 0.0
    e = min(1.0, evacuated / escapable) ** 2
    pc = 0.6 ** math.log2(1 + c) * (1 - c / escapable) ** 4
    ps = 0.8 ** math.log2(1 + s) * (1 - s / escapable) ** 2
    return float(100.0 * e * pc * ps)


@dataclass(frozen=True)
class AgentOutcome:
    agent: str
    evacuated: int
    casualties: int
    stranded: int
    steps: int
    score: float | None
    seconds: float = 0.0


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


def run_outcome(scenario: Scenario, seed: int, agent_name: str, factory: AgentFactory, escapable: int) -> AgentOutcome:
    t0 = time.perf_counter()
    record = run_episode(scenario, agent_policy(factory()), seed=seed)
    seconds = time.perf_counter() - t0
    m = compute_episode_metrics(scenario, record)
    s = m.safety
    score = provisional_score(s.population, escapable, s.evacuated, s.casualties, s.stranded)
    return AgentOutcome(agent_name, s.evacuated, s.casualties, s.stranded, m.time.episode_steps, score, seconds)


def observation_chars(scenario: Scenario, seed: int) -> int:
    """Characters of the serialized step-0 observation: a token-count proxy for API cost."""
    return len(serialize_observation(Simulator(scenario, seed=seed).observe()))


# --------------------------------------------------------------------------- difficulty and families


def difficulty_band(panel_scores: Mapping[str, float]) -> str:
    """Easy / Moderate / Hard / Stress from the reference-panel scores of one world or scenario.

    - Easy: every panel member solves it (score >= SOLVED).
    - Moderate: the best member solves it, at least one does not.
    - Hard: no member solves it; the best keeps >= STRESS.
    - Stress: the best member scores below STRESS.
    """
    values = [panel_scores[a] for a in REFERENCE_PANEL]
    best, worst = max(values), min(values)
    if worst >= SOLVED:
        return "easy"
    if best >= SOLVED:
        return "moderate"
    if best >= STRESS:
        return "hard"
    return "stress"


BANDS: tuple[str, ...] = ("easy", "moderate", "hard", "stress")

# Proposed held-out allocation per primary family (see docs/evaluation-suite-design.md).
PROPOSED_BANDS_PER_FAMILY: tuple[tuple[str, int], ...] = (("easy", 2), ("moderate", 4), ("hard", 8), ("stress", 6))


def family_tags(profile: ScenarioProfile, worlds: Sequence[WorldProfile]) -> frozenset[ScenarioFamily]:
    """Every V1 family whose challenge a scenario measurably contains.

    Structural tags come from :class:`ScenarioProfile`. Fire-dependent tags
    (threatened exit, route invalidation) use the mean over the scenario's
    worlds. Thresholds are fixed here and published with the suite design.
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
    if profile.exit_count > 1 and mean([w.threatened_share for w in worlds]) >= 0.2:
        tags.add(F.THREATENED_NEAREST_EXIT)
    if mean([w.recoverable_invalidation_share for w in worlds]) >= 0.2:
        tags.add(F.ROUTE_INVALIDATION_REPLANNING)
    if not tags and profile.exit_count > 1:
        tags.add(F.MULTIPLE_EXITS_UNEQUAL_DISTANCE)
    return frozenset(tags)


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


def select_suite(
    candidates: Sequence[Candidate],
    quotas: Mapping[ScenarioFamily, Mapping[str, int]],
    selection_key: str,
) -> Selection:
    """Deterministic family x band stratified selection. Independent of candidate order.

    Families are filled scarcest first (fewest eligible candidates; ties by
    name). Each candidate is used once and gets the family it was selected
    for as its primary family. Inside a cell, candidates are taken in the
    order of a keyed hash of their content hash, so the evaluator can fix the
    key privately and nobody can steer which scenario is taken. A cell that
    cannot be filled is reported in ``shortfall``; it is never filled from
    another band silently.
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


def primary_family(tags: frozenset[ScenarioFamily], counts: Mapping[ScenarioFamily, int]) -> ScenarioFamily:
    """For the study only: the scarcest tag (by pool count, ties by name) becomes the primary family."""
    return min(tags, key=lambda f: (counts.get(f, 0), f.value))


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


def family_balanced(scores_by_family: Mapping[Any, Sequence[float]]) -> float:
    return statistics.fmean(statistics.fmean(v) for v in scores_by_family.values() if v)


def stratified_se(scores_by_family: Mapping[Any, Sequence[float]]) -> float:
    """CLT standard error of the family-balanced mean (strata = families)."""
    fams = [v for v in scores_by_family.values() if v]
    var = sum((statistics.variance(v) / len(v) if len(v) > 1 else 0.0) for v in fams)
    return math.sqrt(var) / len(fams)


def one_way_components(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """(between-group variance, within-group variance) from a balanced-enough one-way layout (ANOVA moments)."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return (math.nan, math.nan)
    within = statistics.fmean(statistics.variance(g) for g in groups)
    m = statistics.fmean(len(g) for g in groups)
    between_of_means = statistics.variance([statistics.fmean(g) for g in groups])
    return (max(0.0, between_of_means - within / m), within)


class DrawStream:
    """Deterministic integer draws for the resampling studies (SplitMix64 over a public label)."""

    def __init__(self, label: str) -> None:
        self._rng = SplitMix64(development_seed("draws/" + label))

    def below(self, n: int) -> int:
        return self._rng.next_u64() % n


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
    flammable: bool = True,
    probabilities: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_steps: int = 80,
) -> tuple[str, GeneratorConfig]:
    w, h = grid
    fire: dict[str, Any] = {
        "model": model,
        "spread_interval": {"min": interval[0], "max": interval[1]},
        "exits_flammable": flammable,
    }
    if model == "stochastic":
        fire["spread_probability_choices"] = list(probabilities)
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
        "max_attempts": 2000,
        "constraints": {
            "route_length": {"min": 1, "max": max_steps},
            "obstacle_density": {"min": 0.0, "max": 0.6},
            "crowd_density": {"min": 0.0, "max": 0.9},
            "exit_separation": {"min": 4, "max": w + h},
            "passage_width": {"min": 1, "max": max(w, h)},
            "route_redundancy": {"min": 1, "max": min(4, exits[1])},
            "ignition_person_distance": {"min": 3, "max": w * h},
            "ignition_exit_distance": {"min": 3, "max": w * h},
        },
    }

    return name, GeneratorConfig.from_json(json.dumps(data))


def development_recipes(scale: int) -> list[tuple[str, GeneratorConfig]]:
    """Public development recipes: each aims at some capability axes. Labels come from measurement, not from the recipe."""
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


@dataclass
class PoolScenario:
    recipe: str
    scenario: Scenario
    profile: ScenarioProfile
    worlds: list[WorldProfile] = field(default_factory=list)
    outcomes: dict[str, list[AgentOutcome]] = field(default_factory=dict)  # world_id -> outcomes
    tags: frozenset[ScenarioFamily] = frozenset()
    band: str = ""
    obs_chars: int = 0


def build_pool(scale: int, seeds_per_stochastic: int, candidate_seed_count: int, log: Callable[[str], None]) -> list[PoolScenario]:
    pool: list[PoolScenario] = []
    factories = {bid: baseline_factory(bid) for bid in BASELINE_IDS}
    for name, config in development_recipes(scale):
        t0 = time.perf_counter()
        generated = generate_suite(config, development_seed("generator/" + name), batch_id=f"dev-{name}")
        for s in generated.scenarios:
            prof = profile_scenario(s)
            item = PoolScenario(name, s, prof)
            candidates = [development_seed(f"evaluation/{name}/{s.scenario_id}/{i}") for i in range(candidate_seed_count)]
            seeds = select_distinct_seeds(s, candidates, seeds_per_stochastic)
            for seed in seeds:
                wp = profile_world(s, seed)
                item.worlds.append(wp)
                if wp.escapable == 0:
                    continue
                item.outcomes[wp.world_id] = [run_outcome(s, seed, bid, factories[bid], wp.escapable) for bid in BASELINE_IDS]
            item.obs_chars = observation_chars(s, seeds[0])
            pool.append(item)
        log(f"  recipe {name}: {len(generated.scenarios)} scenarios in {time.perf_counter() - t0:.1f}s")
    return pool


def _panel(outcomes: Sequence[AgentOutcome]) -> dict[str, float]:
    return {o.agent: o.score for o in outcomes if o.score is not None}


def finalize_pool(pool: list[PoolScenario]) -> None:
    for item in pool:
        item.tags = family_tags(item.profile, item.worlds)
        if item.outcomes:
            per_agent: dict[str, list[float]] = defaultdict(list)
            for outs in item.outcomes.values():
                for a, v in _panel(outs).items():
                    per_agent[a].append(v)
            item.band = difficulty_band({a: statistics.fmean(v) for a, v in per_agent.items()})


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


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in r) + " |")
    return "\n".join(out)


def _dist(values: Sequence[float]) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return "—"
    return f"{min(vals):.3g} / {statistics.median(vals):.3g} / {max(vals):.3g}"


# --------------------------------------------------------------------------- audit of the public suite


def audit_public(out: Path, seeds: int) -> str:
    suite = load_suite(PUBLIC_SUITE_DIR)
    factories = {bid: baseline_factory(bid) for bid in BASELINE_IDS}
    lines = [
        "# Public V1 suite audit",
        "",
        f"Analysis version {ANALYSIS_VERSION}; simulator {SIMULATOR_VERSION}; suite hash `{suite.manifest.suite_hash()}`.",
        f"Development seeds: {seeds} per scenario from the public development domain. Scores are PROVISIONAL (safety-only).",
        "",
        "## Structure",
        "",
    ]
    rows = []
    world_rows = []
    baseline_rows = []
    dev_schedule: list[tuple[str, int]] = []
    by_id = {s.scenario_id: s for s in suite.scenarios}
    for entry, s in zip(suite.manifest.entries, suite.scenarios, strict=True):
        p = profile_scenario(s)
        rows.append(
            [
                s.scenario_id, entry.primary_family.value, f"{p.width}x{p.height}", p.population, p.exit_count, p.ignition_count,
                p.fire_model, p.passage_width, p.route_redundancy, p.min_cut, p.bottleneck_severity, p.queue_ratio,
                p.deceptive_share, p.detour_ratio, p.ignition_centrality,
            ]
        )
        seed_list = [development_seed(f"public-audit/{s.scenario_id}/{i}") for i in range(seeds)]
        dev_schedule += [(s.scenario_id, x) for x in seed_list]
        worlds = {}
        for x in seed_list:
            wid = world_id(s, x)
            if wid not in worlds:
                worlds[wid] = x
        wps = [profile_world(s, x) for x in worlds.values()]
        world_rows.append(
            [
                s.scenario_id, len(seed_list), len(worlds), _dist([w.escapable / w.population for w in wps]),
                _dist([w.threatened_share for w in wps]), _dist([w.invalidation_share for w in wps]),
                _dist([w.recoverable_invalidation_share for w in wps]), _dist([w.final_burn_share for w in wps]),
                f"{sum(w.bound_is_loose for w in wps)}/{len(wps)}",
            ]
        )
        per_agent: dict[str, list[float]] = defaultdict(list)
        for wp in wps:
            for bid in BASELINE_IDS:
                o = run_outcome(s, wp.seed, bid, factories[bid], wp.escapable)
                if o.score is not None:
                    per_agent[bid].append(o.score)
        means = {a: statistics.fmean(v) for a, v in per_agent.items()}
        baseline_rows.append([s.scenario_id, *[means.get(b) for b in BASELINE_IDS], difficulty_band(means)])
    lines.append(
        _table(
            ["scenario", "primary family", "grid", "N", "exits", "ign.", "fire", "passage", "redund.", "min cut", "N/cut",
             "queue ratio", "deceptive", "detour", "centrality"],
            rows,
        )
    )
    dedup = dedupe_schedule(by_id, dev_schedule)
    lines += [
        "",
        "## Nominal episodes versus unique worlds",
        "",
        f"A schedule of {seeds} seeds per scenario has {len(dedup.episodes)} nominal episodes and "
        f"**{len(dedup.unique)} unique worlds** ({dedup.duplicate_count} duplicates).",
        "",
        _table(
            ["scenario", "seeds", "unique worlds", "escapable share min/med/max", "threatened", "plan invalidated",
             "recoverable invalidation", "final burn share", "M* above throughput cap"],
            world_rows,
        ),
        "",
        "## Reference panel (provisional score, mean over unique worlds)",
        "",
        _table(["scenario", *BASELINE_IDS, "band"], baseline_rows),
        "",
    ]
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "public-audit.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


# --------------------------------------------------------------------------- the study


@dataclass(frozen=True)
class StudyUnit:
    """One scenario of the pool as a sampling unit: its family and per-world agent scores."""

    family: ScenarioFamily
    stochastic: bool
    scores: Mapping[str, tuple[float, ...]]  # agent -> one score per unique world


def _units(pool: Sequence[PoolScenario]) -> list[StudyUnit]:
    counts: Counter[ScenarioFamily] = Counter(f for item in pool if item.outcomes for f in item.tags)
    units = []
    for item in pool:
        if not item.outcomes or not item.tags:
            continue
        per_agent: dict[str, list[float]] = defaultdict(list)
        for outs in item.outcomes.values():
            for o in outs:
                if o.score is not None:
                    per_agent[o.agent].append(o.score)
        units.append(StudyUnit(primary_family(item.tags, counts), item.profile.fire_model == "stochastic", {a: tuple(v) for a, v in per_agent.items()}))
    return units


def _ci_study(units: Sequence[StudyUnit], sizes: Sequence[int], draws: int, worlds_per_stochastic: int) -> tuple[str, dict[str, list[tuple[int, float]]]]:
    by_family: dict[ScenarioFamily, list[StudyUnit]] = defaultdict(list)
    for u in units:
        by_family[u.family].append(u)
    families = sorted((f for f, us in by_family.items() if len(us) >= 3), key=lambda f: f.value)
    agents = list(BASELINE_IDS)
    population_scores = {
        a: statistics.fmean(statistics.fmean(statistics.fmean(u.scores[a]) for u in by_family[f]) for f in families) for a in agents
    }
    order = sorted(agents, key=lambda a: population_scores[a])
    pairs = list(zip(order, order[1:]))
    stoch_share = statistics.fmean(1.0 if u.stochastic else 0.0 for f in families for u in by_family[f])
    rows = []
    curve: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for m in sizes:  # scenarios per family
        stream = DrawStream(f"ci/{m}")
        estimates: dict[str, list[float]] = defaultdict(list)
        se_pred: dict[str, list[float]] = defaultdict(list)
        reversals: Counter[tuple[str, str]] = Counter()
        exact = 0
        diffs: dict[tuple[str, str], list[float]] = defaultdict(list)
        for _ in range(draws):
            sample = {f: [by_family[f][stream.below(len(by_family[f]))] for _ in range(m)] for f in families}
            point: dict[str, float] = {}
            for a in agents:
                fam_scores = {f: [statistics.fmean(u.scores[a][:worlds_per_stochastic]) for u in us] for f, us in sample.items()}
                point[a] = family_balanced(fam_scores)
                estimates[a].append(point[a])
                se_pred[a].append(stratified_se(fam_scores))
            ranked = sorted(agents, key=lambda a: point[a])
            exact += ranked == order
            for lo, hi in pairs:
                if point[lo] >= point[hi]:
                    reversals[(lo, hi)] += 1
                diffs[(lo, hi)].append(point[hi] - point[lo])
        n_scen = m * len(families)
        n_worlds = round(n_scen * (1 + (worlds_per_stochastic - 1) * stoch_share))
        widths = {a: quantile7(estimates[a], 0.975) - quantile7(estimates[a], 0.025) for a in agents}
        for a in agents:
            curve[a].append((n_worlds, widths[a]))
        rows.append(
            [
                m, n_scen, n_worlds, max(widths.values()), statistics.fmean(widths.values()),
                statistics.fmean(2 * 1.96 * statistics.fmean(se_pred[a]) for a in agents),
                *[reversals[p] / draws for p in pairs], exact / draws,
            ]
        )
    header = [
        "scenarios / family", "scenarios", "unique worlds", "max 95% width", "mean 95% width", "mean CLT width",
        *[f"P(rev) {lo.split('-v1')[0]} / {hi.split('-v1')[0]} (gap {population_scores[hi] - population_scores[lo]:.1f})" for lo, hi in pairs],
        "P(exact ranking)",
    ]
    pop_table = _table(["agent", "pool family-balanced score"], [[a, population_scores[a]] for a in order])
    text = (
        f"Families with >= 3 pool scenarios: {len(families)} ({', '.join(f.value for f in families)}). "
        f"Stochastic share {stoch_share:.2f}; {worlds_per_stochastic} worlds per stochastic scenario. {draws} draws per size.\n\n"
        + pop_table + "\n\n" + _table(header, rows)
    )
    return text, curve


def _seed_study(pool: Sequence[PoolScenario]) -> str:
    rows = []
    for a in BASELINE_IDS:
        groups = []
        for item in pool:
            if item.profile.fire_model != "stochastic":
                continue
            vals = [o.score for outs in item.outcomes.values() for o in outs if o.agent == a and o.score is not None]
            if len(vals) >= 2:
                groups.append(vals)
        between, within = one_way_components(groups)
        ratio = within / between if between and between > 0 else math.inf
        rows.append([a, len(groups), between, within, ratio, math.sqrt(ratio) if math.isfinite(ratio) else math.inf])
    return _table(["agent", "stochastic scenarios", "between-scenario var", "within-scenario (seed) var", "within/between", "S* if a scenario costs one episode"], rows)


def _noisy_factory(base: AgentFactory, epsilon: float, trial: int) -> AgentFactory:
    def make() -> Agent:
        return NoisyAgent(base(), epsilon, trial)

    return make


def _trial_study(pool: Sequence[PoolScenario], trials: int, epsilon: float, limit: int) -> str:
    """Model repetition: a noisy congestion-aware surrogate, K trials on the SAME frozen worlds."""
    base = baseline_factory("congestion-aware-v1")
    per_world: list[list[float]] = []
    picked = [item for item in pool if item.outcomes][:: max(1, len([i for i in pool if i.outcomes]) // limit)][:limit]
    for item in picked:
        wp = next(w for w in item.worlds if w.world_id in item.outcomes)
        scores = []
        for k in range(trials):
            o = run_outcome(item.scenario, wp.seed, "noisy", _noisy_factory(base, epsilon, k), wp.escapable)
            if o.score is not None:
                scores.append(o.score)
        per_world.append(scores)
    between, within = one_way_components(per_world)
    rows = []
    for k in (1, 2, 3, 5, 8):
        inflation = 1 + (within / k) / between if between > 0 else math.inf
        rows.append([k, inflation, math.sqrt(inflation)])
    return (
        f"Surrogate: congestion-aware-v1 with epsilon = {epsilon} random actions, {trials} trials on each of {len(per_world)} worlds.\n"
        f"Between-world variance {between:.1f}; within-world (trial) variance {within:.1f}; ratio {within / between if between else math.inf:.3f}.\n\n"
        + _table(["K trials", "variance inflation vs K=inf", "CI width inflation"], rows)
    )


def _svg(curve: Mapping[str, Sequence[tuple[int, float]]]) -> str:
    """A small dependency-free log-x line chart of 95% sampling-interval width against unique worlds."""
    w, h, left, bottom, top, right = 640, 360, 60, 40, 20, 170
    xs = [x for pts in curve.values() for x, _ in pts]
    ys = [y for pts in curve.values() for _, y in pts]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    y1 = max(ys) * 1.1

    def px(x: float) -> float:
        return left + (math.log10(x) - x0) / (x1 - x0 or 1) * (w - left - right)

    def py(y: float) -> float:
        return h - bottom - y / y1 * (h - top - bottom)

    colors = ["#2a6fdb", "#e07b00", "#1a9e55", "#c43c5c", "#7d5bbe"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="11">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<line x1="{left}" y1="{h - bottom}" x2="{w - right}" y2="{h - bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h - bottom}" stroke="#444"/>',
        f'<text x="{(left + w - right) / 2}" y="{h - 8}" text-anchor="middle">unique worlds (log scale)</text>',
        f'<text x="14" y="{(top + h - bottom) / 2}" transform="rotate(-90 14 {(top + h - bottom) / 2})" text-anchor="middle">95% interval width (provisional points)</text>',
    ]
    for x in sorted(set(xs)):
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


def study(out: Path, quick: bool, log: Callable[[str], None]) -> str:
    scale = 12 if quick else 48
    seeds_per_stochastic = 6
    t0 = time.perf_counter()
    pool = build_pool(scale, seeds_per_stochastic, candidate_seed_count=24, log=log)
    finalize_pool(pool)
    build_seconds = time.perf_counter() - t0
    usable = [p for p in pool if p.outcomes]
    lines = [
        "# Development-pool calibration and sample-size study",
        "",
        f"Analysis version {ANALYSIS_VERSION}; simulator {SIMULATOR_VERSION}; generator recipes at scale {scale}.",
        "All scores are PROVISIONAL safety-only analysis scores, not the EMBER Score. Development seeds only.",
        "",
        "## Pool",
        "",
        f"{len(pool)} generated scenarios, {sum(len(p.worlds) for p in pool)} unique worlds, "
        f"{len(pool) - len(usable)} scenarios without any escapable world (excluded). Build + panel time {build_seconds:.0f} s.",
        "",
    ]
    # Recipes x bands and family availability
    band_rows = []
    for name, _ in development_recipes(scale):
        items = [p for p in usable if p.recipe == name]
        c = Counter(p.band for p in items)
        band_rows.append([name, len(items), *[c.get(b, 0) for b in BANDS]])
    lines += ["### Difficulty band by recipe (scenario level)", "", _table(["recipe", "scenarios", *BANDS], band_rows), ""]
    fam_rows = []
    for fam in ScenarioFamily:
        items = [p for p in usable if fam in p.tags]
        c = Counter(p.band for p in items)
        fam_rows.append([fam.value, len(items), sum(1 for p in items if p.profile.fire_model == "stochastic"), *[c.get(b, 0) for b in BANDS]])
    lines += ["### Family tag availability (a scenario can carry several tags)", "", _table(["family", "tagged", "stochastic", *BANDS], fam_rows), ""]
    # Proxy summary per band
    proxy_rows = []
    for b in BANDS:
        items = [p for p in usable if p.band == b]
        if not items:
            continue

        def mean_w(f: Callable[[WorldProfile], float], its: Sequence[PoolScenario] = items) -> float:
            return statistics.fmean(statistics.fmean(f(w) for w in p.worlds if w.escapable) for p in its)

        proxy_rows.append(
            [
                b, len(items), statistics.median(p.profile.population for p in items), statistics.median(p.profile.crowd_density for p in items),
                statistics.median(p.profile.bottleneck_severity for p in items), statistics.median(p.profile.obstacle_density for p in items),
                mean_w(lambda w: w.escapable / w.population), mean_w(lambda w: w.threatened_share), mean_w(lambda w: w.recoverable_invalidation_share),
                mean_w(lambda w: w.forced_delay_share),
                mean_w(lambda w: 1.0 if w.bound_is_loose else 0.0),
            ]
        )
    lines += [
        "### Measured properties by band (medians / means)",
        "",
        _table(["band", "n", "N", "crowd density", "N/cut", "obstacle density", "escapable share", "threatened", "recoverable invalidation", "forced delay", "M* above throughput cap"], proxy_rows),
        "",
    ]
    # Panel scores per band
    score_rows = []
    for b in BANDS:
        items = [p for p in usable if p.band == b]
        if not items:
            continue
        row: list[Any] = [b, len(items)]
        for a in BASELINE_IDS:
            row.append(statistics.fmean(statistics.fmean(o.score for outs in p.outcomes.values() for o in outs if o.agent == a and o.score is not None) for p in items))
        score_rows.append(row)
    lines += ["### Mean provisional score by band", "", _table(["band", "n", *BASELINE_IDS], score_rows), ""]
    # Discrimination: share of worlds where the panel spread is small
    spreads = []
    for p in usable:
        for outs in p.outcomes.values():
            vals = [v for a, v in _panel(outs).items() if a in REFERENCE_PANEL]
            spreads.append(max(vals) - min(vals))
    allsolved = sum(1 for p in usable for outs in p.outcomes.values() if min(_panel(outs)[a] for a in REFERENCE_PANEL) >= SOLVED)
    allfail = sum(1 for p in usable for outs in p.outcomes.values() if max(_panel(outs)[a] for a in REFERENCE_PANEL) < STRESS)
    total_worlds = sum(len(p.outcomes) for p in usable)
    lines += [
        f"Worlds where every panel member solves (>= {SOLVED:g}): {allsolved}/{total_worlds}. "
        f"Worlds where every panel member is below {STRESS:g}: {allfail}/{total_worlds}. "
        f"Median panel spread (max - min): {statistics.median(spreads):.1f}.",
        "",
    ]
    # Proxy-only prediction of band: rank correlation of proxies with best-panel gap
    def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
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

    gap: list[float] = []
    feats: dict[str, list[float]] = defaultdict(list)
    for p in usable:
        for wp in p.worlds:
            if wp.world_id not in p.outcomes:
                continue
            gap.append(100 - max(_panel(p.outcomes[wp.world_id])[a] for a in REFERENCE_PANEL))
            feats["population"].append(p.profile.population)
            feats["crowd density"].append(p.profile.crowd_density)
            feats["N / min cut"].append(min(p.profile.bottleneck_severity, 1e6))
            feats["obstacle density"].append(p.profile.obstacle_density)
            feats["route redundancy"].append(p.profile.route_redundancy)
            feats["ignition-person distance"].append(p.profile.ignition_person_distance or 1e6)
            feats["escapable share (privileged)"].append(wp.escapable / wp.population)
            feats["threatened share (privileged)"].append(wp.threatened_share)
            feats["recoverable invalidation (privileged)"].append(wp.recoverable_invalidation_share)
            feats["forced delay (privileged)"].append(wp.forced_delay_share)
    lines += [
        "### Do structural proxies predict the best-panel gap? (Spearman, per world)",
        "",
        _table(["proxy", "rho with (100 - best panel score)"], [[k, spearman(v, gap)] for k, v in feats.items()]),
        "",
    ]
    # CI study
    units = _units(pool)
    ci_text, curve = _ci_study(units, sizes=(1, 2, 4, 8, 12, 16, 20, 24, 32), draws=400 if not quick else 100, worlds_per_stochastic=3)
    lines += ["## Sample size, interval width and ranking stability", "", ci_text, ""]
    lines += ["## Seeds per stochastic scenario (variance components)", "", _seed_study(pool), ""]
    lines += ["## Model repetition (K trials)", "", _trial_study(pool, trials=8, epsilon=0.1, limit=40 if not quick else 10), ""]
    # Workload
    steps_c = [o.steps for p in usable for outs in p.outcomes.values() for o in outs if o.agent == "congestion-aware-v1"]
    max_steps = [p.scenario.termination.max_steps for p in usable for _ in p.outcomes]
    chars = [p.obs_chars for p in usable for _ in p.outcomes]
    lines += [
        "## Workload per world (one decision call per step)",
        "",
        _table(
            ["quantity", "min", "median", "mean", "max"],
            [
                ["steps (congestion-aware)", min(steps_c), statistics.median(steps_c), statistics.fmean(steps_c), max(steps_c)],
                ["max_steps (stalling model)", min(max_steps), statistics.median(max_steps), statistics.fmean(max_steps), max(max_steps)],
                ["step-0 observation chars", min(chars), statistics.median(chars), statistics.fmean(chars), max(chars)],
            ],
        ),
        "",
    ]
    # Selection feasibility for the proposed allocation
    quotas = {f: dict(PROPOSED_BANDS_PER_FAMILY) for f in ScenarioFamily}
    cands = [Candidate(p.scenario.scenario_id, p.profile.scenario_hash, p.tags, p.band, p.profile.fire_model == "stochastic") for p in usable]
    sel = select_suite(cands, quotas, selection_key="development-feasibility")
    lines += [
        "## Selection feasibility of the proposed allocation on this pool",
        "",
        f"Quota per family: {dict(PROPOSED_BANDS_PER_FAMILY)}. Selected {len(sel.chosen)} of "
        f"{sum(sum(q.values()) for q in quotas.values())}; shortfall {sum(n for *_, n in sel.shortfall)}.",
        "",
        _table(["family", "band", "missing"], [[f.value, b, n] for f, b, n in sel.shortfall]) if sel.shortfall else "No shortfall.",
        "",
    ]
    file_bytes = [len(json.dumps(p.scenario.to_dict(), indent=2)) + 1 for p in usable]
    secs = [o.seconds for p in usable for outs in p.outcomes.values() for o in outs if o.agent == "congestion-aware-v1"]
    lines += [
        "## Storage and evaluator runtime",
        "",
        _table(
            ["quantity", "min", "median", "mean", "max"],
            [
                ["scenario file bytes", min(file_bytes), statistics.median(file_bytes), statistics.fmean(file_bytes), max(file_bytes)],
                ["congestion-aware episode seconds", min(secs), statistics.median(secs), statistics.fmean(secs), max(secs)],
            ],
        ),
        "",
    ]
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    (out / "ci-width.svg").write_text(_svg(curve), encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=["audit-public", "study"])
    parser.add_argument("--out", type=Path, required=True, help="output directory (keep it outside the repository)")
    parser.add_argument("--quick", action="store_true", help="small pool, for a smoke run")
    parser.add_argument("--seeds", type=int, default=64, help="development seeds per public scenario (audit-public)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    if args.command == "audit-public":
        print(audit_public(args.out, args.seeds))
    else:
        print(study(args.out, args.quick, log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
