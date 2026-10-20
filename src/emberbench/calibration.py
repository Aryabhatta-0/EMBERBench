"""Evaluator-side calibration: the frozen panel, admission, difficulty bands, capacity filter and fire evidence.

This module is the library form of the PR #33 calibration rules
(docs/difficulty-calibration.md; ``tools/analyze_suite_design.py`` imports
them from here). It adds no rule and changes no threshold:

- calibration panel version 1.0: four public, deterministic V1 baselines;
- admission rule version 1.0, in order: not applicable, capacity artifact
  (mean capacity ceiling < 50), saturated, floor, no separation;
- difficulty rule version 1.0: bands on ``B``, the best panel mean official
  EMBER Score.

Every score is the official EMBER Score (``emberbench.scoring``); the
capacity ceiling is the official ``score_evidence`` on synthetic evidence.
There is no second score implementation. No model under evaluation is ever an
input: the only policies are the panel baselines.

Everything here is PRIVILEGED: it uses the evaluator seed and the future fire.
It runs in the evaluator's private workspace and never on a model's path.
"""

import heapq
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .agent import agent_policy
from .baselines import baseline_factory
from .families import FireEvidence
from .grid import Pos, neighbours
from .scenario import Scenario
from .scoring import (
    _UNREACHABLE,
    OFFICIAL_CONSTANTS,
    EpisodeEvidence,
    _References,
    _references,
    episode_evidence,
    score_evidence,
)
from .simulator import run_episode
from .structure import exit_fields
from .worlds import environment_world_digest

# The frozen calibration panel (calibration panel version 1.0). Four fixed,
# deterministic reference policies with different competences; no submitted
# model is ever a member. ``random-legal-v1`` is the floor reference: reported,
# never a band input. The privileged references (the savable count M* and the
# capacity ceiling) are bounds, never panel members.
CALIBRATION_PANEL_VERSION = "1.0"
CALIBRATION_PANEL: tuple[str, ...] = (
    "nearest-exit-greedy-v1",
    "static-shortest-path-v1",
    "fire-aware-shortest-path-v1",
    "congestion-aware-v1",
)
FLOOR_AGENT = "random-legal-v1"

# Difficulty cutpoints (official EMBER points) on B, the best panel mean score.
DIFFICULTY_RULES_VERSION = "1.0"
EASY_MIN_BEST = 80.0  # Easy: the best panel member keeps >= 80 points
MODERATE_MIN_BEST = 30.0  # Moderate: 30 <= B < 80
HARD_MIN_BEST = 10.0  # Hard: 10 <= B < 30; Stress: B < 10
BANDS: tuple[str, ...] = ("easy", "moderate", "hard", "stress")

# Admission rules, applied in EXCLUSIONS order.
ADMISSION_RULES_VERSION = "1.0"
SATURATED = 95.0  # every panel member >= 95: no information, never selected
FLOOR = 1.0  # every panel member < 1 point: no information, never selected
CEILING_MIN = 50.0  # mean capacity ceiling below this: practically unsolvable, never selected
ADMITTED = "admitted"
EXCLUSIONS: tuple[str, ...] = ("not_applicable", "capacity_artifact", "saturated", "floor", "no_separation")


def calibration_rules() -> dict[str, Any]:
    """The canonical description of the panel, band and admission rules (for freeze records)."""
    return {
        "calibration_panel_version": CALIBRATION_PANEL_VERSION,
        "calibration_panel": list(CALIBRATION_PANEL),
        "floor_reference": FLOOR_AGENT,
        "difficulty_rules_version": DIFFICULTY_RULES_VERSION,
        "bands": {"easy_min_best": EASY_MIN_BEST, "moderate_min_best": MODERATE_MIN_BEST, "hard_min_best": HARD_MIN_BEST},
        "admission_rules_version": ADMISSION_RULES_VERSION,
        "admission": {
            "order": list(EXCLUSIONS),
            "capacity_ceiling_min": CEILING_MIN,
            "saturated_min": SATURATED,
            "floor_max_exclusive": FLOOR,
            "separation_min": float(OFFICIAL_CONSTANTS.practical_difference),
        },
    }


# --------------------------------------------------------------------------- capacity filter (M_cap)


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


def capacity_savable(scenario: Scenario, seed: int, *, refs: _References | None = None) -> tuple[int, int]:
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
    ``refs`` may pass the official references when they are already computed.
    """
    refs = _references(scenario, seed) if refs is None else refs
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


# --------------------------------------------------------------------------- fire evidence


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


def savable_ids(scenario: Scenario, refs: _References) -> frozenset[str]:
    """IDs of the people the official ``M*`` counts (they can escape alone). PRIVILEGED."""
    first = refs.earliest[0]
    return frozenset(p.id for p in scenario.people if first[refs.index[p.position]] != _UNREACHABLE)


def fire_evidence(
    scenario: Scenario, seed: int, savable: frozenset[str] | None = None, *, refs: _References | None = None
) -> FireEvidence:
    """The fire-dependent family evidence of the world ``(scenario, seed)``. PRIVILEGED.

    ``savable`` defaults to the official ``M*`` set. For a deterministic-fire
    scenario every seed gives the same value: the scenario has one trajectory.
    """
    refs = _references(scenario, seed) if refs is None else refs
    if savable is None:
        savable = savable_ids(scenario, refs)
    trajectory = refs.trajectory
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
            if p.id in savable:
                recoverable += 1
    free = sum(1 for r in range(scenario.height) for c in range(scenario.width) if Pos(r, c) not in scenario.walls)
    n = len(people)
    return FireEvidence(threatened / n, invalidated / n, recoverable / n, len(trajectory[-1]) / free)


# --------------------------------------------------------------------------- panel runs and world measurements


@dataclass(frozen=True)
class WorldMeasurement:
    """Everything calibration and fire tagging need about one unique world. PRIVILEGED; evaluator-side only."""

    seed: int
    world: str  # environment_world_digest
    savable: int  # M*, the official max_evacuated
    capacity_savable: int  # M_cap
    capacity_ceiling: float | None
    fire: FireEvidence
    panel_scores: tuple[tuple[str, float], ...]  # (panel member, official EMBER Score), panel order

    def score(self, agent: str) -> float:
        return next(s for a, s in self.panel_scores if a == agent)

    def panel(self) -> dict[str, float]:
        return dict(self.panel_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_world_digest": self.world,
            "savable": self.savable,
            "capacity_savable": self.capacity_savable,
            "capacity_ceiling": self.capacity_ceiling,
            "fire": self.fire.to_dict(),
            "panel_scores": dict(self.panel_scores),
        }


def measure_world(scenario: Scenario, seed: int, *, panel: Sequence[str] = CALIBRATION_PANEL) -> WorldMeasurement:
    """Run the panel on one world and measure ``M*``, ``M_cap``, the capacity ceiling and the fire evidence.

    The official references are computed once and shared by every panel run's
    evidence, the capacity bound and the fire evidence, exactly as
    ``score_results`` shares them; each score equals ``score_episode``.
    """
    refs = _references(scenario, seed)
    scores: list[tuple[str, float]] = []
    for agent in panel:
        record = run_episode(scenario, agent_policy(baseline_factory(agent)()), seed=seed)
        scores.append((agent, float(score_evidence(0, episode_evidence(scenario, record, _refs=refs)).score)))
    m_star, m_cap = capacity_savable(scenario, seed, refs=refs)
    return WorldMeasurement(
        seed=seed,
        world=environment_world_digest(scenario, seed),
        savable=m_star,
        capacity_savable=m_cap,
        capacity_ceiling=capacity_ceiling(len(scenario.people), m_star, m_cap),
        fire=fire_evidence(scenario, seed, refs=refs),
        panel_scores=tuple(scores),
    )


# --------------------------------------------------------------------------- admission and bands


class CalibrationWorld(Protocol):
    """What :func:`calibrate` reads from one unique world."""

    @property
    def savable(self) -> int: ...
    @property
    def capacity_savable(self) -> int: ...
    @property
    def capacity_ceiling(self) -> float | None: ...
    def score(self, agent: str) -> float: ...


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_means": dict(self.panel),
            "best": self.best,
            "worst": self.worst,
            "spread": self.spread,
            "capacity_ceiling": self.ceiling,
            "capacity_limited": self.capacity_limited,
            "admission": self.admission,
            "band": self.band,
        }


def difficulty_band(best: float) -> str:
    """Easy / Moderate / Hard / Stress from ``B``, the best calibration-panel score (official points)."""
    if best >= EASY_MIN_BEST:
        return "easy"
    if best >= MODERATE_MIN_BEST:
        return "moderate"
    if best >= HARD_MIN_BEST:
        return "hard"
    return "stress"


def calibrate(worlds: Sequence[CalibrationWorld]) -> Calibration:
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
