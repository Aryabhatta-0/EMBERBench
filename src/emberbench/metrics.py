"""Auditable per-episode metrics computed from a ``Scenario`` and its ``RunRecord``.

The entry point is :func:`compute_episode_metrics`. It is a pure function: it
reads the scenario and the record, never mutates them, and returns an
immutable :class:`EpisodeMetrics`. The full definitions, units, denominators
and edge cases are in ``docs/metrics.md``.

Rules for this module:

- **Raw metrics are canonical.** Every field is a count, a step number, a
  distance or a ratio of named counts. There is no composite score.
- **Fail closed.** A record that does not belong to the scenario, comes from
  another simulator, observation or schema version, breaks a record
  invariant, or fails simulator replay raises :class:`MetricsInputError`.
  Nothing is scored.
- **Exact arithmetic.** Ratios are computed with :class:`fractions.Fraction`
  and converted to ``float`` once, so each value is the correctly rounded
  float of an exact rational. Results do not depend on platform or
  summation order.
- **Proxies are named as proxies.** The fire-exposure and replanning fields
  describe observable events only. They do not claim to measure model intent.

Validation runs in three stages, all mandatory:

1. compatibility: versions, ``scenario_id`` and ``scenario_hash``;
2. structural checks: precise diagnostics for each record invariant that
   the metrics rely on;
3. authentication: :func:`emberbench.verify_record` replays the action log
   and requires a byte-identical record. This rejects records that are
   internally consistent but that the simulator could not have produced.
"""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from . import SIMULATOR_VERSION
from .errors import MetricsInputError, ReplayMismatchError
from .events import CasualtyCause, RunRecord
from .grid import Action, Pos, in_bounds, neighbours
from .movement import MoveOutcome
from .observation import OBSERVATION_VERSION
from .replay import verify_record
from .scenario import SUPPORTED_SCHEMA_VERSIONS, Scenario
from .simulator import compute_trapped
from .state import PersonState, PersonStatus, SimState, TerminationReason

# Bump this whenever the definition of any metric field changes.
METRICS_VERSION = "1.0"

BLOCKED_OUTCOMES: tuple[MoveOutcome, ...] = (
    MoveOutcome.BLOCKED_BOUNDARY,
    MoveOutcome.BLOCKED_WALL,
    MoveOutcome.BLOCKED_FIRE,
    MoveOutcome.BLOCKED_OCCUPIED,
    MoveOutcome.BLOCKED_CONFLICT,
    MoveOutcome.BLOCKED_CYCLE,
)
# Blocks caused by other people, not by the static map or the fire.
CONGESTION_OUTCOMES: frozenset[MoveOutcome] = frozenset(
    {MoveOutcome.BLOCKED_OCCUPIED, MoveOutcome.BLOCKED_CONFLICT, MoveOutcome.BLOCKED_CYCLE}
)


def _ratio(numerator: int | Fraction, denominator: int | Fraction) -> float | None:
    """``numerator / denominator`` as a correctly rounded float, or ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return float(Fraction(numerator) / Fraction(denominator))


def _pairs_to_dict(pairs: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    return {str(k): v for k, v in pairs}


# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class SafetyMetrics:
    """Terminal outcome of every person. ``evacuated + casualties + stranded == population``."""

    population: int
    evacuated: int
    casualties: int
    stranded: int  # still active when the episode terminated
    evacuated_rate: float
    casualty_rate: float
    stranded_rate: float
    terminal_trapped_count: int  # stranded people with no wall- and fire-free path to a usable exit
    terminal_trapped_rate: float
    casualties_by_cause: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "evacuated": self.evacuated,
            "casualties": self.casualties,
            "stranded": self.stranded,
            "evacuated_rate": self.evacuated_rate,
            "casualty_rate": self.casualty_rate,
            "stranded_rate": self.stranded_rate,
            "terminal_trapped_count": self.terminal_trapped_count,
            "terminal_trapped_rate": self.terminal_trapped_rate,
            "casualties_by_cause": _pairs_to_dict(self.casualties_by_cause),
        }


@dataclass(frozen=True)
class TimeMetrics:
    """Timing in simulator steps. Evacuee-only fields always come with ``evacuees`` and ``non_evacuees``."""

    episode_steps: int
    max_steps: int  # the scenario horizon; also the censoring value for person-time-to-safety
    evacuees: int
    non_evacuees: int
    evacuation_steps: tuple[tuple[str, int], ...]  # (person_id, step), sorted by person ID
    first_evacuation_step: int | None
    last_evacuation_step: int | None
    mean_evacuation_step: float | None  # evacuees only
    median_evacuation_step: float | None  # evacuees only
    person_time_to_safety_total: int  # person-steps; non-evacuees count as max_steps
    mean_person_time_to_safety: float  # person_time_to_safety_total / population

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_steps": self.episode_steps,
            "max_steps": self.max_steps,
            "evacuees": self.evacuees,
            "non_evacuees": self.non_evacuees,
            "evacuation_steps": _pairs_to_dict(self.evacuation_steps),
            "first_evacuation_step": self.first_evacuation_step,
            "last_evacuation_step": self.last_evacuation_step,
            "mean_evacuation_step": self.mean_evacuation_step,
            "median_evacuation_step": self.median_evacuation_step,
            "person_time_to_safety_total": self.person_time_to_safety_total,
            "mean_person_time_to_safety": self.mean_person_time_to_safety,
        }


@dataclass(frozen=True)
class MovementMetrics:
    """Counts over decisions: one decision per active person per step."""

    decisions: int
    stay_actions: int
    movement_attempts: int  # decisions whose action is not STAY
    successful_moves: int
    blocked_moves: int
    blocked_by_outcome: tuple[tuple[str, int], ...]  # every blocked outcome, in MoveOutcome order
    congestion_blocks: int  # blocked_occupied + blocked_conflict + blocked_cycle
    stay_rate: float | None  # stay_actions / decisions
    blocked_rate: float | None  # blocked_moves / movement_attempts
    congestion_block_rate: float | None  # congestion_blocks / movement_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "stay_actions": self.stay_actions,
            "movement_attempts": self.movement_attempts,
            "successful_moves": self.successful_moves,
            "blocked_moves": self.blocked_moves,
            "blocked_by_outcome": _pairs_to_dict(self.blocked_by_outcome),
            "congestion_blocks": self.congestion_blocks,
            "stay_rate": self.stay_rate,
            "blocked_rate": self.blocked_rate,
            "congestion_block_rate": self.congestion_block_rate,
        }


@dataclass(frozen=True)
class PathEfficiencyMetrics:
    """Evacuee path length against the static shortest path. This is not a dynamic optimum."""

    evacuees_measured: int
    total_cells_moved: int
    total_static_lower_bound: int
    aggregate_path_efficiency: float | None  # total_static_lower_bound / total_cells_moved
    mean_path_efficiency: float | None  # mean of per-evacuee lower_bound / cells_moved
    mean_detour_ratio: float | None  # mean of per-evacuee cells_moved / lower_bound

    def to_dict(self) -> dict[str, Any]:
        return {
            "evacuees_measured": self.evacuees_measured,
            "total_cells_moved": self.total_cells_moved,
            "total_static_lower_bound": self.total_static_lower_bound,
            "aggregate_path_efficiency": self.aggregate_path_efficiency,
            "mean_path_efficiency": self.mean_path_efficiency,
            "mean_detour_ratio": self.mean_detour_ratio,
        }


@dataclass(frozen=True)
class FireExposureProxy:
    """A proximity proxy over active person-states. ``exposure = sum(1 / d)``; see ``docs/metrics.md``."""

    active_person_states: int
    person_states_with_fire: int
    person_states_without_fire: int
    distance_histogram: tuple[tuple[int, int], ...]  # (Manhattan distance d, person-states), sorted by d
    adjacent_person_states: int  # person-states with d == 1
    min_fire_distance: int | None
    exposure_total: float
    mean_exposure: float  # exposure_total / active_person_states

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_person_states": self.active_person_states,
            "person_states_with_fire": self.person_states_with_fire,
            "person_states_without_fire": self.person_states_without_fire,
            "distance_histogram": _pairs_to_dict(self.distance_histogram),
            "adjacent_person_states": self.adjacent_person_states,
            "min_fire_distance": self.min_fire_distance,
            "exposure_total": self.exposure_total,
            "mean_exposure": self.mean_exposure,
        }


@dataclass(frozen=True)
class ExitUtilization:
    """Use of each exit. ``exit_balance`` is descriptive only: equal use is not always optimal."""

    exit_count: int
    exits_used: int
    evacuations_per_exit: tuple[tuple[str, int], ...]  # every scenario exit, sorted by exit ID
    exit_shares: tuple[tuple[str, float], ...] | None  # None when nobody evacuated
    max_exit_share: float | None
    exit_balance: float | None  # normalized Gini-Simpson index; None for one exit or no evacuations

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_count": self.exit_count,
            "exits_used": self.exits_used,
            "evacuations_per_exit": _pairs_to_dict(self.evacuations_per_exit),
            "exit_shares": None if self.exit_shares is None else _pairs_to_dict(self.exit_shares),
            "max_exit_share": self.max_exit_share,
            "exit_balance": self.exit_balance,
        }


@dataclass(frozen=True)
class ReplanningProxies:
    """Observable reactions to a blocked move. Proxies only: they do not observe model intent.

    A *blocked event* is one decision with a ``blocked_*`` outcome. Its *next
    decision* is the same person's decision at the next step, which exists
    only when the person is still active and the episode continues.
    """

    blocked_events: int
    blocks_with_next_decision: int
    blocks_without_next_decision: int
    moved_after_block: int  # recovery proxy: the next decision moved
    blocked_after_block: int  # repeated-block proxy: the next decision was blocked again
    stayed_after_block: int
    action_changed_after_block: int  # the next action differs from the blocked action
    max_consecutive_blocks: int  # longest run of blocked decisions by one person
    recovery_proxy_rate: float | None  # moved_after_block / blocks_with_next_decision
    repeated_block_proxy_rate: float | None  # blocked_after_block / blocks_with_next_decision
    action_change_proxy_rate: float | None  # action_changed_after_block / blocks_with_next_decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_events": self.blocked_events,
            "blocks_with_next_decision": self.blocks_with_next_decision,
            "blocks_without_next_decision": self.blocks_without_next_decision,
            "moved_after_block": self.moved_after_block,
            "blocked_after_block": self.blocked_after_block,
            "stayed_after_block": self.stayed_after_block,
            "action_changed_after_block": self.action_changed_after_block,
            "max_consecutive_blocks": self.max_consecutive_blocks,
            "recovery_proxy_rate": self.recovery_proxy_rate,
            "repeated_block_proxy_rate": self.repeated_block_proxy_rate,
            "action_change_proxy_rate": self.action_change_proxy_rate,
        }


@dataclass(frozen=True)
class PersonMetrics:
    """Raw per-person values. The episode sections aggregate these."""

    person_id: str
    status: PersonStatus  # terminal status
    initial_position: Pos
    final_position: Pos
    resolution_step: int | None  # evacuation or casualty step; None if stranded
    exit_id: str | None
    casualty_cause: CasualtyCause | None
    decisions: int
    stay_actions: int
    cells_moved: int
    blocked_moves: int
    max_consecutive_blocks: int
    time_to_safety: int  # evacuation step, or max_steps if not evacuated
    static_exit_distance: int | None  # None if no exit is reachable around the walls
    path_efficiency: float | None  # evacuees only
    detour_ratio: float | None  # evacuees only
    min_fire_distance: int | None
    fire_exposure: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "status": self.status.value,
            "initial_position": self.initial_position.to_list(),
            "final_position": self.final_position.to_list(),
            "resolution_step": self.resolution_step,
            "exit_id": self.exit_id,
            "casualty_cause": None if self.casualty_cause is None else self.casualty_cause.value,
            "decisions": self.decisions,
            "stay_actions": self.stay_actions,
            "cells_moved": self.cells_moved,
            "blocked_moves": self.blocked_moves,
            "max_consecutive_blocks": self.max_consecutive_blocks,
            "time_to_safety": self.time_to_safety,
            "static_exit_distance": self.static_exit_distance,
            "path_efficiency": self.path_efficiency,
            "detour_ratio": self.detour_ratio,
            "min_fire_distance": self.min_fire_distance,
            "fire_exposure": self.fire_exposure,
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    """All raw metrics for one episode. There is deliberately no single score."""

    metrics_version: str
    simulator_version: str
    observation_version: str
    schema_version: str
    scenario_id: str
    scenario_hash: str
    seed: int
    termination_reason: TerminationReason
    safety: SafetyMetrics
    time: TimeMetrics
    movement: MovementMetrics
    path_efficiency: PathEfficiencyMetrics
    fire_exposure: FireExposureProxy
    exits: ExitUtilization
    replanning: ReplanningProxies
    people: tuple[PersonMetrics, ...]  # sorted by person ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_version": self.metrics_version,
            "simulator_version": self.simulator_version,
            "observation_version": self.observation_version,
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "termination_reason": self.termination_reason.value,
            "safety": self.safety.to_dict(),
            "time": self.time.to_dict(),
            "movement": self.movement.to_dict(),
            "path_efficiency": self.path_efficiency.to_dict(),
            "fire_exposure": self.fire_exposure.to_dict(),
            "exits": self.exits.to_dict(),
            "replanning": self.replanning.to_dict(),
            "people": [p.to_dict() for p in self.people],
        }


# --------------------------------------------------------------------------- input checks


def _fail(message: str) -> MetricsInputError:
    return MetricsInputError(message)


def _check_compatibility(scenario: Scenario, record: RunRecord) -> None:
    if record.simulator_version != SIMULATOR_VERSION:
        raise _fail(f"record made by simulator {record.simulator_version!r}, metrics support {SIMULATOR_VERSION!r}")
    if record.observation_version != OBSERVATION_VERSION:
        raise _fail(
            f"record made under observation version {record.observation_version!r}, metrics support {OBSERVATION_VERSION!r}"
        )
    if record.schema_version not in SUPPORTED_SCHEMA_VERSIONS or record.schema_version != scenario.schema_version:
        raise _fail(f"record schema version {record.schema_version!r} does not match scenario {scenario.schema_version!r}")
    if record.scenario_id != scenario.scenario_id:
        raise _fail(f"record scenario_id {record.scenario_id!r} does not match scenario {scenario.scenario_id!r}")
    if record.scenario_hash != scenario.content_hash():
        raise _fail("scenario hash does not match the record")


def _check_initial_state(scenario: Scenario, state: SimState) -> None:
    expected_people = tuple(PersonState(p.id, p.position, PersonStatus.ACTIVE) for p in scenario.people)
    if state.step != 0:
        raise _fail(f"initial state has step {state.step}, expected 0")
    if state.people != expected_people:
        raise _fail("initial people do not match the scenario")
    if state.burning != frozenset(scenario.ignitions):
        raise _fail("initial burning cells do not match the scenario ignitions")
    if state.terminated or state.termination_reason is not None:
        raise _fail("initial state is marked terminated")
    if state.trapped != compute_trapped(scenario, state.people, state.burning):
        raise _fail("initial trapped set is inconsistent")


def _check_record(scenario: Scenario, record: RunRecord) -> None:
    """Structural invariants that every simulator-produced record satisfies."""
    _check_initial_state(scenario, record.initial_state)
    if not record.steps:
        raise _fail("record has no steps; the episode never terminated")
    exit_ids = scenario.exit_positions
    walls = scenario.walls
    max_steps = scenario.termination.max_steps
    before = record.initial_state
    for number, step in enumerate(record.steps, start=1):
        where = f"step {number}"
        after = step.state
        if step.step != number or after.step != number:
            raise _fail(f"{where}: step numbers are not consecutive")
        if before.terminated:
            raise _fail(f"{where}: record continues past termination")
        active = {p.id: p for p in before.active}

        # Actions and movements cover exactly the active people.
        action_ids = [pid for pid, _ in step.actions]
        if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(active):
            raise _fail(f"{where}: actions do not cover exactly the active people")
        actions = dict(step.actions)
        moves = {m.person_id: m for m in step.movements}
        if len(moves) != len(step.movements) or set(moves) != set(active):
            raise _fail(f"{where}: movements do not cover exactly the active people")
        for pid, m in moves.items():
            if m.action is not actions[pid] or m.origin != active[pid].position:
                raise _fail(f"{where}: movement of {pid} does not match its action or position")
            if m.target != m.origin.offset(*m.action.delta):
                raise _fail(f"{where}: movement target of {pid} does not match its action")
            if (m.outcome is MoveOutcome.STAYED) != (m.action is Action.STAY):
                raise _fail(f"{where}: outcome of {pid} is inconsistent with its action")
            if m.final != (m.target if m.outcome is MoveOutcome.MOVED else m.origin):
                raise _fail(f"{where}: final cell of {pid} is inconsistent with its outcome")
            if m.outcome is MoveOutcome.MOVED and (not in_bounds(m.final, scenario.width, scenario.height) or m.final in walls):
                raise _fail(f"{where}: {pid} moved into an invalid cell")
        finals = [m.final for m in moves.values()]
        if len(finals) != len(set(finals)):
            raise _fail(f"{where}: two people share a cell")

        # Fire grows monotonically.
        ignited = frozenset(step.ignited)
        if len(ignited) != len(step.ignited) or ignited & before.burning or ignited & walls:
            raise _fail(f"{where}: invalid ignitions")
        if after.burning != before.burning | ignited:
            raise _fail(f"{where}: burning cells do not equal previous burning plus ignitions")

        # Evacuations and casualties match the status transitions.
        evacuated = {e.person_id: e for e in step.evacuations}
        killed = {c.person_id: c for c in step.casualties}
        if len(evacuated) != len(step.evacuations) or len(killed) != len(step.casualties):
            raise _fail(f"{where}: duplicate evacuation or casualty events")
        if set(evacuated) & set(killed) or not (set(evacuated) | set(killed)) <= set(active):
            raise _fail(f"{where}: evacuation and casualty events must name distinct active people")
        for e in evacuated.values():
            if e.position != moves[e.person_id].final or exit_ids.get(e.position) != e.exit_id or e.position in before.burning:
                raise _fail(f"{where}: invalid evacuation of {e.person_id}")
        for c in killed.values():
            expected_cause = CasualtyCause.ENTERED_FIRE if c.position in before.burning else CasualtyCause.FIRE_SPREAD
            if c.position != moves[c.person_id].final or c.position not in after.burning or c.cause is not expected_cause:
                raise _fail(f"{where}: invalid casualty of {c.person_id}")
        if [p.id for p in after.people] != [p.id for p in before.people]:
            raise _fail(f"{where}: the set of people changed")
        for p in after.people:
            prior = before.person(p.id)
            if prior.status is not PersonStatus.ACTIVE:
                if p != prior:
                    raise _fail(f"{where}: resolved person {p.id} changed")
                continue
            if p.id in evacuated:
                expected_status = PersonStatus.EVACUATED
            elif p.id in killed:
                expected_status = PersonStatus.CASUALTY
            else:
                expected_status = PersonStatus.ACTIVE
            if p.status is not expected_status or p.position != moves[p.id].final:
                raise _fail(f"{where}: state of {p.id} does not match its events")
            if expected_status is PersonStatus.ACTIVE and (
                p.position in after.burning or (p.position in exit_ids and p.position not in before.burning)
            ):
                raise _fail(f"{where}: {p.id} is active on a burning cell or a usable exit")

        if after.trapped != compute_trapped(scenario, after.people, after.burning):
            raise _fail(f"{where}: trapped set is inconsistent")

        # Termination: all_resolved wins over max_steps.
        expected_reason: TerminationReason | None = None
        if not after.active:
            expected_reason = TerminationReason.ALL_RESOLVED
        elif number >= max_steps:
            expected_reason = TerminationReason.MAX_STEPS
        if after.termination_reason != expected_reason or after.terminated != (expected_reason is not None):
            raise _fail(f"{where}: termination flags are inconsistent")
        before = after
    if not before.terminated:
        raise _fail("record does not reach termination; metrics need a complete episode")


def _authenticate(scenario: Scenario, record: RunRecord) -> None:
    """Replay the action log and require the exact recorded trajectory."""
    try:
        verify_record(scenario, record)
    except ReplayMismatchError as exc:
        raise _fail(f"record failed simulator replay verification: {exc}") from exc


# --------------------------------------------------------------------------- geometry


def static_exit_distances(scenario: Scenario) -> dict[Pos, int]:
    """Shortest 4-neighbour path length from each cell to the nearest exit, around walls only.

    People and fire (including the initial ignitions) are ignored, so this
    is a static lower bound, not a dynamically feasible or optimal path.
    Cells that cannot reach any exit are absent from the result.
    """
    walls = scenario.walls
    exits = sorted(scenario.exit_positions)
    dist = {p: 0 for p in exits}
    frontier = deque(exits)
    while frontier:
        cell = frontier.popleft()
        for nb in neighbours(cell, scenario.width, scenario.height):
            if nb not in dist and nb not in walls:
                dist[nb] = dist[cell] + 1
                frontier.append(nb)
    return dist


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _nearest_fire(pos: Pos, burning: frozenset[Pos]) -> int | None:
    return min((_manhattan(pos, f) for f in burning), default=None)


def _median(values: Sequence[int]) -> Fraction:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return Fraction(ordered[mid])
    return Fraction(ordered[mid - 1] + ordered[mid], 2)


def _longest_run(flags: Sequence[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------------- computation


def compute_episode_metrics(scenario: Scenario, record: RunRecord) -> EpisodeMetrics:
    """Compute every raw metric for one complete episode.

    Raises :class:`MetricsInputError` if the record does not belong to
    ``scenario``, comes from an incompatible version, is malformed, does not
    reach termination, or is not reproduced exactly by a simulator replay.
    Neither argument is mutated.
    """
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(record, RunRecord):
        raise TypeError("record must be a RunRecord")
    _check_compatibility(scenario, record)
    _check_record(scenario, record)
    _authenticate(scenario, record)

    final = record.final_state
    termination_reason = final.termination_reason
    assert termination_reason is not None  # guaranteed by _check_record
    population = len(scenario.people)
    max_steps = scenario.termination.max_steps
    person_ids = [p.id for p in scenario.people]
    initial_positions = {p.id: p.position for p in scenario.people}

    # Per-person event history.
    resolution_step: dict[str, int] = {}
    exit_of: dict[str, str] = {}
    cause_of: dict[str, CasualtyCause] = {}
    history: dict[str, list[tuple[Action, MoveOutcome]]] = {pid: [] for pid in person_ids}
    for step in record.steps:
        for m in step.movements:
            history[m.person_id].append((m.action, m.outcome))
        for e in step.evacuations:
            resolution_step[e.person_id] = step.step
            exit_of[e.person_id] = e.exit_id
        for c in step.casualties:
            resolution_step[c.person_id] = step.step
            cause_of[c.person_id] = c.cause

    # Fire-exposure proxy over every recorded snapshot s_0 .. s_T.
    distance_counts: dict[int, int] = {}
    no_fire_states = 0
    person_exposure: dict[str, Fraction] = {pid: Fraction(0) for pid in person_ids}
    person_min_distance: dict[str, int | None] = {pid: None for pid in person_ids}
    for snapshot in (record.initial_state, *(s.state for s in record.steps)):
        for p in snapshot.active:
            d = _nearest_fire(p.position, snapshot.burning)
            if d is None:
                no_fire_states += 1
                continue
            distance_counts[d] = distance_counts.get(d, 0) + 1
            person_exposure[p.id] += Fraction(1, d)
            current = person_min_distance[p.id]
            person_min_distance[p.id] = d if current is None else min(current, d)

    static = static_exit_distances(scenario)

    people: list[PersonMetrics] = []
    for pid in person_ids:
        state = final.person(pid)
        events = history[pid]
        outcomes = [o for _, o in events]
        moved = outcomes.count(MoveOutcome.MOVED)
        lower_bound = static.get(initial_positions[pid])
        is_evacuee = state.status is PersonStatus.EVACUATED
        if is_evacuee and (lower_bound is None or moved < lower_bound):
            raise _fail(f"{pid} evacuated in {moved} moves, below the static lower bound {lower_bound}")
        people.append(
            PersonMetrics(
                person_id=pid,
                status=state.status,
                initial_position=initial_positions[pid],
                final_position=state.position,
                resolution_step=resolution_step.get(pid),
                exit_id=exit_of.get(pid),
                casualty_cause=cause_of.get(pid),
                decisions=len(events),
                stay_actions=outcomes.count(MoveOutcome.STAYED),
                cells_moved=moved,
                blocked_moves=sum(1 for o in outcomes if o in BLOCKED_OUTCOMES),
                max_consecutive_blocks=_longest_run([o in BLOCKED_OUTCOMES for o in outcomes]),
                time_to_safety=resolution_step[pid] if is_evacuee else max_steps,
                static_exit_distance=lower_bound,
                path_efficiency=_ratio(lower_bound, moved) if is_evacuee and lower_bound is not None else None,
                detour_ratio=_ratio(moved, lower_bound) if is_evacuee and lower_bound is not None else None,
                min_fire_distance=person_min_distance[pid],
                fire_exposure=float(person_exposure[pid]),
            )
        )

    return EpisodeMetrics(
        metrics_version=METRICS_VERSION,
        simulator_version=record.simulator_version,
        observation_version=record.observation_version,
        schema_version=record.schema_version,
        scenario_id=record.scenario_id,
        scenario_hash=record.scenario_hash,
        seed=record.seed,
        termination_reason=termination_reason,
        safety=_safety(scenario, final, people),
        time=_time(len(record.steps), max_steps, people),
        movement=_movement(people, history),
        path_efficiency=_path_efficiency(people),
        fire_exposure=_fire_exposure(distance_counts, no_fire_states, person_exposure),
        exits=_exits(scenario, people),
        replanning=_replanning(history),
        people=tuple(people),
    )


def _safety(scenario: Scenario, final: SimState, people: Sequence[PersonMetrics]) -> SafetyMetrics:
    population = len(people)
    evacuated = sum(1 for p in people if p.status is PersonStatus.EVACUATED)
    casualties = sum(1 for p in people if p.status is PersonStatus.CASUALTY)
    stranded = sum(1 for p in people if p.status is PersonStatus.ACTIVE)
    if evacuated + casualties + stranded != population or population != len(scenario.people):
        raise _fail("terminal outcomes do not reconcile to the population")
    trapped = len(final.trapped)
    return SafetyMetrics(
        population=population,
        evacuated=evacuated,
        casualties=casualties,
        stranded=stranded,
        evacuated_rate=float(Fraction(evacuated, population)),
        casualty_rate=float(Fraction(casualties, population)),
        stranded_rate=float(Fraction(stranded, population)),
        terminal_trapped_count=trapped,
        terminal_trapped_rate=float(Fraction(trapped, population)),
        casualties_by_cause=tuple((cause.value, sum(1 for p in people if p.casualty_cause is cause)) for cause in CasualtyCause),
    )


def _time(episode_steps: int, max_steps: int, people: Sequence[PersonMetrics]) -> TimeMetrics:
    evac = [(p.person_id, p.resolution_step) for p in people if p.status is PersonStatus.EVACUATED]
    steps = [s for _, s in evac if s is not None]
    total = sum(p.time_to_safety for p in people)
    return TimeMetrics(
        episode_steps=episode_steps,
        max_steps=max_steps,
        evacuees=len(steps),
        non_evacuees=len(people) - len(steps),
        evacuation_steps=tuple((pid, s) for pid, s in evac if s is not None),
        first_evacuation_step=min(steps, default=None),
        last_evacuation_step=max(steps, default=None),
        mean_evacuation_step=_ratio(sum(steps), len(steps)),
        median_evacuation_step=float(_median(steps)) if steps else None,
        person_time_to_safety_total=total,
        mean_person_time_to_safety=float(Fraction(total, len(people))),
    )


def _movement(people: Sequence[PersonMetrics], history: dict[str, list[tuple[Action, MoveOutcome]]]) -> MovementMetrics:
    outcomes = [o for pid in sorted(history) for _, o in history[pid]]
    decisions = len(outcomes)
    stays = outcomes.count(MoveOutcome.STAYED)
    attempts = decisions - stays
    successes = outcomes.count(MoveOutcome.MOVED)
    by_outcome = tuple((o.value, outcomes.count(o)) for o in BLOCKED_OUTCOMES)
    blocked = sum(n for _, n in by_outcome)
    if successes + blocked != attempts:
        raise _fail("movement outcomes do not reconcile to the attempts")
    congestion = sum(1 for o in outcomes if o in CONGESTION_OUTCOMES)
    return MovementMetrics(
        decisions=decisions,
        stay_actions=stays,
        movement_attempts=attempts,
        successful_moves=successes,
        blocked_moves=blocked,
        blocked_by_outcome=by_outcome,
        congestion_blocks=congestion,
        stay_rate=_ratio(stays, decisions),
        blocked_rate=_ratio(blocked, attempts),
        congestion_block_rate=_ratio(congestion, attempts),
    )


def _path_efficiency(people: Sequence[PersonMetrics]) -> PathEfficiencyMetrics:
    pairs = [
        (p.cells_moved, p.static_exit_distance)
        for p in people
        if p.status is PersonStatus.EVACUATED and p.static_exit_distance is not None
    ]
    total_moved = sum(m for m, _ in pairs)
    total_bound = sum(b for _, b in pairs)
    n = len(pairs)
    return PathEfficiencyMetrics(
        evacuees_measured=n,
        total_cells_moved=total_moved,
        total_static_lower_bound=total_bound,
        aggregate_path_efficiency=_ratio(total_bound, total_moved),
        mean_path_efficiency=_ratio(sum((Fraction(b, m) for m, b in pairs), Fraction(0)), n),
        mean_detour_ratio=_ratio(sum((Fraction(m, b) for m, b in pairs), Fraction(0)), n),
    )


def _fire_exposure(distance_counts: dict[int, int], no_fire_states: int, per_person: dict[str, Fraction]) -> FireExposureProxy:
    with_fire = sum(distance_counts.values())
    total_states = with_fire + no_fire_states
    exposure = sum(per_person.values(), Fraction(0))
    histogram = tuple(sorted(distance_counts.items()))
    return FireExposureProxy(
        active_person_states=total_states,
        person_states_with_fire=with_fire,
        person_states_without_fire=no_fire_states,
        distance_histogram=histogram,
        adjacent_person_states=distance_counts.get(1, 0),
        min_fire_distance=histogram[0][0] if histogram else None,
        exposure_total=float(exposure),
        mean_exposure=float(exposure / total_states),  # s_0 holds the whole population, so total_states >= 1
    )


def _exits(scenario: Scenario, people: Sequence[PersonMetrics]) -> ExitUtilization:
    counts = tuple((e.id, sum(1 for p in people if p.exit_id == e.id)) for e in scenario.exits)
    evacuated = sum(n for _, n in counts)
    k = len(counts)
    shares: tuple[tuple[str, float], ...] | None = None
    max_share: float | None = None
    balance: float | None = None
    if evacuated:
        fractions = [Fraction(n, evacuated) for _, n in counts]
        shares = tuple((eid, float(f)) for (eid, _), f in zip(counts, fractions, strict=True))
        max_share = float(max(fractions))
        if k > 1:
            gini_simpson = 1 - sum(f * f for f in fractions)
            balance = float(gini_simpson / (1 - Fraction(1, k)))
    return ExitUtilization(
        exit_count=k,
        exits_used=sum(1 for _, n in counts if n),
        evacuations_per_exit=counts,
        exit_shares=shares,
        max_exit_share=max_share,
        exit_balance=balance,
    )


def _replanning(history: dict[str, list[tuple[Action, MoveOutcome]]]) -> ReplanningProxies:
    blocked_events = with_next = moved_after = blocked_after = stayed_after = changed = longest = 0
    for pid in sorted(history):
        events = history[pid]
        longest = max(longest, _longest_run([o in BLOCKED_OUTCOMES for _, o in events]))
        for i, (action, outcome) in enumerate(events):
            if outcome not in BLOCKED_OUTCOMES:
                continue
            blocked_events += 1
            if i + 1 == len(events):
                continue
            # Decisions are recorded once per active step, so index i + 1 is the next step.
            next_action, next_outcome = events[i + 1]
            with_next += 1
            moved_after += next_outcome is MoveOutcome.MOVED
            blocked_after += next_outcome in BLOCKED_OUTCOMES
            stayed_after += next_outcome is MoveOutcome.STAYED
            changed += next_action is not action
    return ReplanningProxies(
        blocked_events=blocked_events,
        blocks_with_next_decision=with_next,
        blocks_without_next_decision=blocked_events - with_next,
        moved_after_block=moved_after,
        blocked_after_block=blocked_after,
        stayed_after_block=stayed_after,
        action_changed_after_block=changed,
        max_consecutive_blocks=longest,
        recovery_proxy_rate=_ratio(moved_after, with_next),
        repeated_block_proxy_rate=_ratio(blocked_after, with_next),
        action_change_proxy_rate=_ratio(changed, with_next),
    )
