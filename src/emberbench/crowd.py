"""Crowd coordination diagnostics computed from a ``Scenario`` and its ``RunRecord``.

The entry point is :func:`compute_crowd_diagnostics`. It is a pure function
and returns an immutable :class:`CrowdDiagnostics`. The definitions are in
``docs/crowd-diagnostics.md``.

Rules for this module:

- **Observation only.** Every field is read from the recorded movements and
  events. Nothing here changes or re-implements the V1 movement rules
  (``movement.py``). The values describe what happened in the simulator.
  They do not model human crowd behaviour.
- **Raw counts only.** Every field is a count, a step number, a cell or a
  wall-only path distance. There is no crowd score and no ratio.
- **Fail closed.** The record goes through the same three validation stages
  as the metrics (:func:`emberbench.metrics.validate_run_record`), including
  a full simulator replay. A failure raises :class:`MetricsInputError`.
- **Proxies are named as proxies.** The nearest-exit demand proxy uses a
  static, wall-only path distance. It does not observe model intent, and it
  does not say which exit a person should use.
- **Canonical order.** Steps are in step order, cells in row-major order and
  exits in exit-ID order, so equal inputs serialise to equal bytes.

This artifact has its own version, :data:`CROWD_DIAGNOSTICS_VERSION`. It is
separate from ``METRICS_VERSION``: no metric definition changes here.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json, sha256_hex
from .errors import MetricsInputError
from .events import RunRecord
from .grid import Pos, neighbours
from .metrics import BLOCKED_OUTCOMES, CONGESTION_OUTCOMES, validate_run_record
from .movement import MoveOutcome
from .scenario import Scenario
from .state import SimState

# Bump this whenever the definition of any crowd diagnostic field changes.
CROWD_DIAGNOSTICS_VERSION = "1.0"

# Moves that rule 2 rejects against the static map or the fire. They never
# meet another person, so they are not contenders for a cell.
STATIC_BLOCK_OUTCOMES: frozenset[MoveOutcome] = frozenset(
    {MoveOutcome.BLOCKED_BOUNDARY, MoveOutcome.BLOCKED_WALL, MoveOutcome.BLOCKED_FIRE}
)


def _pairs_to_dict(pairs: Sequence[tuple[str, int]]) -> dict[str, int]:
    return dict(pairs)


# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class NearestExitDemandProxy:
    """Active people counted by their static nearest exit. A proxy, not observed intent.

    A person counts toward one exit only when that exit is strictly nearer,
    by wall-only path distance, than every other exit. Ties go to ``tied``.
    People with no wall-free path to any exit go to ``unreachable``.
    ``sum(per_exit) + tied + unreachable`` equals the number of active people.
    """

    per_exit: tuple[tuple[str, int], ...]  # every scenario exit, sorted by exit ID
    tied: int
    unreachable: int

    def to_dict(self) -> dict[str, Any]:
        return {"per_exit": _pairs_to_dict(self.per_exit), "tied": self.tied, "unreachable": self.unreachable}


@dataclass(frozen=True)
class StepCrowdSignals:
    """Raw crowd counts for one step. ``active_people`` is the number of decisions in the step."""

    step: int
    active_people: int
    stays: int
    movement_attempts: int  # decisions whose action is not STAY
    successful_moves: int
    blocked_moves: int
    blocked_by_outcome: tuple[tuple[str, int], ...]  # every blocked outcome, in MoveOutcome order
    congestion_blocks: int  # blocked_occupied + blocked_conflict + blocked_cycle
    contending_moves: int  # movement attempts not blocked by boundary, wall or fire
    contested_cells: int  # target cells with two or more contenders
    max_contenders: int  # largest number of contenders for one target cell; 0 if none
    evacuations_by_exit: tuple[tuple[str, int], ...]  # every scenario exit, sorted by exit ID
    cumulative_evacuations_by_exit: tuple[tuple[str, int], ...]  # totals up to and including this step
    nearest_exit_demand_proxy: NearestExitDemandProxy  # over the people active at decision time

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "active_people": self.active_people,
            "stays": self.stays,
            "movement_attempts": self.movement_attempts,
            "successful_moves": self.successful_moves,
            "blocked_moves": self.blocked_moves,
            "blocked_by_outcome": _pairs_to_dict(self.blocked_by_outcome),
            "congestion_blocks": self.congestion_blocks,
            "contending_moves": self.contending_moves,
            "contested_cells": self.contested_cells,
            "max_contenders": self.max_contenders,
            "evacuations_by_exit": _pairs_to_dict(self.evacuations_by_exit),
            "cumulative_evacuations_by_exit": _pairs_to_dict(self.cumulative_evacuations_by_exit),
            "nearest_exit_demand_proxy": self.nearest_exit_demand_proxy.to_dict(),
        }


@dataclass(frozen=True)
class CellCongestion:
    """Observed entry attempts into one target cell, summed over the episode.

    Only contending moves count: a movement attempt not blocked by boundary,
    wall or fire. So ``attempted_entries == successful_entries + congestion_blocks``.
    """

    cell: Pos
    exit_id: str | None  # the exit on this cell, if any
    static_exit_distance: int | None  # wall-only path distance to the nearest exit; None if unreachable
    attempted_entries: int
    successful_entries: int
    blocked_occupied: int
    blocked_conflict: int
    blocked_cycle: int
    congestion_blocks: int
    contention_steps: int  # steps in which two or more people targeted this cell
    peak_contenders: int  # most people that targeted this cell in one step

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.to_list(),
            "exit_id": self.exit_id,
            "static_exit_distance": self.static_exit_distance,
            "attempted_entries": self.attempted_entries,
            "successful_entries": self.successful_entries,
            "blocked_occupied": self.blocked_occupied,
            "blocked_conflict": self.blocked_conflict,
            "blocked_cycle": self.blocked_cycle,
            "congestion_blocks": self.congestion_blocks,
            "contention_steps": self.contention_steps,
            "peak_contenders": self.peak_contenders,
        }


@dataclass(frozen=True)
class ExitLoad:
    """Measured throughput of one exit and its static nearest-exit demand proxy."""

    exit_id: str
    position: Pos
    evacuations: int
    peak_step_evacuations: int  # at most 1 under V1: one exit cell takes one person per step
    steps_with_evacuation: int
    first_evacuation_step: int | None
    last_evacuation_step: int | None
    initial_demand_proxy: int  # people at step 0 whose unique static nearest exit is this one
    demand_proxy_person_steps: int  # the same count, summed over the decision snapshot of every step

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_id": self.exit_id,
            "position": self.position.to_list(),
            "evacuations": self.evacuations,
            "peak_step_evacuations": self.peak_step_evacuations,
            "steps_with_evacuation": self.steps_with_evacuation,
            "first_evacuation_step": self.first_evacuation_step,
            "last_evacuation_step": self.last_evacuation_step,
            "initial_demand_proxy": self.initial_demand_proxy,
            "demand_proxy_person_steps": self.demand_proxy_person_steps,
        }


@dataclass(frozen=True)
class ExitLoadSignals:
    """Measured exit use plus the demand proxy. Equal load is not assumed to be optimal."""

    exit_count: int
    exits: tuple[ExitLoad, ...]  # sorted by exit ID
    total_evacuations: int
    peak_step_evacuations: int  # most evacuations over all exits in one step
    initial_demand_tied: int
    initial_demand_unreachable: int
    demand_tied_person_steps: int
    demand_unreachable_person_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_count": self.exit_count,
            "exits": [e.to_dict() for e in self.exits],
            "total_evacuations": self.total_evacuations,
            "peak_step_evacuations": self.peak_step_evacuations,
            "initial_demand_tied": self.initial_demand_tied,
            "initial_demand_unreachable": self.initial_demand_unreachable,
            "demand_tied_person_steps": self.demand_tied_person_steps,
            "demand_unreachable_person_steps": self.demand_unreachable_person_steps,
        }


@dataclass(frozen=True)
class CrowdTotals:
    """Episode totals of the step signals. They reconcile with ``MovementMetrics``."""

    decisions: int
    stays: int
    movement_attempts: int
    successful_moves: int
    blocked_moves: int
    blocked_by_outcome: tuple[tuple[str, int], ...]
    congestion_blocks: int
    contending_moves: int
    contested_cell_steps: int  # sum of contested_cells over the steps
    steps_with_contention: int  # steps with at least one contested cell
    steps_with_congestion_blocks: int
    peak_contested_cells: int
    peak_contenders: int
    peak_congestion_blocks: int  # most congestion blocks in one step
    hotspot_cells: int  # cells with at least one congestion block

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "stays": self.stays,
            "movement_attempts": self.movement_attempts,
            "successful_moves": self.successful_moves,
            "blocked_moves": self.blocked_moves,
            "blocked_by_outcome": _pairs_to_dict(self.blocked_by_outcome),
            "congestion_blocks": self.congestion_blocks,
            "contending_moves": self.contending_moves,
            "contested_cell_steps": self.contested_cell_steps,
            "steps_with_contention": self.steps_with_contention,
            "steps_with_congestion_blocks": self.steps_with_congestion_blocks,
            "peak_contested_cells": self.peak_contested_cells,
            "peak_contenders": self.peak_contenders,
            "peak_congestion_blocks": self.peak_congestion_blocks,
            "hotspot_cells": self.hotspot_cells,
        }


@dataclass(frozen=True)
class CrowdDiagnostics:
    """All crowd diagnostics for one authenticated episode. There is deliberately no crowd score."""

    crowd_diagnostics_version: str
    simulator_version: str
    observation_version: str
    schema_version: str
    scenario_id: str
    scenario_hash: str
    seed: int
    record_digest: str  # RunRecord.digest() of the record these values come from
    episode_steps: int
    totals: CrowdTotals
    steps: tuple[StepCrowdSignals, ...]  # in step order
    cells: tuple[CellCongestion, ...]  # cells with at least one attempted entry, row-major order
    exits: ExitLoadSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "crowd_diagnostics_version": self.crowd_diagnostics_version,
            "simulator_version": self.simulator_version,
            "observation_version": self.observation_version,
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "record_digest": self.record_digest,
            "episode_steps": self.episode_steps,
            "totals": self.totals.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "cells": [c.to_dict() for c in self.cells],
            "exits": self.exits.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())

    def cell(self, pos: Pos) -> CellCongestion | None:
        """The entry for ``pos``, or ``None`` if nobody tried to enter it."""
        for c in self.cells:
            if c.cell == pos:
                return c
        return None


# --------------------------------------------------------------------------- geometry


def per_exit_static_distances(scenario: Scenario) -> dict[str, dict[Pos, int]]:
    """For each exit, the shortest 4-neighbour path length to it from every cell, around walls only.

    People, fire and the other exits are ignored. Cells that cannot reach an
    exit are absent from that exit's map.
    """
    walls = scenario.walls
    result: dict[str, dict[Pos, int]] = {}
    for e in scenario.exits:
        dist = {e.position: 0}
        frontier = deque([e.position])
        while frontier:
            cell = frontier.popleft()
            for nb in neighbours(cell, scenario.width, scenario.height):
                if nb not in dist and nb not in walls:
                    dist[nb] = dist[cell] + 1
                    frontier.append(nb)
        result[e.id] = dist
    return result


_TIED = ""  # marker in the nearest-exit table; exit IDs are never empty


def _nearest_exit_table(per_exit: Mapping[str, Mapping[Pos, int]]) -> dict[Pos, str]:
    """Map each reachable cell to its unique nearest exit ID, or to ``_TIED``."""
    table: dict[Pos, str] = {}
    cells = sorted({p for dist in per_exit.values() for p in dist})
    for cell in cells:
        reached = [(dist[cell], eid) for eid, dist in sorted(per_exit.items()) if cell in dist]
        best = min(d for d, _ in reached)
        winners = [eid for d, eid in reached if d == best]
        table[cell] = winners[0] if len(winners) == 1 else _TIED
    return table


def _demand(positions: Sequence[Pos], exit_ids: Sequence[str], table: Mapping[Pos, str]) -> NearestExitDemandProxy:
    counts = dict.fromkeys(exit_ids, 0)
    tied = unreachable = 0
    for pos in positions:
        nearest = table.get(pos)
        if nearest is None:
            unreachable += 1
        elif nearest == _TIED:
            tied += 1
        else:
            counts[nearest] += 1
    return NearestExitDemandProxy(per_exit=tuple((eid, counts[eid]) for eid in exit_ids), tied=tied, unreachable=unreachable)


def _active_positions(state: SimState) -> list[Pos]:
    return [p.position for p in state.active]


# --------------------------------------------------------------------------- computation


def _fail(message: str) -> MetricsInputError:
    return MetricsInputError(message)


@dataclass
class _CellAccumulator:
    attempted: int = 0
    moved: int = 0
    occupied: int = 0
    conflict: int = 0
    cycle: int = 0
    contention_steps: int = 0
    peak: int = 0


def compute_crowd_diagnostics(scenario: Scenario, record: RunRecord) -> CrowdDiagnostics:
    """Compute crowd coordination diagnostics for one complete, authenticated episode.

    Raises :class:`MetricsInputError` if the record does not belong to
    ``scenario``, comes from an incompatible version, is malformed, does not
    reach termination, or is not reproduced exactly by a simulator replay.
    Neither argument is mutated.
    """
    validate_run_record(scenario, record)

    exit_ids = [e.id for e in scenario.exits]
    exit_at = {e.position: e.id for e in scenario.exits}
    per_exit = per_exit_static_distances(scenario)
    table = _nearest_exit_table(per_exit)
    static_distance = {cell: min(d[cell] for d in per_exit.values() if cell in d) for cell in table}

    cells: dict[Pos, _CellAccumulator] = {}
    steps: list[StepCrowdSignals] = []
    cumulative = dict.fromkeys(exit_ids, 0)
    first_evac: dict[str, int] = {}
    last_evac: dict[str, int] = {}
    peak_exit = dict.fromkeys(exit_ids, 0)
    steps_with_evac = dict.fromkeys(exit_ids, 0)

    for step in record.steps:
        outcomes = [m.outcome for m in step.movements]
        stays = outcomes.count(MoveOutcome.STAYED)
        moved = outcomes.count(MoveOutcome.MOVED)
        by_outcome = tuple((o.value, outcomes.count(o)) for o in BLOCKED_OUTCOMES)
        blocked = sum(n for _, n in by_outcome)
        attempts = len(outcomes) - stays
        if moved + blocked != attempts:
            raise _fail(f"step {step.step}: movement outcomes do not reconcile to the attempts")

        # Contenders: movement attempts that passed the static checks, grouped by target cell.
        groups: dict[Pos, list[MoveOutcome]] = {}
        for m in step.movements:
            if m.outcome is MoveOutcome.STAYED or m.outcome in STATIC_BLOCK_OUTCOMES:
                continue
            groups.setdefault(m.target, []).append(m.outcome)
        for target, group in sorted(groups.items()):
            acc = cells.setdefault(target, _CellAccumulator())
            acc.attempted += len(group)
            acc.moved += group.count(MoveOutcome.MOVED)
            acc.occupied += group.count(MoveOutcome.BLOCKED_OCCUPIED)
            acc.conflict += group.count(MoveOutcome.BLOCKED_CONFLICT)
            acc.cycle += group.count(MoveOutcome.BLOCKED_CYCLE)
            if group.count(MoveOutcome.MOVED) > 1:
                raise _fail(f"step {step.step}: two people entered {list(target)}")
            if len(group) > 1:
                acc.contention_steps += 1
            acc.peak = max(acc.peak, len(group))

        evac_counts = dict.fromkeys(exit_ids, 0)
        for e in step.evacuations:
            evac_counts[e.exit_id] += 1
        for eid in exit_ids:
            n = evac_counts[eid]
            cumulative[eid] += n
            if n:
                steps_with_evac[eid] += 1
                first_evac.setdefault(eid, step.step)
                last_evac[eid] = step.step
                peak_exit[eid] = max(peak_exit[eid], n)

        steps.append(
            StepCrowdSignals(
                step=step.step,
                active_people=len(outcomes),
                stays=stays,
                movement_attempts=attempts,
                successful_moves=moved,
                blocked_moves=blocked,
                blocked_by_outcome=by_outcome,
                congestion_blocks=sum(1 for o in outcomes if o in CONGESTION_OUTCOMES),
                contending_moves=sum(len(g) for g in groups.values()),
                contested_cells=sum(1 for g in groups.values() if len(g) > 1),
                max_contenders=max((len(g) for g in groups.values()), default=0),
                evacuations_by_exit=tuple((eid, evac_counts[eid]) for eid in exit_ids),
                cumulative_evacuations_by_exit=tuple((eid, cumulative[eid]) for eid in exit_ids),
                nearest_exit_demand_proxy=_demand([m.origin for m in step.movements], exit_ids, table),
            )
        )

    cell_rows = tuple(
        CellCongestion(
            cell=pos,
            exit_id=exit_at.get(pos),
            static_exit_distance=static_distance.get(pos),
            attempted_entries=acc.attempted,
            successful_entries=acc.moved,
            blocked_occupied=acc.occupied,
            blocked_conflict=acc.conflict,
            blocked_cycle=acc.cycle,
            congestion_blocks=acc.occupied + acc.conflict + acc.cycle,
            contention_steps=acc.contention_steps,
            peak_contenders=acc.peak,
        )
        for pos, acc in sorted(cells.items())
    )
    for c in cell_rows:
        if c.attempted_entries != c.successful_entries + c.congestion_blocks:
            raise _fail(f"cell {list(c.cell)}: entries do not reconcile")

    totals = _totals(steps, cell_rows)
    initial = _demand(_active_positions(record.initial_state), exit_ids, table)
    initial_by_exit = dict(initial.per_exit)
    exits = tuple(
        ExitLoad(
            exit_id=e.id,
            position=e.position,
            evacuations=cumulative[e.id],
            peak_step_evacuations=peak_exit[e.id],
            steps_with_evacuation=steps_with_evac[e.id],
            first_evacuation_step=first_evac.get(e.id),
            last_evacuation_step=last_evac.get(e.id),
            initial_demand_proxy=initial_by_exit[e.id],
            demand_proxy_person_steps=sum(dict(s.nearest_exit_demand_proxy.per_exit)[e.id] for s in steps),
        )
        for e in scenario.exits
    )
    exit_signals = ExitLoadSignals(
        exit_count=len(exits),
        exits=exits,
        total_evacuations=sum(cumulative.values()),
        peak_step_evacuations=max((sum(n for _, n in s.evacuations_by_exit) for s in steps), default=0),
        initial_demand_tied=initial.tied,
        initial_demand_unreachable=initial.unreachable,
        demand_tied_person_steps=sum(s.nearest_exit_demand_proxy.tied for s in steps),
        demand_unreachable_person_steps=sum(s.nearest_exit_demand_proxy.unreachable for s in steps),
    )

    return CrowdDiagnostics(
        crowd_diagnostics_version=CROWD_DIAGNOSTICS_VERSION,
        simulator_version=record.simulator_version,
        observation_version=record.observation_version,
        schema_version=record.schema_version,
        scenario_id=record.scenario_id,
        scenario_hash=record.scenario_hash,
        seed=record.seed,
        record_digest=record.digest(),
        episode_steps=len(record.steps),
        totals=totals,
        steps=tuple(steps),
        cells=cell_rows,
        exits=exit_signals,
    )


def _totals(steps: Sequence[StepCrowdSignals], cells: Sequence[CellCongestion]) -> CrowdTotals:
    by_outcome = tuple(
        (o.value, sum(dict(s.blocked_by_outcome)[o.value] for s in steps)) for o in BLOCKED_OUTCOMES
    )
    return CrowdTotals(
        decisions=sum(s.active_people for s in steps),
        stays=sum(s.stays for s in steps),
        movement_attempts=sum(s.movement_attempts for s in steps),
        successful_moves=sum(s.successful_moves for s in steps),
        blocked_moves=sum(s.blocked_moves for s in steps),
        blocked_by_outcome=by_outcome,
        congestion_blocks=sum(s.congestion_blocks for s in steps),
        contending_moves=sum(s.contending_moves for s in steps),
        contested_cell_steps=sum(s.contested_cells for s in steps),
        steps_with_contention=sum(1 for s in steps if s.contested_cells),
        steps_with_congestion_blocks=sum(1 for s in steps if s.congestion_blocks),
        peak_contested_cells=max((s.contested_cells for s in steps), default=0),
        peak_contenders=max((s.max_contenders for s in steps), default=0),
        peak_congestion_blocks=max((s.congestion_blocks for s in steps), default=0),
        hotspot_cells=sum(1 for c in cells if c.congestion_blocks),
    )
