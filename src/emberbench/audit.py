"""Read-only replay audit of one authenticated episode: timeline, person paths, fire deltas and congestion.

The entry point is :func:`build_replay_audit`. It takes a ``Scenario`` and a
``RunRecord`` and returns an immutable :class:`ReplayAudit`. The definitions
are in ``docs/replay-visualization.md``.

Rules for this module:

- **Authenticated input only.** The record goes through the same three
  validation stages as the metrics (:func:`emberbench.metrics.validate_run_record`),
  including a full simulator replay. A failure raises
  :class:`ReplayAuditError`. There is no path that audits an unchecked record.
- **Read only.** Every field is copied or counted from the recorded steps and
  states. Nothing here runs the simulator rules again, predicts fire or
  infers an event that the record does not hold. Visualization never affects
  benchmark results.
- **One pass.** The timeline, the person paths and the fire deltas are built
  in a single walk over the steps, so the cost is linear in the record size.
- **Canonical order.** Steps are in step order, people in person-ID order,
  cells in row-major order and exits in exit-ID order, so equal inputs
  serialise to equal bytes.

This artifact has its own version, :data:`REPLAY_AUDIT_VERSION`. No
simulator, metric or crowd definition changes here.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json, sha256_hex
from .crowd import STATIC_BLOCK_OUTCOMES, CellCongestion, CrowdDiagnostics, ExitLoadSignals, compute_crowd_diagnostics
from .errors import MetricsInputError, ReplayAuditError
from .events import Casualty, CasualtyCause, Evacuation, RunRecord
from .grid import Action, Pos
from .metrics import validate_run_record
from .movement import MoveOutcome, MoveResolution
from .scenario import Scenario
from .state import PersonState, PersonStatus, SimState, TerminationReason

# Bump this whenever the definition or the serialised layout of any audit field changes.
REPLAY_AUDIT_VERSION = "1.0"


def _cells(cells: Iterable[Pos]) -> list[list[int]]:
    return [p.to_list() for p in sorted(cells)]


# --------------------------------------------------------------------------- map


@dataclass(frozen=True)
class AuditMap:
    """The static map an external viewer needs to draw a frame."""

    width: int
    height: int
    walls: tuple[Pos, ...]  # row-major order
    exits: tuple[tuple[str, Pos], ...]  # sorted by exit ID

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "AuditMap":
        return cls(
            width=scenario.width,
            height=scenario.height,
            walls=tuple(sorted(scenario.obstacles)),
            exits=tuple((e.id, e.position) for e in scenario.exits),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "walls": _cells(self.walls),
            "exits": [{"id": eid, "position": pos.to_list()} for eid, pos in self.exits],
        }


# --------------------------------------------------------------------------- timeline


@dataclass(frozen=True)
class TimelineFrame:
    """One timestep. Frame 0 is the initial state and has no events.

    ``people`` and every count describe the state *after* the step. Each
    movement holds the submitted action and the recorded outcome.
    """

    step: int
    movements: tuple[MoveResolution, ...]  # person-ID order; empty at step 0
    newly_ignited: tuple[Pos, ...]  # row-major order; empty at step 0
    evacuations: tuple[Evacuation, ...]  # person-ID order
    casualties: tuple[Casualty, ...]  # person-ID order
    people: tuple[PersonState, ...]  # every person, person-ID order
    burning_count: int
    active_count: int
    evacuated_total: int  # cumulative, up to and including this step
    casualty_total: int  # cumulative, up to and including this step
    trapped: tuple[str, ...]  # the recorded trapped diagnostic, sorted
    terminated: bool
    termination_reason: TerminationReason | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "movements": [m.to_dict() for m in self.movements],
            "newly_ignited": _cells(self.newly_ignited),
            "evacuations": [e.to_dict() for e in self.evacuations],
            "casualties": [c.to_dict() for c in self.casualties],
            "people": [p.to_dict() for p in self.people],
            "burning_count": self.burning_count,
            "active_count": self.active_count,
            "evacuated_total": self.evacuated_total,
            "casualty_total": self.casualty_total,
            "trapped": list(self.trapped),
            "terminated": self.terminated,
            "termination_reason": None if self.termination_reason is None else self.termination_reason.value,
        }


def _frame(
    state: SimState,
    movements: tuple[MoveResolution, ...],
    ignited: tuple[Pos, ...],
    evacuations: tuple[Evacuation, ...],
    casualties: tuple[Casualty, ...],
) -> TimelineFrame:
    statuses = [p.status for p in state.people]
    return TimelineFrame(
        step=state.step,
        movements=movements,
        newly_ignited=ignited,
        evacuations=evacuations,
        casualties=casualties,
        people=state.people,
        burning_count=len(state.burning),
        active_count=statuses.count(PersonStatus.ACTIVE),
        evacuated_total=statuses.count(PersonStatus.EVACUATED),
        casualty_total=statuses.count(PersonStatus.CASUALTY),
        trapped=tuple(sorted(state.trapped)),
        terminated=state.terminated,
        termination_reason=state.termination_reason,
    )


# --------------------------------------------------------------------------- person paths


@dataclass(frozen=True)
class PathStep:
    """One decision of one person: the step, the attempted action and the recorded outcome."""

    step: int
    action: Action
    outcome: MoveOutcome
    origin: Pos
    target: Pos
    position: Pos  # the cell after movement in this step

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action.value,
            "outcome": self.outcome.value,
            "origin": self.origin.to_list(),
            "target": self.target.to_list(),
            "position": self.position.to_list(),
        }


@dataclass(frozen=True)
class PersonPath:
    """The recorded trajectory of one person.

    ``steps`` holds one entry for every step in which the person was active at
    decision time: steps ``1`` to ``len(steps)``, with no gaps. A resolved
    person's last entry is the resolution step. An unresolved person was
    still active when the episode ended (``final_status`` is ``active``).
    """

    person_id: str
    initial_position: Pos
    steps: tuple[PathStep, ...]
    final_status: PersonStatus
    final_position: Pos
    resolution_step: int | None  # evacuation or casualty step; None if unresolved
    exit_id: str | None  # set only for an evacuation
    casualty_cause: CasualtyCause | None  # set only for a casualty
    trapped_at_end: bool  # unresolved and in the final state's trapped set

    @property
    def unresolved(self) -> bool:
        return self.final_status is PersonStatus.ACTIVE

    def position_at(self, step: int) -> Pos:
        """The cell after ``step``. After resolution this stays the exit or the burning cell."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if step == 0:
            return self.initial_position
        if step <= len(self.steps):
            return self.steps[step - 1].position
        return self.final_position

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "initial_position": self.initial_position.to_list(),
            "steps": [s.to_dict() for s in self.steps],
            "final_status": self.final_status.value,
            "final_position": self.final_position.to_list(),
            "resolution_step": self.resolution_step,
            "exit_id": self.exit_id,
            "casualty_cause": None if self.casualty_cause is None else self.casualty_cause.value,
            "trapped_at_end": self.trapped_at_end,
        }


# --------------------------------------------------------------------------- fire


@dataclass(frozen=True)
class FireDelta:
    """Fire in one step. ``burning_after`` equals ``burning_before | newly_ignited``; the two parts are disjoint."""

    step: int
    burning_before: frozenset[Pos]
    newly_ignited: tuple[Pos, ...]  # row-major order
    burning_after: frozenset[Pos]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "burning_before": _cells(self.burning_before),
            "newly_ignited": _cells(self.newly_ignited),
            "burning_after": _cells(self.burning_after),
        }


# --------------------------------------------------------------------------- congestion overlay


@dataclass(frozen=True)
class BlockedMove:
    """A recorded movement attempt that did not move. ``reason`` is the recorded outcome."""

    person_id: str
    origin: Pos
    target: Pos
    reason: MoveOutcome

    def to_dict(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "origin": self.origin.to_list(), "target": self.target.to_list(), "reason": self.reason.value}


@dataclass(frozen=True)
class ContestedCell:
    """A target cell with two or more contenders in one step, as defined in ``crowd.py``."""

    cell: Pos
    contenders: tuple[tuple[str, MoveOutcome], ...]  # person-ID order, with the recorded outcome

    def to_dict(self) -> dict[str, Any]:
        return {"cell": self.cell.to_list(), "contenders": [{"person_id": pid, "outcome": o.value} for pid, o in self.contenders]}


@dataclass(frozen=True)
class StepCongestion:
    """Congestion markers for one step. The counts reconcile with ``StepCrowdSignals``."""

    step: int
    blocked: tuple[BlockedMove, ...]  # person-ID order
    contested_cells: tuple[ContestedCell, ...]  # row-major order
    congestion_blocks: int
    evacuations_by_exit: tuple[tuple[str, int], ...]  # every exit, sorted by exit ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "blocked": [b.to_dict() for b in self.blocked],
            "contested_cells": [c.to_dict() for c in self.contested_cells],
            "congestion_blocks": self.congestion_blocks,
            "evacuations_by_exit": dict(self.evacuations_by_exit),
        }


@dataclass(frozen=True)
class CrowdOverlay:
    """The optional congestion layer. Every value comes from :func:`compute_crowd_diagnostics` or the record."""

    crowd_diagnostics_version: str
    crowd_diagnostics_digest: str
    steps: tuple[StepCongestion, ...]  # step order, steps 1..N
    hotspots: tuple[CellCongestion, ...]  # cells with at least one congestion block, row-major order
    exit_throughput: ExitLoadSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "crowd_diagnostics_version": self.crowd_diagnostics_version,
            "crowd_diagnostics_digest": self.crowd_diagnostics_digest,
            "steps": [s.to_dict() for s in self.steps],
            "hotspots": [c.to_dict() for c in self.hotspots],
            "exit_throughput": self.exit_throughput.to_dict(),
        }


def _internal(message: str) -> ReplayAuditError:
    return ReplayAuditError(f"congestion overlay does not reconcile with the crowd diagnostics: {message}")


def _crowd_overlay(record: RunRecord, diagnostics: CrowdDiagnostics) -> CrowdOverlay:
    if len(diagnostics.steps) != len(record.steps):
        raise _internal("step counts differ")
    steps = []
    for step, signals in zip(record.steps, diagnostics.steps, strict=True):
        blocked = tuple(
            BlockedMove(m.person_id, m.origin, m.target, m.outcome)
            for m in step.movements
            if m.outcome not in (MoveOutcome.STAYED, MoveOutcome.MOVED)
        )
        groups: dict[Pos, list[tuple[str, MoveOutcome]]] = {}
        for m in step.movements:
            if m.outcome is MoveOutcome.STAYED or m.outcome in STATIC_BLOCK_OUTCOMES:
                continue
            groups.setdefault(m.target, []).append((m.person_id, m.outcome))
        contested = tuple(ContestedCell(cell, tuple(sorted(group))) for cell, group in sorted(groups.items()) if len(group) > 1)
        congestion = sum(1 for b in blocked if b.reason not in STATIC_BLOCK_OUTCOMES)
        max_contenders = max((len(g) for g in groups.values()), default=0)
        if (
            signals.step != step.step
            or len(blocked) != signals.blocked_moves
            or len(contested) != signals.contested_cells
            or max_contenders != signals.max_contenders
            or congestion != signals.congestion_blocks
        ):
            raise _internal(f"step {step.step}")
        steps.append(StepCongestion(step.step, blocked, contested, congestion, signals.evacuations_by_exit))
    return CrowdOverlay(
        crowd_diagnostics_version=diagnostics.crowd_diagnostics_version,
        crowd_diagnostics_digest=diagnostics.digest(),
        steps=tuple(steps),
        hotspots=tuple(c for c in diagnostics.cells if c.congestion_blocks > 0),
        exit_throughput=diagnostics.exits,
    )


# --------------------------------------------------------------------------- the audit


@dataclass(frozen=True)
class ReplayAudit:
    """The read-only audit of one authenticated episode. Build it with :func:`build_replay_audit`."""

    replay_audit_version: str
    simulator_version: str
    observation_version: str
    schema_version: str
    scenario_id: str
    scenario_hash: str
    seed: int
    record_digest: str  # RunRecord.digest() of the audited record
    map: AuditMap
    episode_steps: int
    termination_reason: TerminationReason
    timeline: tuple[TimelineFrame, ...]  # frames 0..episode_steps; timeline[k].step == k
    people: tuple[PersonPath, ...]  # person-ID order
    initial_burning: frozenset[Pos]
    fire: tuple[FireDelta, ...]  # steps 1..episode_steps; fire[k - 1].step == k
    crowd: CrowdOverlay | None  # present only when requested
    _person_index: dict[str, PersonPath] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_person_index", {p.person_id: p for p in self.people})

    def _check_step(self, step: int) -> int:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"step must be an int, got {type(step).__name__}")
        if not 0 <= step <= self.episode_steps:
            raise ValueError(f"step {step} is outside the episode (0..{self.episode_steps})")
        return step

    def frame(self, step: int) -> TimelineFrame:
        """The frame after ``step``; step 0 is the initial state."""
        return self.timeline[self._check_step(step)]

    @property
    def initial_frame(self) -> TimelineFrame:
        return self.timeline[0]

    @property
    def final_frame(self) -> TimelineFrame:
        return self.timeline[-1]

    def fire_delta(self, step: int) -> FireDelta:
        """The fire change in ``step`` (1..episode_steps). Step 0 has no delta; see ``initial_burning``."""
        if self._check_step(step) == 0:
            raise ValueError("step 0 has no fire delta; the initial fire is initial_burning")
        return self.fire[step - 1]

    def burning_at(self, step: int) -> frozenset[Pos]:
        """Cells burning after ``step``."""
        if self._check_step(step) == 0:
            return self.initial_burning
        return self.fire[step - 1].burning_after

    def person(self, person_id: str) -> PersonPath:
        return self._person_index[person_id]

    def step_congestion(self, step: int) -> StepCongestion | None:
        """Congestion markers of ``step``, or ``None`` at step 0 or without the crowd layer."""
        if self._check_step(step) == 0 or self.crowd is None:
            return None
        return self.crowd.steps[step - 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_audit_version": self.replay_audit_version,
            "simulator_version": self.simulator_version,
            "observation_version": self.observation_version,
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "record_digest": self.record_digest,
            "map": self.map.to_dict(),
            "episode_steps": self.episode_steps,
            "termination_reason": self.termination_reason.value,
            "timeline": [f.to_dict() for f in self.timeline],
            "people": [p.to_dict() for p in self.people],
            "fire": {"initial_burning": _cells(self.initial_burning), "steps": [d.to_dict() for d in self.fire]},
            "crowd": None if self.crowd is None else self.crowd.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


def authenticate(scenario: Scenario, record: RunRecord) -> None:
    """Fail closed unless ``record`` is a complete, well-formed episode of ``scenario`` that replays exactly."""
    try:
        validate_run_record(scenario, record)
    except MetricsInputError as exc:
        raise ReplayAuditError(f"run record is not authentic for this scenario: {exc}") from exc


def build_replay_audit(scenario: Scenario, record: RunRecord, *, crowd: bool = False) -> ReplayAudit:
    """Build the audit of one complete, authenticated episode.

    Raises :class:`ReplayAuditError` if the record does not belong to
    ``scenario``, comes from an incompatible version, is malformed, does not
    reach termination, or is not reproduced exactly by a simulator replay.
    With ``crowd=True`` the congestion overlay from
    :func:`compute_crowd_diagnostics` is attached. Neither argument is mutated.
    """
    authenticate(scenario, record)

    initial = record.initial_state
    timeline = [_frame(initial, (), (), (), ())]
    fire: list[FireDelta] = []
    path_steps: dict[str, list[PathStep]] = {p.id: [] for p in initial.people}
    evacuated: dict[str, tuple[int, Evacuation]] = {}
    killed: dict[str, tuple[int, Casualty]] = {}

    before = initial
    for step in record.steps:
        for m in step.movements:
            path_steps[m.person_id].append(PathStep(step.step, m.action, m.outcome, m.origin, m.target, m.final))
        for e in step.evacuations:
            evacuated[e.person_id] = (step.step, e)
        for c in step.casualties:
            killed[c.person_id] = (step.step, c)
        fire.append(FireDelta(step.step, before.burning, step.ignited, step.state.burning))
        timeline.append(_frame(step.state, step.movements, step.ignited, step.evacuations, step.casualties))
        before = step.state

    final = record.final_state
    people = []
    for p in final.people:
        evac = evacuated.get(p.id)
        dead = killed.get(p.id)
        people.append(
            PersonPath(
                person_id=p.id,
                initial_position=initial.person(p.id).position,
                steps=tuple(path_steps[p.id]),
                final_status=p.status,
                final_position=p.position,
                resolution_step=evac[0] if evac else dead[0] if dead else None,
                exit_id=evac[1].exit_id if evac else None,
                casualty_cause=dead[1].cause if dead else None,
                trapped_at_end=p.status is PersonStatus.ACTIVE and p.id in final.trapped,
            )
        )

    reason = final.termination_reason
    if reason is None:  # validate_run_record guarantees termination; keep mypy and readers honest
        raise ReplayAuditError("record does not reach termination")
    return ReplayAudit(
        replay_audit_version=REPLAY_AUDIT_VERSION,
        simulator_version=record.simulator_version,
        observation_version=record.observation_version,
        schema_version=record.schema_version,
        scenario_id=record.scenario_id,
        scenario_hash=record.scenario_hash,
        seed=record.seed,
        record_digest=record.digest(),
        map=AuditMap.from_scenario(scenario),
        episode_steps=len(record.steps),
        termination_reason=reason,
        timeline=tuple(timeline),
        people=tuple(people),
        initial_burning=initial.burning,
        fire=tuple(fire),
        crowd=_crowd_overlay(record, compute_crowd_diagnostics(scenario, record)) if crowd else None,
    )


def _as_mapping(artifact: str | Mapping[str, Any], what: str) -> Mapping[str, Any]:
    if isinstance(artifact, str):
        try:
            artifact = json.loads(artifact)
        except json.JSONDecodeError as exc:
            raise ReplayAuditError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(artifact, Mapping):
        raise ReplayAuditError(f"{what} must be a JSON object")
    return artifact


def verify_replay_audit(scenario: Scenario, record: RunRecord, audit: "ReplayAudit | str | Mapping[str, Any]") -> ReplayAudit:
    """Rebuild the audit from the authenticated record and require byte-identical canonical JSON.

    Returns the rebuilt audit. Raises :class:`ReplayAuditError` on any
    difference, including an incompatible ``replay_audit_version``.
    """
    data = audit.to_dict() if isinstance(audit, ReplayAudit) else _as_mapping(audit, "replay audit")
    version = data.get("replay_audit_version")
    if version != REPLAY_AUDIT_VERSION:
        raise ReplayAuditError(f"replay audit version {version!r} is not supported; this is {REPLAY_AUDIT_VERSION!r}")
    rebuilt = build_replay_audit(scenario, record, crowd=data.get("crowd") is not None)
    try:
        given = canonical_json(data)
    except (TypeError, ValueError) as exc:
        raise ReplayAuditError(f"replay audit is not canonical JSON data: {exc}") from exc
    if given != rebuilt.to_json():
        raise ReplayAuditError("replay audit does not match the authenticated record")
    return rebuilt
