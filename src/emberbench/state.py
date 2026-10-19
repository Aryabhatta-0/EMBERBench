"""Immutable simulator state snapshots.

``SimState`` is evaluator-private: it holds the RNG stream positions and the
``trapped`` diagnostic. Models see :class:`emberbench.observation.Observation`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .grid import CELL_GLYPHS, Cell, Pos
from .scenario import Scenario


class PersonStatus(StrEnum):
    ACTIVE = "active"
    EVACUATED = "evacuated"
    CASUALTY = "casualty"


class TerminationReason(StrEnum):
    ALL_RESOLVED = "all_resolved"  # no active people remain
    MAX_STEPS = "max_steps"  # termination.max_steps reached


@dataclass(frozen=True)
class PersonState:
    """One person. ``position`` is the last cell occupied (the exit or the burning cell for resolved people)."""

    id: str
    position: Pos
    status: PersonStatus

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "position": self.position.to_list(), "status": self.status.value}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PersonState":
        return cls(id=d["id"], position=Pos(*d["position"]), status=PersonStatus(d["status"]))


def _sorted_people(people: tuple[PersonState, ...]) -> tuple[PersonState, ...]:
    return tuple(sorted(people, key=lambda p: p.id))


@dataclass(frozen=True)
class SimState:
    """A full, evaluator-private snapshot of simulator state after a step.

    ``rng_state`` holds the position of every simulator-owned RNG stream, so
    the snapshot fully determines the future given the scenario and actions.
    For that reason a ``SimState`` must never reach a decision model.
    """

    step: int
    people: tuple[PersonState, ...]
    burning: frozenset[Pos]
    trapped: frozenset[str]
    rng_state: tuple[tuple[str, int], ...]
    terminated: bool = False
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "people", _sorted_people(self.people))
        object.__setattr__(self, "rng_state", tuple(sorted(self.rng_state)))

    def person(self, person_id: str) -> PersonState:
        for p in self.people:
            if p.id == person_id:
                return p
        raise KeyError(person_id)

    @property
    def active(self) -> tuple[PersonState, ...]:
        return tuple(p for p in self.people if p.status is PersonStatus.ACTIVE)

    @property
    def active_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.active)

    def ids_with_status(self, status: PersonStatus) -> tuple[str, ...]:
        return tuple(p.id for p in self.people if p.status is status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "people": [p.to_dict() for p in self.people],
            "burning": [p.to_list() for p in sorted(self.burning)],
            "trapped": sorted(self.trapped),
            "rng_state": {name: value for name, value in self.rng_state},
            "terminated": self.terminated,
            "termination_reason": None if self.termination_reason is None else self.termination_reason.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SimState":
        reason = d["termination_reason"]
        return cls(
            step=d["step"],
            people=tuple(PersonState.from_dict(p) for p in d["people"]),
            burning=frozenset(Pos(*p) for p in d["burning"]),
            trapped=frozenset(d["trapped"]),
            rng_state=tuple((str(k), int(v)) for k, v in d["rng_state"].items()),
            terminated=d["terminated"],
            termination_reason=None if reason is None else TerminationReason(reason),
        )


class _Snapshot(Protocol):
    """Anything with people and burning cells: a ``SimState`` or an ``Observation``."""

    @property
    def people(self) -> tuple[PersonState, ...]: ...

    @property
    def burning(self) -> frozenset[Pos]: ...


def cell_at(scenario: Scenario, state: _Snapshot, pos: Pos) -> Cell:
    """Classify a cell. Precedence: wall, then fire, then person, then exit, then empty."""
    if pos in scenario.walls:
        return Cell.WALL
    if pos in state.burning:
        return Cell.FIRE
    if any(p.position == pos for p in state.people if p.status is PersonStatus.ACTIVE):
        return Cell.PERSON
    if pos in scenario.exit_positions:
        return Cell.EXIT
    return Cell.EMPTY


def render(scenario: Scenario, state: _Snapshot) -> str:
    """ASCII view for debugging: ``#`` wall, ``F`` fire, ``P`` person, ``E`` exit, ``.`` empty."""
    rows = []
    for r in range(scenario.height):
        rows.append("".join(CELL_GLYPHS[cell_at(scenario, state, Pos(r, c))] for c in range(scenario.width)))
    return "\n".join(rows)
