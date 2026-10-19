"""The model-visible observation contract (observation version 1.0).

This module defines everything a decision model may see. The normative
contract is in ``docs/v1-fairness-spec.md``. Everything else is
evaluator-private, in particular:

- the evaluator seed and the ``Scenario.seed`` field,
- the RNG stream positions (``SimState.rng_state``) and future draws,
- the scenario ID, description and content hash,
- the fire, movement and termination configuration,
- the ``trapped`` diagnostic,
- ``SimState``, ``RunRecord`` and the ``Scenario`` and ``Simulator`` objects.

A change to the fields or meaning of :class:`Observation` or
:class:`MapView` must bump :data:`OBSERVATION_VERSION`.
"""

from dataclasses import dataclass
from typing import Any

from .grid import Action, Pos
from .scenario import ExitSpec, Scenario
from .state import PersonState, PersonStatus, SimState

OBSERVATION_VERSION = "1.0"

# Every active person may submit any of these actions every step. A move
# that cannot happen is still legal: it resolves to a ``blocked_*`` outcome.
ALLOWED_ACTIONS: tuple[Action, ...] = (Action.STAY, Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)


@dataclass(frozen=True)
class MapView:
    """The static map: grid size, walls and exits. It does not change during a run."""

    width: int
    height: int
    obstacles: tuple[Pos, ...]
    exits: tuple[ExitSpec, ...]

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "MapView":
        """Copy only the static geometry. Seed, IDs and configuration stay behind."""
        return cls(
            width=scenario.width,
            height=scenario.height,
            obstacles=scenario.obstacles,
            exits=tuple(ExitSpec(e.id, e.position) for e in scenario.exits),
        )

    @property
    def walls(self) -> frozenset[Pos]:
        return frozenset(self.obstacles)

    @property
    def exit_positions(self) -> dict[Pos, str]:
        return {e.position: e.id for e in self.exits}

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": [p.to_list() for p in self.obstacles],
            "exits": [{"id": e.id, "position": e.position.to_list()} for e in self.exits],
        }


@dataclass(frozen=True)
class Observation:
    """What a decision model sees before it acts (phase 1).

    The world is fully observable in V1: every burning cell is visible. The
    observation is a pure function of the step number, the static map, the
    people and the burning cells. It holds no reference to the scenario, the
    simulator or any RNG.
    """

    step: int
    map: MapView
    people: tuple[PersonState, ...]
    burning: frozenset[Pos]
    allowed_actions: tuple[Action, ...] = ALLOWED_ACTIONS

    @classmethod
    def from_state(cls, map_view: MapView, state: SimState) -> "Observation":
        return cls(step=state.step, map=map_view, people=state.people, burning=state.burning)

    @property
    def active(self) -> tuple[PersonState, ...]:
        return tuple(p for p in self.people if p.status is PersonStatus.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_version": OBSERVATION_VERSION,
            "step": self.step,
            "map": self.map.to_dict(),
            "people": [p.to_dict() for p in self.people],
            "burning": [p.to_list() for p in sorted(self.burning)],
            "allowed_actions": [a.value for a in self.allowed_actions],
        }
