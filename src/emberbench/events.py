"""Per-step event records and the full run record used for replay and audit.

These records hold every event the metrics module (``metrics.py``) needs:
per-person outcomes, timings, exits used, blocked moves and fire growth.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .canonical import canonical_json, sha256_hex
from .grid import Action, Pos
from .movement import MoveResolution
from .state import SimState


class CasualtyCause(StrEnum):
    ENTERED_FIRE = "entered_fire"  # moved into a cell that was already burning
    FIRE_SPREAD = "fire_spread"  # fire spread onto the person's cell this step


@dataclass(frozen=True)
class Evacuation:
    person_id: str
    exit_id: str
    position: Pos

    def to_dict(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "exit_id": self.exit_id, "position": self.position.to_list()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Evacuation":
        return cls(person_id=d["person_id"], exit_id=d["exit_id"], position=Pos(*d["position"]))


@dataclass(frozen=True)
class Casualty:
    person_id: str
    position: Pos
    cause: CasualtyCause

    def to_dict(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "position": self.position.to_list(), "cause": self.cause.value}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Casualty":
        return cls(person_id=d["person_id"], position=Pos(*d["position"]), cause=CasualtyCause(d["cause"]))


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened in one timestep. ``state`` is the snapshot after the step."""

    step: int
    actions: tuple[tuple[str, Action], ...]
    movements: tuple[MoveResolution, ...]
    ignited: tuple[Pos, ...]
    evacuations: tuple[Evacuation, ...]
    casualties: tuple[Casualty, ...]
    state: SimState

    @property
    def action_map(self) -> dict[str, Action]:
        return dict(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "actions": {pid: a.value for pid, a in self.actions},
            "movements": [m.to_dict() for m in self.movements],
            "ignited": [p.to_list() for p in self.ignited],
            "evacuations": [e.to_dict() for e in self.evacuations],
            "casualties": [c.to_dict() for c in self.casualties],
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StepRecord":
        return cls(
            step=d["step"],
            actions=tuple(sorted((pid, Action(a)) for pid, a in d["actions"].items())),
            movements=tuple(MoveResolution.from_dict(m) for m in d["movements"]),
            ignited=tuple(Pos(*p) for p in d["ignited"]),
            evacuations=tuple(Evacuation.from_dict(e) for e in d["evacuations"]),
            casualties=tuple(Casualty.from_dict(c) for c in d["casualties"]),
            state=SimState.from_dict(d["state"]),
        )


@dataclass(frozen=True)
class RunRecord:
    """A complete, replayable trajectory for one scenario and seed.

    The header binds the record to the simulator semantics
    (``simulator_version``), the model-visible observation contract
    (``observation_version``) and the exact scenario (``scenario_hash``).
    """

    simulator_version: str
    observation_version: str
    schema_version: str
    scenario_id: str
    scenario_hash: str
    seed: int
    initial_state: SimState
    steps: tuple[StepRecord, ...]

    @property
    def final_state(self) -> SimState:
        return self.steps[-1].state if self.steps else self.initial_state

    @property
    def action_log(self) -> list[dict[str, Action]]:
        return [s.action_map for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator_version": self.simulator_version,
            "observation_version": self.observation_version,
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "initial_state": self.initial_state.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        if "observation_version" not in d:
            raise ValueError("run record has no observation_version; it was made before simulator 0.2.0")
        return cls(
            simulator_version=d["simulator_version"],
            observation_version=d["observation_version"],
            schema_version=d["schema_version"],
            scenario_id=d["scenario_id"],
            scenario_hash=d["scenario_hash"],
            seed=d["seed"],
            initial_state=SimState.from_dict(d["initial_state"]),
            steps=tuple(StepRecord.from_dict(s) for s in d["steps"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "RunRecord":
        return cls.from_dict(json.loads(text))
