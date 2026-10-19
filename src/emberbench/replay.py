"""Replay a recorded action log and verify that a run record is authentic."""

from collections.abc import Mapping, Sequence
from typing import Any

from . import SIMULATOR_VERSION
from .errors import InvalidActionError, ReplayMismatchError
from .events import RunRecord
from .observation import OBSERVATION_VERSION
from .scenario import Scenario
from .simulator import Simulator


def replay(scenario: Scenario, action_log: Sequence[Mapping[str, Any]], *, seed: int | None = None) -> RunRecord:
    """Re-run ``scenario`` with the given per-step actions. The log may stop before termination."""
    sim = Simulator(scenario, seed=seed)
    for i, actions in enumerate(action_log, start=1):
        if sim.done:
            raise ReplayMismatchError(f"action log has {len(action_log)} steps but the run terminated at step {i - 1}")
        sim.step(actions)
    return sim.record()


def verify_record(scenario: Scenario, record: RunRecord) -> None:
    """Raise :class:`ReplayMismatchError` unless a fresh replay reproduces ``record`` byte for byte."""
    if record.simulator_version != SIMULATOR_VERSION:
        raise ReplayMismatchError(
            f"record made by simulator {record.simulator_version!r}, this is {SIMULATOR_VERSION!r}"
        )
    if record.observation_version != OBSERVATION_VERSION:
        raise ReplayMismatchError(
            f"record made under observation version {record.observation_version!r}, this is {OBSERVATION_VERSION!r}"
        )
    if record.scenario_hash != scenario.content_hash():
        raise ReplayMismatchError("scenario hash does not match the record")
    sim = Simulator(scenario, seed=record.seed)
    if sim.state != record.initial_state:
        raise ReplayMismatchError("initial state differs")
    for theirs in record.steps:
        if sim.done:
            raise ReplayMismatchError(f"record continues past termination at step {sim.state.step}")
        try:
            ours = sim.step(theirs.action_map)
        except InvalidActionError as exc:
            raise ReplayMismatchError(f"trajectory diverges at step {theirs.step}: {exc}") from exc
        if ours.to_dict() != theirs.to_dict():
            raise ReplayMismatchError(f"trajectory diverges at step {theirs.step}")
    if sim.record().to_json() != record.to_json():
        raise ReplayMismatchError("canonical encodings differ")
