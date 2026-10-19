"""Determinism and replay guarantees: the core benchmark contract."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FIXTURE_NAMES, fixture_scenario, greedy_exit_policy
from emberbench import (
    OBSERVATION_VERSION,
    Action,
    ReplayMismatchError,
    RunRecord,
    Simulator,
    replay,
    run_episode,
    verify_record,
)
from emberbench.rng import SplitMix64

ROOT = Path(__file__).resolve().parents[1]


def scripted_actions(scenario_seed: int, sim: Simulator) -> dict[str, Action]:
    """A fixed pseudo-random action script. It also exercises conflicts, swaps and walls."""
    rng = SplitMix64((scenario_seed * 7919 + sim.state.step) & (2**64 - 1))
    choices = list(Action)
    return {p.id: choices[rng.next_u64() % len(choices)] for p in sim.observe().active}


def run_scripted(name: str, seed: int | None = None) -> RunRecord:
    scenario = fixture_scenario(name)
    sim = Simulator(scenario, seed=seed)
    while not sim.done:
        sim.step(scripted_actions(sim.seed, sim))
    return sim.record()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_two_independent_instances_produce_identical_trajectories(name: str) -> None:
    scenario_a = fixture_scenario(name)
    scenario_b = fixture_scenario(name)  # parsed separately; no shared objects
    sim_a, sim_b = Simulator(scenario_a), Simulator(scenario_b)
    while not sim_a.done:
        assert sim_a.observe() == sim_b.observe()
        actions = scripted_actions(sim_a.seed, sim_a)
        rec_a = sim_a.step(actions)
        rec_b = sim_b.step(dict(reversed(list(actions.items()))))  # insertion order must not matter
        assert rec_a.to_dict() == rec_b.to_dict()
    assert sim_b.done
    assert sim_a.record().to_json().encode() == sim_b.record().to_json().encode()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_same_seed_same_actions_same_trajectory(name: str) -> None:
    first = run_scripted(name)
    second = run_scripted(name)
    assert first.digest() == second.digest()
    assert first == second


def test_different_seeds_change_stochastic_fire_only_through_rng() -> None:
    digests = {run_scripted("stochastic_room", seed).digest() for seed in range(8)}
    assert len(digests) > 1
    for seed in range(4):
        assert run_scripted("stochastic_room", seed).digest() == run_scripted("stochastic_room", seed).digest()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_replay_from_action_log_reproduces_record(name: str) -> None:
    original = run_scripted(name)
    again = replay(fixture_scenario(name), original.action_log, seed=original.seed)
    assert again.to_json() == original.to_json()
    verify_record(fixture_scenario(name), original)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_run_record_serialisation_round_trip(name: str) -> None:
    original = run_scripted(name)
    text = original.to_json()
    restored = RunRecord.from_json(text)
    assert restored == original
    assert restored.to_json() == text
    verify_record(fixture_scenario(name), restored)


def test_tampered_records_are_detected() -> None:
    scenario = fixture_scenario("stochastic_room")
    original = run_scripted("stochastic_room")
    data = original.to_dict()
    data["steps"][2]["ignited"] = data["steps"][2]["ignited"][1:]
    with pytest.raises(ReplayMismatchError, match="step 3"):
        verify_record(scenario, RunRecord.from_dict(data))

    data = original.to_dict()
    data["seed"] = original.seed + 1
    with pytest.raises(ReplayMismatchError):
        verify_record(scenario, RunRecord.from_dict(data))

    data = original.to_dict()
    data["simulator_version"] = "9.9.9"
    with pytest.raises(ReplayMismatchError, match="simulator"):
        verify_record(scenario, RunRecord.from_dict(data))

    data = original.to_dict()
    data["observation_version"] = "0.9"
    with pytest.raises(ReplayMismatchError, match="observation version"):
        verify_record(scenario, RunRecord.from_dict(data))

    with pytest.raises(ReplayMismatchError, match="scenario hash"):
        verify_record(fixture_scenario("one_person_one_exit"), original)


def test_run_record_header_binds_the_observation_contract() -> None:
    record = run_scripted("stochastic_room")
    assert record.observation_version == OBSERVATION_VERSION
    assert list(record.to_dict())[:3] == ["simulator_version", "observation_version", "schema_version"]
    assert '"observation_version":"' + OBSERVATION_VERSION + '"' in record.to_json()
    assert RunRecord.from_json(record.to_json()) == record
    old_format = record.to_dict()
    del old_format["observation_version"]
    with pytest.raises(ValueError, match="observation_version"):
        RunRecord.from_dict(old_format)


def test_replay_rejects_actions_after_termination() -> None:
    scenario = fixture_scenario("one_person_one_exit")
    log = [{"P1": "east"}] * 5
    with pytest.raises(ReplayMismatchError, match="terminated"):
        replay(scenario, log)


def test_simulators_do_not_share_state() -> None:
    scenario = fixture_scenario("stochastic_room")
    lone = run_scripted("stochastic_room")
    # Interleave two runs step by step. The result must not change.
    a, b = Simulator(scenario), Simulator(scenario, seed=123)
    while not a.done:
        a.step(scripted_actions(a.seed, a))
        if not b.done:
            b.step(scripted_actions(b.seed, b))
    assert a.record().digest() == lone.digest()


_SUBPROCESS_SCRIPT = """
import sys
sys.path[:0] = [{src!r}, {tests!r}]
from conftest import FIXTURE_NAMES, fixture_scenario, greedy_exit_policy
from emberbench import run_episode
for name in FIXTURE_NAMES:
    s = fixture_scenario(name)
    print(name, run_episode(s, greedy_exit_policy).digest())
"""


def test_results_identical_across_processes_and_hash_seeds() -> None:
    """String hashing is randomised per process. Output must not depend on set or dict iteration order."""
    script = _SUBPROCESS_SCRIPT.format(src=str(ROOT / "src"), tests=str(ROOT / "tests"))
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    local = "".join(
        f"{n} {run_episode(fixture_scenario(n), greedy_exit_policy).digest()}\n"
        for n in FIXTURE_NAMES
    )
    assert outputs[0].replace("\r\n", "\n") == local


# Golden digests pin the exact rules of SIMULATOR_VERSION 0.2.0. If a change
# alters any of these, the simulator output changed: bump SIMULATOR_VERSION
# and update the digests on purpose.
#
# History: 0.1.0 -> 0.2.0 added "observation_version" to the RunRecord header
# and changed "simulator_version". The dynamics did not change: with the new
# key removed and the version set back to "0.1.0", every record hashes to its
# 0.1.0 digest.
GOLDEN_DIGESTS: dict[str, str] = {
    "blocked_shortest_route": "af2b6e241b1f13bfd41af0d7f54373d8e3e9e0566dc906b935e7d7baea17bdbe",
    "fire_threatening_exit": "d394b40c60da27c8f43963ff9f2a6e855705b593aa808c6fa09af8cc9dbe9878",
    "multiple_exits": "adb60f12ea5b3d5bda0b485b22a1bc4387bcd596730977dfe47d1af78213ab28",
    "narrow_corridor": "ab5657d632eb6c3982be5538240785a32ac3831501fe193af8a5a83b82115cf4",
    "one_person_one_exit": "a91d691f64731ee2d5b5184b95d1313bbd294d6fbd06d44ac91bea2f95a3c990",
    "stochastic_room": "9ee082d50fc30fb2f6c8617c8d55053ff30af958c66eb6eb22f28fb2db6e3a84",
}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_golden_trajectories(name: str) -> None:
    scenario = fixture_scenario(name)
    digest = run_episode(scenario, greedy_exit_policy).digest()
    assert digest == GOLDEN_DIGESTS[name], f"{name}: {digest}"
