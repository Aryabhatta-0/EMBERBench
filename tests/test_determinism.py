"""Determinism and replay guarantees: the core benchmark contract."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FIXTURE_NAMES, fixture_scenario, greedy_exit_policy
from emberbench import (
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

    with pytest.raises(ReplayMismatchError, match="scenario hash"):
        verify_record(fixture_scenario("one_person_one_exit"), original)


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
    print(name, run_episode(s, greedy_exit_policy(s)).digest())
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
        f"{n} {run_episode(fixture_scenario(n), greedy_exit_policy(fixture_scenario(n))).digest()}\n"
        for n in FIXTURE_NAMES
    )
    assert outputs[0].replace("\r\n", "\n") == local


# Golden digests pin the exact rules of SIMULATOR_VERSION 0.1.0. If a change
# alters any of these, the simulator output changed: bump SIMULATOR_VERSION
# and update the digests on purpose.
GOLDEN_DIGESTS: dict[str, str] = {
    "blocked_shortest_route": "bab407562a001ae043c5c4e250dec1fba50afa45253a7ed00fd411543156de26",
    "fire_threatening_exit": "f6720d533dbcf63eb7d5cd10bd3e7760274909bc263a77d334fd102dc54fb576",
    "multiple_exits": "8713a01cfd931063761a8a5458b292b9ab28a928507b4de8a89c91802db105b5",
    "narrow_corridor": "c59376c210d4a2da90bdbf31576851de5be480f5c8e4e9c33be35f1936905903",
    "one_person_one_exit": "d5168c37c8f68b62042da581d777300368d1a75f59f873fba03e42051b0624d4",
    "stochastic_room": "5c0af28522023771eddc84aa6ebfe4e20c938a99ee9a9c6de64e457f62c66305",
}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_golden_trajectories(name: str) -> None:
    scenario = fixture_scenario(name)
    digest = run_episode(scenario, greedy_exit_policy(scenario)).digest()
    assert digest == GOLDEN_DIGESTS[name], f"{name}: {digest}"
