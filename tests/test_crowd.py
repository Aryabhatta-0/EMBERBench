"""Crowd coordination diagnostics (``crowd.py``) and the frozen V1 queue and throughput rules.

The diagnostics only observe. These tests pin the V1 crowd rules through
them and check that every diagnostic reconciles with the RunRecord and with
``compute_episode_metrics``. They never change a rule to make a test pass.
"""

import dataclasses
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from conftest import FIXTURE_NAMES, crowd_stress_scenario, fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    CROWD_DIAGNOSTICS_VERSION,
    OBSERVATION_VERSION,
    SCHEMA_VERSION,
    SIMULATOR_VERSION,
    Action,
    CrowdDiagnostics,
    MetricsInputError,
    MoveOutcome,
    Observation,
    PersonStatus,
    Pos,
    RunRecord,
    Scenario,
    Simulator,
    TerminationConfig,
    TerminationReason,
    compute_crowd_diagnostics,
    compute_episode_metrics,
    load_scenario,
    run_episode,
    verify_record,
)
from emberbench.crowd import STATIC_BLOCK_OUTCOMES, per_exit_static_distances
from emberbench.movement import resolve_movements
from emberbench.rng import SplitMix64

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "scenarios" / "public" / "v1"
CROWD_IDS = ("v1-pub-02-single-exit-dense", "v1-pub-03-corridor-bottleneck", "v1-pub-09-crowd-100-split-flow")
STRESS_SIZES = (50, 100, 200)

N, S, E, W, STAY = Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.STAY

Policy = Callable[[Observation], Mapping[str, Any]]


def public(scenario_id: str) -> Scenario:
    return load_scenario(SUITE_DIR / f"{scenario_id}.json")


def scripted_policy(seed: int) -> Policy:
    """Pseudo-random actions. They hit walls, boundaries, swaps, rotations and conflicts."""
    choices = list(Action)

    def policy(obs: Observation) -> dict[str, Action]:
        rng = SplitMix64((seed * 7919 + obs.step) & (2**64 - 1))
        return {p.id: choices[rng.next_u64() % len(choices)] for p in obs.active}

    return policy


def run_steps(scenario: Scenario, actions: list[dict[str, Action]]) -> RunRecord:
    """Step a scenario through a fixed action script. The scenario must terminate at the end."""
    sim = Simulator(scenario)
    for step_actions in actions:
        sim.step(step_actions)
    assert sim.done
    return sim.record()


def one_step(rows: list[str], actions: dict[str, Action]) -> tuple[RunRecord, CrowdDiagnostics]:
    scenario = layout(rows, termination=TerminationConfig(max_steps=1))
    record = run_steps(scenario, [actions])
    return record, compute_crowd_diagnostics(scenario, record)


# --------------------------------------------------------------------------- independent oracle


def oracle_step(step: Any, exit_ids: list[str]) -> dict[str, Any]:
    """Recompute the step signals straight from the recorded movements, without crowd.py."""
    moves = step.movements
    counts = Counter(m.outcome for m in moves)
    contenders = Counter(m.target for m in moves if m.outcome is not MoveOutcome.STAYED and m.outcome not in STATIC_BLOCK_OUTCOMES)
    evac = Counter(e.exit_id for e in step.evacuations)
    return {
        "active_people": len(moves),
        "stays": counts[MoveOutcome.STAYED],
        "movement_attempts": len(moves) - counts[MoveOutcome.STAYED],
        "successful_moves": counts[MoveOutcome.MOVED],
        "blocked_by_outcome": {o.value: counts[o] for o in MoveOutcome if o.value.startswith("blocked_")},
        "congestion_blocks": counts[MoveOutcome.BLOCKED_OCCUPIED] + counts[MoveOutcome.BLOCKED_CONFLICT] + counts[MoveOutcome.BLOCKED_CYCLE],
        "contending_moves": sum(contenders.values()),
        "contested_cells": sum(1 for n in contenders.values() if n > 1),
        "max_contenders": max(contenders.values(), default=0),
        "evacuations_by_exit": {eid: evac[eid] for eid in exit_ids},
    }


def assert_reconciles(scenario: Scenario, record: RunRecord, diag: CrowdDiagnostics) -> None:
    """Step, cell, exit and episode diagnostics reconcile with the record and with the metrics."""
    exit_ids = [e.id for e in scenario.exits]
    assert diag.episode_steps == len(record.steps) == len(diag.steps)
    cumulative = Counter[str]()
    for step, signals in zip(record.steps, diag.steps, strict=True):
        expected = oracle_step(step, exit_ids)
        got = signals.to_dict()
        assert signals.step == step.step
        for key, value in expected.items():
            assert got[key] == value, (step.step, key)
        assert signals.successful_moves + signals.blocked_moves == signals.movement_attempts
        cumulative.update(e.exit_id for e in step.evacuations)
        assert got["cumulative_evacuations_by_exit"] == {eid: cumulative[eid] for eid in exit_ids}
        demand = signals.nearest_exit_demand_proxy
        assert sum(n for _, n in demand.per_exit) + demand.tied + demand.unreachable == signals.active_people

    # Cells reconcile with the step totals and with the raw movements.
    cells = diag.cells
    assert [c.cell for c in cells] == sorted(c.cell for c in cells)  # row-major, no duplicates
    assert len({c.cell for c in cells}) == len(cells)
    for c in cells:
        assert c.attempted_entries == c.successful_entries + c.congestion_blocks > 0
        assert c.congestion_blocks == c.blocked_occupied + c.blocked_conflict + c.blocked_cycle
        assert c.peak_contenders >= 1
        assert (c.contention_steps > 0) == (c.peak_contenders > 1)
        assert c.exit_id == {e.position: e.id for e in scenario.exits}.get(c.cell)
    t = diag.totals
    assert sum(c.attempted_entries for c in cells) == t.contending_moves
    assert sum(c.successful_entries for c in cells) == t.successful_moves
    assert sum(c.congestion_blocks for c in cells) == t.congestion_blocks
    assert sum(c.contention_steps for c in cells) == t.contested_cell_steps
    assert max((c.peak_contenders for c in cells), default=0) == t.peak_contenders
    static = sum(dict(t.blocked_by_outcome)[o.value] for o in STATIC_BLOCK_OUTCOMES)
    assert t.contending_moves == t.movement_attempts - static

    # Episode totals equal the canonical metrics.
    metrics = compute_episode_metrics(scenario, record)
    mv = metrics.movement
    assert (t.decisions, t.stays, t.movement_attempts, t.successful_moves, t.blocked_moves, t.congestion_blocks) == (
        mv.decisions,
        mv.stay_actions,
        mv.movement_attempts,
        mv.successful_moves,
        mv.blocked_moves,
        mv.congestion_blocks,
    )
    assert t.blocked_by_outcome == mv.blocked_by_outcome
    assert tuple((e.exit_id, e.evacuations) for e in diag.exits.exits) == metrics.exits.evacuations_per_exit
    assert diag.exits.total_evacuations == metrics.safety.evacuated
    for e in diag.exits.exits:
        assert e.steps_with_evacuation == e.evacuations  # follows from one person per exit cell per step
        assert e.peak_step_evacuations <= 1
    assert diag.exits.peak_step_evacuations <= diag.exits.exit_count


# --------------------------------------------------------------------------- schema and validation


def test_result_header_binds_versions_and_record() -> None:
    scenario = fixture_scenario("narrow_corridor")
    record = run_episode(scenario, greedy_exit_policy)
    diag = compute_crowd_diagnostics(scenario, record)
    assert diag.crowd_diagnostics_version == CROWD_DIAGNOSTICS_VERSION == "1.0"
    assert (diag.simulator_version, diag.observation_version, diag.schema_version) == (
        SIMULATOR_VERSION,
        OBSERVATION_VERSION,
        SCHEMA_VERSION,
    )
    assert (diag.scenario_id, diag.scenario_hash, diag.seed) == (scenario.scenario_id, scenario.content_hash(), record.seed)
    assert diag.record_digest == record.digest()


def test_frozen_versions_are_unchanged() -> None:
    # Phase 7 only observes. It must not move the simulator, observation or schema contracts.
    assert (SIMULATOR_VERSION, OBSERVATION_VERSION, SCHEMA_VERSION) == ("0.2.0", "1.0", "1.0")


def test_result_is_immutable_and_inputs_are_untouched() -> None:
    scenario = public("v1-pub-03-corridor-bottleneck")
    record = run_episode(scenario, greedy_exit_policy)
    before = (record.to_json(), scenario.content_hash())
    diag = compute_crowd_diagnostics(scenario, record)
    assert (record.to_json(), scenario.content_hash()) == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        diag.episode_steps = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        diag.cells[0].congestion_blocks = 0  # type: ignore[misc]


def test_serialisation_is_canonical_json() -> None:
    scenario = public("v1-pub-09-crowd-100-split-flow")
    diag = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    text = diag.to_json()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    doc = json.loads(text)
    assert [s["step"] for s in doc["steps"]] == list(range(1, diag.episode_steps + 1))
    assert [c["cell"] for c in doc["cells"]] == sorted(c["cell"] for c in doc["cells"])
    assert [e["exit_id"] for e in doc["exits"]["exits"]] == ["E1", "E2", "E3"]
    assert "score" not in text


def _record_doc() -> tuple[Scenario, dict[str, Any]]:
    scenario = public("v1-pub-03-corridor-bottleneck")
    return scenario, run_episode(scenario, greedy_exit_policy).to_dict()


def test_rejects_wrong_argument_types() -> None:
    scenario, doc = _record_doc()
    with pytest.raises(TypeError):
        compute_crowd_diagnostics(doc, RunRecord.from_dict(doc))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_crowd_diagnostics(scenario, doc)  # type: ignore[arg-type]


def test_rejects_a_record_for_another_scenario() -> None:
    _, doc = _record_doc()
    with pytest.raises(MetricsInputError, match="scenario_id"):
        compute_crowd_diagnostics(public("v1-pub-02-single-exit-dense"), RunRecord.from_dict(doc))


def test_rejects_an_incomplete_record() -> None:
    scenario, doc = _record_doc()
    doc["steps"] = doc["steps"][:-1]
    with pytest.raises(MetricsInputError, match="termination"):
        compute_crowd_diagnostics(scenario, RunRecord.from_dict(doc))


def test_rejects_a_relabelled_block_reason_that_only_replay_can_catch() -> None:
    # Swapping blocked_conflict for blocked_occupied keeps every structural
    # invariant. Only the simulator replay sees that it is false.
    scenario, doc = _record_doc()
    for step in doc["steps"]:
        for m in step["movements"]:
            if m["outcome"] == MoveOutcome.BLOCKED_CONFLICT.value:
                m["outcome"] = MoveOutcome.BLOCKED_OCCUPIED.value
                break
        else:
            continue
        break
    else:
        pytest.fail("fixture run has no conflict")
    with pytest.raises(MetricsInputError, match="replay"):
        compute_crowd_diagnostics(scenario, RunRecord.from_dict(doc))


def test_rejects_a_record_from_another_simulator_version() -> None:
    scenario, doc = _record_doc()
    doc["simulator_version"] = "0.1.0"
    with pytest.raises(MetricsInputError, match="simulator"):
        compute_crowd_diagnostics(scenario, RunRecord.from_dict(doc))


# --------------------------------------------------------------------------- frozen queue and throughput rules


def test_one_exit_cell_evacuates_at_most_one_person_per_step() -> None:
    scenario = layout(["abcdE"])
    record = run_episode(scenario, greedy_exit_policy)
    diag = compute_crowd_diagnostics(scenario, record)
    assert [dict(s.evacuations_by_exit)["E1"] for s in diag.steps] == [1, 1, 1, 1]
    assert [dict(s.cumulative_evacuations_by_exit)["E1"] for s in diag.steps] == [1, 2, 3, 4]
    (exit_load,) = diag.exits.exits
    assert (exit_load.evacuations, exit_load.peak_step_evacuations, exit_load.first_evacuation_step, exit_load.last_evacuation_step) == (4, 1, 1, 4)
    assert diag.totals.congestion_blocks == 0


def test_two_people_converging_on_one_exit_cell_leave_one_per_step() -> None:
    record, diag = one_step(["a.", "Eb"], {"Pa": S, "Pb": W})
    assert [e.person_id for e in record.steps[0].evacuations] == ["Pa"]
    (signals,) = diag.steps
    assert dict(signals.evacuations_by_exit) == {"E1": 1}
    assert (signals.contested_cells, signals.max_contenders) == (1, 2)
    cell = diag.cell(Pos(1, 0))
    assert cell is not None and cell.exit_id == "E1"
    assert (cell.attempted_entries, cell.successful_entries, cell.blocked_conflict) == (2, 1, 1)


def test_two_distinct_exit_cells_evacuate_two_people_in_one_step() -> None:
    scenario = layout(["ab", "EE"])
    record = run_episode(scenario, greedy_exit_policy)
    diag = compute_crowd_diagnostics(scenario, record)
    (signals,) = diag.steps
    assert dict(signals.evacuations_by_exit) == {"E1": 1, "E2": 1}
    assert diag.exits.peak_step_evacuations == 2
    assert [e.peak_step_evacuations for e in diag.exits.exits] == [1, 1]


def test_a_following_chain_advances_without_gaps() -> None:
    scenario = layout(["abcd.E"])
    record = run_episode(scenario, greedy_exit_policy)
    diag = compute_crowd_diagnostics(scenario, record)
    # Step 1: the whole line moves east together; nobody is blocked by the person in front.
    first = record.steps[0]
    assert {m.outcome for m in first.movements} == {MoveOutcome.MOVED}
    assert [p.position for p in first.state.people] == [Pos(0, 1), Pos(0, 2), Pos(0, 3), Pos(0, 4)]
    assert diag.steps[0].successful_moves == 4
    # Then one person leaves per step, with no empty step in between.
    assert [dict(s.evacuations_by_exit)["E1"] for s in diag.steps] == [0, 1, 1, 1, 1]
    assert diag.totals.blocked_moves == diag.totals.contested_cell_steps == 0


def test_same_target_losers_are_blocked_conflict() -> None:
    _, diag = one_step(["...E", "a.b.", ".c.."], {"Pa": E, "Pb": W, "Pc": N})
    (signals,) = diag.steps
    assert (signals.contested_cells, signals.max_contenders, signals.congestion_blocks) == (1, 3, 2)
    assert dict(signals.blocked_by_outcome)["blocked_conflict"] == 2
    centre = diag.cell(Pos(1, 1))
    assert centre is not None
    assert (centre.attempted_entries, centre.successful_entries, centre.blocked_conflict, centre.contention_steps, centre.peak_contenders) == (
        3,
        1,
        2,
        1,
        3,
    )


def test_non_moving_occupants_produce_blocked_occupied() -> None:
    record, diag = one_step(["a.E", "cb.", ".d."], {"Pa": STAY, "Pb": W, "Pc": N, "Pd": N})
    outcomes = {m.person_id: m.outcome for m in record.steps[0].movements}
    # Pb and Pc target cells held by people who do not leave them.
    assert outcomes == {
        "Pa": MoveOutcome.STAYED,
        "Pb": MoveOutcome.BLOCKED_OCCUPIED,
        "Pc": MoveOutcome.BLOCKED_OCCUPIED,
        "Pd": MoveOutcome.BLOCKED_OCCUPIED,
    }
    occupied = {c.cell: c.blocked_occupied for c in diag.cells}
    assert occupied == {Pos(0, 0): 1, Pos(1, 0): 1, Pos(1, 1): 1}
    (signals,) = diag.steps
    assert (signals.stays, signals.movement_attempts, signals.congestion_blocks) == (1, 3, 3)


def test_two_movers_into_a_stayer_are_both_occupied_and_contend() -> None:
    _, diag = one_step(["..E", "abc"], {"Pa": E, "Pb": STAY, "Pc": W})
    centre = diag.cell(Pos(1, 1))
    assert centre is not None
    assert (centre.attempted_entries, centre.blocked_occupied, centre.blocked_conflict) == (2, 2, 0)
    assert (centre.contention_steps, centre.peak_contenders) == (1, 2)


def test_swaps_and_rotations_produce_blocked_cycle() -> None:
    _, swap = one_step(["ab.E"], {"Pa": E, "Pb": W})
    assert {c.cell: c.blocked_cycle for c in swap.cells} == {Pos(0, 0): 1, Pos(0, 1): 1}
    assert swap.steps[0].contested_cells == 0  # a swap is not contention for one cell
    _, rotation = one_step(["ab.E", "dc.."], {"Pa": E, "Pb": S, "Pc": W, "Pd": N})
    assert {c.cell: c.blocked_cycle for c in rotation.cells} == {Pos(0, 0): 1, Pos(0, 1): 1, Pos(1, 0): 1, Pos(1, 1): 1}
    assert rotation.totals.congestion_blocks == dict(rotation.totals.blocked_by_outcome)["blocked_cycle"] == 4


def test_static_blocks_are_not_contenders() -> None:
    _, diag = one_step(["#a.E"], {"Pa": W})
    (signals,) = diag.steps
    assert dict(signals.blocked_by_outcome)["blocked_wall"] == 1
    assert (signals.contending_moves, signals.congestion_blocks, diag.cells) == (0, 0, ())
    _, edge = one_step(["a.E"], {"Pa": N})
    assert dict(edge.steps[0].blocked_by_outcome)["blocked_boundary"] == 1 and edge.cells == ()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
@pytest.mark.parametrize("policy_name", ["greedy", "scripted"])
def test_diagnostics_reconcile_on_every_fixture(name: str, policy_name: str) -> None:
    scenario = fixture_scenario(name)
    policy = greedy_exit_policy if policy_name == "greedy" else scripted_policy(scenario.seed)
    record = run_episode(scenario, policy)
    assert_reconciles(scenario, record, compute_crowd_diagnostics(scenario, record))


@pytest.mark.parametrize("scenario_id", CROWD_IDS)
@pytest.mark.parametrize("policy_name", ["greedy", "scripted"])
def test_diagnostics_reconcile_on_crowd_scenarios(scenario_id: str, policy_name: str) -> None:
    scenario = public(scenario_id)
    policy = greedy_exit_policy if policy_name == "greedy" else scripted_policy(scenario.seed)
    record = run_episode(scenario, policy)
    assert_reconciles(scenario, record, compute_crowd_diagnostics(scenario, record))


# --------------------------------------------------------------------------- nearest-exit demand proxy


def test_demand_proxy_uses_wall_only_path_distance_not_manhattan() -> None:
    # Pa is 2 cells from E1 by Manhattan distance, but the wall forces a 10-cell path.
    # E2 is 4 cells away, so the proxy counts Pa for E2.
    scenario = layout(["E....", "####.", "a...E"], termination=TerminationConfig(max_steps=1))
    dist = per_exit_static_distances(scenario)
    assert (dist["E1"][Pos(2, 0)], dist["E2"][Pos(2, 0)]) == (10, 4)
    diag = compute_crowd_diagnostics(scenario, run_steps(scenario, [{"Pa": STAY}]))
    assert dict(diag.steps[0].nearest_exit_demand_proxy.per_exit) == {"E1": 0, "E2": 1}
    assert [e.initial_demand_proxy for e in diag.exits.exits] == [0, 1]


def test_demand_proxy_counts_ties_and_unreachable_separately() -> None:
    scenario = layout(["E.a.E", "#####", "#b#..", "###.E"], termination=TerminationConfig(max_steps=1))
    diag = compute_crowd_diagnostics(scenario, run_steps(scenario, [{"Pa": STAY, "Pb": STAY}]))
    demand = diag.steps[0].nearest_exit_demand_proxy
    assert (dict(demand.per_exit), demand.tied, demand.unreachable) == ({"E1": 0, "E2": 0, "E3": 0}, 1, 1)
    assert (diag.exits.initial_demand_tied, diag.exits.initial_demand_unreachable) == (1, 1)
    assert (diag.exits.demand_tied_person_steps, diag.exits.demand_unreachable_person_steps) == (1, 1)


def test_demand_proxy_is_independent_of_actual_exit_use() -> None:
    # Pa's nearest exit is E2, but it walks to E1. The proxy and the measured use disagree, as they may.
    scenario = layout(["E..aE"])
    record = run_steps(scenario, [{"Pa": W}, {"Pa": W}, {"Pa": W}])
    diag = compute_crowd_diagnostics(scenario, record)
    assert [e.initial_demand_proxy for e in diag.exits.exits] == [0, 1]
    assert [e.evacuations for e in diag.exits.exits] == [1, 0]
    # Person-steps follow the person: E2 nearest at step 1, tie at step 2, E1 nearest at step 3.
    assert [e.demand_proxy_person_steps for e in diag.exits.exits] == [1, 1]
    assert diag.exits.demand_tied_person_steps == 1


# --------------------------------------------------------------------------- canonical crowd scenarios


def _hottest(diag: CrowdDiagnostics) -> Any:
    return min(diag.cells, key=lambda c: (-c.congestion_blocks, c.cell))


def test_single_exit_dense_hotspots_cluster_on_the_exit_approach() -> None:
    scenario = public("v1-pub-02-single-exit-dense")
    diag = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    exit_cell = scenario.exits[0].position
    # The single exit is the throughput limit: it is used on almost every step, never twice in one step.
    (e1,) = diag.exits.exits
    assert e1.peak_step_evacuations == 1
    assert e1.steps_with_evacuation == e1.evacuations == 34
    assert e1.last_evacuation_step is not None and e1.first_evacuation_step is not None
    assert e1.last_evacuation_step - e1.first_evacuation_step + 1 - e1.steps_with_evacuation <= 3
    # Congestion is heavy and sustained, and its hottest cells sit in the open column before the exit.
    assert diag.totals.congestion_blocks > diag.totals.successful_moves
    assert diag.totals.steps_with_congestion_blocks >= 30
    top = sorted(diag.cells, key=lambda c: (-c.congestion_blocks, c.cell))[:4]
    assert all(c.static_exit_distance is not None and c.static_exit_distance <= 6 for c in top)
    assert sum(1 for c in top if c.cell.col == exit_cell.col - 1) >= 3


def test_corridor_bottleneck_hotspot_is_the_corridor_mouth() -> None:
    scenario = public("v1-pub-03-corridor-bottleneck")
    diag = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    hottest = _hottest(diag)
    # Row 3 is the one-cell corridor from column 7 to the exit; (3, 6) is the last room cell before it.
    assert hottest.cell == Pos(3, 6)
    assert hottest.peak_contenders == diag.totals.peak_contenders == 3
    assert hottest.congestion_blocks * 2 > diag.totals.congestion_blocks  # most congestion is at the mouth
    corridor = [c for c in diag.cells if c.cell.row == 3 and c.cell.col >= 7]
    assert corridor and all(c.congestion_blocks == 0 for c in corridor)  # single file inside the corridor


def test_split_flow_exit_load_and_hotspots_are_measured() -> None:
    scenario = public("v1-pub-09-crowd-100-split-flow")
    diag = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    loads = {e.exit_id: e for e in diag.exits.exits}
    # Proxy at step 0: the block sits nearest the west exit E2; nine people are tied.
    assert {k: v.initial_demand_proxy for k, v in loads.items()} == {"E1": 10, "E2": 81, "E3": 0}
    assert diag.exits.initial_demand_tied == 9
    assert diag.exits.initial_demand_unreachable == 0
    # Measured use under the greedy baseline follows the same imbalance.
    assert {k: v.evacuations for k, v in loads.items()} == {"E1": 15, "E2": 64, "E3": 0}
    assert all(v.peak_step_evacuations <= 1 for v in loads.values())
    # The hottest cells lie on the approach to E2 at (8, 0).
    top = sorted(diag.cells, key=lambda c: (-c.congestion_blocks, c.cell))[:4]
    assert all(c.static_exit_distance is not None and c.static_exit_distance <= 5 for c in top)
    assert all(c.cell.row in (7, 8, 9) and c.cell.col <= 5 for c in top)
    assert diag.totals.peak_contested_cells >= 10


# --------------------------------------------------------------------------- large-population stress


# Pinned outcomes of the greedy baseline on the synthetic stress rooms. A change
# here means the movement rules, the simulator or the diagnostics changed.
STRESS_EXPECTED: dict[int, tuple[int, int, int]] = {
    # population: (episode_steps, evacuated, congestion_blocks)
    50: (27, 50, 232),
    100: (38, 100, 922),
    200: (65, 200, 4115),
}


@pytest.mark.parametrize("population", STRESS_SIZES)
def test_stress_greedy_evacuation_is_correct_and_bounded(population: int) -> None:
    scenario = crowd_stress_scenario(population)
    assert len(scenario.people) == population
    record = run_episode(scenario, greedy_exit_policy)
    verify_record(scenario, record)
    final = record.final_state
    assert final.termination_reason is TerminationReason.ALL_RESOLVED
    assert all(p.status is PersonStatus.EVACUATED for p in final.people)
    diag = compute_crowd_diagnostics(scenario, record)
    assert_reconciles(scenario, record, diag)
    # Four single-cell exits take at most four people per step.
    lower_bound = -(-population // len(scenario.exits))
    assert lower_bound <= diag.episode_steps <= scenario.termination.max_steps
    assert all(sum(n for _, n in s.evacuations_by_exit) <= 4 for s in diag.steps)
    assert (diag.episode_steps, diag.exits.total_evacuations, diag.totals.congestion_blocks) == STRESS_EXPECTED[population]
    assert diag.totals.peak_contenders <= 4  # a cell has four neighbours


@pytest.mark.parametrize("population", STRESS_SIZES)
def test_stress_random_actions_replay_and_reconcile(population: int) -> None:
    scenario = dataclasses.replace(crowd_stress_scenario(population), termination=TerminationConfig(max_steps=30))
    record = run_episode(scenario, scripted_policy(population))
    verify_record(scenario, record)
    assert len(record.steps) == 30
    diag = compute_crowd_diagnostics(scenario, record)
    assert_reconciles(scenario, record, diag)
    by_outcome = dict(diag.totals.blocked_by_outcome)
    # Random actions in a dense block exercise every person-to-person block reason.
    assert by_outcome["blocked_occupied"] > 0 and by_outcome["blocked_conflict"] > 0 and by_outcome["blocked_cycle"] > 0


@pytest.mark.parametrize("population", STRESS_SIZES)
def test_stress_movement_resolution_is_order_independent_and_collision_free(population: int) -> None:
    scenario = crowd_stress_scenario(population)
    positions = {p.id: p.position for p in scenario.people}
    choices = list(Action)
    for trial in range(5):
        rng = SplitMix64(population * 1000 + trial)
        actions = {pid: choices[rng.next_u64() % len(choices)] for pid in sorted(positions)}
        kwargs: dict[str, Any] = dict(
            width=scenario.width, height=scenario.height, walls=scenario.walls, burning=frozenset(), enter_fire=scenario.movement.enter_fire
        )
        forward = resolve_movements(positions, actions, **kwargs)
        backward = resolve_movements(dict(reversed(positions.items())), dict(reversed(actions.items())), **kwargs)
        assert forward == backward
        finals = [m.final for m in forward]
        assert len(finals) == len(set(finals)) == population
        winners = Counter(m.target for m in forward if m.outcome is MoveOutcome.MOVED)
        assert max(winners.values(), default=1) == 1


# --------------------------------------------------------------------------- determinism


def _shuffled(policy: Policy, seed: int) -> Policy:
    """The same actions, submitted in a pseudo-random insertion order."""

    def wrapped(obs: Observation) -> dict[str, Any]:
        items = list(policy(obs).items())
        rng = SplitMix64(seed + obs.step)
        for i in range(len(items) - 1, 0, -1):
            j = rng.next_u64() % (i + 1)
            items[i], items[j] = items[j], items[i]
        return dict(items)

    return wrapped


@pytest.mark.parametrize("scenario_id", CROWD_IDS)
def test_diagnostics_are_identical_across_runs_and_insertion_orders(scenario_id: str) -> None:
    scenario = public(scenario_id)
    baseline = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    again = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy))
    shuffled = compute_crowd_diagnostics(scenario, run_episode(scenario, _shuffled(greedy_exit_policy, 7)))
    reversed_order = compute_crowd_diagnostics(
        scenario, run_episode(scenario, lambda obs: dict(reversed(list(greedy_exit_policy(obs).items()))))
    )
    assert baseline == again == shuffled == reversed_order
    assert baseline.to_json() == again.to_json() == shuffled.to_json() == reversed_order.to_json()


def test_diagnostics_survive_a_record_json_round_trip() -> None:
    scenario = public("v1-pub-09-crowd-100-split-flow")
    record = run_episode(scenario, greedy_exit_policy)
    reloaded = RunRecord.from_json(record.to_json())
    assert compute_crowd_diagnostics(scenario, reloaded).digest() == compute_crowd_diagnostics(scenario, record).digest()


# Golden digests pin crowd diagnostics version 1.0 on SIMULATOR_VERSION 0.2.0
# under the greedy test baseline. CI runs them on Linux, Windows and macOS.
GOLDEN_CROWD_DIGESTS: dict[str, str] = {
    "v1-pub-02-single-exit-dense": "5fd29a9f01fbfc4dea9a3a4921dec5fc8a65de1152dc01642493488169531832",
    "v1-pub-03-corridor-bottleneck": "bf0a69c63168ad036141817169f764712898b01c145e7e503de67a0c27c8562f",
    "v1-pub-09-crowd-100-split-flow": "e26ea26ed2dacf204b4e1b824396786dee50972bc2aa0e9889187c32bc450f3d",
    "stress-50": "38408be8e4188a547d77bc2379de9140a4b6260c7328f354ca9c4ab649d1a576",
    "stress-100": "a5f37dd006d1026c5635350d96f9707dd77dc4f6eec5da5db760208b814913c3",
    "stress-200": "f72c246a3c7db622ea38cc284fdcc3b1d09498afbea24a45942850562e2fdeb7",
}


def _golden_inputs() -> list[tuple[str, Scenario]]:
    return [(i, public(i)) for i in CROWD_IDS] + [(f"stress-{n}", crowd_stress_scenario(n)) for n in STRESS_SIZES]


@pytest.mark.parametrize("name", [*CROWD_IDS, *(f"stress-{n}" for n in STRESS_SIZES)])
def test_golden_crowd_diagnostics(name: str) -> None:
    scenario = dict(_golden_inputs())[name]
    digest = compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy)).digest()
    assert digest == GOLDEN_CROWD_DIGESTS[name], f"{name}: {digest}"


_SUBPROCESS_SCRIPT = """
import sys
sys.path[:0] = [{src!r}, {tests!r}]
from test_crowd import CROWD_IDS, public
from conftest import crowd_stress_scenario, greedy_exit_policy
from emberbench import compute_crowd_diagnostics, run_episode
for name, scenario in [(i, public(i)) for i in CROWD_IDS] + [("stress-100", crowd_stress_scenario(100))]:
    print(name, compute_crowd_diagnostics(scenario, run_episode(scenario, greedy_exit_policy)).digest())
"""


def test_diagnostics_identical_across_processes_and_hash_seeds() -> None:
    script = _SUBPROCESS_SCRIPT.format(src=str(ROOT / "src"), tests=str(ROOT / "tests"))
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout.replace("\r\n", "\n"))
    assert outputs[0] == outputs[1] == outputs[2]
    for line in outputs[0].splitlines():
        name, digest = line.split()
        assert digest == GOLDEN_CROWD_DIGESTS[name], name
