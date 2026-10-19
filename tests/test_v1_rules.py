"""One test group per frozen V1 rule in ``docs/v1-fairness-spec.md`` (section "Frozen V1 rules").

Some rules also have lower-level tests in ``test_movement.py``, ``test_fire.py``
and ``test_simulator.py``. These tests pin each rule at the simulator level.
"""

import pytest

from conftest import fixture_scenario, layout
from emberbench import (
    Action,
    CasualtyCause,
    EnterFirePolicy,
    FireConfig,
    MoveOutcome,
    MovementConfig,
    PersonStatus,
    Pos,
    Scenario,
    ScenarioValidationError,
    SimulationError,
    Simulator,
    TerminationConfig,
    TerminationReason,
)

N, S, E, W, STAY = Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.STAY


def outcomes(sim: Simulator, actions: dict[str, Action]) -> dict[str, MoveOutcome]:
    return {m.person_id: m.outcome for m in sim.step(actions).movements}


# --------------------------------------------------------------------------- R1 collision tie-break

# Four people around the centre cell (1, 1) of a 3x3 block. Each slot moves into the centre.
_SLOTS = {(0, 1): S, (1, 0): E, (1, 2): W, (2, 1): N}


@pytest.mark.parametrize("winner_slot", sorted(_SLOTS))
def test_r1_lowest_id_wins_from_any_side(winner_slot: tuple[int, int]) -> None:
    grid = [list("...E"), list("...."), list("....")]
    others = iter("bcd")
    letters = {slot: ("a" if slot == winner_slot else next(others)) for slot in sorted(_SLOTS)}
    for (r, c), letter in letters.items():
        grid[r][c] = letter
    sim = Simulator(layout(["".join(row) for row in grid]))
    result = outcomes(sim, {f"P{letters[slot]}": action for slot, action in _SLOTS.items()})
    assert result == {"Pa": MoveOutcome.MOVED, "Pb": MoveOutcome.BLOCKED_CONFLICT, "Pc": MoveOutcome.BLOCKED_CONFLICT, "Pd": MoveOutcome.BLOCKED_CONFLICT}
    assert sim.state.person("Pa").position == Pos(1, 1)


# --------------------------------------------------------------------------- R2/R3 swaps and rotations


def test_r2_swap_is_blocked_in_the_simulator() -> None:
    sim = Simulator(layout(["ab.E"]))
    assert outcomes(sim, {"Pa": E, "Pb": W}) == {"Pa": MoveOutcome.BLOCKED_CYCLE, "Pb": MoveOutcome.BLOCKED_CYCLE}
    assert (sim.state.person("Pa").position, sim.state.person("Pb").position) == (Pos(0, 0), Pos(0, 1))


def test_r3_rotation_is_blocked_in_the_simulator() -> None:
    sim = Simulator(layout(["ab.", "dc.", "..E"]))
    result = outcomes(sim, {"Pa": E, "Pb": S, "Pc": W, "Pd": N})
    assert set(result.values()) == {MoveOutcome.BLOCKED_CYCLE}
    assert [p.position for p in sim.state.people] == [Pos(0, 0), Pos(0, 1), Pos(1, 1), Pos(1, 0)]


# --------------------------------------------------------------------------- R4 exit throughput


def test_r4_queue_flows_through_an_exit_one_per_step_without_gaps() -> None:
    sim = Simulator(layout(["..abE"]))
    first = sim.step({"Pa": E, "Pb": E})
    assert [e.person_id for e in first.evacuations] == ["Pb"]
    assert sim.state.person("Pa").position == Pos(0, 3)  # followed into the vacated cell
    second = sim.step({"Pa": E})
    assert [e.person_id for e in second.evacuations] == ["Pa"]


def test_r4_throughput_is_per_exit_cell() -> None:
    sim = Simulator(layout(["ab", "EE"]))
    record = sim.step({"Pa": S, "Pb": S})
    assert [(e.person_id, e.exit_id) for e in record.evacuations] == [("Pa", "E1"), ("Pb", "E2")]


# --------------------------------------------------------------------------- R5 burning exits


def test_r5_exit_that_ignites_on_arrival_step_still_evacuates() -> None:
    sim = Simulator(layout([".aEF"]))
    record = sim.step({"Pa": E})
    assert [e.person_id for e in record.evacuations] == ["Pa"]
    assert Pos(0, 2) in record.ignited  # the exit ignites in phase 7, after evacuation in phase 6
    assert record.casualties == ()


def test_r5_burning_exit_is_refused_under_blocked_policy() -> None:
    s = layout(["aEF"], fire=FireConfig(spread_interval=2), movement=MovementConfig(enter_fire=EnterFirePolicy.BLOCKED))
    sim = Simulator(s)
    sim.step({"Pa": STAY})
    sim.step({"Pa": STAY})  # the exit ignites on step 2; step 3 has no spread
    assert Pos(0, 1) in sim.state.burning
    assert outcomes(sim, {"Pa": E}) == {"Pa": MoveOutcome.BLOCKED_FIRE}
    assert sim.state.person("Pa").status is PersonStatus.ACTIVE


def test_r5_burning_exit_kills_under_casualty_policy() -> None:
    sim = Simulator(layout(["a.EF"]))
    sim.step({"Pa": E})
    record = sim.step({"Pa": E})
    assert record.evacuations == ()
    assert [(c.person_id, c.cause) for c in record.casualties] == [("Pa", CasualtyCause.ENTERED_FIRE)]


# --------------------------------------------------------------------------- R6 enter_fire


def test_r6_enter_fire_default_is_casualty() -> None:
    assert MovementConfig().enter_fire is EnterFirePolicy.CASUALTY


def test_r6_scenario_files_must_state_enter_fire() -> None:
    data = fixture_scenario("one_person_one_exit").to_dict()
    del data["movement"]["enter_fire"]
    with pytest.raises(ScenarioValidationError, match="enter_fire"):
        Scenario.from_dict(data)


# --------------------------------------------------------------------------- R7 ignition timing


def test_r7_initial_ignitions_burn_and_are_visible_at_step_zero() -> None:
    s = layout(["F.a..E"])
    sim = Simulator(s)
    assert sim.state.step == 0
    assert sim.observe().burning == frozenset(s.ignitions) == {Pos(0, 0)}


def test_r7_movement_happens_before_spread_in_the_same_step() -> None:
    sim = Simulator(layout(["Fa...E"]))
    record = sim.step({"Pa": E})  # step away before the fire reaches (0, 1)
    assert Pos(0, 1) in record.ignited
    assert record.casualties == ()
    assert sim.state.person("Pa").status is PersonStatus.ACTIVE


def test_r7_cells_ignited_in_a_step_are_visible_before_the_next_step() -> None:
    sim = Simulator(layout(["F...aE"]))
    record = sim.step({"Pa": STAY})
    assert sim.observe().burning == frozenset({Pos(0, 0)}) | frozenset(record.ignited)


# --------------------------------------------------------------------------- R8 no burn-out


def test_r8_burning_cells_never_burn_out() -> None:
    scenario = fixture_scenario("stochastic_room")
    sim = Simulator(scenario, seed=11)
    previous = sim.state.burning
    while not sim.done:
        sim.step({p.id: STAY for p in sim.observe().active})
        assert previous <= sim.state.burning
        previous = sim.state.burning
    assert frozenset(scenario.ignitions) <= sim.state.burning


# --------------------------------------------------------------------------- R9 trapped


def test_r9_other_people_never_make_someone_trapped() -> None:
    sim = Simulator(layout(["#####", "#abE#", "#####"]))
    assert sim.state.trapped == frozenset()  # Pa's only route goes through Pb


def test_r9_a_route_through_fire_does_not_count() -> None:
    sim = Simulator(layout(["aF.E"]))
    assert sim.state.trapped == {"Pa"}


def test_r9_trapped_changes_no_status_and_ends_no_run() -> None:
    sim = Simulator(layout(["a#.E"], termination=TerminationConfig(max_steps=4)))
    while not sim.done:
        sim.step({"Pa": STAY})
        assert sim.state.trapped == {"Pa"}
        assert sim.state.person("Pa").status is PersonStatus.ACTIVE
    assert sim.state.step == 4
    assert sim.state.termination_reason is TerminationReason.MAX_STEPS


def test_r9_only_active_people_can_be_trapped() -> None:
    sim = Simulator(layout(["a#bE"]))
    sim.step({"Pa": STAY, "Pb": E})
    assert sim.state.person("Pb").status is PersonStatus.EVACUATED
    assert sim.state.trapped == {"Pa"}


# --------------------------------------------------------------------------- R10 termination


def test_r10_all_resolved_wins_over_max_steps_on_the_same_step() -> None:
    sim = Simulator(layout(["aE"], termination=TerminationConfig(max_steps=1)))
    sim.step({"Pa": E})
    assert sim.state.termination_reason is TerminationReason.ALL_RESOLVED


def test_r10_max_steps_counts_completed_steps() -> None:
    sim = Simulator(layout(["a..E"], termination=TerminationConfig(max_steps=1)))
    sim.step({"Pa": STAY})
    assert sim.state.step == 1
    assert sim.state.termination_reason is TerminationReason.MAX_STEPS
    assert sim.state.person("Pa").status is PersonStatus.ACTIVE
    with pytest.raises(SimulationError, match="terminated"):
        sim.step({"Pa": STAY})


def test_r10_mixed_outcomes_still_end_as_all_resolved() -> None:
    sim = Simulator(layout(["Fa.", "..b", "..E"]))
    sim.step({"Pa": STAY, "Pb": S})
    assert sim.state.ids_with_status(PersonStatus.CASUALTY) == ("Pa",)
    assert sim.state.ids_with_status(PersonStatus.EVACUATED) == ("Pb",)
    assert sim.state.termination_reason is TerminationReason.ALL_RESOLVED
