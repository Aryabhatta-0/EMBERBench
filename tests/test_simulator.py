import pytest

from conftest import fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    Action,
    CasualtyCause,
    PersonStatus,
    Pos,
    SimulationError,
    Simulator,
    TerminationReason,
    render,
    run_episode,
)
from emberbench.scenario import EnterFirePolicy, FireConfig, MovementConfig, TerminationConfig

E, W, STAY = Action.EAST, Action.WEST, Action.STAY


def test_one_person_one_exit_evacuates_on_arrival() -> None:
    s = fixture_scenario("one_person_one_exit")
    sim = Simulator(s)
    for expected_col in (2, 3, 4):
        rec = sim.step({"P1": E})
        assert rec.state.person("P1").position == Pos(1, expected_col)
        assert rec.evacuations == ()
    rec = sim.step({"P1": E})
    assert [(e.person_id, e.exit_id) for e in rec.evacuations] == [("P1", "E1")]
    assert rec.state.person("P1").status is PersonStatus.EVACUATED
    assert rec.state.terminated and rec.state.termination_reason is TerminationReason.ALL_RESOLVED
    assert sim.done


def test_exit_throughput_is_one_person_per_exit_per_step() -> None:
    s = fixture_scenario("narrow_corridor")
    record = run_episode(s, greedy_exit_policy)
    per_step = [len(step.evacuations) for step in record.steps]
    assert max(per_step) == 1
    assert sorted(e.person_id for st in record.steps for e in st.evacuations) == ["P1", "P2", "P3", "P4"]
    assert record.final_state.termination_reason is TerminationReason.ALL_RESOLVED


def test_two_people_cannot_enter_one_exit_in_the_same_step() -> None:
    sim = Simulator(layout(["a.", "Eb"]))
    rec = sim.step({"Pa": Action.SOUTH, "Pb": W})
    assert [e.person_id for e in rec.evacuations] == ["Pa"]
    assert rec.state.person("Pb").position == Pos(1, 1)


def test_fire_spread_casualty_and_cause() -> None:
    sim = Simulator(layout(["Fa..E"]))
    rec = sim.step({"Pa": STAY})
    assert [(c.person_id, c.cause) for c in rec.casualties] == [("Pa", CasualtyCause.FIRE_SPREAD)]
    assert rec.state.person("Pa").status is PersonStatus.CASUALTY
    assert rec.state.termination_reason is TerminationReason.ALL_RESOLVED


def test_entering_fire_under_casualty_policy() -> None:
    s = layout(["a.F...E"], fire=FireConfig(spread_interval=50), movement=MovementConfig(enter_fire=EnterFirePolicy.CASUALTY))
    sim = Simulator(s)
    sim.step({"Pa": E})
    rec = sim.step({"Pa": E})
    assert [(c.person_id, c.cause) for c in rec.casualties] == [("Pa", CasualtyCause.ENTERED_FIRE)]


def test_entering_burning_exit_does_not_evacuate() -> None:
    s = layout(["aEF"], fire=FireConfig(spread_interval=1, exits_flammable=True))
    # The fire reaches the exit in phase 7 of step 1, after Pa evacuated in phase 6.
    rec = Simulator(s).step({"Pa": E})
    assert [e.person_id for e in rec.evacuations] == ["Pa"]
    # When the exit is already burning, stepping onto it is a casualty.
    s = layout(["a.EF"], fire=FireConfig(spread_interval=1, exits_flammable=True))
    sim = Simulator(s)
    sim.step({"Pa": E})
    rec = sim.step({"Pa": E})
    assert rec.evacuations == ()
    assert [(c.person_id, c.cause) for c in rec.casualties] == [("Pa", CasualtyCause.ENTERED_FIRE)]


def test_blocked_fire_policy_keeps_person_in_place() -> None:
    s = layout(["aF.E", "...."], fire=FireConfig(spread_interval=99), movement=MovementConfig(enter_fire=EnterFirePolicy.BLOCKED))
    rec = Simulator(s).step({"Pa": E})
    assert rec.movements[0].outcome.value == "blocked_fire"
    assert rec.state.person("Pa").position == Pos(0, 0)


def test_trapped_detection() -> None:
    s = layout(["a#E", ".#.", "F#."], fire=FireConfig(spread_interval=1))
    sim = Simulator(s)
    assert sim.state.trapped == {"Pa"}  # walled off from the only exit
    s2 = layout(["a..", "F#E"], fire=FireConfig(spread_interval=2))
    sim2 = Simulator(s2)
    assert sim2.state.trapped == frozenset()


def test_max_steps_termination_is_deterministic() -> None:
    s = layout(["a...E"], termination=TerminationConfig(max_steps=3))
    sim = Simulator(s)
    for _ in range(3):
        sim.step({"Pa": STAY})
    assert sim.state.termination_reason is TerminationReason.MAX_STEPS
    assert sim.state.step == 3
    with pytest.raises(SimulationError, match="terminated"):
        sim.step({"Pa": STAY})


def test_all_resolved_takes_precedence_over_max_steps() -> None:
    s = layout(["aE"], termination=TerminationConfig(max_steps=1))
    rec = Simulator(s).step({"Pa": E})
    assert rec.state.termination_reason is TerminationReason.ALL_RESOLVED


def test_ids_remain_stable_through_a_run(any_fixture) -> None:  # type: ignore[no-untyped-def]
    record = run_episode(any_fixture, greedy_exit_policy)
    expected = [p.id for p in any_fixture.people]
    assert [p.id for p in record.initial_state.people] == expected
    for step in record.steps:
        assert [p.id for p in step.state.people] == expected
        assert [m.person_id for m in step.movements] == sorted(step.action_map)
    resolved = {e.person_id for st in record.steps for e in st.evacuations} | {
        c.person_id for st in record.steps for c in st.casualties
    }
    assert resolved == {p.id for p in record.final_state.people if p.status is not PersonStatus.ACTIVE}


def test_fixture_outcomes_with_baseline_policy() -> None:
    outcomes = {}
    for name in ("one_person_one_exit", "blocked_shortest_route", "multiple_exits", "narrow_corridor", "fire_threatening_exit"):
        s = fixture_scenario(name)
        final = run_episode(s, greedy_exit_policy).final_state
        outcomes[name] = (final.step, final.ids_with_status(PersonStatus.CASUALTY))
    assert outcomes == {
        "one_person_one_exit": (4, ()),
        "blocked_shortest_route": (9, ()),  # took the long safe route, not the 7-step burning one
        "multiple_exits": (4, ()),
        "narrow_corridor": (7, ()),
        "fire_threatening_exit": (6, ()),
    }


def test_observation_hides_rng_state() -> None:
    sim = Simulator(fixture_scenario("stochastic_room"))
    obs = sim.observe()
    assert "rng_state" not in obs.to_dict()
    assert not hasattr(obs, "rng_state")


def test_render() -> None:
    s = fixture_scenario("one_person_one_exit")
    assert render(s, Simulator(s).state) == "#######\n#P...E#\n#######"


def test_seed_override_is_recorded() -> None:
    s = fixture_scenario("stochastic_room")
    assert Simulator(s, seed=99).record().seed == 99
    assert Simulator(s).record().seed == s.seed
    with pytest.raises(ValueError):
        Simulator(s, seed=-5)
