import itertools
from typing import Any

import pytest

from conftest import layout
from emberbench import Action, InvalidActionError, MoveOutcome, Pos, Scenario, Simulator
from emberbench.movement import parse_actions, resolve_movements
from emberbench.scenario import EnterFirePolicy, MovementConfig

N, S, E, W, STAY = Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.STAY


def resolve(scenario: Scenario, actions: dict[str, Action], burning: frozenset[Pos] = frozenset()) -> dict[str, Any]:
    positions = {p.id: p.position for p in scenario.people}
    result = resolve_movements(
        positions,
        actions,
        width=scenario.width,
        height=scenario.height,
        walls=scenario.walls,
        burning=burning or frozenset(scenario.ignitions),
        enter_fire=scenario.movement.enter_fire,
    )
    return {m.person_id: (m.outcome, m.final) for m in result}


def test_simple_move_and_stay() -> None:
    s = layout(["#####", "#a.b#", "#E..#", "#####"])
    r = resolve(s, {"Pa": E, "Pb": STAY})
    assert r == {"Pa": (MoveOutcome.MOVED, Pos(1, 2)), "Pb": (MoveOutcome.STAYED, Pos(1, 3))}


def test_wall_and_boundary_block() -> None:
    s = layout(["a..", "#.E", "b.."])
    r = resolve(s, {"Pa": N, "Pb": N})
    assert r["Pa"] == (MoveOutcome.BLOCKED_BOUNDARY, Pos(0, 0))
    assert r["Pb"] == (MoveOutcome.BLOCKED_WALL, Pos(2, 0))


def test_conflict_lowest_id_wins_regardless_of_submission_order() -> None:
    s = layout(["a.b", "...", "..E"])
    for order in itertools.permutations([("Pb", W), ("Pa", E)]):
        r = resolve(s, dict(order))
        assert r["Pa"] == (MoveOutcome.MOVED, Pos(0, 1))
        assert r["Pb"] == (MoveOutcome.BLOCKED_CONFLICT, Pos(0, 2))


def test_ids_compare_as_strings() -> None:
    from emberbench import ExitSpec, PersonSpec

    s = Scenario(
        scenario_id="ids",
        width=3,
        height=2,
        obstacles=(),
        exits=(ExitSpec("E1", Pos(1, 1)),),
        people=(PersonSpec("P2", Pos(0, 0)), PersonSpec("P10", Pos(0, 2))),
        ignitions=(),
        seed=0,
    )
    r = resolve(s, {"P2": E, "P10": W})
    assert r["P10"][0] is MoveOutcome.MOVED  # "P10" < "P2"
    assert r["P2"][0] is MoveOutcome.BLOCKED_CONFLICT


def test_swap_is_blocked() -> None:
    s = layout(["ab.", "..E"])
    r = resolve(s, {"Pa": E, "Pb": W})
    assert r == {"Pa": (MoveOutcome.BLOCKED_CYCLE, Pos(0, 0)), "Pb": (MoveOutcome.BLOCKED_CYCLE, Pos(0, 1))}


def test_rotation_is_blocked() -> None:
    # a b      a->E, b->S, d->W, c->N forms a 4-cycle. e (higher ID) also targets a's cell
    # c d      and loses that conflict to c.
    s = layout(["eab", ".cd", "..E"])
    r = resolve(s, {"Pa": E, "Pb": S, "Pd": W, "Pc": N, "Pe": E})
    for pid in ("Pa", "Pb", "Pc", "Pd"):
        assert r[pid][0] is MoveOutcome.BLOCKED_CYCLE
    assert r["Pe"][0] is MoveOutcome.BLOCKED_CONFLICT


def test_lower_id_outsider_breaks_rotation() -> None:
    # b c      Rotation b->E, c->S, d->W, e->N. Pa targets b's cell and beats Pe there,
    # e d      so Pe becomes a non-mover and the whole chain is blocked.
    s = layout(["abc", ".ed", "..E"])
    r = resolve(s, {"Pa": E, "Pb": E, "Pc": S, "Pd": W, "Pe": N})
    assert r["Pe"][0] is MoveOutcome.BLOCKED_CONFLICT
    for pid in ("Pa", "Pb", "Pc", "Pd"):
        assert r[pid][0] is MoveOutcome.BLOCKED_OCCUPIED


def test_following_chain_moves_together() -> None:
    s = layout(["abc..E"])
    r = resolve(s, {"Pa": E, "Pb": E, "Pc": E})
    assert [r[p] for p in ("Pa", "Pb", "Pc")] == [
        (MoveOutcome.MOVED, Pos(0, 1)),
        (MoveOutcome.MOVED, Pos(0, 2)),
        (MoveOutcome.MOVED, Pos(0, 3)),
    ]


def test_chain_into_stayer_is_blocked_all_the_way_back() -> None:
    s = layout(["abc..E"])
    r = resolve(s, {"Pa": E, "Pb": E, "Pc": STAY})
    assert r["Pa"][0] is MoveOutcome.BLOCKED_OCCUPIED
    assert r["Pb"][0] is MoveOutcome.BLOCKED_OCCUPIED
    assert r["Pc"][0] is MoveOutcome.STAYED


def test_chain_into_blocked_mover_is_blocked() -> None:
    s = layout(["ab#.E"])
    r = resolve(s, {"Pa": E, "Pb": E})
    assert r["Pb"][0] is MoveOutcome.BLOCKED_WALL
    assert r["Pa"][0] is MoveOutcome.BLOCKED_OCCUPIED


def test_conflict_loser_is_not_reawarded_when_winner_fails() -> None:
    #  . a .      Pa -> S into (1,1) occupied by Pc, Pc -> S blocked by wall.
    #  b c .      Pb -> E into (1,1) too. Pa wins the conflict, then fails the chain.
    #  . # E
    s = layout([".a.", "bc.", ".#E"])
    r = resolve(s, {"Pa": S, "Pb": E, "Pc": S})
    assert r["Pc"][0] is MoveOutcome.BLOCKED_WALL
    assert r["Pa"][0] is MoveOutcome.BLOCKED_OCCUPIED
    assert r["Pb"][0] is MoveOutcome.BLOCKED_OCCUPIED  # target held by a non-mover


def test_enter_fire_policies() -> None:
    blocked = layout(["aF.E"], movement=MovementConfig(enter_fire=EnterFirePolicy.BLOCKED))
    assert resolve(blocked, {"Pa": E})["Pa"] == (MoveOutcome.BLOCKED_FIRE, Pos(0, 0))
    casualty = layout(["aF.E"], movement=MovementConfig(enter_fire=EnterFirePolicy.CASUALTY))
    assert resolve(casualty, {"Pa": E})["Pa"] == (MoveOutcome.MOVED, Pos(0, 1))


def test_resolution_is_order_independent_for_every_permutation() -> None:
    s = layout(["ab.c", "d.e.", "..E."])
    actions = {"Pa": E, "Pb": E, "Pc": W, "Pd": N, "Pe": W}
    expected = resolve(s, actions)
    for perm in itertools.permutations(actions.items()):
        assert resolve(s, dict(perm)) == expected


def test_final_cells_are_unique_under_exhaustive_actions() -> None:
    s = layout(["ab.", "c.d", ".E."])
    for combo in itertools.product(list(Action), repeat=4):
        actions = dict(zip(["Pa", "Pb", "Pc", "Pd"], combo, strict=True))
        finals = [final for _, final in resolve(s, actions).values()]
        assert len(finals) == len(set(finals))


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ({}, "missing"),
        ({"Pa": "east"}, "missing"),
        ({"Pa": "east", "Pb": "stay", "Pz": "stay"}, "unexpected"),
        ({"Pa": "up", "Pb": "stay"}, "invalid action"),
        ({"Pa": 1, "Pb": "stay"}, "invalid action"),
        ({"Pa": None, "Pb": "stay"}, "invalid action"),
        ({"Pa": "EAST", "Pb": "stay"}, "invalid action"),
        ({"Pa": "east", "Pb": "stay", 3: "stay"}, "keys must be"),
        (["east"], "must be a mapping"),
    ],
)
def test_invalid_action_sets_fail_explicitly(actions: Any, message: str) -> None:
    with pytest.raises(InvalidActionError, match=message):
        parse_actions(actions, frozenset({"Pa", "Pb"}))


def test_invalid_actions_leave_simulator_untouched() -> None:
    sim = Simulator(layout(["a.b.E"]))
    before = sim.state
    with pytest.raises(InvalidActionError):
        sim.step({"Pa": "east"})
    assert sim.state is before
    assert sim.record().steps == ()


def test_actions_for_resolved_people_are_rejected() -> None:
    sim = Simulator(layout(["aE.b"]))
    sim.step({"Pa": "east", "Pb": "stay"})
    with pytest.raises(InvalidActionError, match="unexpected=\\['Pa'\\]"):
        sim.step({"Pa": "stay", "Pb": "west"})
