from conftest import fixture_scenario, layout
from emberbench import Action, Pos, Simulator
from emberbench.fire import DeterministicSpread, FireTerrain, StochasticSpread
from emberbench.rng import SplitMix64
from emberbench.scenario import FireConfig, FireModelName


def terrain_for(rows: list[str], exits_flammable: bool = True) -> tuple[FireTerrain, frozenset[Pos]]:
    s = layout(rows, fire=FireConfig(exits_flammable=exits_flammable))
    return FireTerrain.from_scenario(s), frozenset(s.ignitions)


def test_deterministic_spread_is_von_neumann() -> None:
    terrain, burning = terrain_for([".....", ".....", "..F..", ".....", "a...E"])
    ignited = DeterministicSpread().spread(terrain, burning, 1, SplitMix64(0))
    assert ignited == {Pos(1, 2), Pos(3, 2), Pos(2, 1), Pos(2, 3)}


def test_walls_block_fire_completely() -> None:
    rows = ["F#..", ".#..", ".#.E", "a#.."]
    s = layout(rows)
    sim = Simulator(s)
    for _ in range(10):
        if sim.done:
            break
        sim.step({p.id: Action.STAY for p in sim.observe().active})
    assert all(p.col == 0 for p in sim.state.burning)
    assert not sim.state.burning & s.walls


def test_spread_interval() -> None:
    terrain, burning = terrain_for(["F...", "a..E"])
    model = DeterministicSpread(spread_interval=3)
    assert [bool(model.spread(terrain, burning, t, SplitMix64(0))) for t in range(1, 7)] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_exit_flammability_is_configurable() -> None:
    terrain, burning = terrain_for(["FE", "a."], exits_flammable=True)
    assert Pos(0, 1) in DeterministicSpread().spread(terrain, burning, 1, SplitMix64(0))
    terrain, burning = terrain_for(["FE", "a."], exits_flammable=False)
    assert Pos(0, 1) not in DeterministicSpread().spread(terrain, burning, 1, SplitMix64(0))


def test_stochastic_extremes_match_deterministic() -> None:
    terrain, burning = terrain_for([".....", ".F...", "...F.", "a...E"])
    full = DeterministicSpread().spread(terrain, burning, 1, SplitMix64(0))
    assert StochasticSpread(probability=1.0).spread(terrain, burning, 1, SplitMix64(5)) == full
    assert StochasticSpread(probability=0.0).spread(terrain, burning, 1, SplitMix64(5)) == frozenset()


def test_stochastic_is_seed_determined_and_draw_count_is_geometric() -> None:
    terrain, burning = terrain_for([".......", ".F...F.", ".......", "a.....E"])
    edges = sum(len(srcs) for _, srcs in terrain.frontier(burning))
    model = StochasticSpread(probability=0.5)
    outcomes = set()
    for seed in range(20):
        a, b = SplitMix64(seed), SplitMix64(seed)
        first = model.spread(terrain, burning, 1, a)
        assert first == model.spread(terrain, burning, 1, b)
        assert a.state == b.state
        probe = SplitMix64(seed)
        for _ in range(edges):
            probe.next_u64()
        assert a.state == probe.state  # exactly one draw per edge
        outcomes.add(first)
    assert len(outcomes) > 1  # the seed actually matters


def test_no_draws_on_non_spread_steps() -> None:
    terrain, burning = terrain_for(["F..", "a.E"])
    rng = SplitMix64(3)
    StochasticSpread(probability=0.5, spread_interval=2).spread(terrain, burning, 1, rng)
    assert rng.state == 3


def test_fire_threatening_exit_burns_exit_and_it_stops_evacuating() -> None:
    s = fixture_scenario("fire_threatening_exit")
    sim = Simulator(s)
    near_exit = next(e.position for e in s.exits if e.id == "E1")
    rec = sim.step({"P1": Action.WEST})
    assert near_exit in rec.ignited
    assert near_exit in sim.state.burning
    # The burning exit no longer counts for reachability. E2 is still reachable.
    assert sim.state.trapped == frozenset()


def test_stochastic_fixture_uses_configured_model() -> None:
    s = fixture_scenario("stochastic_room")
    assert s.fire.model is FireModelName.STOCHASTIC
    a, b = Simulator(s), Simulator(s, seed=s.seed + 1)
    stay = {p.id: Action.STAY for p in s.people}
    ra, rb = a.step(stay), b.step(stay)
    assert ra.state.rng_state != rb.state.rng_state
