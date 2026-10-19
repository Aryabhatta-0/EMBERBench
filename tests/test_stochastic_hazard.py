"""The frozen V1 stochastic hazard contract.

These tests pin the fire RNG stream, the exact probability semantics, the
draw-consumption rule, matched-world invariance across actions and a small set
of cross-platform golden fire traces. See ``docs/simulation-contract.md``,
section "Stochastic fire".
"""

import hashlib
import itertools
import json
import math
import os
import random
import secrets
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from conftest import fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    Action,
    FireConfig,
    FireModelName,
    Observation,
    Pos,
    Scenario,
    Simulator,
    TerminationConfig,
    compute_episode_metrics,
    load_suite,
    run_episode,
    verify_record,
)
from emberbench.canonical import canonical_json
from emberbench.fire import DeterministicSpread, FireTerrain, StochasticSpread, build_fire_model
from emberbench.grid import neighbours
from emberbench.rng import MASK64, SplitMix64, derive_stream_seed
from emberbench.simulator import FIRE_STREAM
from emberbench.state import SimState
from test_fairness import random_policy, stay_policy
from test_suite import PINNED_HASHES, SUITE_DIR, SUITE_HASH

Policy = Callable[[Observation], Mapping[str, Any]]

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden" / "stochastic_fire_traces.json"
ROUTE_ID = "v1-pub-10-route-invalidation"
DEV_SEEDS = (0, 1, 7, 20261019)  # fixed development seeds, never scored seeds

# SplitMix64 adds this constant to its state once per draw, so the number of
# draws between two states is (later - earlier) * GAMMA^-1 mod 2**64.
GAMMA = 0x9E3779B97F4A7C15
INV_GAMMA = pow(GAMMA, -1, 1 << 64)


def draws_between(before: int, after: int) -> int:
    return ((after - before) * INV_GAMMA) & MASK64


def fire_state(state: SimState) -> int:
    (label, value), = state.rng_state
    assert label == FIRE_STREAM
    return value


# --------------------------------------------------------------------------- scenarios

_TWO_FIRES = [
    "############",
    "#a........b#",
    "#..F....F..#",
    "#....##....E",
    "#c........d#",
    "#####E######",
]


def two_fires(*, exits_flammable: bool, interval: int = 1, p: float = 0.4) -> Scenario:
    """Two initial ignitions in one room with two exits."""
    return layout(
        _TWO_FIRES,
        scenario_id=f"two-fires-{'hot' if exits_flammable else 'cold'}-exits-i{interval}",
        fire=FireConfig(
            model=FireModelName.STOCHASTIC,
            spread_interval=interval,
            spread_probability=p,
            exits_flammable=exits_flammable,
        ),
        termination=TerminationConfig(max_steps=30),
    )


def stochastic_scenarios() -> dict[str, Scenario]:
    route = load_suite(SUITE_DIR).get(ROUTE_ID)
    return {
        "stochastic_room": fixture_scenario("stochastic_room"),
        "route_invalidation": route,
        "two_fires_hot_exits": two_fires(exits_flammable=True),
        "two_fires_cold_exits": two_fires(exits_flammable=False),
        "two_fires_interval_2": two_fires(exits_flammable=True, interval=2),
        "two_fires_p0": two_fires(exits_flammable=True, p=0.0),
        "two_fires_p1": two_fires(exits_flammable=True, p=1.0),
    }


SCENARIOS = stochastic_scenarios()


# --------------------------------------------------------------------------- policies


def converge_policy(obs: Observation) -> dict[str, Action]:
    """Deliberate conflicts: everyone heads for the lowest-ID person's cell."""
    active = obs.active
    target = active[0].position
    out = {}
    for p in active:
        dr, dc = target.row - p.position.row, target.col - p.position.col
        if dr == dc == 0:
            out[p.id] = Action.STAY
        elif abs(dr) >= abs(dc):
            out[p.id] = Action.SOUTH if dr > 0 else Action.NORTH
        else:
            out[p.id] = Action.EAST if dc > 0 else Action.WEST
    return out


def all_east_policy(obs: Observation) -> dict[str, Action]:
    """Everyone pushes into the same direction: walls, queues and blocked moves."""
    return {p.id: Action.EAST for p in obs.active}


def policies() -> list[Policy]:
    return [stay_policy, greedy_exit_policy, converge_policy, all_east_policy, random_policy(5)]


# --------------------------------------------------------------------------- an independent reference


def edge_count(scenario: Scenario, burning: frozenset[Pos]) -> int:
    """Number of (candidate, burning neighbour) edges, computed without fire.py."""
    exits = frozenset(scenario.exit_positions)
    count = 0
    for src in burning:
        for nb in neighbours(src, scenario.width, scenario.height):
            if nb in burning or nb in scenario.walls:
                continue
            if nb in exits and not scenario.fire.exits_flammable:
                continue
            count += 1
    return count


def fire_only_trace(scenario: Scenario, seed: int, steps: int) -> list[tuple[frozenset[Pos], frozenset[Pos], int]]:
    """The fire with no people at all: (ignited, burning, fire RNG state) for steps 1..steps."""
    terrain = FireTerrain.from_scenario(scenario)
    model = build_fire_model(scenario.fire)
    rng = SplitMix64(derive_stream_seed(seed, FIRE_STREAM))
    burning = frozenset(scenario.ignitions)
    out = []
    for step in range(1, steps + 1):
        ignited = model.spread(terrain, burning, step, rng)
        burning = burning | ignited
        out.append((ignited, burning, rng.state))
    return out


def sim_trace(scenario: Scenario, seed: int, policy: Policy) -> list[tuple[frozenset[Pos], frozenset[Pos], int]]:
    record = run_episode(scenario, policy, seed=seed)
    return [(frozenset(s.ignited), s.state.burning, fire_state(s.state)) for s in record.steps]


# --------------------------------------------------------------------------- 1. the RNG stream


@pytest.mark.parametrize("seed", [0, 1, 20261019, MASK64])
def test_simulator_owns_exactly_one_fire_stream_derived_from_seed_and_label(seed: int) -> None:
    digest = hashlib.sha256(f"emberbench/v1/{seed}/fire".encode("ascii")).digest()
    expected = int.from_bytes(digest[:8], "big")
    assert derive_stream_seed(seed, FIRE_STREAM) == expected
    for scenario in (SCENARIOS["stochastic_room"], fixture_scenario("one_person_one_exit")):
        sim = Simulator(scenario, seed=seed)
        assert sim.state.rng_state == ((FIRE_STREAM, expected),)


def test_deterministic_fire_never_advances_the_stream() -> None:
    scenario = fixture_scenario("fire_threatening_exit")
    assert scenario.fire.model is FireModelName.DETERMINISTIC
    record = run_episode(scenario, greedy_exit_policy, seed=5)
    start = fire_state(record.initial_state)
    assert all(fire_state(s.state) == start for s in record.steps)


def test_no_ambient_entropy_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = SCENARIOS["two_fires_hot_exits"]
    expected = run_episode(scenario, greedy_exit_policy, seed=3)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ambient entropy or clock used")

    with monkeypatch.context() as m:
        for name in ("random", "getrandbits", "randrange", "randint", "choice", "shuffle", "uniform", "seed"):
            m.setattr(random, name, forbidden)
        for name in ("time", "time_ns", "monotonic", "monotonic_ns"):
            m.setattr(time, name, forbidden)
        m.setattr(os, "urandom", forbidden)
        m.setattr(uuid, "uuid1", forbidden)
        m.setattr(uuid, "uuid4", forbidden)
        m.setattr(secrets, "randbits", forbidden)
        m.setattr(secrets, "token_bytes", forbidden)
        record = run_episode(scenario, greedy_exit_policy, seed=3)
        verify_record(scenario, record)
        compute_episode_metrics(scenario, record)
    assert record.to_json() == expected.to_json()


def test_global_random_state_does_not_matter() -> None:
    scenario = SCENARIOS["stochastic_room"]
    random.seed(1)
    a = run_episode(scenario, stay_policy, seed=9)
    random.seed(2)
    random.random()
    b = run_episode(scenario, stay_policy, seed=9)
    assert a.to_json() == b.to_json()


# --------------------------------------------------------------------------- 2. probability semantics


class ScriptedRng(SplitMix64):
    """Returns scripted Bernoulli outcomes and records the requested probabilities."""

    def __init__(self, outcomes: tuple[bool, ...]) -> None:
        super().__init__(0)
        self.outcomes = list(outcomes)
        self.requested: list[float] = []

    def bernoulli(self, p: float) -> bool:
        self.requested.append(p)
        return self.outcomes.pop(0)


def cross_terrain(k: int) -> tuple[FireTerrain, frozenset[Pos], Pos]:
    """A 3x3 grid: one candidate in the centre with exactly ``k`` burning neighbours; all else wall."""
    centre = Pos(1, 1)
    sides = [Pos(0, 1), Pos(1, 0), Pos(1, 2), Pos(2, 1)]
    burning = frozenset(sides[:k])
    cells = {Pos(r, c) for r in range(3) for c in range(3)}
    walls = frozenset(cells - burning - {centre})
    return FireTerrain(3, 3, walls, frozenset(), True), burning, centre


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 10), Fraction(7, 20), Fraction(1, 2), Fraction(1)])
def test_ignition_probability_is_exactly_one_minus_one_minus_p_to_the_k(k: int, p: Fraction) -> None:
    terrain, burning, centre = cross_terrain(k)
    model = StochasticSpread(probability=float(p))
    mass = Fraction(0)
    for outcomes in itertools.product([False, True], repeat=k):
        rng = ScriptedRng(outcomes)
        ignited = model.spread(terrain, burning, 1, rng)
        # One draw per edge, all at the configured p, and no draw is skipped.
        assert rng.requested == [float(p)] * k and rng.outcomes == []
        assert ignited == (frozenset({centre}) if any(outcomes) else frozenset())
        if any(outcomes):
            hits = sum(outcomes)
            mass += p**hits * (1 - p) ** (k - hits)
    assert mass == 1 - (1 - p) ** k


class FixedU64(SplitMix64):
    """Returns one fixed 64-bit output, to probe the Bernoulli threshold exactly."""

    def __init__(self, value: int) -> None:
        super().__init__(0)
        self.value = value

    def next_u64(self) -> int:
        return self.value


@pytest.mark.parametrize("p", [0.0, 2.0**-53, 1e-10, 0.1, 0.35, 0.5, 1 - 2.0**-53, 1.0])
def test_bernoulli_success_probability_is_exact_on_the_53_bit_grid(p: float) -> None:
    # random() is u / 2**53 for u = next_u64() >> 11, uniform on 0 .. 2**53 - 1.
    # bernoulli(p) is random() < p, so exactly ceil(p * 2**53) of the 2**53 values succeed.
    successes = math.ceil(Fraction(p) * (1 << 53))
    if successes > 0:
        assert FixedU64((successes - 1) << 11).bernoulli(p)
    if successes < (1 << 53):
        assert not FixedU64(successes << 11).bernoulli(p)
    assert (successes == 0) == (p == 0.0)
    assert (successes == 1 << 53) == (p == 1.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_ignition_frequency_matches_the_formula(k: int) -> None:
    p, trials = 0.35, 4000
    terrain, burning, _ = cross_terrain(k)
    model = StochasticSpread(probability=p)
    hits = sum(bool(model.spread(terrain, burning, 1, SplitMix64(derive_stream_seed(t, FIRE_STREAM)))) for t in range(trials))
    expected = 1 - (1 - p) ** k
    sigma = math.sqrt(expected * (1 - expected) / trials)
    assert abs(hits / trials - expected) < 5 * sigma


@pytest.mark.parametrize("interval", [1, 2, 3])
@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_p1_matches_deterministic_spread_and_p0_never_spreads(interval: int, seed: int) -> None:
    hot = two_fires(exits_flammable=True, interval=interval, p=1.0)
    cold = two_fires(exits_flammable=True, interval=interval, p=0.0)
    det = layout(
        _TWO_FIRES,
        fire=FireConfig(model=FireModelName.DETERMINISTIC, spread_interval=interval, exits_flammable=True),
        termination=TerminationConfig(max_steps=30),
    )
    steps = 30
    assert [b for _, b, _ in fire_only_trace(hot, seed, steps)] == [b for _, b, _ in fire_only_trace(det, seed, steps)]
    assert all(b == frozenset(cold.ignitions) for _, b, _ in fire_only_trace(cold, seed, steps))


# --------------------------------------------------------------------------- 3. matched worlds


@pytest.mark.parametrize("seed", DEV_SEEDS)
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_fire_is_identical_across_actions_and_equals_the_fire_with_no_people(name: str, seed: int) -> None:
    scenario = SCENARIOS[name]
    traces = [sim_trace(scenario, seed, policy) for policy in policies()]
    longest = max(len(t) for t in traces)
    reference = fire_only_trace(scenario, seed, longest)
    lengths = {len(t) for t in traces}
    for trace in traces:
        # Every step that a run reaches has the reference fire: ignited, burning and RNG state.
        assert trace == reference[: len(trace)]
    assert min(lengths) >= 1
    burning_ever = reference[-1][1]
    assert not burning_ever & scenario.walls
    if not scenario.fire.exits_flammable:
        assert not burning_ever & frozenset(scenario.exit_positions)


def test_flammable_exits_do_burn_for_some_development_seed() -> None:
    scenario = SCENARIOS["two_fires_hot_exits"]
    exits = frozenset(scenario.exit_positions)
    assert any(fire_only_trace(scenario, seed, 30)[-1][1] & exits for seed in DEV_SEEDS)


def test_multiple_ignitions_all_burn_at_step_zero_and_spread() -> None:
    scenario = SCENARIOS["two_fires_hot_exits"]
    assert len(scenario.ignitions) == 2
    sim = Simulator(scenario, seed=0)
    assert sim.state.burning == frozenset(scenario.ignitions)
    trace = fire_only_trace(scenario, 0, 30)
    grown = trace[-1][1]
    for ignition in scenario.ignitions:
        assert any(nb in grown for nb in neighbours(ignition, scenario.width, scenario.height))


def test_burning_cells_never_burn_out() -> None:
    for name, scenario in SCENARIOS.items():
        previous = frozenset(scenario.ignitions)
        for ignited, burning, _ in fire_only_trace(scenario, 7, 40):
            assert previous <= burning and not ignited & previous, name
            assert burning == previous | ignited
            previous = burning


# --------------------------------------------------------------------------- 4. RNG consumption


@pytest.mark.parametrize("seed", DEV_SEEDS)
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_draws_per_step_equal_frontier_edges_on_spread_steps_and_zero_otherwise(name: str, seed: int) -> None:
    scenario = SCENARIOS[name]
    interval = scenario.fire.spread_interval
    record = run_episode(scenario, converge_policy, seed=seed)
    before = record.initial_state
    for s in record.steps:
        expected = edge_count(scenario, before.burning) if s.step % interval == 0 else 0
        assert draws_between(fire_state(before), fire_state(s.state)) == expected, (name, seed, s.step)
        before = s.state


def test_no_candidates_means_no_draws() -> None:
    # A fire sealed in by walls, and a fully burnt grid.
    sealed = FireTerrain(3, 3, frozenset(Pos(r, c) for r in range(3) for c in range(3)) - {Pos(1, 1)}, frozenset(), True)
    full = FireTerrain(2, 2, frozenset(), frozenset(), True)
    for terrain, burning in ((sealed, frozenset({Pos(1, 1)})), (full, frozenset(Pos(r, c) for r in range(2) for c in range(2)))):
        rng = SplitMix64(42)
        assert StochasticSpread(probability=0.5).spread(terrain, burning, 1, rng) == frozenset()
        assert rng.state == 42


def test_non_flammable_exit_is_not_a_candidate_and_costs_no_draw() -> None:
    terrain = FireTerrain(2, 1, frozenset(), frozenset({Pos(0, 1)}), False)
    rng = SplitMix64(42)
    assert StochasticSpread(probability=1.0).spread(terrain, frozenset({Pos(0, 0)}), 1, rng) == frozenset()
    assert rng.state == 42
    hot = FireTerrain(2, 1, frozenset(), frozenset({Pos(0, 1)}), True)
    assert StochasticSpread(probability=1.0).spread(hot, frozenset({Pos(0, 0)}), 1, rng) == {Pos(0, 1)}
    assert draws_between(42, rng.state) == 1


@pytest.mark.parametrize("p", [0.0, 0.35, 1.0])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_every_scheduled_draw_is_consumed_for_any_p_and_neighbour_count(p: float, k: int) -> None:
    terrain, burning, _ = cross_terrain(k)
    rng = SplitMix64(1234)
    StochasticSpread(probability=p).spread(terrain, burning, 1, rng)
    assert draws_between(1234, rng.state) == k  # k edges, one candidate


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6])
def test_non_spread_steps_draw_nothing(step: int) -> None:
    terrain, burning, _ = cross_terrain(4)
    rng = SplitMix64(1234)
    ignited = StochasticSpread(probability=1.0, spread_interval=3).spread(terrain, burning, step, rng)
    expected_draws = 4 if step % 3 == 0 else 0
    assert draws_between(1234, rng.state) == expected_draws
    assert bool(ignited) == (step % 3 == 0)


def _orderings(cells: list[Pos]) -> Iterator[frozenset[Pos]]:
    yield frozenset(cells)
    yield frozenset(reversed(cells))
    shuffled = list(cells)
    random.Random(0).shuffle(shuffled)
    yield frozenset(shuffled)


def test_construction_order_of_the_burning_set_cannot_change_draws() -> None:
    scenario = SCENARIOS["two_fires_hot_exits"]
    terrain = FireTerrain.from_scenario(scenario)
    burning = sorted(fire_only_trace(scenario, 1, 4)[-1][1])
    results = set()
    for b in _orderings(burning):
        rng = SplitMix64(99)
        ignited = StochasticSpread(probability=0.4).spread(terrain, b, 5, rng)
        results.add((ignited, rng.state))
    assert len(results) == 1


def test_action_mapping_order_cannot_change_the_rng_position() -> None:
    scenario = SCENARIOS["two_fires_hot_exits"]
    forward, backward = Simulator(scenario, seed=4), Simulator(scenario, seed=4)
    while not forward.done:
        actions = converge_policy(forward.observe())
        a = forward.step(dict(sorted(actions.items())))
        b = backward.step(dict(sorted(actions.items(), reverse=True)))
        assert a.to_dict() == b.to_dict()
    assert forward.record().to_json() == backward.record().to_json()


def test_rng_state_is_never_in_the_observation() -> None:
    scenario = SCENARIOS["route_invalidation"]
    sim = Simulator(scenario, seed=20261019)
    while not sim.done:
        obs = sim.observe()
        text = canonical_json(obs.to_dict())
        assert "rng" not in text and str(fire_state(sim.state)) not in text
        assert not hasattr(obs, "rng_state")
        sim.step(greedy_exit_policy(obs))


# --------------------------------------------------------------------------- 5. golden traces


def _cells(cells: frozenset[Pos]) -> list[list[int]]:
    return [p.to_list() for p in sorted(cells)]


def golden_scenario(source: str) -> Scenario:
    kind, _, name = source.partition(":")
    if kind == "fixture":
        return fixture_scenario(name)
    if kind == "public":
        return load_suite(SUITE_DIR).get(name)
    if kind == "inline":
        return SCENARIOS[name]
    raise ValueError(source)


def golden_trace(source: str, seed: int, steps: int) -> dict[str, Any]:
    scenario = golden_scenario(source)
    trace = fire_only_trace(scenario, seed, steps)
    start = derive_stream_seed(seed, FIRE_STREAM)
    states = [start] + [s for _, _, s in trace]
    return {
        "source": source,
        "scenario_hash": scenario.content_hash(),
        "seed": seed,
        "steps": steps,
        "ignited": [_cells(i) for i, _, _ in trace],
        "draws": [draws_between(a, b) for a, b in zip(states, states[1:])],
        "final_burning": _cells(trace[-1][1]),
        "fire_rng_state": trace[-1][2],
    }


GOLDEN: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_trace_set_is_small_and_diagnostic() -> None:
    assert 3 <= len(GOLDEN) <= 6
    assert all(g["steps"] <= 12 for g in GOLDEN)
    assert len({(g["source"], g["seed"]) for g in GOLDEN}) == len(GOLDEN)


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: f"{g['source']}@{g['seed']}")
def test_golden_stochastic_fire_trace(golden: dict[str, Any]) -> None:
    assert golden_trace(golden["source"], golden["seed"], golden["steps"]) == golden
    # The simulator reproduces the same fire with people acting, for every step it reaches.
    scenario = golden_scenario(golden["source"])
    record = run_episode(scenario, greedy_exit_policy, seed=golden["seed"])
    for s, ignited in zip(record.steps, golden["ignited"]):
        assert _cells(frozenset(s.ignited)) == ignited
    if len(record.steps) >= golden["steps"]:
        final = record.steps[golden["steps"] - 1].state
        assert _cells(final.burning) == golden["final_burning"]
        assert fire_state(final) == golden["fire_rng_state"]


# --------------------------------------------------------------------------- 7. public suite


def test_public_suite_and_scenario_hashes_are_unchanged() -> None:
    suite = load_suite(SUITE_DIR)
    assert suite.manifest.suite_hash() == SUITE_HASH
    assert {s.scenario_id: s.content_hash() for s in suite.scenarios} == PINNED_HASHES


def _route_cut_step(trace: list[tuple[frozenset[Pos], frozenset[Pos], int]]) -> int | None:
    # The short route is the corridor on row 4; the side shaft meets it at (4, 10).
    for step, (_, burning, _) in enumerate(trace, start=1):
        if any(p.row == 4 and 6 <= p.col <= 17 for p in burning):
            return step
    return None


def test_route_invalidation_fire_depends_on_the_seed_only() -> None:
    scenario = SCENARIOS["route_invalidation"]
    steps = scenario.termination.max_steps
    traces = {seed: fire_only_trace(scenario, seed, steps) for seed in range(16)}
    distinct = {canonical_json([[_cells(b) for _, b, _ in t]]) for t in traces.values()}
    assert len(distinct) >= 2
    # The same seed gives a byte-identical fire trace, by itself and inside the simulator.
    for seed in (0, 5):
        again = fire_only_trace(scenario, seed, steps)
        assert canonical_json([[_cells(i), _cells(b), s] for i, b, s in again]) == canonical_json(
            [[_cells(i), _cells(b), s] for i, b, s in traces[seed]]
        )
        a, b = run_episode(scenario, stay_policy, seed=seed), run_episode(scenario, greedy_exit_policy, seed=seed)
        overlap = min(len(a.steps), len(b.steps))
        assert canonical_json([[x.to_dict()["ignited"], x.state.to_dict()["burning"], x.state.to_dict()["rng_state"]] for x in a.steps[:overlap]]) == canonical_json(
            [[x.to_dict()["ignited"], x.state.to_dict()["burning"], x.state.to_dict()["rng_state"]] for x in b.steps[:overlap]]
        )
    cut_steps = {_route_cut_step(t) for t in traces.values()}
    assert None not in cut_steps
    assert len(cut_steps) >= 3, sorted(cut_steps)  # type: ignore[type-var]
