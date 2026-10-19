"""Fairness invariants: the observation contract and the exogenous fire process.

See ``docs/v1-fairness-spec.md``. These tests check that a decision model
cannot see evaluator-private data, and that every compared model faces the
same hazard trajectory for the same scenario and evaluator seed.
"""

import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from conftest import fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    ALLOWED_ACTIONS,
    OBSERVATION_VERSION,
    Action,
    ExitSpec,
    FireConfig,
    FireModelName,
    MapView,
    Observation,
    PersonState,
    PersonStatus,
    Pos,
    Scenario,
    Simulator,
    run_episode,
)
from emberbench.fire import FireTerrain, build_fire_model
from emberbench.rng import SplitMix64, derive_stream_seed
from emberbench.simulator import FIRE_STREAM

# A distinctive 64-bit seed: its decimal form cannot appear in an observation by chance.
SECRET_SEED = 0xDEADBEEFCAFEF00D
HIDDEN_ID = "hidden_eval_case_0042"
HIDDEN_DESCRIPTION = "hidden test set metadata"

Policy = Callable[[Observation], Mapping[str, Any]]


def secret_scenario(seed: int = SECRET_SEED) -> Scenario:
    return layout(
        [
            "#########",
            "#a.....b#",
            "#...F...E",
            "#c.....d#",
            "#########",
        ],
        scenario_id=HIDDEN_ID,
        description=HIDDEN_DESCRIPTION,
        seed=seed,
        fire=FireConfig(model=FireModelName.STOCHASTIC, spread_probability=0.5),
    )


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def random_policy(policy_seed: int) -> Policy:
    """A model with its own randomness. It never touches the evaluator's streams."""
    rng = SplitMix64(policy_seed)

    def policy(obs: Observation) -> dict[str, Action]:
        return {p.id: obs.allowed_actions[rng.next_u64() % len(obs.allowed_actions)] for p in obs.active}

    return policy


def observations(scenario: Scenario, policy: Policy, *, seed: int | None = None) -> list[Observation]:
    sim = Simulator(scenario, seed=seed)
    seen = [sim.observe()]
    while not sim.done:
        sim.step(policy(seen[-1]))
        seen.append(sim.observe())
    return seen


def walk(value: object) -> Iterator[object]:
    """Yield every object reachable from ``value`` through dataclass fields and containers."""
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from walk(getattr(value, f.name))
    elif isinstance(value, (tuple, frozenset, list)):
        for item in value:
            yield from walk(item)


# --------------------------------------------------------------------------- observation contract


def test_observation_fields_are_exactly_the_declared_contract() -> None:
    assert [f.name for f in dataclasses.fields(Observation)] == ["step", "map", "people", "burning", "allowed_actions"]
    assert [f.name for f in dataclasses.fields(MapView)] == ["width", "height", "obstacles", "exits"]
    assert [f.name for f in dataclasses.fields(PersonState)] == ["id", "position", "status"]
    assert [f.name for f in dataclasses.fields(ExitSpec)] == ["id", "position"]
    obs = Simulator(secret_scenario()).observe()
    assert list(obs.to_dict()) == ["observation_version", "step", "map", "people", "burning", "allowed_actions"]
    assert obs.to_dict()["observation_version"] == OBSERVATION_VERSION == "1.0"


def test_observation_object_graph_holds_only_public_value_types() -> None:
    allowed = {Observation, MapView, ExitSpec, PersonState, PersonStatus, Action, Pos, tuple, frozenset, int, str}
    for obs in observations(secret_scenario(), random_policy(1)):
        for node in walk(obs):
            assert type(node) in allowed, f"observation exposes a {type(node).__name__}"


def test_observation_never_contains_seed_rng_state_or_hidden_metadata() -> None:
    scenario = secret_scenario()
    sim = Simulator(scenario)
    model = random_policy(9)
    secrets: set[int] = {SECRET_SEED, derive_stream_seed(SECRET_SEED, FIRE_STREAM)}
    while True:
        obs = sim.observe()
        secrets.update(value for _name, value in sim.state.rng_state)
        leaves = {n for n in walk(obs) if type(n) is int}
        assert not leaves & secrets
        text = json.dumps(obs.to_dict())
        for forbidden in ("seed", "rng", "trapped", HIDDEN_ID, HIDDEN_DESCRIPTION, scenario.content_hash()):
            assert forbidden not in text
        for value in secrets:
            assert str(value) not in text
        if sim.done:
            break
        sim.step(model(obs))


def test_observation_is_independent_of_seed_and_hidden_metadata() -> None:
    # Deterministic fire: the seed has no effect on the world, so every observation must be equal.
    base = layout(["#######", "#a..F.#", "#b....E", "#######"], seed=1)
    other = dataclasses.replace(base, seed=SECRET_SEED, scenario_id=HIDDEN_ID, description=HIDDEN_DESCRIPTION)
    assert observations(base, greedy_exit_policy) == observations(other, greedy_exit_policy)
    # Stochastic fire: before the first random draw, the seed cannot show either.
    a, b = Simulator(secret_scenario(1)), Simulator(secret_scenario(SECRET_SEED))
    assert a.state.rng_state != b.state.rng_state
    assert a.observe() == b.observe()
    assert a.observe().to_dict() == b.observe().to_dict()


def test_trapped_diagnostic_is_evaluator_private() -> None:
    sim = Simulator(layout(["a#.E"]))
    assert sim.state.trapped == {"Pa"}
    obs = sim.observe()
    assert not hasattr(obs, "trapped")
    assert "trapped" not in obs.to_dict()


def test_observation_shows_full_current_state_and_static_map() -> None:
    scenario = fixture_scenario("fire_threatening_exit")
    sim = Simulator(scenario)
    while True:
        obs, state = sim.observe(), sim.state
        assert obs.step == state.step
        assert obs.people == state.people
        assert obs.burning == state.burning  # V1 is fully observable
        assert obs.map == MapView(scenario.width, scenario.height, scenario.obstacles, scenario.exits)
        assert obs.allowed_actions == ALLOWED_ACTIONS == tuple(Action)
        if sim.done:
            break
        sim.step(greedy_exit_policy(obs))


def test_run_episode_gives_the_policy_only_observations() -> None:
    seen: list[object] = []

    def spy(obs: Observation) -> dict[str, Action]:
        seen.append(obs)
        return greedy_exit_policy(obs)

    run_episode(secret_scenario(), spy)
    assert seen and all(type(obs) is Observation for obs in seen)


def test_observation_is_immutable() -> None:
    obs = Simulator(secret_scenario()).observe()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.step = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.map.width = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- matched evaluation


def test_matched_runs_share_initial_conditions() -> None:
    scenario = secret_scenario()
    a, b = Simulator(scenario, seed=77), Simulator(scenario, seed=77)
    assert a.state == b.state
    assert a.observe() == b.observe()
    ra, rb = a.record(), b.record()
    assert (ra.simulator_version, ra.scenario_hash, ra.seed) == (rb.simulator_version, rb.scenario_hash, rb.seed)


def reference_fire(scenario: Scenario, seed: int, steps: int) -> list[frozenset[Pos]]:
    """The fire trajectory computed with no people at all: a function of scenario, seed and step only."""
    terrain = FireTerrain.from_scenario(scenario)
    model = build_fire_model(scenario.fire)
    rng = SplitMix64(derive_stream_seed(seed, FIRE_STREAM))
    burning = frozenset(scenario.ignitions)
    trajectory = [burning]
    for step in range(1, steps + 1):
        burning = burning | model.spread(terrain, burning, step, rng)
        trajectory.append(burning)
    return trajectory


@pytest.mark.parametrize("seed", [0, 1, 20261019, SECRET_SEED])
@pytest.mark.parametrize("scenario_name", ["stochastic_room", "secret"])
def test_fire_trajectory_does_not_depend_on_actions(scenario_name: str, seed: int) -> None:
    scenario = secret_scenario() if scenario_name == "secret" else fixture_scenario(scenario_name)
    assert scenario.fire.model is FireModelName.STOCHASTIC
    models: list[Policy] = [stay_policy, greedy_exit_policy, random_policy(3), random_policy(4)]
    sims = [Simulator(scenario, seed=seed) for _ in models]
    overlap = 0
    while all(not sim.done for sim in sims):
        records = [sim.step(model(sim.observe())) for sim, model in zip(sims, models, strict=True)]
        overlap += 1
        assert len({r.ignited for r in records}) == 1
        assert len({r.state.burning for r in records}) == 1
        assert len({r.state.rng_state for r in records}) == 1
    assert overlap >= 1
    # The shared trajectory is the one the fire process produces with no people at all.
    expected = reference_fire(scenario, seed, overlap)
    assert [sims[0].record().initial_state.burning] + [s.state.burning for s in sims[0].record().steps[:overlap]] == expected


def test_fire_trajectory_matches_across_runs_of_different_length() -> None:
    scenario = secret_scenario()
    fast = run_episode(scenario, greedy_exit_policy)
    slow = run_episode(scenario, stay_policy)
    overlap = min(len(fast.steps), len(slow.steps))
    assert len(fast.steps) != len(slow.steps)
    for a, b in zip(fast.steps[:overlap], slow.steps[:overlap], strict=True):
        assert (a.ignited, a.state.burning, a.state.rng_state) == (b.ignited, b.state.burning, b.state.rng_state)
