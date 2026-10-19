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
    EnterFirePolicy,
    ExitSpec,
    FireConfig,
    FireModelName,
    MapView,
    MovementConfig,
    Observation,
    PersonState,
    PersonStatus,
    Pos,
    RulesView,
    RunRecord,
    Scenario,
    SimState,
    Simulator,
    StepRecord,
    TerminationConfig,
    run_episode,
)
from emberbench.fire import FireTerrain, build_fire_model
from emberbench.rng import SplitMix64, derive_stream_seed
from emberbench.scenario import ConflictResolution
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
    """Yield every object reachable from ``value`` through instance attributes and containers.

    It reads ``vars()``, not only the declared fields, so an attribute that is
    set outside the dataclass definition is found too. Strings, numbers and
    enum members (which are strings in this package) are leaves.
    """
    yield value
    if isinstance(value, (str, int, float, type(None))):
        return
    if hasattr(value, "__dict__") and not isinstance(value, type):
        for item in vars(value).values():
            yield from walk(item)
    elif isinstance(value, (tuple, frozenset, list)):
        for item in value:
            yield from walk(item)


# --------------------------------------------------------------------------- observation contract


def test_observation_fields_are_exactly_the_declared_contract() -> None:
    assert [f.name for f in dataclasses.fields(Observation)] == ["step", "map", "rules", "people", "burning", "allowed_actions"]
    assert [f.name for f in dataclasses.fields(MapView)] == ["width", "height", "obstacles", "exits"]
    assert [f.name for f in dataclasses.fields(RulesView)] == [
        "fire_model",
        "spread_interval",
        "spread_probability",
        "exits_flammable",
        "enter_fire",
        "conflict_resolution",
        "max_steps",
    ]
    assert [f.name for f in dataclasses.fields(PersonState)] == ["id", "position", "status"]
    assert [f.name for f in dataclasses.fields(ExitSpec)] == ["id", "position"]
    obs = Simulator(secret_scenario()).observe()
    assert list(obs.to_dict()) == ["observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"]
    assert list(obs.to_dict()["rules"]) == [f.name for f in dataclasses.fields(RulesView)]
    assert obs.to_dict()["observation_version"] == OBSERVATION_VERSION == "1.0"


PUBLIC_VALUE_TYPES = {
    Observation,
    MapView,
    RulesView,
    ExitSpec,
    PersonState,
    PersonStatus,
    Action,
    Pos,
    FireModelName,
    EnterFirePolicy,
    ConflictResolution,
    tuple,
    frozenset,
    int,
    float,
    bool,
    str,
    type(None),
}
PRIVATE_TYPES = (
    Scenario,
    Simulator,
    SimState,
    StepRecord,
    RunRecord,
    SplitMix64,
    FireTerrain,
    FireConfig,
    MovementConfig,
    TerminationConfig,
)


@pytest.mark.parametrize("scenario_name", ["secret", "one_person_one_exit"])
def test_observation_object_graph_holds_only_public_value_types(scenario_name: str) -> None:
    scenario = secret_scenario() if scenario_name == "secret" else fixture_scenario(scenario_name)
    for obs in observations(scenario, random_policy(1)):
        for node in walk(obs):
            assert not isinstance(node, PRIVATE_TYPES), f"observation references a {type(node).__name__}"
            assert type(node) in PUBLIC_VALUE_TYPES, f"observation exposes a {type(node).__name__}"


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


BASE_FIRE = FireConfig(model=FireModelName.STOCHASTIC, spread_probability=0.5)
RULE_VARIANTS: dict[str, dict[str, Any]] = {
    "fire_model": {"fire": FireConfig(model=FireModelName.DETERMINISTIC)},
    "spread_interval": {"fire": dataclasses.replace(BASE_FIRE, spread_interval=2)},
    "spread_probability": {"fire": dataclasses.replace(BASE_FIRE, spread_probability=0.25)},
    "exits_flammable": {"fire": dataclasses.replace(BASE_FIRE, exits_flammable=False)},
    "enter_fire": {"movement": MovementConfig(enter_fire=EnterFirePolicy.BLOCKED)},
    "max_steps": {"termination": TerminationConfig(max_steps=7)},
}


def rule_scenario(name: str) -> Scenario:
    """``secret`` (stochastic base rules), one variant of it, or a fixture."""
    if name == "secret":
        return secret_scenario()
    if name in RULE_VARIANTS:
        return dataclasses.replace(secret_scenario(), **RULE_VARIANTS[name])
    return fixture_scenario(name)


@pytest.mark.parametrize("rule", sorted(RULE_VARIANTS))
def test_scenarios_that_differ_only_in_a_public_rule_give_different_observations(rule: str) -> None:
    a, b = Simulator(rule_scenario("secret")).observe(), Simulator(rule_scenario(rule)).observe()
    assert a != b
    assert a.to_dict() != b.to_dict()
    changed = [name for name, value in a.rules.to_dict().items() if b.rules.to_dict()[name] != value]
    assert changed == (["fire_model", "spread_probability"] if rule == "fire_model" else [rule])
    assert dataclasses.replace(a, rules=b.rules) == b  # the rules are the only difference


def test_conflict_resolution_is_public_although_v1_has_one_value() -> None:
    assert list(ConflictResolution) == [ConflictResolution.LOWEST_ID_WINS]
    rules = Simulator(secret_scenario()).observe().to_dict()["rules"]
    assert rules["conflict_resolution"] == "lowest_id_wins"


@pytest.mark.parametrize("scenario_name", ["secret", *sorted(RULE_VARIANTS), "stochastic_room", "one_person_one_exit"])
def test_every_public_rule_matches_the_scenario_config(scenario_name: str) -> None:
    scenario = rule_scenario(scenario_name)
    sim = Simulator(scenario)
    first = sim.observe()
    sim.step(stay_policy(first))
    for obs in (first, sim.observe()):
        rules = obs.rules
        assert rules.fire_model is scenario.fire.model
        assert rules.spread_interval == scenario.fire.spread_interval
        assert rules.spread_probability == scenario.fire.spread_probability
        assert (rules.spread_probability is None) == (scenario.fire.model is FireModelName.DETERMINISTIC)
        assert rules.exits_flammable is scenario.fire.exits_flammable
        assert rules.enter_fire is scenario.movement.enter_fire
        assert rules.conflict_resolution is scenario.movement.conflict_resolution
        assert rules.max_steps == scenario.termination.max_steps


def test_rules_view_is_a_detached_copy() -> None:
    rules = RulesView.from_scenario(secret_scenario())
    assert not any(isinstance(node, PRIVATE_TYPES) for node in walk(rules))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.max_steps = 1  # type: ignore[misc]


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.rules.max_steps = 1  # type: ignore[misc]


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
