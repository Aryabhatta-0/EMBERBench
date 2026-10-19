"""The provider-independent model interface: lifecycle, validation, accounting and serialization."""

import dataclasses
import itertools
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from conftest import greedy_exit_policy, layout
from emberbench import (
    ACTION_SCHEMA_VERSION,
    CROWD_DIAGNOSTICS_VERSION,
    EVALUATION_MANIFEST_VERSION,
    EVALUATION_RESULT_VERSION,
    METRICS_VERSION,
    OBSERVATION_VERSION,
    SCHEMA_VERSION,
    SIMULATOR_VERSION,
    Action,
    ActionOutput,
    ActionValue,
    Agent,
    EvaluationError,
    EvaluationManifest,
    ExecutionMetadata,
    InvalidActionError,
    ModelOutputError,
    Observation,
    ObservationFormatError,
    PersonStatus,
    PolicyAgent,
    Scenario,
    Simulator,
    StepExecution,
    Suite,
    agent_policy,
    agent_policy_factory,
    build_evaluation_manifest,
    deserialize_actions,
    deserialize_observation,
    load_evaluation_result,
    load_suite,
    observation_from_dict,
    run_evaluation,
    serialize_actions,
    serialize_observation,
    validate_actions,
)
from emberbench.agent import ACTION_PAYLOAD_KEYS, EXIT_KEYS, MAP_KEYS, OBSERVATION_KEYS, PERSON_KEYS, RULES_KEYS
from emberbench.canonical import canonical_json
from emberbench.simulator import Policy
from test_evaluation import _ALLOWED_TYPES, _CONTAINERS, SEEDS, SMALL_IDS, STOCHASTIC_ID, _walk, stay_policy, subset
from test_suite import PINNED_HASHES, SUITE_DIR

# --------------------------------------------------------------------------- test-only reference adapters


class AlwaysStayAgent:
    """Stateless: every active person stays."""

    def act(self, observation: Observation) -> dict[str, Action]:
        return {p.id: Action.STAY for p in observation.active}


def greedy_agent() -> Agent:
    """A static callable policy behind the agent protocol."""
    return PolicyAgent(greedy_exit_policy)


class CountingAgent:
    """Stateful: remembers every step it saw, and acts greedily."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def act(self, observation: Observation) -> dict[str, Action]:
        self.steps.append(observation.step)
        return greedy_exit_policy(observation)


class Boom(Exception):
    pass


@pytest.fixture(scope="module")
def public() -> Suite:
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def small(public: Suite) -> Suite:
    return subset(public, SMALL_IDS)


@pytest.fixture(scope="module")
def manifest(small: Suite) -> EvaluationManifest:
    return build_evaluation_manifest(small, SEEDS)


def _artifact(small: Suite, manifest: EvaluationManifest, model_id: str, factory: Callable[[], Policy], path: Path) -> bytes:
    run_evaluation(small, manifest, model_id, factory, path)
    return path.read_bytes()


@pytest.fixture(scope="module")
def plain_artifacts(small: Suite, manifest: EvaluationManifest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Artifacts of the plain-function policies, the reference that adapters must match."""
    root = tmp_path_factory.mktemp("plain")
    return {
        "greedy": _artifact(small, manifest, "greedy", lambda: greedy_exit_policy, root / "greedy.json"),
        "stay": _artifact(small, manifest, "stay", lambda: stay_policy, root / "stay.json"),
    }


def evacuated_state() -> tuple[Simulator, Observation]:
    """A real step-1 observation with one evacuated (inactive) person ``Pa`` and one active ``Pb``."""
    sim = Simulator(layout(["aE", "..", "b."]))
    sim.step({"Pa": Action.EAST, "Pb": Action.STAY})
    obs = sim.observe()
    assert [(p.id, p.status) for p in obs.people] == [("Pa", PersonStatus.EVACUATED), ("Pb", PersonStatus.ACTIVE)]
    return sim, obs


# --------------------------------------------------------------------------- action validation


def test_exact_action_set_is_accepted_without_change() -> None:
    _, obs = evacuated_state()
    raw = {"Pb": Action.WEST}
    out = validate_actions(obs, raw)
    assert out == {"Pb": Action.WEST} and out is not raw
    # String values are accepted exactly as the simulator accepts them.
    assert validate_actions(obs, {"Pb": "north"}) == {"Pb": Action.NORTH}


BAD_OUTPUTS: dict[str, Any] = {
    "missing_id": {},
    "unknown_id": {"Pb": Action.STAY, "Pz": Action.STAY},
    "inactive_id": {"Pa": Action.STAY, "Pb": Action.STAY},
    "only_inactive_id": {"Pa": Action.STAY},
    "unknown_action_name": {"Pb": "up"},
    "wrong_case_action": {"Pb": "NORTH"},
    "integer_action": {"Pb": 1},
    "none_action": {"Pb": None},
    "list_action": {"Pb": ["stay"]},
    "none": None,
    "list_of_pairs": [("Pb", "stay")],
    "tuple_of_pairs": (("Pb", "stay"),),
    "string": "stay",
    "set": {"Pb"},
    "integer_key": {1: "stay"},
}


@pytest.mark.parametrize("name", sorted(BAD_OUTPUTS))
def test_invalid_model_output_fails_closed(name: str) -> None:
    sim, obs = evacuated_state()
    raw = BAD_OUTPUTS[name]
    with pytest.raises(ModelOutputError) as first:
        validate_actions(obs, raw)
    with pytest.raises(ModelOutputError) as second:
        validate_actions(obs, raw)
    assert str(first.value) == str(second.value)  # deterministic
    assert str(first.value).startswith("step 1: ")
    assert isinstance(first.value, InvalidActionError)
    assert isinstance(first.value.__cause__, InvalidActionError)
    # The simulator applies the same rules: it rejects the same output and does not change.
    before = sim.state
    with pytest.raises(InvalidActionError) as direct:
        sim.step(raw)
    assert str(first.value) == f"step 1: {direct.value}"
    assert sim.state is before


def test_actions_outside_allowed_actions_are_rejected() -> None:
    _, obs = evacuated_state()
    restricted = dataclasses.replace(obs, allowed_actions=(Action.STAY,))
    assert validate_actions(restricted, {"Pb": Action.STAY}) == {"Pb": Action.STAY}
    with pytest.raises(ModelOutputError, match=r"step 1: actions not in allowed_actions for \['Pb'\]"):
        validate_actions(restricted, {"Pb": Action.NORTH})


def test_model_exceptions_propagate_unchanged() -> None:
    error = Boom("provider down")

    class FailingAgent:
        def act(self, observation: Observation) -> dict[str, Action]:
            raise error

    _, obs = evacuated_state()
    with pytest.raises(Boom) as caught:
        agent_policy(FailingAgent())(dataclasses.replace(obs, step=0))
    assert caught.value is error


def test_model_exception_stops_evaluation_deterministically(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    def failing_factory() -> Agent:
        built.append(None)
        episode = len(built) - 1

        class Failing(CountingAgent):
            def act(self, observation: Observation) -> dict[str, Action]:
                if episode == 1 and observation.step == 2:
                    raise Boom(f"episode {episode} step {observation.step}")
                return super().act(observation)

        return Failing()

    outcomes = []
    for attempt in ("a", "b"):
        built: list[None] = []
        path = tmp_path / f"{attempt}.json"
        with pytest.raises(Boom) as caught:
            run_evaluation(small, manifest, "failing", agent_policy_factory(failing_factory), path)
        assert sorted(load_evaluation_result(path, small).completed_indices) == [0]
        outcomes.append((str(caught.value), path.read_bytes()))
    assert outcomes[0] == outcomes[1]


def test_invalid_output_stops_evaluation_deterministically(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    class DropsAPerson(CountingAgent):
        def act(self, observation: Observation) -> dict[str, Action]:
            actions = super().act(observation)
            if observation.step == 1:
                del actions[min(actions)]
            return actions

    messages = []
    for attempt in ("a", "b"):
        path = tmp_path / f"{attempt}.json"
        with pytest.raises(ModelOutputError) as caught:
            run_evaluation(small, manifest, "drops", agent_policy_factory(DropsAPerson), path)
        assert load_evaluation_result(path, small).episodes == ()
        messages.append(str(caught.value))
    assert messages[0] == messages[1]
    assert messages[0].startswith("step 1: action set must cover exactly the active people; missing=[")


# --------------------------------------------------------------------------- lifecycle and isolation


def test_deterministic_adapters_give_identical_output(public: Suite) -> None:
    for scenario in public.scenarios:
        sim = Simulator(scenario)
        for _ in range(3):
            obs = sim.observe()
            for make in (AlwaysStayAgent, greedy_agent):
                outputs = [make().act(obs) for _ in range(2)]
                agent = make()
                outputs += [agent.act(obs), agent.act(obs)]
                assert all(o == outputs[0] for o in outputs)
                assert validate_actions(obs, outputs[0]) == outputs[0]
            if sim.done:
                break
            sim.step(greedy_exit_policy(obs))


def test_new_stateful_agent_per_episode(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    instances: list[CountingAgent] = []
    factory_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def factory(*args: Any, **kwargs: Any) -> Agent:
        factory_args.append((args, kwargs))
        instances.append(CountingAgent())
        return instances[-1]

    result = run_evaluation(small, manifest, "counting", agent_policy_factory(factory), tmp_path / "c.json")
    assert factory_args == [((), {})] * len(manifest.episodes)
    assert len({id(a) for a in instances}) == len(instances) == len(manifest.episodes)
    for agent, episode in zip(instances, result.episodes, strict=True):
        # Each agent saw exactly one episode, every step once, from step 0.
        assert agent.steps == list(range(len(episode.record.steps)))


def _returning(agents: list[Agent]) -> Callable[[], Agent]:
    """An agent factory that returns the given objects in order."""
    queue = iter(agents)
    return lambda: next(queue)


@pytest.mark.parametrize(("order", "completed"), [("AA", [0]), ("ABA", [0, 1]), ("ABCB", [0, 1, 2])])
def test_reuse_of_any_earlier_agent_is_rejected(
    order: str, completed: list[int], small: Suite, manifest: EvaluationManifest, tmp_path: Path
) -> None:
    pool = {name: CountingAgent() for name in set(order)}
    path = tmp_path / "reuse.json"
    with pytest.raises(EvaluationError, match="agent of an earlier episode"):
        run_evaluation(small, manifest, "reuse", agent_policy_factory(_returning([pool[n] for n in order])), path)
    assert sorted(load_evaluation_result(path, small).completed_indices) == completed
    # The reused agent was rejected before it saw a step of the new episode.
    assert pool[order[-1]].steps == list(range(len(pool[order[-1]].steps)))


def test_distinct_agents_are_accepted(small: Suite, manifest: EvaluationManifest, plain_artifacts: dict[str, bytes], tmp_path: Path) -> None:
    a, b, c = CountingAgent(), CountingAgent(), CountingAgent()
    factory = agent_policy_factory(_returning([a, b, c]))
    assert [callable(factory()) for _ in range(3)] == [True, True, True]  # A -> B -> C

    class AlwaysEqual(CountingAgent):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 0

    # Reuse is detected by identity, never by equality: equal but distinct agents run.
    run = agent_policy_factory(_returning([AlwaysEqual() for _ in manifest.episodes]))
    assert _artifact(small, manifest, "greedy", run, tmp_path / "eq.json") == plain_artifacts["greedy"]


def test_resume_builds_agents_only_for_missing_episodes(
    small: Suite, manifest: EvaluationManifest, plain_artifacts: dict[str, bytes], tmp_path: Path
) -> None:
    path = tmp_path / "run.json"
    first: list[CountingAgent] = []

    def crashing() -> Agent:
        if len(first) == 2:
            raise Boom("interrupted")
        first.append(CountingAgent())
        return first[-1]

    with pytest.raises(Boom):
        run_evaluation(small, manifest, "greedy", agent_policy_factory(crashing), path)
    assert sorted(load_evaluation_result(path, small).completed_indices) == [0, 1]

    resumed: list[CountingAgent] = []

    def counting() -> Agent:
        resumed.append(CountingAgent())
        return resumed[-1]

    final = run_evaluation(small, manifest, "greedy", agent_policy_factory(counting), path)
    assert len(first) == 2 and len(resumed) == 2  # only episodes 2 and 3 built agents
    for agent, episode in zip(resumed, final.episodes[2:], strict=True):
        assert agent.steps == list(range(len(episode.record.steps)))
    assert path.read_bytes() == plain_artifacts["greedy"]


def test_adapter_timeout_propagates_and_the_episode_is_not_committed(
    small: Suite, manifest: EvaluationManifest, plain_artifacts: dict[str, bytes], tmp_path: Path
) -> None:
    timeout = TimeoutError("provider did not answer within 30 s")

    class TimingOut(CountingAgent):
        """A remote adapter that enforces its own timeout and raises on expiry."""

        def __init__(self, episode: int) -> None:
            super().__init__()
            self.episode = episode

        def act(self, observation: Observation) -> dict[str, Action]:
            if self.episode == 1 and observation.step == 2:
                raise timeout
            return super().act(observation)

    # Simulator level: the step that timed out is never applied.
    policy = agent_policy(TimingOut(1))
    sim = Simulator(small.scenarios[0])
    for _ in range(2):
        sim.step(policy(sim.observe()))
    before = sim.state
    with pytest.raises(TimeoutError):
        sim.step(policy(sim.observe()))
    assert sim.state is before and len(sim.record().steps) == 2

    built: list[TimingOut] = []

    def factory() -> Agent:
        built.append(TimingOut(len(built)))
        return built[-1]

    path = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError) as caught:
        run_evaluation(small, manifest, "greedy", agent_policy_factory(factory), path)
    assert caught.value is timeout  # unchanged, not wrapped
    assert built[1].steps == [0, 1]
    # Episode 1 is not committed. Episode 0 is kept and the run resumes from episode 1.
    assert sorted(load_evaluation_result(path, small).completed_indices) == [0]
    run_evaluation(small, manifest, "greedy", agent_policy_factory(greedy_agent), path)
    assert path.read_bytes() == plain_artifacts["greedy"]


def test_bound_policy_accepts_one_episode_only() -> None:
    scenario = layout(["a..E"])
    policy = agent_policy(CountingAgent())
    sim = Simulator(scenario)
    while not sim.done:
        sim.step(policy(sim.observe()))
    with pytest.raises(EvaluationError, match="bound to one episode: expected step 3, got step 0"):
        policy(Simulator(scenario).observe())
    skipping = agent_policy(CountingAgent())
    sim = Simulator(scenario)
    sim.step(skipping(sim.observe()))
    sim.step({"Pa": Action.STAY})  # the agent did not see step 1
    with pytest.raises(EvaluationError, match="expected step 1, got step 2"):
        skipping(sim.observe())


@pytest.mark.parametrize("made", [greedy_exit_policy, object(), None, {"act": "x"}])
def test_factory_must_return_an_agent(made: Any, small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="not an agent with an act"):
        run_evaluation(small, manifest, "bad", agent_policy_factory(lambda: made), tmp_path / "bad.json")
    with pytest.raises(TypeError):
        agent_policy(made)
    if not callable(made):
        with pytest.raises(TypeError):
            agent_policy_factory(made)


def test_agent_receives_only_the_observation(small: Suite, tmp_path: Path) -> None:
    secret = 0xC0FFEE1234567891  # distinctive, so a leak is easy to spot
    m = build_evaluation_manifest(small, [secret])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class RecordingAgent:
        def act(self, *args: Any, **kwargs: Any) -> dict[str, Action]:
            calls.append((args, kwargs))
            return greedy_exit_policy(args[0])

    run_evaluation(small, m, "recorder", agent_policy_factory(RecordingAgent), tmp_path / "r.json")
    assert calls
    forbidden = (str(secret), hex(secret), m.manifest_hash(), small.manifest.suite_hash(), *PINNED_HASHES.values())
    for args, kwargs in calls:
        assert kwargs == {}
        assert len(args) == 1 and type(args[0]) is Observation
        for node in _walk(args[0]):
            assert isinstance(node, _ALLOWED_TYPES + _CONTAINERS), type(node)
        text = serialize_observation(args[0])
        assert not any(f in text for f in forbidden)
    # The public API of an observation exposes no evaluator-private field.
    obs = calls[0][0][0]
    private = ("seed", "rng_state", "trapped", "scenario", "scenario_id", "scenario_hash", "manifest", "index", "record", "state")
    for view in (obs, obs.map, obs.rules, *obs.people):
        assert not any(hasattr(view, name) for name in private)
    # The seed did reach the simulator.
    assert {e.record.seed for e in load_evaluation_result(tmp_path / "r.json", small).episodes} == {secret}


def test_adapters_reproduce_plain_policy_artifacts(
    small: Suite, manifest: EvaluationManifest, plain_artifacts: dict[str, bytes], tmp_path: Path
) -> None:
    # The interface changes nothing in the artifact: same records, metrics and bytes.
    assert _artifact(small, manifest, "greedy", agent_policy_factory(greedy_agent), tmp_path / "g.json") == plain_artifacts["greedy"]
    assert _artifact(small, manifest, "greedy", agent_policy_factory(CountingAgent), tmp_path / "c.json") == plain_artifacts["greedy"]
    assert _artifact(small, manifest, "stay", agent_policy_factory(AlwaysStayAgent), tmp_path / "s.json") == plain_artifacts["stay"]
    assert load_evaluation_result(tmp_path / "s.json", small).is_complete


# --------------------------------------------------------------------------- execution accounting


class MeteredAgent(AlwaysStayAgent):
    """Reports provider accounting for each call."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, observation: Observation) -> dict[str, Action]:
        self.calls += 1
        return super().act(observation)

    def last_execution(self) -> ExecutionMetadata:
        return ExecutionMetadata(
            provider="test-provider", model="test-model", input_tokens=100 * self.calls, output_tokens=self.calls, cost=0.25, cost_currency="USD"
        )


def test_accounting_is_collected_but_never_changes_outcomes(
    small: Suite, manifest: EvaluationManifest, plain_artifacts: dict[str, bytes], tmp_path: Path
) -> None:
    log: list[StepExecution] = []
    ticks = itertools.count()
    factory = agent_policy_factory(MeteredAgent, on_step=log.append, clock=lambda: float(next(ticks)))
    result = run_evaluation(small, manifest, "stay", factory, tmp_path / "m.json")
    assert (tmp_path / "m.json").read_bytes() == plain_artifacts["stay"]

    lengths = [len(e.record.steps) for e in result.episodes]
    assert len(log) == sum(lengths)
    expected_steps = [s for n in lengths for s in range(n)]
    assert [x.step for x in log] == expected_steps
    assert all(x.latency_seconds == 1.0 for x in log)
    for x in log:
        assert x.reported is not None
        assert (x.reported.input_tokens, x.reported.output_tokens) == (100 * (x.step + 1), x.step + 1)
        assert x.to_dict()["reported"]["provider"] == "test-provider"


def test_accounting_is_optional() -> None:
    class NeverAsked(AlwaysStayAgent):
        def last_execution(self) -> ExecutionMetadata:
            raise AssertionError("accounting must not be read without an observer")

    scenario = layout(["a..E"])
    log: list[StepExecution] = []
    runs: list[tuple[Agent, Callable[[StepExecution], None] | None]] = [(AlwaysStayAgent(), log.append), (NeverAsked(), None)]
    for agent, observer in runs:
        policy = agent_policy(agent, on_step=observer)
        sim = Simulator(scenario)
        while not sim.done:
            sim.step(policy(sim.observe()))
    assert log and all(x.reported is None and x.latency_seconds >= 0 for x in log)
    assert ExecutionMetadata().to_dict() == dict.fromkeys(
        ("provider", "model", "input_tokens", "output_tokens", "cost", "cost_currency")
    )


def test_wrong_accounting_type_is_rejected() -> None:
    class BadReport(AlwaysStayAgent):
        def last_execution(self) -> Any:
            return {"input_tokens": 3}

    policy = agent_policy(BadReport(), on_step=lambda x: None)
    with pytest.raises(EvaluationError, match="must return ExecutionMetadata or None, got dict"):
        policy(Simulator(layout(["a..E"])).observe())


@pytest.mark.parametrize(
    "fields",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"output_tokens": 1.5},
        {"cost": -0.1, "cost_currency": "USD"},
        {"cost": math.nan, "cost_currency": "USD"},
        {"cost": math.inf, "cost_currency": "USD"},
        {"cost": True, "cost_currency": "USD"},
        {"cost": 0.1},
        {"cost_currency": "USD"},
        {"provider": ""},
        {"model": 5},
    ],
)
def test_invalid_accounting_is_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(EvaluationError):
        ExecutionMetadata(**fields)


# --------------------------------------------------------------------------- serialization


def _observations(scenario: Scenario, seed: int | None = None) -> list[Observation]:
    seen: list[Observation] = []

    def policy(obs: Observation) -> Mapping[str, Any]:
        seen.append(obs)
        return greedy_exit_policy(obs)

    sim = Simulator(scenario, seed=seed)
    while not sim.done:
        sim.step(policy(sim.observe()))
    seen.append(sim.observe())  # the terminal observation too
    return seen


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_serialization_round_trips_every_public_observation(public: Suite) -> None:
    for scenario in public.scenarios:
        for obs in _observations(scenario):
            text = serialize_observation(obs)
            assert deserialize_observation(text) == obs
            assert observation_from_dict(json.loads(text)) == obs
            assert serialize_observation(deserialize_observation(text)) == text


def test_serialization_is_canonical(public: Suite) -> None:
    obs = _observations(public.get(STOCHASTIC_ID), seed=5)[-1]
    text = serialize_observation(obs)
    data = json.loads(text)
    assert text == canonical_json(data) and text.isascii() and " " not in text
    assert list(data) == sorted(data)
    assert [p["id"] for p in data["people"]] == sorted(p["id"] for p in data["people"])
    assert data["burning"] == sorted(data["burning"]) and len(data["burning"]) > 1
    assert data["observation_version"] == OBSERVATION_VERSION
    # Equal observations built in another order serialize identically.
    shuffled = dataclasses.replace(obs, burning=frozenset(reversed(sorted(obs.burning))))
    assert serialize_observation(shuffled) == text


def test_serialization_holds_only_model_visible_fields(public: Suite) -> None:
    secret = 0xBADC0FFEE0DDF00D
    base = public.get(STOCHASTIC_ID)
    scenario = dataclasses.replace(base, seed=secret)
    sim = Simulator(scenario)
    for _ in range(4):
        sim.step(greedy_exit_policy(sim.observe()))
    text = serialize_observation(sim.observe())
    assert _keys(json.loads(text)) == OBSERVATION_KEYS | MAP_KEYS | EXIT_KEYS | RULES_KEYS | PERSON_KEYS
    for word in ("seed", "rng", "hash", "trapped", "scenario", "description", "manifest", "episode", "index", "model_id", "metrics"):
        assert word not in text
    rng_positions = [str(v) for _, v in sim.state.rng_state]
    for leak in (str(secret), hex(secret), scenario.content_hash(), base.description, *rng_positions):
        assert leak not in text


def test_serialization_ignores_evaluator_private_data(public: Suite) -> None:
    base = public.get(STOCHASTIC_ID)
    other = dataclasses.replace(base, scenario_id="v1-other-id", description="hidden", seed=12345)
    a, b = Simulator(base, seed=1), Simulator(other, seed=2)
    assert a.state.rng_state != b.state.rng_state
    assert serialize_observation(a.observe()) == serialize_observation(b.observe())


def test_trapped_diagnostic_is_not_serialized() -> None:
    trapped, free = layout(["a#E"]), layout(["a.E"])
    sim = Simulator(trapped)
    assert sim.state.trapped == frozenset({"Pa"})
    text = serialize_observation(sim.observe())
    assert "trapped" not in text
    # Only the wall differs between the two texts, not any trapped marker.
    assert json.loads(text)["people"] == json.loads(serialize_observation(Simulator(free).observe()))["people"]


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(d: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        target: Any = d
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value

    return mutate


def _delete(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda d: d.pop(key)


MALFORMED: dict[str, Callable[[dict[str, Any]], Any]] = {
    "missing_key": _delete("burning"),
    "extra_seed": _set("seed", 1),
    "extra_trapped": _set("trapped", []),
    "other_version": _set("observation_version", "2.0"),
    "bool_step": _set("step", True),
    "negative_step": _set("step", -1),
    "zero_width": _set("map.width", 0),
    "obstacle_out_of_grid": _set("map.obstacles.0", [99, 0]),
    "position_triple": _set("people.0.position", [0, 0, 0]),
    "unsorted_people": lambda d: d["people"].reverse(),
    "duplicate_burning": lambda d: d["burning"].append(d["burning"][-1]),
    "unsorted_obstacles": lambda d: d["map"]["obstacles"].reverse(),
    "bad_status": _set("people.0.status", "trapped"),
    "bad_fire_model": _set("rules.fire_model", "magic"),
    "int_probability": _set("rules.spread_probability", 1),
    "probability_out_of_range": _set("rules.spread_probability", 1.5),
    "probability_for_deterministic": lambda d: d["rules"].update(fire_model="deterministic"),
    "string_flammable": _set("rules.exits_flammable", "yes"),
    "unknown_action": _set("allowed_actions.0", "jump"),
    "duplicate_action": lambda d: d["allowed_actions"].append("stay"),
    "people_not_list": _set("people", {}),
    "exit_extra_key": _set("map.exits.0.hash", "x"),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_serialized_observations_are_rejected(name: str, public: Suite) -> None:
    obs = _observations(public.get(STOCHASTIC_ID), seed=5)[3]
    data = json.loads(serialize_observation(obs))
    MALFORMED[name](data)
    with pytest.raises(ObservationFormatError):
        observation_from_dict(data)
    with pytest.raises(ObservationFormatError):
        deserialize_observation(json.dumps(data, sort_keys=True, separators=(",", ":")))


def test_non_canonical_text_is_rejected(public: Suite) -> None:
    obs = Simulator(public.get(STOCHASTIC_ID)).observe()
    text = serialize_observation(obs)
    data = json.loads(text)
    for variant in (json.dumps(data, indent=2, sort_keys=True), json.dumps(data), text + "\n", "{" + text[1:-1] + ',"step":0}'):
        assert variant != text
        with pytest.raises(ObservationFormatError):
            deserialize_observation(variant)
    with pytest.raises(ObservationFormatError, match="invalid JSON"):
        deserialize_observation(text[:-1])
    with pytest.raises(ObservationFormatError):
        deserialize_observation(text.encode())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        serialize_observation(data)


# --------------------------------------------------------------------------- action typing


class MixedAgent:
    """Returns Action members and action strings: both are valid ActionValues."""

    def act(self, observation: Observation) -> ActionOutput:
        out: dict[str, ActionValue] = {}
        for i, p in enumerate(observation.active):
            out[p.id] = Action.STAY if i % 2 else "stay"
        return out


def test_action_output_type_matches_runtime_acceptance(small: Suite) -> None:
    agent: Agent = MixedAgent()  # checked by mypy --strict
    assert isinstance(agent, Agent)
    obs = Simulator(small.scenarios[0]).observe()
    raw = agent.act(obs)
    assert {type(v) for v in raw.values()} == {Action, str} or len(raw) == 1
    assert validate_actions(obs, raw) == {p.id: Action.STAY for p in obs.active}
    assert agent_policy(agent)(obs) == {p.id: Action.STAY for p in obs.active}


# --------------------------------------------------------------------------- action payload


def test_action_payload_is_canonical_and_versioned() -> None:
    _, obs = evacuated_state()
    text = serialize_actions(obs, {"Pb": "north"})
    assert text == '{"action_schema_version":"1.0","actions":{"Pb":"north"},"step":1}'
    assert deserialize_actions(obs, text) == {"Pb": Action.NORTH}
    assert serialize_actions(obs, {"Pb": Action.NORTH}) == text
    assert ACTION_SCHEMA_VERSION == "1.0" and OBSERVATION_VERSION == "1.0"


def test_action_payload_sorts_person_ids_and_round_trips(public: Suite) -> None:
    scenario = public.get("v1-pub-02-single-exit-dense")
    for obs in _observations(scenario)[:-1]:
        actions = greedy_exit_policy(obs)
        backwards = dict(reversed(list(actions.items())))
        text = serialize_actions(obs, backwards)
        assert text == serialize_actions(obs, actions)
        data = json.loads(text)
        assert list(data["actions"]) == sorted(p.id for p in obs.active) and len(data["actions"]) > 1 or len(obs.active) <= 1
        assert deserialize_actions(obs, text) == validate_actions(obs, actions)
        # Only the declared keys and the active person IDs: no evaluator-private metadata.
        assert _keys(data) == ACTION_PAYLOAD_KEYS | {p.id for p in obs.active}


@pytest.mark.parametrize("name", sorted(BAD_OUTPUTS))
def test_action_payload_cannot_be_built_from_invalid_output(name: str) -> None:
    _, obs = evacuated_state()
    with pytest.raises(ModelOutputError, match=r"^step 1: "):
        serialize_actions(obs, BAD_OUTPUTS[name])


BAD_PAYLOADS: dict[str, Any] = {
    "missing_actions": {"action_schema_version": "1.0", "step": 1},
    "missing_version": {"actions": {"Pb": "stay"}, "step": 1},
    "missing_step": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}},
    "extra_seed": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": 1, "seed": 7},
    "extra_metadata": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": 1, "model": "x"},
    "other_version": {"action_schema_version": "2.0", "actions": {"Pb": "stay"}, "step": 1},
    "float_version": {"action_schema_version": 1.0, "actions": {"Pb": "stay"}, "step": 1},
    "wrong_step": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": 2},
    "earlier_step": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": 0},
    "bool_step": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": True},
    "string_step": {"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": "1"},
    "actions_list": {"action_schema_version": "1.0", "actions": [["Pb", "stay"]], "step": 1},
    "integer_action": {"action_schema_version": "1.0", "actions": {"Pb": 0}, "step": 1},
    "missing_person": {"action_schema_version": "1.0", "actions": {}, "step": 1},
    "unknown_person": {"action_schema_version": "1.0", "actions": {"Pb": "stay", "Pz": "stay"}, "step": 1},
    "inactive_person": {"action_schema_version": "1.0", "actions": {"Pa": "stay", "Pb": "stay"}, "step": 1},
    "invalid_action": {"action_schema_version": "1.0", "actions": {"Pb": "up"}, "step": 1},
    "top_level_list": [{"action_schema_version": "1.0", "actions": {"Pb": "stay"}, "step": 1}],
}


@pytest.mark.parametrize("name", sorted(BAD_PAYLOADS))
def test_malformed_action_payloads_are_rejected(name: str) -> None:
    _, obs = evacuated_state()
    text = canonical_json(BAD_PAYLOADS[name])
    with pytest.raises(ModelOutputError, match=r"^step 1: ") as first:
        deserialize_actions(obs, text)
    with pytest.raises(ModelOutputError) as second:
        deserialize_actions(obs, text)
    assert str(first.value) == str(second.value)


def test_action_payload_checks_allowed_actions() -> None:
    _, obs = evacuated_state()
    restricted = dataclasses.replace(obs, allowed_actions=(Action.STAY,))
    text = serialize_actions(obs, {"Pb": "north"})
    with pytest.raises(ModelOutputError, match="allowed_actions"):
        deserialize_actions(restricted, text)


def test_non_canonical_action_payloads_are_rejected() -> None:
    _, obs = evacuated_state()
    text = serialize_actions(obs, {"Pb": "north"})
    data = json.loads(text)
    variants = [
        json.dumps(data, sort_keys=True),  # spaces
        json.dumps(data, indent=2, sort_keys=True),
        json.dumps({"step": 1, "actions": {"Pb": "north"}, "action_schema_version": "1.0"}, separators=(",", ":")),
        text + "\n",
        '{"action_schema_version":"1.0","actions":{"Pb":"stay","Pb":"north"},"step":1}',  # duplicate key
    ]
    for variant in variants:
        assert variant != text
        with pytest.raises(ModelOutputError, match="not in canonical form"):
            deserialize_actions(obs, variant)
    with pytest.raises(ModelOutputError, match="not valid JSON"):
        deserialize_actions(obs, text[:-1])
    with pytest.raises(ModelOutputError, match="must be str"):
        deserialize_actions(obs, text.encode())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- versions


def test_no_contract_version_changes() -> None:
    assert (SIMULATOR_VERSION, OBSERVATION_VERSION, SCHEMA_VERSION, METRICS_VERSION) == ("0.2.0", "1.0", "1.0", "1.0")
    assert (EVALUATION_MANIFEST_VERSION, EVALUATION_RESULT_VERSION, CROWD_DIAGNOSTICS_VERSION) == ("1.0", "1.0", "1.0")
