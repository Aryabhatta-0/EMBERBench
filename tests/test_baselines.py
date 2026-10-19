"""Reference baselines: registry, evaluation integration, leakage, determinism, sanity and the privileged bound."""

import ast
import dataclasses
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import layout
from emberbench import (
    BASELINE_IDS,
    BASELINES_VERSION,
    Action,
    Agent,
    BaselineError,
    EvaluationManifest,
    EvaluationResult,
    FireConfig,
    Observation,
    Scenario,
    Suite,
    TerminationConfig,
    agent_policy,
    agent_policy_factory,
    baseline_factory,
    build_evaluation_manifest,
    compute_episode_metrics,
    generate_suite,
    freeze_generated_suite,
    get_baseline,
    load_suite,
    run_episode,
    run_evaluation,
)
from emberbench import baselines as baselines_module
from emberbench import oracle
from emberbench.agent import ActionOutput, AgentFactory
from emberbench.baselines import BASELINES, NoConfig, RandomLegalConfig
from emberbench.canonical import canonical_json, sha256_hex
from emberbench.events import RunRecord
from emberbench.generator import GeneratorConfig
from test_evaluation import subset
from test_generator import BASE_CONFIG, BATCH, SEED
from test_suite import SUITE_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent

# Small fixed development manifest: canonical sparse, corridor bottleneck,
# 100-person crowd and stochastic route invalidation, under development seeds.
DEV_IDS = (
    "v1-pub-01-single-exit-sparse",
    "v1-pub-03-corridor-bottleneck",
    "v1-pub-09-crowd-100-split-flow",
    "v1-pub-10-route-invalidation",
)
DEV_SEEDS = (0, 1)
STOCHASTIC_ID = "v1-pub-10-route-invalidation"

# SHA-256 over the record digests of every episode, in manifest order. A change
# means a baseline acts differently: bump BASELINES_VERSION (or the simulator).
GOLDEN_RESULT_DIGESTS = {
    "random-legal-v1": "084330fa3bd69067066ebc0c037fb8ddea9f3bb8092c3b23cd8bb4989142345e",
    "nearest-exit-greedy-v1": "f6e5c147b10f3372b0434082c4e5fc2fdcd69ec93b8bf7c5e2fe9c58c65c0875",
    "static-shortest-path-v1": "f6e5c147b10f3372b0434082c4e5fc2fdcd69ec93b8bf7c5e2fe9c58c65c0875",
    "fire-aware-shortest-path-v1": "bcc4bae71f28831eb343417e5cc081a7c25251613911f38cc7bbb1ebfd96caa1",
    "congestion-aware-v1": "fa4f15eaf5c44efbbeb83512182a599275feece8e2185f195dffa73eff0b1575",
}

# The first two steps of every baseline on the canonical sparse scenario, seed 0.
GOLDEN_SPARSE_TRACES: dict[str, list[dict[str, str]]] = {
    "random-legal-v1": [{"P001": "south", "P002": "north", "P003": "stay", "P004": "north"}, {"P001": "stay", "P002": "south", "P003": "south", "P004": "north"}],
    "nearest-exit-greedy-v1": [{"P001": "south", "P002": "north", "P003": "north", "P004": "south"}, {"P001": "south", "P002": "north", "P003": "east", "P004": "east"}],
    "static-shortest-path-v1": [{"P001": "south", "P002": "north", "P003": "north", "P004": "south"}, {"P001": "south", "P002": "north", "P003": "east", "P004": "east"}],
    "fire-aware-shortest-path-v1": [{"P001": "south", "P002": "north", "P003": "north", "P004": "south"}, {"P001": "south", "P002": "north", "P003": "east", "P004": "east"}],
    "congestion-aware-v1": [{"P001": "south", "P002": "north", "P003": "north", "P004": "south"}, {"P001": "south", "P002": "north", "P003": "east", "P004": "east"}],
}

GOLDEN_CONFIG_DIGESTS = {
    "random-legal-v1": "636a298cab678e283d09f0aaaf5727fb15b845b39a5dcd00a9e643318a6c9632",
    "nearest-exit-greedy-v1": "11a8791644591e60850afb1dc6d1d1a0a0159644592d547549a36004a6e8d5c5",
    "static-shortest-path-v1": "b62da8fe4a8e58f621cabe2a08249ac2381f1daf24522563a0f28c9633d20e60",
    "fire-aware-shortest-path-v1": "983691b4d94c29854afdc4e3cbb58c5c66b6969649cfaac42906befa3c7ce0c7",
    "congestion-aware-v1": "5578eaebf7cd6c05ee71420502ac674e25a666461772dc69fcb240f3e17513a4",
}

FIRE_STILL = FireConfig(spread_interval=99)  # no spread within the short horizons below

# One route passes a burning cell. Static walks into it, fire-aware goes around.
DETOUR = layout(
    [
        "#######",
        "#a.F.E#",
        "#.###.#",
        "#.....#",
        "#######",
    ],
    scenario_id="detour",
    fire=FIRE_STILL,
    termination=TerminationConfig(max_steps=30),
)

# Sixteen people next to a near exit behind a door; a second exit is a little further.
DOOR_BOTTLENECK = layout(
    [
        "##############",
        "#F...........#",
        "#............#",
        "E...abcdefgh.#",
        "#...ijklmnop.#",
        "#######.######",
        "#######E######",
    ],
    scenario_id="door-bottleneck",
    fire=FIRE_STILL,
    termination=TerminationConfig(max_steps=60),
)

# The exit is straight north, behind a wall. A beeline stalls; a path goes around.
WALL_BETWEEN = layout(
    [
        "#######",
        "#..E..#",
        "#.###.#",
        "#..a..#",
        "#F....#",
        "#######",
    ],
    scenario_id="wall-between",
    fire=FIRE_STILL,
    termination=TerminationConfig(max_steps=30),
)


class StayAgent:
    def act(self, observation: Observation) -> dict[str, Action]:
        return {p.id: Action.STAY for p in observation.active}


def run(scenario: Scenario, agent: Agent, *, seed: int | None = None) -> RunRecord:
    return run_episode(scenario, agent_policy(agent), seed=seed)


def action_trace(record: RunRecord) -> list[dict[str, str]]:
    return [{pid: action.value for pid, action in step.actions} for step in record.steps]


def result_digest(result: EvaluationResult) -> str:
    return sha256_hex([e.record.digest() for e in result.episodes])


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def public() -> Suite:
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def dev_suite(public: Suite) -> Suite:
    return subset(public, DEV_IDS)


@pytest.fixture(scope="module")
def dev_manifest(dev_suite: Suite) -> EvaluationManifest:
    return build_evaluation_manifest(dev_suite, DEV_SEEDS)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("baselines")


@pytest.fixture(scope="module")
def results(dev_suite: Suite, dev_manifest: EvaluationManifest, artifacts: Path) -> dict[str, EvaluationResult]:
    """Every baseline and an always-stay control, run unchanged through ``run_evaluation`` on one manifest."""
    factories: dict[str, AgentFactory] = {bid: baseline_factory(bid) for bid in BASELINE_IDS}
    factories["always-stay"] = StayAgent
    return {
        name: run_evaluation(dev_suite, dev_manifest, name, agent_policy_factory(factory), artifacts / f"{name}.json")
        for name, factory in factories.items()
    }


def total(result: EvaluationResult, read: Callable[[Any], int]) -> int:
    return sum(read(e.metrics) for e in result.episodes)


# --------------------------------------------------------------------------- registry


def test_registry_lists_the_five_fair_baselines() -> None:
    assert BASELINES_VERSION == "1.0"
    assert BASELINE_IDS == (
        "random-legal-v1",
        "nearest-exit-greedy-v1",
        "static-shortest-path-v1",
        "fire-aware-shortest-path-v1",
        "congestion-aware-v1",
    )
    for bid in BASELINE_IDS:
        spec = get_baseline(bid)
        assert spec.baseline_id == bid
        assert spec.baseline_version == "1.0"
        assert spec.description and spec.description == spec.description.strip()
        assert spec.privileged is False
        assert spec.descriptor() == {
            "baselines_version": "1.0",
            "baseline_id": bid,
            "baseline_version": "1.0",
            "privileged": False,
            "config": spec.default_config.to_dict(),
        }


def test_config_digests_are_canonical_and_pinned() -> None:
    digests = {bid: get_baseline(bid).config_digest() for bid in BASELINE_IDS}
    assert digests == GOLDEN_CONFIG_DIGESTS
    assert len(set(digests.values())) == len(digests)
    spec = get_baseline("random-legal-v1")
    assert spec.config_digest(RandomLegalConfig(seed=0)) == digests["random-legal-v1"]
    assert spec.config_digest(RandomLegalConfig(seed=1)) != digests["random-legal-v1"]
    assert spec.config_digest() == sha256_hex(json.loads(canonical_json(spec.descriptor())))


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        BASELINES["oracle"] = BASELINES["random-legal-v1"]  # type: ignore[index]


@pytest.mark.parametrize("bad", ["", "oracle", "random-legal", "RANDOM-LEGAL-V1", "emberbench.baselines.RandomLegalAgent", None, 1])
def test_unknown_baseline_ids_fail_closed(bad: Any) -> None:
    with pytest.raises(BaselineError, match="unknown baseline"):
        get_baseline(bad)
    with pytest.raises(BaselineError, match="unknown baseline"):
        baseline_factory(bad)


@pytest.mark.parametrize(
    ("bid", "config", "message"),
    [
        ("random-legal-v1", {"seed": -1}, "seed"),
        ("random-legal-v1", {"seed": 2**64}, "seed"),
        ("random-legal-v1", {"seed": True}, "seed"),
        ("random-legal-v1", {"seed": 1.0}, "seed"),
        ("random-legal-v1", {}, "keys"),
        ("random-legal-v1", {"seed": 1, "extra": 0}, "keys"),
        ("static-shortest-path-v1", {"seed": 1}, "keys"),
        ("static-shortest-path-v1", RandomLegalConfig(), "config must be NoConfig"),
        ("random-legal-v1", NoConfig(), "config must be RandomLegalConfig"),
    ],
)
def test_invalid_configs_fail_closed(bid: str, config: Any, message: str) -> None:
    with pytest.raises(BaselineError, match=message):
        baseline_factory(bid, config)


def test_config_round_trips_through_its_dict_form() -> None:
    config = RandomLegalConfig(seed=2**64 - 1)
    assert RandomLegalConfig.from_dict(json.loads(canonical_json(config.to_dict()))) == config
    a = baseline_factory("random-legal-v1", {"seed": 5})()
    b = baseline_factory("random-legal-v1", RandomLegalConfig(seed=5))()
    obs = run_episode_observation(WALL_BETWEEN)
    assert a.act(obs) == b.act(obs)


def run_episode_observation(scenario: Scenario) -> Observation:
    from emberbench import Simulator

    return Simulator(scenario).observe()


# --------------------------------------------------------------------------- interface and evaluation integration


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_factory_takes_no_arguments_and_builds_a_new_agent_each_time(bid: str) -> None:
    factory = baseline_factory(bid)
    assert inspect.signature(factory).parameters == {}
    first, second = factory(), factory()
    assert first is not second
    assert isinstance(first, Agent) and isinstance(second, Agent)
    # The interface's own guard accepts it for many episodes.
    policy_factory = agent_policy_factory(factory)
    assert policy_factory() is not policy_factory()


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_agent_api_receives_only_config_and_observation(bid: str) -> None:
    cls = get_baseline(bid).build
    assert inspect.isclass(cls)
    assert list(inspect.signature(cls).parameters) == ["config"]
    act = inspect.signature(getattr(cls, "act"))
    assert list(act.parameters) == ["self", "observation"]
    assert act.parameters["observation"].annotation in (Observation, "Observation")


def test_every_baseline_runs_unchanged_through_run_evaluation(
    results: dict[str, EvaluationResult], dev_manifest: EvaluationManifest
) -> None:
    for name, result in results.items():
        assert result.model_id == name
        assert result.manifest == dev_manifest
        assert [e.episode for e in result.episodes] == list(dev_manifest.episodes)
    hashes = {r.manifest_hash for r in results.values()}
    assert hashes == {dev_manifest.manifest_hash()}


def test_golden_result_digests(results: dict[str, EvaluationResult]) -> None:
    assert {bid: result_digest(results[bid]) for bid in BASELINE_IDS} == GOLDEN_RESULT_DIGESTS


def test_golden_sparse_action_traces(results: dict[str, EvaluationResult]) -> None:
    traces = {}
    for bid in BASELINE_IDS:
        episode = next(e for e in results[bid].episodes if e.episode.scenario_id == DEV_IDS[0] and e.episode.seed == 0)
        traces[bid] = action_trace(episode.record)[:2]
    assert traces == GOLDEN_SPARSE_TRACES


# --------------------------------------------------------------------------- leakage


class SpyAgent:
    """Records every argument the evaluator passes to ``act`` and forwards to a baseline."""

    def __init__(self, inner: Agent, calls: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        self._inner = inner
        self._calls = calls

    def act(self, *args: Any, **kwargs: Any) -> ActionOutput:
        self._calls.append((args, kwargs))
        return self._inner.act(*args, **kwargs)


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_factory_and_act_receive_no_evaluator_metadata(
    bid: str, dev_suite: Suite, dev_manifest: EvaluationManifest, tmp_path: Path
) -> None:
    factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    act_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    inner = baseline_factory(bid)

    def factory(*args: Any, **kwargs: Any) -> Agent:
        factory_calls.append((args, kwargs))
        return SpyAgent(inner(), act_calls)

    small = subset(dev_suite, (STOCHASTIC_ID,))
    manifest = build_evaluation_manifest(small, [2**63 + 12345])
    run_evaluation(small, manifest, bid, agent_policy_factory(factory), tmp_path / "spy.json")
    assert factory_calls == [((), {})]
    assert act_calls
    for args, kwargs in act_calls:
        assert kwargs == {} and len(args) == 1 and type(args[0]) is Observation
        text = canonical_json(args[0].to_dict())
        for secret in (str(2**63 + 12345), f"{2**63 + 12345:x}", manifest.manifest_hash(), STOCHASTIC_ID):
            assert secret not in text


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_same_observation_gives_same_actions_under_any_evaluator_seed(bid: str, public: Suite) -> None:
    """Actions are a function of the visible observation only, whatever the evaluator seed."""
    scenario = next(s for s in public.scenarios if s.scenario_id == STOCHASTIC_ID)
    seen: dict[str, str] = {}
    for seed in (0, 1, 2**64 - 1, 987654321):
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        agent = SpyAgent(baseline_factory(bid)(), calls)
        record = run(scenario, agent, seed=seed)
        for (args, _), step in zip(calls, record.steps, strict=True):
            key = canonical_json(args[0].to_dict())
            actions = canonical_json(sorted((pid, a.value) for pid, a in step.actions))
            assert seen.setdefault(key, actions) == actions
    # Seeds that differ only in the hidden stream give identical deterministic-fire runs.
    sparse = next(s for s in public.scenarios if s.scenario_id == DEV_IDS[0])
    traces = {canonical_json(action_trace(run(sparse, baseline_factory(bid)(), seed=seed))) for seed in (0, 7, 2**40)}
    assert len(traces) == 1


FORBIDDEN_NAMES = frozenset(
    {
        "Scenario",
        "EvaluationManifest",
        "EpisodeSpec",
        "RunRecord",
        "SimState",
        "Simulator",
        "run_episode",
        "SplitMix64",
        "derive_stream_seed",
        "FireTerrain",
        "build_fire_model",
        "GenerationManifest",
        "CrowdDiagnostics",
    }
)
FORBIDDEN_MODULES = frozenset(
    {"random", "secrets", "time", "os", "uuid", "emberbench.rng", "emberbench.simulator", "emberbench.state", "emberbench.scenario",
     "emberbench.evaluation", "emberbench.events", "emberbench.generator", "emberbench.crowd", "emberbench.fire", "emberbench.oracle"}
)


def _imports(module: Any) -> tuple[set[str], set[str]]:
    tree = ast.parse(inspect.getsource(module))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.add(f"emberbench.{base}" if node.level else base)
            names.update(alias.name for alias in node.names)
    return modules, names


def test_baseline_module_imports_no_evaluator_or_rng_api() -> None:
    modules, names = _imports(baselines_module)
    assert not modules & FORBIDDEN_MODULES
    assert not names & FORBIDDEN_NAMES
    # Structural check as well: nothing evaluator-side is reachable from the module namespace.
    from emberbench import RunRecord as R, Scenario as S, SimState, Simulator
    from emberbench.evaluation import EvaluationManifest as M
    from emberbench.rng import SplitMix64

    forbidden_objects = {id(o) for o in (R, S, SimState, Simulator, M, SplitMix64, oracle)}
    assert not {id(v) for v in vars(baselines_module).values()} & forbidden_objects


# --------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_repeated_run_gives_byte_identical_artifact(
    bid: str, results: dict[str, EvaluationResult], dev_suite: Suite, dev_manifest: EvaluationManifest, artifacts: Path, tmp_path: Path
) -> None:
    again = run_evaluation(dev_suite, dev_manifest, bid, agent_policy_factory(baseline_factory(bid)), tmp_path / "again.json")
    assert (tmp_path / "again.json").read_bytes() == (artifacts / f"{bid}.json").read_bytes()
    assert again.to_file_text() == results[bid].to_file_text()


class Reordered:
    """Returns the inner baseline's actions in reverse insertion order."""

    def __init__(self, inner: Agent) -> None:
        self._inner = inner

    def act(self, observation: Observation) -> ActionOutput:
        actions = dict(self._inner.act(observation))
        return {pid: actions[pid] for pid in reversed(list(actions))}


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_action_dict_insertion_order_does_not_matter(bid: str, dev_suite: Suite, dev_manifest: EvaluationManifest, artifacts: Path, tmp_path: Path) -> None:
    inner = baseline_factory(bid)
    run_evaluation(dev_suite, dev_manifest, bid, agent_policy_factory(lambda: Reordered(inner())), tmp_path / "reordered.json")
    assert (tmp_path / "reordered.json").read_bytes() == (artifacts / f"{bid}.json").read_bytes()


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_people_order_in_the_observation_does_not_matter(bid: str, public: Suite) -> None:
    scenario = next(s for s in public.scenarios if s.scenario_id == "v1-pub-09-crowd-100-split-flow")
    from emberbench import Simulator

    sim = Simulator(scenario)
    for _ in range(5):
        obs = sim.observe()
        agent = baseline_factory(bid)()
        expected = agent.act(obs)
        shuffled = dataclasses.replace(obs, people=tuple(reversed(obs.people)), burning=frozenset(sorted(obs.burning, reverse=True)))
        assert dict(agent.act(shuffled)) == dict(expected)
        sim.step(expected)


_HASHSEED_SCRIPT = """
import json, sys
from emberbench import BASELINE_IDS, agent_policy, baseline_factory, load_suite, run_episode
from emberbench.canonical import sha256_hex
suite = load_suite(sys.argv[1])
out = {}
for bid in BASELINE_IDS:
    digests = []
    for scenario in suite.scenarios:
        if scenario.scenario_id in sys.argv[2:]:
            for seed in (0, 1):
                digests.append(run_episode(scenario, agent_policy(baseline_factory(bid)()), seed=seed).digest())
    out[bid] = sha256_hex(digests)
print(json.dumps(out, sort_keys=True))
"""


def test_outputs_do_not_depend_on_pythonhashseed() -> None:
    ids = ["v1-pub-01-single-exit-sparse", STOCHASTIC_ID]
    outputs = []
    for hashseed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(REPO_ROOT / "src")}
        done = subprocess.run(
            [sys.executable, "-c", _HASHSEED_SCRIPT, str(SUITE_DIR), *ids], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert set(json.loads(outputs[0])) == set(BASELINE_IDS)


# --------------------------------------------------------------------------- random legal


def test_random_legal_is_deterministic_for_its_public_seed(public: Suite) -> None:
    scenario = next(s for s in public.scenarios if s.scenario_id == "v1-pub-09-crowd-100-split-flow")

    def trace(seed: int) -> str:
        return canonical_json(action_trace(run(scenario, baseline_factory("random-legal-v1", {"seed": seed})())))

    assert trace(0) == trace(0)
    assert trace(0) != trace(1)
    # The baseline seed is model configuration: it does not change the evaluator seed or the fire.
    record_a = run(scenario, baseline_factory("random-legal-v1", {"seed": 0})(), seed=3)
    record_b = run(scenario, baseline_factory("random-legal-v1", {"seed": 1})(), seed=3)
    assert record_a.seed == record_b.seed == 3


def test_random_legal_only_chooses_statically_legal_actions(public: Suite) -> None:
    scenario = next(s for s in public.scenarios if s.scenario_id == "v1-pub-08-maze-deceptive-exit")
    record = run(scenario, baseline_factory("random-legal-v1")())
    walls = scenario.walls
    states = [record.initial_state, *(s.state for s in record.steps)]
    for before, step in zip(states, record.steps):
        for pid, action in step.actions:
            target = before.person(pid).position.offset(*action.delta)
            assert 0 <= target.row < scenario.height and 0 <= target.col < scenario.width
            assert target not in walls


# --------------------------------------------------------------------------- comparative sanity (broad, not a ranking)


def test_shortest_path_baselines_beat_always_stay(results: dict[str, EvaluationResult]) -> None:
    stay = total(results["always-stay"], lambda m: m.safety.evacuated)
    for bid in ("static-shortest-path-v1", "fire-aware-shortest-path-v1", "congestion-aware-v1"):
        assert total(results[bid], lambda m: m.safety.evacuated) > stay


def test_fire_aware_avoids_a_currently_burning_route_that_static_takes() -> None:
    obs = run_episode_observation(DETOUR)
    assert baseline_factory("static-shortest-path-v1")().act(obs) == {"Pa": Action.EAST}
    assert baseline_factory("fire-aware-shortest-path-v1")().act(obs) == {"Pa": Action.SOUTH}
    static = compute_episode_metrics(DETOUR, run(DETOUR, baseline_factory("static-shortest-path-v1")()))
    aware = compute_episode_metrics(DETOUR, run(DETOUR, baseline_factory("fire-aware-shortest-path-v1")()))
    assert (static.safety.evacuated, static.safety.casualties) == (0, 1)
    assert (aware.safety.evacuated, aware.safety.casualties) == (1, 0)


def test_congestion_aware_eases_a_designed_bottleneck() -> None:
    aware = compute_episode_metrics(DOOR_BOTTLENECK, run(DOOR_BOTTLENECK, baseline_factory("fire-aware-shortest-path-v1")()))
    congestion = compute_episode_metrics(DOOR_BOTTLENECK, run(DOOR_BOTTLENECK, baseline_factory("congestion-aware-v1")()))
    assert aware.safety.evacuated == congestion.safety.evacuated == 16
    assert congestion.time.person_time_to_safety_total < aware.time.person_time_to_safety_total
    assert congestion.exits.max_exit_share is not None and aware.exits.max_exit_share is not None
    assert congestion.exits.max_exit_share < aware.exits.max_exit_share


def test_nearest_exit_greedy_is_intentionally_weak_behind_a_wall() -> None:
    greedy = run(WALL_BETWEEN, baseline_factory("nearest-exit-greedy-v1")())
    static = run(WALL_BETWEEN, baseline_factory("static-shortest-path-v1")())
    assert {a for step in greedy.steps for _, a in step.actions} == {Action.STAY}
    assert compute_episode_metrics(WALL_BETWEEN, static).safety.evacuated == 1


def test_fire_aware_falls_back_to_fleeing_when_no_safe_route_exists() -> None:
    enclosed = layout(
        [
            "#######",
            "#F.a.F#",
            "#######",
            "#E....#",
            "#######",
        ],
        fire=FIRE_STILL,
        termination=TerminationConfig(max_steps=5),
    )
    obs = run_episode_observation(enclosed)
    # Every move ties on distance to the fire; ties go to stay.
    for bid in ("fire-aware-shortest-path-v1", "congestion-aware-v1"):
        assert baseline_factory(bid)().act(obs) == {"Pa": Action.STAY}
    shifted = layout(["#######", "#F..a.#", "#######", "#E....#", "#######"], fire=FIRE_STILL)
    for bid in ("fire-aware-shortest-path-v1", "congestion-aware-v1"):
        assert baseline_factory(bid)().act(run_episode_observation(shifted)) == {"Pa": Action.EAST}


# --------------------------------------------------------------------------- performance coverage


@pytest.fixture(scope="module")
def generated_suite(tmp_path_factory: pytest.TempPathFactory) -> Suite:
    generated = generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH)
    return freeze_generated_suite(generated, tmp_path_factory.mktemp("generated") / "frozen").suite


@pytest.mark.parametrize("bid", BASELINE_IDS)
def test_baselines_run_on_generated_scenarios(bid: str, generated_suite: Suite, tmp_path: Path) -> None:
    manifest = build_evaluation_manifest(generated_suite, [5])
    result = run_evaluation(generated_suite, manifest, bid, agent_policy_factory(baseline_factory(bid)), tmp_path / "gen.json")
    assert len(result.episodes) == len(generated_suite.scenarios)
    for episode in result.episodes:
        assert len(episode.record.steps) <= episode.metrics.time.max_steps


def test_dev_manifest_covers_the_required_scenario_kinds(dev_suite: Suite) -> None:
    by_id = {s.scenario_id: s for s in dev_suite.scenarios}
    assert len(by_id["v1-pub-09-crowd-100-split-flow"].people) == 100
    assert by_id[STOCHASTIC_ID].fire.model.value == "stochastic"


# --------------------------------------------------------------------------- privileged escape bound


def test_privileged_bound_is_separate_from_the_fair_interface() -> None:
    assert oracle.PRIVILEGED is True
    bound = oracle.privileged_escape_bound(DETOUR, 0)
    assert bound.privileged is True and bound.to_dict()["privileged"] is True
    assert not isinstance(bound, Agent)
    assert not any(isinstance(v, type) and hasattr(v, "act") for v in vars(oracle).values())
    assert not any("oracle" in bid or "privileged" in bid for bid in BASELINE_IDS)
    assert all(spec.privileged is False for spec in BASELINES.values())
    import emberbench

    assert "oracle" not in emberbench.__all__ and "privileged_escape_bound" not in emberbench.__all__


def test_fire_trajectory_matches_the_simulator(public: Suite) -> None:
    for scenario in public.scenarios:
        for seed in (0, 9):
            trajectory = oracle.fire_trajectory(scenario, seed)
            record = run(scenario, baseline_factory("fire-aware-shortest-path-v1")(), seed=seed)
            assert trajectory[0] == record.initial_state.burning
            for step in record.steps:
                assert trajectory[step.state.step] == step.state.burning


def test_bound_holds_for_every_baseline_and_episode(
    results: dict[str, EvaluationResult], dev_suite: Suite, dev_manifest: EvaluationManifest
) -> None:
    bounds = oracle.privileged_manifest_bounds(dev_suite, dev_manifest)
    for result in results.values():
        for episode, bound in zip(result.episodes, bounds, strict=True):
            assert (bound.scenario_id, bound.seed) == (episode.episode.scenario_id, episode.episode.seed)
            assert episode.metrics.safety.evacuated <= bound.max_evacuated
            assert episode.metrics.time.person_time_to_safety_total >= bound.min_person_time_to_safety_total
            earliest = dict(bound.earliest_evacuation)
            for pid, step in episode.metrics.time.evacuation_steps:
                bound_step = earliest[pid]
                assert bound_step is not None and step >= bound_step


def test_bound_is_tight_for_one_person_and_detects_unescapable_people() -> None:
    bound = oracle.privileged_escape_bound(DETOUR, 0)
    record = run(DETOUR, baseline_factory("fire-aware-shortest-path-v1")())
    assert bound.earliest_evacuation == (("Pa", 8),)
    assert compute_episode_metrics(DETOUR, record).time.evacuation_steps == (("Pa", 8),)
    doomed = layout(["#####", "#.aF#", "#####", "#E..#", "#####"], fire=FireConfig(spread_interval=1))
    assert oracle.privileged_escape_bound(doomed, 0).earliest_evacuation == (("Pa", None),)
    assert oracle.privileged_escape_bound(doomed, 0).max_evacuated == 0


def test_bound_uses_the_evaluator_seed_for_stochastic_fire(public: Suite) -> None:
    scenario = next(s for s in public.scenarios if s.scenario_id == STOCHASTIC_ID)
    assert oracle.fire_trajectory(scenario, 0) != oracle.fire_trajectory(scenario, 1)


def test_public_suite_and_versions_are_unchanged(public: Suite) -> None:
    from emberbench import ACTION_SCHEMA_VERSION, METRICS_VERSION, OBSERVATION_VERSION, SCHEMA_VERSION, SIMULATOR_VERSION, STATISTICS_VERSION
    from test_suite import SUITE_HASH

    assert (SIMULATOR_VERSION, OBSERVATION_VERSION, ACTION_SCHEMA_VERSION, SCHEMA_VERSION, METRICS_VERSION, STATISTICS_VERSION) == (
        "0.2.0", "1.0", "1.0", "1.0", "1.0", "1.0"
    )
    assert public.manifest.suite_hash() == SUITE_HASH

