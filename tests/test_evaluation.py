"""Paired identical evaluation: manifests, seed privacy, episode isolation, resume and pairing."""

import dataclasses
import enum
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from conftest import greedy_exit_policy
from emberbench import (
    METRICS_VERSION,
    OBSERVATION_VERSION,
    SIMULATOR_VERSION,
    Action,
    EpisodeSpec,
    EvaluationError,
    EvaluationManifest,
    EvaluationResult,
    Observation,
    Pos,
    Simulator,
    Suite,
    build_evaluation_manifest,
    load_evaluation_manifest,
    load_evaluation_result,
    load_suite,
    pair_results,
    run_evaluation,
    save_evaluation_manifest,
    save_evaluation_result,
    verify_manifest,
)
from emberbench.canonical import sha256_hex
from emberbench.observation import MapView, RulesView
from emberbench.scenario import ExitSpec
from emberbench.simulator import Policy
from emberbench.state import PersonState
from test_suite import PINNED_HASHES, SUITE_DIR, SUITE_HASH

# A deterministic and a stochastic-fire scenario keep these tests fast.
SMALL_IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation")
STOCHASTIC_ID = "v1-pub-10-route-invalidation"
# Development seeds, deliberately not sorted, to check that their order is kept.
SEEDS = (11, 7)
# Golden hash of the public suite under development seeds (0, 1, 2).
PUBLIC_DEV_MANIFEST_HASH = "7b9aae45e5e2828f881b2f14d8e5585627e5f8bdbb137a1ed2331bf10ce8c876"


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def factory_of(policy: Policy) -> Callable[[], Policy]:
    return lambda: policy


def subset(suite: Suite, ids: tuple[str, ...]) -> Suite:
    entries = tuple(e for e in suite.manifest.entries if e.scenario_id in ids)
    return Suite(
        root=suite.root,
        manifest=dataclasses.replace(suite.manifest, entries=entries),
        scenarios=tuple(s for s in suite.scenarios if s.scenario_id in ids),
    )


@pytest.fixture(scope="module")
def public() -> Suite:
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def small(public: Suite) -> Suite:
    return subset(public, SMALL_IDS)


@pytest.fixture(scope="module")
def manifest(small: Suite) -> EvaluationManifest:
    return build_evaluation_manifest(small, SEEDS)


@pytest.fixture(scope="module")
def greedy_artifact(small: Suite, manifest: EvaluationManifest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("greedy") / "greedy.json"
    run_evaluation(small, manifest, "greedy", factory_of(greedy_exit_policy), path)
    return path


@pytest.fixture(scope="module")
def greedy(greedy_artifact: Path, small: Suite) -> EvaluationResult:
    return load_evaluation_result(greedy_artifact, small)


# --------------------------------------------------------------------------- manifest


def test_manifest_is_deterministic_and_round_trips(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    again = build_evaluation_manifest(small, list(SEEDS))
    assert again == manifest
    assert again.manifest_hash() == manifest.manifest_hash()
    assert EvaluationManifest.from_dict(json.loads(manifest.to_file_text())) == manifest
    save_evaluation_manifest(manifest, tmp_path / "m.json")
    assert load_evaluation_manifest(tmp_path / "m.json") == manifest


def test_manifest_pins_suite_versions_and_episodes(small: Suite, manifest: EvaluationManifest) -> None:
    assert manifest.manifest_version == "1.0"
    assert (manifest.suite_id, manifest.suite_version) == ("emberbench-v1-public", "1.0.0")
    assert manifest.suite_hash == small.manifest.suite_hash()
    assert (manifest.simulator_version, manifest.observation_version, manifest.metrics_version) == (
        SIMULATOR_VERSION,
        OBSERVATION_VERSION,
        METRICS_VERSION,
    )
    # Suite manifest order, then the caller's seed order.
    assert [(e.index, e.scenario_id, e.seed) for e in manifest.episodes] == [
        (0, SMALL_IDS[0], 11),
        (1, SMALL_IDS[0], 7),
        (2, SMALL_IDS[1], 11),
        (3, SMALL_IDS[1], 7),
    ]
    for e in manifest.episodes:
        assert e.scenario_hash == PINNED_HASHES[e.scenario_id]


def test_public_suite_hashes_are_unchanged(public: Suite) -> None:
    m = build_evaluation_manifest(public, [0, 1, 2])
    assert public.manifest.suite_hash() == SUITE_HASH
    assert m.suite_hash == SUITE_HASH
    assert len(m.episodes) == 3 * len(PINNED_HASHES)
    assert {e.scenario_id: e.scenario_hash for e in m.episodes} == PINNED_HASHES
    assert m.manifest_hash() == PUBLIC_DEV_MANIFEST_HASH


def test_seed_order_changes_the_hash(small: Suite, manifest: EvaluationManifest) -> None:
    reordered = build_evaluation_manifest(small, list(reversed(SEEDS)))
    assert [e.seed for e in reordered.episodes] == [7, 11, 7, 11]
    assert reordered.manifest_hash() != manifest.manifest_hash()


def test_episode_order_changes_the_hash(manifest: EvaluationManifest) -> None:
    eps = list(manifest.episodes)
    eps[0], eps[1] = eps[1], eps[0]
    swapped = tuple(dataclasses.replace(e, index=i) for i, e in enumerate(eps))
    assert dataclasses.replace(manifest, episodes=swapped).manifest_hash() != manifest.manifest_hash()


def test_seed_change_changes_the_hash(small: Suite, manifest: EvaluationManifest) -> None:
    assert build_evaluation_manifest(small, [11, 8]).manifest_hash() != manifest.manifest_hash()


@pytest.mark.parametrize(
    "change",
    [
        {"suite_version": "1.0.1"},
        {"suite_hash": "0" * 64},
        {"simulator_version": "9.9.9"},
        {"observation_version": "9.9"},
        {"metrics_version": "9.9"},
    ],
)
def test_every_header_field_changes_the_hash(manifest: EvaluationManifest, change: dict[str, str]) -> None:
    assert dataclasses.replace(manifest, **change).manifest_hash() != manifest.manifest_hash()  # type: ignore[arg-type]


def test_scenario_hash_change_changes_the_hash(manifest: EvaluationManifest) -> None:
    eps = tuple(dataclasses.replace(e, scenario_hash="f" * 64) if e.scenario_id == SMALL_IDS[0] else e for e in manifest.episodes)
    changed = dataclasses.replace(manifest, episodes=eps)
    assert changed.manifest_hash() != manifest.manifest_hash()


def test_seeds_up_to_uint64_max_are_accepted(small: Suite) -> None:
    m = build_evaluation_manifest(small, [0, 2**64 - 1])
    assert [e.seed for e in m.episodes[:2]] == [0, 2**64 - 1]


@pytest.mark.parametrize(
    "seeds",
    [[], [1, 1], [3, 1, 3], [-1], [2**64], [True], [1.0], ["1"], [None]],
    ids=["empty", "duplicate", "duplicate-later", "negative", "too-big", "bool", "float", "str", "none"],
)
def test_invalid_seeds_are_rejected(small: Suite, seeds: list[Any]) -> None:
    with pytest.raises(EvaluationError):
        build_evaluation_manifest(small, seeds)


@pytest.mark.parametrize("seeds", [{1, 2}, "12", frozenset({1}), (s for s in [1, 2])], ids=["set", "str", "frozenset", "generator"])
def test_unordered_seed_collections_are_rejected(small: Suite, seeds: Any) -> None:
    with pytest.raises(TypeError):
        build_evaluation_manifest(small, seeds)


def _manifest_doc(manifest: EvaluationManifest) -> dict[str, Any]:
    return json.loads(manifest.to_file_text())  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["episodes"][1].update(index=5),
        lambda d: d["episodes"][1].update(seed=11),
        lambda d: d["episodes"][1].update(seed=-3),
        lambda d: d["episodes"][1].update(seed=True),
        lambda d: d["episodes"][1].update(scenario_hash="a" * 64),
        lambda d: d["episodes"][1].update(scenario_hash="xyz"),
        lambda d: d["episodes"][1].update(extra=1),
        lambda d: d["episodes"].clear(),
        lambda d: d.update(manifest_version="2.0"),
        lambda d: d.update(suite_version="1.0"),
        lambda d: d.pop("metrics_version"),
        lambda d: d.update(note="x"),
    ],
    ids=[
        "index-gap",
        "duplicate-seed-per-scenario",
        "negative-seed",
        "bool-seed",
        "two-hashes-for-one-scenario",
        "bad-hash",
        "unknown-episode-key",
        "no-episodes",
        "unsupported-version",
        "bad-suite-version",
        "missing-key",
        "unknown-key",
    ],
)
def test_malformed_manifests_are_rejected(manifest: EvaluationManifest, mutate: Callable[[dict[str, Any]], Any]) -> None:
    doc = _manifest_doc(manifest)
    mutate(doc)
    with pytest.raises(EvaluationError):
        EvaluationManifest.from_dict(doc)


def test_the_same_seed_may_be_used_for_different_scenarios(manifest: EvaluationManifest) -> None:
    assert {e.seed for e in manifest.episodes if e.scenario_id == SMALL_IDS[0]} == {
        e.seed for e in manifest.episodes if e.scenario_id == SMALL_IDS[1]
    }


def test_manifest_must_match_suite_and_versions(small: Suite, public: Suite, manifest: EvaluationManifest) -> None:
    verify_manifest(manifest, small)
    with pytest.raises(EvaluationError, match="suite_hash"):
        verify_manifest(manifest, public)
    with pytest.raises(EvaluationError, match="simulator_version"):
        verify_manifest(dataclasses.replace(manifest, simulator_version="0.1.0"), small)
    with pytest.raises(EvaluationError, match="metrics_version"):
        verify_manifest(dataclasses.replace(manifest, metrics_version="0.9"), small)
    eps = tuple(dataclasses.replace(e, scenario_hash="f" * 64) if e.scenario_id == SMALL_IDS[0] else e for e in manifest.episodes)
    with pytest.raises(EvaluationError, match="pinned content hash"):
        verify_manifest(dataclasses.replace(manifest, episodes=eps), small)


# --------------------------------------------------------------------------- seed privacy and isolation

# Every type that may appear anywhere inside an Observation. Containers are walked into.
_ALLOWED_TYPES = (Observation, MapView, RulesView, ExitSpec, PersonState, Pos, enum.Enum, str, int, float, bool, type(None))
_CONTAINERS = (tuple, frozenset)


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _walk(getattr(value, f.name))
    elif isinstance(value, (tuple, list, frozenset, set)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(k)
            yield from _walk(v)


class RecordingPolicy:
    """Records everything it is called with, and acts greedily."""

    def __init__(self, log: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        self.log = log

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Action]:
        self.log.append((args, kwargs))
        return greedy_exit_policy(args[0])


def test_policy_receives_only_observations(small: Suite, tmp_path: Path) -> None:
    secret = 0xC0FFEE1234567891  # distinctive, so a leak is easy to spot
    m = build_evaluation_manifest(small, [secret])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def factory(*args: Any, **kwargs: Any) -> Policy:
        factory_calls.append((args, kwargs))
        return RecordingPolicy(calls)

    run_evaluation(small, m, "recorder", factory, tmp_path / "r.json")

    assert factory_calls == [((), {})] * len(m.episodes)
    assert calls
    for args, kwargs in calls:
        assert kwargs == {}
        assert len(args) == 1 and type(args[0]) is Observation
        for node in _walk(args[0]):
            assert isinstance(node, _ALLOWED_TYPES + _CONTAINERS), type(node)
        text = json.dumps(args[0].to_dict())
        assert str(secret) not in text and hex(secret) not in text
        for forbidden in (m.manifest_hash(), small.manifest.suite_hash(), *PINNED_HASHES.values()):
            assert forbidden not in text
    # The seed did reach the simulator.
    stored = load_evaluation_result(tmp_path / "r.json", small)
    assert {e.record.seed for e in stored.episodes} == {secret}


def test_fresh_policy_per_episode(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    instances: list["CountingPolicy"] = []

    class CountingPolicy:
        def __init__(self) -> None:
            self.steps: list[int] = []
            self.maps: set[int] = set()

        def __call__(self, obs: Observation) -> dict[str, Action]:
            self.steps.append(obs.step)
            self.maps.add(id(obs.map))
            return greedy_exit_policy(obs)

    def factory() -> Policy:
        instances.append(CountingPolicy())
        return instances[-1]

    run_evaluation(small, manifest, "counting", factory, tmp_path / "c.json")
    assert len(instances) == len(manifest.episodes)
    assert len({id(p) for p in instances}) == len(instances)
    for p in instances:
        # Each instance sees exactly one episode from its start.
        assert p.steps == list(range(len(p.steps)))
        assert len(p.maps) == 1


def test_manifest_seed_overrides_scenario_seed(small: Suite, greedy: EvaluationResult) -> None:
    for e in greedy.episodes:
        assert e.record.seed == e.episode.seed
        assert e.record.seed != small.get(e.episode.scenario_id).seed


# --------------------------------------------------------------------------- common random numbers


def _burning_by_step(result: EvaluationResult, index: int) -> list[frozenset[Pos]]:
    r = next(e.record for e in result.episodes if e.episode.index == index)
    return [r.initial_state.burning] + [s.state.burning for s in r.steps]


def test_two_policies_see_identical_stochastic_fire(
    small: Suite, manifest: EvaluationManifest, greedy: EvaluationResult, tmp_path: Path
) -> None:
    stay = run_evaluation(small, manifest, "stay", factory_of(stay_policy), tmp_path / "stay.json")
    assert stay.manifest_hash == greedy.manifest_hash
    stochastic = [e.index for e in manifest.episodes if e.scenario_id == STOCHASTIC_ID]
    assert len(stochastic) == 2
    for index in stochastic:
        a, b = _burning_by_step(greedy, index), _burning_by_step(stay, index)
        n = min(len(a), len(b))
        assert n > 1 and a[:n] == b[:n]
        # The same fire comes from a bare simulator with that seed: it depends on the seed only.
        sim = Simulator(small.get(STOCHASTIC_ID), seed=manifest.episodes[index].seed)
        bare = [sim.state.burning]
        while not sim.done and len(bare) < n:
            sim.step(stay_policy(sim.observe()))
            bare.append(sim.state.burning)
        assert bare == b[: len(bare)]
    # Different seeds really give different fire, so the pairing is not trivial.
    first, second = (_burning_by_step(stay, i) for i in stochastic)
    assert first != second


# --------------------------------------------------------------------------- resume


class Interrupted(Exception):
    pass


def test_interrupted_run_resumes_without_rerunning(
    small: Suite, manifest: EvaluationManifest, greedy_artifact: Path, tmp_path: Path
) -> None:
    path = tmp_path / "run.json"
    built = 0

    def crashing_factory() -> Policy:
        nonlocal built
        built += 1
        if built == 3:
            raise Interrupted
        return greedy_exit_policy

    with pytest.raises(Interrupted):
        run_evaluation(small, manifest, "greedy", crashing_factory, path)
    partial = load_evaluation_result(path, small)
    assert sorted(partial.completed_indices) == [0, 1]
    assert not partial.is_complete

    resumed_for: list[int] = []

    def counting_factory() -> Policy:
        def policy(obs: Observation) -> Mapping[str, Any]:
            if obs.step == 0:
                resumed_for.append(obs.map.width)
            return greedy_exit_policy(obs)

        return policy

    final = run_evaluation(small, manifest, "greedy", counting_factory, path)
    assert len(resumed_for) == 2  # only episodes 2 and 3 ran
    assert final.is_complete
    # Resuming gives exactly the same bytes as an uninterrupted run.
    assert path.read_bytes() == greedy_artifact.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]

    # A complete artifact resumes as a no-op.
    def must_not_run() -> Policy:
        raise AssertionError("no episode should run")

    assert run_evaluation(small, manifest, "greedy", must_not_run, path) == final


def test_crash_inside_an_episode_keeps_only_completed_episodes(small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    built = 0

    def factory() -> Policy:
        nonlocal built
        built += 1
        episode = built

        def policy(obs: Observation) -> Mapping[str, Any]:
            if episode == 2 and obs.step == 3:
                raise Interrupted
            return greedy_exit_policy(obs)

        return policy

    with pytest.raises(Interrupted):
        run_evaluation(small, manifest, "greedy", factory, path)
    assert sorted(load_evaluation_result(path, small).completed_indices) == [0]


def test_resume_rejects_a_changed_schedule(small: Suite, manifest: EvaluationManifest, greedy_artifact: Path) -> None:
    for seeds in ([11, 8], [7, 11], [11]):
        other = build_evaluation_manifest(small, seeds)
        with pytest.raises(EvaluationError, match="evaluation manifest"):
            run_evaluation(small, other, "greedy", factory_of(greedy_exit_policy), greedy_artifact)


def test_resume_rejects_a_changed_model(small: Suite, manifest: EvaluationManifest, greedy_artifact: Path) -> None:
    with pytest.raises(EvaluationError, match="model"):
        run_evaluation(small, manifest, "greedy-v2", factory_of(greedy_exit_policy), greedy_artifact)


def test_existing_artifact_is_never_overwritten(small: Suite, manifest: EvaluationManifest, greedy_artifact: Path) -> None:
    before = greedy_artifact.read_bytes()
    with pytest.raises(EvaluationError, match="already exists"):
        run_evaluation(small, manifest, "greedy", factory_of(greedy_exit_policy), greedy_artifact, resume=False)
    assert greedy_artifact.read_bytes() == before


def test_resume_rejects_changed_versions(small: Suite, manifest: EvaluationManifest, greedy_artifact: Path, tmp_path: Path) -> None:
    for field, value in (("simulator_version", "0.1.0"), ("observation_version", "0.9"), ("metrics_version", "0.9")):
        doc = json.loads(greedy_artifact.read_text())
        doc["manifest"][field] = value
        doc[field] = value
        doc["evaluation_manifest_hash"] = sha256_hex(doc["manifest"])
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(EvaluationError, match=field):
            run_evaluation(small, manifest, "greedy", factory_of(greedy_exit_policy), path)
        # The runner also refuses a manifest made for other versions.
        with pytest.raises(EvaluationError, match=field):
            run_evaluation(small, dataclasses.replace(manifest, **{field: value}), "greedy", factory_of(greedy_exit_policy), tmp_path / "new.json")  # type: ignore[arg-type]


def _first_step_action_changed(doc: dict[str, Any]) -> None:
    actions = doc["episodes"][0]["record"]["steps"][0]["actions"]
    pid = sorted(actions)[0]
    actions[pid] = "stay" if actions[pid] != "stay" else "north"


def _fire_changed_digest_fixed(doc: dict[str, Any]) -> None:
    ep = next(e for e in doc["episodes"] if e["scenario_id"] == STOCHASTIC_ID)
    ep["record"]["steps"][-1]["state"]["burning"] = ep["record"]["steps"][-1]["state"]["burning"][:-1]
    ep["record_digest"] = sha256_hex(ep["record"])


def _action_changed_digest_fixed(doc: dict[str, Any]) -> None:
    _first_step_action_changed(doc)
    doc["episodes"][0]["record_digest"] = sha256_hex(doc["episodes"][0]["record"])


def _seed_changed_everywhere(doc: dict[str, Any]) -> None:
    ep = doc["episodes"][0]
    ep["seed"] = ep["record"]["seed"] = 12
    ep["record_digest"] = sha256_hex(ep["record"])


TAMPERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "duplicate-episode": lambda d: d["episodes"].append(d["episodes"][0]),
    "duplicate-index": lambda d: d["episodes"][1].update(index=0),
    "extra-episode": lambda d: d["episodes"].append({**d["episodes"][0], "index": 4}),
    "out-of-order": lambda d: d["episodes"].reverse(),
    "seed": lambda d: d["episodes"][0].update(seed=12),
    "seed-everywhere": _seed_changed_everywhere,
    "scenario-id": lambda d: d["episodes"][0].update(scenario_id=SMALL_IDS[1]),
    "scenario-hash": lambda d: d["episodes"][0].update(scenario_hash="a" * 64),
    "record-action": _first_step_action_changed,
    "record-action-digest-fixed": _action_changed_digest_fixed,
    "record-fire-digest-fixed": _fire_changed_digest_fixed,
    "record-extra-key": lambda d: d["episodes"][0]["record"].update(note="x"),
    "record-malformed": lambda d: d["episodes"][0].update(record={"steps": 3}),
    "digest": lambda d: d["episodes"][0].update(record_digest="0" * 64),
    "metrics": lambda d: d["episodes"][0]["metrics"]["safety"].update(evacuated=0),
    "episode-extra-key": lambda d: d["episodes"][0].update(note="x"),
    "episode-missing-key": lambda d: d["episodes"][0].pop("metrics"),
    "manifest-hash": lambda d: d.update(evaluation_manifest_hash="0" * 64),
    "manifest-seed": lambda d: d["manifest"]["episodes"][0].update(seed=99),
    "header-suite-hash": lambda d: d.update(suite_hash="0" * 64),
    "header-unknown-key": lambda d: d.update(note="x"),
    "result-version": lambda d: d.update(result_version="2.0"),
    "model-id-blank": lambda d: d.update(model_id=" "),
    "not-an-object": lambda d: d.clear(),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_tampered_artifacts_are_rejected(
    name: str, small: Suite, manifest: EvaluationManifest, greedy_artifact: Path, tmp_path: Path
) -> None:
    doc = json.loads(greedy_artifact.read_text())
    TAMPERS[name](doc)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(EvaluationError):
        load_evaluation_result(path, small)
    before = path.read_bytes()
    with pytest.raises(EvaluationError):
        run_evaluation(small, manifest, "greedy", factory_of(greedy_exit_policy), path)
    assert path.read_bytes() == before  # never repaired


def test_tampered_record_fails_replay_authentication(small: Suite, greedy_artifact: Path, tmp_path: Path) -> None:
    doc = json.loads(greedy_artifact.read_text())
    _fire_changed_digest_fixed(doc)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(EvaluationError, match="authentication"):
        load_evaluation_result(path, small)


def test_record_relabelled_to_another_seed_fails_replay(small: Suite, greedy_artifact: Path, tmp_path: Path) -> None:
    # A consistent forgery: a new manifest, and seed-7 records relabelled as seed 8.
    # Every header and digest agrees, so only the replay can catch it.
    other = build_evaluation_manifest(small, [11, 8])
    doc = json.loads(greedy_artifact.read_text())
    doc["manifest"] = other.to_dict()
    doc["evaluation_manifest_hash"] = other.manifest_hash()
    for ep in doc["episodes"]:
        if ep["seed"] == 7:
            ep["seed"] = ep["record"]["seed"] = 8
            ep["record_digest"] = sha256_hex(ep["record"])
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(EvaluationError, match=r"episodes\[1\]: run record failed authentication"):
        load_evaluation_result(path, small)


def test_invalid_json_is_rejected(small: Suite, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(EvaluationError, match="not valid JSON"):
        load_evaluation_result(path, small)


def test_result_construction_rejects_mismatched_episodes(greedy: EvaluationResult) -> None:
    e0, e1 = greedy.episodes[0], greedy.episodes[1]
    m = greedy.manifest
    with pytest.raises(EvaluationError, match="duplicate"):
        EvaluationResult("1.0", "x", m, (e0, e0))
    with pytest.raises(EvaluationError, match="manifest order"):
        EvaluationResult("1.0", "x", m, (e1, e0))
    with pytest.raises(EvaluationError, match="seed"):
        EvaluationResult("1.0", "x", m, (dataclasses.replace(e1, record=e0.record),))
    with pytest.raises(EvaluationError, match="not in the manifest"):
        EvaluationResult("1.0", "x", m, (dataclasses.replace(e0, episode=EpisodeSpec(9, e0.episode.scenario_id, e0.episode.scenario_hash, 5)),))
    with pytest.raises(EvaluationError, match="model_id"):
        EvaluationResult("1.0", "", m, ())


# --------------------------------------------------------------------------- pairing


def test_identical_schedules_pair_completely(greedy: EvaluationResult, small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    stay = run_evaluation(small, manifest, "stay", factory_of(stay_policy), tmp_path / "stay.json")
    report = pair_results(greedy, stay)
    assert report.total_scheduled == report.paired_count == 4
    assert report.missing_from_a == report.missing_from_b == ()
    assert [k.index for k in report.paired] == [0, 1, 2, 3]
    for k, spec in zip(report.paired, manifest.episodes):
        assert (k.manifest_hash, k.scenario_id, k.scenario_hash, k.seed) == (
            manifest.manifest_hash(),
            spec.scenario_id,
            spec.scenario_hash,
            spec.seed,
        )
    assert report.to_dict()["paired_count"] == 4


def test_partial_pairing_counts_are_exact(greedy: EvaluationResult, small: Suite, manifest: EvaluationManifest, tmp_path: Path) -> None:
    with_gaps = EvaluationResult("1.0", "gappy", manifest, (greedy.episodes[0], greedy.episodes[2]))
    prefix = EvaluationResult("1.0", "prefix", manifest, greedy.episodes[:2])

    report = pair_results(greedy, with_gaps)
    assert (report.total_scheduled, report.paired_count) == (4, 2)
    assert [k.index for k in report.paired] == [0, 2]
    assert report.missing_from_a == ()
    assert [k.index for k in report.missing_from_b] == [1, 3]

    report = pair_results(with_gaps, prefix)
    assert [k.index for k in report.paired] == [0]
    assert [k.index for k in report.missing_from_a] == [1, 3]
    assert [k.index for k in report.missing_from_b] == [2, 3]

    empty = EvaluationResult("1.0", "empty", manifest, ())
    assert pair_results(empty, greedy).paired_count == 0
    assert len(pair_results(empty, greedy).missing_from_a) == 4

    # A saved artifact with gaps loads, and resuming runs only the missing episodes.
    path = tmp_path / "gappy.json"
    save_evaluation_result(with_gaps, path)
    assert load_evaluation_result(path, small) == with_gaps
    ran: list[int] = []

    def factory() -> Policy:
        ran.append(1)
        return greedy_exit_policy

    assert run_evaluation(small, manifest, "gappy", factory, path).is_complete
    assert len(ran) == 2


def test_pairing_rejects_incompatible_manifests(greedy: EvaluationResult, small: Suite, tmp_path: Path) -> None:
    other = build_evaluation_manifest(small, [11, 8])
    result = run_evaluation(small, other, "greedy", factory_of(greedy_exit_policy), tmp_path / "other.json")
    with pytest.raises(EvaluationError, match="incompatible"):
        pair_results(greedy, result)
    with pytest.raises(EvaluationError, match="incompatible"):
        pair_results(result, greedy)


def test_model_id_is_only_a_label(greedy: EvaluationResult) -> None:
    # Two artifacts may carry the same label; pairing still works on episode keys only.
    report = pair_results(greedy, dataclasses.replace(greedy))
    assert report.model_a == report.model_b == "greedy"
    assert report.paired_count == 4
