"""Tests for frozen environment packs (``emberbench.environment_pack``)."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from conftest import greedy_exit_policy
from emberbench import (
    ENVIRONMENT_PACK_VERSION,
    Action,
    EVALUATION_RESULT_VERSION,
    EnvironmentPack,
    EnvironmentPackError,
    EpisodeResult,
    EvaluationManifest,
    EvaluationResult,
    GeneratorConfig,
    Observation,
    SCORING_VERSION,
    ScoringInputError,
    Suite,
    benchmark_versions,
    build_evaluation_manifest,
    compute_episode_metrics,
    fire_world_digest,
    freeze_environment_pack,
    freeze_generated_suite,
    generate_suite,
    load_environment_pack,
    load_generated_suite,
    load_suite,
    run_episode,
    score_result,
    verify_environment_pack,
)
from emberbench.environment_pack import PACK_FILE_NAME, pack_versions
from emberbench.oracle import fire_trajectory
from emberbench.simulator import Policy
from test_generator import BASE_CONFIG, BATCH, SEED

SUITE_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "public" / "v1"
IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation")
SEEDS = (4, 2, 6)
RESAMPLES = 200

# Pinned public hashes. Phase 10 adds scoring and packs; it must not change any of these.
PUBLIC_SUITE_HASH = "5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b"
PUBLIC_DEV_MANIFEST_HASH = "7b9aae45e5e2828f881b2f14d8e5585627e5f8bdbb137a1ed2331bf10ce8c876"


def subset(suite: Suite, ids: tuple[str, ...]) -> Suite:
    entries = tuple(e for e in suite.manifest.entries if e.scenario_id in ids)
    return Suite(suite.root, dataclasses.replace(suite.manifest, entries=entries), tuple(s for s in suite.scenarios if s.scenario_id in ids))


def run_result(pack: EnvironmentPack, model_id: str, policy: Policy) -> EvaluationResult:
    episodes = []
    for spec in pack.manifest.episodes:
        scenario = pack.suite.get(spec.scenario_id)
        record = run_episode(scenario, policy, seed=spec.seed)
        episodes.append(EpisodeResult(spec, record, compute_episode_metrics(scenario, record)))
    return EvaluationResult(EVALUATION_RESULT_VERSION, model_id, pack.manifest, tuple(episodes))


def delayed_policy(obs: Observation) -> dict[str, Action]:
    """A second model: wait four steps, then head for the nearest exit."""
    return {p.id: Action.STAY for p in obs.active} if obs.step < 4 else greedy_exit_policy(obs)


@pytest.fixture(scope="module")
def public() -> Suite:
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory, public: Suite) -> EnvironmentPack:
    """The public suite under a private three-seed schedule."""
    root = tmp_path_factory.mktemp("pack") / "pack"
    return freeze_environment_pack(public, SEEDS, root, pack_id="test-public-v1")


def copy_pack(pack: EnvironmentPack, dest: Path) -> Path:
    for p in sorted(pack.root.rglob("*")):
        target = dest / p.relative_to(pack.root)
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    return dest


# --------------------------------------------------------------------------- public assets unchanged


def test_public_hashes_unchanged(public: Suite) -> None:
    assert public.manifest.suite_hash() == PUBLIC_SUITE_HASH
    assert build_evaluation_manifest(public, (0, 1, 2)).manifest_hash() == PUBLIC_DEV_MANIFEST_HASH
    assert all(s.content_hash() == public.entry(s.scenario_id).content_hash for s in public.scenarios)
    # Benchmark revision 2 pins the scoring and environment-pack contracts in provenance.
    assert benchmark_versions()["scoring"] == SCORING_VERSION
    assert benchmark_versions()["environment_pack"] == ENVIRONMENT_PACK_VERSION


# --------------------------------------------------------------------------- freeze and load


def test_pack_layout_and_record(pack: EnvironmentPack, public: Suite) -> None:
    assert ENVIRONMENT_PACK_VERSION == "1.0"
    files = sorted(p.relative_to(pack.root).as_posix() for p in pack.root.rglob("*") if p.is_file())
    assert PACK_FILE_NAME in files and "evaluation-manifest.json" in files and "suite/manifest.json" in files
    assert len(files) == 3 + len(public.scenarios)
    record = pack.record
    assert record["versions"] == pack_versions()
    assert record["suite"]["suite_hash"] == PUBLIC_SUITE_HASH
    assert [e["seed"] for e in record["episodes"]][:3] == list(SEEDS)
    assert pack.generation is None and pack.suite.manifest == public.manifest
    # Every file is byte-identical to the source.
    for entry in public.manifest.entries:
        assert (pack.root / "suite" / entry.path).read_bytes() == (public.root / entry.path).read_bytes()


def test_pack_is_deterministic(tmp_path: Path, public: Suite, pack: EnvironmentPack) -> None:
    again = freeze_environment_pack(public, SEEDS, tmp_path / "again", pack_id="test-public-v1")
    assert again.digest == pack.digest
    assert (again.root / PACK_FILE_NAME).read_bytes() == (pack.root / PACK_FILE_NAME).read_bytes()
    assert verify_environment_pack(pack.root, expected_digest=pack.digest) == pack.digest
    other = freeze_environment_pack(public, (4, 2, 7), tmp_path / "other", pack_id="test-public-v1")
    assert other.digest != pack.digest  # the seed schedule is part of the environment


def test_fire_worlds_are_model_independent(pack: EnvironmentPack) -> None:
    for episode in pack.episodes[:6]:
        scenario = pack.suite.get(episode.scenario_id)
        assert episode.fire_world_digest == fire_world_digest(scenario, episode.seed)
        trajectory = fire_trajectory(scenario, episode.seed)
        for policy in (greedy_exit_policy, delayed_policy):
            record = run_episode(scenario, policy, seed=episode.seed)
            burning = [record.initial_state.burning, *(s.state.burning for s in record.steps)]
            assert burning == list(trajectory[: len(burning)])


def test_identical_pack_for_two_models(pack: EnvironmentPack) -> None:
    a = score_result(run_result(pack, "model-a", greedy_exit_policy), pack, resamples=RESAMPLES)
    b = score_result(run_result(pack, "model-b", delayed_policy), pack, resamples=RESAMPLES)
    assert a.official and b.official
    assert a.pack_digest == b.pack_digest == pack.digest
    assert a.evaluation_manifest_hash == b.evaluation_manifest_hash == pack.manifest.manifest_hash()
    assert a.layout == b.layout


def test_results_on_another_schedule_are_rejected(pack: EnvironmentPack, public: Suite) -> None:
    other = build_evaluation_manifest(public, (4, 2, 7))
    episodes = []
    for spec in other.episodes[:1]:
        scenario = public.get(spec.scenario_id)
        record = run_episode(scenario, greedy_exit_policy, seed=spec.seed)
        episodes.append(EpisodeResult(spec, record, compute_episode_metrics(scenario, record)))
    result = EvaluationResult(EVALUATION_RESULT_VERSION, "m", other, tuple(episodes))
    with pytest.raises(ScoringInputError):
        score_result(result, pack, resamples=RESAMPLES)


def test_never_overwritten(tmp_path: Path, public: Suite) -> None:
    target = tmp_path / "busy"
    target.mkdir()
    (target / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(EnvironmentPackError):
        freeze_environment_pack(public, SEEDS, target, pack_id="p")


@pytest.mark.parametrize("pack_id", ["", "has space", "x" * 129, "slash/no"])
def test_invalid_pack_id(tmp_path: Path, public: Suite, pack_id: str) -> None:
    with pytest.raises(EnvironmentPackError):
        freeze_environment_pack(public, SEEDS, tmp_path / "p", pack_id=pack_id)


def test_schedule_must_target_the_suite(tmp_path: Path, public: Suite) -> None:
    small = subset(public, IDS)
    foreign: EvaluationManifest = build_evaluation_manifest(small, SEEDS)
    with pytest.raises(EnvironmentPackError):
        freeze_environment_pack(public, foreign, tmp_path / "p", pack_id="p")


def test_generated_suite_pack(tmp_path: Path) -> None:
    generated = generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH)
    freeze_generated_suite(generated, tmp_path / "frozen")
    frozen = load_generated_suite(tmp_path / "frozen")
    pack = freeze_environment_pack(frozen, (1, 2), tmp_path / "pack", pack_id="generated")
    assert pack.generation == frozen.generation
    assert pack.record["source"] == "generated_suite"
    assert load_environment_pack(tmp_path / "pack").digest == pack.digest


# --------------------------------------------------------------------------- tampering fails closed


def _tamper_scenario(root: Path) -> None:
    path = next(p for p in sorted((root / "suite").glob("*.json")) if p.name != "manifest.json")
    path.write_bytes(path.read_bytes().replace(b'"max_steps"', b'"max_steps" ', 1))


def _tamper_manifest(root: Path) -> None:
    path = root / "evaluation-manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["episodes"][0]["seed"] += 1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _tamper_record(key: str, value: object) -> object:
    def apply(root: Path) -> None:
        path = root / PACK_FILE_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[int(p)] if isinstance(node, list) else node[p]
        if isinstance(node, list):
            node[int(leaf)] = value
        else:
            node[leaf] = value
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    return apply


TAMPERING = {
    "scenario bytes": _tamper_scenario,
    "evaluation manifest": _tamper_manifest,
    "extra file": lambda root: (root / "suite" / "extra.json").write_text("{}", encoding="utf-8"),
    "missing file": lambda root: (root / "evaluation-manifest.json").unlink(),
    "pack version": _tamper_record("environment_pack_version", "0.9"),
    "simulator version": _tamper_record("versions.simulator", "9.9.9"),
    "fire world": _tamper_record("episodes.0.fire_world_digest", "0" * 64),
    "seed in record": _tamper_record("episodes.0.seed", 999),
    "file hash": _tamper_record("files.0.1", "f" * 64),
    "unknown key": _tamper_record("extra", 1),
    "pack id": _tamper_record("pack_id", "bad id"),
}


@pytest.mark.parametrize("name", sorted(TAMPERING))
def test_tampering_fails_closed(tmp_path: Path, pack: EnvironmentPack, name: str) -> None:
    root = copy_pack(pack, tmp_path / "copy")
    assert load_environment_pack(root).digest == pack.digest
    TAMPERING[name](root)  # type: ignore[operator]
    with pytest.raises(EnvironmentPackError):
        load_environment_pack(root)


def test_expected_digest_mismatch_fails(pack: EnvironmentPack) -> None:
    with pytest.raises(EnvironmentPackError):
        verify_environment_pack(pack.root, expected_digest="0" * 64)


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need extra privileges on Windows")
def test_symlink_rejected(tmp_path: Path, pack: EnvironmentPack) -> None:
    root = copy_pack(pack, tmp_path / "copy")
    (root / "suite" / "link.json").symlink_to(root / "suite" / "manifest.json")
    with pytest.raises(EnvironmentPackError):
        load_environment_pack(root)
