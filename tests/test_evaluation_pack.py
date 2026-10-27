"""The V1 evaluation pack (tools/experiments/v1): unique worlds, official scoring, reproducibility and fairness.

CI runs the tiny ``smoke`` protocol end to end and compares its canonical
index with the committed golden index byte for byte. The full public
development run is opt-in (``EMBERBENCH_FULL_PACK=1``) because it takes
minutes.
"""

import ast
import dataclasses
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from emberbench import (
    BASELINE_IDS,
    BENCHMARK_REVISION,
    Agent,
    EmberBenchError,
    FireModelName,
    Observation,
    agent_policy_factory,
    benchmark_versions,
    build_evaluation_manifest,
    compare_scores,
    environment_world_digest,
    load_environment_pack,
    load_evaluation_result,
    load_suite,
    run_evaluation,
    score_result,
    serialize_observation,
)
from emberbench.leakguard import check_repository_paths, forbidden_reason

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "tools" / "experiments" / "v1"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "experiments" / "v1"
LABEL = "PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD"

# The canonical public suite and the pack's unique-world manifest. A change to
# either is a benchmark change, not a pack change.
PUBLIC_SUITE_HASH = "5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b"
PUBLIC_MANIFEST_HASH = "3290930556bee1cf0a8232fd9cf2a118b40d4560d79ef18fc23d79f76139b82c"


def _pack_module(name: str) -> ModuleType:
    if str(PACK_DIR) not in sys.path:
        sys.path.insert(0, str(PACK_DIR))
    return importlib.import_module(name)


runner = _pack_module("run_public_benchmark")
protocol_mod = _pack_module("pack_protocol")
pack_agents = _pack_module("pack_agents")
pack_index = _pack_module("pack_index")


def _quiet(message: str) -> None:
    pass


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pack") / "smoke"
    runner.run_pack(protocol_mod.SMOKE, out, log=_quiet)
    return out


def _index(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "index.json").read_text(encoding="ascii"))
    return data


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- pins and unique worlds


def test_public_suite_hash_and_unique_world_manifest_are_unchanged() -> None:
    suite = load_suite(REPO_ROOT / "scenarios" / "public" / "v1")
    assert suite.manifest.suite_hash() == PUBLIC_SUITE_HASH == protocol_mod.PUBLIC_SUITE_HASH == protocol_mod.PUBLIC.suite_hash
    manifest = runner.protocol_manifest(protocol_mod.PUBLIC, runner.protocol_suite(protocol_mod.PUBLIC))
    assert manifest.manifest_hash() == PUBLIC_MANIFEST_HASH == protocol_mod.PUBLIC.manifest_hash
    golden = json.loads((GOLDEN_DIR / "public-development-index.json").read_text(encoding="ascii"))
    assert golden["suite"]["suite_hash"] == PUBLIC_SUITE_HASH
    assert golden["schedule"]["evaluation_manifest"]["manifest_hash"] == PUBLIC_MANIFEST_HASH
    assert golden["protocol"]["candidate_seeds"] == list(protocol_mod.CANDIDATE_SEEDS)


def test_public_pack_schedules_only_unique_worlds() -> None:
    suite = runner.protocol_suite(protocol_mod.PUBLIC)
    manifest = runner.protocol_manifest(protocol_mod.PUBLIC, suite)
    worlds = runner.unique_worlds(suite, manifest)
    digests = [w["environment_world_digest"] for w in worlds]
    assert len(digests) == len(set(digests)) == 9 + protocol_mod.PUBLIC.stochastic_worlds == 19
    for entry in suite.manifest.entries:
        scheduled = [s for s in manifest.episodes if s.scenario_id == entry.scenario_id]
        if suite.get(entry.scenario_id).fire.model is FireModelName.DETERMINISTIC:
            assert len(scheduled) == 1  # never one deterministic world under several seeds
        else:
            assert len(scheduled) == protocol_mod.PUBLIC.stochastic_worlds
    # The nominal candidate product holds many duplicates; none of them is scheduled.
    nominal = build_evaluation_manifest(suite, protocol_mod.CANDIDATE_SEEDS)
    nominal_worlds = [environment_world_digest(suite.get(s.scenario_id), s.seed) for s in nominal.episodes]
    assert len(nominal.episodes) == 160 and len(set(nominal_worlds)) == 25
    assert runner.nominal_schedule(protocol_mod.PUBLIC, suite) == {"nominal_candidate_episodes": 160, "distinct_worlds_among_candidates": 25}


def test_deterministic_seed_duplicates_are_absent(smoke_out: Path) -> None:
    index = _index(smoke_out)
    worlds = index["schedule"]["worlds"]
    assert len({w["environment_world_digest"] for w in worlds}) == len(worlds) == index["schedule"]["unique_worlds"] == 5
    deterministic = [w for w in worlds if w["fire_model"] == "deterministic"]
    assert len(deterministic) == index["schedule"]["deterministic_worlds"] == 1
    assert len({w["scenario_id"] for w in deterministic}) == len(deterministic)
    assert index["schedule"]["nominal_candidate_episodes"] == 12


def test_runner_refuses_a_changed_candidate_list() -> None:
    changed = dataclasses.replace(protocol_mod.SMOKE, candidate_seeds=(1, 0, 2, 3, 4, 5))
    suite = runner.protocol_suite(changed)
    with pytest.raises(runner.PackError, match="manifest hash"):
        runner.protocol_manifest(changed, suite)


# --------------------------------------------------------------------------- reproducibility


def test_smoke_run_reproduces_the_golden_index(smoke_out: Path) -> None:
    text = (smoke_out / "index.json").read_text(encoding="ascii")
    assert text == (GOLDEN_DIR / "smoke-index.json").read_text(encoding="ascii")


def test_rerun_resumes_and_gives_identical_bytes(smoke_out: Path) -> None:
    before = _files(smoke_out)
    index = runner.run_pack(protocol_mod.SMOKE, smoke_out, log=_quiet)
    assert pack_index.canonical(index) + "\n" == (smoke_out / "index.json").read_text(encoding="ascii")
    assert _files(smoke_out) == before


def test_fresh_rerun_is_byte_identical(smoke_out: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    runner.run_pack(protocol_mod.SMOKE, again, log=_quiet)
    assert _files(again) == _files(smoke_out)


def test_existing_output_with_other_bytes_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    assert runner.write_new_or_same(path, "{}\n") == "written"
    assert runner.write_new_or_same(path, "{}\n") == "unchanged"
    with pytest.raises(runner.PackError, match="refusing to overwrite"):
        runner.write_new_or_same(path, "[]\n")
    assert path.read_bytes() == b"{}\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]  # no temporary file left


def test_output_directory_inside_the_repository_is_refused() -> None:
    with pytest.raises(runner.PackError, match="outside the repository"):
        runner.run_pack(protocol_mod.SMOKE, REPO_ROOT / "pack-out", log=_quiet)
    assert not (REPO_ROOT / "pack-out").exists()


def test_runner_denies_network_events() -> None:
    with pytest.raises(runner.PackError, match="network"):
        runner._deny_network("socket.connect", ())
    runner._deny_network("open", ())


def test_index_has_no_absolute_paths_hosts_or_timestamps(smoke_out: Path) -> None:
    text = (smoke_out / "index.json").read_text(encoding="ascii")
    assert str(smoke_out) not in text and str(REPO_ROOT) not in text
    assert not re.search(r"[A-Za-z]:\\\\|\"/", text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)
    import platform
    import tempfile

    assert platform.node() not in text and tempfile.gettempdir() not in text


def test_index_pins_every_version_and_digest(smoke_out: Path) -> None:
    index = _index(smoke_out)
    bench = index["benchmark"]
    for key in ("versions", "scoring_version", "scoring_constants_digest", "environment_world_version", "environment_pack_version"):
        assert bench[key]
    assert "benchmark_revision" in bench["versions"]
    assert index["environment_pack"]["digest"] and len(index["environment_pack"]["fire_world_digests"]) == 5
    for model in index["models"]:
        assert model["config_digest"] and model["result"]["digest"] and model["score_report"]["digest"]
    assert all(p["digest"] for p in index["paired_comparisons"])


def test_provenance_pins_benchmark_revision_3_and_every_current_version(smoke_out: Path) -> None:
    pinned = protocol_mod.BENCHMARK_VERSIONS
    assert protocol_mod.BENCHMARK_REVISION == BENCHMARK_REVISION == "3"
    assert pinned == benchmark_versions()  # the pins match the benchmark on main, key for key
    assert len(pinned) == 28 and pinned["benchmark_revision"] == "3"
    for golden in ("smoke-index.json", "public-development-index.json"):
        bench = json.loads((GOLDEN_DIR / golden).read_text(encoding="ascii"))["benchmark"]
        assert bench["versions"] == pinned, golden
    assert _index(smoke_out)["benchmark"]["versions"] == pinned


def test_runner_fails_closed_on_a_benchmark_version_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    changed = dict(protocol_mod.BENCHMARK_VERSIONS, scoring="9.9")
    monkeypatch.setattr(protocol_mod, "BENCHMARK_VERSIONS", changed)
    with pytest.raises(runner.PackError, match="scoring"):
        runner.run_pack(protocol_mod.SMOKE, tmp_path / "out", log=_quiet)
    assert not (tmp_path / "out").exists()


def test_verify_authenticates_a_finished_run_without_writing(smoke_out: Path) -> None:
    before = _files(smoke_out)
    index = runner.verify_pack(protocol_mod.SMOKE, smoke_out, log=_quiet)
    assert pack_index.canonical(index) + "\n" == (smoke_out / "index.json").read_text(encoding="ascii")
    assert _files(smoke_out) == before


@pytest.mark.parametrize("target", ["scores", "comparisons", "results", "index.json", "REPORT.md", "extra"])
def test_verify_fails_closed_on_any_changed_or_extra_file(smoke_out: Path, tmp_path: Path, target: str) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(smoke_out, copy)
    if target == "extra":
        (copy / "scores" / "unindexed.json").write_text("{}\n", encoding="ascii")
    else:
        path = copy / target if (copy / target).is_file() else sorted((copy / target).iterdir())[0]
        path.write_bytes(path.read_bytes()[:-1] + b" \n")  # one byte added; the JSON content still parses
    with pytest.raises((runner.PackError, EmberBenchError)):
        runner.verify_pack(protocol_mod.SMOKE, copy, log=_quiet)


@pytest.mark.parametrize("extra", ["extra.json", "nested/extra.json"])
def test_verify_fails_closed_on_an_extra_file_in_the_source_suite(smoke_out: Path, tmp_path: Path, extra: str) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(smoke_out, copy)
    source = copy / runner.SOURCE_SUITE_DIR
    assert (source / "manifest.json").is_file()  # the smoke protocol writes its subset suite
    (source / extra).parent.mkdir(parents=True, exist_ok=True)
    (source / extra).write_text("{}\n", encoding="ascii")
    # The expected file set comes from the suite manifest, not from the files found on disk.
    expected = runner._expected_files(copy, _index(copy), runner.protocol_suite(protocol_mod.SMOKE))
    assert source / extra not in expected
    assert {p for p in source.rglob("*") if p.is_file()} - expected == {source / extra}
    # load_suite rejects an unlisted entry of the suite directory; the pack's own check rejects any other extra file.
    with pytest.raises((runner.PackError, EmberBenchError), match="no pack artifact accounts for|not listed in the manifest"):
        runner.verify_pack(protocol_mod.SMOKE, copy, log=_quiet)


def test_expected_files_are_exactly_the_files_of_a_clean_run(smoke_out: Path) -> None:
    expected = runner._expected_files(smoke_out, _index(smoke_out), runner.protocol_suite(protocol_mod.SMOKE))
    assert expected == {p for p in smoke_out.rglob("*") if p.is_file()}


def test_resume_fails_closed_on_a_result_with_non_canonical_bytes(smoke_out: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(smoke_out, copy)
    path = sorted((copy / "results").iterdir())[0]
    canonical_bytes = path.read_bytes()
    # Same JSON content, other whitespace: the result still loads and authenticates by replay.
    reformatted = (json.dumps(json.loads(canonical_bytes), indent=1, sort_keys=True) + "\n").encode("ascii")
    assert reformatted != canonical_bytes
    path.write_bytes(reformatted)
    suite = load_environment_pack(copy / "environment-pack").suite
    assert load_evaluation_result(path, suite).to_file_text().encode("ascii") == canonical_bytes
    before = _files(copy)
    with pytest.raises(runner.PackError, match="differs"):
        runner.run_pack(protocol_mod.SMOKE, copy, log=_quiet)
    assert path.read_bytes() == reformatted  # never rewritten or normalized
    assert _files(copy) == before


# --------------------------------------------------------------------------- one manifest, official scoring, authentic artifacts


def test_every_model_uses_the_identical_manifest(smoke_out: Path) -> None:
    index = _index(smoke_out)
    pack = load_environment_pack(smoke_out / "environment-pack", expected_digest=index["environment_pack"]["digest"])
    manifest_hash = index["schedule"]["evaluation_manifest"]["manifest_hash"]
    assert manifest_hash == protocol_mod.SMOKE.manifest_hash == pack.manifest.manifest_hash()
    ids = [m["model_id"] for m in index["models"]]
    assert ids == ["always-stay", *BASELINE_IDS, pack_agents.FORECAST_ID, pack_agents.COOPERATIVE_ID]
    keys = None
    for model in index["models"]:
        assert model["result"]["evaluation_manifest_hash"] == manifest_hash
        result = load_evaluation_result(smoke_out / model["result"]["file"], pack.suite)  # replays every record
        assert result.manifest == pack.manifest and result.is_complete
        episode_keys = [e.key(manifest_hash) for e in result.episodes]
        keys = keys or episode_keys
        assert episode_keys == keys


def test_score_reports_authenticate_and_come_from_the_official_api(smoke_out: Path) -> None:
    index = _index(smoke_out)
    pack = load_environment_pack(smoke_out / "environment-pack", expected_digest=index["environment_pack"]["digest"])
    for model in index["models"]:
        result = load_evaluation_result(smoke_out / model["result"]["file"], pack.suite)
        text = (smoke_out / model["score_report"]["file"]).read_text(encoding="ascii")
        # Recomputing with emberbench.scoring gives the stored artifact byte for byte: the pack holds no scoring of its own.
        report = score_result(result, pack, resamples=protocol_mod.SMOKE.resamples)
        assert report.to_json() + "\n" == text
        assert report.official and model["score_report"]["official_procedure"] is True
        assert report.digest() == model["score_report"]["digest"] == pack_index.text_digest(text)
        assert list(report.result_digests) == model["score_report"]["result_digests"] == [model["result"]["digest"]]
        assert pack_index.text_digest(pack_index.canonical(model["descriptor"])) == model["config_digest"]
        diag = (smoke_out / model["diagnostics"]["file"]).read_text(encoding="ascii")
        assert pack_index.text_digest(diag) == model["diagnostics"]["digest"]
        assert json.loads(diag)["result_digest"] == model["result"]["digest"]


def test_paired_comparisons_use_the_same_worlds_and_the_official_api(smoke_out: Path) -> None:
    index = _index(smoke_out)
    pack = load_environment_pack(smoke_out / "environment-pack")
    reports = {
        m["model_id"]: score_result(load_evaluation_result(smoke_out / m["result"]["file"], pack.suite), pack, resamples=protocol_mod.SMOKE.resamples)
        for m in index["models"]
    }
    score_digests = {m["model_id"]: m["score_report"]["digest"] for m in index["models"]}
    pairs = [(p["model_a"], p["model_b"]) for p in index["paired_comparisons"]]
    for required in protocol_mod.REQUIRED_PAIRS:
        assert required in pairs
    all_indices = [e.index for e in pack.manifest.episodes]
    for pair in index["paired_comparisons"]:
        text = (smoke_out / pair["file"]).read_text(encoding="ascii")
        assert pack_index.text_digest(text) == pair["digest"]
        doc = json.loads(text)
        a, b = pair["model_a"], pair["model_b"]
        assert doc["score_report_digest_a"] == score_digests[a] and doc["score_report_digest_b"] == score_digests[b]
        assert doc["paired_episode_indices"] == all_indices
        assert doc["evaluation_manifest_hash"] == pack.manifest.manifest_hash() and doc["environment_pack_digest"] == pack.digest
        official = compare_scores(reports[a], reports[b], resamples=protocol_mod.SMOKE.resamples)
        assert doc["comparison"] == official.to_dict() and pair["outcome"] == official.outcome.value


SCORING_INTERNALS = {"score_episode", "score_evidence", "episode_evidence", "quality_components", "severe_failures", "_references", "_stratified_ci", "_hierarchical"}


def test_pack_has_no_scoring_implementation_of_its_own() -> None:
    for path in PACK_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("emberbench"):
                names = {alias.name for alias in node.names}
                assert node.module != "emberbench.scoring", (path.name, node.module)
                assert not names & SCORING_INTERNALS, (path.name, names)
            if isinstance(node, ast.FunctionDef):
                assert not re.fullmatch(r"_?(compute_|calc_)?(ember_)?(score|scores|scoring|episode_score|compare)(_episodes?|_results?)?", node.name), (path.name, node.name)
    runner_tree = ast.parse((PACK_DIR / "run_public_benchmark.py").read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(runner_tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert {"score_result", "compare_scores", "build_unique_world_manifest", "freeze_environment_pack"} <= imported
    assert "compare_results" not in imported  # the old metric comparison is not the headline any more


# --------------------------------------------------------------------------- reports


def test_report_is_labelled_and_reflects_the_index(smoke_out: Path) -> None:
    report = (smoke_out / "REPORT.md").read_text(encoding="utf-8")
    index = _index(smoke_out)
    assert LABEL in report
    assert pack_index.index_digest(index) in report
    assert index["suite"]["suite_hash"] in report and index["schedule"]["evaluation_manifest"]["manifest_hash"] in report
    assert index["environment_pack"]["digest"] in report
    for model in index["models"]:
        assert model["config_digest"] in report
    header = next(line for line in report.splitlines() if line.startswith("| # | Model | EMBER Score"))
    assert "ongestion" not in header and "Block" not in header  # congestion is never a score column


def test_committed_report_matches_the_golden_public_index() -> None:
    report = (PACK_DIR / "REPORT.md").read_text(encoding="utf-8")
    golden = json.loads((GOLDEN_DIR / "public-development-index.json").read_text(encoding="ascii"))
    assert LABEL in report
    assert f"Experiment index digest: `{pack_index.index_digest(golden)}`." in report
    assert golden["schedule"]["unique_worlds"] == 19 and golden["schedule"]["evaluation_manifest"]["episode_count"] == 19
    for model in golden["models"]:
        assert model["config_digest"] in report
    # The golden index pins the current pack agents and protocol.
    specs = {s.model_id: s for s in pack_agents.pack_agent_specs()}
    assert [m["model_id"] for m in golden["models"]] == list(specs)
    for model in golden["models"]:
        assert model["config_digest"] == specs[model["model_id"]].config_digest


# --------------------------------------------------------------------------- fairness and separation


class _SpyAgent:
    def __init__(self, inner: Agent, calls: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        self._inner = inner
        self._calls = calls

    def act(self, *args: Any, **kwargs: Any) -> Any:
        self._calls.append((args, kwargs))
        return self._inner.act(*args, **kwargs)


def test_fair_agents_receive_only_the_observation(tmp_path: Path) -> None:
    suite = runner.protocol_suite(protocol_mod.SMOKE)
    stochastic = dataclasses.replace(
        suite,
        manifest=dataclasses.replace(suite.manifest, entries=suite.manifest.entries[1:]),
        scenarios=suite.scenarios[1:],
    )
    seed = 2**63 + 12345
    manifest = build_evaluation_manifest(stochastic, [seed])
    forbidden = (str(seed), format(seed, "x"), manifest.manifest_hash(), stochastic.manifest.suite_hash())
    for spec in pack_agents.pack_agent_specs():
        factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        act_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        inner = spec.factory

        def factory(*args: Any, **kwargs: Any) -> Agent:
            factory_calls.append((args, kwargs))
            return _SpyAgent(inner(), act_calls)

        out = tmp_path / f"{pack_index.model_slug(spec.model_id)}.json"
        run_evaluation(stochastic, manifest, spec.model_id, agent_policy_factory(factory), out)
        assert factory_calls == [((), {})], spec.model_id
        assert act_calls, spec.model_id
        for args, kwargs in act_calls:
            assert kwargs == {} and len(args) == 1 and type(args[0]) is Observation
            text = serialize_observation(args[0])
            assert not any(token in text for token in forbidden)


ALLOWED_PACK_IMPORTS = {"emberbench"}
PRIVILEGED_MODULES = ("emberbench.oracle", "emberbench.simulator", "emberbench.scenario", "emberbench.rng",
                      "emberbench.fire", "emberbench.state", "emberbench.evaluation", "emberbench.generator",
                      "emberbench.scoring", "emberbench.calibration", "analyze_suite_design")


def test_pack_agents_use_only_the_public_package_namespace() -> None:
    tree = ast.parse((PACK_DIR / "pack_agents.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("emberbench", "analyze_suite_design")):
            assert node.module in ALLOWED_PACK_IMPORTS, node.module
            names = {alias.name for alias in node.names}
            assert names <= {"Action", "Agent", "FireModelName", "Observation", "Pos", "baseline_factory",
                             "BASELINE_IDS", "get_baseline"}, names
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith(PRIVILEGED_MODULES) for alias in node.names)


def test_no_privileged_reference_is_ranked_and_the_references_hold(smoke_out: Path) -> None:
    index = _index(smoke_out)
    fair = [m["model_id"] for m in index["models"]]
    assert all(m["fair"] and m["observation_only"] and m["descriptor"]["privileged"] is False for m in index["models"])
    assert sorted(index["leaderboard"]) == sorted(fair)
    assert not any("privileged" in m or "oracle" in m or "bound" in m for m in index["leaderboard"])
    assert index["privileged_references"] == {
        "competing_model": False,
        "digest": index["privileged_references"]["digest"],
        "fair": False,
        "file": "privileged/references.json",
        "privileged": True,
    }
    text = (smoke_out / "privileged" / "references.json").read_text(encoding="ascii")
    assert pack_index.text_digest(text) == index["privileged_references"]["digest"]
    refs = json.loads(text)
    assert refs["privileged"] is True and refs["fair"] is False and refs["competing_model"] is False
    bound = {w["index"]: w for w in refs["worlds"]}
    pack = load_environment_pack(smoke_out / "environment-pack")
    for model in index["models"]:
        result = load_evaluation_result(smoke_out / model["result"]["file"], pack.suite)
        for episode in result.episodes:
            w = bound[episode.episode.index]
            assert episode.metrics.safety.evacuated <= w["m_cap"] <= w["m_star"] <= w["population"]


def test_pack_holds_no_hidden_or_private_assets(smoke_out: Path) -> None:
    tracked = [p.relative_to(REPO_ROOT).as_posix() for p in (*PACK_DIR.rglob("*"), *GOLDEN_DIR.rglob("*"))
               if p.is_file() and "__pycache__" not in p.parts]
    assert tracked and check_repository_paths(tracked) == []
    assert not [p for p in tracked if p.startswith("tools/") and p.endswith(".json")]
    for path in smoke_out.rglob("*"):
        rel = path.relative_to(smoke_out).as_posix()
        if rel == "environment-pack/evaluation-manifest.json":
            # The frozen pack's schedule is an evaluator workspace file: the leak guard keeps it out of any
            # repository, which is why outputs go outside the repository.
            assert forbidden_reason(rel) is not None
        else:
            assert forbidden_reason(rel) is None, rel
    index = _index(smoke_out)
    assert index["suite"]["visibility"] == "public"
    assert {s["scenario_id"] for s in index["suite"]["scenarios"]} <= {e.scenario_id for e in load_suite(REPO_ROOT / "scenarios" / "public" / "v1").manifest.entries}


# --------------------------------------------------------------------------- opt-in full run


@pytest.mark.skipif(os.environ.get("EMBERBENCH_FULL_PACK") != "1", reason="full pack run takes minutes; set EMBERBENCH_FULL_PACK=1")
def test_full_public_run_reproduces_the_golden_index_and_report(tmp_path: Path) -> None:
    out = tmp_path / "public"
    runner.run_pack(protocol_mod.PUBLIC, out, log=_quiet)
    assert (out / "index.json").read_text(encoding="ascii") == (
        GOLDEN_DIR / "public-development-index.json"
    ).read_text(encoding="ascii")
    assert (out / "REPORT.md").read_bytes() == (PACK_DIR / "REPORT.md").read_bytes()
    runner.verify_pack(protocol_mod.PUBLIC, out, log=_quiet)
