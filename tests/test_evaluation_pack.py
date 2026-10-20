"""The V1 evaluation pack (tools/experiments/v1): reproducibility, fairness and artifact integrity.

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
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from emberbench import (
    BASELINE_IDS,
    Agent,
    Observation,
    agent_policy_factory,
    build_evaluation_manifest,
    load_evaluation_result,
    load_suite,
    run_evaluation,
    serialize_observation,
)
from emberbench.leakguard import check_repository_paths, forbidden_reason

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "tools" / "experiments" / "v1"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "experiments" / "v1"

# The canonical public suite and the pack's development manifest. A change to
# either is a benchmark change, not a pack change.
PUBLIC_SUITE_HASH = "5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b"
PUBLIC_MANIFEST_HASH = "0280777fbbcbdd2dfb9c878394d22a846e8e4dfff41c7f33fd2a73de7659b604"


def _pack_module(name: str) -> ModuleType:
    if str(PACK_DIR) not in sys.path:
        sys.path.insert(0, str(PACK_DIR))
    return importlib.import_module(name)


runner = _pack_module("run_public_benchmark")
protocol_mod = _pack_module("pack_protocol")
pack_agents = _pack_module("pack_agents")
pack_index = _pack_module("pack_index")


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pack") / "smoke"
    runner.run_pack(protocol_mod.SMOKE, out, log=lambda message: None)
    return out


def _index(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "index.json").read_text(encoding="ascii"))
    return data


# --------------------------------------------------------------------------- pins


def test_public_suite_and_manifest_hashes_are_unchanged() -> None:
    suite = load_suite(REPO_ROOT / "scenarios" / "public" / "v1")
    assert suite.manifest.suite_hash() == PUBLIC_SUITE_HASH
    assert protocol_mod.PUBLIC.suite_hash == PUBLIC_SUITE_HASH
    manifest = build_evaluation_manifest(suite, protocol_mod.DEVELOPMENT_SEEDS)
    assert manifest.manifest_hash() == PUBLIC_MANIFEST_HASH == protocol_mod.PUBLIC.manifest_hash
    golden = json.loads((GOLDEN_DIR / "public-development-index.json").read_text(encoding="ascii"))
    assert golden["suite"]["suite_hash"] == PUBLIC_SUITE_HASH
    assert golden["evaluation_manifest"]["manifest_hash"] == PUBLIC_MANIFEST_HASH
    assert golden["protocol"]["seeds"] == list(protocol_mod.DEVELOPMENT_SEEDS)


def test_runner_refuses_a_changed_seed_schedule() -> None:
    changed = dataclasses.replace(protocol_mod.SMOKE, seeds=(0, 1, 3))
    suite = runner.protocol_suite(changed)
    with pytest.raises(runner.PackError, match="manifest hash"):
        runner.protocol_manifest(changed, suite)


# --------------------------------------------------------------------------- reproducibility


def test_smoke_run_reproduces_the_golden_index(smoke_out: Path) -> None:
    text = (smoke_out / "index.json").read_text(encoding="ascii")
    assert text == (GOLDEN_DIR / "smoke-index.json").read_text(encoding="ascii")


def test_rerun_resumes_and_gives_identical_digests(smoke_out: Path) -> None:
    before = {p: p.read_bytes() for p in smoke_out.rglob("*") if p.is_file()}
    index = runner.run_pack(protocol_mod.SMOKE, smoke_out, log=lambda message: None)
    assert pack_index.canonical(index) + "\n" == (smoke_out / "index.json").read_text(encoding="ascii")
    after = {p: p.read_bytes() for p in smoke_out.rglob("*") if p.is_file()}
    assert after == before


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
        runner.run_pack(protocol_mod.SMOKE, REPO_ROOT / "pack-out", log=lambda message: None)
    assert not (REPO_ROOT / "pack-out").exists()


def test_runner_denies_network_events() -> None:
    with pytest.raises(runner.PackError, match="network"):
        runner._deny_network("socket.connect", ())
    runner._deny_network("open", ())


# --------------------------------------------------------------------------- one manifest, authentic artifacts


def test_every_model_uses_the_one_evaluation_manifest(smoke_out: Path) -> None:
    index = _index(smoke_out)
    suite = runner.protocol_suite(protocol_mod.SMOKE)
    manifest_hash = index["evaluation_manifest"]["manifest_hash"]
    assert manifest_hash == protocol_mod.SMOKE.manifest_hash
    ids = [m["model_id"] for m in index["models"]]
    assert ids == ["always-stay", *BASELINE_IDS, pack_agents.FORECAST_ID, pack_agents.COOPERATIVE_ID]
    keys = None
    for model in index["models"]:
        assert model["result"]["evaluation_manifest_hash"] == manifest_hash
        result = load_evaluation_result(smoke_out / model["result"]["file"], suite)  # replays every record
        assert result.manifest_hash == manifest_hash and result.is_complete
        episode_keys = [e.key(manifest_hash) for e in result.episodes]
        keys = keys or episode_keys
        assert episode_keys == keys  # identical scenario/seed order for every model


def test_index_references_authentic_artifacts(smoke_out: Path) -> None:
    index = _index(smoke_out)

    def digest_of(rel: str) -> str:
        digest: str = pack_index.text_digest((smoke_out / rel).read_text(encoding="ascii"))
        return digest

    assert digest_of(index["evaluation_manifest"]["file"]) == index["evaluation_manifest"]["file_digest"]
    result_digests = {}
    for model in index["models"]:
        assert digest_of(model["result"]["file"]) == model["result"]["digest"]
        assert digest_of(model["aggregate_statistics"]["file"]) == model["aggregate_statistics"]["digest"]
        aggregate = json.loads((smoke_out / model["aggregate_statistics"]["file"]).read_text(encoding="ascii"))
        assert aggregate["result_digest"] == model["result"]["digest"]
        assert aggregate["model_id"] == model["model_id"]
        assert pack_index.text_digest(pack_index.canonical(model["descriptor"])) == model["config_digest"]
        result_digests[model["model_id"]] = model["result"]["digest"]
    assert index["paired_statistics"]
    for pair in index["paired_statistics"]:
        assert digest_of(pair["file"]) == pair["digest"]
        doc = json.loads((smoke_out / pair["file"]).read_text(encoding="ascii"))
        assert (doc["model_a"], doc["model_b"]) == (pair["model_a"], pair["model_b"])
        assert doc["result_digest_a"] == result_digests[pair["model_a"]]
        assert doc["result_digest_b"] == result_digests[pair["model_b"]]
    assert digest_of(index["privileged_bound"]["file"]) == index["privileged_bound"]["digest"]


def test_index_has_no_absolute_paths_or_timestamps(smoke_out: Path) -> None:
    text = (smoke_out / "index.json").read_text(encoding="ascii")
    assert str(smoke_out) not in text and str(REPO_ROOT) not in text
    assert not re.search(r"[A-Za-z]:\\\\|\"/", text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)


def test_report_is_labelled_and_reflects_the_index(smoke_out: Path) -> None:
    report = (smoke_out / "REPORT.md").read_text(encoding="utf-8")
    index = _index(smoke_out)
    assert "PUBLIC DEVELOPMENT EVALUATION — NOT HELD-OUT SCORE" in report
    assert pack_index.index_digest(index) in report
    assert index["suite"]["suite_hash"] in report and index["evaluation_manifest"]["manifest_hash"] in report
    for model in index["models"]:
        assert model["config_digest"] in report


def test_committed_report_matches_the_golden_public_index() -> None:
    report = (PACK_DIR / "REPORT.md").read_text(encoding="utf-8")
    golden = json.loads((GOLDEN_DIR / "public-development-index.json").read_text(encoding="ascii"))
    assert "PUBLIC DEVELOPMENT EVALUATION — NOT HELD-OUT SCORE" in report
    assert f"Experiment index digest: `{pack_index.index_digest(golden)}`." in report
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
                      "emberbench.fire", "emberbench.state", "emberbench.evaluation", "emberbench.generator")


def test_pack_agents_use_only_the_public_package_namespace() -> None:
    tree = ast.parse((PACK_DIR / "pack_agents.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("emberbench"):
            assert node.module in ALLOWED_PACK_IMPORTS, node.module
            names = {alias.name for alias in node.names}
            assert names <= {"Action", "Agent", "FireModelName", "Observation", "Pos", "baseline_factory",
                             "BASELINE_IDS", "get_baseline"}, names
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith(PRIVILEGED_MODULES) for alias in node.names)


def test_privileged_bound_is_separate_and_holds(smoke_out: Path) -> None:
    index = _index(smoke_out)
    assert all(m["fair"] and m["descriptor"]["privileged"] is False for m in index["models"])
    assert index["privileged_bound"] == {
        "competing_model": False,
        "digest": index["privileged_bound"]["digest"],
        "fair": False,
        "file": "privileged/escape-bound.json",
        "privileged": True,
    }
    bound = json.loads((smoke_out / "privileged" / "escape-bound.json").read_text(encoding="ascii"))
    assert bound["privileged"] is True and bound["fair"] is False
    max_evacuated = {e["index"]: e["max_evacuated"] for e in bound["episodes"]}
    suite = runner.protocol_suite(protocol_mod.SMOKE)
    for model in index["models"]:
        result = load_evaluation_result(smoke_out / model["result"]["file"], suite)
        for episode in result.episodes:
            assert episode.metrics.safety.evacuated <= max_evacuated[episode.episode.index]


def test_pack_holds_no_hidden_or_private_assets(smoke_out: Path) -> None:
    tracked = [p.relative_to(REPO_ROOT).as_posix() for p in (*PACK_DIR.rglob("*"), *GOLDEN_DIR.rglob("*"))
               if p.is_file() and "__pycache__" not in p.parts]
    assert tracked and check_repository_paths(tracked) == []
    assert not [p for p in tracked if p.startswith("tools/") and p.endswith(".json")]
    for path in smoke_out.rglob("*"):
        assert forbidden_reason(path.relative_to(smoke_out).as_posix()) is None


# --------------------------------------------------------------------------- opt-in full run


@pytest.mark.skipif(os.environ.get("EMBERBENCH_FULL_PACK") != "1", reason="full pack run takes minutes; set EMBERBENCH_FULL_PACK=1")
def test_full_public_run_reproduces_the_golden_index_and_report(tmp_path: Path) -> None:
    out = tmp_path / "public"
    runner.run_pack(protocol_mod.PUBLIC, out, log=lambda message: None)
    assert (out / "index.json").read_text(encoding="ascii") == (
        GOLDEN_DIR / "public-development-index.json"
    ).read_text(encoding="ascii")
    assert (out / "REPORT.md").read_bytes() == (PACK_DIR / "REPORT.md").read_bytes()
