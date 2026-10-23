"""The site consumes official public reports, with operational unknowns preserved."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from build_results_site import build, main, model_entry, parse_price
from emberbench import EVALUATION_RESULT_VERSION, EvaluationResult, freeze_environment_pack, load_suite, score_result


@pytest.fixture(scope="module")
def official_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    suite = load_suite(Path(__file__).resolve().parents[1] / "scenarios" / "public" / "v1")
    pack = freeze_environment_pack(suite, (0,), tmp_path_factory.mktemp("site-pack") / "pack", pack_id="site-test-public")
    # Official failed trials are legitimate score reports too: all scheduled scores are zero,
    # while outcome evidence is absent. The site must not invent people outcomes for them.
    result = EvaluationResult(EVALUATION_RESULT_VERSION, "test-model", pack.manifest, ())
    return score_result(result, pack, resamples=200).to_dict()


def write_run(root: Path, report: dict[str, Any], summary: dict[str, Any] | None = None) -> Path:
    (root / "scores").mkdir(parents=True)
    (root / "scores" / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if summary is not None:
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def remote_summary(report: dict[str, Any], **api: Any) -> dict[str, Any]:
    return {
        "model_id": report["model_id"], "requested_model": "provider/model", "official_procedure": True,
        "pack_digest": report["pack_digest"], "manifest_hash": report["evaluation_manifest_hash"],
        "episodes": len(report["episodes"]), "api": api,
    }


def test_official_failed_report_preserves_missing_outcomes_and_provenance(tmp_path: Path, official_report: dict[str, Any]) -> None:
    run = write_run(tmp_path / "run", official_report)
    data = build([run])
    assert data["world_count"] == len(official_report["episodes"])
    assert data["suite_hash"] == official_report["suite_hash"]
    model = data["models"][0]
    assert model["provenance"]["result_digests"] == official_report["result_digests"]
    assert model["families"] == official_report["families"]
    assert model["episodes"][0]["evacuated"] is None
    assert model["episodes"][0]["failure"] == "missing"
    assert "api" not in model
    out = tmp_path / "site" / "results.js"
    assert main(["--run", str(run), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert main(["--run", str(run), "--out", str(out)]) == 0
    assert out.read_bytes() == first
    assert str(run).encode() not in first


@pytest.mark.parametrize("field,value", [
    ("official", "true"), ("official", False), ("pack_digest", ""), ("suite_hash", None),
    ("evaluation_manifest_hash", "bad"), ("constants_digest", "0" * 64), ("constants", {}),
    ("result_digests", []), ("score", float("nan")), ("score", float("inf")), ("trials", 0),
])
def test_invalid_report_fails_closed(tmp_path: Path, official_report: dict[str, Any], field: str, value: Any) -> None:
    report = copy.deepcopy(official_report)
    report[field] = value
    with pytest.raises(SystemExit):
        build([write_run(tmp_path / "bad", report)])


@pytest.mark.parametrize("field,value", [
    ("pack_id", "other"), ("pack_digest", "0" * 64), ("suite_id", "other"),
    ("scoring_version", "other"), ("suite_hash", "0" * 64), ("evaluation_manifest_hash", "0" * 64),
])
def test_mixed_provenance_fails_closed(tmp_path: Path, official_report: dict[str, Any], field: str, value: Any) -> None:
    other = copy.deepcopy(official_report)
    other["model_id"] = "other-model"
    other[field] = value
    with pytest.raises(SystemExit, match="inconsistent|scoring version"):
        build([write_run(tmp_path / "one", official_report), write_run(tmp_path / "two", other)])


def test_mixed_worlds_duplicate_models_and_unknown_prices_rejected(tmp_path: Path, official_report: dict[str, Any]) -> None:
    first = write_run(tmp_path / "one", official_report)
    with pytest.raises(SystemExit, match="two runs"):
        build([first, first])
    with pytest.raises(SystemExit, match="in no run"):
        build([first], {"unknown": (1, 1)})
    other = copy.deepcopy(official_report)
    other["model_id"] = "other-model"
    other["episodes"][0]["seed"] += 1
    with pytest.raises(SystemExit, match="world schedule"):
        build([first, write_run(tmp_path / "two", other)])


@pytest.mark.parametrize("field,value", [("model_id", "wrong"), ("pack_digest", "0" * 64), ("manifest_hash", "0" * 64), ("official_procedure", False), ("episodes", 1)])
def test_summary_must_match_report(tmp_path: Path, official_report: dict[str, Any], field: str, value: Any) -> None:
    summary = remote_summary(official_report)
    summary[field] = value
    with pytest.raises(SystemExit, match="summary"):
        build([write_run(tmp_path / "bad", official_report, summary)])


def test_api_missing_zero_and_estimated_cost_are_distinct(official_report: dict[str, Any]) -> None:
    missing = model_entry(official_report, remote_summary(official_report))["api"]
    assert missing["input_tokens"] is missing["output_tokens"] is missing["cost_usd"] is None
    assert missing["requests"] is missing["mean_act_latency_s"] is None
    zero = model_entry(official_report, remote_summary(official_report, cost=0, cost_source="reported", input_tokens=0, output_tokens=0))["api"]
    assert zero["cost_usd"] == 0 and zero["cost_source"] == "reported"
    assert zero["input_tokens"] == 0
    summary = remote_summary(official_report, cost=0, cost_source="not reported", input_tokens=1000000, output_tokens=2000000)
    unknown = model_entry(official_report, summary)["api"]
    assert unknown["cost_usd"] is None
    estimated = model_entry(official_report, summary, (1, 2))["api"]
    assert estimated["cost_usd"] == 5 and estimated["cost_source"] == "list price"
    assert model_entry(official_report, remote_summary(official_report), (1, 2))["api"]["cost_usd"] is None


def test_display_rank_uses_official_safety_tiebreak_and_missing_scores_last(tmp_path: Path, official_report: dict[str, Any]) -> None:
    reports = []
    for name, score, safety in (("higher-raw", 10.04, 0.1), ("safer", 10.01, 0.9), ("missing", None, None)):
        report = copy.deepcopy(official_report)
        report.update(model_id=name, score=score, safety=safety)
        reports.append(write_run(tmp_path / name, report))
    assert [m["id"] for m in build(reports)["models"]] == ["safer", "higher-raw", "missing"]
    assert [m["rank"] for m in build(reports)["models"]] == [1, 2, 3]


def test_summary_private_operational_context_is_not_exported(tmp_path: Path, official_report: dict[str, Any]) -> None:
    summary = remote_summary(official_report)
    summary.update(endpoint="https://private.example/endpoint", key="private-token", cache_path="C:/private/cache.jsonl")
    data = build([write_run(tmp_path / "run", official_report, summary)])
    serialized = json.dumps(data)
    assert "private" not in serialized and "endpoint" not in serialized and "cache_path" not in serialized


def test_runner_latency_string_is_accepted(official_report: dict[str, Any]) -> None:
    api = model_entry(official_report, remote_summary(official_report, mean_act_latency_s="0.577"))["api"]
    assert api["mean_act_latency_s"] == 0.577


@pytest.mark.parametrize("key,value", [("input_tokens", -1), ("requests_sent", 1.5), ("cost", float("nan")), ("mean_act_latency_s", float("inf")), ("mean_act_latency_s", "-1"), ("mean_act_latency_s", "nan"), ("wall_clock_s", -1), ("input_tokens", True)])
def test_invalid_operational_metrics_rejected(official_report: dict[str, Any], key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        model_entry(official_report, remote_summary(official_report, **{key: value}))


@pytest.mark.parametrize("text", ["no-price", "=1,2", "model=-1,0", "model=nan,0", "model=0,inf", "model=a,0"])
def test_invalid_list_prices(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_price(text)


def test_exported_rank_precedes_display_rounding(tmp_path: Path, official_report: dict[str, Any]) -> None:
    reports = []
    for name, score, safety in (("lower-bucket", 88.14996, 0.9), ("higher-bucket", 88.15001, 0.1)):
        report = copy.deepcopy(official_report)
        report.update(model_id=name, score=score, safety=safety)
        reports.append(write_run(tmp_path / name, report))
    models = build(reports)["models"]
    assert [m["id"] for m in models] == ["higher-bucket", "lower-bucket"]
    assert [m["score"] for m in models] == [88.15, 88.15]
    assert [m["rank"] for m in models] == [1, 2]
