"""Build the data module of the results website (docs/site/) from finished run directories.

Each run directory holds ``scores/<model-id>.json`` (official ``ScoreReport``s)
and, for a remote or decision-model run, ``summary.json``. The output is one
JavaScript module (``export default {...}``) with the fields the site shows. It
is not a ``.json`` file because the public-asset guard keeps ``.json`` files for
benchmark assets. It has no paths, hosts or timestamps, so the same runs always
give the same bytes.

Every score report must be official and scored against the same environment
pack, or the build stops: the site never mixes packs.

Usage::

    python tools/build_results_site.py --run ../pack-run --run ../jev-run --run ../subq-run \\
        --run ../decision-run-1 --out docs/site/data/results.js \\
        --price typesafe-jev-latest-choice-r1=0.042,0 --price subq-subq-snap-preview-choice-r1=0.05,0

``--price`` gives the list price (USD per million input and output tokens) of a
model whose API reported no cost. The site marks that cost as an estimate.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
from fractions import Fraction

from emberbench.canonical import sha256_hex
from emberbench.scoring import OFFICIAL_CONSTANTS, SCORING_VERSION

LABEL = "PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD"


def _number(value: Any, name: str, *, integer: bool = False) -> int | float | None:
    """Unknown operational measurements stay unknown; zero remains a measurement."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected a finite non-negative number or null")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name}: expected an integer or null")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: expected a lowercase SHA-256 digest")
    return value


def _validate_report(report: dict[str, Any]) -> tuple[Any, ...]:
    """Check comparability without claiming to replay-authenticate a JSON report."""
    if report.get("official") is not True:
        raise ValueError("not an official score report")
    for name in ("pack_id", "suite_id", "scoring_version", "model_id"):
        if not isinstance(report.get(name), str) or not report[name].strip():
            raise ValueError(f"{name}: required nonempty identity")
    hashes = tuple(_hash(report.get(k), k) for k in ("pack_digest", "suite_hash", "evaluation_manifest_hash", "constants_digest"))
    if report.get("constants") != OFFICIAL_CONSTANTS.to_dict() or report["constants_digest"] != OFFICIAL_CONSTANTS.digest():
        raise ValueError("constants do not match the official scoring constants")
    if report["scoring_version"] != SCORING_VERSION:
        raise ValueError("scoring version differs from the installed official scorer")
    digests = report.get("result_digests")
    if not isinstance(digests, list) or not digests:
        raise ValueError("result_digests: required report provenance")
    for value in digests:
        _hash(value, "result_digests")
    trials = _number(report.get("trials"), "trials", integer=True)
    if trials is None or trials < 1 or len(digests) != trials or report["totals"]["trials"] != trials:
        raise ValueError("inconsistent trial counts")
    # JSON's default decoder accepts NaN and Infinity. Reject them anywhere in a report.
    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite report measurement")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(report)
    for name in ("score", "safety", "quality"):
        value = _number(report.get(name), name)
        if value is not None and value > (100 if name == "score" else 1):
            raise ValueError(f"{name}: outside official range")
    worlds = tuple((e["index"], e["scenario_id"], e["seed"]) for e in report["episodes"])
    if not worlds or len(worlds) % trials:
        raise ValueError("inconsistent episode-trial schedule")
    world_count = len(worlds) // int(trials)
    scheduled = worlds[:world_count]
    if len(set(scheduled)) != world_count or any(worlds[t * world_count:(t + 1) * world_count] != scheduled for t in range(int(trials))):
        raise ValueError("inconsistent repeated-trial world identities")
    return (report["pack_id"], report["suite_id"], report["scoring_version"], *hashes, scheduled, tuple(report["not_applicable_episodes"]))


def _round(value: Any, digits: int = 4) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def _name(model_id: str, summary: dict[str, Any] | None) -> tuple[str, str]:
    """The display name and the kind of one model."""
    if summary is None or summary.get("model_id") != model_id:
        return model_id, "Reference agent"
    requested = str(summary.get("requested_model") or model_id)
    if "/" not in requested:
        requested = f"{model_id.split('-', 1)[0]}/{requested}"  # typesafe-jev-latest-... -> typesafe/jev-latest
    return requested, "Decision model"


def model_entry(report: dict[str, Any], summary: dict[str, Any] | None, price: tuple[float, float] | None = None) -> dict[str, Any]:
    name, kind = _name(report["model_id"], summary)
    ci = report.get("ci") or {}
    entry: dict[str, Any] = {
        "id": report["model_id"],
        "name": name,
        "kind": kind,
        "score": _round(report["score"]),
        "ci": [_round(ci.get("lower")), _round(ci.get("upper"))] if ci else None,
        "safety": _round(report["safety"]),
        "quality": _round(report["quality"]),
        "trials": report["trials"],
        "families": report["families"],
        "provenance": {
            "report_digest": sha256_hex(report),
            "evaluation_manifest_hash": report["evaluation_manifest_hash"],
            "suite_hash": report["suite_hash"],
            "constants_digest": report["constants_digest"],
            "result_digests": report["result_digests"],
            "not_applicable_episodes": report["not_applicable_episodes"],
            "ci_method": report.get("ci"),
        },
        "averages": {k: _round(v) for k, v in report["averages"].items()},
        "totals": report["totals"],
        "components": {k: {"value": _round(v["value"]), "defined": v["defined"], "total": v["total"]} for k, v in report["components"].items()},
        "robustness": {k: _round(v) for k, v in report["robustness"].items()},
        "diagnostics": {k: _round(v) for k, v in report["diagnostics"].items()},
        "scenarios": [
            {"name": g["name"], "episodes": g["episodes"], "score": _round(g["score"]), "safety": _round(g["safety"]), "worst": _round(g["worst_episode"])}
            for g in report["scenarios"]
        ],
        "episodes": [
            {
                "index": e["index"],
                "scenario_id": e["scenario_id"],
                "seed": e["seed"],
                "status": e["status"],
                "score": _round(e["score"]),
                "safety": _round(e["safety"]),
                "quality": _round(e["quality"]),
                "failure": e.get("failure"),
                "components": e.get("components"),
                "severity": e.get("severity"),
                "severe": e.get("severe"),
                **{key: (e["evidence"].get(key) if e.get("evidence") is not None else None)
                   for key in ("evacuated", "casualties", "stranded", "population", "max_evacuated")},
            }
            for e in report["episodes"]
        ],
    }
    if kind == "Decision model" and summary is not None:
        api = summary.get("api") or {}
        tokens_in = _number(api.get("input_tokens"), "input_tokens", integer=True)
        tokens_out = _number(api.get("output_tokens"), "output_tokens", integer=True)
        cost = _number(api.get("cost"), "cost")
        source = api.get("cost_source") or ("reported" if cost is not None else "not reported")
        if source not in {"reported", "list price", "not reported"}:
            raise ValueError("unknown API cost source")
        if source == "not reported":
            cost = None
        if cost is None and price is not None and tokens_in is not None and tokens_out is not None:
            cost, source = round((tokens_in * price[0] + tokens_out * price[1]) / 1e6, 6), "list price"
        if source != "not reported" and cost is None:
            raise ValueError("API cost source has no measured or estimated cost")
        latency = _number(api.get("mean_act_latency_s"), "mean_act_latency_s")
        entry["api"] = {
            "resolved_models": summary.get("resolved_models") or [],
            "requests": _number(api.get("requests_sent"), "requests_sent", integer=True),
            "decisions": _number(api.get("act_calls"), "act_calls", integer=True),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": cost if source != "not reported" else None,
            "cost_source": source,
            "wall_clock_s": _number(api.get("wall_clock_s"), "wall_clock_s"),
            "mean_act_latency_s": float(latency) if latency is not None else None,
            "first_response_s": _number(api.get("first_response_s"), "first_response_s"),
            "median_response_s": _number(api.get("median_response_s"), "median_response_s"),
            **{key: _number(api.get(key), key, integer=True) for key in ("cache_hits", "retries", "retries_403", "format_errors")},
            "measurement_scope": "Runner invocation; resumed runs may cover only the final segment. Response timings exclude cache hits.",
        }
        entry["provenance"]["adapter_version"] = summary.get("adapter_version")
        entry["provenance"]["adapter_sha256"] = summary.get("adapter_sha256")
        entry["provenance"]["protocol"] = summary.get("protocol")
    return entry


def parse_price(text: str) -> tuple[str, tuple[float, float]]:
    """``MODEL_ID=IN,OUT``: the list price in USD per million input and output tokens."""
    model_id, sep, prices = text.partition("=")
    parts = prices.split(",")
    if not model_id or not sep or len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{text!r}: expected MODEL_ID=IN,OUT")
    try:
        prices_pair = (float(parts[0]), float(parts[1]))
        for value in prices_pair:
            _number(value, "price")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return model_id, prices_pair


def build(runs: list[Path], prices: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    prices = prices or {}
    models: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    pack: tuple[Any, ...] | None = None
    for run in runs:
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        for path in sorted((run / "scores").glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            try:
                key = _validate_report(report)
                if summary is not None:
                    if summary.get("model_id") != report["model_id"]:
                        raise ValueError("summary model_id differs from score report")
                    for field, report_field in (("pack_digest", "pack_digest"), ("manifest_hash", "evaluation_manifest_hash")):
                        if summary.get(field) != report[report_field]:
                            raise ValueError(f"summary {field} differs from score report")
                    if summary.get("official_procedure") is not True:
                        raise ValueError("summary procedure is not official")
                    if summary.get("episodes") != len(report["episodes"]):
                        raise ValueError("summary episode count differs from score report")
            except (ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"{path}: {exc}") from exc
            if pack is not None and key != pack:
                raise SystemExit(f"{path}: inconsistent pack, scoring version, provenance or world schedule")
            pack = key
            if report["model_id"] in models:
                raise SystemExit(f"{path}: model {report['model_id']} is in two runs")
            try:
                models[report["model_id"]] = model_entry(report, summary, prices.get(report["model_id"]))
                reports[report["model_id"]] = report
            except (ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"{path}: {exc}") from exc
    if pack is None:
        raise SystemExit("no score report found")
    if unknown := sorted(set(prices) - set(models)):
        raise SystemExit(f"--price for a model that is in no run: {', '.join(unknown)}")
    def order(m: dict[str, Any]) -> tuple[Any, ...]:
        # Official descriptive order (scoring.leaderboard_key), never a significance claim.
        raw = reports[m["id"]]
        def descending(value: Any) -> Fraction:
            return Fraction(1) if value is None else -Fraction(str(value))
        score = raw["score"]
        totals = raw["totals"]
        return (
            descending(None if score is None else Fraction(round(Fraction(str(score)) * 10), 10)), descending(raw["safety"]),
            Fraction(totals["avoidable_casualties"], totals["trials"]), Fraction(totals["avoidable_stranded"], totals["trials"]),
            descending(raw["robustness"]["worst_scenario"]), m["id"],
        )
    ranked = sorted(models.values(), key=order)
    for rank, entry in enumerate(ranked, 1):
        entry["rank"] = rank
    return {
        "label": LABEL,
        "pack_id": pack[0],
        "pack_digest": pack[3],
        "suite_id": pack[1],
        "scoring_version": pack[2],
        "suite_hash": pack[4],
        "evaluation_manifest_hash": pack[5],
        "constants_digest": pack[6],
        "world_count": len(pack[7]),
        "generation": {"tool": "tools/build_results_site.py", "schema_version": 2, "source": "Official ScoreReport JSON and matched remote-run summaries", "timestamps": "No evaluation timestamps are available in these source schemas."},
        "unit": "EMBER Score: benchmark-performance points (0-100), not a percentage of people",
        "models": ranked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", required=True, help="a finished run directory with scores/; repeatable")
    parser.add_argument("--out", type=Path, default=Path("docs/site/data/results.js"))
    parser.add_argument(
        "--price", type=parse_price, action="append", default=[],
        help="MODEL_ID=IN,OUT: list price in USD per million input and output tokens, for a run whose API reported no cost; repeatable",
    )  # fmt: skip
    args = parser.parse_args(argv)
    data = build(args.run, dict(args.price))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "// Generated by tools/build_results_site.py. Do not edit.\nexport default " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{args.out}: {len(data['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
