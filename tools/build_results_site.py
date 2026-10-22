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
        --run ../decision-run-1 --out docs/site/data/results.js
"""

import argparse
import json
from pathlib import Path
from typing import Any

LABEL = "PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD"


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


def model_entry(report: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
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
                "evacuated": e["evidence"]["evacuated"],
                "casualties": e["evidence"]["casualties"],
                "stranded": e["evidence"]["stranded"],
                "population": e["evidence"]["population"],
            }
            for e in report["episodes"]
        ],
    }
    if kind == "Decision model" and summary is not None:
        api = summary.get("api") or {}
        entry["api"] = {
            "resolved_models": summary.get("resolved_models") or [],
            "requests": api.get("requests_sent"),
            "input_tokens": api.get("input_tokens"),
            "output_tokens": api.get("output_tokens"),
            "cost_usd": api.get("cost"),
            "mean_act_latency_s": api.get("mean_act_latency_s"),
        }
    return entry


def build(runs: list[Path]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    pack: tuple[str, str, str, str] | None = None
    for run in runs:
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        for path in sorted((run / "scores").glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("official"):
                raise SystemExit(f"{path}: not an official score report")
            key = (report["pack_id"], report["pack_digest"], report["suite_id"], report["scoring_version"])
            if pack is not None and key != pack:
                raise SystemExit(f"{path}: scored against another pack or scoring version ({key[1]})")
            pack = key
            if report["model_id"] in models:
                raise SystemExit(f"{path}: model {report['model_id']} is in two runs")
            models[report["model_id"]] = model_entry(report, summary)
    if pack is None:
        raise SystemExit("no score report found")
    ranked = sorted(models.values(), key=lambda m: (-(m["score"] if m["score"] is not None else -1.0), m["id"]))
    return {
        "label": LABEL,
        "pack_id": pack[0],
        "pack_digest": pack[1],
        "suite_id": pack[2],
        "scoring_version": pack[3],
        "unit": "EMBER Score: benchmark-performance points (0-100), not a percentage of people",
        "models": ranked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", required=True, help="a finished run directory with scores/; repeatable")
    parser.add_argument("--out", type=Path, default=Path("docs/site/data/results.js"))
    args = parser.parse_args(argv)
    data = build(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "// Generated by tools/build_results_site.py. Do not edit.\nexport default " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{args.out}: {len(data['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
