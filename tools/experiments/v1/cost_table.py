"""Cost table of ``emberbench run`` records (research lane; changes no score).

Reads every ``runs/*.json`` under the given run directories and writes one row
per model: score, USD per suite and per episode, calls and tokens per
episode, re-ask, invalid-slot, refusal and horizon-hit rates, and time. The
values come from the run record's accounting block (accounting 1.1), so a
cached answer counts as the original request did.

Time: ``model_s_*`` is the measured model response time per episode;
``runtime_s`` is the wall time of the last run segment only (a resumed run
spent more).

Usage::

    python tools/experiments/v1/cost_table.py --runs DIR [--runs DIR ...] --out OUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

COLUMNS = [
    "model_id", "kind", "score", "episodes", "cost_basis", "usd_suite", "usd_episode_mean", "usd_episode_median", "usd_episode_p95",
    "calls_episode_median", "calls_episode_p95", "input_tokens_episode_mean", "output_tokens_episode_mean", "input_tokens", "output_tokens",
    "reask_rate", "invalid_slot_rate", "refusal_rate", "horizon_hit_rate", "model_s_episode_median", "model_s_episode_p95", "model_s_total", "runtime_s",
]  # fmt: skip


def _stat(acc: dict[str, Any], key: str, field: str) -> Any:
    block = acc.get(key)
    return block.get(field) if isinstance(block, dict) else None


def cost_row(record: dict[str, Any]) -> dict[str, Any]:
    acc = record.get("accounting") or {}
    refusals = record.get("refusals") or {}
    return {
        "model_id": record["model"]["model_id"],
        "kind": record["model"]["kind"],
        "score": record["metrics"]["score"],
        "episodes": acc.get("episodes"),
        "cost_basis": acc.get("cost_basis"),
        "usd_suite": acc.get("total_cost"),
        "usd_episode_mean": _stat(acc, "cost_per_episode", "mean"),
        "usd_episode_median": _stat(acc, "cost_per_episode", "median"),
        "usd_episode_p95": _stat(acc, "cost_per_episode", "p95"),
        "calls_episode_median": _stat(acc, "calls_per_episode", "median"),
        "calls_episode_p95": _stat(acc, "calls_per_episode", "p95"),
        "input_tokens_episode_mean": _stat(acc, "input_tokens_per_episode", "mean"),
        "output_tokens_episode_mean": _stat(acc, "output_tokens_per_episode", "mean"),
        "input_tokens": _stat(acc, "input_tokens_per_episode", "total"),
        "output_tokens": _stat(acc, "output_tokens_per_episode", "total"),
        "reask_rate": acc.get("reask_rate"),
        "invalid_slot_rate": acc.get("invalid_slot_rate"),
        "refusal_rate": refusals.get("rate", 0.0),
        "horizon_hit_rate": acc.get("horizon_hit_rate"),
        "model_s_episode_median": _stat(acc, "response_s_per_episode", "median"),
        "model_s_episode_p95": _stat(acc, "response_s_per_episode", "p95"),
        "model_s_total": _stat(acc, "response_s_per_episode", "total"),
        "runtime_s": record.get("runtime_s"),
    }


def load_rows(dirs: Sequence[Path]) -> list[dict[str, Any]]:
    rows = [cost_row(json.loads(p.read_text(encoding="utf-8"))) for d in dirs for p in sorted((d / "runs").glob("*.json"))]
    return sorted(rows, key=lambda r: (-r["score"], r["model_id"]))


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Cost sheet",
        "",
        "From the `emberbench run` records. USD is the cost the API reported (`cost_basis`).",
        "Model time is the measured response time; runtime is the last run segment only.",
        "",
        "| Model | Score | Episodes | USD / suite | USD / episode (median, p95) | Calls / episode (median, p95) "
        "| Tokens in / out per episode (mean) | Re-ask rate | Refusal rate | Model s / episode (median) | Model time (h) |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        hours = None if r["model_s_total"] is None else r["model_s_total"] / 3600
        lines.append(
            f"| `{r['model_id']}` | {_fmt(r['score'], 2)} | {_fmt(r['episodes'], 0)} | {_fmt(r['usd_suite'], 3)} "
            f"| {_fmt(r['usd_episode_median'], 4)}, {_fmt(r['usd_episode_p95'], 4)} | {_fmt(r['calls_episode_median'], 0)}, {_fmt(r['calls_episode_p95'], 0)} "
            f"| {_fmt(r['input_tokens_episode_mean'], 0)} / {_fmt(r['output_tokens_episode_mean'], 0)} | {_fmt(r['reask_rate'], 4)} "
            f"| {_fmt(r['refusal_rate'], 4)} | {_fmt(r['model_s_episode_median'], 1)} | {_fmt(hours, 2)} |"
        )
    return "\n".join(lines) + "\n"


def write(rows: Sequence[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost.json").write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with (out / "cost.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "COST.md").write_text(markdown(rows), encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", action="append", required=True, help="an `emberbench run --out` directory (repeatable)")
    parser.add_argument("--out", required=True, help="output directory (COST.md, cost.csv, cost.json)")
    args = parser.parse_args(argv)
    rows = load_rows([Path(d) for d in args.runs])
    if not rows:
        raise SystemExit("no run records found under runs/*.json")
    write(rows, Path(args.out))
    print(markdown(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
