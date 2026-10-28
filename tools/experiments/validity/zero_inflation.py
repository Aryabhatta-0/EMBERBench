"""Zero-inflation diagnostic of official EMBER episode scores (research lane; changes no score).

For every agent with a score report under ``--results DIR/scores/*.json``:

- the share of applicable episodes that score 0, split by cause: the
  catastrophic gate, a declared failure (by kind) or a scored episode at 0;
- a histogram of the episode scores (0, then 10-point bins);
- Sarle's bimodality coefficient ``(g1^2 + 1) / (g2 + 3(n-1)^2 / ((n-2)(n-3)))``
  (above 5/9 suggests two modes, but strong skew alone can also pass it);
- the gate cliff: the scores of episodes whose avoidable loss rate
  ``(c + s) / M*`` is just below the gate (0.4 to 0.5). When they are near 0
  already, the gate removes almost nothing and adds no second mode.

Output names no scenario ID or seed, so it is safe for a hidden batch.

Usage::

    python tools/experiments/validity/zero_inflation.py --results DIR [--results DIR ...] --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BC_THRESHOLD = 5 / 9
BINS = [(10 * k, 10 * (k + 1)) for k in range(10)]


def bimodality_coefficient(xs: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 4:
        return None
    mean = statistics.fmean(xs)
    m2 = sum((x - mean) ** 2 for x in xs) / n
    if m2 == 0:
        return None
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    g1 = m3 / m2**1.5 * (n * (n - 1)) ** 0.5 / (n - 2)  # sample skewness
    g2 = ((n + 1) * (m4 / m2**2 - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))  # sample excess kurtosis
    return float(g1**2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def loss_rate(ep: dict[str, Any]) -> float | None:
    ev = ep.get("evidence") or {}
    m = ev.get("max_evacuated")
    if not m:
        return None
    return (len(ev.get("avoidable_casualty_ids", [])) + len(ev.get("avoidable_stranded_ids", []))) / float(m)


def diagnose(report: dict[str, Any]) -> dict[str, Any]:
    eps = [e for e in report["episodes"] if e["status"] != "not_applicable"]
    scores = [float(e["score"]) for e in eps]
    zero = [e for e in eps if float(e["score"]) == 0]
    causes: Counter[str] = Counter()
    for e in zero:
        causes["failed:" + str(e["failure"]) if e["status"] == "failed" else e["status"]] += 1
    hist = {"0": sum(1 for s in scores if s == 0)}
    for lo, hi in BINS:
        hist[f"({lo},{hi}]"] = sum(1 for s in scores if lo < s <= hi)
    near = [float(e["score"]) for e in eps if e["status"] == "scored" and (r := loss_rate(e)) is not None and 0.4 <= r < 0.5]
    positive = [s for s in scores if s > 0]
    bc = bimodality_coefficient(scores)
    return {
        "episodes": len(eps),
        "zero_share": round(len(zero) / len(eps), 4),
        "zero_causes": dict(sorted(causes.items())),
        "histogram": hist,
        "positive_median": None if not positive else round(statistics.median(positive), 3),
        "positive_quartiles": None if len(positive) < 4 else [round(q, 3) for q in statistics.quantiles(positive, n=4)],
        "bimodality_coefficient": None if bc is None else round(bc, 4),
        "bc_above_5_9": None if bc is None else bc > BC_THRESHOLD,
        "below_gate": {"episodes": len(near), "max_score": round(max(near), 3) if near else None,
                       "mean_score": round(statistics.fmean(near), 3) if near else None},  # fmt: skip
    }


def markdown(rec: dict[str, Any]) -> str:
    lines = ["# Zero-inflation diagnostic", "", "Research lane. Changes no score.", "",
             "| Agent | Episodes | Zero share | Zero causes | Positive median | BC | BC > 5/9 | Below-gate eps / max / mean |",
             "|---|---:|---:|---|---:|---:|---|---|"]  # fmt: skip
    for a, d in rec["agents"].items():
        causes = ", ".join(f"{k} {v}" for k, v in d["zero_causes"].items()) or "-"
        g = d["below_gate"]
        lines.append(f"| `{a}` | {d['episodes']} | {d['zero_share']} | {causes} | {d['positive_median']} | "
                     f"{d['bimodality_coefficient']} | {d['bc_above_5_9']} | {g['episodes']} / {g['max_score']} / {g['mean_score']} |")  # fmt: skip
    lines += ["", "## Histograms (episode counts)", "", "| Agent | " + " | ".join(next(iter(rec["agents"].values()))["histogram"]) + " |",
              "|---|" + "---:|" * 11]  # fmt: skip
    lines += [f"| `{a}` | " + " | ".join(str(v) for v in d["histogram"].values()) + " |" for a, d in rec["agents"].items()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", action="append", required=True, help="a run directory with scores/*.json (repeatable)")
    parser.add_argument("--out", required=True, help="output directory (validity-zero.json, validity-zero.md)")
    args = parser.parse_args(argv)
    agents: dict[str, Any] = {}
    for d in args.results:
        for path in sorted(Path(d, "scores").glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            agents[report["model_id"]] = diagnose(report)
    if not agents:
        raise SystemExit("no score reports found")
    agents = dict(sorted(agents.items(), key=lambda kv: kv[1]["zero_share"]))
    rec = {"agents": agents, "bc_threshold": round(BC_THRESHOLD, 4)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validity-zero.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out / "validity-zero.md").write_text(markdown(rec), encoding="utf-8", newline="\n")
    print(markdown(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
