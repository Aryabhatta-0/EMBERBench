"""Compare harness protocols (full joint vs fixed chunks) per model: score, ranking, cost, validity.

Each ``--run`` is a finished ``run_remote.py`` output directory (``environment-pack/``,
``results/``, ``accounting/``). Runs of the same model under different protocols
are paired on identical worlds with the official ``compare_scores``. The output is
one JSON record and a Markdown table; no number is recomputed from summaries.

Usage::

    python tools/experiments/protocol/compare_protocols.py --run ../p1/jev/full_joint --run ../p1/jev/fixed_chunk ... --out ../p1/compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from emberbench import compare_scores, load_environment_pack, load_evaluation_result, score_result
from emberbench.scoring import ScoreReport

PROTOCOL_SUFFIX = re.compile(r"-(full-joint|chunk\d+)$")


def _f(v: Any, d: int = 2) -> str:
    return "-" if v is None else f"{float(v):.{d}f}"


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall tau-b of two paired score lists (ties handled); ``None`` below two items."""
    n = len(a)
    if n < 2:
        return None
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    denom = ((conc + disc + ta) * (conc + disc + tb)) ** 0.5
    return None if denom == 0 else (conc - disc) / denom


def load_run(run: Path, resamples: int) -> list[dict[str, Any]]:
    pack = load_environment_pack(run / "environment-pack")
    rows = []
    for path in sorted((run / "results").glob("*.json")):
        result = load_evaluation_result(path, pack.suite)
        report = score_result(result, pack, resamples=resamples)
        acc_path = run / "accounting" / f"{result.model_id}.json"
        acc = json.loads(acc_path.read_text(encoding="utf-8")) if acc_path.exists() else None
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").exists() else {}
        protocol = (summary.get("harness_protocol") or {})
        label = "full-joint" if protocol.get("joint_mode") == "full_joint" else f"chunk{protocol.get('chunk_size')}"
        rows.append({"run": str(run), "model_id": result.model_id, "model": PROTOCOL_SUFFIX.sub("", result.model_id), "protocol": label,
                     "report": report, "accounting": acc, "pack_digest": pack.digest})  # fmt: skip
    return rows


def describe(r: dict[str, Any]) -> dict[str, Any]:
    rep: ScoreReport = r["report"]
    acc = r["accounting"] or {}
    ci = rep.ci
    fam = {g.name: None if g.score is None else round(float(g.score), 2) for g in rep.families}
    return {
        "model": r["model"],
        "protocol": r["protocol"],
        "score": round(float(rep.score), 3) if rep.score is not None else None,
        "ci95": None if ci is None else [round(float(ci.lower), 3), round(float(ci.upper), 3)],
        "safety": round(float(rep.safety), 4) if rep.safety is not None else None,
        "quality": round(float(rep.quality), 4) if rep.quality is not None else None,
        "worst_scenario": None if rep.robustness.worst_scenario is None else round(float(rep.robustness.worst_scenario), 2),
        "families": fam,
        "calls_per_episode": acc.get("calls_per_episode"),
        "input_tokens_per_episode": acc.get("input_tokens_per_episode"),
        "output_tokens_per_episode": acc.get("output_tokens_per_episode"),
        "cost_per_episode": acc.get("cost_per_episode"),
        "suite_cost": acc.get("total_cost"),
        "response_s_per_episode": acc.get("response_s_per_episode"),
        "invalid_slot_rate": acc.get("invalid_slot_rate"),
        "reask_rate": acc.get("reask_rate"),
        "reask_cost_multiplier": acc.get("reask_cost_multiplier"),
        "horizon_hits": acc.get("horizon_hits"),
        "horizon_hit_rate": acc.get("horizon_hit_rate"),
        "probabilities": acc.get("probabilities"),
    }


def compare(rows: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["protocol"]] = r
    pairs = []
    for model, protos in sorted(by_model.items()):
        names = sorted(protos)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = protos[names[i]], protos[names[j]]
                if a["pack_digest"] != b["pack_digest"]:
                    continue
                cmp = compare_scores(a["report"], b["report"], resamples=resamples)
                pairs.append({
                    "model": model, "a": names[i], "b": names[j],
                    "delta": None if cmp.delta is None else round(float(cmp.delta), 3),
                    "delta_ci95": None if cmp.ci is None else [round(float(cmp.ci.lower), 3), round(float(cmp.ci.upper), 3)],
                    "outcome": cmp.outcome.value, "a_higher": cmp.a_higher, "b_higher": cmp.b_higher, "equal": cmp.equal,
                })  # fmt: skip
    protocols = sorted({r["protocol"] for r in rows})
    complete = [m for m, p in by_model.items() if all(x in p for x in protocols)]
    ranks: dict[str, Any] = {}
    if len(protocols) >= 2 and len(complete) >= 2:
        base = protocols[0]
        for other in protocols[1:]:
            xs = [float(by_model[m][base]["report"].score or 0) for m in complete]
            ys = [float(by_model[m][other]["report"].score or 0) for m in complete]
            order_a = sorted(complete, key=lambda m: -float(by_model[m][base]["report"].score or 0))
            order_b = sorted(complete, key=lambda m: -float(by_model[m][other]["report"].score or 0))
            ranks[f"{base} vs {other}"] = {"kendall_tau_b": kendall_tau(xs, ys), "order_" + base: order_a, "order_" + other: order_b}
    return {"runs": [describe(r) for r in rows], "paired": pairs, "ranking": ranks}


def markdown(record: dict[str, Any]) -> str:
    lines = [
        "| Model | Protocol | Score (95% CI) | Safety | Quality | Calls/ep med / p95 | In tok/ep med | Cost/ep med / p95 (USD) | Suite cost | Resp s/ep med / p95 | Invalid-slot | Re-ask | Horizon hits | p(choice) | Entropy (bits) | Margin |",
        "|---|---|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in record["runs"]:
        ci = r["ci95"]
        calls, tok, cost, resp = (r[k] or {} for k in ("calls_per_episode", "input_tokens_per_episode", "cost_per_episode", "response_s_per_episode"))
        p = r["probabilities"] or {}
        lines.append(
            f"| `{r['model']}` | {r['protocol']} | {_f(r['score'])}" + ("" if ci is None else f" ({_f(ci[0])}-{_f(ci[1])})")
            + f" | {_f(r['safety'], 3)} | {_f(r['quality'], 3)} | {_f(calls.get('median'), 0)} / {_f(calls.get('p95'), 0)}"
            + f" | {_f(tok.get('median'), 0)} | {_f(cost.get('median'), 4)} / {_f(cost.get('p95'), 4)} | {_f(r['suite_cost'], 3)}"
            + f" | {_f(resp.get('median'), 1)} / {_f(resp.get('p95'), 1)} | {_f(r['invalid_slot_rate'], 4)} | {_f(r['reask_rate'], 4)}"
            + f" | {r['horizon_hits']} | {_f(p.get('p_choice_mean'), 3)} | {_f(p.get('entropy_bits_mean'), 3)} | {_f(p.get('margin_mean'), 3)} |"
        )  # fmt: skip
    lines += ["", "| Model | A | B | Delta A-B | 95% CI | Outcome | Worlds A>B / B>A / equal |", "|---|---|---|---:|---|---|---|"]
    for p in record["paired"]:
        ci = p["delta_ci95"]
        lines.append(f"| `{p['model']}` | {p['a']} | {p['b']} | {_f(p['delta'])} | {'-' if ci is None else f'{_f(ci[0])} to {_f(ci[1])}'} | {p['outcome']} | {p['a_higher']} / {p['b_higher']} / {p['equal']} |")
    for name, r in record["ranking"].items():
        lines += ["", f"Ranking {name}: Kendall tau-b = {_f(r['kendall_tau_b'], 3)}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    rows = [r for run in args.run for r in load_run(run, args.resamples)]
    record = compare(rows, args.resamples)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol-comparison.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out / "protocol-comparison.md").write_text(markdown(record), encoding="utf-8", newline="\n")
    sys.stdout.write(markdown(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
