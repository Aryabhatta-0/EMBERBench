"""Validity study of one environment pack: score uncertainty, variance sources, rank stability, power.

Each ``--results`` is a directory with ``results/*.json`` (an ``emberbench run`` output
directory; one result per agent). Every result is scored with the official
``score_result`` against ``--pack``; the analysis then uses the per-episode scores
of the score reports (``ScoreReport.episode_scores``) and the official hierarchy
(primary family -> scenario -> episodes, each level an unweighted mean).

It reports, per agent and for the pack:

- the official score and its 95% CI (from the score report);
- variance shares of the episode scores (eta squared) by primary family,
  scenario, stratum (``--strata``), population tertile and hazard tertile, and
  the within-scenario (seed / world) share;
- rank stability under a paired two-stage bootstrap (families fixed, scenarios
  resampled within family, episodes within scenario; the same draws for every
  agent): the probability of the point ranking and of each adjacent pair order;
- the expected-order check (``--expected-order``, for the baselines);
- ablations: the ranking without one family (or one stratum) at a time;
- power flags: adjacent pairs whose paired 95% interval contains 0, and metrics
  (family scores, safety, quality) whose between-agent spread is below twice
  their median bootstrap standard error;
- with ``--reference`` (the JSON of another analysis, for example the public
  development set), Kendall tau-b over the common agents. This is supporting
  evidence only: two suites with different families need not rank alike.

``--private`` writes no scenario ID, seed or stratum name; use it for a hidden
batch and keep the output with the evaluator. This is an analysis tool: it
changes no score and no official statistic.

Usage::

    python tools/experiments/validity/analyze.py --pack PACK_DIR --results RUN_DIR [--results RUN_DIR ...] \
        [--strata strata.json] [--expected-order a,b,c] [--reference other/validity.json] [--private] --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from emberbench import load_environment_pack, load_evaluation_result, score_result
from emberbench.environment_pack import EnvironmentPack
from emberbench.scoring import ScoreReport
from emberbench.statistics import check_resamples

# family -> scenario -> episode indices (the official layout, in suite order)
Layout = dict[str, dict[str, list[int]]]


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall tau-b of two paired score lists; ``None`` below two items or without variation."""
    conc = disc = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
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


def canonical_id(agent: str) -> str:
    """Runs before harness 1.0 labelled the canonical protocol with a ``-full-joint`` suffix; harness 1.0 IDs have none."""
    return agent.removesuffix("-full-joint")


def layout_of(pack: EnvironmentPack, episodes: set[int]) -> Layout:
    layout: Layout = {}
    for entry in pack.suite.manifest.entries:
        indices = [s.index for s in pack.manifest.episodes if s.scenario_id == entry.scenario_id and s.index in episodes]
        if indices:
            layout.setdefault(entry.primary_family.value, {})[entry.scenario_id] = indices
    return layout


def headline(layout: Layout, scores: Mapping[int, float]) -> float:
    fam = [statistics.fmean(statistics.fmean(scores[i] for i in idx) for idx in scen.values()) for scen in layout.values()]
    return statistics.fmean(fam)


def family_means(layout: Layout, scores: Mapping[int, float]) -> dict[str, float]:
    return {f: statistics.fmean(statistics.fmean(scores[i] for i in idx) for idx in scen.values()) for f, scen in layout.items()}


def eta_squared(values: Mapping[int, float], groups: Mapping[int, str]) -> float | None:
    """Between-group share of the variance of ``values`` (0..1); ``None`` without variance."""
    xs = list(values.values())
    mean = statistics.fmean(xs)
    total = sum((x - mean) ** 2 for x in xs)
    if total == 0:
        return None
    by: dict[str, list[float]] = defaultdict(list)
    for i, v in values.items():
        by[groups[i]].append(v)
    between = sum(len(g) * (statistics.fmean(g) - mean) ** 2 for g in by.values())
    return between / total


def tertiles(feature: Mapping[str, float]) -> dict[str, str]:
    ordered = sorted(set(feature.values()))
    if len(ordered) < 3:
        return {k: f"v{ordered.index(v)}" for k, v in feature.items()}
    lo, hi = ordered[len(ordered) // 3], ordered[(2 * len(ordered)) // 3]
    return {k: "low" if v < lo else ("high" if v >= hi else "mid") for k, v in feature.items()}


def resample_layout(layout: Layout, rng: random.Random) -> list[list[list[int]]]:
    """One two-stage draw: per family, scenarios with replacement, then episodes within each scenario."""
    draw = []
    for scen in layout.values():
        ids = list(scen)
        picked = [scen[rng.choice(ids)] for _ in ids]
        draw.append([[rng.choice(idx) for _ in idx] for idx in picked])
    return draw


def draw_score(draw: list[list[list[int]]], scores: Mapping[int, float]) -> float:
    return statistics.fmean(statistics.fmean(statistics.fmean(scores[i] for i in eps) for eps in fam) for fam in draw)


def ranking(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda a: (-scores[a], a))


def interval(xs: Sequence[float]) -> tuple[float, float]:
    s = sorted(xs)
    return s[int(0.025 * (len(s) - 1))], s[int(0.975 * (len(s) - 1))]


def load_reports(pack: EnvironmentPack, dirs: Sequence[Path], resamples: int) -> dict[str, ScoreReport]:
    reports: dict[str, ScoreReport] = {}
    for d in dirs:
        for path in sorted((d / "results").glob("*.json")):
            report = score_result(load_evaluation_result(path, pack.suite), pack, resamples=resamples)
            if report.model_id in reports:
                raise SystemExit(f"two results for {report.model_id}")
            reports[report.model_id] = report
    if len(reports) < 2:
        raise SystemExit("need at least two agents")
    return reports


def analyze(pack: EnvironmentPack, reports: Mapping[str, ScoreReport], args: argparse.Namespace) -> dict[str, Any]:
    episode = {a: {i: float(s) for i, s in r.episode_scores} for a, r in reports.items()}
    common = set.intersection(*(set(e) for e in episode.values()))
    layout = layout_of(pack, common)
    agents = sorted(reports)
    point = {a: float(s) for a in agents if (s := reports[a].score) is not None}
    order = ranking(point)

    scenario_of = {i: sid for scen in layout.values() for sid, idx in scen.items() for i in idx}
    family_of = {i: f for f, scen in layout.items() for idx in scen.values() for i in idx}
    groups: dict[str, dict[int, str]] = {"family": family_of, "scenario": scenario_of}
    scenarios = {sid: pack.suite.get(sid) for sid in set(scenario_of.values())}
    population = tertiles({sid: float(len(s.people)) for sid, s in scenarios.items()})
    hazard = tertiles({sid: len(s.ignitions) / s.fire.spread_interval for sid, s in scenarios.items()})
    groups["population"] = {i: population[s] for i, s in scenario_of.items()}
    groups["hazard"] = {i: hazard[s] for i, s in scenario_of.items()}
    strata: dict[str, str] = json.loads(Path(args.strata).read_text(encoding="utf-8")) if args.strata else {}
    if strata:
        missing = sorted(set(scenario_of.values()) - set(strata))
        if missing:
            raise SystemExit(f"--strata lacks {len(missing)} scenario(s)")
        groups["stratum"] = {i: strata[s] for i, s in scenario_of.items()}

    # With one episode per scenario, scenario and seed (world) variance cannot be separated.
    repeated = any(len(idx) > 1 for scen in layout.values() for idx in scen.values())
    variance: dict[str, dict[str, float | None]] = {}
    for a in agents:
        values = {i: episode[a][i] for i in common}
        shares = {name: eta_squared(values, g) for name, g in groups.items()}
        scen = shares["scenario"]
        if not repeated:
            shares["scenario"] = scen = None
        shares["within_scenario_seed"] = None if scen is None else 1 - scen
        variance[a] = {k: None if v is None else round(v, 4) for k, v in shares.items()}

    # Paired two-stage bootstrap (same draws for every agent).
    rng = random.Random(f"emberbench-validity/{pack.digest}")
    boot: dict[str, list[float]] = {a: [] for a in agents}
    fam_boot: dict[str, dict[str, list[float]]] = {a: defaultdict(list) for a in agents}
    same_order = 0
    for _ in range(args.resamples):
        draw = resample_layout(layout, rng)
        scores = {a: draw_score(draw, episode[a]) for a in agents}
        for a in agents:
            boot[a].append(scores[a])
            for f, fam_draw in zip(layout, draw):
                fam_boot[a][f].append(statistics.fmean(statistics.fmean(episode[a][i] for i in eps) for eps in fam_draw))
        same_order += ranking(scores) == order
    pairs = []
    for hi, lo in zip(order, order[1:]):
        deltas = [x - y for x, y in zip(boot[hi], boot[lo])]
        low, high = interval(deltas)
        pairs.append({"higher": hi, "lower": lo, "delta": round(point[hi] - point[lo], 3),
                      "ci95": [round(low, 3), round(high, 3)], "p_order_holds": round(sum(d > 0 for d in deltas) / len(deltas), 4),
                      "underpowered": low <= 0 <= high})  # fmt: skip

    expected = None
    if args.expected_order:
        ids = args.expected_order.split(",")
        absent = [x for x in ids if x not in point]
        if absent:
            raise SystemExit(f"--expected-order agents without a result: {absent}")
        holds = all(point[a] < point[b] for a, b in zip(ids, ids[1:]))
        p_boot = sum(all(boot[a][k] < boot[b][k] for a, b in zip(ids, ids[1:])) for k in range(args.resamples)) / args.resamples
        expected = {"order": ids, "holds": holds, "p_bootstrap": round(p_boot, 4)}

    # Ablations: drop one family (or stratum) and rescore the ranking.
    ablations = []
    for f in layout:
        sub = {g: s for g, s in layout.items() if g != f}
        if sub:
            sc = {a: headline(sub, episode[a]) for a in agents}
            ablations.append({"without": "family", "group": f, "tau_vs_full": kendall_tau([point[a] for a in agents], [sc[a] for a in agents]),
                              "top_agent_unchanged": ranking(sc)[0] == order[0]})  # fmt: skip
    if strata:
        for k, st in enumerate(sorted(set(strata[s] for s in scenario_of.values()))):
            keep = {i for i in common if strata[scenario_of[i]] != st}
            sub = layout_of(pack, keep)
            if sub:
                sc = {a: headline(sub, episode[a]) for a in agents}
                ablations.append({"without": "stratum", "group": f"stratum-{k}" if args.private else st,
                                  "tau_vs_full": kendall_tau([point[a] for a in agents], [sc[a] for a in agents]),
                                  "top_agent_unchanged": ranking(sc)[0] == order[0]})  # fmt: skip

    # Power flags per metric: between-agent spread against the median bootstrap standard error.
    power = []
    metric_point = {"score": point}
    metric_se = {"score": {a: statistics.stdev(boot[a]) for a in agents}}
    for f in layout:
        metric_point[f"family:{f}"] = {a: statistics.fmean(fam_boot[a][f]) for a in agents}
        metric_se[f"family:{f}"] = {a: statistics.stdev(fam_boot[a][f]) for a in agents}
    for name, pts in metric_point.items():
        spread = statistics.stdev(pts.values()) if len(pts) > 1 else 0.0
        se = statistics.median(metric_se[name].values())
        power.append({"metric": name, "between_agent_sd": round(spread, 3), "median_bootstrap_se": round(se, 3),
                      "scenarios": len(layout[name.split(":", 1)[1]]) if name.startswith("family:") else len(set(scenario_of.values())),
                      "too_little_power": spread < 2 * se})  # fmt: skip

    tau = None
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        ref_scores = {canonical_id(r["agent"]): r["score"] for r in ref["agents"]}
        both = [a for a in agents if canonical_id(a) in ref_scores and a in point]
        tau = {"reference": ref["pack_digest"], "agents": len(both),
               "kendall_tau_b": kendall_tau([point[a] for a in both], [ref_scores[canonical_id(a)] for a in both]),
               "note": "supporting evidence only"}  # fmt: skip

    return {
        "tool": "emberbench-validity/1.0",
        "pack_digest": pack.digest,
        "pack_id": None if args.private else pack.pack_id,
        "episodes": len(common),
        "families": {f: len(s) for f, s in layout.items()},
        "resamples": args.resamples,
        "agents": [{"agent": a, "score": round(point[a], 3), "ci95": None if (ci := reports[a].ci) is None else [round(float(ci.lower), 3), round(float(ci.upper), 3)],
                    "bootstrap_se": round(statistics.stdev(boot[a]), 3), "safety": None if (sf := reports[a].safety) is None else round(float(sf), 4),
                    "quality": None if (q := reports[a].quality) is None else round(float(q), 4),
                    "families": {f: round(v, 3) for f, v in family_means(layout, episode[a]).items()}} for a in order],  # fmt: skip
        "ranking": order,
        "rank_stability": {"p_point_ranking": round(same_order / args.resamples, 4), "adjacent_pairs": pairs},
        "variance_shares": variance,
        "expected_order": expected,
        "ablations": ablations,
        "power": power,
        "reference_tau": tau,
    }


def markdown(rec: dict[str, Any]) -> str:
    lines = [f"# Validity analysis: `{rec['pack_id'] or 'private pack'}`", "",
             f"Pack digest `{rec['pack_digest'][:16]}`; {rec['episodes']} episodes; families {rec['families']}; {rec['resamples']} paired bootstrap resamples.", "",
             "| Agent | Score | 95% CI | Bootstrap SE | Safety | Quality |", "|---|---:|---|---:|---:|---:|"]  # fmt: skip
    for a in rec["agents"]:
        ci = "-" if a["ci95"] is None else f"{a['ci95'][0]} to {a['ci95'][1]}"
        lines.append(f"| `{a['agent']}` | {a['score']} | {ci} | {a['bootstrap_se']} | {a['safety']} | {a['quality']} |")
    rs = rec["rank_stability"]
    lines += ["", f"P(point ranking) = {rs['p_point_ranking']}", "", "| Higher | Lower | Delta | 95% CI | P(order) | Underpowered |", "|---|---|---:|---|---:|---|"]
    lines += [f"| `{p['higher']}` | `{p['lower']}` | {p['delta']} | {p['ci95'][0]} to {p['ci95'][1]} | {p['p_order_holds']} | {p['underpowered']} |" for p in rs["adjacent_pairs"]]
    names = sorted({k for v in rec["variance_shares"].values() for k in v})
    lines += ["", "Variance shares of the episode scores (eta squared):", "", "| Agent | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    lines += [f"| `{a}` | " + " | ".join("-" if v.get(n) is None else str(v[n]) for n in names) + " |" for a, v in rec["variance_shares"].items()]
    if rec["expected_order"]:
        e = rec["expected_order"]
        lines += ["", f"Expected order {' < '.join(e['order'])}: holds {e['holds']}, bootstrap P {e['p_bootstrap']}."]
    lines += ["", "| Ablation | Group | Tau vs full | Top agent unchanged |", "|---|---|---:|---|"]
    lines += [f"| {x['without']} | {x['group']} | {'-' if x['tau_vs_full'] is None else round(x['tau_vs_full'], 3)} | {x['top_agent_unchanged']} |" for x in rec["ablations"]]
    lines += ["", "| Metric | Between-agent SD | Median bootstrap SE | Scenarios | Too little power |", "|---|---:|---:|---:|---|"]
    lines += [f"| {p['metric']} | {p['between_agent_sd']} | {p['median_bootstrap_se']} | {p['scenarios']} | {p['too_little_power']} |" for p in rec["power"]]
    if rec["reference_tau"]:
        t = rec["reference_tau"]
        lines += ["", f"Kendall tau-b against reference `{t['reference'][:16]}` over {t['agents']} agents: {t['kendall_tau_b']} ({t['note']})."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--strata", help="JSON object: scenario ID -> stratum")
    parser.add_argument("--expected-order", help="comma-separated agent IDs, weakest first")
    parser.add_argument("--reference", help="validity.json of another pack, for Kendall tau")
    parser.add_argument("--private", action="store_true", help="write no scenario ID, seed or stratum name")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        check_resamples(args.resamples)
    except ValueError as exc:
        parser.error(f"--resamples: {exc}")
    pack = load_environment_pack(args.pack)
    record = analyze(pack, load_reports(pack, args.results, min(args.resamples, 2000)), args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "validity.md").write_text(markdown(record), encoding="utf-8")
    print(f"wrote {args.out / 'validity.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
