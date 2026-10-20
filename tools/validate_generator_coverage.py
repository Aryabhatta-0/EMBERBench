"""Phase 11 generator coverage validation on a DEVELOPMENT candidate pool.

Runs the library composition pipeline (emberbench.composition) end to end with
public development seeds and reports whether the generator recipes fill every
family x difficulty x fire-model cell of the frozen V1 design
(docs/evaluation-suite-design.md section 15, "Generator coverage validation").

    python tools/validate_generator_coverage.py run --out <dir outside the repo>
    python tools/validate_generator_coverage.py report --out <same dir>

``run`` generates, measures, calibrates, tags and selects the pool, freezes a
development suite into ``<out>/development-suite`` and checks its unique-world
schedule; it saves the measured pool to ``<out>/pool.pickle`` (a local cache
for ``report``; never commit it). ``report`` (also run at the end of ``run``)
writes ``<out>/generator-coverage.md``.

Every seed and key here is derived from a fixed PUBLIC label. This pool and
the development suite are therefore public development artifacts: they show
that the recipes cover the design, and they must never be used as, or called,
the hidden evaluation suite. A live composition uses private seeds and a
private key with exactly the same code.
"""

import argparse
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from emberbench import load_suite
from emberbench.calibration import ADMITTED, BANDS, EXCLUSIONS
from emberbench.composition import (
    INITIAL_POOL_SIZE,
    INSUFFICIENT_WORLDS,
    V1_DESIGN,
    V1_INITIAL_PLAN,
    V1_RECIPES,
    CompositionResult,
    MeasuredCandidate,
    compose_pool,
    compose_suite,
    eligible,
    freeze_composed_suite,
    rule_set_digest,
    schedule_composed_worlds,
    select_suite,
    versions,
)
from emberbench.families import PRIMARY_FAMILIES
from emberbench.suite import ScenarioFamily

COVERAGE_VALIDATION_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"
DEVELOPMENT_DOMAIN = "emberbench/generator-coverage/development/"
EVALUATION_SEED_COUNT = 64
SUITE_ID = "development-composed-v1"
SUMMARY_NAME = "generator-coverage-summary.json"
F = ScenarioFamily
OVERLAP_PAIRS: tuple[tuple[ScenarioFamily, ScenarioFamily], ...] = (
    (F.CROWD_100, F.SPLIT_FLOW_ACROSS_EXITS),
    (F.THREATENED_NEAREST_EXIT, F.ROUTE_INVALIDATION_REPLANNING),
    (F.MAZE_OBSTACLE_ROUTING, F.DECEPTIVE_NEAREST_EXIT),
)


def development_seed(label: str) -> int:
    """A PUBLIC development seed from a fixed label. Never a hidden-suite seed."""
    return int.from_bytes(hashlib.sha256((DEVELOPMENT_DOMAIN + label).encode("ascii")).digest()[:8], "big")


def development_inputs() -> tuple[int, list[int], str]:
    pool_seed = development_seed("pool")
    seeds = [development_seed(f"evaluation-seed/{i}") for i in range(EVALUATION_SEED_COUNT)]
    return pool_seed, seeds, DEVELOPMENT_DOMAIN + "selection-key"


def run(out: Path, workers: int, log: Callable[[str], None]) -> None:
    if out.resolve().is_relative_to(REPO_ROOT):
        raise SystemExit(f"--out must be outside the repository ({REPO_ROOT})")
    out.mkdir(parents=True, exist_ok=True)
    pool_seed, seeds, key = development_inputs()
    public = load_suite(PUBLIC_SUITE_DIR)
    t0 = time.perf_counter()
    result = compose_pool(
        pool_seed=pool_seed,
        evaluation_seeds=seeds,
        selection_key=key,
        exclude_hashes=[s.content_hash() for s in public.scenarios],
        workers=workers,
        log=log,
    )
    total = time.perf_counter() - t0
    with (out / "pool.pickle").open("wb") as fh:
        pickle.dump({"result": result, "total_seconds": total, "workers": workers}, fh)
    report(out, log)


def _freeze_check(out: Path, result: CompositionResult) -> list[str]:
    pool_seed, seeds, key = development_inputs()
    target = out / "development-suite"
    if target.exists():
        import shutil

        shutil.rmtree(target)
    composed = compose_suite(result, suite_id=SUITE_ID, selection_key=key)
    frozen = freeze_composed_suite(composed, target)
    manifest = schedule_composed_worlds(frozen, seeds)
    worlds = {(e.scenario_id, e.seed) for e in manifest.episodes}
    families = Counter(s.primary_family for s in composed.record.selected)
    return [
        f"- Development suite frozen and reloaded: {len(frozen.suite.scenarios)} scenarios, suite hash `{composed.manifest.suite_hash()[:16]}...`.",
        f"- Unique-world schedule from `build_unique_world_manifest`: {len(manifest.episodes)} episodes, {len(worlds)} distinct (scenario, seed) worlds; "
        f"every world digest equals the frozen calibrated world (checked by `schedule_composed_worlds`).",
        f"- Primary families: {len(families)} families, {min(families.values()) if families else 0}..{max(families.values()) if families else 0} scenarios each; "
        "every selected scenario has exactly one primary family (the cell it filled).",
        f"- Freeze record digest `{composed.record.digest()[:16]}...`, rule-set digest `{composed.record.rule_set_digest[:16]}...`.",
    ]


def _cell(c: MeasuredCandidate) -> str:
    return "stochastic" if c.stochastic else "deterministic"


def _subsample_fill(result: CompositionResult, key: str) -> list[tuple[float, int, int, int]]:
    """Selection on keyed uniform subsamples of the round-0 pool: (fraction, candidates, short slots, short cells)."""
    initial = [c for c in result.candidates if c.origin.batch_index < _round0_batches(result).get(c.origin.recipe_id, 0)]
    rows = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        cut = int(fraction * 2**64)
        sub = [c for c in initial if int.from_bytes(hashlib.sha256(f"emberbench/coverage-subsample/{c.scenario_hash}".encode()).digest()[:8], "big") < cut]
        sel = select_suite(sub, V1_DESIGN.quotas(), key)
        rows.append((fraction, len(sub), sum(n for _, n in sel.shortfall), len(sel.shortfall)))
    return rows


def _round0_batches(result: CompositionResult) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for b in result.rounds[0].batches:
        counts[b.recipe.recipe_id] = max(counts[b.recipe.recipe_id], b.batch_index + 1)
    return dict(counts)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def fmt(x: Any) -> str:
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]


def report(out: Path, log: Callable[[str], None]) -> None:
    with (out / "pool.pickle").open("rb") as fh:
        saved = pickle.load(fh)
    result: CompositionResult = saved["result"]
    _, _, key = development_inputs()
    cands = result.candidates
    n = len(cands)
    quotas = V1_DESIGN.quotas()
    sel = result.selection
    elig = [c for c in cands if eligible(c)]
    lines = [
        "# Generator coverage validation (development pool)",
        "",
        f"Validation tool version {COVERAGE_VALIDATION_VERSION}. **This is a DEVELOPMENT pool built from public seed labels "
        f"(`{DEVELOPMENT_DOMAIN}`). It is not, and must never be used as, the hidden evaluation suite.**",
        "",
        "## Versions",
        "",
        *[f"- {k}: `{v}`" for k, v in versions().items()],
        f"- rule-set digest: `{rule_set_digest()}`",
        "",
        "## Pool",
        "",
        f"- Candidates measured: {n} (initial plan {sum(c for _, c in V1_INITIAL_PLAN)}, protocol size {INITIAL_POOL_SIZE}); "
        f"rounds: {len(result.rounds)} (round 0 + {len(result.rounds) - 1} top-up rounds).",
        f"- Dropped before measurement: {dict(result.dropped) or 'none'} (duplicates by content hash; public-suite hashes).",
        f"- Evaluation candidate seeds: {EVALUATION_SEED_COUNT}; stochastic scenarios are measured on the 3 unique worlds that "
        "`build_unique_world_manifest` schedules.",
        "",
    ]
    for r in result.rounds[1:]:
        lines.append(f"- Top-up round {r.index}: " + ", ".join(f"{b.recipe.recipe_id} x{len(b.scenarios)}" for b in r.batches) + f"; short slots after: {sum(k for _, k in r.shortfall_after)}.")
    lines += ([""] if len(result.rounds) > 1 else []) + ["### Candidates per recipe", ""]
    rows: list[list[Any]] = []
    for recipe in V1_RECIPES:
        rc = [c for c in cands if c.origin.recipe_id == recipe.recipe_id]
        adm = Counter(c.admission for c in rc)
        rows.append([recipe.recipe_id, "stochastic" if recipe.stochastic else "deterministic", len(rc), adm[ADMITTED], sum(1 for c in rc if eligible(c)),
                     adm["capacity_artifact"], adm["saturated"], adm["floor"], adm["no_separation"], adm["not_applicable"] + adm[INSUFFICIENT_WORLDS]])
    lines += _table(["recipe", "fire", "candidates", "admitted", "eligible", "capacity", "saturated", "floor", "no sep.", "n/a + insuff."], rows)
    adm = Counter(c.admission for c in cands)
    lines += ["", "### Admission outcomes (whole pool)", ""]
    lines += _table(["outcome", "count", "rate"], [[k, adm[k], adm[k] / n] for k in (ADMITTED, *EXCLUSIONS, INSUFFICIENT_WORLDS)])
    det = [c for c in cands if not c.stochastic]
    sto = [c for c in cands if c.stochastic]
    lines += [
        "",
        f"- Deterministic candidates: {len(det)} ({sum(1 for c in det if eligible(c))} eligible); stochastic: {len(sto)} ({sum(1 for c in sto if eligible(c))} eligible).",
        f"- Acceptance rate (admitted and at least one primary family): {len(elig) / n:.3f}.",
        f"- Capacity-artifact rejection rate: {adm['capacity_artifact'] / n:.3f} of all candidates "
        f"({adm['capacity_artifact'] / max(1, n - adm['not_applicable'] - adm[INSUFFICIENT_WORLDS]):.3f} of applicable ones).",
        "",
        "### Eligible candidates per family and band",
        "",
        "A candidate counts for every primary family it carries; selection gives it exactly one.",
        "",
    ]
    rows = []
    for f in PRIMARY_FAMILIES:
        row: list[Any] = [f.value]
        for b in BANDS:
            d = sum(1 for c in elig if f in c.tags and c.band == b and not c.stochastic)
            s = sum(1 for c in elig if f in c.tags and c.band == b and c.stochastic)
            row.append(f"{d}+{s}")
        row.append(sum(1 for c in elig if f in c.tags))
        rows.append(row)
    lines += _table(["family", *[f"{b} (det+sto)" for b in BANDS], "total"], rows)
    band_counts = Counter(c.band for c in elig)
    lines += ["", "Eligible per band: " + ", ".join(f"{b} {band_counts[b]}" for b in BANDS) + "."]
    ratios = []
    for cell, q in quotas.items():
        f, b, st = cell
        have = sum(1 for c in elig if f in c.tags and c.band == b and c.stochastic == st)
        ratios.append((have / q, have, q, cell))
    ratios.sort(key=lambda r: (r[0], r[3][0].value, BANDS.index(r[3][1]), r[3][2]))
    lines += ["", "### Rarest quota cells (eligible / quota)", ""]
    lines += _table(["family", "band", "fire", "eligible", "quota", "ratio"], [[c[0].value, c[1], "stochastic" if c[2] else "deterministic", h, q, r] for r, h, q, c in ratios[:12]])
    filled = Counter((c[0], c[1]) for c, _ in sel.chosen)
    cells40 = [(f, b) for f in PRIMARY_FAMILIES for b in BANDS]
    lines += [
        "",
        "## Selection",
        "",
        f"- Quota cells (family x band x fire model): {len(quotas)}; filled completely: {len(quotas) - len(sel.shortfall)}; short slots: {sum(k for _, k in sel.shortfall)}.",
        f"- Family x difficulty cells filled to quota: {sum(1 for fb in cells40 if filled[fb] == dict(V1_DESIGN.bands)[fb[1]])} of {len(cells40)}.",
        f"- Selected: {len(sel.chosen)} scenarios ({sum(1 for _, c in sel.chosen if c.stochastic)} stochastic).",
    ]
    if sel.shortfall:
        lines += ["- Shortfall: " + ", ".join(f"{f.value}/{b}/{'sto' if st else 'det'} {k}" for (f, b, st), k in sel.shortfall) + "."]
    lines += ["", "### Pool size needed (keyed subsamples of the initial pool)", ""]
    lines += _table(["fraction", "candidates", "short slots", "short cells"], _subsample_fill(result, key))
    lines += ["", "## Overlap audit", ""]
    rows = []
    for a, b in OVERLAP_PAIRS:
        ea = {c.scenario_hash for c in elig if a in c.tags}
        eb = {c.scenario_hash for c in elig if b in c.tags}
        sa = [c for cell, c in sel.chosen if cell[0] is a]
        sb = [c for cell, c in sel.chosen if cell[0] is b]
        rows.append([f"{a.value} / {b.value}", len(ea), len(eb), len(ea & eb), len(ea & eb) / max(1, len(ea | eb)),
                     f"{sum(1 for c in sa if b in c.tags)}/{len(sa)}", f"{sum(1 for c in sb if a in c.tags)}/{len(sb)}"])
    lines += _table(["pair A / B", "eligible A", "eligible B", "both", "Jaccard", "selected A also B", "selected B also A"], rows)
    medium_split = sum(1 for c in elig if F.SPLIT_FLOW_ACROSS_EXITS in c.tags and len(c.scenario.people) < 100)
    sel_split = [c for cell, c in sel.chosen if cell[0] is F.SPLIT_FLOW_ACROSS_EXITS]
    lines += [
        "",
        f"- Split flow below 100 people: {medium_split} eligible candidates; selected split-flow populations: "
        f"{sorted(len(c.scenario.people) for c in sel_split)}.",
        f"- Selected scenarios with more than one primary family: {len(sel.chosen) - len({c.scenario_hash for _, c in sel.chosen})} "
        "(each candidate fills exactly one cell).",
    ]
    fire_dep = (F.THREATENED_NEAREST_EXIT, F.ROUTE_INVALIDATION_REPLANNING)
    lines += ["", "### Fire-dependent tags", ""]
    for f in fire_dep:
        d = sum(1 for c in elig if f in c.tags and not c.stochastic)
        s = sum(1 for c in elig if f in c.tags and c.stochastic)
        lines.append(f"- {f.value}: {d} deterministic (one trajectory) and {s} stochastic (mean over the 3 scheduled unique worlds) eligible candidates.")
    secs = result.seconds
    groups: dict[str, list[float]] = {}
    for c in cands:
        size = "crowd_100" if len(c.scenario.people) >= 100 else "small"
        groups.setdefault(f"{size} {_cell(c)}", []).append(c.seconds)
    lines += ["", "## Runtime", ""]
    lines += [
        f"- Workers: {saved['workers']}; total wall clock: {saved['total_seconds'] / 60:.1f} min.",
        f"- Generation: {secs['generation_wall']:.1f} s wall, {secs['generation_cpu']:.1f} s CPU.",
        f"- Measurement (unique worlds, panel runs, official scoring, tags): {secs['measurement_wall']:.1f} s wall, {secs['measurement_cpu']:.1f} s CPU "
        f"({secs['measurement_cpu'] / n:.2f} s per candidate).",
        "",
    ]
    lines += _table(["candidate kind", "count", "mean s", "max s"], [[k, len(v), statistics.fmean(v), max(v)] for k, v in sorted(groups.items())])
    per = secs["measurement_cpu"] / n
    lines += [
        "",
        f"- Estimate for a 4,000-candidate pool with this recipe mix: {4000 * per / 3600:.1f} CPU hours, "
        f"{4000 * per / max(1, saved['workers']) / 60:.0f} min on {saved['workers']} workers.",
        "",
        "## Frozen development suite",
        "",
    ]
    if sel.complete:
        lines += _freeze_check(out, result)
    else:
        lines.append("- Not frozen: the selection is incomplete.")
    text = "\n".join(lines) + "\n"
    (out / "generator-coverage.md").write_text(text, encoding="utf-8", newline="\n")
    log(f"wrote {out / 'generator-coverage.md'}")
    (out / SUMMARY_NAME).write_text(json.dumps(summary(result), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    log(f"wrote {out / SUMMARY_NAME} (copy it to tests/fixtures/ to pin the coverage evidence)")


def summary(result: CompositionResult) -> dict[str, Any]:
    """Counts only (no scenario, seed or hash of a candidate): the coverage evidence pinned by tests/test_composition.py."""
    quotas = V1_DESIGN.quotas()
    elig = [c for c in result.candidates if eligible(c)]
    cells = []
    for (f, b, st), q in sorted(quotas.items(), key=lambda kv: (kv[0][0].value, BANDS.index(kv[0][1]), kv[0][2])):
        have = sum(1 for c in elig if f in c.tags and c.band == b and c.stochastic == st)
        got = sum(1 for cell, _ in result.selection.chosen if cell == (f, b, st))
        cells.append({"family": f.value, "band": b, "stochastic": st, "quota": q, "eligible": have, "selected": got})
    return {
        "coverage_validation_version": COVERAGE_VALIDATION_VERSION,
        "rule_set_digest": rule_set_digest(),
        "candidates": len(result.candidates),
        "rounds": len(result.rounds),
        "selected": len(result.selection.chosen),
        "shortfall_slots": sum(n for _, n in result.selection.shortfall),
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--out", type=Path, required=True, help="output directory (keep it outside the repository)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.command == "run":
        run(args.out, args.workers, log)
    else:
        report(args.out, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
