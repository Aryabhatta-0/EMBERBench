"""Empirical sensitivity study for the EMBER Score (docs/benchmark-methodology.md, section 6).

Run from the repository root:

    python tools/scoring_sensitivity.py [--cache DIR]

It plays the reference baselines and a set of adversarial policies on

- the public V1 suite under the public development seeds 0, 1, 2, and
- a procedurally generated stochastic suite built from a public config and a
  constant, test-only generator seed (no evaluator secret is involved),

then scores every episode under the official constants, under rejected
candidate formulas and under a grid of perturbed constants. It prints the
tables used in the methodology document. Nothing here is part of the score:
the adversarial policies are study instruments, not baselines.

The privileged references are evaluator-side; no policy receives them.
"""

import argparse
import itertools
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from emberbench import (  # noqa: E402
    Action,
    GeneratorConfig,
    Observation,
    Suite,
    agent_policy,
    baseline_factory,
    generate_suite,
    load_suite,
    run_episode,
)
from emberbench.baselines import _descend, _exit_fields, _flee, _heat, _manhattan, _moves, _nearest, _target  # noqa: E402
from emberbench.grid import Pos, in_bounds  # noqa: E402
from emberbench.scoring import (  # noqa: E402
    OFFICIAL_CONSTANTS,
    QUALITY_COMPONENTS,
    EpisodeEvidence,
    ScoringConstants,
    episode_evidence,
    score_evidence,
)

Policy = Callable[[Observation], Mapping[str, Action]]

PUBLIC_SEEDS = (0, 1, 2)
GENERATED_SEEDS = (0, 1, 2)
STUDY_GENERATOR_SEED = 20261020  # test-only constant: this suite is public study material, never a scored suite
STUDY_CONFIG: dict[str, Any] = {
    "config_version": "1.0",
    "scenario_count": 16,
    "grid": {"width": 16, "height": 12},
    "population": {"min": 6, "max": 24},
    "exit_count": {"min": 1, "max": 3},
    "ignition_count": {"min": 1, "max": 2},
    "layout": {
        "partition_count": {"min": 0, "max": 2},
        "doors_per_partition": {"min": 1, "max": 2},
        "door_width": {"min": 1, "max": 2},
        "scatter_density": {"min": 0.0, "max": 0.08},
    },
    "fire": {"model": "stochastic", "spread_interval": {"min": 1, "max": 2}, "exits_flammable": True, "spread_probability_choices": [0.5, 0.75]},
    "movement": {"enter_fire": "casualty"},
    "max_steps": 80,
    "max_attempts": 400,
    "constraints": {
        "route_length": {"min": 1, "max": 80},
        "obstacle_density": {"min": 0.0, "max": 0.5},
        "crowd_density": {"min": 0.0, "max": 0.5},
        "exit_separation": {"min": 4, "max": 27},
        "passage_width": {"min": 1, "max": 16},
        "route_redundancy": {"min": 1, "max": 4},
        "ignition_person_distance": {"min": 2, "max": 192},
        "ignition_exit_distance": {"min": 3, "max": 192},
    },
}


# --------------------------------------------------------------------------- policies (study instruments)


def _stable(pid: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(pid))


def _assign(
    obs: Observation,
    fields: Sequence[tuple[str, Mapping[Pos, int]]],
    *,
    prefer_occupied: bool = False,
) -> dict[str, Action]:
    """Congestion-aware exit assignment (as congestion-aware-v1) over the given distance fields."""
    occupied = frozenset(p.position for p in obs.active)
    actions: dict[str, Action] = {}
    queue: list[tuple[int, str, Pos]] = []
    heat: dict[Pos, int] | None = None
    for person in obs.active:
        f = _nearest(fields, person.position)
        if f is None:
            heat = _heat(obs) if heat is None else heat
            actions[person.id] = _flee(obs, person.position, heat)
        else:
            queue.append((f[person.position], person.id, person.position))
    load = [0] * len(fields)
    inf = float("inf")
    for _d, pid, pos in sorted(queue):
        choice, best = 0, (inf, inf)
        for i, (_eid, f) in enumerate(fields):
            if pos in f:
                key = (max(f[pos], load[i] + 1), f[pos])
                if key < best:
                    choice, best = i, key
        load[choice] += 1
        f = fields[choice][1]
        downhill = [a for a in _moves(obs) if f.get(_target(pos, a)) == f[pos] - 1]
        free = [a for a in downhill if _target(pos, a) not in occupied]
        busy = [a for a in downhill if _target(pos, a) in occupied]
        order = (busy + free) if prefer_occupied else (free + busy)
        actions[pid] = order[0] if order else Action.STAY
    return actions


def _safe_fields(obs: Observation) -> tuple[tuple[str, dict[Pos, int]], ...]:
    return _exit_fields(obs, obs.map.walls | obs.burning, skip_burning=True)


def congestion(obs: Observation) -> dict[str, Action]:
    return _assign(obs, _safe_fields(obs))


def reckless_fast(obs: Observation) -> dict[str, Action]:
    """Load-balanced routing that ignores fire entirely: fast, and walks into fire."""
    return _assign(obs, _exit_fields(obs, obs.map.walls, skip_burning=False))


def abandon_far(obs: Observation) -> dict[str, Action]:
    """Congestion-aware, but the 15% of people farthest from an exit at step 0 are abandoned (always stay)."""
    acts = congestion(obs)
    fields = _safe_fields(obs)
    dist = sorted(((_nearest(fields, p.position) or {}).get(p.position, 10**6), p.id) for p in obs.active)
    if obs.step == 0:
        cut = max(1, len(dist) * 15 // 100)
        abandon_far.left = {pid for _d, pid in dist[-cut:]}  # type: ignore[attr-defined]
    for pid in getattr(abandon_far, "left", set()):
        if pid in acts:
            acts[pid] = Action.STAY
    return acts


def safe_slow(obs: Observation) -> dict[str, Action]:
    """Congestion-aware moves only every third step, except for people within distance 3 of fire."""
    acts = congestion(obs)
    if obs.step % 3 == 0:
        return acts
    for p in obs.active:
        near = min((_manhattan(p.position, f) for f in obs.burning), default=10**6)
        if near > 3:
            acts[p.id] = Action.STAY
    return acts


def late_start(obs: Observation) -> dict[str, Action]:
    """Waits 8 steps, then congestion-aware."""
    if obs.step < 8:
        return {p.id: Action.STAY for p in obs.active}
    return congestion(obs)


def jostler(obs: Observation) -> dict[str, Action]:
    """Congestion-aware routes, but always pushes into an occupied downhill cell when one exists."""
    return _assign(obs, _safe_fields(obs), prefer_occupied=True)


def herd(obs: Observation) -> dict[str, Action]:
    """Fire-aware, but everyone uses the one exit with the smallest total distance: a self-made bottleneck."""
    fields = _safe_fields(obs)
    best: Mapping[Pos, int] | None = None
    best_cost = float("inf")
    for _eid, f in fields:
        cost = sum(f.get(p.position, 10**4) for p in obs.active)
        if cost < best_cost:
            best, best_cost = f, cost
    acts: dict[str, Action] = {}
    heat: dict[Pos, int] | None = None
    for p in obs.active:
        if best is None or p.position not in best:
            heat = _heat(obs) if heat is None else heat
            acts[p.id] = _flee(obs, p.position, heat)
        else:
            acts[p.id] = _descend(obs, p.position, best)
    return acts


def wanderer(obs: Observation) -> dict[str, Action]:
    """Congestion-aware, but every fourth step a person far from fire takes a sideways or backward move."""
    acts = congestion(obs)
    fields = _safe_fields(obs)
    m = obs.map
    for p in obs.active:
        if (obs.step + _stable(p.id)) % 4:
            continue
        if min((_manhattan(p.position, f) for f in obs.burning), default=10**6) < 3:
            continue
        f = _nearest(fields, p.position)
        if f is None:
            continue
        for a in _moves(obs):
            nb = _target(p.position, a)
            if in_bounds(nb, m.width, m.height) and nb not in m.walls and nb not in obs.burning and nb in f and f[nb] >= f[p.position]:
                acts[p.id] = a
                break
    return acts


def hider(obs: Observation) -> dict[str, Action]:
    """Fire-aware for people within 6 steps of an exit; everyone else moves away from the fire and waits."""
    fields = _safe_fields(obs)
    acts = congestion(obs)
    heat: dict[Pos, int] | None = None
    for p in obs.active:
        f = _nearest(fields, p.position)
        if f is None or f[p.position] > 6:
            heat = _heat(obs) if heat is None else heat
            acts[p.id] = _flee(obs, p.position, heat)
    return acts


def always_stay(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def _baseline(name: str) -> Callable[[], Policy]:
    return lambda: agent_policy(baseline_factory(name)())


POLICIES: dict[str, Callable[[], Policy]] = {
    "always-stay": lambda: always_stay,
    "random-legal": _baseline("random-legal-v1"),
    "nearest-greedy": _baseline("nearest-exit-greedy-v1"),
    "static-shortest": _baseline("static-shortest-path-v1"),
    "fire-aware": _baseline("fire-aware-shortest-path-v1"),
    "congestion-aware": _baseline("congestion-aware-v1"),
    "adv-reckless-fast": lambda: reckless_fast,
    "adv-abandon-far": lambda: abandon_far,
    "adv-safe-slow": lambda: safe_slow,
    "adv-late-start": lambda: late_start,
    "adv-jostler": lambda: jostler,
    "adv-herd": lambda: herd,
    "adv-wanderer": lambda: wanderer,
    "adv-hider": lambda: hider,
}


# --------------------------------------------------------------------------- data


def study_suites(tmp: Path) -> dict[str, Suite]:
    public = load_suite(ROOT / "scenarios" / "public" / "v1")
    generated = generate_suite(GeneratorConfig.from_dict(STUDY_CONFIG), STUDY_GENERATOR_SEED, batch_id="study-batch")
    from emberbench.generator import freeze_generated_suite

    frozen_dir = tmp / "generated"
    if not frozen_dir.exists():
        freeze_generated_suite(generated, frozen_dir)
    return {"public": public, "generated": load_suite(frozen_dir / "suite")}


Row = tuple[str, str, str, int, EpisodeEvidence]  # (suite, family, scenario, seed, evidence)


def collect(suites: Mapping[str, Suite], cache: Path) -> dict[str, list[Row]]:
    cache.mkdir(parents=True, exist_ok=True)
    data: dict[str, list[Row]] = {}
    for name, make in POLICIES.items():
        path = cache / f"{name}.json"
        rows: list[Row] = []
        if path.exists():
            for r in json.loads(path.read_text()):
                rows.append((r[0], r[1], r[2], r[3], EpisodeEvidence(**r[4])))
        else:
            for suite_name, suite in suites.items():
                seeds = PUBLIC_SEEDS if suite_name == "public" else GENERATED_SEEDS
                for entry, scenario in zip(suite.manifest.entries, suite.scenarios, strict=True):
                    for seed in seeds:
                        rec = run_episode(scenario, make(), seed=seed)
                        rows.append((suite_name, entry.primary_family.value, scenario.scenario_id, seed, episode_evidence(scenario, rec)))
            path.write_text(json.dumps([[a, b, c, d, e.to_dict()] for a, b, c, d, e in rows]))
        data[name] = rows
        print(f"  collected {name}: {len(rows)} episodes", file=sys.stderr)
    return data


# --------------------------------------------------------------------------- aggregation (as emberbench.scoring)


def hierarchical(rows: Sequence[Row], values: Sequence[float | None]) -> float | None:
    """Episode -> scenario -> primary family -> overall unweighted means, skipping undefined values."""
    by_scen: dict[tuple[str, str], list[float]] = {}
    fam_of: dict[tuple[str, str], str] = {}
    for (suite, fam, scen, _seed, _ev), v in zip(rows, values, strict=True):
        if v is None:
            continue
        by_scen.setdefault((suite, scen), []).append(v)
        fam_of[(suite, scen)] = f"{suite}:{fam}"
    by_fam: dict[str, list[float]] = {}
    for key, vs in by_scen.items():
        by_fam.setdefault(fam_of[key], []).append(sum(vs) / len(vs))
    if not by_fam:
        return None
    fams = [sum(v) / len(v) for v in by_fam.values()]
    return sum(fams) / len(fams)


# --------------------------------------------------------------------------- candidate formulas


Formula = Callable[[EpisodeEvidence], float]


def official(constants: ScoringConstants = OFFICIAL_CONSTANTS) -> Formula:
    def f(ev: EpisodeEvidence) -> float:
        s = score_evidence(0, ev, constants)
        return float(s.score)

    return f


def _parts(ev: EpisodeEvidence) -> dict[str, float | None]:
    s = score_evidence(0, ev, OFFICIAL_CONSTANTS)
    return {k: (None if v is None else float(v)) for k, v in s.components}


def additive(ev: EpisodeEvidence) -> float:
    """REJECTED: 100 * weighted arithmetic mean of evacuated rate and the quality components."""
    if ev.max_evacuated == 0:
        return 0.0
    p = _parts(ev)
    terms = {"evac": (ev.evacuated / ev.max_evacuated, 0.5), **{k: (p[k], w) for k, w in zip(QUALITY_COMPONENTS, (0.15, 0.1, 0.1, 0.05, 0.1), strict=True)}}
    num = sum(v * w for v, w in terms.values() if v is not None)
    den = sum(w for v, w in terms.values() if v is not None)
    return 100 * num / den


def plain_product(ev: EpisodeEvidence) -> float:
    """REJECTED: 100 * evacuated rate * product of quality components (no incident factors, no gate)."""
    if ev.max_evacuated == 0:
        return 0.0
    p = _parts(ev)
    q = 1.0
    for k in QUALITY_COMPONENTS:
        if p[k] is not None:
            q *= p[k]  # type: ignore[operator]
    return 100 * (ev.evacuated / ev.max_evacuated) * q


def rate_only_safety(ev: EpisodeEvidence) -> float:
    """REJECTED: the official form without the discrete incident factors (alpha = 1)."""
    return official(replace(OFFICIAL_CONSTANTS, casualty_incident_factor=Decimal(1), stranding_incident_factor=Decimal(1)))(ev)


def safety_times_arith_quality(ev: EpisodeEvidence) -> float:
    """REJECTED: official Safety times an arithmetic (compensatory) mean of the quality components."""
    s = score_evidence(0, ev, OFFICIAL_CONSTANTS)
    if s.safety is None or s.safety == 0:
        return 0.0
    p = _parts(ev)
    w = {k: float(v) for k, v in OFFICIAL_CONSTANTS.weights().items()}
    num = sum(p[k] * w[k] for k in QUALITY_COMPONENTS if p[k] is not None)  # type: ignore[operator]
    den = sum(w[k] for k in QUALITY_COMPONENTS if p[k] is not None)
    return 100 * float(s.safety) * (num / den if den else 0.0)


def horizon_time(ev: EpisodeEvidence) -> float:
    """REJECTED: official, but Time = (W - A) / (W - L) over all people with censoring at the horizon."""
    s = score_evidence(0, ev, OFFICIAL_CONSTANTS)
    return float(s.score)  # placeholder; the horizon variant is reported separately in time_variants()


CANDIDATES: dict[str, Formula] = {
    "official": official(),
    "additive": additive,
    "plain-product": plain_product,
    "no-incident-factor": rate_only_safety,
    "arith-quality": safety_times_arith_quality,
}


# --------------------------------------------------------------------------- reports


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


def ranks(scores: Mapping[str, float]) -> dict[str, int]:
    order = sorted(scores, key=lambda k: -scores[k])
    return {k: i + 1 for i, k in enumerate(order)}


def kendall_tau(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = sorted(a)
    conc = disc = 0
    for x, y in itertools.combinations(keys, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    total = conc + disc
    return 1.0 if total == 0 else (conc - disc) / total


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def rank(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else float("nan")


def synthetic(population: int, *, casualties: int = 0, stranded: int = 0, time: Fraction = Fraction(1), blocks: Fraction = Fraction(0)) -> EpisodeEvidence:
    """A hand-made evidence record: perfect except for the given failures."""
    evac = population - casualties - stranded
    attempts = 1000
    return EpisodeEvidence(
        scenario_id="synthetic",
        scenario_hash="0" * 64,
        seed=0,
        record_digest="0" * 64,
        population=population,
        max_evacuated=population,
        evacuated=evac,
        casualties=casualties,
        stranded=stranded,
        evacuee_time=int(100 * evac / time) if evac else 0,
        evacuee_time_bound=100 * evac if evac else 0,
        movement_attempts=attempts,
        blocked_occupied=int(attempts * blocks),
        blocked_conflict=0,
        blocked_cycle=0,
        path_cells_moved=10 * evac,
        path_moves_bound=10 * evac,
        path_static_bound=10 * evac,
        evacuee_danger_states=0,
        evacuee_danger_bound=0,
        invalidation_events=0,
        unrecoverable_invalidations=0,
        replanning_remaining=0,
        replanning_remaining_bound=0,
    )


REQUIREMENTS: list[tuple[str, Callable[[Formula], bool]]] = [
    ("R1 one casualty in 100, otherwise perfect, scores <= 70", lambda f: f(synthetic(100, casualties=1)) <= 70),
    ("R2 safe run with Time 0.25 beats perfect run with 1/100 casualty", lambda f: f(synthetic(100, time=Fraction(1, 4))) > f(synthetic(100, casualties=1))),
    ("R3 2 casualties < 1 casualty by more than 1 < 0 does", lambda f: (f(synthetic(100, casualties=1)) - f(synthetic(100, casualties=2))) > 0),
    ("R4 10/100 casualties still >= 5 (unsafe runs stay separable)", lambda f: f(synthetic(100, casualties=10)) >= 5),
    ("R5 10/100 casualties <= 35", lambda f: f(synthetic(100, casualties=10)) <= 35),
    ("R6 one stranded in 100 <= 85 and above one casualty", lambda f: f(synthetic(100, casualties=1)) < f(synthetic(100, stranded=1)) <= 85),
    ("R7 10/100 stranded <= 50", lambda f: f(synthetic(100, stranded=10)) <= 50),
    ("R8 congestion rate 0.3 costs >= 10 points", lambda f: f(synthetic(100)) - f(synthetic(100, blocks=Fraction(3, 10))) >= 10),
    ("R9 50% losses score 0 (catastrophic gate)", lambda f: f(synthetic(100, casualties=50)) == 0),
    ("R10 non-compensation: Time 0.1 with everything else perfect scores <= 60", lambda f: f(synthetic(100, time=Fraction(1, 10))) <= 60),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=ROOT / ".scoring-study")
    args = parser.parse_args()
    suites = study_suites(args.cache)
    data = collect(suites, args.cache / "evidence")
    names = list(POLICIES)

    print("## Raw components (hierarchical means over both suites; Safety parts and quality parts in [0, 1])\n")
    comp_rows = []
    for name in names:
        rows = data[name]
        scored = [score_evidence(0, r[4]) for r in rows]
        line: list[Any] = [name]
        for comp in ("evacuation", "casualty", "stranding"):
            line.append(hierarchical(rows, [None if s.component(comp) is None else 100 * float(s.component(comp)) for s in scored]))  # type: ignore[arg-type]
        line.append(hierarchical(rows, [None if s.safety is None else 100 * float(s.safety) for s in scored]))
        for comp in QUALITY_COMPONENTS:
            vals = [None if s.component(comp) is None else 100 * float(s.component(comp)) for s in scored]  # type: ignore[arg-type]
            v = hierarchical(rows, vals)
            line.append("n/a" if v is None else v)
        line.append(hierarchical(rows, [float(s.score) for s in scored]))
        comp_rows.append(line)
    print(table(["policy", "evac", "casualty", "stranding", "Safety", *QUALITY_COMPONENTS, "EMBER"], comp_rows))

    print("\n## Totals (people over all episodes)\n")
    tot = []
    for name in names:
        rows = data[name]
        tot.append([name, sum(r[4].evacuated for r in rows), sum(r[4].casualties for r in rows), sum(r[4].stranded for r in rows), sum(r[4].max_evacuated for r in rows)])
    print(table(["policy", "evacuated", "casualties", "stranded", "privileged max"], tot))

    print("\n## Candidate formulas: overall scores per suite\n")
    results: dict[str, dict[str, float]] = {}
    for cname, f in CANDIDATES.items():
        results[cname] = {}
        for name in names:
            results[cname][name] = hierarchical(data[name], [f(r[4]) for r in data[name]]) or 0.0
    print(table(["policy", *CANDIDATES], [[n, *(results[c][n] for c in CANDIDATES)] for n in names]))
    for suite_name in ("public", "generated"):
        print(f"\n### official score, {suite_name} suite only\n")
        rows_out = []
        for name in names:
            rows = [r for r in data[name] if r[0] == suite_name]
            rows_out.append([name, hierarchical(rows, [official()(r[4]) for r in rows]) or 0.0])
        print(table(["policy", "EMBER"], rows_out))

    print("\n## Requirement checks per candidate\n")
    req_rows = []
    for label, check in REQUIREMENTS:
        req_rows.append([label, *("pass" if check(f) else "FAIL" for f in CANDIDATES.values())])
    print(table(["requirement", *CANDIDATES], req_rows))

    print("\n## Adversarial checks (official vs candidates): policy must score below congestion-aware\n")
    adv_rows = []
    for name in names:
        if not name.startswith("adv-"):
            continue
        adv_rows.append([name, *(f"{results[c][name] - results[c]['congestion-aware']:+.1f}" for c in CANDIDATES)])
    print(table(["adversary - congestion-aware", *CANDIDATES], adv_rows))

    print("\n## Synthetic single-failure profile (official)\n")
    syn = [
        ("perfect", synthetic(100)),
        ("1 casualty / 100", synthetic(100, casualties=1)),
        ("2 casualties / 100", synthetic(100, casualties=2)),
        ("3 casualties / 100", synthetic(100, casualties=3)),
        ("5 casualties / 100", synthetic(100, casualties=5)),
        ("10 casualties / 100", synthetic(100, casualties=10)),
        ("25 casualties / 100", synthetic(100, casualties=25)),
        ("1 casualty / 4", synthetic(4, casualties=1)),
        ("1 stranded / 100", synthetic(100, stranded=1)),
        ("10 stranded / 100", synthetic(100, stranded=10)),
        ("Time 0.5", synthetic(100, time=Fraction(1, 2))),
        ("Time 0.25", synthetic(100, time=Fraction(1, 4))),
        ("congestion rate 0.1", synthetic(100, blocks=Fraction(1, 10))),
        ("congestion rate 0.3", synthetic(100, blocks=Fraction(3, 10))),
        ("congestion rate 0.6", synthetic(100, blocks=Fraction(6, 10))),
    ]
    print(table(["case", *CANDIDATES], [[label, *(f(ev) for f in CANDIDATES.values())] for label, ev in syn]))

    print("\n## Component correlation (Spearman, episodes with every component defined, moving policies)\n")
    comps = ["safety", *QUALITY_COMPONENTS[:4]]
    vecs: dict[str, list[float]] = {c: [] for c in comps}
    for name in names:
        if name in ("always-stay", "random-legal"):
            continue
        for r in data[name]:
            s = score_evidence(0, r[4])
            vals = {"safety": None if s.safety is None else float(s.safety), **{c: (None if s.component(c) is None else float(s.component(c))) for c in QUALITY_COMPONENTS[:4]}}  # type: ignore[arg-type]
            if all(v is not None for v in vals.values()):
                for c in comps:
                    vecs[c].append(vals[c])  # type: ignore[arg-type]
    print(table(["", *comps], [[a, *(f"{spearman(vecs[a], vecs[b]):.2f}" for b in comps)] for a in comps]))
    print(f"\n(n = {len(vecs['safety'])} episodes)")

    print("\n## Sensitivity grid: constants satisfying every requirement, and rank stability\n")
    base = results["official"]
    grid = {
        "casualty_incident_factor": ["0.5", "0.6", "0.7", "0.8", "0.9"],
        "casualty_rate_exponent": ["2", "4", "6"],
        "evacuation_exponent": ["1", "2", "3"],
        "stranding_incident_factor": ["0.7", "0.8", "0.9"],
        "congestion_exponent": ["1", "2", "3"],
    }
    feasible = []
    taus: list[float] = []
    total = 0
    for combo in itertools.product(*grid.values()):
        total += 1
        c = replace(OFFICIAL_CONSTANTS, **{k: Decimal(v) for k, v in zip(grid, combo, strict=True)})
        f = official(c)
        if not all(check(f) for _label, check in REQUIREMENTS):
            continue
        sc = {n: hierarchical(data[n], [f(r[4]) for r in data[n]]) or 0.0 for n in names}
        tau = kendall_tau(base, sc)
        feasible.append((combo, tau))
        taus.append(tau)
    print(f"feasible: {len(feasible)} of {total} combinations")
    official_combo = tuple(str(getattr(OFFICIAL_CONSTANTS, k)) for k in grid)
    print(f"official constants {dict(zip(grid, official_combo, strict=True))} feasible: {any(c == official_combo for c, _ in feasible)}")
    if taus:
        print(f"Kendall tau vs official ranking over feasible combinations: min {min(taus):.3f}, mean {sum(taus) / len(taus):.3f}")
    per_axis: dict[str, dict[str, int]] = {k: {} for k in grid}
    for combo, _tau in feasible:
        for k, v in zip(grid, combo, strict=True):
            per_axis[k][v] = per_axis[k].get(v, 0) + 1
    for k, counts in per_axis.items():
        print(f"- {k}: " + ", ".join(f"{v}: {counts.get(v, 0)}" for v in grid[k]))

    print("\n## Quality-weight perturbation (each weight x0.5 and x1.5, renormalized)\n")
    w_rows = []
    weights = OFFICIAL_CONSTANTS.weights()
    for comp in QUALITY_COMPONENTS:
        for factor in (Fraction(1, 2), Fraction(3, 2)):
            raw = {k: (w * factor if k == comp else w) for k, w in weights.items()}
            s = sum(raw.values())
            norm = {f"weight_{k}": v / s for k, v in raw.items()}
            c = replace(OFFICIAL_CONSTANTS, **norm)
            f = official(c)
            sc = {n: hierarchical(data[n], [f(r[4]) for r in data[n]]) or 0.0 for n in names}
            w_rows.append([f"{comp} x{float(factor)}", f"{kendall_tau(base, sc):.3f}", f"{max(abs(sc[n] - base[n]) for n in names):.1f}"])
    print(table(["perturbation", "Kendall tau", "max |delta score|"], w_rows))

    print("\n## Official ranking\n")
    rk = ranks(base)
    print(table(["rank", "policy", "EMBER"], [[rk[n], n, base[n]] for n in sorted(names, key=lambda n: rk[n])]))


if __name__ == "__main__":
    main()
