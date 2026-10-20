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
    severe_failures,
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


def fire_hugger(obs: Observation) -> dict[str, Action]:
    """Congestion-aware, but among free shortest-route moves it always takes the one closest to the fire."""
    acts = congestion(obs)
    if not obs.burning:
        return acts
    fields = _safe_fields(obs)
    occupied = frozenset(p.position for p in obs.active)
    for p in obs.active:
        f = _nearest(fields, p.position)
        if f is None:
            continue
        downhill = [a for a in _moves(obs) if f.get(_target(p.position, a)) == f[p.position] - 1 and _target(p.position, a) not in occupied]
        if downhill:
            acts[p.id] = min(downhill, key=lambda a: (min(_manhattan(_target(p.position, a), b) for b in obs.burning), a.value))
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
    "adv-fire-hugger": lambda: fire_hugger,
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
                fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r[4].items()}
                rows.append((r[0], r[1], r[2], r[3], EpisodeEvidence(**fields)))  # type: ignore[arg-type]
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
BETA_STRANDING = math.log(0.8) / math.log(0.5)  # F-strand-floor: the floor is the stranding factor


def official(constants: ScoringConstants = OFFICIAL_CONSTANTS) -> Formula:
    """F (adopted): constrained geometric quality and severe-failure gates."""

    def f(ev: EpisodeEvidence) -> float:
        return float(score_evidence(0, ev, constants).score)

    return f


def _scored(ev: EpisodeEvidence) -> Any:
    return score_evidence(0, ev, OFFICIAL_CONSTANTS)


def _parts(ev: EpisodeEvidence) -> dict[str, float | None]:
    return {k: (None if v is None else float(v)) for k, v in _scored(ev).components}


def _sq(ev: EpisodeEvidence) -> tuple[float, float]:
    """(Safety, unexponentiated weighted geometric quality) under the official constants."""
    s = _scored(ev)
    return (0.0 if s.safety is None else float(s.safety), 0.0 if s.quality is None else float(s.quality))


def a_multiplicative(ev: EpisodeEvidence) -> float:
    """A (rejected): 100 * Safety * Quality, no gates (the first draft of this PR)."""
    s, q = _sq(ev)
    return 100 * s * q


def b_hard_ceilings(ev: EpisodeEvidence) -> float:
    """B (rejected): A, then capped at 80 when any component is severe (a flat cap, no dominance floor)."""
    a = a_multiplicative(ev)
    return min(a, 80.0) if severe_failures(ev) else a


def _one_more_casualty(ev: EpisodeEvidence) -> EpisodeEvidence | None:
    if ev.evacuated == 0:
        return None
    return replace(
        ev,
        evacuated=ev.evacuated - 1,
        casualties=ev.casualties + 1,
        avoidable_casualty_ids=tuple(sorted((*ev.avoidable_casualty_ids, "~extra"))),
    )


def c_safety_tiers(ev: EpisodeEvidence) -> float:
    """C (rejected): quality interpolates between this safety level and the next worse one."""
    s, q = _sq(ev)
    nxt = _one_more_casualty(ev)
    s_next = 0.0 if nxt is None else _sq(nxt)[0]
    return 100 * (s_next + (s - s_next) * q)


def d_lexicographic(ev: EpisodeEvidence) -> float:
    """D (rejected): safety first; quality only moves the score by up to 5% of the safety level."""
    s, q = _sq(ev)
    return 100 * s * (1 - 0.05 * (1 - q))


def e_soft_min(ev: EpisodeEvidence) -> float:
    """E (rejected): weighted power mean (p = -5) of Safety and the quality components (a soft minimum)."""
    s, _q = _sq(ev)
    if s == 0:
        return 0.0
    p = _parts(ev)
    terms = [(s, 0.5)] + [(p[k], 0.5 * float(w)) for k, w in OFFICIAL_CONSTANTS.weights().items() if p[k] is not None]
    if any(v == 0 for v, _w in terms):
        return 0.0
    total = sum(w for _v, w in terms)
    weighted = [float(w) * float(v) ** -5 for v, w in terms]  # type: ignore[arg-type]
    return float(100 * (sum(weighted) / float(total)) ** (-1 / 5))


def f_strand_floor(ev: EpisodeEvidence) -> float:
    """F variant (rejected): as F, but the dominance floor is the stranding factor 0.8 (beta = 0.32)."""
    sc = _scored(ev)
    if sc.safety is None or sc.quality is None:
        return 0.0
    return float(100 * float(sc.safety) * float(sc.quality) ** BETA_STRANDING * float(sc.severity or 0))


def additive(ev: EpisodeEvidence) -> float:
    """REJECTED (first study): 100 * weighted arithmetic mean of evacuated rate and the quality components."""
    if ev.max_evacuated == 0:
        return 0.0
    p = _parts(ev)
    terms = {"evac": (ev.evacuated / ev.max_evacuated, 0.5), **{k: (p[k], w) for k, w in zip(QUALITY_COMPONENTS, (0.15, 0.1, 0.1, 0.05, 0.1), strict=True)}}
    num = sum(v * w for v, w in terms.values() if v is not None)
    den = sum(w for v, w in terms.values() if v is not None)
    return 100 * num / den


def plain_product(ev: EpisodeEvidence) -> float:
    """REJECTED (first study): 100 * evacuated rate * product of quality components (no incident factors, no gate)."""
    if ev.max_evacuated == 0:
        return 0.0
    p = _parts(ev)
    q = 1.0
    for k in QUALITY_COMPONENTS:
        if p[k] is not None:
            q *= p[k]  # type: ignore[operator]
    return 100 * (ev.evacuated / ev.max_evacuated) * q


CANDIDATES: dict[str, Formula] = {
    "F-official": official(),
    "A-multiplicative": a_multiplicative,
    "B-hard-ceilings": b_hard_ceilings,
    "C-safety-tiers": c_safety_tiers,
    "D-lexicographic": d_lexicographic,
    "E-soft-min": e_soft_min,
    "F-strand-floor": f_strand_floor,
    "additive": additive,
    "plain-product": plain_product,
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


# --------------------------------------------------------------------------- synthetic profiles


def synthetic(
    population: int,
    *,
    casualties: int = 0,
    stranded: int = 0,
    unsavable_stranded: int = 0,
    time: Fraction = Fraction(1),
    blocks: Fraction = Fraction(0),
    hazard: Fraction = Fraction(0),
    path: Fraction = Fraction(1),
    replanning: Fraction | None = None,
) -> EpisodeEvidence:
    """A hand-made evidence record: perfect except for the given failures (all components exact)."""
    evac = population - casualties - stranded - unsavable_stranded
    attempts = 10_000
    steps = 100 * evac
    return EpisodeEvidence(
        scenario_id="synthetic",
        scenario_hash="0" * 64,
        seed=0,
        record_digest="0" * 64,
        population=population,
        max_evacuated=population - unsavable_stranded,
        evacuated=evac,
        casualties=casualties,
        stranded=stranded + unsavable_stranded,
        avoidable_casualty_ids=tuple(f"c{i:04d}" for i in range(casualties)),
        avoidable_stranded_ids=tuple(f"s{i:04d}" for i in range(stranded)),
        unavoidable_casualty_ids=(),
        unavoidable_stranded_ids=tuple(f"u{i:04d}" for i in range(unsavable_stranded)),
        evacuee_time=int(steps / time) if evac else 0,
        evacuee_time_bound=steps,
        evacuee_time_queue_bound=steps,
        movement_attempts=attempts,
        blocked_occupied=int(attempts * blocks),
        blocked_conflict=0,
        blocked_cycle=0,
        path_cells_moved=int(10 * evac / path),
        path_moves_bound=10 * evac,
        path_static_bound=10 * evac,
        evacuee_danger_states=int(steps / time * hazard) if evac else 0,
        evacuee_danger_bound=0,
        invalidation_events=0 if replanning is None else 1,
        unrecoverable_invalidations=0,
        replanning_remaining=0 if replanning is None else int(1000 / replanning),
        replanning_remaining_bound=0 if replanning is None else 1000,
    )


def _mixed_identity() -> EpisodeEvidence:
    """P1 savable and dead, P2 unsavable and stranded, 98 evacuated (R11)."""
    return replace(synthetic(100, casualties=1, unsavable_stranded=1))


def _threshold_corner(population: int, stranded: int, c: ScoringConstants = OFFICIAL_CONSTANTS) -> EpisodeEvidence:
    """No avoidable casualty; every quality component exactly at its severe threshold (not severe)."""
    return synthetic(
        population,
        stranded=stranded,
        time=c.severe_time,
        blocks=c.severe_block_rate,
        hazard=c.severe_hazard_rate,
        path=c.severe_path,
        replanning=c.severe_replanning,
    )


def _r12(f: Formula, c: ScoringConstants = OFFICIAL_CONSTANTS) -> bool:
    for m in (2, 4, 10, 100, 1000):
        for s in (0, 1):
            if s and m < 4:
                continue
            if not f(_threshold_corner(m, s, c)) > f(synthetic(m, casualties=1, stranded=s)):
                return False
    return True


REQUIREMENTS: list[tuple[str, Callable[[Formula], bool]]] = [
    ("R1 one avoidable casualty in 100, otherwise perfect, scores <= 60", lambda f: f(synthetic(100, casualties=1)) <= 60),
    ("R2 Time is not irrelevant: Time 0.5 (not severe) costs >= 10 points", lambda f: f(synthetic(100)) - f(synthetic(100, time=Fraction(1, 2))) >= 10),
    ("R3 2 casualties score below 1 casualty", lambda f: f(synthetic(100, casualties=2)) < f(synthetic(100, casualties=1))),
    ("R4 10/100 casualties still >= 5 (unsafe runs stay separable)", lambda f: f(synthetic(100, casualties=10)) >= 5),
    ("R5 10/100 casualties <= 35", lambda f: f(synthetic(100, casualties=10)) <= 35),
    ("R6 one stranded in 100 <= 85 and above one casualty", lambda f: f(synthetic(100, casualties=1)) < f(synthetic(100, stranded=1)) <= 85),
    ("R7 10/100 stranded <= 50", lambda f: f(synthetic(100, stranded=10)) <= 50),
    ("R8 congestion rate 0.3 costs >= 10 points", lambda f: f(synthetic(100)) - f(synthetic(100, blocks=Fraction(3, 10))) >= 10),
    ("R9 50% avoidable losses score 0 (catastrophic gate)", lambda f: f(synthetic(100, casualties=50)) == 0),
    ("R10 Time 0.1 with everything else perfect scores <= 60", lambda f: f(synthetic(100, time=Fraction(1, 10))) <= 60),
    ("R11 a savable casualty is not hidden by another person's unavoidable loss", lambda f: f(_mixed_identity()) <= 60),
    ("R12 no casualty + no severe failure beats perfect + 1 casualty (M* 2..1000, s 0..1)", _r12),
    ("R13 severe bottleneck (rate 0.3), no loss: < 80 and >= 20 points lost", lambda f: f(synthetic(100, blocks=Fraction(3, 10))) < 80 and f(synthetic(100)) - f(synthetic(100, blocks=Fraction(3, 10))) >= 20),
    (
        "R14 severe Time 0.4 / Hazard 0.3 / Replanning 0.4 / Path 0.4 alone: each < 80",
        lambda f: all(
            f(ev) < 80
            for ev in (
                synthetic(100, time=Fraction(2, 5)),
                synthetic(100, hazard=Fraction(3, 10)),
                synthetic(100, replanning=Fraction(2, 5)),
                synthetic(100, path=Fraction(2, 5)),
            )
        ),
    ),
    ("R15 failed / missing / timeout episodes score 0 and stay in every mean (structural; tests)", lambda f: True),
]

# Pre-registered ordering (written before the study was run): each pair must satisfy left > right.
ORDERING: list[tuple[str, EpisodeEvidence, str, EpisodeEvidence]] = [
    ("zero-loss, Time 0.6", synthetic(100, time=Fraction(3, 5)), "perfect quality + 1 casualty", synthetic(100, casualties=1)),
    ("perfect", synthetic(100), "zero-loss but slow (Time 0.5)", synthetic(100, time=Fraction(1, 2))),
    ("perfect", synthetic(100), "zero-loss, severe congestion (0.3)", synthetic(100, blocks=Fraction(3, 10))),
    ("perfect", synthetic(100), "zero-loss, high exposure (0.3)", synthetic(100, hazard=Fraction(3, 10))),
    ("perfect", synthetic(100), "zero-loss, inefficient routing (Path 0.6)", synthetic(100, path=Fraction(3, 5))),
    ("replanning 0.9", synthetic(100, replanning=Fraction(9, 10)), "static-good, invalidation failure (0.3)", synthetic(100, replanning=Fraction(3, 10))),
    ("1 stranded", synthetic(100, stranded=1), "1 casualty", synthetic(100, casualties=1)),
    ("1 casualty", synthetic(100, casualties=1), "2 casualties", synthetic(100, casualties=2)),
    ("mild congestion (0.05)", synthetic(100, blocks=Fraction(1, 20)), "severe congestion (0.4)", synthetic(100, blocks=Fraction(2, 5))),
    ("mild delay (Time 0.8)", synthetic(100, time=Fraction(4, 5)), "extreme delay (Time 0.1)", synthetic(100, time=Fraction(1, 10))),
]


def _old_attribution(ev: EpisodeEvidence) -> tuple[int, int]:
    """The draft's aggregate rule: unavoidable losses offset casualties first, then stranding."""
    u = ev.population - ev.max_evacuated
    return max(0, ev.casualties - u), max(0, ev.stranded - max(0, u - ev.casualties))


def _scores(data: Mapping[str, list[Row]], f: Formula, names: Sequence[str]) -> dict[str, float]:
    return {n: hierarchical(data[n], [f(r[4]) for r in data[n]]) or 0.0 for n in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=ROOT / ".scoring-study")
    args = parser.parse_args()
    suites = study_suites(args.cache)
    data = collect(suites, args.cache / "evidence")
    names = list(POLICIES)
    c = OFFICIAL_CONSTANTS
    print(f"quality exponent beta = {float(c.quality_exponent):.6f}; severe thresholds (component space) " + ", ".join(f"{k} {float(v):.4f}" for k, v in c.severe_thresholds().items()))

    print("\n## Raw components (hierarchical means over both suites; Safety parts and quality parts in [0, 1])\n")
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

    print("\n## Severe quality failures (share of applicable episodes, %)\n")
    sev_rows = []
    for name in names:
        scored = [score_evidence(0, r[4]) for r in data[name]]
        applicable = [s for s in scored if s.safety is not None]
        sev_rows.append([name, *(100 * sum(1 for s in applicable if k in s.severe) / max(1, len(applicable)) for k in QUALITY_COMPONENTS)])
    print(table(["policy", *QUALITY_COMPONENTS], sev_rows))

    print("\n## Totals (people over all episodes) and person-level attribution\n")
    tot = []
    changed_episodes = 0
    all_episodes = 0
    for name in names:
        rows = data[name]
        old_c = old_s = 0
        for r in rows:
            oc, os_ = _old_attribution(r[4])
            old_c += oc
            old_s += os_
            all_episodes += 1
            if (oc, os_) != (r[4].avoidable_casualties, r[4].avoidable_stranded):
                changed_episodes += 1
        tot.append(
            [
                name,
                sum(r[4].evacuated for r in rows),
                sum(r[4].casualties for r in rows),
                sum(r[4].stranded for r in rows),
                sum(r[4].max_evacuated for r in rows),
                sum(r[4].avoidable_casualties for r in rows),
                old_c,
                sum(r[4].avoidable_stranded for r in rows),
                old_s,
            ]
        )
    print(table(["policy", "evacuated", "casualties", "stranded", "savable", "avoidable cas.", "draft rule", "avoidable str.", "draft rule"], tot))
    print(f"\nEpisodes where the draft's aggregate rule differs from person-level attribution: {changed_episodes} of {all_episodes}")

    print("\n## Candidate designs: overall scores (both suites)\n")
    results = {cname: _scores(data, f, names) for cname, f in CANDIDATES.items()}
    print(table(["policy", *CANDIDATES], [[n, *(results[k][n] for k in CANDIDATES)] for n in names]))
    for suite_name in ("public", "generated"):
        print(f"\n### F-official and A-multiplicative, {suite_name} suite only\n")
        rows_out = []
        for name in names:
            rows = [r for r in data[name] if r[0] == suite_name]
            rows_out.append([name, hierarchical(rows, [official()(r[4]) for r in rows]) or 0.0, hierarchical(rows, [a_multiplicative(r[4]) for r in rows]) or 0.0])
        print(table(["policy", "F-official", "A-multiplicative"], rows_out))

    print("\n## Requirement checks per candidate\n")
    print(table(["requirement", *CANDIDATES], [[label, *("pass" if check(f) else "FAIL" for f in CANDIDATES.values())] for label, check in REQUIREMENTS]))

    print("\n## Pre-registered ordering (left must score above right)\n")
    ord_rows = []
    for left, lev, right, rev in ORDERING:
        ord_rows.append([f"{left} > {right}", *("pass" if f(lev) > f(rev) else "FAIL" for f in CANDIDATES.values())])
    print(table(["ordering", *CANDIDATES], ord_rows))
    print("\nF-official values: " + "; ".join(f"{left} {official()(lev):.1f} vs {right} {official()(rev):.1f}" for left, lev, right, rev in ORDERING))

    print("\n## Adversarial checks: adversary minus congestion-aware (gaming check, not a selection criterion)\n")
    adv_rows = [[n, *(f"{results[k][n] - results[k]['congestion-aware']:+.1f}" for k in CANDIDATES)] for n in names if n.startswith("adv-")]
    print(table(["adversary - congestion-aware", *CANDIDATES], adv_rows))

    print("\n## Synthetic single-failure profile\n")
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
        ("1 savable dead + 1 unsavable stranded / 100", _mixed_identity()),
        ("Time 0.8", synthetic(100, time=Fraction(4, 5))),
        ("Time 0.5 (threshold)", synthetic(100, time=Fraction(1, 2))),
        ("Time 0.4 (severe)", synthetic(100, time=Fraction(2, 5))),
        ("Time 0.25", synthetic(100, time=Fraction(1, 4))),
        ("Time 0.1", synthetic(100, time=Fraction(1, 10))),
        ("congestion rate 0.1", synthetic(100, blocks=Fraction(1, 10))),
        ("congestion rate 0.25 (threshold)", synthetic(100, blocks=Fraction(1, 4))),
        ("congestion rate 0.3 (severe)", synthetic(100, blocks=Fraction(3, 10))),
        ("congestion rate 0.5", synthetic(100, blocks=Fraction(1, 2))),
        ("avoidable hazard 0.1", synthetic(100, hazard=Fraction(1, 10))),
        ("avoidable hazard 0.3 (severe)", synthetic(100, hazard=Fraction(3, 10))),
        ("Path 0.6", synthetic(100, path=Fraction(3, 5))),
        ("Path 0.4 (severe)", synthetic(100, path=Fraction(2, 5))),
        ("Replanning 0.8", synthetic(100, replanning=Fraction(4, 5))),
        ("Replanning 0.3 (severe)", synthetic(100, replanning=Fraction(3, 10))),
        ("every component at its threshold", _threshold_corner(100, 0)),
    ]
    print(table(["case", *CANDIDATES], [[label, *(f(ev) for f in CANDIDATES.values())] for label, ev in syn]))

    print("\n## Component correlation (Spearman, episodes with every component defined, moving policies)\n")
    comps = ["safety", *QUALITY_COMPONENTS]
    vecs: dict[str, list[float]] = {k: [] for k in comps}
    for name in names:
        if name in ("always-stay", "random-legal"):
            continue
        for r in data[name]:
            s = score_evidence(0, r[4])
            cvals: dict[str, float | None] = {"safety": None if s.safety is None else float(s.safety)}
            for k in QUALITY_COMPONENTS:
                part = s.component(k)
                cvals[k] = None if part is None else float(part)
            present = [v for v in cvals.values() if v is not None]
            if len(present) == len(comps):
                for k, v in zip(comps, present, strict=True):
                    vecs[k].append(v)
    print(table(["", *comps], [[a, *(f"{spearman(vecs[a], vecs[b]):.2f}" for b in comps)] for a in comps]))
    print(f"\n(n = {len(vecs['safety'])} episodes)")

    print("\n## Score spread (F-official)\n")
    base = results["F-official"]
    moving = [base[n] for n in names if n not in ("always-stay", "random-legal")]
    mean = sum(moving) / len(moving)
    sd = math.sqrt(sum((v - mean) ** 2 for v in moving) / (len(moving) - 1))
    print(f"moving policies: min {min(moving):.1f}, max {max(moving):.1f}, range {max(moving) - min(moving):.1f}, sd {sd:.1f}")
    for k in ("A-multiplicative", "C-safety-tiers", "D-lexicographic", "E-soft-min"):
        mv = [results[k][n] for n in names if n not in ("always-stay", "random-legal")]
        print(f"{k}: range {max(mv) - min(mv):.1f}")

    print("\n## Per-family ranking (F-official; family means over seeds)\n")
    families = sorted({f"{r[0]}:{r[1]}" for r in data[names[0]]})
    fam_rows = []
    for name in names:
        line = [name]
        for fam in families:
            rows = [r for r in data[name] if f"{r[0]}:{r[1]}" == fam]
            line.append(hierarchical(rows, [official()(r[4]) for r in rows]) or 0.0)
        fam_rows.append(line)
    print(table(["policy", *families], fam_rows))
    for fam_i, fam in enumerate(families):
        col = {row[0]: row[fam_i + 1] for row in fam_rows}
        top = sorted(col, key=lambda n: -col[n])[:3]
        print(f"- {fam}: " + ", ".join(f"{n} {col[n]:.1f}" for n in top))

    print("\n## Sensitivity grids: requirement feasibility and rank stability\n")
    grid_runs = [
        (
            "safety",
            {
                "casualty_incident_factor": ["0.5", "0.6", "0.7"],
                "casualty_rate_exponent": ["2", "4", "6"],
                "evacuation_exponent": ["1", "2", "3"],
                "stranding_incident_factor": ["0.7", "0.8", "0.9"],
            },
        ),
        (
            "gates",
            {
                "severe_factor": ["0.7", "0.8", "0.9"],
                "severe_exponent": ["0.5", "1", "2"],
                "severe_time": ["1/3", "1/2", "2/3"],
                "severe_block_rate": ["1/5", "1/4", "1/3"],
                "severe_hazard_rate": ["1/5", "1/4", "1/3"],
            },
        ),
    ]
    for label, grid in grid_runs:
        feasible = []
        every: list[tuple[float, float]] = []
        failing: dict[str, int] = {}
        total = 0
        for combo in itertools.product(*grid.values()):
            total += 1
            change = {k: (Fraction(v) if "/" in v else Decimal(v)) for k, v in zip(grid, combo, strict=True)}
            constants = replace(OFFICIAL_CONSTANTS, **change)  # type: ignore[arg-type]
            f = official(constants)
            sc = _scores(data, f, names)
            tau, delta = kendall_tau(base, sc), max(abs(sc[n] - base[n]) for n in names)
            every.append((tau, delta))
            failed = [lab.split()[0] for lab, check in REQUIREMENTS if not (_r12(f, constants) if lab.startswith("R12") else check(f))]
            for req in failed:
                failing[req] = failing.get(req, 0) + 1
            if not failed:
                feasible.append((combo, tau, delta))
        print(f"### grid {label}: feasible {len(feasible)} of {total}")
        print(f"every combination: Kendall tau min {min(t for t, _d in every):.3f}, mean {sum(t for t, _d in every) / len(every):.3f}; max |delta score| {max(d for _t, d in every):.1f}")
        print("requirements failed (combinations): " + (", ".join(f"{k} {v}" for k, v in sorted(failing.items())) or "none"))
        official_combo = tuple(str(getattr(OFFICIAL_CONSTANTS, k)) for k in grid)
        print(f"official {dict(zip(grid, official_combo, strict=True))} feasible: {any(c_ == official_combo for c_, _t, _d in feasible)}")
        if feasible:
            taus = [t for _c, t, _d in feasible]
            print(f"Kendall tau vs official over feasible: min {min(taus):.3f}, mean {sum(taus) / len(taus):.3f}; max |delta score| {max(d for _c, _t, d in feasible):.1f}")
        for i, k in enumerate(grid):
            counts: dict[str, int] = {}
            for combo, _t, _d in feasible:
                counts[combo[i]] = counts.get(combo[i], 0) + 1
            print(f"- {k}: " + ", ".join(f"{v}: {counts.get(v, 0)}" for v in grid[k]))
        print()

    print("## Quality-weight perturbation (each weight x0.5 and x1.5, renormalized)\n")
    w_rows = []
    weights = OFFICIAL_CONSTANTS.weights()
    for comp in QUALITY_COMPONENTS:
        for factor in (Fraction(1, 2), Fraction(3, 2)):
            raw = {k: (w * factor if k == comp else w) for k, w in weights.items()}
            total_w = sum(raw.values())
            constants = replace(OFFICIAL_CONSTANTS, **{f"weight_{k}": v / total_w for k, v in raw.items()})  # type: ignore[arg-type]
            sc = _scores(data, official(constants), names)
            w_rows.append([f"{comp} x{float(factor)}", f"{kendall_tau(base, sc):.3f}", f"{max(abs(sc[n] - base[n]) for n in names):.1f}"])
    print(table(["perturbation", "Kendall tau", "max |delta score|"], w_rows))

    print("\n## Rank agreement between candidate designs (Kendall tau vs F-official)\n")
    print(table(["candidate", "Kendall tau"], [[k, f"{kendall_tau(base, results[k]):.3f}"] for k in CANDIDATES]))

    print("\n## Official ranking (F-official, both suites)\n")
    rk = ranks(base)
    print(table(["rank", "policy", "EMBER"], [[rk[n], n, base[n]] for n in sorted(names, key=lambda n: rk[n])]))


if __name__ == "__main__":
    main()
