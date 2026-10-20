"""The official EMBER Score (scoring version 1.0): one failure-sensitive number in [0, 100].

The full definition, its justification and the sensitivity study are in
``docs/scoring.md`` and ``docs/benchmark-methodology.md``. In short, for one
episode::

    EMBER = 100 * Safety * Quality

    Safety  = E^kE * P_casualty * P_stranding          (0 when the loss rate is catastrophic)
    Quality = weighted geometric mean of Time, Coordination, Hazard, Path and Replanning

Every component is in ``[0, 1]`` and no component can exceed 1, so good
performance in one dimension never offsets a failure in another. Episodes are
aggregated episode -> scenario -> primary family -> overall, each level an
unweighted mean of the level below.

Rules for this module:

- **Fail closed.** Results are authenticated exactly as in
  :mod:`emberbench.statistics`: the manifest must target the suite, every run
  record is replayed, and the stored metrics must equal the recomputed ones.
- **Failures never disappear.** A scheduled episode without an authenticated
  result scores 0 and is labelled with its failure kind (``missing`` unless
  the evaluator declares ``invalid_output``, ``model_error`` or ``timeout``).
- **Privileged references, evaluator side only.** The time, path, hazard and
  replanning references are relaxed single-person bounds computed from the
  scenario and the evaluator seed (like :mod:`emberbench.oracle`). They never
  reach a model.
- **Deterministic arithmetic.** Counts and references are exact integers and
  fractions. The non-integer powers use :mod:`decimal` (the libmpdec software
  library, identical on every platform), and each episode score is rounded
  half-even to ``1e-9`` points before any aggregation, so aggregation and the
  bootstrap are exact rational arithmetic.
- **Separate from cost.** Latency, tokens and money are not inputs.

``SCORING_VERSION`` changes whenever a definition, a constant or the
serialization changes. It is separate from every simulator, observation,
schema, metrics and statistics version, none of which this module changes.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from decimal import ROUND_HALF_EVEN, Context, Decimal
from enum import StrEnum
from fractions import Fraction
from typing import Any

from .canonical import canonical_json, sha256_hex
from .errors import EmberBenchError, EvaluationError, ScoringInputError
from .environment_pack import EnvironmentPack, trajectory_digest
from .evaluation import EvaluationManifest, EvaluationResult, resolve_scenario, verify_manifest
from .events import RunRecord
from .grid import Pos, neighbours
from .metrics import EpisodeMetrics, compute_episode_metrics
from .oracle import fire_trajectory
from .scenario import Scenario
from .state import PersonStatus
from .statistics import (
    CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    MIN_BOOTSTRAP_SAMPLE_SIZE,
    BootstrapStream,
    ConfidenceInterval,
    bootstrap_seed,
    check_resamples,
    quantile,
)
from .suite import Suite

SCORING_VERSION = "1.0"

# Every value is computed in this context. libmpdec is a software library, so
# the digits are identical on every platform and Python build.
_CTX = Context(prec=50, rounding=ROUND_HALF_EVEN, Emin=-999_999, Emax=999_999)
_SCORE_QUANTUM = Decimal("1e-9")  # episode score resolution, in points
_COMPONENT_QUANTUM = Decimal("1e-12")  # component resolution, in [0, 1]
_UNREACHABLE = 1 << 62
HAZARD_RADIUS = 2  # cells (Manhattan); see docs/scoring.md section 3


def _dec(value: Fraction | int) -> Decimal:
    v = Fraction(value)
    return _CTX.divide(Decimal(v.numerator), Decimal(v.denominator))


def _pow(base: Decimal, exponent: Decimal) -> Decimal:
    if base == 0:
        return Decimal(0) if exponent > 0 else Decimal(1)
    if base == 1 or exponent == 0:
        return Decimal(1)
    return _CTX.exp(_CTX.multiply(exponent, _CTX.ln(base)))


def _fail(message: str) -> ScoringInputError:
    return ScoringInputError(message)


# --------------------------------------------------------------------------- constants


@dataclass(frozen=True)
class ScoringConstants:
    """Every tuning constant of the score. :data:`OFFICIAL_CONSTANTS` is frozen by ``SCORING_VERSION``.

    Exponents and factors are exact decimal strings. The quality weights are
    exact fractions that sum to 1.
    """

    evacuation_exponent: Decimal = Decimal("2")
    casualty_incident_factor: Decimal = Decimal("0.6")
    casualty_rate_exponent: Decimal = Decimal("4")
    stranding_incident_factor: Decimal = Decimal("0.8")
    stranding_rate_exponent: Decimal = Decimal("2")
    catastrophic_loss_rate: Fraction = Fraction(1, 2)
    congestion_exponent: Decimal = Decimal("2")
    hazard_exponent: Decimal = Decimal("1")
    weight_time: Fraction = Fraction(3, 10)
    weight_coordination: Fraction = Fraction(2, 10)
    weight_hazard: Fraction = Fraction(2, 10)
    weight_path: Fraction = Fraction(1, 10)
    weight_replanning: Fraction = Fraction(2, 10)
    # Severe quality failures (docs/scoring.md section 4.3). Each threshold is in natural units.
    severe_time: Fraction = Fraction(1, 2)  # Time below this: person-time above 2x the privileged bound
    severe_block_rate: Fraction = Fraction(1, 4)  # congestion block rate above this
    severe_hazard_rate: Fraction = Fraction(1, 4)  # avoidable danger-zone rate above this
    severe_path: Fraction = Fraction(1, 2)  # Path below this: more than 2x the fewest moves
    severe_replanning: Fraction = Fraction(1, 2)  # Replanning below this: recovery above 2x the privileged bound
    severe_factor: Decimal = Decimal("0.8")  # paid once per severe component
    severe_exponent: Decimal = Decimal("1")  # on (component / threshold) below the threshold
    practical_difference: Fraction = Fraction(1)  # points; smaller paired differences are never "better"

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type in (Fraction, "Fraction"):
                if not isinstance(value, Fraction) or value < 0:
                    raise ValueError(f"{f.name} must be a non-negative Fraction")
            elif not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{f.name} must be a finite non-negative Decimal")
        for name in ("casualty_incident_factor", "stranding_incident_factor", "severe_factor"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 < self.catastrophic_loss_rate <= 1:
            raise ValueError("catastrophic_loss_rate must be in (0, 1]")
        if sum(self.weights().values(), Fraction(0)) != 1:
            raise ValueError("the quality weights must sum to 1")
        if not all(0 < v < 1 for v in self.severe_thresholds().values()):
            raise ValueError("every severe threshold must be in (0, 1)")

    def severe_thresholds(self) -> dict[str, Decimal]:
        """Per quality component: a value strictly below the threshold is a severe failure."""
        return {
            "time": _dec(self.severe_time),
            "coordination": _pow(_dec(1 - self.severe_block_rate), self.congestion_exponent),
            "hazard": _pow(_dec(1 - self.severe_hazard_rate), self.hazard_exponent),
            "path": _dec(self.severe_path),
            "replanning": _dec(self.severe_replanning),
        }

    @property
    def quality_exponent(self) -> Decimal:
        """``beta = ln(casualty_incident_factor) / ln(lowest severe threshold)``, capped at 1. Derived, not tuned.

        Quality enters the score as ``Q ** beta``. Every non-severe component is
        at or above its threshold, so ``Q ** beta >= casualty_incident_factor``:
        a run without avoidable casualties and without a severe failure always
        outscores the same run with an avoidable casualty and perfect quality.
        """
        lowest = min(self.severe_thresholds().values())
        return min(Decimal(1), _CTX.divide(_CTX.ln(self.casualty_incident_factor), _CTX.ln(lowest)))

    def weights(self) -> dict[str, Fraction]:
        return {
            "time": self.weight_time,
            "coordination": self.weight_coordination,
            "hazard": self.weight_hazard,
            "path": self.weight_path,
            "replanning": self.weight_replanning,
        }

    def to_dict(self) -> dict[str, str]:
        out = {f.name: str(getattr(self, f.name)) for f in fields(self)}
        out["quality_exponent"] = str(self.quality_exponent)
        return out

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


OFFICIAL_CONSTANTS = ScoringConstants()

QUALITY_COMPONENTS = ("time", "coordination", "hazard", "path", "replanning")
SAFETY_COMPONENTS = ("evacuation", "casualty", "stranding")


# --------------------------------------------------------------------------- privileged references


@dataclass(frozen=True)
class _References:
    """Relaxed single-person bounds of one (scenario, seed). PRIVILEGED: they use the future fire.

    Each bound is the optimum for one person alone who knows the whole fire
    trajectory and follows the simulator phase order. Other people can only
    block moves, so every bound is a lower bound for every policy.
    """

    trajectory: tuple[frozenset[Pos], ...]
    index: Mapping[Pos, int]
    earliest: tuple[tuple[int, ...], ...]  # earliest[t][cell]: first evacuation step from (cell, snapshot t)
    min_moves: Mapping[Pos, int]  # fewest cell moves of any escape from (cell, snapshot 0)
    min_danger: Mapping[Pos, int]  # fewest danger-zone person-states of any escape from (cell, snapshot 0)
    fire_distance: tuple[tuple[int, ...], ...]  # Manhattan distance to fire per snapshot and cell; 0 when no fire


def _manhattan_field(width: int, height: int, burning: frozenset[Pos]) -> list[int]:
    """Manhattan distance from every grid cell to the nearest burning cell (walls ignored); 0 without fire."""
    if not burning:
        return [0] * (width * height)
    dist = [-1] * (width * height)
    frontier = sorted(burning)
    for p in frontier:
        dist[p.row * width + p.col] = 0
    while frontier:
        nxt: list[Pos] = []
        for p in frontier:
            base = dist[p.row * width + p.col] + 1
            for q in neighbours(p, width, height):
                k = q.row * width + q.col
                if dist[k] < 0:
                    dist[k] = base
                    nxt.append(q)
        frontier = nxt
    return dist


def in_danger_zone(distance: int) -> bool:
    """A person-state is in the danger zone when a burning cell is within ``HAZARD_RADIUS`` (Manhattan)."""
    return 0 < distance <= HAZARD_RADIUS


def _references(scenario: Scenario, seed: int) -> _References:
    width, height = scenario.width, scenario.height
    horizon = scenario.termination.max_steps
    walls = scenario.walls
    exits = frozenset(scenario.exit_positions)
    trajectory = fire_trajectory(scenario, seed)
    cells = [Pos(r, c) for r in range(height) for c in range(width) if Pos(r, c) not in walls]
    index = {p: i for i, p in enumerate(cells)}
    moves = [tuple(index[q] for q in (p, *neighbours(p, width, height)) if q not in walls) for p in cells]
    is_exit = [p in exits for p in cells]

    fields_by_set: dict[frozenset[Pos], tuple[int, ...]] = {}
    distance: list[tuple[int, ...]] = []
    for burning in trajectory:
        cached = fields_by_set.get(burning)
        if cached is None:
            full = _manhattan_field(width, height, burning)
            cached = tuple(full[p.row * width + p.col] for p in cells)
            fields_by_set[burning] = cached
        distance.append(cached)
    burning_idx = [frozenset(index[p] for p in b if p in index) for b in trajectory]

    # Backward induction over (cell, snapshot). From (i, t) one step reaches (j, t + 1) for j in {i} and
    # its open neighbours, unless j burns at the start of the step (refused or fatal), or j is not an
    # exit and burns after the fire phase (a casualty). An exit that does not burn at the start of the
    # step evacuates at once. Snapshot ``horizon`` has no successor: the person is stranded.
    n = len(cells)
    none = _UNREACHABLE
    earliest: list[tuple[int, ...]] = [()] * (horizon + 1)
    earliest[horizon] = (none,) * n
    moves_next = [none] * n
    danger_next = [none] * n
    for t in range(horizon - 1, -1, -1):
        now, after, nxt_e = burning_idx[t], burning_idx[t + 1], earliest[t + 1]
        danger_here = [1 if in_danger_zone(d) else 0 for d in distance[t]]
        cur_e, cur_m, cur_h = [none] * n, [none] * n, [none] * n
        for i in range(n):
            if i in now:
                continue
            best_e = best_m = best_h = none
            for j in moves[i]:
                if j in now:
                    continue
                if is_exit[j]:  # i is never an exit: people on an exit evacuate
                    best_e, best_m, best_h = t + 1, 1, 0
                    break
                if j in after:
                    continue
                best_e = min(best_e, nxt_e[j])
                if moves_next[j] != none:
                    best_m = min(best_m, moves_next[j] + (j != i))
                if danger_next[j] != none:
                    best_h = min(best_h, danger_next[j])
            cur_e[i] = best_e
            cur_m[i] = best_m
            cur_h[i] = none if best_h == none else danger_here[i] + best_h
        earliest[t] = tuple(cur_e)
        moves_next, danger_next = cur_m, cur_h
    min_moves = {cells[i]: v for i, v in enumerate(moves_next) if v != none}
    min_danger = {cells[i]: v for i, v in enumerate(danger_next) if v != none}
    return _References(trajectory, index, tuple(earliest), min_moves, min_danger, tuple(distance))


def _fire_aware_field(scenario: Scenario, burning: frozenset[Pos]) -> dict[Pos, int]:
    """Distance from every cell to the nearest non-burning exit through non-wall, non-burning cells."""
    width, height = scenario.width, scenario.height
    blocked = scenario.walls | burning
    frontier = sorted(p for p in scenario.exit_positions if p not in burning)
    dist = {p: 0 for p in frontier}
    while frontier:
        nxt: list[Pos] = []
        for p in frontier:
            for q in neighbours(p, width, height):
                if q not in dist and q not in blocked:
                    dist[q] = dist[p] + 1
                    nxt.append(q)
        frontier = nxt
    return dist


def _queue_bound(releases: Sequence[int], scenario: Scenario, refs: _References, horizon: int) -> int:
    """Least possible sum of evacuation steps for people with these earliest steps. PRIVILEGED.

    Positions after movement are distinct, so in step ``s`` at most one person
    leaves through each exit cell that is not burning at the start of the
    step. Ignoring which exit each person uses, filling the open exit slots of
    every step with released people (a list schedule) minimizes the sum for
    unit jobs with release dates, so the result is a lower bound for every
    policy and is never below the sum of the releases.
    """
    exits = scenario.exit_positions
    total = waiting = 0
    i = 0
    for step in range(1, horizon + 1):
        while i < len(releases) and releases[i] <= step:
            waiting += 1
            i += 1
        leave = min(waiting, sum(1 for e in exits if e not in refs.trajectory[step - 1]))
        total += leave * step
        waiting -= leave
        if i == len(releases) and waiting == 0:
            return total
    raise _fail("the evacuees cannot all leave within the horizon")


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class EpisodeEvidence:
    """Exact integer evidence for one authenticated episode. Every score input is here.

    ``*_bound`` values are privileged relaxed references (single person,
    other people ignored, future fire known), so each is a lower bound on the
    matching actual value of every policy.
    """

    scenario_id: str
    scenario_hash: str
    seed: int
    record_digest: str
    population: int
    max_evacuated: int  # privileged: people who can escape at all (the savable people)
    evacuated: int
    casualties: int
    stranded: int
    avoidable_casualty_ids: tuple[str, ...]  # savable people who died
    avoidable_stranded_ids: tuple[str, ...]  # savable people still inside at the end
    unavoidable_casualty_ids: tuple[str, ...]  # unsavable people who died
    unavoidable_stranded_ids: tuple[str, ...]  # unsavable people still inside at the end
    evacuee_time: int  # sum of evacuation steps over evacuees
    evacuee_time_bound: int  # sum of each evacuee's earliest possible evacuation step (single person; diagnostic)
    evacuee_time_queue_bound: int  # the same, with at most one evacuation per open exit cell per step
    movement_attempts: int
    blocked_occupied: int
    blocked_conflict: int
    blocked_cycle: int
    path_cells_moved: int  # sum of cells moved over evacuees
    path_moves_bound: int  # sum over evacuees of the fewest moves of any escape (fire-aware, privileged)
    path_static_bound: int  # sum over evacuees of the wall-only exit distance (the metrics bound; diagnostic)
    evacuee_danger_states: int  # active evacuee person-states with fire within HAZARD_RADIUS
    evacuee_danger_bound: int  # sum over evacuees of the fewest such states of any escape (privileged)
    invalidation_events: int
    unrecoverable_invalidations: int
    replanning_remaining: int  # sum over recoverable events of (censored resolution step - t)
    replanning_remaining_bound: int  # sum over recoverable events of (earliest escape step - t)

    def __post_init__(self) -> None:
        groups = (self.avoidable_casualty_ids, self.avoidable_stranded_ids, self.unavoidable_casualty_ids, self.unavoidable_stranded_ids)
        ids = [pid for group in groups for pid in group]
        if any(list(group) != sorted(set(group)) for group in groups) or len(set(ids)) != len(ids):
            raise _fail("loss identities must be sorted, unique and disjoint")
        if (
            self.casualties != len(self.avoidable_casualty_ids) + len(self.unavoidable_casualty_ids)
            or self.stranded != len(self.avoidable_stranded_ids) + len(self.unavoidable_stranded_ids)
            or self.population != self.evacuated + self.casualties + self.stranded
            or self.max_evacuated != self.population - self.unavoidable_losses
        ):
            raise _fail("loss identities do not match the outcome counts")
        if not (
            0 <= self.evacuee_time_bound <= self.evacuee_time_queue_bound <= self.evacuee_time
            and 0 <= self.evacuee_danger_bound <= self.evacuee_danger_states <= self.evacuee_time
            and 0 <= self.congestion_blocks <= self.movement_attempts
        ):
            raise _fail("evidence bounds are inconsistent")

    @property
    def unavoidable_losses(self) -> int:
        """Lost people whom no policy could have saved (privileged, per person)."""
        return len(self.unavoidable_casualty_ids) + len(self.unavoidable_stranded_ids)

    @property
    def avoidable_casualties(self) -> int:
        """Savable people who died. Never offset by another person's unavoidable loss."""
        return len(self.avoidable_casualty_ids)

    @property
    def avoidable_stranded(self) -> int:
        """Savable people still inside at the end. Never offset by another person's unavoidable loss."""
        return len(self.avoidable_stranded_ids)

    @property
    def congestion_blocks(self) -> int:
        return self.blocked_occupied + self.blocked_conflict + self.blocked_cycle

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def episode_evidence(
    scenario: Scenario,
    record: RunRecord,
    metrics: EpisodeMetrics | None = None,
    *,
    _refs: _References | None = None,
) -> EpisodeEvidence:
    """Authenticate ``record`` and compute the exact evidence for scoring. Privileged: uses the seed.

    ``metrics`` may be passed when the caller has already computed them from the
    same pair; they are recomputed and compared otherwise.
    """
    recomputed = compute_episode_metrics(scenario, record)  # authenticates by replay
    if metrics is not None and canonical_json(metrics.to_dict()) != canonical_json(recomputed.to_dict()):
        raise _fail("metrics do not match the authenticated record")
    m = recomputed
    refs = _references(scenario, record.seed) if _refs is None else _refs
    horizon = scenario.termination.max_steps
    snapshots = (record.initial_state, *(s.state for s in record.steps))
    for t, snap in enumerate(snapshots):
        if snap.burning != refs.trajectory[t]:
            raise _fail(f"snapshot {t}: burning set differs from the reference fire trajectory")

    start = {p.id: p.position for p in scenario.people}
    resolved = {p.person_id: p for p in m.people}
    evacuees = [pid for pid, p in resolved.items() if p.status is PersonStatus.EVACUATED]
    earliest0 = {pid: refs.earliest[0][refs.index[pos]] for pid, pos in start.items()}
    savable = frozenset(pid for pid, e in earliest0.items() if e != _UNREACHABLE)
    max_evacuated = len(savable)

    # Person-level attribution: a loss is avoidable exactly when that person was savable.
    losses: dict[tuple[PersonStatus, bool], list[str]] = {}
    for pid, outcome in resolved.items():
        if outcome.status is not PersonStatus.EVACUATED:
            losses.setdefault((outcome.status, pid in savable), []).append(pid)

    def lost(status: PersonStatus, is_savable: bool) -> tuple[str, ...]:
        return tuple(sorted(losses.get((status, is_savable), ())))

    time_actual = time_bound = 0
    for pid in evacuees:
        step = resolved[pid].time_to_safety
        bound = earliest0[pid]
        if bound == _UNREACHABLE or step < bound:
            raise _fail(f"{pid} evacuated at step {step}, before the privileged bound {bound}")
        time_actual += step
        time_bound += bound
    queue_bound = _queue_bound(sorted(earliest0[pid] for pid in evacuees), scenario, refs, horizon)
    if time_actual < queue_bound:
        raise _fail("evacuation person-time below the privileged exit-capacity bound")

    # Danger zone: active evacuee person-states with a burning cell within HAZARD_RADIUS.
    evac_set = frozenset(evacuees)
    danger = {pid: 0 for pid in evacuees}
    for t, snap in enumerate(snapshots):
        dist = refs.fire_distance[t]
        for p in snap.active:
            if p.id in evac_set and in_danger_zone(dist[refs.index[p.position]]):
                danger[p.id] += 1
    danger_bound = moves_actual = moves_bound = 0
    for pid in evacuees:
        least_danger = refs.min_danger.get(start[pid])
        least_moves = refs.min_moves.get(start[pid])
        moved = resolved[pid].cells_moved
        if least_danger is None or least_danger > danger[pid]:
            raise _fail(f"{pid}: danger-zone exposure below the privileged minimum")
        if least_moves is None or least_moves > moved:
            raise _fail(f"{pid}: fewer moves than the privileged minimum")
        danger_bound += least_danger
        moves_actual += moved
        moves_bound += least_moves

    # Route invalidation: fire spread in step t lengthened the person's best fire-aware route.
    field_cache: dict[frozenset[Pos], dict[Pos, int]] = {}

    def fire_field(burning: frozenset[Pos]) -> dict[Pos, int]:
        if burning not in field_cache:
            field_cache[burning] = _fire_aware_field(scenario, burning)
        return field_cache[burning]

    events = unrecoverable = remaining = remaining_bound = 0
    for t in range(1, len(snapshots)):
        before_b, after_b = snapshots[t - 1].burning, snapshots[t].burning
        if before_b == after_b:
            continue
        before, after = fire_field(before_b), fire_field(after_b)
        for p in snapshots[t].active:
            old = before.get(p.position)
            new = after.get(p.position)
            if old is None or (new is not None and new <= old):
                continue
            events += 1
            bound = refs.earliest[t][refs.index[p.position]]
            if bound == _UNREACHABLE:
                unrecoverable += 1
                continue
            person = resolved[p.id]
            resolution = person.time_to_safety  # evacuation step, or the horizon (censored)
            if person.status is PersonStatus.EVACUATED and resolution < bound:
                raise _fail(f"{p.id} evacuated before the privileged bound after invalidation")
            remaining += resolution - t
            remaining_bound += bound - t
    if remaining < remaining_bound:
        raise _fail("replanning time below its privileged bound")

    mv = dict(m.movement.blocked_by_outcome)
    return EpisodeEvidence(
        scenario_id=record.scenario_id,
        scenario_hash=record.scenario_hash,
        seed=record.seed,
        record_digest=record.digest(),
        population=m.safety.population,
        max_evacuated=max_evacuated,
        evacuated=m.safety.evacuated,
        casualties=m.safety.casualties,
        stranded=m.safety.stranded,
        avoidable_casualty_ids=lost(PersonStatus.CASUALTY, True),
        avoidable_stranded_ids=lost(PersonStatus.ACTIVE, True),
        unavoidable_casualty_ids=lost(PersonStatus.CASUALTY, False),
        unavoidable_stranded_ids=lost(PersonStatus.ACTIVE, False),
        evacuee_time=time_actual,
        evacuee_time_bound=time_bound,
        evacuee_time_queue_bound=queue_bound,
        movement_attempts=m.movement.movement_attempts,
        blocked_occupied=mv.get("blocked_occupied", 0),
        blocked_conflict=mv.get("blocked_conflict", 0),
        blocked_cycle=mv.get("blocked_cycle", 0),
        path_cells_moved=moves_actual,
        path_moves_bound=moves_bound,
        path_static_bound=m.path_efficiency.total_static_lower_bound,
        evacuee_danger_states=sum(danger.values()),
        evacuee_danger_bound=danger_bound,
        invalidation_events=events,
        unrecoverable_invalidations=unrecoverable,
        replanning_remaining=remaining,
        replanning_remaining_bound=remaining_bound,
    )


# --------------------------------------------------------------------------- episode score


class EpisodeStatus(StrEnum):
    SCORED = "scored"  # a complete, authenticated episode
    CATASTROPHIC = "catastrophic"  # authenticated, but the avoidable loss rate reached the gate: score 0
    FAILED = "failed"  # no authenticated result: score 0 (see FailureKind)
    NOT_APPLICABLE = "not_applicable"  # nobody could escape under any policy; excluded from every mean


class FailureKind(StrEnum):
    MISSING = "missing"  # scheduled, no result, no declared reason
    INVALID_OUTPUT = "invalid_output"  # ModelOutputError or another rejected action set
    MODEL_ERROR = "model_error"  # the model or its adapter raised
    TIMEOUT = "timeout"  # the adapter's own deadline expired


def _q(value: Decimal) -> Fraction:
    return Fraction(value.quantize(_COMPONENT_QUANTUM, rounding=ROUND_HALF_EVEN, context=_CTX))


def _incident(factor: Decimal, rate_exponent: Decimal, count: int, denominator: int) -> Decimal:
    """``factor ** log2(1 + count) * (1 - count / denominator) ** rate_exponent``; 1 when ``count == 0``.

    The discrete factor is paid once for the first person and again for every
    doubling of the count, so each additional loss costs more than the one
    before in relative terms; the rate term grows with the share of people.
    """
    if count == 0:
        return Decimal(1)
    progression = _CTX.divide(_CTX.ln(Decimal(1 + count)), _CTX.ln(Decimal(2)))
    rate = _dec(Fraction(denominator - count, denominator))
    return _CTX.multiply(_pow(factor, progression), _pow(rate, rate_exponent))


def _out(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class EpisodeScore:
    """The score of one scheduled episode, with every component. Fractions are exact; ``None`` is undefined."""

    index: int
    scenario_id: str
    seed: int
    status: EpisodeStatus
    failure: FailureKind | None
    score: Fraction  # points in [0, 100], rounded half-even to 1e-9
    safety: Fraction | None
    quality: Fraction | None  # the weighted geometric mean, before the quality exponent
    components: tuple[tuple[str, Fraction | None], ...]  # SAFETY_COMPONENTS + QUALITY_COMPONENTS, in that order
    evidence: EpisodeEvidence | None
    severity: Fraction | None = None  # product of the severe-failure factors; 1 without a severe failure
    severe: tuple[str, ...] = ()  # quality components below their severe threshold

    def component(self, name: str) -> Fraction | None:
        return dict(self.components)[name]

    @property
    def safety_ceiling(self) -> Fraction | None:
        """The best score this episode could have reached with perfect quality: ``100 * Safety``."""
        return None if self.safety is None else 100 * self.safety

    @property
    def ceiling(self) -> Fraction | None:
        """The best score this episode could have reached with the same failures: ``100 * Safety * severity``."""
        return None if self.safety is None or self.severity is None else 100 * self.safety * self.severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "status": self.status.value,
            "failure": None if self.failure is None else self.failure.value,
            "score": float(self.score),
            "safety": _out(self.safety),
            "quality": _out(self.quality),
            "severity": _out(self.severity),
            "severe": list(self.severe),
            "safety_ceiling": _out(self.safety_ceiling),
            "ceiling": _out(self.ceiling),
            "components": {name: _out(v) for name, v in self.components},
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
        }


def quality_components(ev: EpisodeEvidence, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> dict[str, Fraction | None]:
    """The five quality components in ``[0, 1]``; ``None`` where the episode gives no evidence."""
    out: dict[str, Fraction | None] = {name: None for name in QUALITY_COMPONENTS}
    if ev.evacuated:
        out["time"] = Fraction(ev.evacuee_time_queue_bound, ev.evacuee_time)
        out["path"] = Fraction(ev.path_moves_bound, ev.path_cells_moved)
        out["hazard"] = _q(_pow(_dec(1 - _hazard_rate(ev)), constants.hazard_exponent))
    if ev.movement_attempts:
        out["coordination"] = _q(_pow(_dec(1 - _block_rate(ev)), constants.congestion_exponent))
    if ev.replanning_remaining:
        out["replanning"] = Fraction(ev.replanning_remaining_bound, ev.replanning_remaining)
    return out


def _hazard_rate(ev: EpisodeEvidence) -> Fraction:
    """Avoidable danger-zone person-states per evacuee person-step."""
    return Fraction(ev.evacuee_danger_states - ev.evacuee_danger_bound, ev.evacuee_time)


def _block_rate(ev: EpisodeEvidence) -> Fraction:
    """Movement attempts blocked by other people (occupied, conflict, cycle) per movement attempt."""
    return Fraction(ev.congestion_blocks, ev.movement_attempts)


def severe_failures(ev: EpisodeEvidence, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> dict[str, Decimal]:
    """Per severe quality component, ``component / threshold`` (in ``[0, 1)``). Compared exactly in natural units."""
    out: dict[str, Decimal] = {}
    if ev.evacuated:
        time = Fraction(ev.evacuee_time_queue_bound, ev.evacuee_time)
        if time < constants.severe_time:
            out["time"] = _dec(time / constants.severe_time)
        rate = _hazard_rate(ev)
        if rate > constants.severe_hazard_rate:
            out["hazard"] = _pow(_dec((1 - rate) / (1 - constants.severe_hazard_rate)), constants.hazard_exponent)
        path = Fraction(ev.path_moves_bound, ev.path_cells_moved)
        if path < constants.severe_path:
            out["path"] = _dec(path / constants.severe_path)
    if ev.movement_attempts:
        rate = _block_rate(ev)
        if rate > constants.severe_block_rate:
            out["coordination"] = _pow(_dec((1 - rate) / (1 - constants.severe_block_rate)), constants.congestion_exponent)
    if ev.replanning_remaining:
        replan = Fraction(ev.replanning_remaining_bound, ev.replanning_remaining)
        if replan < constants.severe_replanning:
            out["replanning"] = _dec(replan / constants.severe_replanning)
    return {k: out[k] for k in QUALITY_COMPONENTS if k in out}


def safety_components(ev: EpisodeEvidence, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> dict[str, Fraction]:
    """Evacuation, casualty and stranding factors in ``[0, 1]``. Requires ``max_evacuated >= 1``."""
    m = ev.max_evacuated
    evac = _pow(_dec(Fraction(ev.evacuated, m)), constants.evacuation_exponent)
    cas = _incident(constants.casualty_incident_factor, constants.casualty_rate_exponent, ev.avoidable_casualties, m)
    strand = _incident(constants.stranding_incident_factor, constants.stranding_rate_exponent, ev.avoidable_stranded, m)
    return {"evacuation": _q(evac), "casualty": _q(cas), "stranding": _q(strand)}


def score_evidence(index: int, ev: EpisodeEvidence, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> EpisodeScore:
    """Score one authenticated episode: ``100 * Safety * Quality ** beta * severity`` (docs/scoring.md section 4)."""
    if ev.max_evacuated == 0:
        empty = tuple((name, None) for name in (*SAFETY_COMPONENTS, *QUALITY_COMPONENTS))
        return EpisodeScore(index, ev.scenario_id, ev.seed, EpisodeStatus.NOT_APPLICABLE, None, Fraction(0), None, None, empty, ev)
    safety_parts = safety_components(ev, constants)
    quality_parts = quality_components(ev, constants)
    loss_rate = Fraction(ev.avoidable_casualties + ev.avoidable_stranded, ev.max_evacuated)
    catastrophic = loss_rate >= constants.catastrophic_loss_rate
    safety = Decimal(1)
    for value in safety_parts.values():
        safety = _CTX.multiply(safety, _dec(value))
    if catastrophic:
        safety = Decimal(0)

    weights = constants.weights()
    defined = {k: v for k, v in quality_parts.items() if v is not None and weights[k] > 0}
    total_weight = sum((weights[k] for k in defined), Fraction(0))
    if not defined:
        quality = Decimal(0)  # only possible with nobody evacuated, so Safety is 0 as well
    elif any(v == 0 for v in defined.values()):
        quality = Decimal(0)
    else:
        log_sum = Decimal(0)
        for k, v in defined.items():
            log_sum = _CTX.add(log_sum, _CTX.multiply(_dec(weights[k] / total_weight), _CTX.ln(_dec(v))))
        quality = _CTX.exp(log_sum)

    severe = severe_failures(ev, constants)
    severity = Decimal(1)
    for ratio in severe.values():
        severity = _CTX.multiply(severity, _CTX.multiply(constants.severe_factor, _pow(ratio, constants.severe_exponent)))

    points = _CTX.multiply(Decimal(100), _CTX.multiply(_CTX.multiply(safety, _pow(quality, constants.quality_exponent)), severity))
    score = Fraction(points.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN, context=_CTX))
    comps = tuple((name, safety_parts[name]) for name in SAFETY_COMPONENTS) + tuple((name, quality_parts[name]) for name in QUALITY_COMPONENTS)
    return EpisodeScore(
        index=index,
        scenario_id=ev.scenario_id,
        seed=ev.seed,
        status=EpisodeStatus.CATASTROPHIC if catastrophic else EpisodeStatus.SCORED,
        failure=None,
        score=score,
        safety=_q(safety),
        quality=_q(quality),
        components=comps,
        evidence=ev,
        severity=_q(severity),
        severe=tuple(severe),
    )


def failed_episode(index: int, scenario_id: str, seed: int, kind: FailureKind) -> EpisodeScore:
    """A scheduled episode without an authenticated result: score 0, Safety 0, quality undefined."""
    empty = tuple((name, None) for name in (*SAFETY_COMPONENTS, *QUALITY_COMPONENTS))
    return EpisodeScore(index, scenario_id, seed, EpisodeStatus.FAILED, kind, Fraction(0), Fraction(0), None, empty, None)


def score_episode(scenario: Scenario, record: RunRecord, *, index: int = 0, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> EpisodeScore:
    """Authenticate one run record and score it. Privileged: reads the scenario and the evaluator seed."""
    return score_evidence(index, episode_evidence(scenario, record), constants)


# --------------------------------------------------------------------------- authenticated results


def _score_trial(
    result: EvaluationResult,
    suite: Suite,
    refs: Mapping[int, _References],
    applicable: Mapping[int, bool],
    declared: Mapping[int, FailureKind],
    constants: ScoringConstants,
) -> dict[int, EpisodeScore]:
    """Authenticate one result from scratch (as :mod:`emberbench.statistics` does) and score every scheduled episode."""
    if not isinstance(result, EvaluationResult):
        raise TypeError("results must be EvaluationResult objects")
    try:
        manifest = EvaluationManifest.from_dict(result.manifest.to_dict())
        if manifest != result.manifest:
            raise _fail("manifest does not round-trip")
        verify_manifest(manifest, suite)
        checked = EvaluationResult(result.result_version, result.model_id, manifest, result.episodes)
    except ScoringInputError:
        raise
    except (EvaluationError, TypeError, ValueError, AttributeError) as exc:
        raise _fail(f"result for {getattr(result, 'model_id', None)!r} is not valid: {exc}") from exc

    committed = {e.episode.index: e for e in checked.episodes}
    scheduled = {spec.index for spec in manifest.episodes}
    for index, kind in declared.items():
        if not isinstance(kind, FailureKind):
            raise _fail(f"episode {index}: failure kind must be a FailureKind")
        if index not in scheduled:
            raise _fail(f"episode {index}: a failure is declared for an episode that is not scheduled")
        if index in committed:
            raise _fail(f"episode {index}: a failure is declared for an episode with a committed result")

    scores: dict[int, EpisodeScore] = {}
    for spec in manifest.episodes:
        e = committed.get(spec.index)
        if e is None:
            if applicable[spec.index]:
                scores[spec.index] = failed_episode(spec.index, spec.scenario_id, spec.seed, declared.get(spec.index, FailureKind.MISSING))
            continue
        where = f"episode {spec.index}"
        try:
            scenario = resolve_scenario(suite, spec)
            metrics = compute_episode_metrics(scenario, e.record)  # authenticates by replay
            evidence = episode_evidence(scenario, e.record, metrics, _refs=refs[spec.index])
        except ScoringInputError as exc:
            raise _fail(f"{where}: {exc}") from exc
        except (EmberBenchError, TypeError, ValueError, AttributeError, KeyError, IndexError) as exc:
            raise _fail(f"{where}: run record failed authentication: {exc}") from exc
        try:
            stored_ok = isinstance(e.metrics, EpisodeMetrics) and canonical_json(e.metrics.to_dict()) == canonical_json(metrics.to_dict())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _fail(f"{where}: stored metrics are malformed: {exc}") from exc
        if not stored_ok:
            raise _fail(f"{where}: stored metrics differ from the metrics of the authenticated record")
        if applicable[spec.index]:
            scores[spec.index] = score_evidence(spec.index, evidence, constants)
    return scores


# --------------------------------------------------------------------------- hierarchy

# primary family -> scenario -> applicable episode indices, in suite order (families in first-appearance order).
Layout = tuple[tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...]


def _layout(manifest: EvaluationManifest, suite: Suite, applicable: Mapping[int, bool]) -> Layout:
    families: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for entry in suite.manifest.entries:
        indices = tuple(s.index for s in manifest.episodes if s.scenario_id == entry.scenario_id and applicable[s.index])
        if indices:
            families.setdefault(entry.primary_family.value, []).append((entry.scenario_id, indices))
    return tuple((family, tuple(scenarios)) for family, scenarios in families.items())


def _mean(values: Sequence[Fraction]) -> Fraction | None:
    return sum(values, Fraction(0)) / len(values) if values else None


def _hierarchical(
    layout: Layout, value: Mapping[int, Fraction | None]
) -> tuple[Fraction | None, dict[str, Fraction | None], dict[str, Fraction | None]]:
    """``(overall, by family, by scenario)``: each level is the unweighted mean of the defined values below it."""
    by_scenario: dict[str, Fraction | None] = {}
    by_family: dict[str, Fraction | None] = {}
    for family, scenarios in layout:
        for scenario_id, indices in scenarios:
            by_scenario[scenario_id] = _mean([v for i in indices if (v := value[i]) is not None])
        by_family[family] = _mean([v for sid, _ in scenarios if (v := by_scenario[sid]) is not None])
    overall = _mean([v for v in by_family.values() if v is not None])
    return overall, by_family, by_scenario


def _micro(layout: Layout, value: Mapping[int, Fraction | None]) -> Fraction | None:
    return _mean([v for _, scenarios in layout for _, indices in scenarios for i in indices if (v := value[i]) is not None])


def _scenario_balanced(layout: Layout, value: Mapping[int, Fraction | None]) -> Fraction | None:
    by_scenario = _hierarchical(layout, value)[2]
    return _mean([v for v in by_scenario.values() if v is not None])


def _stratified_ci(layout: Layout, value: Mapping[int, Fraction], seed: bytes, resamples: int, statistic: str) -> ConfidenceInterval | None:
    """Stratified two-stage percentile bootstrap of the family-balanced mean.

    Strata are primary families. Each resample draws, within every family, as
    many scenarios as it has, with replacement; then, within every drawn
    scenario, as many episodes as it has, with replacement; and recomputes the
    family-balanced mean. ``None`` below :data:`MIN_BOOTSTRAP_SAMPLE_SIZE`
    episodes. The interval describes this evaluation sample only.
    """
    check_resamples(resamples)
    indices = [i for _, scenarios in layout for _, idx in scenarios for i in idx]
    if len(indices) < MIN_BOOTSTRAP_SAMPLE_SIZE:
        return None
    denominator = math.lcm(*(value[i].denominator for i in indices))
    nums = {i: value[i].numerator * (denominator // value[i].denominator) for i in indices}
    strata = [[[nums[i] for i in idx] for _, idx in scenarios] for _, scenarios in layout]
    stream = BootstrapStream(seed)
    stats: list[Fraction] = []
    for _ in range(resamples):
        total = Fraction(0)
        for scenarios in strata:
            k = len(scenarios)
            family_sum = Fraction(0)
            for _ in range(k):
                episodes = scenarios[stream.below(k)]
                n = len(episodes)
                family_sum += Fraction(sum(episodes[stream.below(n)] for _ in range(n)), n)
            total += family_sum / k
        stats.append(total / (len(strata) * denominator))
    stats.sort()
    alpha = 1 - CONFIDENCE_LEVEL
    lower, upper = quantile(stats, alpha / 2), quantile(stats, 1 - alpha / 2)
    assert lower is not None and upper is not None
    return ConfidenceInterval(statistic, CONFIDENCE_LEVEL, resamples, seed.hex(), lower, upper)


# --------------------------------------------------------------------------- report


@dataclass(frozen=True)
class ComponentAggregate:
    """One component, family-balanced over the episode-trials where it is defined."""

    name: str
    value: Fraction | None
    defined: int  # episode-trials with the component defined
    total: int  # applicable episode-trials

    def to_dict(self) -> dict[str, Any]:
        return {"value": _out(self.value), "defined": self.defined, "total": self.total}


@dataclass(frozen=True)
class GroupScore:
    """The score of one primary family or one scenario."""

    kind: str  # "family" or "scenario"
    name: str
    score: Fraction | None
    safety: Fraction | None
    episodes: int  # applicable scheduled episodes
    worst_episode: Fraction | None  # lowest trial-mean episode score

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "score": _out(self.score),
            "safety": _out(self.safety),
            "episodes": self.episodes,
            "worst_episode": _out(self.worst_episode),
        }


@dataclass(frozen=True)
class Robustness:
    """Lower-tail views. A high mean with a bad tail is visible here."""

    median_scenario: Fraction | None
    worst_scenario: Fraction | None
    worst_scenario_id: str | None
    p10_episode: Fraction | None  # type-7 10th percentile of trial-mean episode scores
    cvar10_episode: Fraction | None  # mean of the lowest ceil(n / 10) trial-mean episode scores
    zero_score_episode_trials: int
    catastrophic_episode_trials: int
    failed_episode_trials: tuple[tuple[str, int], ...]  # per FailureKind, in enum order
    severe_episode_trials: tuple[tuple[str, int], ...] = ()  # per quality component, in QUALITY_COMPONENTS order

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_scenario": _out(self.median_scenario),
            "worst_scenario": _out(self.worst_scenario),
            "worst_scenario_id": self.worst_scenario_id,
            "p10_episode": _out(self.p10_episode),
            "cvar10_episode": _out(self.cvar10_episode),
            "zero_score_episode_trials": self.zero_score_episode_trials,
            "catastrophic_episode_trials": self.catastrophic_episode_trials,
            "failed_episode_trials": dict(self.failed_episode_trials),
            "severe_episode_trials": dict(self.severe_episode_trials),
        }


@dataclass(frozen=True)
class SafetyTotals:
    """People counts summed over every applicable scored episode-trial (failed trials add nothing)."""

    trials: int
    escapable: int  # sum of max_evacuated
    evacuated: int
    avoidable_casualties: int
    avoidable_stranded: int
    unavoidable_losses: int

    def per_trial(self, value: int) -> Fraction:
        return Fraction(value, self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ScoreReport:
    """The EMBER Score of one model on one schedule. Built only by :func:`score_results`.

    ``score`` is the headline: the family-balanced mean of trial-mean episode
    scores, in benchmark-performance points. It is not a percentage of people.
    """

    scoring_version: str
    constants: ScoringConstants
    model_id: str
    trials: int
    evaluation_manifest_hash: str
    suite_id: str
    suite_hash: str
    pack_id: str | None
    pack_digest: str | None  # set only when scored against an EnvironmentPack (an official score)
    result_digests: tuple[str, ...]
    score: Fraction | None
    ci: ConfidenceInterval | None
    averages: tuple[tuple[str, Fraction | None], ...]  # family_balanced (headline), scenario_balanced, micro
    safety: Fraction | None
    quality: Fraction | None
    components: tuple[ComponentAggregate, ...]
    families: tuple[GroupScore, ...]
    scenarios: tuple[GroupScore, ...]
    robustness: Robustness
    totals: SafetyTotals
    not_applicable_episodes: tuple[int, ...]
    episode_scores: tuple[tuple[int, Fraction], ...]  # applicable index -> trial-mean score
    episodes: tuple[EpisodeScore, ...]  # every applicable episode-trial, trial-major
    layout: Layout

    @property
    def official(self) -> bool:
        """True only for a score against an environment pack under the official constants."""
        return self.pack_digest is not None and self.constants == OFFICIAL_CONSTANTS

    def component(self, name: str) -> ComponentAggregate:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    def family(self, name: str) -> GroupScore:
        for g in self.families:
            if g.name == name:
                return g
        raise KeyError(name)

    def scenario(self, name: str) -> GroupScore:
        for g in self.scenarios:
            if g.name == name:
                return g
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring_version": self.scoring_version,
            "constants": self.constants.to_dict(),
            "constants_digest": self.constants.digest(),
            "official": self.official,
            "model_id": self.model_id,
            "trials": self.trials,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "suite_id": self.suite_id,
            "suite_hash": self.suite_hash,
            "pack_id": self.pack_id,
            "pack_digest": self.pack_digest,
            "result_digests": list(self.result_digests),
            "unit": "benchmark-performance points (0-100), not a percentage of people",
            "score": _out(self.score),
            "ci": None if self.ci is None else self.ci.to_dict(),
            "averages": {name: _out(v) for name, v in self.averages},
            "safety": _out(self.safety),
            "quality": _out(self.quality),
            "components": {c.name: c.to_dict() for c in self.components},
            "families": [g.to_dict() for g in self.families],
            "scenarios": [g.to_dict() for g in self.scenarios],
            "robustness": self.robustness.to_dict(),
            "totals": self.totals.to_dict(),
            "not_applicable_episodes": list(self.not_applicable_episodes),
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


def _checked_failures(failures: Sequence[Mapping[int, FailureKind]] | None, trials: int) -> list[Mapping[int, FailureKind]]:
    if failures is None:
        return [{} for _ in range(trials)]
    if isinstance(failures, Mapping) or not isinstance(failures, Sequence) or len(failures) != trials:
        raise _fail("failures must be a sequence with one mapping per trial")
    for declared in failures:
        if not isinstance(declared, Mapping) or not all(isinstance(i, int) and not isinstance(i, bool) for i in declared):
            raise _fail("each failure declaration must map episode indices to FailureKind values")
    return list(failures)


def score_results(
    trials: Sequence[EvaluationResult],
    target: Suite | EnvironmentPack,
    *,
    failures: Sequence[Mapping[int, FailureKind]] | None = None,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    constants: ScoringConstants = OFFICIAL_CONSTANTS,
) -> ScoreReport:
    """Authenticate every trial of one model and compute its EMBER Score report.

    ``trials`` are repeated runs of the same model on the same manifest (one
    run for a deterministic policy). ``target`` is the suite, or the frozen
    :class:`~emberbench.environment_pack.EnvironmentPack`; only a pack whose
    manifest equals the trials' manifest gives an official score.
    ``failures[t]`` declares why scheduled episodes of trial ``t`` have no
    result; an undeclared gap is ``missing``. Every gap scores 0.
    """
    resamples = check_resamples(resamples)
    if isinstance(trials, EvaluationResult) or not isinstance(trials, Sequence) or not trials:
        raise _fail("trials must be a non-empty sequence of EvaluationResult objects")
    if not isinstance(constants, ScoringConstants):
        raise TypeError("constants must be ScoringConstants")
    pack = target if isinstance(target, EnvironmentPack) else None
    suite = pack.suite if pack is not None else target
    if not isinstance(suite, Suite):
        raise TypeError("target must be a Suite or an EnvironmentPack")
    declared = _checked_failures(failures, len(trials))
    first = trials[0]
    if not isinstance(first, EvaluationResult):
        raise TypeError("trials must be EvaluationResult objects")
    for t in trials[1:]:
        if not isinstance(t, EvaluationResult) or t.model_id != first.model_id or t.manifest != first.manifest:
            raise _fail("every trial must be a result of the same model on the same evaluation manifest")
    manifest = first.manifest
    if pack is not None and manifest != pack.manifest:
        raise _fail("the results were not produced on the environment pack's evaluation manifest")
    try:
        verify_manifest(manifest, suite)
    except (EvaluationError, TypeError, ValueError, AttributeError) as exc:
        raise _fail(f"manifest does not target the suite: {exc}") from exc

    # Privileged references: one per scheduled episode, shared by every trial.
    refs: dict[int, _References] = {}
    applicable: dict[int, bool] = {}
    for spec in manifest.episodes:
        scenario = resolve_scenario(suite, spec)
        r = _references(scenario, spec.seed)
        if pack is not None and trajectory_digest(spec.scenario_hash, spec.seed, r.trajectory) != pack.episodes[spec.index].fire_world_digest:
            raise _fail(f"episode {spec.index}: fire world differs from the environment pack")
        refs[spec.index] = r
        applicable[spec.index] = any(r.earliest[0][r.index[p.position]] != _UNREACHABLE for p in scenario.people)

    per_trial = [_score_trial(t, suite, refs, applicable, d, constants) for t, d in zip(trials, declared, strict=True)]
    layout = _layout(manifest, suite, applicable)
    ordered = [i for _, scenarios in layout for _, idx in scenarios for i in idx]
    n_trials = len(trials)

    def trial_mean(get: Callable[[EpisodeScore], Fraction | None]) -> dict[int, Fraction | None]:
        return {i: _mean([v for s in per_trial if (v := get(s[i])) is not None]) for i in ordered}

    score_by = trial_mean(lambda s: s.score)
    safety_by = trial_mean(lambda s: s.safety)
    headline, fam_scores, scen_scores = _hierarchical(layout, score_by)
    safety, fam_safety, scen_safety = _hierarchical(layout, safety_by)
    quality = _hierarchical(layout, trial_mean(lambda s: s.quality))[0]

    components: list[ComponentAggregate] = []
    for name in (*SAFETY_COMPONENTS, *QUALITY_COMPONENTS):
        by = trial_mean(lambda s, n=name: s.component(n))  # type: ignore[misc]
        defined = sum(1 for s in per_trial for i in ordered if s[i].component(name) is not None)
        components.append(ComponentAggregate(name, _hierarchical(layout, by)[0], defined, len(ordered) * n_trials))

    episode_values: dict[int, Fraction] = {i: v for i, v in score_by.items() if v is not None}
    seed_material = {
        "domain": "emberbench/scoring/score-ci",
        "scoring_version": SCORING_VERSION,
        "constants_digest": constants.digest(),
        "evaluation_manifest_hash": manifest.manifest_hash(),
        "model_id": first.model_id,
        "episode_scores": [[i, str(episode_values[i])] for i in ordered],
    }
    ci = _stratified_ci(layout, episode_values, bootstrap_seed(seed_material), resamples, "family_balanced_mean")

    families = tuple(
        GroupScore(
            "family",
            fam,
            fam_scores[fam],
            fam_safety[fam],
            sum(len(idx) for _, idx in scen),
            min((episode_values[i] for _, idx in scen for i in idx), default=None),
        )
        for fam, scen in layout
    )
    scenarios = tuple(
        GroupScore("scenario", sid, scen_scores[sid], scen_safety[sid], len(idx), min((episode_values[i] for i in idx), default=None))
        for _, scen in layout
        for sid, idx in scen
    )

    flat = [s[i] for s in per_trial for i in ordered]
    ep_sorted = sorted(episode_values.values())
    scen_values = sorted((v, sid) for sid, v in scen_scores.items() if v is not None)
    tail = ep_sorted[: math.ceil(len(ep_sorted) / 10)]
    robustness = Robustness(
        median_scenario=quantile([v for v, _ in scen_values], Fraction(1, 2)),
        worst_scenario=scen_values[0][0] if scen_values else None,
        worst_scenario_id=scen_values[0][1] if scen_values else None,
        p10_episode=quantile(ep_sorted, Fraction(1, 10)),
        cvar10_episode=_mean(tail),
        zero_score_episode_trials=sum(1 for e in flat if e.score == 0),
        catastrophic_episode_trials=sum(1 for e in flat if e.status is EpisodeStatus.CATASTROPHIC),
        failed_episode_trials=tuple((k.value, sum(1 for e in flat if e.failure is k)) for k in FailureKind),
        severe_episode_trials=tuple((k, sum(1 for e in flat if k in e.severe)) for k in QUALITY_COMPONENTS),
    )
    evidence = [e.evidence for e in flat if e.evidence is not None]
    totals = SafetyTotals(
        trials=n_trials,
        escapable=sum(ev.max_evacuated for ev in evidence),
        evacuated=sum(ev.evacuated for ev in evidence),
        avoidable_casualties=sum(ev.avoidable_casualties for ev in evidence),
        avoidable_stranded=sum(ev.avoidable_stranded for ev in evidence),
        unavoidable_losses=sum(ev.unavoidable_losses for ev in evidence),
    )
    return ScoreReport(
        scoring_version=SCORING_VERSION,
        constants=constants,
        model_id=first.model_id,
        trials=n_trials,
        evaluation_manifest_hash=manifest.manifest_hash(),
        suite_id=manifest.suite_id,
        suite_hash=manifest.suite_hash,
        pack_id=None if pack is None else pack.pack_id,
        pack_digest=None if pack is None else pack.digest,
        result_digests=tuple(sha256_hex(t.to_dict()) for t in trials),
        score=headline,
        ci=ci,
        averages=(
            ("family_balanced", headline),
            ("scenario_balanced", _scenario_balanced(layout, score_by)),
            ("micro", _micro(layout, score_by)),
        ),
        safety=safety,
        quality=quality,
        components=tuple(components),
        families=families,
        scenarios=scenarios,
        robustness=robustness,
        totals=totals,
        not_applicable_episodes=tuple(spec.index for spec in manifest.episodes if not applicable[spec.index]),
        episode_scores=tuple((i, episode_values[i]) for i in ordered),
        episodes=tuple(flat),
        layout=layout,
    )


def score_result(
    result: EvaluationResult,
    target: Suite | EnvironmentPack,
    *,
    failures: Mapping[int, FailureKind] | None = None,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    constants: ScoringConstants = OFFICIAL_CONSTANTS,
) -> ScoreReport:
    """:func:`score_results` for a single trial."""
    return score_results([result], target, failures=[failures or {}], resamples=resamples, constants=constants)


# --------------------------------------------------------------------------- comparison and leaderboard


class ComparisonOutcome(StrEnum):
    A_BETTER = "a_better"  # interval of the paired difference above 0 and the difference at least the practical margin
    B_BETTER = "b_better"
    INCONCLUSIVE = "inconclusive"  # anything else, including too few episodes for an interval
    EXACT_TIE = "exact_tie"  # every paired episode score is identical


@dataclass(frozen=True)
class ScoreComparison:
    """A paired comparison of two models on the same schedule. Separate from leaderboard order."""

    model_a: str
    model_b: str
    score_a: Fraction | None
    score_b: Fraction | None
    delta: Fraction | None  # score_a - score_b: the family-balanced mean of paired episode differences
    ci: ConfidenceInterval | None
    practical_difference: Fraction
    outcome: ComparisonOutcome
    episodes: int
    a_higher: int  # paired episodes where A scored higher
    b_higher: int
    equal: int
    family_deltas: tuple[tuple[str, Fraction | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "score_a": _out(self.score_a),
            "score_b": _out(self.score_b),
            "delta": _out(self.delta),
            "ci": None if self.ci is None else self.ci.to_dict(),
            "practical_difference": float(self.practical_difference),
            "outcome": self.outcome.value,
            "episodes": self.episodes,
            "a_higher": self.a_higher,
            "b_higher": self.b_higher,
            "equal": self.equal,
            "family_deltas": {name: _out(v) for name, v in self.family_deltas},
        }


def _comparable(a: ScoreReport, b: ScoreReport) -> bool:
    return (a.scoring_version, a.constants, a.evaluation_manifest_hash, a.pack_digest, a.layout) == (
        b.scoring_version,
        b.constants,
        b.evaluation_manifest_hash,
        b.pack_digest,
        b.layout,
    )


def compare_scores(a: ScoreReport, b: ScoreReport, *, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES) -> ScoreComparison:
    """Compare two reports episode by episode with a stratified paired bootstrap.

    ``A_BETTER`` needs both a 95% interval of the paired difference above 0
    and a difference of at least ``practical_difference`` points.
    ``B_BETTER`` is symmetric. ``EXACT_TIE`` needs every paired episode score
    equal. Everything else is ``INCONCLUSIVE``.
    """
    resamples = check_resamples(resamples)
    if not isinstance(a, ScoreReport) or not isinstance(b, ScoreReport):
        raise TypeError("a and b must be ScoreReport objects")
    if not _comparable(a, b):
        raise _fail("reports are not comparable: scoring version, constants, manifest, pack and applicable episodes must match")
    margin = a.constants.practical_difference
    va, vb = dict(a.episode_scores), dict(b.episode_scores)
    deltas = {i: va[i] - vb[i] for i in va}
    delta, fam, _ = _hierarchical(a.layout, dict(deltas))
    seed = bootstrap_seed(
        {
            "domain": "emberbench/scoring/paired-ci",
            "scoring_version": a.scoring_version,
            "constants_digest": a.constants.digest(),
            "evaluation_manifest_hash": a.evaluation_manifest_hash,
            "models": [a.model_id, b.model_id],
            "deltas": [[i, str(deltas[i])] for i in sorted(deltas)],
        }
    )
    ci = _stratified_ci(a.layout, deltas, seed, resamples, "family_balanced_mean_delta")
    if all(d == 0 for d in deltas.values()):
        outcome = ComparisonOutcome.EXACT_TIE
    elif ci is None or delta is None:
        outcome = ComparisonOutcome.INCONCLUSIVE
    elif ci.lower > 0 and delta >= margin:
        outcome = ComparisonOutcome.A_BETTER
    elif ci.upper < 0 and -delta >= margin:
        outcome = ComparisonOutcome.B_BETTER
    else:
        outcome = ComparisonOutcome.INCONCLUSIVE
    return ScoreComparison(
        model_a=a.model_id,
        model_b=b.model_id,
        score_a=a.score,
        score_b=b.score,
        delta=delta,
        ci=ci,
        practical_difference=margin,
        outcome=outcome,
        episodes=len(deltas),
        a_higher=sum(1 for d in deltas.values() if d > 0),
        b_higher=sum(1 for d in deltas.values() if d < 0),
        equal=sum(1 for d in deltas.values() if d == 0),
        family_deltas=tuple(fam.items()),
    )


def _desc(value: Fraction | None) -> Fraction:
    return Fraction(1) if value is None else -value  # every defined value is >= 0, so undefined sorts last


def leaderboard_key(report: ScoreReport) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, str]:
    """Sort key: score to 0.1 points (desc), Safety (desc), avoidable casualties and stranded per trial (asc), worst scenario (desc), model ID.

    Safety decides every tie of the displayed score, so a less safe model
    never ranks above a safer one with the same displayed score. Order is not
    significance: use :func:`compare_scores` for that.
    """
    rounded = None if report.score is None else Fraction(round(report.score * 10), 10)
    t = report.totals
    return (
        _desc(rounded),
        _desc(report.safety),
        t.per_trial(t.avoidable_casualties),
        t.per_trial(t.avoidable_stranded),
        _desc(report.robustness.worst_scenario),
        report.model_id,
    )


def leaderboard_order(reports: Iterable[ScoreReport]) -> list[ScoreReport]:
    """Reports in leaderboard order. Every report must use the same scoring version, constants, manifest and pack."""
    items = list(reports)
    if any(not isinstance(r, ScoreReport) for r in items):
        raise TypeError("reports must be ScoreReport objects")
    if len({(r.scoring_version, r.constants.digest(), r.evaluation_manifest_hash, r.pack_digest) for r in items}) > 1:
        raise _fail("a leaderboard needs reports with the same scoring version, constants, manifest and pack")
    return sorted(items, key=leaderboard_key)


# --------------------------------------------------------------------------- operational view


def inference_config_digest(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a model's pinned inference configuration (model, version, sampling, prompt)."""
    return sha256_hex(dict(config))


@dataclass(frozen=True)
class OperationalRecord:
    """Evaluator-measured operational facts of one submission. Never an input to the EMBER Score.

    Every value is measured or configured by the evaluator, never taken from a
    provider's own report of success, accuracy or safety. ``None`` means not
    measured (for example, tokens of a local policy).
    """

    model_id: str
    provider_id: str
    inference_config_digest: str
    trials: int
    episodes: int  # scheduled episode-trials, including failed ones
    latency_seconds: float  # total wall-clock time of every decision call
    input_tokens: int | None
    output_tokens: int | None
    cost: Fraction | None
    cost_unit: str | None
    retries: int
    timeouts: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.provider_id:
            raise _fail("model_id and provider_id must be non-empty")
        if len(self.inference_config_digest) != 64 or any(c not in "0123456789abcdef" for c in self.inference_config_digest):
            raise _fail("inference_config_digest must be a lowercase SHA-256 hex digest")
        counts = (self.trials, self.episodes, self.retries, self.timeouts, self.input_tokens or 0, self.output_tokens or 0)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts) or self.trials < 1:
            raise _fail("operational counts must be non-negative integers and trials >= 1")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise _fail("latency_seconds must be finite and non-negative")
        if (self.cost is None) != (self.cost_unit is None) or (self.cost is not None and self.cost < 0):
            raise _fail("cost and cost_unit go together, and cost is non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "inference_config_digest": self.inference_config_digest,
            "trials": self.trials,
            "episodes": self.episodes,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": None if self.cost is None else str(self.cost),
            "cost_unit": self.cost_unit,
            "retries": self.retries,
            "timeouts": self.timeouts,
        }

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class EfficiencyView:
    """EMBER-Efficiency: an optional view that sets the score beside a decision cost. Never part of the score."""

    model_id: str
    score: Fraction | None
    cost_per_episode: Fraction
    unit: str  # for example "USD", "output tokens" or "seconds", measured by the evaluator
    points_per_unit: Fraction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "score": _out(self.score),
            "cost_per_episode": float(self.cost_per_episode),
            "unit": self.unit,
            "points_per_unit": _out(self.points_per_unit),
        }


def efficiency_view(report: ScoreReport, cost_per_episode: Fraction | int, unit: str) -> EfficiencyView:
    """Points per unit of evaluator-measured decision cost. Report it next to the score, never instead of it."""
    if isinstance(cost_per_episode, bool) or not isinstance(cost_per_episode, (Fraction, int)):
        raise TypeError("cost_per_episode must be an int or a Fraction")
    cost = Fraction(cost_per_episode)
    if cost < 0 or not isinstance(unit, str) or not unit:
        raise ValueError("cost_per_episode must be non-negative and unit must be a non-empty string")
    per_unit = None if report.score is None or cost == 0 else report.score / cost
    return EfficiencyView(report.model_id, report.score, cost, unit, per_unit)


# --------------------------------------------------------------------------- text summary


def _fmt(value: Fraction | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def format_summary(report: ScoreReport) -> str:
    """A plain-text summary. The headline is in benchmark-performance points, never a percentage of people."""
    r = report
    rb, t = r.robustness, r.totals
    lines = [f"EMBER Score {_fmt(r.score)} / 100 benchmark-performance points under EMBER scoring version {r.scoring_version}"]
    if r.ci is not None:
        lines.append(f"95% CI [{_fmt(r.ci.lower)}, {_fmt(r.ci.upper)}] (stratified two-stage bootstrap, {r.ci.resamples} resamples; this sample only)")
    else:
        lines.append("95% CI n/a (fewer than 5 applicable episodes)")
    status = f"official; environment pack {r.pack_id} sha256:{r.pack_digest}" if r.official else "UNOFFICIAL: not scored against an environment pack with the official constants"
    lines += [
        f"Model {r.model_id}; trials {r.trials}; {status}",
        f"Safety factor {_fmt(r.safety, 3)}; quality factor {_fmt(r.quality, 3)}",
        "Components (0-1, defined/total): " + "; ".join(f"{c.name} {_fmt(c.value, 3)} ({c.defined}/{c.total})" for c in r.components),
        "Primary families: " + "; ".join(f"{g.name} {_fmt(g.score)}" for g in r.families),
        f"Tail: median scenario {_fmt(rb.median_scenario)}; worst scenario {_fmt(rb.worst_scenario)} ({rb.worst_scenario_id});"
        f" p10 episode {_fmt(rb.p10_episode)}; CVaR10 {_fmt(rb.cvar10_episode)}",
        f"Zero-score episode-trials {rb.zero_score_episode_trials} of {len(r.episodes)}: catastrophic {rb.catastrophic_episode_trials}; "
        + "; ".join(f"{k} {n}" for k, n in rb.failed_episode_trials),
        "Severe quality failures (episode-trials): " + "; ".join(f"{k} {n}" for k, n in rb.severe_episode_trials),
        f"People per trial: escapable {_fmt(t.per_trial(t.escapable))}; evacuated {_fmt(t.per_trial(t.evacuated))};"
        f" avoidable casualties {_fmt(t.per_trial(t.avoidable_casualties))}; avoidable stranded {_fmt(t.per_trial(t.avoidable_stranded))}",
        "The EMBER Score is not a percentage of people evacuated.",
    ]
    return "\n".join(lines) + "\n"
