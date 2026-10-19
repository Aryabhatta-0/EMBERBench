"""Deterministic aggregate and paired statistics over authenticated evaluation results.

The entry points are :func:`aggregate_result` (one model) and
:func:`compare_results` (two models under the same evaluation manifest). Both
return immutable, versioned artifacts with a canonical serialization and a
SHA-256 digest. The full definitions are in ``docs/statistics.md``.

Rules for this module:

- **Fail closed.** An :class:`~emberbench.evaluation.EvaluationResult` is not
  trusted as given. Its manifest must target the suite, every episode must
  match its manifest entry, every run record is authenticated by replay and
  the stored metrics must equal the recomputed metrics exactly.
- **No imputation.** A scheduled episode with no result is *missing*. A
  metric that is undefined for an episode (``None``) is *undefined*. Both are
  counted and reported. Neither is replaced by a value or silently dropped.
- **Exact arithmetic.** Per-episode values are rebuilt as exact
  :class:`fractions.Fraction` values from the integer counts in the metrics.
  Means, medians, variances, quantiles, deltas and bootstrap means are exact.
  Values become ``float`` only in ``to_dict``. The standard deviation is the
  one value that is not rational; it is ``math.sqrt`` of the float variance.
- **Deterministic bootstrap.** Confidence intervals come from a percentile
  bootstrap whose random stream is SHAKE-256 of the input artifact digests and
  the group and metric names. It never uses :mod:`random`, the time or OS
  entropy, and it is separate from the simulator's SplitMix64 streams.
- **No score.** There is no EMBER score, composite, ranking or global winner.
  Paired win/loss/tie counts exist per metric only, and only for metrics with
  an explicit optimization direction.
"""

import hashlib
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any, TypeVar

from .canonical import canonical_bytes, canonical_json, sha256_hex
from .errors import EmberBenchError, EvaluationError, StatisticsInputError
from .evaluation import (
    EpisodeKey,
    EpisodeSpec,
    EvaluationManifest,
    EvaluationResult,
    pair_results,
    resolve_scenario,
    verify_manifest,
)
from .metrics import EpisodeMetrics, compute_episode_metrics
from .state import TerminationReason
from .suite import ScenarioFamily, Suite

# Bump this whenever any statistic, its serialization or the bootstrap stream changes.
STATISTICS_VERSION = "1.0"

# Fixed quantile levels. Interpolation is Hyndman-Fan type 7; see quantile().
QUANTILE_LEVELS: tuple[tuple[str, Fraction], ...] = (
    ("p25", Fraction(1, 4)),
    ("p50", Fraction(1, 2)),
    ("p75", Fraction(3, 4)),
    ("p90", Fraction(9, 10)),
    ("p95", Fraction(19, 20)),
)

CONFIDENCE_LEVEL = Fraction(95, 100)
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
MIN_BOOTSTRAP_RESAMPLES = 100
MAX_BOOTSTRAP_RESAMPLES = 1_000_000
# A confidence interval is None below this many observations.
MIN_BOOTSTRAP_SAMPLE_SIZE = 5
BOOTSTRAP_METHOD = "percentile_bootstrap"

_STREAM_DOMAIN = b"emberbench/statistics/bootstrap/v1\x00"
_STREAM_CHUNK_WORDS = 512
_TWO_POW_64 = 1 << 64


def _fail(message: str) -> StatisticsInputError:
    return StatisticsInputError(message)


def _out(value: Fraction | None) -> float | None:
    """The documented output boundary: an exact value becomes its correctly rounded float."""
    return None if value is None else float(value)


# --------------------------------------------------------------------------- metric definitions


class MetricDirection(StrEnum):
    """How a paired difference is read. ``descriptive`` metrics have no winner."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DESCRIPTIVE = "descriptive"


class MetricKind(StrEnum):
    """What a metric measures. Proxies and descriptive fields are labelled as such."""

    OUTCOME = "outcome"  # terminal safety or time outcome
    BEHAVIOUR = "behaviour"  # observed movement behaviour
    PROXY = "proxy"  # an approximation of a quantity that is not measured directly
    DESCRIPTIVE = "descriptive"  # describes the run; not an objective


@dataclass(frozen=True)
class MetricDefinition:
    """One supported scalar metric: where it comes from, how it is read and how it is rebuilt exactly."""

    name: str
    source: str  # the EpisodeMetrics field, as "section.field"
    kind: MetricKind
    direction: MetricDirection
    definition: str
    exact: Callable[[EpisodeMetrics], Fraction | None]
    stored: Callable[[EpisodeMetrics], float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "kind": self.kind.value,
            "direction": self.direction.value,
            "definition": self.definition,
        }


def _frac(numerator: int, denominator: int) -> Fraction | None:
    return None if denominator == 0 else Fraction(numerator, denominator)


def _exposure_total(m: EpisodeMetrics) -> Fraction:
    total = Fraction(0)
    for distance, count in m.fire_exposure.distance_histogram:
        if distance < 1:
            raise _fail(f"{m.scenario_id}: fire distance histogram has distance {distance} < 1")
        total += Fraction(count, distance)
    return total


def _exit_counts(m: EpisodeMetrics) -> list[int]:
    return [n for _, n in m.exits.evacuations_per_exit]


def _max_exit_share(m: EpisodeMetrics) -> Fraction | None:
    counts = _exit_counts(m)
    return _frac(max(counts, default=0), sum(counts))


def _exit_balance(m: EpisodeMetrics) -> Fraction | None:
    counts = _exit_counts(m)
    evacuated, k = sum(counts), len(counts)
    if evacuated == 0 or k < 2:
        return None
    gini_simpson = 1 - sum(Fraction(n, evacuated) ** 2 for n in counts)
    return gini_simpson / (1 - Fraction(1, k))


_H, _L, _D = MetricDirection.HIGHER_IS_BETTER, MetricDirection.LOWER_IS_BETTER, MetricDirection.DESCRIPTIVE

METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "evacuated_rate", "safety.evacuated_rate", MetricKind.OUTCOME, _H,
        "evacuated / population",
        lambda m: Fraction(m.safety.evacuated, m.safety.population),
        lambda m: m.safety.evacuated_rate,
    ),
    MetricDefinition(
        "casualty_rate", "safety.casualty_rate", MetricKind.OUTCOME, _L,
        "casualties / population",
        lambda m: Fraction(m.safety.casualties, m.safety.population),
        lambda m: m.safety.casualty_rate,
    ),
    MetricDefinition(
        "stranded_rate", "safety.stranded_rate", MetricKind.OUTCOME, _L,
        "stranded / population (still active at termination)",
        lambda m: Fraction(m.safety.stranded, m.safety.population),
        lambda m: m.safety.stranded_rate,
    ),
    MetricDefinition(
        "terminal_trapped_rate", "safety.terminal_trapped_rate", MetricKind.OUTCOME, _L,
        "terminal_trapped_count / population",
        lambda m: Fraction(m.safety.terminal_trapped_count, m.safety.population),
        lambda m: m.safety.terminal_trapped_rate,
    ),
    MetricDefinition(
        "mean_person_time_to_safety", "time.mean_person_time_to_safety", MetricKind.OUTCOME, _L,
        "person_time_to_safety_total / population; non-evacuees are censored at max_steps",
        lambda m: Fraction(m.time.person_time_to_safety_total, m.safety.population),
        lambda m: m.time.mean_person_time_to_safety,
    ),
    MetricDefinition(
        "blocked_rate", "movement.blocked_rate", MetricKind.BEHAVIOUR, _L,
        "blocked_moves / movement_attempts; undefined with no attempts",
        lambda m: _frac(m.movement.blocked_moves, m.movement.movement_attempts),
        lambda m: m.movement.blocked_rate,
    ),
    MetricDefinition(
        "congestion_block_rate", "movement.congestion_block_rate", MetricKind.BEHAVIOUR, _L,
        "congestion_blocks / movement_attempts; undefined with no attempts",
        lambda m: _frac(m.movement.congestion_blocks, m.movement.movement_attempts),
        lambda m: m.movement.congestion_block_rate,
    ),
    MetricDefinition(
        "aggregate_path_efficiency", "path_efficiency.aggregate_path_efficiency", MetricKind.PROXY, _H,
        "total_static_lower_bound / total_cells_moved over measured evacuees; undefined with no movement",
        lambda m: _frac(m.path_efficiency.total_static_lower_bound, m.path_efficiency.total_cells_moved),
        lambda m: m.path_efficiency.aggregate_path_efficiency,
    ),
    MetricDefinition(
        "mean_exposure", "fire_exposure.mean_exposure", MetricKind.PROXY, _L,
        "sum over fire distance histogram of count / d, divided by active_person_states",
        lambda m: _exposure_total(m) / m.fire_exposure.active_person_states,
        lambda m: m.fire_exposure.mean_exposure,
    ),
    MetricDefinition(
        "exits_used", "exits.exits_used", MetricKind.DESCRIPTIVE, _D,
        "number of exits with at least one evacuation",
        lambda m: Fraction(sum(1 for n in _exit_counts(m) if n)),
        lambda m: m.exits.exits_used,
    ),
    MetricDefinition(
        "max_exit_share", "exits.max_exit_share", MetricKind.DESCRIPTIVE, _D,
        "max exit evacuations / evacuations; undefined with no evacuations",
        _max_exit_share,
        lambda m: m.exits.max_exit_share,
    ),
    MetricDefinition(
        "exit_balance", "exits.exit_balance", MetricKind.DESCRIPTIVE, _D,
        "normalized Gini-Simpson index of exit shares; undefined with one exit or no evacuations",
        _exit_balance,
        lambda m: m.exits.exit_balance,
    ),
)

METRIC_NAMES: tuple[str, ...] = tuple(d.name for d in METRIC_DEFINITIONS)


def metric_definition(name: str) -> MetricDefinition:
    for d in METRIC_DEFINITIONS:
        if d.name == name:
            return d
    raise KeyError(name)


def _exact_values(index: int, m: EpisodeMetrics) -> tuple[Fraction | None, ...]:
    """Rebuild every supported metric exactly and require it to match the stored float."""
    values = []
    for d in METRIC_DEFINITIONS:
        exact = d.exact(m)
        stored = d.stored(m)
        if (exact is None) != (stored is None) or (exact is not None and float(exact) != stored):
            raise _fail(f"episode {index}: {d.source} is {stored!r}, its exact definition gives {_out(exact)!r}")
        values.append(exact)
    return tuple(values)


# --------------------------------------------------------------------------- descriptive statistics


def quantile(ordered: Sequence[Fraction], p: Fraction) -> Fraction | None:
    """Hyndman-Fan type 7 quantile of an ascending sequence, computed exactly.

    With ``n`` values ``x[0] <= ... <= x[n-1]``, ``h = (n - 1) * p``,
    ``j = floor(h)`` and ``g = h - j``: ``Q(p) = x[j] + g * (x[j+1] - x[j])``
    (just ``x[j]`` when ``g == 0``). ``None`` for an empty sequence.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"quantile level must be in [0, 1], got {p}")
    n = len(ordered)
    if n == 0:
        return None
    h = (n - 1) * Fraction(p)
    j = h.numerator // h.denominator
    g = h - j
    if g == 0:
        return ordered[j]
    return ordered[j] + g * (ordered[j + 1] - ordered[j])


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Exact summary of ``n`` observed values. Every field is ``None`` where it is undefined."""

    n: int
    mean: Fraction | None  # n >= 1
    median: Fraction | None  # n >= 1; equals the p50 quantile
    variance: Fraction | None  # sample variance, denominator n - 1; n >= 2
    std: float | None  # sqrt(float(variance)); n >= 2
    minimum: Fraction | None
    maximum: Fraction | None
    quantiles: tuple[tuple[str, Fraction | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": _out(self.mean),
            "median": _out(self.median),
            "variance": _out(self.variance),
            "std": self.std,
            "min": _out(self.minimum),
            "max": _out(self.maximum),
            "quantiles": {label: _out(q) for label, q in self.quantiles},
        }


def describe(values: Sequence[Fraction]) -> DescriptiveStatistics:
    """Mean, median, sample variance and std, min, max and fixed quantiles of exact values."""
    xs = [Fraction(v) for v in values]
    n = len(xs)
    ordered = sorted(xs)
    mean = sum(xs, Fraction(0)) / n if n else None
    variance = None
    if n >= 2 and mean is not None:
        variance = sum(((x - mean) ** 2 for x in xs), Fraction(0)) / (n - 1)
    return DescriptiveStatistics(
        n=n,
        mean=mean,
        median=quantile(ordered, Fraction(1, 2)),
        variance=variance,
        std=None if variance is None else math.sqrt(float(variance)),
        minimum=ordered[0] if n else None,
        maximum=ordered[-1] if n else None,
        quantiles=tuple((label, quantile(ordered, p)) for label, p in QUANTILE_LEVELS),
    )


# --------------------------------------------------------------------------- bootstrap


def check_resamples(resamples: object) -> int:
    if not isinstance(resamples, int) or isinstance(resamples, bool):
        raise TypeError(f"resamples must be an int, got {type(resamples).__name__}")
    if not MIN_BOOTSTRAP_RESAMPLES <= resamples <= MAX_BOOTSTRAP_RESAMPLES:
        raise ValueError(f"resamples must be in [{MIN_BOOTSTRAP_RESAMPLES}, {MAX_BOOTSTRAP_RESAMPLES}], got {resamples}")
    return resamples


class BootstrapStream:
    """A counter-mode SHAKE-256 stream of unsigned 64-bit words.

    Block ``k`` is ``SHAKE-256(domain || seed || k as 8 big-endian bytes)``,
    read as ``_STREAM_CHUNK_WORDS`` big-endian words. The stream is fully
    specified here, so it is identical on every platform and Python version.
    It shares nothing with the simulator's random streams.
    """

    __slots__ = ("_seed", "_counter", "_words", "_pos")

    def __init__(self, seed: bytes) -> None:
        if not isinstance(seed, bytes) or len(seed) != 32:
            raise ValueError("seed must be 32 bytes")
        self._seed = seed
        self._counter = 0
        self._words: tuple[int, ...] = ()
        self._pos = 0

    def next_u64(self) -> int:
        if self._pos == len(self._words):
            block = hashlib.shake_256(_STREAM_DOMAIN + self._seed + self._counter.to_bytes(8, "big"))
            self._words = struct.unpack(f">{_STREAM_CHUNK_WORDS}Q", block.digest(8 * _STREAM_CHUNK_WORDS))
            self._counter += 1
            self._pos = 0
        word = self._words[self._pos]
        self._pos += 1
        return word

    def below(self, n: int) -> int:
        """A uniform integer in ``[0, n)`` by rejection sampling, so there is no modulo bias."""
        if n < 1:
            raise ValueError("n must be positive")
        limit = _TWO_POW_64 - _TWO_POW_64 % n
        while True:
            word = self.next_u64()
            if word < limit:
                return word % n


def bootstrap_seed(material: dict[str, Any]) -> bytes:
    """The 32-byte stream seed: SHA-256 of the canonical JSON of ``material``."""
    return hashlib.sha256(canonical_bytes(material)).digest()


@dataclass(frozen=True)
class ConfidenceInterval:
    """A percentile-bootstrap interval for a mean. It describes this evaluation sample only."""

    statistic: str  # "mean" or "mean_delta"
    level: Fraction
    resamples: int
    seed: str  # hex of the 32-byte stream seed, for audit
    lower: Fraction
    upper: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": BOOTSTRAP_METHOD,
            "statistic": self.statistic,
            "level": float(self.level),
            "resamples": self.resamples,
            "seed": self.seed,
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


def bootstrap_mean_ci(values: Sequence[Fraction], seed: bytes, resamples: int, *, statistic: str = "mean") -> ConfidenceInterval | None:
    """Percentile bootstrap interval for the mean of ``values``; ``None`` below the minimum sample size.

    Each of ``resamples`` resamples draws ``n`` indices uniformly with
    replacement from ``seed``'s stream. The bounds are the type-7 quantiles
    at ``(1 - level) / 2`` and ``(1 + level) / 2`` of the sorted resample means.
    """
    check_resamples(resamples)
    xs = [Fraction(v) for v in values]
    n = len(xs)
    if n < MIN_BOOTSTRAP_SAMPLE_SIZE:
        return None
    # Put every value over one common denominator, so each resample sum is an integer sum.
    denominator = math.lcm(*(x.denominator for x in xs))
    numerators = [x.numerator * (denominator // x.denominator) for x in xs]
    stream = BootstrapStream(seed)
    sums = sorted(sum(numerators[stream.below(n)] for _ in range(n)) for _ in range(resamples))
    means = [Fraction(s, n * denominator) for s in sums]
    alpha = 1 - CONFIDENCE_LEVEL
    lower, upper = quantile(means, alpha / 2), quantile(means, 1 - alpha / 2)
    assert lower is not None and upper is not None
    return ConfidenceInterval(statistic, CONFIDENCE_LEVEL, resamples, seed.hex(), lower, upper)


# --------------------------------------------------------------------------- authenticated input


@dataclass(frozen=True)
class _Observation:
    """One authenticated episode, reduced to the values this module uses."""

    index: int
    termination_reason: TerminationReason
    evacuated: int
    casualties: int
    stranded: int
    values: tuple[Fraction | None, ...]  # aligned with METRIC_DEFINITIONS


def _authenticate(result: EvaluationResult, suite: Suite) -> dict[int, _Observation]:
    """Check ``result`` against ``suite`` from scratch and return its observations by episode index."""
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    if not isinstance(suite, Suite):
        raise TypeError("suite must be a Suite")
    try:
        manifest = EvaluationManifest.from_dict(result.manifest.to_dict())
        if manifest != result.manifest:
            raise _fail("manifest does not round-trip")
        verify_manifest(manifest, suite)
        # Construction repeats the structural checks: order, duplicates and record headers.
        checked = EvaluationResult(result.result_version, result.model_id, manifest, result.episodes)
    except StatisticsInputError:
        raise
    except (EvaluationError, TypeError, ValueError, AttributeError) as exc:
        raise _fail(f"result for {getattr(result, 'model_id', None)!r} is not valid: {exc}") from exc

    observations: dict[int, _Observation] = {}
    for e in checked.episodes:
        spec = e.episode
        where = f"episode {spec.index}"
        try:
            scenario = resolve_scenario(suite, spec)
            metrics = compute_episode_metrics(scenario, e.record)
        except (EmberBenchError, TypeError, ValueError, AttributeError, KeyError, IndexError) as exc:
            raise _fail(f"{where}: run record failed authentication: {exc}") from exc
        try:
            stored_ok = isinstance(e.metrics, EpisodeMetrics) and canonical_json(e.metrics.to_dict()) == canonical_json(metrics.to_dict())
        except (TypeError, ValueError, AttributeError) as exc:
            raise _fail(f"{where}: stored metrics are malformed: {exc}") from exc
        if not stored_ok:
            raise _fail(f"{where}: stored metrics differ from the metrics of the authenticated record")
        observations[spec.index] = _Observation(
            index=spec.index,
            termination_reason=metrics.termination_reason,
            evacuated=metrics.safety.evacuated,
            casualties=metrics.safety.casualties,
            stranded=metrics.safety.stranded,
            values=_exact_values(spec.index, metrics),
        )
    return observations


def _result_digest(result: EvaluationResult) -> str:
    """SHA-256 of the canonical per-model artifact, as saved by ``save_evaluation_result``."""
    return sha256_hex(result.to_dict())


# --------------------------------------------------------------------------- groups


class GroupKind(StrEnum):
    OVERALL = "overall"
    SCENARIO = "scenario"
    FAMILY = "family"


@dataclass(frozen=True)
class _Group:
    kind: GroupKind
    name: str
    specs: tuple[EpisodeSpec, ...]  # in manifest order


def _groups(manifest: EvaluationManifest, suite: Suite) -> list[_Group]:
    """Overall, then scenarios in suite order, then families in ScenarioFamily order.

    A scenario with several families belongs to every one of them.
    """
    groups = [_Group(GroupKind.OVERALL, "overall", manifest.episodes)]
    scheduled = {spec.scenario_id for spec in manifest.episodes}
    entries = [e for e in suite.manifest.entries if e.scenario_id in scheduled]
    for entry in entries:
        specs = tuple(s for s in manifest.episodes if s.scenario_id == entry.scenario_id)
        groups.append(_Group(GroupKind.SCENARIO, entry.scenario_id, specs))
    for family in ScenarioFamily:
        members = {e.scenario_id for e in entries if family in e.families}
        if members:
            specs = tuple(s for s in manifest.episodes if s.scenario_id in members)
            groups.append(_Group(GroupKind.FAMILY, family.value, specs))
    return groups


def _header(manifest: EvaluationManifest) -> dict[str, Any]:
    return {
        "evaluation_manifest_hash": manifest.manifest_hash(),
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "suite_hash": manifest.suite_hash,
        "simulator_version": manifest.simulator_version,
        "observation_version": manifest.observation_version,
        "metrics_version": manifest.metrics_version,
    }


def _methods(resamples: int) -> dict[str, Any]:
    return {
        "quantiles": {
            "rule": "hyndman_fan_type_7",
            "levels": {label: float(p) for label, p in QUANTILE_LEVELS},
        },
        "variance": "sample variance, sum((x - mean)^2) / (n - 1)",
        "std": "sqrt of the sample variance",
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "level": float(CONFIDENCE_LEVEL),
            "resamples": resamples,
            "min_sample_size": MIN_BOOTSTRAP_SAMPLE_SIZE,
            "stream": "shake_256 counter mode",
        },
    }


# --------------------------------------------------------------------------- aggregate

_G = TypeVar("_G", "AggregateGroup", "PairedGroup")


@dataclass(frozen=True)
class RobustnessCounts:
    """Raw episode counts. A missing episode is reported as missing, not as a failure."""

    scheduled: int
    observed: int
    missing: int
    all_resolved: int
    max_steps: int
    with_casualties: int
    with_stranded: int
    zero_evacuation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled": self.scheduled,
            "observed": self.observed,
            "missing": self.missing,
            "all_resolved": self.all_resolved,
            "max_steps": self.max_steps,
            "with_casualties": self.with_casualties,
            "with_stranded": self.with_stranded,
            "zero_evacuation": self.zero_evacuation,
        }


@dataclass(frozen=True)
class MetricSummary:
    """One metric over one group. ``statistics.n == observed - undefined``."""

    metric: str
    scheduled: int
    observed: int  # episodes with a result
    missing: int  # scheduled episodes with no result
    undefined: int  # episodes with a result for which this metric is None
    statistics: DescriptiveStatistics
    mean_ci: ConfidenceInterval | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "scheduled": self.scheduled,
            "observed": self.observed,
            "missing": self.missing,
            "undefined": self.undefined,
            "statistics": self.statistics.to_dict(),
            "mean_ci": None if self.mean_ci is None else self.mean_ci.to_dict(),
        }


@dataclass(frozen=True)
class AggregateGroup:
    kind: GroupKind
    name: str
    episode_indices: tuple[int, ...]  # scheduled episodes in this group, in manifest order
    missing_indices: tuple[int, ...]
    robustness: RobustnessCounts
    metrics: tuple[MetricSummary, ...]  # in METRIC_DEFINITIONS order

    def metric(self, name: str) -> MetricSummary:
        for m in self.metrics:
            if m.metric == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "episode_indices": list(self.episode_indices),
            "missing_indices": list(self.missing_indices),
            "robustness": self.robustness.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass(frozen=True)
class AggregateStatistics:
    """The versioned per-model statistics artifact. It holds no score and no ranking."""

    statistics_version: str
    model_id: str
    manifest: EvaluationManifest
    result_digest: str  # SHA-256 of the canonical input EvaluationResult
    resamples: int
    missing_episodes: tuple[EpisodeKey, ...]
    overall: AggregateGroup
    scenarios: tuple[AggregateGroup, ...]  # suite order
    families: tuple[AggregateGroup, ...]  # ScenarioFamily order

    def scenario(self, scenario_id: str) -> AggregateGroup:
        return _find(self.scenarios, scenario_id)

    def family(self, family: ScenarioFamily | str) -> AggregateGroup:
        return _find(self.families, str(family))

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "artifact": "aggregate",
            "model_id": self.model_id,
            **_header(self.manifest),
            "result_digest": self.result_digest,
            "scheduled_episodes": len(self.manifest.episodes),
            "observed_episodes": len(self.manifest.episodes) - len(self.missing_episodes),
            "missing_episodes": [k.to_dict() for k in self.missing_episodes],
            "metric_definitions": [d.to_dict() for d in METRIC_DEFINITIONS],
            "methods": _methods(self.resamples),
            "overall": self.overall.to_dict(),
            "scenarios": [g.to_dict() for g in self.scenarios],
            "families": [g.to_dict() for g in self.families],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


def _find(groups: Sequence[_G], name: str) -> _G:
    for g in groups:
        if g.name == name:
            return g
    raise KeyError(name)


def aggregate_result(result: EvaluationResult, suite: Suite, *, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES) -> AggregateStatistics:
    """Summarize one model's authenticated result: overall, per scenario and per family."""
    check_resamples(resamples)
    observations = _authenticate(result, suite)
    manifest = result.manifest
    manifest_hash = manifest.manifest_hash()
    digest = _result_digest(result)

    built: list[AggregateGroup] = []
    for group in _groups(manifest, suite):
        present = [observations[s.index] for s in group.specs if s.index in observations]
        missing = tuple(s.index for s in group.specs if s.index not in observations)
        robustness = RobustnessCounts(
            scheduled=len(group.specs),
            observed=len(present),
            missing=len(missing),
            all_resolved=sum(1 for o in present if o.termination_reason is TerminationReason.ALL_RESOLVED),
            max_steps=sum(1 for o in present if o.termination_reason is TerminationReason.MAX_STEPS),
            with_casualties=sum(1 for o in present if o.casualties > 0),
            with_stranded=sum(1 for o in present if o.stranded > 0),
            zero_evacuation=sum(1 for o in present if o.evacuated == 0),
        )
        summaries = []
        for position, d in enumerate(METRIC_DEFINITIONS):
            values = [v for o in present if (v := o.values[position]) is not None]
            seed = bootstrap_seed(
                {
                    "statistics_version": STATISTICS_VERSION,
                    "purpose": "aggregate_mean",
                    "evaluation_manifest_hash": manifest_hash,
                    "result_digests": [digest],
                    "group": [group.kind.value, group.name],
                    "metric": d.name,
                    "resamples": resamples,
                }
            )
            summaries.append(
                MetricSummary(
                    metric=d.name,
                    scheduled=len(group.specs),
                    observed=len(present),
                    missing=len(missing),
                    undefined=len(present) - len(values),
                    statistics=describe(values),
                    mean_ci=bootstrap_mean_ci(values, seed, resamples),
                )
            )
        built.append(
            AggregateGroup(group.kind, group.name, tuple(s.index for s in group.specs), missing, robustness, tuple(summaries))
        )

    return AggregateStatistics(
        statistics_version=STATISTICS_VERSION,
        model_id=result.model_id,
        manifest=manifest,
        result_digest=digest,
        resamples=resamples,
        missing_episodes=tuple(s.key(manifest_hash) for s in manifest.episodes if s.index not in observations),
        overall=built[0],
        scenarios=tuple(g for g in built if g.kind is GroupKind.SCENARIO),
        families=tuple(g for g in built if g.kind is GroupKind.FAMILY),
    )


# --------------------------------------------------------------------------- paired comparison


@dataclass(frozen=True)
class PairedMetricComparison:
    """Per-episode differences ``A - B`` for one metric over exactly paired episodes.

    ``a_better``, ``b_better`` and ``ties`` follow the metric's direction. They
    are ``None`` for a descriptive metric, which has no winner. The sign
    counts describe ``A - B`` only and carry no judgement.
    """

    metric: str
    direction: MetricDirection
    paired: int  # episodes completed by both models
    missing_from_a: int
    missing_from_b: int
    undefined: int  # paired episodes where the metric is None for A or for B
    deltas: tuple[tuple[int, Fraction], ...]  # (episode index, A - B), in manifest order
    statistics: DescriptiveStatistics  # of the deltas
    mean_delta_ci: ConfidenceInterval | None
    delta_positive: int
    delta_negative: int
    delta_zero: int
    a_better: int | None
    b_better: int | None
    ties: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "paired": self.paired,
            "missing_from_a": self.missing_from_a,
            "missing_from_b": self.missing_from_b,
            "undefined": self.undefined,
            "deltas": [{"index": i, "delta": float(d)} for i, d in self.deltas],
            "statistics": self.statistics.to_dict(),
            "mean_delta_ci": None if self.mean_delta_ci is None else self.mean_delta_ci.to_dict(),
            "delta_positive": self.delta_positive,
            "delta_negative": self.delta_negative,
            "delta_zero": self.delta_zero,
            "a_better": self.a_better,
            "b_better": self.b_better,
            "ties": self.ties,
        }


@dataclass(frozen=True)
class PairedGroup:
    kind: GroupKind
    name: str
    episode_indices: tuple[int, ...]
    paired_indices: tuple[int, ...]
    missing_from_a: tuple[int, ...]
    missing_from_b: tuple[int, ...]
    metrics: tuple[PairedMetricComparison, ...]  # in METRIC_DEFINITIONS order

    def metric(self, name: str) -> PairedMetricComparison:
        for m in self.metrics:
            if m.metric == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "episode_indices": list(self.episode_indices),
            "paired_indices": list(self.paired_indices),
            "missing_from_a": list(self.missing_from_a),
            "missing_from_b": list(self.missing_from_b),
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass(frozen=True)
class PairedStatistics:
    """The versioned paired-comparison artifact. There is no cross-metric winner."""

    statistics_version: str
    model_a: str
    model_b: str
    manifest: EvaluationManifest
    result_digest_a: str
    result_digest_b: str
    resamples: int
    paired: tuple[EpisodeKey, ...]
    missing_from_a: tuple[EpisodeKey, ...]
    missing_from_b: tuple[EpisodeKey, ...]
    overall: PairedGroup
    scenarios: tuple[PairedGroup, ...]
    families: tuple[PairedGroup, ...]

    def scenario(self, scenario_id: str) -> PairedGroup:
        return _find(self.scenarios, scenario_id)

    def family(self, family: ScenarioFamily | str) -> PairedGroup:
        return _find(self.families, str(family))

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics_version": self.statistics_version,
            "artifact": "paired",
            "model_a": self.model_a,
            "model_b": self.model_b,
            **_header(self.manifest),
            "result_digest_a": self.result_digest_a,
            "result_digest_b": self.result_digest_b,
            "delta": "A - B",
            "scheduled_episodes": len(self.manifest.episodes),
            "paired_count": len(self.paired),
            "missing_from_a": [k.to_dict() for k in self.missing_from_a],
            "missing_from_b": [k.to_dict() for k in self.missing_from_b],
            "metric_definitions": [d.to_dict() for d in METRIC_DEFINITIONS],
            "methods": _methods(self.resamples),
            "overall": self.overall.to_dict(),
            "scenarios": [g.to_dict() for g in self.scenarios],
            "families": [g.to_dict() for g in self.families],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return sha256_hex(self.to_dict())


def _wins(direction: MetricDirection, deltas: Sequence[Fraction]) -> tuple[int | None, int | None, int | None]:
    if direction is MetricDirection.DESCRIPTIVE:
        return None, None, None
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - positive - negative
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return positive, negative, ties
    return negative, positive, ties


def compare_results(
    a: EvaluationResult, b: EvaluationResult, suite: Suite, *, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
) -> PairedStatistics:
    """Compare two authenticated results of one manifest, metric by metric, on exactly paired episodes."""
    check_resamples(resamples)
    pairing = pair_results(a, b)  # rejects different manifests before any replay
    obs_a = _authenticate(a, suite)
    obs_b = _authenticate(b, suite)
    manifest = a.manifest
    manifest_hash = manifest.manifest_hash()
    digests = [_result_digest(a), _result_digest(b)]
    paired_set = {k.index for k in pairing.paired}

    built: list[PairedGroup] = []
    for group in _groups(manifest, suite):
        indices = tuple(s.index for s in group.specs)
        paired = tuple(i for i in indices if i in paired_set)
        missing_a = tuple(i for i in indices if i not in obs_a)
        missing_b = tuple(i for i in indices if i not in obs_b)
        comparisons = []
        for position, d in enumerate(METRIC_DEFINITIONS):
            deltas: list[tuple[int, Fraction]] = []
            for i in paired:
                va, vb = obs_a[i].values[position], obs_b[i].values[position]
                if va is not None and vb is not None:
                    deltas.append((i, va - vb))
            values = [delta for _, delta in deltas]
            seed = bootstrap_seed(
                {
                    "statistics_version": STATISTICS_VERSION,
                    "purpose": "paired_mean_delta",
                    "evaluation_manifest_hash": manifest_hash,
                    "result_digests": digests,
                    "group": [group.kind.value, group.name],
                    "metric": d.name,
                    "resamples": resamples,
                }
            )
            a_better, b_better, ties = _wins(d.direction, values)
            comparisons.append(
                PairedMetricComparison(
                    metric=d.name,
                    direction=d.direction,
                    paired=len(paired),
                    missing_from_a=len(missing_a),
                    missing_from_b=len(missing_b),
                    undefined=len(paired) - len(deltas),
                    deltas=tuple(deltas),
                    statistics=describe(values),
                    mean_delta_ci=bootstrap_mean_ci(values, seed, resamples, statistic="mean_delta"),
                    delta_positive=sum(1 for v in values if v > 0),
                    delta_negative=sum(1 for v in values if v < 0),
                    delta_zero=sum(1 for v in values if v == 0),
                    a_better=a_better,
                    b_better=b_better,
                    ties=ties,
                )
            )
        built.append(PairedGroup(group.kind, group.name, indices, paired, missing_a, missing_b, tuple(comparisons)))

    return PairedStatistics(
        statistics_version=STATISTICS_VERSION,
        model_a=a.model_id,
        model_b=b.model_id,
        manifest=manifest,
        result_digest_a=digests[0],
        result_digest_b=digests[1],
        resamples=resamples,
        paired=pairing.paired,
        missing_from_a=pairing.missing_from_a,
        missing_from_b=pairing.missing_from_b,
        overall=built[0],
        scenarios=tuple(g for g in built if g.kind is GroupKind.SCENARIO),
        families=tuple(g for g in built if g.kind is GroupKind.FAMILY),
    )
