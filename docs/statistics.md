# Aggregate and paired statistics (statistics version 1.0)

This document defines the statistical summaries that EMBERBench computes over
evaluation results. The code is in
[`src/emberbench/statistics.py`](../src/emberbench/statistics.py). The tests are
in [`tests/test_statistics.py`](../tests/test_statistics.py).

**There is no EMBER score or global winner in this module.** (The EMBER
Score is a separate module: see [scoring.md](scoring.md).) This module does not rank
models, combine metrics or produce a leaderboard. Every number belongs to one
named metric. Win/loss/tie counts exist per metric only.

Out of scope here: leaderboards, a composite score, model rankings, provider
adapters, procedural generation and a CLI.

## 1. Entry points

```python
from emberbench import aggregate_result, compare_results, load_evaluation_result, load_suite

suite = load_suite("scenarios/public/v1")
a = load_evaluation_result("results/model-a.json", suite)
b = load_evaluation_result("results/model-b.json", suite)

stats_a = aggregate_result(a, suite)             # one model
paired = compare_results(a, b, suite)            # A - B, per metric
print(stats_a.digest(), paired.digest())
```

| Function | Returns | Artifact field `artifact` |
|---|---|---|
| `aggregate_result(result, suite, *, resamples=2000)` | `AggregateStatistics` | `"aggregate"` |
| `compare_results(a, b, suite, *, resamples=2000)` | `PairedStatistics` | `"paired"` |

Both artifacts are immutable dataclasses with `to_dict()`, `to_json()`
(canonical JSON) and `digest()` (SHA-256 of the canonical JSON). They carry
`statistics_version = "1.0"`. This version is separate from
`SIMULATOR_VERSION`, `OBSERVATION_VERSION`, `SCHEMA_VERSION`,
`METRICS_VERSION` and the evaluation artifact versions, which this module does
not change. Bump `STATISTICS_VERSION` when any statistic, its serialization or
the bootstrap stream changes.

## 2. Fail-closed input

An `EvaluationResult` in memory is **not trusted as given**. Before any
episode is used, both functions:

1. rebuild the evaluation manifest from its dictionary and require that it is
   equal to the original;
2. call `verify_manifest(manifest, suite)`: the suite ID, version and hash and
   the simulator, observation and metrics versions must match the running code;
3. rebuild the `EvaluationResult`, which repeats its structural checks
   (manifest order, no duplicates, every episode equal to its manifest entry,
   record headers equal to the manifest);
4. for every episode: resolve its scenario from the suite by ID **and** content
   hash, recompute `EpisodeMetrics` from the run record (this authenticates the
   record by simulator replay), and require that the stored metrics equal the
   recomputed metrics exactly (canonical JSON).

Any failure raises `StatisticsInputError` (a subclass of `EvaluationError`).
Nothing is repaired, skipped or partly summarized.

`compare_results` first calls `pair_results(a, b)`. Results from different
evaluation manifests raise `EvaluationError` before any replay.

## 3. Supported metrics and directions

Each metric is a per-episode scalar from `EpisodeMetrics`. The module rebuilds
it as an exact `Fraction` from the integer counts in the metrics (see §10) and
requires that the float of that fraction equals the stored value.

| Metric | Source field | Kind | Direction | Exact definition (per episode) | Undefined when |
|---|---|---|---|---|---|
| `evacuated_rate` | `safety.evacuated_rate` | outcome | higher is better | `evacuated / population` | never |
| `casualty_rate` | `safety.casualty_rate` | outcome | lower is better | `casualties / population` | never |
| `stranded_rate` | `safety.stranded_rate` | outcome | lower is better | `stranded / population` | never |
| `terminal_trapped_rate` | `safety.terminal_trapped_rate` | outcome | lower is better | `terminal_trapped_count / population` | never |
| `mean_person_time_to_safety` | `time.mean_person_time_to_safety` | outcome | lower is better | `person_time_to_safety_total / population` | never |
| `blocked_rate` | `movement.blocked_rate` | behaviour | lower is better | `blocked_moves / movement_attempts` | no movement attempts |
| `congestion_block_rate` | `movement.congestion_block_rate` | behaviour | lower is better | `congestion_blocks / movement_attempts` | no movement attempts |
| `aggregate_path_efficiency` | `path_efficiency.aggregate_path_efficiency` | proxy | higher is better | `total_static_lower_bound / total_cells_moved` | no measured movement |
| `mean_exposure` | `fire_exposure.mean_exposure` | proxy | lower is better | `(Σ_d count_d / d) / active_person_states` | never |
| `exits_used` | `exits.exits_used` | descriptive | none | number of exits with `n_e >= 1` | never |
| `max_exit_share` | `exits.max_exit_share` | descriptive | none | `max_e n_e / E` | `E = 0` |
| `exit_balance` | `exits.exit_balance` | descriptive | none | normalized Gini-Simpson index | `K = 1` or `E = 0` |

The per-episode definitions are in [metrics.md](metrics.md). Notes:

- **Kinds.** *Outcome* metrics are terminal safety and time results.
  *Behaviour* metrics describe observed movement. *Proxy* metrics approximate a
  quantity that is not measured directly (path optimality, fire danger).
  *Descriptive* metrics describe the run and are not objectives.
- **Directions** are used only to read the sign of a paired difference (§8).
  They do not make a metric an objective by itself.
- `mean_person_time_to_safety` counts every non-evacuee at `max_steps`
  (censoring), so it mixes speed and safety. Read it with the safety rates.
- `blocked_rate` and `congestion_block_rate` are lower-is-better per metric,
  but a run can lower them by not moving. Never read them alone.
- **Exit metrics have no direction.** Equal exit use is often not optimal (one
  exit can be much closer, or threatened by fire). `exit_balance` is never an
  objective, and no win/loss/tie count exists for any exit metric.

## 4. Groups: overall, per scenario, per family

Each artifact holds the same summaries for three kinds of group:

| Group | Episodes | Order |
|---|---|---|
| `overall` | every scheduled episode | — |
| `scenarios` | every scheduled episode of one scenario | suite manifest order |
| `families` | every scheduled episode of every scenario in the family | `ScenarioFamily` declaration order |

A scenario with several families contributes all of its episodes to **every**
one of those families. Family views can therefore overlap, and the family
counts do not add up to the overall counts. A family with no scheduled
scenario is not listed.

**Overall statistics use the episode observations directly.** They are never an
average of scenario or family means. With unequal group sizes (for example
after missing episodes) the two are different, and only the episode-level value
is reported. Each episode has equal weight; a scenario with more scheduled
seeds has more weight in `overall`.

## 5. Descriptive statistics

For one metric in one group, let `x_1, …, x_n` be the **defined** values of the
**observed** episodes, and `x_(1) <= … <= x_(n)` the same values in ascending
order.

| Field | Formula | Defined when |
|---|---|---|
| `n` | number of defined values | always |
| `mean` | `(Σ_i x_i) / n` | `n >= 1` |
| `median` | the type-7 quantile at `p = 1/2` (the mean of the two middle values when `n` is even) | `n >= 1` |
| `variance` | **sample variance** `Σ_i (x_i - mean)² / (n - 1)` | `n >= 2` |
| `std` | `sqrt(variance)` | `n >= 2` |
| `min`, `max` | `x_(1)`, `x_(n)` | `n >= 1` |
| `quantiles` | type-7 quantiles at `p25, p50, p75, p90, p95` | `n >= 1` |

An undefined statistic is `null` (`None`). It is never `0`.

### Quantile rule (Hyndman-Fan type 7)

With 0-based sorted values `y[0] <= … <= y[n-1]` and a level `p` in `[0, 1]`:

```
h = (n - 1) · p
j = floor(h)
g = h - j
Q(p) = y[j] + g · (y[j+1] - y[j])        (Q(p) = y[j] when g = 0)
```

This is the default of R (`type = 7`) and of NumPy (`method = "linear"`). The
levels are fixed: `p25 = 1/4`, `p50 = 1/2`, `p75 = 3/4`, `p90 = 9/10`,
`p95 = 19/20`. With `n = 1` every quantile is the single value.

Example: for `1, 2, 3, 4` the quantiles are `p25 = 1.75`, `p50 = 2.5`,
`p75 = 3.25`, `p90 = 3.7`, `p95 = 3.85`, and the sample variance is `5/3`.

## 6. Robustness and completeness counts

Each aggregate group has a `robustness` object of raw **episode** counts. There
are no rates here, so nothing hides the denominators.

| Field | Counts episodes … |
|---|---|
| `scheduled` | in the manifest for this group |
| `observed` | with an authenticated result |
| `missing` | scheduled, with no result (`scheduled - observed`) |
| `all_resolved` | observed, terminated with `all_resolved` |
| `max_steps` | observed, terminated at the step limit |
| `with_casualties` | observed, `casualties > 0` |
| `with_stranded` | observed, `stranded > 0` |
| `zero_evacuation` | observed, `evacuated = 0` |

`all_resolved + max_steps = observed`. An episode can count in several of the
last four rows.

## 7. Missing and undefined data

There are two different kinds of absent value. Both are reported; neither is
imputed.

- **Missing episode.** The manifest schedules it but the result has no entry
  for it (for example, an interrupted run). It is *missing*, not a model
  failure: it is not counted as a casualty, a timeout or zero evacuation. The
  artifact lists every missing episode key (`missing_episodes`), and every
  group gives `missing_indices` and `robustness.missing`.
- **Undefined metric.** The episode has a result, but the metric is `None` for
  it (for example `blocked_rate` when nobody tried to move). The per-metric
  summary counts it in `undefined`.

For each metric summary:

```
scheduled = observed + missing
n         = observed - undefined
```

The statistics use the `n` defined values only. No value is filled in with
zero, a mean, a worst case or a value from another episode.

## 8. Paired comparison

`compare_results(a, b, suite)` uses `pair_results(a, b)` for the exact episode
join. An episode is *paired* only when both results contain the same key
`(manifest hash, index, scenario ID, scenario hash, seed)`. Unpaired episodes
are never compared; they are listed in `missing_from_a` and `missing_from_b`.

For one metric in one group, with paired episodes `k` where the metric is
defined for **both** A and B:

```
delta_k = value_A(k) - value_B(k)        (always A - B)
```

Each `PairedMetricComparison` reports:

| Field | Meaning |
|---|---|
| `paired` | paired episodes in the group |
| `missing_from_a`, `missing_from_b` | scheduled episodes in the group that A or B does not have |
| `undefined` | paired episodes where the metric is `None` for A or for B |
| `deltas` | every `(episode index, delta_k)`, in manifest order |
| `statistics` | §5 statistics of the deltas: mean, median, std, quantiles, … (`n = paired - undefined`) |
| `mean_delta_ci` | §9 interval for the mean delta |
| `delta_positive`, `delta_negative`, `delta_zero` | counts of `delta > 0`, `< 0`, `= 0` |
| `a_better`, `b_better`, `ties` | win/loss/tie counts by the metric's direction |

Win/loss/tie, **per metric**:

| Direction | `a_better` | `b_better` | `ties` |
|---|---|---|---|
| higher is better | `delta > 0` | `delta < 0` | `delta = 0` |
| lower is better | `delta < 0` | `delta > 0` | `delta = 0` |
| descriptive | `null` | `null` | `null` |

There is no count across metrics, no majority vote and no composite. Swapping
A and B negates every delta and swaps `a_better` and `b_better`.

## 9. Confidence intervals

Each metric summary has `mean_ci` (aggregate) or `mean_delta_ci` (paired): a
**95 % percentile bootstrap** interval for the mean.

1. Let `x_1, …, x_n` be the defined values (aggregate) or the paired deltas
   (paired). If `n < 5` the interval is `null`.
2. For each of `R` resamples, draw `n` indices uniformly with replacement from
   `0 … n-1` and take the mean of the selected values. In a paired comparison
   the resampling unit is the paired episode, so A and B stay together in the
   delta.
3. Sort the `R` resample means. The bounds are their type-7 quantiles at
   `0.025` and `0.975`.

`R` is `resamples`: the default is **2000**; any integer in `[100, 1 000 000]`
is accepted. All means are exact fractions; only the bounds become floats.

**Determinism.** The bootstrap never uses `random`, the time or OS entropy.
Its stream seed is

```
seed = SHA-256( canonical JSON of {
    statistics_version, purpose ("aggregate_mean" or "paired_mean_delta"),
    evaluation_manifest_hash, result_digests, group [kind, name], metric, resamples } )
```

where each result digest is the SHA-256 of the canonical `EvaluationResult`
(the artifact as saved). The 64-bit words come from SHAKE-256 in counter mode:
block `k` is `SHAKE-256("emberbench/statistics/bootstrap/v1\0" || seed || k)`
(`k` as 8 big-endian bytes), read as 512 big-endian unsigned 64-bit words. An
index below `n` is drawn by rejection sampling (`word < 2^64 - 2^64 mod n`,
then `word mod n`), so there is no modulo bias. The seed is in the output for
audit. Identical inputs give identical intervals on every platform. Different
data, labels or manifests give different streams. This stream shares nothing
with the simulator's SplitMix64 streams.

**Interpretation.** An interval describes the uncertainty of the mean **over
this evaluation sample** of scenarios and seeds, resampled at the episode
level. It does not prove that one model is better in general, on other
scenarios or on other seeds. Episodes of one scenario are not independent of
each other, and the public suite is small, so treat intervals as descriptive.

## 10. Exact arithmetic and serialization

- Per-episode values are exact `Fraction`s rebuilt from integer counts (for
  `mean_exposure`, from the fire-distance histogram: `Σ count_d / d`).
- Means, medians, variances, quantiles, min/max, deltas and bootstrap means
  are exact `Fraction`s in the dataclasses.
- The **output boundary** is `to_dict()`: each exact value becomes its
  correctly rounded `float`. The standard deviation is the one irrational
  value: it is `math.sqrt(float(variance))`, which IEEE 754 rounds correctly.
- Every list is in manifest, suite or `ScenarioFamily` order. No result
  depends on `dict` or `set` iteration order or on `PYTHONHASHSEED`.
- `to_json()` is canonical JSON (sorted keys, no whitespace, ASCII, no NaN).
  `digest()` is its SHA-256. The model ID label, the input result digests and
  `resamples` are part of the artifact, so a change to any of them changes the
  digest.

## 11. Artifact layout

`AggregateStatistics.to_dict()`:

```
statistics_version, artifact = "aggregate", model_id,
evaluation_manifest_hash, suite_id, suite_version, suite_hash,
simulator_version, observation_version, metrics_version,
result_digest, scheduled_episodes, observed_episodes, missing_episodes[],
metric_definitions[], methods{quantiles, variance, std, bootstrap},
overall{kind, name, episode_indices, missing_indices, robustness, metrics[]},
scenarios[], families[]
```

Each `metrics[]` entry: `metric, scheduled, observed, missing, undefined,
statistics{n, mean, median, variance, std, min, max, quantiles{p25…p95}},
mean_ci{method, statistic, level, resamples, seed, lower, upper} | null`.

`PairedStatistics.to_dict()` has the same header with `model_a`, `model_b`,
`result_digest_a`, `result_digest_b`, `delta = "A - B"`, `paired_count`,
`missing_from_a[]`, `missing_from_b[]`, and groups with `paired_indices`,
`missing_from_a`, `missing_from_b` and the §8 fields per metric.

## 12. Summary of what each section is

| Concept | Where | Is it a judgement? |
|---|---|---|
| Descriptive statistics | §5, `statistics` | no: a summary of observed values |
| Robustness counts | §6, `robustness` | no: raw episode counts |
| Missing / undefined | §7 | no: completeness accounting |
| Paired differences | §8, `deltas` | no: signed `A - B` values |
| Per-metric win/loss/tie | §8 | per metric only, by its direction |
| Metrics with no direction | §3, exit metrics | never a winner |
| Confidence intervals | §9 | uncertainty over this sample only |

There is no EMBER score or global winner in this module.
