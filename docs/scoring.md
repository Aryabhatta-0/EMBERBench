# The EMBER Score (scoring version 1.0)

This document is the normative definition of the EMBER Score. The code is
`src/emberbench/scoring.py`. The tests are `tests/test_scoring.py`. The
research, the candidate formulas and the sensitivity study that justify each
choice are in [benchmark-methodology.md](benchmark-methodology.md). The frozen
environment that every model is scored on is in
[environment-pack.md](environment-pack.md).

The EMBER Score is one number in `[0, 100]`. Its unit is
**benchmark-performance points under EMBER scoring version 1.0**. It is not a
percentage of people evacuated. A report says "72.4 points under EMBER
scoring version 1.0", never "72.4% evacuated".

## 1. Status and versioning

| Item | Value |
| --- | --- |
| Version | `SCORING_VERSION = "1.0"` |
| Constants | `OFFICIAL_CONSTANTS` (section 4); `ScoringConstants.digest()` pins them |
| Separate from | simulator, observation, action schema, scenario schema, metrics, crowd diagnostics, statistics, generator and evaluation versions. Scoring changes none of them. |
| Not in | `benchmark_versions()` (the provenance version set is unchanged) |

`SCORING_VERSION` changes when a definition, a constant, the arithmetic or
the serialization changes. A score is comparable with another score only
when both have the same scoring version, the same constants digest, the same
evaluation manifest and the same environment pack.

A score is **official** only when it is computed against an
[environment pack](environment-pack.md) whose evaluation manifest equals the
results' manifest, under `OFFICIAL_CONSTANTS`. `ScoreReport.official` tells
which. A score against a bare suite is labelled UNOFFICIAL.

## 2. Inputs and authentication

`score_results(trials, target, *, failures=None, resamples=2000)` takes:

- `trials`: one or more `EvaluationResult` objects of the **same model on the
  same evaluation manifest**. Use one trial for a deterministic policy and
  several for a stochastic model (section 12).
- `target`: the `Suite`, or the `EnvironmentPack`.
- `failures[t]`: an optional map from episode index to `FailureKind` that
  says why trial `t` has no result for a scheduled episode.

Every result is authenticated from scratch, exactly as in
[statistics.md](statistics.md): the manifest must round-trip and target the
suite, every run record is replayed (`verify_record`), and the stored metrics
must equal the recomputed metrics. With a pack, every scheduled episode's
fire trajectory must also match the pack's `fire_world_digest`, and every
recorded burning set must match that trajectory. Any difference raises
`ScoringInputError`. There is no partial score.

Provider-reported success, accuracy or cost is never an input. Scoring is
local and uses only replayed records.

## 3. Privileged references

Several components compare the model with a **relaxed single-person
optimum**. The optimum is computed by backward induction over (cell, step)
for one person alone who knows the whole future fire and follows the
simulator phase order. Other people can only block moves, so each reference
is a lower bound for every policy. The references use the evaluator seed.
They are evaluator-side only, like `emberbench.oracle`, and never reach a
model.

For each scheduled episode and each person `i` at start cell `x_i`:

| Reference | Definition |
| --- | --- |
| `M*` (`max_evacuated`) | number of people with a finite earliest evacuation step. Equal to `privileged_escape_bound(...).earliest_evacuation` (tested on every public scenario). |
| `L_i` | earliest evacuation step from `(x_i, 0)` |
| `m*_i` | fewest cell moves of any escape from `(x_i, 0)` (the move onto the exit counts) |
| `D*_i` | fewest danger-zone person-states of any escape from `(x_i, 0)` |
| `L(x, t)` | earliest evacuation step from cell `x` at snapshot `t` |

A person-state is in the **danger zone** when the Manhattan distance `d` from
the person's cell to the nearest burning cell satisfies `0 < d <= 2`
(`HAZARD_RADIUS = 2`). The distance ignores walls, like the metrics exposure
proxy.

The transition rule is the simulator's: from `(x, t)` a person may stay or
move to an open neighbour `y`, unless `y` burns at the start of the step;
a non-burning exit evacuates at once; any other `y` must not burn after the
fire phase. Snapshot `H` (`max_steps`) has no successor.

## 4. The episode score

For one authenticated episode with `M* >= 1`:

```text
EMBER = 100 * Safety * Quality

Safety  = E * P_casualty * P_stranding             (0 at the catastrophic gate)
Quality = weighted geometric mean of the defined quality components
```

Every component is in `[0, 1]`. No component can exceed 1, so no strength
offsets a failure.

### 4.1 Safety

Let `N` be the population, `U = N - M*` the people nobody could save,
`C` and `S` the casualties and stranded people. Unavoidable losses are charged
to casualties first:

```text
c = max(0, C - U)                      avoidable casualties
s = max(0, S - max(0, U - C))          avoidable stranded
```

| Factor | Formula | Official constants |
| --- | --- | --- |
| Evacuation `E` | `(evacuated / M*) ^ kE` | `kE = 2` |
| Casualty `P_casualty` | `alpha_c ^ log2(1 + c) * (1 - c / M*) ^ k_c` | `alpha_c = 0.6`, `k_c = 4` |
| Stranding `P_stranding` | `alpha_s ^ log2(1 + s) * (1 - s / M*) ^ k_s` | `alpha_s = 0.8`, `k_s = 2` |
| Catastrophic gate | `Safety = 0` when `(c + s) / M* >= 1/2` | `catastrophic_loss_rate = 1/2` |

The **incident factor** `alpha ^ log2(1 + n)` is paid in full for the first
loss and again for every doubling of `1 + n`. It makes the first casualty
expensive in any population. The **rate term** `(1 - n / M*) ^ k` grows with
the share of people lost. A casualty costs more than a stranded person.
Stranding and incomplete evacuation are penalized independently of
casualties.

### 4.2 Quality components

| Component | Formula | Defined when | Weight |
| --- | --- | --- | --- |
| Time `T` | `sum(L_i) / sum(A_i)` over evacuees; `A_i` is the evacuation step | at least one evacuee | 3/10 |
| Coordination `C` | `(1 - cbr) ^ kC`, `cbr = (blocked_occupied + blocked_conflict + blocked_cycle) / movement_attempts` | at least one movement attempt | 2/10 |
| Hazard `H` | `(1 - (D - D*) / sum(A_i)) ^ kH` over evacuees; `D` = their danger-zone states, `D* = sum(D*_i)` | at least one evacuee | 2/10 |
| Path `P` | `sum(m*_i) / sum(m_i)` over evacuees; `m_i` = cells moved | at least one evacuee | 1/10 |
| Replanning `R` | `sum(L(x_e, t_e) - t_e) / sum(r_e - t_e)` over recoverable route-invalidation events | at least one recoverable event | 2/10 |

Constants: `kC = 2`, `kH = 1`.

Every ratio is a privileged lower bound over the actual value, so it is in
`(0, 1]`. If an actual value is below its bound, the record is rejected: that
can only happen with a forged record.

**Route-invalidation event.** At step `t >= 1` the burning set changed and,
for a person active in snapshot `t`, the fire-aware distance to the nearest
non-burning exit (shortest path through non-wall, non-burning cells) became
longer or infinite, where it was finite before. The event is **recoverable**
when `L(x_e, t)` is finite. Its resolution step `r_e` is the person's
evacuation step, or the horizon `H` when the person is lost (censored). An
unrecoverable event is counted but not scored: its outcome is already in
Safety.

**Quality** is the weighted geometric mean over the defined components, with
the weights renormalized over them:

```text
Quality = exp( sum_k (w_k / W) * ln q_k ),   W = sum of the weights of the defined components
Quality = 0 when any defined component is 0
```

An undefined component is never set to 1. It is left out, and its weight is
shared by the others. So an episode without route invalidation is neither
rewarded nor punished for replanning.

### 4.3 Episode statuses

| Status | Meaning | Score |
| --- | --- | --- |
| `scored` | authenticated, below the gate | `100 * Safety * Quality` |
| `catastrophic` | authenticated, avoidable loss rate `>= 1/2` | 0 |
| `failed` | scheduled, no authenticated result | 0 |
| `not_applicable` | `M* = 0`: no policy can save anyone | excluded from every mean, for every model |

A failed episode has a `FailureKind`: `missing` (no declared reason),
`invalid_output` (`ModelOutputError` or another rejected action set),
`model_error` (the model or its adapter raised) or `timeout` (the adapter's
deadline expired). All four score 0 and are counted in the report. A failure
can never disappear: a gap without a declaration is `missing`. A declaration
for an unscheduled episode, or for an episode that has a committed result, is
rejected.

`not_applicable` depends only on the scenario and the seed, never on the
model.

### 4.4 Arithmetic

Counts and references are exact integers. Non-integer powers are computed
with Python `decimal` (libmpdec) at 50 significant digits, round-half-even,
which gives identical digits on every platform. Components are rounded to
`1e-12` and the episode score to `1e-9` points. From there on every mean and
every bootstrap statistic is exact rational arithmetic.

## 5. Exact penalties and ceilings

Single-failure profile under the official constants. Everything else is
perfect; replanning is undefined.

| Case (population 100 unless stated) | Score |
| --- | --- |
| perfect | 100.0 |
| 1 avoidable casualty | 56.5 |
| 2 casualties | 39.4 |
| 3 casualties | 30.0 |
| 5 casualties | 19.6 |
| 10 casualties | 9.1 |
| 25 casualties | 1.6 |
| 1 casualty in a population of 4 | 10.7 |
| 1 stranded | 76.8 |
| 10 stranded | 30.3 |
| 50 losses (the gate) | 0.0 |
| Time 0.5 | 77.1 |
| Time 0.25 | 59.5 |
| Time 0.1 | 42.2 |
| congestion block rate 0.1 / 0.3 / 0.6 | 94.9 / 83.7 / 63.2 |
| danger-zone rate 0.3 above the privileged minimum | 91.5 |
| Path 0.625 | 94.3 |
| Replanning 0.25 (all five components defined) | 75.8 |

Ceilings that follow from the formula:

- **Any avoidable casualty:** the episode score is at most `100 * 0.6 = 60`,
  for every population size.
- **Any avoidable stranded person:** at most `100 * 0.8 * (1 - 1/M*)^4 < 80`.
- **No 95 or more after a meaningful safety failure:** any avoidable
  casualty or stranded person gives a score below 80.
- **Catastrophic failure:** half or more of the people who could be saved
  are lost: 0.
- **Invalid output, model error, missing result:** 0.
- **Timeout:** 0, the most severe penalty.

## 6. Aggregation

Episodes are aggregated in a fixed hierarchy. Each level is the unweighted
mean of the defined values of the level below:

```text
episode-trial  ->  episode (mean over trials)
episode        ->  scenario (mean over its applicable seeds)
scenario       ->  primary family (mean over its scenarios)
family         ->  overall (mean over families)   = the EMBER Score
```

The primary family is `SuiteEntry.primary_family`. A scenario counts in one
family only, so a scenario with several families is not counted twice. A
scenario whose episodes are all `not_applicable` is excluded.

The report also gives the **scenario-balanced** mean (mean of scenario means)
and the **micro** mean (mean of all episodes). They are diagnostics. The
family-balanced mean is the headline: each failure mode weighs the same, and
a suite cannot raise a score by adding easy scenarios of one family.

Components, Safety and Quality are aggregated with the same hierarchy over
the episode-trials where they are defined. Each component reports its
coverage `defined / total`.

**Lower tail.** The report gives the median scenario score, the worst
scenario and its ID, the type-7 10th percentile of episode scores, CVaR10
(the mean of the lowest `ceil(n/10)` episode scores), and the number of
zero-score episode-trials by cause (catastrophic and each `FailureKind`).

**Totals.** Escapable, evacuated, avoidable casualties, avoidable stranded
and unavoidable losses, summed over scored episode-trials, also per trial.

## 7. Confidence interval

The 95% interval of the headline is a **stratified two-stage percentile
bootstrap**:

1. Strata are the primary families.
2. In each resample, inside each family, draw as many scenarios as it has,
   with replacement; inside each drawn scenario, draw as many episodes as it
   has, with replacement.
3. Recompute the family-balanced mean. Bounds are the type-7 2.5% and 97.5%
   quantiles of `resamples` (default 2000) statistics.

The random stream is the SHAKE-256 counter-mode `BootstrapStream` of
[statistics.md](statistics.md), seeded with SHA-256 of the canonical JSON of
the scoring version, the constants digest, the manifest hash, the model ID
and the exact episode scores. The interval is identical on every platform.
With fewer than 5 applicable episodes there is no interval.

The interval describes **this evaluation sample**: these scenarios and these
seeds. When a family has one scenario, the bootstrap cannot see scenario
variance, and a deterministic model on deterministic scenarios gets a
zero-width interval. To make claims about unseen environments, use a
generated suite with several scenarios per family (see
[procedural-generation.md](procedural-generation.md)).

## 8. Comparison and leaderboard

`compare_scores(a, b)` pairs two reports episode by episode. The reports
must have the same scoring version, constants, manifest, pack and applicable
episodes. Paired differences `d_e = a_e - b_e` are aggregated with the same
hierarchy (so `delta = score_a - score_b` exactly), and the interval is the
same stratified bootstrap applied to the differences. The outcome is one of
four:

| Outcome | Rule |
| --- | --- |
| `exact_tie` | every paired episode score is identical |
| `a_better` | interval lower bound `> 0` **and** `delta >= 1.0` point |
| `b_better` | interval upper bound `< 0` **and** `-delta >= 1.0` point |
| `inconclusive` | anything else, including fewer than 5 applicable episodes |

`practical_difference = 1` point is the smallest difference that counts as
"better". A difference that is statistically clear but smaller is
`inconclusive`.

**Leaderboard order** is separate from significance. `leaderboard_order`
sorts by:

1. the score rounded half-even to 0.1 point, descending;
2. the Safety factor, descending;
3. avoidable casualties per trial, ascending;
4. avoidable stranded per trial, ascending;
5. the worst scenario score, descending;
6. the model ID.

So safety decides every tie of the displayed score. Adjacent leaderboard
entries can still be `inconclusive`; publish the comparison next to the
order.

## 9. Metric classification

Every existing metric of [metrics.md](metrics.md) has one role:

| Metric | Role | Use in the score |
| --- | --- | --- |
| evacuated / evacuation rate | core (Safety) | `E`, relative to `M*` |
| casualties (fatality) | safety gate and core (Safety) | `P_casualty`, catastrophic gate |
| stranded | safety gate and core (Safety) | `P_stranding`, catastrophic gate |
| time to safety (`time.*`) | core (Quality) | `T`, evacuees only, relative to `L_i` |
| congestion / bottleneck (`blocked_*`, `congestion_block_rate`) | core (Quality) | `C` from occupied, conflict and cycle blocks |
| fire exposure (`fire_exposure`, `min_fire_distance`) | diagnostic | replaced by the avoidable danger-zone rate `H` |
| path efficiency (`path_efficiency.*`, static bound) | diagnostic | replaced by `P` with the fire-aware privileged bound |
| replanning proxies (`recovery_proxy_rate` and others) | diagnostic | replaced by `R` on route-invalidation events |
| exit utilization / balance | diagnostic only | none; a 50/50 split is never rewarded |
| crowd diagnostics (hotspots, contested cells, throughput) | diagnostic only | none |
| decisions, latency, tokens, money | operational only | none; see section 11 |

Reasons for each role are in
[benchmark-methodology.md](benchmark-methodology.md), section 4.

## 10. Interpretation: what 70/100 means

- 70 points is a **benchmark-performance** value, not 70% of people.
- An episode with any avoidable casualty scores at most 60. So an episode
  score of 70 means nobody who could be saved died.
- The headline is a mean over families. 70 can be "every family safe with
  quality 0.7" (for example, evacuation about 2.6 times slower than the
  privileged bound and nothing else wrong) or "seven families perfect and
  three at zero". The family scores and the lower-tail statistics tell
  which. Always read them with the headline.
- Anchors on the public suite with the development seeds 0, 1, 2:
  `congestion-aware-v1` 49.6, `fire-aware-shortest-path-v1` 39.6,
  `static-shortest-path-v1` 36.7, `nearest-exit-greedy-v1` 26.8,
  `random-legal-v1` 0.0, always-stay 0.0. The privileged bound would score
  100 but is not a policy.

A report has this shape (`format_summary(report)`):

```text
EMBER Score 49.6 / 100 benchmark-performance points under EMBER scoring version 1.0
95% CI [47.1, 54.6] (stratified two-stage bootstrap, 500 resamples; this sample only)
Model congestion-aware-v1; trials 1; official; environment pack <id> sha256:<digest>
Safety factor 0.556; quality factor 0.828
Components (0-1, defined/total): evacuation 0.864 (30/30); casualty 0.590 (30/30); ...
Primary families: single_exit_low_density 99.1; single_exit_high_density 13.3; ...
Tail: median scenario 34.5; worst scenario 7.1 (v1-pub-05-central-ignition); p10 episode 9.7; CVaR10 7.1
Zero-score episode-trials 0 of 30: catastrophic 0; missing 0; invalid_output 0; model_error 0; timeout 0
People per trial: escapable 561.0; evacuated 526.0; avoidable casualties 35.0; avoidable stranded 0.0
The EMBER Score is not a percentage of people evacuated.
```

## 11. Operational metrics and EMBER-Efficiency

Tokens, money and latency are never inputs to the score. Report them next
to it. `efficiency_view(report, cost_per_episode, unit)` gives an optional
EMBER-Efficiency view: points per unit of **evaluator-measured** cost. It
does not change the score or the order.

## 12. Model stochasticity

A deterministic policy needs one trial. A model that can give different
actions for the same observation (any sampled LLM) needs repeated trials on
the same pack. `temperature = 0` is not a guarantee of determinism: batching,
kernel choice and provider changes can alter outputs. The protocol is in
[environment-pack.md](environment-pack.md), section 6. Each trial's
episodes are averaged before aggregation, so trials never count as extra
scenarios.

## 13. Relation to the metrics rules

[metrics.md](metrics.md), section 11, requires a composite to be versioned,
decomposable, and to never let speed buy safety. Version 1.0 meets the first
two: it has its own version, and every report holds the components and the
per-episode evidence. For the third rule:

- Within one episode, a run with an avoidable casualty scores at most 60.
  A safe run beats it unless the safe run is more than about 4.6 times slower
  than the privileged bound (Time below 0.22 when replanning is undefined).
  This bounded exception is deliberate: the specification asks for a
  multiplicative score in which an extremely slow evacuation also fails.
- Leaderboard ties of the displayed score are always broken by safety
  first (section 8).
