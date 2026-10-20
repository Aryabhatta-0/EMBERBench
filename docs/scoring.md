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
EMBER    = 100 * Safety * Quality ^ beta * Severity

Safety   = E * P_casualty * P_stranding             (0 at the catastrophic gate)
Quality  = weighted geometric mean of the defined quality components
beta     = min(1, ln(alpha_c) / ln(q_min))          = ln 0.6 / ln 0.5 = 0.7369655941662...
Severity = product over severe components k of  phi * (q_k / q_k_severe) ^ gamma
```

Every component, Safety, Quality and Severity are in `[0, 1]`. No component
can exceed 1, so no strength offsets a failure. Safety caps the score
(section 5), and each severe quality failure caps it again (section 4.3).

### 4.1 Safety and person-level attribution

Every loss is attributed **per person**. A person `i` is **savable** when the
privileged earliest escape `L_i` (section 3) is finite. Then, for every
person at the end of the episode:

| Outcome | Savable | Not savable |
| --- | --- | --- |
| evacuated | evacuated | evacuated (cannot happen: `L_i` would be finite) |
| died | **avoidable casualty** | unavoidable casualty |
| still active or stranded at the horizon | **avoidable stranded** | unavoidable stranded |

`c` and `s` are the numbers of avoidable casualties and avoidable stranded
people. The evidence stores the sorted person IDs of all four loss classes
(`avoidable_casualty_ids`, `avoidable_stranded_ids`,
`unavoidable_casualty_ids`, `unavoidable_stranded_ids`), so each count can be
audited person by person.

An unavoidable loss is never transferred between people. An earlier draft
computed `c = max(0, C - U)` with `U = N - M*` from aggregate counts. That
rule lets the unavoidable loss of one person hide the avoidable death of
another: with person `a` savable but dead and person `b` unsavable and
stranded, the draft gives `c = 0`, but the correct value is `c = 1`.
`test_real_episode_person_level_attribution` reproduces this case with a real
replayed episode. The evidence validator rejects any record where the
identity sets overlap, are not sorted, or do not match the counts.

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
| Time `T` | `Q* / sum(A_i)` over evacuees; `A_i` is the evacuation step, `Q*` the exit-capacity queue bound (below) | at least one evacuee | 3/10 |
| Coordination `C` | `(1 - cbr) ^ kC`, `cbr = (blocked_occupied + blocked_conflict + blocked_cycle) / movement_attempts` | at least one movement attempt | 2/10 |
| Hazard `H` | `(1 - (D - D*) / sum(A_i)) ^ kH` over evacuees; `D` = their danger-zone states, `D* = sum(D*_i)` | at least one evacuee | 2/10 |
| Path `P` | `sum(m*_i) / sum(m_i)` over evacuees; `m_i` = cells moved | at least one evacuee | 1/10 |
| Replanning `R` | `sum(L(x_e, t_e) - t_e) / sum(r_e - t_e)` over recoverable route-invalidation events | at least one recoverable event | 2/10 |

Constants: `kC = 2`, `kH = 1`.

Every ratio is a privileged lower bound over the actual value, so it is in
`(0, 1]`. If an actual value is below its bound, the record is rejected: that
can only happen with a forged record.

**Time queue bound `Q*`.** The sum of single-person optimum steps `sum(L_i)`
ignores that people queue at exits. Positions after movement are distinct,
so in each step at most one person leaves through each exit cell that is not
burning at the start of the step. `Q*` is the list schedule of the evacuees'
release steps `L_i` over these exit slots: in each step, the open exit slots
are filled with released people who are still waiting. For unit jobs with
release dates, this schedule minimizes the sum of finish steps, so
`sum(L_i) <= Q* <= sum(A_i)` for every policy. A bottleneck that no policy can
avoid is therefore not charged to Time.

**Exit utilization** is diagnostic only. A balanced split between exits is
not a goal: the right split depends on distances, fire and capacity. The
queue bound already measures the only effect of exit allocation that the
score needs, the time cost of a bad allocation, against an allocation-optimal
bound. A separate balance or allocation-regret term would count that cost
twice.

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

Path uses the fire-aware privileged bound `m*_i`, so a detour that the fire
makes necessary is not charged. Hazard is measured above the privileged
minimum `D*`, so exposure that no policy can avoid is not charged. Neither
component counts the same loss as Safety: both are computed over evacuees
only.

### 4.3 Quality exponent and severe failures

Two rules make Safety dominate without making Quality irrelevant.

**Quality exponent `beta`.** Quality enters as `Quality ^ beta`, with

```text
beta = min(1, ln(alpha_c) / ln(q_min)),   q_min = smallest severe threshold in component space
```

`beta` is derived, not tuned. With every quality component at or above its
severe threshold, `Quality >= q_min`, so `Quality ^ beta >= alpha_c = 0.6`.
A perfect run with one avoidable casualty has `Safety < 0.6`. So a run with
no avoidable loss and no severe failure always scores above a perfect run
with one avoidable casualty, in every population (requirement R12, tested
for `M*` from 2 to 1000, also with one extra stranded person on both sides).

**Severe failures.** A quality component is **severe** when it crosses a
threshold in natural units. The comparison is exact and strict:

| Component | Severe when | Threshold in component space `q_k_severe` |
| --- | --- | --- |
| Time | `T < 1/2` (more than twice the queue bound) | 0.5 |
| Coordination | block rate `> 1/4` | `(1 - 1/4) ^ 2 = 0.5625` |
| Hazard | avoidable danger-zone rate `> 1/4` | `1 - 1/4 = 0.75` |
| Path | `P < 1/2` (more than twice the fewest cells) | 0.5 |
| Replanning | `R < 1/2` | 0.5 |

Each severe component `k` multiplies the score by
`phi * (q_k / q_k_severe) ^ gamma`, with `phi = 0.8` and `gamma = 1`. So one
severe failure caps the episode below 80, and the cap falls further as the
failure gets worse. Two severe failures cap it below 64. Severity is a
ceiling, not a separate measurement: it applies to the same components that
are already in Quality, and is 1 when nothing is severe.

The report counts severe episode-trials per component
(`Robustness.severe_episode_trials`). The episode record gives `severity`,
the list `severe` and the `ceiling = 100 * Safety * Severity`.

The thresholds are set by normative requirements, not by a policy ranking
(see [benchmark-methodology.md](benchmark-methodology.md), section 6):
evacuating more than twice as slowly as an exit-capacity optimum, one blocked
move in four, one avoidable danger-zone step in four, twice the needed cells
and twice the needed recovery time are each a failure of that dimension.

### 4.4 Episode statuses

| Status | Meaning | Score |
| --- | --- | --- |
| `scored` | authenticated, below the gate | `100 * Safety * Quality ^ beta * Severity` |
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

### 4.5 Arithmetic

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
| 1 savable casualty + 1 unsavable stranded person | 56.5 |
| 50 losses (the gate) | 0.0 |
| Time 0.8 / 0.5 (threshold) | 94.0 / 82.6 |
| Time 0.4 / 0.25 / 0.1 (severe) | 49.7 / 27.3 / 8.5 |
| block rate 0.1 / 0.25 (threshold) | 96.2 / 89.9 |
| block rate 0.3 / 0.5 (severe bottleneck) | 61.1 / 27.5 |
| avoidable danger-zone rate 0.1 / 0.3 (severe) | 98.1 / 69.9 |
| Path 0.6 / 0.4 (severe) | 95.4 / 58.8 |
| Replanning 0.8 / 0.3 (severe), replanning defined | 96.8 / 40.2 |
| every quality component exactly at its threshold | 64.8 |

The jump at a threshold is the severe factor `phi = 0.8` times the ratio
`q / q_severe`. Below the thresholds, Quality still matters: Time 0.5 alone
costs 17.4 points, and a block rate of 0.25 costs 10.1 points.

Ceilings that follow from the formula:

- **Any avoidable casualty:** the episode score is at most `100 * 0.6 = 60`,
  for every population size.
- **Any avoidable stranded person:** at most `100 * 0.8 * (1 - 1/M*)^4 < 80`.
- **No 95 or more after a meaningful safety failure:** any avoidable
  casualty or stranded person gives a score below 80.
- **A casualty is worse than a stranding:** one avoidable casualty scores
  below one avoidable stranded person in the same episode.
- **Any severe quality failure:** below `100 * 0.8 = 80`; two severe
  failures, below 64. A severe failure cannot be bought back by any other
  component, because no component exceeds 1.
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
| casualties (fatality) | safety gate and core (Safety) | `P_casualty` on avoidable (savable) casualties, catastrophic gate |
| stranded | safety gate and core (Safety) | `P_stranding` on avoidable stranded people, catastrophic gate |
| time to safety (`time.*`) | core (Quality) and severe gate | `T`, evacuees only, relative to the queue bound `Q*` |
| congestion / bottleneck (`blocked_*`, `congestion_block_rate`) | core (Quality) and severe gate | `C` from occupied, conflict and cycle blocks per movement attempt |
| fire exposure (`fire_exposure`, `min_fire_distance`) | diagnostic | replaced by the avoidable danger-zone rate `H` (core and severe gate) |
| path efficiency (`path_efficiency.*`, static bound) | diagnostic | replaced by `P` with the fire-aware privileged bound (core, low weight, severe gate) |
| replanning proxies (`recovery_proxy_rate` and others) | diagnostic | replaced by `R` on route-invalidation events (core and severe gate) |
| exit utilization / balance | diagnostic only | none; a 50/50 split is never rewarded; the time cost of a bad allocation is in `T` through `Q*` |
| crowd diagnostics (hotspots, contested cells, throughput) | diagnostic only | none |
| decisions, latency, input/output tokens, cost, retries, timeouts | operational only | none; `OperationalRecord`, see section 11 |

Reasons for each role are in
[benchmark-methodology.md](benchmark-methodology.md), section 4.

## 10. Interpretation: what 70/100 means

- **70.0/100 is 70.0 benchmark-performance points under `SCORING_VERSION`
  1.0. It is not "70% of people evacuated".**
- An episode with any avoidable casualty scores below 60. So an episode
  score of 70 means nobody who could be saved died.
- An episode with a severe quality failure scores below 80. So an episode
  score of 80 or more means no avoidable loss and no severe failure.
- The headline is a mean over families. 70 can be "every family safe with
  moderate quality" or "seven families perfect and three at zero". The
  family scores and the lower-tail statistics tell which. Always read them
  with the headline.
- Anchors on the public suite with the development seeds 0, 1, 2:
  `congestion-aware-v1` 49.1, `fire-aware-shortest-path-v1` 39.8,
  `static-shortest-path-v1` 36.9, `nearest-exit-greedy-v1` 26.9,
  `random-legal-v1` 0.0, always-stay 0.0. The privileged bound would score
  100 but is not a policy.

A report has this shape (`format_summary(report)`, family list shortened):

```text
EMBER Score 49.1 / 100 benchmark-performance points under EMBER scoring version 1.0
95% CI [46.5, 51.7] (stratified two-stage bootstrap, 500 resamples; this sample only)
Model congestion-aware-v1; trials 1; official; environment pack <id> sha256:<digest>
Safety factor 0.556; quality factor 0.870
Components (0-1, defined/total): evacuation 0.864 (30/30); casualty 0.590 (30/30); stranding 1.000 (30/30); time 0.936 (30/30); coordination 0.784 (30/30); hazard 0.896 (30/30); path 0.932 (30/30); replanning 1.000 (8/30)
Primary families: single_exit_low_density 99.4; single_exit_high_density 4.2; narrow_corridor_bottleneck 91.4; ...
Tail: median scenario 32.5; worst scenario 4.2 (v1-pub-02-single-exit-dense); p10 episode 6.8; CVaR10 4.2
Zero-score episode-trials 0 of 30: catastrophic 0; missing 0; invalid_output 0; model_error 0; timeout 0
Severe quality failures (episode-trials): time 0; coordination 6; hazard 3; path 0; replanning 0
People per trial: escapable 561.0; evacuated 526.0; avoidable casualties 35.0; avoidable stranded 0.0
The EMBER Score is not a percentage of people evacuated.
```

## 11. Operational metrics and EMBER-Efficiency

Tokens, money and latency are never inputs to the score. `score_results` has
no parameter for them. Record them in an `OperationalRecord`, next to the
score:

| Field | Meaning |
| --- | --- |
| `model_id`, `provider_id` | the exact model snapshot and the provider |
| `inference_config_digest` | `inference_config_digest(config)`: SHA-256 of the canonical JSON of every inference setting (temperature, top-p, max tokens, seed if any, system prompt version, adapter version) |
| `trials`, `episodes` | the work the record covers |
| `latency_seconds` | evaluator-measured wall time |
| `input_tokens`, `output_tokens` | kept separate, never summed |
| `cost`, `cost_unit` | evaluator-computed cost (exact fraction) and its unit; `None` when unknown |
| `retries`, `timeouts` | adapter retries and expired deadlines |

The record is validated (no empty IDs, a 64-hex digest, non-negative finite
values) and has its own digest. `efficiency_view(report, cost_per_episode,
unit)` gives an optional EMBER-Efficiency view: points per unit of
**evaluator-measured** cost. It does not change the score or the order.

## 12. Model stochasticity

A deterministic policy needs one trial. A model that can give different
actions for the same observation (any sampled LLM) needs **K >= 3** repeated
trials on the same pack, with pinned inference settings (one
`inference_config_digest`). `temperature = 0` is not a guarantee of
determinism: batching, kernel choice and provider changes can alter outputs.
Every trial is scored; the best trial is never selected. Each episode's
trials are averaged before aggregation, so trials never count as extra
scenarios. The protocol is in [environment-pack.md](environment-pack.md),
section 6.

## 13. Relation to the metrics rules

[metrics.md](metrics.md), section 11, requires a composite to be versioned,
decomposable, and to never let speed buy safety. Version 1.0 meets all
three:

- It has its own version and constants digest, and every report holds the
  components, the severe-failure list and the per-person evidence.
- Within one episode, a run with no avoidable loss and no severe quality
  failure always scores above a perfect run with one avoidable casualty
  (section 4.3). Speed can only help a run with a casualty against a safe run
  that itself fails a dimension severely, for example one that evacuates
  more than twice as slowly as the exit-capacity bound. Such a run is not
  "safe and good"; the severe gate makes that explicit.
- Leaderboard ties of the displayed score are always broken by safety
  first (section 8).
