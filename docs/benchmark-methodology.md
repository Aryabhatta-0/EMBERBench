# EMBERBench scientific evaluation methodology

This document explains **why** the EMBER Score (scoring version 1.0) has its
form. The normative definition is [scoring.md](scoring.md). The frozen
environment protocol is [environment-pack.md](environment-pack.md). The full
output of the sensitivity study is
[scoring-study-results.md](scoring-study-results.md).

Nothing here changes the simulator, the observation, the schemas or the
metrics. The score is computed only from authenticated, replayed run records.

## 1. Summary of decisions

| Question | Decision |
| --- | --- |
| Form | `EMBER = 100 * Safety * Quality`; both factors in `[0, 1]` |
| Safety | `E^2 * P_casualty * P_stranding`, with a discrete incident factor plus a rate term for each loss type, and a catastrophic gate at half the savable people |
| Quality | weighted geometric mean of Time, Coordination, Hazard, Path and Replanning (weights 0.3 / 0.2 / 0.2 / 0.1 / 0.2) |
| Normalization | each quality component is a privileged relaxed optimum divided by the actual value, so it is relative to the difficulty of that exact episode |
| Failures | missing, invalid output, model error and timeout score 0 and stay in every mean |
| Aggregation | episode (mean over trials) -> scenario -> primary family -> overall, each an unweighted mean |
| Uncertainty | stratified two-stage percentile bootstrap, 95% |
| Comparison | paired; four outcomes; "better" needs the interval to exclude 0 and a difference of at least 1 point |
| Leaderboard | score to 0.1, then safety-first tie-breaks; separate from significance |
| Environment | one frozen environment pack per evaluation, identical for every model |

## 2. Literature

Each source below is a primary source. The right column says what EMBERBench
takes from it.

| Source | Relevant finding | Use in EMBERBench |
| --- | --- | --- |
| Mattson et al., *MLPerf Training Benchmark*, MLSys 2020, arXiv:1910.01500 | A result counts only when the model reaches a fixed quality target; time-to-train is measured only after that gate; closed division rules fix the environment; multiple runs are required because of run-to-run variance. | Safety acts as a gate and a multiplier, not as one more additive term. Fixed environment (pack). Repeated trials for stochastic models. |
| Reddi et al., *MLPerf Inference Benchmark*, ISCA 2020, arXiv:1911.02549 | Scenarios have fixed quality targets and latency bounds; results that miss the target are invalid, not "partially scored". | Invalid output and timeouts score 0 and are never dropped. Cost and latency are reported separately from quality. |
| Liang et al., *Holistic Evaluation of Language Models* (HELM), TMLR 2023, arXiv:2211.09110 | A single number hides trade-offs; report many metrics side by side under standardized conditions; aggregate with explicit, documented rules. | The headline ships with every component, family score, tail statistic and coverage count. Operational metrics are separate. |
| Cobbe et al., *Leveraging Procedural Generation to Benchmark RL* (Procgen), ICML 2020, arXiv:1912.01588 | Normalize each environment's return with per-environment constants so environments are comparable; evaluate generalization on held-out levels. | Each component is normalized per episode by a privileged reference. Held-out generated suites in packs. |
| Bellemare et al., *The Arcade Learning Environment*, JAIR 2013; Mnih et al., *Human-level control through deep RL*, Nature 2015 | Human-normalized score `(agent - random) / (human - random)`; aggregate across games after normalization. | Baseline/oracle normalization was considered (section 5.6); the privileged optimum is used as the upper reference instead of a human. |
| Machado et al., *Revisiting the Arcade Learning Environment*, JAIR 2018, arXiv:1709.06009 | Evaluation protocols must be fixed and reported (stochasticity, episode termination, number of runs); the median of normalized scores is more robust than the mean. | Fixed pack and stochasticity protocol. Median and lower-tail statistics reported next to the mean. |
| Ray, Achiam and Amodei, *Benchmarking Safe Exploration in Deep RL* (Safety Gym), OpenAI 2019 | Report task return and constraint cost separately; a policy that violates constraints is not "better" because of a higher return. | Safety is separate from quality and multiplies it; a loss can never be bought back by speed beyond the documented bound. |
| Ji et al., *Safety-Gymnasium*, NeurIPS 2023 Datasets and Benchmarks, arXiv:2310.12567 | Unified safe-RL benchmark; reward and cost are reported together; safety is evaluated per task. | Per-family reporting of the Safety factor. |
| Altman, *Constrained Markov Decision Processes*, 1999; Achiam et al., *Constrained Policy Optimization*, ICML 2017, arXiv:1705.10528 | Safety as a constraint, not as a weighted reward term; constraint satisfaction comes before return. | Catastrophic gate; leaderboard tie-breaks are lexicographic with safety first. |
| Roijers et al., *A Survey of Multi-Objective Sequential Decision-Making*, JAIR 2013; Hayes et al., *A practical guide to multi-objective RL and planning*, AAMAS journal 2022, arXiv:2103.09568 | A scalar utility must be stated explicitly; linear scalarization assumes compensation between objectives; non-linear utilities express non-compensation. | The score is an explicit non-linear utility: a product, plus a geometric mean for quality, so objectives do not compensate linearly. |
| Fleming and Wallace, *How not to lie with statistics: the correct way to summarize benchmark results*, CACM 1986 | For normalized ratios, the geometric mean is the consistent summary; the arithmetic mean of ratios depends on the reference. | Quality is a weighted geometric mean of normalized ratios. |
| Henderson et al., *Deep RL that Matters*, AAAI 2018, arXiv:1709.06560; Colas et al., *How many random seeds?*, arXiv:1806.08295 | Results vary strongly across seeds; few runs give misleading conclusions; report variance and use significance tests. | Multiple seeds per scenario, intervals, and an explicit "inconclusive" outcome. |
| Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021, arXiv:2108.13264 | Point estimates over few runs are unreliable; use stratified bootstrap confidence intervals across tasks, robust aggregates, performance profiles and probability of improvement. | Stratified (by family) two-stage bootstrap; median and tail statistics; paired comparison. |
| Chan et al., *Measuring the Reliability of RL Algorithms*, ICLR 2020, arXiv:1912.05663 | Report dispersion and risk (for example CVaR of the worst runs), not only the mean. | CVaR10, p10, worst scenario and zero-score counts in every report. |
| Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021, arXiv:2103.03098 | Randomize and pair sources of variance; compare with paired tests and a meaningful effect threshold. | Paired episodes (common random numbers through the frozen fire); 1-point practical margin. |
| Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018 | Use paired tests on the same test items; bootstrap is a valid non-parametric choice. | Paired bootstrap over the same scheduled episodes. |
| Efron and Tibshirani, *An Introduction to the Bootstrap*, 1993 | Percentile intervals; stratified and multi-stage resampling for grouped data. | Two-stage resampling (scenarios within family, episodes within scenario). |
| Kruschke, *Rejecting or accepting parameter values in Bayesian estimation*, AMPPS 2018 | A region of practical equivalence separates "statistically different" from "practically different". | `practical_difference = 1` point. |
| Sokolova and Lapalme, *A systematic analysis of performance measures for classification tasks*, IP&M 2009 | Micro averages weight large classes; macro averages weight each class equally. | Macro (family-balanced) headline; micro and scenario-balanced reported as diagnostics (section 7). |
| Rockafellar and Uryasev, *Optimization of conditional value-at-risk*, J. Risk 2000 | CVaR is a coherent measure of the tail. | CVaR10 of episode scores. |
| Manheim and Garrabrant, *Categorizing Variants of Goodhart's Law*, arXiv:1803.04585, 2018; Skalse et al., *Defining and Characterizing Reward Hacking*, NeurIPS 2022, arXiv:2209.13085 | An optimized proxy diverges from the goal; any scalar can be gamed unless its failure modes are tested. | Adversarial policies and requirement checks (section 6). |
| Helbing, Farkas and Vicsek, *Simulating dynamical features of escape panic*, Nature 2000 | "Faster is slower": pushing at bottlenecks reduces the outflow. | Congestion is penalized non-linearly; the "jostler" adversary tests it. |
| ISO 13571:2012, *Life-threatening components of fire*; the ASET/RSET concept of fire-safety engineering | Exposure accumulates with time near the hazard; safety needs the required egress time below the available egress time. | Hazard is a rate of avoidable danger-zone states; time is relative to the privileged earliest escape. |
| Ouyang et al., *An empirical study of the non-determinism of ChatGPT in code generation*, ACM TOSEM 2024, arXiv:2308.02828 | Outputs vary between identical requests even at temperature 0. | Repeated trials; temperature 0 is not a determinism guarantee. |

## 3. Design principles and how each is met

| Principle | Mechanism |
| --- | --- |
| Failure-sensitive | multiplicative form; a zero component gives 0 |
| Every component in `[0, 1]`; no bonus above 1 | each component is a privileged bound over the actual value, or a probability-like factor |
| Safety dominates | Safety multiplies Quality; an avoidable casualty caps the episode at 60 |
| A casualty hurts more than a success helps | incident factor 0.6 on the first casualty; the last evacuee adds at most `1 - (1 - 1/M*)^2` |
| Multiple casualties progressively worse | the incident factor is paid again at every doubling of `1 + c`, and the rate term grows |
| Stranding and incomplete evacuation independent | separate `P_stranding` and `E` |
| Non-linear congestion | `(1 - cbr)^2` |
| Difficulty-relative time | `T = sum(L_i) / sum(A_i)` with the privileged earliest escape |
| Exposure penalized without deaths | `H` counts danger-zone states of evacuees above their privileged minimum |
| Bad routing penalized | `P` against the fire-aware privileged minimum of moves |
| Failure to replan penalized | `R` on route-invalidation events, plus the losses in Safety |
| Invalid output, model failure and timeout never disappear | scheduled episodes without a result score 0 and are counted |
| No blind double counting | Time, Hazard and Path use evacuees only, so losses are only in Safety; correlations checked (section 6.5) |
| Cost separate | tokens, money and latency are never inputs |

## 4. Metric classification

| Metric | Role | Reason |
| --- | --- | --- |
| evacuation rate | core | the primary outcome; normalized by `M*` so impossible rescues are not charged |
| casualties | safety gate and core | the most severe outcome |
| stranded | safety gate and core | a person still inside at the horizon is a failure, but less certain than a casualty |
| time to safety | core | speed matters once people are safe; evacuees only, so a death never "improves" time |
| congestion block rate | core | measures coordination failure directly; the three people-caused block types only (static wall and fire blocks are not coordination) |
| fire-exposure proxy (`sum 1/d`) | diagnostic | it accumulates with time, so it double-counts speed (Spearman 0.87 with Time) and rewards reckless fast policies (section 5.5) |
| static path efficiency | diagnostic | it penalizes necessary fire detours (section 5.7) |
| replanning proxies | diagnostic | they react to any block, not to route invalidation by the fire |
| exit utilization / balance | diagnostic only | the right split depends on geometry and fire; rewarding balance would reward a 50/50 split when one exit is better |
| contested cells and hotspots | diagnostic only | contested cells overlap conflict blocks |
| decision cost | operational only | not a property of evacuation quality |

## 5. Candidate formulas and why they were rejected

Every candidate was scored on the same evidence (section 6). The full
tables are in [scoring-study-results.md](scoring-study-results.md).

### 5.1 Additive weighted sum

`100 * sum(w_k * x_k)` over evacuation, survival, stranding and the quality
components. **Rejected.** One casualty in 100 scores 99.4 (R1 fails), 10
casualties in 100 score 94.4, and 50% losses still score above 0. The
"abandon the farthest 15%" adversary is only 7.2 points below
congestion-aware. A weighted sum lets speed and order buy back lives, which
violates the Safety Gym and constrained-MDP view.

### 5.2 Plain product of rates

`100 * (evacuated / N) * (1 - c / N) * ...`. **Rejected.** One casualty in
100 scores 99.0. A product of rates is failure-sensitive only when a rate is
near 0, so isolated deaths in large crowds are almost free. It also fails
R5, R6, R7 and R9.

### 5.3 Official form without the incident factor (`alpha = 1`)

**Rejected.** One casualty in 100 scores 94.1 and one stranded person 96.1,
so a run with a death keeps a "95-level" score. R1, R5, R6 and R7 fail. The
rate term alone scales with the population; the discrete incident factor is
needed so that the first loss is expensive in any crowd.

### 5.4 Arithmetic (compensatory) quality mean

**Rejected.** Time 0.1 with everything else perfect scores 73 (R10 fails):
a very slow evacuation is hidden by perfect coordination and path. The
geometric mean gives 42 and keeps each component's failure visible, as
Fleming and Wallace recommend for normalized ratios.

### 5.5 Dose-based hazard (`sum 1/d` relative to the privileged minimum dose)

**Rejected.** The first study run used the metrics exposure proxy. It has
Spearman 0.87 with Time, so it counts speed twice. It also rewarded the
reckless fire-ignoring adversary: Hazard 69.1 against 64.5 for
congestion-aware, because a fast run accumulates less dose even when it
passes close to the fire. The replacement is the **avoidable danger-zone
rate**: states within 2 cells of fire above the privileged minimum, divided
by the evacuee states. Its Spearman correlation with Time is 0.14, and
reckless now scores below congestion-aware on Hazard (89.4 against 89.9).

### 5.6 Horizon-normalized time `(W - A) / (W - L)`

`W` is the horizon, `A` the actual and `L` the lower bound. **Rejected.** It
depends on the scenario horizon and compresses differences: with
`H = 200`, `L = 20` and `A = 40` it gives 0.89, while the ratio `L / A` gives
0.5. A long horizon would make slow runs look good. Baseline normalization
`(A - random) / (best - random)` was rejected for the same reason: the
reference baseline changes the scale, and random scores 0 on almost every
scenario. The degenerate cases of `L / A` are handled: `L >= 1` for every
evacuee (an exit is never a start cell), and with no evacuee Time is
undefined, not 1.

### 5.7 Static path efficiency

The metrics path efficiency uses the wall-only distance. **Rejected** as a
score input: it charges a policy for a detour that the fire made necessary
(first study: fire-aware 95.3 against static 100). The replacement bound is
the privileged fire-aware minimum number of moves. It reduces that
unfairness (fire-aware 97.3); the rest is the extra distance of the
fire-aware baseline over the privileged optimum. Path has Spearman 0.48 with
Time and the smallest weight (0.1).

### 5.8 Replanning set to 1 when nothing was invalidated

**Rejected.** It would fabricate a perfect component and raise the scores of
easy scenarios. Replanning is undefined there and its weight is shared by
the other components.

## 6. Sensitivity study

### 6.1 Setup

`python tools/scoring_sensitivity.py` plays 14 policies on two suites:

- the public V1 suite with the public development seeds 0, 1, 2 (30
  episodes), and
- a generated stochastic study suite: 16 scenarios on a 16x12 grid with 6 to
  24 people, stochastic fire and a horizon of 80, from a public config and a
  constant test-only generator seed, with seeds 0, 1, 2 (48 episodes).

Policies:

- reference baselines: always-stay, `random-legal-v1`,
  `nearest-exit-greedy-v1`, `static-shortest-path-v1`,
  `fire-aware-shortest-path-v1`, `congestion-aware-v1`;
- adversaries built to game a score: **reckless-fast** (load-balanced
  shortest paths that ignore the fire), **abandon-far** (the 15% farthest
  people stay), **safe-slow** (moves every third step unless the fire is
  close), **late-start** (waits 8 steps), **jostler** (prefers occupied
  cells), **herd** (everyone to one exit), **wanderer** (sideways moves when
  far from the fire) and **hider** (runs from the fire instead of to an
  exit unless an exit is close).

The privileged bound is a reference, not a policy.

### 6.2 Requirements

A candidate is acceptable only if it meets all of these on hand-made
evidence:

| ID | Requirement |
| --- | --- |
| R1 | one casualty in 100, otherwise perfect, scores at most 70 |
| R2 | a safe run with Time 0.25 beats a perfect run with one casualty in 100 |
| R3 | two casualties score below one |
| R4 | 10 casualties in 100 still score at least 5 (unsafe runs stay separable) |
| R5 | 10 casualties in 100 score at most 35 |
| R6 | one stranded in 100 scores at most 85 and above one casualty |
| R7 | 10 stranded in 100 score at most 50 |
| R8 | a congestion block rate of 0.3 costs at least 10 points |
| R9 | 50% losses score 0 |
| R10 | Time 0.1 with everything else perfect scores at most 60 (non-compensation) |

Results: the official form passes all ten. Additive passes 2 (R3, R4).
Plain-product passes 4. No-incident-factor passes 5. Arithmetic quality
passes 9 and fails R10.

### 6.3 Adversaries

Score of each adversary minus congestion-aware (both suites):

| Adversary | Official | Additive | Plain product | No incident factor | Arithmetic quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| reckless-fast | -3.6 | -3.4 | -2.9 | -6.1 | -3.8 |
| abandon-far | -31.4 | -7.2 | -9.0 | -31.5 | -31.8 |
| safe-slow | -30.0 | -24.1 | -36.7 | -36.5 | -29.5 |
| late-start | -20.3 | -29.6 | -35.9 | -27.2 | -19.9 |
| jostler | -8.4 | -2.9 | -2.7 | -9.9 | -8.6 |
| herd | -12.0 | -8.7 | -12.0 | -16.4 | -12.2 |
| wanderer | -29.1 | -21.5 | -42.3 | -34.9 | -28.8 |
| hider | -24.0 | -30.2 | -29.9 | -29.7 | -23.7 |

Every adversary scores below the best baseline under the official form.
Abandoning people costs 31 points, not 7 as under the additive form.

One result needs a note. On the generated suite alone, reckless-fast (31.2)
scores above congestion-aware (29.6). There, reckless-fast loses fewer people
(283 against 294 casualties of 570 savable). The score follows safety; this
is not a gaming failure. On the public suite, reckless-fast is 5.1 points
below congestion-aware.

### 6.4 Official ranking (both suites)

| Rank | Policy | EMBER |
| ---: | --- | ---: |
| 1 | congestion-aware | 45.0 |
| 2 | adv-reckless-fast | 41.4 |
| 3 | adv-jostler | 36.7 |
| 4 | fire-aware | 36.5 |
| 5 | static-shortest | 34.7 |
| 6 | adv-herd | 33.0 |
| 7 | nearest-greedy | 25.1 |
| 8 | adv-late-start | 24.7 |
| 9 | adv-hider | 21.0 |
| 10 | adv-wanderer | 15.9 |
| 11 | adv-safe-slow | 15.0 |
| 12 | adv-abandon-far | 13.6 |
| 13 | always-stay | 0.0 |
| 14 | random-legal | 0.0 |

The baselines are in the expected order: congestion-aware > fire-aware >
static > greedy > random = stay.

### 6.5 Redundancy

Spearman rank correlation over 908 episodes of moving policies where every
component is defined:

|  | Safety | Time | Coordination | Hazard | Path |
| --- | ---: | ---: | ---: | ---: | ---: |
| Safety | 1.00 | 0.30 | 0.33 | 0.13 | 0.09 |
| Time | | 1.00 | 0.37 | 0.14 | 0.48 |
| Coordination | | | 1.00 | 0.08 | -0.01 |
| Hazard | | | | 1.00 | 0.05 |
| Path | | | | | 1.00 |

No pair is above 0.5. Path and Time are the most related (0.48); Path keeps
the smallest weight. Replanning is defined on fewer episodes; it overlaps
with Safety on purpose, because a failure to replan both delays and kills,
and the two parts measure different things (delay of the survivors, and the
losses).

### 6.6 Constant grid and rank stability

405 combinations: `alpha_c` in {0.5 ... 0.9}, `k_c` in {2, 4, 6}, `kE` in {1,
2, 3}, `alpha_s` in {0.7, 0.8, 0.9}, `kC` in {1, 2, 3}. 52 meet every
requirement. All feasible combinations have `alpha_c` in {0.5, 0.6},
`alpha_s` in {0.7, 0.8} and `kC` in {2, 3}. The official constants
(`alpha_c = 0.6`, `k_c = 4`, `kE = 2`, `alpha_s = 0.8`, `kC = 2`) are
feasible and central in this region. Kendall tau between the official
ranking and the ranking of each feasible combination: minimum 0.956, mean
0.982. The first draft used `alpha_c = 0.7`; it failed R2 (one casualty in
100 scored 65.9, above a safe run with Time 0.25), so it was rejected.

### 6.7 Weight perturbation

Each quality weight multiplied by 0.5 and by 1.5, then renormalized: Kendall
tau is at least 0.978 and no policy moves by more than 1.9 points. The
ranking does not depend on the exact weights.

## 7. Aggregation

| Option | Effect | Decision |
| --- | --- | --- |
| micro (mean of all episodes) | large scenario sets and many seeds dominate | diagnostic |
| scenario-balanced | each scenario weighs the same; a family with many scenarios dominates | diagnostic |
| family-balanced | each failure mode weighs the same | **headline** |

Public suite, development seeds 0, 1, 2 (each public scenario is its own
primary family, so family-balanced equals scenario-balanced there):

| Baseline | Family-balanced | Scenario-balanced | Micro | 95% CI | Worst scenario | CVaR10 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| always-stay (control) | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| `random-legal-v1` | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| `nearest-exit-greedy-v1` | 26.8 | 26.8 | 26.8 | [23.6, 32.9] | 0.0 | 0.0 |
| `static-shortest-path-v1` | 36.7 | 36.7 | 36.7 | [33.5, 42.8] | 0.0 | 0.0 |
| `fire-aware-shortest-path-v1` | 39.6 | 39.6 | 39.6 | [37.1, 44.6] | 0.4 | 0.4 |
| `congestion-aware-v1` | 49.6 | 49.6 | 49.6 | [47.1, 54.6] | 7.1 | 7.1 |

On the public suite the three means are equal, because every scenario has
three seeds and its own primary family. They differ on a suite where a family
has several scenarios or scenarios have different numbers of applicable
episodes.

Trials are averaged first, so repeated trials never act as extra scenarios.
Seeds are averaged inside each scenario. A suite cannot raise a model's
score by adding easy scenarios of one family.

## 8. Statistical rule

- **Interval.** Stratified two-stage percentile bootstrap, 95%, 2000
  resamples, deterministic SHAKE-256 stream (scoring.md, section 7).
- **Comparison.** Paired by episode. `a_better` needs the interval of the
  paired difference to exclude 0 **and** a difference of at least 1 point.
  Otherwise `inconclusive`, unless every paired score is identical
  (`exact_tie`).
- **Leaderboard.** Order by the score to 0.1, then Safety, avoidable
  casualties, avoidable stranded, worst scenario, model ID. Order is not
  significance; publish both.
- **Interpretation of the interval.** It describes this sample of scenarios
  and seeds. With one scenario per family it cannot see scenario variance.
  Claims about unseen environments need held-out generated suites with
  several scenarios per family.

Paired comparisons of adjacent baselines on the public suite (seeds 0, 1, 2):

| A vs B | Delta | 95% CI | Outcome |
| --- | ---: | --- | --- |
| static vs greedy | 9.90 | [9.90, 9.90] | `a_better` |
| fire-aware vs static | 2.84 | [1.78, 3.53] | `a_better` |
| congestion-aware vs fire-aware | 10.06 | [10.06, 10.06] | `a_better` |

The zero-width intervals come from deterministic policies on scenarios whose
fire is deterministic: only `v1-pub-10-route-invalidation` varies by seed,
and there two of the pairs score the same.

## 9. Limitations

- The references are single-person relaxations. They ignore other people, so
  Time and Path can stay below 1 even for an optimal crowd policy in a dense
  scenario. This makes the scale conservative, not biased between models.
- Hazard uses a fixed radius of 2 cells and ignores walls in the distance,
  like the metrics proxy.
- With one scenario per family the interval reflects only seed and trial
  variance (section 8).
- The constants are chosen by explicit requirements and checked for rank
  stability; other reasonable requirements would give other constants. The
  study and the requirements are published so that they can be challenged.
- The score is not lexicographic in safety. Within one episode, a safe run
  more than about 4.6 times slower than the privileged bound can score below
  a perfect run with one casualty in a large crowd (scoring.md, section 13).

## 10. Reproduce

```bash
python tools/scoring_sensitivity.py --cache .scoring-study
```

The cache directory holds the generated study suite and per-episode
evidence; it is ignored by git. Delete it after any change to the evidence
definition.
