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
| Form | `EMBER = 100 * Safety * Quality^beta * Severity`; every factor in `[0, 1]` (design F of section 5.9) |
| Safety | `E^2 * P_casualty * P_stranding`, with a discrete incident factor plus a rate term for each loss type, and a catastrophic gate at half the savable people |
| Attribution | per person: a loss is avoidable when that person's privileged earliest escape exists; unavoidable losses are never transferred between people |
| Quality | weighted geometric mean of Time, Coordination, Hazard, Path and Replanning (weights 0.3 / 0.2 / 0.2 / 0.1 / 0.2), raised to the derived exponent `beta = ln 0.6 / ln 0.5` |
| Severe failures | per quality component, a threshold in natural units; each severe component multiplies the score by `0.8 * (q / q_severe)` |
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
| Safety dominates | Safety multiplies Quality; an avoidable casualty caps the episode below 60; `beta` makes every run without loss and without a severe failure score above a perfect run with one casualty (R12) |
| Time and quality still matter | below the thresholds Quality is continuous (Time 0.5 costs 17.4 points); severe failures cap the score below 80 (R2, R13, R14) |
| A casualty hurts more than a success helps | incident factor 0.6 on the first casualty; the last evacuee adds at most `1 - (1 - 1/M*)^2` |
| Multiple casualties progressively worse | the incident factor is paid again at every doubling of `1 + c`, and the rate term grows |
| Stranding and incomplete evacuation independent | separate `P_stranding` and `E` |
| Non-linear congestion | `(1 - cbr)^2`, plus the severe gate at `cbr > 1/4` |
| Difficulty-relative time | `T = Q* / sum(A_i)` with the exit-capacity queue bound over the privileged earliest escapes; severe below 1/2 |
| Exposure penalized without deaths | `H` counts danger-zone states of evacuees above their privileged minimum; severe above 1/4 |
| Bad routing penalized | `P` against the fire-aware privileged minimum of moves |
| Failure to replan penalized | `R` on route-invalidation events, plus the losses in Safety; severe below 1/2 |
| Invalid output, model failure and timeout never disappear | scheduled episodes without a result score 0 and are counted |
| No blind double counting | Time, Hazard and Path use evacuees only, so losses are only in Safety; Path and Hazard are relative to fire-aware privileged minima, so necessary detours and unavoidable exposure are not charged; correlations checked (section 6.5) |
| Cost separate | tokens, money and latency are never inputs; they are in a separate `OperationalRecord` |

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
| exit utilization / balance | diagnostic only | the right split depends on geometry and fire; rewarding balance would reward a 50/50 split when one exit is better. Allocation regret was considered: its cost is the extra time of a bad allocation, which `T` already measures against the allocation-optimal queue bound `Q*`, so a separate term would double-count it |
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
geometric mean keeps each component's failure visible, as Fleming and
Wallace recommend for normalized ratios: it gives 42 alone (design A,
section 5.9), and 8.5 with the severe gate of the official form.

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

### 5.9 How Safety dominates: designs A to F

The first version 1.0 draft was `100 * Safety * Quality` (design A). It has
a safety ceiling (one casualty: at most 60) but not safety dominance: a safe
run with Time below about 0.22 scored below a perfect run with one casualty
in a large crowd. A pure lexicographic order fixes that but makes Time
irrelevant. Six designs were compared on the same evidence and the same
requirements (section 6.2, R1 to R15) and pre-registered orderings
(section 6.3):

| Design | Definition | Fails | Decision |
| --- | --- | --- | --- |
| A multiplicative (draft) | `100 * Safety * Quality` | R12, R13, R14 | rejected: a severe bottleneck (rate 0.3) costs only 16 points; Time 0.1 still scores 42 |
| B hard ceilings | A, capped at 80 when any component is severe | R12, R13, R14 | rejected: a flat cap does not grow with severity, and A's dominance gap remains |
| C safety tiers | Quality interpolates between this Safety level and the next worse one | R2, R8, R10, R13, R14 | rejected: Quality moves the score too little; Time 0.1 scores 75 |
| D lexicographic (epsilon 0.05) | `100 * Safety * (1 - 0.05 * (1 - Quality))` | R2, R8, R10, R13, R14 | rejected: Time becomes irrelevant (Time 0.1 scores 97) |
| E soft minimum | weighted power mean (p = -5) of Safety and the quality components | R1, R11, R12, R14 | rejected: one casualty scores 63, and a severe exposure (rate 0.3) still scores 92 |
| F constrained geometric | `100 * Safety * Quality^beta * Severity` | none | **adopted** |

Two variants of F were also checked. With the dominance floor set at the
stranding factor 0.8 instead of the casualty factor 0.6 (`beta = 0.32`),
Quality is too flat: Time 0.5 costs only 8 points (R2 fails). Plain
products and additive forms fail the safety requirements (sections 5.1 and
5.2).

Why F works:

- `beta = ln(0.6) / ln(0.5)` is derived from the casualty incident factor
  and the smallest severe threshold. Every run whose quality components are
  all at or above their thresholds has `Quality^beta >= 0.6`, which is above
  the Safety of any run with an avoidable casualty. So R12 holds by
  construction, in every population, not only in the tested cases.
- Below the thresholds, Quality is continuous and still costs points: Time
  0.5 costs 17.4 points, a block rate of 0.25 costs 10.1 points.
- At a threshold, the severe factor `0.8 * (q / q_severe)` adds a step and a
  slope. A severe failure caps the score below 80 and keeps falling with
  severity, so a severe bottleneck, delay, exposure or replanning failure
  is a substantial, permanent loss (R13, R14).

### 5.10 Aggregate attribution of unavoidable losses

The draft charged unavoidable losses to casualties first, from counts:
`c = max(0, C - U)` with `U = N - M*`. **Rejected.** It transfers an
"allowance" between people. If person `a` could escape but dies, and person
`b` could not escape and is stranded, the draft gives `c = 0`: the
avoidable death disappears. The official rule is per person (scoring.md,
section 4.1). On the study episodes the two rules happen to agree (0 of
1170 episodes differ), but `test_real_episode_person_level_attribution`
builds a real replayed episode where the draft hides a casualty. R11 checks
the same case on synthetic evidence.

### 5.11 Exit balance and allocation regret

A 50/50 exit balance is never a goal and is not scored. An allocation
regret term (extra time over an allocation-optimal schedule) was
considered. **Not adopted as a separate term:** the time cost of a bad
allocation is already in `T`, because `T` compares with the exit-capacity
queue bound `Q*`, which is allocation-optimal. A separate term would charge
the same seconds twice. Exit utilization stays a diagnostic.

## 6. Sensitivity study

### 6.1 Setup

`python tools/scoring_sensitivity.py` plays 15 policies on two suites:

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
  far from the fire), **hider** (runs from the fire instead of to an exit
  unless an exit is close) and **fire-hugger** (congestion-aware routing
  that prefers cells next to the fire).

The privileged bound is a reference, not a policy. All 1170 episodes
(15 policies x 78 episodes) are scored with person-level attribution.

### 6.2 Requirements

A candidate is acceptable only if it meets all of these on hand-made
evidence. They were written before the candidates were scored, and they
pin the thresholds normatively: R2 says that Time must matter, R13 and R14
say what a severe failure must cost.

| ID | Requirement |
| --- | --- |
| R1 | one avoidable casualty in 100, otherwise perfect, scores at most 60 |
| R2 | Time is not irrelevant: Time 0.5 (not severe) costs at least 10 points |
| R3 | two casualties score below one |
| R4 | 10 casualties in 100 still score at least 5 (unsafe runs stay separable) |
| R5 | 10 casualties in 100 score at most 35 |
| R6 | one stranded in 100 scores at most 85 and above one casualty |
| R7 | 10 stranded in 100 score at most 50 |
| R8 | a block rate of 0.3 costs at least 10 points |
| R9 | 50% avoidable losses score 0 (catastrophic gate) |
| R10 | Time 0.1 with everything else perfect scores at most 60 (non-compensation) |
| R11 | a savable casualty is not hidden by another person's unavoidable loss |
| R12 | a run with no casualty and no severe failure beats a perfect run with one casualty (`M*` 2 to 1000, 0 or 1 extra stranded) |
| R13 | a severe bottleneck (block rate 0.3) without loss scores below 80 and loses at least 20 points |
| R14 | severe Time 0.4, Hazard 0.3, Replanning 0.4 and Path 0.4 alone each score below 80 |
| R15 | failed, missing and timed-out episodes score 0 and stay in every mean (structural; tested in `tests/test_scoring.py`) |

Results: the official form (F) passes all 15. The other designs fail as
listed in section 5.9.

### 6.3 Pre-registered orderings

These orderings were fixed before the study. Left must score above right.
The official form passes all ten:

| Ordering | Official scores |
| --- | --- |
| zero-loss with Time 0.6 > perfect quality with 1 casualty | 86.8 > 56.5 |
| perfect > zero-loss but slow (Time 0.5) | 100 > 82.6 |
| perfect > zero-loss with severe congestion (0.3) | 100 > 61.1 |
| perfect > zero-loss with high exposure (0.3) | 100 > 69.9 |
| perfect > zero-loss with inefficient routing (Path 0.6) | 100 > 95.4 |
| Replanning 0.9 > static-good with invalidation failure (Replanning 0.3) | 98.5 > 40.2 |
| 1 stranded > 1 casualty | 76.8 > 56.5 |
| 1 casualty > 2 casualties | 56.5 > 39.4 |
| mild congestion (0.05) > severe congestion (0.4) | 98.1 > 42.4 |
| mild delay (Time 0.8) > extreme delay (Time 0.1) | 94.0 > 8.5 |

### 6.4 Adversaries

Score of each adversary minus congestion-aware (both suites). This is a
gaming check, not a selection criterion:

| Adversary | Official F | A | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| reckless-fast | -2.3 | -3.7 | -4.1 | -5.0 | -4.6 |
| abandon-far | -31.1 | -32.4 | -33.6 | -35.0 | -33.0 |
| safe-slow | -33.0 | -31.2 | -31.9 | -29.2 | -34.3 |
| late-start | -21.7 | -21.3 | -21.5 | -18.3 | -24.3 |
| jostler | -8.0 | -9.3 | -10.3 | -11.1 | -10.0 |
| herd | -11.0 | -12.9 | -14.0 | -13.7 | -14.1 |
| wanderer | -30.8 | -30.2 | -30.4 | -27.0 | -32.0 |
| hider | -24.6 | -24.4 | -24.9 | -24.4 | -26.9 |
| fire-hugger | -1.9 | -1.7 | -1.4 | -1.5 | -2.8 |

Every adversary scores below the best baseline under every multiplicative
design. Reckless-fast and fire-hugger are close to congestion-aware because
they reuse its routing; their extra losses and exposure cost about 2 points.
The severe gates make safe-slow lose more than under A, because its Time is
severe in about half of its episodes.

### 6.5 Official ranking (both suites)

| Rank | Policy | EMBER | Public only | Generated only |
| ---: | --- | ---: | ---: | ---: |
| 1 | congestion-aware | 43.8 | 49.1 | 26.0 |
| 2 | adv-fire-hugger | 41.9 | 46.1 | 27.9 |
| 3 | adv-reckless-fast | 41.5 | 44.5 | 31.4 |
| 4 | adv-jostler | 35.8 | 39.9 | 22.2 |
| 5 | fire-aware | 35.7 | 39.8 | 22.1 |
| 6 | static-shortest | 34.8 | 36.9 | 28.1 |
| 7 | adv-herd | 32.8 | 37.0 | 18.9 |
| 8 | nearest-greedy | 25.2 | 26.9 | 19.3 |
| 9 | adv-late-start | 22.0 | 28.4 | 0.8 |
| 10 | adv-hider | 19.1 | 22.5 | 8.1 |
| 11 | adv-wanderer | 12.9 | 13.8 | 9.9 |
| 12 | adv-abandon-far | 12.7 | 11.1 | 18.2 |
| 13 | adv-safe-slow | 10.8 | 11.2 | 9.3 |
| 14 | always-stay | 0.0 | 0.0 | 0.0 |
| 15 | random-legal | 0.0 | 0.0 | 0.0 |

Over both suites and on the public suite, the baselines are in the expected
order: congestion-aware > fire-aware > static > greedy > random = stay.

**Generated suite alone: the order differs, and this is reported, not
tuned away.** There, reckless-fast (31.4) and static-shortest (28.1) score
above congestion-aware (26.0), and fire-aware (22.1) is lowest of the four.
Congestion-aware is still the safest of the four (91 avoidable casualties
against 109 for static and 102 for reckless-fast; Safety 0.348 against
0.304). It loses its lead through the severe coordination gate: block rate
above 1/4 in 7 of 48 generated episodes, against 0 for static and
reckless-fast. Routing around the fire concentrates people into fewer
corridors, and a policy that keeps more people alive also has more people
in the queues. Under design A (no gates) congestion-aware is third on that
suite (30.1, behind reckless-fast 31.6 and fire-hugger 30.4) and above
static (28.3). The thresholds were not changed to restore the expected
order: they are fixed by R13 and R14. Section 9 records the limitation
behind this result. The constants were not chosen to produce
this order. They are fixed by R1 to R15, and section 6.7 reports the rank
stability over every combination of the grids, feasible or not.

**Per family.** The family table is in
[scoring-study-results.md](scoring-study-results.md). On
`route_invalidation_replanning`, static-shortest scores 32.6, and fire-aware
and congestion-aware score 44.0: the static policy fails to replan. Over
both suites, Replanning is severe in 19.2% of the static policy's episodes,
against 9.0% for congestion-aware. Late-start is best on that family
(70.1), because waiting lets the fire reveal the blocked route. It pays for
that elsewhere: Time is severe in 52.6% of its episodes.

**Severe failures.** Congestion-aware has a severe coordination failure in
16.7% of the episodes, a severe hazard failure in 6.4% and a severe
replanning failure in 9.0%. Safe-slow and late-start have severe Time in
51.3% and 52.6%. Hider has severe coordination in 60.3%.

### 6.6 Redundancy and spread

Spearman rank correlation over 220 episodes of moving policies where every
component, including Replanning, is defined:

|  | Safety | Time | Coordination | Hazard | Path | Replanning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Safety | 1.00 | 0.05 | 0.23 | -0.08 | -0.34 | 0.49 |
| Time | | 1.00 | 0.20 | 0.05 | 0.55 | -0.12 |
| Coordination | | | 1.00 | -0.23 | -0.15 | 0.35 |
| Hazard | | | | 1.00 | 0.24 | -0.51 |
| Path | | | | | 1.00 | -0.56 |
| Replanning | | | | | | 1.00 |

No pair is above 0.56 in absolute value. Path and Time are the most related
positive pair (0.55); Path keeps the smallest weight (0.1). Replanning
overlaps with Safety (0.49) on purpose, because a failure to replan both
delays and kills, and the two parts measure different things (the delay of
survivors and the losses).

**Spread.** Over the moving policies, the official scores range from 10.8
to 43.8 (range 33.0, standard deviation 11.9). The other multiplicative
designs have a similar range (32.4 to 35.0), so the gates do not compress
the scale.

### 6.7 Constant grids and rank stability

Two grids vary the constants around the official values. For each
combination the study checks R1 to R14 and the Kendall tau of the policy
ranking against the official ranking. Tau is reported for **every**
combination, feasible or not, so the ranking is not a selection criterion.

**Safety grid** (81 combinations): `alpha_c` in {0.5, 0.6, 0.7}, `k_c` in
{2, 4, 6}, `kE` in {1, 2, 3}, `alpha_s` in {0.7, 0.8, 0.9}.

- 26 of 81 meet every requirement; the official constants are feasible.
- Failures: R1 in 27 (all with `alpha_c = 0.7`, one casualty above 60),
  R6 in 30, R11 in 27, R4 in 15, R7 in 9.
- Every combination: tau min 0.923, mean 0.979, largest score change 12.0
  points. Feasible combinations: tau min 0.923, mean 0.982, largest change
  7.8 points.

**Gate grid** (243 combinations): `phi` in {0.7, 0.8, 0.9}, `gamma` in
{0.5, 1, 2}, severe Time in {1/3, 1/2, 2/3}, severe block rate in {1/5,
1/4, 1/3}, severe hazard rate in {1/5, 1/4, 1/3}. The quality exponent
`beta` is re-derived for each combination, and R12 is checked at that
combination's own threshold corner.

- 68 of 243 meet every requirement; the official gates are feasible.
- Failures: R13 in 81 (all with block rate 1/3: a rate of 0.3 is then not
  severe), R14 in 141 (severe Time 1/3 or hazard rate 1/3), R8 in 27.
- Every combination: tau min 0.923, mean 0.963, largest score change 7.4
  points. Feasible combinations: tau min 0.942, mean 0.972.
- The official thresholds (Time 1/2, block rate 1/4, hazard rate 1/4) are
  the most lenient grid values that still meet R13 and R14. Stricter
  values also pass, but mark more episodes as severe. So the thresholds
  are pinned by the requirements, not by the ranking. `phi` and `gamma` are
  not pinned: all three values of each pass.

The severe Path and Replanning thresholds (1/2) are not in the grid. They
follow the same rule as Time: twice the privileged need is a failure.

### 6.8 Weight perturbation

Each quality weight multiplied by 0.5 and by 1.5, then renormalized:
Kendall tau is at least 0.981 and no policy moves by more than 1.1 points.
The ranking does not depend on the exact weights.

### 6.9 Agreement between designs

Kendall tau of each design's ranking against the official one: A 0.981,
B 0.981, C 0.962, D 0.942, E 1.000, F with the stranding floor 1.000,
additive 0.808, plain product 0.865. The multiplicative designs rank the
policies almost alike. They differ in the requirements, which is why the
requirements, not the ranking, select the design.

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
| `nearest-exit-greedy-v1` | 26.9 | 26.9 | 26.9 | [23.7, 33.2] | 0.0 | 0.0 |
| `static-shortest-path-v1` | 36.9 | 36.9 | 36.9 | [33.6, 43.2] | 0.0 | 0.0 |
| `fire-aware-shortest-path-v1` | 39.8 | 39.8 | 39.8 | [37.2, 45.0] | 0.0 | 0.0 |
| `congestion-aware-v1` | 49.1 | 49.1 | 49.1 | [46.5, 54.3] | 4.2 | 4.2 |

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

| A vs B | Delta | 95% CI | Outcome | Episodes A higher / B higher / equal |
| --- | ---: | --- | --- | --- |
| static vs greedy | 9.93 | [9.93, 9.93] | `a_better` | 3 / 0 / 27 |
| fire-aware vs static | 2.98 | [1.83, 3.65] | `a_better` | 11 / 3 / 16 |
| congestion-aware vs fire-aware | 9.26 | [9.26, 9.26] | `a_better` | 15 / 3 / 12 |

The zero-width intervals come from deterministic policies on scenarios whose
fire is deterministic: only `v1-pub-10-route-invalidation` varies by seed,
and there two of the pairs score the same.

## 9. Limitations

- The references are single-person relaxations. Path and the release steps
  of the Time bound ignore other people. The Time queue bound adds exit
  capacity, but not corridor capacity. So Time and Path can stay below 1 even
  for an optimal crowd policy in a dense scenario. This makes the scale
  conservative, not biased between models.
- Hazard uses a fixed radius of 2 cells and ignores walls in the distance,
  like the metrics proxy.
- With one scenario per family the interval reflects only seed and trial
  variance (section 8).
- The constants are chosen by explicit requirements and checked for rank
  stability; other reasonable requirements would give other constants. The
  study and the requirements are published so that they can be challenged.
- The score is not purely lexicographic in safety, on purpose. A safe run
  can score below a perfect run with one casualty only when the safe run
  itself has a severe quality failure (R12 and scoring.md, section 4.3).
- Coordination is a block rate per movement attempt without a
  capacity-aware privileged reference. A policy that keeps more people alive
  in a dense scenario also has more people in the queues, so it can cross
  the severe coordination threshold where a policy that loses people does
  not. On the generated study suite this puts congestion-aware below
  static-shortest and reckless-fast (section 6.5). A capacity-relative
  block-rate reference, like the queue bound used for Time, is the natural
  next step; it would need a new scoring version.
- The severe thresholds are normative choices pinned by R13 and R14. The
  grids show which values meet them (section 6.7), but other reasonable
  requirements would give other thresholds.

## 10. Reproduce

```bash
python tools/scoring_sensitivity.py --cache .scoring-study
```

The cache directory holds the generated study suite and per-episode
evidence; it is ignored by git. Delete it after any change to the evidence
definition.
