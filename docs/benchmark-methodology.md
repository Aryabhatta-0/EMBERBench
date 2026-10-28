# EMBERBench scientific evaluation methodology

> **Scope:** These are public development or archived research rules. They do
> not define the official hidden evaluation distribution. Previously disclosed
> recipes and study seeds are exposed research, not secret evaluation assets.
> See [Evaluation boundary](evaluation-boundary.md).


This document explains **why** the EMBER Score (scoring version 1.1) has its
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
| Quality | weighted geometric mean of Time, Hazard, Path and Replanning (weights 3/8, 2/8, 1/8, 2/8), raised to the derived exponent `beta = ln 0.6 / ln 0.5` |
| Trajectory invariance | every score input is a function of the authenticated state trajectory; identical trajectories score identically (R16) |
| Congestion | not a score input; block rate, contested cells, hotspots and exit throughput are mandatory report diagnostics; a bottleneck is charged by its realized delay (Time against the cut-capacity bound `Q_cut`), exposure and losses (section 5.12) |
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
| Cobbe et al., *Leveraging Procedural Generation to Benchmark RL* (Procgen), ICML 2020, arXiv:1912.01588 | Normalize each environment's return with per-environment constants so environments are comparable; evaluate generalization on archived levels. | Each component is normalized per episode by a privileged reference. Archived generated suites in packs. |
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
| Helbing, Farkas and Vicsek, *Simulating dynamical features of escape panic*, Nature 2000 | "Faster is slower": pushing at bottlenecks reduces the outflow. | A bottleneck is charged by the outflow it loses: Time against a cut-capacity bound. The "jostler" adversary tests it. |
| ISO 13571:2012, *Life-threatening components of fire*; the ASET/RSET concept of fire-safety engineering | Exposure accumulates with time near the hazard; safety needs the required egress time below the available egress time. | Hazard is a rate of avoidable danger-zone states; time is relative to the privileged earliest escape. |
| Ouyang et al., *An empirical study of the non-determinism of ChatGPT in code generation*, ACM TOSEM 2024, arXiv:2308.02828 | Outputs vary between identical requests even at temperature 0. | Repeated trials; temperature 0 is not a determinism guarantee. |

## 3. Design principles and how each is met

| Principle | Mechanism |
| --- | --- |
| Failure-sensitive | multiplicative form; a zero component gives 0 |
| Every component in `[0, 1]`; no bonus above 1 | each component is a privileged bound over the actual value, or a probability-like factor |
| Safety dominates | Safety multiplies Quality; an avoidable casualty caps the episode below 60; `beta` makes every run without loss and without a severe failure score above a perfect run with one casualty (R12) |
| Time and quality still matter | below the thresholds Quality is continuous (Time 0.5 costs 22.5 points); severe failures cap the score below 80 (R2, R13, R14) |
| A casualty hurts more than a success helps | incident factor 0.6 on the first casualty; the last evacuee adds at most `1 - (1 - 1/M*)^2` |
| Multiple casualties progressively worse | the incident factor is paid again at every doubling of `1 + c`, and the rate term grows |
| Stranding and incomplete evacuation independent | separate `P_stranding` and `E` |
| Trajectory invariance | every input is a function of the replayed state trajectory, never of the action log alone (R16; the block-scrub adversary gains exactly 0) |
| Bottlenecks charged by their realized effect | a severe realized bottleneck is a severe Time failure (`T < 1/2` against `Q_cut`); its exposure and losses are in Hazard and Safety; blocked actions are diagnostics |
| Difficulty-relative time | `T = Q_cut / sum(A_i)` with the exit- and cut-capacity bound over the privileged earliest escapes, so unavoidable queues at exits and doors are not charged; severe below 1/2 |
| Exposure penalized without deaths | `H` counts danger-zone states of evacuees above their privileged minimum; severe above 1/4 |
| Bad routing penalized | `P` against the fire-aware privileged minimum of moves |
| Failure to replan penalized | `R` on route-invalidation events, plus the losses in Safety; severe below 1/2 |
| Invalid output, model failure and timeout never disappear | scheduled episodes without a result score 0 and are counted |
| No blind double counting | Time, Hazard and Path use evacuees only, so losses are only in Safety; the blocked moves of people who later die are not an input; Path and Hazard are relative to fire-aware privileged minima, so necessary detours and unavoidable exposure are not charged; correlations checked (section 6.5) |
| Cost separate | tokens, money and latency are never inputs; they are in a separate `OperationalRecord` |

## 4. Metric classification

| Metric | Role | Reason |
| --- | --- | --- |
| evacuation rate | core | the primary outcome; normalized by `M*` so impossible rescues are not charged |
| casualties | safety gate and core | the most severe outcome |
| stranded | safety gate and core | a person still inside at the horizon is a failure, but less certain than a casualty |
| time to safety | core | speed matters once people are safe; evacuees only, so a death never "improves" time |
| congestion block rate | diagnostic only, mandatory in every report | an action-log count: replacing each blocked move with STAY gives the same trajectory with zero blocks, so it has no lower bound and is gameable at no cost (section 5.12) |
| fire-exposure proxy (`sum 1/d`) | diagnostic | it accumulates with time, so it double-counts speed (Spearman 0.87 with Time) and rewards reckless fast policies (section 5.5) |
| static path efficiency | diagnostic | it penalizes necessary fire detours (section 5.7) |
| replanning proxies | diagnostic | they react to any block, not to route invalidation by the fire |
| exit utilization / balance | diagnostic only | the right split depends on geometry and fire; rewarding balance would reward a 50/50 split when one exit is better. Allocation regret was considered: its cost is the extra time of a bad allocation, which `T` already measures against the allocation-optimal capacity bound `Q_cut`, so a separate term would double-count it |
| contested cells, hotspots, exit throughput | diagnostic only, mandatory in every report | contested cells and hotspots come from attempted moves (the block-scrub sets them to 0 on an identical trajectory); exit throughput is a trajectory property, but its cost is the delay already in `T` |
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
| A multiplicative (draft) | `100 * Safety * Quality` | R12, R14 | rejected: a severe exposure (rate 0.3) still scores 89; Time 0.1 still scores 32 |
| B hard ceilings | A, capped at 80 when any component is severe | R12, R14 | rejected: a flat cap does not grow with severity, and A's dominance gap remains |
| C safety tiers | Quality interpolates between this Safety level and the next worse one | R10, R13, R14 | rejected: Quality moves the score too little; Time 0.1 scores 70 |
| D lexicographic (epsilon 0.05) | `100 * Safety * (1 - 0.05 * (1 - Quality))` | R2, R8, R10, R13, R14 | rejected: Time becomes irrelevant (Time 0.1 scores 97) |
| E soft minimum | weighted power mean (p = -5) of Safety and the quality components | R1, R11, R12, R14 | rejected: one casualty scores 63, and a severe exposure (rate 0.3) still scores 90 |
| F constrained geometric | `100 * Safety * Quality^beta * Severity` | none | **adopted** |

Two variants of F were also checked. With the dominance floor set at the
stranding factor 0.8 instead of the casualty factor 0.6 (`beta = 0.32`),
Quality is flatter: Time 0.5 costs 10.6 points instead of 22.5, and a severe
Time 0.4 scores 55 instead of 46. It meets R1 to R16 with the current
weights, but only just (R2 asks for 10 points). It is not adopted: the
official `beta` is the **largest** exponent for which R12 holds by
construction (`q_min ^ beta >= alpha_c`), so it gives Quality the most
influence that safety dominance allows; any smaller exponent is an arbitrary
flattening. Plain products and additive forms fail the safety requirements
(sections 5.1 and 5.2).

Why F works:

- `beta = ln(0.6) / ln(0.5)` is derived from the casualty incident factor
  and the smallest severe threshold. Every run whose quality components are
  all at or above their thresholds has `Quality^beta >= 0.6`, which is above
  the Safety of any run with an avoidable casualty. So R12 holds by
  construction, in every population, not only in the tested cases.
- Below the thresholds, Quality is continuous and still costs points: Time
  0.5 costs 22.5 points.
- At a threshold, the severe factor `0.8 * (q / q_severe)` adds a step and a
  slope. A severe failure caps the score below 80 and keeps falling with
  severity, so a severe realized bottleneck, delay, exposure or replanning
  failure is a substantial, permanent loss (R13, R14).

### 5.10 Aggregate attribution of unavoidable losses

The draft charged unavoidable losses to casualties first, from counts:
`c = max(0, C - U)` with `U = N - M*`. **Rejected.** It transfers an
"allowance" between people. If person `a` could escape but dies, and person
`b` could not escape and is stranded, the draft gives `c = 0`: the
avoidable death disappears. The official rule is per person (scoring.md,
section 4.1). On the study episodes the two rules happen to agree (0 of
1248 episodes differ), but `test_real_episode_person_level_attribution`
builds a real replayed episode where the draft hides a casualty. R11 checks
the same case on synthetic evidence.

### 5.11 Exit balance and allocation regret

A 50/50 exit balance is never a goal and is not scored. An allocation
regret term (extra time over an allocation-optimal schedule) was
considered. **Not adopted as a separate term:** the time cost of a bad
allocation is already in `T`, because `T` compares with the capacity bound
`Q_cut`, which is allocation-optimal at the exits and at every door-like
cut. A separate term would charge the same seconds twice. Exit utilization
stays a diagnostic.

### 5.12 Coordination: the metric contract

Scoring 1.0 at commit `ce5e322` had a Coordination component
`C = (1 - block rate)^2` (weight 0.2) and a severe gate at a block rate above
1/4. This section tests whether any Coordination component can be measured
without rewarding a different action encoding of the same realized
trajectory. Full tables: [scoring-study-results.md](scoring-study-results.md).

**Invariant (adopted, R16).** If two policies produce byte-identical
authenticated state trajectories (positions, statuses and fire at every
step), their core EMBER Score is identical. It is desirable because the
benchmark claims to measure evacuation outcomes: the people, the fire and
the timing are the same, so nothing the score is meant to measure differs.
Only the action log differs, and the action log is an encoding choice.

**Block-scrub reproduction.** The adversary `adv-block-scrub` runs
`congestion-aware-v1`, predicts the public resolver, and replaces every move
that would be blocked by a person (occupied, conflict, cycle) with `STAY`.

| Suite | Episodes | Identical state trajectories | Identical trajectory evidence | Blocks (congestion-aware) | Blocks (scrub) |
| --- | ---: | ---: | ---: | ---: | ---: |
| public | 30 | 30 | 30 | 4449 | 0 |
| generated | 48 | 48 | 48 | 844 | 0 |
| archived | 48 | 48 | 48 | 1211 | 0 |

Every observation, state, fire set, evacuation, casualty, stranding,
person-time, path and danger-zone count is identical; only the action labels,
the movement outcomes, the block counts and the action-based crowd counts
(contested cells, hotspots) differ. Under the 1.0 score the scrub gains
+4.5 points on the public suite, +7.4 on the generated suite, +5.2 on both
and +3.1 on the archived suite, and up to 70.9 points in one episode (the
block-rate gate). `test_block_scrub_keeps_trajectory_and_score` is the
permanent regression test.

**The concepts.** "Coordination" mixes eight things:

| Concept | What it is | Trajectory property? | In the score? |
| --- | --- | --- | --- |
| A attempted coordination / action intent | which moves a policy asks for | no (action log) | **no**: the scrub changes it without changing the outcome |
| B realized coordination outcome | who leaves when, and who survives | yes | **yes**, as Time, Hazard and Safety |
| C unavoidable capacity congestion | queues that every policy must form at exits and doors | yes | **no charge**: inside the bound `Q_cut` |
| D avoidable congestion | queues a policy causes, for example everyone at one exit | yes | **yes**, as Time above `Q_cut` (and Hazard while waiting) |
| E queueing delay | waiting person-steps | yes | **yes**, once: it is part of person-time, so in Time |
| F exit / cut throughput loss | fewer evacuations per open slot than possible | yes | **yes**, once: lost throughput is extra person-time, so in Time; throughput is reported |
| G conflicts that change the trajectory | a contest that moves a person later | yes, through its effect | **yes**, through its effect on Time, Hazard, Safety |
| H conflicts that do not change the trajectory | a blocked move where `STAY` gives the same state | no | **no** |

So the score measures B to G through their realized effects, and A and H not
at all. Blocked moves are not all bad: in a door queue that no policy can
avoid, a move into the queue is blocked or not depending only on the
resolver's order, and the evacuation is optimal either way.

**Why blocks cannot be scored.**

1. *No lower bound.* Each other component is a privileged lower bound over
   the actual value. The block count has no positive lower bound: the scrub
   reaches 0 on every trajectory, so "bound / actual" does not exist.
2. *Gameable at no cost.* The resolver is public and pure, and its inputs are
   all in the observation. Any agent can scrub, with no change in outcome.
3. *Not defensible as decision quality.* Keeping an action-based component
   would need the statement "a policy that predicts a blocked move and emits
   STAY is genuinely better than a policy that attempts it, even when both
   produce the same trajectory". It is not defended here: predicting the
   resolver's tie-breaks is knowledge of the simulator, not evacuation
   skill, and it changes no person's fate.
4. *Dead-person double charge.* On the generated study suite, 598 of the 844
   blocks of `congestion-aware-v1` belong to people who end as casualties or
   stranded; their losses are already in Safety. G6/G7 remove these blocks,
   but the remaining blocks still fail points 1 to 3.

**Candidates.** Each candidate is scored on the same evidence (study suites
and archived suite). `G5-official` is the adopted form.

| Candidate | Definition | R16 (scrub delta) | Best adversary minus congestion-aware | Congestion-aware rank (study / archived / generated only) | tau vs official (study / archived) | Fails |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 (`ce5e322`) | `C = (1 - cbr)^2`, gate at `cbr > 1/4`, exit-capacity Time | FAIL (+5.2; max 70.9 per episode) | +5.2 (the scrub) | 2 / 2 / 5 | 0.966 / 0.966 | R16 |
| G1 | 1.0 with cut-capacity Time | FAIL (+5.2) | +5.2 | 2 / 2 / 5 | 0.966 / 0.966 | R16 |
| G4 | no Coordination; its weight moved to Time (0.5) | pass (0) | 0 | 1 / 1 / 1 | 1.000 / 1.000 | none |
| **G5 (official)** | no Coordination; weights renormalized (3/8, 2/8, 1/8, 2/8); cut-capacity Time | pass (0) | 0 | 1 / 1 / 1 | 1.000 / 1.000 | none |
| G6 | 1.0, blocks of evacuees only | FAIL (+3.6) | +3.6 | 2 / 2 / 2 | 0.983 / 1.000 | R16 |
| G7 | G6 with cut-capacity Time and the capacity-forced delay credited against blocks | FAIL (+0.1; max 3.2 per episode) | +0.1 | 2 / 2 / 3 | 1.000 / 1.000 | R16 |
| T-wait | trajectory-only: `1 - max(0, (T - m) - (Q_cut - L)) / T` | pass (0) | 0 | 1 / 1 / 1 | 1.000 / 1.000 | R2, R4, R8, R13 on synthetic profiles; a transform of Time and Path |

All candidates meet R1 to R15 in their restated form except T-wait. The 1.0
statements of R8 and R13 in action-log units ("a block rate of 0.3 alone
costs 10 points") are withdrawn: G4 and G5 fail them by design, because a
block rate without delay is not a failure of any outcome.

- *Overlap with Time.* Spearman over 1064 moving-policy episodes: Time and
  the 1.0 Coordination 0.14, Time and the G7 Coordination 0.67, Time and
  T-wait 0.88. The 1.0 component carries information that Time does not,
  but the scrub shows that this information is the action encoding. The
  more a block measure is cleaned of encoding (G7), the more it becomes
  Time.
- *Overlap with Safety.* 1.0 and G1 count the blocks of people who die (the
  dead-person double charge). G6, G7 and G5 do not.
- *Interpretability.* G5 is the simplest: delay is in one place, and the
  block rate is a reported explanation. G7 needs the cut bound, per-person
  block attribution and a credit rule, and is still not invariant.
- *Cost.* G1, G4, G5 and G7 need the cut bound (one BFS per threshold and
  one forward reachability per evacuee, well below a second per episode on
  the study suites). G6 and G7 also need per-person action attribution.
- *Generated suite.* Under 1.0 the generated suite alone ranked
  reckless-fast (31.4) and static-shortest (28.1) above congestion-aware
  (26.0), through the block-rate gate. Under G5, congestion-aware is first
  on the generated suite (33.1), on the public suite (53.2) and on the
  archived suite (40.5). This was **not** the selection criterion: G7 also
  ranks congestion-aware second only behind its own scrub, and is rejected
  for R16.

**A trajectory-only Coordination (attempted).** The natural trajectory-only
definition is the avoidable waiting of evacuees: waiting person-steps above
the capacity-forced delay, `(T - m) - (Q_cut - L)`, where `T` is person-time,
`m` cells moved, `Q_cut` the capacity bound and `L` the free-flow bound.
Every term is a Time or Path input, so the measure is an algebraic
combination of the Time and Path components, and it correlates 0.88 with
Time. Excess person-time over `Q_cut` is `T - Q_cut`, a monotone transform
of Time itself. Density-based measures (occupied neighbours, occupancy near
cuts) have no privileged lower bound that separates necessary crowding at an
exit from avoidable crowding, so they cannot be normalized per episode like
the other components. No trajectory-based, non-redundant Coordination
component exists under V1 semantics.

**Decision: D, remove Coordination from the headline score for V1.** The
quality weights are renormalized over the remaining components in their old
proportions (Time 3/8, Hazard 2/8, Path 1/8, Replanning 2/8). Time uses the
cut-capacity bound `Q_cut` (scoring.md, section 4.2), so unavoidable queues
at exits and doors are not charged and avoidable queues are. A severe
realized bottleneck is a severe Time failure. The congestion diagnostics
are mandatory in every report (`ScoreReport.diagnostics`). A future scoring
version can add a Coordination component only with a privileged lower bound
on a trajectory property that is not a function of Time.

## 6. Sensitivity study

### 6.1 Setup

`python tools/scoring_sensitivity.py` plays 16 policies on three suites:

- the public V1 suite with the public development seeds 0, 1, 2 (30
  episodes),
- a generated stochastic study suite: 16 scenarios on a 16x12 grid with 6 to
  24 people, stochastic fire and a horizon of 80, from a public config and a
  constant test-only generator seed (20261020), with seeds 0, 1, 2 (48
  episodes), and
- an **archived research-holdout** generated suite from the same config and a fresh,
  pre-registered generator seed (20261021), with seeds 0, 1, 2 (48
  episodes). It is used only for validation (section 6.10), never to choose
  a constant or a candidate.

The public and generated suites together are the **study suites**.

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
  unless an exit is close), **fire-hugger** (congestion-aware routing
  that prefers cells next to the fire) and **block-scrub**
  (congestion-aware with every person-blocked move replaced by `STAY`;
  section 5.12).

The privileged bound is a reference, not a policy. All 2016 episodes
(16 policies x 126 episodes; 1248 in the study suites) are scored with
person-level attribution.

### 6.2 Requirements

A candidate is acceptable only if it meets all of these on hand-made
evidence. They were written before the candidates were scored, and they
pin the thresholds normatively: R2 says that Time must matter, R13 and R14
say what a severe failure must cost. R8 and R13 are stated in realized
units: a bottleneck is described by the delay it causes against the
capacity bound, not by blocked actions (section 5.12). R16 was added by the
coordination contract.

| ID | Requirement |
| --- | --- |
| R1 | one avoidable casualty in 100, otherwise perfect, scores at most 60 |
| R2 | Time is not irrelevant: Time 0.5 (not severe) costs at least 10 points |
| R3 | two casualties score below one |
| R4 | 10 casualties in 100 still score at least 5 (unsafe runs stay separable) |
| R5 | 10 casualties in 100 score at most 35 |
| R6 | one stranded in 100 scores at most 85 and above one casualty |
| R7 | 10 stranded in 100 score at most 50 |
| R8 | a realized bottleneck that doubles evacuee person-time over the capacity bound (Time 0.5, block rate 0.3) costs at least 10 points |
| R9 | 50% avoidable losses score 0 (catastrophic gate) |
| R10 | Time 0.1 with everything else perfect scores at most 60 (non-compensation) |
| R11 | a savable casualty is not hidden by another person's unavoidable loss |
| R12 | a run with no casualty and no severe failure beats a perfect run with one casualty (`M*` 2 to 1000, 0 or 1 extra stranded) |
| R13 | a severe realized bottleneck (Time 0.4 against the capacity bound, block rate 0.3) without loss scores below 80 and loses at least 20 points |
| R14 | severe Time 0.4, Hazard 0.3, Replanning 0.4 and Path 0.4 alone each score below 80 |
| R15 | failed, missing and timed-out episodes score 0 and stay in every mean (structural; tested in `tests/test_scoring.py`) |
| R16 | trajectory invariance: identical authenticated state trajectories score identically; the block-scrub adversary gains exactly 0 in every episode |

Results: the official form (F, G5) passes all 16. The other designs fail as
listed in sections 5.9 and 5.12.

### 6.3 Pre-registered orderings

These orderings were fixed before the study. Left must score above right.
The official form passes all ten:

| Ordering | Official scores |
| --- | --- |
| zero-loss with Time 0.6 > perfect quality with 1 casualty | 82.8 > 56.5 |
| perfect > zero-loss but slow (Time 0.5) | 100 > 77.5 |
| perfect > zero-loss with a severe realized bottleneck (Time 0.4) | 100 > 45.7 |
| perfect > zero-loss with high exposure (0.3) | 100 > 68.4 |
| perfect > zero-loss with inefficient routing (Path 0.6) | 100 > 93.9 |
| Replanning 0.9 > static-good with invalidation failure (Replanning 0.3) | 98.1 > 38.5 |
| 1 stranded > 1 casualty | 76.8 > 56.5 |
| 1 casualty > 2 casualties | 56.5 > 39.4 |
| mild realized congestion (Time 0.9) > severe realized bottleneck (Time 0.4) | 96.2 > 45.7 |
| mild delay (Time 0.8) > extreme delay (Time 0.1) | 92.1 > 6.8 |

The two congestion orderings were stated as block rates (0.3; 0.05 against
0.4) before the coordination contract. They are restated in realized delay,
with the same block rates attached; the block rates no longer change the
score.

### 6.4 Adversaries

Score of each adversary minus congestion-aware (study suites). This is a
gaming check, not a selection criterion:

| Adversary | Official F | A | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| reckless-fast | -4.2 | -4.4 | -4.7 | -5.1 | -5.8 |
| abandon-far | -33.9 | -33.8 | -34.3 | -35.0 | -34.7 |
| safe-slow | -38.0 | -34.5 | -34.0 | -29.4 | -37.2 |
| late-start | -26.8 | -25.0 | -24.2 | -18.5 | -27.8 |
| jostler | -10.4 | -10.5 | -10.8 | -11.2 | -11.3 |
| herd | -14.0 | -14.7 | -14.8 | -13.8 | -15.6 |
| wanderer | -36.0 | -33.8 | -32.9 | -27.2 | -35.1 |
| hider | -24.1 | -24.7 | -24.9 | -24.4 | -25.6 |
| fire-hugger | -1.3 | -1.4 | -1.4 | -1.5 | -1.9 |
| block-scrub | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Every adversary except block-scrub scores below the best baseline under
every multiplicative design. Block-scrub scores exactly the same as
congestion-aware, as R16 requires: it is the same evacuation. Reckless-fast
and fire-hugger are close to congestion-aware because they reuse its
routing; their extra losses and exposure cost 1 to 4 points. The severe
gates make safe-slow lose more than under A, because its Time is severe in
about half of its episodes.

### 6.5 Official ranking

| Rank | Policy | Study suites | Public only | Generated only | Archived |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | congestion-aware | 48.6 | 53.2 | 33.1 | 40.5 |
| 1 | adv-block-scrub | 48.6 | 53.2 | 33.1 | 40.5 |
| 3 | adv-fire-hugger | 47.3 | 52.0 | 31.6 | 38.0 |
| 4 | adv-reckless-fast | 44.4 | 48.2 | 31.9 | 31.6 |
| 5 | adv-jostler | 38.2 | 40.7 | 29.8 | 37.1 |
| 6 | fire-aware | 37.9 | 40.4 | 29.8 | 37.1 |
| 7 | static-shortest | 35.5 | 37.4 | 28.9 | 33.2 |
| 8 | adv-herd | 34.6 | 37.0 | 26.6 | 27.6 |
| 9 | nearest-greedy | 25.8 | 27.5 | 20.2 | 9.3 |
| 10 | adv-hider | 24.4 | 25.7 | 20.2 | 5.5 |
| 11 | adv-late-start | 21.8 | 28.1 | 0.9 | 2.5 |
| 12 | adv-abandon-far | 14.7 | 12.0 | 23.6 | 18.3 |
| 13 | adv-wanderer | 12.6 | 13.1 | 10.9 | 11.0 |
| 14 | adv-safe-slow | 10.6 | 10.6 | 10.7 | 3.6 |
| 15 | always-stay | 0.0 | 0.0 | 0.0 | 0.0 |
| 15 | random-legal | 0.0 | 0.0 | 0.0 | 0.0 |

On every suite, including the archived suite, the baselines are in the
expected order: congestion-aware > fire-aware > static > greedy > random =
stay. Block-scrub ties congestion-aware exactly (R16).

**Generated suite.** Under the 1.0 score at `ce5e322`, the generated suite
alone ranked reckless-fast (31.4) and static-shortest (28.1) above
congestion-aware (26.0), through the severe coordination gate (block rate
above 1/4 in 7 of 48 episodes). Section 5.12 shows that this gate charged
an action encoding and, on that suite, mostly the blocks of people who
later died. With Coordination removed, congestion-aware is first there
(33.1), ahead of reckless-fast (31.9), fire-hugger (31.6), fire-aware
(29.8) and static (28.9). The constants were not chosen to produce this
order; the decision of section 5.12 rests on R16 and the scrub proof.

**Per family.** The family table is in
[scoring-study-results.md](scoring-study-results.md). On
`route_invalidation_replanning`, static-shortest scores 32.9, and fire-aware
and congestion-aware score 44.2: the static policy fails to replan. Over the
study suites, Replanning is severe in 19.2% of the static policy's episodes,
against 9.0% for congestion-aware. Late-start is best on that family
(68.0), because waiting lets the fire reveal the blocked route. It pays for
that elsewhere: Time is severe in 51.3% of its episodes. On the archived
`split_flow_across_exits` family, fire-aware (46.3) and jostler (46.2) are
above congestion-aware (45.2); on archived `single_exit_low_density`,
fire-hugger (72.6) is above congestion-aware (68.1).

**Severe failures.** Share of applicable episodes with at least one severe
gate (study suites / archived): congestion-aware 15.4% / 12.5% (hazard
6.4% / 2.1%, replanning 9.0% / 10.4%, Time 0 / 0), fire-aware 15.4% /
16.7%, static 25.6% / 12.5%, greedy 26.9% / 12.5%. Safe-slow and
late-start have severe Time in 51.3% of the study episodes. No moving
baseline has a severe Time failure, so mild, unavoidable congestion never
triggers a gate.

### 6.6 Redundancy and spread

Spearman rank correlation over 238 episodes of moving policies where every
component, including Replanning, is defined (study suites):

|  | Safety | Time | Hazard | Path | Replanning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Safety | 1.00 | 0.09 | -0.13 | -0.33 | 0.51 |
| Time | | 1.00 | 0.03 | 0.54 | -0.13 |
| Hazard | | | 1.00 | 0.23 | -0.52 |
| Path | | | | 1.00 | -0.55 |
| Replanning | | | | | 1.00 |

No pair is above 0.55 in absolute value. Path and Time are the most related
positive pair (0.54); Path keeps the smallest weight (1/8). Replanning
overlaps with Safety (0.51) on purpose, because a failure to replan both
delays and kills, and the two parts measure different things (the delay of
survivors and the losses). The correlations of the rejected Coordination
measures with Time are in section 5.12.

**Spread.** Over the moving policies, the official scores range from 10.6
to 48.6 (range 38.0, standard deviation 13.5). The other multiplicative
designs have a similar range (34.3 to 37.2), so the gates do not compress
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
- Every combination: tau min 0.983, mean 0.996, largest score change 13.2
  points. Feasible combinations: tau min 0.983, mean 0.996, largest change
  8.4 points.

**Gate grid** (81 combinations): `phi` in {0.7, 0.8, 0.9}, `gamma` in
{0.5, 1, 2}, severe Time in {1/3, 1/2, 2/3}, severe hazard rate in {1/5,
1/4, 1/3}. The quality exponent `beta` is re-derived for each combination,
and R12 is checked at that combination's own threshold corner. (The 1.0
grid also varied the severe block rate; that threshold no longer exists.)

- 36 of 81 meet every requirement; the official gates are feasible.
- Failures: R13 in 27 (all with severe Time 1/3: Time 0.4 is then not
  severe), R14 in 45 (severe Time 1/3 or hazard rate 1/3).
- Every combination: tau min 0.898, mean 0.971, largest score change 6.9
  points. Feasible combinations: tau min 0.983, mean 0.999, largest change
  6.8 points.
- The official thresholds (Time 1/2, hazard rate 1/4) are the most lenient
  grid values that still meet R13 and R14. Stricter values also pass, but
  mark more episodes as severe. So the thresholds are pinned by the
  requirements, not by the ranking. `phi` and `gamma` are not pinned: all
  three values of each pass.

The severe Path and Replanning thresholds (1/2) are not in the grid. They
follow the same rule as Time: twice the privileged need is a failure.

### 6.8 Weight perturbation

Each quality weight multiplied by 0.5 and by 1.5, then renormalized:
Kendall tau is 1.000 in every case and no policy moves by more than 1.3
points. The ranking does not depend on the exact weights.

### 6.9 Agreement between designs

Kendall tau of each design's ranking against the official one: A 0.983,
B 0.983, C 0.949, D 0.915, E 1.000, F with the stranding floor 0.966,
additive 0.831, plain product 0.898. The multiplicative designs rank the
policies almost alike. They differ in the requirements, which is why the
requirements, not the ranking, select the design.

### 6.10 Archived research-holdout validation

The archived research holdout (publicly disclosed generator seed 20261021, 48 episodes per policy) was
generated after every decision and checked once.

- Baseline order: congestion-aware 40.5 > fire-aware 37.1 > static 33.2 >
  greedy 9.3 > random 0.0 = stay 0.0, the same order as on the study suites.
- Kendall tau of the 16-policy ranking, study suites against archived:
  0.814. The changes are among adversaries and greedy: greedy falls to
  9.3 (it stalls in the archived high-density layouts), reckless-fast falls
  from 4th to 7th, and abandon-far rises from 12th to 9th.
- Block-scrub: 48 of 48 identical state trajectories, 1211 blocks against
  0, score delta exactly 0 (R16). Under the 1.0 score the scrub would gain
  +3.1 points there.
- Cut-capacity bound: `Q_cut <= person-time` in every one of the 1773
  episodes with evacuees (all suites, all policies); `Q_cut > Q_exit` in 104
  generated and 60 archived episodes, never on the public suite.
- Severe gates (any): congestion-aware 12.5%, fire-aware 16.7%, static
  12.5%; Time is never severe for a moving baseline.

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
| `nearest-exit-greedy-v1` | 27.5 | 27.5 | 27.5 | [24.2, 33.9] | 0.0 | 0.0 |
| `static-shortest-path-v1` | 37.4 | 37.4 | 37.4 | [34.1, 43.8] | 0.0 | 0.0 |
| `fire-aware-shortest-path-v1` | 40.4 | 40.4 | 40.4 | [37.7, 45.6] | 1.0 | 1.0 |
| `congestion-aware-v1` | 53.2 | 53.2 | 53.2 | [50.6, 58.5] | 7.4 | 7.4 |

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
  Claims about unseen environments need archived generated suites with
  several scenarios per family.

Paired comparisons of adjacent baselines on the public suite (seeds 0, 1, 2):

| A vs B | Delta | 95% CI | Outcome | Episodes A higher / B higher / equal |
| --- | ---: | --- | --- | --- |
| static vs greedy | 9.90 | [9.90, 9.90] | `a_better` | 3 / 0 / 27 |
| fire-aware vs static | 2.94 | [1.81, 3.61] | `a_better` | 11 / 0 / 19 |
| congestion-aware vs fire-aware | 12.87 | [12.87, 12.87] | `a_better` | 15 / 3 / 12 |

The zero-width intervals come from deterministic policies on scenarios whose
fire is deterministic: only `v1-pub-10-route-invalidation` varies by seed,
and there two of the pairs score the same.

## 9. Limitations

- The references are relaxations. Path and the release steps of the Time
  bound ignore other people. The Time bound `Q_cut` adds exit capacity and
  the capacity of each wall-distance cut, one cut at a time; it ignores
  interactions between successive cuts and the flow inside rooms. So Time
  and Path can stay below 1 even for an optimal crowd policy in a dense
  scenario. This makes the scale conservative, not biased between models.
  `Q_cut` is proved to be a lower bound (scoring.md, section 4.2) and is
  never above the realized person-time in any study or archived episode.
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
- Congestion is scored only through its realized effects (section 5.12).
  A policy that causes many blocked moves but no extra delay, exposure or
  loss is not charged, by design (R16). Coordination quality beyond those
  effects is not measured by V1; a future component needs a privileged
  lower bound on a trajectory property that is not a function of Time.
- The severe thresholds are normative choices pinned by R13 and R14. The
  grids show which values meet them (section 6.7), but other reasonable
  requirements would give other thresholds.

## 10. Reproduce

```bash
python tools/scoring_sensitivity.py --cache .scoring-study
```

The cache directory holds the generated study and archived suites, the
per-episode evidence and the block-scrub trajectory checks; it is ignored by
git. Delete it after any change to the evidence
definition.
