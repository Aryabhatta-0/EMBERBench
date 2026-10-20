# Difficulty tiers and calibration (research, V1)

| Field | Value |
| --- | --- |
| Status | **Research proposal. Not merged.** Cutpoints must be re-anchored on the final EMBER Score (PR #32). |
| Code | `difficulty_band`, `family_tags`, `select_suite` in [`tools/analyze_suite_design.py`](../tools/analyze_suite_design.py) |
| Evidence | [suite-design-results.md](suite-design-results.md) (development pool: 400 generated scenarios, 880 unique worlds) |
| Score used | The **provisional** safety-only analysis score. It is not the EMBER Score. |

This document defines how a world or scenario gets a difficulty band, and
how the evaluator uses bands to compose the unseen suite. The parent design is
[evaluation-suite-design.md](evaluation-suite-design.md).

## 1. Requirements

1. A band is computed from measurements. Nobody labels a scenario by hand.
2. The computation is deterministic and reproducible from the scenario, the
   seed and the public reference panel.
3. The rule does not depend on any model under evaluation.
4. The rule is calibrated on **development** worlds only. Hidden worlds are
   only measured, evaluator-side, with the rule frozen beforehand.
5. The composed suite separates methods: it must not be mostly worlds where
   every reference agent scores about 100, or about 0.

## 2. Candidate inputs

Three kinds of measurement are available in V1.

| Kind | Examples | Needs the seed? | Cost |
| --- | --- | --- | --- |
| Structural proxies | population, crowd density, `N / min cut` (drain time through the narrowest cut), obstacle density, route redundancy, passage width, ignition distance, exit-distance gap, deceptive-exit share | no | milliseconds |
| Privileged fire-dynamics proxies | escapable share `M*/N`, threatened-exit share, share of people whose every static shortest route is cut (recoverable invalidation), forced-delay share, exit throughput cap | yes (evaluator only) | milliseconds |
| Reference-panel performance | provisional (later: final) score of `nearest-exit-greedy-v1`, `static-shortest-path-v1`, `fire-aware-shortest-path-v1`, `congestion-aware-v1` relative to the privileged reference | yes (evaluator only) | about 0.03 s per episode (median) |

`random-legal-v1` scores about 0 on almost every world. It is the floor, the
analogue of Procgen's `R_min`, and it does not set bands.

## 3. Evidence: proxies alone do not predict difficulty

On the development pool, the rank correlation (Spearman, per world) between
each proxy and the **oracle gap of the best fair baseline**
(`100 - max panel score`) is:

| Proxy | rho |
| --- | --- |
| population | 0.23 |
| crowd density | 0.24 |
| N / min cut | 0.14 |
| obstacle density | -0.08 |
| route redundancy | 0.07 |
| ignition-person distance | -0.18 |
| escapable share (privileged) | -0.03 |
| threatened share (privileged) | 0.24 |
| recoverable invalidation (privileged) | 0.63 |
| forced delay (privileged) | 0.61 |

- Every **structural** proxy has `|rho| <= 0.24`. A band from structural
  proxies would be mostly noise. Safety Gym style levels ("more hazards is
  harder") do not transfer to EMBER: hazard timing, not count, decides.
- The two **privileged fire-dynamics** proxies reach about 0.6. They measure
  what actually happens to the static plans. They need the seed, so they are
  evaluator-side.
- No proxy measures crowd interaction well. The privileged bound `M*` is a
  per-person relaxation, so the escapable share is uncorrelated with the gap.

## 4. Options considered

| Option | For | Against | Decision |
| --- | --- | --- | --- |
| Structural proxies only | Seed-free; cheap; public | Weak evidence (section 3) | Use for **family tags** and diagnostics, not for bands |
| Pool quantiles of baseline scores | Equal band sizes by construction | Bands drift with the generator config and the pool; a "hard" world in one pool is "easy" in another; quantiles hide saturation | Rejected as the definition; reported as a diagnostic |
| Oracle gap of the reference panel, with fixed cutpoints | Direct measurement of difficulty relative to the privileged reference (regret, as in PAIRED); stable across pools; interpretable | Needs evaluator-side runs; selection on the panel's outcome biases the panel's own scores on the selected suite | **Adopted** |
| Combination | Best of both | More moving parts | Adopted in this form: bands from the panel gap; tags from proxies; proxies as QA |

## 5. Band definition

For one world, let `best` and `worst` be the highest and lowest score of the
four panel members. For a scenario with several worlds (stochastic fire), each
member's score is first averaged over the scenario's unique worlds.

| Band | Rule | Meaning |
| --- | --- | --- |
| Easy | `worst >= 90` | Every competent heuristic solves it. A control item: a failure means a basic fault. |
| Moderate | `best >= 90 > worst` | Some heuristic solves it and some does not. It separates simple from informed policies. |
| Hard | `30 <= best < 90` | No heuristic solves it. The best keeps part of the attainable score. |
| Stress | `best < 30` | Even the best heuristic loses most of the attainable score. Headroom against saturation. |

The cutpoints `SOLVED = 90` and `STRESS = 30` are anchored to the score
scale, not fitted to a pool. Under the provisional safety score, 90 means "no
avoidable loss and nearly full evacuation" (one avoidable casualty out of 10
already gives about 39), and 30 means "the best heuristic loses about two
people out of ten or more".

**After PR #32 merges,** recompute the bands with the final EMBER Score and
re-anchor the two cutpoints to the same meaning on that scale. Record the
cutpoints in the freeze record of every suite.

## 6. Calibration results on the development pool

The pool is 9 public development recipes of the existing generator
(open-sparse, dense-single, partitioned, multi-exit, multi-ignition, maze,
crowd, stochastic-open, stochastic-partitioned), 400 scenarios, 880 unique
worlds, generated from public development seeds. One scenario had no
escapable person in any world and is excluded.

Mean provisional score by band (scenario level):

| Band | Scenarios | random | nearest-exit | static | fire-aware | congestion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 107 | 1.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| Moderate | 117 | 0.0 | 18.0 | 88.4 | 94.4 | 96.1 |
| Hard | 61 | 0.0 | 28.3 | 38.5 | 51.1 | 53.3 |
| Stress | 114 | 0.0 | 5.5 | 7.5 | 10.1 | 10.6 |

- The bands order every panel member monotonically. The panel order
  (random < nearest-exit < static < fire-aware <= congestion) holds inside
  every band.
- **Without calibration the pool does not separate methods.** In 286 of 869
  worlds every panel member scores at least 90. In 244 of 869 every member
  scores below 30. So 61% of raw generated worlds carry almost no ranking
  information among the fair baselines. The median panel spread
  (`max - min`) per world is 0.
- Recipes do not determine bands. For example `maze` gives 12 easy, 32
  moderate, 0 hard and 4 stress scenarios; `dense-single` gives 9, 13, 2 and
  24. A recipe is a way to make a family likely, not a difficulty label.
- Families have different natural band mixes. `route_invalidation_replanning`
  has 0 easy and 0 moderate scenarios among 30 tagged; `maze_obstacle_routing`
  has 1 hard among 54. A uniform family x band grid is not achievable; see
  section 7.

Measured properties by band (medians of structural proxies, means of
privileged proxies over worlds):

| Band | N | crowd density | N / cut | recoverable invalidation | forced delay | threatened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 8 | 0.07 | 5.0 | 0.00 | 0.00 | 0.06 |
| Moderate | 9 | 0.08 | 4.7 | 0.01 | 0.01 | 0.03 |
| Hard | 13 | 0.08 | 6.5 | 0.05 | 0.06 | 0.11 |
| Stress | 13 | 0.09 | 7.0 | 0.11 | 0.11 | 0.22 |

The privileged fire-dynamics proxies and the crowd size rise with the band.
This is the QA check: if a future generator or simulator revision breaks this
monotone pattern, the band rule must be reviewed before it is used.

## 7. Calibration rule for composing the suite

1. **Freeze the rule** (panel IDs and versions, cutpoints, tag thresholds,
   scoring version) and the generator recipes on development worlds.
2. **Measure** every hidden candidate evaluator-side: tags, unique worlds, panel
   scores, band. Store the result only in the private freeze record.
3. **Target mix per family**: Easy 2, Moderate 4, Hard 8, Stress 6 out of 20
   (10% / 20% / 40% / 30%). Easy is a small control share. Moderate and Hard
   carry most of the discrimination between current methods. Stress keeps
   headroom so that the suite does not saturate.
4. **Select** with `select_suite`: families from scarcest to most common, cells
   filled in keyed-hash order. A cell that cannot be filled is reported, never
   silently filled.
5. **Shortfall** (decided before any model runs, and recorded): first draw
   more candidates for that family from the same recipe and generator seed
   stream; if the cell is still empty, move its quota to the nearest band in
   the order Hard, Moderate, Stress, Easy. The realized mix per family is
   published with the round.
6. **Discrimination floor**: across the whole suite, at most 15% Easy and at
   most 30% Stress worlds, and at least half the worlds with a panel spread of
   at least 20 points or a Hard band.

## 8. Known limits

- **Selection on the panel's outcome.** The panel helps choose the worlds, so
  the panel's own scores on the selected suite are not unbiased estimates of
  its performance on the generator distribution. They are reference points.
  New models are unaffected, because the rule does not depend on them; but if
  a model is a near copy of a panel member, say so when it is reported.
- **Loose privileged bound.** `M*` relaxes congestion. Some Stress worlds may
  be Stress because nobody can reach `M*`, not because the task is hard. The
  tool reports a simple exit-throughput cap (one person per exit per step
  until the exit burns). A world whose `M*` exceeds that cap has a provably
  loose bound and should not be selected as Stress. On the development pool
  this applies to 11% of Stress worlds (mean share) and to no Easy, Moderate
  or Hard world, and to no public world. A tighter joint bound is future work
  and needs no simulator change.
- **Panel coverage.** All four panel members are heuristics without
  look-ahead. A world that is Stress for them may be Moderate for a planner.
  When a stronger public reference exists, add it to the panel under a new
  rule version. Never change the rule inside a round.
- **Provisional score.** The provisional score ignores the quality components
  and is harsher on single casualties than a quality-weighted score would be
  near the top. Bands must be recomputed on the final score.

## 9. Reproducibility and tests

`tests/test_suite_design.py` checks that:

- the band rule gives the documented band for fixed panel scores;
- the band of public scenarios is reproducible (`v1-pub-01` is Easy,
  `v1-pub-08` is Moderate);
- selection is deterministic, independent of candidate order, uses each
  scenario once, and changes only with the private selection key;
- shortfall is reported, not filled from another band.
