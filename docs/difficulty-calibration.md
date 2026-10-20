# Difficulty bands and calibration (research, V1)

| Field | Value |
| --- | --- |
| Status | **Research design. Not merged.** Uses the official EMBER Score (scoring version 1.0). |
| Code | `CALIBRATION_PANEL`, `calibrate`, `difficulty_band`, `capacity_savable`, `capacity_ceiling`, `family_tags`, `select_suite` in [`tools/analyze_suite_design.py`](../tools/analyze_suite_design.py) |
| Evidence | [suite-design-results.md](suite-design-results.md) (development pool: 1,120 generated scenarios, 2,320 unique worlds) |
| Score used | The official EMBER Score from `emberbench.scoring` (`score_episode`, `score_evidence`). The tool has no other score. |

This document defines how a scenario gets an admission decision and a
difficulty band, and how the evaluator uses the bands to compose the unseen
suite. The parent design is [evaluation-suite-design.md](evaluation-suite-design.md).

## 1. Requirements

1. A band is computed from measurements. Nobody labels a scenario by hand.
2. The computation is deterministic and reproducible from the scenario, its
   unique worlds and the frozen calibration panel.
3. No model under evaluation is an input. The rule is not fitted to any model.
4. The rule is calibrated on **development** worlds (public development seeds)
   only. Hidden worlds are only measured, evaluator-side, with the rule frozen
   beforehand.
5. The composed suite separates methods. It must not be mostly worlds where
   every reference policy scores about 100, or about 0.

## 2. The frozen calibration panel (version 1.0)

| Role | Policy | Use |
| --- | --- | --- |
| Panel member | `nearest-exit-greedy-v1` | band input |
| Panel member | `static-shortest-path-v1` | band input |
| Panel member | `fire-aware-shortest-path-v1` | band input |
| Panel member | `congestion-aware-v1` | band input |
| Floor reference | `random-legal-v1` | reported only (the analogue of Procgen's `R_min`); it scores 0 on every public world |
| Privileged reference | `M*` (official `max_evacuated`), `M_cap` (section 6), official time and path bounds | a **bound only**. It normalizes the official score and excludes capacity artifacts. It never sets a band. |

The panel is the four public, non-random V1 baselines, in the order of
increasing information use. Its members, its version (`CALIBRATION_PANEL_VERSION
= "1.0"`) and the band thresholds are frozen and recorded in the freeze record
of every suite. A change to any of them is a new panel version and needs new
suites.

**No circular calibration.**

- The panel is fixed before any model is evaluated. A model under evaluation
  is never a panel member and never an input to admission, banding or
  selection (tests check the signatures).
- The thresholds are fixed numbers with a stated meaning on the official
  score scale (section 4). They are not quantiles of a pool and not fitted to
  make any policy look good.
- The panel members define the bands, so their own scores on a selected suite
  are biased by the selection. The suite-design study therefore measures rank
  stability also with two comparators that are **not** panel members: a
  congestion-aware policy with an ε share of uniformly random legal actions,
  ε = 0.05 and ε = 0.2.
- `fire-aware-shortest-path-v1` and `congestion-aware-v1` are practically tied
  on the pool (mean 61.1 and 61.6). The design study does not count their
  order as evidence.

## 3. Evidence: structural proxies do not predict difficulty

Rank correlation (Spearman) between each proxy and `100 - B` over the admitted
development scenarios, where `B` is the best panel score (section 4):

| Proxy | rho |
| --- | --- |
| population | 0.35 |
| crowd density | 0.17 |
| N / min cut | 0.20 |
| obstacle density | -0.48 |
| route redundancy | 0.34 |
| ignition-person distance | -0.30 |
| threatened share (privileged) | 0.27 |
| recoverable invalidation (privileged) | 0.57 |
| savable share `M*/N` (privileged) | 0.04 |

- No structural proxy reaches `|rho| = 0.5`. Obstacle density correlates
  **negatively**: mazes slow the fire and are easy for any planner. A
  Safety-Gym-style level ("more obstacles is harder") would be wrong here.
- The best single predictor is privileged: the share of savable people whose
  every static shortest route the fire cuts (0.57). It needs the seed, so it
  is evaluator-side.
- Proxies are therefore used for **family tags** and quality checks, not for
  bands.

## 4. Band definition

For a scenario, take each panel member's mean official EMBER Score over the
scenario's applicable unique worlds (worlds with `M* > 0`). `B` is the
highest of the four means.

| Band | Rule (official points) | Best panel member, development pool (median avoidable losses) |
| --- | --- | --- |
| Easy | `B >= 80` | 0 |
| Moderate | `30 <= B < 80` | 1 |
| Hard | `10 <= B < 30` | 1.75 |
| Stress | `B < 10` | 3 |

**Why these cutpoints.** The official score is steep in avoidable losses.
With perfect quality, one avoidable loss gives at most 52.5 when `M* = 10`,
65.2 when `M* = 20`, 76.8 when `M* = 100`; two avoidable losses out of 10 give
28.8, three give 15.4 (computed with `score_evidence`). So:

- `B >= 80` means that the best fair heuristic has **no** avoidable loss and a
  good quality. On the pool, the mean avoidable loss of the best member in the
  Easy band is 0.01.
- `30 <= B < 80` means about one avoidable loss (or none with a severe quality
  failure).
- `10 <= B < 30` means about two avoidable losses.
- `B < 10` means three or more: the best V1 heuristic fails badly. This band
  is the headroom against saturation.

The earlier proposal (before PR #32) used `worst >= 90` and `best >= 90` on a
safety-only analysis score. On the official score, the old Easy rule
`worst >= 90` mostly selects saturated worlds (every panel member is at 95 or
more on 28.6% of the pool worlds), which carry no ranking information. The rule
is replaced by the `B` rule above and an explicit saturation exclusion
(section 5).

**Why `B`, not the mean or the worst member.** A band states how much of the
attainable score the best fair V1 heuristic keeps: the regret of the strongest
available reference, as in PAIRED (Dennis et al., 2020). The spread of the
panel is used for admission (section 5), not for the band.

## 5. Admission rule (deterministic, in this order)

| Order | Exclusion | Rule | Pool share |
| --- | --- | --- | --- |
| 1 | `not_applicable` | no world has a savable person (`M* = 0`); the official score is NOT_APPLICABLE | 0.9% |
| 2 | `capacity_artifact` | mean capacity ceiling `< 50` (section 6) | 4.1% |
| 3 | `saturated` | every panel member `>= 95` | 24.1% |
| 4 | `floor` | every panel member `< 1` | 2.2% |
| 5 | `no_separation` | panel spread `B - worst <` the official practical difference (1 point) | 12.9% |
| — | `admitted` | every other scenario; banded by `B` | 55.7% |

A saturated or floor scenario carries no ranking information for V1-level
policies. A no-separation scenario cannot rank the panel itself. All three
still count in the pool statistics, so the acceptance rates of section 7 are
honest.

## 6. Practical unsolvability: the capacity filter (`M_cap`)

The official `M*` counts the people who can escape **alone** against the
precomputed fire. It ignores that people share exits and cuts. In a dense
single-exit room, `M*` can be 35 while at most 14 people can physically leave
before the fire closes the exit. Every policy then "loses" 21 savable people,
and the official score is 0 for every policy: the world measures the bound,
not the policy.

`capacity_savable(scenario, seed)` returns `(M*, M_cap)`. `M_cap <= M*` adds
the two capacity facts behind the official time bound `Q_cut`:

- at most one person leaves through an open exit cell per step;
- at most `k_s` people stand on the cut `N(C)` of a component `C` of the cells
  with wall distance `> d` at step `s` (the same cut groups as the official
  `_cut_bound`).

Each savable person gets an interval of feasible steps per exit and per cut.
The maximum number of people who fit is an interval scheduling problem, solved
exactly by Glover's earliest-deadline greedy. `M_cap` is still an upper bound
(the relaxation ignores movement conflicts), so it is safe to use as a filter.

`capacity_ceiling(N, M*, M_cap)` is the best official score that a perfect
policy can reach when `M* - M_cap` savable people are capacity-limited:
`score_evidence` on synthetic evidence with `M* - M_cap` avoidable stranded
people and perfect quality. It uses the official formula; the tool has no
second implementation.

**Rule:** a scenario whose mean capacity ceiling is below `CEILING_MIN = 50`
is excluded as a capacity artifact. The official score is not changed:
`M*` stays the official reference. The filter only stops such scenarios from
entering a suite.

| Measurement (development pool) | Value |
| --- | --- |
| applicable worlds with `M_cap < M*` | 92 of 2,286 (4.0%) |
| scenarios with such a world | 75 |
| scenarios excluded (ceiling `< 50`) | 46 (41 of them would otherwise be Stress) |
| panel runs that evacuated more than `M_cap` | 0 (2,320 worlds x 4 members): the bound holds |
| capacity-limited worlds where the best member reaches `M_cap` exactly | 50: the bound is tight there |
| public V1 suite | `M_cap = M*` on all 73 unique worlds |

The artifacts come mostly from the `dense-single` recipe (34 of its 120
scenarios). Without the filter they would enter the Stress band, where every
policy scores about 0 for a reason that no policy controls.

## 7. Calibration results on the development pool

Mean official score of each comparator over the admitted units (scenarios with
a final primary family) per band:

| Band | Units | nearest-exit-greedy | static-shortest-path | fire-aware-shortest-path | congestion-aware | noisy ε=0.05 | noisy ε=0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Easy | 264 | 14.2 | 76.6 | 93.0 | 94.0 | 85.7 | 49.9 |
| Moderate | 69 | 10.7 | 23.3 | 40.9 | 42.4 | 35.8 | 13.0 |
| Hard | 79 | 4.1 | 6.4 | 14.4 | 17.4 | 17.2 | 9.0 |
| Stress | 105 | 0.9 | 1.3 | 3.9 | 4.5 | 5.2 | 2.4 |

- The bands are ordered for every comparator, including the two non-panel
  comparators. The calibration is not circular in effect either.
- The Easy band still separates policies strongly (nearest-exit-greedy 14.2
  against congestion-aware 94.0): "Easy" means easy for an informed planner,
  not for every policy.
- The official score is bimodal on single scenarios (histogram of `B`: 224 of
  1,110 applicable scenarios below 10, 595 at 90 or more). The Moderate band is
  the scarcest (7.9% of candidates).

Public V1 suite, same rule (one world per deterministic scenario, 64 worlds
for the stochastic one): pub-01 and pub-04 saturated; pub-03 and pub-08 Easy;
pub-02 and pub-10 Moderate; pub-06, pub-07 and pub-09 Hard; pub-05 Stress.

## 8. Composition rule

1. Every candidate gets its `calibrate` result and its family tags
   (`family_tags`, fixed thresholds on the structural profile and on the
   evaluator-side fire profile).
2. `select_suite(candidates, quotas, selection_key)` fills the
   family x band quotas. Families are filled scarcest first; each candidate is
   used once and its selected family becomes its primary family; inside a
   cell the order is a keyed SHA-256 of the scenario content hash, so the
   private key, not the candidate order or a model, decides which candidate is
   taken.
3. A cell that cannot be filled is reported as a **shortfall**. It is never
   filled from another band in silence. The evaluator generates more candidates
   for that cell (targeted recipe), or the freeze record states the shortfall.
4. The fire-model quota (stochastic slots per family) is applied in the same
   way, as part of the cell key.

## 9. Known limits

- Bands are relative to V1 heuristics. A strong model can find Stress worlds
  solvable. That is the intended headroom; saturation triggers a new panel
  version (evaluation-suite-design.md, section 16).
- `M_cap` is a relaxation. Some admitted Stress worlds can still be harder than
  any policy can solve. The filter removes the clear capacity artifacts only.
- Family tags use fixed thresholds on measured properties. The generator's own
  `describe_families` names fewer families; the evaluator-side tags are the
  stratification key (see the gap table in evaluation-suite-design.md).
- The noisy comparators share their base policy with a panel member. They test
  rank stability; they are not independent agents.

## 10. Reproducibility and tests

Every band and admission decision is a pure function of the scenario, its
unique worlds (by `environment_world_digest`) and the frozen panel. The tool
caches expensive runs outside the repository, keyed by every version; the
reported numbers are recomputed from the final code. `tests/test_suite_design.py`
checks the band thresholds, the admission order, the capacity bound on a
scenario with `M_cap < M*`, Glover's greedy, the capacity ceiling against the
official score, the official scoring of panel runs and the selector rules.
