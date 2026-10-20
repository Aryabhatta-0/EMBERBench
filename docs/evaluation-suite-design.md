# Unseen evaluation suite: scientific design (research, V1)

| Field | Value |
| --- | --- |
| Status | **Research design. Not merged.** Final for V1 on the official EMBER Score. |
| Scope | Design of the official V1 unseen (hidden) evaluation suite. No simulator, observation, scenario-format, generator or scoring change. |
| Score | The official EMBER Score (`emberbench.scoring`, scoring version 1.0). Every number in this document comes from it. |
| Companion documents | [difficulty-calibration.md](difficulty-calibration.md), [suite-design-results.md](suite-design-results.md) |
| Library support | `environment_world_digest`, `build_unique_world_manifest` (`emberbench.worlds`) |
| Analysis tool | [`tools/analyze_suite_design.py`](../tools/analyze_suite_design.py) (`audit-public`, `study`, `gap-probe`) |
| Tests | [`tests/test_suite_design.py`](../tests/test_suite_design.py) |

This document is named "evaluation suite", not "held-out suite", on purpose:
the leak guard (`emberbench.leakguard`) rejects any repository path with a
`heldout-` component, so that private material cannot be committed by
accident. The document holds no private scenario, seed or result. Every
number comes from public development seeds.

## 0. Summary

- **One design:** 10 primary families x 16 scenarios = **160 scenarios,
  240 unique worlds**. Per family: Easy 2, Moderate 4, Hard 6, Stress 4;
  12 deterministic-fire scenarios (1 world each) and 4 stochastic-fire
  scenarios (3 distinct fire worlds each).
- **World identity:** `environment_world_digest`, a seed-free digest of the
  scenario and its burning trajectory. The public suite is 640 nominal episodes
  but only **73 unique worlds** (88.6% duplicates).
- **Difficulty:** bands from the best official score `B` of a frozen
  four-baseline calibration panel: Easy `B >= 80`, Moderate `[30, 80)`, Hard
  `[10, 30)`, Stress `< 10`. A capacity filter (`M_cap`, official
  `score_evidence`) excludes worlds that are hard only because `M*` ignores
  exit and cut capacity.
- **Model trials:** K = 1 for a verified-deterministic model, K = 3 for every
  other model, on the same frozen worlds: 240 or 720 episodes.
- **Precision:** within-suite 95% CI about 7.9 points wide (about +-3.9);
  paired 95% width between adjacent comparators at most 5.8 points; comparator
  pairs more than 1.12 points apart keep their order in at least 95% of
  simulated suites.
- **Candidate pool:** about 4,000 candidates (25x the suite) with the current
  recipes plus the probe recipes, then targeted top-up batches for any
  shortfall cell. The earlier "10x" figure is **not** supported: 1,120
  candidates (7x) filled 87.6% of the quotas and never all of them.
- **Generator gap:** the generator names 3 primary families. Nine of the ten
  primary families need some generator work (section 6): new proxies or
  constraints for three, fire-dependent evaluator-side tags for two, labels or
  recipes for four. Until then the evaluator assigns families by measured tags.

## 1. Literature review

| Source | Finding | Implication for EMBERBench |
| --- | --- | --- |
| Cobbe, Hesse, Hilton, Schulman. *Leveraging Procedural Generation to Benchmark RL* (Procgen), ICML 2020, arXiv:1912.01588 | 16 procedurally generated environments. Train on a finite level set, test on the full distribution; agents need many thousands of levels to close the generalization gap. Scores are normalized per environment by `R_min` and `R_max`. | A fixed public suite of 10 scenarios is memorizable. Scored evaluation must draw from a procedural distribution that no model saw. Per-world normalization by a privileged reference (EMBER's `M*`) is the analogue of `R_max`; `random-legal-v1` is the analogue of `R_min`. |
| Cobbe, Klimov, Hesse, Kim, Schulman. *Quantifying Generalization in RL* (CoinRun), ICML 2019, arXiv:1812.02341 | Agents overfit to surprisingly large training-level sets. | The number of distinct worlds, not the number of episodes, is the unit of evidence. |
| Machado et al. *Revisiting the Arcade Learning Environment*, JAIR 2018, arXiv:1709.06009 | A deterministic environment lets an agent memorize an open-loop action sequence; sticky actions break it. Separate tuning games from reporting games. | Deterministic worlds are memorizable when public. Repeating a deterministic world with a deterministic policy adds no evidence. Calibration and reporting use different worlds. |
| Agarwal et al. *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021, arXiv:2108.13264 | With few runs, point estimates mislead. Use stratified intervals, robust aggregates and the probability of improvement. | Stratify by family. A stratum with one unit has no variance estimate. Every family needs several independent scenarios. |
| Henderson et al. *Deep RL that Matters*, AAAI 2018, arXiv:1709.06560 | Two groups of 5 seeds of one algorithm can differ significantly. | Seeds are a variance source to control and freeze, not a free sample-size multiplier. |
| Colas, Sigaud, Oudeyer. *How Many Random Seeds?*, 2018, arXiv:1806.08295 | Power analysis for RL comparisons; small-sample tests are anti-conservative. | Size the suite from a precision target, with margin, not from cost. |
| Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021, arXiv:2103.03098 | Data sampling is the largest variance source. | In EMBER the analogue is **which worlds are drawn**. The study measures suite-sampling variance directly (section 11). |
| Miller. *Adding Error Bars to Evals*, 2024, arXiv:2411.00640 | Clustered standard errors when items are grouped; K resamples per item cut within-item variance to sigma^2/K; paired differences. | Worlds of one scenario are a cluster. K model trials on the same frozen worlds. Paired comparison on one pack. |
| Ray, Achiam, Amodei. *Safety Gym*, 2019; Ji et al. *Safety-Gymnasium*, 2023, arXiv:2310.12567 | Difficulty levels by hazard count. | EMBER needs evidence that its proxies track difficulty. They do not (difficulty-calibration.md, section 3): obstacle density correlates negatively. |
| Mattson et al. *MLPerf Training*, MLSys 2020, arXiv:1910.01500; Reddi et al. *MLPerf Inference*, ISCA 2020, arXiv:1911.02549 | Run counts from measured variance; frozen rules per round; stated statistical confidence. | Choose K and worlds per stochastic scenario from measured variance. Freeze rules and suite per round. |
| Liang et al. *HELM*, TMLR 2023, arXiv:2211.09110 | Every model on the same instances; repeated trials for model randomness. | One frozen pack for all models; model randomness by trials, not by new worlds. |
| Recht et al. *Do ImageNet Classifiers Generalize to ImageNet?*, ICML 2019, arXiv:1902.10811 | A new test set from the same process changes absolute accuracy but mostly keeps the ranking. | Scores are comparable only inside one suite revision. |
| Blum, Hardt. *The Ladder*, ICML 2015; Dwork et al. *The Reusable Holdout*, Science 2015 | Repeated adaptive queries overfit a holdout. | Limit scored submissions per suite; no per-world results while a suite is live. |
| Kiela et al. *Dynabench*, NAACL 2021; Ott et al., Nature Communications 2022 | Static benchmarks saturate. | Define saturation triggers in advance (section 16). |
| Jacovi et al. *Stop Uploading Test Data in Plain Text*, EMNLP 2023, arXiv:2305.10160 | Public plain-text test data contaminates later models. | Never publish a live pack. Publish digests only. |
| Zhang et al. *GSM1k*, 2024, arXiv:2405.00332 | Some models drop on a fresh equivalent test set. | A fresh suite from the same frozen rules is the contamination check (section 15). |
| White et al. *LiveBench*, 2024; Jain et al. *LiveCodeBench*, 2024 | Dated, refreshed releases limit contamination. | Each round gets a new suite identity; old suites are retired, not edited. |
| Rodriguez et al., ACL 2021; Polo et al. *tinyBenchmarks*, ICML 2024 | Items differ in difficulty and discrimination. | Prefer worlds that separate methods (admission rule). EMBER scores are bimodal, so a tiny suite is not enough. |
| Dennis et al. *PAIRED*, NeurIPS 2020, arXiv:2012.02096 | Environment difficulty as regret against a strong reference. | Band = score kept by the best fair reference heuristic (difficulty-calibration.md). |
| Kirk et al. *A Survey of Zero-Shot Generalisation in Deep RL*, JAIR 2023 | Train and test contexts disjoint and drawn by a stated process. | Development (public) and evaluation (private) seeds come from disjoint domains. |
| Card et al. *With Little Power Comes Great Responsibility*, EMNLP 2020 | Many test sets are too small for the effects they report. | State the smallest difference the suite resolves (section 11). |

## 2. Audit of the public V1 suite

Full tables: [suite-design-results.md, part A](suite-design-results.md#part-a-public-v1-suite-audit).
Schedule: `build_evaluation_manifest` with 64 public development seeds.

| Quantity | Value |
| --- | --- |
| nominal scheduled episodes | 640 |
| unique scenarios | 10 |
| unique environment worlds | **73** |
| deterministic-fire worlds | 9 (one per deterministic scenario) |
| stochastic-fire worlds | 64 (all from `v1-pub-10`) |
| duplicate episodes | 567 (88.6%) |
| `build_unique_world_manifest`, 3 worlds per stochastic scenario | 12 episodes |
| worlds per primary family | 1 for nine families; 64 for `route_invalidation_replanning` |

Official score of the calibration panel (difficulty-calibration.md):
pub-01 and pub-04 are saturated (every member at 95 or more); pub-03 and pub-08
Easy; pub-02 and pub-10 Moderate; pub-06, pub-07 and pub-09 Hard; pub-05
Stress. `random-legal-v1` scores 0 on every world. `M_cap = M*` on every
world (no capacity artifact).

The public suite is a good development suite and a poor scored suite: one
scenario per family (family intervals not estimable), 73 worlds of which 64
belong to one scenario, two saturated scenarios, small populations (seven of
ten have 9 people or fewer), and every scenario is public.

## 3. World identity and unique-world scheduling

In V1 the episode seed reaches a run only through the simulator's fire stream.
The fire ignores the agents' actions, and the observation contains no seed or
RNG state. A policy's view of an episode is therefore fixed by the scenario
and the burning trajectory:

> **Environment world** = (simulator version, observation version, scenario
> content hash, burning cells at every snapshot `0..H`).

```text
environment_world_digest = sha256(canonical_json({
    environment_world_version: "1.0", simulator_version, observation_version,
    scenario_hash, burning: [sorted [row, col] per snapshot 0..H]}))
```

- The seed is **not** an input. Two seeds with equal trajectories are one
  world; a deterministic-fire scenario is one world for every seed.
- The model is **not** an input. Tests check that every baseline sees exactly
  the precomputed trajectory.
- PR #32's `fire_world_digest` is unchanged. It hashes the seed with the
  trajectory and stays the per-episode pack attribute. It is not used for
  deduplication.
- `build_unique_world_manifest(suite, candidate_seeds, *, stochastic_worlds)`
  schedules each deterministic scenario once (the first candidate seed) and
  each stochastic scenario with the first `stochastic_worlds` candidate seeds
  of pairwise-different worlds. It raises `EvaluationError` when the
  candidates give fewer worlds. The result is a normal, validated
  `EvaluationManifest`; it freezes into an `EnvironmentPack` unchanged. It
  takes no model input. `build_evaluation_manifest` is unchanged.
- Every count in this document is a count of unique worlds. Seed duplicates
  are never counted as evidence.

## 4. Families: audit and final list

Audit criteria (the brief's five): V1-supported (needs only V1 mechanics: grid,
walls, exits, people, deterministic or stochastic fire), independently
meaningful (tests a distinct decision capability), non-redundant (tag overlap,
Jaccard), scoreable (admitted scenarios exist and the panel separates on them),
generatable (the current generator produces it without hand-authoring). No
V2 concept (smoke, psychology, floors, communication) is used.

| Family | Capability tested | Tag rule (measured) | Supported | Meaningful | Non-redundant (max Jaccard) | Scoreable (admitted) | Generatable now | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_exit_low_density` | Basic evacuation planning: route every person to the only exit before the fire cuts the route. The control family. | one exit, crowd density < 0.15 | yes | yes | 0.23 | 89 | yes | **primary** |
| `single_exit_high_density` | Queue management at one exit: order and timing so that the queue drains before the exit burns; conflict-free moves. | one exit, density >= 0.15 | yes | yes | 0.07 | 59 | yes (28% capacity artifacts) | **primary** |
| `narrow_corridor_bottleneck` | Sequencing through a width-1 cut without blocked or conflicting moves. | passage width 1 and `N / cut >= 4` | yes | yes | 0.23 | 200 | yes | **primary** |
| `split_flow_across_exits` | Load balancing: assign people to exits by expected queue time, not by distance. | several exits, split pressure > 1, `N >= 20` | yes | yes | **0.87 with `crowd_100`** in the pool | 39 | partly (only at crowd scale) | **primary**; needs a medium-population recipe |
| `crowd_100` | Scale: one decision per step for 100 or more people; long observations and action lists. | `N >= 100` | yes | yes | 0.87 (see above) | 35 | yes (slow) | **primary** |
| `maze_obstacle_routing` | Path finding with detours in obstacle-rich layouts. | obstacle density >= 0.25 or detour ratio >= 1.5 | yes | yes | 0.24 | 97 | yes, but no Moderate band in the default recipe | **primary** |
| `deceptive_nearest_exit` | Exit choice by true path length, not by straight-line distance. | share of people whose nearest exit by distance is a dead end or longer by route >= 0.25 | yes | yes | 0.24 | 58 | only by rejection sampling | **primary** |
| `threatened_nearest_exit` | Fire anticipation: avoid the nearest exit when the fire reaches it first. | mean threatened share >= 0.2 (evaluator-side, needs the trajectory) | yes | yes | 0.30 | 61 | only by rejection sampling | **primary** |
| `multiple_ignitions` | Reasoning about several fire fronts at once. | 2 or more ignitions | yes | yes | 0.22 | 125 | yes | **primary** |
| `route_invalidation_replanning` | Replanning when the fire cuts the planned route while another route stays open. | mean recoverable invalidation share >= 0.2 (evaluator-side) | yes | yes | 0.30 | 113 | only by rejection sampling | **primary** |
| `central_ignition` | A fire geometry, not a separate decision capability: its failures are the threatened-exit and invalidation failures. | ignition centrality <= 0.34 | yes | weak | 0.22 | 89 | no label | secondary tag |
| `multiple_exits_unequal_distance` | A layout property of almost every multi-exit scenario (49.5% of the pool); exit choice is tested by the deceptive, threatened and split-flow families. | exit-gap median >= 4 | yes | weak | 0.22 | 346 | implicit | secondary tag |

**Final list: 10 primary families.** Twelve is not correct for V1: two values
of `ScenarioFamily` are scenario attributes, not capabilities. They stay valid
`ScenarioFamily` values and secondary tags; no enum changes. `crowd_100` and
`split_flow_across_exits` overlap 0.87 in the development pool only because
the crowd recipe is the only source of split pressure. The capabilities differ
(scale against load balancing), so both stay primary, and the gap table asks
for a medium-population split-flow recipe.

What fails per family (congestion-aware-v1 on admitted tagged worlds, official
score, Safety, Quality): the families separate both in Safety (0.33 for route
invalidation to 0.92 for mazes) and in Quality; severe replanning failures
range from 2.6% (maze) to 29.8% (threatened exit). Table:
[suite-design-results.md, part B, section 6](suite-design-results.md).

Stochastic fire is an orthogonal stratum, not a family.

## 5. Difficulty

The full rule and its evidence are in
[difficulty-calibration.md](difficulty-calibration.md). In short:

- Frozen calibration panel 1.0: `nearest-exit-greedy-v1`,
  `static-shortest-path-v1`, `fire-aware-shortest-path-v1`,
  `congestion-aware-v1`. `random-legal-v1` is the floor reference only. The
  privileged reference is a bound only.
- `B` = best panel mean official score over the scenario's worlds. Easy
  `B >= 80` (best member has no avoidable loss), Moderate `[30, 80)` (about
  one), Hard `[10, 30)` (about two), Stress `< 10` (three or more).
- Admission, in order: not applicable; capacity artifact (mean capacity
  ceiling < 50); saturated (every member >= 95); floor (every member < 1);
  no separation (spread < 1 point, the official practical difference).
- Development pool: 55.7% admitted. Structural proxies do not predict
  difficulty (largest `|rho|` 0.48, and negative for obstacle density).
- Not circular: no model under evaluation is an input; thresholds are fixed
  numbers with a meaning on the score scale; rank stability is checked with
  two non-panel comparators.

## 6. Generator Coverage Gap

The current generator (version 1.0.0) produces scenarios from a config
(grid, population, exits, ignitions, partitions, doors, door width, scatter,
fire model, spread interval and probability, flammable exits) and accepts or
rejects them by structural constraints (route length, obstacle density, crowd
density, exit separation, passage width, route redundancy, ignition-person and
ignition-exit distance). Its `describe_families` names only **three primary
families** (`single_exit_low_density`, `single_exit_high_density`,
`split_flow_across_exits`; every multi-exit scenario becomes
`split_flow_across_exits`) and four secondary labels. It cannot target a
difficulty band.

Evidence: the development pool (10 recipes, 1,120 scenarios) and the
`gap-probe` command (four config-only recipes, development seeds; results in
[suite-design-results.md, part C](suite-design-results.md#part-c-generator-coverage-probe)).

| Family | Current generator support | Missing generator capability | Procedural generation feasible? | Required engineering work |
| --- | --- | --- | --- | --- |
| `single_exit_low_density` | Primary label; any recipe with one exit and few people | none | yes (shown) | none |
| `single_exit_high_density` | Primary label; `dense-single` recipe | A capacity check at generation time: 34 of 120 deterministic and 24 of 40 stochastic dense scenarios are capacity artifacts | yes (shown; stochastic variant shown by the probe) | Optional constraint "capacity ceiling >= 50" in the generator, or keep the evaluator-side filter; add a stochastic dense recipe |
| `narrow_corridor_bottleneck` | Secondary label; `door_width` and `passage_width` controls | Primary-family assignment | yes (shown) | Label only |
| `split_flow_across_exits` | Primary label, but given to **every** multi-exit scenario | A split-pressure proxy and constraint (exit load imbalance at `N >= 20`); a medium-population multi-exit recipe. In the pool the tag appears almost only at crowd scale | yes (proxy is structural and seed-free; the probe's crowded-maze recipe gives 5 split-flow scenarios with `N < 100`) | New proxy, new constraint, fixed label rule, new recipe (new generator version) |
| `crowd_100` | Secondary label; population control | none (stochastic crowd shown by the probe) | yes (shown; slow: about 30x the CPU of a small scenario) | Recipe only |
| `maze_obstacle_routing` | Secondary label; partitions and scatter | The default recipe never gives the Moderate band (0 of 120) | yes: the probe's crowded maze (more people, faster fire) and stochastic maze recipes give Moderate mazes | Recipe only |
| `deceptive_nearest_exit` | None (no label, proxy or constraint) | A deceptive-exit proxy (share of people whose nearest exit by distance is not the nearest by route) and a constraint | yes (structural, seed-free) | New proxy and constraint; label (new generator version) |
| `threatened_nearest_exit` | Indirect: `ignition_exit_distance` constraint | A fire-aware tag. It needs the fire trajectory, so for stochastic fire it depends on the evaluation seed | yes for deterministic fire (rejection on the one trajectory); for stochastic fire only evaluator-side | Constraint for deterministic fire; keep the evaluator-side tag (this tool's `family_tags`) for stochastic fire and promote it to a supported API |
| `multiple_ignitions` | Secondary label; ignition-count control | Primary-family assignment | yes (shown) | Label only |
| `route_invalidation_replanning` | None | An invalidation tag; it needs the trajectory (same limit as above) | yes, by evaluator-side rejection sampling (10.1% of the pool qualifies) | Same as `threatened_nearest_exit` |
| `central_ignition` (secondary) | None | not needed for V1 strata | yes | none for V1 |
| `multiple_exits_unequal_distance` (secondary) | Implicit (`exit_separation`) | not needed for V1 strata | yes | none for V1 |

Cross-cutting work for the next generator PR:

1. A family-assignment rule for all ten primary families (today: three), with
   the thresholds frozen in a generator version.
2. Recipes that target cells: a medium-population split-flow recipe, a
   crowded-maze recipe and stochastic variants of the dense, maze and crowd
   recipes. The probe shows that the last three need configuration only.
3. Fire-dependent tags (threatened exit, route invalidation) cannot be a
   seed-free generator property for stochastic fire. They stay evaluator-side,
   measured on the selected unique worlds, and the freeze record stores them.
4. Difficulty cannot be targeted by the generator. Bands come from the
   evaluator's calibration (section 5) and rejection sampling.

**Until this work lands, the V1 suite is composed by evaluator-side tagging
and rejection sampling from a large pool** (section 12). This works for every
cell (shown by the pool and the probe), but it is not yet a supported library
feature. The design does not claim that the generator alone produces the
10-family suite.

## 7. Suite size

400 simulated suites per size, drawn cell by cell from the 517 admitted
development units, with the final shares (2/4/6/4 of 16, 25% stochastic, 3
worlds each). Six comparators: the four panel members and two non-panel
noisy policies. Widths are in official EMBER points.

| Scenarios / family | Scenarios | Unique worlds (simulated) | Max 95% suite-sampling width | Within-suite CI width (mean) | Max paired width | Largest gap reversing in > 5% of suites | Kendall tau | Family-mean width | Decision calls / model run | Evaluator CPU / run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 100 | 124 | 6.59 | 9.26 | 7.02 | 1.16 | 0.88 | 17.40 | 3,054 | 2.4 min |
| 12 | 120 | 158 | 6.46 | 10.02 | 7.20 | 2.08 | 0.89 | 14.49 | 3,843 | 3.0 min |
| **16** | **160** | **210** | **5.36** | **7.91** | **5.74** | **1.12** | **0.90** | **14.20** | **5,108** | **3.9 min** |
| 20 | 200 | 264 | 4.67 | 7.44 | 5.47 | 1.77 | 0.92 | 11.48 | 6,497 | 5.0 min |
| 24 | 240 | 316 | 4.24 | 6.44 | 4.85 | 1.27 | 0.92 | 11.06 | 7,746 | 5.9 min |

(Simulated suites have fewer worlds than the design because four families have
no stochastic candidate in the development pool and fall back to deterministic
slots; with the probe recipes the design has 24 worlds per family.)

**Decision: 16 per family.**

- 16 is the smallest size with a paired width of about 6 points or less
  (5.7 to 6.2 in independent simulations; half-width about 3 points) for every
  adjacent comparator pair; 10 and 12 are above 7.
- It gives every family at least 4 scenarios in each of Moderate, Hard and
  Stress and 2 Easy controls, so every family x band cell has more than one
  unit.
- 20 per family narrows the widths by about 5% to 13% for 25% more episodes,
  API calls and rare-cell candidates. That is not worth its cost in V1.
- The "largest gap reversing" metric depends on which comparator pairs exist,
  so it is not monotone in size; at 16 every pair more than 1.12 points apart
  kept its order in at least 95% of suites.
- Family coverage: all 10 families x 4 bands at every size from 10.

## 8. Difficulty allocation

Same simulation, 16 per family, different band shares:

| Allocation (E/M/H/S) | nearest-exit | static | fire-aware | congestion | noisy 0.05 | noisy 0.2 | Scores < 1 | Mean P(paired CI resolves) | Mean 95% width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform 4/4/4/4 | 7.1 | 23.6 | 36.5 | 38.2 | 35.3 | 19.1 | 20.3% | 0.57 | 4.50 |
| previous 2/4/8/6 (scaled to 2/3/6/5) | 5.0 | 15.0 | 25.1 | 26.9 | 25.8 | 14.4 | 25.2% | 0.70 | 3.89 |
| **final 2/4/6/4** | **5.6** | **16.3** | **27.1** | **28.9** | **28.0** | **15.2** | **22.4%** | **0.68** | **3.91** |
| hard-heavy 1/3/8/4 | 4.4 | 11.9 | 20.9 | 23.1 | 23.0 | 13.2 | 23.3% | 0.56 | 3.52 |
| stress-heavy 1/3/4/8 | 3.9 | 10.5 | 18.6 | 20.0 | 19.1 | 10.6 | 27.5% | 1.00 | 3.22 |
| no easy 0/6/6/4 | 4.4 | 10.5 | 20.0 | 22.0 | 21.4 | 10.9 | 23.3% | 0.70 | 3.16 |

Why 2/4/6/4 is scientifically useful:

- **Headroom.** The best V1 comparator scores about 29 points. A strong model
  has 70 points of room before saturation.
- **Not a floor suite.** 22.4% of comparator x world scores are below 1
  point, against 27.5% for the stress-heavy mix. Stress worlds give headroom,
  but every Stress world scores about 0 for weak policies and carries no
  information about them.
- **Separation.** Moderate and Hard are where the panel spreads most (mean
  spread 36 and 15 points). The final mix has the most Moderate slots of the
  separating mixes. It matches the previous 2/4/8/6 in resolving power with
  fewer floor scores.
- **Controls.** Two Easy worlds per family detect basic faults (a model that
  fails an Easy world has a bug, not a weakness). "No easy" removes them for
  no gain.
- The stress-heavy mix resolves every counted pair only because three of its
  five adjacent pairs fall below the 1-point practical difference there and are
  not counted; it is not more precise.

## 9. Deterministic and stochastic fire

Variance components over admitted stochastic units (up to 6 worlds each):
the world-to-world share of the variance is 24% to 42% depending on the
comparator. So for a fixed episode budget, more scenarios always beat more
worlds per scenario (`Var = sigma_b^2 / S + sigma_w^2 / (S W)`); extra worlds
are worth their cost only to measure robustness to fire realisations inside a
scenario.

| Stochastic scenarios x worlds (per family) | Unique worlds | Max 95% width | Max paired width | CVaR10 width | Fallback slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 x 1 | 160 | 5.12 | 6.19 | 0.45 | 7.5% |
| 4 x 1 | 160 | 5.76 | 6.22 | 0.40 | 16.9% |
| 4 x 2 | 185 | 5.65 | 6.02 | 0.36 | 16.9% |
| **4 x 3** | **210** | **5.85** | **6.19** | **0.33** | **16.9%** |
| 4 x 5 | 260 | 5.03 | 5.17 | 0.32 | 16.9% |
| 8 x 3 | 264 | 5.08 | 5.42 | 0.26 | 25.0% |
| 16 x 3 | 368 | 5.71 | 5.38 | 0.18 | 42.5% |

(Simulated, 400 suites each; differences below about 0.5 points are Monte
Carlo noise.)

**Decision: 12 deterministic + 4 stochastic scenarios per family, 3 unique
fire worlds per stochastic scenario.**

- Precision barely depends on the mix; the decision rests on coverage. Four
  stochastic scenarios per family (25% of scenarios, 50% of worlds) test
  robustness to fire randomness in every family, and 3 worlds give each one a
  within-scenario spread (the lower-tail CVaR10 width falls from 0.40 at 1
  world to 0.33 at 3; 5 worlds add little).
- More stochastic scenarios need stochastic candidates in every family. The
  development pool has none for four families (fallback 17% to 43%); the probe
  shows they need recipes only.
- The previous 14/6 x 3 out of 20 is not kept: at 16 per family, 4 x 3 gives
  the same precision with fewer rare stochastic candidates.
- Worlds come from `build_unique_world_manifest`: three seeds with pairwise
  different trajectories, never duplicates.

## 10. Model trials (K)

Model randomness is separate from environment randomness: trials repeat the
same frozen worlds. Surrogate: congestion-aware-v1 with ε random actions, 10
trials on 60 admitted worlds; trial noise of the suite mean for 240 worlds.

| ε | K | Trial-noise SD of the suite mean (points) | CI width inflation vs K = infinity | Width saved vs K = 1 |
| --- | ---: | ---: | ---: | ---: |
| 0.05 | 1 | 1.00 | 1.073 | — |
| 0.05 | 3 | 0.58 | 1.025 | 4.8% |
| 0.05 | 5 | 0.45 | 1.015 | 5.8% |
| 0.05 | 10 | 0.32 | 1.008 | 6.6% |
| 0.2 | 1 | 1.21 | 1.146 | — |
| 0.2 | 3 | 0.70 | 1.051 | 9.5% |
| 0.2 | 5 | 0.54 | 1.031 | 11.5% |
| 0.2 | 10 | 0.38 | 1.016 | 13.0% |

**Decision: K = 3 for every model that is not verified deterministic; K = 1
for a verified-deterministic model** (two runs on the first 10 worlds give
identical action sequences; otherwise K = 3). K = 3 removes about 70% to 75% of
the possible gain; K = 5 adds 1 to 2 percentage points for 67% more calls.
Re-check with the measured trial variance of the first real models: if the
within-world share of a model's variance is above 0.5, use K = 5 for it.

## 11. Sample size, CI and rank stability (final design)

All values for 16 per family, 400 simulated suites, paired worlds (every
comparator on the same worlds), unique worlds only.

| Quantity | Value |
| --- | --- |
| 95% suite-sampling width of the family-balanced score (max over comparators) | 5.63 points |
| within-suite 95% CI width (stratified CLT, mean) | 7.89 points (about +-3.9) |
| paired 95% width, adjacent comparators (max) | 5.76 points |
| largest expected gap that reverses in > 5% of suites | 1.12 points |
| P(every adjacent pair >= 1 point apart in the right order) | 0.63 |
| mean Kendall tau to the expected order (6 comparators) | 0.90 |
| family-mean 95% width (one family, best comparator) | 13.9 points |
| CVaR10 95% width over unique worlds | 0.33 points |

- **Overall score:** about +-4 points. Two models whose paired difference is
  more than about 3 points are resolved. Differences near the official
  practical difference (1 point) are not: the two comparator pairs about
  1.1 points apart reverse in about 20% of suites. Reports must call such pairs
  ties, not rank them.
- **Family scores** are coarse (about +-7 points with 16 scenarios). Report
  them with intervals; never rank models on one family.
- **Lower tail:** p10 and CVaR10 are 0 for every V1 comparator (the worst 10%
  of worlds are lost by every heuristic), so the lower-tail statistics are
  stable but carry information only for strong models.
- Use paired differences on the same pack for model comparison.

## 12. Candidate-pool size

Acceptance per generated candidate = admitted, tagged with the family and in
the band (1,120 development candidates, current recipe mix):

| Band | Acceptance (any family) |
| --- | ---: |
| Easy | 29.5% |
| Moderate | 7.9% |
| Hard | 8.4% |
| Stress | 10.0% |

Per family: 3.1% (`crowd_100`) to 17.9% (`narrow_corridor_bottleneck`).
Rarest cells (current mix): `maze_obstacle_routing`/Moderate (0 hits),
`crowd_100`/Moderate, `deceptive_nearest_exit`/Stress and
`maze_obstacle_routing`/Stress (0.36%), the deceptive and maze Hard cells
(0.45%).

Binomial requirement per cell at 99.875% (a 95% family-wise guarantee over 40
cells, Bonferroni): **3,611 candidates** (22.6x the suite) for all cells
except maze Moderate, which the current mix never produced. The probe's maze
recipes produce it at 10% (crowded maze) and 20% (stochastic maze), so a
targeted batch of 59 to 123 candidates fills it. The real selector filled
30%, 49%, 69% and 88% of the quotas from 140, 280, 560 and 1,120 candidates,
and never all of them.

**Decision:** generate **4,000 candidates** (25x the suite; 11% margin over
the binomial requirement) with the current recipes plus the probe recipes.
Then, for every cell with a shortfall, generate targeted batches of 200 with
the best recipe for that cell until the cell is full or 10,000 candidates in
total are reached. The stopping rule depends only on cell counts, never on a
model. A cell still short at the cap is reported in the freeze record. The
binomial model ignores competition between cells (one candidate serves one
cell), so the empirical fill, not the binomial number, is the reason for the
top-up rule. Evaluator cost: about 6.4 CPU hours for the panel on 4,000
candidates (1.79 h for 1,120).

## 13. API workload (final design)

| Quantity | Value |
| --- | --- |
| episodes, verified-deterministic model (K = 1) | 240 |
| episodes, other model (K = 3) | 720 |
| expected decision calls (one per simulator step; congestion-aware step counts) | 5,825 (K = 1); 17,475 (K = 3) |
| worst-case calls (every episode to `max_steps`) | 21,972 (K = 1); 65,916 (K = 3) |
| step-0 observation size | 1,906 characters mean |
| approximate input tokens (calls x step-0 characters / 4) | 2.8 M (K = 1); 8.3 M (K = 3) |
| evaluator CPU (simulator + official scoring) | about 4 minutes per model run |

Observation size grows with population: `crowd_100` worlds dominate the
token count. Cost does not decide validity; it is shown for planning.

## 14. Hidden evaluation protocol

In this order. No step uses any model under evaluation, and there is no
model-specific regeneration.

1. **Freeze versions.** Benchmark, simulator, observation, scoring
   (`SCORING_VERSION`), generator and generator-config versions; calibration
   panel 1.0; environment-world version 1.0; analysis tool version. Publish a
   hash of this rule set.
2. **Freeze quotas.** 10 families x 16 (2/4/6/4), 4 stochastic x 3 worlds,
   admission rule, band thresholds, tag thresholds, pool size and top-up rule.
   Include them in the rule-set hash.
3. **Choose private seeds.** From a CSPRNG in the evaluator's private
   workspace: one high-entropy generator seed per recipe, one evaluation-seed
   stream, one selection key. Publish SHA-256 commitments only.
4. **Generate the candidate pool** (section 12). Drop duplicate scenario
   hashes and any scenario equal to a public or earlier-round scenario.
5. **Run the fixed calibration panel** on every candidate's unique worlds
   (`distinct_world_seeds`, 3 candidate worlds for stochastic fire): official
   EMBER Score, `M*`, `M_cap`, capacity ceiling, admission, band, tags.
6. **Stratify** admitted candidates by family x band x fire model.
7. **Select deterministically** with `select_suite` and the private key.
   Report every shortfall; never borrow another band.
8. **Freeze the unique evaluator worlds** with `build_unique_world_manifest`
   (1 world per deterministic scenario, 3 per stochastic one). Check that every
   `environment_world_digest` is unique.
9. **Create one immutable `EnvironmentPack`** (`freeze_environment_pack`) with
   a new suite ID, version and batch ID.
10. **Commit only digests publicly before evaluation:** pack digest, suite
    hash, manifest hash, rule-set hash, counts per family x band x fire model.
    Seeds, key, scenarios and measurements stay in the private freeze record.
11. **Evaluate every model on the same pack**, with its K, replay-verified.
    Record provider, model ID and version, date and K for every run.

## 15. Leakage and contamination

- **No hidden material to developers.** They get the public suite, the
  generator, the rules and the commitments. They never get hidden scenarios,
  seeds, world digests, bands or per-world results of a live suite. The
  observation contains no seed.
- **No per-world results while a suite is live.** Publish overall and
  per-family scores with intervals only.
- **Provider and model identity** are recorded per run (section 14, step 11).
  A suite is treated as exposed to every provider that served it.
- **No hidden suite in release artifacts.** Private material lives only under
  leak-guarded paths; `tools/check_public_assets.py` checks the built
  distribution; tests check that no study cache or generated suite is
  committed and that the tool and `worlds.py` contain no seed literal.
- **Reveal only after retirement:** generator seeds, selection key, pack and
  freeze record, so that anybody can check the suite with
  `verify_regeneration`.
- **Canary policy.** A canary string in prompts or archives can show, after
  the fact, that a **copy of the material** appeared somewhere it was searched
  for (a crawl, a dataset, a model output that reproduces it verbatim). It
  cannot show that a model was trained on the suite, and its absence proves
  nothing. EMBER observations are grid states, which a model can learn from
  without reproducing any string. This design therefore does **not** use a
  canary as a contamination test. A harness may add one to pack archives for
  provenance tracing only, and must describe it that way.
- **Contamination check (when required).** Before a round closes, if a model
  is suspected (for example a large jump between rounds), generate a fresh
  check suite with the same frozen rules, a new seed and a new key, and run
  that model and the calibration panel on it. A model whose paired score
  (against the panel) on the live suite exceeds its paired score on the check
  suite by more than the paired 95% interval is flagged for review. This is a
  statistical check of overfitting to the live suite, not proof of training
  contamination.

## 16. Refresh and version policy

A live suite is replaced, never edited, and never replaced silently. Every
replacement has a **new suite ID and version, a new pack digest and a
published reason**. Scores on different suites are not pooled or ranked
together. The scoring version is separate from the suite version: a suite
records the scoring version it was calibrated with, and a scoring change does
not rename a suite by itself.

| Trigger | Action |
| --- | --- |
| Leak of hidden material | Retire at once; new suite; re-run affected models |
| Contamination flag confirmed (section 15) | Retire; new suite |
| Saturation: the best model scores >= 90 overall or on Stress | New calibration-panel version with a stronger reference; new suite |
| Generator bug that affects selected worlds | Retire; fix; new generator version; new suite |
| Scoring or benchmark revision (`SCORING_VERSION` or rule change) | Re-calibrate the bands on the new score; new suite |
| Simulator or observation revision | New suite (world digests change) |
| Invalid hidden world (for example a wrong `M*` or a capacity artifact found later) | Retire the suite and publish the defect; never drop a world from a live suite |
| Scheduled | New suite each evaluation round (at least every 6 months); retire and reveal the previous one |

To compare rounds, run the calibration panel on every suite and report models
relative to it (a linking set), not raw scores across suites.

## 17. Final V1 proposal

| Item | Value |
| --- | --- |
| Primary families | **10**: `single_exit_low_density` (basic evacuation planning), `single_exit_high_density` (queue management at one exit), `narrow_corridor_bottleneck` (sequencing through a width-1 cut), `split_flow_across_exits` (load balancing across exits), `crowd_100` (scale), `maze_obstacle_routing` (path finding with detours), `deceptive_nearest_exit` (exit choice by true route length), `threatened_nearest_exit` (fire anticipation), `multiple_ignitions` (several fire fronts), `route_invalidation_replanning` (replanning). `central_ignition` and `multiple_exits_unequal_distance` are secondary tags. |
| Scenarios per family | 16 |
| Difficulty per family | Easy 2, Moderate 4, Hard 6, Stress 4 |
| Deterministic / stochastic per family | 12 / 4 |
| Unique fire trajectories per stochastic scenario | 3 |
| Total scenarios | 160 |
| Total unique worlds | **240** (120 deterministic, 120 stochastic) |
| Model trials | K = 1 verified-deterministic; K = 3 otherwise |
| Model episodes | 240 (K = 1); 720 (K = 3) |
| Expected 95% CI width | about 7.9 points within a suite (about +-3.9); paired width between models at most about 5.8 points |
| Candidate pool | 4,000 candidates, then targeted top-up batches of 200 per short cell, cap 10,000 |
| API calls per model | about 5,800 (K = 1) or 17,500 (K = 3); worst case 22,000 or 66,000 |
| Calibration panel | version 1.0: nearest-exit-greedy, static-shortest-path, fire-aware-shortest-path, congestion-aware; random-legal as floor; privileged reference as bound only |
| Admission and bands | difficulty-calibration.md sections 4 to 6 |
| World identity | `environment_world_digest` (version 1.0); schedule `build_unique_world_manifest` |
| Generation and freeze | Section 14 (11 ordered steps) |
| Refresh | Section 16 |
| Prerequisite | Generator work in section 6 for a library-supported pipeline; until then evaluator-side tagging and rejection sampling |
