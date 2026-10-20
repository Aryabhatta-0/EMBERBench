# Generator coverage validation (development pool)

Validation tool version 1.0.0. **This is a DEVELOPMENT pool built from public seed labels (`emberbench/generator-coverage/development/`). It is not, and must never be used as, the hidden evaluation suite.**

## Versions

- composition: `1.0`
- generator: `1.1.0`
- generator_config: `1.1`
- generation_manifest: `1.1`
- recipe_catalog: `1.0`
- family_rules: `1.0`
- selection_rules: `1.0`
- simulator: `0.2.0`
- observation: `1.0`
- scenario_schema: `1.0`
- scoring: `1.0`
- environment_world: `1.0`
- calibration_panel_version: `1.0`
- difficulty_rules_version: `1.0`
- admission_rules_version: `1.0`
- rule-set digest: `61985f4711e28ebd68c7d58140f015e9569ed4eeda819420a1b7e7f8ced0f8e1`

## Pool

- Candidates measured: 4000 (initial plan 4000, protocol size 4000); rounds: 1 (round 0 + 0 top-up rounds).
- Dropped before measurement: none (duplicates by content hash; public-suite hashes).
- Evaluation candidate seeds: 64; stochastic scenarios are measured on the 3 unique worlds that `build_unique_world_manifest` schedules.

### Candidates per recipe

| recipe | fire | candidates | admitted | eligible | capacity | saturated | floor | no sep. | n/a + insuff. |
|---|---|---|---|---|---|---|---|---|---|
| open-sparse | deterministic | 120 | 24 | 24 | 1 | 73 | 7 | 13 | 2 |
| dense-single | deterministic | 160 | 69 | 69 | 46 | 38 | 3 | 4 | 0 |
| partitioned | deterministic | 160 | 100 | 94 | 3 | 37 | 7 | 12 | 1 |
| multi-exit | deterministic | 160 | 89 | 38 | 0 | 47 | 4 | 20 | 0 |
| multi-ignition | deterministic | 160 | 76 | 76 | 1 | 35 | 8 | 40 | 0 |
| maze | deterministic | 120 | 99 | 93 | 1 | 17 | 0 | 3 | 0 |
| exit-threat | deterministic | 160 | 89 | 45 | 0 | 31 | 11 | 29 | 0 |
| crowd | deterministic | 140 | 118 | 118 | 7 | 6 | 8 | 1 | 0 |
| stochastic-open | stochastic | 160 | 55 | 40 | 2 | 64 | 7 | 31 | 1 |
| stochastic-partitioned | stochastic | 160 | 97 | 87 | 0 | 35 | 3 | 21 | 4 |
| maze-crowded | deterministic | 200 | 144 | 143 | 23 | 25 | 0 | 7 | 1 |
| maze-stochastic | stochastic | 200 | 168 | 163 | 2 | 26 | 0 | 3 | 1 |
| dense-single-stochastic | stochastic | 160 | 49 | 49 | 94 | 11 | 1 | 5 | 0 |
| crowd-stochastic | stochastic | 120 | 93 | 93 | 9 | 3 | 13 | 2 | 0 |
| split-flow-medium | deterministic | 200 | 159 | 159 | 6 | 25 | 5 | 5 | 0 |
| split-flow-medium-stochastic | stochastic | 160 | 128 | 128 | 4 | 17 | 6 | 5 | 0 |
| deceptive-exit | deterministic | 200 | 149 | 149 | 0 | 35 | 5 | 11 | 0 |
| deceptive-exit-stochastic | stochastic | 200 | 153 | 153 | 4 | 30 | 3 | 10 | 0 |
| exit-threat-stochastic | stochastic | 200 | 135 | 57 | 0 | 30 | 5 | 30 | 0 |
| route-invalidation | deterministic | 160 | 96 | 50 | 8 | 31 | 1 | 23 | 1 |
| route-invalidation-stochastic | stochastic | 180 | 130 | 63 | 0 | 28 | 1 | 21 | 0 |
| narrow-corridor | deterministic | 120 | 82 | 82 | 8 | 14 | 1 | 13 | 2 |
| narrow-corridor-stochastic | stochastic | 120 | 90 | 90 | 5 | 13 | 1 | 11 | 0 |
| multi-ignition-stochastic | stochastic | 160 | 98 | 98 | 2 | 28 | 3 | 26 | 3 |
| open-sparse-stochastic | stochastic | 120 | 32 | 32 | 1 | 53 | 1 | 30 | 3 |

### Admission outcomes (whole pool)

| outcome | count | rate |
|---|---|---|
| admitted | 2522 | 0.630 |
| not_applicable | 19 | 0.005 |
| capacity_artifact | 227 | 0.057 |
| saturated | 752 | 0.188 |
| floor | 104 | 0.026 |
| no_separation | 376 | 0.094 |
| insufficient_worlds | 0 | 0.000 |

- Deterministic candidates: 2060 (1140 eligible); stochastic: 1940 (1053 eligible).
- Acceptance rate (admitted and at least one primary family): 0.548.
- Capacity-artifact rejection rate: 0.057 of all candidates (0.057 of applicable ones).

### Eligible candidates per family and band

A candidate counts for every primary family it carries; selection gives it exactly one.

| family | easy (det+sto) | moderate (det+sto) | hard (det+sto) | stress (det+sto) | total |
|---|---|---|---|---|---|
| single_exit_low_density | 79+58 | 12+53 | 5+13 | 8+7 | 235 |
| single_exit_high_density | 26+12 | 21+18 | 14+9 | 13+14 | 127 |
| narrow_corridor_bottleneck | 405+253 | 107+166 | 66+49 | 74+66 | 1186 |
| split_flow_across_exits | 160+92 | 52+60 | 48+36 | 56+51 | 555 |
| crowd_100 | 53+26 | 20+21 | 18+16 | 27+30 | 211 |
| maze_obstacle_routing | 182+118 | 26+47 | 25+6 | 10+4 | 418 |
| deceptive_nearest_exit | 191+137 | 38+73 | 26+16 | 22+16 | 519 |
| threatened_nearest_exit | 69+35 | 25+41 | 37+46 | 60+68 | 381 |
| multiple_ignitions | 66+89 | 21+85 | 25+32 | 34+27 | 379 |
| route_invalidation_replanning | 30+12 | 21+52 | 43+39 | 59+58 | 314 |

Eligible per band: easy 1082, moderate 502, hard 294, stress 315.

### Rarest quota cells (eligible / quota)

| family | band | fire | eligible | quota | ratio |
|---|---|---|---|---|---|
| single_exit_low_density | hard | deterministic | 5 | 4 | 1.250 |
| single_exit_low_density | stress | deterministic | 8 | 3 | 2.667 |
| maze_obstacle_routing | hard | stochastic | 6 | 2 | 3.000 |
| maze_obstacle_routing | stress | deterministic | 10 | 3 | 3.333 |
| single_exit_high_density | hard | deterministic | 14 | 4 | 3.500 |
| maze_obstacle_routing | stress | stochastic | 4 | 1 | 4.000 |
| single_exit_low_density | moderate | deterministic | 12 | 3 | 4.000 |
| single_exit_high_density | stress | deterministic | 13 | 3 | 4.333 |
| crowd_100 | hard | deterministic | 18 | 4 | 4.500 |
| single_exit_high_density | hard | stochastic | 9 | 2 | 4.500 |
| maze_obstacle_routing | hard | deterministic | 25 | 4 | 6.250 |
| multiple_ignitions | hard | deterministic | 25 | 4 | 6.250 |

## Selection

- Quota cells (family x band x fire model): 70; filled completely: 70; short slots: 0.
- Family x difficulty cells filled to quota: 40 of 40.
- Selected: 160 scenarios (40 stochastic).

### Pool size needed (keyed subsamples of the initial pool)

| fraction | candidates | short slots | short cells |
|---|---|---|---|
| 0.250 | 1021 | 18 | 12 |
| 0.500 | 1970 | 3 | 1 |
| 0.750 | 2978 | 0 | 0 |
| 1.000 | 4000 | 0 | 0 |

## Overlap audit

| pair A / B | eligible A | eligible B | both | Jaccard | selected A also B | selected B also A |
|---|---|---|---|---|---|---|
| crowd_100 / split_flow_across_exits | 211 | 555 | 211 | 0.380 | 16/16 | 2/16 |
| threatened_nearest_exit / route_invalidation_replanning | 381 | 314 | 174 | 0.334 | 6/16 | 9/16 |
| maze_obstacle_routing / deceptive_nearest_exit | 418 | 519 | 176 | 0.231 | 9/16 | 2/16 |

- Split flow below 100 people: 344 eligible candidates; selected split-flow populations: [20, 20, 24, 25, 29, 32, 38, 39, 39, 40, 43, 45, 52, 56, 105, 115].
- Selected scenarios with more than one primary family: 0 (each candidate fills exactly one cell).

### Fire-dependent tags

- threatened_nearest_exit: 191 deterministic (one trajectory) and 190 stochastic (mean over the 3 scheduled unique worlds) eligible candidates.
- route_invalidation_replanning: 153 deterministic (one trajectory) and 161 stochastic (mean over the 3 scheduled unique worlds) eligible candidates.

## Runtime

- Workers: 11; total wall clock: 48.8 min.
- Generation: 52.4 s wall, 446.1 s CPU.
- Measurement (unique worlds, panel runs, official scoring, tags): 2863.1 s wall, 31430.3 s CPU (7.86 s per candidate).

| candidate kind | count | mean s | max s |
|---|---|---|---|
| crowd_100 deterministic | 140 | 30.103 | 62.224 |
| crowd_100 stochastic | 120 | 125.245 | 286.887 |
| small deterministic | 1920 | 1.527 | 10.593 |
| small stochastic | 1820 | 5.085 | 55.466 |

- Estimate for a 4,000-candidate pool with this recipe mix: 8.7 CPU hours, 48 min on 11 workers.

## Frozen development suite

- Development suite frozen and reloaded: 160 scenarios, suite hash `008bd10f80cf33da...`.
- Unique-world schedule from `build_unique_world_manifest`: 240 episodes, 240 distinct (scenario, seed) worlds; every world digest equals the frozen calibrated world (checked by `schedule_composed_worlds`).
- Primary families: 10 families, 16..16 scenarios each; every selected scenario has exactly one primary family (the cell it filled).
- Freeze record digest `8aaec19bc7e8a473...`, rule-set digest `61985f4711e28ebd...`.
