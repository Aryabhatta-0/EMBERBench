# Generator suite coverage (Phase 11)

| Field | Value |
| --- | --- |
| Generator | `1.1.0` (`GENERATOR_VERSION`), config `1.1` (`GENERATOR_CONFIG_VERSION`; `1.0` still accepted), freeze record `1.1` (`GENERATION_MANIFEST_VERSION`) |
| Family rules | `1.0` (`FAMILY_RULES_VERSION`) |
| Calibration | panel `1.0`, difficulty rules `1.0`, admission rules `1.0` |
| Composition | `1.0` (`COMPOSITION_VERSION`); selection rules `1.0`; recipe catalog `1.0`; suite freeze record `1.0` |
| Modules | [`structure`](../src/emberbench/structure.py), [`families`](../src/emberbench/families.py), [`calibration`](../src/emberbench/calibration.py), [`composition`](../src/emberbench/composition.py), [`generator`](../src/emberbench/generator.py) |
| Validation | [`tools/validate_generator_coverage.py`](../tools/validate_generator_coverage.py), results in [generator-coverage-results.md](generator-coverage-results.md) |
| Unchanged | simulator, observation, action schema, scenario schema, metrics, statistics, scoring (`SCORING_VERSION` 1.0), environment world 1.0 |
| Benchmark revision | `2` (`BENCHMARK_REVISION`). `benchmark_versions()` pins every version above, so provenance from other rules fails verification ([integrity.md](integrity.md) section 12) |

PR #33 ([evaluation-suite-design.md](evaluation-suite-design.md)) froze the V1
evaluation-suite design: 10 primary families x 16 scenarios (Easy 2,
Moderate 4, Hard 6, Stress 4), 12 deterministic and 4 stochastic scenarios per
family, 3 unique fire worlds per stochastic scenario: 160 scenarios and 240
unique worlds. It also showed a gap: the generator labelled only three primary
families and called every multi-exit scenario split flow (section 6 there).
Phase 11 closes the gap without changing any of those numbers and without
changing the simulator or the score. The PR #33 rules moved, unchanged, from
the research tool into the library, and the research tool now imports them.

## 1. Who computes what

| Part | Side | Seed or world needed? | Private during a live evaluation? |
| --- | --- | --- | --- |
| Structural proxies (`measure_proxies`, `profile_structure`) | generator | no (seed-free, scenario only) | the values of hidden scenarios are |
| Structural family tags (`structural_tags`): 8 primary families and both secondary tags | generator | no | the tags of hidden scenarios are |
| Generation constraints and recipes | generator | no | the recipes and rules are **public**; the generator seeds are private |
| Fire evidence (`fire_evidence`): threatened share, (recoverable) invalidation share | evaluator | yes: one fire trajectory per world | yes |
| Fire-dependent tags (`threatened_nearest_exit`, `route_invalidation_replanning`) | evaluator | yes: deterministic fire uses its one trajectory; stochastic fire uses the mean over the scenario's selected unique worlds | yes |
| Calibration (panel scores, `M*`, `M_cap`, capacity ceiling), admission, band | evaluator | yes: every unique world | yes |
| Selection, top-up, neutral relabeling, freeze record | evaluator | private pool seed, private evaluation seeds, private selection key | yes; only `SuiteFreezeRecord.public_commitment()` may be published before the evaluation |

The generator never labels difficulty, never sees an evaluation seed and never
reads a model. A stochastic scenario's family never depends on a model: its
fire-dependent tags are measured on exactly the unique worlds that
`build_unique_world_manifest` will schedule for it (the same candidate-seed
list and the same walk, `world_seeds_for`), and the evidence is frozen into
the freeze record.

## 2. The ten family rules (family rules 1.0, the exact PR #33 thresholds)

| Family | Rule | Side |
| --- | --- | --- |
| `single_exit_low_density` | one exit and crowd density < 0.15 | generator |
| `single_exit_high_density` | one exit and crowd density >= 0.15 | generator |
| `narrow_corridor_bottleneck` | `passage_width <= 1` and `bottleneck_severity = N / min_cut >= 4` | generator |
| `split_flow_across_exits` | several exits, `split_pressure > 1` and `N >= 20` | generator |
| `crowd_100` | `N >= 100` | generator |
| `maze_obstacle_routing` | obstacle density >= 0.25 or detour ratio >= 1.5 | generator |
| `deceptive_nearest_exit` | several exits and `deceptive_share >= 0.25` | generator |
| `threatened_nearest_exit` | several exits and mean threatened share >= 0.2 | evaluator (fire) |
| `multiple_ignitions` | at least 2 ignitions | generator |
| `route_invalidation_replanning` | mean recoverable invalidation share >= 0.2 | evaluator (fire) |
| `central_ignition` (secondary) | ignition centrality <= 0.34 | generator |
| `multiple_exits_unequal_distance` (secondary) | several exits and median exit gap >= 4; also the fallback tag of a multi-exit scenario with no other tag | generator |

Secondary tags never become a primary family: `primary_candidates` drops them
and the design has no cell for them. `family_rules()` returns this table in
canonical form; it is part of the frozen rule set and its digest.

**Split flow (the PR #33 fix).** Several exits are not enough. `split_pressure`
is the busiest route-nearest exit's load divided by 1 plus the median extra
route its people would walk to their second-nearest exit. A low-pressure
multi-exit scenario (people already spread over far-apart exits) is not split
flow; a scenario with 20 or more people crowding one exit while another is
almost as close is. The `split-flow-medium` recipes produce split flow at 24-60
people, so the family is not only a `crowd_100` side effect.

**Deceptive nearest exit.** `deceptive_share` is the share of people whose
Manhattan-nearest exit is a dead end or at least 2 steps longer by route than
their route-nearest exit. It is seed-free. Config 1.1 can constrain it.

**Narrow corridor.** A door is not a bottleneck by itself: the passage must be
1 cell wide for some person and the crowd must need at least 4 steps to drain
through the narrowest cut (`N / min_cut >= 4`). Two people behind a 1-cell door
are not a bottleneck; eight people behind a 3-cell door are not either.

## 3. Generator config 1.1

Config 1.1 adds three optional constraint ranges: `split_pressure`,
`deceptive_share` and `bottleneck_severity`. They aim a recipe at one
structural family by rejection sampling. Details and the new proxies:
[procedural-generation.md, section 6](procedural-generation.md#6-generation-constraints-and-proxies).
The generator stream label contains `GENERATOR_VERSION`, so generator 1.1.0
produces different scenarios than 1.0.0 for the same seed and config (the
sampling code itself did not change). The public suite does not come from the
generator and its hash is unchanged.

## 4. Recipes (recipe catalog 1.0)

A recipe is a generator config without a scenario count: a distribution, not a
test item. No recipe names or hard-codes a hidden scenario. `targets` is used
only to document the recipe and to choose top-up batches; a candidate's
families are always measured.

| Recipe | Fire | Targets | Grid, population, exits, ignitions, partitions | Extra constraints | Initial candidates |
| --- | --- | --- | --- | --- | --- |
| `open-sparse` | deterministic | `single_exit_low_density` | 14x10, N 3-8, exits 1-1, ignitions 1-1, partitions 0-0 | - | 120 |
| `dense-single` | deterministic | `single_exit_high_density` | 14x10, N 20-40, exits 1-1, ignitions 1-1, partitions 0-0 | - | 160 |
| `partitioned` | deterministic | `narrow_corridor_bottleneck` | 18x12, N 6-16, exits 1-2, ignitions 1-1, partitions 1-2 | - | 160 |
| `multi-exit` | deterministic | `threatened_nearest_exit`, `route_invalidation_replanning` | 20x12, N 6-20, exits 2-3, ignitions 1-1, partitions 0-1 | - | 160 |
| `multi-ignition` | deterministic | `multiple_ignitions` | 18x12, N 6-16, exits 2-3, ignitions 2-4, partitions 0-0 | - | 160 |
| `maze` | deterministic | `maze_obstacle_routing` | 16x12, N 4-10, exits 2-3, ignitions 1-1, partitions 2-4 | - | 120 |
| `exit-threat` | deterministic | `threatened_nearest_exit` | 20x12, N 6-16, exits 2-3, ignitions 1-1, partitions 0-1 | ignition_exit_distance <= 6 | 160 |
| `crowd` | deterministic | `crowd_100` | 30x18, N 100-120, exits 2-4, ignitions 1-1, partitions 0-0 | - | 140 |
| `stochastic-open` | stochastic | `single_exit_low_density`, `threatened_nearest_exit` | 18x12, N 5-14, exits 1-3, ignitions 1-2, partitions 0-0 | - | 160 |
| `stochastic-partitioned` | stochastic | `narrow_corridor_bottleneck`, `route_invalidation_replanning` | 18x12, N 5-14, exits 1-3, ignitions 1-2, partitions 1-2 | - | 160 |
| `maze-crowded` | deterministic | `maze_obstacle_routing` | 16x12, N 12-24, exits 2-3, ignitions 1-2, partitions 2-4 | - | 200 |
| `maze-stochastic` | stochastic | `maze_obstacle_routing` | 16x12, N 6-16, exits 2-3, ignitions 1-2, partitions 2-4 | - | 200 |
| `dense-single-stochastic` | stochastic | `single_exit_high_density` | 14x10, N 20-40, exits 1-1, ignitions 1-1, partitions 0-0 | - | 160 |
| `crowd-stochastic` | stochastic | `crowd_100` | 30x18, N 100-120, exits 2-4, ignitions 1-1, partitions 0-0 | - | 120 |
| `split-flow-medium` | deterministic | `split_flow_across_exits` | 20x14, N 24-60, exits 2-3, ignitions 1-1, partitions 0-1 | split_pressure 1.05..inf | 200 |
| `split-flow-medium-stochastic` | stochastic | `split_flow_across_exits` | 20x14, N 24-60, exits 2-3, ignitions 1-1, partitions 0-1 | split_pressure 1.05..inf | 160 |
| `deceptive-exit` | deterministic | `deceptive_nearest_exit` | 18x12, N 8-20, exits 2-3, ignitions 1-1, partitions 1-3 | deceptive_share 0.25..1 | 200 |
| `deceptive-exit-stochastic` | stochastic | `deceptive_nearest_exit` | 18x12, N 8-20, exits 2-3, ignitions 1-1, partitions 1-3 | deceptive_share 0.25..1 | 200 |
| `exit-threat-stochastic` | stochastic | `threatened_nearest_exit` | 20x12, N 6-16, exits 2-3, ignitions 1-1, partitions 0-1 | ignition_exit_distance <= 6 | 200 |
| `route-invalidation` | deterministic | `route_invalidation_replanning` | 18x12, N 6-18, exits 1-3, ignitions 1-1, partitions 1-2 | route_redundancy >= 2 | 160 |
| `route-invalidation-stochastic` | stochastic | `route_invalidation_replanning` | 18x12, N 6-18, exits 1-3, ignitions 1-1, partitions 1-2 | route_redundancy >= 2 | 180 |
| `narrow-corridor` | deterministic | `narrow_corridor_bottleneck` | 18x12, N 8-24, exits 1-2, ignitions 1-1, partitions 1-2 | bottleneck_severity 4..inf; passage_width 1 | 120 |
| `narrow-corridor-stochastic` | stochastic | `narrow_corridor_bottleneck` | 18x12, N 8-24, exits 1-2, ignitions 1-1, partitions 1-2 | bottleneck_severity 4..inf; passage_width 1 | 120 |
| `multi-ignition-stochastic` | stochastic | `multiple_ignitions` | 18x12, N 6-16, exits 2-3, ignitions 2-4, partitions 0-0 | - | 160 |
| `open-sparse-stochastic` | stochastic | `single_exit_low_density` | 14x10, N 3-8, exits 1-1, ignitions 1-1, partitions 0-0 | - | 120 |

The first ten are the PR #33 development recipes, the next four its gap-probe
recipes (crowded maze; stochastic maze, dense single exit and crowd), the rest
are new targeted recipes. Threatened exit and route invalidation cannot be
constrained without a fire trajectory: their recipes raise the rate (exit
close to an ignition; two doors per partition so a route can be cut and still
be recovered) and evaluator-side tagging plus rejection does the rest. The
initial plan totals 4,000 candidates (`V1_INITIAL_PLAN`). The top-up table
`V1_TOP_UP` maps each family and fire model to one dedicated recipe.

## 5. Pipeline (`compose_pool`, `compose_suite`, `freeze_composed_suite`)

1. **Generate** every planned recipe batch (at most 200 candidates per batch)
   with `generate_suite`. The generator seed of a batch is
   `SHA-256("emberbench/candidate-pool/1.0/<recipe>/<version>/<batch>/" || pool_seed)`;
   only its commitment is recorded. Drop duplicate content hashes and every
   excluded hash (the public suite, earlier rounds).
2. **Unique worlds.** For each candidate, the indices of the private
   evaluation seeds that `build_unique_world_manifest` will schedule: the first
   seed for deterministic fire, the first 3 seeds with pairwise-different
   `environment_world_digest` for stochastic fire. A stochastic candidate with
   fewer distinct worlds is dropped as `insufficient_worlds`.
3. **Frozen calibration panel** on every unique world (`measure_world`): the
   official EMBER Score of each panel member (`score_evidence` on
   `episode_evidence`, the same references shared as in `score_results`),
   `M*`, `M_cap`, the capacity ceiling and the fire evidence.
4. **Admission** (`calibrate`, in order): not applicable, capacity artifact
   (mean capacity ceiling < 50, before any other rule), saturated, floor, no
   separation, admitted.
5. **Difficulty band** on `B`, the best panel mean: Easy >= 80, Moderate >= 30,
   Hard >= 10, Stress below.
6. **Family tags**: `family_tags(profile, [fire evidence per world])`.
7. **Quotas** (`select_suite`, selection rules 1.0): families scarcest first
   (fewest eligible candidates, ties by name); in a family, bands Easy..Stress
   and the stochastic cell before the deterministic one; in a cell, candidates
   ordered by `SHA-256("emberbench/suite-selection/" + key + "/" + hash)`. Each
   candidate fills one cell, and that cell's family becomes its one primary
   family. A short cell is reported, never filled from another band or fire
   model.
8. **Top-up**: while a cell is short and the pool is below 10,000 candidates,
   one round generates one 200-candidate batch of the `V1_TOP_UP` recipe for
   each short cell's family and fire model, measures it and fills only the
   short cells (the earlier selection stays fixed). The stopping rule reads
   cell counts only.
9. **Relabel and freeze**: neutral IDs `gen-NNNN` in the order of
   `SHA-256("emberbench/suite-order/" + key + "/" + hash)`; each relabeled
   scenario is checked to keep every fire trajectory; then
   `<dir>/suite/` (a normal suite) and `<dir>/suite-freeze.json` are written
   once and loaded back verified. `schedule_composed_worlds` builds the
   evaluation manifest with `build_unique_world_manifest` and checks every
   world digest against the freeze record.

The result is a pure function of the candidate content, the frozen rules, the
private pool seed, evaluation seeds and selection key, and the quotas. Worker
count, candidate order, clock and model identity do not change it (tested).

## 6. Suite freeze record (`suite-freeze.json`, version 1.0)

| Key | Content |
| --- | --- |
| `suite` | suite ID, version, hash |
| `rule_set`, `rule_set_digest` | every version (generator, generator config, generation manifest, recipe catalog, family rules, selection rules, calibration panel, difficulty rules, admission rules, simulator, observation, scenario schema, scoring, environment world), the family rules, the calibration rules, the selection rules with the top-up table, the design with every cell quota, every recipe with its config and config hash |
| `commitments` | SHA-256 commitments to the pool seed, the evaluation seed list and the selection key |
| `batches` | per batch: recipe, version, batch index and ID, candidates, config hash, generator-seed commitment, generation hash |
| `pool` | candidates per recipe, admission outcomes, eligible candidates per cell, dropped duplicates |
| `selected` | per scenario: neutral ID, final hash, candidate hash, origin (recipe, batch, ordinal), primary family, band, fire model, all tags, structural tags, proxies, calibration, and per world the seed index, `environment_world_digest`, `M*`, `M_cap`, capacity ceiling, fire evidence and panel scores |
| `shortfall` | every short cell |

It holds no seed and no key. It is still **private while the suite is live**:
it identifies the scenarios. `SuiteFreezeRecord.public_commitment()` (suite
hash, freeze-record digest, rule-set digest, commitments, cell counts, shortfall
size) is what may be published before the evaluation. The leak guard rejects
`suite-freeze.json` anywhere in the repository or a distribution.

## 7. Generator coverage validation

`tools/validate_generator_coverage.py run --out <dir outside the repo>` runs
the whole pipeline on a **development** pool built from public seed labels
(`emberbench/generator-coverage/development/`). It reports candidates per
recipe, admission outcomes, eligible candidates per family, band and fire
model, the rarest cells, the selection and its shortfall, keyed-subsample pool
sizes, the overlap audit, fire-dependent tag counts and runtime, then freezes a
development suite and checks its 240-world schedule. This pool is not, and
must never be called, a hidden suite. Results:
[generator-coverage-results.md](generator-coverage-results.md).
