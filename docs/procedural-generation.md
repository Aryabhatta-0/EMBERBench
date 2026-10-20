# Procedural unseen scenario generation

| Field | Value |
| --- | --- |
| Module | `emberbench.generator` |
| Generator version | `1.1.0` (`GENERATOR_VERSION`) |
| Config version | `1.1` (`GENERATOR_CONFIG_VERSION`); `1.0` configs are still accepted |
| Freeze record version | `1.1` (`GENERATION_MANIFEST_VERSION`) |
| Family rules | `1.0` (`FAMILY_RULES_VERSION`, [`emberbench.families`](../src/emberbench/families.py)) |
| Scenario schema | `1.0` (unchanged) |
| Suite visibility | `generated` |

The public V1 suite is fixed and open, so a model can be tuned to it. Scored
evaluation also needs scenarios that no model has seen. This module generates
such scenarios from a public config and an evaluator-private seed, and freezes
them before any model runs.

The generator does not change the scenario schema, the simulator, the
observation or the metrics. A generated scenario is an ordinary V1 scenario
file.

## 1. Two kinds of seed

| Seed | Owner | Selects | Where it goes |
| --- | --- | --- | --- |
| **Generator seed** | Evaluator, private until the evaluation is over | The scenarios: layout, exits, ignitions, people, fire parameters | Only into `generate_suite` and `verify_regeneration`. It is never stored. |
| **Evaluation seed** | Evaluator, in the evaluation manifest | The fire trajectory of one episode | `build_evaluation_manifest(suite, seeds)` and then `run_evaluation` |

The two seeds are independent. Use different high-entropy values (for example
`secrets.randbits(256)` for the generator seed and `secrets.randbits(64)` for
each evaluation seed). The `seed` field of a generated scenario file is always
`0`. It is a development default and `run_evaluation` always overrides it.

### Separate RNG domain

The generator does not use `random`, the clock, UUIDs, `os.urandom`, hash order
or any other ambient entropy. It uses the SplitMix64 generator from
`emberbench.rng` on its own streams:

```
state = first 8 bytes of SHA-256(
    "emberbench/generator/" + GENERATOR_VERSION + "/" + config_hash + "/"
    + ordinal + "/" + attempt + "/" + generator_seed as 32 big-endian bytes)
```

- The prefix `emberbench/generator/` never occurs in the simulator labels
  (`emberbench/v1/<seed>/<label>`). Generation draws and fire draws can never
  share a stream.
- Each scenario (ordinal) and each candidate (attempt) has its own stream. A
  rejected candidate does not shift the draws of any other scenario.
- Integers are drawn by rejection sampling, so there is no modulo bias.
  Shuffles are Fisher-Yates over lists in canonical (row-major) order.
- The generator seed is an integer in `[0, 2**256 - 1]`.

## 2. Generate first, freeze second, evaluate third

```python
import secrets
from emberbench import GeneratorConfig, build_evaluation_manifest, freeze_generated_suite, generate_suite, run_evaluation

config = GeneratorConfig.from_json(open("generator-config.json").read())   # public
generator_seed = secrets.randbits(256)                                      # private, kept by the evaluator

generated = generate_suite(config, generator_seed, batch_id="eval-2026-q4-a")
frozen = freeze_generated_suite(generated, "/evaluator/private/eval-2026-q4-a")

manifest = build_evaluation_manifest(frozen.suite, evaluation_seeds)        # private seeds too
for model_id, factory in models.items():
    run_evaluation(frozen.suite, manifest, model_id, factory, f"/evaluator/results/{model_id}.json")
```

The rules:

1. **Generate once.** Never generate a suite per model. Every model gets the
   same frozen suite and the same evaluation manifest. Paired scenario and seed
   fairness (`docs/evaluation.md`) is unchanged.
2. **Freeze before any model runs.** `freeze_generated_suite` writes into a new
   or empty directory and never overwrites. It writes the freeze record last,
   so an interrupted freeze does not load.
3. **Load verified.** `load_generated_suite(directory)` checks every file, every
   pinned hash and every proxy, and returns the `Suite` for evaluation. The
   evaluation manifest then pins the suite hash, which the freeze record also
   pins.

## 3. Frozen generated-suite format

```
<directory>/
  generation.json          freeze record (evaluator-side)
  suite/
    manifest.json          normal suite manifest, visibility "generated"
    gen-0000.json          one V1 scenario file per ordinal
    gen-0001.json
    ...
```

`suite/` is a normal suite: `load_suite` accepts it unchanged. Its manifest has
`"visibility": "generated"` and `"seed_role": "development"`, and
`suite_id` is the batch ID. Each entry has descriptive family labels from fixed
thresholds (see `describe_families`). They describe the scenario; they are not
a difficulty claim. Since generator 1.1.0 the labels are the **structural**
part of the frozen V1 family rules (`structural_tags`, family rules 1.0, see
[generator.md](generator.md)). The fire-dependent families
(`threatened_nearest_exit`, `route_invalidation_replanning`) are never
generator labels: they need a fire trajectory, so the evaluator tags them on
the selected worlds.

`generation.json` pins:

| Field | Meaning |
| --- | --- |
| `generation_manifest_version` | `1.0` |
| `generator_version` | The code version that made the suite |
| `batch_id` | Evaluator-owned label. It must match the ID pattern and must not contain the generator seed (decimal or hex). |
| `config`, `config_hash` | The full public config and its SHA-256 |
| `suite_id`, `suite_version`, `suite_hash` | The suite, pinned by hash |
| `schema_version` | `1.0` |
| `scenarios[]` | Per scenario, in order: `ordinal`, `scenario_id`, `content_hash`, `proxies` |

It does **not** hold the generator seed, the candidate or rejection history,
the attempt numbers or the stream states.

`load_generated_suite` rejects a frozen suite when:

- a scenario file or the suite manifest changed (`SuiteValidationError`), or
- the freeze record is malformed, has unknown keys, or its config hash, suite
  hash, ordinals, IDs, content hashes, proxies or family labels do not match
  the suite (`GenerationError`), or
- a scenario breaks the config's constraints, or the directory has extra
  files (`GenerationError`).

### Neutral IDs

- Scenario IDs are `gen-0000`, `gen-0001`, ... They encode only the suite
  position.
- Person IDs are `p00`, `p01`, ... and exit IDs are `x00`, `x01`, ... The
  numbers follow a random permutation, so an ID says nothing about a position,
  a distance or a solution. This matters because V1 conflicts are resolved by
  lowest ID.
- The description is empty.

## 4. Reproducibility

The output is a pure function of `GENERATOR_VERSION`, the config, the
generator seed and the batch ID. With identical inputs:

- every scenario file has the same bytes (LF line endings on every platform),
- the suite order is the same (ordinal order),
- the config hash, suite hash and freeze-record hash are the same.

The test suite pins golden hashes for a small config and checks them on Linux,
Windows and macOS, and in interpreters with different `PYTHONHASHSEED`
values.

What changes what:

| Change | Scenario hashes | Suite hash | Freeze-record hash |
| --- | --- | --- | --- |
| Generator seed | change | change | change |
| Any config field | change | change | change (and `config_hash`) |
| Batch ID only | same | change | change |
| Order of entries | rejected on load | rejected on load | rejected on load |

The suite hash is a commitment to the private seed. After the evaluation the
evaluator can publish the seed. Anyone can then run
`verify_regeneration(directory, seed)`, which regenerates the suite and
compares every byte. The evaluator can also run it before the evaluation as a
self-check.

## 5. Generator config

All keys are required and unknown keys are rejected. A range is
`{"min": a, "max": b}` with `a <= b`.

```json
{
  "config_version": "1.0",
  "scenario_count": 4,
  "grid": {"width": 12, "height": 9},
  "population": {"min": 3, "max": 8},
  "exit_count": {"min": 1, "max": 3},
  "ignition_count": {"min": 1, "max": 2},
  "layout": {
    "partition_count": {"min": 0, "max": 2},
    "doors_per_partition": {"min": 1, "max": 2},
    "door_width": {"min": 1, "max": 2},
    "scatter_density": {"min": 0.0, "max": 0.1}
  },
  "fire": {
    "model": "stochastic",
    "spread_interval": {"min": 1, "max": 3},
    "exits_flammable": true,
    "spread_probability_choices": [0.25, 0.5, 0.75]
  },
  "movement": {"enter_fire": "casualty"},
  "max_steps": 60,
  "max_attempts": 200,
  "constraints": {
    "route_length": {"min": 1, "max": 60},
    "obstacle_density": {"min": 0.0, "max": 0.5},
    "crowd_density": {"min": 0.0, "max": 0.5},
    "exit_separation": {"min": 4, "max": 21},
    "passage_width": {"min": 1, "max": 12},
    "route_redundancy": {"min": 1, "max": 4},
    "ignition_person_distance": {"min": 2, "max": 108},
    "ignition_exit_distance": {"min": 3, "max": 108}
  }
}
```

| Key | Effect |
| --- | --- |
| `grid` | 5 to 128 cells per side. The border is wall; exits replace border cells. |
| `population`, `exit_count`, `ignition_count` | Drawn per scenario from the range. |
| `layout.partition_count` | Straight interior walls, vertical or horizontal, at least two cells apart. |
| `layout.doors_per_partition`, `layout.door_width` | Gaps in each partition: the route-redundancy and corridor-width controls. |
| `layout.scatter_density` | Share of the remaining free interior cells that become walls, in steps of 1/1000. |
| `fire` | V1 fire rules. `spread_probability_choices` is required for `stochastic` and forbidden for `deterministic`. |
| `movement.enter_fire` | `casualty` or `blocked`. Conflict resolution is always `lowest_id_wins`. |
| `max_steps` | The scenario's `termination.max_steps`. |
| `max_attempts` | Candidates per scenario before generation fails. |
| `constraints` | Accepted ranges of the measured proxies (section 6). |

The default example above was written for tests. It was not fitted or tuned to
the public suite, and generated scenarios are tested to differ from every
public scenario.

## 6. Generation constraints and proxies

Each candidate is built, then measured by `measure_proxies(scenario)`. The
values come from the scenario alone, so anyone can re-derive them. They are
**generation constraints and proxies**, not a universal measure of difficulty.
"Interior" means every cell that is not on the grid border.

| Proxy | Definition |
| --- | --- |
| Route existence (always required) | Every person has a route to an exit that avoids walls and ignition cells, of at most `max_steps` moves. |
| `route_length` | The longest of the people's shortest such routes. |
| `obstacle_density` | Interior wall cells / interior cells. |
| `crowd_density` | People / free interior cells. |
| `exit_separation` | Smallest Manhattan distance between two exits. Not checked with one exit. Always at least 2, so exits never touch. |
| `passage_width` (corridor width) | A cell's width is the shorter of its horizontal and vertical free runs. Each person takes the route whose narrowest cell is widest. The value is the minimum over people: the narrowest passage that some person cannot avoid. |
| `route_redundancy` | The minimum over people of the number of vertex-disjoint routes to the exit set on the static map, capped at 4. A value of 1 means one blocked cell can cut a person off from every exit. |
| `ignition_person_distance` | Shortest wall-free distance from any ignition to any person. |
| `ignition_exit_distance` | Shortest wall-free distance from any ignition to any exit. |
| `min_cut` | The maximum number of vertex-disjoint routes from the whole crowd to the exit set (every free cell has capacity 1). |
| `bottleneck_severity` | `population / min_cut`: the fewest steps the crowd needs to drain through its narrowest cut (`null` when `min_cut` is 0). |
| `max_nearest_exit_load` | The most people that share one route-nearest exit. |
| `split_pressure` | The busiest route-nearest exit's load / (1 + the median extra route to its people's second-nearest exit). `null` with one exit. |
| `deceptive_share` | Share of people whose Manhattan-nearest exit is a dead end or at least 2 steps longer by route than their route-nearest exit. |
| `exit_gap_median` | Median over people of (second - first) route distance to an exit. `null` with one exit. |
| `detour_ratio` | Mean over people of route / Manhattan distance to the route-nearest exit. |
| `ignition_centrality` | 0 for an ignition at the grid centre, 1 on the border (the most central ignition counts). |

The proxies from `min_cut` on were added in generator 1.1.0. They are the PR #33
family measurements, so a family rule and a generation constraint read the
same number.

**Config 1.1 constraints.** A `1.1` config may also set three optional ranges
in `constraints`: `split_pressure`, `deceptive_share` and
`bottleneck_severity` (floats; the ratios are capped at 1,000,000). A `1.0`
config with any of them is rejected. A positive `split_pressure` or
`deceptive_share` minimum needs `exit_count.max >= 2`. A `null` proxy meets no
`split_pressure` range (it reads as 0) and no `bottleneck_severity` range.
They let a recipe aim at one structural family by rejection sampling; they do
not label difficulty.

A distance of `null` (the fire can never reach a person or exit) counts as
infinite: it meets any minimum and breaks any maximum.

Exits, ignitions and people are placed constructively: a cell that would break
the `exit_separation`, `ignition_exit_distance` or `ignition_person_distance`
minimum is never drawn. Every proxy is then checked on the finished candidate.

### Failing closed

The generator never weakens a constraint.

- **Before any draw**, `GeneratorConfig` rejects malformed and impossible
  configs: a range with `min > max`, more people and ignitions than interior
  cells, a `route_length` above `max_steps`, a `route_redundancy` above 4 or
  above `exit_count.max` (each border exit has one approach cell), doors that
  do not fit in a partition, more partitions than fit, and bad fire settings.
- **During generation**, if no candidate for a scenario passes within
  `max_attempts`, `generate_suite` raises `GenerationError`. The message gives
  the rejection counts per constraint. This is the only place where rejection
  history appears. It is for the evaluator and is never written to disk.

## 7. Leakage prevention

A decision model receives only `Observation` values (`docs/v1-fairness-spec.md`).
For generated suites this means:

- The generator seed is never in an observation, a scenario file, the suite
  manifest, the freeze record, an evaluation manifest or a result artifact.
  The tests check this byte by byte.
- The config hash, batch ID, scenario ID, content hash, proxies and family
  labels are evaluator-side. The observation contract does not include them.
- The generator code and config format are public. Knowing them does not
  reveal the scenarios: without the seed, the next suite cannot be predicted.
- **Do not publish a scored generator seed** before the evaluation is over.
  This repository and its tests contain only test seeds. Keep the frozen
  directory, like the evaluation seeds, outside the repository and out of
  reach of the models under evaluation.
- Generate a new batch (a new seed and batch ID) for each scored evaluation
  round. A suite whose scenarios were shown to model developers is no longer
  unseen.
