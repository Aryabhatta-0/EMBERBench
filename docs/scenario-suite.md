# Canonical V1 public scenario suite

| Field | Value |
| --- | --- |
| Suite ID | `emberbench-v1-public` |
| Suite version | `1.0.0` |
| Visibility | `public` (development and diagnostics) |
| Scenario schema | `1.0` |
| Location | [`scenarios/public/v1/`](../scenarios/public/v1/) |
| Manifest | [`scenarios/public/v1/manifest.json`](../scenarios/public/v1/manifest.json) |
| Suite hash | `5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b` |

This suite is the fixed, public set of V1 diagnostic scenarios. Each scenario
isolates one challenge, so a failure points to one missing capability. The
suite uses only the frozen V1 rules in
[simulation-contract.md](simulation-contract.md) and
[v1-fairness-spec.md](v1-fairness-spec.md).

The suite is **not** the hidden evaluation set. It contains no hidden data, no
evaluator seed schedule and no paired seed manifest. A model can be developed
against it in the open. Scored evaluation must also use scenarios that are not
in this repository.

## Seeds

Every scenario file has a `seed`, because the schema requires one. In this
suite every seed is `0`, and the manifest declares `"seed_role": "development"`.

- The scenario seed is a **development default**. It makes a local run
  reproducible. It is **not** a benchmark seed.
- Scored seed schedules belong to the **evaluator**. The evaluator passes them
  with `Simulator(scenario, seed=...)`. They are never stored here.
- Paired seed manifests for matched comparison belong to **Issue #6**.
- For the nine deterministic scenarios the seed has no effect on the run. For
  the stochastic scenario (`v1-pub-10-route-invalidation`) it selects the fire
  trajectory.

A model never receives any seed (see `v1-fairness-spec.md`, section 1.3).

## Manifest

`manifest.json` is canonical JSON with a two-space indent and a final newline.
`emberbench.load_suite(directory)` reads it strictly: unknown or missing keys
are rejected. It then loads every scenario file with the strict schema parser
and checks every pinned field against the file.

| Key | Meaning |
| --- | --- |
| `suite_id`, `suite_version` | Suite identity. `suite_version` is `MAJOR.MINOR.PATCH`. |
| `visibility` | Always `public`. |
| `seed_role` | Always `development`. See [Seeds](#seeds). |
| `schema_version` | Scenario schema of every entry. |
| `scenarios[]` | One entry per scenario, sorted by `scenario_id`. |

Each entry has:

| Key | Meaning | Checked against the file |
| --- | --- | --- |
| `scenario_id` | Stable ID. Unique. | yes |
| `path` | File name, relative to the manifest. Must be `<scenario_id>.json`. | file must exist |
| `schema_version` | Scenario schema version. | yes |
| `content_hash` | `Scenario.content_hash()`: SHA-256 of the canonical scenario JSON. Unique. | yes |
| `primary_family` | The main challenge. It is also in `families`. | — |
| `families` | Every family the scenario covers, sorted. | — |
| `population` | Number of people. | yes |
| `exit_count` | Number of exits. | yes |
| `ignition_count` | Number of initial fire cells. | yes |
| `fire_model` | `deterministic` or `stochastic`. | yes |
| `purpose` | One-line diagnostic purpose. | — |

The suite directory must contain only `manifest.json` and the files that it
lists. The suite hash is the SHA-256 of the canonical manifest JSON. It pins
every scenario hash and all metadata at once.

## Design rules

Every scenario in the suite follows these rules. `tests/test_suite.py` checks
each of them.

- It parses under the strict schema V1 parser.
- It states every configurable rule explicitly: `fire.model`,
  `fire.spread_interval`, `fire.exits_flammable`, `fire.spread_probability`
  (stochastic only), `movement.conflict_resolution`, `movement.enter_fire` and
  `termination.max_steps`. Every scenario uses `enter_fire: "casualty"`, the V1
  default.
- The file is canonical: it is byte-identical to
  `json.dumps(scenario.to_dict(), indent=2) + "\n"`, and so a load and save
  round trip changes nothing.
- No person starts on a wall, an exit or a fire cell. No two people, exits or
  ignitions share an ID or a position.
- **Neutral person IDs.** IDs are `P001` to `Pnnn`. The number is the rank of
  `sha256("emberbench/v1/public/person-id/{scenario_id}/{row}/{col}")`. The
  ID order is therefore not row-major, and it does not follow the distance to
  an exit. This matters because the collision rule (R1) uses the ID order.
  Exit IDs are `E1`, `E2`, ... in row-major order of the exit cells. Exit
  positions are visible to the model anyway.
- **No person is doomed by construction.** In every deterministic scenario,
  each person alone has a fire-free route to some exit in time. Interaction
  with other people can still cause losses. That is part of the challenge.
- A scenario adds only the complexity that its challenge needs.

This document does not give an optimal action sequence for any scenario, and
the tests do not name a winning policy.

## Family coverage

There are twelve required families and ten scenarios. Two scenarios each
cover two families. In both cases, the second family is a direct consequence
of the first challenge, not an extra one.

| Family | Scenario |
| --- | --- |
| `single_exit_low_density` | `v1-pub-01-single-exit-sparse` |
| `single_exit_high_density` | `v1-pub-02-single-exit-dense` |
| `narrow_corridor_bottleneck` | `v1-pub-03-corridor-bottleneck` |
| `multiple_exits_unequal_distance` | `v1-pub-04-unequal-exits` |
| `central_ignition` | `v1-pub-05-central-ignition` |
| `threatened_nearest_exit` | `v1-pub-06-threatened-exit` |
| `multiple_ignitions` | `v1-pub-07-multiple-ignitions` |
| `maze_obstacle_routing` | `v1-pub-08-maze-deceptive-exit` (primary) |
| `deceptive_nearest_exit` | `v1-pub-08-maze-deceptive-exit` |
| `crowd_100` | `v1-pub-09-crowd-100-split-flow` (primary) |
| `split_flow_across_exits` | `v1-pub-09-crowd-100-split-flow` |
| `route_invalidation_replanning` | `v1-pub-10-route-invalidation` |

- **Maze and deceptive exit.** In V1 an exit can only "deceive" through
  geometry: the observation shows the full map and the fire. A deceptive exit
  is therefore one that is near in straight-line distance but far by path.
  That needs walls that force detours, which is the maze challenge. One
  scenario tests both: plan with path distance, not straight-line distance.
- **100-person crowd and split flow.** Each exit cell passes one person per
  step (R4). With 100 people, the queue at one exit is longer than the walk to
  another exit, so flow allocation is the main difficulty at that scale. A
  100-person crowd with one exit would only measure a 100-step queue.

## Scenarios

Hazard notation: `det/k` is the deterministic model with
`spread_interval = k`. `sto/k p` is the stochastic model with
`spread_interval = k` and `spread_probability = p`. "Flammable exits" is
`fire.exits_flammable`.

| Scenario ID | Family | Pop. | Exits | Hazard | Intended challenge | Why it exists | Capability it diagnoses |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `v1-pub-01-single-exit-sparse` | single exit, low density | 4 | 1 | 1 ignition, det/3, flammable exits | Walk each person to the only exit before a slow fire arrives. | Control case. A model that fails here has a basic action or navigation fault. | Basic movement, valid actions, shortest-path navigation. |
| `v1-pub-02-single-exit-dense` | single exit, high density | 36 | 1 | 1 ignition behind the crowd, det/4, flammable exits | Drain a packed room through one exit. More people than the longest walk. | Separates congestion handling from distance. | Queue ordering, collision avoidance, use of the one-person-per-step exit rate. |
| `v1-pub-03-corridor-bottleneck` | narrow corridor bottleneck | 8 | 1 | 1 ignition at the back of the room, det/2, flammable exits | Feed eight people into a one-cell-wide corridor before fire reaches its entrance. | Every route shares the same cells, so ordering decides the outcome. | Sequencing through restricted geometry under time pressure. |
| `v1-pub-04-unequal-exits` | multiple exits, unequal distance | 5 | 2 | 1 ignition, det/3, flammable exits | Choose between two exits whose path distances differ by at least 4 for each person. | Simplest multi-exit decision, with no queue or fire trap. | Exit selection by path distance. |
| `v1-pub-05-central-ignition` | central ignition | 8 | 2 | 1 ignition at the grid centre, det/2, flammable exits | Move away from a front that grows in every direction and go around it. | The front grows toward people on every side at once. | Hazard-aware routing around an expanding front. |
| `v1-pub-06-threatened-exit` | threatened / lost nearest exit | 6 | 2 | 1 ignition beside the nearest exit, det/2, flammable exits | Judge who can still use the nearest exit before it burns, and send the others to the far exit. | The nearest exit is a trap for some people and safe for others. | Prediction of exit loss from known fire rules; commitment versus switching. |
| `v1-pub-07-multiple-ignitions` | multiple ignitions | 9 | 2 | 3 separate ignitions, det/3, flammable exits | Pass between three growing fronts before they merge. | One fire front is not enough to test multi-source threat reasoning. | Reasoning about several hazard sources at once. |
| `v1-pub-08-maze-deceptive-exit` | maze / obstacle routing; deceptive nearest exit | 5 | 2 | 1 ignition in the outer ring, det/3, flammable exits | Route through a serpentine maze. The straight-line-nearest exit is far by path. | Separates path planning from straight-line heuristics. | Path distance around walls; resistance to geometric heuristics. |
| `v1-pub-09-crowd-100-split-flow` | 100-person crowd; split flow | 100 | 3 | 1 ignition, det/4, flammable exits | Evacuate 100 people. Most are nearest to one exit, and its queue is longer than the walk to another exit. | Scale test, and the only scenario where flow allocation dominates. | Coordination at scale; load balancing across exits. |
| `v1-pub-10-route-invalidation` | route invalidation requiring replanning | 6 | 1 | 1 ignition in a side shaft, sto/1 p=0.5, non-flammable exits | Use or abandon a short corridor that stochastic fire may cut at an unknown time. A longer route stays open. | The short route can fail after people commit to it, at a seed-dependent step. | Replanning when the current plan becomes invalid; risk under stochastic spread. |

Each layout below is the static map at step 0: `#` wall, `E` exit, `F` fire
(ignition), `P` person, `.` floor. A test checks that every layout here
matches its scenario file.

### v1-pub-01-single-exit-sparse

Hash `d70b5c78b24e3020ebd7c3b3ddd7cd6ccc5ec6232c051cd2aaea2d007b84e59e`. 14 × 9, `max_steps` 60.

```text
##############
#............#
#..P.........#
#F.......P...#
#............E
#.....P......#
#..........P.#
#............#
##############
```

### v1-pub-02-single-exit-dense

Hash `8fc36c0b598f123e59cfd102feb2ed48276b3ee9da78e62f72853105b2bd7042`. 12 × 9, `max_steps` 100.

```text
############
#.PPPPPP...#
#.PPPPPP...#
#.PPPPPP...#
#.PPPPPP...E
#.PPPPPP...#
#.PPPPPP...#
#F.........#
############
```

### v1-pub-03-corridor-bottleneck

Hash `0690eff35dba913fcdccef5d44eca0b360d4ece5b70be0e1616db61cc95e1b9a`. 15 × 7, `max_steps` 60.

```text
###############
#......########
#.P.P.P########
#F.P.P........E
#.P.P.P########
#......########
###############
```

### v1-pub-04-unequal-exits

Hash `f5b630f390b69cf758df56907037e032455ab0dff06f2a615bdaead2ec50b64e`. 20 × 9, `max_steps` 60.

```text
####################
#..................#
#...P..........P...#
E.......P..........#
#..................#
#.....P.......P....#
#..................E
#.........F........#
####################
```

### v1-pub-05-central-ignition

Hash `e1077d245b8c395403a1836578227479e2a32248a496fd301e68275186d0b937`. 15 × 11, `max_steps` 40.

```text
###############
#.............#
#......P......#
#....P...P....#
#.............#
E..P...F...P..E
#.............#
#....P...P....#
#......P......#
#.............#
###############
```

### v1-pub-06-threatened-exit

Hash `413bca35898f45a9b04446d513342e48d7389dff344520a41184531b593c3d34`. 17 × 9, `max_steps` 50.

```text
#################
#...............#
#F...P..........#
#...P...........#
E..P...P........E
#..P............#
#.........P.....#
#...............#
#################
```

### v1-pub-07-multiple-ignitions

Hash `cab42b2e0ab0324575b55fd5feb17070e4b73b7002b2375e156beedc6ce4c954`. 19 × 11, `max_steps` 50.

```text
###################
#.................#
#..F.....P.....F..#
#.....P.......P...#
#...P.............#
E........F........E
#.......P...P.....#
#..P..............#
#........P.....P..#
#.................#
###################
```

### v1-pub-08-maze-deceptive-exit

Hash `6aaf5079a9833ee862b3b07d7c300ef54b3bccba6e9f4f73e97e779d27a53445`. 19 × 11, `max_steps` 120.

```text
###################
#.................#
#.###############.#
#.#.P.#......#....E
#.#...#.P.#..#..#.#
E.#P..#...#..#..#.#
#.#..P#.P.#..#..#.#
#.#.......#.....#.#
#.###############.#
#........F........#
###################
```

### v1-pub-09-crowd-100-split-flow

Hash `4cae3e2f63610053571e764305adec41f0820dc7ebd0dc658fc1760bbecf9964`. 26 × 16, `max_steps` 200.

```text
##################E#######
#........................#
#........................#
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
E..PPPPPPPPPP............#
#..PPPPPPPPPP............E
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
#..PPPPPPPPPP............#
#........................#
#............F...........#
##########################
```

### v1-pub-10-route-invalidation

Hash `ee407099e5a8900e0d1d0fd7987e375aefda31d74bf31f74d6040e2f50e0ebef`. 19 × 11, `max_steps` 80.

```text
##########F########
#.....####.########
#..P..####.########
#.P...####.########
#...P.............E
#..P..############.
#.P...############.
#...P.############.
#.....############.
#..................
###################
```

## Fixtures are not canonical scenarios

[`tests/fixtures/scenarios/`](../tests/fixtures/scenarios/) holds small **test
fixtures** for simulator behaviour. Their descriptions start with
`TEST FIXTURE.`. They are not in this suite, and the tests check both
directions: no canonical scenario is under `tests/fixtures`, and no fixture
(by ID or by content hash) is in the manifest. Canonical descriptions start
with `CANONICAL V1 PUBLIC SCENARIO.`

## Change control

The suite is frozen at version `1.0.0`. `tests/test_suite.py` pins the suite
hash and every scenario hash.

- Any change to a scenario file changes its content hash. It needs a new
  `suite_version`, a new manifest entry and updated pins. Do not edit a
  scenario in place under the same version.
- A rule change that changes simulator output bumps `SIMULATOR_VERSION`. It
  does not change scenario hashes, because a scenario hash covers only the
  scenario inputs.
- A change to the scenario file format bumps `SCHEMA_VERSION` and needs a new
  suite.
