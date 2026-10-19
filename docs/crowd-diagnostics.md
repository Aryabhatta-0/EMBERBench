# Crowd coordination diagnostics (crowd diagnostics version 1.0)

This document defines every field returned by
`emberbench.compute_crowd_diagnostics(scenario, record) -> CrowdDiagnostics`.
The code is in `src/emberbench/crowd.py`. The tests are in
`tests/test_crowd.py`.

The diagnostics make congestion and bottlenecks measurable. They **observe**
a recorded episode. They do not change the V1 movement rules, and they do
not model human behaviour. There is no crowd score.

The document has four parts, kept apart on purpose:

1. [Simulator crowd rules](#1-simulator-crowd-rules): the frozen V1 rules
   that produce the data.
2. [Raw observed signals](#3-raw-observed-signals): counts read from the
   recorded movements and events.
3. [Proxies](#4-proxy-the-static-nearest-exit-demand): values that use an
   assumption, named as proxies.
4. [What EMBERBench does not model](#7-what-emberbench-does-not-model).

## 1. Simulator crowd rules

These rules are frozen in V1. They are defined in
[simulation-contract.md](simulation-contract.md) and
[v1-fairness-spec.md](v1-fairness-spec.md). They are repeated here only so
that the signals below make sense. Phase 7 changes none of them.

- People are homogeneous. Each person occupies one cell and moves at most
  one cell (4-neighbour) per step.
- One person per cell after every step.
- A move into a cell held by a person who does not move gives
  `blocked_occupied`. A chain that ends at such a person is also
  `blocked_occupied` all the way back.
- When several movers target the same cell, the lowest person ID (string
  order) wins. The others get `blocked_conflict`.
- A closed chain (a swap or a longer rotation) is never allowed. Every
  member gets `blocked_cycle`.
- A person may follow another person into the cell that person leaves in
  the same step, so a queue advances without gaps.
- An exit is one cell. A person who ends a step on a non-burning exit
  evacuates. So one exit cell evacuates at most one person per step, and
  `k` distinct exit cells evacuate at most `k` people per step.

The six blocked reasons are canonical: `boundary`, `wall` and `fire` (static
blocks, from the map or the fire), and `occupied`, `conflict` and `cycle`
(congestion blocks, from other people).

## 2. Inputs, validation and versions

| Item | Value |
| --- | --- |
| Inputs | a `Scenario` and a complete `RunRecord` of that scenario |
| Validation | `emberbench.validate_run_record`, the same three stages as the metrics: compatibility, structural checks, and full simulator replay (`verify_record`) |
| Failure | `MetricsInputError`; no partial result |
| Result | an immutable (frozen) `CrowdDiagnostics` |
| Artifact version | `CROWD_DIAGNOSTICS_VERSION = "1.0"`, separate from `METRICS_VERSION` |
| Binding | the result holds the simulator, observation and schema versions, `scenario_id`, `scenario_hash`, `seed` and `record_digest` (`RunRecord.digest()`) |

A record whose block reasons were relabelled (for example `blocked_conflict`
changed to `blocked_occupied`) still passes the structural checks. The
replay stage rejects it. So every count below is a count of what the
simulator really did.

## 3. Raw observed signals

### 3.1 Contenders

A **contending move** is a movement attempt (an action that is not `stay`)
whose outcome is not `blocked_boundary`, `blocked_wall` or
`blocked_fire`. It passed the static checks, so it ends as `moved`,
`blocked_occupied`, `blocked_conflict` or `blocked_cycle`.

The **contenders for a cell `c` at step `t`** are the contending moves of
step `t` whose recorded target is `c`. A cell is **contested** at step `t`
when it has two or more contenders. Two movers that target a cell held by
a person who stays are contenders too: they both want the same cell.

A swap is not contention: each member targets a different cell.

### 3.2 Step signals (`steps`)

One entry per recorded step, in step order. The counts cover the people
active when the step was decided.

| Field | Definition |
| --- | --- |
| `step` | the step number `t` |
| `active_people` | decisions in the step (one per active person) |
| `stays` | outcome `stayed` |
| `movement_attempts` | `active_people - stays` |
| `successful_moves` | outcome `moved` |
| `blocked_moves` | outcome `blocked_*` |
| `blocked_by_outcome` | count per blocked outcome, in `MoveOutcome` order |
| `congestion_blocks` | `blocked_occupied + blocked_conflict + blocked_cycle` |
| `contending_moves` | `movement_attempts - (blocked_boundary + blocked_wall + blocked_fire)` |
| `contested_cells` | cells with two or more contenders in this step |
| `max_contenders` | the largest contender count of one cell in this step; `0` if there are no contending moves |
| `evacuations_by_exit` | evacuations in this step, per exit (every exit, sorted by exit ID) |
| `cumulative_evacuations_by_exit` | evacuations up to and including this step, per exit |
| `nearest_exit_demand_proxy` | see [section 4](#4-proxy-the-static-nearest-exit-demand) |

Checked: `successful_moves + blocked_moves = movement_attempts`.

### 3.3 Spatial congestion hotspots (`cells`)

One entry per cell that had at least one contender in the episode. The
entries are in canonical **row-major** order (`(row, col)` ascending).
Static blocks are not counted here, so wall and boundary cells never appear.

| Field | Definition |
| --- | --- |
| `cell` | `[row, col]` |
| `exit_id` | the exit on this cell, or `null` |
| `static_exit_distance` | wall-only path distance from this cell to the nearest exit (section 4), or `null` |
| `attempted_entries` | contenders for this cell, summed over all steps |
| `successful_entries` | contenders with outcome `moved` |
| `blocked_occupied` | contenders with outcome `blocked_occupied` |
| `blocked_conflict` | contenders with outcome `blocked_conflict` |
| `blocked_cycle` | contenders with outcome `blocked_cycle` |
| `congestion_blocks` | `blocked_occupied + blocked_conflict + blocked_cycle` |
| `contention_steps` | steps in which the cell had two or more contenders |
| `peak_contenders` | the largest contender count of this cell in one step |

Checked: `attempted_entries = successful_entries + congestion_blocks`, and
at most one contender enters a cell per step.

A **hotspot** is a cell with `congestion_blocks > 0`. The API returns the
raw table. It does not rank cells. To find the worst hotspots, sort by
`congestion_blocks` (or by `contention_steps`) and break ties by cell.

Hotspots are **observed runtime hotspots** of one episode under one
decision model. They are not claims about where real crowds congest.

### 3.4 Exit throughput (`exits`)

Per exit (`exits.exits`, sorted by exit ID):

| Field | Definition |
| --- | --- |
| `exit_id`, `position` | the exit |
| `evacuations` | people who evacuated through this exit |
| `peak_step_evacuations` | most evacuations through this exit in one step; at most `1` under V1 |
| `steps_with_evacuation` | steps with at least one evacuation here; equals `evacuations` under V1 |
| `first_evacuation_step`, `last_evacuation_step` | or `null` if unused |
| `initial_demand_proxy` | see section 4 |
| `demand_proxy_person_steps` | see section 4 |

Episode level: `exit_count`, `total_evacuations`, and
`peak_step_evacuations` (most evacuations over all exits in one step).

Cumulative throughput per step is in `steps[t].cumulative_evacuations_by_exit`.

### 3.5 Episode totals (`totals`)

| Field | Definition |
| --- | --- |
| `decisions`, `stays`, `movement_attempts`, `successful_moves`, `blocked_moves`, `blocked_by_outcome`, `congestion_blocks`, `contending_moves` | sums of the step fields |
| `contested_cell_steps` | sum of `contested_cells` over the steps |
| `steps_with_contention` | steps with `contested_cells > 0` |
| `steps_with_congestion_blocks` | steps with `congestion_blocks > 0` |
| `peak_contested_cells`, `peak_contenders`, `peak_congestion_blocks` | maxima over the steps |
| `hotspot_cells` | cells with `congestion_blocks > 0` |

## 4. Proxy: the static nearest-exit demand

The **nearest-exit demand proxy** counts active people by the exit that is
nearest to them on the static map. It is a **proxy**. It does not observe
which exit a decision model intends to use, and it does not say which exit
a person should use.

Distance is the **static wall-only path distance**: the length of the
shortest 4-neighbour path from the cell to the exit that does not cross a
wall or leave the grid. People, fire (also the initial ignitions) and the
other exits are ignored. It is computed once per exit by breadth-first
search. It is not Manhattan distance: a wall can make a close exit far.

For each active person:

- if one exit is strictly nearer than every other exit, the person counts
  for that exit;
- if two or more exits share the smallest distance, the person counts as
  `tied`;
- if no exit is reachable around the walls, the person counts as
  `unreachable`.

So `sum(per_exit) + tied + unreachable = active people`.

| Field | Definition |
| --- | --- |
| `steps[t].nearest_exit_demand_proxy` | `per_exit`, `tied`, `unreachable` for the people active when step `t` was decided |
| `exits.exits[i].initial_demand_proxy` | the per-exit count at step 0 (the whole population) |
| `exits.exits[i].demand_proxy_person_steps` | the per-exit count summed over all steps |
| `exits.initial_demand_tied`, `exits.initial_demand_unreachable` | the step-0 counts |
| `exits.demand_tied_person_steps`, `exits.demand_unreachable_person_steps` | summed over all steps |

Comparing the proxy with the measured `evacuations` shows how the actual
load differs from the static nearest-exit load. A difference is not an
error. Fire, congestion and queue length can make another exit the better
choice. **Equal load across exits is not assumed to be optimal**, and
nothing here rewards it.

## 5. Determinism and serialisation

- Every value is an integer, a cell or `null`. There is no floating point.
- Steps are in step order, cells in row-major order, exits in exit-ID
  order. `to_json()` is canonical JSON (sorted keys, no spaces), and
  `digest()` is its SHA-256.
- The result does not depend on the insertion order of the action
  mappings, on `PYTHONHASHSEED`, or on the platform. The tests check
  repeated runs, shuffled and reversed action maps, three hash seeds in
  subprocesses, and pinned golden digests (run on Linux, Windows and macOS
  in CI).

## 6. Tested queue and throughput invariants

`tests/test_crowd.py` pins these frozen V1 rules through the diagnostics:

- one exit cell evacuates at most one person per step;
- two distinct exit cells can evacuate two people in one step;
- a following chain advances without gaps;
- same-target losers are `blocked_conflict`;
- movers into a person who does not move are `blocked_occupied`;
- swaps and rotations are `blocked_cycle`;
- step signals reconcile exactly with the recorded movements (an
  independent recomputation);
- episode totals reconcile exactly with `MovementMetrics` and
  `ExitUtilization` from `compute_episode_metrics`.

Large-population stress tests run 50, 100 and 200 people in a synthetic
room (`crowd_stress_scenario` in `tests/conftest.py`). They check movement
resolution (order independence, one person per cell), simulator execution,
replay authentication and the diagnostics. They use bounded step counts and
exact pinned results, not wall-clock limits. For manual timing, run
`python tools/profile_crowd.py`; it is not a CI gate.

## 7. What EMBERBench does not model

The simulator is a discrete grid with homogeneous agents. It does **not**
model, and the diagnostics do not measure:

- panic, stress or any other emotional state;
- pushing force, crowd pressure or crush;
- body size, shape or the space a person takes beyond one cell;
- group behaviour (families, following a leader, herding);
- social dynamics (helping, yielding, competition, norms);
- heterogeneous speeds, mobility or reaction times.

The signals describe how one decision model's actions interact under the V1
rules. They are useful to compare models on the same episode. They are not
evidence about real human evacuations.
