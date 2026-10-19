# Per-episode metrics (metrics version 1.0)

This document defines every field returned by
`emberbench.compute_episode_metrics(scenario, record) -> EpisodeMetrics`.
The code is in `src/emberbench/metrics.py`. The exact-value tests are in
`tests/test_metrics.py`.

The metrics describe **one episode**: one scenario, one seed, one action
log. Aggregation across seeds or scenarios, robustness statistics and
leaderboards are out of scope here (see Issue #12).

## 1. Principles

1. **Raw metrics are canonical.** Every field is a count, a step number, a
   cell distance or a ratio of named counts. Each ratio states its
   numerator and denominator.
2. **No composite score.** There is no single EMBER score in this version.
   See [section 11](#11-rules-for-any-future-composite-or-ranking).
3. **Safety first.** Speed fields that only cover evacuees always come with
   the counts of people who did not evacuate. The person-time-to-safety
   metric counts every person.
4. **Proxies are named as proxies.** The fire-exposure and replanning fields
   describe observable events. They do not measure risk or model intent.
5. **Fail closed.** Inputs that do not belong together, a malformed record,
   or a record that the simulator could not have produced raise
   `MetricsInputError`. No partial result is returned.
6. **Exact arithmetic.** Every ratio is computed as an exact
   `fractions.Fraction` and converted to `float` once. A value is the
   correctly rounded float of an exact rational, on every platform.
7. **Pure function.** The scenario and the record are not mutated.
   `METRICS_VERSION` changes whenever a definition in this document changes.

## 2. Inputs and validation

| Source | Used for |
| --- | --- |
| `Scenario` | population and person IDs, initial positions, walls, grid size, exit IDs and positions, `termination.max_steps` |
| `RunRecord` header | versions, `scenario_id`, `scenario_hash`, `seed` |
| `RunRecord.initial_state` | snapshot `s_0` (positions and initial burning cells) |
| `RunRecord.steps[t]` | actions, movement outcomes, ignitions, evacuations, casualties and snapshot `s_t` |
| `RunRecord.final_state` | terminal status of each person, terminal trapped set, termination reason |

`compute_episode_metrics` validates its inputs in three stages, in this
order. All stages always run; there is no option to skip one. It raises
`MetricsInputError` when any check fails.

**Stage 1 and 2: compatibility and structural validation.**

- **Compatibility.** `simulator_version` equals `SIMULATOR_VERSION`,
  `observation_version` equals `OBSERVATION_VERSION`, and `schema_version`
  is supported and equals the scenario's. `scenario_id` equals the
  scenario's. `scenario_hash` equals `scenario.content_hash()`.
- **Initial state.** Step 0, every scenario person active at the scenario
  position, burning cells equal to the ignitions, not terminated, and a
  trapped set equal to a recomputation.
- **Each step `t`.** Step numbers are `1, 2, ..., T`. Actions and movements
  cover exactly the people active in `s_{t-1}`. Each movement agrees with
  its action, origin and outcome; a `moved` target is in bounds and not a
  wall; no two people end in one cell. Ignitions are new, non-wall cells,
  and `burning(s_t) = burning(s_{t-1}) ∪ ignited_t`. Every evacuation and
  casualty event matches one status change (with the correct exit ID and
  casualty cause), resolved people never change, no active person stands on
  a burning cell or a usable exit, and the trapped set equals a
  recomputation.
- **Termination.** Only the last snapshot is terminated. Its reason is
  `all_resolved` when nobody is active, otherwise `max_steps` when
  `T = max_steps`. A record with no steps, or one that does not reach
  termination, is rejected: metrics need a complete episode.

These checks give precise diagnostics and confirm every assumption the
metrics rely on. On their own they only prove internal consistency.

**Stage 3: simulator replay authentication.** `compute_episode_metrics`
then calls `verify_record(scenario, record)`. It replays the recorded action
log from the recorded seed and requires a byte-identical record. A
`ReplayMismatchError` becomes a `MetricsInputError` whose message starts
with `record failed simulator replay verification:`; the original error is
kept as `__cause__`. This stage rejects records that pass every structural
check but are impossible under the simulator, for example a `blocked_wall`
outcome for a move into an open cell, or an ignition that the configured
fire model and RNG stream would not produce. Callers do not need to call
`verify_record` themselves.

## 3. Notation

- `N` = population = number of people in the scenario (always `>= 1`).
- `T` = `episode_steps` = number of recorded steps.
- `H` = `scenario.termination.max_steps` (the horizon).
- `s_0` = initial snapshot; `s_t` = snapshot after step `t`, for `t = 1..T`.
- `A(s)` = people with status `active` in snapshot `s`.
- A **decision** is one movement record: one active person at one step.
  The number of decisions is `D = Σ_{t=1..T} |A(s_{t-1})|`.
- `ratio(a, b)` = `a / b`, or `null` (`None`) when `b = 0`.

## 4. Safety (`safety`)

| Field | Definition | Unit |
| --- | --- | --- |
| `population` | `N` | people |
| `evacuated` | people with terminal status `evacuated` | people |
| `casualties` | people with terminal status `casualty` | people |
| `stranded` | people still `active` at termination | people |
| `evacuated_rate` | `evacuated / N` | fraction |
| `casualty_rate` | `casualties / N` | fraction |
| `stranded_rate` | `stranded / N` | fraction |
| `terminal_trapped_count` | `|final_state.trapped|`: stranded people with no wall-free, fire-free path to a non-burning exit (rule R9) | people |
| `terminal_trapped_rate` | `terminal_trapped_count / N` | fraction |
| `casualties_by_cause` | casualties per `CasualtyCause` (`entered_fire`, `fire_spread`) | people |

Invariants: `evacuated + casualties + stranded = N` exactly (checked; a
mismatch raises), and `terminal_trapped_count <= stranded`. When the episode
ends with `all_resolved`, `stranded = 0` and the trapped set is empty.

## 5. Time (`time`)

| Field | Definition | Unit |
| --- | --- | --- |
| `episode_steps` | `T` | steps |
| `max_steps` | `H` (from the scenario) | steps |
| `evacuees` | `E` = number of evacuated people | people |
| `non_evacuees` | `N - E` (casualties and stranded) | people |
| `evacuation_steps` | for each evacuee, the step `t` whose `evacuations` list names it; sorted by person ID | steps |
| `first_evacuation_step` | `min` of `evacuation_steps`; `null` if `E = 0` | step |
| `last_evacuation_step` | `max` of `evacuation_steps`; `null` if `E = 0` | step |
| `mean_evacuation_step` | `ratio(Σ evacuation_steps, E)` | steps |
| `median_evacuation_step` | middle value; for even `E`, the mean of the two middle values; `null` if `E = 0` | steps |
| `person_time_to_safety_total` | `Σ_i τ_i` (see below) | person-steps |
| `mean_person_time_to_safety` | `person_time_to_safety_total / N` | steps |

**Person-time-to-safety.** For each person `i`:

```
τ_i = evacuation step of i     if i evacuated
τ_i = H                        otherwise (casualty or stranded)
```

Non-evacuees are censored at the horizon `H`. Every evacuation step is
`<= H`, so a casualty or a stranded person can never look faster than any
evacuee. The sum covers all `N` people, so this metric cannot hide a
casualty. The censoring value is the scenario horizon, not `T`: an episode
that ends early because the last people died does not get a smaller value.

`mean_evacuation_step` and `median_evacuation_step` are **evacuee-only**
statistics. They always come with `evacuees` and `non_evacuees`, and a run
must never be compared on them alone. A run that loses its slowest people
gets a lower evacuee mean.

## 6. Movement and congestion (`movement`)

Each decision has exactly one `MoveOutcome`: `stayed`, `moved`, or one of
the six `blocked_*` outcomes.

| Field | Definition | Denominator |
| --- | --- | --- |
| `decisions` | `D` | — |
| `stay_actions` | decisions with action `stay` (outcome `stayed`) | — |
| `movement_attempts` | `D - stay_actions` (action is a direction) | — |
| `successful_moves` | outcome `moved` | — |
| `blocked_moves` | outcome `blocked_*` | — |
| `blocked_by_outcome` | count per blocked outcome: `boundary`, `wall`, `fire`, `occupied`, `conflict`, `cycle` | — |
| `congestion_blocks` | `blocked_occupied + blocked_conflict + blocked_cycle` (blocks caused by other people) | — |
| `stay_rate` | `ratio(stay_actions, decisions)` | decisions |
| `blocked_rate` | `ratio(blocked_moves, movement_attempts)` | movement attempts |
| `congestion_block_rate` | `ratio(congestion_blocks, movement_attempts)` | movement attempts |

Invariant: `successful_moves + blocked_moves = movement_attempts` (checked).
Under `enter_fire: casualty`, a move into fire is `moved`; the person then
becomes a casualty in the same step.

## 7. Path efficiency (`path_efficiency` and per person)

For each person `i`:

- `cells_moved_i` = number of the person's decisions with outcome `moved`.
  Each such move changes the cell by exactly one 4-neighbour step.
- `static_exit_distance_i` = `L_i` = length of the shortest 4-neighbour
  path from the initial cell to **any** exit, through any non-wall cell in
  the grid. It ignores **all** fire (including the initial ignitions) and
  **all** people. It is `null` when no exit can be reached around the walls.

`L_i` uses only the scenario geometry. It is a **static lower bound**: it
is not a dynamically feasible path and **not** an oracle-optimal dynamic
path. Under spreading fire and congestion, the best achievable path can be
longer.

For each **evacuee** (`L_i` is always defined and `>= 1`, and
`cells_moved_i >= L_i`; a violation raises):

```
path_efficiency_i = L_i / cells_moved_i        in (0, 1]
detour_ratio_i    = cells_moved_i / L_i        in [1, ∞)
```

Both are `null` for casualties and stranded people. Episode aggregates, over
the `M` evacuees:

| Field | Definition |
| --- | --- |
| `evacuees_measured` | `M` |
| `total_cells_moved` | `Σ cells_moved_i` |
| `total_static_lower_bound` | `Σ L_i` |
| `aggregate_path_efficiency` | `ratio(Σ L_i, Σ cells_moved_i)` (ratio of sums) |
| `mean_path_efficiency` | `ratio(Σ path_efficiency_i, M)` |
| `mean_detour_ratio` | `ratio(Σ detour_ratio_i, M)` |

Path efficiency measures distance, not time. A blocked or `stay` step costs
time but no distance.

## 8. Fire-exposure proxy (`fire_exposure` and per person)

This is a deterministic **proximity proxy**, not a physical model of heat or
smoke. It has no tuned constants.

The person-states are all pairs `(t, i)` with `t = 0..T` and `i ∈ A(s_t)`:
every active person in every recorded snapshot, including `s_0` and the
terminal snapshot. Let `B_t = burning(s_t)`. For each person-state with
`B_t ≠ ∅`:

```
d(t, i) = min_{f ∈ B_t} ( |row_i - row_f| + |col_i - col_f| )
```

This is the Manhattan distance on the grid. It ignores walls: a wall does
not shield a person in this proxy. An active person is never on a burning
cell in a snapshot, so `d >= 1`. The exposure is:

```
exposure_total = Σ_{(t, i) : B_t ≠ ∅} 1 / d(t, i)
mean_exposure  = exposure_total / active_person_states
```

| Field | Definition |
| --- | --- |
| `active_person_states` | `Σ_{t=0..T} |A(s_t)|` (`>= N`, because `s_0` holds everyone) |
| `person_states_with_fire` | person-states with `B_t ≠ ∅` |
| `person_states_without_fire` | person-states with `B_t = ∅` (they add 0) |
| `distance_histogram` | `(d, count)` for each observed `d`, sorted by `d` |
| `adjacent_person_states` | count with `d = 1` |
| `min_fire_distance` | smallest observed `d`; `null` if no fire was ever present with someone active |
| `exposure_total` | as above (unit: person-states weighted by `1/d`) |
| `mean_exposure` | as above |

Per person: `min_fire_distance` and `fire_exposure` (the same sum, limited
to that person's states).

**Recomputing alternatives.** `distance_histogram` and
`person_states_without_fire` are the raw components. Any proxy of the form
`Σ g(d)` can be recomputed from them, for example a threshold count
(`Σ_{d <= k} count`) or another decay. `exposure_total` equals
`Σ count / d` over the histogram.

**Limitation.** People who leave the grid stop accumulating exposure. A
casualty therefore can have a low exposure. Always read the proxy together
with the safety counts.

## 9. Exit utilization (`exits`)

Let `n_e` be the number of evacuations through exit `e`, `E = Σ n_e`, and
`K` the number of exits in the scenario.

| Field | Definition |
| --- | --- |
| `exit_count` | `K` |
| `exits_used` | number of exits with `n_e >= 1` |
| `evacuations_per_exit` | `(exit_id, n_e)` for every exit, sorted by exit ID (zeros included) |
| `exit_shares` | `(exit_id, n_e / E)`; `null` when `E = 0` |
| `max_exit_share` | `max_e n_e / E`; `null` when `E = 0` |
| `exit_balance` | normalized Gini-Simpson index (below); `null` when `K = 1` or `E = 0` |

```
p_e          = n_e / E
exit_balance = (1 - Σ_e p_e²) / (1 - 1/K)        in [0, 1]
```

`exit_balance = 1` when every exit has the same share, and `0` when one
exit takes every evacuee. **This is descriptive only.** Equal use is often
not optimal: one exit can be much closer, or threatened by fire. Do not
reward or penalize a run for its balance.

## 10. Replanning and recovery proxies (`replanning`)

These fields count observable reactions to a blocked move. They are
**proxies**. They do not observe model intent, and a model may change its
action for reasons that have nothing to do with the block.

A **blocked event** is a decision with a `blocked_*` outcome. Its **next
decision** is the same person's decision at the next step. It exists only
when the person is still active and the episode has not terminated.

| Field | Definition |
| --- | --- |
| `blocked_events` | number of blocked events (equals `movement.blocked_moves`) |
| `blocks_with_next_decision` | `B'` = blocked events that have a next decision |
| `blocks_without_next_decision` | `blocked_events - B'` |
| `moved_after_block` | next decision has outcome `moved` (recovery proxy) |
| `blocked_after_block` | next decision is blocked again (repeated-block proxy) |
| `stayed_after_block` | next decision is `stay` |
| `action_changed_after_block` | next action differs from the blocked action (a change to `stay` counts) |
| `max_consecutive_blocks` | longest run of consecutive blocked decisions by one person |
| `recovery_proxy_rate` | `ratio(moved_after_block, B')` |
| `repeated_block_proxy_rate` | `ratio(blocked_after_block, B')` |
| `action_change_proxy_rate` | `ratio(action_changed_after_block, B')` |

Invariant: `moved_after_block + blocked_after_block + stayed_after_block = B'`.
Per person: `blocked_moves` and `max_consecutive_blocks`.

## 11. Rules for any future composite or ranking

This version deliberately has no single score. Any future ranking or
composite score MUST:

1. be **versioned**, separately from `METRICS_VERSION`, and state the exact
   raw fields and formula it uses;
2. be **decomposable**: every published composite value must ship with the
   raw metrics it was computed from;
3. **never let speed buy safety**: a faster run that is materially less safe
   (more casualties or stranded people) must never outrank a safer run
   because of speed alone. Safety outcomes take precedence, for example
   through a lexicographic order with safety first, not a weighted sum in
   which time can offset lives.

Cross-seed robustness and statistical aggregation are in [statistics.md](statistics.md).

## 12. Per-person fields (`people`)

One `PersonMetrics` per scenario person, sorted by person ID:

| Field | Definition |
| --- | --- |
| `person_id`, `initial_position`, `final_position` | from the scenario and `final_state` |
| `status` | terminal status |
| `resolution_step` | evacuation or casualty step; `null` if stranded |
| `exit_id` | exit used; `null` unless evacuated |
| `casualty_cause` | cause; `null` unless a casualty |
| `decisions`, `stay_actions`, `cells_moved`, `blocked_moves`, `max_consecutive_blocks` | section 6 and 10, for this person |
| `time_to_safety` | `τ_i` (section 5) |
| `static_exit_distance`, `path_efficiency`, `detour_ratio` | section 7 |
| `min_fire_distance`, `fire_exposure` | section 8 |

## 13. Edge cases

| Case | Behaviour |
| --- | --- |
| Nobody evacuates | evacuee statistics, path-efficiency ratios, exit shares and balance are `null`; `person_time_to_safety_total = N · H` |
| One exit | `exit_balance` is `null` |
| No fire ever | every person-state is without fire; `exposure_total = 0`, `min_fire_distance = null` |
| No movement attempts | `blocked_rate` and `congestion_block_rate` are `null` |
| No blocked event with a next decision | the three replanning rates are `null` |
| Person walled off from every exit | `static_exit_distance = null`; the person can never evacuate |
| Episode ends with `all_resolved` before `H` | non-evacuees are still censored at `H`, not `T` |
| Record not terminated, empty, or from another scenario or version | `MetricsInputError` |
| Record consistent but not reproduced by a replay | `MetricsInputError` (`record failed simulator replay verification: ...`) |

## 14. Serialisation

`EpisodeMetrics.to_dict()` returns plain JSON types. Pairs become objects
keyed by ID, outcome, cause or distance (as strings). `null` marks an
undefined ratio. No value is `NaN` or infinite, so the result encodes with
`emberbench.canonical.canonical_json`.
