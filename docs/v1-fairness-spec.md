# EMBERBench V1 fairness and semantics specification

This document is normative. It freezes the V1 fairness contract, the
model-visible observation and the V1 simulator rules. The words MUST, MUST NOT
and SHOULD have their RFC 2119 meaning.

| Contract | Version | Defined in |
| --- | --- | --- |
| Simulator semantics and run record format | `SIMULATOR_VERSION = "0.2.0"` | `src/emberbench/__init__.py`, [simulation-contract.md](simulation-contract.md) |
| Scenario schema | `SCHEMA_VERSION = "1.0"` | `src/emberbench/scenario.py`, [scenario-schema.md](scenario-schema.md) |
| Model-visible observation | `OBSERVATION_VERSION = "1.0"` | `src/emberbench/observation.py`, this document |

If the code and this document disagree, it is a bug.

**V1 in one sentence.** A model sees the full current state and knows the
episode dynamics (the rules). The evaluator's randomness (the seed, the RNG
state and every future draw) stays hidden. V1 evaluates decisions under known
rules, not inference of hidden rules.

## 1. Fairness contract

### 1.1 Matched evaluation

Every compared model MUST run on exactly the same:

- scenario (the same `Scenario.content_hash()`),
- simulator version (`SIMULATOR_VERSION`),
- initial population, with the same person IDs and positions,
- hazard configuration (ignitions, fire model and its parameters),
- evaluator seed schedule (the same list of seeds for each scenario),
- exogenous stochastic process (the fire RNG stream derived from each seed).

The only input that differs between two matched runs is the sequence of
actions that the models submit. A `RunRecord` header (`simulator_version`,
`observation_version`, `scenario_hash`, `seed`) identifies the matched episode
and the observation contract the model received.

### 1.2 Evaluator-private and model-visible state

The simulator state has two separate parts.

**Evaluator-private state.** Only the evaluator and the harness may read it.
A decision model MUST NOT receive any of it, in any form, before or during a run:

| Item | Where it lives |
| --- | --- |
| Evaluator seed, and `Scenario.seed` | `Simulator.seed`, `Scenario`, `RunRecord.seed` |
| RNG stream positions and stream seeds | `SimState.rng_state` |
| Future random draws | the fire stream (`SplitMix64`) |
| Scenario ID, description, content hash, schema version | `Scenario`, `RunRecord` header |
| `trapped` diagnostic | `SimState.trapped` |
| The full state, the step log and the run record | `SimState`, `StepRecord`, `RunRecord` |
| The `Scenario` and `Simulator` objects | evaluator side |

**Model-visible state.** A decision model receives only the `Observation`
returned by `Simulator.observe()`. Section 4 defines its fields. It has the
full current state (map, people, fire) and the episode rules (`RulesView`):
every action-relevant rule that a scenario can vary. The rules are copied
from the scenario, so the model never holds a `Scenario` reference.

The harness boundary is `Simulator.observe()` and `Simulator.step(actions)`.
A harness MUST give a model only `Observation` objects (or their
`to_dict()` form). It MUST NOT give a model the `Simulator`, the `Scenario`, a
`SimState`, a `StepRecord` or a `RunRecord` until the evaluation is complete.
`run_episode(scenario, policy)` obeys this rule: the policy receives only
`Observation` objects.

### 1.3 The seed

The seed exists for reproducibility and for matched evaluation only. It is
not a model input.

- The evaluator owns the seed schedule. The schedule MUST NOT be disclosed to
  a model before the evaluation is complete.
- Scored evaluations SHOULD draw seeds uniformly from `[0, 2**64)` with a
  cryptographic generator. A model that knows the simulator code could test
  small seeds (0, 1, 2, ...) against the fire it observes and then predict
  future fire. A 64-bit seed space, with stream seeds derived by SHA-256,
  makes that search impractical. The small seeds in `tests/fixtures/` are for
  tests only.
- A model sees the *realised* fire (the burning cells). That is the intended
  information. It does not see the draws that produced the fire, or any
  future draw.

### 1.4 Person and exit IDs

Person IDs and exit IDs are model-visible, because actions are keyed by
person ID. Scenario authors MUST NOT encode hidden test-set metadata in them.
IDs SHOULD carry no information about the solution (for example, IDs SHOULD
NOT be ordered by distance to an exit), because the collision rule (R1) uses
the ID order.

## 2. Stochastic-world fairness

Fire is exogenous in V1. People and actions do not affect fire propagation:

- The fire model receives only the static terrain, the burning cells, the
  step number and the fire RNG stream (`FireModel.spread`). It does not
  receive people or actions.
- People do not block fire, and fire spread does not depend on where people are.
- The stochastic model takes one draw per (candidate cell, burning neighbour)
  edge on every spread step. The number of draws depends only on the geometry
  and the burning set.

Therefore, for the same scenario and the same evaluator seed, two runs with
different actions MUST produce identical ignitions, burning sets and fire RNG
positions on every step that both runs complete. The fire trajectory is equal
to the trajectory the fire process produces with no people at all. This is
the "common random numbers" property: a difference in outcome between two
matched runs comes from the decisions only, not from luck.

A run can end earlier than another run (for example when everyone evacuates).
The invariant holds for the overlapping steps.

A future subsystem that needs randomness MUST use a new stream label (see
[simulation-contract.md](simulation-contract.md#randomness)). If its draws
depend on actions, it MUST NOT share a stream with the fire.

## 3. Frozen V1 rules

Each rule has a short rationale and the tests that pin it. V1 keeps the
Phase 1 behaviour for every rule. No rule changed in Phase 2: the simulator
dynamics are the same as in `0.1.0`. `SIMULATOR_VERSION` is `0.2.0` only
because the `RunRecord` header gained `observation_version` (see section 6).

Every rule that a scenario can vary is in `Observation.rules`. The model
knows the rules it plays under.

### R1. Collision tie-break: lowest person ID wins

If two or more movers target the same cell, the mover with the lowest person
ID wins. IDs compare as plain strings by code point (`"P10" < "P2"`). Every
other mover gets `blocked_conflict`. The decision uses intent only. A loser is
not re-awarded the cell if the winner fails later.

*Rationale.* The rule is deterministic, simple and visible to the model (IDs
are in the observation, and `rules.conflict_resolution` names the rule). It is identical for every model on a scenario. A
random tie-break would need RNG draws whose count depends on the actions. That
would either couple the fire stream to the actions (section 2) or add a second
stream with no benefit for fairness. The known bias (low IDs win) is the same
for every compared model.

*Tests:* `test_v1_rules.py::test_r1_*`,
`test_movement.py::test_conflict_lowest_id_wins_regardless_of_submission_order`,
`test_movement.py::test_ids_compare_as_strings`,
`test_movement.py::test_conflict_loser_is_not_reawarded_when_winner_fails`.

### R2. Swaps are blocked

Two people who try to exchange cells (`A <-> B`) both get `blocked_cycle` and
stay in place.

*Rationale.* A cell holds one person. A swap needs two people to pass through
each other on a shared edge. This is not possible in a one-cell-wide corridor,
and the grid cannot tell a wide passage from a narrow one. Blocking is the
conservative, simple rule.

*Tests:* `test_v1_rules.py::test_r2_*`, `test_movement.py::test_swap_is_blocked`.

### R3. Rotations are blocked

Any closed chain of movers (a rotation of length 3 or more) is blocked. Every
member gets `blocked_cycle`. An outside mover with a lower ID than the cycle
member it meets wins the conflict (R1). That cycle member then becomes a
non-mover, and the chain resolves as `blocked_occupied`.

*Rationale.* Same rule as R2, with no special case. Open chains (people who
follow each other in one step) are allowed, so a queue still moves at full
speed.

*Tests:* `test_v1_rules.py::test_r3_*`, `test_movement.py::test_rotation_is_blocked`,
`test_movement.py::test_lower_id_outsider_breaks_rotation`,
`test_movement.py::test_following_chain_moves_together`.

### R4. Exit throughput: one person per exit cell per step

An exit is a normal cell for movement, with capacity one. A person who moves
onto a usable exit is evacuated in the same step (phase 6). The exit is
therefore empty at the start of the next step. Each exit cell lets at most one
person through per step. A queue flows with no gaps: the next person can move
into the vacated cell in the same step. Two adjacent exit cells let two
people through per step.

*Rationale.* Throughput follows from the one-person-per-cell rule. It needs no
extra parameter. Congestion at exits is part of what V1 measures.

*Tests:* `test_v1_rules.py::test_r4_*`,
`test_simulator.py::test_exit_throughput_is_one_person_per_exit_per_step`,
`test_simulator.py::test_two_people_cannot_enter_one_exit_in_the_same_step`.

### R5. Burning exits are unusable

An exit evacuates a person only if the exit was not burning at the start of
the step. A burning exit never evacuates anyone. Moving onto a burning exit
follows R6: it is refused (`blocked`) or the person becomes a casualty
(`casualty`). An exit can burn only if `fire.exits_flammable` is `true`. A
person who reaches an exit in the same step that the fire reaches it is
evacuated, because evacuation (phase 6) comes before fire spread (phase 7).

*Rationale.* A burning exit is the main "exit becomes unusable" event that V1
measures. Evacuation before spread gives the model the benefit of the doubt
for a move that was valid when it was chosen: the model saw a usable exit.

*Tests:* `test_v1_rules.py::test_r5_*`,
`test_simulator.py::test_entering_burning_exit_does_not_evacuate`,
`test_fire.py::test_exit_flammability_is_configurable`,
`test_fire.py::test_fire_threatening_exit_burns_exit_and_it_stops_evacuating`.

### R6. `enter_fire`: explicit in every scenario, default `casualty`

`movement.enter_fire` is `blocked` or `casualty`. With `blocked`, a move into
a cell that was burning at the start of the step gives `blocked_fire`, and the
person stays in place. With `casualty`, the person moves and becomes a
casualty with cause `entered_fire`. Scenario files MUST state the value (the
key is required). The Python default (`MovementConfig()`) is `casualty`.
Scored V1 scenarios SHOULD use `casualty`.

*Rationale.* `casualty` makes a bad decision have its real consequence, so it
measures decision quality. `blocked` hides some bad decisions behind a safety
net. Both values stay available for tests and ablations, and the value is part
of the scenario hash, so two runs can never silently use different policies.

*Tests:* `test_v1_rules.py::test_r6_*`, `test_movement.py::test_enter_fire_policies`,
`test_simulator.py::test_entering_fire_under_casualty_policy`,
`test_simulator.py::test_blocked_fire_policy_keeps_person_in_place`.

### R7. Ignition timing

- The scenario ignitions burn at step 0. They are in the first observation.
  No person can start on an ignition.
- Spread happens in phase 7 of step `t` (1-based), only when
  `t % fire.spread_interval == 0`. The first possible spread is in step
  `spread_interval`.
- Movement (phase 5) happens before spread (phase 7) in the same step. A person
  who moves away from a cell that ignites in this step survives. A person who
  stays on it becomes a casualty with cause `fire_spread`.
- Cells that ignite in step `t` are in the observation before step `t + 1`.

*Rationale.* The model always acts on the fire it can see. No cell burns
"between" the observation and the move.

*Tests:* `test_v1_rules.py::test_r7_*`, `test_fire.py::test_spread_interval`,
`test_simulator.py::test_fire_spread_casualty_and_cause`.

### R8. No burn-out

A burning cell burns until the end of the run. Burn-out does not exist in V1,
for any cell type (floor or exit). The burning set never shrinks.

*Rationale.* Simple and monotone. A burned-out cell that becomes passable
again would add a timing rule and new routing behaviour that V1 does not
need. A later version can add burn-out with a `SIMULATOR_VERSION` bump.

*Tests:* `test_v1_rules.py::test_r8_*`.

### R9. Trapped definition

`trapped` is the set of **active** people who have no path to an exit that
is not burning. A path uses 4-neighbour steps and may cross any cell that is
not a wall and not burning. Other people are ignored. The set is computed at
step 0 and after phase 8 of every step.

`trapped` is an evaluator diagnostic. It is recorded in `SimState` and
`RunRecord`. It is **not** model-visible in V1. It does not change a
person's status, and it does not end the run.

*Rationale.* Walls and fire are permanent in V1 (R8). People are not, so
congestion does not count as trapped. The diagnostic is for metrics (for
example "trapped from the start" against "trapped by a bad decision"). It is
kept out of the observation because it is a pre-computed planning aid. A
model can compute the same set from the observation (the map plus the
burning cells), so hiding it removes no information.

*Tests:* `test_v1_rules.py::test_r9_*`, `test_simulator.py::test_trapped_detection`,
`test_fairness.py::test_trapped_diagnostic_is_evaluator_private`.

### R10. Termination precedence

After phase 8 of step `t`:

1. If no person is active, the run ends with `all_resolved`.
2. Else, if `t >= termination.max_steps`, the run ends with `max_steps`.

`all_resolved` wins when both hold. Inside a step, evacuation (phase 6) comes
before casualties (phase 8). A run does not end early because every active
person is trapped. A terminated simulator raises `SimulationError` on
`step()`.

*Rationale.* A run that resolves everyone on its last step is a full success,
so `all_resolved` is the more informative reason. Early stop on "all trapped"
would make the run length depend on a diagnostic. It would not change any
outcome, because a trapped person cannot evacuate.

*Tests:* `test_v1_rules.py::test_r10_*`,
`test_simulator.py::test_all_resolved_takes_precedence_over_max_steps`,
`test_simulator.py::test_max_steps_termination_is_deterministic`.

## 4. Observation contract (version 1.0)

`Simulator.observe()` returns an `Observation`. In V1 the current state is
fully observable (the observation shows every burning cell) and the episode
dynamics are known (`rules`). The observation holds no reference to the
scenario, the simulator or any RNG. Its object graph contains only these
types: `Observation`, `MapView`, `RulesView`, `ExitSpec`, `PersonState`,
`PersonStatus`, `Action`, `Pos`, `FireModelName`, `EnterFirePolicy`,
`ConflictResolution`, tuples, frozensets, integers, floats, booleans, strings
and `None`.

| Field | Type | Meaning |
| --- | --- | --- |
| `step` | `int` | Number of completed steps. `0` before the first step. |
| `map.width`, `map.height` | `int` | Grid size. |
| `map.obstacles` | `tuple[Pos, ...]` | Wall cells, row-major. |
| `map.exits` | `tuple[ExitSpec, ...]` | Exit ID and position, sorted by ID. |
| `rules` | `RulesView` | The episode dynamics. See the next table. |
| `people` | `tuple[PersonState, ...]` | Every person: `id`, `position`, `status` (`active`, `evacuated`, `casualty`), sorted by ID. |
| `burning` | `frozenset[Pos]` | Every burning cell now. |
| `allowed_actions` | `tuple[Action, ...]` | `stay`, `north`, `south`, `east`, `west`. Every active person may submit any of them every step. |

`RulesView` holds every action-relevant rule that a scenario can vary. Each
value is copied from the scenario when the simulator is built:

| Field | Type | Copied from |
| --- | --- | --- |
| `fire_model` | `FireModelName` | `fire.model` (`deterministic` or `stochastic`) |
| `spread_interval` | `int` | `fire.spread_interval` |
| `spread_probability` | `float \| None` | `fire.spread_probability`. `None` (JSON `null`) unless the model is `stochastic`. |
| `exits_flammable` | `bool` | `fire.exits_flammable` |
| `enter_fire` | `EnterFirePolicy` | `movement.enter_fire` (`blocked` or `casualty`) |
| `conflict_resolution` | `ConflictResolution` | `movement.conflict_resolution` (V1: `lowest_id_wins` only) |
| `max_steps` | `int` | `termination.max_steps` |

The spread probability is a parameter of the hazard distribution. It is not
evaluator randomness: it tells the model how likely spread is, but not which
draws will succeed.

`Observation.to_dict()` gives the serialised form, with the extra key
`observation_version`. The action contract is unchanged: `step(actions)`
needs exactly one action from `allowed_actions` for each active person. A move
that cannot happen is legal and resolves to a `blocked_*` outcome.

The observation does **not** contain: the seed, RNG state, stream seeds,
future draws, scenario ID, description, content hash, schema version, or
`trapped`.

A change to the fields or the meaning of the observation MUST bump
`OBSERVATION_VERSION`.

*Tests:* `tests/test_fairness.py`. The tests check:

- the exact field lists of `Observation`, `MapView` and `RulesView`, and of
  the serialised form,
- that the object graph (walked through every instance attribute, not only
  the declared fields) holds only the public types above, and never a
  `Scenario`, `Simulator`, `SimState`, `StepRecord`, `RunRecord`, RNG or
  configuration object,
- that the seed, stream seeds, RNG positions and hidden metadata are absent
  from the observation and its JSON form,
- that observations are identical for different seeds and different hidden
  metadata,
- that two scenarios that differ only in one public rule give different
  observations, and that the rules are the only difference,
- that every `RulesView` value equals the scenario configuration.

## 5. Metrics input

A metric MAY need both the `Scenario` and the `RunRecord`. The `RunRecord`
has the actions, movements, ignitions, evacuations, casualties, trapped sets
and states of each step. The static geometry (walls, exit positions, grid
size) and the configuration (for example `max_steps`) are in the `Scenario`.
The `scenario_hash` in the record binds the two together. Not every metric can
be computed from a `RunRecord` alone.

## 6. Provenance and change control

Every `RunRecord` header stores `simulator_version`, `observation_version`,
`schema_version`, `scenario_id`, `scenario_hash` and `seed`. A new record
always stores the active `OBSERVATION_VERSION`. `verify_record` rejects a
record with a different `simulator_version` or `observation_version`, or a
different scenario hash, with an explicit `ReplayMismatchError`.
`RunRecord.from_dict` rejects a record that has no `observation_version`.

- A change that changes simulator output (any `RunRecord` byte) for the same
  scenario, seed and actions MUST bump `SIMULATOR_VERSION`. The golden digest
  test fails until the bump is made.
- A change to the observation fields or their meaning MUST bump
  `OBSERVATION_VERSION`.
- A change to the scenario file format MUST bump `SCHEMA_VERSION`.
- A documentation-only clarification of visibility does not need a bump.

**History.**

| `SIMULATOR_VERSION` | Change |
| --- | --- |
| `0.1.0` | Phase 1 deterministic core. |
| `0.2.0` | The `RunRecord` header gained `observation_version`. The dynamics did not change. With that key removed and the version set back to `0.1.0`, every golden record hashes to its `0.1.0` digest. `OBSERVATION_VERSION` stays `1.0`, because V1 was not released before this change. |
