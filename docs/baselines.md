# Reference baselines (baselines version 1.0)

**Reference baselines are anchors, not benchmark winners.**

EMBERBench ships five deterministic, non-LLM reference agents. They make
benchmark results readable: a model that does no better than
`static-shortest-path-v1` on a scenario family has not shown more than a
textbook routing rule. The baselines are not tuned to win, and no result here
is a leaderboard entry or a ranking. Their EMBER Scores
([scoring.md](scoring.md)) are reference anchors, not entries.

The module is `emberbench.baselines`. A separate, **privileged** upper-bound
analysis is in `emberbench.oracle` (section 5). It is not a baseline and not a
model.

## 1. The fair-baseline contract

Every registered baseline is a *fair* model. It uses the public model
interface only ([model-interface.md](model-interface.md)):

- A zero-argument `AgentFactory` builds a new agent for every episode. The
  agent is built from its public baseline configuration only.
- `act(observation)` receives one `Observation` and returns one action for
  every active person.
- It never receives the scenario, the evaluator seed, the
  `EvaluationManifest`, a `RunRecord`, `SimState`, RNG state, future fire,
  the scenario ID or hash, or generation metadata.

Every decision is a pure function of the observation and the configuration.
No baseline keeps state between steps. No baseline reads the clock, the OS,
environment variables, the `random` module or any simulator RNG. No output
depends on hash order, dict insertion order or the order of people in the
observation.

Baselines run through the normal evaluation path with no special case:

```python
from emberbench import agent_policy_factory, baseline_factory, run_evaluation

factory = baseline_factory("fire-aware-shortest-path-v1")
result = run_evaluation(suite, manifest, "fire-aware-shortest-path-v1",
                        agent_policy_factory(factory), "fire-aware.json")
```

Use the same `EvaluationManifest` for every baseline and every model, so the
results pair exactly ([evaluation.md](evaluation.md)).

## 2. Registry

| ID | Version | Config | Description |
|---|---|---|---|
| `random-legal-v1` | 1.0 | `{"seed": 0}` | Uniform seeded choice among stay and moves into in-bounds non-wall cells. |
| `nearest-exit-greedy-v1` | 1.0 | `{}` | Beeline (Manhattan) toward the exit with the shortest wall-only path; ignores fire and people. |
| `static-shortest-path-v1` | 1.0 | `{}` | Shortest wall-only path to the nearest exit, replanned each step; ignores fire and people. |
| `fire-aware-shortest-path-v1` | 1.0 | `{}` | Shortest path avoiding currently burning cells and exits; flees the fire if no safe route exists. |
| `congestion-aware-v1` | 1.0 | `{}` | Fire-aware shortest paths with exits assigned by current load (one person per exit per step). |

The registry is a fixed, read-only table (`emberbench.baselines.BASELINES`).
Nothing is imported or instantiated from a string.

- `get_baseline(baseline_id)` returns a `BaselineSpec`. An unknown ID (or a
  value that is not a string) raises `BaselineError`. There is no fuzzy
  match and no default.
- `baseline_factory(baseline_id, config=None)` returns a zero-argument
  factory. `config` is the baseline's config object, its `to_dict()` form
  (parsed strictly: exact keys, exact types, ranges) or `None` for the
  default. A config of the wrong type raises `BaselineError`.
- `BaselineSpec.descriptor(config)` is the canonical description:
  `baselines_version`, `baseline_id`, `baseline_version`, `privileged`
  (always `false` here) and `config`.
- `BaselineSpec.config_digest(config)` is the SHA-256 of the canonical JSON of
  the descriptor. Record it next to the `model_id` when you publish a
  baseline result. The default digests are pinned in
  `tests/test_baselines.py`.

`model_id` in `run_evaluation` stays a free-text label. A baseline run with
a non-default config should use a label that says so, for example
`random-legal-v1+seed=7`.

## 3. Shared conventions

- **Distance.** Breadth-first search over 4-neighbours, one BFS per exit,
  from the observation's static map. "Wall-only" means walls block and
  nothing else does.
- **Move order.** Moves are tried in the order of
  `observation.allowed_actions` without `stay`: north, south, east, west in
  V1. The first move that meets the rule wins. This is how equal shortest
  paths are broken: the path is the one whose next move comes first in that
  order.
- **Equal exits.** The exit with the lowest exit ID (plain string order) wins.
- **Burning cells.** V1 rules R5 and R6: a burning exit never evacuates, and
  a move into a burning cell is refused (`blocked`) or fatal (`casualty`).
  The fire-aware baselines therefore treat every currently burning cell as
  impassable and every currently burning exit as unusable, under both
  `enter_fire` policies.
- **Fallback (fire-aware and congestion-aware).** When a person has no
  currently safe route to any usable exit, the person flees: among `stay` and
  every move into an in-bounds, non-wall, non-burning cell, it takes the one
  with the largest wall-only distance to the nearest burning cell. Ties go to
  `stay`, then to the move order.

Complexity is per step. `N` is the number of grid cells, `E` the number of
exits and `P` the number of active people.

## 4. The baselines

### `random-legal-v1`

- **Visible information:** map walls and size, `allowed_actions`, each
  active person's ID and position, the step number.
- **Algorithm:** the legal set of a person is `stay` plus every move into an
  in-bounds, non-wall cell, in `allowed_actions` order. One element is
  chosen uniformly.
- **Randomness:** no evaluator randomness. The draw is SHA-256 of the
  canonical JSON of
  `["emberbench/baseline/random-legal/1", seed, step, person_id, counter]`.
  The first 8 bytes are a 64-bit word. Words at or above the largest
  multiple of the set size are rejected (the counter goes up), so the choice
  is exactly uniform. `seed` is a **public baseline parameter** in
  `[0, 2**64 - 1]` (default 0). It is model configuration, not an evaluator
  seed, and its domain label differs from every simulator stream label. The
  same observation and config always give the same actions. Different
  people and steps get independent draws.
- **Tie-breaking:** none needed.
- **Complexity:** `O(P)`. Each draw needs about one hash.
- **Known weakness:** it does not try to leave. It ignores exits, fire and
  people. It anchors the floor of the scale.
- **Fair or privileged:** fair.

### `nearest-exit-greedy-v1`

- **Visible information:** map walls, size and exits; active people.
- **Algorithm:** the target is the exit with the shortest wall-only path from
  the person. The person then takes the first move into an in-bounds,
  non-wall cell that lowers the straight-line (Manhattan) distance to that
  exit. If no move does, the person stays.
- **Tie-breaking:** equal exits by exit ID; equal moves by move order.
- **Complexity:** `O(E * N + P * E)`.
- **Known weakness (intentional):** it is a beeline. It stalls for good when
  a wall lies between the person and the exit (it scores 0 on
  `v1-pub-08-maze-deceptive-exit`). It ignores fire and people.
- **Interpretation note:** the exit is chosen by wall-only path distance, but
  each step only has to lower the straight-line distance. If each step had to
  lower the path distance, this baseline would be identical to
  `static-shortest-path-v1`.
- **Fair or privileged:** fair.

### `static-shortest-path-v1`

- **Visible information:** map walls, size and exits; active people.
- **Algorithm:** one BFS per exit over non-wall cells. The target is the exit
  with the shortest path from the person's current position. The person
  takes a move that lowers that distance by one. It replans from the observed
  position every step. Fire, burning exits and other people are ignored. A
  person with no wall-only route to any exit stays.
- **Tie-breaking:** equal exits by exit ID; equal shortest paths and equal
  actions by move order.
- **Complexity:** `O(E * N + P * E)`.
- **Known weakness:** it walks into fire and toward burning exits, and it
  sends every person to the same nearest exit, whatever the queue.
- **Fair or privileged:** fair.

### `fire-aware-shortest-path-v1`

- **Visible information:** as above, plus the **currently** burning cells.
- **Algorithm:** like `static-shortest-path-v1`, but every currently burning
  cell is impassable and every currently burning exit is unusable. It
  recomputes from each observation. It never predicts future spread, and it
  never uses the evaluator seed or the spread probability. With no currently
  safe route the person uses the flee fallback.
- **Tie-breaking:** as `static-shortest-path-v1`; the fallback as in
  section 3.
- **Complexity:** `O(E * N + P * E)`, plus `O(N)` for the fallback map when
  at least one person needs it.
- **Known weakness:** a route that is safe now can burn before the person
  passes. It reacts only after the fire blocks a route, and it ignores
  congestion.
- **Fair or privileged:** fair.

### `congestion-aware-v1`

- **Visible information:** map, currently burning cells, and the current
  positions of all active people. It does **not** use `CrowdDiagnostics`,
  which are evaluator-side and never reach a model.
- **Algorithm:**
  1. The fire-aware distance fields, one per usable exit.
  2. People with a safe route are sorted by their nearest-exit distance, then
     by person ID. In that order, each person takes the exit with the lowest
     estimated evacuation step `max(distance, load + 1)`, where `load` is the
     number of people already assigned to that exit. An exit lets one person
     through per step (V1 rule R4), so this is the earliest step that exit
     could take the person if everyone ahead used it.
  3. The person takes a move that lowers the distance to the assigned exit by
     one. A move into a cell that no active person occupies now is
     preferred.
  4. People with no safe route use the flee fallback.
- **Tie-breaking:** equal estimates go to the shorter distance, then to the
  lowest exit ID; equal moves to a free cell first, then by move order.
- **Complexity:** `O(E * N + P log P + P * E)`.
- **Known weakness:** the load is a snapshot count, not a forecast. People
  can switch exits from one step to the next as the counts change. It does
  not plan around other people's paths, and it is not a model of human
  crowd behaviour.
- **Fair or privileged:** fair.

## 5. The privileged escape bound (not a baseline)

`emberbench.oracle` is **privileged**. It reads the `Scenario` and the
evaluator seed, and therefore the whole future fire. It is an upper-bound
reference analysis. It is **not** an agent: it has no `act` method, it is not
in the registry, it is not exported from the top-level package, and it
produces no actions and no `EvaluationResult`. Its numbers must never be
shown as a model competitor.

**Why it is correct.** In V1 the fire never sees people or actions
(`emberbench.fire`). Under one scenario and seed, the burning set at every
step is the same for every policy. `fire_trajectory(scenario, seed)`
reproduces that set from the fire model and the seed-derived fire stream. A
test checks it against the simulator on every public scenario.

For each person **alone**, a time-expanded BFS over `(cell, step)` finds the
earliest step at which the person can reach a usable exit. It follows the
simulator's phase order: a move into a cell that burns at the start of the
step is never useful, an exit that is not burning at the start of the step
evacuates at once, and a person on a cell that burns after fire spread is a
casualty. Other people can only block moves (rules R1-R4), so ignoring them is
a relaxation. Hence, for every policy and every episode:

- `max_evacuated` (people who can escape alone) is an upper bound on
  `safety.evacuated`,
- `min_person_time_to_safety_total` is a lower bound on
  `time.person_time_to_safety_total`, and
- each evacuated person's step is at least that person's bound.

The tests check these three facts for every baseline on the development
manifest. The bound is tight for a single person. With crowds it is usually
not attainable, because people compete for exits. It says nothing about
casualties versus stranded people.

**Cost.** `O(T * N)` per person, where `T` is `max_steps`.

```python
from emberbench.oracle import privileged_manifest_bounds  # privileged: evaluator-side only

bounds = privileged_manifest_bounds(suite, manifest)
```

A full multi-person oracle (a joint plan with the known fire) is not
included. An optimal joint plan is a combinatorial multi-agent search, and a
prioritized planner is not an upper bound, so it would be a misleading
"oracle".

## 6. Development results

Public V1 suite, development seeds `(0, 1, 2)` (manifest hash
`7b9aae45e5e2828f881b2f14d8e5585627e5f8bdbb137a1ed2331bf10ce8c876`), 30
episodes, 561 person-episodes. These are development numbers for
orientation only. They are not scored results, and the order between
baselines can differ on other scenarios. `always-stay` is a test control, not
a registered baseline.

| Model | Evacuated | Casualties | Stranded | Person-time to safety |
|---|---:|---:|---:|---:|
| `always-stay` (control) | 0 | 555 | 6 | 80310 |
| `random-legal-v1` | 12 | 546 | 3 | 78540 |
| `nearest-exit-greedy-v1` | 404 | 148 | 9 | 31358 |
| `static-shortest-path-v1` | 419 | 142 | 0 | 29933 |
| `fire-aware-shortest-path-v1` | 433 | 128 | 0 | 29084 |
| `congestion-aware-v1` | 526 | 35 | 0 | 14960 |
| privileged escape bound | ≤ 561 | – | – | ≥ 5432 |

Evacuated people per scenario (sum over the three seeds):

| Scenario | People | stay | random | greedy | static | fire-aware | congestion | bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1-pub-01-single-exit-sparse` | 12 | 0 | 0 | 12 | 12 | 12 | 12 | 12 |
| `v1-pub-02-single-exit-dense` | 108 | 0 | 3 | 81 | 81 | 81 | 102 | 108 |
| `v1-pub-03-corridor-bottleneck` | 24 | 0 | 0 | 21 | 21 | 21 | 24 | 24 |
| `v1-pub-04-unequal-exits` | 15 | 0 | 0 | 15 | 15 | 15 | 15 | 15 |
| `v1-pub-05-central-ignition` | 24 | 0 | 0 | 18 | 18 | 18 | 18 | 24 |
| `v1-pub-06-threatened-exit` | 18 | 0 | 0 | 9 | 9 | 15 | 15 | 18 |
| `v1-pub-07-multiple-ignitions` | 27 | 0 | 0 | 21 | 21 | 21 | 24 | 27 |
| `v1-pub-08-maze-deceptive-exit` | 15 | 0 | 0 | 0 | 15 | 15 | 15 | 15 |
| `v1-pub-09-crowd-100-split-flow` | 300 | 0 | 9 | 216 | 216 | 219 | 285 | 300 |
| `v1-pub-10-route-invalidation` | 18 | 0 | 0 | 11 | 11 | 16 | 16 | 18 |

How to read this:

- Every person is escapable alone in every development episode (the bound
  is 561 of 561). Every loss is a matter of routing, fire timing and
  coordination, not of an impossible start.
- Fire awareness matters where the fire threatens a route or an exit
  (`threatened-exit`, `route-invalidation`). Exit load matters in dense and
  crowded scenarios. Even the best baseline leaves a gap to the bound in
  person-time to safety (14960 against at least 5432).
- On the single-exit scenarios, `congestion-aware-v1` gains from its
  free-cell move preference, not from exit choice.

The tests do not pin this table. They pin only broad, robust facts (section
7), plus exact digests of the baseline runs on a smaller development
manifest.

## 7. What the tests prove

`tests/test_baselines.py`:

- **Registry:** the five IDs, versions, descriptions and descriptors; pinned
  config digests; a read-only registry; unknown IDs and invalid configs fail
  closed.
- **Integration:** every baseline runs unchanged through
  `run_evaluation(..., agent_policy_factory(...))` on one shared manifest,
  including the canonical sparse, corridor-bottleneck, 100-person and
  stochastic route-invalidation scenarios, and a generated suite.
- **Leakage:** the factory is called with no arguments; `act` gets exactly
  one `Observation` and nothing else; the evaluator seed, the manifest hash
  and the scenario ID never occur in it; the same observation gives the same
  actions under four different evaluator seeds; the module imports no
  scenario, manifest, run-record, simulator, state, RNG, fire, generator,
  crowd-diagnostics or oracle API (an AST check plus a check of the module
  namespace).
- **Determinism:** a repeated run gives a byte-identical artifact; reversing
  the action dict's insertion order gives a byte-identical artifact;
  reordering people and burning cells in the observation gives the same
  actions; three `PYTHONHASHSEED` values in fresh processes give identical
  outputs; golden digests of all runs and golden two-step action traces are
  pinned, and CI checks them on Linux, Windows and macOS with Python 3.12
  and 3.13.
- **Sanity (no total ranking):** the shortest-path baselines evacuate more
  than always-stay; fire-aware avoids a currently burning route that static
  takes; congestion-aware lowers person-time and the largest exit share in a
  designed door bottleneck; greedy stalls behind a wall that static goes
  around; the random baseline is deterministic for its seed and changes with
  it.
- **Privileged bound:** it is separated from the fair interface; its fire
  trajectory matches the simulator; it holds for every baseline and episode;
  it is tight for one person and finds a person who cannot escape.

## 8. Versioning

`BASELINES_VERSION = "1.0"` is independent of every other version. Bump it
when any baseline can act differently for the same observation and
configuration, or when the registry or config format changes. A baseline's
own `baseline_version` (and its `-vN` ID suffix) changes when that
baseline's algorithm changes.

This phase changes no other version: `SIMULATOR_VERSION`,
`OBSERVATION_VERSION`, `ACTION_SCHEMA_VERSION`, `SCHEMA_VERSION`,
`METRICS_VERSION`, `STATISTICS_VERSION` and the suite formats are unchanged,
and all public scenario and suite hashes are unchanged.
