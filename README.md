# EMBERBench

EMBERBench is a benchmark for **decision-making models on dynamic fire-evacuation tasks**.
A model controls the people in a 2D grid building. Fire spreads through the building, and
the model must decide each timestep how every person moves. The benchmark measures how
well the model's decisions get people out through the exits before the fire reaches them.

> **Status: Phase 3.** This repository contains the benchmark contract, the
> deterministic simulation core, the frozen V1 fairness and semantics
> specification and auditable per-episode metrics. Model integrations, LLM APIs, a UI, procedural map generation,
> leaderboards, a composite score, cross-seed aggregation and human-behaviour models are out of scope for now.

## What EMBERBench measures

The benchmark looks at the quality of sequential decisions under a changing hazard:

- routing people to exits that are reachable and that stay usable,
- avoiding routes that the fire will cut off,
- managing congestion at corridors and exits,
- reacting when the fire makes an exit unusable.

Per-episode raw metrics (evacuated, casualties, stranded, timing, congestion,
path efficiency, a fire-exposure proxy, exit use and replanning proxies) are
computed from the `Scenario` and the `RunRecord` with `compute_episode_metrics`.
It scores only records that pass structural validation and a full simulator
replay (`verify_record`). See [docs/metrics.md](docs/metrics.md). There is no single score: raw metrics
are canonical, and safety always comes before speed.

## Fairness principle

Every compared model runs on **exactly the same episode**:

- the same scenario: map, obstacles, exits, initial population (with stable
  person IDs), ignition points and hazard configuration,
- the same simulator version and rules,
- the same evaluator seed schedule, and therefore the same exogenous fire process.

The evaluator seed is for reproducibility and matched evaluation only. **A model
never receives the seed**, the RNG state, future random draws or hidden scenario
metadata. A model receives only the model-visible `Observation`: the current
step, the static map, the episode rules (fire model and parameters, fire-entry
policy, conflict rule, step limit), the people, the burning cells and the
allowed actions. V1 evaluates decisions under known rules; only the
evaluator randomness is hidden.

Fire is exogenous in V1: actions do not change it. For the same scenario and
seed, every model faces the same fire trajectory, so a difference in outcome
comes from the decisions only. Model outputs are the only thing that differs
between two matched runs.

The normative contract is in [docs/v1-fairness-spec.md](docs/v1-fairness-spec.md).

## Deterministic scenario + seed evaluation

A **(scenario, seed)** pair fully defines a benchmark episode. The seed belongs to
the evaluator, not to the model:

- All randomness comes from simulator-owned [SplitMix64](src/emberbench/rng.py)
  streams. Each stream is derived from the seed and a subsystem label (for example
  `fire`). The code does not use `random`, the clock, UUIDs or any global state.
- Iteration over sets and dicts is always sorted first, so results do not depend on
  `PYTHONHASHSEED`. A test checks this across processes.
- The same scenario, seed and action sequence produce **byte-identical** canonical
  JSON run records.
- `verify_record()` replays a stored run and rejects any mismatch. It also rejects a
  different scenario hash, a different `SIMULATOR_VERSION` or a different
  `OBSERVATION_VERSION`.

### Development scenarios are not the test set

The scenarios in [`tests/fixtures/scenarios/`](tests/fixtures/scenarios/) are tiny
**test fixtures**. They exist to check simulator behaviour. They are **not**
benchmark scoring scenarios, and they are **not intended to become the final hidden
test set**. The scored evaluation set must be kept separate from development
scenarios, so that models cannot be tuned to it.

## V1 scope

| Area | V1 behaviour |
| --- | --- |
| World | 2D grid. Cells: empty, wall/obstacle, exit, fire, person. |
| People | One cell each. Stable unique IDs. One move per step (N/S/E/W/stay). No panic, health, groups or speed differences. |
| Collisions | Deterministic and documented. The lowest ID wins a contested cell. Swaps and rotations are blocked. Following another person in the same step is allowed. |
| Exits | Entering a non-burning exit evacuates the person in the same step. Each exit cell lets one person through per step. |
| Fire | Spreads to 4-neighbours. It never passes through walls. Model: `deterministic` or seeded `stochastic`. `exits_flammable` is explicit in config. No burn-out. |
| Casualties | An active person on a burning cell becomes a casualty. Moving into fire is either `blocked` or a `casualty` (config). |
| Termination | `all_resolved` (nobody is active) or `max_steps`. |

The full rules are in [docs/simulation-contract.md](docs/simulation-contract.md). The
frozen V1 decisions and their rationale are in
[docs/v1-fairness-spec.md](docs/v1-fairness-spec.md). The scenario file format is in
[docs/scenario-schema.md](docs/scenario-schema.md).

### Deterministic phase order (per timestep)

1. Observation (state snapshot before the step)
2. Submitted movement actions (validated, exactly one per active person)
3. Movement validation (boundary, wall, fire policy)
4. Collision and congestion resolution
5. Person movement (simultaneous)
6. Evacuation through exits
7. Fire propagation
8. Casualties and trapping updates
9. Termination check

## Architecture

```
src/emberbench/
  scenario.py   versioned scenario schema, strict parsing and validation, content hash
  grid.py       positions, cell kinds, actions, neighbourhoods
  rng.py        simulation-owned SplitMix64 streams, seed derivation
  fire.py       FireModel protocol, deterministic and stochastic spread
  movement.py   action parsing, movement validation, collision resolution
  state.py      immutable, evaluator-private SimState snapshots, render()
  observation.py the versioned model-visible Observation, MapView and RulesView
  simulator.py  Simulator: runs the 9 phases, run_episode()
  events.py     StepRecord / RunRecord, the replay and audit log
  replay.py     replay() and verify_record()
  metrics.py    compute_episode_metrics(): auditable raw per-episode metrics
  canonical.py  canonical JSON and SHA-256 hashing
  errors.py     explicit exception types
```

The concepts stay separate. The **scenario** is static input. The **simulator** owns
the mutable run and produces immutable **state** snapshots. **Fire dynamics** and
**movement resolution** are pure functions of their inputs (and of the passed RNG
stream). **Events** record what happened. **Metrics** (`metrics.py`) use the
`Scenario` and the `RunRecord` together. Some metrics need the static map or the
configuration from the scenario, so not every metric can come from `RunRecord` alone.

### Minimal use

```python
from emberbench import Simulator, load_scenario

scenario = load_scenario("tests/fixtures/scenarios/one_person_one_exit.json")
sim = Simulator(scenario)            # or Simulator(scenario, seed=123)
while not sim.done:
    obs = sim.observe()              # phase 1: the only thing a model may see
    sim.step({p.id: "east" for p in obs.active})
record = sim.record()                # replayable, canonical RunRecord
print(record.final_state.termination_reason, record.digest())
```

## Replay and events

A `RunRecord` holds everything needed to replay and audit a run:

- `simulator_version`, `observation_version`, `schema_version`, `scenario_id`,
  `scenario_hash`, `seed`
- `initial_state`
- for each step: the submitted `actions`, the resolved `movements` (intent, final
  cell and outcome such as `moved` or `blocked_conflict`), the `ignited` cells,
  the `evacuations` (person and exit), the `casualties` (person, cell and cause),
  and the full post-step `state` (people, burning cells, trapped set, RNG
  position and termination).

All records serialise to canonical JSON (`record.to_json()`) and load back with no
loss (`RunRecord.from_json`).

## Running the tests

Requires Python 3.12+ and has no runtime dependencies.

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m pytest
```

```bash
python -m mypy
```

The suite covers deterministic replay, two independent simulator instances,
cross-process `PYTHONHASHSEED` independence, serialisation round trips, walls blocking
movement and fire, exit behaviour, explicit rejection of invalid scenarios and
actions, collision rules, termination, ID stability, golden trajectory digests,
every frozen V1 rule, the observation contract (no seed, RNG state or hidden
metadata) and the fire trajectory invariance across different actions.
Any change to simulator output fails the golden test until you bump
`SIMULATOR_VERSION` on purpose.
