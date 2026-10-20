# EMBERBench

EMBERBench is a benchmark for **decision-making models on dynamic fire-evacuation tasks**.
A model controls the people in a 2D grid building. Fire spreads through the building, and
the model must decide each timestep how every person moves. The benchmark measures how
well the model's decisions get people out through the exits before the fire reaches them.

> **Status: Phase 7.** This repository contains the benchmark contract, the
> deterministic simulation core, the frozen V1 fairness and semantics
> specification, the finalized matched stochastic hazard contract, auditable
> per-episode metrics, crowd coordination diagnostics, the canonical V1
> public scenario suite and resumable paired evaluation on exact scenario +
> seed schedules. Wind, delayed
> ignitions, burn-out, smoke and material flammability are deferred to a
> future benchmark version. Model integrations, LLM APIs, a UI, procedural map generation,
> cross-seed aggregation, win rates and confidence intervals, and
> human-behaviour models are out of scope for now. Phase 10 adds the
> official EMBER Score and frozen environment packs (see below).

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
replay (`verify_record`). See [docs/metrics.md](docs/metrics.md). The metrics have no single score: raw metrics
are canonical, and safety always comes before speed.

Crowd coordination diagnostics come from `compute_crowd_diagnostics` on the
same authenticated inputs: per-step movement and block counts, contested
cells, a row-major table of congestion hotspots, per-exit throughput and a
static nearest-exit demand proxy. They observe the frozen V1 crowd rules and
do not model human behaviour. See [docs/crowd-diagnostics.md](docs/crowd-diagnostics.md).

`build_replay_audit` turns the same authenticated inputs into a read-only
replay audit: a step-by-step timeline, per-person paths, per-step fire deltas
and an optional congestion overlay. Deterministic ASCII frames
(`render_frame`) and a portable JSON animation export (`export_animation`)
come from that audit. Visualization never affects benchmark results. See
[docs/replay-visualization.md](docs/replay-visualization.md).

Five deterministic, non-LLM reference baselines (random legal, nearest-exit
greedy, static shortest-path, fire-aware shortest-path and congestion-aware
routing) run through the same model interface as any evaluated model and see
only the observation. Reference baselines are anchors, not benchmark winners.
A separate, privileged escape bound gives an evaluator-side upper bound on
evacuations. See [docs/baselines.md](docs/baselines.md).

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

### Canonical V1 public suite

[`scenarios/public/v1/`](scenarios/public/v1/) holds the fixed, public V1
diagnostic suite (`emberbench-v1-public`, suite version `1.0.0`): ten scenarios
that cover twelve diagnostic families, from a sparse single-exit room to a
100-person crowd and a stochastic route invalidation. A manifest pins each
scenario's content hash. `load_suite()` verifies the manifest against the files.
The suite is public by design. It is not the hidden evaluation set, and its
seeds are development defaults, not evaluator seeds. See
[docs/scenario-suite.md](docs/scenario-suite.md).

```python
from emberbench import load_suite

suite = load_suite("scenarios/public/v1")
for scenario in suite.scenarios:
    print(scenario.scenario_id, scenario.content_hash())
```

### Procedural unseen scenarios

`generate_suite()` builds unseen V1 scenarios from a public `GeneratorConfig`
and an evaluator-private generator seed, with no ambient entropy. Every
scenario must meet explicit generation constraints (route existence, passage
width, route redundancy, densities, exit separation, ignition distances), or
generation fails. `freeze_generated_suite()` writes the suite once, before any
model runs, and `load_generated_suite()` verifies it and returns a normal
`Suite` for `build_evaluation_manifest()` and `run_evaluation()`. The seed is
never stored. See [docs/procedural-generation.md](docs/procedural-generation.md).

`emberbench.composition` composes the V1 evaluation suite from a generated
candidate pool on the evaluator side: unique worlds, the frozen calibration
panel, admission, difficulty band, the ten V1 family rules, deterministic
family x band x fire-model quotas with targeted top-up, and a private freeze
record. See [docs/generator.md](docs/generator.md).

### Hidden-test integrity

`emberbench.integrity` defines the held-out protocol: which assets are public
and which stay with the evaluator, an evaluator workspace outside the
repository, a provider-neutral `SubmissionIdentity`, and an immutable
`EvaluationProvenance` record that pins the versions, the frozen suite, the
evaluation manifest hash, the submission and the result digest by SHA-256. It
holds no seeds and no timestamps. Signatures are optional and external.
`verify_evaluation_provenance()` re-derives every pin and fails closed.
Post-submission generation binds the generator seed to a frozen submission
roster. `emberbench.leakguard` keeps held-out material out of the repository,
the wheel and the sdist. Hashes prove content identity, not authorship. See
[docs/integrity.md](docs/integrity.md).

## The EMBER Score

`score_results(trials, pack)` gives one failure-sensitive number in
`[0, 100]`: **benchmark-performance points under EMBER scoring version 1.0**,
never a percentage of people evacuated. It is
`100 * Safety * Quality^beta * Severity`. Safety multiplies evacuation, a
casualty factor and a stranding factor over losses attributed person by
person (a loss is avoidable when that person could have escaped), and is 0
when half of the savable people are lost. Quality is a weighted geometric
mean of time, hazard, path and replanning, each relative to a privileged
optimum of the same episode; time is relative to a capacity bound that
counts unavoidable queues at exits and doors. Every input is a function of
the replayed state trajectory, so identical trajectories score identically.
Congestion (block rate, contested cells, hotspots, exit throughput) is
reported in every score report but is never a score input: a bottleneck is
charged by the delay, exposure and losses it causes. An avoidable casualty
caps an episode below 60; a severe delay (including a severe realized
bottleneck), exposure, path or replanning failure caps it below 80. Missing, invalid,
failed and timed-out episodes score 0 and are counted. Episodes are averaged
by scenario, then by primary family; the report gives a stratified
bootstrap 95% interval, every component, per-family scores and lower-tail
statistics. `compare_scores` gives a paired outcome (`a_better`,
`b_better`, `inconclusive` or `exact_tie`), and `leaderboard_order` breaks
ties by safety. Every model is scored on the same frozen
`EnvironmentPack` (`freeze_environment_pack`, `verify_environment_pack`),
whose digest pins the scenario bytes, the seed schedule, the versions and
the fire world of every episode. See [docs/scoring.md](docs/scoring.md),
[docs/environment-pack.md](docs/environment-pack.md) and the research and
sensitivity study in
[docs/benchmark-methodology.md](docs/benchmark-methodology.md).

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

### Paired evaluation

An `EvaluationManifest` pins an ordered schedule of (scenario, seed) episodes,
the suite hash and the simulator, observation and metrics versions. Every model
runs the same manifest, so all models face the same fire (common random
numbers). `run_evaluation()` builds a fresh policy per episode, gives it only
`Observation` values, saves an authenticated per-model artifact after every
episode and resumes it after an interruption. `pair_results()` reports exactly
which episodes two artifacts can compare. The seeds come from the evaluator;
this repository does not generate scored seeds. See
[docs/evaluation.md](docs/evaluation.md).

```python
from emberbench import build_evaluation_manifest, load_suite, pair_results, run_evaluation

suite = load_suite("scenarios/public/v1")
manifest = build_evaluation_manifest(suite, [3, 1, 4])   # development seeds only
a = run_evaluation(suite, manifest, "model-a", make_policy_a, "model-a.json")
b = run_evaluation(suite, manifest, "model-b", make_policy_b, "model-b.json")
print(pair_results(a, b).paired_count)
```

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
  crowd.py      compute_crowd_diagnostics(): congestion, hotspots, exit load signals
  baselines.py  fair reference baseline agents and their registry
  oracle.py     PRIVILEGED evaluator-side escape bound (not a model)
  suite.py      scenario suite manifests: strict parsing, load_suite() verification
  evaluation.py evaluation manifests, run_evaluation() with resume, pair_results()
  integrity.py  held-out protocol: asset classes, submission identity, provenance, verification
  leakguard.py  path allowlist and leak checks for the repository, wheel and sdist
  canonical.py  canonical JSON and SHA-256 hashing
  cli.py        the `emberbench` command line (also `python -m emberbench`)
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

## Command-line interface

The package installs an `emberbench` command (also `python -m emberbench`). It is a thin,
deterministic, provider-free front end to the library: no network access, no ambient
configuration, no generated seeds and no implicit code loading.

```bash
emberbench suite validate scenarios/public/v1
```

```bash
emberbench evaluation manifest --suite scenarios/public/v1 --seed 0 --seed 1 --seed 2 --out manifest.json
```

```bash
emberbench evaluation run --suite scenarios/public/v1 --manifest manifest.json --model-id my-model --agent my_agents:make_agent --agent-path . --out result.json
```

Every command supports `--json`, refuses to overwrite files and exits with a stable code.
See [docs/cli.md](docs/cli.md).

## Running the tests

Requires Python 3.12+ and has no runtime dependencies. The development tools are pinned
in `tools/dev-constraints.txt`, the same versions CI uses.

```bash
python -m pip install -c tools/dev-constraints.txt -e ".[dev]"
```

```bash
python -m pytest
```

```bash
python -m mypy
```

```bash
python tools/check_public_assets.py
```

The suite covers deterministic replay, two independent simulator instances,
cross-process `PYTHONHASHSEED` independence, serialisation round trips, walls blocking
movement and fire, exit behaviour, explicit rejection of invalid scenarios and
actions, collision rules, termination, ID stability, golden trajectory digests,
every frozen V1 rule, the observation contract (no seed, RNG state or hidden
metadata), the fire trajectory invariance across different actions, and the
canonical public suite (pinned hashes, strict manifest checks and per-scenario
diagnostic properties), and paired evaluation (manifest hashing, seed privacy,
fresh policies per episode, identical fire across models, resume and tamper
rejection, exact pairing), and the stochastic hazard contract (exact ignition
probability, one draw per frontier edge, matched fire across policies and
pinned cross-platform golden fire traces), and crowd diagnostics (the frozen
queue and throughput rules, exact reconciliation with the run record and the
metrics, hotspots in the public crowd scenarios, 50/100/200-person stress
runs and golden diagnostic digests).
Any change to simulator output fails the golden test until you bump
`SIMULATOR_VERSION` on purpose.

## Versions and releases

The package version (`emberbench.__version__`), the benchmark revision
(`BENCHMARK_REVISION`) and the component contract versions are separate; `emberbench
version --json` reports all of them. The rules are in [docs/versioning.md](docs/versioning.md),
the history in [docs/changelog.md](docs/changelog.md) and the release steps, including the
regression digest registry and `tools/check_release.py`, in
[docs/release-checklist.md](docs/release-checklist.md).
