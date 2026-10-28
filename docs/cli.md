# Command-line interface

`CLI_VERSION = "1.1"` (1.1 adds `run`)

The `emberbench` command exposes the library contracts as a deterministic,
provider-free command line. Every command calls the existing library API
(`load_scenario`, `load_suite`, `load_generated_suite`,
`build_evaluation_manifest`, `run_episode`, `verify_record`,
`run_evaluation`, `load_evaluation_result`, `aggregate_result`,
`compare_results`). The CLI adds argument parsing, fail-closed input loading,
no-overwrite atomic output and stable exit codes. It does not add a second
simulator, a second validator or a second artifact format.

The CLI uses only the standard library (`argparse`).

## Entry points

```bash
emberbench version --json
```

```bash
python -m emberbench version --json
```

Both run the same function, `emberbench.cli:main`. The console script is
installed by the package (`[project.scripts]` in `pyproject.toml`).

`python -m` normally puts the current directory first on `sys.path`. The
module entry point removes it, so the current directory never decides which
agent code is imported (see [Agent factories](#agent-factories)).

## Command tree

```
emberbench
  version                     contract versions
  run                         evaluate one model on a frozen environment pack, score it, write a run record
  scenario validate FILE      one scenario file
  suite validate DIR          a suite directory (manifest.json + scenario files)
  generated validate DIR      a frozen generated suite (generation.json + suite/)
  evaluation manifest         build an EvaluationManifest from a suite and ordered seeds
  evaluation run              run every manifest episode with a local agent (resumable)
  evaluation verify           authenticate an EvaluationResult
  episode run                 run one scenario with one seed and a local agent
  replay verify               authenticate one RunRecord by replay
  statistics aggregate        AggregateStatistics of one result
  statistics compare          PairedStatistics of two results (A - B)
```

Every command accepts:

- `--json`: print only the result, as one canonical JSON line on stdout.
- `--debug`: print a Python traceback to stderr on failure.
- `--help`: print the usage of the command.

Options go after the command words (`emberbench suite validate DIR --json`).
Option names are never abbreviated.

Commands that take a suite accept exactly one of:

- `--suite DIR`: a suite directory, loaded with `load_suite`.
- `--generated-suite DIR`: a frozen generated suite, loaded with
  `load_generated_suite`. The freeze record is checked as well.

## Commands

### `version`

```bash
emberbench version --json
```

Prints the package version (`package_version`, from
`src/emberbench/_version.py`, see [versioning.md](versioning.md)) and every
contract version of this build:

- `benchmark_revision` and every benchmark-defining version, exactly the
  entries of `benchmark_versions()` that evaluation provenance pins:
  `simulator_version`, `observation_version`, `action_schema_version`,
  `schema_version` (the scenario schema), `metrics_version`,
  `crowd_diagnostics_version`, `statistics_version`, `generator_version`,
  `generator_config_version`, `generation_manifest_version`,
  `evaluation_manifest_version`, `evaluation_result_version`,
  `scoring_version`, `environment_pack_version`,
  `environment_world_version`, `family_rules_version`,
  `calibration_panel_version`, `difficulty_rules_version`,
  `admission_rules_version`, `composition_version`,
  `selection_rules_version`, `recipe_catalog_version` and
  `suite_freeze_record_version`;
- the tooling contracts that do not change what a score means:
  `cli_version`, `baselines_version`, `integrity_protocol_version`,
  `replay_audit_version` and `animation_format_version`.

The benchmark-defining part is built from `benchmark_versions()`, so a
version that provenance pins is always reported (see
[versioning.md](versioning.md), section 3, and `docs/integrity.md`,
section 12). No field depends on the environment (no paths, platform,
interpreter or install location). These keys were added before any
release, so `CLI_VERSION` stays `1.0`.

### `scenario validate FILE`

```bash
emberbench scenario validate scenarios/public/v1/v1-pub-01-single-exit-sparse.json --json
```

Loads the file with the strict scenario parser. Reports `scenario_id`,
`scenario_hash` (the content hash) and `schema_version`.

### `suite validate DIR`

```bash
emberbench suite validate scenarios/public/v1 --json
```

Loads the suite with `load_suite`: every listed file must exist, no file may
be unlisted, and every pinned hash and derived field must match. Reports
`suite_id`, `suite_version`, `suite_hash`, `visibility`, `seed_role`,
`schema_version`, `scenario_count` and `scenarios` (each `scenario_id` and
`content_hash`, in manifest order).

### `generated validate DIR`

```bash
emberbench generated validate path/to/frozen-batch --json
```

Loads a frozen generated suite with `load_generated_suite` and checks the
freeze record against every scenario. Reports the suite fields above and
`generation_manifest_version`, `generator_version`, `batch_id` and
`config_hash`. The private generator seed is not in the freeze record, so it
is never shown.

### `evaluation manifest`

```bash
emberbench evaluation manifest --suite scenarios/public/v1 --seed 1 --seed 7 --seed 42 --out manifest.json
```

Builds the manifest with `build_evaluation_manifest(suite, seeds)` and writes
its canonical file text (`EvaluationManifest.to_file_text()`, the bytes that
`save_evaluation_manifest` writes).

- `--seed N` is repeatable and required. The order is kept exactly. The CLI
  never generates, sorts or removes seeds.
- A seed is a decimal integer in `[0, 2**64 - 1]`. A duplicate seed is a
  usage error.
- `--out FILE` is required. `--force` replaces an existing file.

Reports `manifest_hash`, `manifest_version`, the suite fields,
`episode_count` and `seed_count`. The seed values are never printed: they
stay in the file the operator asked for.

### `episode run`

```bash
emberbench episode run --scenario scenarios/public/v1/v1-pub-01-single-exit-sparse.json --seed 3 --agent my_agents:make_agent --out record.json
```

Runs one episode through the single simulator path:
`run_episode(scenario, agent_policy(agent), seed=SEED)`. The agent is built
fresh with the factory, receives only `Observation` values and its output is
validated by `validate_actions`. The record is then checked with
`verify_record` and written as canonical JSON plus one LF
(`RunRecord.to_json() + "\n"`).

- `--scenario FILE`, `--seed N`, `--agent MODULE:FACTORY` and `--out FILE`
  are required. The scenario file's development seed is not used.
- `--agent-path DIR` (repeatable) and `--force` are optional.

Reports `scenario_id`, `scenario_hash`, `simulator_version`,
`observation_version`, `record_digest`, `steps` and `termination_reason`.

### `replay verify`

```bash
emberbench replay verify --scenario scenarios/public/v1/v1-pub-01-single-exit-sparse.json --record record.json --json
```

1. The record file must be JSON that `RunRecord.from_dict` accepts and that
   round-trips exactly (no unknown or changed fields). Otherwise: exit 3.
2. `verify_record(scenario, record)` replays the action log. Any version,
   scenario-hash, initial-state or trajectory difference: exit 5.
3. `validate_run_record(scenario, record)` requires a complete, well-formed
   episode. A replayable prefix that stops before termination: exit 3.

Reports `verified: true`, `scenario_id`, `scenario_hash`, `record_digest`,
`simulator_version`, `observation_version`, `steps` and `termination_reason`.
There is no best-effort recovery.

### `evaluation run`

```bash
emberbench evaluation run --suite scenarios/public/v1 --manifest manifest.json --model-id my-model --agent my_agents:make_agent --out result.json
```

Calls `run_evaluation(suite, manifest, model_id, agent_policy_factory(factory), out, resume=...)`
unchanged. Every episode gets a new agent.

- `--manifest FILE`, `--model-id ID`, `--agent MODULE:FACTORY` and
  `--out FILE` are required. `--model-id` must pass the library model ID
  check (non-empty, printable, at most 256 characters, no outer whitespace).
- An existing `--out` file is never overwritten. Without `--resume` it is an
  output conflict (exit 6). With `--resume` it must load and authenticate
  completely and carry the same manifest hash and model ID (otherwise exit 3
  or 6). Completed episodes are not run again. This is the library resume
  behaviour; the CLI does not change it.
- `run_evaluation` saves the artifact atomically after every episode. If the
  agent fails, the last checkpoint stays on disk so that `--resume` can
  continue it. This is the only command that leaves an artifact behind after
  a failure, and that artifact is always a complete, valid checkpoint.

Reports `model_id`, `result_version`, `manifest_hash`, the suite fields,
`result_digest`, `scheduled_episodes`, `completed_episodes` and `complete`.

### `evaluation verify`

```bash
emberbench evaluation verify --suite scenarios/public/v1 --result result.json --manifest manifest.json --require-complete --json
```

Calls `load_evaluation_result(result, suite)`: the embedded manifest must
match the suite and the running versions, and every episode's run record is
authenticated by replay and its metrics are recomputed. A malformed,
duplicate, unknown, non-canonical or tampered episode fails the whole
artifact (exit 3, or exit 5 when the cause is a replay mismatch). No episode
is dropped or repaired.

- `--manifest FILE` also requires this exact evaluation manifest (exit 5 if
  the embedded manifest differs).
- `--require-complete` also requires every scheduled episode (exit 5 if any
  is missing). Without it, a partial result that authenticates is valid.

Reports `verified: true` and the same fields as `evaluation run`.

### `statistics aggregate`

```bash
emberbench statistics aggregate --suite scenarios/public/v1 --result result.json --out stats.json
```

Loads and authenticates the result, then writes
`aggregate_result(result, suite).to_json() + "\n"`.

### `statistics compare`

```bash
emberbench statistics compare --suite scenarios/public/v1 --a a.json --b b.json --out paired.json
```

Loads and authenticates both results, then writes
`compare_results(a, b, suite).to_json() + "\n"`. Deltas are A - B. Results of
different manifests are rejected.

Both statistics commands accept `--resamples N`. Without it the library
default (`DEFAULT_BOOTSTRAP_RESAMPLES`, 2000) is used: the CLI does not pass
the argument at all. Out-of-range values are a usage error. They report
`statistics_version`, `digest` (the artifact's `digest()`), the model IDs,
`manifest_hash` and `resamples`.

## Exit codes

| Code | Name          | Meaning |
|-----:|---------------|---------|
| 0    | `OK`          | Success. |
| 1    | `INTERNAL`    | Unexpected failure (a bug). Rerun with `--debug` for a traceback. |
| 2    | `USAGE`       | Malformed command line or argument value: unknown command or option, missing option, bad seed, duplicate seed, bad `--resamples`, bad `--model-id`, malformed `--agent`, missing `--agent-path` directory. |
| 3    | `VALIDATION`  | An input is missing, unreadable, not JSON, malformed, not canonical, tampered, incomplete where completeness is required, or fails a library check (`ScenarioValidationError`, `SuiteValidationError`, `GenerationError`, `EvaluationError`, `StatisticsInputError`, `MetricsInputError`). |
| 4    | `AGENT`       | The agent module cannot be imported, the factory is missing or not callable, the factory raises or returns a non-agent or a reused agent, `act` raises, or the model output is rejected (`ModelOutputError`). |
| 5    | `MISMATCH`    | A replay does not reproduce the record (`ReplayMismatchError`, also when it is the cause of another error), or an evaluation result does not match the required manifest or completeness. |
| 6    | `OUTPUT`      | The output file exists (without `--force` or `--resume`), is a directory, is also an input file, its parent directory does not exist, or it cannot be written. |
| 130  | `INTERRUPTED` | Interrupted (`KeyboardInterrupt`). |

The codes are `emberbench.cli.ExitCode` and are pinned by tests.

A Python traceback is never printed unless `--debug` is given.

## Output

### JSON mode (`--json`)

- On success, stdout holds exactly one line: the result object as canonical
  JSON (sorted keys, no whitespace, ASCII). Nothing else is printed to stdout.
- Every result has a `command` field. Results never include input or output
  paths, timestamps, hostnames or environment details.
- On failure, stdout is empty and stderr holds exactly one line:
  `{"error":{"exit_code":N,"kind":"...","message":"..."}}`. With `--debug`
  the traceback follows on stderr.
- There are no ANSI escape codes or colours.

Usage errors found while parsing the command line, before the command is
known, are also reported in this JSON form when `--json` is present.

### Human mode

A short `key: value` list on stdout. Errors go to stderr as
`emberbench: error: MESSAGE`.

Error messages are deterministic for the same inputs. Library messages can
contain the input paths as given on the command line.

## Files

### Inputs are never written

Validation, verification and statistics commands only read their inputs.
An output path that is the same file as an input is refused, even with
`--force`.

### Output rules

- An existing output file is never replaced unless `--force` is given.
  `evaluation run` has no `--force`: an existing result can only be resumed.
- Parent directories are never created. A missing parent is exit 6.
- The output is written to a temporary file in the same directory, flushed
  and fsynced. With `--force` the temporary file is then renamed over the
  output (`os.replace`). Without `--force` it is hard-linked to the output
  path. The link fails if the file exists, so a file that another process
  creates in the meantime is never replaced either.
- If the filesystem has no hard links, the CLI creates the output with
  `O_CREAT | O_EXCL` (this also fails if the file exists), writes the bytes,
  flushes and fsyncs. On such a filesystem the output is visible while it
  is written. If that write fails, the CLI removes the file, but only if it
  is still the file that the CLI created. There is never a check-then-replace
  step.
- On any failure, no output file and no temporary file are left behind
  (`evaluation run` checkpoints are the documented exception above).
- Inputs are validated and the output path is checked before an agent is
  imported or an episode runs.

### Artifact bytes

| Command | File content |
|---------|--------------|
| `evaluation manifest` | `EvaluationManifest.to_file_text()` (indented JSON, LF) |
| `episode run` | `RunRecord.to_json() + "\n"` |
| `evaluation run` | `EvaluationResult.to_file_text()` (written by `run_evaluation`) |
| `statistics aggregate` | `AggregateStatistics.to_json() + "\n"` |
| `statistics compare` | `PairedStatistics.to_json() + "\n"` |

All artifacts are ASCII with LF line endings on every platform. They hold no
absolute paths and no timestamps. A record or statistics `digest` is the
SHA-256 of the file without its final LF.

## Agent factories

`episode run` and `evaluation run` import the operator's agent code. This is
the only place where the CLI imports code that is not part of EMBERBench, and
it happens only when `--agent` is given.

```
--agent MODULE:FACTORY
```

- `MODULE` is a dotted Python module path (`my_pkg.agents`).
- `FACTORY` is an attribute path in that module (`make_agent` or
  `Agents.make`).
- Both parts use ASCII identifiers only. Anything else is a usage error
  (exit 2).

The CLI imports exactly `MODULE` with `importlib.import_module` and reads
`FACTORY` from it. There is no entry-point discovery, plugin registry,
fallback module, network location or search of the current directory.
The module must be importable from `sys.path` as the interpreter sets it, or
from a directory given explicitly with `--agent-path DIR` (repeatable; the
directories are put first, in the order given, for this command only).

The factory must satisfy the `AgentFactory` contract
(`docs/model-interface.md`):

- It is a zero-argument callable.
- Every call returns a new object with an `act(observation)` method. A
  returned agent that was already used for an earlier episode is rejected.
- `act` receives only the `Observation`. It never receives the scenario,
  seed, manifest, episode index or run record.

Example agent module:

```python
# my_agents.py
from emberbench import Action, PolicyAgent


def make_agent():
    return PolicyAgent(lambda obs: {p.id: Action.STAY for p in obs.active})
```

```bash
emberbench episode run --scenario scenarios/public/v1/v1-pub-01-single-exit-sparse.json --seed 3 --agent my_agents:make_agent --agent-path . --out record.json
```

Importing a module runs its code. Use `--agent` only with code that you trust.

## `run`: evaluate one model on a frozen environment pack

```bash
emberbench run --suite PACK_DIR --model typesafe/jev-1.14 --expected-digest HEX --out RUN_DIR
```

```bash
emberbench run --suite PACK_DIR --baseline congestion-aware-v1 --out RUN_DIR
```

`--suite` is an environment pack directory (`pack.json`, the suite and the
evaluation manifest). The pack is verified before any episode runs; with
`--expected-digest` its digest must match, and with `--commitment HEX
--salt-env VAR` it must open the published suite commitment (the salt is read
from the environment variable, never from the command line).

Exactly one model source:

- `--model NAME`: a remote model, run through the harness
  ([harness-protocol.md](harness-protocol.md)). Defaults: the decision API
  style and the OpenRouter decisions endpoint (`--api`, `--base-url`), the key
  in `EMBERBENCH_API_KEY` (`--key-env`), the canonical full-joint protocol and
  at most 2 re-asks. `--joint-mode fixed_chunk --chunk-size N` is the declared
  variant for a provider with a request cap; its model ID ends in `-chunkN`.
  `--budget-usd` stops the run (resumable) at a spend limit.
- `--baseline ID`: a registered baseline.
- `--agent MODULE:FACTORY --model-id ID`: a local agent (as in
  `evaluation run`).

`RUN_DIR` must be outside the source repository; a rerun with the same model
resumes. It receives `results/<model_id>.json`, `scores/`, `accounting/` and
`runs/<model_id>.json`, the run record (`RUN_RECORD_VERSION = "1.1"`): every
`benchmark_versions()` entry, the pack digest, manifest hash, seeds (or only
their SHA-256 with `--private`), the commitment check, the model and its
settings, the harness protocol and the SHA-256 of the harness module, start
and end time, calls, tokens, cost and runtime, the accounting summary, the
result SHA-256, the metrics and (1.1) `refusals`: the count, rate and
indexes of refused episodes. A refused episode
([harness-protocol.md](harness-protocol.md), section 3) is listed in
`refusals/<model_id>.json`, scores 0 as the failure `refusal` and is not run
again on resume. A remote model also gets `ledger/` and
`cache.jsonl`; the cache holds the model-visible state text, so a private run
directory stays with the evaluator.

`--private` (hidden suites) needs `--expected-digest` or `--commitment`. The
run record and the printed summary then hold no pack ID, scenario ID or seed,
and harness errors hold no answer text. Deterministic baselines reproduce
their result bytes exactly.

### Docker

The `Dockerfile` at the repository root builds the runner on a base image
pinned by digest (`python:3.12.16-slim-bookworm`); the package has no runtime
dependencies, and the build tools come from `tools/dev-constraints.txt`.

```text
docker build -t emberbench .
docker run --rm -v PACK_DIR:/suite:ro -v RUN_DIR:/out -e EMBERBENCH_API_KEY emberbench run --suite /suite --model typesafe/jev-1.14 --expected-digest HEX --out /out
docker build --target test -t emberbench-test .
docker run --rm emberbench-test
```

The pack is mounted read-only; nothing of a private suite is part of the
image. The `test` image runs the test suite in a clean environment.

## Configuration, determinism and network policy

- There is no configuration file, no home-directory configuration, no
  `.env` file and no environment variable that changes a result. Every value
  that affects a benchmark result comes from an explicit argument, an
  explicit input file or a versioned library default.
- The current directory does not change results. Relative paths are resolved
  against it, as usual, and nothing else is taken from it.
- The same command with the same input files gives byte-identical output
  files and stdout, under any `PYTHONHASHSEED`, from any current directory and
  on Linux, Windows and macOS. The order of options does not matter. The
  order of `--seed` options does: it is the manifest order.
- The CLI core never opens a network connection: it does not download
  models, contact providers, check for updates or fetch URLs. A test runs
  every command under an audit hook that fails on any socket or URL event.
  An agent that the operator loads with `--agent` may use the network itself.
  That is outside the CLI core and is the operator's explicit choice.
- `emberbench run --model` is the one command that contacts a model API and
  reads an environment variable (the API key named by `--key-env`, and the
  commitment salt named by `--salt-env`). Its code is in `emberbench.run`
  and `emberbench.harness`; `run --baseline` and `run --agent` stay
  provider-free.

## Examples (public development assets)

```bash
emberbench suite validate scenarios/public/v1
```

```bash
emberbench evaluation manifest --suite scenarios/public/v1 --seed 0 --seed 1 --seed 2 --out manifest.json --json
```

```bash
emberbench evaluation run --suite scenarios/public/v1 --manifest manifest.json --model-id stay --agent my_agents:make_agent --agent-path . --out stay.json
```

```bash
emberbench evaluation verify --suite scenarios/public/v1 --result stay.json --manifest manifest.json --require-complete
```

```bash
emberbench statistics aggregate --suite scenarios/public/v1 --result stay.json --out stay-stats.json
```

The public seeds `0, 1, 2` give the evaluation manifest hash
`7b9aae45e5e2828f881b2f14d8e5585627e5f8bdbb137a1ed2331bf10ce8c876`.
Development seeds are for development only; scored seed schedules belong to
the evaluator (`docs/evaluation.md`).
