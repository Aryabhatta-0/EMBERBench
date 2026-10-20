# Versioning and release policy

EMBERBench has three kinds of version. They answer different questions and
change for different reasons. Never infer one from another.

| Kind | Question it answers | Where it lives |
|------|---------------------|----------------|
| Package version | Which build of the Python distribution is installed? | `src/emberbench/_version.py` (`emberbench.__version__`) |
| Benchmark revision | What does a score mean? Which scores may be compared? | `BENCHMARK_REVISION` in `src/emberbench/integrity.py` |
| Component contract versions | Which exact rules or file format produced this artifact? | One constant per component, listed below |

`emberbench version --json` reports all three: `package_version`,
`benchmark_revision` and every component version. The output depends on no
environment value. It reads the package version from the source, not from
installed metadata.

## 1. Package version

- **One source.** `src/emberbench/_version.py` holds `__version__`.
  `pyproject.toml` reads it (`dynamic = ["version"]`,
  `[tool.setuptools.dynamic]`), so the wheel and sdist metadata, the
  runtime attribute and the CLI cannot disagree. Do not write the version in
  any other file.
- **Format.** `MAJOR.MINOR.PATCH` (Semantic Versioning 2.0.0), with an
  optional pre-release suffix (`1.0.0rc1`) that is also a valid
  [PEP 440](https://peps.python.org/pep-0440/) version.
- **Not in any artifact.** No run record, manifest, result, statistics file,
  generated suite or provenance record holds the package version. A package
  release alone therefore never changes a pinned hash.
- **Not the benchmark revision.** Package `2.3.1` can implement benchmark
  revision `1`. A new benchmark revision always needs a new package MAJOR
  version, but a new package MAJOR version does not need a new benchmark
  revision.

The current development version is `0.1.0`. Changing it is a release step
(see [release-checklist.md](release-checklist.md)), not part of feature work.

### Release rules

| Release | Allowed changes | Pinned hashes |
|---------|-----------------|---------------|
| **PATCH** (`x.y.Z`) | Bug fixes that change no artifact byte, documentation, packaging, CI and development tooling, performance work, clearer error messages. | None may change. |
| **MINOR** (`x.Y.0`) | Backward-compatible additions: new library API, new CLI commands or options, new baselines under new IDs, new diagnostic artifacts, a component version bump that does not change what any existing score means (for example `CLI_VERSION`, `BASELINES_VERSION`, `INTEGRITY_PROTOCOL_VERSION` for a new optional record field). | Only the hashes owned by a bumped component, as listed in its changelog entry. |
| **MAJOR** (`X.0.0`) | A new `BENCHMARK_REVISION`, or an incompatible change to the public Python API or to CLI behaviour (removed command, changed exit code, changed artifact bytes of an existing command). | Any, as listed in the changelog entry. |

A change that would need a MAJOR release but is not ready for one waits.
Do not ship it in a MINOR or PATCH release.

## 2. Benchmark revision

`BENCHMARK_REVISION = "1"` is V1. The revision defines what a score means.
Scores and provenance from different revisions are never compared, and
provenance of another revision fails verification
([integrity.md](integrity.md), section 12).

A new benchmark revision is **mandatory** for:

- a simulator semantic change (`SIMULATOR_VERSION`): any change to a run
  record for the same scenario, seed and actions;
- an observation, action or scenario schema change (`OBSERVATION_VERSION`,
  `ACTION_SCHEMA_VERSION`, `SCHEMA_VERSION`);
- a metric or statistic definition change (`METRICS_VERSION`,
  `STATISTICS_VERSION`, `CROWD_DIAGNOSTICS_VERSION`);
- a generator semantics or config format change (`GENERATOR_VERSION`,
  `GENERATOR_CONFIG_VERSION`), because held-out batches come from it;
- a change to the content of the canonical public suite that scored
  comparisons use (a new `suite_version`);
- discovered held-out leakage that the current protocol permits, or any
  other flaw in the evaluation protocol;
- a material bug that affects ranking or comparison, even when the fix is
  small.

A new revision is **not** needed for: package-only changes, documentation,
CLI presentation, new or changed reference baselines (they are not scored
submissions; bump `BASELINES_VERSION`), replay and audit tooling that does
not change what `verify_record` accepts, and a regenerated held-out batch
under the same code (see [integrity.md](integrity.md), section 12).

## 3. Component contract versions

Each component owns one version. Its document gives the exact bump rule.

| Constant | Value | Owner | Owned hashes and artifacts |
|----------|-------|-------|----------------------------|
| `SIMULATOR_VERSION` | `0.2.0` | `src/emberbench/__init__.py`, [simulation-contract.md](simulation-contract.md) | Run record bytes and digests, golden trajectories, stochastic golden traces |
| `SCHEMA_VERSION` | `1.0` | `src/emberbench/scenario.py`, [scenario-schema.md](scenario-schema.md) | Scenario content hashes |
| `OBSERVATION_VERSION` | `1.0` | `src/emberbench/observation.py`, [v1-fairness-spec.md](v1-fairness-spec.md) | Run record header, model input |
| `ACTION_SCHEMA_VERSION` | `1.0` | `src/emberbench/agent.py`, [model-interface.md](model-interface.md) | Model output format |
| `METRICS_VERSION` | `1.0` | `src/emberbench/metrics.py`, [metrics.md](metrics.md) | Episode metrics |
| `CROWD_DIAGNOSTICS_VERSION` | `1.0` | `src/emberbench/crowd.py`, [crowd-diagnostics.md](crowd-diagnostics.md) | Crowd diagnostic digests |
| `STATISTICS_VERSION` | `1.0` | `src/emberbench/statistics.py`, [statistics.md](statistics.md) | Aggregate and paired statistics digests |
| `EVALUATION_MANIFEST_VERSION` | `1.0` | `src/emberbench/evaluation.py`, [evaluation.md](evaluation.md) | Evaluation manifest hash and file bytes |
| `EVALUATION_RESULT_VERSION` | `1.0` | `src/emberbench/evaluation.py`, [evaluation.md](evaluation.md) | Evaluation result files |
| `GENERATOR_VERSION` | `1.0.0` | `src/emberbench/generator.py`, [procedural-generation.md](procedural-generation.md) | Generated scenarios, generation and generated suite hashes |
| `GENERATOR_CONFIG_VERSION` | `1.0` | `src/emberbench/generator.py` | Generator config hash |
| `GENERATION_MANIFEST_VERSION` | `1.0` | `src/emberbench/generator.py` | `generation.json` format |
| `GENERATED_SUITE_VERSION` | `1.0.0` | `src/emberbench/generator.py` | `suite_version` of every generated suite manifest; it moves with `GENERATOR_VERSION` and is not reported separately |
| `BASELINES_VERSION` | `1.0` | `src/emberbench/baselines.py`, [baselines.md](baselines.md) | Baseline config and result digests |
| `INTEGRITY_PROTOCOL_VERSION` | `1.0` | `src/emberbench/integrity.py`, [integrity.md](integrity.md) | Provenance record format and digests |
| `CLI_VERSION` | `1.0` | `src/emberbench/cli.py`, [cli.md](cli.md) | Command tree, exit codes, JSON output keys |
| `BENCHMARK_REVISION` | `1` | `src/emberbench/integrity.py`, [integrity.md](integrity.md) | Meaning of every score |

The public suite has its own `suite_version` (`1.0.0`) in
`scenarios/public/v1/manifest.json`. A released suite is never edited; a
changed suite is a new suite version ([scenario-suite.md](scenario-suite.md)).

Rules for every component version:

1. **Bump by hand, on purpose.** No tool, script or CI step bumps a
   component version. A failing pinned-hash test is the signal; the fix is a
   deliberate bump in the same change, never an edited pin alone.
2. **Bump only the owner.** A change in one component does not bump the
   others. A package release does not bump any component.
3. **Report it.** Every component version appears in
   `emberbench version --json` (except `GENERATED_SUITE_VERSION`, see the
   table) and in the table above. `tests/test_release.py` fails when a new
   `*_VERSION` constant is missing from either.
4. **Record it.** The changelog entry names every bumped component, its old
   and new value, and every pinned hash that changed.

## 4. When pinned hashes may change

The pinned hashes are listed in the regression digest registry
(`tests/fixtures/release/digest_registry.json`, see
[release-checklist.md](release-checklist.md)). A pinned hash may change only
when **all** of these hold:

- the component that owns it (table above) has a new version in the same
  change;
- the change is not a PATCH release;
- the registry entry, the test pin that it mirrors and the changelog are
  updated together, and the changelog says why;
- if the change alters what a score means, `BENCHMARK_REVISION` is bumped
  too (section 2).

Platform, Python version (within `requires-python`), `PYTHONHASHSEED`,
locale, working directory, path spelling and line endings of accepted inputs
never change a pinned hash. If one of them does, that is a bug.

## 5. Components added in parallel work

A branch that adds a component with its own version (for example replay or
audit artifacts) must, before it merges or when release work syncs with it:

- add the constant to the table in section 3 and to `emberbench version`;
- add its frozen digests to the registry;
- add a changelog entry.
