# Versioning and release policy

EMBERBench has three kinds of version. They answer different questions and
change for different reasons. Never infer one from another.

| Kind | Question it answers | Where it lives |
|------|---------------------|----------------|
| Package version | Which build of the Python distribution is installed? | `src/emberbench/_version.py` (`emberbench.__version__`) |
| Benchmark revision | What does a score mean? Which scores may be compared? | `BENCHMARK_REVISION` in `src/emberbench/integrity.py` |
| Component contract versions | Which exact rules or file format produced this artifact? | One constant per component, listed below |

`emberbench version --json` reports all three: `package_version`,
`benchmark_revision` and every component version (section 3). The output
depends on no environment value. It reads the package version from the
source, not from installed metadata.

## 1. Package version

- **One source.** `src/emberbench/_version.py` holds `__version__`.
  `pyproject.toml` reads it (`dynamic = ["version"]`,
  `[tool.setuptools.dynamic]`), so the wheel and sdist metadata, the
  runtime attribute and the CLI cannot disagree. Do not write the version in
  any other file.
- **Format.** `MAJOR.MINOR.PATCH` (Semantic Versioning 2.0.0), with an
  optional pre-release suffix (`1.0.0rc1`) that is also a valid
  [PEP 440](https://peps.python.org/pep-0440/) version.
- **Not in any benchmark artifact.** No run record, manifest, result,
  statistics file, score report, environment pack, generated or composed
  suite or provenance record holds the package version. A package release
  alone therefore never changes a benchmark hash. The only registry entry
  that moves with it is `cli.version-json`, the bytes of
  `emberbench version --json` itself.
- **Not the benchmark revision.** The two are independent numbers. Package
  `0.1.0` implements benchmark revision `2`. A benchmark revision never
  sets the package version, and a package release never sets the revision.

**Current decision.** The package version stays `0.1.0`. No version of
EMBERBench has been tagged or published, so `0.1.0` is still the first
release. Benchmark revision 2 is a pre-release revision: it replaced
revision 1 before anything was released, and no published package or score
used revision 1. Choosing a different first version (for example `1.0.0`
for the V1 benchmark) is a maintainer decision at tag time; until one is
recorded, the version stays `0.1.0`. Changing it is a release step (see
[release-checklist.md](release-checklist.md)), not part of feature work.

### Release rules

| Release | Allowed changes | Pinned hashes |
|---------|-----------------|---------------|
| **PATCH** (`x.y.Z`) | Bug fixes that change no artifact byte, documentation, packaging, CI and development tooling, performance work, clearer error messages. | None may change. |
| **MINOR** (`x.Y.0`) | Backward-compatible additions: new library API, new CLI commands or options, new baselines under new IDs, new diagnostic artifacts, a tooling version bump that does not change what any score means (`CLI_VERSION`, `BASELINES_VERSION`, `INTEGRITY_PROTOCOL_VERSION` for a new optional record field, `REPLAY_AUDIT_VERSION`, `ANIMATION_FORMAT_VERSION`). | Only the hashes owned by a bumped component, as listed in its changelog entry. |
| **MAJOR** (`X.0.0`) | A new `BENCHMARK_REVISION` after the first release, or an incompatible change to the public Python API or to CLI behaviour (removed command, changed exit code, changed artifact bytes of an existing command). | Any, as listed in the changelog entry. |

While the package is below `1.0.0`, a change that needs a MAJOR release
increments MINOR instead (`0.Y.0`), as Semantic Versioning allows for
`0.y.z`. A change that would need a MAJOR (or pre-1.0 MINOR) release but is
not ready for one waits. Do not ship it in a smaller release.

## 2. Benchmark revision

`BENCHMARK_REVISION = "2"` is the V1 benchmark. The revision defines what a
score means. Scores and provenance from different revisions are never
compared, and provenance of another revision fails verification
([integrity.md](integrity.md), section 12).

Revision 2 is one pre-release step from revision 1. It takes in the official
EMBER Score (`SCORING_VERSION` `1.0`), environment packs and unique-world
identity (`ENVIRONMENT_PACK_VERSION`, `ENVIRONMENT_WORLD_VERSION`), generator
1.1 (`GENERATOR_VERSION` `1.1.0`, config and generation manifest `1.1`) and
the suite-definition contracts (family rules, calibration panel, difficulty
and admission rules, composition, selection, recipe catalog, suite freeze
record). Revision 1 was never released.

A new benchmark revision is **mandatory** for:

- a change to any version that `benchmark_versions()` returns (the
  "In provenance" column of section 3). The full list, with the reason for
  each, is in [integrity.md](integrity.md), section 12;
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
"In provenance" is **yes** exactly for the versions that
`benchmark_versions()` returns: evaluation provenance records them and
`verify_evaluation_provenance` fails closed on any difference. A change to
one of them needs a new `BENCHMARK_REVISION` (section 2). The other
versions are tooling contracts: they never change what a score means.

| Constant | Value | In provenance | Owner | Owned hashes and artifacts |
|----------|-------|---------------|-------|----------------------------|
| `BENCHMARK_REVISION` | `2` | yes | `src/emberbench/integrity.py`, [integrity.md](integrity.md) | Meaning of every score |
| `SIMULATOR_VERSION` | `0.2.0` | yes | `src/emberbench/__init__.py`, [simulation-contract.md](simulation-contract.md) | Run record bytes and digests, golden trajectories, stochastic golden traces |
| `OBSERVATION_VERSION` | `1.0` | yes | `src/emberbench/observation.py`, [v1-fairness-spec.md](v1-fairness-spec.md) | Run record header, model input |
| `ACTION_SCHEMA_VERSION` | `1.0` | yes | `src/emberbench/agent.py`, [model-interface.md](model-interface.md) | Model output format |
| `SCHEMA_VERSION` | `1.0` | yes | `src/emberbench/scenario.py`, [scenario-schema.md](scenario-schema.md) | Scenario content hashes |
| `METRICS_VERSION` | `1.0` | yes | `src/emberbench/metrics.py`, [metrics.md](metrics.md) | Episode metrics |
| `CROWD_DIAGNOSTICS_VERSION` | `1.0` | yes | `src/emberbench/crowd.py`, [crowd-diagnostics.md](crowd-diagnostics.md) | Crowd diagnostic digests |
| `STATISTICS_VERSION` | `1.0` | yes | `src/emberbench/statistics.py`, [statistics.md](statistics.md) | Aggregate and paired statistics digests |
| `GENERATOR_VERSION` | `1.1.0` | yes | `src/emberbench/generator.py`, [procedural-generation.md](procedural-generation.md) | Generated scenarios, generation and generated suite hashes |
| `GENERATOR_CONFIG_VERSION` | `1.1` | yes | `src/emberbench/generator.py` | Generator config hash |
| `GENERATION_MANIFEST_VERSION` | `1.1` | yes | `src/emberbench/generator.py` | `generation.json` format and generation hash |
| `EVALUATION_MANIFEST_VERSION` | `1.0` | yes | `src/emberbench/evaluation.py`, [evaluation.md](evaluation.md) | Evaluation manifest hash and file bytes |
| `EVALUATION_RESULT_VERSION` | `1.0` | yes | `src/emberbench/evaluation.py`, [evaluation.md](evaluation.md) | Evaluation result files |
| `SCORING_VERSION` | `1.0` | yes | `src/emberbench/scoring.py`, [scoring.md](scoring.md) | Scoring constants digest, episode scores, score report digests |
| `ENVIRONMENT_PACK_VERSION` | `1.0` | yes | `src/emberbench/environment_pack.py`, [environment-pack.md](environment-pack.md) | `pack.json` and the pack digest |
| `ENVIRONMENT_WORLD_VERSION` | `1.0` | yes | `src/emberbench/worlds.py`, [evaluation-suite-design.md](evaluation-boundary.md) | Environment world digests (unique-world identity) |
| `FAMILY_RULES_VERSION` | `1.0` | yes | `src/emberbench/families.py`, [generator.md](generator.md) | Family rule record, family tags |
| `CALIBRATION_PANEL_VERSION` | `1.0` | yes | `src/emberbench/calibration.py`, [difficulty-calibration.md](difficulty-calibration.md) | Calibration rule record (panel) |
| `DIFFICULTY_RULES_VERSION` | `1.0` | yes | `src/emberbench/calibration.py`, [difficulty-calibration.md](difficulty-calibration.md) | Calibration rule record (bands) |
| `ADMISSION_RULES_VERSION` | `1.0` | yes | `src/emberbench/calibration.py`, [difficulty-calibration.md](difficulty-calibration.md) | Calibration rule record (admission) |
| `COMPOSITION_VERSION` | `1.0` | yes | `src/emberbench/composition.py`, [generator.md](generator.md) | Composition rule set digest |
| `SELECTION_RULES_VERSION` | `1.0` | yes | `src/emberbench/composition.py`, [generator.md](generator.md) | Composition rule set digest (selection) |
| `RECIPE_CATALOG_VERSION` | `1.0` | yes | `src/emberbench/composition.py`, [generator.md](generator.md) | Composition rule set digest (recipes) |
| `SUITE_FREEZE_RECORD_VERSION` | `1.0` | yes | `src/emberbench/composition.py`, [generator.md](generator.md) | Suite freeze record format |
| `CLI_VERSION` | `1.0` | no | `src/emberbench/cli.py`, [cli.md](cli.md) | Command tree, exit codes, JSON output keys, CLI artifact bytes |
| `BASELINES_VERSION` | `1.0` | no | `src/emberbench/baselines.py`, [baselines.md](baselines.md) | Baseline config and result digests |
| `INTEGRITY_PROTOCOL_VERSION` | `1.0` | no | `src/emberbench/integrity.py`, [integrity.md](integrity.md) | Provenance record format and digests |
| `REPLAY_AUDIT_VERSION` | `1.0` | no | `src/emberbench/audit.py`, [replay-visualization.md](replay-visualization.md) | Replay audit digests |
| `ANIMATION_FORMAT_VERSION` | `1.0` | no | `src/emberbench/visualization.py`, [replay-visualization.md](replay-visualization.md) | Animation JSON digests |
| `GENERATED_SUITE_VERSION` | `1.0.0` | no | `src/emberbench/generator.py` | `suite_version` label of every generated suite manifest; content identity is the generated suite hash, owned by `GENERATOR_VERSION`. Not reported separately |
| `COMPOSED_SUITE_VERSION` | `1.0.0` | no | `src/emberbench/composition.py` | `suite_version` label of every composed suite manifest; content identity is owned by the composition versions. Not reported separately |

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
   `emberbench version --json` (except the two `suite_version` labels, see
   the table) and in the table above. The benchmark-defining part of
   `emberbench version` is built from `benchmark_versions()`, so a version
   that provenance pins is reported by construction.
4. **Record it.** The changelog entry names every bumped component, its old
   and new value, and every pinned hash that changed.

`tests/test_release.py` enforces this. It imports every `emberbench` module
and collects each `*_VERSION` string constant and `BENCHMARK_REVISION`. It
fails when one of them is missing from `emberbench version` (unless it is a
listed `suite_version` label), from the table above (with the right
"In provenance" value), from the changelog, or from the release checklist,
and when a `benchmark_versions()` entry has no owning constant.

## 4. When pinned hashes may change

The pinned hashes are listed in the regression digest registry
(`tests/fixtures/release/digest_registry.json`, see
[release-checklist.md](release-checklist.md)). A pinned hash may change only
when **all** of these hold:

- the component that owns it (table above, and the `owner` field of the
  registry entry) has a new version in the same change;
- the change is not a PATCH release;
- the registry entry, the test pin that it mirrors and the changelog are
  updated together, and the changelog says why;
- if the change alters what a score means, `BENCHMARK_REVISION` is bumped
  too (section 2).

Platform, Python version (within `requires-python`), `PYTHONHASHSEED`,
locale, time zone, working directory, path spelling and line endings of
accepted inputs never change a pinned hash. If one of them does, that is a
bug. The one deliberate exception is an environment pack: it freezes the
exact bytes of its files ([environment-pack.md](environment-pack.md)), so
its digest, and a score report that names it, follow the file bytes. The
repository's `.gitattributes` keeps those files LF on every platform.

## 5. Components added in parallel work

A branch that adds a component with its own version (for example a new
artifact format) must, before it merges or when release work syncs with it:

- add the constant to the table in section 3 and, if it defines what a
  score means, to `benchmark_versions()` (then `emberbench version` reports
  it automatically); otherwise add it to the tooling part of
  `version_info()` in `src/emberbench/cli.py`;
- add it to the version review in [release-checklist.md](release-checklist.md);
- add its frozen digests to the registry;
- add a changelog entry.
