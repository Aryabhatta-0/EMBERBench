# Changelog

Notable changes to EMBERBench. The version policy is in
[versioning.md](versioning.md). Each release entry names the package version,
the benchmark revision, every component version that changed and every pinned
hash that changed.

Entries reference the pull requests in `Aryabhatta-0/EMBERBench` that
introduced each part. Dates are added only when a release is tagged.

## Unreleased: V1 (package `0.1.0`, benchmark revision `1`)

No version of EMBERBench has been released yet. This entry collects the V1
milestones that are in `main`. Release them by following
[release-checklist.md](release-checklist.md).

Component versions at this point: `SIMULATOR_VERSION` `0.2.0`,
`SCHEMA_VERSION` `1.0`, `OBSERVATION_VERSION` `1.0`,
`ACTION_SCHEMA_VERSION` `1.0`, `METRICS_VERSION` `1.0`,
`CROWD_DIAGNOSTICS_VERSION` `1.0`, `STATISTICS_VERSION` `1.0`,
`EVALUATION_MANIFEST_VERSION` `1.0`, `EVALUATION_RESULT_VERSION` `1.0`,
`GENERATOR_VERSION` `1.0.0`, `GENERATOR_CONFIG_VERSION` `1.0`,
`GENERATION_MANIFEST_VERSION` `1.0`, `GENERATED_SUITE_VERSION` `1.0.0`,
`BASELINES_VERSION` `1.0`, `INTEGRITY_PROTOCOL_VERSION` `1.0`,
`CLI_VERSION` `1.0`. Public suite `suite_version` `1.0.0`.

### Milestones

- **Deterministic core** (Phase 1, PR #2): benchmark contract, grid
  scenarios, the deterministic simulator, run records with digests, and
  replay.
- **Fairness contract** (Phase 2, PR #17): the frozen V1 fairness and
  semantics specification ([v1-fairness-spec.md](v1-fairness-spec.md)).
  `SIMULATOR_VERSION` went from `0.1.0` to `0.2.0`: the run record header
  gained `observation_version`; the dynamics did not change.
- **Metrics** (Phase 3, PR #18): auditable per-episode metrics
  ([metrics.md](metrics.md)).
- **Fixed suite** (Phase 4, PR #19): the canonical V1 public scenario suite
  with pinned hashes ([scenario-suite.md](scenario-suite.md)).
- **Paired evaluation** (Phase 5, PR #20): exact scenario + seed manifests and
  resumable, paired evaluation runs ([evaluation.md](evaluation.md)).
- **Stochastic hazard** (Phase 6, PR #21): the matched V1 stochastic fire
  contract with cross-platform golden traces
  ([simulation-contract.md](simulation-contract.md)).
- **Crowd diagnostics** (Phase 7, PR #22): queue and throughput diagnostics
  ([crowd-diagnostics.md](crowd-diagnostics.md)).
- **Model interface** (Phase 7, PR #23): the provider-independent agent,
  observation and action contract ([model-interface.md](model-interface.md)).
- **Statistics** (Phase 8, PR #24): deterministic aggregate and paired
  statistics ([statistics.md](statistics.md)).
- **Generator** (Phase 8, PR #25): the procedural unseen scenario generator
  ([procedural-generation.md](procedural-generation.md)).
- **Integrity** (Phase 9, PR #26, issue #14): the fail-closed hidden-test integrity
  protocol, provenance records and the public-asset leak guard
  ([integrity.md](integrity.md)).
- **Baselines** (Phase 9, PR #27): deterministic reference baseline agents
  ([baselines.md](baselines.md)).
- **CLI** (Phase 9, PR #28): the reproducible, provider-free `emberbench`
  command ([cli.md](cli.md)).

### Release hardening (Phase 10, issue #16)

- One authoritative package version in `src/emberbench/_version.py`, exposed
  as `emberbench.__version__` and as `package_version` in
  `emberbench version`. No component version changed and no pinned hash
  changed.
- Version and release policy ([versioning.md](versioning.md)) and release
  checklist ([release-checklist.md](release-checklist.md)).
- Development tooling pinned by `tools/dev-constraints.txt`, with bounded
  ranges in `pyproject.toml`.
- CI packaging job: two clean builds compared, wheel built from the sdist
  offline, clean-venv install, console script and public-suite smoke.
- Regression digest registry (`tests/fixtures/release/digest_registry.json`)
  checked in fresh interpreters.
