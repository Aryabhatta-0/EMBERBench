# Changelog

Notable changes to EMBERBench. The version policy is in
[versioning.md](versioning.md). Each release entry names the package version,
the benchmark revision, every component version that changed and every pinned
hash that changed.

Entries reference the pull requests in `Aryabhatta-0/EMBERBench` that
introduced each part. Dates are added only when a release is tagged.

## Unreleased: V1 (package `0.1.0`, benchmark revision `2`)

No version of EMBERBench has been released yet. This entry collects the V1
milestones that are in `main`. Release them by following
[release-checklist.md](release-checklist.md). The package version stays
`0.1.0` until the maintainers record a different first release version
([versioning.md](versioning.md), section 1). License: Apache License 2.0
(`Apache-2.0`), selected by the maintainers: `LICENSE` at the repository
root, and `license = "Apache-2.0"` with `license-files = ["LICENSE"]` in
`pyproject.toml`, so the wheel and sdist carry `License-Expression`,
`License-File` and the license text.

Benchmark-defining versions (all pinned by evaluation provenance through
`benchmark_versions()`): `SIMULATOR_VERSION` `0.2.0`,
`OBSERVATION_VERSION` `1.0`, `ACTION_SCHEMA_VERSION` `1.0`,
`SCHEMA_VERSION` `1.0`, `METRICS_VERSION` `1.0`,
`CROWD_DIAGNOSTICS_VERSION` `1.0`, `STATISTICS_VERSION` `1.0`,
`GENERATOR_VERSION` `1.1.0`, `GENERATOR_CONFIG_VERSION` `1.1`,
`GENERATION_MANIFEST_VERSION` `1.1`, `EVALUATION_MANIFEST_VERSION` `1.0`,
`EVALUATION_RESULT_VERSION` `1.0`, `SCORING_VERSION` `1.0`,
`ENVIRONMENT_PACK_VERSION` `1.0`, `ENVIRONMENT_WORLD_VERSION` `1.0`,
`FAMILY_RULES_VERSION` `1.0`, `CALIBRATION_PANEL_VERSION` `1.0`,
`DIFFICULTY_RULES_VERSION` `1.0`, `ADMISSION_RULES_VERSION` `1.0`,
`COMPOSITION_VERSION` `1.0`, `SELECTION_RULES_VERSION` `1.0`,
`RECIPE_CATALOG_VERSION` `1.0`, `SUITE_FREEZE_RECORD_VERSION` `1.0`.

Tooling versions: `CLI_VERSION` `1.0`, `BASELINES_VERSION` `1.0`,
`INTEGRITY_PROTOCOL_VERSION` `1.0`, `REPLAY_AUDIT_VERSION` `1.0`,
`ANIMATION_FORMAT_VERSION` `1.0`. Suite labels:
`GENERATED_SUITE_VERSION` `1.0.0`, `COMPOSED_SUITE_VERSION` `1.0.0`. Public
suite `suite_version` `1.0.0`.

### Milestones

- **Deterministic simulator** (Phase 1, PR #2): benchmark contract, grid
  scenarios, the deterministic simulator, run records with digests, and
  replay.
- **Fairness contract** (Phase 2, PR #17): the frozen V1 fairness and
  semantics specification ([v1-fairness-spec.md](v1-fairness-spec.md)).
  `SIMULATOR_VERSION` went from `0.1.0` to `0.2.0`: the run record header
  gained `observation_version`; the dynamics did not change.
- **Metrics** (Phase 3, PR #18): auditable per-episode metrics
  ([metrics.md](metrics.md)).
- **Fixed public suite** (Phase 4, PR #19): the canonical V1 public scenario
  suite with pinned hashes ([scenario-suite.md](scenario-suite.md)). It is a
  public development and diagnostic suite, not the hidden scored suite.
- **Paired evaluation** (Phase 5, PR #20): exact scenario + seed manifests and
  resumable, paired evaluation runs ([evaluation.md](evaluation.md)).
- **Stochastic hazard** (Phase 6, PR #21): the matched V1 stochastic fire
  contract with cross-platform golden traces
  ([simulation-contract.md](simulation-contract.md)).
- **Crowd diagnostics** (Phase 7, PR #22): queue and throughput diagnostics
  ([crowd-diagnostics.md](crowd-diagnostics.md)).
- **Observation/action contract** (Phase 7, PR #23): the provider-independent
  agent, observation and action contract ([model-interface.md](model-interface.md)).
- **Statistics** (Phase 8, PR #24): deterministic aggregate and paired
  statistics ([statistics.md](statistics.md)).
- **Generator** (Phase 8, PR #25): the procedural unseen scenario generator
  ([procedural-generation.md](procedural-generation.md)).
- **Integrity and provenance** (Phase 9, PR #26, issue #14): the fail-closed
  hidden-test integrity protocol, provenance records and the public-asset
  leak guard ([integrity.md](integrity.md)).
- **Baselines** (Phase 9, PR #27): deterministic reference baseline agents
  ([baselines.md](baselines.md)).
- **CLI** (Phase 9, PR #28): the reproducible, provider-free `emberbench`
  command ([cli.md](cli.md)).
- **Replay and visualization** (Phase 10, PR #29): read-only replay audit,
  ASCII frames and the JSON animation export (`REPLAY_AUDIT_VERSION` `1.0`,
  `ANIMATION_FORMAT_VERSION` `1.0`; [replay-visualization.md](replay-visualization.md)).
- **Official EMBER Score and EnvironmentPack** (Phase 10, PR #32): scoring
  1.0 with person-level loss attribution and severe quality gates, frozen
  environment packs that pin scenario bytes, schedule, versions and fire
  worlds ([scoring.md](scoring.md), [environment-pack.md](environment-pack.md)).
  Coordination is not a score component: it was removed before scoring 1.0
  was frozen ([benchmark-methodology.md](benchmark-methodology.md), section 5.12).
- **Scientific suite design and unique-world identity** (PR #33):
  `environment_world_digest` and `build_unique_world_manifest`, so a
  deterministic scenario is evaluated once instead of once per seed
  ([evaluation-suite-design.md](evaluation-suite-design.md)).
- **Generator 1.1 and 10-family coverage** (Phase 11, PR #34): generator
  `1.1.0` (config and generation manifest `1.1`), the ten V1 family rules,
  calibration panel, difficulty and admission rules, composition, selection,
  recipe catalog and suite freeze record ([generator.md](generator.md),
  [difficulty-calibration.md](difficulty-calibration.md)).
- **Benchmark revision 2** (PR #34): `BENCHMARK_REVISION` went from `1` to
  `2` before any release, and `benchmark_versions()` pins every
  benchmark-defining version above. Revision 1 was never released.
- **Public development evaluation pack** (research lane, PR #31): the
  official EMBER Score of the fair reference agents on the 19 unique worlds
  of the public suite, one frozen environment pack, fail-closed resume and
  `--verify`, and the golden index and `REPORT.md`
  ([tools/experiments/v1/README.md](../tools/experiments/v1/README.md)). It
  is not a hidden-suite leaderboard and changes no version or pinned hash.
- **Remote-model runner** (research lane): any model behind a base URL and
  an API key (OpenAI-compatible chat, Anthropic Messages or the Jev/SubQ
  decision API) on the same environment pack, with the same official scoring
  and paired comparisons against the reference runs
  ([tools/experiments/remote/README.md](../tools/experiments/remote/README.md)).
  It changes no version or pinned hash.
- **Results website**: moved to its own repository,
  [Aryabhatta-0/EMBERBench-site](https://github.com/Aryabhatta-0/EMBERBench-site).
  It builds its data from the official score reports of EMBERBench runs, with
  the `emberbench` package pinned to a commit of this repository. The first
  version (`docs/site`, `tools/build_results_site.py` and
  `.github/workflows/pages.yml`, PR #36) is removed from this repository.

### Release hardening (Phase 10, issue #16, PR #30)

- One authoritative package version in `src/emberbench/_version.py`, exposed
  as `emberbench.__version__` and as `package_version` in
  `emberbench version`.
- `emberbench version --json` reports every version of `benchmark_versions()`
  by construction, plus the tooling versions. No component version changed.
- Version and release policy ([versioning.md](versioning.md)) and release
  checklist ([release-checklist.md](release-checklist.md)).
- Development tooling pinned by `tools/dev-constraints.txt`, with bounded
  ranges in `pyproject.toml`.
- `tools/check_release.py`: version-contract and docs consistency, two
  clean builds compared, wheel built from the sdist offline, wheel content
  and size limits, clean-venv install, registry recomputed with the
  installed wheel, leak guard, and the license blocker.
- Regression digest registry (`tests/fixtures/release/digest_registry.json`,
  31 entries) rebuilt for the V1 contracts; it adds scoring, environment
  pack, environment world, rule set, replay, version and CLI pins. The
  generator and provenance entries changed with generator 1.1 and
  revision 2 (PR #34); no pinned hash changed in this PR.
