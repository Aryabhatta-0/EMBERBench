# Changelog

Notable changes to EMBERBench. The version policy is in
[versioning.md](versioning.md). Each release entry names the package version,
the benchmark revision, every component version that changed and every pinned
hash that changed.

Entries reference the pull requests in `Aryabhatta-0/EMBERBench` that
introduced each part. Dates are added only when a release is tagged.

## Unreleased: V1 (package `1.0.0rc1`, benchmark revision `3`)

No version of EMBERBench has been released yet. This entry collects the V1
milestones that are in `main`. Release them by following
[release-checklist.md](release-checklist.md). The maintainers chose `1.0.0`
as the first release version; the candidate is `1.0.0rc1` (tag
`v1.0.0-rc.1`), and `1.0.0` follows when the hidden-batch statistics hold
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
`EVALUATION_RESULT_VERSION` `1.0`, `SCORING_VERSION` `1.1`,
`ENVIRONMENT_PACK_VERSION` `1.0`, `ENVIRONMENT_WORLD_VERSION` `1.0`,
`FAMILY_RULES_VERSION` `1.0`, `CALIBRATION_PANEL_VERSION` `1.0`,
`DIFFICULTY_RULES_VERSION` `1.0`, `ADMISSION_RULES_VERSION` `1.0`,
`COMPOSITION_VERSION` `1.0`, `SELECTION_RULES_VERSION` `1.0`,
`RECIPE_CATALOG_VERSION` `1.0`, `SUITE_FREEZE_RECORD_VERSION` `1.0`,
`HARNESS_VERSION` `1.1`, `ADAPTER_VERSION` `r1`, `ACCOUNTING_VERSION` `1.1`,
`COMMITMENT_VERSION` `1.0`.

Tooling versions: `CLI_VERSION` `1.1` (was `1.0`; adds `emberbench run`), `RUN_RECORD_VERSION` `1.1`, `BASELINES_VERSION` `1.0`,
`INTEGRITY_PROTOCOL_VERSION` `1.0`, `REPLAY_AUDIT_VERSION` `1.0`,
`ANIMATION_FORMAT_VERSION` `1.0`. Suite labels:
`GENERATED_SUITE_VERSION` `1.0.0`. Public
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
  ([evaluation-suite-design.md](evaluation-boundary.md)).
- **Generator 1.1 and 10-family coverage** (Phase 11, PR #34): generator
  `1.1.0` (config and generation manifest `1.1`), the ten V1 family rules,
  calibration panel, difficulty and admission rules, composition, selection,
  recipe catalog and suite freeze record ([generator.md](generator.md),
  [difficulty-calibration.md](difficulty-calibration.md)).
- **Benchmark revision 2** (PR #34): `BENCHMARK_REVISION` went from `1` to
  `2` before any release, and `benchmark_versions()` pins every
  benchmark-defining version above. Revision 1 was never released.
- **Benchmark revision 3** (harness 1.0): `BENCHMARK_REVISION` went from `2`
  to `3` before any release. `benchmark_versions()` now also pins the
  evaluator protocol for remote models: `HARNESS_VERSION` `1.0` (canonical
  full-joint protocol for every API style, fixed chunks only as a declared
  `-chunkN` variant, re-ask of invalid slots only, at most 2 re-asks, then
  fail-closed; the executed action is the model's `choice`;
  [harness-protocol.md](harness-protocol.md)), `ADAPTER_VERSION` `r1` (the
  prompt design, unchanged), `ACCOUNTING_VERSION` `1.0`
  (calls, tokens, cost, time, re-asks and horizon hits per episode) and
  `COMMITMENT_VERSION` `1.0` (salted hidden-suite commitment). No simulator,
  scenario, metric or scoring rule changed; the golden pack indexes change
  only in their `versions` block.
- **Refusals** (harness 1.1, scoring 1.1, accounting 1.1, run record 1.1;
  still revision 3, which is unreleased): a model or provider-policy refusal
  of a valid question raises `EpisodeRefused`. The harness does not retry or
  re-prompt it. `emberbench run` records the episode in
  `refusals/<model_id>.json`, scores it as the declared failure `refusal`
  (score 0), keeps every other episode and does not run it again on resume.
  The run record and the accounting report the refusal count and rate.
  Transient provider failures keep the same fixed retry policy. Every score
  of scoring version `1.0` is unchanged; score reports gain the count
  `failed_episode_trials.refusal`. The run record also gives
  `score_excluding_refusals`, a diagnostic that never ranks models
  (`score_excluding` in `emberbench.scoring`).
- **Hidden exposure log** (run record 1.1): `emberbench run --private
  --model` needs `--exposure-log`; each run segment that sends episodes logs
  the router, upstream provider, endpoint, recorded retention and episode
  count first. `--private --model` also needs `--zdr`: every request
  carries OpenRouter's `provider.zdr: true`. Rotation rule (8 providers, one
  retention strike, 2 quarters or any leak) and pilot batches:
  [integrity.md](integrity.md), section 12.
- **`emberbench run`** (CLI `1.1`): one command evaluates a remote model
  (`--model`), a registered baseline (`--baseline`) or a local agent
  (`--agent`) on any frozen environment pack, verifies the pack digest or
  suite commitment first, scores it and writes a run record
  (`RUN_RECORD_VERSION` `1.1`; [cli.md](cli.md)). `--private` keeps scenario
  IDs, seeds and answer text out of every output.
- **Public development evaluation pack** (research lane, PR #31): the
  official EMBER Score of the fair reference agents on the 19 unique worlds
  of the public suite, one frozen environment pack, fail-closed resume and
  `--verify`, and the golden index and `REPORT.md`
  ([tools/experiments/v1/README.md](../tools/experiments/v1/README.md)). It
  is not a hidden-suite leaderboard and changes no version or pinned hash.
  The smoke protocol (5 episodes, 100 resamples) now gives no
  `a_better`/`inconclusive` verdicts: its comparison table has the mean
  difference and interval only, and its index has `"verdicts": false`.
- **Validity analysis** (research lane): a models-only view (adjacent model
  pairs with the paired interval and bootstrap P(higher > lower), P(the whole
  model order), the models-only tau and the size of one adjacent swap) and
  the zero-score episode share of every agent. It changes no score.
  `zero_inflation.py` splits zero scores by cause (catastrophic gate,
  declared failure) and reports the score histogram, the bimodality
  coefficient and the scores just below the gate.
- **Cost sheet and demo** (research lane): `tools/experiments/v1/cost_table.py`
  (USD per suite and episode, calls, tokens, re-ask and refusal rates, model
  time, from the run records) and `demo.py` (public pack -> `REPORT.md` and
  `COST.md` in one command).
- **Fix:** `emberbench run` without `--json` no longer fails when the
  summary holds a list of values (`ci95`).
- **Leaderboard tiers and outcome profile** (presentation; scoring version
  unchanged): `leaderboard_tiers` groups `leaderboard_order` into tied tiers
    by paired, family-stratified bootstrap tests against the tier leader,
    with Holm correction at 5% within each step of the tier walk. Marginal
    CI overlap does not decide a tier. `outcome_profile` gives
  the zero-score share and the survival rate to publish next to each score
  ([scoring.md](scoring.md), section 8). No score, rank key or pinned hash
  changes.
  - **V1 coverage limitation:** retain the existing cells; document the four
    deviating families and the 140/83/77 stratum totals in
    [benchmark-methodology.md](benchmark-methodology.md). `crowd_100` must
    not be compared across strata.
  - **Container provenance:** add the OCI source label linking the image to
    `Aryabhatta-0/EMBERBench`. The public Linux/amd64 runtime from source
    `9d9999b913bc304001e6cf41f6b34a970109f296` is published at
    `ghcr.io/aryabhatta-0/emberbench@sha256:1142df88f0d68b5a3d34323481168c6d3253ef6cef4e7879e5de25f9b2b712d8`.
    The registry manifest bytes, source/revision labels, anonymous digest
    pull and public-only baseline smoke were independently verified. The
    [publication record](container-publication.md) pins the source and CI;
    later documentation commits do not change this image's source identity.
    The test stage installs Git and indexes its copied public sources so the
    leak-guard and release tests can enumerate them inside the container.
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
