# EMBERBench V1 evaluation pack (research lane)

**PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD**

This pack shows how EMBERBench compares decision-making methods with the
official EMBER Score (`emberbench.scoring`, scoring version 1.1) on the public
V1 suite. It is not a core benchmark phase. It changes no simulator,
scenario, metric, statistic, scoring, generator, integrity, baseline,
observation or artifact format, and it adds no runtime dependency. It uses
public EMBERBench APIs only and has no scoring code of its own.

| File | Content |
|---|---|
| `pack_agents.py` | The evaluated fair agents: the `always-stay` control, the five registered baselines (by ID) and two pack agents, with their descriptors and config digests. |
| `pack_protocol.py` | The committed protocol: the canonical public suite, the public development candidate seeds, the number of stochastic worlds, the pinned suite and unique-world manifest hashes, the bootstrap size and the paired comparisons. Also the tiny CI `smoke` protocol. |
| `run_public_benchmark.py` | The runner: `build_unique_world_manifest`, `freeze_environment_pack`, `run_evaluation` per model (resumable), authentication by replay, `score_result` against the pack, `compare_scores`, `aggregate_result` diagnostics, the separate privileged references, the index and the report. |
| `pack_index.py` | The canonical reproducibility index (versions and digests of every input and output; no paths, hosts or timestamps). |
| `pack_report.py` | Renders `REPORT.md` from authenticated artifacts only. |
| `REPORT.md` | The report of the public development run. |

The golden indexes are in `tests/fixtures/experiments/v1/`. The tests are in
`tests/test_evaluation_pack.py`: CI runs the `smoke` protocol end to end. Set
`EMBERBENCH_FULL_PACK=1` to also run the full public protocol and compare it
with the golden index and `REPORT.md`.

## Protocol

- **Unique worlds.** The schedule is
  `build_unique_world_manifest(suite, CANDIDATE_SEEDS, stochastic_worlds=10)`
  with the public development candidates `0..15`: each of the nine
  deterministic-fire scenarios once, and 10 distinct fire worlds of
  `v1-pub-10-route-invalidation`. That is 19 unique environment worlds out of
  160 nominal candidate episodes. Duplicate deterministic seeds are never
  scheduled.
- **One environment pack.** The runner freezes the suite and the manifest
  into an environment pack. Every model runs it, and every model is scored
  against it, so each `ScoreReport` is produced by the official procedure
  (`official: true`). The suite is public, so the numbers are still not an
  official hidden-suite score.
- **Fair methods only in the ranking.** The privileged references (M*,
  M_cap, time and path lower bounds) are in a separate artifact and a
  separate report section.

## Outputs

| Path (relative to `--out`) | Content |
|---|---|
| `environment-pack/` | The frozen pack: `pack.json`, `evaluation-manifest.json`, `suite/` |
| `source-suite/` | The scenario subset of the `smoke` protocol, written so that the pack can freeze it |
| `models/` | Model descriptors (configuration and code digests) |
| `results/` | `EvaluationResult` per model |
| `scores/` | Official `ScoreReport` per model |
| `comparisons/` | Official paired `ScoreComparison`s, with the two score-report digests |
| `diagnostics/` | `aggregate_result` statistics (diagnostics, not score components) |
| `privileged/references.json` | M*, M_cap and the time bound per world (not models) |
| `index.json`, `REPORT.md` | The reproducibility index and the report |

## Run

```bash
python tools/experiments/v1/run_public_benchmark.py --out ../emberbench-pack-out
```

```bash
python tools/experiments/v1/run_public_benchmark.py --protocol smoke --out ../emberbench-pack-smoke
```

The output directory must be outside the repository. On Windows keep its path
short: the environment pack adds about 70 characters below it, and paths are
limited to 260 characters. A rerun with the same directory resumes:
per-model results continue through `run_evaluation`, and every other output
must be byte-identical, or the run stops. An existing result must also hold
the canonical bytes of its authenticated content before it is resumed; a
result that is valid but formatted differently stops the run and is not
rewritten. Nothing is overwritten. The runner opens no network connection.

```bash
python tools/experiments/v1/run_public_benchmark.py --out ../emberbench-pack-out --verify
```

`--verify` runs and writes nothing. It authenticates every result by replay
against the frozen environment pack, recomputes every score report,
comparison, diagnostic, reference, the index and the report from the
authenticated inputs, and requires each file to be byte-identical. A missing,
changed or extra file stops it with exit status 1. The expected files of
`source-suite/` are the suite manifest and exactly the scenario files it
lists.

## Performance

Measured on one Windows 11 laptop core (Python 3.12), full public protocol,
fresh output directory. Wall-clock values are logged by the runner only; no
pinned artifact holds them, because they are not reproducible.

| Quantity | Value |
|---|---|
| unique worlds | 19 |
| episodes per model | 19 (8 models, 152 episodes) |
| total runtime | 113 s and 114 s (two clean runs; 115 s and 116 s process wall time) |
| episodes (run and replay authentication) | 52 s to 54 s |
| official scoring (8 score reports, 19 paired comparisons) | 39 s to 40 s |
| diagnostics (`aggregate_result`) | 12 s to 13 s |
| privileged references | 2.2 s to 2.4 s |
| `--verify` of a finished run (no episodes are run) | 76 s |
| artifact size | 14,138,424 bytes, 67 files (13 MB of it `results/`) |

The earlier pack ran 100 episodes per model (every scenario under 10 seeds)
in about 19 minutes (800 episodes); 81 of the 100 were duplicates of a deterministic world.
The unique-world schedule runs 152 episodes with the same 19 worlds, about 10x faster.
Replay authentication and fire-world checks against the pack are unchanged.

## Version handling

`pack_protocol.BENCHMARK_VERSIONS` pins all 24 keys of `benchmark_versions()`
for benchmark revision 3. The runner and `--verify` fail closed when the
running code differs in any key or value. The index also pins the
environment-pack versions (including the generator versions), the scoring
version and constants digest, the suite and manifest hashes, every world
digest and every model, result, score-report and comparison digest.

A new benchmark revision needs a new pack run, never a mix: update
`BENCHMARK_VERSIONS`, rerun the full and the smoke protocols, and commit the
new index files and report. Nothing else in the pack needs to change.

## Why `tools/experiments/v1/`

The leak guard (`emberbench.leakguard`, docs/integrity.md section 10) allows
only fixed top-level directories, and JSON files only under
`scenarios/public/` and `tests/fixtures/`. A top-level `experiments/`
directory would fail it, and changing the leak guard is an integrity change.
The pack therefore lives under `tools/`, its golden JSON indexes under
`tests/fixtures/`, and run outputs outside the repository.
