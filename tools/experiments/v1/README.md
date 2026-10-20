# EMBERBench V1 evaluation pack (research lane)

**PUBLIC DEVELOPMENT EVALUATION — NOT HELD-OUT SCORE**

This pack shows how EMBERBench compares decision-making methods. It is not a
core benchmark phase. It changes no simulator, scenario, metric, statistic,
generator, integrity, baseline, observation or artifact format, and it adds no
runtime dependency. It uses public EMBERBench APIs only.

| File | Content |
|---|---|
| `pack_agents.py` | The evaluated fair agents: the `always-stay` control, the five registered baselines (by ID) and two pack agents, with their descriptors and config digests. |
| `pack_protocol.py` | The committed protocol: the canonical public suite, the development seed schedule, the pinned suite and manifest hashes, the bootstrap size and the paired comparisons. Also the tiny CI `smoke` protocol. |
| `run_public_benchmark.py` | The runner: `run_evaluation` per model (resumable), authentication by replay, `aggregate_result`, `compare_results`, the separate privileged bound, the index and the report. |
| `pack_index.py` | The canonical experiment index (digests of every input and output; no paths, no timestamps). |
| `pack_report.py` | Renders `REPORT.md` from authenticated artifacts only. |
| `REPORT.md` | The report of the public development run. |

The golden indexes are in `tests/fixtures/experiments/v1/`. The tests are in
`tests/test_evaluation_pack.py`: CI runs the `smoke` protocol end to end. Set
`EMBERBENCH_FULL_PACK=1` to also run the full public protocol and compare it
with the golden index and `REPORT.md`.

## Run

```bash
python tools/experiments/v1/run_public_benchmark.py --out ../emberbench-pack-out
```

```bash
python tools/experiments/v1/run_public_benchmark.py --protocol smoke --out ../emberbench-pack-smoke
```

The output directory must be outside the repository. A rerun with the same
directory resumes: per-model results continue through `run_evaluation`, and
every other output must be byte-identical, or the run stops. Nothing is
overwritten. The runner opens no network connection.

## Why `tools/experiments/v1/`

The leak guard (`emberbench.leakguard`, docs/integrity.md section 10) allows
only fixed top-level directories, and JSON files only under
`scenarios/public/` and `tests/fixtures/`. A top-level `experiments/`
directory would fail it, and changing the leak guard is an integrity change.
The pack therefore lives under `tools/`, its golden JSON indexes under
`tests/fixtures/`, and run outputs outside the repository.
