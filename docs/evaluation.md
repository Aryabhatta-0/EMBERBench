# Paired identical evaluation

This document specifies how EMBERBench runs several models on **exactly the same
episodes** and how their results are matched. The code is in
[`src/emberbench/evaluation.py`](../src/emberbench/evaluation.py). The tests are in
[`tests/test_evaluation.py`](../tests/test_evaluation.py).

Out of scope here: model-provider adapters, win rates, confidence intervals and
other aggregate statistics (see [statistics.md](statistics.md)), procedural scenario generation,
leaderboards, hidden scenarios and a composite score.

## 1. Common random numbers

An episode is fully defined by a scenario and a seed. All randomness comes from
simulator-owned streams derived from the seed (see
[simulation-contract.md](simulation-contract.md)). In V1 the fire does not depend
on the actions, so two models that play the same scenario with the same seed
face **the same fire, cell for cell and step for step**.

Paired evaluation uses this. Every model plays the same ordered list of
(scenario, seed) episodes. A difference between two models on one episode then
comes from their decisions only, not from different fire. This is the
*common-random-numbers* method. It removes the between-seed noise from each
paired difference, so fewer episodes are necessary to compare two models.

The test `test_two_policies_see_identical_stochastic_fire` checks this on the
stochastic scenario `v1-pub-10-route-invalidation`. A greedy policy and an
always-stay policy see identical burning sets at every step that both runs
reach, and two different seeds give different fire.

## 2. The evaluation manifest

An `EvaluationManifest` is an immutable, versioned episode schedule.

```json
{
  "manifest_version": "1.0",
  "suite_id": "emberbench-v1-public",
  "suite_version": "1.0.0",
  "suite_hash": "<sha256 of the canonical suite manifest>",
  "simulator_version": "0.2.0",
  "observation_version": "1.0",
  "metrics_version": "1.0",
  "episodes": [
    {"index": 0, "scenario_id": "v1-pub-01-single-exit-sparse", "scenario_hash": "<sha256>", "seed": 3},
    {"index": 1, "scenario_id": "v1-pub-01-single-exit-sparse", "scenario_hash": "<sha256>", "seed": 1}
  ]
}
```

Rules (all enforced on construction and on parsing):

- `manifest_version` must be a supported version (`1.0`).
- `suite_id`, `suite_version` and `suite_hash` must be well formed.
- `episodes` is non-empty. `index` equals the position in the list (0, 1, 2, ...).
- Each `seed` is an integer in `[0, 2**64 - 1]`. Booleans and floats are rejected.
- A seed occurs at most once per scenario. The same seed may be used for
  different scenarios.
- One scenario ID is pinned to exactly one content hash.
- Unknown and missing keys are rejected.

`manifest_hash()` is the SHA-256 of the canonical JSON of the whole manifest.
A change to any field, any seed, any scenario hash or to the episode order gives
a different hash.

### Building a manifest

`build_evaluation_manifest(suite, seeds)` schedules every suite scenario under
every seed: the outer order is the suite manifest order, the inner order is the
order of `seeds` as given. The function does not sort, remove or generate seeds.
An empty list, a duplicate or an out-of-range seed is an error. A `set`, a
`str` or a generator is a `TypeError`, because it has no reliable order.

`verify_manifest(manifest, suite)` requires that the manifest targets exactly
this suite (ID, version, hash) and the running code's simulator, observation
and metrics versions, and that every pinned scenario hash matches both the
suite entry and the scenario file. `run_evaluation` and
`load_evaluation_result` always call it.

`save_evaluation_manifest` and `load_evaluation_manifest` store a manifest as
JSON. The loaded manifest is checked with the same rules.

## 3. Public development manifests and private scored manifests

The seeds decide the fire. If a model developer knows the scored seeds, they
can tune a model to that exact fire. Therefore:

- **Development manifests** use small, fixed, public seeds (for example
  `[0, 1, 2]`). They are fine in tests, examples and documentation. The test
  suite pins one: the public suite under seeds `(0, 1, 2)` has manifest hash
  `7b9aae45e5e2828f881b2f14d8e5585627e5f8bdbb137a1ed2331bf10ce8c876`.
- **Scored manifests** use high-entropy seeds chosen by the evaluator (for
  example 64-bit values from `secrets.randbits(64)`). The evaluator keeps the
  seed schedule outside this repository and publishes the manifest, if at all,
  only after the evaluation.

This repository never generates scored seeds. The `seed` in a scenario file is
a development default only. `run_evaluation` always overrides it with the
manifest seed.

A published manifest hash lets anyone check later that a published schedule is
the one that was used, without revealing the seeds before the evaluation.

## 4. Seed privacy and episode isolation

The evaluator knows the manifest and the seeds. The decision policy does not.

`run_evaluation(suite, manifest, model_id, policy_factory, output_path, resume=True)`
does this for each episode:

1. `policy = policy_factory()`. The factory receives **no arguments**: no
   scenario, seed, episode index or other metadata.
2. `run_episode(scenario, policy, seed=episode.seed)`. The policy is called
   with exactly one positional argument, an `Observation`, once per step.

The policy never receives the `EvaluationManifest`, an episode entry or index,
the seed, the `Scenario`, the `Simulator`, the `RunRecord` or any evaluator
metadata. The `Observation` contains no seed and no RNG state (see
[v1-fairness-spec.md](v1-fairness-spec.md)).

A new policy is built for every episode, so state that a policy keeps cannot
carry over into the next episode. This also makes resume correct: an episode
gives the same result whether it runs in a fresh process or after other
episodes. The factory must return a new policy each time. A plain stateless
function is also acceptable.

The tests prove this:

- `test_policy_receives_only_observations` uses a policy that records every
  argument it receives and a factory that records its own arguments. The
  factory is called with `()` once per episode. Each policy call has exactly one
  argument of type `Observation` and no keyword arguments. Every object inside
  the observation is one of the known observation types. The 64-bit seed (in
  decimal and hex), the manifest hash, the suite hash and the scenario hashes do
  not occur in its JSON form.
- `test_fresh_policy_per_episode` checks that there is one distinct policy
  instance per episode and that each instance sees one episode, from step 0.

Limit: this is an interface contract for in-process Python policies, not a
sandbox. Python code can inspect its caller's stack. A real model adapter must
send only the serialized observation to the model.

For per-episode agents, action validation, optional execution accounting and
the canonical serialized observation, see [model-interface.md](model-interface.md).

## 5. The per-model result artifact

`run_evaluation` writes one JSON artifact per model run. It is canonical JSON
(sorted keys, no extra whitespace) followed by a newline.

```json
{
  "result_version": "1.0",
  "model_id": "model-a",
  "evaluation_manifest_hash": "<sha256>",
  "suite_id": "emberbench-v1-public",
  "suite_version": "1.0.0",
  "suite_hash": "<sha256>",
  "simulator_version": "0.2.0",
  "observation_version": "1.0",
  "metrics_version": "1.0",
  "manifest": { "...": "the complete evaluation manifest" },
  "episodes": [
    {
      "index": 0,
      "scenario_id": "v1-pub-01-single-exit-sparse",
      "scenario_hash": "<sha256>",
      "seed": 3,
      "record_digest": "<sha256 of the canonical run record>",
      "record": { "...": "the complete RunRecord" },
      "metrics": { "...": "EpisodeMetrics.to_dict()" }
    }
  ]
}
```

- The header fields repeat the embedded manifest so a reader can see them at
  once. They must equal the manifest.
- `episodes` holds completed episodes only, in manifest order. Each one names
  its exact manifest entry and contains the full `RunRecord`, so anyone can
  replay and audit it, plus the raw metrics computed from it.
- `model_id` is a label chosen by the evaluator (1 to 256 printable characters,
  no leading or trailing whitespace). It is not proof of which model produced
  the actions, and it is not a unique key.

`load_evaluation_result(path, suite)` accepts an artifact only if **all** of
these hold:

- the JSON has exactly the expected keys at every level;
- the embedded manifest is valid, its hash equals `evaluation_manifest_hash`,
  and every header field equals the manifest;
- the manifest passes `verify_manifest` against `suite` and the running code;
- every episode index is in the manifest, occurs once, and episodes are in
  manifest order;
- every episode's `scenario_id`, `scenario_hash` and `seed` equal its manifest
  entry, and the record's header agrees with them;
- every record is in canonical form and matches `record_digest`;
- every record is **authenticated**: `compute_episode_metrics` checks its
  structure and replays its action log in a fresh simulator, which must
  reproduce it byte for byte;
- every stored `metrics` object equals the metrics recomputed from the
  authenticated record;
- the artifact as a whole re-serializes to exactly the loaded JSON.

Any failure raises `EvaluationError`. Nothing is repaired, skipped or ignored.

## 6. Resume guarantees

`run_evaluation` saves the artifact after every completed episode. Each save
writes a temporary file in the same directory, flushes and fsyncs it, and
renames it over the artifact with `os.replace`. A crash therefore leaves either
the previous artifact or the new one, never a partial file. After a hard crash
(for example a power loss) a stray `.<name>.*.tmp` file can remain next to the
artifact. It is never read and can be deleted. If the output path
does not exist, an artifact with zero episodes is written before the first
episode runs.

If `output_path` exists:

- With `resume=False` it is an error. An existing artifact is never
  overwritten.
- With `resume=True` (the default) the artifact is loaded and authenticated
  completely, as in section 5. Then it must have the same evaluation manifest
  hash and the same model ID as the call. A changed schedule, seed order,
  scenario, suite or version, or a different model ID, is rejected.
- Only the episodes that are not in the artifact run, in manifest order. A
  completed episode is never run again.

Because each episode uses a fresh policy and its own seed, an interrupted and
resumed run gives the same artifact, byte for byte, as an uninterrupted run
with a deterministic policy. `test_interrupted_run_resumes_without_rerunning`
checks this.

A policy exception or an invalid action stops the run with that exception. The
artifact keeps the episodes completed before it.

Cost note: the whole artifact is rewritten after each episode, so total bytes
written grow with the square of the number of episodes. This is acceptable for
V1 schedules. A sharded layout can replace it later without changing the
pairing rules.

## 7. Exact pairing

`pair_results(a, b)` compares two loaded artifacts. It first requires that both
use the same evaluation manifest (equal hash and equal content). Otherwise it
raises `EvaluationError`: results from different schedules are never compared.

Each scheduled episode has the key

```
(evaluation manifest hash, episode index, scenario ID, scenario hash, seed)
```

**Definition.** An episode is a *comparable pair* exactly when both artifacts
contain a result whose key equals that scheduled key. Nothing else (model ID,
file name, run order or time) is part of the key.

The function returns a `PairingReport`:

| Field | Meaning |
|---|---|
| `manifest_hash` | the shared evaluation manifest hash |
| `model_a`, `model_b` | the two model ID labels |
| `total_scheduled` | number of episodes in the manifest |
| `paired` | keys present in both artifacts, in manifest order |
| `paired_count` | `len(paired)` |
| `missing_from_a` | scheduled keys that `a` does not have |
| `missing_from_b` | scheduled keys that `b` does not have |

An episode that neither artifact has is in both missing lists. The report holds
no scores, win rates or statistics. [statistics.md](statistics.md) defines the paired statistics; they use only
`paired` episodes.

## 8. Example

```python
from emberbench import build_evaluation_manifest, load_suite, pair_results, run_evaluation

suite = load_suite("scenarios/public/v1")
manifest = build_evaluation_manifest(suite, [0, 1, 2])  # development seeds


def make_policy():
    # Return a NEW policy. It receives only Observation values.
    return MyPolicy()


a = run_evaluation(suite, manifest, "model-a", make_policy, "results/model-a.json")
b = run_evaluation(suite, manifest, "model-b", make_other_policy, "results/model-b.json")
report = pair_results(a, b)
print(report.total_scheduled, report.paired_count)
```

To compare saved artifacts later, load each with
`load_evaluation_result(path, suite)`, which authenticates every record, and
then call `pair_results`.
