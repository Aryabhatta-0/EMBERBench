# Frozen environment packs (environment pack version 1.0)

An environment pack freezes the complete evaluation environment **once,
before any model runs**. Every model is then evaluated on the same bytes,
the same seed schedule and the same fire worlds. The code is
`src/emberbench/environment_pack.py`. The tests are
`tests/test_environment_pack.py`. The score that uses a pack is in
[scoring.md](scoring.md).

## 1. What a pack freezes

| Part | Content |
| --- | --- |
| Scenario bytes | an exact byte copy of every scenario file and of the suite `manifest.json` |
| Generator record | for a generated suite, the exact `generation.json` (generator version, config, batch, pinned proxies) |
| Seed schedule | the ordered `evaluation-manifest.json` (episode index, scenario ID, scenario hash, seed) |
| Fire worlds | one `fire_world_digest` per scheduled episode |
| Versions | `pack_versions()`: environment pack, simulator, observation, action schema, scenario schema, metrics, generator, generator config, generation manifest and evaluation manifest |
| File hashes | the SHA-256 of every file in the pack |
| Pack digest | SHA-256 of the canonical JSON of `pack.json` |

`pack.json` lists the SHA-256 of every other file, so the pack digest pins
every byte of the pack.

## 2. Layout

```text
<pack>/
  pack.json                    the record; its canonical digest is the pack digest
  evaluation-manifest.json     the ordered seed schedule
  suite/                       a plain suite: manifest.json + scenario files
  generated-suite/             or a generated suite:
    generation.json
    suite/
```

`pack.json` has exactly these keys: `environment_pack_version`, `pack_id`,
`source` (`suite` or `generated_suite`), `versions`, `suite` (ID, version,
suite hash, and per scenario its ID, content hash and primary family),
`evaluation_manifest_hash`, `episodes` (per episode: index, scenario ID,
scenario hash, seed, `fire_world_digest`) and `files` (sorted
`[path, sha256]` pairs).

## 3. API

```python
from emberbench import freeze_environment_pack, load_environment_pack, verify_environment_pack

pack = freeze_environment_pack(suite_or_frozen_generated_suite, [seed0, seed1, ...], "packs/2026-q4", pack_id="2026-q4")
digest = pack.digest                                   # publish this before the evaluation
pack = load_environment_pack("packs/2026-q4", expected_digest=digest)
verify_environment_pack("packs/2026-q4", expected_digest=digest)   # returns the digest or raises
```

- `freeze_environment_pack(source, schedule, directory, *, pack_id)`.
  `source` is a loaded `Suite` or a `FrozenGeneratedSuite`. `schedule` is an
  ordered seed list (scheduled with `build_evaluation_manifest`) or an
  existing `EvaluationManifest` that targets the suite. The directory must
  not exist or must be empty: a pack is never overwritten. The pack is
  written to a temporary sibling, renamed into place, then loaded back and
  verified from disk.
- `load_environment_pack(directory, *, expected_digest=None)` verifies and
  returns an `EnvironmentPack` (`suite`, `manifest`, `generation`,
  `episodes`, `record`, `digest`).
- `verify_environment_pack(directory, *, expected_digest=None)` returns the
  digest or raises.
- `fire_world_digest(scenario, seed)` is the SHA-256 of the canonical JSON of
  the scenario hash, the seed and the sorted burning cells of every snapshot
  `0..H`.

The pack ID is 1 to 128 ASCII letters, digits, `-`, `_` or `.`.

## 4. Verification fails closed

`load_environment_pack` raises `EnvironmentPackError` when any of these is
true:

- `pack.json` is missing, is not JSON, or has a missing or extra key;
- the pack version or any entry of `versions` differs from the running
  code;
- a listed file is missing, a file is not listed, a path is unsafe, or a
  symbolic link is present;
- a file's SHA-256 differs from its listed value;
- the suite does not load (`load_suite`) or the generated suite does not
  verify (`load_generated_suite`);
- the evaluation manifest does not target the suite (`verify_manifest`) or
  its hash differs;
- the record differs from the record rebuilt from the files, including any
  fire world recomputed from the scenario and the seed;
- `expected_digest` is given and differs.

`score_results(..., pack)` also refuses results whose manifest is not the
pack's manifest, and checks every episode's fire trajectory against the
pack.

## 5. Identical environments for every model

- **Never regenerated per model.** A pack is written once. Models only read
  it. A model's score names the pack digest; two scores are comparable only
  with the same digest.
- **No model-dependent state.** In V1 the fire ignores people and actions.
  The burning set of every snapshot is a function of the scenario and the
  seed (`oracle.fire_trajectory`). The pack stores its digest, and scoring
  checks every recorded snapshot against it. So every model meets the same
  fire at the same step: common random numbers by construction. Nothing in
  the pack depends on a model or on any mid-episode state that a model can
  change.
- **People are the model's.** Positions, blocks and outcomes depend on the
  model's actions. They are in the run record and are authenticated by
  replay; they are not in the pack.

## 6. API and external models

An external model (an LLM API or any remote service) is a policy behind the
adapter of [model-interface.md](model-interface.md):

1. The evaluator holds the pack. The model never receives the pack, the
   seed, the fire trajectory or any privileged reference.
2. At each step the model receives one `Observation` and returns one
   `ActionOutput`. Invalid output raises `ModelOutputError`; the episode is
   declared `invalid_output` and scores 0.
3. The evaluator runs the simulator locally and records the run. Scoring is
   local, from replayed records only.
4. Provider-reported success, accuracy, safety or cost is never trusted and
   never used. Cost for the EMBER-Efficiency view is measured by the
   evaluator.
5. A timeout is the adapter's own deadline. A timed-out episode is declared
   `timeout` and scores 0. Provider retries are an adapter policy, recorded
   in the run's configuration, not hidden.
6. A model error (the provider or the adapter raised) is declared
   `model_error` and scores 0. A failure can never disappear: an episode
   without a result and without a declaration is `missing` and scores 0.
7. Operational data is recorded separately in an `OperationalRecord`
   ([scoring.md](scoring.md), section 11): model ID, provider ID, the
   inference configuration digest, latency, input tokens, output tokens,
   cost and its unit, retries and timeouts. It is never a score input.

**Stochasticity protocol.**

- Pin the inference configuration (model name and version or snapshot,
  temperature, top-p, max tokens, system prompt, adapter version) and record
  it with the submission as `inference_config_digest(config)`. All trials of
  one submission use the same digest.
- `temperature = 0` is not a determinism guarantee. Batching, numerics and
  provider-side changes can change outputs.
- Run `K >= 3` trials of the full pack for any sampled model (`K = 1` only
  for a policy shown to be deterministic). Score them together with
  `score_results([trial_1, ..., trial_K], pack)`. Each episode is the mean of
  its trials. Report `K`.
- Never select the best trial. Every trial that started is scored; a trial
  that failed contributes zeros.

## 7. Privacy and publication

A pack contains the evaluator seeds and, for a held-out generated suite,
the private scenarios. It is a **private asset** until the evaluation ends
(see [integrity.md](integrity.md)). Publish the pack digest first, as a
commitment. Publish the pack after the evaluation; anyone can then run
`verify_environment_pack` with the published digest and re-score every
submitted result.

## 8. Versioning

`ENVIRONMENT_PACK_VERSION = "1.0"`. It changes when the layout, the record
or the fire-world digest changes. The scoring correction pass (person-level
attribution, severe quality gates, the exit-capacity time bound) changed
none of these, so the version stays `"1.0"`: the privileged references are
recomputed by the scorer from the pack, and are not stored in it. A pack is valid only under the exact
versions in its `versions` entry; a simulator or schema change invalidates
every pack, by design.
