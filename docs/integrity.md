# Hidden-test integrity

> **Scope:** These are public development or archived research rules. They do
> not define the official hidden evaluation distribution. Previously disclosed
> recipes and study seeds are exposed research, not secret evaluation assets.
> See [Evaluation boundary](evaluation-boundary.md).


| Field | Value |
| --- | --- |
| Modules | `emberbench.integrity`, `emberbench.leakguard` |
| Integrity protocol version | `1.0` (`INTEGRITY_PROTOCOL_VERSION`) |
| Benchmark revision | `2` (`BENCHMARK_REVISION`); section 12 |
| Pinned versions | every key of `benchmark_versions()`: revision, simulator, interface, metrics, statistics, scoring, environment, generator, suite-definition and evaluation contracts (section 12) |
| Runtime dependencies | none (no cryptography library) |

The simulator, the public V1 suite and the generator are open source. A model
can be tuned on all of them. A score is meaningful only on a **held-out**
suite that no evaluated model has seen and that every model gets in exactly
the same form. This document defines how an evaluator produces, uses and
publishes such a suite, and what the published records prove.

The protocol adds no new information to the model interface. An evaluated
agent still receives only `Observation` (see
[model-interface.md](model-interface.md)).

## 1. Threat model

| Actor | Trust | What the protocol defends against |
| --- | --- | --- |
| Model submitter | Not trusted | Tuning on held-out scenarios or seeds; changing the model after seeing held-out material; claiming a result for another model version. |
| Evaluator | Trusted to keep private material private and to run the protocol | Accidental leakage (committed files, packaged files, model inputs); accidental reuse of a different suite or manifest; silent repair of broken artifacts. |
| Third-party reader | Verifies published records | Edited or mixed-up result, suite, manifest or version claims. |
| Repository contributor | Not trusted with held-out data | Committing or packaging held-out material. |

Out of scope: a malicious evaluator, a compromised evaluator machine, a model
that memorised scenarios from a leak outside the protocol, side channels in
model-provider infrastructure, and proof of authorship (see section 12).

## 2. Asset classes

`emberbench.integrity.ASSET_POLICIES` encodes this table. The repository and
the Python package contain **public development assets only**.

| Class (`AssetClass`) | Contents | Release | In repository / package | Given to an agent |
| --- | --- | --- | --- | --- |
| `public_development` | Simulator, generator code, public configs, the public V1 suite, docs, tests, test-only fixtures | always | yes | yes (public scenarios reach a scored agent only as observations) |
| `private_heldout_suite` | Generator seed, rejection history, frozen generated suite, its batch ID and directory | optional audit reveal after the batch is retired | **no** | no |
| `private_evaluation_manifest` | Evaluation manifest and its seeds; full per-model result artifacts, because they embed the manifest | optional audit reveal after the batch is retired | **no** | no |
| `private_submission_metadata` | Submitted code, weights, endpoints, credentials, contact data | never (only identity digests are published) | **no** | no |
| `public_provenance` | Provenance records, submission rosters, aggregate and paired statistics | after evaluation | no (published elsewhere) | no |

Two facts follow from the existing formats:

- A full `EvaluationResult` artifact embeds the evaluation manifest, and so
  its seeds. Publish provenance and statistics after evaluation; publish full
  results only when the batch is retired.
- `AggregateStatistics` and `PairedStatistics` hold the manifest hash and the
  result digest, not the seeds. They are publishable after evaluation.

The repository contains public development/test assets and disclosed historical research. Published research-holdout seeds are exposed, not secret. No future official policy or live hidden suite belongs in the tests, fixtures, docs or package data. Tests use constants marked *test-only*.

## 3. Evaluation lifecycle

Each step finishes before the next one starts. Generation finishes before any
evaluated model sees the suite. The suite is never regenerated per model.

| # | Step | API | Output |
| --- | --- | --- | --- |
| 1 | Freeze the benchmark revision | a release with fixed `benchmark_versions()` | versions to pin |
| 2 | Freeze each submitted model and code | `SubmissionIdentity`, `freeze_submission(identity, workspace)` | `submission/identity.json` |
| 3 | Generate the held-out suite from a private generator seed | `generate_suite(config, seed, batch_id=...)` (or `generate_post_submission_suite`, section 8) | in-memory suite |
| 4 | Freeze the generated suite | `freeze_generated_suite(generated, workspace / "generated-suite")` | `generated-suite/` |
| 5 | Create the private paired evaluation manifest | `build_evaluation_manifest(suite, seeds)`, `save_evaluation_manifest` | `evaluation-manifest.json` |
| 6 | Evaluate every model on the same frozen suite and manifest | `run_evaluation(suite, manifest, model_id, factory, path)` | `result.json` per model |
| 7 | Hash (and optionally sign) the provenance before any statistic is computed | `build_evaluation_provenance(...)`, external signer, `save_provenance` | `provenance.json` |
| 8 | Publish allowed provenance and statistics | `aggregate_result`, `compare_results` | public records |
| 9 | Optionally reveal generation material for audit | `verify_regeneration` / `verify_post_submission_generation` | audit |

Use high-entropy seeds: `secrets.randbits(256)` for a generator seed and
`secrets.randbits(64)` for each evaluation seed. Low-entropy seeds can be
guessed, and the contamination scanner (section 9) cannot search for short
values.

`build_evaluation_provenance` refuses to pin an incomplete result, a public
suite, a manifest that does not verify against the frozen suite, a result of
another manifest, a result that fails authentication by replay, a submission
whose label is not the result's `model_id`, and any artifact of another
benchmark revision.

## 4. Evaluator workspace

Private material lives in an evaluator-side directory **outside** the
repository checkout. One workspace per evaluated model:

```
evaluation-run/
  submission/
    identity.json             SubmissionIdentity, written by freeze_submission
    ...                       the submitted material (not constrained)
  submission-roster.json      only in post-submission mode
  generated-suite/            frozen generated suite: suite/ + generation.json
  evaluation-manifest.json
  result.json
  provenance.json
```

`workspace_stage(root)` validates the layout and returns the
`WorkspaceStage` reached: `empty`, `submission_frozen`, `suite_frozen`,
`manifest_frozen`, `evaluated`, `provenance_recorded`. It raises
`IntegrityError` if the workspace is inside an EMBERBench source checkout,
holds an unexpected entry, or has a later artifact without an earlier one.
Every model of one comparison has a byte-identical `generated-suite/` and
`evaluation-manifest.json`; their provenance records then pin equal suite and
manifest hashes.

`verify_evaluation_workspace(root)` loads every artifact from disk, verifies
the frozen suite, authenticates the result by replay, and runs
`verify_evaluation_provenance` with the identity in `submission/identity.json`
pinned. Identity, roster and provenance files are written once; an existing
file is never overwritten.

## 5. Provenance

`EvaluationProvenance` is a versioned, immutable, public record of one
model's evaluation. Its canonical form:

```json
{
  "integrity_protocol_version": "1.0",
  "versions": {
    "benchmark_revision": "2", "simulator": "0.2.0", "observation": "1.0",
    "action_schema": "1.0", "scenario_schema": "1.0", "metrics": "1.0",
    "crowd_diagnostics": "1.0", "statistics": "1.0", "generator": "1.1.0",
    "generator_config": "1.1", "generation_manifest": "1.1",
    "evaluation_manifest": "1.0", "evaluation_result": "1.0",
    "scoring": "1.0", "environment_pack": "1.0", "environment_world": "1.0",
    "family_rules": "1.0", "calibration_panel": "1.0",
    "difficulty_rules": "1.0", "admission_rules": "1.0",
    "composition": "1.0", "selection_rules": "1.0", "recipe_catalog": "1.0",
    "suite_freeze_record": "1.0"
  },
  "protocol": {"mode": "shared_heldout_batch", "submission_roster_digest": null},
  "suite": {
    "suite_id": "<batch ID>", "suite_version": "1.0.0",
    "suite_hash": "<64 hex>", "generation_hash": "<64 hex>",
    "generator_version": "1.1.0", "generator_config_hash": "<64 hex>",
    "scenario_count": 40
  },
  "evaluation_manifest": {"manifest_hash": "<64 hex>", "episode_count": 400},
  "submission": {
    "identity": {
      "model_id": "model-a", "code_commit": "<40 hex>", "image_digest": null,
      "adapter_config_digest": "<64 hex>", "dependency_lock_digest": null
    },
    "identity_digest": "<64 hex>"
  },
  "evaluator": {"code_digest": null},
  "result": {"result_digest": "<64 hex>", "result_version": "1.0", "episode_count": 400}
}
```

- `provenance_digest()` is the SHA-256 of `signing_payload()`, the canonical
  JSON bytes of this object (sorted keys, no whitespace, ASCII). Equal content
  gives an equal digest on every platform; a one-bit change in any field
  changes it.
- There is **no generator seed, no evaluation seed, no scenario content and no
  timestamp**. A time is not part of the content identity; record it, if
  needed, in an external signed or timestamped envelope.
- `suite.generation_hash` pins the freeze record (config, batch ID, every
  scenario hash and proxy). `suite.suite_hash` pins the suite manifest.
- `result.result_digest` is the SHA-256 of the canonical result artifact. It
  equals the `result_digest` that `aggregate_result` and `compare_results`
  report, so published statistics link to the provenance.
- `evaluator.code_digest` is `null` unless the evaluator supplies it, for
  example with `evaluator_code_digest()` (SHA-256 of the installed
  `emberbench/*.py` files).

The stored file is a `ProvenanceRecord`:

```json
{"provenance": {...}, "provenance_digest": "<64 hex>", "signing": "unsigned", "signatures": []}
```

`ProvenanceRecord.from_dict` rejects unknown or missing fields at every
level, wrong types, a digest that does not match, and a `signing` value that
does not match the signature list.

## 6. Submission identity

`SubmissionIdentity` is provider-neutral explicit metadata:

| Field | Format | Required |
| --- | --- | --- |
| `model_id` | the label used in `run_evaluation` | yes |
| `code_commit` | git commit SHA, 40 or 64 lowercase hex | no |
| `image_digest` | `sha256:<64 hex>` container image digest | no |
| `adapter_config_digest` | SHA-256 of the adapter or model config (`digest_file`, `digest_directory`) | no |
| `dependency_lock_digest` | SHA-256 of the dependency lock file | no |

A missing field is `null`. Nothing is inferred. `pinned_fields` lists the
supplied digests; `is_pinned` is false for a bare label. These values identify
content. They do **not** prove that this content is what ran, or who built it,
unless an actual signature or trusted build process vouches for them.

## 7. Signatures

Signing is optional and external. EMBERBench implements no cryptography and
has no crypto dependency.

- Hashing works without signatures. A signature never changes
  `provenance_digest`.
- An external process signs `provenance.signing_payload()` and the evaluator
  records the result with
  `record.with_signature(SignatureRecord(algorithm, key_id, signature_bytes))`.
  The file stores the algorithm label, key ID and base64 signature bytes.
- An unsigned record says `"signing": "unsigned"`. A signed record says
  `"externally_signed"`. A file can never claim that a signature is valid.
- `verify_evaluation_provenance(..., signature_verifier=f)` calls
  `f(signature, payload)` for every signature. Only `True` accepts. Without a
  verifier, signatures are reported as `signed_unverified` and are not
  trusted. With `require_signatures=True`, an unsigned or unverified record is
  rejected.

## 8. Post-submission generation

The strongest contamination-resistant mode. It makes it impossible for a
submitter to change a model after held-out material exists:

1. Freeze every submission's identity. Each must be pinned (at least one
   content digest).
2. Freeze the roster: `SubmissionRoster.of(identities)`, `save_roster`.
   Publish or escrow `roster.roster_digest()`.
3. Only now choose the private seed (`secrets.randbits(256)`).
4. Generate with `generate_post_submission_suite(config, private_seed, roster, batch_id=...)`.
   The generator seed is
   `SHA-256(domain, protocol version, private seed, roster digest)`, so the
   suite content depends on the frozen roster: no scenario of the batch can
   exist before every listed submission is frozen.
5. Freeze the suite, then freeze the evaluation manifest, then evaluate.
6. `build_evaluation_provenance(..., roster=roster)` records
   `"mode": "post_submission_generation"` and the roster digest.
   Verification needs the roster and checks that the submission is in it with
   this exact identity.
7. After the batch is retired, reveal the private seed;
   `verify_post_submission_generation(suite_dir, private_seed, roster)`
   regenerates every byte. A submission changed after the roster was frozen
   gives another roster digest, another seed and a failed audit.

No remote submission infrastructure is part of EMBERBench.

## 9. Contamination controls

An evaluated agent receives only `Observation` (step, map, rules, people,
burning cells, allowed actions). `MODEL_VISIBLE_OBSERVATION_FIELDS` lists
exactly these fields, and `Agent.act` takes only the observation. The tests
check that an agent never receives:

- the generator seed or the private seed,
- the evaluation seeds,
- the generation rejection history (it is never stored),
- the batch ID, the suite directory or path,
- scenario IDs or content hashes,
- the suite, generation, config or manifest hashes,
- provenance metadata or identity digests,
- prior hidden evaluation results or their digests.

`sensitive_tokens(frozen_suite, manifest, generator_seed=..., extra=...)` and
`find_leaks(text, tokens)` let an evaluator scan anything it sends to a model
(for example a provider prompt built from a serialized observation). Values
shorter than 8 characters are not scanned for.

Limitations: a scanner finds only literal copies. It cannot detect a leaked
scenario layout that was re-encoded, a model trained on a leaked batch, or a
provider that logs prompts. The map, rules and people of each held-out
scenario necessarily reach the model through observations; if the provider
retains them, treat the batch as exposed after evaluation.

## 10. Leak guard

`emberbench.leakguard` checks file paths, not file contents:

- **Allowlist.** Top-level entries: `.github/`, `docs/`, `scenarios/`, `src/`,
  `tests/`, `tools/`, `.gitattributes`, `.gitignore`, `LICENSE`, `README.md`,
  `pyproject.toml`. Under `scenarios/` only `scenarios/public/`. Benchmark
  asset files (`*.json`) only under `PUBLIC_ASSET_ROOTS`:
  `scenarios/public/` and `tests/fixtures/`.
- **Forbidden components** (case-insensitive, `_` read as `-`): `hidden`,
  `heldout`, `held-out`, `scored`, `private`, `secret(s)`, and names starting
  with `hidden-`, `heldout-`, `held-out-`, `scored-`, `private-seed`,
  `private-eval`, `secret-manifest`, `production-eval`, `evaluation-run`.
- **Forbidden files**: evaluator workspace and freeze-record names
  (`generation.json`, `evaluation-manifest.json`, `submission-roster.json`,
  `identity.json`, `provenance.json`, `result.json`) and seed or key types
  (`.seed`, `.seeds`, `.key`, `.pem`, `.p12`, `.pfx`).
- **Wheel**: only top-level `emberbench/*.py`, `emberbench/py.typed` and
  `.dist-info` metadata. **Sdist**: the repository rules plus setuptools
  metadata.

`tests/test_leakguard.py` checks the checkout and builds a wheel and an sdist
from a copy of the tracked files. CI also runs:

```bash
python -m build --no-isolation --outdir dist .
```

```bash
python tools/check_public_assets.py --dist dist
```

Add `--untracked` to also check untracked, not ignored files before a commit.

## 11. Fail-closed verification

`verify_evaluation_provenance(record, frozen_suite=..., manifest=..., result=..., expected_submission=..., roster=..., expected_evaluator_code_digest=..., signature_verifier=..., require_signatures=...)`
re-derives every pin from the actual artifacts and raises `IntegrityError` on:

- a wrong suite hash, generation hash, config hash or scenario count,
- a wrong evaluation manifest hash, or a manifest that does not verify against
  the frozen suite,
- a wrong result digest, a result of another manifest, an incomplete result,
  or a result that fails authentication by replay,
- any version that differs from this benchmark revision,
- a submission identity that differs from the pinned one (when pinned), or a
  label that is not the result's `model_id`,
- an evaluator code digest that differs or is missing (when pinned),
- a missing or wrong roster in post-submission mode, or a roster in shared
  mode,
- a public suite or any other incompatible artifact,
- malformed, missing or unknown fields at any level,
- a signature the verifier rejects, or missing or unverified signatures when
  they are required.

Nothing is repaired, skipped or re-derived into the record.

## 12. Benchmark revision policy

A **benchmark revision** (`BENCHMARK_REVISION`) defines what a score means.
The benchmark revision changes when a released contract changes
incompatibly and that contract defines one of these:

- the meaning of a score,
- the identity of an environment (the world an agent is evaluated in),
- the composition or admission of a suite (which worlds are selected and
  evaluated),
- the semantics of the generator,
- the evaluator protocol.

Bump the revision, together with the component version, for:

- a simulator semantic change (`SIMULATOR_VERSION`),
- an observation or action schema change (`OBSERVATION_VERSION`,
  `ACTION_SCHEMA_VERSION`, `SCHEMA_VERSION`),
- a metric or statistic definition change (`METRICS_VERSION`,
  `STATISTICS_VERSION`, `CROWD_DIAGNOSTICS_VERSION`),
- a change to the EMBER Score formula, its constants, its arithmetic or its
  serialization (`SCORING_VERSION`),
- an environment identity change (`ENVIRONMENT_PACK_VERSION`,
  `ENVIRONMENT_WORLD_VERSION`),
- a generator semantics or config format change (`GENERATOR_VERSION`,
  `GENERATOR_CONFIG_VERSION`, `GENERATION_MANIFEST_VERSION`),
- a change to the suite-definition rules: family rules
  (`FAMILY_RULES_VERSION`), the calibration panel, difficulty or admission
  rules (`CALIBRATION_PANEL_VERSION`, `DIFFICULTY_RULES_VERSION`,
  `ADMISSION_RULES_VERSION`), composition, selection, the recipe catalog or
  the suite freeze record (`COMPOSITION_VERSION`, `SELECTION_RULES_VERSION`,
  `RECIPE_CATALOG_VERSION`, `SUITE_FREEZE_RECORD_VERSION`),
- an evaluation manifest or result format change
  (`EVALUATION_MANIFEST_VERSION`, `EVALUATION_RESULT_VERSION`),
- discovered held-out leakage that the current protocol permits,
- an invalid evaluation protocol,
- a material bug that affects ranking or comparison.

`benchmark_versions()` pins every one of these versions, and each value is the
owning module's constant. Evaluation provenance records all of them.
`verify_evaluation_provenance` fails closed when any recorded value differs
from the running benchmark, or when a key is missing or unknown.

Examples:

| Change | Revision |
| --- | --- |
| A new private suite, generated under the identical revision and rules | Same revision |
| New evaluator-private seeds (generator or evaluation) under unchanged rules | Same revision |
| Changed family, calibration-panel, difficulty, admission, composition or selection rules | New revision |
| Changed scoring formula or scoring constants | New revision |
| Changed generator semantics or config format | New revision |

Scores from different revisions are never compared. Provenance of another
revision fails verification.

**Revision history.**

| Revision | Content |
| --- | --- |
| `1` | First integrity protocol. Never released as a scored benchmark. |
| `2` | The official EMBER Score (`SCORING_VERSION` 1.0) and environment packs, and the generator 1.1 suite-definition contracts (family rules, calibration, admission, composition and selection, all 1.0). These changes came between two releases, so one pre-release step covers them. |

A **regenerated held-out batch under the same revision** is different: new
private seed, new batch ID, new frozen suite and new manifest, with the same
code and versions. Use it when a batch is exposed (provider retention, a
leak, a reveal for audit) or retired on schedule. Rules:

- Never replace a compromised batch silently. Announce it, keep its
  provenance, and mark its results as compromised.
- Results on different batches are separate comparisons. Every model in one
  comparison is evaluated on one batch and one manifest.
- A protocol flaw that let leakage happen is a revision change, not just a
  new batch.

## 13. What EMBERBench cannot prove

- **Hashes establish content identity and integrity, not authorship.** A
  matching digest shows that two artifacts are byte-identical, not who made
  them or when.
- A `SubmissionIdentity` names content. It does not prove that the named code
  produced the actions, unless a trusted build or signature process stands
  behind the evaluation.
- An unverified signature proves nothing. EMBERBench never checks a
  signature itself.
- The evaluator is trusted. A malicious evaluator can leak a batch, choose a
  favourable seed, or run another model; the protocol makes these harder to
  hide, not impossible.
- Without an external timestamp, records do not prove when a step happened.
  Post-submission generation proves an order only through the roster digest
  and the later seed reveal.
- A clean scan for sensitive tokens does not prove that a model never saw
  held-out material elsewhere.
