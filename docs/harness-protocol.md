# Harness protocol (harness 1.0)

This document freezes how a remote model (a decision model or a chat LLM) plays
an EMBERBench episode. The code is `src/emberbench/harness.py`
(`HARNESS_VERSION = "1.0"`, prompt design `ADAPTER_VERSION = "r1"`). The
versions are pinned by `benchmark_versions()["harness"]` and
`benchmark_versions()["adapter"]` from benchmark revision 3 (see
[integrity.md](integrity.md), section 12). Any change to a rule in this
document is a semantic change: bump `HARNESS_VERSION` (or `ADAPTER_VERSION`
for a prompt change) and the benchmark revision. `emberbench run --model`
([cli.md](cli.md)) is the canonical entry point.

The harness does not change the simulator, the scenarios, the metrics or the
EMBER Score. Offline agents (the baselines) do not use it.

## 1. Inputs

Every step the harness gets one `Observation` and nothing else. A prompt never
contains the scenario, its ID, the seed, the evaluation manifest or a suite
commitment. The state text, the questions and the system prompt are the `r1`
prompt design (`state_text`, `person_question`, `chat_prompt`, `SYSTEM_PROMPT`).

## 2. Joint protocol

The active people of a step are sorted by person ID (code-point order).

| Mode | Rule | Status |
|---|---|---|
| `full_joint` | All active people of the step in one request. | **Canonical** for every API style. |
| `fixed_chunk(N)` | Consecutive chunks of `N` people in the sorted order; one request per chunk, all with the same state. | Declared variant, only for a provider that caps the questions per request. |

- A `fixed_chunk` run records `joint_mode` and `chunk_size` in its provenance,
  and its model ID gets the suffix `-chunkN`. Its scores are reported apart
  from canonical scores.
- The protocol does not change during a run. Each joint mode and chunk size
  gives its own model ID, so its own result file: a resume never mixes two
  joint protocols in one result.
- Before harness 1.0 the decision style used `fixed_chunk(32)` by default.

**Why full joint.** Phase 1 ran both modes on the 10 public scenarios (19
unique worlds) with four decision models through the OpenRouter decisions API.

| Model | Full joint | Chunk 32 | Suite cost full / chunk (USD) |
|---|---:|---:|---:|
| `typesafe/jev-1.14` | 49.79 | 49.52 | 0.089 / 0.093 |
| `openai/gpt-6-luna-decisions` | 30.75 | 30.75 | 0.28 / - |
| `liquid/d1-ultra` | 25.71 | 25.73 | 0.10 / - |
| `inception/mercury-decide` | 20.89 | 20.89 | 0.059 / - |

- Every score difference was 0.29 points or less, and no paired difference
  was significant. The ranking did not change (Kendall tau-b = 1.0).
- Full joint used the fewest requests, the lowest cost and the lowest response
  time for every model (Jev p95 calls per episode: 71 full joint, 119 chunk 32,
  130 chunk 16).
- All four models answered 100 questions in one request without an invalid
  answer. Full joint is the smallest protocol that gives every person the same
  request, so it is the canonical choice.
- One provider (`nace-ai/drex-v2.0`) rejects more than 16 questions per
  request. It can run only as the declared variant `fixed_chunk(16)`: score
  32.97 (95% CI 31.09 to 35.73), suite cost 0.12 USD, no invalid slot. This
  score is a variant score and is reported apart from the canonical scores.

## 3. Validation, re-asks and fail-closed

- **Decision style.** One `choice` question per active person. The options are
  the allowed actions of that person. An answer is valid only when its
  `choice` is one of the options of its question. A missing question, an
  unknown option or a malformed response is an invalid slot.
- **Chat styles.** One JSON object that maps every person ID of the request to
  one allowed move, exactly as written. A missing person, an extra key, a wrong
  case or an unknown move makes the answer invalid. Nothing is repaired.
- **Re-ask.** For the decision style only the invalid questions are re-asked,
  with the same state. For chat styles the request is re-asked in the same
  conversation with the reason. At most `max_reasks = 2` re-asks per request.
  A re-ask is always a new model request, also when its body equals the
  first one; the answer cache keys it by its attempt number.
- **Fail-closed.** After the last re-ask the harness raises `AdapterError`. The
  episode is not committed, no action is guessed, and the run can be resumed.
- **Transport errors.** HTTP 429, 5xx and network errors are retried up to 5
  times with back-off. They are not re-asks and are not invalid slots.
- **Budget.** An optional spend limit (`--budget-usd`) stops the run before the
  next request; this also fails closed.

## 4. Executed action and probabilities

- The executed action is the model's `choice`. The harness never samples from
  the probabilities and never replaces the choice with their argmax.
- When a decision model returns `probabilities`, the harness records, per slot,
  the probability of the choice, the entropy (bits), the margin (top 1 minus
  top 2) and whether the choice is the argmax. A distribution is invalid when
  its keys are not exactly the options, a value is negative or not finite, or
  the sum is not positive; an invalid distribution is counted and does not
  invalidate the answer.
- These are **descriptive probability diagnostics, not calibration**: there are
  no target labels for the correct action. They never change a score.

## 5. Suite commitment

An evaluator publishes `suite_commitment(pack_digest, salt)` before an
evaluation (`src/emberbench/commitment.py`, `COMMITMENT_VERSION = "1.0"`):

```text
sha256("emberbench-suite-commitment/1.0\n" + pack_digest + "\n" + salt + "\n")
```

`pack_digest` is the environment-pack digest; it binds every scenario file and
the seed schedule. `salt` is 256 random bits, kept private with the pack. The
commitment reveals nothing about the suite. A later reveal of the pack and the
salt lets anyone check it with `verify_suite_commitment`. A new hidden batch
publishes a new commitment; the scoring contract does not change.

## 6. Termination and horizon

- An episode ends when no active people remain (`all_resolved`) or at the
  scenario's own `termination.max_steps` (`max_steps`). The harness never
  stops an episode early and never changes the horizon.
- The public scenarios have horizons from 40 to 200 steps (the generated suites
  use 100). Phase 1 evidence: the strongest baselines resolve every public
  scenario in at most 43 steps; the only horizon hit of any model was
  `v1-pub-08-maze-deceptive-exit` (horizon 120, solved by baselines in 30
  steps), with 3 or 4 people stranded. A horizon hit is a model failure, not a
  short horizon, so the horizon stays as it is.
- The horizon is never lowered to save cost. A budget variant (for example a
  lower step cap) is a separate, labelled protocol and never the main
  benchmark.

## 7. Accounting

Every request (live or from the cache) is recorded in a ledger: attempt,
questions, invalid slots, tokens, cost, latency, cache flag. The accounting
summary (`src/emberbench/accounting.py`, `ACCOUNTING_VERSION = "1.0"`) gives
median and p95 per episode of calls, steps, tokens, cost and response time;
the re-ask rate, the invalid-slot rate and the re-ask cost multiplier
(total cost / first-attempt cost); and horizon hits.

## 8. Provenance

A run records `HarnessProtocol.to_dict()`: `harness_version`,
`adapter_version`, `joint_mode`, `chunk_size`, `person_order`, `max_reasks` and
`executed_action`; and the SHA-256 of the harness module. Together with
`benchmark_versions()`, the pack digest and the model settings, this defines
the run.
