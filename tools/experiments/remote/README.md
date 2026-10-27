# Any remote model on the V1 public development pack (research lane)

**PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD**

`run_remote.py` evaluates any model behind an HTTP API. You give a base URL, an API key and a model
name. The runner uses the same environments and the same evaluation method as the TypeSafe Jev and
SubQ Snap runs:

- the same frozen environment pack of the public V1 suite (digest `ca18c895…b506e`; the runner stops
  if the digest is different),
- the same unique-world manifest (19 worlds: the nine deterministic scenarios once, 10 fire worlds of
  `v1-pub-10-route-invalidation`),
- `run_evaluation` with replay authentication of every episode,
- the official `score_result` (EMBER Score, scoring version 1.0) and the paired `compare_scores`
  against the reference runs.

It is not a registered baseline and not part of the release. It needs network access, so it does not
run under the evaluation pack's no-network rule.

## API styles

| `--api` | Request | Use it for |
|---|---|---|
| `openai` (default) | `POST {base_url}/chat/completions`, `Authorization: Bearer <key>` | OpenAI and every OpenAI-compatible server: OpenRouter, Together, Groq, DeepSeek, Mistral, vLLM, Ollama, LM Studio |
| `anthropic` | `POST {base_url}/messages`, `x-api-key: <key>` | The Anthropic Messages API |
| `decision` | `POST {base_url}/systemone`, body `{model, state, questions}` | TypeSafe Jev, SubQ Snap and other decision APIs with this body |

If the base URL already ends with the endpoint path, the runner uses it as it is.

## Adapter (version `r1`)

- **Input.** Each `act` call gets one `Observation` and nothing else. The prompt never contains the
  scenario, its ID, the seed or the manifest.
- **State text.** The same text as the Jev adapter: the rules, an ASCII map (`#` wall, `F` fire, `E`
  exit, `@` person, `.` floor), the exits and the people. For each person, every allowed move with the
  cell it leads to and the straight-line distances to the nearest exit and fire.
- **Harness protocol (harness `1.0`, [docs/harness-protocol.md](../../../docs/harness-protocol.md)).**
  The canonical protocol is full joint: all active people of a step go in one request, for every API
  style. `--joint-mode fixed_chunk --chunk-size N` (or `--max-people N`) is a declared variant for a
  provider with a request cap; its model ID gets the suffix `-chunkN`. Before harness `1.0` the decision
  style used chunks of 32 by default.
- **Chat styles.** The model must answer with one JSON object: each key is a person ID, each value is
  an allowed move. Code fences and `<think>` blocks are skipped. Nothing is repaired: a wrong case, a
  missing person or an unknown move is an invalid answer. The runner re-asks up to `--format-retries`
  times (default 2) with the reason. If the answer is still invalid, the run stops (fail-closed) and
  you can resume it.
- **Decision style.** One `choice` question per person. The request bodies are byte-identical to the
  Jev adapter `r1`. Only the invalid questions are re-asked, with the same state, up to
  `--format-retries` times; then the run stops (fail-closed).
- **Errors.** Up to 5 retries on HTTP 429, 5xx and network errors. `--retry-403` adds long waits on
  HTTP 403, for a provider that throttles with 403 (SubQ preview).
- **Defaults.** `temperature` 0 (`--temperature omit` for models that reject it). `max_tokens` is not
  sent, except 2048 for `anthropic`. `--extra-body '{"reasoning_effort": "low"}'` adds provider fields.
- **Cost.** The runner adds up the cost that the API reports (OpenRouter: `usage.cost`). If the API
  reports no cost, give the list price with `--input-price` and `--output-price` (USD per million
  tokens). `summary.json` records the cost and its source: `reported`, `list price` or `not reported`.
- **Time.** `summary.json` records the wall-clock time of the run, the mean time per `act` call and the
  first and median response times. No API style streams, so the first token arrives with the whole
  answer: the response time is the time to first token. Answers from the cache are not timed.

## Reproducibility

Every first request body is canonical JSON. The runner caches each accepted answer by the SHA-256 of
that body in `<out>/cache.jsonl`. A resume, or a new run with only the cache file, sends no request and
gives byte-identical result and score files. An answer that the runner did not accept is not cached.
`summary.json` records the models that answered.

A check of the decision style: the cache of the Jev run, replayed through this runner with
`--api decision --provider typesafe --model jev-latest`, sent 0 requests. It gave a result file and a
score report that are byte-identical to the Jev run (EMBER Score 48.27).

## Run

The key comes only from the environment variable named by `--key-env` (default
`EMBERBENCH_API_KEY`). It is never written to a file or printed. The output directory must be outside
the repository. On Windows keep its path short.

```bash
export EMBERBENCH_API_KEY=...   # your key; do not commit it
PYTHONPATH=src python tools/experiments/remote/run_remote.py --base-url https://api.openai.com/v1 --model gpt-4o-mini --out ../remote-run --reference-run ../pack-run
```

```bash
PYTHONPATH=src python tools/experiments/remote/run_remote.py --api anthropic --base-url https://api.anthropic.com/v1 --model claude-sonnet-5-5 --out ../claude-run --reference-run ../pack-run
```

```bash
PYTHONPATH=src python tools/experiments/remote/run_remote.py --api decision --base-url https://api.subq.ai/v1 --model subq-snap-preview --provider subq --max-people 16 --retry-403 --out ../subq-run --reference-run ../pack-run
```

`--reference-run` is optional and can be given more than once. It is a finished run directory with
`environment-pack/` and `results/`: the evaluation-pack output of
`tools/experiments/v1/run_public_benchmark.py`, or an earlier Jev, SubQ or remote run. If the directory
does not exist, the runner first makes it with the offline evaluation pack. `--protocol smoke` runs the
tiny CI subset; its numbers mean nothing.

The model ID is `<provider>-<model>-chat-r1` (`-choice-r1` for the decision style). The provider comes
from the base URL host unless you give `--provider`.

## Outputs

| Path (relative to `--out`) | Content |
|---|---|
| `environment-pack/`, `source-suite/` | The frozen pack (the same digest as the evaluation pack) |
| `results/<model-id>.json` | The `EvaluationResult` |
| `scores/<model-id>.json` | The official `ScoreReport` |
| `cache.jsonl` | The accepted answers, by request SHA-256 |
| `summary.json` | Score, safety, quality, totals, per-scenario scores, API use and every paired comparison |
| `REPORT.md` | The result, the comparison table with the reference runs and the score per scenario |

## Limits

- Public suite only. These are development results, not hidden-suite or leaderboard scores.
- One prompt design (`r1`). The score measures the model with this adapter, not the model alone.
- One run per world. The cache makes the artifact reproducible, not the model.
