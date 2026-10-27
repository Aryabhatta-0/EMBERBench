"""Evaluate any remote model, given a base URL and an API key, on the V1 public development pack.

This is the general form of the Jev and SubQ Snap runs (research lane, not a
registered baseline). It uses the same frozen environment pack, the same
unique-world manifest, the same ``run_evaluation`` replay-authenticated
results and the same official ``score_result`` / ``compare_scores`` procedure.
Only the transport changes. Three API styles are supported:

``openai`` (default)
    ``POST {base_url}/chat/completions`` with ``Authorization: Bearer``. Any
    OpenAI-compatible server (OpenAI, OpenRouter, Together, Groq, DeepSeek,
    vLLM, Ollama, LM Studio, ...). One request per step asks for one JSON
    object that maps every person ID to one allowed move.
``anthropic``
    ``POST {base_url}/messages`` with ``x-api-key`` (the Anthropic Messages API).
    The same prompt and the same JSON answer.
``decision``
    ``POST {base_url}/systemone`` with the ``{model, state, questions}`` body of
    TypeSafe Jev and SubQ Snap. The request bodies are byte-identical to the
    Jev adapter ``r1``, so a Jev or SubQ cache gives the same answers here.

Every prompt is built from the serialized :class:`~emberbench.Observation`
only (model-interface.md section 3, "future external or remote adapter"):
never the scenario, its ID, the seed or the manifest.

Reproducibility: every first request body is canonical JSON, and the accepted
responses are cached by its SHA-256 in ``<out>/cache.jsonl``. A resume or a
second run with the same cache sends no request and gives byte-identical
artifacts. A fresh cache can give other answers; the models that answered are
recorded in ``summary.json``.

Failure is fail-closed (model-interface.md section 6): an HTTP error after the
retries, a timeout, or an answer that is still not valid after
``--format-retries`` corrective re-asks raises; the episode is not committed and
``--out`` can be resumed. Only accepted answers are cached.

The key is read from the environment variable named by ``--key-env``
(default ``EMBERBENCH_API_KEY``). It is never written to a file or printed.

Usage (``--out`` must be outside the repository)::

    EMBERBENCH_API_KEY=... python tools/experiments/remote/run_remote.py \\
        --base-url https://api.openai.com/v1 --model gpt-4o-mini --out ../remote-run \\
        --reference-run ../pack-run
"""

import argparse
import datetime
import hashlib
import json
import os
import statistics
import sys
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from emberbench import (  # noqa: E402
    ScoreReport,
    compare_scores,
    load_environment_pack,
    load_evaluation_result,
    score_result,
)
from emberbench import harness  # noqa: E402
from emberbench.harness import (  # noqa: E402,F401  (re-exported: the runner's public names)
    ADAPTER_VERSION,
    ANTHROPIC_VERSION_HEADER,
    API_STYLES,
    DEFAULT_KEY_ENV,
    ENDPOINTS,
    FORBIDDEN_RETRIES,
    FORBIDDEN_WAITS,
    GOAL,
    RETRIES,
    SYSTEM_PROMPT,
    TIMEOUT_SECONDS,
    AdapterError,
    Exchange,
    HarnessProtocol,
    JointMode,
    Ledger,
    RemoteAgent,
    RemoteClient,
    chat_prompt,
    label,
    move_options,
    parse_moves,
    person_question,
    provider_label,
    state_text,
)
from emberbench.run import RunError, evaluate_pack  # noqa: E402

import pack_protocol  # noqa: E402
import run_public_benchmark as rpb  # noqa: E402

EXPECTED_PACK_DIGEST = "ca18c895f6f655f85448dd3bcd8a02a776a2c21bd28e13d5a3aa91b3955b506e"
MAX_QUESTIONS = harness.DEFAULT_CHUNK_SIZE  # decision style default: questions per request (fixed chunks)
FORMAT_RETRIES = harness.MAX_REASKS


def endpoint_url(base_url: str, api: str) -> str:
    """``harness.endpoint_url``; a bad URL is a usage error of this runner."""
    try:
        return harness.endpoint_url(base_url, api)
    except ValueError:
        raise SystemExit(f"--base-url must be an http(s) URL, not {base_url!r}") from None


# --------------------------------------------------------------------------- results and report


def _f(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _pts(value: Fraction | None) -> str:
    return "-" if value is None else f"{float(value):.2f}"  # scores are on the 0-100 scale


def _signed(value: Fraction | None) -> str:
    return "-" if value is None else f"{float(value):+.2f}"


OUTCOMES = {"a_better": "model better", "b_better": "reference better", "inconclusive": "inconclusive", "exact_tie": "exact tie"}


def score_summary(report: ScoreReport) -> dict[str, Any]:
    ci = report.ci
    return {
        "score": _f(report.score),
        "ci95": None if ci is None else [_f(ci.lower), _f(ci.upper)],
        "safety": _f(report.safety),
        "quality": _f(report.quality),
        "evacuated": report.totals.evacuated,
        "avoidable_casualties": report.totals.avoidable_casualties,
        "avoidable_stranded": report.totals.avoidable_stranded,
        "unavoidable_losses": report.totals.unavoidable_losses,
        "worst_scenario": _f(report.robustness.worst_scenario),
        "worst_scenario_id": report.robustness.worst_scenario_id,
    }


def reference_results(paths: Sequence[Path], pack: Any, protocol: pack_protocol.Protocol, own_id: str) -> list[tuple[Any, ScoreReport]]:
    """Every result file in the ``results/`` of each reference run, scored again against ``pack``."""
    scored: list[tuple[Any, ScoreReport]] = []
    seen: set[str] = {own_id}
    for run in paths:
        ref_pack = load_environment_pack(run / rpb.PACK_DIR)
        if ref_pack.digest != pack.digest:
            raise SystemExit(f"{run}: the reference run used another environment pack ({ref_pack.digest})")
        for ref_path in sorted((run / "results").glob("*.json")):
            ref = load_evaluation_result(ref_path, pack.suite)  # authenticate every episode by replay
            if ref.model_id in seen or not ref.is_complete:
                continue
            seen.add(ref.model_id)
            scored.append((ref, score_result(ref, pack, resamples=protocol.resamples)))
    return scored


def render_report(summary: Mapping[str, Any], report: ScoreReport, refs: Sequence[tuple[ScoreReport, Any]]) -> str:
    """A short Markdown report: the result, the paired comparisons and the score per scenario."""
    model_id = summary["model_id"]
    ci = report.ci
    api = summary["api"]
    lines = [
        f"# `{model_id}` on the V1 public development pack",
        "",
        f"**{pack_protocol.RUN_LABEL}**",
        "",
        "Official EMBER Score procedure (scoring version 1.0) on the frozen environment pack of the **public**",
        "V1 suite: the same pack, manifest and scoring as the Jev and SubQ Snap runs. The suite is public, so",
        "these are development results only.",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Model ID (evaluator label) | `{model_id}` |",
        f"| API style / endpoint | `{summary['api_style']}` / `{summary['endpoint']}` |",
        f"| Requested model | `{summary['requested_model']}` |",
        f"| Model that answered | {', '.join(f'`{m}`' for m in summary['resolved_models']) or '- (all from the cache)'} |",
        f"| **EMBER Score (family-balanced)** | **{_pts(report.score)}**"
        + ("" if ci is None else f" (95% CI {_pts(ci.lower)} to {_pts(ci.upper)})")
        + " |",
        f"| Safety | {_f(report.safety)} |",
        f"| Quality | {_f(report.quality)} |",
        f"| Worst scenario | {_pts(report.robustness.worst_scenario)} (`{report.robustness.worst_scenario_id}`) |",
        f"| Evacuated | {report.totals.evacuated} |",
        f"| Avoidable casualties / stranded | {report.totals.avoidable_casualties} / {report.totals.avoidable_stranded} |",
        f"| Episodes (unique worlds) | {summary['episodes']} |",
        f"| Official-procedure score | {'yes' if report.official else 'no'} |",
        f"| Pack digest | `{summary['pack_digest']}` |",
        "",
        "| API item | Value |",
        "|---|---|",
        f"| `act` calls (steps) | {api['act_calls']} |",
        f"| HTTP requests | {api['requests_sent']} |",
        f"| Answers from the cache | {api['cache_hits']} |",
        f"| Transport retries (429/5xx/network, 403) | {api['retries']}, {api['retries_403']} |",
        f"| Invalid answers re-asked | {api['format_errors']} |",
        f"| Input / output tokens | {api['input_tokens']:,} / {api['output_tokens']:,} |",
        f"| Cost (USD, {api['cost_source']}) | {api['cost']} |",
        f"| Mean `act` latency | {api['mean_act_latency_s']} s |",
        f"| First / median response time | {api['first_response_s']} s / {api['median_response_s']} s |",
        f"| Wall-clock time | {api['wall_clock_s']} s |",
    ]
    if refs:
        rows = sorted([(report, None), *refs], key=lambda r: -(float(r[0].score) if r[0].score is not None else -1.0))
        lines += [
            "",
            "## Comparison with the reference runs",
            "",
            "Delta is this model minus the reference: the paired, family-balanced difference with a 95%",
            "bootstrap interval. Worlds = paired worlds where this model / the reference scored higher / equal.",
            "",
            "| Model | EMBER Score | Evacuated | Avoidable casualties | Delta | Delta 95% CI | Outcome | Worlds |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
        for r, cmp in rows:
            if cmp is None:
                lines.append(f"| **`{r.model_id}`** | **{_pts(r.score)}** | **{r.totals.evacuated}** | **{r.totals.avoidable_casualties}** | | | | |")
                continue
            span = "-" if cmp.ci is None else f"{_signed(cmp.ci.lower)} to {_signed(cmp.ci.upper)}"
            lines.append(
                f"| `{r.model_id}` | {_pts(r.score)} | {r.totals.evacuated} | {r.totals.avoidable_casualties} | {_signed(cmp.delta)} "
                f"| {span} | {OUTCOMES[cmp.outcome.value]} | {cmp.a_higher} / {cmp.b_higher} / {cmp.equal} |"
            )
    best: dict[str, tuple[Fraction, str]] = {}
    for r, _ in refs:
        for g in r.scenarios:
            if g.score is not None and (g.name not in best or g.score > best[g.name][0]):
                best[g.name] = (g.score, r.model_id)
    lines += ["", "## Score per scenario", ""]
    if refs:
        lines += ["| Scenario | Worlds | This model | Best reference | Best reference model |", "|---|---:|---:|---:|---|"]
        for g in report.scenarios:
            b_score, b_id = best.get(g.name, (None, "-"))
            lines.append(f"| `{g.name}` | {g.episodes} | {_pts(g.score)} | {_pts(b_score)} | `{b_id}` |")
    else:
        lines += ["| Scenario | Worlds | Score | Safety |", "|---|---:|---:|---:|"]
        lines += [f"| `{g.name}` | {g.episodes} | {_pts(g.score)} | {_f(g.safety)} |" for g in report.scenarios]
    lines += [
        "",
        "## Limits",
        "",
        "- Public suite only: development results, not hidden-suite or leaderboard scores.",
        f"- One prompt design (`{ADAPTER_VERSION}`). The score measures the model with this adapter, not the model alone.",
        "- One run per world. The cache makes the artifact reproducible, not the model.",
        "",
    ]
    return "\n".join(lines)


def _temperature(text: str) -> float | None:
    return None if text.lower() in ("omit", "none") else float(text)


def _style_default(api: str, max_reasks: int) -> HarnessProtocol:
    """The canonical harness 1.0 protocol for every API style: full joint (docs/harness-protocol.md, section 2).

    Before harness 1.0 the decision style used fixed chunks of 32. A chunked run is
    now a declared variant: its model ID carries the ``-chunkN`` suffix.
    """
    del api
    return HarnessProtocol(JointMode.FULL_JOINT, None, max_reasks)


def _harness_protocol(args: argparse.Namespace) -> HarnessProtocol:
    try:
        if args.max_people is not None:
            if args.joint_mode is not None or args.chunk_size is not None:
                raise SystemExit("give --max-people or --joint-mode/--chunk-size, not both")
            if args.max_people == 0:
                return HarnessProtocol(JointMode.FULL_JOINT, None, args.format_retries)
            return HarnessProtocol(JointMode.FIXED_CHUNK, args.max_people, args.format_retries)
        if args.joint_mode is None:
            if args.chunk_size is not None:
                return HarnessProtocol(JointMode.FIXED_CHUNK, args.chunk_size, args.format_retries)
            return _style_default(args.api, args.format_retries)
        mode = JointMode(args.joint_mode)
        size = (args.chunk_size or MAX_QUESTIONS) if mode is JointMode.FIXED_CHUNK else args.chunk_size
        return HarnessProtocol(mode, size, args.format_retries)
    except ValueError as error:
        raise SystemExit(f"harness protocol: {error}") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", required=True, help="API base URL, for example https://api.openai.com/v1")
    parser.add_argument("--model", required=True, help="model name sent in every request")
    parser.add_argument("--out", type=Path, required=True, help="output directory, outside the repository")
    parser.add_argument("--api", choices=API_STYLES, default="openai", help="API style (default: openai-compatible chat)")
    parser.add_argument("--key-env", default=DEFAULT_KEY_ENV, help=f"environment variable that holds the API key (default {DEFAULT_KEY_ENV})")
    parser.add_argument("--provider", help="provider label in the model ID (default: from the base URL host)")
    parser.add_argument("--protocol", choices=sorted(pack_protocol.PROTOCOLS), default=pack_protocol.PUBLIC.name)
    parser.add_argument(
        "--reference-run", type=Path, action="append", default=[],
        help="a finished run directory (environment-pack/ and results/) to compare against; repeatable. "
        "A missing directory is first filled with the offline evaluation-pack run.",
    )  # fmt: skip
    parser.add_argument("--temperature", type=_temperature, default=0.0, help="chat temperature, or 'omit' (default 0)")
    parser.add_argument("--max-tokens", type=int, help="chat max_tokens (default: omitted; 2048 for anthropic)")
    parser.add_argument("--extra-body", default="{}", help="JSON object merged into every chat request body")
    parser.add_argument("--joint-mode", choices=[m.value for m in JointMode], help="full_joint: every active person in one request; "
                        "fixed_chunk: --chunk-size people per request (default: full_joint, the canonical harness 1.0 protocol)")
    parser.add_argument("--chunk-size", type=int, help=f"people per request in fixed_chunk mode (default {MAX_QUESTIONS})")
    parser.add_argument("--max-people", type=int, help="historical form: 0 is full_joint, n is fixed_chunk with n people")
    parser.add_argument("--format-retries", type=int, default=FORMAT_RETRIES, help="re-asks after an invalid answer")
    parser.add_argument("--cache", type=Path, help="answer cache file (default <out>/cache.jsonl); can be shared between protocol runs")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="seconds per HTTP request")
    parser.add_argument("--retry-403", action="store_true", help="retry HTTP 403 with long waits (a provider that throttles with 403)")
    parser.add_argument("--budget-usd", type=float, help="stop (resumable) once the API-reported cost of this invocation reaches this many USD")
    parser.add_argument("--input-price", type=float, help="USD per million input tokens, used when the API reports no cost")
    parser.add_argument("--output-price", type=float, help="USD per million output tokens, used when the API reports no cost")
    args = parser.parse_args(argv)

    key = os.environ.get(args.key_env, "")
    if not key and urllib.parse.urlsplit(args.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"set the API key in the environment variable {args.key_env}")
    extra_body = json.loads(args.extra_body)
    if not isinstance(extra_body, dict):
        raise SystemExit("--extra-body must be a JSON object")
    harness_protocol = _harness_protocol(args)

    protocol = pack_protocol.PROTOCOLS[args.protocol]
    rpb.check_benchmark_versions()
    out = rpb._check_out_dir(args.out)
    suite = rpb.protocol_suite(protocol)
    manifest = rpb.protocol_manifest(protocol, suite)
    pack = rpb._environment_pack(protocol, suite, manifest, out)
    suite, manifest = pack.suite, pack.manifest
    if protocol is pack_protocol.PUBLIC and pack.digest != EXPECTED_PACK_DIGEST:
        raise SystemExit(f"environment pack digest {pack.digest} is not the released {EXPECTED_PACK_DIGEST}")
    print(f"pack     {pack.pack_id} sha256:{pack.digest}: {len(manifest.episodes)} episodes", flush=True)

    for run in args.reference_run:
        if not run.exists():
            print(f"reference {run}: running the offline evaluation pack first", flush=True)
            rpb.run_pack(protocol, run)

    client = RemoteClient(
        api=args.api,
        base_url=args.base_url,
        model=args.model,
        key=key,
        cache_path=args.cache or out / "cache.jsonl",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        format_retries=args.format_retries,
        timeout=args.timeout,
        retry_403=args.retry_403,
        budget_usd=args.budget_usd,
    )
    provider = label(args.provider or provider_label(args.base_url))
    model_id = harness.remote_model_id(provider, args.model, args.api, harness_protocol)
    print(f"model    {model_id} via {client.url}", flush=True)
    segment = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    prices = (args.input_price or 0.0, args.output_price or 0.0) if args.input_price is not None or args.output_price is not None else None

    def make_factory(ledger: Ledger | None) -> Callable[[], RemoteAgent]:
        return lambda: RemoteAgent(client, provider=provider, protocol=harness_protocol, ledger=ledger)

    try:
        evaluation = evaluate_pack(
            pack, out, model_id, make_factory,
            ledger_factory=lambda path, pending: Ledger(path, segment, pending), resamples=protocol.resamples, prices=prices,
        )  # fmt: skip
    except RunError as error:
        raise SystemExit(error.message) from None
    result, report, accounting, steps, elapsed = evaluation.result, evaluation.report, evaluation.accounting, evaluation.steps, evaluation.elapsed_s

    refs = [(r, compare_scores(report, r, resamples=protocol.resamples)) for _, r in reference_results(args.reference_run, pack, protocol, model_id)]
    cost, cost_source = client.cost, "reported"
    if cost == 0 and (args.input_price is not None or args.output_price is not None):
        cost = (client.input_tokens * (args.input_price or 0) + client.output_tokens * (args.output_price or 0)) / 1e6
        cost_source = "list price"
    elif cost == 0:
        cost_source = "not reported"
    summary: dict[str, Any] = {
        "model_id": model_id,
        "api_style": args.api,
        "endpoint": client.url,
        "requested_model": args.model,
        "resolved_models": sorted(client.resolved_models),
        "adapter_version": ADAPTER_VERSION,
        "harness_protocol": harness_protocol.to_dict(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(harness.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "protocol": protocol.name,
        "pack_digest": pack.digest,
        "manifest_hash": manifest.manifest_hash(),
        "episodes": len(result.episodes),
        "official_procedure": report.official,
        "result": score_summary(report),
        "scenarios": {g.name: _f(g.score) for g in report.scenarios},
        "api": {
            "requests_sent": client.calls,
            "cache_hits": client.cache_hits,
            "retries": client.retries,
            "retries_403": client.retries_403,
            "format_errors": client.format_errors,
            "input_tokens": client.input_tokens,
            "output_tokens": client.output_tokens,
            "cost": round(cost, 6),
            "cost_source": cost_source,
            "act_calls": len(steps),
            "mean_act_latency_s": _f(sum(s.latency_seconds for s in steps) / len(steps), 3) if steps else None,
            "first_response_s": round(client.response_seconds[0], 3) if client.response_seconds else None,
            "median_response_s": round(statistics.median(client.response_seconds), 3) if client.response_seconds else None,
            "wall_clock_s": round(elapsed, 1),
        },
        "accounting": {k: v for k, v in accounting.items() if k != "per_episode"},
        "references": {
            r.model_id: {
                **score_summary(r),
                "delta": _f(cmp.delta),
                "delta_ci95": None if cmp.ci is None else [_f(cmp.ci.lower), _f(cmp.ci.upper)],
                "outcome": cmp.outcome.value,
                "model_higher": cmp.a_higher,
                "ref_higher": cmp.b_higher,
                "equal": cmp.equal,
            }
            for r, cmp in refs
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii", newline="\n")
    (out / "REPORT.md").write_text(render_report(summary, report, refs), encoding="utf-8", newline="\n")
    ci = report.ci
    print(
        f"score    {_pts(report.score)}" + ("" if ci is None else f" (95% CI {_pts(ci.lower)} to {_pts(ci.upper)})")
        + f"  safety {_f(report.safety)}  quality {_f(report.quality)}  evacuated {report.totals.evacuated}"
        + f"  avoidable casualties {report.totals.avoidable_casualties}",
        flush=True,
    )  # fmt: skip
    for r, cmp in refs:
        print(f"  vs {r.model_id:<40} {_pts(r.score):>6}  delta {_signed(cmp.delta):>7}  {OUTCOMES[cmp.outcome.value]}", flush=True)
    print(f"report   {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
