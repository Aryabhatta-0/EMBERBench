"""``emberbench run``: evaluate one model on one frozen environment pack, score it and record provenance.

One command for every kind of model::

    emberbench run --suite PACK_DIR --model typesafe/jev-1.14 --out RUN_DIR      # a remote decision model
    emberbench run --suite PACK_DIR --baseline greedy-v1 --out RUN_DIR             # a registered baseline
    emberbench run --suite PACK_DIR --agent mod:factory --model-id ID --out RUN_DIR  # a local agent

``PACK_DIR`` is any frozen environment pack: the public development pack, the
public validation set or an evaluator-private hidden batch. The pack is
verified before any episode runs (``--expected-digest``; for a hidden batch
also ``--commitment`` with the salt in an environment variable). Remote models
run through the harness (docs/harness-protocol.md); this module imports it only
for ``--model``, so a local run loads no network code.

``RUN_DIR`` gets ``results/``, ``scores/``, ``accounting/`` and ``runs/`` (the
run record), plus ``ledger/`` and ``cache.jsonl`` for a remote model, and
``refusals/`` when the model refused a question. A refused episode scores 0
(failure kind ``refusal``) and the run continues; a rerun with the same model
resumes and does not run a refused episode again. With ``--private`` the record and the printed
summary hold no scenario ID, seed or answer text; errors are redacted.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .accounting import accounting_summary, pending_episodes, read_ledger
from .agent import AgentFactory, StepExecution, agent_policy_factory
from .baselines import BASELINE_IDS, baseline_factory
from .commitment import suite_commitment
from .environment_pack import EnvironmentPack, load_environment_pack
from .errors import EmberBenchError, EpisodeRefused
from .evaluation import EpisodeSpec, EvaluationResult, check_model_id, load_evaluation_result, run_evaluation
from .integrity import benchmark_versions
from .scoring import FailureKind, ScoreReport, score_result
from .statistics import check_resamples

RUN_RECORD_VERSION = "1.1"  # 1.1: refusals
DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_RESAMPLES = 2000
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


class RunError(Exception):
    """A refused or failed run. ``kind`` is ``usage``, ``mismatch``, ``output`` or ``agent``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class Evaluation:
    result: EvaluationResult
    report: ScoreReport
    accounting: dict[str, Any]
    steps: list[StepExecution]
    elapsed_s: float
    result_path: Path
    refused: tuple[int, ...] = ()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="ascii", newline="\n")


def _f(v: Any, digits: int = 4) -> float | None:
    return None if v is None else round(float(v), digits)


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings (stable across checkouts)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def evaluate_pack(
    pack: EnvironmentPack,
    out: Path,
    model_id: str,
    make_factory: Callable[[Any], AgentFactory],
    *,
    ledger_factory: Callable[[Path, list[int]], Any] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    prices: tuple[float, float] | None = None,
) -> Evaluation:
    """Run (or resume) ``model_id`` on every pack episode, authenticate by replay, account and score.

    ``make_factory(ledger)`` returns the agent factory; ``ledger`` is ``None``
    unless ``ledger_factory(path, pending_episode_indexes)`` is given (remote
    models). Writes ``results/``, ``scores/`` and ``accounting/`` under ``out``.

    An episode whose policy raises :class:`EpisodeRefused` is recorded in
    ``refusals/<model_id>.json`` (episode index, step, question; no text), is
    not run again on resume and is scored as the declared failure ``refusal``
    (score 0). It never changes another episode's result.
    """
    path = out / "results" / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    refusal_path = out / "refusals" / f"{model_id}.json"
    refusals = load_refusals(refusal_path, pack, model_id)
    ledger_path = out / "ledger" / f"{model_id}.jsonl"
    ledger = None
    if ledger_factory is not None:
        ledger_path.parent.mkdir(exist_ok=True)
        ledger = ledger_factory(ledger_path, pending_episodes(path, pack.suite, pack.manifest, skip=refusals))

    def on_refusal(spec: EpisodeSpec, refusal: EpisodeRefused) -> None:
        refusals[spec.index] = {"index": spec.index, "step": refusal.step, "question": refusal.question}
        log = {"model_id": model_id, "manifest_hash": pack.manifest.manifest_hash(), "refused": [refusals[i] for i in sorted(refusals)]}
        _write_json(refusal_path, log)

    steps: list[StepExecution] = []
    started = time.perf_counter()
    policy = agent_policy_factory(make_factory(ledger), on_step=steps.append)
    run_evaluation(pack.suite, pack.manifest, model_id, policy, path, resume=True, on_refusal=on_refusal, skip=frozenset(refusals))
    elapsed = time.perf_counter() - started
    result = load_evaluation_result(path, pack.suite)  # authenticates every episode by replay
    done = {e.episode.index for e in result.episodes}
    if len(done) + len(refusals) != len(pack.manifest.episodes) or done & set(refusals):
        raise RunError("output", f"{model_id}: {len(done)} of {len(pack.manifest.episodes)} episodes completed, {len(refusals)} refused")
    refused = tuple(sorted(refusals))
    rows = read_ledger(ledger_path) if ledger_factory is not None else []
    accounting = accounting_summary(rows, result, prices=prices, refused=refused)
    _write_json(out / "accounting" / f"{model_id}.json", accounting)
    report = score_result(result, pack, failures={i: FailureKind.REFUSAL for i in refused}, resamples=resamples)
    (out / "scores").mkdir(exist_ok=True)
    (out / "scores" / f"{model_id}.json").write_text(report.to_json() + "\n", encoding="ascii", newline="\n")
    return Evaluation(result, report, accounting, steps, elapsed, path, refused)


def load_refusals(path: Path, pack: EnvironmentPack, model_id: str) -> dict[int, dict[str, Any]]:
    """The refusal log of ``model_id`` on ``pack`` (empty when there is none); it must belong to both."""
    if not path.exists():
        return {}
    log = json.loads(path.read_text(encoding="ascii"))
    if log.get("model_id") != model_id or log.get("manifest_hash") != pack.manifest.manifest_hash():
        raise RunError("output", f"{path}: refusal log of another model or manifest")
    entries = {int(e["index"]): e for e in log["refused"]}
    if not set(entries) <= {e.index for e in pack.manifest.episodes}:
        raise RunError("output", f"{path}: refusal log names unscheduled episodes")
    return entries


def refusal_summary(evaluation: Evaluation) -> dict[str, Any]:
    """Refused episodes, reported apart from the score: count, rate over the scheduled episodes, indices."""
    scheduled = len(evaluation.result.manifest.episodes)
    return {
        "count": len(evaluation.refused),
        "rate": round(len(evaluation.refused) / scheduled, 6) if scheduled else None,
        "episodes": list(evaluation.refused),
        "failure_kind": FailureKind.REFUSAL.value,
    }


def score_summary(report: ScoreReport) -> dict[str, Any]:
    ci = report.ci
    return {
        "score": _f(report.score, 3),
        "ci95": None if ci is None else [_f(ci.lower, 3), _f(ci.upper, 3)],
        "safety": _f(report.safety),
        "quality": _f(report.quality),
        "evacuated": report.totals.evacuated,
        "avoidable_casualties": report.totals.avoidable_casualties,
        "official": report.official,
        "families": {g.name: _f(g.score, 3) for g in report.families},
        "score_report_digest": report.digest(),
    }


# --------------------------------------------------------------------------- arguments


def add_arguments(p: argparse.ArgumentParser) -> None:
    """The ``emberbench run`` options (registered by the CLI)."""
    p.add_argument("--suite", required=True, metavar="DIR", help="a frozen environment pack directory")
    p.add_argument("--expected-digest", metavar="HEX", help="refuse unless the pack digest is exactly this")
    p.add_argument("--commitment", metavar="HEX", help="refuse unless the pack opens this suite commitment (needs --salt-env)")
    p.add_argument("--salt-env", metavar="VAR", help="environment variable that holds the commitment salt (never on the command line)")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", metavar="NAME", help="a remote model, sent through the harness")
    who.add_argument("--baseline", choices=BASELINE_IDS, metavar="ID", help=f"a registered baseline: {', '.join(BASELINE_IDS)}")
    who.add_argument("--agent", metavar="MODULE:FACTORY", help="a local zero-argument AgentFactory (needs --model-id)")
    p.add_argument("--agent-path", action="append", default=[], metavar="DIR", help="prepend DIR to the import path for --agent (repeatable)")
    p.add_argument("--model-id", metavar="ID", help="result label (default: derived from the model, protocol and provider)")
    p.add_argument("--out", required=True, metavar="DIR", help="run directory, outside the source repository; a rerun resumes")
    p.add_argument("--private", action="store_true", help="hidden suite: needs a pinned pack; no scenario ID, seed or answer text in any output")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, metavar="N", help=f"bootstrap resamples (default {DEFAULT_RESAMPLES})")
    remote = p.add_argument_group("remote model (--model)")
    remote.add_argument("--api", choices=("decision", "openai", "anthropic"), default="decision", help="API style (default decision)")
    remote.add_argument("--base-url", default=DEFAULT_DECISIONS_URL, metavar="URL", help="API base URL (default: the OpenRouter decisions API)")
    remote.add_argument("--key-env", default="EMBERBENCH_API_KEY", metavar="VAR", help="environment variable that holds the API key")
    remote.add_argument("--provider", metavar="LABEL", help="provider label in the model ID (default: from the base URL host)")
    remote.add_argument("--joint-mode", choices=("full_joint", "fixed_chunk"), default="full_joint", help="harness joint protocol (canonical: full_joint)")
    remote.add_argument("--chunk-size", type=int, metavar="N", help="people per request with --joint-mode fixed_chunk (a declared variant)")
    remote.add_argument("--max-reasks", type=int, default=2, metavar="N", help="re-asks after an invalid answer (harness 1.1: 2)")
    remote.add_argument("--temperature", type=float, metavar="T", help="chat temperature (default 0)")
    remote.add_argument("--timeout", type=float, default=120.0, metavar="S", help="seconds per HTTP request")
    remote.add_argument("--budget-usd", type=float, metavar="USD", help="stop (resumable) once the reported cost of this invocation reaches this")
    remote.add_argument("--input-price", type=float, metavar="USD", help="USD per million input tokens, when the API reports no cost")
    remote.add_argument("--output-price", type=float, metavar="USD", help="USD per million output tokens, when the API reports no cost")


# --------------------------------------------------------------------------- checks


def _check_out(text: str) -> Path:
    out = Path(text).resolve()
    if (_SOURCE_ROOT / "pyproject.toml").is_file() and (out == _SOURCE_ROOT or _SOURCE_ROOT in out.parents):
        raise RunError("output", f"{text}: the run directory must be outside the source repository ({_SOURCE_ROOT})")
    if out.exists() and not out.is_dir():
        raise RunError("output", f"{text}: not a directory")
    if not out.parent.is_dir():
        raise RunError("output", f"{text}: parent directory does not exist (it is never created)")
    return out


def _load_pack(args: argparse.Namespace) -> tuple[EnvironmentPack, dict[str, Any] | None]:
    if args.private and not (args.expected_digest or args.commitment):
        raise RunError("usage", "--private needs a pinned pack: --expected-digest or --commitment")
    if bool(args.commitment) != bool(args.salt_env):
        raise RunError("usage", "--commitment and --salt-env go together")
    try:
        pack = load_environment_pack(args.suite, expected_digest=args.expected_digest)
    except EmberBenchError as exc:
        raise RunError("mismatch", str(exc) if not args.private else "the environment pack failed verification") from None
    commitment = None
    if args.commitment:
        salt = os.environ.get(args.salt_env, "")
        try:
            opened = suite_commitment(pack.digest, salt) == args.commitment
        except ValueError as exc:
            raise RunError("usage", f"--salt-env {args.salt_env}: {exc}") from None
        if not opened:
            raise RunError("mismatch", "the environment pack does not open the given suite commitment")
        commitment = {"commitment": args.commitment, "verified": True}
    return pack, commitment


# --------------------------------------------------------------------------- execute


def execute(args: argparse.Namespace, *, load_agent: Callable[[argparse.Namespace], AgentFactory] | None = None) -> dict[str, Any]:
    """Run ``emberbench run``. Returns the summary printed by the CLI; raises :class:`RunError`."""
    out = _check_out(args.out)
    pack, commitment = _load_pack(args)
    try:
        check_resamples(args.resamples)
    except ValueError as exc:
        raise RunError("usage", f"--resamples: {exc}") from None
    versions = benchmark_versions()
    started_utc = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    model: dict[str, Any]
    usage: dict[str, Any] | None = None
    if args.model:
        evaluation, model, usage = _run_remote(args, pack, out)
    else:
        if args.baseline:
            model_id = check_model_id(args.model_id or args.baseline)
            factory = baseline_factory(args.baseline)
            model = {"kind": "baseline", "model_id": model_id, "baseline_id": args.baseline}
        else:
            if not args.model_id:
                raise RunError("usage", "--agent needs --model-id")
            if load_agent is None:
                raise RunError("usage", "--agent is supported by the emberbench CLI only")
            model_id = check_model_id(args.model_id)
            factory = load_agent(args)
            model = {"kind": "agent", "model_id": model_id, "agent": args.agent}
        evaluation = evaluate_pack(pack, out, model_id, lambda _ledger: factory, resamples=args.resamples)
    result_digest = file_sha256(evaluation.result_path)
    seeds = [e.seed for e in pack.manifest.episodes]
    seed_text = ",".join(str(s) for s in seeds)
    record = {
        "run_record_version": RUN_RECORD_VERSION,
        "benchmark_versions": versions,
        "started_utc": started_utc,
        "finished_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "private": bool(args.private),
        "suite": {
            "pack_id": None if args.private else pack.pack_id,
            "pack_digest": pack.digest,
            "manifest_hash": pack.manifest.manifest_hash(),
            "episodes": len(pack.manifest.episodes),
            "seeds": None if args.private else seeds,
            "seeds_sha256": hashlib.sha256(seed_text.encode("ascii")).hexdigest(),
            "commitment": commitment,
        },
        "model": model,
        "usage": usage,
        "runtime_s": round(evaluation.elapsed_s, 3),
        "accounting": {k: v for k, v in evaluation.accounting.items() if k != "per_episode"},
        "result_sha256": result_digest,
        "refusals": refusal_summary(evaluation),
        "metrics": score_summary(evaluation.report),
    }
    if not args.private:
        record["scenarios"] = {g.name: _f(g.score, 3) for g in evaluation.report.scenarios}
    model_id = model["model_id"]
    _write_json(out / "runs" / f"{model_id}.json", record)
    metrics = score_summary(evaluation.report)
    return {
        "model_id": model_id,
        "pack_digest": pack.digest,
        "episodes": len(evaluation.result.episodes),
        "score": metrics["score"],
        "ci95": metrics["ci95"],
        "safety": metrics["safety"],
        "quality": metrics["quality"],
        "cost_usd": evaluation.accounting.get("total_cost"),
        "horizon_hits": evaluation.accounting.get("horizon_hits"),
        "refusals": len(evaluation.refused),
        "run_record": str(out / "runs" / f"{model_id}.json"),
    }


def _run_remote(args: argparse.Namespace, pack: EnvironmentPack, out: Path) -> tuple[Evaluation, dict[str, Any], dict[str, Any]]:
    from . import harness  # the network harness, loaded only for a remote model

    try:
        mode = harness.JointMode(args.joint_mode)
        protocol = harness.HarnessProtocol(mode, args.chunk_size if mode is harness.JointMode.FIXED_CHUNK else None, args.max_reasks)
        url = harness.endpoint_url(args.base_url, args.api)
    except ValueError as exc:
        raise RunError("usage", str(exc)) from None
    if args.chunk_size is not None and mode is harness.JointMode.FULL_JOINT:
        raise RunError("usage", "--chunk-size needs --joint-mode fixed_chunk")
    key = os.environ.get(args.key_env, "")
    host = url.split("/")[2].split(":")[0]
    if not key and host not in ("localhost", "127.0.0.1", "[::1]"):
        raise RunError("usage", f"set the API key in the environment variable {args.key_env}")
    client = harness.RemoteClient(
        api=args.api,
        base_url=args.base_url,
        model=args.model,
        key=key,
        cache_path=out / "cache.jsonl",
        temperature=0.0 if args.temperature is None else args.temperature,
        format_retries=args.max_reasks,
        timeout=args.timeout,
        budget_usd=args.budget_usd,
        redact=args.private,
    )
    provider = harness.label(args.provider or harness.provider_label(args.base_url))
    model_id = check_model_id(args.model_id or harness.remote_model_id(provider, args.model, args.api, protocol))
    segment = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    prices = (args.input_price or 0.0, args.output_price or 0.0) if args.input_price is not None or args.output_price is not None else None

    def make_factory(ledger: Any) -> AgentFactory:
        return lambda: harness.RemoteAgent(client, provider=provider, protocol=protocol, ledger=ledger)

    try:
        evaluation = evaluate_pack(
            pack, out, model_id, make_factory,
            ledger_factory=lambda path, pending: harness.Ledger(path, segment, pending),
            resamples=args.resamples, prices=prices,
        )  # fmt: skip
    except harness.AdapterError as exc:
        raise RunError("agent", f"{model_id}: {exc} (the run can be resumed)") from None
    model = {
        "kind": "remote",
        "model_id": model_id,
        "requested_model": args.model,
        "resolved_models": sorted(client.resolved_models),
        "api": args.api,
        "endpoint": url,
        "provider": provider,
        "settings": {
            "temperature": 0.0 if args.temperature is None else args.temperature,
            "timeout_s": args.timeout,
            "budget_usd": args.budget_usd,
            "input_price": args.input_price,
            "output_price": args.output_price,
        },
        "harness_protocol": protocol.to_dict(),
        "harness_sha256": file_sha256(Path(harness.__file__)),
    }
    usage = {
        "requests_sent": client.calls,
        "cache_hits": client.cache_hits,
        "transport_retries": client.retries,
        "reasks": client.format_errors,
        "input_tokens": client.input_tokens,
        "output_tokens": client.output_tokens,
        "reported_cost_usd": round(client.cost, 6),
        "act_calls": len(evaluation.steps),
    }
    return evaluation, model, usage


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m emberbench.run``: the same command without the CLI wrapper."""
    parser = argparse.ArgumentParser(prog="emberbench run", description=(__doc__ or "").splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        summary = execute(args)
    except RunError as exc:
        parser.exit(2 if exc.kind == "usage" else 1, f"emberbench run: {exc.message}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
