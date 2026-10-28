"""Cost, call and horizon accounting of a harness run (docs/harness-protocol.md, section 6).

The harness writes one ledger row per accepted ``act`` call
(:class:`emberbench.harness.Ledger`). This module joins those rows with the
authenticated :class:`~emberbench.EvaluationResult` and reports, per episode
and over the run: calls, re-asks, invalid slots, tokens, cost, response time,
steps and horizon hits, each with its median and 95th percentile. Accounting
never changes a score.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluation import EvaluationManifest, EvaluationResult, load_evaluation_result
from .state import TerminationReason
from .suite import Suite

ACCOUNTING_VERSION = "1.1"  # 1.1: the refusals block


def pending_episodes(path: Path, suite: Suite, manifest: EvaluationManifest, skip: Collection[int] = ()) -> list[int]:
    """Manifest episode indices that a ``run_evaluation`` call with ``resume=True`` and ``skip`` will run, in order."""
    done: set[int] = set(skip)
    if path.exists():
        done |= {e.episode.index for e in load_evaluation_result(path, suite).episodes}
    return [e.index for e in manifest.episodes if e.index not in done]


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_rows(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    """Ledger rows per episode: only the rows of the last segment that ran the episode (a resume re-runs it)."""
    by_episode: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    last: dict[int, str] = {}
    for row in rows:
        episode = int(row["episode"])
        by_episode.setdefault(episode, {}).setdefault(str(row["segment"]), []).append(row)
        last[episode] = str(row["segment"])
    return {e: segs[last[e]] for e, segs in sorted(by_episode.items())}


def quantile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile with linear interpolation between order statistics; ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _dist(values: Sequence[float], digits: int = 4) -> dict[str, Any]:
    def r(v: float | None) -> float | None:
        return None if v is None else round(v, digits)

    return {
        "n": len(values),
        "total": r(sum(values)) if values else 0,
        "mean": r(statistics.fmean(values)) if values else None,
        "median": r(quantile(values, 0.5)),
        "p95": r(quantile(values, 0.95)),
        "max": r(max(values)) if values else None,
    }


def _request_cost(req: Mapping[str, Any], prices: tuple[float, float] | None) -> float | None:
    if prices is not None:
        return float(req["input_tokens"] * prices[0] + req["output_tokens"] * prices[1]) / 1e6
    cost = req.get("cost")
    return None if cost is None else float(cost)


def accounting_summary(
    rows: Iterable[Mapping[str, Any]],
    result: EvaluationResult,
    *,
    prices: tuple[float, float] | None = None,
    refused: Collection[int] = (),
) -> dict[str, Any]:
    """Per-episode and run-level accounting of ``result`` from its ledger ``rows``.

    Every distribution and rate is over the completed episodes. ``refused``
    (episode indices recorded as refused) is reported apart in ``refusals``:
    their count, their rate over the scheduled episodes and the cost of their
    requests before the refusal (not part of ``total_cost``).

    ``prices`` (USD per million input and output tokens) replaces the reported
    cost when given (list-price accounting). Without it, an episode's cost is
    the sum of reported costs, or ``None`` when any request reported none.
    """
    per_episode = episode_rows(rows)
    episodes: list[dict[str, Any]] = []
    first_slots = first_invalid = first_requests = reask_requests = 0
    first_cost = total_cost = 0.0
    cost_known = True
    probs: list[Mapping[str, Any]] = []
    for ep in result.episodes:
        index = ep.episode.index
        acts = per_episode.get(index, [])
        requests = [req for row in acts for req in row["requests"]]
        costs = [_request_cost(req, prices) for req in requests]
        known = all(c is not None for c in costs)
        cost_known &= known
        latencies = [req["latency_s"] for req in requests if req.get("latency_s") is not None]
        for req, cost in zip(requests, costs, strict=True):
            if req["attempt"] == 0:
                first_requests += 1
                first_slots += req["questions"]
                first_invalid += req["invalid_slots"]
                first_cost += cost or 0.0
            else:
                reask_requests += 1
            total_cost += cost or 0.0
        for row in acts:
            probs.extend(row.get("probabilities") or [])
        m = ep.metrics
        episodes.append(
            {
                "index": index,
                "acts": len(acts),
                "steps": m.time.episode_steps,
                "max_steps": m.time.max_steps,
                "hit_horizon": m.termination_reason is TerminationReason.MAX_STEPS,
                "stranded": m.safety.stranded,
                "calls": len(requests),
                "reasks": sum(1 for r in requests if r["attempt"] > 0),
                "slots": sum(r["questions"] for r in requests if r["attempt"] == 0),
                "invalid_slots": sum(r["invalid_slots"] for r in requests),
                "input_tokens": sum(r["input_tokens"] for r in requests),
                "output_tokens": sum(r["output_tokens"] for r in requests),
                "cost": round(sum(c for c in costs if c is not None), 8) if known else None,
                "response_s": round(sum(latencies), 4),
                "latency_complete": len(latencies) == len(requests),
                "accounted": index in per_episode,
            }
        )
    accounted = [e for e in episodes if e["accounted"]]
    valid_probs = [p for p in probs if p.get("valid")]
    horizon = [e for e in episodes if e["hit_horizon"]]
    refused_costs = [_request_cost(req, prices) for i in sorted(refused) for row in per_episode.get(i, []) for req in row["requests"]]
    scheduled = len(result.manifest.episodes)
    return {
        "accounting_version": ACCOUNTING_VERSION,
        "cost_basis": "list price" if prices is not None else "reported",
        "episodes": len(episodes),
        "episodes_accounted": len(accounted),
        "calls_per_episode": _dist([e["calls"] for e in accounted], 2),
        "steps_per_episode": _dist([e["steps"] for e in episodes], 2),
        "input_tokens_per_episode": _dist([e["input_tokens"] for e in accounted], 1),
        "output_tokens_per_episode": _dist([e["output_tokens"] for e in accounted], 1),
        "cost_per_episode": _dist([e["cost"] for e in accounted], 6) if cost_known else None,
        "response_s_per_episode": _dist([e["response_s"] for e in accounted], 3),
        "response_time_complete": all(e["latency_complete"] for e in accounted),
        "total_cost": round(total_cost, 6) if cost_known else None,
        "first_attempt_cost": round(first_cost, 6) if cost_known else None,
        "reask_cost_multiplier": round(total_cost / first_cost, 6) if cost_known and first_cost > 0 else None,
        "requests": first_requests + reask_requests,
        "reask_requests": reask_requests,
        "reask_rate": round(reask_requests / first_requests, 6) if first_requests else None,
        "slots": first_slots,
        "invalid_slots_first_attempt": first_invalid,
        "invalid_slot_rate": round(first_invalid / first_slots, 6) if first_slots else None,
        "horizon_hits": len(horizon),
        "horizon_hit_rate": round(len(horizon) / len(episodes), 6) if episodes else None,
        "horizon_hits_with_stranded": sum(1 for e in horizon if e["stranded"] > 0),
        "refusals": {
            "episodes": len(refused),
            "rate": round(len(refused) / scheduled, 6) if scheduled else None,
            "cost": round(sum(c for c in refused_costs if c is not None), 6) if all(c is not None for c in refused_costs) else None,
        },
        "probabilities": None
        if not probs
        else {
            "slots": len(probs),
            "invalid_distributions": len(probs) - len(valid_probs),
            "p_choice_mean": _mean(valid_probs, "p_choice"),
            "p_choice_median": _round(quantile([p["p_choice"] for p in valid_probs], 0.5)),
            "entropy_bits_mean": _mean(valid_probs, "entropy_bits"),
            "margin_mean": _mean(valid_probs, "margin"),
            "choice_not_argmax": sum(1 for p in valid_probs if not p["argmax_agrees"]),
            "note": "descriptive option-probability diagnostics; not calibration (no target labels)",
        },
        "per_episode": episodes,
    }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 6)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return round(statistics.fmean(r[key] for r in rows), 6) if rows else None


def load_accounting(ledger: Path, result_path: Path, suite: Suite, *, prices: tuple[float, float] | None = None) -> dict[str, Any]:
    return accounting_summary(read_ledger(ledger), load_evaluation_result(result_path, suite), prices=prices)
