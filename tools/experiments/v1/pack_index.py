"""The canonical, machine-readable experiment index of the V1 evaluation pack.

The index pins every input and output of one pack run: the benchmark revision
and component versions, the suite and evaluation manifest hashes, each model's
configuration digest and each result, statistics and bound digest. It holds no
absolute path, no timestamp, no host or interpreter detail. File references
are relative to the output directory.

Every artifact digest is the SHA-256 of the artifact file without its final
LF. For results and statistics this equals the library digest
(``AggregateStatistics.digest()``, ``PairedStatistics.digest()``, and the
SHA-256 of the canonical ``EvaluationResult`` that the statistics record as
``result_digest``). The index digest is the SHA-256 of the canonical JSON of
the index.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version as package_version
from typing import Any

from emberbench import BASELINES_VERSION, EvaluationManifest, EvaluationResult, Suite, benchmark_versions
from emberbench.cli import CLI_VERSION
from emberbench.oracle import EscapeBound

import pack_agents
import pack_protocol

INDEX_VERSION = "1"


def canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, ASCII, no NaN (the EMBERBench encoding)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def text_digest(text: str) -> str:
    """SHA-256 of an artifact's file text without its final LF."""
    body = text[:-1] if text.endswith("\n") else text
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def index_digest(index: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(index).encode("ascii")).hexdigest()


def model_slug(model_id: str) -> str:
    """A file-name-safe form of a model ID. Distinct IDs of this pack give distinct slugs."""
    slug = "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in model_id)
    if not slug or slug.startswith("."):
        raise ValueError(f"model ID {model_id!r} has no safe file name")
    return slug


def paired_file(a: str, b: str) -> str:
    """Relative file of the paired statistics ``a - b``, named by the short labels."""
    return f"statistics/paired/{model_slug(pack_agents.SHORT_NAMES.get(a, a))}--{model_slug(pack_agents.SHORT_NAMES.get(b, b))}.json"


def bound_document(manifest: EvaluationManifest, bounds: Sequence[EscapeBound]) -> dict[str, Any]:
    """The privileged escape bound per manifest episode. Reference context only; never a model row."""
    episodes = []
    for spec, bound in zip(manifest.episodes, bounds, strict=True):
        if bound.scenario_id != spec.scenario_id or bound.seed != spec.seed:
            raise ValueError("bounds are not in manifest order")
        entry = bound.to_dict()
        entry["index"] = spec.index
        episodes.append(entry)
    return {
        "artifact": "privileged-escape-bound",
        "privileged": True,
        "fair": False,
        "note": (
            "Privileged relaxed upper bound from emberbench.oracle: reads the scenario and the evaluator seed "
            "(the whole future fire) and ignores other people. It is not a model, not a baseline and not a "
            "leaderboard row."
        ),
        "evaluation_manifest_hash": manifest.manifest_hash(),
        "episodes": episodes,
    }


def build_index(
    protocol: pack_protocol.Protocol,
    suite: Suite,
    manifest: EvaluationManifest,
    specs: Sequence[pack_agents.PackAgentSpec],
    results: Mapping[str, EvaluationResult],
    aggregates: Mapping[str, str],
    paired: Mapping[tuple[str, str], str],
    bound_text: str,
) -> dict[str, Any]:
    models = []
    for spec in specs:
        result = results[spec.model_id]
        slug = model_slug(spec.model_id)
        models.append(
            {
                "model_id": spec.model_id,
                "kind": spec.kind,
                "fair": True,
                "observation_only": True,
                "description": spec.description,
                "descriptor": dict(spec.descriptor),
                "config_digest": spec.config_digest,
                "result": {
                    "file": f"results/{slug}.json",
                    "digest": text_digest(result.to_file_text()),
                    "evaluation_manifest_hash": result.manifest_hash,
                    "episodes": len(result.episodes),
                },
                "aggregate_statistics": {
                    "file": f"statistics/aggregate/{slug}.json",
                    "digest": text_digest(aggregates[spec.model_id]),
                },
            }
        )
    return {
        "index_version": INDEX_VERSION,
        "pack_id": pack_agents.PACK_ID,
        "pack_version": pack_protocol.PACK_VERSION,
        "pack_agents_version": pack_agents.PACK_AGENTS_VERSION,
        "label": pack_protocol.RUN_LABEL,
        "protocol": {
            "name": protocol.name,
            "scenario_ids": None if protocol.scenario_ids is None else list(protocol.scenario_ids),
            "seeds": list(protocol.seeds),
            "resamples": protocol.resamples,
        },
        "benchmark": {
            "package": "emberbench",
            "package_version": package_version("emberbench"),
            "versions": benchmark_versions(),
            "baselines_version": BASELINES_VERSION,
            "cli_version": CLI_VERSION,
        },
        "suite": {
            "suite_id": suite.manifest.suite_id,
            "suite_version": suite.manifest.suite_version,
            "suite_hash": suite.manifest.suite_hash(),
            "visibility": suite.manifest.visibility.value,
            "seed_role": suite.manifest.seed_role.value,
            "canonical_public_suite_hash": pack_protocol.PUBLIC.suite_hash,
            "scenarios": [{"scenario_id": e.scenario_id, "content_hash": e.content_hash} for e in suite.manifest.entries],
        },
        "evaluation_manifest": {
            "file": "manifest/public-dev-manifest.json",
            "manifest_hash": manifest.manifest_hash(),
            "file_digest": text_digest(manifest.to_file_text()),
            "episode_count": len(manifest.episodes),
        },
        "models": models,
        "paired_statistics": [
            {
                "model_a": a,
                "model_b": b,
                "delta": "A - B",
                "file": paired_file(a, b),
                "digest": text_digest(text),
            }
            for (a, b), text in paired.items()
        ],
        "privileged_bound": {
            "file": "privileged/escape-bound.json",
            "digest": text_digest(bound_text),
            "privileged": True,
            "fair": False,
            "competing_model": False,
        },
    }
