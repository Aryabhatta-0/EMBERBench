"""The canonical, machine-readable reproducibility index of the V1 evaluation pack.

The index pins every input and output of one pack run: the benchmark revision
and component versions, the scoring version and constants digest, the
environment-world and environment-pack versions, the suite, unique-world
manifest and environment-pack digests, every unique-world digest, each model's
configuration digest and each result, score-report, paired-comparison,
diagnostics and privileged-reference digest. It holds no absolute path, no
timestamp, no host, interpreter or temporary-directory detail. File references
are relative to the output directory.

Every artifact digest is the SHA-256 of the artifact file without its final
LF. For results, score reports and statistics this equals the library digest
(the canonical ``EvaluationResult``, ``ScoreReport.digest()``,
``AggregateStatistics.digest()``). The index digest is the SHA-256 of the
canonical JSON of the index.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version as package_version
from typing import Any

from emberbench import (
    BASELINES_VERSION,
    ENVIRONMENT_PACK_VERSION,
    ENVIRONMENT_WORLD_VERSION,
    OFFICIAL_CONSTANTS,
    SCORING_VERSION,
    EnvironmentPack,
    EvaluationManifest,
    EvaluationResult,
    ScoreComparison,
    ScoreReport,
    Suite,
    benchmark_versions,
    leaderboard_order,
)
from emberbench.cli import CLI_VERSION
from emberbench.environment_pack import pack_versions

import pack_agents
import pack_protocol

INDEX_VERSION = "2"


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


def _short_slug(model_id: str) -> str:
    return model_slug(pack_agents.SHORT_NAMES.get(model_id, model_id))


def score_file(model_id: str) -> str:
    return f"scores/{model_slug(model_id)}.json"


def diagnostics_file(model_id: str) -> str:
    return f"diagnostics/{model_slug(model_id)}.json"


def comparison_file(a: str, b: str) -> str:
    """Relative file of the paired score comparison ``a - b``, named by the short labels."""
    return f"comparisons/{_short_slug(a)}--{_short_slug(b)}.json"


def comparison_document(a: ScoreReport, b: ScoreReport, comparison: ScoreComparison) -> dict[str, Any]:
    """The official comparison with the digests of the two score reports it pairs."""
    if (comparison.model_a, comparison.model_b) != (a.model_id, b.model_id):
        raise ValueError("comparison does not belong to these reports")
    return {
        "artifact": "paired-score-comparison",
        "delta": "A - B",
        "scoring_version": a.scoring_version,
        "constants_digest": a.constants.digest(),
        "evaluation_manifest_hash": a.evaluation_manifest_hash,
        "environment_pack_digest": a.pack_digest,
        "score_report_digest_a": a.digest(),
        "score_report_digest_b": b.digest(),
        "paired_episode_indices": [i for i, _ in a.episode_scores],
        "comparison": comparison.to_dict(),
    }


def build_index(
    protocol: pack_protocol.Protocol,
    suite: Suite,
    manifest: EvaluationManifest,
    pack: EnvironmentPack,
    worlds: Sequence[Mapping[str, Any]],
    nominal: Mapping[str, int],
    specs: Sequence[pack_agents.PackAgentSpec],
    results: Mapping[str, EvaluationResult],
    reports: Mapping[str, ScoreReport],
    comparisons: Mapping[tuple[str, str], str],
    diagnostics: Mapping[str, str],
    references_text: str,
) -> dict[str, Any]:
    models = []
    for spec in specs:
        result, report = results[spec.model_id], reports[spec.model_id]
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
                "score_report": {
                    "file": score_file(spec.model_id),
                    "digest": report.digest(),
                    "official_procedure": report.official,
                    "result_digests": list(report.result_digests),
                },
                "diagnostics": {
                    "file": diagnostics_file(spec.model_id),
                    "digest": text_digest(diagnostics[spec.model_id]),
                    "score_component": False,
                },
            }
        )
    stochastic = sum(1 for w in worlds if w["fire_model"] == "stochastic")
    return {
        "index_version": INDEX_VERSION,
        "pack_id": pack_agents.PACK_ID,
        "pack_version": pack_protocol.PACK_VERSION,
        "pack_agents_version": pack_agents.PACK_AGENTS_VERSION,
        "label": pack_protocol.RUN_LABEL,
        "protocol": {
            "name": protocol.name,
            "scenario_ids": None if protocol.scenario_ids is None else list(protocol.scenario_ids),
            "candidate_seeds": list(protocol.candidate_seeds),
            "stochastic_worlds": protocol.stochastic_worlds,
            "resamples": protocol.resamples,
        },
        "benchmark": {
            "package": "emberbench",
            "package_version": package_version("emberbench"),
            "versions": benchmark_versions(),
            "scoring_version": SCORING_VERSION,
            "scoring_constants_digest": OFFICIAL_CONSTANTS.digest(),
            "environment_world_version": ENVIRONMENT_WORLD_VERSION,
            "environment_pack_version": ENVIRONMENT_PACK_VERSION,
            "environment_pack_versions": pack_versions(),
            "baselines_version": BASELINES_VERSION,
            "cli_version": CLI_VERSION,
        },
        "suite": {
            "suite_id": suite.manifest.suite_id,
            "suite_version": suite.manifest.suite_version,
            "suite_hash": suite.manifest.suite_hash(),
            "visibility": suite.manifest.visibility.value,
            "seed_role": suite.manifest.seed_role.value,
            "canonical_public_suite_hash": pack_protocol.PUBLIC_SUITE_HASH,
            "scenarios": [
                {"scenario_id": e.scenario_id, "content_hash": e.content_hash, "primary_family": e.primary_family.value}
                for e in suite.manifest.entries
            ],
        },
        "schedule": {
            "builder": "build_unique_world_manifest",
            "nominal_candidate_episodes": nominal["nominal_candidate_episodes"],
            "distinct_worlds_among_candidates": nominal["distinct_worlds_among_candidates"],
            "unique_worlds": len(worlds),
            "deterministic_worlds": len(worlds) - stochastic,
            "stochastic_worlds": stochastic,
            "evaluation_manifest": {
                "file": "environment-pack/evaluation-manifest.json",
                "manifest_hash": manifest.manifest_hash(),
                "file_digest": text_digest(manifest.to_file_text()),
                "episode_count": len(manifest.episodes),
            },
            "worlds": [dict(w) for w in worlds],
        },
        "environment_pack": {
            "file": "environment-pack/pack.json",
            "pack_id": pack.pack_id,
            "digest": pack.digest,
            "fire_world_digests": [e.fire_world_digest for e in pack.episodes],
        },
        "models": models,
        "leaderboard": [r.model_id for r in leaderboard_order(reports[s.model_id] for s in specs)],
        "paired_comparisons": [
            {
                "model_a": a,
                "model_b": b,
                "delta": "A - B",
                "file": comparison_file(a, b),
                "digest": text_digest(text),
                "outcome": json.loads(text)["comparison"]["outcome"],
            }
            for (a, b), text in comparisons.items()
        ],
        "privileged_references": {
            "file": "privileged/references.json",
            "digest": text_digest(references_text),
            "privileged": True,
            "fair": False,
            "competing_model": False,
        },
    }
