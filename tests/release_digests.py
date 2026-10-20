"""Recompute every entry of the regression digest registry.

The registry (``tests/fixtures/release/digest_registry.json``) lists one
frozen digest per release-critical contract. Each value here is computed the
same way as the test that pins it, from public development assets and
test-only seeds. ``tests/test_release.py`` compares the registry with those
test pins and with this module run in fresh interpreters.

Run directly, it prints the computed digests as canonical JSON on stdout.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "fixtures" / "release" / "digest_registry.json"

# Public suite under development seeds (0, 1, 2): the manifest that the CLI example builds.
DEV_MANIFEST_SEEDS = (0, 1, 2)
REGISTRY_SCENARIOS = ("v1-pub-01-single-exit-sparse", "v1-pub-09-crowd-100-split-flow", "v1-pub-10-route-invalidation")
REGISTRY_GOLDEN_RECORDS = ("one_person_one_exit", "stochastic_room")
REGISTRY_BASELINE = "fire-aware-shortest-path-v1"


def load_registry() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["entries"]
    return entries


def _suite_and_scenarios() -> dict[str, str]:
    from emberbench import load_suite
    from test_suite import SUITE_DIR

    suite = load_suite(SUITE_DIR)
    out = {"suite.public-v1.suite_hash": suite.manifest.suite_hash()}
    for scenario_id in REGISTRY_SCENARIOS:
        out[f"scenario.{scenario_id}.content_hash"] = suite.get(scenario_id).content_hash()
    return out


def _golden_records() -> dict[str, str]:
    from conftest import fixture_scenario, greedy_exit_policy
    from emberbench import run_episode

    return {f"simulator.golden-record.{name}": run_episode(fixture_scenario(name), greedy_exit_policy).digest() for name in REGISTRY_GOLDEN_RECORDS}


def _stochastic_traces() -> dict[str, str]:
    from emberbench.canonical import sha256_hex
    from test_stochastic_hazard import GOLDEN, golden_trace

    traces = [golden_trace(g["source"], g["seed"], g["steps"]) for g in GOLDEN]
    return {"hazard.stochastic-golden-traces": sha256_hex(traces)}


def _evaluation_manifest() -> dict[str, str]:
    from emberbench import build_evaluation_manifest, load_suite
    from test_suite import SUITE_DIR

    return {"evaluation.public-v1-dev-seeds.manifest_hash": build_evaluation_manifest(load_suite(SUITE_DIR), DEV_MANIFEST_SEEDS).manifest_hash()}


def _baseline() -> dict[str, str]:
    from emberbench import agent_policy_factory, baseline_factory, build_evaluation_manifest, get_baseline, load_suite, run_evaluation
    from test_baselines import DEV_IDS, DEV_SEEDS, result_digest
    from test_evaluation import subset
    from test_suite import SUITE_DIR

    suite = subset(load_suite(SUITE_DIR), DEV_IDS)
    manifest = build_evaluation_manifest(suite, DEV_SEEDS)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_evaluation(suite, manifest, REGISTRY_BASELINE, agent_policy_factory(baseline_factory(REGISTRY_BASELINE)), Path(tmp) / "out.json")
    return {
        f"baseline.{REGISTRY_BASELINE}.config_digest": get_baseline(REGISTRY_BASELINE).config_digest(),
        f"baseline.{REGISTRY_BASELINE}.result_digest": result_digest(result),
    }


def _generator() -> dict[str, str]:
    from emberbench.generator import GeneratorConfig, generate_suite
    from test_generator import BASE_CONFIG, BATCH, SEED

    config = GeneratorConfig.from_dict(BASE_CONFIG)
    generated = generate_suite(config, SEED, batch_id=BATCH)
    return {
        "generator.test-config.config_hash": config.config_hash(),
        "generator.test-config.generation_hash": generated.generation.generation_hash(),
        "generator.test-config.suite_hash": generated.suite_manifest.suite_hash(),
    }


def _statistics() -> dict[str, str]:
    from test_statistics import golden_digests

    return {f"statistics.{name}": digest for name, digest in golden_digests()}


def _provenance() -> dict[str, str]:
    from conftest import greedy_exit_policy
    from emberbench import build_evaluation_manifest, build_evaluation_provenance, run_evaluation
    from emberbench.generator import GeneratorConfig, freeze_generated_suite, generate_suite
    from test_generator import BASE_CONFIG, BATCH, SEED
    from test_integrity import EVAL_SEEDS, MODEL, factory_of, identity

    with tempfile.TemporaryDirectory(prefix="emberbench provenance ") as tmp:
        root = Path(tmp)
        frozen = freeze_generated_suite(generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH), root / "frozen")
        manifest = build_evaluation_manifest(frozen.suite, EVAL_SEEDS)
        result = run_evaluation(frozen.suite, manifest, MODEL, factory_of(greedy_exit_policy), root / "greedy.json")
        provenance = build_evaluation_provenance(frozen, manifest, result, identity())
    return {"integrity.test-run.provenance_digest": provenance.provenance_digest()}


def _cli_artifact() -> dict[str, str]:
    from emberbench.cli import main
    from emberbench.integrity import digest_file
    from test_suite import SUITE_DIR

    with tempfile.TemporaryDirectory(prefix="emberbench cli ") as tmp:
        out = Path(tmp) / "manifest.json"
        seeds = [arg for seed in DEV_MANIFEST_SEEDS for arg in ("--seed", str(seed))]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["evaluation", "manifest", "--suite", str(SUITE_DIR), *seeds, "--out", str(out), "--json"])
        if code != 0:
            raise RuntimeError(f"emberbench evaluation manifest exited with {code}")
        return {"cli.evaluation-manifest.file_sha256": digest_file(out)}


COMPUTERS: tuple[Callable[[], dict[str, str]], ...] = (
    _suite_and_scenarios,
    _golden_records,
    _stochastic_traces,
    _evaluation_manifest,
    _baseline,
    _generator,
    _statistics,
    _provenance,
    _cli_artifact,
)


def compute_registry_digests() -> dict[str, str]:
    """Every registry digest, computed now. Keys are the registry entry names."""
    out: dict[str, str] = {}
    for compute in COMPUTERS:
        out.update(compute())
    return dict(sorted(out.items()))


if __name__ == "__main__":
    sys.stdout.write(json.dumps(compute_registry_digests(), sort_keys=True, separators=(",", ":")) + "\n")
