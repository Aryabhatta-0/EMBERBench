"""The regression digest registry: its specification, its computation and its file.

The registry (``tests/fixtures/release/digest_registry.json``) lists one
frozen digest per release-visible contract. :data:`SPEC` is the source of
truth for each entry: the owner version, the test pin it mirrors and what it
covers. Every value is computed only from public development assets
(``scenarios/public/v1``), test fixtures and test-only seeds. No entry holds
a path, a timestamp, a host value, a private seed or held-out material.

``tests/test_release.py`` checks the registry in-process against the test
pins it mirrors and recomputes it in fresh interpreters under different hash
seeds, locales, time zones, working directories and line endings.
``tools/check_release.py`` recomputes it once more against the installed
wheel.

Usage::

    python tests/release_digests.py --check   # recompute and compare; exit 1 with a diff on drift
    python tests/release_digests.py --write   # rewrite the registry file from SPEC and the computed values
    python tests/release_digests.py           # print the computed digests as JSON

``--write`` is for a deliberate change only: a bumped owner version and a
changelog entry (docs/versioning.md, section 4).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

if __name__ == "__main__":  # run as a script: the test helpers live next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))

REGISTRY_PATH = Path(__file__).resolve().parent / "fixtures" / "release" / "digest_registry.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
# Set to a copy of the public suite (for example with CRLF line endings) to show that the copy gives the same digests.
PUBLIC_SUITE_ENV = "EMBERBENCH_REGISTRY_PUBLIC_SUITE"

# Public suite under development seeds (0, 1, 2): the manifest that the CLI example builds.
DEV_MANIFEST_SEEDS = (0, 1, 2)
REGISTRY_SCENARIOS = ("v1-pub-01-single-exit-sparse", "v1-pub-09-crowd-100-split-flow", "v1-pub-10-route-invalidation")
REGISTRY_GOLDEN_RECORDS = ("one_person_one_exit", "stochastic_room")
REGISTRY_BASELINE = "fire-aware-shortest-path-v1"
# The stochastic public scenario, the fire-aware baseline and seed 1: the replay, world and episode score pins.
STOCHASTIC_ID = "v1-pub-10-route-invalidation"
STOCHASTIC_SEED = 1
REPLAY_CASE = "public:route-invalidation:fire-aware:s1"  # its row in tests/test_replay_audit.py GOLDEN_DIGESTS
# An official environment pack: the public suite under a three-seed schedule (as tests/test_environment_pack.py).
PACK_SEEDS = (4, 2, 6)
PACK_ID = "registry-public-v1"
PACK_RESAMPLES = 200
# The unique-world schedule of the public development evaluation: candidate seeds 0..15, ten stochastic worlds.
UNIQUE_WORLD_SEEDS = tuple(range(16))
UNIQUE_STOCHASTIC_WORLDS = 10

REGISTRY_ONLY = "this registry only"
# An environment pack freezes the exact bytes of its files (docs/environment-pack.md), so these two entries,
# unlike every other one, change with a CRLF copy of the public suite. .gitattributes keeps the files LF everywhere.
BYTE_PINNED = frozenset({f"environment-pack.{PACK_ID}.pack_digest", f"scoring.score-report.{PACK_ID}"})


def _entry(owner: str, mirrors: str, what: str) -> dict[str, str]:
    return {"owner": owner, "mirrors": mirrors, "what": what}


# One entry per registry key: what it pins, who owns it and which test pin it mirrors.
SPEC: dict[str, dict[str, str]] = {
    "baseline.fire-aware-shortest-path-v1.config_digest": _entry(
        "BASELINES_VERSION", "tests/test_baselines.py GOLDEN_CONFIG_DIGESTS", "Baseline descriptor digest."
    ),
    "baseline.fire-aware-shortest-path-v1.result_digest": _entry(
        "BASELINES_VERSION; SIMULATOR_VERSION",
        "tests/test_baselines.py GOLDEN_RESULT_DIGESTS",
        "Record digests of every episode of the baseline development manifest (the baseline's actions).",
    ),
    "benchmark.versions": _entry(
        "BENCHMARK_REVISION; every version in benchmark_versions()",
        REGISTRY_ONLY,
        "sha256_hex(benchmark_versions()): every version that provenance pins, with its value.",
    ),
    "cli.evaluation-manifest.file_sha256": _entry(
        "CLI_VERSION; EVALUATION_MANIFEST_VERSION",
        REGISTRY_ONLY,
        "SHA-256 of the file bytes of 'emberbench evaluation manifest --suite scenarios/public/v1 --seed 0 --seed 1 --seed 2'.",
    ),
    "cli.version-json.stdout_sha256": _entry(
        "package version; CLI_VERSION; every version that 'emberbench version' reports",
        REGISTRY_ONLY,
        "SHA-256 of the stdout bytes of 'emberbench version --json'. The only entry that moves with the package version.",
    ),
    "environment-pack.registry-public-v1.pack_digest": _entry(
        "ENVIRONMENT_PACK_VERSION; every version in pack_versions(); SCHEMA_VERSION",
        REGISTRY_ONLY,
        "Digest of the pack frozen from the public suite and the schedule 4, 2, 6 (scenario bytes, schedule, versions, fire worlds).",
    ),
    "environment-world.v1-pub-10-route-invalidation.seed-1": _entry(
        "ENVIRONMENT_WORLD_VERSION; SIMULATOR_VERSION; OBSERVATION_VERSION",
        REGISTRY_ONLY,
        "environment_world_digest of the stochastic public scenario at seed 1 (unique-world identity).",
    ),
    "evaluation.baseline-dev.result_file_sha256": _entry(
        "EVALUATION_RESULT_VERSION; BASELINES_VERSION; SIMULATOR_VERSION",
        REGISTRY_ONLY,
        "SHA-256 of the evaluation result file of the baseline development manifest.",
    ),
    "evaluation.public-v1-dev-seeds.manifest_hash": _entry(
        "EVALUATION_MANIFEST_VERSION", "tests/test_evaluation.py PUBLIC_DEV_MANIFEST_HASH", "Public suite under development seeds 0, 1, 2."
    ),
    "evaluation.public-v1-unique-worlds.manifest_hash": _entry(
        "EVALUATION_MANIFEST_VERSION; ENVIRONMENT_WORLD_VERSION",
        REGISTRY_ONLY,
        "build_unique_world_manifest(public suite, candidate seeds 0..15, stochastic_worlds=10): no duplicate world.",
    ),
    "generator.test-config.config_hash": _entry("GENERATOR_CONFIG_VERSION", "tests/test_generator.py GOLDEN_CONFIG_HASH", "Test-only generator config."),
    "generator.test-config.generation_hash": _entry(
        "GENERATOR_VERSION; GENERATION_MANIFEST_VERSION",
        "tests/test_generator.py GOLDEN_GENERATION_HASH",
        "Generation record of the test-only config, seed and batch.",
    ),
    "generator.test-config.suite_hash": _entry("GENERATOR_VERSION", "tests/test_generator.py GOLDEN_SUITE_HASH", "Generated suite manifest hash."),
    "hazard.stochastic-golden-traces": _entry(
        "SIMULATOR_VERSION", "tests/fixtures/golden/stochastic_fire_traces.json", "sha256_hex of the recomputed list of stochastic golden fire traces."
    ),
    "integrity.test-run.provenance_digest": _entry(
        "INTEGRITY_PROTOCOL_VERSION; BENCHMARK_REVISION; every version in benchmark_versions()",
        REGISTRY_ONLY,
        "Provenance of the tests/test_integrity.py run fixture (test-only seeds and identity).",
    ),
    "replay.route-invalidation-fire-aware-s1.animation_sha256": _entry(
        "ANIMATION_FORMAT_VERSION; REPLAY_AUDIT_VERSION",
        "tests/test_replay_audit.py GOLDEN_DIGESTS",
        "SHA-256 of the animation JSON (with crowd overlay) of the fire-aware baseline on the stochastic public scenario, seed 1.",
    ),
    "replay.route-invalidation-fire-aware-s1.audit_digest": _entry(
        "REPLAY_AUDIT_VERSION",
        "tests/test_replay_audit.py GOLDEN_DIGESTS",
        "Replay audit digest (with crowd overlay) of the same episode.",
    ),
    "rules.calibration": _entry(
        "CALIBRATION_PANEL_VERSION; DIFFICULTY_RULES_VERSION; ADMISSION_RULES_VERSION",
        REGISTRY_ONLY,
        "sha256_hex(calibration_rules()): panel, difficulty bands and admission thresholds.",
    ),
    "rules.composition.rule_set_digest": _entry(
        "COMPOSITION_VERSION; SELECTION_RULES_VERSION; RECIPE_CATALOG_VERSION; FAMILY_RULES_VERSION; calibration versions",
        "tests/fixtures/generator-coverage-summary.json rule_set_digest",
        "rule_set_digest(): the public V1 composition rule set (versions, family, calibration and selection rules, design, recipes).",
    ),
    "rules.family": _entry("FAMILY_RULES_VERSION", REGISTRY_ONLY, "sha256_hex(family_rules()): the ten V1 family rules and thresholds."),
    "scenario.v1-pub-01-single-exit-sparse.content_hash": _entry("SCHEMA_VERSION", "tests/test_suite.py PINNED_HASHES", "Canonical sparse single-exit scenario."),
    "scenario.v1-pub-09-crowd-100-split-flow.content_hash": _entry("SCHEMA_VERSION", "tests/test_suite.py PINNED_HASHES", "100-person crowd scenario."),
    "scenario.v1-pub-10-route-invalidation.content_hash": _entry("SCHEMA_VERSION", "tests/test_suite.py PINNED_HASHES", "Stochastic route-invalidation scenario."),
    "scoring.constants_digest": _entry("SCORING_VERSION", REGISTRY_ONLY, "OFFICIAL_CONSTANTS.digest(): weights, thresholds and caps of the EMBER Score."),
    "scoring.episode-score.route-invalidation-fire-aware-s1": _entry(
        "SCORING_VERSION; SIMULATOR_VERSION; BASELINES_VERSION",
        REGISTRY_ONLY,
        "sha256_hex(score_episode(...).to_dict()) of the fire-aware baseline on the stochastic public scenario, seed 1.",
    ),
    "scoring.score-report.registry-public-v1": _entry(
        "SCORING_VERSION; STATISTICS_VERSION; ENVIRONMENT_PACK_VERSION; BASELINES_VERSION",
        REGISTRY_ONLY,
        "Official ScoreReport digest of the fire-aware baseline on the registry pack (200 bootstrap resamples).",
    ),
    "simulator.golden-record.one_person_one_exit": _entry(
        "SIMULATOR_VERSION", "tests/test_determinism.py GOLDEN_DIGESTS", "Run record digest, greedy policy, deterministic fire."
    ),
    "simulator.golden-record.stochastic_room": _entry(
        "SIMULATOR_VERSION", "tests/test_determinism.py GOLDEN_DIGESTS", "Run record digest, greedy policy, stochastic fire."
    ),
    "statistics.aggregate": _entry("STATISTICS_VERSION", "tests/test_statistics.py GOLDEN_STATISTICS_DIGESTS", "Aggregate statistics of the tiny fixed input."),
    "statistics.paired": _entry("STATISTICS_VERSION", "tests/test_statistics.py GOLDEN_STATISTICS_DIGESTS", "Paired statistics of the tiny fixed input."),
    "suite.public-v1.suite_hash": _entry("SCHEMA_VERSION; suite_version 1.0.0", "tests/test_suite.py SUITE_HASH", "Manifest hash of scenarios/public/v1."),
}

_COMMENT = [
    "Regression digest registry: one frozen digest per release-visible contract.",
    "Generated by 'python tests/release_digests.py --write' from SPEC in that file; do not edit by hand.",
    "tests/test_release.py checks it in-process against the test pins and in fresh interpreters; tools/check_release.py against the installed wheel.",
    "Change an entry only with a deliberate bump of its owner version (docs/versioning.md, section 4) and a changelog entry.",
]


def load_registry() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["entries"]
    return entries


def registry_text(digests: dict[str, str]) -> str:
    """The registry file for ``digests``: SPEC metadata plus the digest of each entry, LF line endings."""
    entries = {name: {"digest": digests[name], **SPEC[name]} for name in sorted(SPEC)}
    return json.dumps({"_comment": _COMMENT, "entries": entries}, indent=2, ensure_ascii=True) + "\n"


def public_suite_dir() -> Path:
    override = os.environ.get(PUBLIC_SUITE_ENV)
    return Path(override) if override else REPO_ROOT / "scenarios" / "public" / "v1"


def _suite_and_scenarios() -> dict[str, str]:
    from emberbench import load_suite

    suite = load_suite(public_suite_dir())
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


def _evaluation_manifests() -> dict[str, str]:
    from emberbench import build_evaluation_manifest, build_unique_world_manifest, load_suite

    suite = load_suite(public_suite_dir())
    unique = build_unique_world_manifest(suite, UNIQUE_WORLD_SEEDS, stochastic_worlds=UNIQUE_STOCHASTIC_WORLDS)
    return {
        "evaluation.public-v1-dev-seeds.manifest_hash": build_evaluation_manifest(suite, DEV_MANIFEST_SEEDS).manifest_hash(),
        "evaluation.public-v1-unique-worlds.manifest_hash": unique.manifest_hash(),
    }


def _baseline() -> dict[str, str]:
    from emberbench import agent_policy_factory, baseline_factory, build_evaluation_manifest, get_baseline, load_suite, run_evaluation
    from emberbench.integrity import digest_file
    from test_baselines import DEV_IDS, DEV_SEEDS, result_digest
    from test_evaluation import subset

    suite = subset(load_suite(public_suite_dir()), DEV_IDS)
    manifest = build_evaluation_manifest(suite, DEV_SEEDS)
    with tempfile.TemporaryDirectory(prefix="emberbench baseline ") as tmp:
        out = Path(tmp) / "result.json"
        result = run_evaluation(suite, manifest, REGISTRY_BASELINE, agent_policy_factory(baseline_factory(REGISTRY_BASELINE)), out)
        file_digest = digest_file(out)
    return {
        f"baseline.{REGISTRY_BASELINE}.config_digest": get_baseline(REGISTRY_BASELINE).config_digest(),
        f"baseline.{REGISTRY_BASELINE}.result_digest": result_digest(result),
        "evaluation.baseline-dev.result_file_sha256": file_digest,
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
    from emberbench import benchmark_versions, build_evaluation_manifest, build_evaluation_provenance, run_evaluation
    from emberbench.canonical import sha256_hex
    from emberbench.generator import GeneratorConfig, freeze_generated_suite, generate_suite
    from test_generator import BASE_CONFIG, BATCH, SEED
    from test_integrity import EVAL_SEEDS, MODEL, factory_of, identity

    with tempfile.TemporaryDirectory(prefix="emberbench provenance ") as tmp:
        root = Path(tmp)
        frozen = freeze_generated_suite(generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH), root / "frozen")
        manifest = build_evaluation_manifest(frozen.suite, EVAL_SEEDS)
        result = run_evaluation(frozen.suite, manifest, MODEL, factory_of(greedy_exit_policy), root / "greedy.json")
        provenance = build_evaluation_provenance(frozen, manifest, result, identity())
    return {"integrity.test-run.provenance_digest": provenance.provenance_digest(), "benchmark.versions": sha256_hex(benchmark_versions())}


def _cli() -> dict[str, str]:
    from emberbench.cli import main
    from emberbench.integrity import digest_file

    with tempfile.TemporaryDirectory(prefix="emberbench cli ") as tmp:
        out = Path(tmp) / "manifest.json"
        seeds = [arg for seed in DEV_MANIFEST_SEEDS for arg in ("--seed", str(seed))]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["evaluation", "manifest", "--suite", str(public_suite_dir()), *seeds, "--out", str(out), "--json"])
        if code != 0:
            raise RuntimeError(f"emberbench evaluation manifest exited with {code}")
        manifest_digest = digest_file(out)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        code = main(["version", "--json"])
    if code != 0:
        raise RuntimeError(f"emberbench version exited with {code}")
    return {
        "cli.evaluation-manifest.file_sha256": manifest_digest,
        "cli.version-json.stdout_sha256": hashlib.sha256(stdout.getvalue().encode("utf-8")).hexdigest(),
    }


def _replay_world_and_episode_score() -> dict[str, str]:
    from emberbench import build_replay_audit, environment_world_digest, load_suite, run_episode, score_episode
    from emberbench import agent_policy, baseline_factory
    from emberbench.canonical import sha256_hex
    from emberbench.visualization import animation_json

    scenario = load_suite(public_suite_dir()).get(STOCHASTIC_ID)
    record = run_episode(scenario, agent_policy(baseline_factory(REGISTRY_BASELINE)()), seed=STOCHASTIC_SEED)
    audit = build_replay_audit(scenario, record, crowd=True)
    return {
        "replay.route-invalidation-fire-aware-s1.audit_digest": audit.digest(),
        "replay.route-invalidation-fire-aware-s1.animation_sha256": hashlib.sha256(animation_json(audit).encode("ascii")).hexdigest(),
        f"environment-world.{STOCHASTIC_ID}.seed-{STOCHASTIC_SEED}": environment_world_digest(scenario, STOCHASTIC_SEED),
        "scoring.episode-score.route-invalidation-fire-aware-s1": sha256_hex(score_episode(scenario, record).to_dict()),
    }


def _pack_and_score_report() -> dict[str, str]:
    from emberbench import (
        OFFICIAL_CONSTANTS,
        agent_policy_factory,
        baseline_factory,
        freeze_environment_pack,
        load_suite,
        run_evaluation,
        score_result,
    )
    suite = load_suite(public_suite_dir())
    with tempfile.TemporaryDirectory(prefix="emberbench pack ") as tmp:
        pack = freeze_environment_pack(suite, PACK_SEEDS, Path(tmp) / "pack", pack_id=PACK_ID)
        result = run_evaluation(pack.suite, pack.manifest, REGISTRY_BASELINE, agent_policy_factory(baseline_factory(REGISTRY_BASELINE)), Path(tmp) / "result.json")
        report = score_result(result, pack, resamples=PACK_RESAMPLES)
    if not report.official:
        raise RuntimeError("the registry score report is not official")
    return {
        f"environment-pack.{PACK_ID}.pack_digest": pack.digest,
        f"scoring.score-report.{PACK_ID}": report.digest(),
        "scoring.constants_digest": OFFICIAL_CONSTANTS.digest(),
    }


def _rules() -> dict[str, str]:
    from emberbench.calibration import calibration_rules
    from emberbench.canonical import sha256_hex
    from emberbench.composition import rule_set_digest
    from emberbench.families import family_rules

    return {
        "rules.family": sha256_hex(family_rules()),
        "rules.calibration": sha256_hex(calibration_rules()),
        "rules.composition.rule_set_digest": rule_set_digest(),
    }


COMPUTERS: tuple[Callable[[], dict[str, str]], ...] = (
    _suite_and_scenarios,
    _golden_records,
    _stochastic_traces,
    _evaluation_manifests,
    _baseline,
    _generator,
    _statistics,
    _provenance,
    _cli,
    _replay_world_and_episode_score,
    _pack_and_score_report,
    _rules,
)


def compute_registry_digests() -> dict[str, str]:
    """Every registry digest, computed now. Keys are the registry entry names."""
    out: dict[str, str] = {}
    for compute in COMPUTERS:
        out.update(compute())
    return dict(sorted(out.items()))


def registry_drift(expected: dict[str, str], actual: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """``{name: {"registry": ..., "computed": ...}}`` for every entry that differs or exists on one side only."""
    return {
        name: {"registry": expected.get(name), "computed": actual.get(name)}
        for name in sorted(set(expected) | set(actual))
        if expected.get(name) != actual.get(name)
    }


def main(argv: list[str]) -> int:
    computed = compute_registry_digests()
    if set(computed) != set(SPEC):
        sys.stderr.write(f"SPEC and the computed digests disagree: {sorted(set(computed) ^ set(SPEC))}\n")
        return 1
    if "--write" in argv:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_bytes(registry_text(computed).encode("ascii"))
        sys.stdout.write(f"wrote {len(computed)} entries to {REGISTRY_PATH}\n")
        return 0
    if "--check" in argv:
        expected = {name: entry["digest"] for name, entry in load_registry().items()}
        drift = registry_drift(expected, computed)
        if drift:
            sys.stderr.write(f"registry drift in {len(drift)} entr{'y' if len(drift) == 1 else 'ies'}:\n{json.dumps(drift, indent=2)}\n")
            return 1
        if REGISTRY_PATH.read_bytes() != registry_text(computed).encode("ascii"):
            sys.stderr.write("the registry file is not the output of --write (metadata, order or formatting differs)\n")
            return 1
        sys.stdout.write(f"registry verified: {len(computed)} entries\n")
        return 0
    sys.stdout.write(json.dumps(computed, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
