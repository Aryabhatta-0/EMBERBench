"""Run the V1 evaluation pack: PUBLIC DEVELOPMENT EVALUATION -- NOT A HIDDEN-SUITE LEADERBOARD.

Usage::

    python tools/experiments/v1/run_public_benchmark.py --out DIR
    python tools/experiments/v1/run_public_benchmark.py --out DIR --protocol smoke

The runner uses only public EMBERBench APIs and adds no simulator, metric,
statistic or score of its own:

- schedule: ``build_unique_world_manifest`` (one episode per unique
  environment world) and ``freeze_environment_pack``;
- runs: ``run_evaluation`` (resumable) and ``load_evaluation_result``
  (authentication by replay);
- the headline: ``emberbench.scoring`` -- ``score_result`` against the frozen
  environment pack, ``compare_scores`` and ``leaderboard_order``;
- diagnostics (never the headline): ``aggregate_result``;
- separate privileged references: ``emberbench.oracle.privileged_manifest_bounds``
  and the capacity-aware bound ``emberbench.calibration.capacity_savable``.

Rules:

- No network. An audit hook stops the process on any socket or URL event.
- Deterministic inputs: the suite and the candidate seeds are pinned in
  ``pack_protocol.py``; a suite or manifest hash mismatch stops the run.
- Fail closed: any library error stops the run with exit status 1.
- Resumable: per-model results are resumed by ``run_evaluation``; every other
  output that already exists must be byte-identical to the recomputed bytes.
- Never overwrites: an existing output with different bytes is an error.
- ``DIR`` must be outside the repository, so outputs are never committed by
  accident and never land inside the Python package.
"""

import argparse
import dataclasses
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from emberbench import (
    EnvironmentPack,
    EvaluationManifest,
    EvaluationResult,
    ScoreReport,
    Suite,
    agent_policy_factory,
    aggregate_result,
    benchmark_versions,
    build_evaluation_manifest,
    build_unique_world_manifest,
    compare_scores,
    environment_world_digest,
    freeze_environment_pack,
    load_environment_pack,
    load_evaluation_result,
    load_suite,
    run_evaluation,
    score_result,
)
from emberbench.calibration import capacity_savable  # privileged: evaluator-side references only
from emberbench.evaluation import resolve_scenario
from emberbench.oracle import privileged_manifest_bounds  # privileged: evaluator-side references only

import pack_agents
import pack_index
import pack_protocol
import pack_report

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK_DIR = "environment-pack"
SOURCE_SUITE_DIR = "source-suite"


class PackError(Exception):
    """The pack cannot run or its outputs do not match. Nothing is repaired."""


# --------------------------------------------------------------------------- no network


def _deny_network(event: str, args: tuple[Any, ...]) -> None:
    if event.startswith(("socket.", "urllib.", "http.client.")) and event != "socket.__new__":
        raise PackError(f"network access is not allowed in the evaluation pack ({event})")


# --------------------------------------------------------------------------- outputs


def write_new_or_same(path: Path, text: str, *, encoding: str = "ascii") -> str:
    """Create ``path`` with ``text`` (LF; ASCII unless ``encoding`` says otherwise), or accept an identical file.

    Returns ``"written"`` or ``"unchanged"``. An existing file with other
    bytes raises :class:`PackError`; it is never replaced.
    """
    data = text.encode(encoding)
    if path.exists():
        if path.read_bytes() != data:
            raise PackError(f"{path}: exists with different content; refusing to overwrite (use a new --out)")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)  # fails if the path appeared meanwhile: never replaces
        except FileExistsError:
            raise PackError(f"{path}: appeared during the run; refusing to overwrite") from None
    finally:
        tmp.unlink(missing_ok=True)
    return "written"


def check_same(path: Path, text: str, *, encoding: str = "ascii") -> str:
    """Verification mode: ``path`` must exist with exactly the recomputed bytes. Nothing is written."""
    if not path.is_file() or path.read_bytes() != text.encode(encoding):
        raise PackError(f"{path}: missing or differs from the artifact recomputed from the authenticated inputs")
    return "unchanged"


def check_benchmark_versions() -> dict[str, str]:
    """``benchmark_versions()`` of the running code; fail closed unless it equals the pinned revision exactly."""
    versions = benchmark_versions()
    pinned = pack_protocol.BENCHMARK_VERSIONS
    if versions != pinned:
        differ = sorted(k for k in versions.keys() | pinned.keys() if versions.get(k) != pinned.get(k))
        raise PackError(f"benchmark versions differ from the pinned benchmark revision {pack_protocol.BENCHMARK_REVISION}: {differ}")
    return versions


def _check_out_dir(out: Path) -> Path:
    resolved = out.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise PackError(f"{out}: the output directory must be outside the repository ({REPO_ROOT})")
    if resolved.exists() and not resolved.is_dir():
        raise PackError(f"{out}: not a directory")
    if not resolved.parent.is_dir():
        raise PackError(f"{out}: parent directory does not exist")
    resolved.mkdir(exist_ok=True)
    return resolved


# --------------------------------------------------------------------------- protocol inputs


def protocol_suite(protocol: pack_protocol.Protocol) -> Suite:
    """The canonical public suite, or the protocol's subset of it, checked against the pinned hash."""
    suite = load_suite(REPO_ROOT / pack_protocol.PUBLIC_SUITE_DIR)
    if suite.manifest.suite_hash() != pack_protocol.PUBLIC_SUITE_HASH:
        raise PackError(f"canonical public suite hash changed: {suite.manifest.suite_hash()}")
    if protocol.scenario_ids is not None:
        known = {e.scenario_id for e in suite.manifest.entries}
        missing = [sid for sid in protocol.scenario_ids if sid not in known]
        if missing:
            raise PackError(f"unknown scenarios in protocol {protocol.name}: {missing}")
        ids = set(protocol.scenario_ids)
        suite = Suite(
            root=suite.root,
            manifest=dataclasses.replace(
                suite.manifest, entries=tuple(e for e in suite.manifest.entries if e.scenario_id in ids)
            ),
            scenarios=tuple(s for s in suite.scenarios if s.scenario_id in ids),
        )
    if suite.manifest.suite_hash() != protocol.suite_hash:
        raise PackError(f"protocol {protocol.name}: suite hash {suite.manifest.suite_hash()} is not the pinned hash")
    return suite


def protocol_manifest(protocol: pack_protocol.Protocol, suite: Suite) -> EvaluationManifest:
    """The unique-world manifest of the protocol, checked against the pinned hash."""
    manifest = build_unique_world_manifest(suite, protocol.candidate_seeds, stochastic_worlds=protocol.stochastic_worlds)
    if manifest.manifest_hash() != protocol.manifest_hash:
        raise PackError(f"protocol {protocol.name}: manifest hash {manifest.manifest_hash()} is not the pinned hash")
    return manifest


def _source_suite(suite: Suite, out: Path, write: Callable[..., str] = write_new_or_same) -> Suite:
    """The suite on disk, as the environment pack needs it. A subset is written to ``out/source-suite``."""
    if suite.root.resolve() == (REPO_ROOT / pack_protocol.PUBLIC_SUITE_DIR).resolve() and suite.manifest == load_suite(suite.root).manifest:
        return suite
    target = out / SOURCE_SUITE_DIR
    write(target / "manifest.json", suite.manifest.to_file_text())
    for entry in suite.manifest.entries:
        write(target / entry.path, (suite.root / entry.path).read_bytes().decode("utf-8"), encoding="utf-8")
    copy = load_suite(target)
    if copy.manifest.suite_hash() != suite.manifest.suite_hash():
        raise PackError(f"{target}: the written subset suite differs from the protocol suite")
    return copy


def _environment_pack(
    protocol: pack_protocol.Protocol, suite: Suite, manifest: EvaluationManifest, out: Path, *, verify: bool = False
) -> EnvironmentPack:
    """Freeze the environment pack, or load and check the one a previous run froze (the only option when verifying)."""
    root = out / PACK_DIR
    pack_id = f"emberbench-v1-{protocol.name}-unique-worlds"
    source = _source_suite(suite, out, check_same if verify else write_new_or_same)
    if root.exists() and any(root.iterdir()):
        pack = load_environment_pack(root)  # rejects a missing, extra or changed file
    elif verify:
        raise PackError(f"{root}: the environment pack is missing")
    else:
        pack = freeze_environment_pack(source, manifest, root, pack_id=pack_id)
    if pack.pack_id != pack_id or pack.manifest != manifest or pack.suite.manifest.suite_hash() != suite.manifest.suite_hash():
        raise PackError(f"{root}: the environment pack does not match protocol {protocol.name}")
    return pack


def unique_worlds(suite: Suite, manifest: EvaluationManifest) -> list[dict[str, Any]]:
    """One record per scheduled episode with its environment-world digest. Every world must be distinct."""
    worlds = []
    for spec in manifest.episodes:
        scenario = resolve_scenario(suite, spec)
        worlds.append(
            {
                "index": spec.index,
                "scenario_id": spec.scenario_id,
                "seed": spec.seed,
                "fire_model": scenario.fire.model.value,
                "environment_world_digest": environment_world_digest(scenario, spec.seed),
            }
        )
    digests = [w["environment_world_digest"] for w in worlds]
    if len(set(digests)) != len(digests):
        raise PackError("the manifest schedules a duplicate environment world")
    return worlds


def nominal_schedule(protocol: pack_protocol.Protocol, suite: Suite) -> dict[str, int]:
    """The full scenario x candidate-seed product and how many distinct worlds it holds (counted, never scheduled)."""
    nominal = build_evaluation_manifest(suite, protocol.candidate_seeds)
    distinct = {environment_world_digest(resolve_scenario(suite, spec), spec.seed) for spec in nominal.episodes}
    return {"nominal_candidate_episodes": len(nominal.episodes), "distinct_worlds_among_candidates": len(distinct)}


# --------------------------------------------------------------------------- privileged references


def privileged_references(suite: Suite, manifest: EvaluationManifest, pack: EnvironmentPack) -> dict[str, Any]:
    """M*, the capacity-aware bound M_cap and the single-person time bound per world. Never a model row."""
    worlds = []
    for spec, bound in zip(manifest.episodes, privileged_manifest_bounds(suite, manifest), strict=True):
        if (bound.scenario_id, bound.seed) != (spec.scenario_id, spec.seed):
            raise PackError("privileged bounds are not in manifest order")
        savable, cap = capacity_savable(resolve_scenario(suite, spec), spec.seed)
        if savable != bound.max_evacuated or cap > savable:
            raise PackError(f"episode {spec.index}: inconsistent privileged references (M* {savable} vs {bound.max_evacuated}, M_cap {cap})")
        worlds.append(
            {
                "index": spec.index,
                "scenario_id": spec.scenario_id,
                "seed": spec.seed,
                "population": bound.population,
                "max_steps": bound.max_steps,
                "m_star": bound.max_evacuated,
                "m_cap": cap,
                "min_person_time_to_safety_total": bound.min_person_time_to_safety_total,
            }
        )
    return {
        "artifact": "privileged-references",
        "label": pack_protocol.RUN_LABEL,
        "privileged": True,
        "fair": False,
        "competing_model": False,
        "note": (
            "Privileged relaxed references. They read the scenario and the evaluator seed (the whole future fire). "
            "M* is the official savable count (people who can escape alone); M_cap <= M* adds exit and cut capacity "
            "(emberbench.calibration). They are bounds, not models, not baselines and not leaderboard rows."
        ),
        "evaluation_manifest_hash": manifest.manifest_hash(),
        "environment_pack_digest": pack.digest,
        "worlds": worlds,
    }


# --------------------------------------------------------------------------- the run


def _log(message: str) -> None:
    print(message, flush=True)


def run_pack(protocol: pack_protocol.Protocol, out: Path, *, log: Callable[[str], None] = _log, verify: bool = False) -> dict[str, Any]:
    """Run the whole pack into ``out`` and return the canonical index. See the module docstring.

    With ``verify=True`` nothing is run or written: every result in ``out`` is
    authenticated by replay against the frozen environment pack, every other
    artifact is recomputed from the authenticated inputs and must equal the
    file byte for byte, and ``out`` must hold no other file. Any difference
    stops with :class:`PackError` (fail closed).
    """
    started = time.perf_counter()
    check_benchmark_versions()
    write = check_same if verify else write_new_or_same
    if verify:
        out = out.resolve()
        if not (out / "index.json").is_file():
            raise PackError(f"{out}: no pack run to verify (index.json is missing)")
    else:
        out = _check_out_dir(out)
    suite = protocol_suite(protocol)
    manifest = protocol_manifest(protocol, suite)
    pack = _environment_pack(protocol, suite, manifest, out, verify=verify)
    suite, manifest = pack.suite, pack.manifest  # every model runs the frozen pack
    worlds = unique_worlds(suite, manifest)
    nominal = nominal_schedule(protocol, suite)
    log(f"pack     {pack.pack_id} sha256:{pack.digest}: {len(worlds)} unique worlds of {nominal['nominal_candidate_episodes']} nominal episodes")

    specs = pack_agents.pack_agent_specs()
    model_ids = tuple(s.model_id for s in specs)
    if len({pack_index.model_slug(m) for m in model_ids}) != len(model_ids):
        raise PackError("duplicate model IDs")

    timings: dict[str, float] = {}
    results: dict[str, EvaluationResult] = {}
    t = time.perf_counter()
    for spec in specs:
        slug = pack_index.model_slug(spec.model_id)
        # The descriptor pins the configuration (and, for pack agents, the code)
        # before any episode runs, so a resume can never mix two agent versions.
        write(out / "models" / f"{slug}.json", pack_index.canonical(dict(spec.descriptor)) + "\n")
        path = out / "results" / f"{slug}.json"
        if verify:
            if not path.is_file():
                raise PackError(f"{path}: result is missing")
        else:
            path.parent.mkdir(exist_ok=True)
            log(f"run      {spec.model_id}")
            run_evaluation(suite, manifest, spec.model_id, agent_policy_factory(spec.factory), path, resume=True)
        result = load_evaluation_result(path, suite)  # authenticate every episode by replay
        if not result.is_complete or result.manifest_hash != manifest.manifest_hash():
            raise PackError(f"{path}: result is incomplete or belongs to another manifest")
        if verify:
            check_same(path, result.to_file_text())  # the canonical bytes that run_evaluation writes
        results[spec.model_id] = result
    timings["episodes"] = time.perf_counter() - t

    t = time.perf_counter()
    reports: dict[str, ScoreReport] = {}
    for spec in specs:
        log(f"score    {spec.model_id}")
        report = score_result(results[spec.model_id], pack, resamples=protocol.resamples)
        if not report.official:
            raise PackError(f"{spec.model_id}: the score report is not an official-procedure score")
        write(out / pack_index.score_file(spec.model_id), report.to_json() + "\n")
        reports[spec.model_id] = report

    comparisons: dict[tuple[str, str], str] = {}
    for a, b in pack_protocol.paired_comparisons(model_ids):
        log(f"compare  {a} - {b}")
        doc = pack_index.comparison_document(reports[a], reports[b], compare_scores(reports[a], reports[b], resamples=protocol.resamples))
        text = pack_index.canonical(doc) + "\n"
        write(out / pack_index.comparison_file(a, b), text)
        comparisons[(a, b)] = text
    timings["scoring"] = time.perf_counter() - t

    t = time.perf_counter()
    diagnostics: dict[str, str] = {}
    for spec in specs:
        log(f"diag     {spec.model_id}")
        text = aggregate_result(results[spec.model_id], suite, resamples=protocol.resamples).to_json() + "\n"
        write(out / pack_index.diagnostics_file(spec.model_id), text)
        diagnostics[spec.model_id] = text
    timings["diagnostics"] = time.perf_counter() - t

    t = time.perf_counter()
    log("refs     privileged references (separate context, not models)")
    references = privileged_references(suite, manifest, pack)
    references_text = pack_index.canonical(references) + "\n"
    write(out / "privileged" / "references.json", references_text)
    timings["privileged"] = time.perf_counter() - t

    index = pack_index.build_index(
        protocol, suite, manifest, pack, worlds, nominal, specs, results, reports, comparisons, diagnostics, references_text
    )
    write(out / "index.json", pack_index.canonical(index) + "\n")
    report_text = pack_report.render_report(protocol, index, results, reports, comparisons, references)
    write(out / "REPORT.md", report_text, encoding="utf-8")
    if verify:
        extra = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p not in _expected_files(out, index))
        if extra:
            raise PackError(f"{out}: files that no pack artifact accounts for: {extra}")
    timings["total"] = time.perf_counter() - started
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"index    {pack_index.index_digest(index)}")
    # Wall-clock values are logged only: they are not reproducible, so no pinned artifact holds them.
    log("time     " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()) + f"; artifacts {size} bytes")
    return index


def _expected_files(out: Path, index: Mapping[str, Any]) -> set[Path]:
    """Every file a pack run writes: the indexed artifacts, the per-model descriptors, the frozen pack and the subset suite."""
    files = {out / "index.json", out / "REPORT.md", out / "privileged" / "references.json"}
    for model in index["models"]:
        slug = pack_index.model_slug(model["model_id"])
        files |= {out / "models" / f"{slug}.json", out / model["result"]["file"], out / model["score_report"]["file"], out / model["diagnostics"]["file"]}
    files |= {out / c["file"] for c in index["paired_comparisons"]}
    for tree in (PACK_DIR, SOURCE_SUITE_DIR):  # the pack verifies its own file list on load
        files |= {p for p in (out / tree).rglob("*") if p.is_file()}
    return files


def verify_pack(protocol: pack_protocol.Protocol, out: Path, *, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Re-authenticate a finished run in ``out`` without running or writing anything. See :func:`run_pack`."""
    return run_pack(protocol, out, log=log, verify=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory, outside the repository")
    parser.add_argument("--protocol", choices=sorted(pack_protocol.PROTOCOLS), default=pack_protocol.PUBLIC.name)
    parser.add_argument("--verify", action="store_true", help="re-authenticate a finished run in --out; run and write nothing")
    args = parser.parse_args(argv)
    sys.addaudithook(_deny_network)
    try:
        (verify_pack if args.verify else run_pack)(pack_protocol.PROTOCOLS[args.protocol], args.out)
    except PackError as exc:
        print(f"run_public_benchmark: error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # fail closed on any library error, with its message
        print(f"run_public_benchmark: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
