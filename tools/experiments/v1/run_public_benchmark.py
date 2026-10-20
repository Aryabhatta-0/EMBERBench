"""Run the V1 evaluation pack: PUBLIC DEVELOPMENT EVALUATION -- NOT HELD-OUT SCORE.

Usage::

    python tools/experiments/v1/run_public_benchmark.py --out DIR
    python tools/experiments/v1/run_public_benchmark.py --out DIR --protocol smoke

The runner uses only public EMBERBench APIs: ``load_suite``,
``build_evaluation_manifest``, ``run_evaluation`` (resumable),
``load_evaluation_result`` (authentication by replay), ``aggregate_result``,
``compare_results`` and, for the separate privileged context,
``emberbench.oracle.privileged_manifest_bounds``. It adds no simulator,
metric or statistic of its own.

Rules:

- No network. An audit hook stops the process on any socket or URL event.
- Deterministic inputs: the suite and the seed schedule are pinned in
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from emberbench import (
    EvaluationManifest,
    EvaluationResult,
    Suite,
    agent_policy_factory,
    aggregate_result,
    build_evaluation_manifest,
    compare_results,
    load_evaluation_result,
    load_suite,
    run_evaluation,
)
from emberbench.oracle import privileged_manifest_bounds  # privileged: evaluator-side context only

import pack_agents
import pack_index
import pack_protocol
import pack_report

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    if suite.manifest.suite_hash() != pack_protocol.PUBLIC.suite_hash:
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
    manifest = build_evaluation_manifest(suite, protocol.seeds)
    if manifest.manifest_hash() != protocol.manifest_hash:
        raise PackError(f"protocol {protocol.name}: manifest hash {manifest.manifest_hash()} is not the pinned hash")
    return manifest


# --------------------------------------------------------------------------- the run


def _log(message: str) -> None:
    print(message, flush=True)


def run_pack(protocol: pack_protocol.Protocol, out: Path, *, log: Any = _log) -> dict[str, Any]:
    """Run the whole pack into ``out`` and return the canonical index. See the module docstring."""
    out = _check_out_dir(out)
    suite = protocol_suite(protocol)
    manifest = protocol_manifest(protocol, suite)
    write_new_or_same(out / "manifest" / "public-dev-manifest.json", manifest.to_file_text())

    specs = pack_agents.pack_agent_specs()
    model_ids = tuple(s.model_id for s in specs)
    if len({pack_index.model_slug(m) for m in model_ids}) != len(model_ids):
        raise PackError("duplicate model IDs")

    results: dict[str, EvaluationResult] = {}
    for spec in specs:
        slug = pack_index.model_slug(spec.model_id)
        # The descriptor pins the configuration (and, for pack agents, the code)
        # before any episode runs, so a resume can never mix two agent versions.
        write_new_or_same(out / "models" / f"{slug}.json", pack_index.canonical(dict(spec.descriptor)) + "\n")
        path = out / "results" / f"{slug}.json"
        path.parent.mkdir(exist_ok=True)
        log(f"run      {spec.model_id}")
        run_evaluation(suite, manifest, spec.model_id, agent_policy_factory(spec.factory), path, resume=True)
        result = load_evaluation_result(path, suite)  # authenticate every episode by replay
        if not result.is_complete or result.manifest_hash != manifest.manifest_hash():
            raise PackError(f"{path}: result is incomplete or belongs to another manifest")
        results[spec.model_id] = result

    aggregates: dict[str, str] = {}
    for spec in specs:
        log(f"stats    {spec.model_id}")
        stats = aggregate_result(results[spec.model_id], suite, resamples=protocol.resamples)
        text = stats.to_json() + "\n"
        write_new_or_same(out / "statistics" / "aggregate" / f"{pack_index.model_slug(spec.model_id)}.json", text)
        aggregates[spec.model_id] = text

    paired: dict[tuple[str, str], str] = {}
    for a, b in pack_protocol.paired_comparisons(model_ids):
        log(f"compare  {a} - {b}")
        comparison = compare_results(results[a], results[b], suite, resamples=protocol.resamples)
        text = comparison.to_json() + "\n"
        write_new_or_same(out / pack_index.paired_file(a, b), text)
        paired[(a, b)] = text

    log("bound    privileged escape bound (separate context, not a model)")
    bounds = privileged_manifest_bounds(suite, manifest)
    bound_doc = pack_index.bound_document(manifest, bounds)
    bound_text = pack_index.canonical(bound_doc) + "\n"
    write_new_or_same(out / "privileged" / "escape-bound.json", bound_text)

    index = pack_index.build_index(protocol, suite, manifest, specs, results, aggregates, paired, bound_text)
    write_new_or_same(out / "index.json", pack_index.canonical(index) + "\n")
    report = pack_report.render_report(protocol, index, results, aggregates, paired, bound_doc)
    write_new_or_same(out / "REPORT.md", report, encoding="utf-8")
    log(f"index    {pack_index.index_digest(index)}")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory, outside the repository")
    parser.add_argument("--protocol", choices=sorted(pack_protocol.PROTOCOLS), default=pack_protocol.PUBLIC.name)
    args = parser.parse_args(argv)
    sys.addaudithook(_deny_network)
    try:
        run_pack(pack_protocol.PROTOCOLS[args.protocol], args.out)
    except PackError as exc:
        print(f"run_public_benchmark: error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # fail closed on any library error, with its message
        print(f"run_public_benchmark: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
