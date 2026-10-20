"""Frozen environment packs (environment pack version 1.0): one immutable environment for every model.

An environment pack freezes everything that defines *where* models are
evaluated, before any model runs:

- the exact bytes of every scenario file and of the suite manifest;
- the ordered evaluation manifest (the seed schedule);
- the generator freeze record, when the suite is procedurally generated;
- the simulator, observation, action-schema, scenario-schema, metrics,
  generator and evaluation-manifest versions;
- one ``fire_world_digest`` per scheduled episode: the SHA-256 of the
  complete burning-set trajectory that the simulator will produce.

The pack digest is the SHA-256 of the canonical ``pack.json``. ``pack.json``
pins the SHA-256 of every other file, so the digest pins every byte.

Rules for this module:

- **Never regenerated per model.** A pack is written once and only read
  afterwards. Every model is scored against the same pack.
- **Model-independent.** In V1 the fire ignores actions and people, so the
  burning-set trajectory of an episode is a function of the scenario and the
  seed only. The pack stores its digest; scoring checks every run record
  against the same trajectory. Nothing in the pack depends on a model or on
  mid-episode state that a model can change.
- **Fail closed.** :func:`load_environment_pack` rejects a missing, extra or
  changed file, a symbolic link, a version mismatch, a schedule that does not
  target the suite, and a fire world that does not match.
- **Private asset.** A pack contains the evaluator seeds. Publish only its
  digest before the evaluation, as a commitment; publish the pack after it.

The definition is in ``docs/environment-pack.md``.
"""

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import SIMULATOR_VERSION
from .agent import ACTION_SCHEMA_VERSION
from .canonical import canonical_json, sha256_hex
from .errors import EmberBenchError, EnvironmentPackError
from .evaluation import (
    EVALUATION_MANIFEST_VERSION,
    EvaluationManifest,
    build_evaluation_manifest,
    resolve_scenario,
    verify_manifest,
)
from .generator import (
    GENERATION_MANIFEST_NAME,
    GENERATION_MANIFEST_VERSION,
    GENERATOR_CONFIG_VERSION,
    GENERATOR_VERSION,
    SUITE_DIR_NAME,
    FrozenGeneratedSuite,
    GenerationManifest,
    load_generated_suite,
)
from .metrics import METRICS_VERSION
from .observation import OBSERVATION_VERSION
from .oracle import fire_trajectory
from .scenario import SCHEMA_VERSION, Scenario
from .suite import HASH_PATTERN, MANIFEST_NAME, Suite, load_suite

ENVIRONMENT_PACK_VERSION = "1.0"
PACK_FILE_NAME = "pack.json"
PACK_MANIFEST_NAME = "evaluation-manifest.json"
PACK_SUITE_DIR = "suite"  # a plain suite: manifest.json and the scenario files
PACK_GENERATED_DIR = "generated-suite"  # a generated suite: generation.json and suite/

_PACK_KEYS = ("environment_pack_version", "pack_id", "source", "versions", "suite", "evaluation_manifest_hash", "episodes", "files")


def _fail(message: str) -> EnvironmentPackError:
    return EnvironmentPackError(message)


def pack_versions() -> dict[str, str]:
    """Every version that defines the environment. A pack is valid only under exactly these versions."""
    return {
        "environment_pack": ENVIRONMENT_PACK_VERSION,
        "simulator": SIMULATOR_VERSION,
        "observation": OBSERVATION_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "scenario_schema": SCHEMA_VERSION,
        "metrics": METRICS_VERSION,
        "generator": GENERATOR_VERSION,
        "generator_config": GENERATOR_CONFIG_VERSION,
        "generation_manifest": GENERATION_MANIFEST_VERSION,
        "evaluation_manifest": EVALUATION_MANIFEST_VERSION,
    }


def trajectory_digest(scenario_hash: str, seed: int, trajectory: Sequence[frozenset[Any]]) -> str:
    """SHA-256 of a burning-set trajectory: one sorted ``[row, col]`` list per snapshot ``0..H``."""
    return sha256_hex(
        {
            "scenario_hash": scenario_hash,
            "seed": seed,
            "burning": [[[p.row, p.col] for p in sorted(burning)] for burning in trajectory],
        }
    )


def fire_world_digest(scenario: Scenario, seed: int) -> str:
    """The digest of the exogenous fire of one episode. Identical for every model, by construction."""
    return trajectory_digest(scenario.content_hash(), seed, fire_trajectory(scenario, seed))


@dataclass(frozen=True)
class PackEpisode:
    """One scheduled episode of the pack, with its fire world."""

    index: int
    scenario_id: str
    scenario_hash: str
    seed: int
    fire_world_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
            "fire_world_digest": self.fire_world_digest,
        }


@dataclass(frozen=True)
class EnvironmentPack:
    """A verified, frozen environment pack."""

    root: Path
    pack_id: str
    suite: Suite
    manifest: EvaluationManifest
    generation: GenerationManifest | None
    episodes: tuple[PackEpisode, ...]
    record: dict[str, Any]  # the parsed pack.json
    digest: str  # SHA-256 of the canonical pack.json

    def episode(self, index: int) -> PackEpisode:
        return self.episodes[index]


# --------------------------------------------------------------------------- freeze


def _source_files(source: Suite | FrozenGeneratedSuite) -> tuple[str, list[tuple[str, Path]]]:
    """``(source kind, [(path inside the pack, source path)])`` for every file of the environment."""
    suite = source.suite if isinstance(source, FrozenGeneratedSuite) else source
    files = [(MANIFEST_NAME, suite.root / MANIFEST_NAME)]
    files += [(e.path, suite.root / e.path) for e in suite.manifest.entries]
    if isinstance(source, FrozenGeneratedSuite):
        out = [(f"{PACK_GENERATED_DIR}/{SUITE_DIR_NAME}/{rel}", src) for rel, src in files]
        out.append((f"{PACK_GENERATED_DIR}/{GENERATION_MANIFEST_NAME}", source.root / GENERATION_MANIFEST_NAME))
        return "generated_suite", out
    return "suite", [(f"{PACK_SUITE_DIR}/{rel}", src) for rel, src in files]


def _pack_record(pack_id: str, source: str, suite: Suite, manifest: EvaluationManifest, episodes: Sequence[PackEpisode], files: dict[str, str]) -> dict[str, Any]:
    m = suite.manifest
    return {
        "environment_pack_version": ENVIRONMENT_PACK_VERSION,
        "pack_id": pack_id,
        "source": source,
        "versions": pack_versions(),
        "suite": {
            "suite_id": m.suite_id,
            "suite_version": m.suite_version,
            "suite_hash": m.suite_hash(),
            "scenarios": [
                {"scenario_id": e.scenario_id, "content_hash": e.content_hash, "primary_family": e.primary_family.value}
                for e in m.entries
            ],
        },
        "evaluation_manifest_hash": manifest.manifest_hash(),
        "episodes": [e.to_dict() for e in episodes],
        "files": [[path, files[path]] for path in sorted(files)],
    }


def _episodes(suite: Suite, manifest: EvaluationManifest) -> tuple[PackEpisode, ...]:
    return tuple(
        PackEpisode(spec.index, spec.scenario_id, spec.scenario_hash, spec.seed, fire_world_digest(resolve_scenario(suite, spec), spec.seed))
        for spec in manifest.episodes
    )


def _check_pack_id(pack_id: object) -> str:
    if not isinstance(pack_id, str) or not pack_id or len(pack_id) > 128 or not pack_id.isascii() or not all(c.isalnum() or c in "-_." for c in pack_id):
        raise _fail("pack_id must be 1-128 ASCII letters, digits, '-', '_' or '.'")
    return pack_id


def freeze_environment_pack(
    source: Suite | FrozenGeneratedSuite,
    schedule: Sequence[int] | EvaluationManifest,
    directory: str | Path,
    *,
    pack_id: str,
) -> EnvironmentPack:
    """Freeze ``source`` and its seed schedule into ``directory`` and return the verified pack.

    ``schedule`` is either an ordered seed list (scheduled with
    :func:`~emberbench.build_evaluation_manifest`) or an existing manifest
    that targets the suite. ``directory`` must not exist or must be empty: a
    pack is never overwritten. The pack is written to a temporary sibling and
    renamed into place, then loaded back and verified from disk.
    """
    pack_id = _check_pack_id(pack_id)
    if not isinstance(source, (Suite, FrozenGeneratedSuite)):
        raise TypeError("source must be a Suite or a FrozenGeneratedSuite")
    suite = source.suite if isinstance(source, FrozenGeneratedSuite) else source
    try:
        manifest = schedule if isinstance(schedule, EvaluationManifest) else build_evaluation_manifest(suite, schedule)
        verify_manifest(manifest, suite)
    except EmberBenchError as exc:
        raise _fail(f"schedule does not target the suite: {exc}") from exc
    root = Path(directory)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise _fail(f"{root}: already exists and is not empty; a pack is never overwritten")

    kind, sources = _source_files(source)
    blobs: dict[str, bytes] = {}
    for rel, src in sources:
        if src.is_symlink() or not src.is_file():
            raise _fail(f"{src}: not a regular file")
        blobs[rel] = src.read_bytes()
    blobs[PACK_MANIFEST_NAME] = manifest.to_file_text().encode("utf-8")
    files = {rel: hashlib.sha256(data).hexdigest() for rel, data in blobs.items()}
    record = _pack_record(pack_id, kind, suite, manifest, _episodes(suite, manifest), files)

    staging = root.parent / f".{root.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        for rel, data in blobs.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / PACK_FILE_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if root.exists():
            root.rmdir()  # empty, checked above
        os.replace(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    pack = load_environment_pack(root)
    # The source suite was verified when it was loaded; the copy must be the same suite.
    if pack.suite.manifest != suite.manifest or tuple(s.content_hash() for s in pack.suite.scenarios) != tuple(s.content_hash() for s in suite.scenarios):
        raise _fail(f"{root}: the frozen suite differs from the source suite; its files changed after loading")
    return pack


# --------------------------------------------------------------------------- load and verify


def _listed_files(record: dict[str, Any]) -> dict[str, str]:
    raw = record["files"]
    if not isinstance(raw, list):
        raise _fail("files: expected a list")
    files: dict[str, str] = {}
    for i, item in enumerate(raw):
        if not (isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)):
            raise _fail(f"files[{i}]: expected [path, sha256]")
        path, digest = item
        if not HASH_PATTERN.match(digest) or path in files or path == PACK_FILE_NAME:
            raise _fail(f"files[{i}]: invalid or duplicate entry")
        parts = path.split("/")
        if path.startswith("/") or any(p in ("", ".", "..") for p in parts) or "\\" in path:
            raise _fail(f"files[{i}]: unsafe path {path!r}")
        files[path] = digest
    if list(files) != sorted(files):
        raise _fail("files: must be sorted by path")
    return files


def _present_files(root: Path) -> set[str]:
    present: set[str] = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise _fail(f"{p}: symbolic links are not allowed in a pack")
        if p.is_file():
            present.add(p.relative_to(root).as_posix())
        elif not p.is_dir():
            raise _fail(f"{p}: not a regular file or directory")
    return present


def load_environment_pack(directory: str | Path, *, expected_digest: str | None = None) -> EnvironmentPack:
    """Load and fully verify a pack. Raises :class:`EnvironmentPackError` on any difference.

    ``expected_digest`` is the published commitment; when given, the pack
    digest must equal it.
    """
    root = Path(directory)
    try:
        return _load(root, expected_digest)
    except EnvironmentPackError:
        raise
    except (EmberBenchError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise _fail(f"{root}: {exc}") from exc


def _load(root: Path, expected_digest: str | None) -> EnvironmentPack:
    if not root.is_dir() or root.is_symlink():
        raise _fail(f"{root}: not a directory")
    text = (root / PACK_FILE_NAME).read_text(encoding="utf-8")
    record = json.loads(text)
    if not isinstance(record, dict) or sorted(record) != sorted(_PACK_KEYS):
        raise _fail(f"{PACK_FILE_NAME}: expected exactly the keys {sorted(_PACK_KEYS)}")
    if record["environment_pack_version"] != ENVIRONMENT_PACK_VERSION:
        raise _fail(f"environment pack version {record['environment_pack_version']!r} is not {ENVIRONMENT_PACK_VERSION!r}")
    if record["versions"] != pack_versions():
        raise _fail("versions: the pack was frozen under other benchmark versions")
    pack_id = _check_pack_id(record["pack_id"])
    digest = sha256_hex(record)
    if expected_digest is not None and digest != expected_digest:
        raise _fail(f"pack digest {digest} does not match the expected {expected_digest}")

    files = _listed_files(record)
    present = _present_files(root) - {PACK_FILE_NAME}
    if present != set(files):
        raise _fail(f"files: missing {sorted(set(files) - present)}, unexpected {sorted(present - set(files))}")
    for path, want in files.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != want:
            raise _fail(f"{path}: content does not match its pinned SHA-256")

    source = record["source"]
    generation: GenerationManifest | None = None
    if source == "suite":
        suite = load_suite(root / PACK_SUITE_DIR)
    elif source == "generated_suite":
        frozen = load_generated_suite(root / PACK_GENERATED_DIR)
        suite, generation = frozen.suite, frozen.generation
    else:
        raise _fail(f"source: unknown source kind {source!r}")

    manifest = EvaluationManifest.from_dict(json.loads((root / PACK_MANIFEST_NAME).read_text(encoding="utf-8")))
    verify_manifest(manifest, suite)
    if manifest.manifest_hash() != record["evaluation_manifest_hash"]:
        raise _fail("evaluation_manifest_hash does not match the evaluation manifest")
    episodes = _episodes(suite, manifest)
    expected = _pack_record(pack_id, source, suite, manifest, episodes, files)
    if canonical_json(expected) != canonical_json(record):
        for key in _PACK_KEYS:
            if canonical_json(expected[key]) != canonical_json(record[key]):
                raise _fail(f"{key}: does not match the environment (for 'episodes', a fire world differs)")
    return EnvironmentPack(root, pack_id, suite, manifest, generation, episodes, record, digest)


def verify_environment_pack(directory: str | Path, *, expected_digest: str | None = None) -> str:
    """Verify a pack from disk and return its digest. Raises :class:`EnvironmentPackError` on any difference."""
    return load_environment_pack(directory, expected_digest=expected_digest).digest
