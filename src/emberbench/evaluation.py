"""Paired identical evaluation: evaluation manifests, a resumable runner and exact pairing.

An :class:`EvaluationManifest` pins an ordered list of episodes. Each episode
is one scenario (by ID and content hash) and one evaluator seed. Every model
evaluated under the same manifest sees the same scenarios with the same
seeds, in the same order, so the stochastic fire is identical for all of
them (common random numbers).

The evaluator owns the manifest and its seeds. A decision policy receives only
:class:`~emberbench.observation.Observation` values. It never receives the
manifest, an episode entry or index, a seed, a scenario, a run record or any
evaluator metadata. ``run_evaluation`` builds a fresh policy for every episode
from a zero-argument factory, so one episode cannot leak into the next.

This module does not generate seeds, call model providers or compute
aggregate statistics. See ``docs/evaluation.md``.
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import SIMULATOR_VERSION
from .canonical import canonical_json, sha256_hex
from .errors import EmberBenchError, EvaluationError
from .events import RunRecord
from .metrics import METRICS_VERSION, EpisodeMetrics, compute_episode_metrics
from .observation import OBSERVATION_VERSION
from .rng import MAX_SEED
from .scenario import ID_PATTERN, Scenario
from .simulator import Policy, run_episode
from .suite import HASH_PATTERN, SUITE_VERSION_PATTERN, Suite

EVALUATION_MANIFEST_VERSION = "1.0"
EVALUATION_RESULT_VERSION = "1.0"
MAX_MODEL_ID_LENGTH = 256

# Called once per episode with no arguments. It must return a new policy.
PolicyFactory = Callable[[], Policy]

_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "suite_id",
        "suite_version",
        "suite_hash",
        "simulator_version",
        "observation_version",
        "metrics_version",
        "episodes",
    }
)
_EPISODE_KEYS = frozenset({"index", "scenario_id", "scenario_hash", "seed"})
_RESULT_KEYS = frozenset(
    {
        "result_version",
        "model_id",
        "evaluation_manifest_hash",
        "suite_id",
        "suite_version",
        "suite_hash",
        "simulator_version",
        "observation_version",
        "metrics_version",
        "manifest",
        "episodes",
    }
)
_EPISODE_RESULT_KEYS = _EPISODE_KEYS | {"record_digest", "record", "metrics"}


# --------------------------------------------------------------------------- strict field checks


def _fail(where: str, message: str) -> EvaluationError:
    return EvaluationError(f"{where}: {message}")


def _expect_object(value: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    missing = sorted(keys - value.keys())
    if missing:
        raise _fail(where, f"missing required keys {missing}")
    unknown = sorted(value.keys() - keys)
    if unknown:
        raise _fail(where, f"unknown keys {unknown}")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "expected a non-empty string")
    return value


def _expect_index(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(where, f"expected a non-negative integer, got {value!r}")
    return value


def _expect_seed(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(where, f"seed must be an integer, got {type(value).__name__}")
    if not 0 <= value <= MAX_SEED:
        raise _fail(where, f"seed must be in [0, 2**64 - 1], got {value}")
    return value


def _expect_hash(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HASH_PATTERN.match(value):
        raise _fail(where, "must be 64 lowercase hex characters")
    return value


def _expect_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(where, f"expected a list, got {type(value).__name__}")
    return value


def check_model_id(model_id: object) -> str:
    """A model ID is a free-text label. It is never proof of identity or uniqueness."""
    if not isinstance(model_id, str) or not model_id:
        raise EvaluationError("model_id must be a non-empty string")
    if len(model_id) > MAX_MODEL_ID_LENGTH:
        raise EvaluationError(f"model_id must have at most {MAX_MODEL_ID_LENGTH} characters")
    if model_id != model_id.strip() or not model_id.isprintable():
        raise EvaluationError("model_id must be printable, with no leading or trailing whitespace")
    return model_id


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class EpisodeSpec:
    """One scheduled episode: a pinned scenario and an evaluator seed."""

    index: int
    scenario_id: str
    scenario_hash: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
        }

    def key(self, manifest_hash: str) -> "EpisodeKey":
        return EpisodeKey(manifest_hash, self.index, self.scenario_id, self.scenario_hash, self.seed)


@dataclass(frozen=True, order=True)
class EpisodeKey:
    """The exact identity of a scheduled episode. Two results are comparable only on equal keys."""

    manifest_hash: str
    index: int
    scenario_id: str
    scenario_hash: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "index": self.index,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class EvaluationManifest:
    """An immutable, ordered episode schedule pinned to one suite and one set of versions.

    ``manifest_hash()`` covers every field, so a change to any seed, scenario,
    version or to the episode order gives a different hash.
    """

    manifest_version: str
    suite_id: str
    suite_version: str
    suite_hash: str
    simulator_version: str
    observation_version: str
    metrics_version: str
    episodes: tuple[EpisodeSpec, ...]

    def __post_init__(self) -> None:
        _validate_manifest(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "suite_hash": self.suite_hash,
            "simulator_version": self.simulator_version,
            "observation_version": self.observation_version,
            "metrics_version": self.metrics_version,
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def manifest_hash(self) -> str:
        """SHA-256 of the canonical manifest."""
        return sha256_hex(self.to_dict())

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "EvaluationManifest":
        obj = _expect_object(data, _MANIFEST_KEYS, "manifest")
        episodes = []
        for i, raw in enumerate(_expect_list(obj["episodes"], "manifest.episodes")):
            where = f"manifest.episodes[{i}]"
            e = _expect_object(raw, _EPISODE_KEYS, where)
            episodes.append(
                EpisodeSpec(
                    index=_expect_index(e["index"], f"{where}.index"),
                    scenario_id=_expect_str(e["scenario_id"], f"{where}.scenario_id"),
                    scenario_hash=_expect_hash(e["scenario_hash"], f"{where}.scenario_hash"),
                    seed=_expect_seed(e["seed"], f"{where}.seed"),
                )
            )
        return cls(
            manifest_version=_expect_str(obj["manifest_version"], "manifest.manifest_version"),
            suite_id=_expect_str(obj["suite_id"], "manifest.suite_id"),
            suite_version=_expect_str(obj["suite_version"], "manifest.suite_version"),
            suite_hash=_expect_str(obj["suite_hash"], "manifest.suite_hash"),
            simulator_version=_expect_str(obj["simulator_version"], "manifest.simulator_version"),
            observation_version=_expect_str(obj["observation_version"], "manifest.observation_version"),
            metrics_version=_expect_str(obj["metrics_version"], "manifest.metrics_version"),
            episodes=tuple(episodes),
        )


def _validate_manifest(m: EvaluationManifest) -> None:
    if m.manifest_version != EVALUATION_MANIFEST_VERSION:
        raise _fail("manifest_version", f"unsupported version {m.manifest_version!r}")
    if not isinstance(m.suite_id, str) or not ID_PATTERN.match(m.suite_id):
        raise _fail("suite_id", f"must match {ID_PATTERN.pattern}, got {m.suite_id!r}")
    if not isinstance(m.suite_version, str) or not SUITE_VERSION_PATTERN.match(m.suite_version):
        raise _fail("suite_version", f"must be MAJOR.MINOR.PATCH, got {m.suite_version!r}")
    _expect_hash(m.suite_hash, "suite_hash")
    for name in ("simulator_version", "observation_version", "metrics_version"):
        _expect_str(getattr(m, name), name)
    if not isinstance(m.episodes, tuple) or not m.episodes:
        raise _fail("episodes", "expected a non-empty tuple")
    hash_of: dict[str, str] = {}
    seen: set[tuple[str, int]] = set()
    for position, e in enumerate(m.episodes):
        where = f"episodes[{position}]"
        if not isinstance(e, EpisodeSpec):
            raise _fail(where, "expected an EpisodeSpec")
        if _expect_index(e.index, f"{where}.index") != position:
            raise _fail(f"{where}.index", f"must equal its position {position}, got {e.index}")
        if not isinstance(e.scenario_id, str) or not ID_PATTERN.match(e.scenario_id):
            raise _fail(f"{where}.scenario_id", f"must match {ID_PATTERN.pattern}, got {e.scenario_id!r}")
        _expect_hash(e.scenario_hash, f"{where}.scenario_hash")
        _expect_seed(e.seed, f"{where}.seed")
        if hash_of.setdefault(e.scenario_id, e.scenario_hash) != e.scenario_hash:
            raise _fail(where, f"scenario {e.scenario_id!r} is pinned to two different content hashes")
        if (e.scenario_id, e.seed) in seen:
            raise _fail(where, f"seed {e.seed} is scheduled twice for scenario {e.scenario_id!r}")
        seen.add((e.scenario_id, e.seed))


def build_evaluation_manifest(suite: Suite, seeds: Sequence[int]) -> EvaluationManifest:
    """Schedule every suite scenario under every seed: suite manifest order, then ``seeds`` order.

    ``seeds`` are evaluator-owned. This function never generates, sorts or
    removes seeds; a duplicate or invalid seed is rejected.
    """
    if not isinstance(suite, Suite):
        raise TypeError("suite must be a Suite")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be an ordered sequence of integers (a list or tuple)")
    seed_list = [_expect_seed(s, f"seeds[{i}]") for i, s in enumerate(seeds)]
    if not seed_list:
        raise EvaluationError("seeds: at least one seed is required")
    duplicates = sorted({s for s in seed_list if seed_list.count(s) > 1})
    if duplicates:
        raise EvaluationError(f"seeds: duplicate seeds {duplicates}")
    episodes: list[EpisodeSpec] = []
    for entry in suite.manifest.entries:
        if suite.get(entry.scenario_id).content_hash() != entry.content_hash:
            raise EvaluationError(f"{entry.scenario_id}: scenario does not match its suite entry")
        for seed in seed_list:
            episodes.append(EpisodeSpec(len(episodes), entry.scenario_id, entry.content_hash, seed))
    return EvaluationManifest(
        manifest_version=EVALUATION_MANIFEST_VERSION,
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.suite_version,
        suite_hash=suite.manifest.suite_hash(),
        simulator_version=SIMULATOR_VERSION,
        observation_version=OBSERVATION_VERSION,
        metrics_version=METRICS_VERSION,
        episodes=tuple(episodes),
    )


def verify_manifest(manifest: EvaluationManifest, suite: Suite) -> None:
    """Require that ``manifest`` targets exactly ``suite`` and the running code's versions."""
    if not isinstance(manifest, EvaluationManifest):
        raise TypeError("manifest must be an EvaluationManifest")
    if not isinstance(suite, Suite):
        raise TypeError("suite must be a Suite")
    pinned = {
        "suite_id": suite.manifest.suite_id,
        "suite_version": suite.manifest.suite_version,
        "suite_hash": suite.manifest.suite_hash(),
        "simulator_version": SIMULATOR_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "metrics_version": METRICS_VERSION,
    }
    for name, expected in pinned.items():
        actual = getattr(manifest, name)
        if actual != expected:
            raise _fail(f"manifest.{name}", f"is {actual!r}, expected {expected!r}")
    for spec in manifest.episodes:
        resolve_scenario(suite, spec)


def resolve_scenario(suite: Suite, spec: EpisodeSpec) -> Scenario:
    """The suite scenario that ``spec`` pins. Raises unless both the entry and the file match its hash."""
    where = f"episode {spec.index}"
    try:
        entry = suite.entry(spec.scenario_id)
        scenario = suite.get(spec.scenario_id)
    except KeyError:
        raise _fail(where, f"scenario {spec.scenario_id!r} is not in suite {suite.manifest.suite_id!r}") from None
    if entry.content_hash != spec.scenario_hash or scenario.content_hash() != spec.scenario_hash:
        raise _fail(where, f"scenario {spec.scenario_id!r} does not match the pinned content hash")
    return scenario


def save_evaluation_manifest(manifest: EvaluationManifest, path: str | Path) -> None:
    _atomic_write_text(Path(path), manifest.to_file_text())


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    return EvaluationManifest.from_dict(_read_json(Path(path)))


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class EpisodeResult:
    """One completed episode: its manifest entry, the full run record and its raw metrics."""

    episode: EpisodeSpec
    record: RunRecord
    metrics: EpisodeMetrics

    def key(self, manifest_hash: str) -> EpisodeKey:
        return self.episode.key(manifest_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.episode.to_dict(),
            "record_digest": self.record.digest(),
            "record": self.record.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """The per-model artifact: completed episodes of one manifest, in manifest order.

    ``model_id`` is a label chosen by the evaluator. Nothing here proves which
    model produced the actions. Construction checks structure and consistency
    with the manifest. :func:`load_evaluation_result` also replays every
    record, so a loaded result is authenticated.
    """

    result_version: str
    model_id: str
    manifest: EvaluationManifest
    episodes: tuple[EpisodeResult, ...]

    def __post_init__(self) -> None:
        _validate_result(self)

    @property
    def manifest_hash(self) -> str:
        return self.manifest.manifest_hash()

    @property
    def completed_indices(self) -> frozenset[int]:
        return frozenset(e.episode.index for e in self.episodes)

    @property
    def is_complete(self) -> bool:
        return len(self.episodes) == len(self.manifest.episodes)

    def to_dict(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "result_version": self.result_version,
            "model_id": self.model_id,
            "evaluation_manifest_hash": m.manifest_hash(),
            "suite_id": m.suite_id,
            "suite_version": m.suite_version,
            "suite_hash": m.suite_hash,
            "simulator_version": m.simulator_version,
            "observation_version": m.observation_version,
            "metrics_version": m.metrics_version,
            "manifest": m.to_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def to_file_text(self) -> str:
        return canonical_json(self.to_dict()) + "\n"


def _validate_result(r: EvaluationResult) -> None:
    if r.result_version != EVALUATION_RESULT_VERSION:
        raise _fail("result_version", f"unsupported version {r.result_version!r}")
    check_model_id(r.model_id)
    if not isinstance(r.manifest, EvaluationManifest):
        raise _fail("manifest", "expected an EvaluationManifest")
    if not isinstance(r.episodes, tuple):
        raise _fail("episodes", "expected a tuple")
    m = r.manifest
    previous = -1
    for position, e in enumerate(r.episodes):
        where = f"episodes[{position}]"
        if not isinstance(e, EpisodeResult):
            raise _fail(where, "expected an EpisodeResult")
        spec = e.episode
        if not 0 <= spec.index < len(m.episodes):
            raise _fail(where, f"index {spec.index} is not in the manifest (it has {len(m.episodes)} episodes)")
        if spec.index == previous:
            raise _fail(where, f"duplicate result for episode {spec.index}")
        if spec.index < previous:
            raise _fail(where, "results must be in manifest order")
        previous = spec.index
        if spec != m.episodes[spec.index]:
            raise _fail(where, f"does not match manifest episode {spec.index}")
        rec = e.record
        header = {
            "scenario_id": (rec.scenario_id, spec.scenario_id),
            "scenario_hash": (rec.scenario_hash, spec.scenario_hash),
            "seed": (rec.seed, spec.seed),
            "simulator_version": (rec.simulator_version, m.simulator_version),
            "observation_version": (rec.observation_version, m.observation_version),
        }
        for name, (actual, expected) in header.items():
            if actual != expected:
                raise _fail(where, f"record {name} is {actual!r}, manifest gives {expected!r}")


def _parse_result(data: Any, suite: Suite) -> EvaluationResult:
    obj = _expect_object(data, _RESULT_KEYS, "result")
    manifest = EvaluationManifest.from_dict(obj["manifest"])
    if obj["evaluation_manifest_hash"] != manifest.manifest_hash():
        raise _fail("result.evaluation_manifest_hash", "does not match the embedded manifest")
    for name in ("suite_id", "suite_version", "suite_hash", "simulator_version", "observation_version", "metrics_version"):
        if obj[name] != getattr(manifest, name):
            raise _fail(f"result.{name}", f"is {obj[name]!r}, the embedded manifest gives {getattr(manifest, name)!r}")
    verify_manifest(manifest, suite)

    episodes = []
    seen: set[int] = set()
    for position, raw in enumerate(_expect_list(obj["episodes"], "result.episodes")):
        where = f"result.episodes[{position}]"
        e = _expect_object(raw, _EPISODE_RESULT_KEYS, where)
        index = _expect_index(e["index"], f"{where}.index")
        if index >= len(manifest.episodes):
            raise _fail(where, f"unknown episode {index}: the manifest has {len(manifest.episodes)} episodes")
        if index in seen:
            raise _fail(where, f"duplicate result for episode {index}")
        seen.add(index)
        spec = manifest.episodes[index]
        stored = {k: e[k] for k in _EPISODE_KEYS}
        if canonical_json(stored) != canonical_json(spec.to_dict()):
            raise _fail(where, f"does not match manifest episode {index}")
        try:
            record = RunRecord.from_dict(e["record"])
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, EmberBenchError) as exc:
            raise _fail(where, f"malformed run record: {exc!r}") from exc
        if canonical_json(record.to_dict()) != canonical_json(e["record"]):
            raise _fail(where, "run record is not in canonical form")
        if e["record_digest"] != record.digest():
            raise _fail(where, "record_digest does not match the run record")
        if (record.scenario_id, record.scenario_hash, record.seed) != (spec.scenario_id, spec.scenario_hash, spec.seed):
            raise _fail(where, "run record does not belong to its manifest episode")
        scenario = resolve_scenario(suite, spec)
        try:
            metrics = compute_episode_metrics(scenario, record)
        except EmberBenchError as exc:
            raise _fail(where, f"run record failed authentication: {exc}") from exc
        if canonical_json(metrics.to_dict()) != canonical_json(e["metrics"]):
            raise _fail(where, "stored metrics differ from the metrics of the authenticated record")
        episodes.append(EpisodeResult(spec, record, metrics))

    result = EvaluationResult(
        result_version=_expect_str(obj["result_version"], "result.result_version"),
        model_id=obj["model_id"],
        manifest=manifest,
        episodes=tuple(episodes),
    )
    if canonical_json(result.to_dict()) != canonical_json(data):
        raise _fail("result", "artifact is not in canonical form")
    return result


def load_evaluation_result(path: str | Path, suite: Suite) -> EvaluationResult:
    """Load a per-model artifact and authenticate every stored run record by replay.

    Raises :class:`EvaluationError` on any malformed, inconsistent, duplicate,
    unknown or tampered entry. Nothing is repaired or skipped.
    """
    return _parse_result(_read_json(Path(path)), suite)


def save_evaluation_result(result: EvaluationResult, path: str | Path) -> None:
    _atomic_write_text(Path(path), result.to_file_text())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvaluationError(f"{path}: file not found") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"{path}: not valid JSON: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file in the same directory, then rename it over ``path``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- runner


def run_evaluation(
    suite: Suite,
    manifest: EvaluationManifest,
    model_id: str,
    policy_factory: PolicyFactory,
    output_path: str | Path,
    *,
    resume: bool = True,
) -> EvaluationResult:
    """Run every episode of ``manifest`` in order and save the artifact after each one.

    For each episode the scenario is resolved and checked against its pinned
    hash, a new policy is built with ``policy_factory()``, and the episode
    runs with the manifest seed (the scenario file's development seed is not
    used). The run record is then authenticated by replay and scored.

    If ``output_path`` exists, it is resumed only when ``resume`` is true:
    the artifact must load and authenticate completely and carry the same
    manifest hash and model ID. Completed episodes are not run again. With
    ``resume=False`` an existing file is an error, never overwritten.
    """
    verify_manifest(manifest, suite)
    check_model_id(model_id)
    if not callable(policy_factory):
        raise TypeError("policy_factory must be a zero-argument callable that returns a policy")
    path = Path(output_path)
    if path.exists():
        if not resume:
            raise EvaluationError(f"{path}: already exists; pass resume=True to continue it")
        result = load_evaluation_result(path, suite)
        if result.manifest_hash != manifest.manifest_hash():
            raise EvaluationError(f"{path}: artifact belongs to evaluation manifest {result.manifest_hash}, not {manifest.manifest_hash()}")
        if result.model_id != model_id:
            raise EvaluationError(f"{path}: artifact is for model {result.model_id!r}, not {model_id!r}")
    else:
        result = EvaluationResult(EVALUATION_RESULT_VERSION, model_id, manifest, ())
        save_evaluation_result(result, path)

    completed = {e.episode.index: e for e in result.episodes}
    for spec in manifest.episodes:
        if spec.index in completed:
            continue
        scenario = resolve_scenario(suite, spec)
        policy = policy_factory()
        if not callable(policy):
            raise EvaluationError(f"episode {spec.index}: policy_factory did not return a callable policy")
        # The policy sees only Observation values. The seed stays inside the simulator.
        record = run_episode(scenario, policy, seed=spec.seed)
        metrics = compute_episode_metrics(scenario, record)
        completed[spec.index] = EpisodeResult(spec, record, metrics)
        result = EvaluationResult(
            EVALUATION_RESULT_VERSION, model_id, manifest, tuple(completed[i] for i in sorted(completed))
        )
        save_evaluation_result(result, path)
    return result


# --------------------------------------------------------------------------- pairing


@dataclass(frozen=True)
class PairingReport:
    """Which scheduled episodes both models completed. It holds no scores or statistics."""

    manifest_hash: str
    model_a: str
    model_b: str
    total_scheduled: int
    paired: tuple[EpisodeKey, ...]
    missing_from_a: tuple[EpisodeKey, ...]
    missing_from_b: tuple[EpisodeKey, ...]

    @property
    def paired_count(self) -> int:
        return len(self.paired)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "total_scheduled": self.total_scheduled,
            "paired_count": self.paired_count,
            "paired": [k.to_dict() for k in self.paired],
            "missing_from_a": [k.to_dict() for k in self.missing_from_a],
            "missing_from_b": [k.to_dict() for k in self.missing_from_b],
        }


def pair_results(a: EvaluationResult, b: EvaluationResult) -> PairingReport:
    """Pair two per-model artifacts episode by episode.

    Two episode results form a comparable pair only when they have the same
    evaluation manifest hash, episode index, scenario ID, scenario hash and
    seed. Artifacts from different manifests are rejected, not compared.
    """
    if not isinstance(a, EvaluationResult) or not isinstance(b, EvaluationResult):
        raise TypeError("pair_results needs two EvaluationResult values")
    manifest_hash = a.manifest_hash
    if b.manifest_hash != manifest_hash or a.manifest != b.manifest:
        raise EvaluationError(
            f"incompatible evaluation manifests: {a.model_id!r} uses {manifest_hash}, {b.model_id!r} uses {b.manifest_hash}"
        )
    scheduled = [spec.key(manifest_hash) for spec in a.manifest.episodes]
    keys_a = {e.key(manifest_hash) for e in a.episodes}
    keys_b = {e.key(manifest_hash) for e in b.episodes}
    unknown = (keys_a | keys_b) - set(scheduled)
    if unknown:
        raise EvaluationError(f"results contain episodes that the manifest does not schedule: {sorted(unknown)}")
    return PairingReport(
        manifest_hash=manifest_hash,
        model_a=a.model_id,
        model_b=b.model_id,
        total_scheduled=len(scheduled),
        paired=tuple(k for k in scheduled if k in keys_a and k in keys_b),
        missing_from_a=tuple(k for k in scheduled if k not in keys_a),
        missing_from_b=tuple(k for k in scheduled if k not in keys_b),
    )
