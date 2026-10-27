"""Hidden-test integrity: asset classes, submission identity, provenance and fail-closed verification.

The simulator, the public suite and the generator are open, so a score is
generalization claim requires a separately maintained private distribution. Public development scores remain useful but are not contamination-resistant. This
module defines the evaluator-side protocol around that suite. It does not
change the simulator, the observation, the metrics, the statistics or the
generator, and it never gives an agent anything other than an
:class:`~emberbench.observation.Observation`.

* :class:`AssetClass` and :data:`ASSET_POLICIES` say which material is public,
  which stays with the evaluator and which may be published afterwards.
* :class:`SubmissionIdentity` pins a submitted model by explicit, optional
  digests. A missing field is ``None``, never a guess.
* :class:`EvaluationProvenance` pins one evaluation: versions, the frozen
  suite, the evaluation manifest hash, the submission, the evaluator code and
  the result digest. It holds no generator seed, no evaluation seed and no
  timestamp, so its SHA-256 digest is a pure function of its content.
* :class:`ProvenanceRecord` adds optional signatures made by an external
  process. A signature is recorded, never trusted, until a caller-supplied
  verifier accepts it. Unsigned provenance says ``"unsigned"``.
* :func:`verify_evaluation_provenance` re-derives every pin from the actual
  artifacts and raises :class:`~emberbench.errors.IntegrityError` on any
  difference. It never repairs anything.
* :class:`SubmissionRoster` and :func:`post_submission_generator_seed` bind
  the generator seed to frozen submissions (post-submission generation).
* :func:`workspace_stage` and :func:`verify_evaluation_workspace` check the
  evaluator-side directory layout, which lives outside the repository.

A hash establishes content identity and integrity only. It does not prove who
made an artifact, when, or which model produced the actions. See
``docs/integrity.md``.
"""

import base64
import binascii
import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

from . import SIMULATOR_VERSION
from .agent import ACTION_SCHEMA_VERSION
from .calibration import ADMISSION_RULES_VERSION, CALIBRATION_PANEL_VERSION, DIFFICULTY_RULES_VERSION
from .accounting import ACCOUNTING_VERSION
from .canonical import canonical_bytes, canonical_json, sha256_hex
from .commitment import COMMITMENT_VERSION
from .composition import COMPOSITION_VERSION, RECIPE_CATALOG_VERSION, SELECTION_RULES_VERSION, SUITE_FREEZE_RECORD_VERSION
from .crowd import CROWD_DIAGNOSTICS_VERSION
from .environment_pack import ENVIRONMENT_PACK_VERSION
from .errors import EmberBenchError, IntegrityError
from .evaluation import (
    EVALUATION_MANIFEST_VERSION,
    EVALUATION_RESULT_VERSION,
    EvaluationManifest,
    EvaluationResult,
    check_model_id,
    load_evaluation_manifest,
    load_evaluation_result,
    verify_manifest,
)
from .families import FAMILY_RULES_VERSION
from .generator import (
    GENERATION_MANIFEST_VERSION,
    GENERATOR_CONFIG_VERSION,
    GENERATOR_VERSION,
    FrozenGeneratedSuite,
    GeneratedSuite,
    GeneratorConfig,
    generate_suite,
    load_generated_suite,
    validate_generator_seed,
    verify_regeneration,
)
from .harness import ADAPTER_VERSION, HARNESS_VERSION
from .metrics import METRICS_VERSION
from .observation import OBSERVATION_VERSION
from .scenario import ID_PATTERN, SCHEMA_VERSION
from .scoring import SCORING_VERSION
from .statistics import STATISTICS_VERSION
from .statistics import _authenticate as _authenticate_result
from .suite import HASH_PATTERN, Visibility
from .worlds import ENVIRONMENT_WORLD_VERSION

INTEGRITY_PROTOCOL_VERSION = "1.0"

# The benchmark revision. Bump it for any change listed under "Benchmark
# revision policy" in docs/integrity.md. A new held-out batch under the same
# rules is not a new revision. Revision 2 is the one pre-release step that
# takes in the official scoring contract (scoring 1.0, environment packs) and
# the generator 1.1 suite-definition contracts (families, calibration,
# admission, composition). Revision 3 adds the evaluator protocol for remote
# models: the harness 1.0 joint protocol and the r1 prompt design
# (docs/harness-protocol.md), the run accounting and the suite commitment. No simulator, scenario or scoring rule
# changed, so revision-2 offline baseline results are identical under revision 3.
BENCHMARK_REVISION = "3"

IMAGE_DIGEST_PREFIX = "sha256:"
MAX_KEY_ID_LENGTH = 256
MAX_ALGORITHM_LENGTH = 64
# Values shorter than this are not scanned for by :func:`find_leaks`; they occur by chance in ordinary text.
MIN_SENSITIVE_TOKEN_LENGTH = 8

# The only fields an evaluated agent receives. Integrity metadata is never added here.
MODEL_VISIBLE_OBSERVATION_FIELDS = frozenset({"step", "map", "rules", "people", "burning", "allowed_actions"})


def _fail(where: str, message: str) -> IntegrityError:
    return IntegrityError(f"{where}: {message}")


def benchmark_versions() -> dict[str, str]:
    """Every version that defines what a score means. Provenance pins all of them.

    The values are the authoritative constants of each owning module; no
    version string is repeated here. :func:`verify_evaluation_provenance`
    fails closed when any recorded value differs from these.
    """
    return {
        "benchmark_revision": BENCHMARK_REVISION,
        "simulator": SIMULATOR_VERSION,
        "observation": OBSERVATION_VERSION,
        "action_schema": ACTION_SCHEMA_VERSION,
        "scenario_schema": SCHEMA_VERSION,
        "metrics": METRICS_VERSION,
        "crowd_diagnostics": CROWD_DIAGNOSTICS_VERSION,
        "statistics": STATISTICS_VERSION,
        "generator": GENERATOR_VERSION,
        "generator_config": GENERATOR_CONFIG_VERSION,
        "generation_manifest": GENERATION_MANIFEST_VERSION,
        "evaluation_manifest": EVALUATION_MANIFEST_VERSION,
        "evaluation_result": EVALUATION_RESULT_VERSION,
        "scoring": SCORING_VERSION,
        "environment_pack": ENVIRONMENT_PACK_VERSION,
        "environment_world": ENVIRONMENT_WORLD_VERSION,
        # Public development / archived research contract identities; not a future private policy.
        "family_rules": FAMILY_RULES_VERSION,
        "calibration_panel": CALIBRATION_PANEL_VERSION,
        "difficulty_rules": DIFFICULTY_RULES_VERSION,
        "admission_rules": ADMISSION_RULES_VERSION,
        "composition": COMPOSITION_VERSION,
        "selection_rules": SELECTION_RULES_VERSION,
        "recipe_catalog": RECIPE_CATALOG_VERSION,
        "suite_freeze_record": SUITE_FREEZE_RECORD_VERSION,
        # The evaluator protocol for remote models (revision 3).
        "harness": HARNESS_VERSION,
        "adapter": ADAPTER_VERSION,
        "accounting": ACCOUNTING_VERSION,
        "commitment": COMMITMENT_VERSION,
    }


# --------------------------------------------------------------------------- asset classes


class AssetClass(StrEnum):
    PUBLIC_DEVELOPMENT = "public_development"
    PRIVATE_HELDOUT_SUITE = "private_heldout_suite"
    PRIVATE_EVALUATION_MANIFEST = "private_evaluation_manifest"
    PRIVATE_SUBMISSION_METADATA = "private_submission_metadata"
    PUBLIC_PROVENANCE = "public_provenance"


class Release(StrEnum):
    """When an asset class may become public."""

    ALWAYS = "always"
    AFTER_EVALUATION = "after_evaluation"
    OPTIONAL_AUDIT_REVEAL = "optional_audit_reveal"
    NEVER = "never"


@dataclass(frozen=True)
class AssetPolicy:
    asset_class: AssetClass
    contents: str
    release: Release
    # May be committed to the repository or shipped in the package.
    in_repository: bool
    # May be given to an evaluated agent. Even public suites reach a scored agent only as observations.
    model_visible: bool


ASSET_POLICIES: Mapping[AssetClass, AssetPolicy] = {
    p.asset_class: p
    for p in (
        AssetPolicy(
            AssetClass.PUBLIC_DEVELOPMENT,
            "Simulator, generator code, public configs, the public V1 suite, docs, tests and test-only fixtures.",
            Release.ALWAYS,
            in_repository=True,
            model_visible=True,
        ),
        AssetPolicy(
            AssetClass.PRIVATE_HELDOUT_SUITE,
            "Generator seed, rejection history, frozen generated suite, its batch ID and its directory.",
            Release.OPTIONAL_AUDIT_REVEAL,
            in_repository=False,
            model_visible=False,
        ),
        AssetPolicy(
            AssetClass.PRIVATE_EVALUATION_MANIFEST,
            "Evaluation manifest and its seeds, and full per-model result artifacts (they embed the manifest).",
            Release.OPTIONAL_AUDIT_REVEAL,
            in_repository=False,
            model_visible=False,
        ),
        AssetPolicy(
            AssetClass.PRIVATE_SUBMISSION_METADATA,
            "Submitted code, weights, endpoints, credentials and contact data. Only identity digests are published.",
            Release.NEVER,
            in_repository=False,
            model_visible=False,
        ),
        AssetPolicy(
            AssetClass.PUBLIC_PROVENANCE,
            "Provenance records, submission rosters, aggregate and paired statistics.",
            Release.AFTER_EVALUATION,
            in_repository=False,
            model_visible=False,
        ),
    )
}


# --------------------------------------------------------------------------- digests


def digest_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def digest_file(path: str | Path) -> str:
    return digest_bytes(Path(path).read_bytes())


def digest_directory(path: str | Path) -> str:
    """SHA-256 over every regular file below ``path``: sorted POSIX relative paths and file digests.

    Symbolic links are rejected, because their target can change without
    changing the tree.
    """
    root = Path(path)
    if not root.is_dir():
        raise _fail(str(root), "not a directory")
    entries: list[list[str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise _fail(str(p), "symbolic links are not digested")
        if p.is_file():
            entries.append([p.relative_to(root).as_posix(), digest_file(p)])
    return sha256_hex(sorted(entries))


def evaluator_code_digest() -> str:
    """SHA-256 of the installed ``emberbench`` source files (``*.py`` and ``py.typed``).

    It identifies the evaluator code that ran. It is only as trustworthy as
    the environment that computed it.
    """
    package = Path(__file__).resolve().parent
    entries = [[p.name, digest_file(p)] for p in sorted(package.iterdir()) if p.is_file() and (p.suffix == ".py" or p.name == "py.typed")]
    return sha256_hex(entries)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None


def _expect_hash(value: object, where: str) -> str:
    if not _is_hash(value):
        raise _fail(where, "must be 64 lowercase hex characters")
    assert isinstance(value, str)
    return value


def _expect_optional_hash(value: object, where: str) -> str | None:
    return None if value is None else _expect_hash(value, where)


def _expect_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise _fail(where, f"expected a string, got {type(value).__name__}")
    return value


def _expect_count(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(where, f"expected a non-negative integer, got {value!r}")
    return value


def _expect_object(value: object, keys: Iterable[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    expected = set(keys)
    missing = sorted(expected - value.keys())
    unknown = sorted(set(value) - expected)
    if missing:
        raise _fail(where, f"missing fields {missing}")
    if unknown:
        raise _fail(where, f"unknown fields {unknown}")
    return value


# --------------------------------------------------------------------------- submission identity


@dataclass(frozen=True)
class SubmissionIdentity:
    """Explicit, immutable metadata that pins a submitted model.

    ``model_id`` is the label used in the evaluation result. Every other field
    is optional and ``None`` when not supplied; nothing is inferred. These
    values identify content. They do not prove who built it or that it is
    what ran, unless an external signature or trusted build vouches for them.
    """

    model_id: str
    code_commit: str | None = None  # git commit SHA (40 or 64 hex)
    image_digest: str | None = None  # container image digest, "sha256:<64 hex>"
    adapter_config_digest: str | None = None  # SHA-256 of the adapter or model config
    dependency_lock_digest: str | None = None  # SHA-256 of the dependency lock file

    def __post_init__(self) -> None:
        try:
            check_model_id(self.model_id)
        except EmberBenchError as exc:
            raise _fail("submission.model_id", str(exc)) from None
        if self.code_commit is not None:
            text = _expect_str(self.code_commit, "submission.code_commit")
            if len(text) not in (40, 64) or any(c not in "0123456789abcdef" for c in text):
                raise _fail("submission.code_commit", "must be a 40 or 64 character lowercase hex commit SHA")
        if self.image_digest is not None:
            text = _expect_str(self.image_digest, "submission.image_digest")
            if not text.startswith(IMAGE_DIGEST_PREFIX) or not _is_hash(text[len(IMAGE_DIGEST_PREFIX) :]):
                raise _fail("submission.image_digest", "must be 'sha256:' followed by 64 lowercase hex characters")
        _expect_optional_hash(self.adapter_config_digest, "submission.adapter_config_digest")
        _expect_optional_hash(self.dependency_lock_digest, "submission.dependency_lock_digest")

    @property
    def pinned_fields(self) -> tuple[str, ...]:
        """The optional content fields that are supplied, in a fixed order."""
        return tuple(f for f in _IDENTITY_OPTIONAL_FIELDS if getattr(self, f) is not None)

    @property
    def is_pinned(self) -> bool:
        """True if at least one content digest is supplied. A label alone pins nothing."""
        return bool(self.pinned_fields)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, **{f: getattr(self, f) for f in _IDENTITY_OPTIONAL_FIELDS}}

    def identity_digest(self) -> str:
        """SHA-256 of the canonical identity. Any changed or added field changes it."""
        return sha256_hex(self.to_dict())

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "SubmissionIdentity":
        obj = _expect_object(data, ("model_id", *_IDENTITY_OPTIONAL_FIELDS), "submission")
        return cls(**obj)


_IDENTITY_OPTIONAL_FIELDS = ("code_commit", "image_digest", "adapter_config_digest", "dependency_lock_digest")


@dataclass(frozen=True)
class SubmissionRoster:
    """The submissions frozen before a post-submission held-out batch is generated.

    Sorted by ``model_id``; model IDs are unique and every identity must be
    pinned, because a bare label freezes nothing.
    """

    submissions: tuple[SubmissionIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.submissions, tuple) or not self.submissions:
            raise _fail("roster", "needs at least one submission, as a tuple")
        ids = [s.model_id for s in self.submissions if isinstance(s, SubmissionIdentity)]
        if len(ids) != len(self.submissions):
            raise _fail("roster", "every entry must be a SubmissionIdentity")
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise _fail("roster", "model IDs must be unique and sorted")
        for s in self.submissions:
            if not s.is_pinned:
                raise _fail("roster", f"submission {s.model_id!r} pins no code or model digest")

    @classmethod
    def of(cls, submissions: Iterable[SubmissionIdentity]) -> "SubmissionRoster":
        return cls(tuple(sorted(submissions, key=lambda s: s.model_id)))

    def contains(self, identity: SubmissionIdentity) -> bool:
        return identity in self.submissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "integrity_protocol_version": INTEGRITY_PROTOCOL_VERSION,
            "submissions": [s.to_dict() for s in self.submissions],
        }

    def roster_digest(self) -> str:
        return sha256_hex(self.to_dict())

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "SubmissionRoster":
        obj = _expect_object(data, ("integrity_protocol_version", "submissions"), "roster")
        if obj["integrity_protocol_version"] != INTEGRITY_PROTOCOL_VERSION:
            raise _fail("roster.integrity_protocol_version", f"unsupported version {obj['integrity_protocol_version']!r}")
        if not isinstance(obj["submissions"], list):
            raise _fail("roster.submissions", "expected a list")
        return cls(tuple(SubmissionIdentity.from_dict(s) for s in obj["submissions"]))


# --------------------------------------------------------------------------- post-submission generation

_POST_SUBMISSION_DOMAIN = "emberbench/integrity/post-submission-generator-seed"


def post_submission_generator_seed(private_seed: int, roster: SubmissionRoster) -> int:
    """The generator seed of a post-submission batch: SHA-256 of the private seed and the roster digest.

    The suite then depends on the frozen roster, so no scenario of the batch
    can exist before every listed submission is frozen. The private seed must
    be chosen after the roster is frozen and kept private until the batch is
    retired. Only the roster digest goes into provenance.
    """
    seed = validate_generator_seed(private_seed)
    if not isinstance(roster, SubmissionRoster):
        raise TypeError("roster must be a SubmissionRoster")
    material = canonical_bytes(
        {
            "domain": _POST_SUBMISSION_DOMAIN,
            "integrity_protocol_version": INTEGRITY_PROTOCOL_VERSION,
            "private_seed": format(seed, "064x"),
            "roster_digest": roster.roster_digest(),
        }
    )
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def generate_post_submission_suite(config: GeneratorConfig, private_seed: int, roster: SubmissionRoster, *, batch_id: str) -> GeneratedSuite:
    """Generate a roster-bound RESEARCH holdout from public development rules.

    This utility is not the official evaluator. A private seed does not make the
    public generator distribution secret. Freeze it before research evaluation.
    """
    return generate_suite(config, post_submission_generator_seed(private_seed, roster), batch_id=batch_id)


def verify_post_submission_generation(directory: str | Path, private_seed: int, roster: SubmissionRoster) -> FrozenGeneratedSuite:
    """Audit after reveal: the frozen suite is exactly what ``private_seed`` and ``roster`` generate."""
    return verify_regeneration(directory, post_submission_generator_seed(private_seed, roster))


# --------------------------------------------------------------------------- provenance


class ProtocolMode(StrEnum):
    # One held-out batch, generated and frozen in advance, kept private and shared by every submission evaluated on it.
    SHARED_HELDOUT_BATCH = "shared_heldout_batch"
    # The generator seed is bound to a frozen roster of submissions (strongest mode).
    POST_SUBMISSION_GENERATION = "post_submission_generation"


@dataclass(frozen=True)
class SuitePin:
    suite_id: str
    suite_version: str
    suite_hash: str
    generation_hash: str
    generator_version: str
    generator_config_hash: str
    scenario_count: int

    @classmethod
    def of(cls, frozen: FrozenGeneratedSuite) -> "SuitePin":
        g = frozen.generation
        return cls(g.suite_id, g.suite_version, g.suite_hash, g.generation_hash(), g.generator_version, g.config_hash, len(g.entries))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ManifestPin:
    manifest_hash: str
    episode_count: int

    @classmethod
    def of(cls, manifest: EvaluationManifest) -> "ManifestPin":
        return cls(manifest.manifest_hash(), len(manifest.episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ResultPin:
    result_digest: str
    result_version: str
    episode_count: int

    @classmethod
    def of(cls, result: EvaluationResult) -> "ResultPin":
        # Equal to the result digest that aggregate and paired statistics report.
        return cls(sha256_hex(result.to_dict()), result.result_version, len(result.episodes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EvaluationProvenance:
    """The immutable, public record of one model's evaluation on one frozen held-out suite.

    It holds hashes and versions only: no generator seed, no evaluation seed,
    no scenario content and no timestamp. :meth:`provenance_digest` is the
    SHA-256 of :meth:`signing_payload`, the canonical JSON of :meth:`to_dict`.
    """

    integrity_protocol_version: str
    versions: Mapping[str, str]
    mode: ProtocolMode
    roster_digest: str | None
    suite: SuitePin
    evaluation_manifest: ManifestPin
    submission: SubmissionIdentity
    evaluator_code_digest: str | None
    result: ResultPin

    def to_dict(self) -> dict[str, Any]:
        return {
            "integrity_protocol_version": self.integrity_protocol_version,
            "versions": dict(self.versions),
            "protocol": {"mode": str(self.mode), "submission_roster_digest": self.roster_digest},
            "suite": self.suite.to_dict(),
            "evaluation_manifest": self.evaluation_manifest.to_dict(),
            "submission": {"identity": self.submission.to_dict(), "identity_digest": self.submission.identity_digest()},
            "evaluator": {"code_digest": self.evaluator_code_digest},
            "result": self.result.to_dict(),
        }

    def signing_payload(self) -> bytes:
        """The exact bytes an external signer signs."""
        return canonical_bytes(self.to_dict())

    def provenance_digest(self) -> str:
        return digest_bytes(self.signing_payload())

    @classmethod
    def from_dict(cls, data: Any) -> "EvaluationProvenance":
        return _parse_provenance(data)


_PROVENANCE_KEYS = ("integrity_protocol_version", "versions", "protocol", "suite", "evaluation_manifest", "submission", "evaluator", "result")


def _parse_provenance(data: Any) -> EvaluationProvenance:
    obj = _expect_object(data, _PROVENANCE_KEYS, "provenance")
    if obj["integrity_protocol_version"] != INTEGRITY_PROTOCOL_VERSION:
        raise _fail("provenance.integrity_protocol_version", f"unsupported version {obj['integrity_protocol_version']!r}")
    versions = _expect_object(obj["versions"], benchmark_versions(), "provenance.versions")
    for name, value in versions.items():
        _expect_str(value, f"provenance.versions.{name}")

    protocol = _expect_object(obj["protocol"], ("mode", "submission_roster_digest"), "provenance.protocol")
    try:
        mode = ProtocolMode(_expect_str(protocol["mode"], "provenance.protocol.mode"))
    except ValueError:
        raise _fail("provenance.protocol.mode", f"unknown mode {protocol['mode']!r}") from None
    roster_digest = _expect_optional_hash(protocol["submission_roster_digest"], "provenance.protocol.submission_roster_digest")
    if (mode is ProtocolMode.POST_SUBMISSION_GENERATION) != (roster_digest is not None):
        raise _fail("provenance.protocol", "a roster digest is required exactly in post-submission mode")

    s = _expect_object(obj["suite"], SuitePin.__dataclass_fields__, "provenance.suite")
    suite = SuitePin(
        suite_id=_expect_str(s["suite_id"], "provenance.suite.suite_id"),
        suite_version=_expect_str(s["suite_version"], "provenance.suite.suite_version"),
        suite_hash=_expect_hash(s["suite_hash"], "provenance.suite.suite_hash"),
        generation_hash=_expect_hash(s["generation_hash"], "provenance.suite.generation_hash"),
        generator_version=_expect_str(s["generator_version"], "provenance.suite.generator_version"),
        generator_config_hash=_expect_hash(s["generator_config_hash"], "provenance.suite.generator_config_hash"),
        scenario_count=_expect_count(s["scenario_count"], "provenance.suite.scenario_count"),
    )
    if not ID_PATTERN.fullmatch(suite.suite_id):
        raise _fail("provenance.suite.suite_id", f"must match {ID_PATTERN.pattern}")

    m = _expect_object(obj["evaluation_manifest"], ManifestPin.__dataclass_fields__, "provenance.evaluation_manifest")
    manifest = ManifestPin(
        manifest_hash=_expect_hash(m["manifest_hash"], "provenance.evaluation_manifest.manifest_hash"),
        episode_count=_expect_count(m["episode_count"], "provenance.evaluation_manifest.episode_count"),
    )

    sub = _expect_object(obj["submission"], ("identity", "identity_digest"), "provenance.submission")
    identity = SubmissionIdentity.from_dict(sub["identity"])
    if _expect_hash(sub["identity_digest"], "provenance.submission.identity_digest") != identity.identity_digest():
        raise _fail("provenance.submission.identity_digest", "does not match the identity")

    ev = _expect_object(obj["evaluator"], ("code_digest",), "provenance.evaluator")
    evaluator_digest = _expect_optional_hash(ev["code_digest"], "provenance.evaluator.code_digest")

    r = _expect_object(obj["result"], ResultPin.__dataclass_fields__, "provenance.result")
    result = ResultPin(
        result_digest=_expect_hash(r["result_digest"], "provenance.result.result_digest"),
        result_version=_expect_str(r["result_version"], "provenance.result.result_version"),
        episode_count=_expect_count(r["episode_count"], "provenance.result.episode_count"),
    )
    return EvaluationProvenance(
        integrity_protocol_version=INTEGRITY_PROTOCOL_VERSION,
        versions=dict(versions),
        mode=mode,
        roster_digest=roster_digest,
        suite=suite,
        evaluation_manifest=manifest,
        submission=identity,
        evaluator_code_digest=evaluator_digest,
        result=result,
    )


# --------------------------------------------------------------------------- signatures


class SigningState(StrEnum):
    """What a stored record claims. It never claims that a signature is valid."""

    UNSIGNED = "unsigned"
    EXTERNALLY_SIGNED = "externally_signed"


class SignatureStatus(StrEnum):
    """The outcome of verification."""

    UNSIGNED = "unsigned"
    SIGNED_UNVERIFIED = "signed_unverified"  # signatures present, no verifier supplied
    SIGNED_VERIFIED = "signed_verified"  # every signature accepted by the supplied verifier


@dataclass(frozen=True)
class SignatureRecord:
    """A signature made by an external process over :meth:`EvaluationProvenance.signing_payload`.

    EMBERBench stores the algorithm label, key ID and signature bytes. It does
    not implement or check any signature scheme; see :data:`SignatureVerifier`.
    """

    algorithm: str
    key_id: str
    signature: bytes

    def __post_init__(self) -> None:
        algorithm = _expect_str(self.algorithm, "signature.algorithm")
        if not 0 < len(algorithm) <= MAX_ALGORITHM_LENGTH or not all(c.isalnum() or c in "._:+-" for c in algorithm) or not algorithm.isascii():
            raise _fail("signature.algorithm", f"must be 1-{MAX_ALGORITHM_LENGTH} ASCII letters, digits or '._:+-'")
        key_id = _expect_str(self.key_id, "signature.key_id")
        if not 0 < len(key_id) <= MAX_KEY_ID_LENGTH or not key_id.isprintable() or key_id != key_id.strip():
            raise _fail("signature.key_id", f"must be 1-{MAX_KEY_ID_LENGTH} printable characters with no outer whitespace")
        if not isinstance(self.signature, bytes) or not self.signature:
            raise _fail("signature.signature", "must be non-empty bytes")

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "key_id": self.key_id, "signature_base64": base64.b64encode(self.signature).decode("ascii")}

    @classmethod
    def from_dict(cls, data: Any) -> "SignatureRecord":
        obj = _expect_object(data, ("algorithm", "key_id", "signature_base64"), "signature")
        text = _expect_str(obj["signature_base64"], "signature.signature_base64")
        try:
            raw = base64.b64decode(text.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError):
            raise _fail("signature.signature_base64", "is not valid base64") from None
        if base64.b64encode(raw).decode("ascii") != text:
            raise _fail("signature.signature_base64", "is not canonical base64")
        return cls(obj["algorithm"], obj["key_id"], raw)


# Called with one signature and the exact signed bytes. Return True only if the signature is valid for a trusted key.
SignatureVerifier = Callable[[SignatureRecord, bytes], bool]


@dataclass(frozen=True)
class ProvenanceRecord:
    """The stored provenance file: the provenance, its digest and zero or more external signatures."""

    provenance: EvaluationProvenance
    signatures: tuple[SignatureRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, EvaluationProvenance):
            raise TypeError("provenance must be an EvaluationProvenance")
        if not isinstance(self.signatures, tuple) or not all(isinstance(s, SignatureRecord) for s in self.signatures):
            raise TypeError("signatures must be a tuple of SignatureRecord")
        keys = [(s.algorithm, s.key_id) for s in self.signatures]
        if len(set(keys)) != len(keys):
            raise _fail("signatures", "two signatures with the same algorithm and key ID")

    @property
    def signing(self) -> SigningState:
        return SigningState.EXTERNALLY_SIGNED if self.signatures else SigningState.UNSIGNED

    def with_signature(self, signature: SignatureRecord) -> "ProvenanceRecord":
        """A new record with ``signature`` added. The provenance and its digest do not change."""
        return ProvenanceRecord(self.provenance, (*self.signatures, signature))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "provenance_digest": self.provenance.provenance_digest(),
            "signing": str(self.signing),
            "signatures": [s.to_dict() for s in self.signatures],
        }

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "ProvenanceRecord":
        obj = _expect_object(data, ("provenance", "provenance_digest", "signing", "signatures"), "record")
        provenance = _parse_provenance(obj["provenance"])
        if _expect_hash(obj["provenance_digest"], "record.provenance_digest") != provenance.provenance_digest():
            raise _fail("record.provenance_digest", "does not match the provenance")
        if not isinstance(obj["signatures"], list):
            raise _fail("record.signatures", "expected a list")
        record = cls(provenance, tuple(SignatureRecord.from_dict(s) for s in obj["signatures"]))
        if obj["signing"] != str(record.signing):
            raise _fail("record.signing", f"is {obj['signing']!r}, but the record is {str(record.signing)!r}")
        # Anything the typed form does not reproduce exactly (an unknown nested field, a non-canonical value) is rejected.
        if canonical_json(record.to_dict()) != canonical_json(data):
            raise _fail("record", "does not round-trip exactly")
        return record


def save_provenance(record: ProvenanceRecord, path: str | Path) -> None:
    """Write ``record`` once. An existing file is never overwritten."""
    _write_once(Path(path), record.to_file_text())


def load_provenance(path: str | Path) -> ProvenanceRecord:
    return ProvenanceRecord.from_dict(_read_json(Path(path)))


def save_roster(roster: SubmissionRoster, path: str | Path) -> None:
    _write_once(Path(path), roster.to_file_text())


def load_roster(path: str | Path) -> SubmissionRoster:
    return SubmissionRoster.from_dict(_read_json(Path(path)))


def _write_once(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="ascii", newline="\n") as f:
            f.write(text)
    except FileExistsError:
        raise _fail(str(path), "already exists; frozen integrity artifacts are never overwritten") from None


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _fail(str(path), "file not found") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fail(str(path), f"invalid JSON: {exc}") from None


# --------------------------------------------------------------------------- build and verify


def _check_inputs(frozen: object, manifest: object, result: object) -> None:
    if not isinstance(frozen, FrozenGeneratedSuite):
        raise TypeError("frozen_suite must be a FrozenGeneratedSuite (from load_generated_suite)")
    if not isinstance(manifest, EvaluationManifest):
        raise TypeError("manifest must be an EvaluationManifest")
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")


def _check_artifacts(frozen: FrozenGeneratedSuite, manifest: EvaluationManifest, result: EvaluationResult) -> None:
    """The frozen suite, the manifest and the result belong together and to this benchmark revision."""
    if frozen.suite.manifest.visibility != Visibility.GENERATED:
        raise _fail("suite", "held-out provenance needs a frozen generated suite, not a public suite")
    current = benchmark_versions()
    pinned = {
        "manifest.simulator_version": (manifest.simulator_version, current["simulator"]),
        "manifest.observation_version": (manifest.observation_version, current["observation"]),
        "manifest.metrics_version": (manifest.metrics_version, current["metrics"]),
        "suite.generator_version": (frozen.generation.generator_version, current["generator"]),
        "result.result_version": (result.result_version, current["evaluation_result"]),
    }
    for where, (actual, expected) in pinned.items():
        if actual != expected:
            raise _fail(where, f"is {actual!r}, this benchmark revision uses {expected!r}")
    try:
        verify_manifest(manifest, frozen.suite)
    except EmberBenchError as exc:
        raise _fail("evaluation_manifest", f"does not match the frozen suite: {exc}") from None
    if result.manifest_hash != manifest.manifest_hash():
        raise _fail("result", "was produced from a different evaluation manifest")
    if not result.is_complete:
        raise _fail("result", f"is incomplete: {len(result.episodes)} of {len(manifest.episodes)} episodes")
    try:
        _authenticate_result(result, frozen.suite)
    except EmberBenchError as exc:
        raise _fail("result", f"failed authentication: {exc}") from None


def _check_roster(mode: ProtocolMode, submission: SubmissionIdentity, roster: SubmissionRoster | None, roster_digest: str | None) -> None:
    if mode is ProtocolMode.SHARED_HELDOUT_BATCH:
        if roster is not None:
            raise _fail("roster", "supplied, but the provenance is not in post-submission mode")
        return
    if roster is None:
        raise _fail("roster", "post-submission provenance needs the frozen submission roster")
    if not isinstance(roster, SubmissionRoster):
        raise TypeError("roster must be a SubmissionRoster")
    if roster.roster_digest() != roster_digest:
        raise _fail("roster", "digest does not match the provenance")
    if not roster.contains(submission):
        raise _fail("roster", f"does not contain the submission {submission.model_id!r} with this exact identity")


def build_evaluation_provenance(
    frozen_suite: FrozenGeneratedSuite,
    manifest: EvaluationManifest,
    result: EvaluationResult,
    submission: SubmissionIdentity,
    *,
    roster: SubmissionRoster | None = None,
    evaluator_code_digest: str | None = None,
) -> EvaluationProvenance:
    """Pin one complete, authenticated evaluation. Build it before any statistic is computed.

    With ``roster`` the provenance is in post-submission mode and the
    submission must be in the roster. ``evaluator_code_digest`` is ``None``
    unless the caller supplies it (for example :func:`evaluator_code_digest`).
    """
    _check_inputs(frozen_suite, manifest, result)
    if not isinstance(submission, SubmissionIdentity):
        raise TypeError("submission must be a SubmissionIdentity")
    _check_artifacts(frozen_suite, manifest, result)
    if result.model_id != submission.model_id:
        raise _fail("submission.model_id", f"is {submission.model_id!r}, the result is for {result.model_id!r}")
    mode = ProtocolMode.SHARED_HELDOUT_BATCH if roster is None else ProtocolMode.POST_SUBMISSION_GENERATION
    roster_digest = None if roster is None else roster.roster_digest()
    _check_roster(mode, submission, roster, roster_digest)
    return EvaluationProvenance(
        integrity_protocol_version=INTEGRITY_PROTOCOL_VERSION,
        versions=benchmark_versions(),
        mode=mode,
        roster_digest=roster_digest,
        suite=SuitePin.of(frozen_suite),
        evaluation_manifest=ManifestPin.of(manifest),
        submission=submission,
        evaluator_code_digest=_expect_optional_hash(evaluator_code_digest, "evaluator_code_digest"),
        result=ResultPin.of(result),
    )


@dataclass(frozen=True)
class ProvenanceVerification:
    """What :func:`verify_evaluation_provenance` established. Anything else raised."""

    provenance_digest: str
    mode: ProtocolMode
    signature_status: SignatureStatus
    verified_key_ids: tuple[str, ...]
    submission_pinned_fields: tuple[str, ...]
    evaluator_code_pinned: bool


def _first_difference(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str | None:
    for key in sorted(set(expected) | set(actual)):
        if canonical_json(expected.get(key)) != canonical_json(actual.get(key)):
            return key
    return None


def verify_evaluation_provenance(
    record: ProvenanceRecord | Mapping[str, Any],
    *,
    frozen_suite: FrozenGeneratedSuite,
    manifest: EvaluationManifest,
    result: EvaluationResult,
    expected_submission: SubmissionIdentity | None = None,
    roster: SubmissionRoster | None = None,
    expected_evaluator_code_digest: str | None = None,
    signature_verifier: SignatureVerifier | None = None,
    require_signatures: bool = False,
) -> ProvenanceVerification:
    """Check a provenance record against the actual frozen suite, manifest and result. Fail closed.

    Every pin is re-derived from the artifacts and compared. The result is
    re-authenticated by replay against the suite. Raises
    :class:`~emberbench.errors.IntegrityError` on a wrong suite, manifest or
    result digest, a version of another benchmark revision, a submission or
    evaluator digest that differs from the expected one, a malformed or
    unknown field, or a signature the verifier rejects. Nothing is repaired.

    Signatures are never trusted on their own. Without ``signature_verifier``
    a signed record is reported as ``signed_unverified``; with
    ``require_signatures`` that is an error, as is an unsigned record.
    """
    # An in-memory record is re-parsed too, so it gets exactly the checks a loaded file gets.
    record = ProvenanceRecord.from_dict(record.to_dict() if isinstance(record, ProvenanceRecord) else record)
    _check_inputs(frozen_suite, manifest, result)
    p = record.provenance

    if p.integrity_protocol_version != INTEGRITY_PROTOCOL_VERSION:
        raise _fail("provenance.integrity_protocol_version", f"unsupported version {p.integrity_protocol_version!r}")
    diff = _first_difference(benchmark_versions(), p.versions)
    if diff is not None:
        raise _fail(f"provenance.versions.{diff}", "differs from this benchmark revision")
    _check_artifacts(frozen_suite, manifest, result)

    pins: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "suite": (SuitePin.of(frozen_suite).to_dict(), p.suite.to_dict()),
        "evaluation_manifest": (ManifestPin.of(manifest).to_dict(), p.evaluation_manifest.to_dict()),
        "result": (ResultPin.of(result).to_dict(), p.result.to_dict()),
    }
    for name, (expected, actual) in pins.items():
        diff = _first_difference(expected, actual)
        if diff is not None:
            raise _fail(f"provenance.{name}.{diff}", "does not match the actual artifact")
    if result.model_id != p.submission.model_id:
        raise _fail("provenance.submission.model_id", f"is {p.submission.model_id!r}, the result is for {result.model_id!r}")
    if expected_submission is not None:
        if not isinstance(expected_submission, SubmissionIdentity):
            raise TypeError("expected_submission must be a SubmissionIdentity")
        diff = _first_difference(expected_submission.to_dict(), p.submission.to_dict())
        if diff is not None:
            raise _fail(f"provenance.submission.{diff}", "does not match the pinned submission")
    _check_roster(p.mode, p.submission, roster, p.roster_digest)
    if expected_evaluator_code_digest is not None:
        _expect_hash(expected_evaluator_code_digest, "expected_evaluator_code_digest")
        if p.evaluator_code_digest is None:
            raise _fail("provenance.evaluator.code_digest", "is not recorded, but an evaluator digest is pinned")
        if p.evaluator_code_digest != expected_evaluator_code_digest:
            raise _fail("provenance.evaluator.code_digest", "does not match the pinned evaluator digest")

    status, verified = _check_signatures(record, signature_verifier, require_signatures)
    return ProvenanceVerification(
        provenance_digest=p.provenance_digest(),
        mode=p.mode,
        signature_status=status,
        verified_key_ids=verified,
        submission_pinned_fields=p.submission.pinned_fields,
        evaluator_code_pinned=p.evaluator_code_digest is not None,
    )


def _check_signatures(record: ProvenanceRecord, verifier: SignatureVerifier | None, required: bool) -> tuple[SignatureStatus, tuple[str, ...]]:
    if not record.signatures:
        if required:
            raise _fail("signatures", "the record is unsigned, but signatures are required")
        return SignatureStatus.UNSIGNED, ()
    if verifier is None:
        if required:
            raise _fail("signatures", "signatures are required, but no verifier was supplied; they are not trusted")
        return SignatureStatus.SIGNED_UNVERIFIED, ()
    payload = record.provenance.signing_payload()
    for s in record.signatures:
        if verifier(s, payload) is not True:
            raise _fail("signatures", f"{s.algorithm} signature by key {s.key_id!r} was rejected by the verifier")
    return SignatureStatus.SIGNED_VERIFIED, tuple(s.key_id for s in record.signatures)


# --------------------------------------------------------------------------- evaluator workspace

SUBMISSION_DIR = "submission"
SUBMISSION_IDENTITY_NAME = "identity.json"
SUBMISSION_ROSTER_NAME = "submission-roster.json"
GENERATED_SUITE_DIR = "generated-suite"
EVALUATION_MANIFEST_NAME = "evaluation-manifest.json"
RESULT_NAME = "result.json"
PROVENANCE_NAME = "provenance.json"


class WorkspaceStage(IntEnum):
    """Lifecycle stages of an evaluation workspace. Each stage needs every earlier one."""

    EMPTY = 0
    SUBMISSION_FROZEN = 1  # submission/identity.json
    SUITE_FROZEN = 2  # generated-suite/
    MANIFEST_FROZEN = 3  # evaluation-manifest.json
    EVALUATED = 4  # result.json
    PROVENANCE_RECORDED = 5  # provenance.json


_STAGE_ENTRIES: tuple[tuple[WorkspaceStage, str], ...] = (
    (WorkspaceStage.SUBMISSION_FROZEN, f"{SUBMISSION_DIR}/{SUBMISSION_IDENTITY_NAME}"),
    (WorkspaceStage.SUITE_FROZEN, GENERATED_SUITE_DIR),
    (WorkspaceStage.MANIFEST_FROZEN, EVALUATION_MANIFEST_NAME),
    (WorkspaceStage.EVALUATED, RESULT_NAME),
    (WorkspaceStage.PROVENANCE_RECORDED, PROVENANCE_NAME),
)
_WORKSPACE_ENTRIES = frozenset({SUBMISSION_DIR, SUBMISSION_ROSTER_NAME, GENERATED_SUITE_DIR, EVALUATION_MANIFEST_NAME, RESULT_NAME, PROVENANCE_NAME})


def _inside_source_checkout(path: Path) -> Path | None:
    """The EMBERBench source checkout that contains ``path``, if any."""
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "emberbench" / "__init__.py").is_file():
            return parent
    return None


def workspace_stage(root: str | Path) -> WorkspaceStage:
    """Validate an evaluator workspace layout and return the lifecycle stage it has reached.

    Raises :class:`~emberbench.errors.IntegrityError` if the workspace is
    inside an EMBERBench source checkout, has unexpected entries, or has a
    later artifact without an earlier one (for example a result without a
    frozen manifest).
    """
    path = Path(root).resolve()
    if not path.is_dir():
        raise _fail(str(path), "workspace is not a directory")
    checkout = _inside_source_checkout(path)
    if checkout is not None:
        raise _fail(str(path), f"workspace is inside the EMBERBench source checkout {checkout}; keep private material outside it")
    unexpected = sorted(p.name for p in path.iterdir() if p.name not in _WORKSPACE_ENTRIES)
    if unexpected:
        raise _fail(str(path), f"unexpected workspace entries {unexpected}")
    present = [(stage, (path / name).exists()) for stage, name in _STAGE_ENTRIES]
    reached = WorkspaceStage.EMPTY
    for stage, exists in present:
        if exists and stage != reached + 1:
            raise _fail(str(path), f"{stage.name.lower()} artifact present before {WorkspaceStage(reached + 1).name.lower()}")
        if exists:
            reached = stage
    if (path / SUBMISSION_DIR).exists() and reached < WorkspaceStage.SUBMISSION_FROZEN:
        raise _fail(str(path), f"{SUBMISSION_DIR}/ has no {SUBMISSION_IDENTITY_NAME}")
    return reached


def freeze_submission(identity: SubmissionIdentity, root: str | Path) -> Path:
    """First lifecycle step: write ``submission/identity.json`` once into a new or empty workspace."""
    if not isinstance(identity, SubmissionIdentity):
        raise TypeError("identity must be a SubmissionIdentity")
    path = Path(root)
    checkout = _inside_source_checkout(path.resolve())
    if checkout is not None:
        raise _fail(str(path), f"workspace is inside the EMBERBench source checkout {checkout}; keep private material outside it")
    path.mkdir(parents=True, exist_ok=True)
    if workspace_stage(path) != WorkspaceStage.EMPTY:
        raise _fail(str(path), "the submission is already frozen")
    (path / SUBMISSION_DIR).mkdir(exist_ok=True)
    target = path / SUBMISSION_DIR / SUBMISSION_IDENTITY_NAME
    _write_once(target, identity.to_file_text())
    return target


def verify_evaluation_workspace(
    root: str | Path,
    *,
    expected_submission: SubmissionIdentity | None = None,
    expected_evaluator_code_digest: str | None = None,
    signature_verifier: SignatureVerifier | None = None,
    require_signatures: bool = False,
) -> ProvenanceVerification:
    """Load every artifact of a complete workspace from disk and verify its provenance.

    The suite is re-verified with :func:`~emberbench.generator.load_generated_suite`,
    the result is authenticated by replay, and the provenance must pin the
    identity in ``submission/identity.json`` exactly.
    """
    path = Path(root)
    if workspace_stage(path) != WorkspaceStage.PROVENANCE_RECORDED:
        raise _fail(str(path), "workspace is not complete")
    identity = SubmissionIdentity.from_dict(_read_json(path / SUBMISSION_DIR / SUBMISSION_IDENTITY_NAME))
    if expected_submission is not None and expected_submission != identity:
        raise _fail(f"{SUBMISSION_DIR}/{SUBMISSION_IDENTITY_NAME}", "does not match the expected submission")
    roster_path = path / SUBMISSION_ROSTER_NAME
    roster = load_roster(roster_path) if roster_path.exists() else None
    try:
        frozen = load_generated_suite(path / GENERATED_SUITE_DIR)
        manifest = load_evaluation_manifest(path / EVALUATION_MANIFEST_NAME)
        result = load_evaluation_result(path / RESULT_NAME, frozen.suite)
    except IntegrityError:
        raise
    except EmberBenchError as exc:
        raise _fail(str(path), f"artifact failed verification: {exc}") from None
    return verify_evaluation_provenance(
        load_provenance(path / PROVENANCE_NAME),
        frozen_suite=frozen,
        manifest=manifest,
        result=result,
        expected_submission=identity,
        roster=roster,
        expected_evaluator_code_digest=expected_evaluator_code_digest,
        signature_verifier=signature_verifier,
        require_signatures=require_signatures,
    )


# --------------------------------------------------------------------------- contamination scanning


def sensitive_tokens(
    frozen_suite: FrozenGeneratedSuite,
    manifest: EvaluationManifest | None = None,
    *,
    generator_seed: int | None = None,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """Strings that must never appear in anything given to an evaluated agent.

    They cover the batch ID, suite and generation hashes, the config hash,
    scenario IDs and content hashes, the workspace path, the manifest hash and
    seeds, and the generator seed if supplied. Values shorter than
    :data:`MIN_SENSITIVE_TOKEN_LENGTH` are left out, so low-entropy seeds
    cannot be scanned for; use high-entropy seeds.
    """
    g = frozen_suite.generation
    values: list[str] = [g.batch_id, g.suite_hash, g.generation_hash(), g.config_hash, str(frozen_suite.root), frozen_suite.root.name]
    for entry in g.entries:
        values += [entry.scenario_id, entry.content_hash]
    seeds: list[int] = [] if generator_seed is None else [generator_seed]
    if manifest is not None:
        values.append(manifest.manifest_hash())
        seeds += [e.seed for e in manifest.episodes]
    for seed in seeds:
        values += [str(seed), format(seed, "x")]
    values += list(extra)
    return frozenset(v for v in values if len(v) >= MIN_SENSITIVE_TOKEN_LENGTH)


def find_leaks(text: str, tokens: Iterable[str]) -> tuple[str, ...]:
    """The sensitive tokens that occur in ``text`` (case-insensitive), sorted."""
    lowered = text.lower()
    return tuple(sorted(t for t in set(tokens) if t.lower() in lowered))
