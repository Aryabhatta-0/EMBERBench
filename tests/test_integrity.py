"""Hidden-test integrity: provenance, submission identity, signatures, contamination and post-submission generation."""

import copy
import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import greedy_exit_policy
from emberbench import (
    ACTION_SCHEMA_VERSION,
    CROWD_DIAGNOSTICS_VERSION,
    EVALUATION_MANIFEST_VERSION,
    EVALUATION_RESULT_VERSION,
    GENERATION_MANIFEST_VERSION,
    GENERATOR_CONFIG_VERSION,
    GENERATOR_VERSION,
    METRICS_VERSION,
    OBSERVATION_VERSION,
    SCHEMA_VERSION,
    SIMULATOR_VERSION,
    STATISTICS_VERSION,
    Action,
    Agent,
    EvaluationManifest,
    EvaluationResult,
    Observation,
    aggregate_result,
    build_evaluation_manifest,
    load_evaluation_manifest,
    load_evaluation_result,
    load_suite,
    run_evaluation,
    save_evaluation_manifest,
)
from emberbench.canonical import canonical_bytes, canonical_json, sha256_hex
from emberbench.errors import GenerationError, IntegrityError
from emberbench.generator import FrozenGeneratedSuite, GeneratorConfig, freeze_generated_suite, generate_suite, load_generated_suite
from emberbench.integrity import (
    ASSET_POLICIES,
    BENCHMARK_REVISION,
    EVALUATION_MANIFEST_NAME,
    GENERATED_SUITE_DIR,
    INTEGRITY_PROTOCOL_VERSION,
    MODEL_VISIBLE_OBSERVATION_FIELDS,
    PROVENANCE_NAME,
    RESULT_NAME,
    SUBMISSION_ROSTER_NAME,
    AssetClass,
    EvaluationProvenance,
    ProtocolMode,
    ProvenanceRecord,
    Release,
    SignatureRecord,
    SignatureStatus,
    SigningState,
    SubmissionIdentity,
    SubmissionRoster,
    WorkspaceStage,
    benchmark_versions,
    build_evaluation_provenance,
    digest_bytes,
    digest_directory,
    evaluator_code_digest,
    find_leaks,
    freeze_submission,
    generate_post_submission_suite,
    load_provenance,
    post_submission_generator_seed,
    save_provenance,
    save_roster,
    sensitive_tokens,
    verify_evaluation_provenance,
    verify_evaluation_workspace,
    verify_post_submission_generation,
    workspace_stage,
)
from emberbench.simulator import Policy
from test_generator import BASE_CONFIG, BATCH, GOLDEN_CONFIG_HASH, GOLDEN_GENERATION_HASH, GOLDEN_SUITE_HASH, OTHER_SEED, SEED
from test_suite import REPO_ROOT, SUITE_DIR, SUITE_HASH

# Test-only values. Real generator, private and scored evaluation seeds are evaluator-private and never in the repository.
EVAL_SEEDS = (0x7E57_0000_0000_0001, 0x7E57_0000_0000_0002)
PRIVATE_TEST_SEED = 0x7E57_5EED_0000_0000_0000_0000_0000_0001
MODEL = "greedy-test-model"


def _test_digest(label: str) -> str:
    return hashlib.sha256(f"test-only {label}".encode()).hexdigest()


def identity(model_id: str = MODEL, **changes: Any) -> SubmissionIdentity:
    base = SubmissionIdentity(
        model_id=model_id,
        code_commit=hashlib.sha1(f"test-only commit {model_id}".encode()).hexdigest(),
        image_digest=None,
        adapter_config_digest=_test_digest(f"adapter {model_id}"),
        dependency_lock_digest=None,
    )
    return dataclasses.replace(base, **changes)


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def factory_of(policy: Policy) -> Callable[[], Policy]:
    return lambda: policy


@dataclasses.dataclass(frozen=True)
class Run:
    frozen: FrozenGeneratedSuite
    manifest: EvaluationManifest
    result: EvaluationResult
    other_result: EvaluationResult  # same manifest, another policy
    submission: SubmissionIdentity
    record: ProvenanceRecord


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return GeneratorConfig.from_dict(BASE_CONFIG)


@pytest.fixture(scope="module")
def run(config: GeneratorConfig, tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("integrity")
    frozen = freeze_generated_suite(generate_suite(config, SEED, batch_id=BATCH), root / "frozen")
    manifest = build_evaluation_manifest(frozen.suite, EVAL_SEEDS)
    result = run_evaluation(frozen.suite, manifest, MODEL, factory_of(greedy_exit_policy), root / "greedy.json")
    other = run_evaluation(frozen.suite, manifest, MODEL, factory_of(stay_policy), root / "stay.json")
    submission = identity()
    provenance = build_evaluation_provenance(frozen, manifest, result, submission)
    return Run(frozen, manifest, result, other, submission, ProvenanceRecord(provenance))


def verify(run: Run, record: ProvenanceRecord | dict[str, Any], **kwargs: Any) -> Any:
    args: dict[str, Any] = {"frozen_suite": run.frozen, "manifest": run.manifest, "result": run.result}
    args.update(kwargs)
    return verify_evaluation_provenance(record, **args)


def resealed(data: dict[str, Any]) -> dict[str, Any]:
    """A record dict whose digests are recomputed after an edit, so only the pin checks can catch it."""
    out = copy.deepcopy(data)
    sub = out["provenance"]["submission"]
    sub["identity_digest"] = sha256_hex(sub["identity"])
    out["provenance_digest"] = sha256_hex(out["provenance"])
    return out


def edited(run: Run, edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data = run.record.to_dict()
    edit(data["provenance"])
    return resealed(data)


# --------------------------------------------------------------------------- versions and unchanged public behaviour


def test_versions_are_pinned_and_unchanged() -> None:
    assert INTEGRITY_PROTOCOL_VERSION == "1.0"
    assert BENCHMARK_REVISION == "1"
    unchanged = {
        "simulator": (SIMULATOR_VERSION, "0.2.0"),
        "observation": (OBSERVATION_VERSION, "1.0"),
        "action_schema": (ACTION_SCHEMA_VERSION, "1.0"),
        "scenario_schema": (SCHEMA_VERSION, "1.0"),
        "metrics": (METRICS_VERSION, "1.0"),
        "crowd_diagnostics": (CROWD_DIAGNOSTICS_VERSION, "1.0"),
        "statistics": (STATISTICS_VERSION, "1.0"),
        "generator": (GENERATOR_VERSION, "1.1.0"),
        "generator_config": (GENERATOR_CONFIG_VERSION, "1.1"),
        "generation_manifest": (GENERATION_MANIFEST_VERSION, "1.1"),
        "evaluation_manifest": (EVALUATION_MANIFEST_VERSION, "1.0"),
        "evaluation_result": (EVALUATION_RESULT_VERSION, "1.0"),
    }
    for name, (actual, expected) in unchanged.items():
        assert actual == expected, name
    assert benchmark_versions() == {"benchmark_revision": "1", **{k: v for k, (v, _) in unchanged.items()}}


def test_public_suite_and_generator_hashes_are_unchanged(run: Run) -> None:
    assert load_suite(SUITE_DIR).manifest.suite_hash() == SUITE_HASH
    g = run.frozen.generation
    assert (g.config_hash, g.generation_hash(), g.suite_hash) == (GOLDEN_CONFIG_HASH, GOLDEN_GENERATION_HASH, GOLDEN_SUITE_HASH)


def test_asset_policies_keep_private_material_out_of_the_repository() -> None:
    assert set(ASSET_POLICIES) == set(AssetClass)
    in_repo = {c for c, p in ASSET_POLICIES.items() if p.in_repository}
    assert in_repo == {AssetClass.PUBLIC_DEVELOPMENT}
    visible = {c for c, p in ASSET_POLICIES.items() if p.model_visible}
    assert visible == {AssetClass.PUBLIC_DEVELOPMENT}
    assert ASSET_POLICIES[AssetClass.PRIVATE_SUBMISSION_METADATA].release is Release.NEVER
    assert ASSET_POLICIES[AssetClass.PUBLIC_PROVENANCE].release is Release.AFTER_EVALUATION
    for c in (AssetClass.PRIVATE_HELDOUT_SUITE, AssetClass.PRIVATE_EVALUATION_MANIFEST):
        assert ASSET_POLICIES[c].release is Release.OPTIONAL_AUDIT_REVEAL


# --------------------------------------------------------------------------- canonical provenance


def test_provenance_digest_is_canonical(run: Run, tmp_path: Path) -> None:
    p = run.record.provenance
    assert p.provenance_digest() == hashlib.sha256(canonical_bytes(p.to_dict())).hexdigest()
    assert p.signing_payload() == canonical_bytes(p.to_dict())
    again = build_evaluation_provenance(run.frozen, run.manifest, run.result, identity())
    assert again == p and again.provenance_digest() == p.provenance_digest()
    # Key order and whitespace in the file do not matter; content does.
    data = run.record.to_dict()
    shuffled = json.loads(json.dumps(data, indent=4, sort_keys=False))
    shuffled["provenance"] = dict(reversed(list(shuffled["provenance"].items())))
    assert ProvenanceRecord.from_dict(shuffled).provenance.provenance_digest() == p.provenance_digest()
    save_provenance(run.record, tmp_path / PROVENANCE_NAME)
    assert load_provenance(tmp_path / PROVENANCE_NAME) == run.record
    assert EvaluationProvenance.from_dict(p.to_dict()) == p


def test_provenance_pins_everything_required(run: Run) -> None:
    d = run.record.provenance.to_dict()
    assert d["integrity_protocol_version"] == INTEGRITY_PROTOCOL_VERSION
    assert d["versions"] == benchmark_versions()
    g = run.frozen.generation
    assert d["suite"] == {
        "suite_id": BATCH,
        "suite_version": g.suite_version,
        "suite_hash": g.suite_hash,
        "generation_hash": g.generation_hash(),
        "generator_version": GENERATOR_VERSION,
        "generator_config_hash": g.config_hash,
        "scenario_count": 4,
    }
    assert d["evaluation_manifest"] == {"manifest_hash": run.manifest.manifest_hash(), "episode_count": 8}
    assert d["submission"]["identity"] == run.submission.to_dict()
    assert d["submission"]["identity_digest"] == run.submission.identity_digest()
    assert d["evaluator"] == {"code_digest": None}
    assert d["protocol"] == {"mode": "shared_heldout_batch", "submission_roster_digest": None}
    # The result digest is the one statistics report for the same artifact.
    assert d["result"]["result_digest"] == aggregate_result(run.result, run.frozen.suite, resamples=100).result_digest


def _flip_hex(text: str) -> str:
    return text[:-1] + format(int(text[-1], 16) ^ 1, "x")


def _flip_char(text: str) -> str:
    return text[:-1] + chr(ord(text[-1]) ^ 1)


@pytest.mark.parametrize(
    "path,flip",
    [
        (("submission", "identity", "model_id"), _flip_char),
        (("submission", "identity", "code_commit"), _flip_hex),
        (("submission", "identity", "adapter_config_digest"), _flip_hex),
        (("suite", "suite_hash"), _flip_hex),
        (("suite", "generation_hash"), _flip_hex),
        (("suite", "generator_config_hash"), _flip_hex),
        (("evaluation_manifest", "manifest_hash"), _flip_hex),
        (("result", "result_digest"), _flip_hex),
        (("versions", "simulator"), _flip_char),
        (("suite", "scenario_count"), lambda n: n ^ 1),
    ],
)
def test_one_bit_metadata_change_changes_digest(run: Run, path: tuple[str, ...], flip: Callable[[Any], Any]) -> None:
    data = run.record.provenance.to_dict()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = flip(node[path[-1]])
    if path[0] == "submission":
        data["submission"]["identity_digest"] = sha256_hex(data["submission"]["identity"])
    assert sha256_hex(data) != run.record.provenance.provenance_digest()
    assert EvaluationProvenance.from_dict(data).provenance_digest() != run.record.provenance.provenance_digest()


def test_identity_digest_changes_with_every_field() -> None:
    base = identity()
    digests = {base.identity_digest()}
    for changed in (
        dataclasses.replace(base, model_id=_flip_char(base.model_id)),
        dataclasses.replace(base, code_commit=_flip_hex(base.code_commit or "")),
        dataclasses.replace(base, image_digest="sha256:" + _test_digest("image")),
        dataclasses.replace(base, adapter_config_digest=None),
        dataclasses.replace(base, dependency_lock_digest=_test_digest("lock")),
    ):
        digests.add(changed.identity_digest())
    assert len(digests) == 6


# --------------------------------------------------------------------------- fail-closed verification


def test_valid_provenance_verifies(run: Run) -> None:
    v = verify(run, run.record, expected_submission=identity())
    assert v.provenance_digest == run.record.provenance.provenance_digest()
    assert v.mode is ProtocolMode.SHARED_HELDOUT_BATCH
    assert v.signature_status is SignatureStatus.UNSIGNED
    assert v.submission_pinned_fields == ("code_commit", "adapter_config_digest")
    assert not v.evaluator_code_pinned
    # A loaded dict verifies the same way as the typed record.
    assert verify(run, json.loads(run.record.to_file_text())) == v


def test_result_tampering_is_rejected(run: Run) -> None:
    # Another valid result of the same manifest and model label is not the pinned result.
    with pytest.raises(IntegrityError, match=r"provenance\.result\.result_digest"):
        verify(run, run.record, result=run.other_result)
    with pytest.raises(IntegrityError, match=r"provenance\.result\.result_digest"):
        verify(run, edited(run, lambda p: p["result"].update(result_digest=_flip_hex(p["result"]["result_digest"]))))
    # A record whose stored metrics were altered fails authentication by replay.
    first = run.result.episodes[0]
    safety = first.metrics.safety
    bad_metrics = dataclasses.replace(first.metrics, safety=dataclasses.replace(safety, evacuated=safety.evacuated + 1))
    forged = dataclasses.replace(run.result, episodes=(dataclasses.replace(first, metrics=bad_metrics), *run.result.episodes[1:]))
    with pytest.raises(IntegrityError, match="failed authentication"):
        verify(run, run.record, result=forged)


def test_suite_tampering_is_rejected(run: Run, config: GeneratorConfig, tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match=r"provenance\.suite\.suite_hash"):
        verify(run, edited(run, lambda p: p["suite"].update(suite_hash=_flip_hex(p["suite"]["suite_hash"]))))
    with pytest.raises(IntegrityError, match=r"provenance\.suite\.generation_hash"):
        verify(run, edited(run, lambda p: p["suite"].update(generation_hash=_flip_hex(p["suite"]["generation_hash"]))))
    # Another suite of the same batch ID (another seed) does not match the manifest or the provenance.
    other = freeze_generated_suite(generate_suite(config, OTHER_SEED, batch_id=BATCH), tmp_path / "other")
    with pytest.raises(IntegrityError, match="evaluation_manifest: does not match the frozen suite"):
        verify(run, run.record, frozen_suite=other)


def test_evaluation_manifest_tampering_is_rejected(run: Run) -> None:
    with pytest.raises(IntegrityError, match=r"provenance\.evaluation_manifest\.manifest_hash"):
        verify(run, edited(run, lambda p: p["evaluation_manifest"].update(manifest_hash=_flip_hex(p["evaluation_manifest"]["manifest_hash"]))))
    reseeded = build_evaluation_manifest(run.frozen.suite, (EVAL_SEEDS[0], EVAL_SEEDS[1] + 1))
    with pytest.raises(IntegrityError, match="different evaluation manifest"):
        verify(run, run.record, manifest=reseeded)


def test_wrong_model_or_code_digest_is_rejected(run: Run) -> None:
    for wrong in (
        identity(code_commit=_flip_hex(identity().code_commit or "")),
        identity(adapter_config_digest=None),
        identity(image_digest="sha256:" + _test_digest("other image")),
    ):
        with pytest.raises(IntegrityError, match=r"provenance\.submission\..*does not match the pinned submission"):
            verify(run, run.record, expected_submission=wrong)
    # A provenance that claims another code commit does not match the pinned submission either.
    claimed = edited(run, lambda p: p["submission"]["identity"].update(code_commit="0" * 40))
    with pytest.raises(IntegrityError, match="code_commit"):
        verify(run, claimed, expected_submission=identity())
    # The submission label must be the result's model ID.
    with pytest.raises(IntegrityError, match="model_id"):
        build_evaluation_provenance(run.frozen, run.manifest, run.result, identity("someone-else"))
    with pytest.raises(IntegrityError, match="model_id"):
        verify(run, edited(run, lambda p: p["submission"]["identity"].update(model_id="someone-else")))


def test_evaluator_digest_is_checked_when_pinned(run: Run) -> None:
    code = evaluator_code_digest()
    assert code == evaluator_code_digest() and len(code) == 64
    with pytest.raises(IntegrityError, match="not recorded"):
        verify(run, run.record, expected_evaluator_code_digest=code)
    pinned = ProvenanceRecord(build_evaluation_provenance(run.frozen, run.manifest, run.result, identity(), evaluator_code_digest=code))
    assert verify(run, pinned, expected_evaluator_code_digest=code).evaluator_code_pinned
    with pytest.raises(IntegrityError, match="evaluator digest"):
        verify(run, pinned, expected_evaluator_code_digest=_flip_hex(code))


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda p: p["versions"].update(simulator="0.1.0"), r"versions\.simulator"),
        (lambda p: p["versions"].update(generator="2.0.0"), r"versions\.generator"),
        (lambda p: p["versions"].update(benchmark_revision="2"), r"versions\.benchmark_revision"),
        (lambda p: p["suite"].update(generator_version="0.9.0"), r"suite\.generator_version"),
        (lambda p: p["result"].update(result_version="0.9"), r"result\.result_version"),
        (lambda p: p.update(integrity_protocol_version="2.0"), "unsupported version"),
    ],
)
def test_version_mismatch_is_rejected(run: Run, edit: Callable[[dict[str, Any]], None], message: str) -> None:
    with pytest.raises(IntegrityError, match=message):
        verify(run, edited(run, edit))


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda p: p.update(timestamp="2026-10-19T00:00:00Z"), r"unknown fields \['timestamp'\]"),
        (lambda p: p["versions"].update(extra="1.0"), r"unknown fields \['extra'\]"),
        (lambda p: p["suite"].pop("suite_hash"), r"missing fields \['suite_hash'\]"),
        (lambda p: p["submission"]["identity"].update(signer="me"), r"unknown fields \['signer'\]"),
        (lambda p: p["suite"].update(scenario_count=True), "non-negative integer"),
        (lambda p: p["suite"].update(suite_hash="ABC"), "64 lowercase hex"),
        (lambda p: p["protocol"].update(mode="trust-me"), "unknown mode"),
        (lambda p: p["protocol"].update(submission_roster_digest="0" * 64), "roster digest is required exactly"),
        (lambda p: p["evaluator"].update(code_digest=""), "64 lowercase hex"),
        (lambda p: p["submission"]["identity"].update(image_digest="latest"), "sha256:"),
    ],
)
def test_malformed_or_unknown_fields_are_rejected(run: Run, edit: Callable[[dict[str, Any]], None], message: str) -> None:
    with pytest.raises(IntegrityError, match=message):
        verify(run, edited(run, edit))


def test_record_envelope_is_strict(run: Run) -> None:
    data = run.record.to_dict()
    with pytest.raises(IntegrityError, match="provenance_digest"):
        verify(run, {**data, "provenance_digest": _flip_hex(data["provenance_digest"])})
    with pytest.raises(IntegrityError, match="identity_digest"):
        broken = copy.deepcopy(data)
        broken["provenance"]["submission"]["identity_digest"] = _flip_hex(broken["provenance"]["submission"]["identity_digest"])
        broken["provenance_digest"] = sha256_hex(broken["provenance"])
        verify(run, broken)
    with pytest.raises(IntegrityError, match="unknown fields"):
        verify(run, {**data, "verified": True})
    # A file can never claim that its signatures are valid.
    with pytest.raises(IntegrityError, match="signing"):
        verify(run, {**data, "signing": "signed_verified"})


def test_incompatible_artifacts_are_rejected(run: Run) -> None:
    public = FrozenGeneratedSuite(SUITE_DIR, load_suite(SUITE_DIR), run.frozen.generation)
    with pytest.raises(IntegrityError, match="not a public suite"):
        verify(run, run.record, frozen_suite=public)
    partial = EvaluationResult(run.result.result_version, MODEL, run.manifest, run.result.episodes[:3])
    with pytest.raises(IntegrityError, match="incomplete"):
        build_evaluation_provenance(run.frozen, run.manifest, partial, identity())
    with pytest.raises(IntegrityError, match="incomplete"):
        verify(run, run.record, result=partial)
    with pytest.raises(TypeError):
        verify(run, run.record, frozen_suite=load_suite(SUITE_DIR))


# --------------------------------------------------------------------------- signatures


def test_unsigned_provenance_says_so(run: Run) -> None:
    assert run.record.signing is SigningState.UNSIGNED
    data = run.record.to_dict()
    assert data["signing"] == "unsigned" and data["signatures"] == []
    assert verify(run, run.record).signature_status is SignatureStatus.UNSIGNED
    with pytest.raises(IntegrityError, match="unsigned"):
        verify(run, run.record, require_signatures=True)


def test_external_signature_round_trips_without_being_trusted(run: Run, tmp_path: Path) -> None:
    # Placeholder bytes standing in for an external signer's output. EMBERBench never checks them itself.
    sig = SignatureRecord("ed25519", "test-key-1", bytes(range(64)))
    signed = run.record.with_signature(sig)
    assert signed.signing is SigningState.EXTERNALLY_SIGNED
    assert signed.provenance.provenance_digest() == run.record.provenance.provenance_digest()
    save_provenance(signed, tmp_path / PROVENANCE_NAME)
    loaded = load_provenance(tmp_path / PROVENANCE_NAME)
    assert loaded == signed and loaded.signatures[0].signature == bytes(range(64))
    assert loaded.to_dict()["signing"] == "externally_signed"

    unverified = verify(run, loaded)
    assert unverified.signature_status is SignatureStatus.SIGNED_UNVERIFIED and unverified.verified_key_ids == ()
    with pytest.raises(IntegrityError, match="no verifier"):
        verify(run, loaded, require_signatures=True)

    calls: list[tuple[SignatureRecord, bytes]] = []

    def accepting(s: SignatureRecord, payload: bytes) -> bool:
        calls.append((s, payload))
        return True

    ok = verify(run, loaded, signature_verifier=accepting, require_signatures=True)
    assert ok.signature_status is SignatureStatus.SIGNED_VERIFIED and ok.verified_key_ids == ("test-key-1",)
    assert calls == [(sig, run.record.provenance.signing_payload())]
    with pytest.raises(IntegrityError, match="rejected by the verifier"):
        verify(run, loaded, signature_verifier=lambda s, p: False)
    with pytest.raises(IntegrityError, match="rejected by the verifier"):
        verify(run, loaded, signature_verifier=lambda s, p: 1)  # truthy, but not True


def test_signature_records_are_strict(run: Run) -> None:
    sig = SignatureRecord("ed25519", "test-key-1", b"\x01\x02")
    with pytest.raises(IntegrityError, match="same algorithm and key ID"):
        run.record.with_signature(sig).with_signature(sig)
    for bad in (
        {"algorithm": "ed25519", "key_id": "k", "signature_base64": "not base64!"},
        {"algorithm": "ed25519", "key_id": "k", "signature_base64": "AQI"},  # missing padding: not canonical
        {"algorithm": "ed 25519", "key_id": "k", "signature_base64": "AQI="},
        {"algorithm": "ed25519", "key_id": " k", "signature_base64": "AQI="},
        {"algorithm": "ed25519", "key_id": "k", "signature_base64": ""},
        {"algorithm": "ed25519", "key_id": "k", "signature_base64": "AQI=", "trusted": True},
    ):
        with pytest.raises(IntegrityError):
            SignatureRecord.from_dict(bad)
    assert SignatureRecord.from_dict(sig.to_dict()) == sig


# --------------------------------------------------------------------------- contamination controls


def test_provenance_holds_no_private_seed_or_timestamp(run: Run) -> None:
    text = run.record.to_file_text()
    for seed in (SEED, *EVAL_SEEDS):
        for form in (str(seed), format(seed, "x"), format(seed, "X")):
            assert form not in text

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    assert not {k for k in keys(run.record.to_dict()) if "seed" in k or "time" in k or "date" in k}
    assert "episodes" not in text and "scenarios" not in text


def test_observation_carries_no_integrity_fields() -> None:
    assert {f.name for f in dataclasses.fields(Observation)} == MODEL_VISIBLE_OBSERVATION_FIELDS
    params = list(inspect.signature(Agent.act).parameters)
    assert params == ["self", "observation"]


def test_agents_never_receive_private_material(run: Run, tmp_path: Path) -> None:
    seen: list[str] = []

    def recording_policy(obs: Observation) -> dict[str, Action]:
        # The serialized form adds only its own version tag.
        assert set(obs.to_dict()) == MODEL_VISIBLE_OBSERVATION_FIELDS | {"observation_version"}
        seen.append(canonical_json(obs.to_dict()))
        return greedy_exit_policy(obs)

    prior = sha256_hex(run.result.to_dict())
    tokens = sensitive_tokens(
        run.frozen,
        run.manifest,
        generator_seed=SEED,
        extra=(prior, run.record.provenance.provenance_digest(), run.submission.identity_digest(), str(tmp_path)),
    )
    # Every kind of private material is covered.
    g = run.frozen.generation
    expected = {BATCH, g.suite_hash, g.generation_hash(), g.config_hash, run.manifest.manifest_hash(), str(SEED), format(SEED, "x"), prior}
    expected |= {e.scenario_id for e in g.entries} | {e.content_hash for e in g.entries} | {str(s) for s in EVAL_SEEDS}
    assert expected <= tokens
    run_evaluation(run.frozen.suite, run.manifest, "recorder", factory_of(recording_policy), tmp_path / "recorder.json")
    assert len(seen) > len(run.manifest.episodes)
    assert find_leaks("\n".join(seen), tokens) == ()
    # Positive control: a planted value is found.
    assert find_leaks(seen[0] + BATCH.upper(), tokens) == (BATCH,)


# --------------------------------------------------------------------------- workspace and post-submission generation


def _complete_workspace(
    root: Path, config: GeneratorConfig, submission: SubmissionIdentity, roster: SubmissionRoster | None, private_seed: int
) -> ProvenanceRecord:
    assert workspace_stage(freeze_submission(submission, root).parent.parent) is WorkspaceStage.SUBMISSION_FROZEN
    if roster is not None:
        save_roster(roster, root / SUBMISSION_ROSTER_NAME)
        generated = generate_post_submission_suite(config, private_seed, roster, batch_id="post-submission-test")
    else:
        generated = generate_suite(config, private_seed, batch_id="shared-test")
    frozen = freeze_generated_suite(generated, root / GENERATED_SUITE_DIR)
    assert workspace_stage(root) is WorkspaceStage.SUITE_FROZEN
    manifest = build_evaluation_manifest(frozen.suite, EVAL_SEEDS)
    save_evaluation_manifest(manifest, root / EVALUATION_MANIFEST_NAME)
    assert workspace_stage(root) is WorkspaceStage.MANIFEST_FROZEN
    result = run_evaluation(frozen.suite, manifest, submission.model_id, factory_of(greedy_exit_policy), root / RESULT_NAME)
    assert workspace_stage(root) is WorkspaceStage.EVALUATED
    record = ProvenanceRecord(build_evaluation_provenance(frozen, manifest, result, submission, roster=roster))
    save_provenance(record, root / PROVENANCE_NAME)
    assert workspace_stage(root) is WorkspaceStage.PROVENANCE_RECORDED
    return record


def test_post_submission_generation_workflow(config: GeneratorConfig, tmp_path: Path) -> None:
    submissions = [identity("model-b"), identity("model-a")]
    roster = SubmissionRoster.of(submissions)
    assert [s.model_id for s in roster.submissions] == ["model-a", "model-b"]
    derived = post_submission_generator_seed(PRIVATE_TEST_SEED, roster)
    assert derived != PRIVATE_TEST_SEED and 0 <= derived < 1 << 256

    # Workspaces for both models: their suites are byte-identical, because the batch is bound to the shared roster.
    records = {s.model_id: _complete_workspace(tmp_path / s.model_id, config, s, roster, PRIVATE_TEST_SEED) for s in submissions}
    for model_id, record in records.items():
        v = verify_evaluation_workspace(tmp_path / model_id)
        assert v.mode is ProtocolMode.POST_SUBMISSION_GENERATION
        assert record.provenance.roster_digest == roster.roster_digest()
    assert len({r.provenance.suite.generation_hash for r in records.values()}) == 1
    assert len({r.provenance.evaluation_manifest.manifest_hash for r in records.values()}) == 1

    suite_dir = tmp_path / "model-a" / GENERATED_SUITE_DIR
    # Audit after reveal: the frozen batch is exactly what the private seed and roster generate ...
    verify_post_submission_generation(suite_dir, PRIVATE_TEST_SEED, roster)
    # ... and not what the private seed alone generates.
    plain = generate_suite(config, PRIVATE_TEST_SEED, batch_id="post-submission-test")
    assert plain.generation.suite_hash != records["model-a"].provenance.suite.suite_hash
    # A submission changed after the batch was frozen breaks both the audit and the provenance.
    changed = SubmissionRoster.of([identity("model-a", code_commit="f" * 40), identity("model-b")])
    with pytest.raises(GenerationError):
        verify_post_submission_generation(suite_dir, PRIVATE_TEST_SEED, changed)
    frozen = load_generated_suite(suite_dir)
    manifest = load_evaluation_manifest(tmp_path / "model-a" / EVALUATION_MANIFEST_NAME)
    result = load_evaluation_result(tmp_path / "model-a" / RESULT_NAME, frozen.suite)
    args: dict[str, Any] = {"frozen_suite": frozen, "manifest": manifest, "result": result}
    with pytest.raises(IntegrityError, match="roster: digest"):
        verify_evaluation_provenance(records["model-a"], roster=changed, **args)
    with pytest.raises(IntegrityError, match="needs the frozen submission roster"):
        verify_evaluation_provenance(records["model-a"], **args)
    with pytest.raises(IntegrityError, match="does not contain the submission"):
        build_evaluation_provenance(frozen, manifest, result, identity("model-a", dependency_lock_digest=_test_digest("late")), roster=roster)


def test_roster_is_strict() -> None:
    with pytest.raises(IntegrityError, match="pins no code"):
        SubmissionRoster.of([SubmissionIdentity("label-only")])
    with pytest.raises(IntegrityError, match="unique"):
        SubmissionRoster.of([identity("a"), identity("a", code_commit="0" * 40)])
    with pytest.raises(IntegrityError, match="sorted"):
        SubmissionRoster((identity("b"), identity("a")))
    with pytest.raises(IntegrityError, match="at least one"):
        SubmissionRoster(())
    roster = SubmissionRoster.of([identity("a"), identity("b")])
    assert SubmissionRoster.from_dict(json.loads(roster.to_file_text())) == roster
    assert SubmissionRoster.of([identity("a")]).roster_digest() != roster.roster_digest()


def test_shared_workspace_verifies_and_detects_tampering(config: GeneratorConfig, tmp_path: Path) -> None:
    root = tmp_path / "evaluation-run"
    record = _complete_workspace(root, config, identity(), None, SEED)
    v = verify_evaluation_workspace(root, expected_submission=identity())
    assert v.mode is ProtocolMode.SHARED_HELDOUT_BATCH and v.provenance_digest == record.provenance.provenance_digest()
    with pytest.raises(IntegrityError, match="expected submission"):
        verify_evaluation_workspace(root, expected_submission=identity(adapter_config_digest=None))
    # Frozen artifacts are written once.
    with pytest.raises(IntegrityError, match="never overwritten"):
        save_provenance(record, root / PROVENANCE_NAME)
    with pytest.raises(IntegrityError, match="already frozen"):
        freeze_submission(identity(), root)
    # Any edit to the stored result is caught.
    result_path = root / RESULT_NAME
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["model_id"] = "someone-else"
    result_path.write_text(canonical_json(data) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_evaluation_workspace(root)


def test_workspace_layout_is_validated(tmp_path: Path) -> None:
    assert workspace_stage(tmp_path) is WorkspaceStage.EMPTY
    (tmp_path / RESULT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(IntegrityError, match="evaluated artifact present before submission_frozen"):
        workspace_stage(tmp_path)
    (tmp_path / RESULT_NAME).unlink()
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    with pytest.raises(IntegrityError, match="unexpected workspace entries"):
        workspace_stage(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IntegrityError, match="not complete"):
        verify_evaluation_workspace(empty)


def test_workspace_must_live_outside_the_repository() -> None:
    with pytest.raises(IntegrityError, match="inside the EMBERBench source checkout"):
        workspace_stage(REPO_ROOT / "docs")
    target = REPO_ROOT / "evaluation-run-should-not-exist"
    with pytest.raises(IntegrityError, match="inside the EMBERBench source checkout"):
        freeze_submission(identity(), target)
    assert not target.exists()


def test_directory_digest_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_bytes(b"one")
    (tmp_path / "y.txt").write_bytes(b"two")
    first = digest_directory(tmp_path)
    assert first == digest_directory(tmp_path)
    assert first == sha256_hex(sorted([["a/x.txt", digest_bytes(b"one")], ["y.txt", digest_bytes(b"two")]]))
    (tmp_path / "y.txt").write_bytes(b"twO")
    assert digest_directory(tmp_path) != first
