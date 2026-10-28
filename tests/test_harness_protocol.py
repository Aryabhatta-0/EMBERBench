"""Harness 1.1 (docs/harness-protocol.md), accounting, suite commitments and ``emberbench run``.

A local HTTP server plays a decision model, so no test opens an external
network connection. The small environment pack is built from two public
scenarios in a temporary directory.
"""

import contextlib
import dataclasses
import io
import json
import math
import shutil
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from emberbench import (
    Action,
    EnvironmentPack,
    EpisodeRefused,
    Observation,
    agent_policy_factory,
    build_unique_world_manifest,
    freeze_environment_pack,
    load_evaluation_result,
    load_suite,
    run_episode,
    run_evaluation,
    score_result,
)
from emberbench.baselines import baseline_factory
from emberbench.cli import main as cli_main
from emberbench.commitment import suite_commitment, verify_suite_commitment
from emberbench.harness import (
    AdapterError,
    HarnessProtocol,
    JointMode,
    RemoteAgent,
    RemoteClient,
    probability_diagnostics,
    refused_question,
    remote_model_id,
)
from emberbench.integrity import benchmark_versions
from emberbench.scoring import score_excluding
from emberbench.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"
SMALL_IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation")
CROWD = SUITE_DIR / "v1-pub-09-crowd-100-split-flow.json"
ALLOWED = {a.value: a for a in Action}
KEY_ENV = "EMBERBENCH_TEST_HARNESS_KEY"
SALT = "ab" * 32


# --------------------------------------------------------------------------- fixtures


class FakeDecisionModel:
    """Answers every question with ``stay``; the first ``invalid_rounds`` requests answer ``q0`` with ``fly``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.invalid_rounds = 0
        self.refuse_requests: set[int] = set()  # 0-based request numbers the "provider policy" refuses
        self.lock = threading.Lock()

    def answer(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """The answer, or ``None`` for a refusal (HTTP 502 with the provider's refusal message)."""
        with self.lock:
            self.requests.append(body)
            if len(self.requests) - 1 in self.refuse_requests:
                return None
            invalid = self.invalid_rounds > 0
            self.invalid_rounds -= 1 if invalid else 0
        answers = {}
        for name, question in body["questions"].items():
            options = list(question["criteria"])
            rest = 0.4 / max(len(options) - 1, 1)
            probabilities = {o: (0.6 if o == "stay" else rest) for o in options}
            answers[name] = {"choice": "fly" if invalid and name == "q0" else "stay", "probabilities": probabilities}
        return {"model": "fake-decision-1", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.001}}


@pytest.fixture()
def server() -> Iterator[tuple[str, FakeDecisionModel]]:
    model = FakeDecisionModel()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            answer = model.answer(body)
            refusal = {"error": {"message": f'OpenAI refused to answer question "{min(body["questions"])}"', "code": 502}}
            reply = json.dumps(refusal if answer is None else answer).encode()
            self.send_response(502 if answer is None else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", model
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="module")
def pack(tmp_path_factory: pytest.TempPathFactory) -> EnvironmentPack:
    root = tmp_path_factory.mktemp("pack")
    public = load_suite(SUITE_DIR)
    entries = tuple(e for e in public.manifest.entries if e.scenario_id in SMALL_IDS)
    (root / "suite").mkdir()
    for e in entries:
        shutil.copyfile(SUITE_DIR / e.path, root / "suite" / e.path)
    (root / "suite" / "manifest.json").write_bytes(dataclasses.replace(public.manifest, entries=entries).to_file_text().encode("ascii"))
    suite = load_suite(root / "suite")
    manifest = build_unique_world_manifest(suite, range(4), stochastic_worlds=2)
    return freeze_environment_pack(suite, manifest, root / "environment-pack", pack_id="harness-test-pack")


def _cli(*args: str | Path) -> tuple[int, dict[str, Any] | None]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_main([str(a) for a in args])
    text = stdout.getvalue().strip()
    return code, (json.loads(text) if code == 0 and text else None)


def _run(pack: EnvironmentPack, out: Path, *extra: str | Path) -> tuple[int, dict[str, Any] | None]:
    return _cli("run", "--suite", pack.root, "--out", out, "--resamples", "100", "--json", *extra)


def _record(out: Path, model_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "runs" / f"{model_id}.json").read_text(encoding="ascii"))
    return data


def _question(options: tuple[str, ...] = ("stay", "north")) -> dict[str, Any]:
    return {"type": "choice", "instructions": "move?", "criteria": {o: o for o in options}}


class _Stop(Exception):
    pass


def _first_observation(path: Path) -> Observation:
    seen: list[Observation] = []

    def policy(observation: Observation) -> dict[str, Action]:
        seen.append(observation)
        raise _Stop

    with contextlib.suppress(Exception):
        run_episode(load_scenario(path), policy, seed=0)
    return seen[0]


# --------------------------------------------------------------------------- joint protocol


def test_protocol_rules_and_chunking() -> None:
    people = list(range(5))
    assert HarnessProtocol().chunks(people) == [people]
    assert HarnessProtocol(JointMode.FIXED_CHUNK, 2).chunks(people) == [[0, 1], [2, 3], [4]]
    assert HarnessProtocol().chunks([]) == []
    for bad in ({"joint_mode": JointMode.FULL_JOINT, "chunk_size": 4}, {"joint_mode": JointMode.FIXED_CHUNK}, {"max_reasks": -1}):
        with pytest.raises(ValueError):
            HarnessProtocol(**bad)
    assert HarnessProtocol().to_dict() == {
        "harness_version": "1.1",
        "adapter_version": "r1",
        "joint_mode": "full_joint",
        "chunk_size": None,
        "person_order": "person_id",
        "max_reasks": 2,
        "executed_action": "choice",
    }


def test_only_a_declared_variant_changes_the_model_id() -> None:
    assert remote_model_id("openrouter", "typesafe/jev-1.14", "decision", HarnessProtocol()) == "openrouter-typesafe_jev-1.14-choice-r1"
    assert remote_model_id("p", "m", "decision", HarnessProtocol(JointMode.FIXED_CHUNK, 16)) == "p-m-choice-r1-chunk16"
    assert remote_model_id("p", "m", "openai", HarnessProtocol(max_reasks=0)) == "p-m-chat-r1"


def test_remote_agent_splits_people_in_person_id_order(server: tuple[str, FakeDecisionModel], tmp_path: Path) -> None:
    base_url, model = server
    observation = _first_observation(CROWD)
    ids = sorted(p.id for p in observation.active)
    assert len(ids) == 100
    for protocol, sizes in ((HarnessProtocol(), [100]), (HarnessProtocol(JointMode.FIXED_CHUNK, 32), [32, 32, 32, 4])):
        model.requests.clear()
        client = RemoteClient(api="decision", base_url=base_url, model="m", key="", cache_path=tmp_path / f"{protocol.label}.jsonl")
        actions = RemoteAgent(client, provider="t", protocol=protocol).act(observation)
        assert sorted(actions) == ids and set(actions.values()) == {Action.STAY}
        assert [len(r["questions"]) for r in model.requests] == sizes
        # The body is canonical JSON with sorted keys (q0, q1, q10, ...), so order by question number.
        asked = [
            r["questions"][name]["instructions"].split("person ")[1].split(",")[0]
            for r in model.requests
            for name in sorted(r["questions"], key=lambda n: int(n[1:]))
        ]
        assert asked == ids
        assert len({r["state"] for r in model.requests}) == 1  # every chunk sees the same state


# --------------------------------------------------------------------------- validation, re-ask, fail-closed


def test_decision_reasks_only_the_invalid_questions(server: tuple[str, FakeDecisionModel], tmp_path: Path) -> None:
    base_url, model = server
    model.invalid_rounds = 1
    client = RemoteClient(api="decision", base_url=base_url, model="m", key="", cache_path=tmp_path / "c.jsonl")
    body = {"model": "m", "state": "S", "questions": {"q0": _question(), "q1": _question()}}
    exchange = client.ask_decision(body, {"q0": "p1", "q1": "p2"}, ALLOWED)
    assert exchange.moves == {"p1": Action.STAY, "p2": Action.STAY}
    assert [sorted(r["questions"]) for r in model.requests] == [["q0", "q1"], ["q0"]]
    assert model.requests[1]["state"] == "S"
    assert [(r.attempt, r.questions, r.invalid_slots) for r in exchange.requests] == [(0, 2, 1), (1, 1, 0)]
    assert client.format_errors == 1


@pytest.mark.parametrize("redact", [False, True])
def test_decision_fails_closed_after_the_last_reask(server: tuple[str, FakeDecisionModel], tmp_path: Path, redact: bool) -> None:
    base_url, model = server
    model.invalid_rounds = 99
    client = RemoteClient(api="decision", base_url=base_url, model="m", key="", cache_path=tmp_path / "c.jsonl", format_retries=2, redact=redact)
    body = {"model": "m", "state": "S", "questions": {"q0": _question()}}
    with pytest.raises(AdapterError, match="after 2 re-ask") as caught:
        client.ask_decision(body, {"q0": "p1"}, ALLOWED)
    assert len(model.requests) == 3
    assert ("fly" in str(caught.value)) is not redact


def test_probability_diagnostics_are_descriptive_and_strict() -> None:
    options = ["stay", "north", "south"]
    assert probability_diagnostics({"choice": "stay"}, options) is None
    diag = probability_diagnostics({"choice": "north", "probabilities": {"stay": 2, "north": 1, "south": 1}}, options)
    assert diag is not None and diag["valid"] and not diag["argmax_agrees"]
    assert diag["p_choice"] == pytest.approx(0.25) and diag["margin"] == pytest.approx(0.25)
    assert diag["entropy_bits"] == pytest.approx(1.5)
    for bad in ({"stay": 1.0}, {"stay": -0.1, "north": 0.6, "south": 0.5}, {"stay": math.nan, "north": 0.5, "south": 0.5}, {o: 0 for o in options}):
        assert probability_diagnostics({"choice": "stay", "probabilities": bad}, options) == {"valid": False}


# --------------------------------------------------------------------------- suite commitment


def test_suite_commitment_opens_only_with_the_salt(pack: EnvironmentPack) -> None:
    commitment = suite_commitment(pack.digest, SALT)
    assert verify_suite_commitment(pack.root, SALT, commitment)
    assert not verify_suite_commitment(pack.root, "cd" * 32, commitment)
    assert suite_commitment("0" * 64, SALT) != commitment
    for digest, salt in ((pack.digest, "AB" * 32), (pack.digest, "ab"), ("x" * 64, SALT)):
        with pytest.raises(ValueError):
            suite_commitment(digest, salt)


# --------------------------------------------------------------------------- emberbench run


def test_run_baseline_reproduces_the_library_result_and_resumes(pack: EnvironmentPack, tmp_path: Path) -> None:
    out = tmp_path / "run"
    code, summary = _run(pack, out, "--baseline", "congestion-aware-v1", "--expected-digest", pack.digest)
    assert code == 0 and summary is not None and summary["model_id"] == "congestion-aware-v1"
    result = out / "results" / "congestion-aware-v1.json"
    direct = tmp_path / "direct.json"
    run_evaluation(pack.suite, pack.manifest, "congestion-aware-v1", agent_policy_factory(baseline_factory("congestion-aware-v1")), direct)
    assert result.read_bytes() == direct.read_bytes()
    first = result.read_bytes()
    assert _run(pack, out, "--baseline", "congestion-aware-v1")[0] == 0
    assert result.read_bytes() == first
    record = _record(out, "congestion-aware-v1")
    assert record["run_record_version"] == "1.1" and record["benchmark_versions"] == benchmark_versions()
    assert record["suite"]["pack_digest"] == pack.digest and record["suite"]["seeds"] == [e.seed for e in pack.manifest.episodes]
    assert record["model"] == {"kind": "baseline", "model_id": "congestion-aware-v1", "baseline_id": "congestion-aware-v1"}
    assert record["metrics"]["official"] is True and record["usage"] is None
    assert len(record["result_sha256"]) == 64


def test_run_verifies_the_pack_before_any_episode(pack: EnvironmentPack, tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert _run(pack, out, "--baseline", "random-legal-v1", "--expected-digest", "0" * 64)[0] == 5
    assert _run(pack, out, "--baseline", "random-legal-v1", "--commitment", "0" * 64)[0] == 2  # no --salt-env
    assert _run(pack, out, "--baseline", "random-legal-v1", "--private")[0] == 2  # a private run needs a pinned pack
    assert not (out / "results").exists()
    inside = REPO_ROOT / "build" / "emberbench-run-test"
    assert _run(pack, inside, "--baseline", "random-legal-v1")[0] == 6
    assert not inside.exists()


def test_private_run_records_no_scenario_or_seed(pack: EnvironmentPack, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBERBENCH_TEST_SALT", SALT)
    commitment = suite_commitment(pack.digest, SALT)
    out = tmp_path / "private"
    code, summary = _run(pack, out, "--baseline", "nearest-exit-greedy-v1", "--private", "--commitment", commitment, "--salt-env", "EMBERBENCH_TEST_SALT")
    assert code == 0 and summary is not None
    text = (out / "runs" / "nearest-exit-greedy-v1.json").read_text(encoding="ascii")
    record = json.loads(text)
    assert record["private"] and record["suite"]["seeds"] is None and record["suite"]["pack_id"] is None and "scenarios" not in record
    assert record["suite"]["commitment"] == {"commitment": commitment, "verified": True}
    for secret in (*SMALL_IDS, pack.pack_id, SALT):
        assert secret not in text and secret not in json.dumps(summary)
    assert _run(pack, out, "--baseline", "nearest-exit-greedy-v1", "--private", "--commitment", "1" * 64, "--salt-env", "EMBERBENCH_TEST_SALT")[0] == 5


def test_run_never_stops_an_episode_before_the_horizon(pack: EnvironmentPack, tmp_path: Path) -> None:
    out = tmp_path / "stay"
    assert _run(pack, out, "--agent", "cli_agents:stay", "--agent-path", Path(__file__).parent, "--model-id", "stay")[0] == 0
    accounting = json.loads((out / "accounting" / "stay.json").read_text(encoding="ascii"))
    for episode in accounting["per_episode"]:
        assert episode["hit_horizon"] == (episode["steps"] == episode["max_steps"])
    assert accounting["horizon_hits"] == sum(e["hit_horizon"] for e in accounting["per_episode"])


def test_run_remote_decision_model_accounts_for_reasks(
    pack: EnvironmentPack, server: tuple[str, FakeDecisionModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, model = server
    monkeypatch.setenv(KEY_ENV, "test-key")
    model.invalid_rounds = 1
    out = tmp_path / "remote"
    code, summary = _run(pack, out, "--model", "fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV)
    assert code == 0 and summary is not None and summary["model_id"] == "t-fake-choice-r1"
    record = _record(out, "t-fake-choice-r1")
    assert record["model"]["harness_protocol"]["joint_mode"] == "full_joint"
    assert record["model"]["endpoint"] == f"{base_url}/systemone" and len(record["model"]["harness_sha256"]) == 64
    acc = record["accounting"]
    assert acc["reask_requests"] == 1 and acc["invalid_slots_first_attempt"] == 1
    assert acc["reask_cost_multiplier"] > 1 and acc["total_cost"] == pytest.approx(0.001 * acc["requests"])
    assert acc["probabilities"]["note"].endswith("not calibration (no target labels)")
    assert acc["probabilities"]["p_choice_mean"] == pytest.approx(0.6)
    # A declared chunked variant is a separate model ID and result.
    code, summary = _run(pack, out, "--model", "fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV,
                         "--joint-mode", "fixed_chunk", "--chunk-size", "1")  # fmt: skip
    assert code == 0 and summary is not None and summary["model_id"] == "t-fake-choice-r1-chunk1"
    assert (out / "results" / "t-fake-choice-r1.json").exists()
    assert _run(pack, out, "--model", "fake", "--base-url", base_url, "--key-env", KEY_ENV, "--chunk-size", "4")[0] == 2


# --------------------------------------------------------------------------- refusals


def test_refused_question_parses_only_the_provider_refusal_message() -> None:
    assert refused_question('{"error": {"message": "OpenAI refused to answer question \\"q29\\"", "code": 502}}') == "q29"
    for body in ("not json", "[]", '{"error": "x"}', '{"error": {"message": 7}}', '{"error": {"message": "Bad gateway", "code": 502}}'):
        assert refused_question(body) is None


def test_a_refusal_is_not_retried_or_reprompted(server: tuple[str, FakeDecisionModel], tmp_path: Path) -> None:
    base_url, model = server
    model.refuse_requests = {0}
    client = RemoteClient(api="decision", base_url=base_url, model="m", key="", cache_path=tmp_path / "c.jsonl")
    body = {"model": "m", "state": "S", "questions": {"q0": _question()}}
    with pytest.raises(EpisodeRefused) as caught:
        client.ask_decision(body, {"q0": "p1"}, ALLOWED)
    assert caught.value.question == "q0" and len(model.requests) == 1 and client.retries == 0


def test_run_evaluation_without_a_refusal_handler_propagates(pack: EnvironmentPack, tmp_path: Path) -> None:
    def policy(observation: Observation) -> dict[str, Action]:
        raise EpisodeRefused("refused", question="q0")

    path = tmp_path / "result.json"
    with pytest.raises(EpisodeRefused):
        run_evaluation(pack.suite, pack.manifest, "refuser", lambda: policy, path)
    refused: list[int] = []
    result = run_evaluation(pack.suite, pack.manifest, "refuser", lambda: policy, tmp_path / "handled.json",
                            on_refusal=lambda spec, _: refused.append(spec.index))  # fmt: skip
    assert refused == [e.index for e in pack.manifest.episodes] and result.episodes == ()


def test_run_scores_a_refused_episode_0_and_keeps_the_others(
    pack: EnvironmentPack, server: tuple[str, FakeDecisionModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, model = server
    monkeypatch.setenv(KEY_ENV, "test-key")
    model_id = "t-fake-choice-r1"
    first = pack.manifest.episodes[0].index
    scheduled = len(pack.manifest.episodes)
    reference = tmp_path / "reference"
    assert _run(pack, reference, "--model", "fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV)[0] == 0
    model.requests.clear()
    model.refuse_requests = {0}  # the first request of the first episode
    out = tmp_path / "refused"
    code, summary = _run(pack, out, "--model", "fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV)
    assert code == 0 and summary is not None and summary["refusals"] == 1
    log = json.loads((out / "refusals" / f"{model_id}.json").read_text(encoding="ascii"))
    assert log["refused"] == [{"index": first, "step": 0, "question": "q0"}]
    record = _record(out, model_id)
    diagnostic = record["refusals"].pop("score_excluding_refusals")
    assert record["refusals"] == {"count": 1, "rate": round(1 / scheduled, 6), "episodes": [first], "failure_kind": "refusal"}
    report = score_result(load_evaluation_result(reference / "results" / f"{model_id}.json", pack.suite), pack, resamples=100)
    excluded = score_excluding(report, [first])
    assert excluded is not None and diagnostic == round(float(excluded), 3)
    assert record["accounting"]["refusals"]["episodes"] == 1
    assert record["accounting"]["refusals"]["rate"] == pytest.approx(1 / scheduled)
    report = json.loads((out / "scores" / f"{model_id}.json").read_text(encoding="ascii"))
    assert report["robustness"]["failed_episode_trials"]["refusal"] == 1
    refused_score = next(e for e in report["episodes"] if e["index"] == first)
    assert refused_score["status"] == "failed" and refused_score["failure"] == "refusal" and refused_score["score"] == 0
    # The other episodes keep exactly the results of a run without the refusal.
    kept = json.loads((out / "results" / f"{model_id}.json").read_text(encoding="ascii"))["episodes"]
    full = json.loads((reference / "results" / f"{model_id}.json").read_text(encoding="ascii"))["episodes"]
    assert len(kept) == scheduled - 1 and kept == full[1:]
    # A resume never asks the refused question again.
    model.requests.clear()
    assert _run(pack, out, "--model", "fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV)[0] == 0
    assert model.requests == []


# --------------------------------------------------------------------------- hidden exposure log


def test_private_remote_runs_log_every_exposure(
    pack: EnvironmentPack, server: tuple[str, FakeDecisionModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, _ = server
    monkeypatch.setenv(KEY_ENV, "test-key")
    monkeypatch.setenv("EMBERBENCH_TEST_SALT", SALT)
    commitment = suite_commitment(pack.digest, SALT)
    out, log = tmp_path / "hidden", tmp_path / "exposure.jsonl"
    remote = ("--model", "lab/fake", "--base-url", base_url, "--provider", "t", "--key-env", KEY_ENV,
              "--private", "--commitment", commitment, "--salt-env", "EMBERBENCH_TEST_SALT")  # fmt: skip
    assert _run(pack, out, *remote)[0] == 2  # a private remote run must log its exposure
    assert not log.exists() and not (out / "results").exists()
    assert _run(pack, out, *remote, "--exposure-log", REPO_ROOT / "exposure.jsonl")[0] == 6
    assert _run(pack, out, *remote, "--exposure-log", log, "--retention", "zero")[0] == 0
    text = log.read_text(encoding="ascii")
    line = json.loads(text)
    assert line["suite"] == {"commitment": commitment} and line["episodes_sent"] == len(pack.manifest.episodes)
    assert (line["router"], line["upstream"], line["retention"], line["endpoint_host"]) == ("t", "lab", "zero", "127.0.0.1")
    for secret in (*SMALL_IDS, pack.pack_id, SALT):
        assert secret not in text
    assert _run(pack, out, *remote, "--exposure-log", log)[0] == 0  # nothing pending: nothing sent, nothing logged
    assert log.read_text(encoding="ascii") == text
