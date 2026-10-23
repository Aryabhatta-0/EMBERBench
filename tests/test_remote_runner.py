"""The remote-model runner (tools/experiments/remote): any base URL and API key on the V1 evaluation pack.

A local HTTP server plays the model, so these tests open no external network
connection. The server answers in the OpenAI, Anthropic and decision styles.
"""

import importlib
import json
import re
import shutil
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from emberbench import Action

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_DIR = REPO_ROOT / "tools" / "experiments" / "remote"
KEY = "test-key-0123456789"
KEY_ENV = "EMBERBENCH_TEST_REMOTE_KEY"


def _remote_module() -> ModuleType:
    if str(REMOTE_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_DIR))
    return importlib.import_module("run_remote")


remote = _remote_module()
ALLOWED = {a.value: a for a in Action}


class FakeModel:
    """The server state: every request, and how many invalid chat answers to give first."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.invalid_first = 0
        self.lock = threading.Lock()

    def answer(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.requests.append((path, headers, body))
            invalid = self.invalid_first > 0
            if invalid and not path.endswith("/systemone"):
                self.invalid_first -= 1
        if path.endswith("/systemone"):
            return {"model": "fake-decision-1", "answers": {q: {"choice": "stay"} for q in body["questions"]}, "usage": {"input_tokens": 3, "output_tokens": 1}}
        first = next(m["content"] for m in body["messages"] if m["role"] == "user")
        ids = re.search(r"exactly these keys: (.*)\.\n", first)
        assert ids is not None
        moves = {pid: ("NORTH" if invalid else "stay") for pid in ids.group(1).split(", ")}
        text = f"<think>plan</think>\n```json\n{json.dumps(moves)}\n```"
        if path.endswith("/messages"):
            return {"model": "fake-claude-1", "content": [{"type": "text", "text": text}], "usage": {"input_tokens": 5, "output_tokens": 2}}
        return {"model": "fake-gpt-1", "choices": [{"message": {"role": "assistant", "content": text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


@pytest.fixture()
def server() -> Iterator[tuple[str, FakeModel]]:
    model = FakeModel()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps(model.answer(self.path, {k.lower(): v for k, v in self.headers.items()}, body)).encode()
            self.send_response(200)
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


def _run(base_url: str, out: Path, *extra: str) -> dict[str, Any]:
    assert remote.main(["--base-url", base_url, "--out", str(out), "--key-env", KEY_ENV, "--protocol", "smoke", *extra]) == 0
    summary: dict[str, Any] = json.loads((out / "summary.json").read_text(encoding="ascii"))
    return summary


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- parsing and URLs


def test_parse_moves_reads_the_last_object_with_exactly_the_person_ids() -> None:
    text = '<think>{"p1": "west"}</think> first {"x": 1} then ```json\n{"p1": "north", "p2": "stay"}\n```'
    assert remote.parse_moves(text, ["p1", "p2"], ALLOWED) == {"p1": Action.NORTH, "p2": Action.STAY}


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("no json here", "no JSON object"),
        ('{"p1": "north"}', "missing"),
        ('{"p1": "north", "p2": "stay", "p3": "stay"}', "extra"),
        ('{"p1": "NORTH", "p2": "stay"}', "not one of"),
        ('{"p1": null, "p2": "stay"}', "not one of"),
    ],
)
def test_parse_moves_repairs_nothing(text: str, reason: str) -> None:
    with pytest.raises(remote.AdapterError, match=reason):
        remote.parse_moves(text, ["p1", "p2"], ALLOWED)


def test_endpoint_url_and_provider_label() -> None:
    assert remote.endpoint_url("https://api.openai.com/v1/", "openai") == "https://api.openai.com/v1/chat/completions"
    assert remote.endpoint_url("https://x.test/v1/chat/completions", "openai") == "https://x.test/v1/chat/completions"
    assert remote.endpoint_url("https://api.anthropic.com/v1", "anthropic") == "https://api.anthropic.com/v1/messages"
    assert remote.endpoint_url("https://api.typesafe.ai/v1", "decision") == "https://api.typesafe.ai/v1/systemone"
    assert remote.endpoint_url("https://openrouter.ai/api/alpha/decisions", "decision") == "https://openrouter.ai/api/alpha/decisions"
    with pytest.raises(SystemExit):
        remote.endpoint_url("api.openai.com/v1", "openai")
    assert remote.provider_label("https://api.openai.com/v1") == "openai"
    assert remote.provider_label("https://openrouter.ai/api/v1") == "openrouter"
    assert remote.provider_label("http://127.0.0.1:8000/v1") == "127.0.0.1"
    assert remote.label("meta-llama/Llama-3.3 70B") == "meta-llama_Llama-3.3_70B"


# --------------------------------------------------------------------------- end to end


def test_openai_run_is_official_cached_and_byte_identical(server: tuple[str, FakeModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, model = server
    monkeypatch.setenv(KEY_ENV, KEY)
    out = tmp_path / "run"
    summary = _run(base_url, out, "--model", "gpt/test", "--reference-run", str(tmp_path / "ref"))

    assert summary["model_id"] == "127.0.0.1-gpt_test-chat-r1"
    assert summary["official_procedure"] is True
    assert summary["episodes"] == 5
    assert summary["resolved_models"] == ["fake-gpt-1"]
    assert summary["api"]["requests_sent"] == len(model.requests) > 0
    assert summary["api"]["input_tokens"] == 5 * len(model.requests)
    assert summary["api"]["cost"] == 0 and summary["api"]["cost_source"] == "not reported"
    # Measured, but rounded to 3 decimals: a fast local fake server can give 0.0.
    assert 0 <= summary["api"]["first_response_s"] and 0 <= summary["api"]["median_response_s"]
    assert "always-stay" in summary["references"]  # the missing reference run was made first, offline
    assert (out / "REPORT.md").read_text(encoding="utf-8").startswith("# `127.0.0.1-gpt_test-chat-r1`")
    path, headers, body = model.requests[0]
    assert path == "/v1/chat/completions" and headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "gpt/test" and body["temperature"] == 0.0 and "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": remote.SYSTEM_PROMPT}
    # The prompt is built from the observation only: never the scenario ID.
    assert all("v1-pub" not in m["content"] for _, _, b in model.requests for m in b["messages"])
    assert all(KEY.encode() not in data for data in _files(out).values())

    # A second run with only the cache sends no request and gives the same result and score bytes.
    again = tmp_path / "again"
    again.mkdir()
    shutil.copyfile(out / "cache.jsonl", again / "cache.jsonl")
    sent = len(model.requests)
    summary2 = _run(base_url, again, "--model", "gpt/test")
    assert len(model.requests) == sent and summary2["api"]["requests_sent"] == 0
    assert summary2["api"]["first_response_s"] is None  # answers from the cache are not timed
    for sub in ("results", "scores"):
        assert _files(again / sub) == _files(out / sub)


def test_anthropic_style_headers_and_body(server: tuple[str, FakeModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, model = server
    monkeypatch.setenv(KEY_ENV, KEY)
    summary = _run(base_url, tmp_path / "run", "--api", "anthropic", "--model", "claude-test", "--provider", "anthropic")
    assert summary["model_id"] == "anthropic-claude-test-chat-r1"
    assert summary["resolved_models"] == ["fake-claude-1"]
    path, headers, body = model.requests[0]
    assert path == "/v1/messages"
    assert headers["x-api-key"] == KEY and headers["anthropic-version"] == remote.ANTHROPIC_VERSION
    assert "authorization" not in headers
    assert body["system"] == remote.SYSTEM_PROMPT and body["max_tokens"] == 2048
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_decision_style_matches_the_jev_request(server: tuple[str, FakeModel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, model = server
    monkeypatch.setenv(KEY_ENV, KEY)
    summary = _run(base_url, tmp_path / "run", "--api", "decision", "--model", "jev-latest", "--provider", "typesafe", "--input-price", "0.042")
    assert summary["model_id"] == "typesafe-jev-latest-choice-r1"
    # The API reports no cost, so the cost comes from the list price of the input tokens.
    assert summary["api"]["cost_source"] == "list price"
    assert summary["api"]["cost"] == round(summary["api"]["input_tokens"] * 0.042 / 1e6, 6) > 0
    path, _, body = model.requests[0]
    assert path == "/v1/systemone"
    assert sorted(body) == ["model", "questions", "state"]
    question = next(iter(body["questions"].values()))
    assert question["type"] == "choice" and set(question["criteria"]) <= set(ALLOWED)
    rows = [json.loads(line) for line in (tmp_path / "run" / "cache.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(sorted(row) == ["request_sha256", "response"] for row in rows)  # the Jev cache format


def test_invalid_answer_is_re_asked_then_accepted(server: tuple[str, FakeModel], tmp_path: Path) -> None:
    base_url, model = server
    model.invalid_first = 1
    client = remote.RemoteClient(api="openai", base_url=base_url, model="m", key=KEY, cache_path=tmp_path / "cache.jsonl")
    prompt = "State.\nAnswer with one JSON object with exactly these keys: p1, p2.\n"
    exchange = client.ask_chat(prompt, ["p1", "p2"], ALLOWED)
    assert exchange.moves == {"p1": Action.STAY, "p2": Action.STAY}
    assert client.format_errors == 1 and client.calls == 2 and len(exchange.responses) == 2
    retry = model.requests[1][2]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "not valid" in retry[-1]["content"]


def test_answer_still_invalid_fails_closed_and_is_not_cached(server: tuple[str, FakeModel], tmp_path: Path) -> None:
    base_url, model = server
    model.invalid_first = 10
    cache = tmp_path / "cache.jsonl"
    client = remote.RemoteClient(api="openai", base_url=base_url, model="m", key=KEY, cache_path=cache, format_retries=1)
    prompt = "State.\nAnswer with one JSON object with exactly these keys: p1.\n"
    with pytest.raises(remote.AdapterError, match="NORTH"):
        client.ask_chat(prompt, ["p1"], ALLOWED)
    assert client.calls == 2 and not cache.exists()


def test_missing_key_is_refused_for_a_remote_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match=KEY_ENV):
        remote.main(["--base-url", "https://api.example.test/v1", "--model", "m", "--out", str(tmp_path / "o"), "--key-env", KEY_ENV])
    assert not (tmp_path / "o").exists()
