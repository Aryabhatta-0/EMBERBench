"""Evaluate any remote model, given a base URL and an API key, on the V1 public development pack.

This is the general form of the Jev and SubQ Snap runs (research lane, not a
registered baseline). It uses the same frozen environment pack, the same
unique-world manifest, the same ``run_evaluation`` replay-authenticated
results and the same official ``score_result`` / ``compare_scores`` procedure.
Only the transport changes. Three API styles are supported:

``openai`` (default)
    ``POST {base_url}/chat/completions`` with ``Authorization: Bearer``. Any
    OpenAI-compatible server (OpenAI, OpenRouter, Together, Groq, DeepSeek,
    vLLM, Ollama, LM Studio, ...). One request per step asks for one JSON
    object that maps every person ID to one allowed move.
``anthropic``
    ``POST {base_url}/messages`` with ``x-api-key`` (the Anthropic Messages API).
    The same prompt and the same JSON answer.
``decision``
    ``POST {base_url}/systemone`` with the ``{model, state, questions}`` body of
    TypeSafe Jev and SubQ Snap. The request bodies are byte-identical to the
    Jev adapter ``r1``, so a Jev or SubQ cache gives the same answers here.

Every prompt is built from the serialized :class:`~emberbench.Observation`
only (model-interface.md section 3, "future external or remote adapter"):
never the scenario, its ID, the seed or the manifest.

Reproducibility: every first request body is canonical JSON, and the accepted
responses are cached by its SHA-256 in ``<out>/cache.jsonl``. A resume or a
second run with the same cache sends no request and gives byte-identical
artifacts. A fresh cache can give other answers; the models that answered are
recorded in ``summary.json``.

Failure is fail-closed (model-interface.md section 6): an HTTP error after the
retries, a timeout, or an answer that is still not valid after
``--format-retries`` corrective re-asks raises; the episode is not committed and
``--out`` can be resumed. Only accepted answers are cached.

The key is read from the environment variable named by ``--key-env``
(default ``EMBERBENCH_API_KEY``). It is never written to a file or printed.

Usage (``--out`` must be outside the repository)::

    EMBERBENCH_API_KEY=... python tools/experiments/remote/run_remote.py \\
        --base-url https://api.openai.com/v1 --model gpt-4o-mini --out ../remote-run \\
        --reference-run ../pack-run
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from emberbench import (  # noqa: E402
    Action,
    ExecutionMetadata,
    Observation,
    Pos,
    ScoreReport,
    StepExecution,
    agent_policy_factory,
    compare_scores,
    load_environment_pack,
    load_evaluation_result,
    run_evaluation,
    score_result,
)

import pack_protocol  # noqa: E402
import run_public_benchmark as rpb  # noqa: E402

ADAPTER_VERSION = "r1"
API_STYLES = ("openai", "anthropic", "decision")
ENDPOINTS = {"openai": "/chat/completions", "anthropic": "/messages", "decision": "/systemone"}
ANTHROPIC_VERSION = "2023-06-01"
EXPECTED_PACK_DIGEST = "ca18c895f6f655f85448dd3bcd8a02a776a2c21bd28e13d5a3aa91b3955b506e"
DEFAULT_KEY_ENV = "EMBERBENCH_API_KEY"
MAX_QUESTIONS = 32  # decision style: questions per request; larger crowds are split into several requests
TIMEOUT_SECONDS = 120
RETRIES = 5  # 429, 5xx and network errors only
FORBIDDEN_WAITS = (15, 30, 60, 120)  # seconds; only with --retry-403
FORBIDDEN_RETRIES = 15  # per request; about 25 minutes in total
FORMAT_RETRIES = 2  # chat styles: corrective re-asks after an answer that is not valid
LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")

GOAL = (
    "Goal: move every person to an exit before the fire reaches them. A person who stands on a burning cell "
    "at the end of a step is a casualty. A person who reaches an exit cell evacuates. An exit lets one person "
    "through per step. Two people cannot share a cell. People still inside after the last step are stranded."
)

SYSTEM_PROMPT = (
    "You control the people in a grid fire-evacuation simulation. Every step you choose one move for every "
    "person. Answer with one JSON object only: each key is a person ID and each value is one of the allowed "
    "moves, written exactly as given (lowercase). Do not add other keys."
)


class AdapterError(RuntimeError):
    """A model answer that the adapter cannot accept, or a transport failure after the retries."""


# --------------------------------------------------------------------------- state text (Jev adapter r1)


def _cell_text(observation: Observation, pos: Pos, occupied: Mapping[Pos, str]) -> str:
    m = observation.map
    if not (0 <= pos.row < m.height and 0 <= pos.col < m.width):
        return "outside the grid, so the move is blocked"
    if pos in m.walls:
        return "a wall, so the move is blocked"
    exit_id = m.exit_positions.get(pos)
    if pos in observation.burning:
        return "a burning cell" + (f" (exit {exit_id})" if exit_id else "") + ": the person dies there"
    if exit_id is not None:
        return f"exit {exit_id}: the person evacuates"
    if pos in occupied:
        return f"occupied now by person {occupied[pos]}"
    return "open floor"


def _manhattan(a: Pos, cells: Any) -> int | None:
    return min((abs(a.row - c.row) + abs(a.col - c.col) for c in cells), default=None)


def state_text(observation: Observation) -> str:
    """The whole observation as plain text: rules, an ASCII map and the people."""
    m, rules = observation.map, observation.rules
    people = {p.position: p.id for p in observation.active}
    rows = []
    for r in range(m.height):
        line = []
        for c in range(m.width):
            pos = Pos(r, c)
            if pos in m.walls:
                ch = "#"
            elif pos in observation.burning:
                ch = "F"
            elif pos in m.exit_positions:
                ch = "E"
            elif pos in people:
                ch = "@"
            else:
                ch = "."
            line.append(ch)
        rows.append(f"{r:>3} " + "".join(line))
    lines = [
        f"EMBERBench fire evacuation. Step {observation.step}; the episode ends after step {rules.max_steps}.",
        GOAL,
        f"Fire: {rules.fire_model.value} model; it can spread to the 4 neighbour cells every {rules.spread_interval} step(s)"
        + (f" with probability {rules.spread_probability}" if rules.spread_probability is not None else "")
        + f". Walls never burn. Exits {'can' if rules.exits_flammable else 'cannot'} burn."
        + f" Entering fire: {rules.enter_fire.value}. Move conflicts: {rules.conflict_resolution.value}.",
        f"Grid: {m.width} columns x {m.height} rows. Row 0 is the north edge, column 0 the west edge."
        " north = row - 1, south = row + 1, west = column - 1, east = column + 1.",
        "Map legend: # wall, F fire, E exit, @ person, . floor.",
        *rows,
        "Exits: " + ", ".join(f"{e.id} at (row {e.position.row}, col {e.position.col})" for e in m.exits),
        "People: " + ", ".join(f"{p.id} at (row {p.position.row}, col {p.position.col})" for p in observation.active),
    ]
    return "\n".join(lines)


def move_options(observation: Observation, pos: Pos, occupied: Mapping[Pos, str]) -> dict[str, str]:
    """Every allowed action with the cell it leads to and two straight-line distances (observation facts only)."""
    exits = list(observation.map.exit_positions)
    options: dict[str, str] = {}
    for action in observation.allowed_actions:
        dest = pos.offset(*action.delta)
        if action is Action.STAY:
            where = f"stay at (row {pos.row}, col {pos.col})"
        else:
            where = f"move {action.value} to (row {dest.row}, col {dest.col}): " + _cell_text(observation, dest, occupied)
        facts = []
        d_exit, d_fire = _manhattan(dest, exits), _manhattan(dest, observation.burning)
        if d_exit is not None:
            facts.append(f"{d_exit} cells (straight line, ignoring walls) from the nearest exit")
        if d_fire is not None:
            facts.append(f"{d_fire} cells from the nearest fire")
        options[action.value] = where + ("; " + ", ".join(facts) if facts else "")
    return options


def person_question(observation: Observation, pid: str, pos: Pos, occupied: Mapping[Pos, str]) -> dict[str, Any]:
    """One decision-style ``choice`` question; the options are the allowed actions."""
    return {
        "type": "choice",
        "instructions": f"Which move should person {pid}, now at (row {pos.row}, col {pos.col}), make this step "
        "so that they reach an exit safely and soon?",
        "criteria": move_options(observation, pos, occupied),
    }


def chat_prompt(observation: Observation, people: Sequence[Any], occupied: Mapping[Pos, str]) -> str:
    """The user message of the chat styles: the state, then the options of every person in ``people``."""
    lines = [state_text(observation), "", "Choose one move for each of these people. The allowed moves and where they lead:"]
    for p in people:
        lines.append(f"Person {p.id} at (row {p.position.row}, col {p.position.col}):")
        lines.extend(f"  - {name}: {text}" for name, text in move_options(observation, p.position, occupied).items())
    example = ", ".join(f'"{p.id}": "<move>"' for p in people[:2])
    lines += [
        "",
        f"Answer with one JSON object with exactly these keys: {', '.join(p.id for p in people)}.",
        f"Each value is one of: {', '.join(a.value for a in observation.allowed_actions)}. Example: {{{example}}}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- answer parsing

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _json_objects(text: str) -> list[Any]:
    """Every top-level JSON object in ``text``, in order (code fences and reasoning blocks are skipped)."""
    text = _THINK_RE.sub("", text)
    decoder = json.JSONDecoder()
    found: list[Any] = []
    i = 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        found.append(obj)
        i = end
    return found


def parse_moves(text: str, person_ids: Sequence[str], allowed: Mapping[str, Action]) -> dict[str, Action]:
    """The moves of one chat answer, or :class:`AdapterError` with the reason.

    The answer must hold a JSON object whose keys are exactly ``person_ids`` and
    whose values are allowed action names as written. If there are several
    objects, the last one with exactly these keys is used. Nothing is repaired:
    a wrong case, a missing person or an unknown move is an error.
    """
    wanted = set(person_ids)
    candidates = [o for o in _json_objects(text) if isinstance(o, dict) and set(o) == wanted]
    if not candidates:
        objects = [o for o in _json_objects(text) if isinstance(o, dict)]
        if not objects:
            raise AdapterError("the answer holds no JSON object")
        keys = set(objects[-1])
        missing, extra = sorted(wanted - keys), sorted(keys - wanted)
        raise AdapterError(f"the JSON object must have exactly the person IDs as keys (missing {missing}, extra {extra})")
    moves: dict[str, Action] = {}
    for pid, value in candidates[-1].items():
        if not isinstance(value, str) or value not in allowed:
            raise AdapterError(f"person {pid}: {value!r} is not one of {sorted(allowed)}")
        moves[pid] = allowed[value]
    return moves


# --------------------------------------------------------------------------- client


def endpoint_url(base_url: str, api: str) -> str:
    """``base_url`` with the endpoint path of ``api`` (kept when the URL already ends with it).

    A decision-style URL that ends with ``/decisions`` (OpenRouter's ``/api/alpha/decisions``) is kept too.
    """
    base = base_url.rstrip("/")
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise SystemExit(f"--base-url must be an http(s) URL, not {base_url!r}")
    path = ENDPOINTS[api]
    if base.endswith(path) or api == "decision" and base.endswith("/decisions"):
        return base
    return base + path


def provider_label(base_url: str) -> str:
    """A short provider label from the host: ``api.openai.com`` -> ``openai``, ``localhost:8000`` -> ``localhost``."""
    host = urllib.parse.urlsplit(base_url).hostname or "remote"
    parts = [p for p in host.split(".") if p not in ("api", "www", "com", "ai", "io", "org", "net", "dev", "co")]
    return parts[0] if parts and not host.replace(".", "").isdigit() else host


def label(text: str) -> str:
    """``text`` as a model-ID and file-name part: runs of other characters become one ``_``."""
    return LABEL_RE.sub("_", text).strip("_") or "model"


@dataclass(frozen=True)
class Exchange:
    """One accepted answer: the moves, the raw responses (the last one valid) and its usage."""

    moves: dict[str, Action]
    responses: tuple[dict[str, Any], ...]
    input_tokens: int
    output_tokens: int
    models: frozenset[str]


class RemoteClient:
    """HTTP client with a content-addressed cache of accepted answers. Shared by every episode; holds no episode state."""

    def __init__(
        self,
        *,
        api: str,
        base_url: str,
        model: str,
        key: str,
        cache_path: Path,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        format_retries: int = FORMAT_RETRIES,
        timeout: float = TIMEOUT_SECONDS,
        retry_403: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if api not in API_STYLES:
            raise ValueError(f"api must be one of {API_STYLES}")
        self.api = api
        self.url = endpoint_url(base_url, api)
        self.model = model
        self._key = key
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None or api != "anthropic" else 2048
        self.extra_body = dict(extra_body or {})
        self.format_retries = format_retries
        self.timeout = timeout
        self.retry_403 = retry_403
        self._sleep = sleep
        self._cache_path = cache_path
        self._cache: dict[str, list[dict[str, Any]]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                # A decision-style row is {"request_sha256", "response"} (the Jev cache format); a chat row has "responses".
                self._cache[row["request_sha256"]] = row["responses"] if "responses" in row else [row["response"]]
        self.calls = self.cache_hits = self.retries = self.retries_403 = self.format_errors = 0
        self.input_tokens = self.output_tokens = 0
        self.cost = 0.0  # the provider's reported cost (OpenRouter: usage.cost), 0 when not reported
        self.response_seconds: list[float] = []  # one per answered HTTP request, in request order
        self.resolved_models: set[str] = set()

    # ------------------------------------------------------------ request bodies

    def chat_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model}
        if self.api == "anthropic":
            body["system"] = SYSTEM_PROMPT
            body["messages"] = messages
        else:
            body["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        return body

    def response_text(self, response: Mapping[str, Any]) -> str:
        if self.api == "anthropic":
            content = response.get("content") or []
            return "".join(str(c.get("text") or "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        choices = response.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else None
        return str((message or {}).get("content") or "")

    @staticmethod
    def usage(response: Mapping[str, Any]) -> tuple[int, int]:
        u = response.get("usage") or {}
        tokens_in = u.get("input_tokens", u.get("prompt_tokens"))
        tokens_out = u.get("output_tokens", u.get("completion_tokens"))
        return int(tokens_in or 0), int(tokens_out or 0)

    # ------------------------------------------------------------ asking

    @staticmethod
    def _digest(body: Mapping[str, Any]) -> tuple[str, str]:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest(), text

    def _store(self, digest: str, responses: list[dict[str, Any]]) -> None:
        self._cache[digest] = responses
        row: dict[str, Any] = {"request_sha256": digest}
        if self.api == "decision":
            row["response"] = responses[-1]
        else:
            row["responses"] = responses
        with self._cache_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _count(self, response: Mapping[str, Any]) -> None:
        self.calls += 1
        tokens_in, tokens_out = self.usage(response)
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.resolved_models.add(str(response.get("model")))
        self.cost += float((response.get("usage") or {}).get("cost") or 0)

    def _exchange(self, moves: dict[str, Action], responses: list[dict[str, Any]]) -> Exchange:
        totals = [self.usage(r) for r in responses]
        return Exchange(
            moves=moves,
            responses=tuple(responses),
            input_tokens=sum(t[0] for t in totals),
            output_tokens=sum(t[1] for t in totals),
            models=frozenset(str(r.get("model")) for r in responses),
        )

    def ask_decision(self, body: dict[str, Any], names: Mapping[str, str], allowed: Mapping[str, Action]) -> Exchange:
        """One decision-style request. ``names`` maps question names to person IDs."""
        digest, text = self._digest(body)
        cached = self._cache.get(digest)
        if cached is not None:
            self.cache_hits += 1
            response = cached[-1]
        else:
            response = self._post(text)
            self._count(response)
            self._store(digest, [response])  # the Jev adapter caches before it checks; kept for cache compatibility
        answers = response.get("answers") or {}
        moves: dict[str, Action] = {}
        for name, pid in names.items():
            answer = answers.get(name) or {}
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in allowed:
                raise AdapterError(f"the answer for {pid} is not an allowed action: {answer!r}")
            moves[pid] = allowed[choice]
        return self._exchange(moves, [response])

    def ask_chat(self, prompt: str, person_ids: Sequence[str], allowed: Mapping[str, Action]) -> Exchange:
        """One chat request, with up to ``format_retries`` corrective re-asks. Only an accepted answer is cached."""
        messages = [{"role": "user", "content": prompt}]
        digest, _ = self._digest(self.chat_body(messages))
        cached = self._cache.get(digest)
        if cached is not None:
            self.cache_hits += 1
            return self._exchange(parse_moves(self.response_text(cached[-1]), person_ids, allowed), cached)
        responses: list[dict[str, Any]] = []
        while True:
            _, text = self._digest(self.chat_body(messages))
            response = self._post(text)
            self._count(response)
            responses.append(response)
            answer = self.response_text(response)
            try:
                moves = parse_moves(answer, person_ids, allowed)
            except AdapterError as error:
                self.format_errors += 1
                if len(responses) > self.format_retries:
                    raise AdapterError(f"{error}; answer: {answer[:300]!r}") from None
                messages = [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"That answer is not valid: {error}. Reply with only the JSON object."},
                ]
                continue
            self._store(digest, responses)
            return self._exchange(moves, responses)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "emberbench-remote/r1"}
        if self.api == "anthropic":
            headers["x-api-key"] = self._key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        elif self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def _post(self, text: str) -> dict[str, Any]:
        request = urllib.request.Request(self.url, data=text.encode("ascii"), headers=self._headers(), method="POST")
        delay = 1.0
        attempt = forbidden = 0
        while True:
            try:
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                    body = json.loads(reply.read().decode("utf-8"))
                    if not isinstance(body, dict):
                        raise AdapterError(f"{self.url}: the response is not a JSON object")
                    # No style here streams, so the first token arrives with the whole answer: this is time to first token.
                    self.response_seconds.append(time.perf_counter() - started)
                    return body
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:500]
                if error.code == 403 and self.retry_403 and forbidden < FORBIDDEN_RETRIES:
                    # A provider that throttles with 403 (SubQ preview capacity): wait longer, then retry.
                    wait = FORBIDDEN_WAITS[min(forbidden, len(FORBIDDEN_WAITS) - 1)]
                    forbidden += 1
                    self.retries_403 += 1
                    print(f"  HTTP 403, retry {forbidden}/{FORBIDDEN_RETRIES} in {wait}s", file=sys.stderr, flush=True)
                    self._sleep(wait)
                    continue
                if error.code != 429 and error.code < 500 or attempt == RETRIES:
                    raise AdapterError(f"{self.url}: HTTP {error.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == RETRIES:
                    raise AdapterError(f"{self.url}: request failed: {error}") from None
            attempt += 1
            self.retries += 1
            self._sleep(delay)
            delay = min(delay * 2, 16.0)


class RemoteAgent:
    """One episode. ``act`` sends only the observation text; every answer must be an allowed action."""

    def __init__(self, client: RemoteClient, *, provider: str, max_people: int) -> None:
        self._client = client
        self._provider = provider
        self._max_people = max_people
        self._last: ExecutionMetadata | None = None

    def act(self, observation: Observation) -> dict[str, Action]:
        people = sorted(observation.active, key=lambda p: p.id)
        occupied = {p.position: p.id for p in people}
        allowed = {a.value: a for a in observation.allowed_actions}
        state = state_text(observation) if self._client.api == "decision" else ""
        actions: dict[str, Action] = {}
        tokens_in = tokens_out = 0
        models: set[str] = set()
        size = self._max_people or max(len(people), 1)
        for start in range(0, len(people), size):
            chunk = people[start : start + size]
            if self._client.api == "decision":
                names = {f"q{i}": p.id for i, p in enumerate(chunk)}
                body = {
                    "model": self._client.model,
                    "state": state,
                    "questions": {f"q{i}": person_question(observation, p.id, p.position, occupied) for i, p in enumerate(chunk)},
                }
                exchange = self._client.ask_decision(body, names, allowed)
            else:
                exchange = self._client.ask_chat(chat_prompt(observation, chunk, occupied), [p.id for p in chunk], allowed)
            actions.update(exchange.moves)
            tokens_in += exchange.input_tokens
            tokens_out += exchange.output_tokens
            models |= exchange.models
        self._last = ExecutionMetadata(
            provider=self._provider,
            model=",".join(sorted(models)) if models else None,
            input_tokens=tokens_in if people else None,
            output_tokens=tokens_out if people else None,
        )
        return actions

    def last_execution(self) -> ExecutionMetadata | None:
        return self._last


# --------------------------------------------------------------------------- results and report


def _f(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _pts(value: Fraction | None) -> str:
    return "-" if value is None else f"{float(value):.2f}"  # scores are on the 0-100 scale


def _signed(value: Fraction | None) -> str:
    return "-" if value is None else f"{float(value):+.2f}"


OUTCOMES = {"a_better": "model better", "b_better": "reference better", "inconclusive": "inconclusive", "exact_tie": "exact tie"}


def score_summary(report: ScoreReport) -> dict[str, Any]:
    ci = report.ci
    return {
        "score": _f(report.score),
        "ci95": None if ci is None else [_f(ci.lower), _f(ci.upper)],
        "safety": _f(report.safety),
        "quality": _f(report.quality),
        "evacuated": report.totals.evacuated,
        "avoidable_casualties": report.totals.avoidable_casualties,
        "avoidable_stranded": report.totals.avoidable_stranded,
        "unavoidable_losses": report.totals.unavoidable_losses,
        "worst_scenario": _f(report.robustness.worst_scenario),
        "worst_scenario_id": report.robustness.worst_scenario_id,
    }


def reference_results(paths: Sequence[Path], pack: Any, protocol: pack_protocol.Protocol, own_id: str) -> list[tuple[Any, ScoreReport]]:
    """Every result file in the ``results/`` of each reference run, scored again against ``pack``."""
    scored: list[tuple[Any, ScoreReport]] = []
    seen: set[str] = {own_id}
    for run in paths:
        ref_pack = load_environment_pack(run / rpb.PACK_DIR)
        if ref_pack.digest != pack.digest:
            raise SystemExit(f"{run}: the reference run used another environment pack ({ref_pack.digest})")
        for ref_path in sorted((run / "results").glob("*.json")):
            ref = load_evaluation_result(ref_path, pack.suite)  # authenticate every episode by replay
            if ref.model_id in seen or not ref.is_complete:
                continue
            seen.add(ref.model_id)
            scored.append((ref, score_result(ref, pack, resamples=protocol.resamples)))
    return scored


def render_report(summary: Mapping[str, Any], report: ScoreReport, refs: Sequence[tuple[ScoreReport, Any]]) -> str:
    """A short Markdown report: the result, the paired comparisons and the score per scenario."""
    model_id = summary["model_id"]
    ci = report.ci
    api = summary["api"]
    lines = [
        f"# `{model_id}` on the V1 public development pack",
        "",
        f"**{pack_protocol.RUN_LABEL}**",
        "",
        "Official EMBER Score procedure (scoring version 1.0) on the frozen environment pack of the **public**",
        "V1 suite: the same pack, manifest and scoring as the Jev and SubQ Snap runs. The suite is public, so",
        "these are development results only.",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Model ID (evaluator label) | `{model_id}` |",
        f"| API style / endpoint | `{summary['api_style']}` / `{summary['endpoint']}` |",
        f"| Requested model | `{summary['requested_model']}` |",
        f"| Model that answered | {', '.join(f'`{m}`' for m in summary['resolved_models']) or '- (all from the cache)'} |",
        f"| **EMBER Score (family-balanced)** | **{_pts(report.score)}**"
        + ("" if ci is None else f" (95% CI {_pts(ci.lower)} to {_pts(ci.upper)})")
        + " |",
        f"| Safety | {_f(report.safety)} |",
        f"| Quality | {_f(report.quality)} |",
        f"| Worst scenario | {_pts(report.robustness.worst_scenario)} (`{report.robustness.worst_scenario_id}`) |",
        f"| Evacuated | {report.totals.evacuated} |",
        f"| Avoidable casualties / stranded | {report.totals.avoidable_casualties} / {report.totals.avoidable_stranded} |",
        f"| Episodes (unique worlds) | {summary['episodes']} |",
        f"| Official-procedure score | {'yes' if report.official else 'no'} |",
        f"| Pack digest | `{summary['pack_digest']}` |",
        "",
        "| API item | Value |",
        "|---|---|",
        f"| `act` calls (steps) | {api['act_calls']} |",
        f"| HTTP requests | {api['requests_sent']} |",
        f"| Answers from the cache | {api['cache_hits']} |",
        f"| Transport retries (429/5xx/network, 403) | {api['retries']}, {api['retries_403']} |",
        f"| Invalid answers re-asked | {api['format_errors']} |",
        f"| Input / output tokens | {api['input_tokens']:,} / {api['output_tokens']:,} |",
        f"| Cost (USD, {api['cost_source']}) | {api['cost']} |",
        f"| Mean `act` latency | {api['mean_act_latency_s']} s |",
        f"| First / median response time | {api['first_response_s']} s / {api['median_response_s']} s |",
        f"| Wall-clock time | {api['wall_clock_s']} s |",
    ]
    if refs:
        rows = sorted([(report, None), *refs], key=lambda r: -(float(r[0].score) if r[0].score is not None else -1.0))
        lines += [
            "",
            "## Comparison with the reference runs",
            "",
            "Delta is this model minus the reference: the paired, family-balanced difference with a 95%",
            "bootstrap interval. Worlds = paired worlds where this model / the reference scored higher / equal.",
            "",
            "| Model | EMBER Score | Evacuated | Avoidable casualties | Delta | Delta 95% CI | Outcome | Worlds |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
        for r, cmp in rows:
            if cmp is None:
                lines.append(f"| **`{r.model_id}`** | **{_pts(r.score)}** | **{r.totals.evacuated}** | **{r.totals.avoidable_casualties}** | | | | |")
                continue
            span = "-" if cmp.ci is None else f"{_signed(cmp.ci.lower)} to {_signed(cmp.ci.upper)}"
            lines.append(
                f"| `{r.model_id}` | {_pts(r.score)} | {r.totals.evacuated} | {r.totals.avoidable_casualties} | {_signed(cmp.delta)} "
                f"| {span} | {OUTCOMES[cmp.outcome.value]} | {cmp.a_higher} / {cmp.b_higher} / {cmp.equal} |"
            )
    best: dict[str, tuple[Fraction, str]] = {}
    for r, _ in refs:
        for g in r.scenarios:
            if g.score is not None and (g.name not in best or g.score > best[g.name][0]):
                best[g.name] = (g.score, r.model_id)
    lines += ["", "## Score per scenario", ""]
    if refs:
        lines += ["| Scenario | Worlds | This model | Best reference | Best reference model |", "|---|---:|---:|---:|---|"]
        for g in report.scenarios:
            b_score, b_id = best.get(g.name, (None, "-"))
            lines.append(f"| `{g.name}` | {g.episodes} | {_pts(g.score)} | {_pts(b_score)} | `{b_id}` |")
    else:
        lines += ["| Scenario | Worlds | Score | Safety |", "|---|---:|---:|---:|"]
        lines += [f"| `{g.name}` | {g.episodes} | {_pts(g.score)} | {_f(g.safety)} |" for g in report.scenarios]
    lines += [
        "",
        "## Limits",
        "",
        "- Public suite only: development results, not hidden-suite or leaderboard scores.",
        f"- One prompt design (`{ADAPTER_VERSION}`). The score measures the model with this adapter, not the model alone.",
        "- One run per world. The cache makes the artifact reproducible, not the model.",
        "",
    ]
    return "\n".join(lines)


def _temperature(text: str) -> float | None:
    return None if text.lower() in ("omit", "none") else float(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", required=True, help="API base URL, for example https://api.openai.com/v1")
    parser.add_argument("--model", required=True, help="model name sent in every request")
    parser.add_argument("--out", type=Path, required=True, help="output directory, outside the repository")
    parser.add_argument("--api", choices=API_STYLES, default="openai", help="API style (default: openai-compatible chat)")
    parser.add_argument("--key-env", default=DEFAULT_KEY_ENV, help=f"environment variable that holds the API key (default {DEFAULT_KEY_ENV})")
    parser.add_argument("--provider", help="provider label in the model ID (default: from the base URL host)")
    parser.add_argument("--protocol", choices=sorted(pack_protocol.PROTOCOLS), default=pack_protocol.PUBLIC.name)
    parser.add_argument(
        "--reference-run", type=Path, action="append", default=[],
        help="a finished run directory (environment-pack/ and results/) to compare against; repeatable. "
        "A missing directory is first filled with the offline evaluation-pack run.",
    )  # fmt: skip
    parser.add_argument("--temperature", type=_temperature, default=0.0, help="chat temperature, or 'omit' (default 0)")
    parser.add_argument("--max-tokens", type=int, help="chat max_tokens (default: omitted; 2048 for anthropic)")
    parser.add_argument("--extra-body", default="{}", help="JSON object merged into every chat request body")
    parser.add_argument("--max-people", type=int, help="people per request (default: all for chat, 32 for decision)")
    parser.add_argument("--format-retries", type=int, default=FORMAT_RETRIES, help="chat re-asks after an invalid answer")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="seconds per HTTP request")
    parser.add_argument("--retry-403", action="store_true", help="retry HTTP 403 with long waits (a provider that throttles with 403)")
    parser.add_argument("--input-price", type=float, help="USD per million input tokens, used when the API reports no cost")
    parser.add_argument("--output-price", type=float, help="USD per million output tokens, used when the API reports no cost")
    args = parser.parse_args(argv)

    key = os.environ.get(args.key_env, "")
    if not key and urllib.parse.urlsplit(args.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"set the API key in the environment variable {args.key_env}")
    extra_body = json.loads(args.extra_body)
    if not isinstance(extra_body, dict):
        raise SystemExit("--extra-body must be a JSON object")
    max_people = args.max_people if args.max_people is not None else (MAX_QUESTIONS if args.api == "decision" else 0)

    protocol = pack_protocol.PROTOCOLS[args.protocol]
    rpb.check_benchmark_versions()
    out = rpb._check_out_dir(args.out)
    suite = rpb.protocol_suite(protocol)
    manifest = rpb.protocol_manifest(protocol, suite)
    pack = rpb._environment_pack(protocol, suite, manifest, out)
    suite, manifest = pack.suite, pack.manifest
    if protocol is pack_protocol.PUBLIC and pack.digest != EXPECTED_PACK_DIGEST:
        raise SystemExit(f"environment pack digest {pack.digest} is not the released {EXPECTED_PACK_DIGEST}")
    print(f"pack     {pack.pack_id} sha256:{pack.digest}: {len(manifest.episodes)} episodes", flush=True)

    for run in args.reference_run:
        if not run.exists():
            print(f"reference {run}: running the offline evaluation pack first", flush=True)
            rpb.run_pack(protocol, run)

    client = RemoteClient(
        api=args.api,
        base_url=args.base_url,
        model=args.model,
        key=key,
        cache_path=out / "cache.jsonl",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        format_retries=args.format_retries,
        timeout=args.timeout,
        retry_403=args.retry_403,
    )
    provider = label(args.provider or provider_label(args.base_url))
    kind = "choice" if args.api == "decision" else "chat"
    model_id = f"{provider}-{label(args.model)}-{kind}-{ADAPTER_VERSION}"
    print(f"model    {model_id} via {client.url}", flush=True)
    steps: list[StepExecution] = []
    path = out / "results" / f"{model_id}.json"
    path.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    factory: Callable[[], RemoteAgent] = lambda: RemoteAgent(client, provider=provider, max_people=max_people)  # noqa: E731
    run_evaluation(suite, manifest, model_id, agent_policy_factory(factory, on_step=steps.append), path, resume=True)
    elapsed = time.perf_counter() - started
    result = load_evaluation_result(path, suite)  # authenticate every episode by replay
    if not result.is_complete:
        raise SystemExit(f"{path}: incomplete")
    report = score_result(result, pack, resamples=protocol.resamples)
    (out / "scores").mkdir(exist_ok=True)
    (out / "scores" / f"{model_id}.json").write_text(report.to_json() + "\n", encoding="ascii", newline="\n")

    refs = [(r, compare_scores(report, r, resamples=protocol.resamples)) for _, r in reference_results(args.reference_run, pack, protocol, model_id)]
    cost, cost_source = client.cost, "reported"
    if cost == 0 and (args.input_price is not None or args.output_price is not None):
        cost = (client.input_tokens * (args.input_price or 0) + client.output_tokens * (args.output_price or 0)) / 1e6
        cost_source = "list price"
    elif cost == 0:
        cost_source = "not reported"
    summary: dict[str, Any] = {
        "model_id": model_id,
        "api_style": args.api,
        "endpoint": client.url,
        "requested_model": args.model,
        "resolved_models": sorted(client.resolved_models),
        "adapter_version": ADAPTER_VERSION,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "protocol": protocol.name,
        "pack_digest": pack.digest,
        "manifest_hash": manifest.manifest_hash(),
        "episodes": len(result.episodes),
        "official_procedure": report.official,
        "result": score_summary(report),
        "scenarios": {g.name: _f(g.score) for g in report.scenarios},
        "api": {
            "requests_sent": client.calls,
            "cache_hits": client.cache_hits,
            "retries": client.retries,
            "retries_403": client.retries_403,
            "format_errors": client.format_errors,
            "input_tokens": client.input_tokens,
            "output_tokens": client.output_tokens,
            "cost": round(cost, 6),
            "cost_source": cost_source,
            "act_calls": len(steps),
            "mean_act_latency_s": _f(sum(s.latency_seconds for s in steps) / len(steps), 3) if steps else None,
            "first_response_s": round(client.response_seconds[0], 3) if client.response_seconds else None,
            "median_response_s": round(statistics.median(client.response_seconds), 3) if client.response_seconds else None,
            "wall_clock_s": round(elapsed, 1),
        },
        "references": {
            r.model_id: {
                **score_summary(r),
                "delta": _f(cmp.delta),
                "delta_ci95": None if cmp.ci is None else [_f(cmp.ci.lower), _f(cmp.ci.upper)],
                "outcome": cmp.outcome.value,
                "model_higher": cmp.a_higher,
                "ref_higher": cmp.b_higher,
                "equal": cmp.equal,
            }
            for r, cmp in refs
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii", newline="\n")
    (out / "REPORT.md").write_text(render_report(summary, report, refs), encoding="utf-8", newline="\n")
    ci = report.ci
    print(
        f"score    {_pts(report.score)}" + ("" if ci is None else f" (95% CI {_pts(ci.lower)} to {_pts(ci.upper)})")
        + f"  safety {_f(report.safety)}  quality {_f(report.quality)}  evacuated {report.totals.evacuated}"
        + f"  avoidable casualties {report.totals.avoidable_casualties}",
        flush=True,
    )  # fmt: skip
    for r, cmp in refs:
        print(f"  vs {r.model_id:<40} {_pts(r.score):>6}  delta {_signed(cmp.delta):>7}  {OUTCOMES[cmp.outcome.value]}", flush=True)
    print(f"report   {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
