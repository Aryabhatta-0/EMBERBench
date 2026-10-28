"""The model harness: how a remote LLM or decision model plays an EMBERBench episode.

The harness turns each :class:`~emberbench.Observation` into one or more model
requests, validates every answer and returns the joint action of the step. It
is the canonical runner behind ``emberbench run`` and
``tools/experiments/remote/run_remote.py`` (see docs/harness-protocol.md).

Three API styles are supported:

``openai``
    ``POST {base_url}/chat/completions`` with ``Authorization: Bearer``. One
    request asks for one JSON object that maps every person ID to one allowed move.
``anthropic``
    ``POST {base_url}/messages`` with ``x-api-key``; the same prompt and answer.
``decision``
    ``POST {base_url}/systemone`` (or ``.../decisions``) with the
    ``{model, state, questions}`` body: one named ``choice`` question per
    active person, whose options are the allowed actions.

The joint protocol (:class:`HarnessProtocol`) decides how the active people of
one step are split into requests: ``full_joint`` puts all of them in one
request; ``fixed_chunk`` splits them, in person-ID order, into consecutive
chunks of ``chunk_size``. Every prompt is built from the observation only:
never the scenario, its ID, the seed or the manifest.

Invalid output is never repaired. A chat answer that is not valid is re-asked
in the same conversation; for the decision style only the invalid questions
are re-asked with the same state. After ``max_reasks`` re-asks the request
fails closed with :class:`AdapterError`: the episode is not committed and the
run can be resumed.

Accounting: every HTTP request (live or replayed from the cache) yields a
:class:`RequestRecord` with tokens, cost, latency and invalid-slot counts.
Decision-model option probabilities are summarized per slot
(:func:`probability_diagnostics`); they never change the executed action,
which is the model's ``choice``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .agent import ExecutionMetadata
from .errors import EpisodeRefused
from .grid import Action, Pos
from .observation import Observation

HARNESS_VERSION = "1.1"  # the joint protocol, validation, re-ask, refusal and fail-closed rules of this module
ADAPTER_VERSION = "r1"  # the prompt design: state text, questions, chat prompt and system prompt
API_STYLES = ("openai", "anthropic", "decision")
ENDPOINTS = {"openai": "/chat/completions", "anthropic": "/messages", "decision": "/systemone"}
ANTHROPIC_VERSION_HEADER = "2023-06-01"
DEFAULT_KEY_ENV = "EMBERBENCH_API_KEY"
DEFAULT_CHUNK_SIZE = 32  # the historical decision-style cap: questions per request
TIMEOUT_SECONDS = 120
RETRIES = 5  # 429, 5xx and network errors only; never a refusal
# The OpenRouter decisions API reports a refusal of one question (by the model or its provider's policy)
# as an error whose message reads '... refused to answer question "qN"'.
_REFUSAL = re.compile(r"\brefused to answer question\W+(q\d+)\b")
FORBIDDEN_WAITS = (15, 30, 60, 120)  # seconds; only with retry_403
FORBIDDEN_RETRIES = 15  # per request; about 25 minutes in total
MAX_REASKS = 2  # corrective re-asks after an invalid answer
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


def refused_question(body: str) -> str | None:
    """The question that an HTTP error ``body`` reports as refused (see ``_REFUSAL``), or ``None``."""
    try:
        message = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None
    match = _REFUSAL.search(message) if isinstance(message, str) else None
    return match.group(1) if match else None


class AdapterError(RuntimeError):
    """A model answer that the harness cannot accept, or a transport failure after the retries."""


# --------------------------------------------------------------------------- joint protocol


class JointMode(StrEnum):
    FULL_JOINT = "full_joint"  # every active person of the step in one request
    FIXED_CHUNK = "fixed_chunk"  # consecutive chunks of chunk_size people, in person-ID order


@dataclass(frozen=True)
class HarnessProtocol:
    """How the active people of one step become model requests, and the invalid-output policy."""

    joint_mode: JointMode = JointMode.FULL_JOINT
    chunk_size: int | None = None
    max_reasks: int = MAX_REASKS

    def __post_init__(self) -> None:
        if not isinstance(self.joint_mode, JointMode):
            raise ValueError(f"joint_mode must be one of {[m.value for m in JointMode]}")
        if self.joint_mode is JointMode.FIXED_CHUNK:
            if not isinstance(self.chunk_size, int) or isinstance(self.chunk_size, bool) or self.chunk_size < 1:
                raise ValueError("fixed_chunk needs a chunk_size >= 1")
        elif self.chunk_size is not None:
            raise ValueError("full_joint takes no chunk_size")
        if not isinstance(self.max_reasks, int) or isinstance(self.max_reasks, bool) or self.max_reasks < 0:
            raise ValueError("max_reasks must be an integer >= 0")

    @property
    def label(self) -> str:
        return "full-joint" if self.joint_mode is JointMode.FULL_JOINT else f"chunk{self.chunk_size}"

    def chunks(self, people: Sequence[Any]) -> list[list[Any]]:
        """``people`` (already in person-ID order) as the requests of one step; none for no people."""
        size = len(people) if self.chunk_size is None else self.chunk_size
        return [list(people[i : i + size]) for i in range(0, len(people), max(size, 1))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_version": HARNESS_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "joint_mode": self.joint_mode.value,
            "chunk_size": self.chunk_size,
            "person_order": "person_id",
            "max_reasks": self.max_reasks,
            "executed_action": "choice",
        }


FULL_JOINT = HarnessProtocol()
CHUNK_32 = HarnessProtocol(JointMode.FIXED_CHUNK, DEFAULT_CHUNK_SIZE)


# --------------------------------------------------------------------------- state text (adapter r1)


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


def invalid_chat_slots(text: str, person_ids: Sequence[str], allowed: Mapping[str, Action]) -> int:
    """The invalid slots of a rejected chat answer: people without an allowed move in its last JSON object (at least 1)."""
    objects = [o for o in _json_objects(text) if isinstance(o, dict)]
    if not objects:
        return len(person_ids)
    last = objects[-1]
    bad = sum(1 for pid in person_ids if not (isinstance(last.get(pid), str) and last.get(pid) in allowed))
    return max(bad, 1)


def probability_diagnostics(answer: Mapping[str, Any], options: Sequence[str]) -> dict[str, Any] | None:
    """Diagnostics of one decision answer's option probabilities, or ``None`` when it reports none.

    ``valid`` is false when the probabilities are not a mapping from exactly the
    offered options to non-negative numbers with a positive sum. For a valid
    distribution (normalized when its sum is not 1): the probability of the
    chosen option, the entropy in bits, the margin between the two largest
    probabilities and whether the choice is an argmax. These are descriptive
    diagnostics, not a calibration measure: no target label is involved.
    """
    probs = answer.get("probabilities")
    if probs is None:
        return None
    choice = answer.get("choice")
    values: dict[str, float] = {}
    ok = isinstance(probs, Mapping) and set(probs) == set(options)
    if ok:
        for key in options:
            v = probs[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                ok = False
                break
            values[key] = float(v)
    total = sum(values.values())
    if not ok or total <= 0 or choice not in values:
        return {"valid": False}
    p = {k: v / total for k, v in values.items()}
    ranked = sorted(p.values(), reverse=True)
    top = ranked[0]
    return {
        "valid": True,
        "sum": round(total, 6),
        "p_choice": round(p[choice], 6),
        "entropy_bits": round(-sum(x * math.log2(x) for x in p.values() if x > 0), 6),
        "margin": round(top - (ranked[1] if len(ranked) > 1 else 0.0), 6),
        "argmax_agrees": p[choice] >= top - 1e-12,
    }


# --------------------------------------------------------------------------- client


def endpoint_url(base_url: str, api: str) -> str:
    """``base_url`` with the endpoint path of ``api`` (kept when the URL already ends with it).

    A decision-style URL that ends with ``/decisions`` (OpenRouter's ``/api/alpha/decisions``) is kept too.
    """
    base = base_url.rstrip("/")
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"the base URL must be an http(s) URL, not {base_url!r}")
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


def remote_model_id(provider: str, model: str, api: str, protocol: HarnessProtocol) -> str:
    """``<provider>-<model>-<choice|chat>-r1``, plus ``-<label>`` for a declared non-canonical joint protocol."""
    kind = "choice" if api == "decision" else "chat"
    model_id = f"{label(provider)}-{label(model)}-{kind}-{ADAPTER_VERSION}"
    return model_id if protocol.joint_mode is JointMode.FULL_JOINT else f"{model_id}-{protocol.label}"


@dataclass(frozen=True)
class RequestRecord:
    """One HTTP request of an exchange, live or replayed from the cache."""

    attempt: int  # 0 for the first request, then 1, 2, ... for re-asks
    questions: int  # slots asked in this request
    invalid_slots: int  # slots without a valid answer in this request's response
    input_tokens: int
    output_tokens: int
    cost: float | None  # as reported by the API; None when not reported
    latency_s: float | None  # the measured response time; None for an old cache row without it
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "questions": self.questions,
            "invalid_slots": self.invalid_slots,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_s": None if self.latency_s is None else round(self.latency_s, 4),
            "cached": self.cached,
        }


@dataclass(frozen=True)
class Exchange:
    """One accepted answer: the moves, the raw responses (the last one valid) and its accounting."""

    moves: dict[str, Action]
    responses: tuple[dict[str, Any], ...]
    input_tokens: int
    output_tokens: int
    models: frozenset[str]
    requests: tuple[RequestRecord, ...] = ()
    probabilities: tuple[dict[str, Any], ...] = ()


def usage(response: Mapping[str, Any]) -> tuple[int, int]:
    u = response.get("usage") or {}
    tokens_in = u.get("input_tokens", u.get("prompt_tokens"))
    tokens_out = u.get("output_tokens", u.get("completion_tokens"))
    return int(tokens_in or 0), int(tokens_out or 0)


def reported_cost(response: Mapping[str, Any]) -> float | None:
    value = (response.get("usage") or {}).get("cost")
    return None if value is None else float(value)


class RemoteClient:
    """HTTP client with a content-addressed cache of answers. Shared by every episode; holds no episode state.

    With ``redact`` true, error messages carry no model answer and no provider
    error body, so a private suite's state text cannot reach a log through them.
    """

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
        format_retries: int = MAX_REASKS,
        timeout: float = TIMEOUT_SECONDS,
        retry_403: bool = False,
        redact: bool = False,
        budget_usd: float | None = None,
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
        self.redact = redact
        self.budget_usd = budget_usd  # stop (fail closed, resumable) once the reported cost reaches it
        self._sleep = sleep
        self._cache_path = cache_path
        self._cache: dict[str, tuple[list[dict[str, Any]], list[float | None]]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                # A decision-style row is {"request_sha256", "response"} (the Jev cache format); a chat row has "responses".
                responses = row["responses"] if "responses" in row else [row["response"]]
                latency = row.get("latency_s")
                latencies = latency if isinstance(latency, list) else [latency] * len(responses)
                self._cache[row["request_sha256"]] = (responses, latencies)
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
        return usage(response)

    # ------------------------------------------------------------ asking

    @staticmethod
    def _digest(body: Mapping[str, Any]) -> tuple[str, str]:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest(), text

    def _store(self, digest: str, responses: list[dict[str, Any]], latencies: list[float | None]) -> None:
        self._cache[digest] = (responses, latencies)
        row: dict[str, Any] = {"request_sha256": digest}
        if self.api == "decision":
            row["response"] = responses[-1]
            row["latency_s"] = latencies[-1]
        else:
            row["responses"] = responses
            row["latency_s"] = latencies
        with self._cache_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def _count(self, response: Mapping[str, Any]) -> None:
        self.calls += 1
        tokens_in, tokens_out = usage(response)
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.resolved_models.add(str(response.get("model")))
        self.cost += reported_cost(response) or 0.0

    @staticmethod
    def _record(response: Mapping[str, Any], *, attempt: int, questions: int, invalid: int, latency: float | None, cached: bool) -> RequestRecord:
        tokens_in, tokens_out = usage(response)
        return RequestRecord(attempt, questions, invalid, tokens_in, tokens_out, reported_cost(response), latency, cached)

    def _exchange(self, moves: dict[str, Action], responses: list[dict[str, Any]], records: list[RequestRecord], probs: list[dict[str, Any]]) -> Exchange:
        totals = [usage(r) for r in responses]
        return Exchange(
            moves=moves,
            responses=tuple(responses),
            input_tokens=sum(t[0] for t in totals),
            output_tokens=sum(t[1] for t in totals),
            models=frozenset(str(r.get("model")) for r in responses),
            requests=tuple(records),
            probabilities=tuple(probs),
        )

    def _decision_response(self, body: Mapping[str, Any], attempt: int) -> tuple[dict[str, Any], float | None, bool]:
        if self.extra_body:  # routing fields (OpenRouter ``provider``); none keeps the r1 cache keys
            body = {**body, **self.extra_body}
        digest, text = self._digest(body)
        if attempt > 0:
            # A re-ask is a new model request even when its body equals the first one (every question
            # was invalid), so it gets its own cache key; first attempts keep the r1 key.
            digest = hashlib.sha256(f"{digest}/reask/{attempt}".encode("ascii")).hexdigest()
        cached = self._cache.get(digest)
        if cached is not None:
            self.cache_hits += 1
            return cached[0][-1], cached[1][-1], True
        response, latency = self._post(text)
        self._count(response)
        self._store(digest, [response], [latency])  # cached before it is checked, as the Jev adapter did
        return response, latency, False

    def ask_decision(self, body: dict[str, Any], names: Mapping[str, str], allowed: Mapping[str, Action]) -> Exchange:
        """One decision-style request. ``names`` maps question names to person IDs.

        Questions whose answer is not an allowed ``choice`` are re-asked (only
        those, with the same state) up to ``format_retries`` times; then the
        request fails closed.
        """
        questions: Mapping[str, Any] = body["questions"]
        pending = dict(names)
        moves: dict[str, Action] = {}
        responses: list[dict[str, Any]] = []
        records: list[RequestRecord] = []
        probs: list[dict[str, Any]] = []
        attempt = 0
        while True:
            request = body if attempt == 0 else {**body, "questions": {n: questions[n] for n in pending}}
            response, latency, cached = self._decision_response(request, attempt)
            responses.append(response)
            answers = response.get("answers") or {}
            invalid: list[str] = []
            for name, pid in pending.items():
                answer = answers.get(name) if isinstance(answers, Mapping) else None
                choice = answer.get("choice") if isinstance(answer, Mapping) else None
                if not isinstance(choice, str) or choice not in allowed:
                    invalid.append(name)
                    continue
                moves[pid] = allowed[choice]
                assert isinstance(answer, Mapping)
                diag = probability_diagnostics(answer, list(questions[name]["criteria"]))
                if diag is not None:
                    probs.append(diag)
            records.append(self._record(response, attempt=attempt, questions=len(pending), invalid=len(invalid), latency=latency, cached=cached))
            if not invalid:
                return self._exchange(moves, responses, records, probs)
            self.format_errors += 1
            if attempt >= self.format_retries:
                detail = "" if self.redact else f": {answers.get(invalid[0]) if isinstance(answers, Mapping) else answers!r}"
                raise AdapterError(f"{len(invalid)} answer(s) still not an allowed action after {attempt} re-ask(s){detail}")
            pending = {n: pending[n] for n in invalid}
            attempt += 1

    def ask_chat(self, prompt: str, person_ids: Sequence[str], allowed: Mapping[str, Action]) -> Exchange:
        """One chat request, with up to ``format_retries`` corrective re-asks. Only an accepted answer is cached."""
        messages = [{"role": "user", "content": prompt}]
        digest, _ = self._digest(self.chat_body(messages))
        cached = self._cache.get(digest)
        if cached is not None:
            self.cache_hits += 1
            responses, cached_latencies = cached
            records = []
            for i, (response, latency) in enumerate(zip(responses, cached_latencies, strict=True)):
                last = i == len(responses) - 1
                bad = 0 if last else invalid_chat_slots(self.response_text(response), person_ids, allowed)
                records.append(self._record(response, attempt=i, questions=len(person_ids), invalid=bad, latency=latency, cached=True))
            return self._exchange(parse_moves(self.response_text(responses[-1]), person_ids, allowed), responses, records, [])
        responses = []
        latencies: list[float | None] = []
        records = []
        while True:
            _, text = self._digest(self.chat_body(messages))
            response, latency = self._post(text)
            self._count(response)
            responses.append(response)
            latencies.append(latency)
            answer = self.response_text(response)
            try:
                moves = parse_moves(answer, person_ids, allowed)
            except AdapterError as error:
                self.format_errors += 1
                bad = invalid_chat_slots(answer, person_ids, allowed)
                records.append(self._record(response, attempt=len(records), questions=len(person_ids), invalid=bad, latency=latency, cached=False))
                if len(responses) > self.format_retries:
                    detail = "" if self.redact else f"; answer: {answer[:300]!r}"
                    raise AdapterError(f"{'invalid answer' if self.redact else error}{detail}") from None
                messages = [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": f"That answer is not valid: {error}. Reply with only the JSON object."},
                ]
                continue
            records.append(self._record(response, attempt=len(records), questions=len(person_ids), invalid=0, latency=latency, cached=False))
            self._store(digest, responses, latencies)
            return self._exchange(moves, responses, records, [])

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "emberbench-remote/r1"}
        if self.api == "anthropic":
            headers["x-api-key"] = self._key
            headers["anthropic-version"] = ANTHROPIC_VERSION_HEADER
        elif self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def _post(self, text: str) -> tuple[dict[str, Any], float]:
        if self.budget_usd is not None and self.cost >= self.budget_usd:
            raise AdapterError(f"the spend budget of {self.budget_usd} USD is reached (reported cost {self.cost:.6f})")
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
                    elapsed = time.perf_counter() - started
                    self.response_seconds.append(elapsed)
                    return body, elapsed
            except urllib.error.HTTPError as error:
                raw = error.read().decode("utf-8", "replace")
                question = refused_question(raw)
                if question is not None:
                    # A refusal is the model's answer, not a transient failure: no retry, no re-prompt.
                    raise EpisodeRefused(f"{self.url}: the model refused question {question} (HTTP {error.code})", question=question) from None
                detail = "" if self.redact else ": " + raw[:500]
                if error.code == 403 and self.retry_403 and forbidden < FORBIDDEN_RETRIES:
                    # A provider that throttles with 403 (SubQ preview capacity): wait longer, then retry.
                    wait = FORBIDDEN_WAITS[min(forbidden, len(FORBIDDEN_WAITS) - 1)]
                    forbidden += 1
                    self.retries_403 += 1
                    print(f"  HTTP 403, retry {forbidden}/{FORBIDDEN_RETRIES} in {wait}s", file=sys.stderr, flush=True)
                    self._sleep(wait)
                    continue
                if error.code != 429 and error.code < 500 or attempt == RETRIES:
                    raise AdapterError(f"{self.url}: HTTP {error.code}{detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == RETRIES:
                    raise AdapterError(f"{self.url}: request failed: {'network error' if self.redact else error}") from None
            attempt += 1
            self.retries += 1
            self._sleep(delay)
            delay = min(delay * 2, 16.0)


# --------------------------------------------------------------------------- agent and ledger


@dataclass
class Ledger:
    """Machine-readable accounting: one JSON line per accepted ``act`` call.

    ``episodes`` maps the agent ordinal of this run segment to the manifest
    episode index (a resumed run skips completed episodes). Rows carry the
    segment label, so a reader keeps, per episode, the rows of the last segment
    that ran it.
    """

    path: Path | None
    segment: str
    episodes: list[int]
    _next: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def next_episode(self) -> int:
        index = self.episodes[self._next] if self._next < len(self.episodes) else -1
        self._next += 1
        return index

    def write(self, row: dict[str, Any]) -> None:
        row = {"segment": self.segment, **row}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


class RemoteAgent:
    """One episode. ``act`` sends only the observation text; every answer must be an allowed action."""

    def __init__(
        self,
        client: RemoteClient,
        *,
        provider: str,
        protocol: HarnessProtocol = FULL_JOINT,
        max_people: int | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        if max_people is not None:  # the historical knob: 0 is full joint, n is fixed chunks of n
            mode, size = (JointMode.FULL_JOINT, None) if max_people == 0 else (JointMode.FIXED_CHUNK, max_people)
            protocol = HarnessProtocol(mode, size, protocol.max_reasks)
        self._client = client
        self._provider = provider
        self.protocol = protocol
        self._ledger = ledger
        self._episode = ledger.next_episode() if ledger is not None else -1
        self._last: ExecutionMetadata | None = None

    def act(self, observation: Observation) -> dict[str, Action]:
        people = sorted(observation.active, key=lambda p: p.id)
        occupied = {p.position: p.id for p in people}
        allowed = {a.value: a for a in observation.allowed_actions}
        state = state_text(observation) if self._client.api == "decision" else ""
        actions: dict[str, Action] = {}
        tokens_in = tokens_out = 0
        models: set[str] = set()
        requests: list[dict[str, Any]] = []
        probs: list[dict[str, Any]] = []
        for chunk in self.protocol.chunks(people):
            try:
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
            except EpisodeRefused as refusal:
                refusal.step = observation.step
                raise
            actions.update(exchange.moves)
            tokens_in += exchange.input_tokens
            tokens_out += exchange.output_tokens
            models |= exchange.models
            requests.extend(r.to_dict() for r in exchange.requests)
            probs.extend(exchange.probabilities)
        self._last = ExecutionMetadata(
            provider=self._provider,
            model=",".join(sorted(models)) if models else None,
            input_tokens=tokens_in if people else None,
            output_tokens=tokens_out if people else None,
        )
        if self._ledger is not None:
            self._ledger.write({"episode": self._episode, "step": observation.step, "people": len(people), "requests": requests, "probabilities": probs})
        return actions

    def last_execution(self) -> ExecutionMetadata | None:
        return self._last
