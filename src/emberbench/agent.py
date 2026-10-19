"""The provider-independent model interface: ``Observation -> {person_id: Action}``.

Every evaluated decision model (a planner, an LLM, an RL agent or a future
remote adapter) runs through the same boundary:

- It is built fresh for one episode by a zero-argument :data:`AgentFactory`.
- :meth:`Agent.act` receives one :class:`~emberbench.observation.Observation`
  per timestep and returns one action for every active person.
- It is discarded after the episode. It is never reused.

The model receives only the observation. It never receives the scenario, the
evaluator seed, the evaluation manifest, an episode index, a run record, the
simulator state, RNG state or hidden metadata.

Model output is validated with the simulator's own action rules
(:func:`emberbench.movement.parse_actions`). Invalid output raises
:class:`~emberbench.errors.ModelOutputError`. It is never repaired. An
exception raised by the model propagates unchanged. Both stop the evaluation.

:class:`ExecutionMetadata` and :class:`StepExecution` carry optional,
evaluator-side accounting (latency, token counts, labels, cost). They never
reach the simulator, the run record or the result artifact.

:func:`serialize_observation` gives the canonical JSON text of an observation
for a future external adapter. It holds only the model-visible fields.

The core has no runtime dependencies and no async or network API. See
``docs/model-interface.md``.
"""

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .canonical import canonical_json
from .errors import EvaluationError, InvalidActionError, ModelOutputError, ObservationFormatError
from .grid import Action, Pos
from .movement import parse_actions
from .observation import OBSERVATION_VERSION, MapView, Observation, RulesView
from .scenario import ConflictResolution, EnterFirePolicy, ExitSpec, FireModelName
from .simulator import Policy
from .state import PersonState, PersonStatus

# --------------------------------------------------------------------------- interface


@runtime_checkable
class Agent(Protocol):
    """A decision model for one episode.

    ``act`` is called once per timestep, in step order, with only the
    model-visible observation. It must return a mapping with exactly one
    action for every active person (``observation.active``). Values are
    :class:`~emberbench.grid.Action` members or their string values, as the
    simulator accepts.
    """

    def act(self, observation: Observation) -> Mapping[str, Action]: ...


# Called once per episode with no arguments. It must return a new agent.
AgentFactory = Callable[[], Agent]


class PolicyAgent:
    """Adapt a plain ``Observation -> actions`` callable to :class:`Agent`.

    The callable is shared by every agent that wraps it, so it must be stateless.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: Policy) -> None:
        if not callable(policy):
            raise TypeError("policy must be a callable that maps an Observation to an action mapping")
        self._policy = policy

    def act(self, observation: Observation) -> Mapping[str, Any]:
        return self._policy(observation)


# --------------------------------------------------------------------------- validation


def validate_actions(observation: Observation, actions: Any) -> dict[str, Action]:
    """Check model output against ``observation`` with the simulator's action rules.

    The output must be a mapping with exactly one valid action for every
    active person: no missing, unknown or inactive IDs, no other values.
    Returns a new ``dict`` in person-ID order. Raises :class:`ModelOutputError`.
    Nothing is repaired, dropped or defaulted.
    """
    active_ids = frozenset(p.id for p in observation.active)
    try:
        parsed = parse_actions(actions, active_ids)
    except InvalidActionError as exc:
        raise ModelOutputError(f"step {observation.step}: {exc}") from exc
    disallowed = sorted(pid for pid, action in parsed.items() if action not in observation.allowed_actions)
    if disallowed:
        raise ModelOutputError(f"step {observation.step}: actions not in allowed_actions for {disallowed}")
    return parsed


# --------------------------------------------------------------------------- execution accounting


def _check_label(value: Any, where: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise EvaluationError(f"{where}: must be a non-empty string or None, got {value!r}")


def _check_count(value: Any, where: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise EvaluationError(f"{where}: must be a non-negative integer or None, got {value!r}")


def _check_amount(value: Any, where: str) -> None:
    if value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise EvaluationError(f"{where}: must be a finite non-negative number or None, got {value!r}")


@dataclass(frozen=True)
class ExecutionMetadata:
    """Accounting that an adapter reports for one ``act`` call. Every field is optional.

    ``None`` means unknown. An adapter must report only values that its
    provider returned. It must not estimate them. ``cost`` needs a
    ``cost_currency`` (for example ``"USD"``), and the other way round.
    These values are evaluator-side only. They never change simulator state
    or any evacuation outcome.
    """

    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    cost_currency: str | None = None

    def __post_init__(self) -> None:
        _check_label(self.provider, "provider")
        _check_label(self.model, "model")
        _check_count(self.input_tokens, "input_tokens")
        _check_count(self.output_tokens, "output_tokens")
        _check_amount(self.cost, "cost")
        _check_label(self.cost_currency, "cost_currency")
        if (self.cost is None) != (self.cost_currency is None):
            raise EvaluationError("cost and cost_currency must be given together")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "cost_currency": self.cost_currency,
        }


@runtime_checkable
class ReportsExecution(Protocol):
    """Optional: an agent that reports accounting for its most recent ``act`` call."""

    def last_execution(self) -> ExecutionMetadata | None: ...


@dataclass(frozen=True)
class StepExecution:
    """Evaluator-side record of one accepted ``act`` call.

    ``latency_seconds`` is the wall-clock time of the call, measured by the
    evaluator. ``reported`` is what the agent reported, or ``None``.
    """

    step: int
    latency_seconds: float
    reported: ExecutionMetadata | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "latency_seconds": self.latency_seconds,
            "reported": None if self.reported is None else self.reported.to_dict(),
        }


StepObserver = Callable[[StepExecution], None]
Clock = Callable[[], float]


def _reported(agent: Agent) -> ExecutionMetadata | None:
    if not isinstance(agent, ReportsExecution):
        return None
    reported = agent.last_execution()
    if reported is not None and not isinstance(reported, ExecutionMetadata):
        raise EvaluationError(f"last_execution() must return ExecutionMetadata or None, got {type(reported).__name__}")
    return reported


# --------------------------------------------------------------------------- evaluation adapters


def agent_policy(agent: Agent, *, on_step: StepObserver | None = None, clock: Clock = time.perf_counter) -> Policy:
    """Bind ``agent`` to one episode and return a simulator :data:`Policy`.

    Each call passes the observation to ``agent.act`` and validates the result
    with :func:`validate_actions`. The policy accepts the steps of one episode
    only, in order from step 0. A second episode raises :class:`EvaluationError`.

    ``on_step``, if given, receives a :class:`StepExecution` after each
    accepted call. Accounting is read only then, and never changes the actions.
    """
    act = getattr(agent, "act", None)
    if not callable(act):
        raise TypeError("agent must have an act(observation) method")
    expected_step = 0

    def policy(observation: Observation) -> dict[str, Action]:
        nonlocal expected_step
        if observation.step != expected_step:
            raise EvaluationError(
                f"agent is bound to one episode: expected step {expected_step}, got step {observation.step}"
            )
        started = clock()
        raw = act(observation)
        latency = clock() - started
        actions = validate_actions(observation, raw)
        expected_step += 1
        if on_step is not None:
            on_step(StepExecution(observation.step, latency, _reported(agent)))
        return actions

    return policy


def agent_policy_factory(
    agent_factory: AgentFactory, *, on_step: StepObserver | None = None, clock: Clock = time.perf_counter
) -> Callable[[], Policy]:
    """Turn an :data:`AgentFactory` into a policy factory for ``run_evaluation``.

    Every call builds a new agent with ``agent_factory()`` (no arguments) and
    binds it to one episode with :func:`agent_policy`. A factory that returns
    the agent of the previous episode again raises :class:`EvaluationError`.
    """
    if not callable(agent_factory):
        raise TypeError("agent_factory must be a zero-argument callable that returns a new agent")
    previous: list[Agent] = []  # only the agent of the last episode, to detect reuse

    def factory() -> Policy:
        agent = agent_factory()
        if not isinstance(agent, Agent) or not callable(agent.act):
            raise EvaluationError(f"agent_factory returned {type(agent).__name__}, not an agent with an act(observation) method")
        if previous and agent is previous[0]:
            raise EvaluationError("agent_factory returned the agent of the previous episode; it must build a new agent")
        previous[:] = [agent]
        return agent_policy(agent, on_step=on_step, clock=clock)

    return factory


# --------------------------------------------------------------------------- serialization

OBSERVATION_KEYS = frozenset({"observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"})
MAP_KEYS = frozenset({"width", "height", "obstacles", "exits"})
EXIT_KEYS = frozenset({"id", "position"})
RULES_KEYS = frozenset(
    {"fire_model", "spread_interval", "spread_probability", "exits_flammable", "enter_fire", "conflict_resolution", "max_steps"}
)
PERSON_KEYS = frozenset({"id", "position", "status"})


def serialize_observation(observation: Observation) -> str:
    """The canonical JSON text of ``observation``: sorted keys, no whitespace, ASCII only.

    It is :meth:`Observation.to_dict` and nothing else, so it holds only
    model-visible fields. Equal observations give equal text.
    """
    if type(observation) is not Observation:
        raise TypeError(f"expected an Observation, got {type(observation).__name__}")
    return canonical_json(observation.to_dict())


def deserialize_observation(text: str) -> Observation:
    """Parse canonical observation text. Text that is not exactly canonical is rejected."""
    if not isinstance(text, str):
        raise ObservationFormatError(f"expected str, got {type(text).__name__}")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ObservationFormatError(f"invalid JSON: {exc}") from exc
    observation = observation_from_dict(data)
    if serialize_observation(observation) != text:
        raise ObservationFormatError("observation text is not in canonical form")
    return observation


def _fail(where: str, message: str) -> ObservationFormatError:
    return ObservationFormatError(f"{where}: {message}")


def _object(value: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    if value.keys() != keys:
        missing = sorted(keys - value.keys())
        unknown = sorted(value.keys() - keys)
        raise _fail(where, f"keys must be exactly {sorted(keys)}; missing={missing} unknown={unknown}")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(where, f"expected a list, got {type(value).__name__}")
    return value


def _int(value: Any, where: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(where, f"expected an integer >= {minimum}, got {value!r}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, f"expected a non-empty string, got {value!r}")
    return value


def _pos(value: Any, where: str, width: int, height: int) -> Pos:
    items = _list(value, where)
    if len(items) != 2:
        raise _fail(where, f"position must be [row, col], got {value!r}")
    pos = Pos(_int(items[0], f"{where}[0]", 0), _int(items[1], f"{where}[1]", 0))
    if pos.row >= height or pos.col >= width:
        raise _fail(where, f"position {pos.to_list()} is outside the {width}x{height} grid")
    return pos


def _enum[E: StrEnum](cls: type[E], value: Any, where: str) -> E:
    if not isinstance(value, str) or value not in cls._value2member_map_:
        raise _fail(where, f"expected one of {[m.value for m in cls]}, got {value!r}")
    return cls(value)


def _strictly_sorted(items: list[Any], where: str) -> None:
    if any(a >= b for a, b in zip(items, items[1:])):
        raise _fail(where, "items must be unique and in canonical (sorted) order")


def observation_from_dict(data: Any) -> Observation:
    """Strictly parse the :meth:`Observation.to_dict` form. Raises :class:`ObservationFormatError`."""
    obj = _object(data, OBSERVATION_KEYS, "observation")
    if obj["observation_version"] != OBSERVATION_VERSION:
        raise _fail("observation_version", f"unsupported version {obj['observation_version']!r}; expected {OBSERVATION_VERSION!r}")

    m = _object(obj["map"], MAP_KEYS, "map")
    width = _int(m["width"], "map.width", 1)
    height = _int(m["height"], "map.height", 1)
    obstacles = [_pos(p, f"map.obstacles[{i}]", width, height) for i, p in enumerate(_list(m["obstacles"], "map.obstacles"))]
    _strictly_sorted(obstacles, "map.obstacles")
    exits = []
    for i, e in enumerate(_list(m["exits"], "map.exits")):
        e = _object(e, EXIT_KEYS, f"map.exits[{i}]")
        exits.append(ExitSpec(_str(e["id"], f"map.exits[{i}].id"), _pos(e["position"], f"map.exits[{i}].position", width, height)))
    _strictly_sorted([e.id for e in exits], "map.exits")

    r = _object(obj["rules"], RULES_KEYS, "rules")
    fire_model = _enum(FireModelName, r["fire_model"], "rules.fire_model")
    probability = r["spread_probability"]
    if fire_model is FireModelName.STOCHASTIC:
        if type(probability) is not float or not 0.0 <= probability <= 1.0:
            raise _fail("rules.spread_probability", f"expected a float in [0, 1] for a stochastic fire, got {probability!r}")
    elif probability is not None:
        raise _fail("rules.spread_probability", f"must be null unless fire_model is stochastic, got {probability!r}")
    if not isinstance(r["exits_flammable"], bool):
        raise _fail("rules.exits_flammable", f"expected a boolean, got {r['exits_flammable']!r}")
    rules = RulesView(
        fire_model=fire_model,
        spread_interval=_int(r["spread_interval"], "rules.spread_interval", 1),
        spread_probability=probability,
        exits_flammable=r["exits_flammable"],
        enter_fire=_enum(EnterFirePolicy, r["enter_fire"], "rules.enter_fire"),
        conflict_resolution=_enum(ConflictResolution, r["conflict_resolution"], "rules.conflict_resolution"),
        max_steps=_int(r["max_steps"], "rules.max_steps", 1),
    )

    people = []
    for i, p in enumerate(_list(obj["people"], "people")):
        p = _object(p, PERSON_KEYS, f"people[{i}]")
        people.append(
            PersonState(
                _str(p["id"], f"people[{i}].id"),
                _pos(p["position"], f"people[{i}].position", width, height),
                _enum(PersonStatus, p["status"], f"people[{i}].status"),
            )
        )
    _strictly_sorted([p.id for p in people], "people")
    burning = [_pos(p, f"burning[{i}]", width, height) for i, p in enumerate(_list(obj["burning"], "burning"))]
    _strictly_sorted(burning, "burning")
    allowed = [_enum(Action, a, f"allowed_actions[{i}]") for i, a in enumerate(_list(obj["allowed_actions"], "allowed_actions"))]
    if len(set(allowed)) != len(allowed):
        raise _fail("allowed_actions", "actions must be unique")

    return Observation(
        step=_int(obj["step"], "step", 0),
        map=MapView(width=width, height=height, obstacles=tuple(obstacles), exits=tuple(exits)),
        rules=rules,
        people=tuple(people),
        burning=frozenset(burning),
        allowed_actions=tuple(allowed),
    )
