"""Deterministic, non-LLM reference baseline agents (baselines version 1.0).

Reference baselines are anchors, not benchmark winners. They give fixed
points of comparison that make benchmark difficulty readable: a model that
does no better than ``random-legal-v1`` or ``static-shortest-path-v1`` on a
family has learned little that a simple rule does not already do.

Every baseline here is a *fair* model. It uses the public interface only:

- It is built by a zero-argument :data:`~emberbench.agent.AgentFactory` from
  its public configuration, fresh for every episode.
- :meth:`act` receives one :class:`~emberbench.observation.Observation` and
  returns one action per active person.
- It never receives the scenario, the evaluator seed, the evaluation
  manifest, a run record, simulator state, RNG state, future fire, the
  scenario ID or hash, or generation metadata.

Every decision is a pure function of the observation and the baseline
configuration. No baseline keeps state between steps, reads the clock, the
OS or a global RNG, or iterates over a hash-ordered container in a way that
could change its output.

Shared tie-breaking (all baselines):

- Moves are tried in the order of ``observation.allowed_actions`` without
  ``stay``: north, south, east, west in V1. The first move that meets the
  rule wins.
- Equal exits: the exit with the lowest exit ID (plain string order) wins.
- Each person is decided independently, except in ``congestion-aware-v1``,
  which assigns exits in order of distance, then person ID. Output never
  depends on the order of people in the observation or on any dict or set
  iteration order.

The privileged, evaluator-side escape bound is in :mod:`emberbench.oracle`.
It is not an agent and is not in this registry. See ``docs/baselines.md``.
"""

import hashlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .agent import Agent, AgentFactory
from .canonical import canonical_bytes, sha256_hex
from .errors import BaselineError
from .grid import Action, Pos, in_bounds, neighbours
from .observation import Observation

# Independent of every other version. Bump it when any baseline's actions can
# change for the same observation and configuration, or when the registry or
# configuration format changes.
BASELINES_VERSION = "1.0"

# Upper bound for integer configuration values (the random seed range).
_MAX_U64 = (1 << 64) - 1

# Domain separator for the random-legal draws. It differs from every
# simulator stream label, so a baseline seed never shares draws with the
# evaluator even when the two integers are equal.
_RANDOM_DOMAIN = "emberbench/baseline/random-legal/1"

_INF = float("inf")


# --------------------------------------------------------------------------- shared grid helpers


def _moves(observation: Observation) -> tuple[Action, ...]:
    return tuple(a for a in observation.allowed_actions if a is not Action.STAY)


def _target(pos: Pos, action: Action) -> Pos:
    return pos.offset(*action.delta)


def _bfs(source: Pos, blocked: frozenset[Pos], width: int, height: int) -> dict[Pos, int]:
    """Unit-cost distance from every reachable cell to ``source``. ``blocked`` cells are never entered."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for nb in neighbours(cell, width, height):
            if nb not in dist and nb not in blocked:
                dist[nb] = dist[cell] + 1
                queue.append(nb)
    return dist


def _exit_fields(
    observation: Observation, blocked: frozenset[Pos], *, skip_burning: bool
) -> tuple[tuple[str, dict[Pos, int]], ...]:
    """One unit-cost distance field per exit, sorted by exit ID."""
    m = observation.map
    exits = sorted(m.exits, key=lambda e: e.id)
    return tuple(
        (e.id, _bfs(e.position, blocked, m.width, m.height))
        for e in exits
        if not (skip_burning and e.position in observation.burning)
    )


def _nearest(fields: Iterable[tuple[str, Mapping[Pos, int]]], pos: Pos) -> Mapping[Pos, int] | None:
    """The field of the exit with the lowest cost from ``pos`` (ties: lowest exit ID), or ``None``."""
    best: Mapping[Pos, int] | None = None
    best_cost: float = _INF
    for _exit_id, field in fields:  # already in exit-ID order, so ``<`` keeps the lowest ID on ties
        cost = field.get(pos, _INF)
        if cost < best_cost:
            best, best_cost = field, cost
    return best


def _descend(observation: Observation, pos: Pos, field: Mapping[Pos, int]) -> Action:
    """The first move that lowers the unit-cost distance by one."""
    here = field[pos]
    for action in _moves(observation):
        if field.get(_target(pos, action)) == here - 1:
            return action
    return Action.STAY  # unreachable for a BFS field; kept as a safe default


def _heat(observation: Observation) -> dict[Pos, int]:
    """Wall-only distance from every cell to the nearest burning cell (multi-source BFS)."""
    m = observation.map
    walls = m.walls
    heat: dict[Pos, int] = {}
    queue = deque(sorted(observation.burning))
    for cell in queue:
        heat[cell] = 0
    while queue:
        cell = queue.popleft()
        for nb in neighbours(cell, m.width, m.height):
            if nb not in heat and nb not in walls:
                heat[nb] = heat[cell] + 1
                queue.append(nb)
    return heat


def _flee(observation: Observation, pos: Pos, heat: Mapping[Pos, int]) -> Action:
    """Fallback with no safe route: move to maximise the wall-only distance to the nearest burning cell.

    ``heat`` is :func:`_heat` of the same observation. Candidates are
    ``stay`` and every move into an in-bounds, non-wall, non-burning cell.
    Ties go to ``stay``, then to the move order. With no burning cell every
    candidate ties, so the person stays.
    """
    m = observation.map
    walls = m.walls
    best, best_heat = Action.STAY, heat.get(pos, _INF)
    for action in _moves(observation):
        nb = _target(pos, action)
        if not in_bounds(nb, m.width, m.height) or nb in walls or nb in observation.burning:
            continue
        if heat.get(nb, _INF) > best_heat:
            best, best_heat = action, heat.get(nb, _INF)
    return best


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


# --------------------------------------------------------------------------- configuration


class BaselineConfig(Protocol):
    """A baseline's public, immutable configuration. It is model configuration, never evaluator data."""

    def to_dict(self) -> dict[str, Any]: ...


def _expect_keys(data: object, keys: frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise BaselineError(f"{where}: config must be an object, got {type(data).__name__}")
    if set(data) != keys:
        raise BaselineError(f"{where}: config keys must be exactly {sorted(keys)}, got {sorted(map(str, data))}")
    return data


def _expect_int(value: object, low: int, high: int, where: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BaselineError(f"{where}: must be an integer in [{low}, {high}], got {value!r}")
    return value


@dataclass(frozen=True)
class NoConfig:
    """Configuration of a baseline that has no parameters."""

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: object) -> "NoConfig":
        _expect_keys(data, frozenset(), "config")
        return cls()


@dataclass(frozen=True)
class RandomLegalConfig:
    """``seed`` is a public baseline parameter in ``[0, 2**64 - 1]``. It is not an evaluator seed."""

    seed: int = 0

    def __post_init__(self) -> None:
        _expect_int(self.seed, 0, _MAX_U64, "seed")

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    @classmethod
    def from_dict(cls, data: object) -> "RandomLegalConfig":
        d = _expect_keys(data, frozenset({"seed"}), "config")
        return cls(seed=d["seed"])


# --------------------------------------------------------------------------- baselines


class RandomLegalAgent:
    """``random-legal-v1``: a uniform choice among the statically legal actions of each person.

    The legal set is ``stay`` plus every move into an in-bounds, non-wall
    cell, in ``allowed_actions`` order. Fire and people are ignored. The
    choice is drawn from SHA-256 of ``(domain, seed, step, person ID)``, so it
    depends only on the public baseline seed and the observation.
    """

    __slots__ = ("_seed",)

    def __init__(self, config: RandomLegalConfig) -> None:
        self._seed = config.seed

    def act(self, observation: Observation) -> dict[str, Action]:
        m = observation.map
        walls = m.walls
        actions: dict[str, Action] = {}
        for person in observation.active:
            legal = [
                a
                for a in observation.allowed_actions
                if a is Action.STAY or (in_bounds(_target(person.position, a), m.width, m.height) and _target(person.position, a) not in walls)
            ]
            actions[person.id] = legal[_uniform_index(self._seed, observation.step, person.id, len(legal))]
        return actions


def _uniform_index(seed: int, step: int, person_id: str, n: int) -> int:
    """An exactly uniform index in ``[0, n)`` by rejection sampling over 64-bit SHA-256 words."""
    limit = (1 << 64) - (1 << 64) % n
    counter = 0
    while True:
        digest = hashlib.sha256(canonical_bytes([_RANDOM_DOMAIN, seed, step, person_id, counter])).digest()
        word = int.from_bytes(digest[:8], "big")
        if word < limit:
            return word % n
        counter += 1


class NearestExitGreedyAgent:
    """``nearest-exit-greedy-v1``: pick the nearest exit, then beeline toward it.

    The exit is the one with the shortest wall-only path from the person
    (ties: lowest exit ID). The person then takes the first move into an
    in-bounds, non-wall cell that lowers the straight-line (Manhattan)
    distance to that exit, or stays if no such move exists. Fire and people
    are ignored. Intentionally weak: it stalls behind any wall that lies
    between the person and the exit.
    """

    __slots__ = ()

    def __init__(self, config: NoConfig) -> None:
        pass

    def act(self, observation: Observation) -> dict[str, Action]:
        m = observation.map
        walls = m.walls
        fields = _exit_fields(observation, walls, skip_burning=False)
        exit_pos = {e.id: e.position for e in m.exits}
        actions: dict[str, Action] = {}
        for person in observation.active:
            pos = person.position
            action = Action.STAY
            best_id: str | None = None
            best_cost = _INF
            for exit_id, field in fields:
                if field.get(pos, _INF) < best_cost:
                    best_id, best_cost = exit_id, field[pos]
            if best_id is not None:
                goal = exit_pos[best_id]
                here = _manhattan(pos, goal)
                for move in _moves(observation):
                    nb = _target(pos, move)
                    if in_bounds(nb, m.width, m.height) and nb not in walls and _manhattan(nb, goal) < here:
                        action = move
                        break
            actions[person.id] = action
        return actions


class StaticShortestPathAgent:
    """``static-shortest-path-v1``: follow a shortest wall-only path to the nearest exit.

    Distances come from one breadth-first search per exit over non-wall
    cells. Fire and people are ignored, including burning exits. A person
    with no wall-only route to any exit stays.
    """

    __slots__ = ()

    def __init__(self, config: NoConfig) -> None:
        pass

    def act(self, observation: Observation) -> dict[str, Action]:
        fields = _exit_fields(observation, observation.map.walls, skip_burning=False)
        actions: dict[str, Action] = {}
        for person in observation.active:
            field = _nearest(fields, person.position)
            actions[person.id] = Action.STAY if field is None else _descend(observation, person.position, field)
        return actions


class FireAwareShortestPathAgent:
    """``fire-aware-shortest-path-v1``: follow a shortest path that avoids the cells burning now.

    Currently burning cells are impassable and currently burning exits are
    unusable (V1 rules R5 and R6). Nothing about future fire is predicted.
    With no currently safe route the person falls back to :func:`_flee`.
    """

    __slots__ = ()

    def __init__(self, config: NoConfig) -> None:
        pass

    def act(self, observation: Observation) -> dict[str, Action]:
        blocked = observation.map.walls | observation.burning
        fields = _exit_fields(observation, blocked, skip_burning=True)
        actions: dict[str, Action] = {}
        heat: dict[Pos, int] | None = None  # built once, only if someone has no safe route
        for person in observation.active:
            field = _nearest(fields, person.position)
            if field is None:
                heat = _heat(observation) if heat is None else heat
                actions[person.id] = _flee(observation, person.position, heat)
            else:
                actions[person.id] = _descend(observation, person.position, field)
        return actions


class CongestionAwareAgent:
    """``congestion-aware-v1``: fire-aware routing with current exit load balancing.

    It uses the fire-aware distance fields (burning cells impassable,
    burning exits unusable). People are then assigned to exits one at a
    time, in order of their nearest-exit distance (ties: person ID). Each
    person takes the exit with the lowest estimated evacuation step
    ``max(distance, load + 1)``, where ``load`` is the number of people
    already assigned to that exit, because an exit lets one person through
    per step (V1 rule R4). Ties go to the shorter distance, then to the
    lowest exit ID. The person then takes a shortest step toward that exit,
    preferring a target cell that no active person occupies now; the move
    order breaks remaining ties. People with no currently safe route use
    the fire-aware fallback.

    The load is a count of people in the current observation. It is not a
    model of human crowd behaviour, and it does not use evaluator-side
    crowd diagnostics.
    """

    __slots__ = ()

    def __init__(self, config: NoConfig) -> None:
        pass

    def act(self, observation: Observation) -> dict[str, Action]:
        blocked = observation.map.walls | observation.burning
        fields = _exit_fields(observation, blocked, skip_burning=True)
        occupied = frozenset(p.position for p in observation.active)
        actions: dict[str, Action] = {}
        queue: list[tuple[int, str, Pos]] = []
        heat: dict[Pos, int] | None = None  # built once, only if someone has no safe route
        for person in observation.active:
            field = _nearest(fields, person.position)
            if field is None:
                heat = _heat(observation) if heat is None else heat
                actions[person.id] = _flee(observation, person.position, heat)
            else:
                queue.append((field[person.position], person.id, person.position))
        load = [0] * len(fields)
        for _distance, person_id, pos in sorted(queue):
            choice, best = 0, (_INF, _INF)
            for index, (_exit_id, field) in enumerate(fields):  # exit-ID order: ``<`` keeps the lowest ID on ties
                if pos in field:
                    key = (max(field[pos], load[index] + 1), field[pos])
                    if key < best:
                        choice, best = index, key
            load[choice] += 1
            actions[person_id] = _descend_free(observation, pos, fields[choice][1], occupied)
        return {pid: actions[pid] for pid in sorted(actions)}


def _descend_free(observation: Observation, pos: Pos, field: Mapping[Pos, int], occupied: frozenset[Pos]) -> Action:
    """A move that lowers the distance by one, preferring a cell that is free now."""
    here = field[pos]
    downhill = [a for a in _moves(observation) if field.get(_target(pos, a)) == here - 1]
    for action in downhill:
        if _target(pos, action) not in occupied:
            return action
    return downhill[0] if downhill else Action.STAY


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class BaselineSpec:
    """One registered fair baseline: identity, version, description, default config and builder.

    ``build`` is called with a validated config and must return a new agent.
    """

    baseline_id: str
    baseline_version: str
    description: str
    default_config: BaselineConfig
    parse_config: Callable[[object], BaselineConfig]
    build: Callable[[Any], Agent]
    privileged: bool = False

    def _config(self, config: BaselineConfig | None) -> BaselineConfig:
        if config is None:
            return self.default_config
        if type(config) is not type(self.default_config):
            raise BaselineError(
                f"{self.baseline_id}: config must be {type(self.default_config).__name__}, got {type(config).__name__}"
            )
        return config

    def descriptor(self, config: BaselineConfig | None = None) -> dict[str, Any]:
        """The canonical, JSON-ready description of this baseline under ``config``."""
        return {
            "baselines_version": BASELINES_VERSION,
            "baseline_id": self.baseline_id,
            "baseline_version": self.baseline_version,
            "privileged": self.privileged,
            "config": self._config(config).to_dict(),
        }

    def config_digest(self, config: BaselineConfig | None = None) -> str:
        """SHA-256 of the canonical JSON of :meth:`descriptor`."""
        return sha256_hex(self.descriptor(config))

    def factory(self, config: BaselineConfig | None = None) -> AgentFactory:
        """A zero-argument factory that builds a new agent on every call."""
        cfg = self._config(config)
        build = self.build

        def make() -> Agent:
            return build(cfg)

        return make


_REGISTRY: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        baseline_id="random-legal-v1",
        baseline_version="1.0",
        description="Uniform seeded choice among stay and moves into in-bounds non-wall cells.",
        default_config=RandomLegalConfig(),
        parse_config=RandomLegalConfig.from_dict,
        build=RandomLegalAgent,
    ),
    BaselineSpec(
        baseline_id="nearest-exit-greedy-v1",
        baseline_version="1.0",
        description="Beeline (Manhattan) toward the exit with the shortest wall-only path; ignores fire and people.",
        default_config=NoConfig(),
        parse_config=NoConfig.from_dict,
        build=NearestExitGreedyAgent,
    ),
    BaselineSpec(
        baseline_id="static-shortest-path-v1",
        baseline_version="1.0",
        description="Shortest wall-only path to the nearest exit, replanned each step; ignores fire and people.",
        default_config=NoConfig(),
        parse_config=NoConfig.from_dict,
        build=StaticShortestPathAgent,
    ),
    BaselineSpec(
        baseline_id="fire-aware-shortest-path-v1",
        baseline_version="1.0",
        description="Shortest path avoiding currently burning cells and exits; flees the fire if no safe route exists.",
        default_config=NoConfig(),
        parse_config=NoConfig.from_dict,
        build=FireAwareShortestPathAgent,
    ),
    BaselineSpec(
        baseline_id="congestion-aware-v1",
        baseline_version="1.0",
        description="Fire-aware shortest paths with exits assigned by current load (one person per exit per step).",
        default_config=NoConfig(),
        parse_config=NoConfig.from_dict,
        build=CongestionAwareAgent,
    ),
)

# Read-only: the registry is a fixed table, never filled from strings at runtime.
BASELINES: Mapping[str, BaselineSpec] = MappingProxyType({spec.baseline_id: spec for spec in _REGISTRY})
BASELINE_IDS: tuple[str, ...] = tuple(spec.baseline_id for spec in _REGISTRY)


def get_baseline(baseline_id: object) -> BaselineSpec:
    """Look up a registered fair baseline. An unknown ID raises :class:`BaselineError`."""
    if not isinstance(baseline_id, str) or baseline_id not in BASELINES:
        raise BaselineError(f"unknown baseline {baseline_id!r}; known: {list(BASELINE_IDS)}")
    return BASELINES[baseline_id]


def baseline_factory(baseline_id: str, config: BaselineConfig | Mapping[str, Any] | None = None) -> AgentFactory:
    """A zero-argument agent factory for a registered baseline.

    ``config`` may be the baseline's config object, its ``to_dict`` form
    (parsed strictly) or ``None`` for the default.
    """
    spec = get_baseline(baseline_id)
    if isinstance(config, Mapping):
        config = spec.parse_config(config)
    return spec.factory(config)
