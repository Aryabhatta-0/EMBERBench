"""Fair research agents of the V1 evaluation pack (not registered baselines).

Every agent here is *fair*: a zero-argument factory builds a new agent per
episode, and ``act`` receives one :class:`~emberbench.Observation` and nothing
else. The agents use only model-visible data: the static map, the episode
rules (``observation.rules``), the people and the currently burning cells.
They never read the scenario, the evaluator seed, the manifest, a run record
or any RNG. None of them is an oracle: the privileged escape bound in
``emberbench.oracle`` is the only upper-bound reference, and the pack reports
it separately.

Agents
------

``always-stay``
    Test control. Every active person stays.

``forecast-shortest-path-r1``
    Independent per-person planner. It forecasts the earliest step at which
    every cell *can* burn from the observation and the public fire rules, and
    takes the shortest path to an exit that no forecast fire closes before
    the person passes. It ignores other people.

``cooperative-forecast-whca-r1``
    Cooperative receding-horizon planner: forecast-safe exit slots assigned by
    a queue model, then prioritized windowed space-time A* (WHCA*) with
    vertex, swap and exit-throughput reservations. It replans every step.

Fire forecast
-------------
``ignite[c]`` is the first step number whose fire phase can ignite cell
``c`` (0 for a cell that burns now, ``None`` for a cell that cannot burn).
Fire spreads only through flammable cells (walls never burn; exits only when
``rules.exits_flammable``), one neighbour layer per spread step, and spread
steps are the steps ``s`` with ``s % spread_interval == 0``. For the
deterministic model the forecast is exact. For the stochastic model it is the
earliest possible ignition (every edge succeeds), so it is pessimistic. A
second, *stretched* tier gives each layer ``ceil(1 / p)`` spread steps (the
expected wait of one edge). A third tier uses the current fire only. A planner
uses the first tier that leaves the person a route.

With the V1 phase order a person who ends step ``t`` on a non-exit cell ``c``
survives only if ``t < ignite[c]``, and a person who enters exit ``e`` at
step ``t`` evacuates only if ``e`` is not burning at the start of the step,
that is ``t <= ignite[e]``. Fire never depends on people, so waiting never
helps a person who is alone; a breadth-first search in arrival time therefore
finds the earliest forecast-safe arrival.
"""

import hashlib
import heapq
import json
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emberbench import Action, Agent, FireModelName, Observation, Pos, baseline_factory

PACK_ID = "emberbench-v1-evaluation-pack"
PACK_AGENTS_VERSION = "1"

ALWAYS_STAY_ID = "always-stay"
FORECAST_ID = "forecast-shortest-path-r1"
COOPERATIVE_ID = "cooperative-forecast-whca-r1"

# Short labels for tables and file names (Windows paths are limited to 260 characters).
SHORT_NAMES = {
    ALWAYS_STAY_ID: "stay",
    "random-legal-v1": "random",
    "nearest-exit-greedy-v1": "greedy",
    "static-shortest-path-v1": "static",
    "fire-aware-shortest-path-v1": "fire-aware",
    "congestion-aware-v1": "congestion",
    FORECAST_ID: "forecast",
    COOPERATIVE_ID: "coop-whca",
}

_INF = math.inf
Ignite = Mapping[Pos, int | None]  # missing key: the cell can never burn


# --------------------------------------------------------------------------- grid helpers


def _moves(observation: Observation) -> tuple[Action, ...]:
    return tuple(a for a in observation.allowed_actions if a is not Action.STAY)


def _step(pos: Pos, action: Action) -> Pos:
    return pos.offset(*action.delta)


def _open(observation: Observation, pos: Pos, walls: frozenset[Pos]) -> bool:
    m = observation.map
    return 0 <= pos.row < m.height and 0 <= pos.col < m.width and pos not in walls


def _wall_field(observation: Observation, source: Pos, walls: frozenset[Pos], exits: frozenset[Pos]) -> dict[Pos, int]:
    """Wall-only BFS distance to ``source``. Other exits are terminal: a person evacuates on them."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        if cell != source and cell in exits:
            continue
        for action in _moves(observation):
            nb = _step(cell, action)
            if nb not in dist and _open(observation, nb, walls):
                dist[nb] = dist[cell] + 1
                queue.append(nb)
    return dist


# --------------------------------------------------------------------------- fire forecast


def _forecast_tiers(observation: Observation) -> tuple[dict[Pos, int], ...]:
    """The forecast tiers described in the module docstring, most cautious first, without duplicates."""
    rules = observation.rules
    stretches: list[float] = [1]
    if rules.fire_model is FireModelName.STOCHASTIC:
        p = rules.spread_probability
        stretch = math.inf if not p else float(math.ceil(1 / p))
        if stretch != 1:
            stretches.append(stretch)
    tiers: list[dict[Pos, int]] = []
    for stretch in stretches:
        tiers.append(_spread_forecast(observation, stretch))
    tiers.append({cell: 0 for cell in observation.burning})  # current fire only
    unique: list[dict[Pos, int]] = []
    for tier in tiers:
        if tier not in unique:
            unique.append(tier)
    return tuple(unique)


def _spread_forecast(observation: Observation, stretch: float) -> dict[Pos, int]:
    """``ignite`` when each burning layer takes ``stretch`` spread steps (``inf``: no spread)."""
    m, rules = observation.map, observation.rules
    walls = m.walls
    exits = frozenset(m.exit_positions)
    ignite: dict[Pos, int] = {cell: 0 for cell in observation.burning}
    if stretch == math.inf:
        return ignite
    interval = rules.spread_interval
    first = (observation.step // interval + 1) * interval  # first spread step after now
    layer = sorted(observation.burning)
    depth = 0
    while layer:
        depth += 1
        when = first + (int(stretch) * depth - 1) * interval
        nxt: list[Pos] = []
        for cell in layer:
            for action in _moves(observation):
                nb = _step(cell, action)
                if nb in ignite or not _open(observation, nb, walls):
                    continue
                if nb in exits and not rules.exits_flammable:
                    continue
                ignite[nb] = when
                nxt.append(nb)
        layer = sorted(nxt)
    return ignite


def _can_stand(ignite: Ignite, cell: Pos, t: int) -> bool:
    when = ignite.get(cell)
    return when is None or t < when


def _can_evacuate(ignite: Ignite, exit_cell: Pos, t: int) -> bool:
    when = ignite.get(exit_cell)
    return when is None or t <= when


def _arrival_search(
    observation: Observation, start: Pos, ignite: Ignite, walls: frozenset[Pos], exits: Mapping[Pos, str]
) -> tuple[dict[str, int], dict[str, Action]]:
    """Earliest forecast-safe arrival step at every exit, and the first move of that path.

    Breadth-first in arrival time from ``start`` at ``observation.step``.
    Moves are expanded in ``allowed_actions`` order, so equal paths are
    broken by the first move. Exits are terminal.
    """
    now = observation.step
    arrival: dict[str, int] = {}
    first: dict[str, Action] = {}
    seen = {start}
    queue: deque[tuple[Pos, int, Action | None]] = deque([(start, now, None)])
    while queue:
        cell, t, head = queue.popleft()
        for action in _moves(observation):
            nb = _step(cell, action)
            if nb in seen or not _open(observation, nb, walls):
                continue
            move = head if head is not None else action
            exit_id = exits.get(nb)
            if exit_id is not None:
                if _can_evacuate(ignite, nb, t + 1) and exit_id not in arrival:
                    arrival[exit_id] = t + 1
                    first[exit_id] = move
                continue
            if not _can_stand(ignite, nb, t + 1):
                continue
            seen.add(nb)
            queue.append((nb, t + 1, move))
    return arrival, first


@dataclass(frozen=True)
class _Route:
    tier: int
    ignite: Ignite
    arrival: dict[str, int]
    first: dict[str, Action]


def _route(
    observation: Observation,
    start: Pos,
    tiers: tuple[dict[Pos, int], ...],
    walls: frozenset[Pos],
    exits: Mapping[Pos, str],
) -> _Route | None:
    """The forecast-safe routes of the most cautious tier that leaves any route, or ``None``."""
    for index, ignite in enumerate(tiers):
        arrival, first = _arrival_search(observation, start, ignite, walls, exits)
        if arrival:
            return _Route(index, ignite, arrival, first)
    return None


def _best_exit(route: _Route) -> str:
    return min(route.arrival, key=lambda exit_id: (route.arrival[exit_id], exit_id))


def _flee(observation: Observation, pos: Pos, walls: frozenset[Pos], ignite: Ignite) -> Action:
    """No route at all: take the candidate cell that the forecast ignites last (ties: stay, then move order)."""

    def safety(cell: Pos) -> float:
        when = ignite.get(cell)
        return _INF if when is None else when

    best, best_safety = Action.STAY, safety(pos)
    for action in _moves(observation):
        nb = _step(pos, action)
        if not _open(observation, nb, walls) or nb in observation.burning:
            continue
        if safety(nb) > best_safety:
            best, best_safety = action, safety(nb)
    return best


# --------------------------------------------------------------------------- agents


class AlwaysStayAgent:
    """Control: every active person stays."""

    def act(self, observation: Observation) -> dict[str, Action]:
        return {p.id: Action.STAY for p in observation.active}


class ForecastShortestPathAgent:
    """Independent forecast-safe shortest paths (``forecast-shortest-path-r1``)."""

    def act(self, observation: Observation) -> dict[str, Action]:
        walls = observation.map.walls
        exits = observation.map.exit_positions
        tiers = _forecast_tiers(observation)
        actions: dict[str, Action] = {}
        for person in sorted(observation.active, key=lambda p: p.id):
            route = _route(observation, person.position, tiers, walls, exits)
            if route is None:
                actions[person.id] = _flee(observation, person.position, walls, tiers[0])
            else:
                actions[person.id] = route.first[_best_exit(route)]
        return actions


@dataclass(frozen=True)
class CooperativeConfig:
    """Public configuration of ``cooperative-forecast-whca-r1``."""

    window: int = 10  # reservation window W in steps
    max_expansions: int = 4000  # A* node budget per person and step

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window, "max_expansions": self.max_expansions}


class CooperativeForecastAgent:
    """Forecast-safe exit slots plus prioritized windowed space-time A* (``cooperative-forecast-whca-r1``).

    Per step:

    1. Each person gets forecast-safe arrival times to every exit (the first
       forecast tier that leaves a route).
    2. People are sorted by their earliest arrival, then ID. In that order each
       person takes the exit with the earliest free *slot* ``s >= arrival``
       (an exit lets one person through per step, rule R4) that the forecast
       keeps open (``s <= ignite[exit]``); ties go to the earlier arrival, then
       the lower exit ID. Slots are the priority order for planning.
    3. In slot order, each person runs A* over ``(cell, t)`` for
       ``t <= now + W``: moves and waits, forecast-safe, never into a cell
       or exit another planned person holds at ``t``, never swapping with a
       planned person, and never at ``now + 1`` into the current cell of a
       person who is not planned yet. The goal is any forecast-open exit; at
       the window end the cost to go is the wall-only distance to the assigned
       exit. The plan is reserved.
    4. Without a plan the person makes the forecast shortest-path move if its
       target is free, else stays; with no route at all the person flees.

    The static wall-only distance fields are cached per episode (a new agent
    is built for every episode). Every choice has an explicit tie-break, so
    the actions never depend on hash or dict order.
    """

    def __init__(self, config: CooperativeConfig) -> None:
        self._config = config
        self._fields: dict[Pos, dict[Pos, int]] | None = None

    def _wall_fields(self, observation: Observation) -> dict[Pos, dict[Pos, int]]:
        if self._fields is None:
            walls = observation.map.walls
            exits = frozenset(observation.map.exit_positions)
            self._fields = {e.position: _wall_field(observation, e.position, walls, exits) for e in observation.map.exits}
        return self._fields

    def act(self, observation: Observation) -> dict[str, Action]:
        m = observation.map
        walls = m.walls
        exits = m.exit_positions
        exit_pos = {e.id: e.position for e in m.exits}
        fields = self._wall_fields(observation)
        tiers = _forecast_tiers(observation)
        now = observation.step
        people = sorted(observation.active, key=lambda p: p.id)
        routes = {p.id: _route(observation, p.position, tiers, walls, exits) for p in people}

        # 2. Exit slots.
        def earliest(pid: str) -> float:
            route = routes[pid]
            return _INF if route is None else min(route.arrival.values())

        used: dict[str, set[int]] = {e.id: set() for e in m.exits}
        assigned: dict[str, tuple[str, int]] = {}
        for person in sorted(people, key=lambda p: (earliest(p.id), p.id)):
            route = routes[person.id]
            if route is None:
                continue
            best: tuple[int, int, str] | None = None
            for exit_id in sorted(route.arrival):
                slot = route.arrival[exit_id]
                while slot in used[exit_id]:
                    slot += 1
                if not _can_evacuate(route.ignite, exit_pos[exit_id], slot):
                    continue
                key = (slot, route.arrival[exit_id], exit_id)
                if best is None or key < best:
                    best = key
            if best is None:  # every slot is too late: fall back to the earliest arrival
                exit_id = _best_exit(route)
                best = (route.arrival[exit_id], route.arrival[exit_id], exit_id)
            used[best[2]].add(best[0])
            assigned[person.id] = (best[2], best[0])

        def priority(pid: str) -> tuple[float, float, str]:
            if pid in assigned:
                return (assigned[pid][1], earliest(pid), pid)
            return (_INF, _INF, pid)

        # 3. Prioritized windowed space-time A*.
        order = sorted(people, key=lambda p: priority(p.id))
        vertex: set[tuple[Pos, int]] = set()
        edges: set[tuple[Pos, Pos, int]] = set()
        unplanned = {p.position for p in people}
        actions: dict[str, Action] = {}
        for person in order:
            unplanned.discard(person.position)
            route = routes[person.id]
            path: list[Pos] | None = None
            if route is not None:
                target = exit_pos[assigned[person.id][0]]
                path = self._plan(observation, person.position, route.ignite, fields[target], vertex, edges, unplanned)
            if path is None:
                path = [person.position, self._fallback(observation, person.position, route, tiers, vertex, edges, unplanned)]
            for t, cell in enumerate(path[1:], start=now + 1):
                vertex.add((cell, t))
                edges.add((path[t - now - 1], cell, t))
            actions[person.id] = _action_between(person.position, path[1])
        return actions

    def _fallback(
        self,
        observation: Observation,
        pos: Pos,
        route: _Route | None,
        tiers: tuple[dict[Pos, int], ...],
        vertex: set[tuple[Pos, int]],
        edges: set[tuple[Pos, Pos, int]],
        unplanned: set[Pos],
    ) -> Pos:
        walls = observation.map.walls
        t = observation.step + 1
        if route is None:
            action = _flee(observation, pos, walls, tiers[0])
        else:
            action = route.first[_best_exit(route)]
        target = _step(pos, action)
        if (target, t) in vertex or (target, pos, t) in edges or target in unplanned:
            return pos
        return target

    def _plan(
        self,
        observation: Observation,
        start: Pos,
        ignite: Ignite,
        field: Mapping[Pos, int],
        vertex: set[tuple[Pos, int]],
        edges: set[tuple[Pos, Pos, int]],
        unplanned: set[Pos],
    ) -> list[Pos] | None:
        """Windowed space-time A*; the path ``[start, cell at now+1, ...]`` or ``None``."""
        walls = observation.map.walls
        exits = observation.map.exit_positions
        now = observation.step
        horizon = min(now + self._config.window, observation.rules.max_steps)
        if start not in field:
            return None
        candidates = (Action.STAY, *_moves(observation))
        parent: dict[tuple[Pos, int], tuple[Pos, int] | None] = {(start, now): None}
        heap: list[tuple[float, int, int, Pos, int]] = [(now + field[start], 0, 0, start, now)]
        counter = 0
        expansions = 0
        while heap:
            f, _rank, _n, cell, t = heapq.heappop(heap)
            if cell in exits or t == horizon:
                return _unwind(parent, (cell, t))
            expansions += 1
            if expansions > self._config.max_expansions:
                return None
            for action in candidates:
                nb = _step(cell, action)
                node = (nb, t + 1)
                if node in parent or not _open(observation, nb, walls) or nb not in field:
                    continue
                if node in vertex or (nb, cell, t + 1) in edges:
                    continue
                if t == now and nb != start and nb in unplanned:
                    continue
                if nb in exits:
                    if not _can_evacuate(ignite, nb, t + 1):
                        continue
                    cost = float(t + 1)
                else:
                    if not _can_stand(ignite, nb, t + 1):
                        continue
                    cost = float(t + 1 + field[nb])
                parent[node] = (cell, t)
                counter += 1
                # Equal cost: prefer the node closer to the exit, then expansion order.
                heapq.heappush(heap, (cost, field[nb] if nb not in exits else -1, counter, nb, t + 1))
        return None


def _unwind(parent: Mapping[tuple[Pos, int], tuple[Pos, int] | None], node: tuple[Pos, int]) -> list[Pos]:
    path = [node[0]]
    step = parent[node]
    while step is not None:
        path.append(step[0])
        step = parent[step]
    path.reverse()
    return path


def _action_between(a: Pos, b: Pos) -> Action:
    delta = (b.row - a.row, b.col - a.col)
    for action in Action:
        if action.delta == delta:
            return action
    raise AssertionError(f"not adjacent: {a} -> {b}")


# --------------------------------------------------------------------------- registry and digests


def _implementation_digest() -> str:
    """SHA-256 of this file with LF line endings, so the digest pins the exact algorithm."""
    text = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PackAgentSpec:
    """One evaluated agent of the pack: a fair model or the control."""

    model_id: str
    kind: str  # "control", "registered-baseline" or "pack-agent"
    description: str
    factory: Callable[[], Agent]
    descriptor: Mapping[str, Any]

    @property
    def config_digest(self) -> str:
        return _sha256(dict(self.descriptor))


def _pack_descriptor(agent_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pack_id": PACK_ID,
        "pack_agents_version": PACK_AGENTS_VERSION,
        "agent_id": agent_id,
        "privileged": False,
        "config": dict(config),
        "implementation_sha256": _implementation_digest(),
    }


def pack_agent_specs() -> tuple[PackAgentSpec, ...]:
    """Every agent of the pack in report order: control, the five registered baselines, then pack agents."""
    from emberbench import BASELINE_IDS, get_baseline

    specs = [
        PackAgentSpec(
            ALWAYS_STAY_ID,
            "control",
            "Test control: every active person stays.",
            AlwaysStayAgent,
            _pack_descriptor(ALWAYS_STAY_ID, {}),
        )
    ]
    for baseline_id in BASELINE_IDS:
        spec = get_baseline(baseline_id)
        specs.append(
            PackAgentSpec(
                baseline_id, "registered-baseline", spec.description, baseline_factory(baseline_id), spec.descriptor()
            )
        )
    specs.append(
        PackAgentSpec(
            FORECAST_ID,
            "pack-agent",
            "Independent shortest path that avoids cells the public fire rules can ignite before the person passes.",
            ForecastShortestPathAgent,
            _pack_descriptor(FORECAST_ID, {}),
        )
    )
    config = CooperativeConfig()
    specs.append(
        PackAgentSpec(
            COOPERATIVE_ID,
            "pack-agent",
            "Forecast-safe exit slots plus prioritized windowed space-time A* with reservations, replanned every step.",
            lambda: CooperativeForecastAgent(config),
            _pack_descriptor(COOPERATIVE_ID, config.to_dict()),
        )
    )
    return tuple(specs)
