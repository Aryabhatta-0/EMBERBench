"""Seed-free structural measurements of a scenario: generation proxies and the family-rule profile.

Everything here is a pure function of the scenario bytes. No seed, fire
trajectory, policy or model is an input, so anyone can re-derive every value
from a frozen scenario. The generator uses :func:`measure_proxies` as its
generation constraints (``emberbench.generator``); the family rules use
:func:`profile_structure` (``emberbench.families``).

The definitions of ``min_cut``, ``split_pressure``, ``deceptive_share``,
``exit_gap_median``, ``detour_ratio`` and ``ignition_centrality`` are the
PR #33 structural profile (docs/evaluation-suite-design.md section 4); this
module only moves them from the research tool into the package. See
``docs/generator.md``.
"""

import math
import statistics
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .grid import Pos, neighbours
from .scenario import Scenario

# A person has at most four neighbours, so at most four vertex-disjoint routes.
ROUTE_REDUNDANCY_CAP = 4
# A person counts as deceived when the exit nearest by Manhattan distance is at
# least this many steps longer by route than the exit nearest by route.
DECEPTIVE_ROUTE_EXCESS = 2


@dataclass(frozen=True)
class GenerationProxies:
    """Deterministic measurements of one scenario, used as generation constraints and family evidence.

    All values are computed from the scenario alone, so anyone can re-derive
    them. "Interior" means every cell that is not on the grid border. "Route"
    distances are static: walls only, no fire.

    - ``obstacle_density``: interior wall cells / interior cells.
    - ``crowd_density``: people / free interior cells.
    - ``exit_separation``: smallest Manhattan distance between two exits
      (``None`` with one exit).
    - ``passage_width``: the narrowest passage that some person cannot avoid.
      A cell's width is the shorter of its horizontal and vertical free runs;
      each person takes the route whose narrowest cell is widest, and the
      value is the minimum over people (0 if a person has no route).
    - ``route_redundancy``: the minimum over people of the number of
      vertex-disjoint routes to the exits on the static map, capped at
      :data:`ROUTE_REDUNDANCY_CAP`. A value of 1 means one blocked cell can
      cut a person off from every exit.
    - ``ignition_person_distance`` / ``ignition_exit_distance``: shortest
      wall-free distance from any ignition to any person / exit (``None`` if
      the fire can never reach one).
    - ``route_length``: the longest shortest fire-free route over people,
      avoiding ignition cells (``None`` if a person has no such route).
    - ``min_cut``: the maximum number of vertex-disjoint routes from the whole
      crowd to the exit set (every free cell, exits included, has capacity 1).
    - ``bottleneck_severity``: ``population / min_cut``, the fewest steps the
      crowd needs to drain through its narrowest cut (``None`` when
      ``min_cut`` is 0: infinite).
    - ``max_nearest_exit_load``: the most people that share one route-nearest
      exit (ties to the smaller exit ID).
    - ``split_pressure``: the busiest route-nearest exit's load divided by
      ``1 +`` the median extra route length its people would walk to their
      second-nearest exit. High when many people crowd one exit while another
      is almost as close. ``None`` with one exit or when no person of the
      busiest exit has a second reachable exit.
    - ``deceptive_share``: the share of people whose exit nearest by
      Manhattan distance is a dead end or at least
      :data:`DECEPTIVE_ROUTE_EXCESS` steps longer by route than their
      route-nearest exit (with several Manhattan-nearest exits, the best of
      them counts).
    - ``exit_gap_median``: the median over people of (second - first) route
      distance to an exit (``None`` with one exit or no such person).
    - ``detour_ratio``: the mean over people of route distance / Manhattan
      distance to the route-nearest exit (``None`` if no person can reach an
      exit; the family rule reads ``None`` as infinite).
    - ``ignition_centrality``: 0 for an ignition at the grid centre, 1 on the
      border (Chebyshev, normalized); the minimum over ignitions.
    """

    obstacle_density: float
    crowd_density: float
    exit_separation: int | None
    passage_width: int
    route_redundancy: int
    ignition_person_distance: int | None
    ignition_exit_distance: int | None
    route_length: int | None
    min_cut: int
    bottleneck_severity: float | None
    max_nearest_exit_load: int
    split_pressure: float | None
    deceptive_share: float
    exit_gap_median: float | None
    detour_ratio: float | None
    ignition_centrality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_density": self.obstacle_density,
            "crowd_density": self.crowd_density,
            "exit_separation": self.exit_separation,
            "passage_width": self.passage_width,
            "route_redundancy": self.route_redundancy,
            "ignition_person_distance": self.ignition_person_distance,
            "ignition_exit_distance": self.ignition_exit_distance,
            "route_length": self.route_length,
            "min_cut": self.min_cut,
            "bottleneck_severity": self.bottleneck_severity,
            "max_nearest_exit_load": self.max_nearest_exit_load,
            "split_pressure": self.split_pressure,
            "deceptive_share": self.deceptive_share,
            "exit_gap_median": self.exit_gap_median,
            "detour_ratio": self.detour_ratio,
            "ignition_centrality": self.ignition_centrality,
        }


def _bfs(sources: Iterable[Pos], passable: Callable[[Pos], bool], width: int, height: int) -> dict[Pos, int]:
    dist: dict[Pos, int] = {}
    queue: deque[Pos] = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        cell = queue.popleft()
        for nb in neighbours(cell, width, height):
            if nb not in dist and passable(nb):
                dist[nb] = dist[cell] + 1
                queue.append(nb)
    return dist


def exit_fields(scenario: Scenario) -> dict[str, dict[Pos, int]]:
    """Per exit ID, the static route distance (walls only) from every reachable cell."""
    walls = scenario.walls

    def open_cell(p: Pos) -> bool:
        return p not in walls

    return {e.id: _bfs([e.position], open_cell, scenario.width, scenario.height) for e in scenario.exits}


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _passage_width(s: Scenario) -> int:
    walls = s.walls
    exits = set(s.exit_positions)
    free = [Pos(r, c) for r in range(s.height) for c in range(s.width) if Pos(r, c) not in walls]
    run_h: dict[Pos, int] = {}
    run_v: dict[Pos, int] = {}
    for r in range(s.height):
        _assign_runs([Pos(r, c) for c in range(s.width)], walls, run_h)
    for c in range(s.width):
        _assign_runs([Pos(r, c) for r in range(s.height)], walls, run_v)
    width = {p: min(run_h[p], run_v[p]) for p in free if p not in exits}

    parent: dict[Pos, Pos] = {}
    has_exit: dict[Pos, bool] = {}

    def find(p: Pos) -> Pos:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def add(p: Pos, is_exit: bool) -> None:
        parent[p] = p
        has_exit[p] = is_exit
        for nb in neighbours(p, s.width, s.height):
            if nb in parent:
                a, b = find(p), find(nb)
                if a != b:
                    parent[b] = a
                    has_exit[a] = has_exit[a] or has_exit[b]

    for e in sorted(exits):
        add(e, True)
    unresolved = {p.position for p in s.people}
    order = sorted(width, key=lambda p: (-width[p], p))
    i = 0
    while i < len(order) and unresolved:
        level = width[order[i]]
        while i < len(order) and width[order[i]] == level:
            add(order[i], False)
            i += 1
        unresolved = {p for p in unresolved if not (p in parent and has_exit[find(p)])}
        if not unresolved:
            return level
    return 0


def _assign_runs(line: list[Pos], walls: frozenset[Pos], out: dict[Pos, int]) -> None:
    run: list[Pos] = []
    for p in [*line, None]:
        if p is not None and p not in walls:
            run.append(p)
            continue
        for q in run:
            out[q] = len(run)
        run = []


class _FlowNetwork:
    """A unit-capacity flow network with residual edges, for small vertex-cut problems."""

    __slots__ = ("heads", "to", "cap")

    def __init__(self, nodes: int) -> None:
        self.heads: list[list[int]] = [[] for _ in range(nodes)]
        self.to: list[int] = []
        self.cap: list[int] = []

    def edge(self, u: int, v: int, cap: int) -> None:
        self.heads[u].append(len(self.to))
        self.to.append(v)
        self.cap.append(cap)
        self.heads[v].append(len(self.to))
        self.to.append(u)
        self.cap.append(0)

    def max_flow(self, source: int, sink: int, cap: list[int], limit: int) -> int:
        """Augment shortest paths on ``cap`` (modified in place) until ``limit`` or no path is left."""
        heads, to = self.heads, self.to
        flow = 0
        while flow < limit:
            via = {source: -1}
            queue = deque([source])
            while queue and sink not in via:
                u = queue.popleft()
                for e in heads[u]:
                    if cap[e] > 0 and to[e] not in via:
                        via[to[e]] = e
                        queue.append(to[e])
            if sink not in via:
                break
            v = sink
            while v != source:
                e = via[v]
                cap[e] -= 1
                cap[e ^ 1] += 1
                v = to[e ^ 1]
            flow += 1
        return flow


def _route_redundancy(s: Scenario) -> int:
    """Minimum over people of vertex-disjoint routes to the exit set (unit vertex capacities)."""
    walls = s.walls
    exits = set(s.exit_positions)
    free = [Pos(r, c) for r in range(s.height) for c in range(s.width) if Pos(r, c) not in walls]
    index = {p: k for k, p in enumerate(free)}
    sink = 2 * len(free)
    big = 1 << 30
    net = _FlowNetwork(sink + 1)
    for k, p in enumerate(free):
        if p in exits:
            net.edge(2 * k, sink, big)
            continue
        net.edge(2 * k, 2 * k + 1, 1)
        for nb in neighbours(p, s.width, s.height):
            if nb in index:
                net.edge(2 * k + 1, 2 * index[nb], big)

    best = ROUTE_REDUNDANCY_CAP
    for person in sorted(p.position for p in s.people):
        best = min(best, net.max_flow(2 * index[person] + 1, sink, list(net.cap), best))
        if best == 0:
            break
    return best


def _min_cut(s: Scenario) -> int:
    """Max number of vertex-disjoint crowd-to-exit routes on the static map (every free cell has capacity 1).

    Every exit cell has capacity 1, so the value is at most the exit count.
    """
    walls = s.walls
    exits = set(s.exit_positions)
    free = [Pos(r, c) for r in range(s.height) for c in range(s.width) if Pos(r, c) not in walls]
    index = {p: k for k, p in enumerate(free)}
    source, sink = 2 * len(free), 2 * len(free) + 1
    net = _FlowNetwork(sink + 1)
    for k, p in enumerate(free):
        net.edge(2 * k, 2 * k + 1, 1)
        if p in exits:
            net.edge(2 * k + 1, sink, 1)
            continue
        for nb in neighbours(p, s.width, s.height):
            if nb in index:
                net.edge(2 * k + 1, 2 * index[nb], 1)
    for p in sorted({p.position for p in s.people}):
        net.edge(source, 2 * index[p], 1)
    return net.max_flow(source, sink, list(net.cap), len(exits))


def _exit_profile(s: Scenario) -> tuple[int, float | None, float, float | None, float | None]:
    """``(max_nearest_exit_load, split_pressure, deceptive_share, exit_gap_median, detour_ratio)``."""
    fields = exit_fields(s)
    exit_ids = sorted(fields)
    pos_of = {e.id: e.position for e in s.exits}
    people = sorted(s.people, key=lambda p: p.id)
    nearest: dict[str, str] = {}
    gaps: list[int] = []
    detours: list[float] = []
    deceptive = 0
    for p in people:
        dists = sorted((fields[e].get(p.position, math.inf), e) for e in exit_ids)
        d0, e0 = dists[0]
        nearest[p.id] = e0
        if len(dists) > 1 and math.isfinite(dists[1][0]) and math.isfinite(d0):
            gaps.append(int(dists[1][0] - d0))
        if math.isfinite(d0):
            man = _manhattan(p.position, pos_of[e0])
            detours.append(d0 / man if man else 1.0)
            man_best = min(_manhattan(p.position, pos_of[e]) for e in exit_ids)
            man_nearest = [e for e in exit_ids if _manhattan(p.position, pos_of[e]) == man_best]
            if min(fields[e].get(p.position, math.inf) for e in man_nearest) - d0 >= DECEPTIVE_ROUTE_EXCESS:
                deceptive += 1
    load = Counter(nearest.values())
    q_max = max(load.values()) if load else 0
    split: float | None = None
    if len(exit_ids) > 1 and load:
        busiest = min(e for e, n in load.items() if n == q_max)
        extra = []
        for p in people:
            if nearest[p.id] != busiest:
                continue
            d = sorted(fields[e].get(p.position, math.inf) for e in exit_ids)
            if math.isfinite(d[1]):
                extra.append(d[1] - d[0])
        split = q_max / (1 + statistics.median(extra)) if extra else None
    return (
        q_max,
        split,
        deceptive / len(people) if people else 0.0,
        float(statistics.median(gaps)) if gaps else None,
        statistics.fmean(detours) if detours else None,
    )


def _ignition_centrality(s: Scenario) -> float:
    h, w = s.height, s.width
    return min(max(abs(i.row - (h - 1) / 2) / ((h - 1) / 2), abs(i.col - (w - 1) / 2) / ((w - 1) / 2)) for i in s.ignitions)


def measure_proxies(scenario: Scenario) -> GenerationProxies:
    """Measure every structural proxy of ``scenario``. Pure and deterministic."""
    s = scenario
    walls = s.walls
    exits = s.exit_positions
    interior = [Pos(r, c) for r in range(1, s.height - 1) for c in range(1, s.width - 1)]
    interior_walls = sum(1 for p in interior if p in walls)
    free_interior = len(interior) - interior_walls
    exit_list = sorted(exits)
    separations = [abs(a.row - b.row) + abs(a.col - b.col) for i, a in enumerate(exit_list) for b in exit_list[i + 1 :]]

    def open_cell(p: Pos) -> bool:
        return p not in walls

    from_fire = _bfs(s.ignitions, open_cell, s.width, s.height)
    person_cells = [p.position for p in s.people]
    to_person = [from_fire[p] for p in person_cells if p in from_fire]
    to_exit = [from_fire[e] for e in exit_list if e in from_fire]

    ignitions = set(s.ignitions)
    route = _bfs(exit_list, lambda p: p not in walls and p not in ignitions and p not in exits, s.width, s.height)
    route_lengths = [route.get(p) for p in person_cells]
    load, split, deceptive, gap, detour = _exit_profile(s)
    cut = _min_cut(s)

    return GenerationProxies(
        obstacle_density=interior_walls / len(interior) if interior else 0.0,
        crowd_density=len(s.people) / free_interior if free_interior else 0.0,
        exit_separation=min(separations) if separations else None,
        passage_width=_passage_width(s),
        route_redundancy=_route_redundancy(s),
        ignition_person_distance=min(to_person) if to_person else None,
        ignition_exit_distance=min(to_exit) if to_exit else None,
        route_length=None if any(d is None for d in route_lengths) else max(d for d in route_lengths if d is not None),
        min_cut=cut,
        bottleneck_severity=len(s.people) / cut if cut else None,
        max_nearest_exit_load=load,
        split_pressure=split,
        deceptive_share=deceptive,
        exit_gap_median=gap,
        detour_ratio=detour,
        ignition_centrality=_ignition_centrality(s),
    )


@dataclass(frozen=True)
class StructuralProfile:
    """The seed-free inputs of the structural family rules, for one scenario.

    It is :class:`GenerationProxies` plus the scenario counts and fire
    parameters. ``bottleneck_severity`` and ``detour_ratio`` are ``inf`` when
    no route exists. Field names match the PR #33 research tool's profile.
    """

    scenario_id: str
    scenario_hash: str
    width: int
    height: int
    free_cells: int
    population: int
    exit_count: int
    ignition_count: int
    fire_model: str
    spread_interval: int
    spread_probability: float | None
    max_steps: int
    obstacle_density: float
    crowd_density: float
    passage_width: int
    route_redundancy: int
    route_length: int | None
    ignition_person_distance: int | None
    ignition_exit_distance: int | None
    min_cut: int
    bottleneck_severity: float  # population / min_cut: drain steps through the narrowest cut
    max_nearest_exit_load: int
    split_pressure: float | None  # busiest nearest-exit queue / (1 + median detour to the next exit)
    deceptive_share: float  # people whose straight-line-nearest exit is >= 2 steps worse by route
    exit_gap_median: float | None  # median (second - first) route distance to exits
    detour_ratio: float  # mean route / Manhattan distance to the route-nearest exit
    ignition_centrality: float  # 0 = an ignition at the grid centre, 1 = on the border


def profile_structure(scenario: Scenario, proxies: GenerationProxies | None = None) -> StructuralProfile:
    """The structural profile of ``scenario``. Pass ``proxies`` when already measured, to skip the work."""
    s = scenario
    p = measure_proxies(s) if proxies is None else proxies
    free = sum(1 for r in range(s.height) for c in range(s.width) if Pos(r, c) not in s.walls)
    population = len(s.people)
    return StructuralProfile(
        scenario_id=s.scenario_id,
        scenario_hash=s.content_hash(),
        width=s.width,
        height=s.height,
        free_cells=free,
        population=population,
        exit_count=len(s.exits),
        ignition_count=len(s.ignitions),
        fire_model=s.fire.model.value,
        spread_interval=s.fire.spread_interval,
        spread_probability=s.fire.spread_probability,
        max_steps=s.termination.max_steps,
        obstacle_density=p.obstacle_density,
        crowd_density=p.crowd_density,
        passage_width=p.passage_width,
        route_redundancy=p.route_redundancy,
        route_length=p.route_length,
        ignition_person_distance=p.ignition_person_distance,
        ignition_exit_distance=p.ignition_exit_distance,
        min_cut=p.min_cut,
        bottleneck_severity=math.inf if p.bottleneck_severity is None else p.bottleneck_severity,
        max_nearest_exit_load=p.max_nearest_exit_load,
        split_pressure=p.split_pressure,
        deceptive_share=p.deceptive_share,
        exit_gap_median=p.exit_gap_median,
        detour_ratio=math.inf if p.detour_ratio is None else p.detour_ratio,
        ignition_centrality=p.ignition_centrality,
    )
