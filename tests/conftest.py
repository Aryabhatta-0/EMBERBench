from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from emberbench import Action, Observation, Pos, Scenario, load_scenario
from emberbench.grid import neighbours

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scenarios"
FIXTURE_NAMES = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def fixture_scenario(name: str) -> Scenario:
    return load_scenario(FIXTURE_DIR / f"{name}.json")


@pytest.fixture(params=FIXTURE_NAMES)
def any_fixture(request: pytest.FixtureRequest) -> Scenario:
    return fixture_scenario(request.param)


def layout(
    rows: Iterable[str],
    *,
    scenario_id: str = "test",
    seed: int = 0,
    **kwargs: Any,
) -> Scenario:
    """Build a scenario from ASCII: ``#`` wall, ``E`` exit, ``F`` ignition, ``.`` empty.

    People are letters ``a``-``z`` and get the ID ``P<letter>``, so tests can pick
    the ID ordering on purpose (``Pa < Pb``).
    """
    from emberbench import ExitSpec, PersonSpec

    rows = list(rows)
    obstacles: list[Pos] = []
    exits: list[ExitSpec] = []
    people: list[PersonSpec] = []
    ignitions: list[Pos] = []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            p = Pos(r, c)
            if ch == "#":
                obstacles.append(p)
            elif ch == "E":
                exits.append(ExitSpec(f"E{len(exits) + 1}", p))
            elif ch == "F":
                ignitions.append(p)
            elif ch.islower():
                people.append(PersonSpec(f"P{ch}", p))
            elif ch != ".":
                raise ValueError(ch)
    return Scenario(
        scenario_id=scenario_id,
        width=len(rows[0]),
        height=len(rows),
        obstacles=tuple(obstacles),
        exits=tuple(exits),
        people=tuple(people),
        ignitions=tuple(ignitions),
        seed=seed,
        **kwargs,
    )


def crowd_stress_scenario(population: int, *, max_steps: int = 400, seed: int = 0) -> Scenario:
    """A synthetic walled room with one exit in the middle of each wall and a dense block of people.

    People fill a block 20 columns wide in row-major order and get the IDs
    ``P000``, ``P001``, ... One ignition sits in the top-left corner and
    spreads slowly. Used by the large-population stress tests only.
    """
    from emberbench import ExitSpec, FireConfig, PersonSpec, TerminationConfig

    cols = 20
    rows = -(-population // cols)
    width, height = cols + 14, rows + 12
    exits = (
        ExitSpec("E1", Pos(0, width // 2)),
        ExitSpec("E2", Pos(height // 2, width - 1)),
        ExitSpec("E3", Pos(height - 1, width // 2)),
        ExitSpec("E4", Pos(height // 2, 0)),
    )
    exit_cells = {e.position for e in exits}
    border = [
        Pos(r, c)
        for r in range(height)
        for c in range(width)
        if (r in (0, height - 1) or c in (0, width - 1)) and Pos(r, c) not in exit_cells
    ]
    people = tuple(PersonSpec(f"P{i:03d}", Pos(6 + i // cols, 7 + i % cols)) for i in range(population))
    return Scenario(
        scenario_id=f"stress-{population}",
        width=width,
        height=height,
        obstacles=tuple(border),
        exits=exits,
        people=people,
        ignitions=(Pos(1, 1),),
        seed=seed,
        fire=FireConfig(spread_interval=8),
        termination=TerminationConfig(max_steps=max_steps),
    )


_STEP_TO_ACTION = {
    (-1, 0): Action.NORTH,
    (1, 0): Action.SOUTH,
    (0, 1): Action.EAST,
    (0, -1): Action.WEST,
}


def greedy_exit_policy(obs: Observation) -> dict[str, Action]:
    """Test-only baseline: step along a shortest fire-free path to the nearest usable exit.

    It reads only the model-visible observation, like a real decision model.
    """
    width, height = obs.map.width, obs.map.height
    blocked = obs.map.walls | obs.burning
    dist: dict[Pos, int] = {}
    queue = deque(sorted(p for p in obs.map.exit_positions if p not in obs.burning))
    for p in queue:
        dist[p] = 0
    while queue:
        cell = queue.popleft()
        for nb in neighbours(cell, width, height):
            if nb not in dist and nb not in blocked:
                dist[nb] = dist[cell] + 1
                queue.append(nb)
    actions = {}
    for person in obs.active:
        best = Action.STAY
        here = dist.get(person.position)
        for nb in neighbours(person.position, width, height):
            if nb in dist and (here is None or dist[nb] < here):
                if best is Action.STAY or dist[nb] < dist[person.position.offset(*best.delta)]:
                    best = _STEP_TO_ACTION[(nb.row - person.position.row, nb.col - person.position.col)]
        actions[person.id] = best
    return actions
