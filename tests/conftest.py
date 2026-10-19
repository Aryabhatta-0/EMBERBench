from collections import deque
from collections.abc import Callable, Iterable
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
    obstacles, exits, people, ignitions = [], [], [], []
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


_STEP_TO_ACTION = {
    (-1, 0): Action.NORTH,
    (1, 0): Action.SOUTH,
    (0, 1): Action.EAST,
    (0, -1): Action.WEST,
}


def greedy_exit_policy(scenario: Scenario) -> Callable[[Observation], dict[str, Action]]:
    """Test-only baseline: step along a shortest fire-free path to the nearest usable exit."""

    def policy(obs: Observation) -> dict[str, Action]:
        blocked = scenario.walls | obs.burning
        dist: dict[Pos, int] = {}
        queue = deque(sorted(p for p in scenario.exit_positions if p not in obs.burning))
        for p in queue:
            dist[p] = 0
        while queue:
            cell = queue.popleft()
            for nb in neighbours(cell, scenario.width, scenario.height):
                if nb not in dist and nb not in blocked:
                    dist[nb] = dist[cell] + 1
                    queue.append(nb)
        actions = {}
        for person in obs.active:
            best = Action.STAY
            here = dist.get(person.position)
            for nb in neighbours(person.position, scenario.width, scenario.height):
                if nb in dist and (here is None or dist[nb] < here):
                    if best is Action.STAY or dist[nb] < dist[person.position.offset(*best.delta)]:
                        best = _STEP_TO_ACTION[(nb.row - person.position.row, nb.col - person.position.col)]
            actions[person.id] = best
        return actions

    return policy
