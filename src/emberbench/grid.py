"""Grid primitives: positions, cell kinds and movement actions.

Coordinates are ``(row, col)`` with ``(0, 0)`` at the top-left corner.
``NORTH`` decreases the row, ``SOUTH`` increases it, ``WEST`` decreases the
column and ``EAST`` increases it.
"""

from enum import StrEnum
from typing import NamedTuple


class Pos(NamedTuple):
    """A grid cell. Tuple ordering (row-major) is the canonical sort order."""

    row: int
    col: int

    def offset(self, d_row: int, d_col: int) -> "Pos":
        return Pos(self.row + d_row, self.col + d_col)

    def to_list(self) -> list[int]:
        return [self.row, self.col]


class Cell(StrEnum):
    """What occupies a cell in a rendered view of the state."""

    EMPTY = "empty"
    WALL = "wall"
    EXIT = "exit"
    FIRE = "fire"
    PERSON = "person"


CELL_GLYPHS: dict[Cell, str] = {
    Cell.EMPTY: ".",
    Cell.WALL: "#",
    Cell.EXIT: "E",
    Cell.FIRE: "F",
    Cell.PERSON: "P",
}


class Action(StrEnum):
    """A movement action for one person for one timestep."""

    STAY = "stay"
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def delta(self) -> tuple[int, int]:
        return _ACTION_DELTAS[self]


_ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}

# Von Neumann neighbourhood, used by fire spread and reachability checks.
# The order is fixed so that iteration never depends on hashing.
NEIGHBOUR_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))


def in_bounds(pos: Pos, width: int, height: int) -> bool:
    return 0 <= pos.row < height and 0 <= pos.col < width


def neighbours(pos: Pos, width: int, height: int) -> list[Pos]:
    """In-bounds 4-neighbours of ``pos`` in canonical (row-major) order."""
    result = [pos.offset(dr, dc) for dr, dc in NEIGHBOUR_DELTAS]
    return [p for p in result if in_bounds(p, width, height)]
