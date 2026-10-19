"""Fire dynamics contract and V1 spread models.

A fire model is called once per timestep, in simulator phase 7. It gets the
static terrain, the set of burning cells, the 1-based step number and the
simulator-owned fire RNG stream. It returns the set of newly ignited cells.

Contract every model must honour:

* Walls never burn, and fire never passes through them. Spread is only into
  flammable 4-neighbours of burning cells.
* Exits burn only when ``fire.exits_flammable`` is true.
* Burning cells stay burning (V1 has no burn-out).
* Randomness comes only from the ``rng`` argument. The model must use the
  stream in a deterministic order (canonical row-major cell order).
* The model is pure apart from advancing ``rng``. It keeps no hidden state.
* The model never sees people or actions, so the fire is the same for every
  action sequence under the same scenario and seed.

V1 has no delayed ignitions after step 0, no wind or directional spread, no
burn-out, no smoke, no material-specific flammability and no fire that
depends on people. These are future-version features; see
``docs/simulation-contract.md``, "Not in V1".
"""

from dataclasses import dataclass
from typing import Protocol

from .grid import Pos, neighbours
from .rng import SplitMix64
from .scenario import FireConfig, FireModelName, Scenario


@dataclass(frozen=True)
class FireTerrain:
    width: int
    height: int
    walls: frozenset[Pos]
    exits: frozenset[Pos]
    exits_flammable: bool

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "FireTerrain":
        return cls(
            width=scenario.width,
            height=scenario.height,
            walls=scenario.walls,
            exits=frozenset(scenario.exit_positions),
            exits_flammable=scenario.fire.exits_flammable,
        )

    def is_flammable(self, pos: Pos) -> bool:
        if pos in self.walls:
            return False
        if pos in self.exits and not self.exits_flammable:
            return False
        return True

    def frontier(self, burning: frozenset[Pos]) -> list[tuple[Pos, list[Pos]]]:
        """Unburnt flammable cells next to fire, with their burning neighbours.

        Both levels are in canonical row-major order.
        """
        candidates: dict[Pos, list[Pos]] = {}
        for src in sorted(burning):
            for nb in neighbours(src, self.width, self.height):
                if nb not in burning and self.is_flammable(nb):
                    candidates.setdefault(nb, []).append(src)
        return [(cell, sorted(srcs)) for cell, srcs in sorted(candidates.items())]


class FireModel(Protocol):
    @property
    def name(self) -> str: ...

    def spread(self, terrain: FireTerrain, burning: frozenset[Pos], step: int, rng: SplitMix64) -> frozenset[Pos]:
        """Return cells that ignite at ``step``. These cells must not be burning already."""
        ...


def _is_spread_step(step: int, interval: int) -> bool:
    return step % interval == 0


@dataclass(frozen=True)
class DeterministicSpread:
    """Every flammable neighbour of fire ignites on each ``spread_interval``-th step."""

    spread_interval: int = 1
    name: str = FireModelName.DETERMINISTIC.value

    def spread(self, terrain: FireTerrain, burning: frozenset[Pos], step: int, rng: SplitMix64) -> frozenset[Pos]:
        if not _is_spread_step(step, self.spread_interval):
            return frozenset()
        return frozenset(cell for cell, _ in terrain.frontier(burning))


@dataclass(frozen=True)
class StochasticSpread:
    """Each (burning neighbour -> candidate) edge ignites the candidate with ``probability``.

    On a spread step there is exactly one Bernoulli draw per edge. The draws
    go in canonical order: candidate cell first, then source cell. A candidate
    ignites if any of its edges succeeds. All draws are taken (no
    short-circuit), so the stream position depends only on the fire geometry.

    A candidate with ``k`` burning neighbours therefore ignites with
    probability ``1 - (1 - p)**k``. A spread step advances ``rng`` by exactly
    the number of (candidate, burning neighbour) edges; any other step does
    not advance it. This holds for every ``p``, including 0 and 1.
    """

    probability: float
    spread_interval: int = 1
    name: str = FireModelName.STOCHASTIC.value

    def spread(self, terrain: FireTerrain, burning: frozenset[Pos], step: int, rng: SplitMix64) -> frozenset[Pos]:
        if not _is_spread_step(step, self.spread_interval):
            return frozenset()
        ignited = []
        for cell, sources in terrain.frontier(burning):
            hits = [rng.bernoulli(self.probability) for _ in sources]
            if any(hits):
                ignited.append(cell)
        return frozenset(ignited)


def build_fire_model(config: FireConfig) -> FireModel:
    if config.model is FireModelName.DETERMINISTIC:
        return DeterministicSpread(spread_interval=config.spread_interval)
    if config.model is FireModelName.STOCHASTIC:
        assert config.spread_probability is not None  # guaranteed by scenario validation
        return StochasticSpread(probability=config.spread_probability, spread_interval=config.spread_interval)
    raise ValueError(f"unknown fire model {config.model!r}")
