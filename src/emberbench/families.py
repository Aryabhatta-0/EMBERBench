"""The V1 family rules (family-rule version 1.0): fixed thresholds on measured scenario and world properties.

The rules and thresholds are the ones frozen by PR #33
(docs/evaluation-suite-design.md section 4, ``family_tags`` in
``tools/analyze_suite_design.py``). This module makes them a versioned
library API. It contains no measurement code with a seed: structural inputs
come from :class:`~emberbench.structure.StructuralProfile` (seed-free) and
fire inputs from :class:`FireEvidence` values that the evaluator measures on
the scenario's selected unique worlds (``emberbench.calibration``).

Two kinds of tag:

- **Structural** (generator-side, seed-free): ``single_exit_low_density``,
  ``single_exit_high_density``, ``narrow_corridor_bottleneck``,
  ``split_flow_across_exits``, ``crowd_100``, ``maze_obstacle_routing``,
  ``deceptive_nearest_exit``, ``multiple_ignitions`` and the secondary tags
  ``central_ignition`` and ``multiple_exits_unequal_distance``.
- **Fire-dependent** (evaluator-side): ``threatened_nearest_exit`` and
  ``route_invalidation_replanning``. They depend on the realized fire
  trajectory. A deterministic-fire scenario has one trajectory, so its tags
  are fixed by the scenario; a stochastic-fire scenario is tagged over the
  unique worlds the evaluator selected for it. No model is ever an input.

A scenario may carry several tags. In a composed suite each selected
scenario gets exactly one primary family: the family whose quota cell it was
selected for (``emberbench.composition``). :func:`descriptive_primary` is the
fixed-priority label that the generator writes into its own manifests.
"""

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .structure import DECEPTIVE_ROUTE_EXCESS, StructuralProfile
from .suite import ScenarioFamily

FAMILY_RULES_VERSION = "1.0"

# The ten V1 primary families (docs/evaluation-suite-design.md section 17), in design order.
PRIMARY_FAMILIES: tuple[ScenarioFamily, ...] = (
    ScenarioFamily.SINGLE_EXIT_LOW_DENSITY,
    ScenarioFamily.SINGLE_EXIT_HIGH_DENSITY,
    ScenarioFamily.NARROW_CORRIDOR_BOTTLENECK,
    ScenarioFamily.SPLIT_FLOW_ACROSS_EXITS,
    ScenarioFamily.CROWD_100,
    ScenarioFamily.MAZE_OBSTACLE_ROUTING,
    ScenarioFamily.DECEPTIVE_NEAREST_EXIT,
    ScenarioFamily.THREATENED_NEAREST_EXIT,
    ScenarioFamily.MULTIPLE_IGNITIONS,
    ScenarioFamily.ROUTE_INVALIDATION_REPLANNING,
)
# Valid ScenarioFamily values that are scenario attributes, not capabilities: tags only, never primary.
SECONDARY_FAMILIES: tuple[ScenarioFamily, ...] = (
    ScenarioFamily.CENTRAL_IGNITION,
    ScenarioFamily.MULTIPLE_EXITS_UNEQUAL_DISTANCE,
)
FIRE_DEPENDENT_FAMILIES: frozenset[ScenarioFamily] = frozenset(
    {ScenarioFamily.THREATENED_NEAREST_EXIT, ScenarioFamily.ROUTE_INVALIDATION_REPLANNING}
)
STRUCTURAL_PRIMARY_FAMILIES: tuple[ScenarioFamily, ...] = tuple(f for f in PRIMARY_FAMILIES if f not in FIRE_DEPENDENT_FAMILIES)

# PR #33 thresholds (docs/evaluation-suite-design.md section 4). Frozen with FAMILY_RULES_VERSION.
HIGH_CROWD_DENSITY = 0.15  # single exit: density >= this is high density
SPLIT_PRESSURE_MIN = 1.0  # split flow: split pressure strictly above this ...
SPLIT_FLOW_MIN_POPULATION = 20  # ... and at least this many people
CROWD_100_POPULATION = 100
MULTIPLE_IGNITIONS_MIN = 2
NARROW_PASSAGE_WIDTH = 1  # narrow corridor: unavoidable passage width <= this ...
BOTTLENECK_SEVERITY_MIN = 4.0  # ... and population / min cut >= this
MAZE_OBSTACLE_DENSITY = 0.25  # maze: obstacle density >= this ...
MAZE_DETOUR_RATIO = 1.5  # ... or mean route / Manhattan distance >= this
DECEPTIVE_SHARE_MIN = 0.25  # deceptive exit: several exits and deceptive share >= this
CENTRAL_IGNITION_MAX = 0.34  # secondary: ignition centrality <= this
UNEQUAL_EXIT_GAP_MIN = 4.0  # secondary: several exits and median exit gap >= this
THREATENED_SHARE_MIN = 0.2  # threatened exit: several exits and mean threatened share >= this
RECOVERABLE_INVALIDATION_MIN = 0.2  # route invalidation: mean recoverable invalidation share >= this

# The generator's descriptive primary label: the first tag in this order. It is the PR #33
# admitted-candidate count per family in ascending order (docs/evaluation-suite-design.md,
# section 4, "Scoreable (admitted)"), so the rarest family names a scenario. It never
# decides a composed suite's primary family; quota selection does.
DESCRIPTIVE_PRIORITY: tuple[ScenarioFamily, ...] = (
    ScenarioFamily.CROWD_100,
    ScenarioFamily.SPLIT_FLOW_ACROSS_EXITS,
    ScenarioFamily.DECEPTIVE_NEAREST_EXIT,
    ScenarioFamily.SINGLE_EXIT_HIGH_DENSITY,
    ScenarioFamily.THREATENED_NEAREST_EXIT,
    ScenarioFamily.SINGLE_EXIT_LOW_DENSITY,
    ScenarioFamily.MAZE_OBSTACLE_ROUTING,
    ScenarioFamily.ROUTE_INVALIDATION_REPLANNING,
    ScenarioFamily.MULTIPLE_IGNITIONS,
    ScenarioFamily.NARROW_CORRIDOR_BOTTLENECK,
    ScenarioFamily.CENTRAL_IGNITION,
    ScenarioFamily.MULTIPLE_EXITS_UNEQUAL_DISTANCE,
)


@dataclass(frozen=True)
class FireEvidence:
    """Fire-dependent measurements of ONE unique world. Privileged: they need the fire trajectory.

    - ``threatened_share``: people whose route-nearest exit burns no later than
      ``max(route distance, people sharing that exit)`` steps.
    - ``invalidation_share``: people for whom no static shortest route to
      their route-nearest exit survives the fire when walked alone.
    - ``recoverable_invalidation_share``: invalidated people who are still
      savable (the official ``M*`` counts them): replanning can save them.
    - ``final_burn_share``: burning cells at the horizon / free cells.
    """

    threatened_share: float
    invalidation_share: float
    recoverable_invalidation_share: float
    final_burn_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threatened_share": self.threatened_share,
            "invalidation_share": self.invalidation_share,
            "recoverable_invalidation_share": self.recoverable_invalidation_share,
            "final_burn_share": self.final_burn_share,
        }


def _structural(profile: StructuralProfile) -> set[ScenarioFamily]:
    F = ScenarioFamily
    tags: set[ScenarioFamily] = set()
    if profile.exit_count == 1:
        tags.add(F.SINGLE_EXIT_HIGH_DENSITY if profile.crowd_density >= HIGH_CROWD_DENSITY else F.SINGLE_EXIT_LOW_DENSITY)
    else:
        if profile.exit_gap_median is not None and profile.exit_gap_median >= UNEQUAL_EXIT_GAP_MIN:
            tags.add(F.MULTIPLE_EXITS_UNEQUAL_DISTANCE)
        if is_split_flow(profile):
            tags.add(F.SPLIT_FLOW_ACROSS_EXITS)
    if profile.population >= CROWD_100_POPULATION:
        tags.add(F.CROWD_100)
    if profile.ignition_count >= MULTIPLE_IGNITIONS_MIN:
        tags.add(F.MULTIPLE_IGNITIONS)
    if is_narrow_corridor(profile):
        tags.add(F.NARROW_CORRIDOR_BOTTLENECK)
    if profile.obstacle_density >= MAZE_OBSTACLE_DENSITY or profile.detour_ratio >= MAZE_DETOUR_RATIO:
        tags.add(F.MAZE_OBSTACLE_ROUTING)
    if is_deceptive(profile):
        tags.add(F.DECEPTIVE_NEAREST_EXIT)
    if profile.ignition_centrality <= CENTRAL_IGNITION_MAX:
        tags.add(F.CENTRAL_IGNITION)
    return tags


def is_split_flow(profile: StructuralProfile) -> bool:
    """Genuine load balancing: several exits, split pressure above 1, at least 20 people.

    Having several exits is not enough: the busiest route-nearest exit must
    carry more people than one plus the median extra walk to the next exit.
    """
    return (
        profile.exit_count > 1
        and profile.split_pressure is not None
        and profile.split_pressure > SPLIT_PRESSURE_MIN
        and profile.population >= SPLIT_FLOW_MIN_POPULATION
    )


def is_narrow_corridor(profile: StructuralProfile) -> bool:
    """An unavoidable width-1 passage and at least 4 people per unit of the crowd's minimum cut."""
    return profile.passage_width <= NARROW_PASSAGE_WIDTH and profile.bottleneck_severity >= BOTTLENECK_SEVERITY_MIN


def is_deceptive(profile: StructuralProfile) -> bool:
    """Several exits, and at least a quarter of the people are misled by straight-line distance."""
    return profile.exit_count > 1 and profile.deceptive_share >= DECEPTIVE_SHARE_MIN


def _fire(profile: StructuralProfile, fires: Sequence[FireEvidence]) -> set[ScenarioFamily]:
    def mean(xs: Sequence[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    tags: set[ScenarioFamily] = set()
    if profile.exit_count > 1 and mean([f.threatened_share for f in fires]) >= THREATENED_SHARE_MIN:
        tags.add(ScenarioFamily.THREATENED_NEAREST_EXIT)
    if mean([f.recoverable_invalidation_share for f in fires]) >= RECOVERABLE_INVALIDATION_MIN:
        tags.add(ScenarioFamily.ROUTE_INVALIDATION_REPLANNING)
    return tags


def _with_fallback(profile: StructuralProfile, tags: set[ScenarioFamily]) -> frozenset[ScenarioFamily]:
    # PR #33: a multi-exit scenario without any other tag is described by its exits.
    if not tags and profile.exit_count > 1:
        tags.add(ScenarioFamily.MULTIPLE_EXITS_UNEQUAL_DISTANCE)
    return frozenset(tags)


def structural_tags(profile: StructuralProfile) -> frozenset[ScenarioFamily]:
    """The seed-free tags of a scenario: every family that its structure alone decides. Generator-side."""
    return _with_fallback(profile, _structural(profile))


def family_tags(profile: StructuralProfile, fires: Sequence[FireEvidence]) -> frozenset[ScenarioFamily]:
    """Every V1 family whose challenge a scenario measurably contains (PR #33 rule, exact).

    ``fires`` holds one :class:`FireEvidence` per unique world of the scenario
    (one for deterministic fire, the selected worlds for stochastic fire); the
    fire-dependent tags use their mean. Evaluator-side.
    """
    return _with_fallback(profile, _structural(profile) | _fire(profile, fires))


def descriptive_primary(tags: Iterable[ScenarioFamily]) -> ScenarioFamily:
    """The fixed-priority label of a tag set (see :data:`DESCRIPTIVE_PRIORITY`).

    A primary family wins over any secondary tag. A tag set with no primary
    family keeps its secondary tag as the label (a suite manifest needs one);
    such a scenario is not eligible for any V1 primary stratum.
    """
    present = set(tags)
    for family in DESCRIPTIVE_PRIORITY:
        if family in present:
            return family
    raise ValueError("a family tag set must not be empty")


def primary_candidates(tags: Iterable[ScenarioFamily]) -> frozenset[ScenarioFamily]:
    """The V1 primary families a scenario is eligible for (secondary tags removed)."""
    return frozenset(f for f in tags if f in PRIMARY_FAMILIES)


def family_rules() -> dict[str, Any]:
    """The canonical, versioned description of every rule threshold (for freeze records)."""
    return {
        "family_rules_version": FAMILY_RULES_VERSION,
        "primary_families": [f.value for f in PRIMARY_FAMILIES],
        "secondary_families": [f.value for f in SECONDARY_FAMILIES],
        "fire_dependent_families": sorted(f.value for f in FIRE_DEPENDENT_FAMILIES),
        "descriptive_priority": [f.value for f in DESCRIPTIVE_PRIORITY],
        "thresholds": {
            "high_crowd_density": HIGH_CROWD_DENSITY,
            "split_pressure_min_exclusive": SPLIT_PRESSURE_MIN,
            "split_flow_min_population": SPLIT_FLOW_MIN_POPULATION,
            "crowd_100_population": CROWD_100_POPULATION,
            "multiple_ignitions_min": MULTIPLE_IGNITIONS_MIN,
            "narrow_passage_width": NARROW_PASSAGE_WIDTH,
            "bottleneck_severity_min": BOTTLENECK_SEVERITY_MIN,
            "maze_obstacle_density": MAZE_OBSTACLE_DENSITY,
            "maze_detour_ratio": MAZE_DETOUR_RATIO,
            "deceptive_share_min": DECEPTIVE_SHARE_MIN,
            "deceptive_route_excess": DECEPTIVE_ROUTE_EXCESS,
            "central_ignition_max": CENTRAL_IGNITION_MAX,
            "unequal_exit_gap_min": UNEQUAL_EXIT_GAP_MIN,
            "threatened_share_min": THREATENED_SHARE_MIN,
            "recoverable_invalidation_min": RECOVERABLE_INVALIDATION_MIN,
        },
    }
