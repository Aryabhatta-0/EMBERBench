"""Evaluator-side composition of the V1 scientific suite: recipes, candidate pool, selection and freeze record.

This is the library form of the PR #33 hidden-suite composition protocol
(docs/evaluation-suite-design.md section 14). The pipeline is::

    generate candidates (recipes, private pool seed)
    -> compute each candidate's unique worlds (private evaluation seeds)
    -> run the frozen calibration panel (emberbench.calibration)
    -> apply the admission rules -> assign the difficulty band
    -> tag families (structural + fire-dependent, emberbench.families)
    -> fill the family x band x fire-model quotas (keyed, deterministic)
    -> top up only the short cells with targeted batches
    -> relabel neutrally and freeze (suite directory + private freeze record)

The generator never labels difficulty and never sees a seed of the
evaluation worlds. Fire-dependent tags of stochastic scenarios are measured on
exactly the unique worlds that :func:`~emberbench.build_unique_world_manifest`
later schedules (same candidate seed list, same walk), so the frozen family
evidence describes the evaluated worlds. No model is an input anywhere: the
only policies are the calibration-panel baselines.

Everything a composition produces is PRIVATE while the suite is live: the
seeds, the selection key, the scenarios, the freeze record and the per-world
measurements. Only :meth:`SuiteFreezeRecord.public_commitment` may be
published before the evaluation.
"""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import SIMULATOR_VERSION
from .calibration import (
    ADMITTED,
    BANDS,
    Calibration,
    WorldMeasurement,
    calibrate,
    calibration_rules,
    measure_world,
)
from .canonical import sha256_hex
from .errors import GenerationError
from .evaluation import EvaluationManifest
from .families import (
    FAMILY_RULES_VERSION,
    PRIMARY_FAMILIES,
    family_rules,
    family_tags,
    primary_candidates,
    structural_tags,
)
from .generator import (
    GENERATION_MANIFEST_VERSION,
    GENERATOR_CONFIG_VERSION,
    GENERATOR_VERSION,
    GeneratorConfig,
    generate_suite,
    scenario_file_bytes,
    scenario_id_for,
    validate_generator_seed,
)
from .observation import OBSERVATION_VERSION
from .oracle import fire_trajectory
from .rng import validate_seed
from .scenario import ID_PATTERN, SCHEMA_VERSION, FireModelName, Scenario
from .scoring import SCORING_VERSION
from .structure import GenerationProxies, StructuralProfile, profile_structure
from .suite import MANIFEST_NAME, ScenarioFamily, SeedRole, Suite, SuiteEntry, SuiteManifest, Visibility, load_suite
from .worlds import ENVIRONMENT_WORLD_VERSION, build_unique_world_manifest, environment_world_digest

COMPOSITION_VERSION = "1.0"
SELECTION_RULES_VERSION = "1.0"
RECIPE_CATALOG_VERSION = "1.0"
SUITE_FREEZE_RECORD_VERSION = "1.0"
SUITE_FREEZE_RECORD_NAME = "suite-freeze.json"
COMPOSED_SUITE_DIR_NAME = "suite"
COMPOSED_SUITE_VERSION = "1.0.0"
COMPOSED_PURPOSE = "Composed unseen evaluation scenario."

# PR #33 pool protocol (docs/evaluation-suite-design.md section 12).
INITIAL_POOL_SIZE = 4000
TOP_UP_BATCH_SIZE = 200
POOL_CAP = 10_000
# Candidate-pool exclusion that precedes admission: build_unique_world_manifest could not schedule it.
INSUFFICIENT_WORLDS = "insufficient_worlds"

_POOL_DOMAIN = "emberbench/candidate-pool/"
_COMMIT_DOMAIN = "emberbench/commitment/"

QuotaCell = tuple[ScenarioFamily, str, bool]  # (primary family, band, stochastic fire)


# --------------------------------------------------------------------------- suite design


@dataclass(frozen=True)
class SuiteDesign:
    """The family x band x fire-model quotas. :data:`V1_DESIGN` is the frozen PR #33 design."""

    families: tuple[ScenarioFamily, ...]
    bands: tuple[tuple[str, int], ...]  # (band, scenarios per family), in BANDS order
    stochastic_per_family: int
    stochastic_worlds: int  # unique fire worlds per stochastic scenario

    def __post_init__(self) -> None:
        if tuple(b for b, _ in self.bands) != BANDS or any(n < 0 for _, n in self.bands):
            raise ValueError(f"bands must list {BANDS} in order with non-negative counts")
        if not set(self.families) <= set(PRIMARY_FAMILIES) or len(set(self.families)) != len(self.families):
            raise ValueError("families must be distinct V1 primary families")
        if not 0 <= self.stochastic_per_family <= self.per_family or self.stochastic_worlds < 1:
            raise ValueError("invalid stochastic allocation")

    @property
    def per_family(self) -> int:
        return sum(n for _, n in self.bands)

    @property
    def scenario_count(self) -> int:
        return self.per_family * len(self.families)

    @property
    def world_count(self) -> int:
        return len(self.families) * (self.per_family + self.stochastic_per_family * (self.stochastic_worlds - 1))

    def slots(self) -> tuple[tuple[str, bool], ...]:
        """One family's ``(band, stochastic)`` slots: stochastic slots spread evenly over the band-ordered slots.

        This is the PR #33 study rule (``_slots``). For the V1 design the four
        stochastic slots fall in Moderate (1), Hard (2) and Stress (1).
        """
        bands = [b for b, n in self.bands for _ in range(n)]
        m, s = len(bands), self.stochastic_per_family
        return tuple((b, (i + 1) * s // m > i * s // m) for i, b in enumerate(bands))

    def quotas(self) -> dict[QuotaCell, int]:
        counts = Counter(self.slots())
        return {(f, b, st): counts[(b, st)] for f in self.families for b in BANDS for st in (True, False) if counts[(b, st)]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": [f.value for f in self.families],
            "bands": {b: n for b, n in self.bands},
            "per_family": self.per_family,
            "stochastic_per_family": self.stochastic_per_family,
            "stochastic_worlds": self.stochastic_worlds,
            "scenarios": self.scenario_count,
            "unique_worlds": self.world_count,
            "cells": [[f.value, b, st, n] for (f, b, st), n in sorted(self.quotas().items(), key=lambda kv: _cell_key(kv[0]))],
        }


V1_DESIGN = SuiteDesign(
    families=PRIMARY_FAMILIES,
    bands=(("easy", 2), ("moderate", 4), ("hard", 6), ("stress", 4)),
    stochastic_per_family=4,
    stochastic_worlds=3,
)


def _cell_key(cell: QuotaCell) -> tuple[int, int, bool]:
    family, band, stochastic = cell
    return PRIMARY_FAMILIES.index(family), BANDS.index(band), not stochastic


# --------------------------------------------------------------------------- recipes


@dataclass(frozen=True)
class Recipe:
    """A reproducible candidate distribution: a generator config without its scenario count.

    A recipe defines a distribution, never a test item. ``targets`` names the
    primary families it aims at; it is used only to choose top-up batches. The
    families a candidate counts for are always measured, never taken from the
    recipe.
    """

    recipe_id: str
    recipe_version: str
    targets: tuple[ScenarioFamily, ...]
    config: Mapping[str, Any]  # a generator config dict without "scenario_count"

    def __post_init__(self) -> None:
        if not ID_PATTERN.fullmatch(self.recipe_id):
            raise ValueError(f"recipe_id must match {ID_PATTERN.pattern}")
        self.config_for(1)  # validates the config once

    @property
    def stochastic(self) -> bool:
        return bool(self.config["fire"]["model"] == FireModelName.STOCHASTIC.value)

    def config_for(self, count: int) -> GeneratorConfig:
        return GeneratorConfig.from_dict({**self.config, "scenario_count": count})

    def config_hash(self) -> str:
        """The hash of the recipe's distribution (the config with scenario count 1)."""
        return self.config_for(1).config_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "targets": [f.value for f in self.targets],
            "stochastic": self.stochastic,
            "config_hash": self.config_hash(),
            "config": dict(self.config),
        }


def _recipe(
    recipe_id: str,
    targets: Iterable[ScenarioFamily],
    *,
    grid: tuple[int, int],
    population: tuple[int, int],
    exits: tuple[int, int],
    ignitions: tuple[int, int] = (1, 1),
    partitions: tuple[int, int] = (0, 0),
    doors: tuple[int, int] = (1, 2),
    door_width: tuple[int, int] = (1, 2),
    scatter: tuple[float, float] = (0.0, 0.05),
    model: str = "deterministic",
    interval: tuple[int, int] = (2, 3),
    probabilities: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_steps: int = 80,
    max_attempts: int = 4000,
    ignition_exit: tuple[int, int] | None = None,
    route_redundancy: tuple[int, int] | None = None,
    passage_width: tuple[int, int] | None = None,
    split_pressure: tuple[float, float] | None = None,
    deceptive_share: tuple[float, float] | None = None,
    bottleneck_severity: tuple[float, float] | None = None,
) -> Recipe:
    """A recipe in the PR #33 config style (the research tool's ``_config``), plus the 1.1 constraints."""
    w, h = grid
    fire: dict[str, Any] = {"model": model, "spread_interval": {"min": interval[0], "max": interval[1]}, "exits_flammable": True}
    if model == FireModelName.STOCHASTIC.value:
        fire["spread_probability_choices"] = list(probabilities)
    ign_exit = ignition_exit or (3, w * h)
    redundancy = route_redundancy or (1, min(4, exits[1]))
    width = passage_width or (1, max(w, h))
    constraints: dict[str, Any] = {
        "route_length": {"min": 1, "max": max_steps},
        "obstacle_density": {"min": 0.0, "max": 0.6},
        "crowd_density": {"min": 0.0, "max": 0.9},
        "exit_separation": {"min": 4, "max": w + h},
        "passage_width": {"min": width[0], "max": width[1]},
        "route_redundancy": {"min": redundancy[0], "max": redundancy[1]},
        "ignition_person_distance": {"min": 3, "max": w * h},
        "ignition_exit_distance": {"min": ign_exit[0], "max": ign_exit[1]},
    }
    for name, value in (("split_pressure", split_pressure), ("deceptive_share", deceptive_share), ("bottleneck_severity", bottleneck_severity)):
        if value is not None:
            constraints[name] = {"min": value[0], "max": value[1]}
    config = {
        "config_version": GENERATOR_CONFIG_VERSION,
        "grid": {"width": w, "height": h},
        "population": {"min": population[0], "max": population[1]},
        "exit_count": {"min": exits[0], "max": exits[1]},
        "ignition_count": {"min": ignitions[0], "max": ignitions[1]},
        "layout": {
            "partition_count": {"min": partitions[0], "max": partitions[1]},
            "doors_per_partition": {"min": doors[0], "max": doors[1]},
            "door_width": {"min": door_width[0], "max": door_width[1]},
            "scatter_density": {"min": scatter[0], "max": scatter[1]},
        },
        "fire": fire,
        "movement": {"enter_fire": "casualty"},
        "max_steps": max_steps,
        "max_attempts": max_attempts,
        "constraints": constraints,
    }
    return Recipe(recipe_id, "1.0", tuple(targets), config)


def _catalog() -> tuple[Recipe, ...]:
    F = ScenarioFamily
    S = "stochastic"
    big = 1_000_000.0
    maze: dict[str, Any] = {"partitions": (2, 4), "doors": (1, 1), "door_width": (1, 1), "scatter": (0.1, 0.25)}
    return (
        # The ten PR #33 development recipes (tools/analyze_suite_design.py, development_recipes).
        _recipe("open-sparse", [F.SINGLE_EXIT_LOW_DENSITY], grid=(14, 10), population=(3, 8), exits=(1, 1), interval=(2, 4), max_steps=60),
        _recipe("dense-single", [F.SINGLE_EXIT_HIGH_DENSITY], grid=(14, 10), population=(20, 40), exits=(1, 1), interval=(3, 8), max_steps=90),
        _recipe("partitioned", [F.NARROW_CORRIDOR_BOTTLENECK], grid=(18, 12), population=(6, 16), exits=(1, 2), partitions=(1, 2), doors=(1, 1), door_width=(1, 1)),
        _recipe("multi-exit", [F.THREATENED_NEAREST_EXIT, F.ROUTE_INVALIDATION_REPLANNING], grid=(20, 12), population=(6, 20), exits=(2, 3), partitions=(0, 1)),
        _recipe("multi-ignition", [F.MULTIPLE_IGNITIONS], grid=(18, 12), population=(6, 16), exits=(2, 3), ignitions=(2, 4), interval=(3, 4)),
        _recipe("maze", [F.MAZE_OBSTACLE_ROUTING], grid=(16, 12), population=(4, 10), exits=(2, 3), interval=(3, 4), max_steps=100, **maze),
        _recipe("exit-threat", [F.THREATENED_NEAREST_EXIT], grid=(20, 12), population=(6, 16), exits=(2, 3), partitions=(0, 1), interval=(2, 4), ignition_exit=(3, 6)),
        _recipe("crowd", [F.CROWD_100], grid=(30, 18), population=(100, 120), exits=(2, 4), interval=(4, 8), max_steps=200),
        _recipe("stochastic-open", [F.SINGLE_EXIT_LOW_DENSITY, F.THREATENED_NEAREST_EXIT], grid=(18, 12), population=(5, 14), exits=(1, 3), ignitions=(1, 2), model=S, interval=(1, 2)),
        _recipe(
            "stochastic-partitioned", [F.NARROW_CORRIDOR_BOTTLENECK, F.ROUTE_INVALIDATION_REPLANNING], grid=(18, 12), population=(5, 14), exits=(1, 3),
            ignitions=(1, 2), partitions=(1, 2), doors=(1, 2), door_width=(1, 1), model=S, interval=(1, 2),
        ),
        # The four PR #33 gap-probe recipes (crowded maze; stochastic maze, dense single exit, crowd).
        _recipe("maze-crowded", [F.MAZE_OBSTACLE_ROUTING], grid=(16, 12), population=(12, 24), exits=(2, 3), ignitions=(1, 2), interval=(1, 2), max_steps=100, **maze),
        _recipe("maze-stochastic", [F.MAZE_OBSTACLE_ROUTING], grid=(16, 12), population=(6, 16), exits=(2, 3), ignitions=(1, 2), model=S, interval=(1, 2), max_steps=100, **maze),
        _recipe("dense-single-stochastic", [F.SINGLE_EXIT_HIGH_DENSITY], grid=(14, 10), population=(20, 40), exits=(1, 1), model=S, interval=(1, 3), max_steps=90),
        _recipe("crowd-stochastic", [F.CROWD_100], grid=(30, 18), population=(100, 120), exits=(2, 4), model=S, interval=(2, 4), max_steps=200),
        # Phase 11 targeted recipes: the new constraints aim them at one structural family each.
        _recipe(
            "split-flow-medium", [F.SPLIT_FLOW_ACROSS_EXITS], grid=(20, 14), population=(24, 60), exits=(2, 3), partitions=(0, 1), door_width=(1, 3),
            interval=(3, 6), max_steps=120, split_pressure=(1.05, big),
        ),
        _recipe(
            "split-flow-medium-stochastic", [F.SPLIT_FLOW_ACROSS_EXITS], grid=(20, 14), population=(24, 60), exits=(2, 3), partitions=(0, 1),
            door_width=(1, 3), model=S, interval=(2, 4), max_steps=120, split_pressure=(1.05, big),
        ),
        _recipe(
            "deceptive-exit", [F.DECEPTIVE_NEAREST_EXIT], grid=(18, 12), population=(8, 20), exits=(2, 3), partitions=(1, 3), doors=(1, 1),
            door_width=(1, 2), interval=(2, 4), max_steps=100, max_attempts=20000, deceptive_share=(0.25, 1.0),
        ),
        _recipe(
            "deceptive-exit-stochastic", [F.DECEPTIVE_NEAREST_EXIT], grid=(18, 12), population=(8, 20), exits=(2, 3), partitions=(1, 3), doors=(1, 1),
            door_width=(1, 2), model=S, interval=(1, 3), max_steps=100, max_attempts=20000, deceptive_share=(0.25, 1.0),
        ),
        _recipe(
            "exit-threat-stochastic", [F.THREATENED_NEAREST_EXIT], grid=(20, 12), population=(6, 16), exits=(2, 3), partitions=(0, 1), model=S,
            interval=(1, 3), ignition_exit=(3, 6),
        ),
        _recipe(
            "route-invalidation", [F.ROUTE_INVALIDATION_REPLANNING], grid=(18, 12), population=(6, 18), exits=(1, 3), partitions=(1, 2), doors=(2, 2),
            door_width=(1, 2), interval=(1, 3), max_steps=100, route_redundancy=(2, 4),
        ),
        _recipe(
            "route-invalidation-stochastic", [F.ROUTE_INVALIDATION_REPLANNING], grid=(18, 12), population=(6, 18), exits=(1, 3), partitions=(1, 2),
            doors=(2, 2), door_width=(1, 2), model=S, interval=(1, 2), max_steps=100, route_redundancy=(2, 4),
        ),
        _recipe(
            "narrow-corridor", [F.NARROW_CORRIDOR_BOTTLENECK], grid=(18, 12), population=(8, 24), exits=(1, 2), partitions=(1, 2), doors=(1, 1),
            door_width=(1, 1), interval=(2, 4), max_steps=100, passage_width=(1, 1), bottleneck_severity=(4.0, big),
        ),
        _recipe(
            "narrow-corridor-stochastic", [F.NARROW_CORRIDOR_BOTTLENECK], grid=(18, 12), population=(8, 24), exits=(1, 2), partitions=(1, 2),
            doors=(1, 1), door_width=(1, 1), model=S, interval=(1, 3), max_steps=100, passage_width=(1, 1), bottleneck_severity=(4.0, big),
        ),
        _recipe("multi-ignition-stochastic", [F.MULTIPLE_IGNITIONS], grid=(18, 12), population=(6, 16), exits=(2, 3), ignitions=(2, 4), model=S, interval=(1, 3)),
        _recipe("open-sparse-stochastic", [F.SINGLE_EXIT_LOW_DENSITY], grid=(14, 10), population=(3, 8), exits=(1, 1), model=S, interval=(1, 2), max_steps=60),
    )


V1_RECIPES: tuple[Recipe, ...] = _catalog()

# The initial 4,000-candidate plan (recipe ID, candidates). Rare cells get more candidates; crowd recipes fewer (cost).
V1_INITIAL_PLAN: tuple[tuple[str, int], ...] = (
    ("open-sparse", 120),
    ("dense-single", 160),
    ("partitioned", 160),
    ("multi-exit", 160),
    ("multi-ignition", 160),
    ("maze", 120),
    ("exit-threat", 160),
    ("crowd", 140),
    ("stochastic-open", 160),
    ("stochastic-partitioned", 160),
    ("maze-crowded", 200),
    ("maze-stochastic", 200),
    ("dense-single-stochastic", 160),
    ("crowd-stochastic", 120),
    ("split-flow-medium", 200),
    ("split-flow-medium-stochastic", 160),
    ("deceptive-exit", 200),
    ("deceptive-exit-stochastic", 200),
    ("exit-threat-stochastic", 200),
    ("route-invalidation", 160),
    ("route-invalidation-stochastic", 180),
    ("narrow-corridor", 120),
    ("narrow-corridor-stochastic", 120),
    ("multi-ignition-stochastic", 160),
    ("open-sparse-stochastic", 120),
)


def recipe_by_id(recipe_id: str, recipes: Sequence[Recipe] = V1_RECIPES) -> Recipe:
    for r in recipes:
        if r.recipe_id == recipe_id:
            return r
    raise KeyError(recipe_id)


def _v1_top_up() -> dict[tuple[ScenarioFamily, bool], str]:
    F = ScenarioFamily
    pairs = {
        F.SINGLE_EXIT_LOW_DENSITY: ("open-sparse", "open-sparse-stochastic"),
        F.SINGLE_EXIT_HIGH_DENSITY: ("dense-single", "dense-single-stochastic"),
        F.NARROW_CORRIDOR_BOTTLENECK: ("narrow-corridor", "narrow-corridor-stochastic"),
        F.SPLIT_FLOW_ACROSS_EXITS: ("split-flow-medium", "split-flow-medium-stochastic"),
        F.CROWD_100: ("crowd", "crowd-stochastic"),
        F.MAZE_OBSTACLE_ROUTING: ("maze-crowded", "maze-stochastic"),
        F.DECEPTIVE_NEAREST_EXIT: ("deceptive-exit", "deceptive-exit-stochastic"),
        F.THREATENED_NEAREST_EXIT: ("exit-threat", "exit-threat-stochastic"),
        F.MULTIPLE_IGNITIONS: ("multi-ignition", "multi-ignition-stochastic"),
        F.ROUTE_INVALIDATION_REPLANNING: ("route-invalidation", "route-invalidation-stochastic"),
    }
    return {(f, st): ids[st] for f, ids in pairs.items() for st in (False, True)}


# The frozen top-up table of the V1 catalog: one dedicated recipe per primary family and fire model.
V1_TOP_UP: dict[tuple[ScenarioFamily, bool], str] = _v1_top_up()


def top_up_recipe(cell: QuotaCell, recipes: Sequence[Recipe] = V1_RECIPES) -> Recipe:
    """The fixed top-up recipe of a quota cell (family and fire model; the band does not matter).

    For the V1 catalog this is :data:`V1_TOP_UP`. For another catalog it is
    the first recipe that targets the family with the same fire model, else
    the first recipe of that fire model. The choice depends only on the cell,
    never on a model or a score.
    """
    family, _, stochastic = cell
    if tuple(recipes) == V1_RECIPES:
        return recipe_by_id(V1_TOP_UP[(family, stochastic)], recipes)
    same_model = [r for r in recipes if r.stochastic == stochastic]
    for r in same_model:
        if family in r.targets:
            return r
    if not same_model:
        raise ValueError(f"no recipe with stochastic={stochastic}")
    return same_model[0]


# --------------------------------------------------------------------------- private seeds and commitments


def batch_seed(pool_seed: int, recipe: Recipe, batch_index: int) -> int:
    """The private generator seed of one recipe batch: a SHA-256 derivation from the private pool seed."""
    seed = validate_generator_seed(pool_seed)
    label = f"{_POOL_DOMAIN}{COMPOSITION_VERSION}/{recipe.recipe_id}/{recipe.recipe_version}/{batch_index}/".encode("ascii")
    return int.from_bytes(hashlib.sha256(label + seed.to_bytes(32, "big")).digest(), "big")


def commit(kind: str, value: int | str | Sequence[int]) -> str:
    """A SHA-256 commitment to a private value. Publishing it reveals nothing; revealing the value proves it."""
    payload = json.dumps({"domain": _COMMIT_DOMAIN + kind, "value": value if isinstance(value, (int, str)) else list(value)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_rank(selection_key: str, scenario_hash: str) -> str:
    # PR #33 keyed order (tools/analyze_suite_design.py, _selection_rank).
    return hashlib.sha256(f"emberbench/suite-selection/{selection_key}/{scenario_hash}".encode("utf-8")).hexdigest()


def _order_rank(selection_key: str, scenario_hash: str) -> str:
    return hashlib.sha256(f"emberbench/suite-order/{selection_key}/{scenario_hash}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class CandidateOrigin:
    """Where a candidate came from: recipe, batch and ordinal. Private while the suite is live."""

    recipe_id: str
    recipe_version: str
    batch_index: int
    batch_id: str
    ordinal: int

    def sort_key(self) -> tuple[str, int, int]:
        return self.recipe_id, self.batch_index, self.ordinal

    def to_dict(self) -> dict[str, Any]:
        return {"recipe_id": self.recipe_id, "recipe_version": self.recipe_version, "batch_index": self.batch_index, "batch_id": self.batch_id, "ordinal": self.ordinal}


@dataclass(frozen=True)
class MeasuredCandidate:
    """One candidate with its structural profile, unique-world measurements, calibration and family tags."""

    origin: CandidateOrigin
    scenario: Scenario
    proxies: GenerationProxies
    world_seed_indices: tuple[int, ...]  # positions in the evaluation candidate-seed list
    worlds: tuple[WorldMeasurement, ...]
    exclusion: str | None  # INSUFFICIENT_WORLDS, or None
    seconds: float  # evaluator CPU for this candidate (diagnostic only; never frozen)

    @property
    def scenario_hash(self) -> str:
        return self.scenario.content_hash()

    @property
    def stochastic(self) -> bool:
        return self.scenario.fire.model is FireModelName.STOCHASTIC

    @property
    def profile(self) -> StructuralProfile:
        return profile_structure(self.scenario, self.proxies)

    @property
    def calibration(self) -> Calibration:
        return calibrate(self.worlds)

    @property
    def admission(self) -> str:
        return self.exclusion or self.calibration.admission

    @property
    def band(self) -> str | None:
        return None if self.exclusion else self.calibration.band

    @property
    def structural_tags(self) -> frozenset[ScenarioFamily]:
        return structural_tags(self.profile)

    @property
    def tags(self) -> frozenset[ScenarioFamily]:
        return family_tags(self.profile, [w.fire for w in self.worlds])


def world_seeds_for(scenario: Scenario, evaluation_seeds: Sequence[int], stochastic_worlds: int) -> tuple[int, ...]:
    """Indices of the seeds that :func:`~emberbench.build_unique_world_manifest` schedules for ``scenario``.

    Deterministic fire: the first seed. Stochastic fire: the first
    ``stochastic_worlds`` seeds with pairwise-different worlds (fewer if the
    list runs out). The same walk as the manifest builder, so calibration and
    tags are measured on exactly the evaluated worlds.
    """
    if scenario.fire.model is FireModelName.DETERMINISTIC:
        return (0,)
    chosen: list[int] = []
    seen: set[str] = set()
    for i, seed in enumerate(evaluation_seeds):
        world = environment_world_digest(scenario, seed)
        if world not in seen:
            seen.add(world)
            chosen.append(i)
            if len(chosen) == stochastic_worlds:
                break
    return tuple(chosen)


def measure_candidate(
    origin: CandidateOrigin, scenario: Scenario, proxies: GenerationProxies, evaluation_seeds: Sequence[int], stochastic_worlds: int
) -> MeasuredCandidate:
    """Measure one candidate on its unique worlds (a worker task). PRIVILEGED; no model input."""
    t0 = time.perf_counter()
    indices = world_seeds_for(scenario, evaluation_seeds, stochastic_worlds)
    needed = 1 if scenario.fire.model is FireModelName.DETERMINISTIC else stochastic_worlds
    if len(indices) < needed:
        return MeasuredCandidate(origin, scenario, proxies, indices, (), INSUFFICIENT_WORLDS, time.perf_counter() - t0)
    worlds = tuple(measure_world(scenario, evaluation_seeds[i]) for i in indices)
    return MeasuredCandidate(origin, scenario, proxies, indices, worlds, None, time.perf_counter() - t0)


@dataclass(frozen=True)
class GeneratedBatch:
    recipe: Recipe
    batch_index: int
    batch_id: str
    seed_commitment: str
    generation_hash: str
    scenarios: tuple[tuple[Scenario, GenerationProxies], ...]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe.recipe_id,
            "recipe_version": self.recipe.recipe_version,
            "batch_index": self.batch_index,
            "batch_id": self.batch_id,
            "candidates": len(self.scenarios),
            "config_hash": self.recipe.config_for(len(self.scenarios)).config_hash(),
            "generator_seed_commitment": self.seed_commitment,
            "generation_hash": self.generation_hash,
        }


def batch_id_for(recipe: Recipe, batch_index: int) -> str:
    return f"pool-{recipe.recipe_id}-{batch_index:04d}"[:64]


def generate_batch(recipe: Recipe, pool_seed: int, batch_index: int, count: int) -> GeneratedBatch:
    """One recipe batch: a normal generated suite from the derived private seed (re-checkable with verify_regeneration)."""
    t0 = time.perf_counter()
    seed = batch_seed(pool_seed, recipe, batch_index)
    batch_id = batch_id_for(recipe, batch_index)
    generated = generate_suite(recipe.config_for(count), seed, batch_id=batch_id)
    pairs = tuple(zip(generated.scenarios, (e.proxies for e in generated.generation.entries), strict=True))
    return GeneratedBatch(recipe, batch_index, batch_id, commit("generator-seed", seed), generated.generation.generation_hash(), pairs, time.perf_counter() - t0)


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """Selected candidates per quota cell, in selection order, and the cells still short."""

    chosen: tuple[tuple[QuotaCell, MeasuredCandidate], ...]
    shortfall: tuple[tuple[QuotaCell, int], ...]

    @property
    def complete(self) -> bool:
        return not self.shortfall


def eligible(candidate: MeasuredCandidate) -> bool:
    return candidate.admission == ADMITTED and bool(primary_candidates(candidate.tags))


def select_suite(
    candidates: Sequence[MeasuredCandidate],
    quotas: Mapping[QuotaCell, int],
    selection_key: str,
    *,
    previous: Selection | None = None,
) -> Selection:
    """Deterministic family x band x fire-model selection (selection rule version 1.0).

    The PR #33 rule with the fire model as part of the cell key
    (docs/difficulty-calibration.md section 8): families are filled scarcest
    first (fewest eligible candidates; ties by name); inside a family, bands in
    Easy..Stress order, stochastic before deterministic; inside a cell,
    candidates in the order of a keyed SHA-256 of their content hash. Each
    candidate is used once; the family it fills becomes its primary family. A
    cell that cannot be filled is reported in ``shortfall``, never filled from
    another band or fire model.

    ``previous`` keeps an earlier selection fixed and fills only its shortfall
    (the top-up rule). The result depends only on the candidates' content and
    measurements, the quotas and the key: never on the candidate order, a
    model, a clock or the environment.
    """
    pool = sorted((c for c in candidates if eligible(c)), key=lambda c: (_selection_rank(selection_key, c.scenario_hash), c.scenario_hash))
    if len({c.scenario_hash for c in pool}) != len(pool):
        raise ValueError("duplicate scenario content hash in the candidate pool")
    chosen: list[tuple[QuotaCell, MeasuredCandidate]] = list(previous.chosen) if previous else []
    used = {c.scenario_hash for _, c in chosen}
    filled = Counter(cell for cell, _ in chosen)
    tags = {c.scenario_hash: c.tags for c in pool}
    families = sorted({f for f, _, _ in quotas}, key=lambda f: (sum(1 for c in pool if f in tags[c.scenario_hash]), f.value))
    shortfall: list[tuple[QuotaCell, int]] = []
    for family in families:
        for band in BANDS:
            for stochastic in (True, False):
                cell = (family, band, stochastic)
                need = quotas.get(cell, 0) - filled[cell]
                for c in pool:
                    if need <= 0:
                        break
                    if c.scenario_hash in used or family not in tags[c.scenario_hash] or c.band != band or c.stochastic != stochastic:
                        continue
                    used.add(c.scenario_hash)
                    chosen.append((cell, c))
                    need -= 1
                if need > 0:
                    shortfall.append((cell, need))
    return Selection(tuple(chosen), tuple(shortfall))


# --------------------------------------------------------------------------- the pool workflow


@dataclass(frozen=True)
class PoolRound:
    """One generation round: the batches it generated and the shortfall after selecting with them."""

    index: int
    batches: tuple[GeneratedBatch, ...]
    shortfall_after: tuple[tuple[QuotaCell, int], ...]


@dataclass(frozen=True)
class CompositionResult:
    """A complete pool run: every candidate (deduplicated), every round and the final selection."""

    design: SuiteDesign
    recipes: tuple[Recipe, ...]
    candidates: tuple[MeasuredCandidate, ...]
    dropped: Mapping[str, int]  # "duplicate" / "excluded_hash" -> count
    rounds: tuple[PoolRound, ...]
    selection: Selection
    commitments: Mapping[str, str]
    seconds: Mapping[str, float]  # diagnostic wall-clock timings; never frozen


def _run(fn: Callable[..., Any], tasks: Sequence[tuple[Any, ...]], workers: int) -> list[Any]:
    if workers <= 1 or len(tasks) <= 1:
        return [fn(*t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, *zip(*tasks, strict=True), chunksize=1))


def _batch_tasks(plan: Sequence[tuple[Recipe, int]], next_index: dict[str, int], batch_size: int) -> list[tuple[Recipe, int, int]]:
    tasks: list[tuple[Recipe, int, int]] = []
    for recipe, count in plan:
        while count > 0:
            n = min(batch_size, count)
            index = next_index.get(recipe.recipe_id, 0)
            next_index[recipe.recipe_id] = index + 1
            tasks.append((recipe, index, n))
            count -= n
    return tasks


def compose_pool(
    *,
    pool_seed: int,
    evaluation_seeds: Sequence[int],
    selection_key: str,
    design: SuiteDesign = V1_DESIGN,
    recipes: Sequence[Recipe] = V1_RECIPES,
    initial_plan: Sequence[tuple[str, int]] = V1_INITIAL_PLAN,
    top_up_batch: int = TOP_UP_BATCH_SIZE,
    pool_cap: int = POOL_CAP,
    exclude_hashes: Iterable[str] = (),
    workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> CompositionResult:
    """Generate, measure, calibrate, tag and select a candidate pool, topping up only the short cells.

    Inputs are evaluator-private: ``pool_seed`` (the generator seed of every
    batch is derived from it), ``evaluation_seeds`` (the ordered candidate seed
    list of the evaluation worlds, also given to
    :func:`~emberbench.build_unique_world_manifest`) and ``selection_key``.
    ``exclude_hashes`` lists scenario hashes that may never enter the suite
    (the public suite, earlier rounds).

    Round 0 generates ``initial_plan``. While a cell is short and the pool is
    below ``pool_cap``, each later round generates one batch of
    ``top_up_batch`` candidates with :func:`top_up_recipe` for every short
    cell (one batch per distinct recipe) and fills only the short cells. The
    stopping rule reads cell counts only. The result is a pure function of
    the inputs; ``workers`` changes only the wall-clock time.
    """
    say = log or (lambda _msg: None)
    validate_generator_seed(pool_seed)
    seeds = [validate_seed(s) for s in evaluation_seeds]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise GenerationError("evaluation_seeds must be a non-empty list of distinct seeds")
    if not isinstance(selection_key, str) or not selection_key:
        raise GenerationError("selection_key must be a non-empty string")
    catalog = {r.recipe_id: r for r in recipes}
    if len(catalog) != len(recipes):
        raise GenerationError("recipe IDs must be unique")
    excluded = frozenset(exclude_hashes)
    quotas = design.quotas()
    next_index: dict[str, int] = {}
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    candidates: list[MeasuredCandidate] = []
    rounds: list[PoolRound] = []
    selection: Selection | None = None
    timing: dict[str, float] = dict.fromkeys(("generation_wall", "generation_cpu", "measurement_wall", "measurement_cpu"), 0.0)
    plan = [(catalog[rid], n) for rid, n in initial_plan]
    while True:
        tasks = _batch_tasks(plan, next_index, top_up_batch if rounds else max(top_up_batch, 1))
        t0 = time.perf_counter()
        batches = _run(generate_batch, [(r, pool_seed, i, n) for r, i, n in tasks], workers)
        timing["generation_wall"] += time.perf_counter() - t0
        measure: list[tuple[Any, ...]] = []
        for b in sorted(batches, key=lambda b: (b.recipe.recipe_id, b.batch_index)):
            timing["generation_cpu"] += b.seconds
            for ordinal, (scenario, proxies) in enumerate(b.scenarios):
                h = scenario.content_hash()
                if h in excluded:
                    dropped["excluded_hash"] += 1
                    continue
                if h in seen:
                    dropped["duplicate"] += 1
                    continue
                seen.add(h)
                origin = CandidateOrigin(b.recipe.recipe_id, b.recipe.recipe_version, b.batch_index, b.batch_id, ordinal)
                measure.append((origin, scenario, proxies, seeds, design.stochastic_worlds))
        # Heaviest first, so the slow crowd candidates do not finish last.
        measure.sort(key=lambda t: (-len(t[1].people) * (design.stochastic_worlds if t[1].fire.model is FireModelName.STOCHASTIC else 1), t[0].sort_key()))
        t0 = time.perf_counter()
        measured = _run(measure_candidate, measure, workers)
        timing["measurement_wall"] += time.perf_counter() - t0
        timing["measurement_cpu"] += sum(m.seconds for m in measured)
        candidates.extend(measured)
        candidates.sort(key=lambda c: c.origin.sort_key())
        selection = select_suite(candidates, quotas, selection_key, previous=selection)
        rounds.append(PoolRound(len(rounds), tuple(batches), selection.shortfall))
        say(f"round {len(rounds) - 1}: {len(candidates)} candidates, {sum(n for _, n in selection.shortfall)} slots short")
        if selection.complete or len(candidates) >= pool_cap:
            break
        recipes_needed: list[Recipe] = []
        for cell, _ in sorted(selection.shortfall, key=lambda cn: _cell_key(cn[0])):
            r = top_up_recipe(cell, recipes)
            if r not in recipes_needed:
                recipes_needed.append(r)
        room = pool_cap - len(candidates)
        plan = []
        for r in recipes_needed:
            if room <= 0:
                break
            plan.append((r, min(top_up_batch, room)))
            room -= top_up_batch
    assert selection is not None
    commitments = {
        "pool_seed": commit("pool-seed", pool_seed),
        "evaluation_seeds": commit("evaluation-seeds", seeds),
        "selection_key": commit("selection-key", selection_key),
    }
    return CompositionResult(design, tuple(recipes), tuple(candidates), dict(dropped), tuple(rounds), selection, commitments, dict(timing))


# --------------------------------------------------------------------------- freeze record


def versions() -> dict[str, str]:
    """Every version a composed suite depends on."""
    return {
        "composition": COMPOSITION_VERSION,
        "generator": GENERATOR_VERSION,
        "generator_config": GENERATOR_CONFIG_VERSION,
        "generation_manifest": GENERATION_MANIFEST_VERSION,
        "recipe_catalog": RECIPE_CATALOG_VERSION,
        "family_rules": FAMILY_RULES_VERSION,
        "selection_rules": SELECTION_RULES_VERSION,
        "simulator": SIMULATOR_VERSION,
        "observation": OBSERVATION_VERSION,
        "scenario_schema": SCHEMA_VERSION,
        "scoring": SCORING_VERSION,
        "environment_world": ENVIRONMENT_WORLD_VERSION,
        **{k: v for k, v in calibration_rules().items() if k.endswith("_version")},
    }


def rule_set(design: SuiteDesign = V1_DESIGN, recipes: Sequence[Recipe] = V1_RECIPES) -> dict[str, Any]:
    """The frozen rule set of a composition: versions, rules, design and recipes. Public."""
    return {
        "versions": versions(),
        "family_rules": family_rules(),
        "calibration_rules": calibration_rules(),
        "selection_rules": {
            "selection_rules_version": SELECTION_RULES_VERSION,
            "family_order": "scarcest eligible first, ties by name",
            "cell_order": "bands easy..stress; stochastic before deterministic",
            "candidate_order": "sha256('emberbench/suite-selection/' + key + '/' + scenario_hash)",
            "top_up": {
                "batch": TOP_UP_BATCH_SIZE,
                "cap": POOL_CAP,
                "initial": INITIAL_POOL_SIZE,
                "recipes": sorted([f.value, st, rid] for (f, st), rid in V1_TOP_UP.items()) if tuple(recipes) == V1_RECIPES else None,
            },
        },
        "design": design.to_dict(),
        "recipes": [r.to_dict() for r in recipes],
    }


def rule_set_digest(design: SuiteDesign = V1_DESIGN, recipes: Sequence[Recipe] = V1_RECIPES) -> str:
    return sha256_hex(rule_set(design, recipes))


@dataclass(frozen=True)
class SelectedScenario:
    """One frozen suite scenario with its candidate origin, assignment and evidence."""

    scenario_id: str  # neutral suite ID
    scenario_hash: str  # hash of the relabeled suite scenario
    candidate_hash: str  # hash of the candidate as generated and measured
    origin: CandidateOrigin
    primary_family: ScenarioFamily
    band: str
    stochastic: bool
    tags: tuple[ScenarioFamily, ...]
    structural_tags: tuple[ScenarioFamily, ...]
    proxies: GenerationProxies
    calibration: Calibration
    world_seed_indices: tuple[int, ...]
    world_digests: tuple[str, ...]  # environment_world_digest of the suite scenario, per world
    worlds: tuple[WorldMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "candidate_hash": self.candidate_hash,
            "origin": self.origin.to_dict(),
            "primary_family": self.primary_family.value,
            "band": self.band,
            "fire_model": "stochastic" if self.stochastic else "deterministic",
            "tags": [f.value for f in self.tags],
            "structural_tags": [f.value for f in self.structural_tags],
            "proxies": self.proxies.to_dict(),
            "calibration": self.calibration.to_dict(),
            "worlds": [
                {"seed_index": i, "environment_world_digest": d, **{k: v for k, v in w.to_dict().items() if k != "environment_world_digest"}}
                for i, d, w in zip(self.world_seed_indices, self.world_digests, self.worlds, strict=True)
            ],
        }


@dataclass(frozen=True)
class SuiteFreezeRecord:
    """The canonical, PRIVATE record of one composed suite (suite freeze record version 1.0).

    It holds the versions, the rule set and its digest, the recipes, every
    batch with its generator-seed commitment and generation hash, the
    commitments to the pool seed, the evaluation seeds and the selection key,
    pool statistics, the selected scenarios with their family and band
    assignments, unique-world digests and evidence, and every shortfall. It
    holds no seed and no key. Publish only :meth:`public_commitment` while the
    suite is live.
    """

    suite_id: str
    suite_version: str
    suite_hash: str
    rule_set_digest: str
    rule_set: Mapping[str, Any]
    batches: tuple[Mapping[str, Any], ...]
    commitments: Mapping[str, str]
    pool: Mapping[str, Any]
    selected: tuple[SelectedScenario, ...]
    shortfall: tuple[tuple[QuotaCell, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_freeze_record_version": SUITE_FREEZE_RECORD_VERSION,
            "suite": {"suite_id": self.suite_id, "suite_version": self.suite_version, "suite_hash": self.suite_hash},
            "rule_set_digest": self.rule_set_digest,
            "rule_set": dict(self.rule_set),
            "commitments": dict(self.commitments),
            "batches": [dict(b) for b in self.batches],
            "pool": dict(self.pool),
            "selected": [s.to_dict() for s in self.selected],
            "shortfall": [[f.value, b, st, n] for (f, b, st), n in self.shortfall],
        }

    def digest(self) -> str:
        return sha256_hex(self.to_dict())

    def cell_counts(self) -> dict[str, int]:
        counts = Counter(f"{s.primary_family.value}/{s.band}/{'stochastic' if s.stochastic else 'deterministic'}" for s in self.selected)
        return dict(sorted(counts.items()))

    def public_commitment(self) -> dict[str, Any]:
        """What may be published before evaluation: digests and cell counts only (PR #33 section 14, step 10)."""
        return {
            "suite_freeze_record_version": SUITE_FREEZE_RECORD_VERSION,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "suite_hash": self.suite_hash,
            "freeze_record_digest": self.digest(),
            "rule_set_digest": self.rule_set_digest,
            "commitments": dict(self.commitments),
            "cell_counts": self.cell_counts(),
            "shortfall_slots": sum(n for _, n in self.shortfall),
        }

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def pool_statistics(result: CompositionResult) -> dict[str, Any]:
    """Deterministic pool counts: per recipe, per admission outcome, per band and per family x band x fire model."""
    per_recipe: Counter[str] = Counter(c.origin.recipe_id for c in result.candidates)
    admission: Counter[str] = Counter(c.admission for c in result.candidates)
    cells: Counter[str] = Counter()
    for c in result.candidates:
        if eligible(c):
            for f in primary_candidates(c.tags):
                cells[f"{f.value}/{c.band}/{'stochastic' if c.stochastic else 'deterministic'}"] += 1
    return {
        "candidates": len(result.candidates),
        "dropped": dict(sorted(result.dropped.items())),
        "rounds": len(result.rounds),
        "per_recipe": dict(sorted(per_recipe.items())),
        "admission": dict(sorted(admission.items())),
        "eligible_by_cell": dict(sorted(cells.items())),
    }


@dataclass(frozen=True)
class ComposedSuite:
    """An in-memory composed suite, not yet written: the suite manifest, the scenarios and the freeze record."""

    manifest: SuiteManifest
    scenarios: tuple[Scenario, ...]
    record: SuiteFreezeRecord


def compose_suite(result: CompositionResult, *, suite_id: str, selection_key: str) -> ComposedSuite:
    """Relabel the selected candidates neutrally and build the suite and its freeze record.

    The suite order is a keyed hash of the candidate hashes, so a position or
    an ID says nothing about a family, a band or a recipe. Each scenario keeps
    its bytes except the ID. The fire stream does not read the ID, so every
    world keeps its trajectory; this is checked, and the world digests of the
    relabeled scenarios are recorded.
    """
    if not isinstance(suite_id, str) or not ID_PATTERN.fullmatch(suite_id):
        raise GenerationError(f"suite_id must match {ID_PATTERN.pattern}")
    if commit("selection-key", selection_key) != result.commitments["selection_key"]:
        raise GenerationError("selection_key does not match the composition's commitment")
    chosen = sorted(result.selection.chosen, key=lambda cc: _order_rank(selection_key, cc[1].scenario_hash))
    scenarios: list[Scenario] = []
    entries: list[SuiteEntry] = []
    selected: list[SelectedScenario] = []
    for position, (cell, c) in enumerate(chosen):
        family, band, stochastic = cell
        final = replace(c.scenario, scenario_id=scenario_id_for(position))
        digests = []
        for w in c.worlds:
            if fire_trajectory(final, w.seed) != fire_trajectory(c.scenario, w.seed):
                raise GenerationError(f"{final.scenario_id}: relabeling changed a fire trajectory")
            digests.append(environment_world_digest(final, w.seed))
        tags = c.tags
        scenarios.append(final)
        entries.append(SuiteEntry.describe(final, families=tuple(sorted(tags)), primary=family, purpose=COMPOSED_PURPOSE))
        selected.append(
            SelectedScenario(
                scenario_id=final.scenario_id,
                scenario_hash=final.content_hash(),
                candidate_hash=c.scenario_hash,
                origin=c.origin,
                primary_family=family,
                band=band,
                stochastic=stochastic,
                tags=tuple(sorted(tags)),
                structural_tags=tuple(sorted(c.structural_tags)),
                proxies=c.proxies,
                calibration=c.calibration,
                world_seed_indices=c.world_seed_indices,
                world_digests=tuple(digests),
                worlds=c.worlds,
            )
        )
    manifest = SuiteManifest(
        suite_id=suite_id,
        suite_version=COMPOSED_SUITE_VERSION,
        visibility=Visibility.GENERATED,
        seed_role=SeedRole.DEVELOPMENT,
        schema_version=SCHEMA_VERSION,
        entries=tuple(entries),
    )
    manifest = SuiteManifest.from_dict(manifest.to_dict())
    batches = [b.to_dict() for r in result.rounds for b in sorted(r.batches, key=lambda b: (b.recipe.recipe_id, b.batch_index))]
    record = SuiteFreezeRecord(
        suite_id=suite_id,
        suite_version=COMPOSED_SUITE_VERSION,
        suite_hash=manifest.suite_hash(),
        rule_set_digest=rule_set_digest(result.design, result.recipes),
        rule_set=rule_set(result.design, result.recipes),
        batches=tuple(batches),
        commitments=dict(result.commitments),
        pool=pool_statistics(result),
        selected=tuple(selected),
        shortfall=result.selection.shortfall,
    )
    return ComposedSuite(manifest, tuple(scenarios), record)


@dataclass(frozen=True)
class FrozenComposedSuite:
    root: Path
    suite: Suite
    record: Mapping[str, Any] = field(repr=False)


def freeze_composed_suite(composed: ComposedSuite, directory: str | Path) -> FrozenComposedSuite:
    """Write the suite once into an empty or new ``directory`` (outside the repository) and load it back verified.

    Layout: ``directory/suite/`` is a normal suite directory and
    ``directory/suite-freeze.json`` is the private freeze record (the leak
    guard rejects that file name in the repository and in distributions).
    """
    root = Path(directory)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise GenerationError(f"{root}: already exists and is not empty; a frozen suite is never overwritten")
    (root / COMPOSED_SUITE_DIR_NAME).mkdir(parents=True)
    for s in composed.scenarios:
        (root / COMPOSED_SUITE_DIR_NAME / f"{s.scenario_id}.json").write_bytes(scenario_file_bytes(s))
    (root / COMPOSED_SUITE_DIR_NAME / MANIFEST_NAME).write_bytes(composed.manifest.to_file_text().encode("ascii"))
    # The freeze record is written last, so an interrupted freeze cannot load.
    (root / SUITE_FREEZE_RECORD_NAME).write_bytes(composed.record.to_file_text().encode("ascii"))
    return load_composed_suite(root)


def load_composed_suite(directory: str | Path) -> FrozenComposedSuite:
    """Load a frozen composed suite and check it against its freeze record (hashes, IDs, structural tags)."""
    root = Path(directory)
    path = root / SUITE_FREEZE_RECORD_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GenerationError(f"{path}: freeze record not found") from None
    if record.get("suite_freeze_record_version") != SUITE_FREEZE_RECORD_VERSION:
        raise GenerationError(f"{path}: unsupported freeze record version")
    suite = load_suite(root / COMPOSED_SUITE_DIR_NAME)
    if suite.manifest.suite_hash() != record["suite"]["suite_hash"]:
        raise GenerationError(f"{root}: suite hash does not match the freeze record")
    selected = record["selected"]
    if [s.scenario_id for s in suite.scenarios] != [e["scenario_id"] for e in selected]:
        raise GenerationError(f"{root}: scenario IDs do not match the freeze record")
    for scenario, entry, suite_entry in zip(suite.scenarios, selected, suite.manifest.entries, strict=True):
        if scenario.content_hash() != entry["scenario_hash"]:
            raise GenerationError(f"{root}: {scenario.scenario_id} does not match the freeze record")
        structural = sorted(f.value for f in structural_tags(profile_structure(scenario)))
        if structural != entry["structural_tags"] or not set(structural) <= {f.value for f in suite_entry.families}:
            raise GenerationError(f"{root}: {scenario.scenario_id} structural tags do not match the scenario")
        if suite_entry.primary_family.value != entry["primary_family"] or entry["primary_family"] not in entry["tags"]:
            raise GenerationError(f"{root}: {scenario.scenario_id} primary family does not match the freeze record")
    return FrozenComposedSuite(root, suite, record)


def schedule_composed_worlds(frozen: FrozenComposedSuite, evaluation_seeds: Sequence[int]) -> EvaluationManifest:
    """The unique-world evaluation manifest of a frozen composed suite, checked against the frozen world digests.

    It is :func:`~emberbench.build_unique_world_manifest` with the private
    evaluation seeds of the composition; every scheduled world must be the
    world the suite was calibrated and tagged on.
    """
    if commit("evaluation-seeds", [validate_seed(s) for s in evaluation_seeds]) != frozen.record["commitments"]["evaluation_seeds"]:
        raise GenerationError("evaluation_seeds do not match the freeze record's commitment")
    stochastic_worlds = frozen.record["rule_set"]["design"]["stochastic_worlds"]
    manifest = build_unique_world_manifest(frozen.suite, evaluation_seeds, stochastic_worlds=stochastic_worlds)
    expected = {e["scenario_id"]: [w["environment_world_digest"] for w in e["worlds"]] for e in frozen.record["selected"]}
    got: dict[str, list[str]] = {}
    for spec in manifest.episodes:
        got.setdefault(spec.scenario_id, []).append(environment_world_digest(frozen.suite.get(spec.scenario_id), spec.seed))
    if got != expected:
        raise GenerationError("the scheduled unique worlds differ from the frozen calibrated worlds")
    return manifest
