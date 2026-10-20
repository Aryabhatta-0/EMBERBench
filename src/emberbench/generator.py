"""Procedural unseen scenarios: a versioned generator config, a private-seeded generator and frozen suites.

The generator code and the :class:`GeneratorConfig` are public. The
generator seed is evaluator-private until the evaluation is over. A
generated suite is a pure function of the config, the seed and the
generator version: no global ``random``, clock, UUID, hash order or OS
entropy is used. Every draw comes from a SplitMix64 stream in its own
domain (``emberbench/generator/...``), separate from the simulator's fire
streams.

The workflow is: generate, freeze, evaluate. :func:`generate_suite` builds
the scenarios in memory, :func:`freeze_generated_suite` writes them once as a
normal suite directory plus a ``generation.json`` freeze record, and
:func:`load_generated_suite` verifies both and returns a
:class:`~emberbench.suite.Suite` for ``build_evaluation_manifest`` and
``run_evaluation``. Every model is then run against that one frozen suite.

A candidate scenario is accepted only if every generation constraint holds.
The constraints are measured proxies (see :class:`GenerationProxies`), not a
universal difficulty truth. If no candidate passes within
``max_attempts``, generation fails with :class:`GenerationError`; a
constraint is never weakened. See ``docs/procedural-generation.md``.
"""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .canonical import sha256_hex
from .errors import GenerationError
from .families import descriptive_primary, structural_tags
from .grid import Pos
from .rng import MASK64, SplitMix64
from .scenario import (
    ID_PATTERN,
    MAX_STEPS_LIMIT,
    SCHEMA_VERSION,
    EnterFirePolicy,
    ExitSpec,
    FireConfig,
    FireModelName,
    MovementConfig,
    PersonSpec,
    Scenario,
    TerminationConfig,
)
from .structure import ROUTE_REDUNDANCY_CAP, GenerationProxies, _bfs, measure_proxies, profile_structure
from .suite import HASH_PATTERN, MANIFEST_NAME, ScenarioFamily, SeedRole, Suite, SuiteEntry, SuiteManifest, Visibility, load_suite

# 1.1.0: family labels follow the frozen V1 family rules (emberbench.families,
# family-rule version 1.0); new structural proxies and constraints.
GENERATOR_VERSION = "1.1.0"
# 1.1 adds the optional split_pressure, deceptive_share and bottleneck_severity
# constraints. A 1.0 config is still accepted and has the same canonical form.
GENERATOR_CONFIG_VERSION = "1.1"
SUPPORTED_CONFIG_VERSIONS: tuple[str, ...] = ("1.0", GENERATOR_CONFIG_VERSION)
# 1.1: the pinned proxies hold the new structural measurements.
GENERATION_MANIFEST_VERSION = "1.1"
GENERATION_MANIFEST_NAME = "generation.json"
SUITE_DIR_NAME = "suite"
GENERATED_SUITE_VERSION = "1.0.0"
GENERATED_PURPOSE = "Procedurally generated unseen evaluation scenario."

MAX_GENERATOR_SEED = (1 << 256) - 1
MIN_GENERATED_GRID_SIDE = 5
MAX_GENERATED_GRID_SIDE = 128
MAX_SCENARIO_COUNT = 9999
MAX_ATTEMPTS_LIMIT = 100_000
# Scatter density is drawn in steps of 1/1000.
DENSITY_RESOLUTION = 1000
# Upper limit of the unbounded ratio constraints (split pressure, bottleneck severity).
MAX_RATIO_CONSTRAINT = 1_000_000.0

_DOMAIN = b"emberbench/generator/"
_E = TypeVar("_E")


# --------------------------------------------------------------------------- private randomness


def validate_generator_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"generator seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_GENERATOR_SEED:
        raise ValueError("generator seed must be in [0, 2**256 - 1]")
    return seed


def derive_generation_stream(generator_seed: int, config_hash: str, ordinal: int, attempt: int) -> int:
    """The 64-bit SplitMix64 state for one candidate of one generated scenario.

    The domain prefix ``emberbench/generator/`` never occurs in the simulator's
    stream labels (``emberbench/v1/<seed>/<label>``), so generation draws and
    fire draws can never share a stream. Each ordinal and each attempt gets
    its own stream, so a rejection never shifts another scenario's draws.
    """
    seed = validate_generator_seed(generator_seed)
    label = f"{GENERATOR_VERSION}/{config_hash}/{ordinal}/{attempt}/".encode("ascii")
    digest = hashlib.sha256(_DOMAIN + label + seed.to_bytes(32, "big")).digest()
    return int.from_bytes(digest[:8], "big")


class GeneratorRng:
    """Uniform integer draws on a SplitMix64 stream owned by one candidate."""

    __slots__ = ("_stream",)

    def __init__(self, state: int) -> None:
        self._stream = SplitMix64(state)

    def below(self, n: int) -> int:
        """An unbiased integer in ``[0, n)`` (rejection sampling, no modulo bias)."""
        if not 1 <= n <= MASK64 + 1:
            raise ValueError(f"n must be in [1, 2**64], got {n}")
        limit = ((MASK64 + 1) // n) * n
        while True:
            x = self._stream.next_u64()
            if x < limit:
                return x % n

    def integer(self, lo: int, hi: int) -> int:
        """An integer in ``[lo, hi]``."""
        return lo + self.below(hi - lo + 1)

    def coin(self) -> bool:
        return self.below(2) == 1

    def choice(self, items: list[_E]) -> _E:
        return items[self.below(len(items))]

    def shuffled(self, items: Iterable[_E]) -> list[_E]:
        """A Fisher-Yates permutation of ``items``. The input order must already be canonical."""
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class IntRange:
    lo: int
    hi: int

    def to_dict(self) -> dict[str, Any]:
        return {"min": self.lo, "max": self.hi}

    def contains(self, value: int) -> bool:
        return self.lo <= value <= self.hi


@dataclass(frozen=True)
class FloatRange:
    lo: float
    hi: float

    def to_dict(self) -> dict[str, Any]:
        return {"min": self.lo, "max": self.hi}

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclass(frozen=True)
class LayoutConfig:
    """Obstacle layout: straight partition walls with doors, then scattered wall cells.

    ``door_width`` is the corridor-width control and ``doors_per_partition``
    the route-redundancy control. ``scatter_density`` is the share of the
    remaining free interior cells that become walls, in steps of 1/1000.
    """

    partition_count: IntRange
    doors_per_partition: IntRange
    door_width: IntRange
    scatter_density: FloatRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_count": self.partition_count.to_dict(),
            "doors_per_partition": self.doors_per_partition.to_dict(),
            "door_width": self.door_width.to_dict(),
            "scatter_density": self.scatter_density.to_dict(),
        }


@dataclass(frozen=True)
class FireGenerationConfig:
    """Fire rules for generated scenarios. ``spread_probability_choices`` is set iff the model is stochastic."""

    model: FireModelName
    spread_interval: IntRange
    exits_flammable: bool
    spread_probability_choices: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model.value,
            "spread_interval": self.spread_interval.to_dict(),
            "exits_flammable": self.exits_flammable,
        }
        if self.model is FireModelName.STOCHASTIC:
            d["spread_probability_choices"] = list(self.spread_probability_choices)
        return d


@dataclass(frozen=True)
class GenerationConstraints:
    """Accepted ranges for the measured proxies of :class:`GenerationProxies`.

    These are generation constraints, not a difficulty truth. Route existence
    (every person has a fire-free route of at most ``max_steps`` cells) is
    always required and is not configurable.

    The three optional constraints (config version 1.1) let a recipe target a
    structural family: ``split_pressure`` (``None`` counts as 0),
    ``deceptive_share`` and ``bottleneck_severity`` (population / min cut;
    infinite when no route exists). ``None`` means unconstrained.
    """

    route_length: IntRange
    obstacle_density: FloatRange
    crowd_density: FloatRange
    exit_separation: IntRange
    passage_width: IntRange
    route_redundancy: IntRange
    ignition_person_distance: IntRange
    ignition_exit_distance: IntRange
    split_pressure: FloatRange | None = None
    deceptive_share: FloatRange | None = None
    bottleneck_severity: FloatRange | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name).to_dict() for name in _CONSTRAINT_NAMES}
        for name in _OPTIONAL_CONSTRAINT_NAMES:
            value = getattr(self, name)
            if value is not None:
                out[name] = value.to_dict()
        return out


_CONSTRAINT_NAMES = (
    "route_length",
    "obstacle_density",
    "crowd_density",
    "exit_separation",
    "passage_width",
    "route_redundancy",
    "ignition_person_distance",
    "ignition_exit_distance",
)
_FLOAT_CONSTRAINTS = frozenset({"obstacle_density", "crowd_density"})
# Config version 1.1 only. Float ranges; split pressure and bottleneck severity are unbounded ratios.
_OPTIONAL_CONSTRAINT_NAMES = ("split_pressure", "deceptive_share", "bottleneck_severity")
_RATIO_CONSTRAINTS = frozenset({"split_pressure", "bottleneck_severity"})


@dataclass(frozen=True)
class GeneratorConfig:
    """The public, versioned input of the generator. Together with the private seed it fixes every byte."""

    scenario_count: int
    width: int
    height: int
    population: IntRange
    exit_count: IntRange
    ignition_count: IntRange
    layout: LayoutConfig
    fire: FireGenerationConfig
    enter_fire: EnterFirePolicy
    max_steps: int
    max_attempts: int
    constraints: GenerationConstraints
    config_version: str = GENERATOR_CONFIG_VERSION

    def __post_init__(self) -> None:
        _validate_config(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "scenario_count": self.scenario_count,
            "grid": {"width": self.width, "height": self.height},
            "population": self.population.to_dict(),
            "exit_count": self.exit_count.to_dict(),
            "ignition_count": self.ignition_count.to_dict(),
            "layout": self.layout.to_dict(),
            "fire": self.fire.to_dict(),
            "movement": {"enter_fire": self.enter_fire.value},
            "max_steps": self.max_steps,
            "max_attempts": self.max_attempts,
            "constraints": self.constraints.to_dict(),
        }

    def config_hash(self) -> str:
        """SHA-256 of the canonical config. It is public and seeds no stream on its own."""
        return sha256_hex(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> "GeneratorConfig":
        return _parse_config(data)

    @classmethod
    def from_json(cls, text: str) -> "GeneratorConfig":
        return _parse_config(_loads(text, "generator config"))


# --------------------------------------------------------------------------- config parsing and validation


def _fail(where: str, message: str) -> GenerationError:
    return GenerationError(f"{where}: {message}")


def _loads(text: str, what: str) -> Any:
    def reject(name: str) -> Any:
        raise GenerationError(f"{what}: non-finite number {name!r} is not allowed")

    try:
        return json.loads(text, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"{what}: not valid JSON: {exc}") from exc


def _expect_object(value: Any, where: str, required: set[str], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    missing = sorted(required - value.keys())
    if missing:
        raise _fail(where, f"missing required keys {missing}")
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        raise _fail(where, f"unknown keys {unknown}")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _expect_int(value: Any, where: str) -> int:
    if not _is_int(value):
        raise _fail(where, f"expected an integer, got {type(value).__name__}")
    return int(value)


def _expect_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(where, f"expected a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise _fail(where, "must be finite")
    return result


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise _fail(where, f"expected a string, got {type(value).__name__}")
    return value


def _parse_int_range(value: Any, where: str) -> IntRange:
    obj = _expect_object(value, where, {"min", "max"})
    return IntRange(_expect_int(obj["min"], f"{where}.min"), _expect_int(obj["max"], f"{where}.max"))


def _parse_float_range(value: Any, where: str) -> FloatRange:
    obj = _expect_object(value, where, {"min", "max"})
    return FloatRange(_expect_float(obj["min"], f"{where}.min"), _expect_float(obj["max"], f"{where}.max"))


def _parse_config(data: Any) -> GeneratorConfig:
    keys = {
        "config_version",
        "scenario_count",
        "grid",
        "population",
        "exit_count",
        "ignition_count",
        "layout",
        "fire",
        "movement",
        "max_steps",
        "max_attempts",
        "constraints",
    }
    obj = _expect_object(data, "config", keys)
    grid = _expect_object(obj["grid"], "grid", {"width", "height"})
    lay = _expect_object(obj["layout"], "layout", {"partition_count", "doors_per_partition", "door_width", "scatter_density"})
    fire = _expect_object(obj["fire"], "fire", {"model", "spread_interval", "exits_flammable"}, frozenset({"spread_probability_choices"}))
    movement = _expect_object(obj["movement"], "movement", {"enter_fire"})
    optional = frozenset(_OPTIONAL_CONSTRAINT_NAMES) if obj["config_version"] != "1.0" else frozenset()
    cons = _expect_object(obj["constraints"], "constraints", set(_CONSTRAINT_NAMES), optional)

    try:
        model = FireModelName(_expect_str(fire["model"], "fire.model"))
    except ValueError:
        raise _fail("fire.model", f"must be one of {[m.value for m in FireModelName]}") from None
    try:
        enter_fire = EnterFirePolicy(_expect_str(movement["enter_fire"], "movement.enter_fire"))
    except ValueError:
        raise _fail("movement.enter_fire", f"must be one of {[m.value for m in EnterFirePolicy]}") from None
    if not isinstance(fire["exits_flammable"], bool):
        raise _fail("fire.exits_flammable", "expected a boolean")
    choices: tuple[float, ...] = ()
    if "spread_probability_choices" in fire:
        raw = fire["spread_probability_choices"]
        if not isinstance(raw, list):
            raise _fail("fire.spread_probability_choices", "expected a list")
        choices = tuple(_expect_float(p, f"fire.spread_probability_choices[{i}]") for i, p in enumerate(raw))
    elif model is FireModelName.STOCHASTIC:
        raise _fail("fire.spread_probability_choices", "required when fire.model is 'stochastic'")
    if model is FireModelName.DETERMINISTIC and "spread_probability_choices" in fire:
        raise _fail("fire.spread_probability_choices", "only allowed when fire.model is 'stochastic'")

    def parse_constraint(name: str) -> Any:
        if name in _FLOAT_CONSTRAINTS:
            return _parse_float_range(cons[name], f"constraints.{name}")
        return _parse_int_range(cons[name], f"constraints.{name}")

    constraint_ranges: dict[str, Any] = {name: parse_constraint(name) for name in _CONSTRAINT_NAMES}
    constraint_ranges.update({name: _parse_float_range(cons[name], f"constraints.{name}") for name in _OPTIONAL_CONSTRAINT_NAMES if name in cons})
    return GeneratorConfig(
        config_version=_expect_str(obj["config_version"], "config_version"),
        scenario_count=_expect_int(obj["scenario_count"], "scenario_count"),
        width=_expect_int(grid["width"], "grid.width"),
        height=_expect_int(grid["height"], "grid.height"),
        population=_parse_int_range(obj["population"], "population"),
        exit_count=_parse_int_range(obj["exit_count"], "exit_count"),
        ignition_count=_parse_int_range(obj["ignition_count"], "ignition_count"),
        layout=LayoutConfig(
            partition_count=_parse_int_range(lay["partition_count"], "layout.partition_count"),
            doors_per_partition=_parse_int_range(lay["doors_per_partition"], "layout.doors_per_partition"),
            door_width=_parse_int_range(lay["door_width"], "layout.door_width"),
            scatter_density=_parse_float_range(lay["scatter_density"], "layout.scatter_density"),
        ),
        fire=FireGenerationConfig(
            model=model,
            spread_interval=_parse_int_range(fire["spread_interval"], "fire.spread_interval"),
            exits_flammable=fire["exits_flammable"],
            spread_probability_choices=choices,
        ),
        enter_fire=enter_fire,
        max_steps=_expect_int(obj["max_steps"], "max_steps"),
        max_attempts=_expect_int(obj["max_attempts"], "max_attempts"),
        constraints=GenerationConstraints(**constraint_ranges),
    )


def _check_int(value: object, where: str, lo: int, hi: int) -> int:
    if not _is_int(value):
        raise _fail(where, f"expected an integer, got {type(value).__name__}")
    assert isinstance(value, int)
    if not lo <= value <= hi:
        raise _fail(where, f"must be in [{lo}, {hi}], got {value}")
    return value


def _check_int_range(r: object, where: str, lo: int, hi: int) -> IntRange:
    if not isinstance(r, IntRange):
        raise _fail(where, "expected an IntRange")
    _check_int(r.lo, f"{where}.min", lo, hi)
    _check_int(r.hi, f"{where}.max", lo, hi)
    if r.lo > r.hi:
        raise _fail(where, f"min {r.lo} is greater than max {r.hi}")
    return r


def _check_float_range(r: object, where: str, cap: float = 1.0) -> FloatRange:
    if not isinstance(r, FloatRange):
        raise _fail(where, "expected a FloatRange")
    for name, value in (("min", r.lo), ("max", r.hi)):
        if not isinstance(value, float) or not 0.0 <= value <= cap:
            raise _fail(f"{where}.{name}", f"must be a float in [0, {cap:g}], got {value!r}")
    if r.lo > r.hi:
        raise _fail(where, f"min {r.lo} is greater than max {r.hi}")
    return r


def _validate_config(c: GeneratorConfig) -> None:
    """Reject malformed configs and constraint sets that can never be met, before any draw."""
    if c.config_version not in SUPPORTED_CONFIG_VERSIONS:
        raise _fail("config_version", f"unsupported version {c.config_version!r}; supported: {list(SUPPORTED_CONFIG_VERSIONS)}")
    _check_int(c.scenario_count, "scenario_count", 1, MAX_SCENARIO_COUNT)
    w = _check_int(c.width, "grid.width", MIN_GENERATED_GRID_SIDE, MAX_GENERATED_GRID_SIDE)
    h = _check_int(c.height, "grid.height", MIN_GENERATED_GRID_SIDE, MAX_GENERATED_GRID_SIDE)
    max_steps = _check_int(c.max_steps, "max_steps", 1, MAX_STEPS_LIMIT)
    _check_int(c.max_attempts, "max_attempts", 1, MAX_ATTEMPTS_LIMIT)
    interior = (w - 2) * (h - 2)
    perimeter = 2 * (w - 2) + 2 * (h - 2)
    big = w * h

    population = _check_int_range(c.population, "population", 1, interior)
    exits = _check_int_range(c.exit_count, "exit_count", 1, perimeter)
    ignitions = _check_int_range(c.ignition_count, "ignition_count", 1, interior)
    if population.hi + ignitions.hi > interior:
        raise _fail("population", f"population.max + ignition_count.max ({population.hi + ignitions.hi}) exceeds the {interior} interior cells")

    lay = c.layout
    if not isinstance(lay, LayoutConfig):
        raise _fail("layout", "expected a LayoutConfig")
    max_partitions = (w - 3) // 2 + (h - 3) // 2
    _check_int_range(lay.partition_count, "layout.partition_count", 0, max_partitions)
    doors = _check_int_range(lay.doors_per_partition, "layout.doors_per_partition", 1, interior)
    door_width = _check_int_range(lay.door_width, "layout.door_width", 1, max(w, h))
    if lay.partition_count.hi > 0 and doors.hi * door_width.hi + doors.hi - 1 > min(w, h) - 2:
        raise _fail("layout", f"{doors.hi} doors of width {door_width.hi} do not fit in a partition of length {min(w, h) - 2}")
    scatter = _check_float_range(lay.scatter_density, "layout.scatter_density")
    for name, value in (("min", scatter.lo), ("max", scatter.hi)):
        if abs(value * DENSITY_RESOLUTION - round(value * DENSITY_RESOLUTION)) > 1e-9:
            raise _fail(f"layout.scatter_density.{name}", f"must be a multiple of 1/{DENSITY_RESOLUTION}, got {value!r}")

    fire = c.fire
    if not isinstance(fire, FireGenerationConfig) or not isinstance(fire.model, FireModelName):
        raise _fail("fire", "expected a FireGenerationConfig")
    _check_int_range(fire.spread_interval, "fire.spread_interval", 1, MAX_STEPS_LIMIT)
    if not isinstance(fire.exits_flammable, bool):
        raise _fail("fire.exits_flammable", "must be a boolean")
    choices = fire.spread_probability_choices
    if fire.model is FireModelName.STOCHASTIC:
        if not choices:
            raise _fail("fire.spread_probability_choices", "needs at least one probability when fire.model is 'stochastic'")
        for p in choices:
            if not isinstance(p, float) or not 0.0 <= p <= 1.0:
                raise _fail("fire.spread_probability_choices", f"must be floats in [0, 1], got {p!r}")
        if list(choices) != sorted(set(choices)):
            raise _fail("fire.spread_probability_choices", "must be sorted and unique")
    elif choices:
        raise _fail("fire.spread_probability_choices", "only allowed when fire.model is 'stochastic'")
    if not isinstance(c.enter_fire, EnterFirePolicy):
        raise _fail("movement.enter_fire", "expected an EnterFirePolicy")

    k = c.constraints
    if not isinstance(k, GenerationConstraints):
        raise _fail("constraints", "expected GenerationConstraints")
    route = _check_int_range(k.route_length, "constraints.route_length", 1, big)
    if route.lo > max_steps:
        raise _fail("constraints.route_length", f"min {route.lo} exceeds max_steps {max_steps}: no person could evacuate")
    if route.hi > max_steps:
        raise _fail("constraints.route_length", f"max {route.hi} exceeds max_steps {max_steps}; set it to at most max_steps")
    _check_float_range(k.obstacle_density, "constraints.obstacle_density")
    _check_float_range(k.crowd_density, "constraints.crowd_density")
    _check_int_range(k.exit_separation, "constraints.exit_separation", 2, w + h)
    _check_int_range(k.passage_width, "constraints.passage_width", 1, max(w, h))
    redundancy = _check_int_range(k.route_redundancy, "constraints.route_redundancy", 1, ROUTE_REDUNDANCY_CAP)
    if redundancy.lo > exits.hi:
        # Every exit is one border cell with one interior approach cell, so a
        # person who is not next to an exit has at most one route per exit.
        raise _fail("constraints.route_redundancy", f"min {redundancy.lo} exceeds exit_count.max {exits.hi}")
    _check_int_range(k.ignition_person_distance, "constraints.ignition_person_distance", 1, big)
    _check_int_range(k.ignition_exit_distance, "constraints.ignition_exit_distance", 1, big)
    for name in _OPTIONAL_CONSTRAINT_NAMES:
        value = getattr(k, name)
        if value is None:
            continue
        if c.config_version == "1.0":
            raise _fail(f"constraints.{name}", "needs config_version '1.1'")
        _check_float_range(value, f"constraints.{name}", MAX_RATIO_CONSTRAINT if name in _RATIO_CONSTRAINTS else 1.0)
    if k.deceptive_share is not None and k.deceptive_share.lo > 0 and exits.hi < 2:
        raise _fail("constraints.deceptive_share", "a positive minimum needs exit_count.max >= 2")
    if k.split_pressure is not None and k.split_pressure.lo > 0 and exits.hi < 2:
        raise _fail("constraints.split_pressure", "a positive minimum needs exit_count.max >= 2")


# --------------------------------------------------------------------------- constraints and family labels


def constraint_violations(proxies: GenerationProxies, config: GeneratorConfig) -> tuple[str, ...]:
    """Names of every generation constraint that ``proxies`` breaks, in a fixed order.

    ``None`` distances count as infinite: they meet any minimum and break any maximum.
    """
    k = config.constraints
    out: list[str] = []
    if proxies.route_length is None or proxies.route_length > config.max_steps:
        out.append("route_exists")
    elif not k.route_length.contains(proxies.route_length):
        out.append("route_length")
    if not k.obstacle_density.contains(proxies.obstacle_density):
        out.append("obstacle_density")
    if not k.crowd_density.contains(proxies.crowd_density):
        out.append("crowd_density")
    if proxies.exit_separation is not None and not k.exit_separation.contains(proxies.exit_separation):
        out.append("exit_separation")
    if not k.passage_width.contains(proxies.passage_width):
        out.append("passage_width")
    if not k.route_redundancy.contains(proxies.route_redundancy):
        out.append("route_redundancy")
    for name, value in (
        ("ignition_person_distance", proxies.ignition_person_distance),
        ("ignition_exit_distance", proxies.ignition_exit_distance),
    ):
        bounds: IntRange = getattr(k, name)
        if value is None or not bounds.contains(value):
            out.append(name)
    if k.split_pressure is not None and not k.split_pressure.contains(proxies.split_pressure or 0.0):
        out.append("split_pressure")
    if k.deceptive_share is not None and not k.deceptive_share.contains(proxies.deceptive_share):
        out.append("deceptive_share")
    if k.bottleneck_severity is not None:
        severity = proxies.bottleneck_severity
        if severity is None or not k.bottleneck_severity.contains(severity):
            out.append("bottleneck_severity")
    return tuple(out)


def describe_families(scenario: Scenario, proxies: GenerationProxies) -> tuple[ScenarioFamily, tuple[ScenarioFamily, ...]]:
    """Structural family labels for the suite manifest: ``(primary, sorted families)``.

    The tags are the seed-free part of the frozen V1 family rules
    (:func:`emberbench.families.structural_tags`). The fire-dependent families
    (threatened nearest exit, route invalidation) are never generator labels:
    they need a fire trajectory, so the evaluator tags them on the selected
    worlds. ``primary`` is :func:`emberbench.families.descriptive_primary`.
    The labels describe the scenario; they are not a claim about its difficulty.
    """
    tags = structural_tags(profile_structure(scenario, proxies))
    return descriptive_primary(tags), tuple(sorted(tags))


# --------------------------------------------------------------------------- generation


def scenario_id_for(ordinal: int) -> str:
    """The neutral ID of a generated scenario. It encodes only the suite position."""
    return f"gen-{ordinal:04d}"


def _neutral_ids(prefix: str, count: int) -> list[str]:
    digits = max(2, len(str(count - 1)))
    return [f"{prefix}{i:0{digits}d}" for i in range(count)]


def _build_candidate(c: GeneratorConfig, rng: GeneratorRng, scenario_id: str) -> Scenario | str:
    """One candidate scenario, or the name of the construction step that could not be completed."""
    w, h = c.width, c.height
    walls: set[Pos] = {Pos(r, col) for r in range(h) for col in range(w) if r in (0, h - 1) or col in (0, w - 1)}
    interior = [Pos(r, col) for r in range(1, h - 1) for col in range(1, w - 1)]

    used: dict[bool, list[int]] = {True: [], False: []}
    for _ in range(rng.integer(c.layout.partition_count.lo, c.layout.partition_count.hi)):
        vertical = rng.coin()
        span = w if vertical else h
        lines = [x for x in range(2, span - 2) if all(abs(x - u) >= 2 for u in used[vertical])]
        if not lines:
            return "layout"
        line = rng.choice(lines)
        used[vertical].append(line)
        length = (h if vertical else w) - 2
        opened: set[int] = set()
        for _ in range(rng.integer(c.layout.doors_per_partition.lo, c.layout.doors_per_partition.hi)):
            dw = rng.integer(c.layout.door_width.lo, c.layout.door_width.hi)
            starts = [x for x in range(length - dw + 1) if not any(i in opened for i in range(x - 1, x + dw + 1))]
            if not starts:
                return "layout"
            start = rng.choice(starts)
            opened.update(range(start, start + dw))
        for i in range(length):
            if i not in opened:
                walls.add(Pos(i + 1, line) if vertical else Pos(line, i + 1))

    free = [p for p in interior if p not in walls]
    permille = rng.integer(round(c.layout.scatter_density.lo * DENSITY_RESOLUTION), round(c.layout.scatter_density.hi * DENSITY_RESOLUTION))
    walls.update(rng.shuffled(free)[: permille * len(free) // DENSITY_RESOLUTION])

    border = [p for p in sorted(walls) if p.row in (0, h - 1) or p.col in (0, w - 1)]
    corners = {Pos(0, 0), Pos(0, w - 1), Pos(h - 1, 0), Pos(h - 1, w - 1)}

    def inner(p: Pos) -> Pos:
        return Pos(min(max(p.row, 1), h - 2), min(max(p.col, 1), w - 2))

    exit_candidates = [p for p in border if p not in corners and inner(p) not in walls]
    n_exits = rng.integer(c.exit_count.lo, c.exit_count.hi)
    separation = c.constraints.exit_separation.lo
    exits: list[Pos] = []
    for p in rng.shuffled(exit_candidates):
        if len(exits) == n_exits:
            break
        if all(abs(p.row - q.row) + abs(p.col - q.col) >= separation for q in exits):
            exits.append(p)
    if len(exits) < n_exits:
        return "exit_placement"
    walls.difference_update(exits)

    def open_cell(p: Pos) -> bool:
        return p not in walls

    # Constructive placement: cells that would break a distance minimum are never drawn.
    from_exit = _bfs(exits, open_cell, w, h)
    ignition_candidates = [
        p for p in interior if p not in walls and from_exit.get(p, math.inf) >= c.constraints.ignition_exit_distance.lo
    ]
    n_ignitions = rng.integer(c.ignition_count.lo, c.ignition_count.hi)
    if len(ignition_candidates) < n_ignitions:
        return "ignition_placement"
    ignitions = rng.shuffled(ignition_candidates)[:n_ignitions]

    from_fire = _bfs(ignitions, open_cell, w, h)
    burning = set(ignitions)
    person_candidates = [
        p
        for p in interior
        if p not in walls and p not in burning and from_fire.get(p, math.inf) >= c.constraints.ignition_person_distance.lo
    ]
    population = rng.integer(c.population.lo, c.population.hi)
    if len(person_candidates) < population:
        return "person_placement"
    # The shuffled order is the ID order, so an ID says nothing about a position.
    people = rng.shuffled(person_candidates)[:population]

    fire = c.fire
    interval = rng.integer(fire.spread_interval.lo, fire.spread_interval.hi)
    probability = rng.choice(list(fire.spread_probability_choices)) if fire.model is FireModelName.STOCHASTIC else None
    return Scenario(
        scenario_id=scenario_id,
        width=w,
        height=h,
        obstacles=tuple(sorted(walls)),
        exits=tuple(ExitSpec(i, p) for i, p in zip(_neutral_ids("x", len(exits)), exits, strict=True)),
        people=tuple(PersonSpec(i, p) for i, p in zip(_neutral_ids("p", len(people)), people, strict=True)),
        ignitions=tuple(ignitions),
        # The scenario seed is a development default. Scored runs take their
        # seed from the evaluation manifest, never from the generator seed.
        seed=0,
        fire=FireConfig(model=fire.model, spread_interval=interval, exits_flammable=fire.exits_flammable, spread_probability=probability),
        movement=MovementConfig(enter_fire=c.enter_fire),
        termination=TerminationConfig(max_steps=c.max_steps),
        description="",
    )


def _generate_one(
    c: GeneratorConfig, config_hash: str, seed: int, ordinal: int, taken: set[str]
) -> tuple[Scenario, GenerationProxies]:
    rejections: Counter[str] = Counter()
    scenario_id = scenario_id_for(ordinal)
    for attempt in range(c.max_attempts):
        rng = GeneratorRng(derive_generation_stream(seed, config_hash, ordinal, attempt))
        candidate = _build_candidate(c, rng, scenario_id)
        if isinstance(candidate, str):
            rejections[candidate] += 1
            continue
        proxies = measure_proxies(candidate)
        violations = constraint_violations(proxies, c)
        if violations:
            rejections.update(violations)
            continue
        if candidate.content_hash() in taken:
            rejections["duplicate_scenario"] += 1
            continue
        return candidate, proxies
    # The rejection counts go to the evaluator in this error only. They are never stored.
    raise GenerationError(
        f"{scenario_id}: no candidate met every generation constraint in {c.max_attempts} attempts; "
        f"rejections by constraint: {dict(sorted(rejections.items()))}. Constraints are never weakened: "
        "change the config or raise max_attempts."
    )


def _check_batch_id(batch_id: object, seed: int) -> str:
    if not isinstance(batch_id, str) or not ID_PATTERN.fullmatch(batch_id):
        raise GenerationError(f"batch_id must match {ID_PATTERN.pattern}, got {batch_id!r}")
    lowered = batch_id.lower()
    for text in (str(seed), format(seed, "x")):
        if len(text) >= 6 and text in lowered:
            raise GenerationError("batch_id must not contain the generator seed")
    return batch_id


# --------------------------------------------------------------------------- generated-suite manifest


@dataclass(frozen=True)
class GeneratedEntry:
    ordinal: int
    scenario_id: str
    content_hash: str
    proxies: GenerationProxies

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "scenario_id": self.scenario_id,
            "content_hash": self.content_hash,
            "proxies": self.proxies.to_dict(),
        }


@dataclass(frozen=True)
class GenerationManifest:
    """The freeze record of a generated suite. It holds no generator seed and no rejection history.

    It pins the generator version, the public config and its hash, the
    evaluator-owned batch ID, the suite hash and, per scenario, the ordinal,
    ID, content hash and measured proxies.
    """

    generation_manifest_version: str
    generator_version: str
    batch_id: str
    config: GeneratorConfig
    config_hash: str
    suite_id: str
    suite_version: str
    suite_hash: str
    schema_version: str
    entries: tuple[GeneratedEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_manifest_version": self.generation_manifest_version,
            "generator_version": self.generator_version,
            "batch_id": self.batch_id,
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "suite_hash": self.suite_hash,
            "schema_version": self.schema_version,
            "scenarios": [e.to_dict() for e in self.entries],
        }

    def generation_hash(self) -> str:
        """SHA-256 of the canonical freeze record. Equal config, seed and batch ID give an equal hash."""
        return sha256_hex(self.to_dict())

    def to_file_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> "GenerationManifest":
        return _parse_generation(data)


def _expect_hash(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not HASH_PATTERN.fullmatch(text):
        raise _fail(where, "must be 64 lowercase hex characters")
    return text


def _expect_optional_int(value: Any, where: str) -> int | None:
    return None if value is None else _expect_int(value, where)


def _parse_proxies(value: Any, where: str) -> GenerationProxies:
    keys = set(GenerationProxies.__dataclass_fields__)
    obj = _expect_object(value, where, keys)
    return GenerationProxies(
        obstacle_density=_expect_float(obj["obstacle_density"], f"{where}.obstacle_density"),
        crowd_density=_expect_float(obj["crowd_density"], f"{where}.crowd_density"),
        exit_separation=_expect_optional_int(obj["exit_separation"], f"{where}.exit_separation"),
        passage_width=_expect_int(obj["passage_width"], f"{where}.passage_width"),
        route_redundancy=_expect_int(obj["route_redundancy"], f"{where}.route_redundancy"),
        ignition_person_distance=_expect_optional_int(obj["ignition_person_distance"], f"{where}.ignition_person_distance"),
        ignition_exit_distance=_expect_optional_int(obj["ignition_exit_distance"], f"{where}.ignition_exit_distance"),
        route_length=_expect_optional_int(obj["route_length"], f"{where}.route_length"),
        min_cut=_expect_int(obj["min_cut"], f"{where}.min_cut"),
        bottleneck_severity=_expect_optional_float(obj["bottleneck_severity"], f"{where}.bottleneck_severity"),
        max_nearest_exit_load=_expect_int(obj["max_nearest_exit_load"], f"{where}.max_nearest_exit_load"),
        split_pressure=_expect_optional_float(obj["split_pressure"], f"{where}.split_pressure"),
        deceptive_share=_expect_float(obj["deceptive_share"], f"{where}.deceptive_share"),
        exit_gap_median=_expect_optional_float(obj["exit_gap_median"], f"{where}.exit_gap_median"),
        detour_ratio=_expect_optional_float(obj["detour_ratio"], f"{where}.detour_ratio"),
        ignition_centrality=_expect_float(obj["ignition_centrality"], f"{where}.ignition_centrality"),
    )


def _expect_optional_float(value: Any, where: str) -> float | None:
    return None if value is None else _expect_float(value, where)


def _parse_generation(data: Any) -> GenerationManifest:
    keys = {
        "generation_manifest_version",
        "generator_version",
        "batch_id",
        "config",
        "config_hash",
        "suite_id",
        "suite_version",
        "suite_hash",
        "schema_version",
        "scenarios",
    }
    obj = _expect_object(data, "generation", keys)
    for name, expected in (
        ("generation_manifest_version", GENERATION_MANIFEST_VERSION),
        ("generator_version", GENERATOR_VERSION),
        ("suite_version", GENERATED_SUITE_VERSION),
        ("schema_version", SCHEMA_VERSION),
    ):
        if obj[name] != expected:
            raise _fail(name, f"is {obj[name]!r}, expected {expected!r}")
    config = _parse_config(obj["config"])
    config_hash = _expect_hash(obj["config_hash"], "config_hash")
    if config_hash != config.config_hash():
        raise _fail("config_hash", "does not match the config")
    batch_id = _expect_str(obj["batch_id"], "batch_id")
    if not ID_PATTERN.fullmatch(batch_id):
        raise _fail("batch_id", f"must match {ID_PATTERN.pattern}")
    if obj["suite_id"] != batch_id:
        raise _fail("suite_id", f"is {obj['suite_id']!r}, expected the batch ID {batch_id!r}")
    raw = obj["scenarios"]
    if not isinstance(raw, list) or len(raw) != config.scenario_count:
        raise _fail("scenarios", f"expected a list of {config.scenario_count} entries (config.scenario_count)")
    entries = []
    for i, value in enumerate(raw):
        where = f"scenarios[{i}]"
        e = _expect_object(value, where, {"ordinal", "scenario_id", "content_hash", "proxies"})
        if e["ordinal"] != i or not _is_int(e["ordinal"]):
            raise _fail(f"{where}.ordinal", f"is {e['ordinal']!r}, expected {i}")
        if e["scenario_id"] != scenario_id_for(i):
            raise _fail(f"{where}.scenario_id", f"is {e['scenario_id']!r}, expected {scenario_id_for(i)!r}")
        entries.append(GeneratedEntry(i, scenario_id_for(i), _expect_hash(e["content_hash"], f"{where}.content_hash"), _parse_proxies(e["proxies"], f"{where}.proxies")))
    return GenerationManifest(
        generation_manifest_version=GENERATION_MANIFEST_VERSION,
        generator_version=GENERATOR_VERSION,
        batch_id=batch_id,
        config=config,
        config_hash=config_hash,
        suite_id=batch_id,
        suite_version=GENERATED_SUITE_VERSION,
        suite_hash=_expect_hash(obj["suite_hash"], "suite_hash"),
        schema_version=SCHEMA_VERSION,
        entries=tuple(entries),
    )


# --------------------------------------------------------------------------- generate, freeze, load


@dataclass(frozen=True)
class GeneratedSuite:
    """An in-memory generated suite, not yet frozen. It does not hold the generator seed."""

    generation: GenerationManifest
    suite_manifest: SuiteManifest
    scenarios: tuple[Scenario, ...]


@dataclass(frozen=True)
class FrozenGeneratedSuite:
    """A verified, frozen generated suite: the loaded :class:`Suite` and its freeze record."""

    root: Path
    suite: Suite
    generation: GenerationManifest


def generate_suite(config: GeneratorConfig, generator_seed: int, *, batch_id: str) -> GeneratedSuite:
    """Generate ``config.scenario_count`` scenarios in ordinal order from the private seed.

    The result depends only on ``config``, ``generator_seed``, ``batch_id`` and
    :data:`GENERATOR_VERSION`. Raises :class:`GenerationError` if a scenario
    cannot meet every constraint within ``config.max_attempts``.
    """
    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be a GeneratorConfig")
    seed = validate_generator_seed(generator_seed)
    batch = _check_batch_id(batch_id, seed)
    config_hash = config.config_hash()
    scenarios: list[Scenario] = []
    entries: list[GeneratedEntry] = []
    suite_entries: list[SuiteEntry] = []
    taken: set[str] = set()
    for ordinal in range(config.scenario_count):
        scenario, proxies = _generate_one(config, config_hash, seed, ordinal, taken)
        taken.add(scenario.content_hash())
        scenarios.append(scenario)
        entries.append(GeneratedEntry(ordinal, scenario.scenario_id, scenario.content_hash(), proxies))
        primary, families = describe_families(scenario, proxies)
        suite_entries.append(SuiteEntry.describe(scenario, families=families, primary=primary, purpose=GENERATED_PURPOSE))
    manifest = SuiteManifest(
        suite_id=batch,
        suite_version=GENERATED_SUITE_VERSION,
        visibility=Visibility.GENERATED,
        seed_role=SeedRole.DEVELOPMENT,
        schema_version=SCHEMA_VERSION,
        entries=tuple(suite_entries),
    )
    # Round-trip through the strict suite parser, so a generated manifest can never be one load_suite rejects.
    manifest = SuiteManifest.from_dict(manifest.to_dict())
    generation = GenerationManifest(
        generation_manifest_version=GENERATION_MANIFEST_VERSION,
        generator_version=GENERATOR_VERSION,
        batch_id=batch,
        config=config,
        config_hash=config_hash,
        suite_id=batch,
        suite_version=GENERATED_SUITE_VERSION,
        suite_hash=manifest.suite_hash(),
        schema_version=SCHEMA_VERSION,
        entries=tuple(entries),
    )
    return GeneratedSuite(generation=generation, suite_manifest=manifest, scenarios=tuple(scenarios))


def scenario_file_bytes(scenario: Scenario) -> bytes:
    """The exact bytes of a generated scenario file (LF line endings on every platform)."""
    return (json.dumps(scenario.to_dict(), indent=2) + "\n").encode("ascii")


def _suite_files(generated: GeneratedSuite) -> dict[str, bytes]:
    files = {f"{SUITE_DIR_NAME}/{s.scenario_id}.json": scenario_file_bytes(s) for s in generated.scenarios}
    files[f"{SUITE_DIR_NAME}/{MANIFEST_NAME}"] = generated.suite_manifest.to_file_text().encode("ascii")
    files[GENERATION_MANIFEST_NAME] = generated.generation.to_file_text().encode("ascii")
    return files


def freeze_generated_suite(generated: GeneratedSuite, directory: str | Path) -> FrozenGeneratedSuite:
    """Write ``generated`` once into an empty or new ``directory`` and load it back verified.

    Layout: ``directory/suite/`` is a normal suite (``manifest.json`` plus one
    file per scenario) and ``directory/generation.json`` is the freeze record.
    An existing non-empty directory is never overwritten.
    """
    if not isinstance(generated, GeneratedSuite):
        raise TypeError("generated must be a GeneratedSuite")
    root = Path(directory)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise GenerationError(f"{root}: already exists and is not empty; a frozen suite is never overwritten")
    (root / SUITE_DIR_NAME).mkdir(parents=True)
    # The freeze record is written last, so an interrupted freeze cannot load.
    for name, data in sorted(_suite_files(generated).items(), key=lambda kv: kv[0] == GENERATION_MANIFEST_NAME):
        (root / name).write_bytes(data)
    return load_generated_suite(root)


def load_generated_suite(directory: str | Path) -> FrozenGeneratedSuite:
    """Load a frozen generated suite and verify the freeze record against every scenario.

    Raises :class:`~emberbench.errors.SuiteValidationError` if the suite
    directory is tampered with, and :class:`GenerationError` if the freeze
    record is malformed or does not match the suite.
    """
    root = Path(directory)
    path = root / GENERATION_MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise GenerationError(f"{path}: freeze record not found") from None
    generation = _parse_generation(_loads(text, str(path)))
    extra = sorted({p.name for p in root.iterdir()} - {GENERATION_MANIFEST_NAME, SUITE_DIR_NAME})
    if extra:
        raise GenerationError(f"{root}: unexpected files {extra}")
    suite = load_suite(root / SUITE_DIR_NAME)
    m = suite.manifest
    if m.suite_hash() != generation.suite_hash:
        raise GenerationError(f"{root}: suite hash {m.suite_hash()} does not match the frozen {generation.suite_hash}")
    if (m.suite_id, m.suite_version, m.visibility, m.seed_role) != (
        generation.suite_id,
        generation.suite_version,
        Visibility.GENERATED,
        SeedRole.DEVELOPMENT,
    ):
        raise GenerationError(f"{root}: suite header does not match the freeze record")
    if len(suite.scenarios) != len(generation.entries):
        raise GenerationError(f"{root}: suite has {len(suite.scenarios)} scenarios, freeze record has {len(generation.entries)}")
    config = generation.config
    for entry, scenario, suite_entry in zip(generation.entries, suite.scenarios, m.entries, strict=True):
        where = f"{root}: {entry.scenario_id}"
        if (scenario.scenario_id, scenario.content_hash()) != (entry.scenario_id, entry.content_hash):
            raise GenerationError(f"{where}: scenario does not match the freeze record")
        if (scenario.width, scenario.height, scenario.seed) != (config.width, config.height, 0):
            raise GenerationError(f"{where}: grid or seed does not match the generator contract")
        proxies = measure_proxies(scenario)
        if proxies != entry.proxies:
            raise GenerationError(f"{where}: pinned proxies do not match the scenario")
        violations = constraint_violations(proxies, config)
        if violations:
            raise GenerationError(f"{where}: breaks generation constraints {list(violations)}")
        primary, families = describe_families(scenario, proxies)
        if (suite_entry.primary_family, suite_entry.families) != (primary, families):
            raise GenerationError(f"{where}: family labels do not match the scenario")
    return FrozenGeneratedSuite(root=root, suite=suite, generation=generation)


def verify_regeneration(directory: str | Path, generator_seed: int) -> FrozenGeneratedSuite:
    """Evaluator audit: regenerate from the frozen config and ``generator_seed`` and compare every byte.

    Use it after the seed is revealed, or before evaluation to confirm that
    the frozen suite is exactly what the private seed produces.
    """
    frozen = load_generated_suite(directory)
    regenerated = generate_suite(frozen.generation.config, generator_seed, batch_id=frozen.generation.batch_id)
    for name, data in sorted(_suite_files(regenerated).items()):
        if (frozen.root / name).read_bytes() != data:
            raise GenerationError(f"{frozen.root}: {name} differs from the regenerated suite")
    return frozen
