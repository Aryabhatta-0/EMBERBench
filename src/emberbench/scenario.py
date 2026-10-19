"""Versioned scenario schema: parsing, validation and canonical serialisation.

A scenario is a JSON object. Every key is required unless noted otherwise,
and unknown keys are rejected. This prevents a typo from silently becoming
a default. See ``docs/scenario-schema.md`` for the full field reference.
"""

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from .canonical import canonical_json, sha256_hex
from .errors import ScenarioValidationError
from .grid import Pos, in_bounds
from .rng import MAX_SEED

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

MAX_GRID_SIDE = 1024
MAX_STEPS_LIMIT = 100_000
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_E = TypeVar("_E", bound=StrEnum)


class FireModelName(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class ConflictResolution(StrEnum):
    # V1 supports one rule. It is still an explicit config value, so that
    # every scenario states the rule it runs under.
    LOWEST_ID_WINS = "lowest_id_wins"


class EnterFirePolicy(StrEnum):
    BLOCKED = "blocked"
    CASUALTY = "casualty"


@dataclass(frozen=True)
class FireConfig:
    model: FireModelName = FireModelName.DETERMINISTIC
    spread_interval: int = 1
    exits_flammable: bool = True
    spread_probability: float | None = None  # required iff model is stochastic

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model.value,
            "spread_interval": self.spread_interval,
            "exits_flammable": self.exits_flammable,
        }
        if self.spread_probability is not None:
            d["spread_probability"] = self.spread_probability
        return d


@dataclass(frozen=True)
class MovementConfig:
    conflict_resolution: ConflictResolution = ConflictResolution.LOWEST_ID_WINS
    enter_fire: EnterFirePolicy = EnterFirePolicy.CASUALTY

    def to_dict(self) -> dict[str, Any]:
        return {"conflict_resolution": self.conflict_resolution.value, "enter_fire": self.enter_fire.value}


@dataclass(frozen=True)
class TerminationConfig:
    max_steps: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps}


@dataclass(frozen=True)
class ExitSpec:
    id: str
    position: Pos


@dataclass(frozen=True)
class PersonSpec:
    id: str
    position: Pos


@dataclass(frozen=True)
class Scenario:
    """A validated, immutable scenario.

    Collections are stored in canonical order: obstacles and ignitions are
    sorted by position, and exits and people are sorted by ID. Two scenarios
    that differ only in input ordering therefore compare equal and hash
    identically.
    """

    scenario_id: str
    width: int
    height: int
    obstacles: tuple[Pos, ...]
    exits: tuple[ExitSpec, ...]
    people: tuple[PersonSpec, ...]
    ignitions: tuple[Pos, ...]
    seed: int
    fire: FireConfig = field(default_factory=FireConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    description: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacles", tuple(sorted(Pos(*p) for p in self.obstacles)))
        object.__setattr__(self, "ignitions", tuple(sorted(Pos(*p) for p in self.ignitions)))
        object.__setattr__(self, "exits", tuple(sorted(self.exits, key=lambda e: e.id)))
        object.__setattr__(self, "people", tuple(sorted(self.people, key=lambda p: p.id)))
        _validate(self)

    @property
    def walls(self) -> frozenset[Pos]:
        return frozenset(self.obstacles)

    @property
    def exit_positions(self) -> dict[Pos, str]:
        return {e.position: e.id for e in self.exits}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "grid": {"width": self.width, "height": self.height},
            "obstacles": [p.to_list() for p in self.obstacles],
            "exits": [{"id": e.id, "position": e.position.to_list()} for e in self.exits],
            "people": [{"id": p.id, "position": p.position.to_list()} for p in self.people],
            "ignitions": [p.to_list() for p in self.ignitions],
            "fire": self.fire.to_dict(),
            "movement": self.movement.to_dict(),
            "termination": self.termination.to_dict(),
            "seed": self.seed,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON. It identifies the exact benchmark inputs."""
        return sha256_hex(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> "Scenario":
        return _parse(data)

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ScenarioValidationError(f"scenario is not valid JSON: {exc}") from exc
        return _parse(data)


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.from_json(Path(path).read_text(encoding="utf-8"))


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scenario.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- parsing


def _reject_constant(name: str) -> Any:
    raise ScenarioValidationError(f"non-finite number {name!r} is not allowed")


def _fail(where: str, message: str) -> ScenarioValidationError:
    return ScenarioValidationError(f"{where}: {message}")


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


def _expect_int(value: Any, where: str, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(where, f"expected an integer, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise _fail(where, f"must be in [{lo}, {hi}], got {value}")
    return value


def _expect_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(where, f"expected a boolean, got {type(value).__name__}")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise _fail(where, f"expected a string, got {type(value).__name__}")
    return value


def _expect_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(where, f"expected a list, got {type(value).__name__}")
    return value


def _expect_enum(enum_cls: type[_E], value: Any, where: str) -> _E:
    text = _expect_str(value, where)
    try:
        return enum_cls(text)
    except ValueError:
        allowed = [m.value for m in enum_cls]
        raise _fail(where, f"must be one of {allowed}, got {text!r}") from None


def _expect_pos(value: Any, where: str) -> Pos:
    items = _expect_list(value, where)
    if len(items) != 2:
        raise _fail(where, f"position must be [row, col], got {value!r}")
    row = _expect_int(items[0], f"{where}[0]", -(2**31), 2**31)
    col = _expect_int(items[1], f"{where}[1]", -(2**31), 2**31)
    return Pos(row, col)


def _expect_id(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not ID_PATTERN.fullmatch(text):
        raise _fail(where, f"id must match {ID_PATTERN.pattern}, got {text!r}")
    return text


def _parse_fire(value: Any) -> FireConfig:
    obj = _expect_object(value, "fire", {"model", "spread_interval", "exits_flammable"}, frozenset({"spread_probability"}))
    model = _expect_enum(FireModelName, obj["model"], "fire.model")
    probability: float | None = None
    if "spread_probability" in obj:
        raw = obj["spread_probability"]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise _fail("fire.spread_probability", f"expected a number, got {type(raw).__name__}")
        probability = float(raw)
    return FireConfig(
        model=model,
        spread_interval=_expect_int(obj["spread_interval"], "fire.spread_interval", 1, MAX_STEPS_LIMIT),
        exits_flammable=_expect_bool(obj["exits_flammable"], "fire.exits_flammable"),
        spread_probability=probability,
    )


def _parse_movement(value: Any) -> MovementConfig:
    obj = _expect_object(value, "movement", {"conflict_resolution", "enter_fire"})
    return MovementConfig(
        conflict_resolution=_expect_enum(ConflictResolution, obj["conflict_resolution"], "movement.conflict_resolution"),
        enter_fire=_expect_enum(EnterFirePolicy, obj["enter_fire"], "movement.enter_fire"),
    )


def _parse_termination(value: Any) -> TerminationConfig:
    obj = _expect_object(value, "termination", {"max_steps"})
    return TerminationConfig(max_steps=_expect_int(obj["max_steps"], "termination.max_steps", 1, MAX_STEPS_LIMIT))


_TOP_LEVEL_KEYS = {
    "schema_version",
    "scenario_id",
    "grid",
    "obstacles",
    "exits",
    "people",
    "ignitions",
    "fire",
    "movement",
    "termination",
    "seed",
}


def _parse(data: Any) -> Scenario:
    obj = _expect_object(data, "scenario", _TOP_LEVEL_KEYS, frozenset({"description"}))
    version = _expect_str(obj["schema_version"], "schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise _fail("schema_version", f"unsupported version {version!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}")
    grid = _expect_object(obj["grid"], "grid", {"width", "height"})

    exits = []
    for i, raw in enumerate(_expect_list(obj["exits"], "exits")):
        e = _expect_object(raw, f"exits[{i}]", {"id", "position"})
        exits.append(ExitSpec(_expect_id(e["id"], f"exits[{i}].id"), _expect_pos(e["position"], f"exits[{i}].position")))

    people = []
    for i, raw in enumerate(_expect_list(obj["people"], "people")):
        p = _expect_object(raw, f"people[{i}]", {"id", "position"})
        people.append(PersonSpec(_expect_id(p["id"], f"people[{i}].id"), _expect_pos(p["position"], f"people[{i}].position")))

    return Scenario(
        schema_version=version,
        scenario_id=_expect_id(obj["scenario_id"], "scenario_id"),
        description=_expect_str(obj.get("description", ""), "description"),
        width=_expect_int(grid["width"], "grid.width", 1, MAX_GRID_SIDE),
        height=_expect_int(grid["height"], "grid.height", 1, MAX_GRID_SIDE),
        obstacles=tuple(_expect_pos(p, f"obstacles[{i}]") for i, p in enumerate(_expect_list(obj["obstacles"], "obstacles"))),
        exits=tuple(exits),
        people=tuple(people),
        ignitions=tuple(_expect_pos(p, f"ignitions[{i}]") for i, p in enumerate(_expect_list(obj["ignitions"], "ignitions"))),
        fire=_parse_fire(obj["fire"]),
        movement=_parse_movement(obj["movement"]),
        termination=_parse_termination(obj["termination"]),
        seed=_expect_int(obj["seed"], "seed", 0, MAX_SEED),
    )


# --------------------------------------------------------------------------- invariants


def _check_unique_positions(positions: list[Pos], where: str) -> None:
    seen: set[Pos] = set()
    for p in positions:
        if p in seen:
            raise _fail(where, f"duplicate position {p.to_list()}")
        seen.add(p)


def _validate(s: Scenario) -> None:
    """Semantic checks that also run for scenarios built directly in Python."""
    if s.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise _fail("schema_version", f"unsupported version {s.schema_version!r}")
    if not ID_PATTERN.fullmatch(s.scenario_id):
        raise _fail("scenario_id", f"id must match {ID_PATTERN.pattern}, got {s.scenario_id!r}")
    for name, value in (("grid.width", s.width), ("grid.height", s.height)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_GRID_SIDE:
            raise _fail(name, f"must be an integer in [1, {MAX_GRID_SIDE}], got {value!r}")
    if not isinstance(s.seed, int) or isinstance(s.seed, bool) or not 0 <= s.seed <= MAX_SEED:
        raise _fail("seed", f"must be an integer in [0, 2**64 - 1], got {s.seed!r}")

    _validate_config(s)

    def check_bounds(p: Pos, where: str) -> None:
        if not in_bounds(p, s.width, s.height):
            raise _fail(where, f"position {p.to_list()} is outside the {s.width}x{s.height} grid")

    for p in s.obstacles:
        check_bounds(p, "obstacles")
    _check_unique_positions(list(s.obstacles), "obstacles")
    walls = frozenset(s.obstacles)

    if not s.exits:
        raise _fail("exits", "a scenario needs at least one exit")
    if not s.people:
        raise _fail("people", "a scenario needs at least one person")

    for kind, specs in (("exits", s.exits), ("people", s.people)):
        ids = [x.id for x in specs]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise _fail(kind, f"duplicate ids {dupes}")
        for x in specs:
            if not ID_PATTERN.fullmatch(x.id):
                raise _fail(kind, f"id must match {ID_PATTERN.pattern}, got {x.id!r}")
            check_bounds(x.position, f"{kind}[{x.id}]")
            if x.position in walls:
                raise _fail(f"{kind}[{x.id}]", f"position {x.position.to_list()} is an obstacle")
        _check_unique_positions([x.position for x in specs], kind)

    exit_cells = frozenset(e.position for e in s.exits)
    ignitions = frozenset(s.ignitions)
    for person in s.people:
        if person.position in exit_cells:
            raise _fail(f"people[{person.id}]", "a person cannot start on an exit")
        if person.position in ignitions:
            raise _fail(f"people[{person.id}]", "a person cannot start on an ignition point")

    _check_unique_positions(list(s.ignitions), "ignitions")
    for p in s.ignitions:
        check_bounds(p, "ignitions")
        if p in walls:
            raise _fail("ignitions", f"position {p.to_list()} is an obstacle; walls cannot burn")
        if p in exit_cells and not s.fire.exits_flammable:
            raise _fail("ignitions", f"position {p.to_list()} is an exit but fire.exits_flammable is false")


def _validate_config(s: Scenario) -> None:
    fire = s.fire
    if not isinstance(fire.model, FireModelName):
        raise _fail("fire.model", f"invalid model {fire.model!r}")
    if not isinstance(fire.spread_interval, int) or isinstance(fire.spread_interval, bool) or fire.spread_interval < 1:
        raise _fail("fire.spread_interval", f"must be an integer >= 1, got {fire.spread_interval!r}")
    if not isinstance(fire.exits_flammable, bool):
        raise _fail("fire.exits_flammable", "must be a boolean")
    if fire.model is FireModelName.STOCHASTIC:
        p = fire.spread_probability
        if p is None:
            raise _fail("fire.spread_probability", "required when fire.model is 'stochastic'")
        if not isinstance(p, float) or not 0.0 <= p <= 1.0:
            raise _fail("fire.spread_probability", f"must be a float in [0, 1], got {p!r}")
    elif fire.spread_probability is not None:
        raise _fail("fire.spread_probability", "only allowed when fire.model is 'stochastic'")
    if not isinstance(s.movement.conflict_resolution, ConflictResolution):
        raise _fail("movement.conflict_resolution", f"invalid value {s.movement.conflict_resolution!r}")
    if not isinstance(s.movement.enter_fire, EnterFirePolicy):
        raise _fail("movement.enter_fire", f"invalid value {s.movement.enter_fire!r}")
    m = s.termination.max_steps
    if not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= MAX_STEPS_LIMIT:
        raise _fail("termination.max_steps", f"must be an integer in [1, {MAX_STEPS_LIMIT}], got {m!r}")
