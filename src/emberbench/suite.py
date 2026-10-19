"""Scenario suite manifests: strict parsing and verification against the scenario files.

A suite is a directory with a ``manifest.json`` and one scenario file per
entry. The manifest pins, for every scenario, its content hash and the
metadata derived from it (population, exit count, ignition count, fire
model). ``load_suite`` re-derives all of them from the files and rejects any
mismatch, so a stale or edited manifest cannot load.

This module only reads a suite. It holds no scenario data, no seed schedule
and no hidden evaluation data. See ``docs/scenario-suite.md``.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from .canonical import sha256_hex
from .errors import ScenarioValidationError, SuiteValidationError
from .scenario import ID_PATTERN, SUPPORTED_SCHEMA_VERSIONS, FireModelName, Scenario, load_scenario

MANIFEST_NAME = "manifest.json"
SUITE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_E = TypeVar("_E", bound=StrEnum)


class ScenarioFamily(StrEnum):
    """The diagnostic families that the canonical V1 suite must cover."""

    SINGLE_EXIT_LOW_DENSITY = "single_exit_low_density"
    SINGLE_EXIT_HIGH_DENSITY = "single_exit_high_density"
    MULTIPLE_EXITS_UNEQUAL_DISTANCE = "multiple_exits_unequal_distance"
    NARROW_CORRIDOR_BOTTLENECK = "narrow_corridor_bottleneck"
    CENTRAL_IGNITION = "central_ignition"
    THREATENED_NEAREST_EXIT = "threatened_nearest_exit"
    MULTIPLE_IGNITIONS = "multiple_ignitions"
    MAZE_OBSTACLE_ROUTING = "maze_obstacle_routing"
    DECEPTIVE_NEAREST_EXIT = "deceptive_nearest_exit"
    CROWD_100 = "crowd_100"
    SPLIT_FLOW_ACROSS_EXITS = "split_flow_across_exits"
    ROUTE_INVALIDATION_REPLANNING = "route_invalidation_replanning"


class Visibility(StrEnum):
    # V1 has only the public suite. Hidden evaluation sets are never stored
    # in this repository or package.
    PUBLIC = "public"


class SeedRole(StrEnum):
    # The ``seed`` in each scenario file is a reproducible development
    # default. Scored seed schedules belong to the evaluator.
    DEVELOPMENT = "development"


@dataclass(frozen=True)
class SuiteEntry:
    scenario_id: str
    path: str
    schema_version: str
    content_hash: str
    primary_family: ScenarioFamily
    families: tuple[ScenarioFamily, ...]
    population: int
    exit_count: int
    ignition_count: int
    fire_model: FireModelName
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "path": self.path,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "primary_family": self.primary_family.value,
            "families": [f.value for f in self.families],
            "population": self.population,
            "exit_count": self.exit_count,
            "ignition_count": self.ignition_count,
            "fire_model": self.fire_model.value,
            "purpose": self.purpose,
        }

    @classmethod
    def describe(cls, scenario: Scenario, *, families: tuple[ScenarioFamily, ...], primary: ScenarioFamily, purpose: str) -> "SuiteEntry":
        """Build the entry for ``scenario``. Every derived field comes from the scenario itself."""
        return cls(
            scenario_id=scenario.scenario_id,
            path=f"{scenario.scenario_id}.json",
            schema_version=scenario.schema_version,
            content_hash=scenario.content_hash(),
            primary_family=primary,
            families=tuple(sorted(families)),
            population=len(scenario.people),
            exit_count=len(scenario.exits),
            ignition_count=len(scenario.ignitions),
            fire_model=scenario.fire.model,
            purpose=purpose,
        )


@dataclass(frozen=True)
class SuiteManifest:
    suite_id: str
    suite_version: str
    visibility: Visibility
    seed_role: SeedRole
    schema_version: str
    entries: tuple[SuiteEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "visibility": self.visibility.value,
            "seed_role": self.seed_role.value,
            "schema_version": self.schema_version,
            "scenarios": [e.to_dict() for e in self.entries],
        }

    def to_file_text(self) -> str:
        """The exact text of a canonical ``manifest.json``."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def suite_hash(self) -> str:
        """SHA-256 of the canonical manifest. It pins every scenario hash and all metadata."""
        return sha256_hex(self.to_dict())

    @property
    def families(self) -> frozenset[ScenarioFamily]:
        return frozenset(f for e in self.entries for f in e.families)

    @classmethod
    def from_dict(cls, data: Any) -> "SuiteManifest":
        return _parse_manifest(data)


@dataclass(frozen=True)
class Suite:
    """A manifest together with its verified scenarios, in manifest order."""

    root: Path
    manifest: SuiteManifest
    scenarios: tuple[Scenario, ...]

    def get(self, scenario_id: str) -> Scenario:
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        raise KeyError(scenario_id)

    def entry(self, scenario_id: str) -> SuiteEntry:
        for e in self.manifest.entries:
            if e.scenario_id == scenario_id:
                return e
        raise KeyError(scenario_id)


def load_suite(directory: str | Path) -> Suite:
    """Load ``directory/manifest.json`` and verify every scenario it lists.

    Raises ``SuiteValidationError`` if the manifest is malformed, if a file is
    missing or extra, or if any pinned hash or derived field does not match.
    """
    root = Path(directory)
    manifest_path = root / MANIFEST_NAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SuiteValidationError(f"{manifest_path}: manifest not found") from None
    except json.JSONDecodeError as exc:
        raise SuiteValidationError(f"{manifest_path}: not valid JSON: {exc}") from exc
    manifest = _parse_manifest(data)

    listed = {e.path for e in manifest.entries} | {MANIFEST_NAME}
    present = {p.name for p in root.iterdir()}
    extra = sorted(present - listed)
    if extra:
        raise SuiteValidationError(f"{root}: files not listed in the manifest: {extra}")

    scenarios = []
    for e in manifest.entries:
        path = root / e.path
        if not path.is_file():
            raise SuiteValidationError(f"{e.scenario_id}: file {e.path} does not exist")
        try:
            scenario = load_scenario(path)
        except ScenarioValidationError as exc:
            raise SuiteValidationError(f"{e.scenario_id}: invalid scenario: {exc}") from exc
        _verify_entry(e, scenario)
        scenarios.append(scenario)
    return Suite(root=root, manifest=manifest, scenarios=tuple(scenarios))


def _verify_entry(e: SuiteEntry, s: Scenario) -> None:
    actual = {
        "scenario_id": s.scenario_id,
        "schema_version": s.schema_version,
        "content_hash": s.content_hash(),
        "population": len(s.people),
        "exit_count": len(s.exits),
        "ignition_count": len(s.ignitions),
        "fire_model": s.fire.model.value,
    }
    pinned = e.to_dict()
    for key, value in actual.items():
        if pinned[key] != value:
            raise SuiteValidationError(f"{e.scenario_id}: manifest {key} is {pinned[key]!r}, scenario file gives {value!r}")


# --------------------------------------------------------------------------- parsing


def _fail(where: str, message: str) -> SuiteValidationError:
    return SuiteValidationError(f"{where}: {message}")


def _expect_object(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    missing = sorted(keys - value.keys())
    if missing:
        raise _fail(where, f"missing required keys {missing}")
    unknown = sorted(value.keys() - keys)
    if unknown:
        raise _fail(where, f"unknown keys {unknown}")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "expected a non-empty string")
    return value


def _expect_count(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _fail(where, f"expected a non-negative integer, got {value!r}")
    return value


def _expect_member(enum_cls: type[_E], value: Any, where: str) -> _E:
    text = _expect_str(value, where)
    try:
        return enum_cls(text)
    except ValueError:
        raise _fail(where, f"must be one of {[m.value for m in enum_cls]}, got {text!r}") from None


def _expect_schema_version(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if text not in SUPPORTED_SCHEMA_VERSIONS:
        raise _fail(where, f"unsupported schema version {text!r}")
    return text


_ENTRY_KEYS = {
    "scenario_id",
    "path",
    "schema_version",
    "content_hash",
    "primary_family",
    "families",
    "population",
    "exit_count",
    "ignition_count",
    "fire_model",
    "purpose",
}


def _parse_entry(value: Any, where: str) -> SuiteEntry:
    obj = _expect_object(value, where, _ENTRY_KEYS)
    scenario_id = _expect_str(obj["scenario_id"], f"{where}.scenario_id")
    if not ID_PATTERN.fullmatch(scenario_id):
        raise _fail(f"{where}.scenario_id", f"must match {ID_PATTERN.pattern}, got {scenario_id!r}")
    path = _expect_str(obj["path"], f"{where}.path")
    # A plain file name inside the suite directory: no subdirectories, no
    # "..", no absolute paths. This keeps every suite self-contained.
    if path != f"{scenario_id}.json":
        raise _fail(f"{where}.path", f"must be {scenario_id + '.json'!r}, got {path!r}")
    content_hash = _expect_str(obj["content_hash"], f"{where}.content_hash")
    if not HASH_PATTERN.fullmatch(content_hash):
        raise _fail(f"{where}.content_hash", "must be 64 lowercase hex characters")
    raw_families = obj["families"]
    if not isinstance(raw_families, list) or not raw_families:
        raise _fail(f"{where}.families", "expected a non-empty list")
    families = tuple(_expect_member(ScenarioFamily, f, f"{where}.families") for f in raw_families)
    if list(families) != sorted(set(families)):
        raise _fail(f"{where}.families", "must be sorted and unique")
    primary = _expect_member(ScenarioFamily, obj["primary_family"], f"{where}.primary_family")
    if primary not in families:
        raise _fail(f"{where}.primary_family", f"{primary.value!r} is not in families")
    return SuiteEntry(
        scenario_id=scenario_id,
        path=path,
        schema_version=_expect_schema_version(obj["schema_version"], f"{where}.schema_version"),
        content_hash=content_hash,
        primary_family=primary,
        families=families,
        population=_expect_count(obj["population"], f"{where}.population"),
        exit_count=_expect_count(obj["exit_count"], f"{where}.exit_count"),
        ignition_count=_expect_count(obj["ignition_count"], f"{where}.ignition_count"),
        fire_model=_expect_member(FireModelName, obj["fire_model"], f"{where}.fire_model"),
        purpose=_expect_str(obj["purpose"], f"{where}.purpose"),
    )


def _parse_manifest(data: Any) -> SuiteManifest:
    obj = _expect_object(data, "manifest", {"suite_id", "suite_version", "visibility", "seed_role", "schema_version", "scenarios"})
    suite_id = _expect_str(obj["suite_id"], "suite_id")
    if not ID_PATTERN.fullmatch(suite_id):
        raise _fail("suite_id", f"must match {ID_PATTERN.pattern}, got {suite_id!r}")
    version = _expect_str(obj["suite_version"], "suite_version")
    if not SUITE_VERSION_PATTERN.fullmatch(version):
        raise _fail("suite_version", f"must be MAJOR.MINOR.PATCH, got {version!r}")
    schema_version = _expect_schema_version(obj["schema_version"], "schema_version")
    raw = obj["scenarios"]
    if not isinstance(raw, list) or not raw:
        raise _fail("scenarios", "expected a non-empty list")
    entries = tuple(_parse_entry(e, f"scenarios[{i}]") for i, e in enumerate(raw))
    ids = [e.scenario_id for e in entries]
    if len(set(ids)) != len(ids):
        raise _fail("scenarios", f"duplicate scenario ids {sorted({i for i in ids if ids.count(i) > 1})}")
    if ids != sorted(ids):
        raise _fail("scenarios", "entries must be sorted by scenario_id")
    hashes = [e.content_hash for e in entries]
    if len(set(hashes)) != len(hashes):
        raise _fail("scenarios", "two entries have the same content hash")
    for e in entries:
        if e.schema_version != schema_version:
            raise _fail(e.scenario_id, f"schema_version {e.schema_version!r} differs from the suite's {schema_version!r}")
    return SuiteManifest(
        suite_id=suite_id,
        suite_version=version,
        visibility=_expect_member(Visibility, obj["visibility"], "visibility"),
        seed_role=_expect_member(SeedRole, obj["seed_role"], "seed_role"),
        schema_version=schema_version,
        entries=entries,
    )
