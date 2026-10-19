"""Procedural unseen scenario generation: determinism, validity, constraints, freezing and leakage."""

import copy
import json
import os
import random
import re
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import greedy_exit_policy
from emberbench import (
    Action,
    Observation,
    Pos,
    Scenario,
    build_evaluation_manifest,
    load_scenario,
    load_suite,
    pair_results,
    run_episode,
    run_evaluation,
)
from emberbench.canonical import canonical_json
from emberbench.errors import GenerationError, SuiteValidationError
from emberbench.generator import (
    GENERATION_MANIFEST_NAME,
    GENERATOR_VERSION,
    MAX_GENERATOR_SEED,
    SUITE_DIR_NAME,
    FrozenGeneratedSuite,
    GeneratedSuite,
    GeneratorConfig,
    GeneratorRng,
    constraint_violations,
    derive_generation_stream,
    freeze_generated_suite,
    generate_suite,
    load_generated_suite,
    measure_proxies,
    scenario_file_bytes,
    verify_regeneration,
)
from emberbench.rng import derive_stream_seed
from emberbench.scenario import ExitSpec, FireModelName, PersonSpec
from emberbench.simulator import Policy
from emberbench.suite import MANIFEST_NAME, Visibility
from test_suite import PINNED_HASHES, SUITE_DIR, SUITE_HASH

# Test-only seeds. Real scored generator seeds are evaluator-private and never in the repository.
SEED = 0x5EED_1234_ABCD_9876_0FED_CBA9
OTHER_SEED = SEED + 1
BATCH = "test-batch-a"

# Golden hashes for BASE_CONFIG + SEED + BATCH. They must be equal on every platform and Python version.
GOLDEN_CONFIG_HASH = "a464c56981fbfcfe976ec4114a9194ce408aebf53348e135d3515a3902f12704"
GOLDEN_GENERATION_HASH = "f9fac2b98f45fb36186504cff4826e26eda2a9780470727507f5ca69cfe9271d"
GOLDEN_SUITE_HASH = "78319258f35ba44a6139a370233350ca25830cacad2181e16c625b9ef20ae1dd"

BASE_CONFIG: dict[str, Any] = {
    "config_version": "1.0",
    "scenario_count": 4,
    "grid": {"width": 12, "height": 9},
    "population": {"min": 3, "max": 8},
    "exit_count": {"min": 1, "max": 3},
    "ignition_count": {"min": 1, "max": 2},
    "layout": {
        "partition_count": {"min": 0, "max": 2},
        "doors_per_partition": {"min": 1, "max": 2},
        "door_width": {"min": 1, "max": 2},
        "scatter_density": {"min": 0.0, "max": 0.1},
    },
    "fire": {
        "model": "stochastic",
        "spread_interval": {"min": 1, "max": 3},
        "exits_flammable": True,
        "spread_probability_choices": [0.25, 0.5, 0.75],
    },
    "movement": {"enter_fire": "casualty"},
    "max_steps": 60,
    "max_attempts": 200,
    "constraints": {
        "route_length": {"min": 1, "max": 60},
        "obstacle_density": {"min": 0.0, "max": 0.5},
        "crowd_density": {"min": 0.0, "max": 0.5},
        "exit_separation": {"min": 4, "max": 21},
        "passage_width": {"min": 1, "max": 12},
        "route_redundancy": {"min": 1, "max": 4},
        "ignition_person_distance": {"min": 2, "max": 108},
        "ignition_exit_distance": {"min": 3, "max": 108},
    },
}


def config_with(**changes: Any) -> GeneratorConfig:
    """BASE_CONFIG with dotted-path overrides, for example ``config_with(**{"grid.width": 14})``."""
    data = copy.deepcopy(BASE_CONFIG)
    for path, value in changes.items():
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        if value is _DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return GeneratorConfig.from_dict(data)


_DELETE = object()


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return GeneratorConfig.from_dict(BASE_CONFIG)


@pytest.fixture(scope="module")
def generated(config: GeneratorConfig) -> GeneratedSuite:
    return generate_suite(config, SEED, batch_id=BATCH)


@pytest.fixture
def frozen(generated: GeneratedSuite, tmp_path: Path) -> FrozenGeneratedSuite:
    return freeze_generated_suite(generated, tmp_path / "frozen")


def all_files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- reproducibility


def test_golden_hashes_are_platform_independent(config: GeneratorConfig, generated: GeneratedSuite) -> None:
    assert config.config_hash() == GOLDEN_CONFIG_HASH
    assert generated.generation.generation_hash() == GOLDEN_GENERATION_HASH
    assert generated.suite_manifest.suite_hash() == GOLDEN_SUITE_HASH


def test_regeneration_is_byte_identical(config: GeneratorConfig, generated: GeneratedSuite, tmp_path: Path) -> None:
    again = generate_suite(GeneratorConfig.from_dict(copy.deepcopy(BASE_CONFIG)), SEED, batch_id=BATCH)
    assert again == generated
    assert [s.scenario_id for s in again.scenarios] == [s.scenario_id for s in generated.scenarios]
    assert again.generation.generation_hash() == generated.generation.generation_hash()
    a = freeze_generated_suite(generated, tmp_path / "a")
    b = freeze_generated_suite(again, tmp_path / "b")
    assert all_files(a.root) == all_files(b.root)
    assert all_files(a.root)[f"{SUITE_DIR_NAME}/gen-0000.json"] == scenario_file_bytes(generated.scenarios[0])
    assert b"\r" not in b"".join(all_files(a.root).values())


_SUBPROCESS_SCRIPT = """
import json, sys
from emberbench.generator import GeneratorConfig, generate_suite
config = GeneratorConfig.from_json(sys.stdin.read())
g = generate_suite(config, int(sys.argv[1]), batch_id=sys.argv[2])
print(g.generation.generation_hash())
"""


@pytest.mark.parametrize("hash_seed", ["0", "4242"])
def test_regeneration_in_a_fresh_interpreter(generated: GeneratedSuite, hash_seed: str) -> None:
    # A different PYTHONHASHSEED changes set and dict-of-str ordering; the output must not move.
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(src)}
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(SEED), BATCH],
        input=canonical_json(BASE_CONFIG),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert out.stdout.strip() == generated.generation.generation_hash()


def test_different_seeds_create_variation(config: GeneratorConfig, generated: GeneratedSuite) -> None:
    other = generate_suite(config, OTHER_SEED, batch_id=BATCH)
    ours = {s.content_hash() for s in generated.scenarios}
    theirs = {s.content_hash() for s in other.scenarios}
    assert not ours & theirs
    assert other.suite_manifest.suite_hash() != generated.suite_manifest.suite_hash()
    assert other.generation.generation_hash() != generated.generation.generation_hash()
    # The public config is the same; only the private seed differs.
    assert other.generation.config_hash == generated.generation.config_hash
    # Scenarios within one suite also differ from each other.
    assert len(ours) == len(generated.scenarios)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 201},
        {"grid.width": 13},
        {"population.max": 9},
        {"fire.spread_probability_choices": [0.25, 0.5]},
        {"constraints.passage_width.max": 11},
    ],
)
def test_config_changes_change_output_and_hashes(generated: GeneratedSuite, changes: dict[str, Any]) -> None:
    changed = generate_suite(config_with(**changes), SEED, batch_id=BATCH)
    assert changed.generation.config_hash != generated.generation.config_hash
    assert changed.generation.generation_hash() != generated.generation.generation_hash()
    assert changed.suite_manifest.suite_hash() != generated.suite_manifest.suite_hash()
    assert {s.content_hash() for s in changed.scenarios}.isdisjoint(s.content_hash() for s in generated.scenarios)


def test_batch_id_changes_suite_hash_but_not_scenarios(config: GeneratorConfig, generated: GeneratedSuite) -> None:
    other = generate_suite(config, SEED, batch_id="test-batch-b")
    assert other.scenarios == generated.scenarios
    assert other.suite_manifest.suite_hash() != generated.suite_manifest.suite_hash()
    assert other.generation.generation_hash() != generated.generation.generation_hash()


def test_suite_order_is_ordinal_order(generated: GeneratedSuite) -> None:
    ids = [f"gen-{i:04d}" for i in range(4)]
    assert [s.scenario_id for s in generated.scenarios] == ids
    assert [e.scenario_id for e in generated.suite_manifest.entries] == ids
    assert [(e.ordinal, e.scenario_id) for e in generated.generation.entries] == list(enumerate(ids))
    assert [e.content_hash for e in generated.generation.entries] == [s.content_hash() for s in generated.scenarios]


def test_no_ambient_entropy(generated: GeneratedSuite, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError("the generator used ambient entropy")

    for module, name in [
        (random, "random"),
        (random, "randrange"),
        (random, "randint"),
        (random, "choice"),
        (random, "shuffle"),
        (random, "sample"),
        (random, "getrandbits"),
        (random, "seed"),
        (os, "urandom"),
        (os, "getrandom"),
        (secrets, "token_bytes"),
        (secrets, "randbits"),
        (secrets, "randbelow"),
        (uuid, "uuid1"),
        (uuid, "uuid4"),
        (time, "time"),
        (time, "time_ns"),
    ]:
        monkeypatch.setattr(module, name, forbidden, raising=False)
    again = generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH)
    assert again == generated


# --------------------------------------------------------------------------- private randomness


def test_generator_streams_are_separate_from_simulator_streams() -> None:
    config_hash = "0" * 64
    gen = {derive_generation_stream(s, config_hash, o, a) for s in (0, 1) for o in (0, 1) for a in (0, 1)}
    assert len(gen) == 8
    sim = {derive_stream_seed(s, label) for s in (0, 1) for label in ("fire", "generator", "fire/0")}
    assert gen.isdisjoint(sim)
    assert derive_generation_stream(MAX_GENERATOR_SEED, config_hash, 0, 0) != derive_generation_stream(0, config_hash, 0, 0)


@pytest.mark.parametrize("seed", [-1, MAX_GENERATOR_SEED + 1])
def test_generator_seed_range(config: GeneratorConfig, seed: int) -> None:
    with pytest.raises(ValueError):
        generate_suite(config, seed, batch_id=BATCH)


@pytest.mark.parametrize("seed", [True, 1.0, "1"])
def test_generator_seed_type(config: GeneratorConfig, seed: Any) -> None:
    with pytest.raises(TypeError):
        generate_suite(config, seed, batch_id=BATCH)


def test_generator_rng_is_uniform_and_deterministic() -> None:
    a, b = GeneratorRng(7), GeneratorRng(7)
    draws = [a.below(3) for _ in range(3000)]
    assert draws == [b.below(3) for _ in range(3000)]
    counts = [draws.count(k) for k in range(3)]
    assert set(draws) == {0, 1, 2} and min(counts) > 900
    assert sorted(GeneratorRng(1).shuffled(range(10))) == list(range(10))
    with pytest.raises(ValueError):
        a.below(0)


def test_batch_id_must_not_carry_the_seed(config: GeneratorConfig) -> None:
    with pytest.raises(GenerationError, match="must not contain the generator seed"):
        generate_suite(config, SEED, batch_id=f"batch-{SEED}")
    with pytest.raises(GenerationError, match="must not contain the generator seed"):
        generate_suite(config, SEED, batch_id=f"batch-{SEED:x}")
    with pytest.raises(GenerationError, match="batch_id"):
        generate_suite(config, SEED, batch_id="bad batch")


# --------------------------------------------------------------------------- validity


def test_generated_scenarios_are_strictly_valid(generated: GeneratedSuite, frozen: FrozenGeneratedSuite) -> None:
    for scenario in generated.scenarios:
        assert Scenario.from_json(scenario.to_json()) == scenario
        assert load_scenario(frozen.suite.root / f"{scenario.scenario_id}.json") == scenario
        walls = scenario.walls
        exits = scenario.exit_positions
        people = [p.position for p in scenario.people]
        assert scenario.exits and scenario.people and scenario.ignitions
        assert len(set(people)) == len(people) and len(exits) == len(scenario.exits)
        assert len(set(scenario.ignitions)) == len(scenario.ignitions)
        for p in people:
            assert p not in walls and p not in exits and p not in scenario.ignitions
        for p in scenario.ignitions:
            assert p not in walls and p not in exits
        for p in [*people, *exits, *scenario.ignitions, *walls]:
            assert 0 <= p.row < scenario.height and 0 <= p.col < scenario.width
        # Explicit V1 rules and canonical ordering.
        d = scenario.to_dict()
        assert d["movement"] == {"conflict_resolution": "lowest_id_wins", "enter_fire": "casualty"}
        assert set(d["fire"]) == {"model", "spread_interval", "exits_flammable", "spread_probability"}
        assert d["schema_version"] == "1.0" and d["seed"] == 0 and d["description"] == ""
        assert d["obstacles"] == sorted(d["obstacles"]) and d["ignitions"] == sorted(d["ignitions"])
        assert [e["id"] for e in d["exits"]] == sorted(e["id"] for e in d["exits"])
        assert [p["id"] for p in d["people"]] == sorted(p["id"] for p in d["people"])


def test_deterministic_fire_config() -> None:
    config = config_with(**{"fire.model": "deterministic", "fire.spread_probability_choices": _DELETE, "scenario_count": 2})
    for scenario in generate_suite(config, SEED, batch_id=BATCH).scenarios:
        assert scenario.fire.model is FireModelName.DETERMINISTIC
        assert scenario.fire.spread_probability is None
        assert "spread_probability" not in scenario.to_dict()["fire"]


def test_ids_are_neutral_and_non_spatial(generated: GeneratedSuite) -> None:
    row_major_people = 0
    for scenario in generated.scenarios:
        n = len(scenario.people)
        assert [p.id for p in scenario.people] == [f"p{i:02d}" for i in range(n)]
        assert [e.id for e in scenario.exits] == [f"x{i:02d}" for i in range(len(scenario.exits))]
        assert re.fullmatch(r"gen-\d{4}", scenario.scenario_id)
        positions = [p.position for p in scenario.people]
        row_major_people += positions == sorted(positions)
    # IDs follow a random permutation, so they do not follow the row-major position order.
    assert row_major_people < len(generated.scenarios)


def test_generated_scenarios_differ_from_the_public_suite(generated: GeneratedSuite) -> None:
    assert {s.content_hash() for s in generated.scenarios}.isdisjoint(PINNED_HASHES.values())


def test_public_suite_hashes_are_unchanged() -> None:
    public = load_suite(SUITE_DIR)
    assert public.manifest.suite_hash() == SUITE_HASH
    assert public.manifest.visibility is Visibility.PUBLIC
    assert {e.scenario_id: e.content_hash for e in public.manifest.entries} == PINNED_HASHES


# --------------------------------------------------------------------------- measured proxies and constraints


def _scenario(rows: list[str], *, people: list[Pos], exits: list[Pos], ignitions: list[Pos]) -> Scenario:
    walls = tuple(Pos(r, c) for r, line in enumerate(rows) for c, ch in enumerate(line) if ch == "#")
    return Scenario(
        scenario_id="proxy-test",
        width=len(rows[0]),
        height=len(rows),
        obstacles=walls,
        exits=tuple(ExitSpec(f"x{i}", p) for i, p in enumerate(exits)),
        people=tuple(PersonSpec(f"p{i}", p) for i, p in enumerate(people)),
        ignitions=tuple(ignitions),
        seed=0,
    )


def test_proxies_on_a_hand_built_door_map() -> None:
    rows = [
        "#######",
        "#..#..#",
        "#.....E",
        "#..#..#",
        "#######",
    ]
    rows[2] = rows[2][:6] + "."
    s = _scenario(rows, people=[Pos(2, 1)], exits=[Pos(2, 6)], ignitions=[Pos(3, 1)])
    p = measure_proxies(s)
    assert p.obstacle_density == 2 / 15
    assert p.crowd_density == 1 / 13
    assert p.exit_separation is None
    assert p.passage_width == 1  # the one-cell door at (2, 3)
    assert p.route_redundancy == 1
    assert p.ignition_person_distance == 1
    assert p.ignition_exit_distance == 6
    assert p.route_length == 5


def test_proxies_on_a_hand_built_open_map_with_two_exits() -> None:
    rows = [
        "#######",
        "#.....#",
        "#......",
        "#.....#",
        "#######",
    ]
    rows[0] = "###.###"
    s = _scenario(rows, people=[Pos(2, 1), Pos(3, 2)], exits=[Pos(2, 6), Pos(0, 3)], ignitions=[Pos(3, 5)])
    p = measure_proxies(s)
    assert p.obstacle_density == 0.0
    assert p.exit_separation == 5
    assert p.passage_width == 3
    assert p.route_redundancy == 2
    assert p.ignition_person_distance == 3
    assert p.ignition_exit_distance == 2
    assert p.route_length == 4


def test_unreachable_person_has_no_route() -> None:
    rows = [
        "#######",
        "#.#...#",
        "###...E",
        "#.....#",
        "#######",
    ]
    rows[2] = rows[2][:6] + "."
    s = _scenario(rows, people=[Pos(1, 1), Pos(3, 3)], exits=[Pos(2, 6)], ignitions=[Pos(3, 1)])
    p = measure_proxies(s)
    assert p.route_length is None and p.passage_width == 0 and p.route_redundancy == 0
    assert "route_exists" in constraint_violations(p, GeneratorConfig.from_dict(BASE_CONFIG))


def test_every_generated_scenario_meets_every_constraint(generated: GeneratedSuite, config: GeneratorConfig) -> None:
    k = config.constraints
    for scenario, entry in zip(generated.scenarios, generated.generation.entries, strict=True):
        p = measure_proxies(scenario)
        assert p == entry.proxies
        assert constraint_violations(p, config) == ()
        assert p.route_length is not None and p.route_length <= config.max_steps
        assert k.obstacle_density.contains(p.obstacle_density)
        assert p.exit_separation is None or p.exit_separation >= k.exit_separation.lo
        assert p.ignition_person_distance is not None and p.ignition_person_distance >= k.ignition_person_distance.lo
        assert config.population.contains(len(scenario.people))
        assert config.exit_count.contains(len(scenario.exits))
        assert config.ignition_count.contains(len(scenario.ignitions))
        assert (scenario.width, scenario.height, scenario.termination.max_steps) == (12, 9, 60)


@pytest.mark.parametrize(
    ("changes", "check"),
    [
        (
            {"layout.scatter_density": {"min": 0.15, "max": 0.3}, "constraints.obstacle_density": {"min": 0.15, "max": 0.3}},
            lambda p: 0.15 <= p.obstacle_density <= 0.3,
        ),
        ({"constraints.passage_width": {"min": 2, "max": 12}}, lambda p: p.passage_width >= 2),
        (
            {"exit_count": {"min": 2, "max": 3}, "constraints.route_redundancy": {"min": 2, "max": 4}},
            lambda p: p.route_redundancy >= 2,
        ),
        (
            {"exit_count": {"min": 2, "max": 2}, "constraints.exit_separation": {"min": 10, "max": 21}},
            lambda p: p.exit_separation is not None and p.exit_separation >= 10,
        ),
        ({"constraints.ignition_person_distance": {"min": 4, "max": 6}}, lambda p: 4 <= (p.ignition_person_distance or 0) <= 6),
        ({"constraints.ignition_exit_distance": {"min": 6, "max": 108}}, lambda p: (p.ignition_exit_distance or 0) >= 6),
        (
            {"population": {"min": 12, "max": 20}, "constraints.crowd_density": {"min": 0.2, "max": 0.5}},
            lambda p: 0.2 <= p.crowd_density <= 0.5,
        ),
        ({"constraints.route_length": {"min": 8, "max": 60}}, lambda p: (p.route_length or 0) >= 8),
    ],
)
def test_generation_constraints_are_enforced(changes: dict[str, Any], check: Callable[[Any], bool]) -> None:
    config = config_with(scenario_count=3, max_attempts=2000, **changes)
    for scenario in generate_suite(config, SEED, batch_id=BATCH).scenarios:
        assert check(measure_proxies(scenario))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"population": {"min": 5, "max": 4}}, "greater than max"),
        ({"population": {"min": 1, "max": 69}}, "exceeds the 70 interior cells"),
        ({"constraints.route_length": {"min": 61, "max": 61}}, "exceeds max_steps"),
        ({"constraints.route_redundancy": {"min": 4, "max": 4}}, "exceeds exit_count.max"),
        ({"constraints.route_redundancy": {"min": 1, "max": 5}}, "route_redundancy.max"),
        ({"layout.door_width": {"min": 4, "max": 4}, "layout.doors_per_partition": {"min": 2, "max": 2}}, "do not fit"),
        ({"layout.scatter_density": {"min": 0.0, "max": 0.0005}}, "multiple of 1/1000"),
        ({"layout.partition_count": {"min": 0, "max": 9}}, "partition_count.max"),
        ({"fire.model": "deterministic"}, "only allowed when fire.model is 'stochastic'"),
        ({"fire.spread_probability_choices": _DELETE}, "required when fire.model is 'stochastic'"),
        ({"fire.spread_probability_choices": [0.5, 0.25]}, "sorted and unique"),
        ({"fire.spread_probability_choices": [1.5]}, "in [0, 1]"),
        ({"constraints.exit_separation": {"min": 1, "max": 5}}, "exit_separation.min"),
        ({"constraints.obstacle_density": {"min": -0.1, "max": 0.5}}, "obstacle_density.min"),
        ({"grid.width": 4}, "grid.width"),
        ({"max_attempts": True}, "max_attempts"),
        ({"config_version": "2.0"}, "unsupported version"),
        ({"surprise": 1}, "unknown keys"),
        ({"max_steps": _DELETE}, "missing required keys"),
        ({"movement.enter_fire": "teleport"}, "movement.enter_fire"),
    ],
)
def test_invalid_or_impossible_configs_fail_explicitly(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(GenerationError, match=re.escape(message)):
        config_with(**changes)


def test_unsatisfiable_constraints_fail_closed() -> None:
    # Scatter can never reach this obstacle density. The generator must fail, not relax the bound.
    config = config_with(
        max_attempts=5,
        **{
            "layout.partition_count": {"min": 0, "max": 0},
            "layout.scatter_density": {"min": 0.0, "max": 0.0},
            "constraints.obstacle_density": {"min": 0.4, "max": 0.5},
        },
    )
    with pytest.raises(GenerationError, match="never weakened") as info:
        generate_suite(config, SEED, batch_id=BATCH)
    text = str(info.value)
    assert "gen-0000" in text and "'obstacle_density':" in text
    assert str(SEED) not in text and f"{SEED:x}" not in text


def test_constructive_placement_failure_is_reported() -> None:
    config = config_with(max_attempts=3, **{"constraints.ignition_exit_distance": {"min": 100, "max": 108}})
    with pytest.raises(GenerationError, match="ignition_placement"):
        generate_suite(config, SEED, batch_id=BATCH)


def test_config_round_trips_canonically(config: GeneratorConfig) -> None:
    assert GeneratorConfig.from_json(canonical_json(config.to_dict())) == config
    assert config.to_dict() == BASE_CONFIG
    with pytest.raises(GenerationError, match="not valid JSON"):
        GeneratorConfig.from_json("{")
    with pytest.raises(GenerationError, match="non-finite"):
        GeneratorConfig.from_json(canonical_json(BASE_CONFIG).replace('"max_steps":60', '"max_steps":NaN'))


# --------------------------------------------------------------------------- freezing


def test_frozen_suite_layout_and_contents(frozen: FrozenGeneratedSuite, generated: GeneratedSuite) -> None:
    files = all_files(frozen.root)
    assert sorted(files) == sorted(
        [GENERATION_MANIFEST_NAME, f"{SUITE_DIR_NAME}/{MANIFEST_NAME}", *(f"{SUITE_DIR_NAME}/gen-{i:04d}.json" for i in range(4))]
    )
    assert frozen.generation == generated.generation
    assert frozen.suite.manifest == generated.suite_manifest
    assert frozen.suite.scenarios == generated.scenarios
    assert frozen.suite.manifest.visibility is Visibility.GENERATED
    record = json.loads(files[GENERATION_MANIFEST_NAME])
    assert record["generator_version"] == GENERATOR_VERSION
    assert record["config_hash"] == frozen.generation.config.config_hash()
    assert record["suite_hash"] == frozen.suite.manifest.suite_hash()
    assert [e["ordinal"] for e in record["scenarios"]] == [0, 1, 2, 3]
    assert load_generated_suite(frozen.root) == frozen
    assert verify_regeneration(frozen.root, SEED) == frozen


def test_freeze_never_overwrites(frozen: FrozenGeneratedSuite, generated: GeneratedSuite) -> None:
    before = all_files(frozen.root)
    with pytest.raises(GenerationError, match="never overwritten"):
        freeze_generated_suite(generated, frozen.root)
    assert all_files(frozen.root) == before


def _rewrite_json(path: Path, edit: Callable[[Any], None]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _move_first_person(data: Any) -> None:
    taken = {tuple(p) for p in data["obstacles"]} | {tuple(e["position"]) for e in data["exits"]}
    taken |= {tuple(p["position"]) for p in data["people"]} | {tuple(p) for p in data["ignitions"]}
    for r in range(1, data["grid"]["height"] - 1):
        for c in range(1, data["grid"]["width"] - 1):
            if (r, c) not in taken:
                data["people"][0]["position"] = [r, c]
                return


def test_tampered_scenario_file_is_detected(frozen: FrozenGeneratedSuite) -> None:
    _rewrite_json(frozen.suite.root / "gen-0001.json", _move_first_person)
    with pytest.raises(SuiteValidationError, match="content_hash"):
        load_generated_suite(frozen.root)


def test_consistently_retampered_suite_is_detected(frozen: FrozenGeneratedSuite) -> None:
    # Edit a scenario and re-pin its hash in the suite manifest; the freeze record still catches it.
    path = frozen.suite.root / "gen-0001.json"
    _rewrite_json(path, _move_first_person)
    new_hash = load_scenario(path).content_hash()

    def repin(data: Any) -> None:
        data["scenarios"][1]["content_hash"] = new_hash

    _rewrite_json(frozen.suite.root / MANIFEST_NAME, repin)
    load_suite(frozen.suite.root)  # the suite on its own is consistent again
    with pytest.raises(GenerationError, match="suite hash"):
        load_generated_suite(frozen.root)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["scenarios"][0]["proxies"].update(passage_width=99), "pinned proxies"),
        (lambda d: d["config"].update(max_attempts=1), "config_hash"),
        (lambda d: d.update(suite_hash="0" * 64), "suite hash"),
        (lambda d: d["scenarios"].reverse(), "ordinal"),
        (lambda d: d["scenarios"].pop(), "scenario_count"),
        (lambda d: d.update(generator_version="0.9.0"), "generator_version"),
        (lambda d: d.update(generator_seed=SEED), "unknown keys"),
        (lambda d: d["scenarios"][0].update(content_hash="f" * 64), "does not match the freeze record"),
    ],
)
def test_tampered_freeze_record_is_detected(frozen: FrozenGeneratedSuite, edit: Callable[[Any], None], message: str) -> None:
    _rewrite_json(frozen.root / GENERATION_MANIFEST_NAME, edit)
    with pytest.raises(GenerationError, match=re.escape(message)):
        load_generated_suite(frozen.root)


def test_extra_or_missing_files_are_detected(frozen: FrozenGeneratedSuite) -> None:
    (frozen.root / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(GenerationError, match="unexpected files"):
        load_generated_suite(frozen.root)
    (frozen.root / "notes.txt").unlink()
    (frozen.root / GENERATION_MANIFEST_NAME).unlink()
    with pytest.raises(GenerationError, match="freeze record not found"):
        load_generated_suite(frozen.root)


def test_regeneration_with_the_wrong_seed_is_detected(frozen: FrozenGeneratedSuite) -> None:
    with pytest.raises(GenerationError, match="differs from the regenerated suite"):
        verify_regeneration(frozen.root, OTHER_SEED)


# --------------------------------------------------------------------------- evaluation integration and leakage


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def factory_of(policy: Policy) -> Callable[[], Policy]:
    return lambda: policy


def test_same_frozen_suite_serves_two_policies(frozen: FrozenGeneratedSuite, tmp_path: Path) -> None:
    suite = load_generated_suite(frozen.root).suite
    manifest = build_evaluation_manifest(suite, [11, 7])
    assert manifest.suite_hash == frozen.generation.suite_hash
    assert [e.scenario_hash for e in manifest.episodes[::2]] == [e.content_hash for e in frozen.generation.entries]
    policies: dict[str, Policy] = {"stay": stay_policy, "greedy": greedy_exit_policy}
    results = {
        name: run_evaluation(suite, manifest, name, factory_of(policy), tmp_path / f"{name}.json")
        for name, policy in policies.items()
    }
    assert results["stay"].manifest_hash == results["greedy"].manifest_hash == manifest.manifest_hash()
    report = pair_results(results["stay"], results["greedy"])
    assert report.paired_count == len(manifest.episodes) == 8
    assert sum(e.metrics.safety.evacuated for e in results["greedy"].episodes) > sum(e.metrics.safety.evacuated for e in results["stay"].episodes)
    # Neither the evaluation manifest nor the result artifacts carry the generator seed.
    artifacts = [manifest.to_file_text().encode(), *((tmp_path / f"{name}.json").read_bytes() for name in policies)]
    for data in artifacts:
        assert str(SEED).encode() not in data and f"{SEED:x}".encode() not in data


def test_generator_seed_never_reaches_the_model(frozen: FrozenGeneratedSuite, tmp_path: Path) -> None:
    seen: list[str] = []

    def recording_policy(obs: Observation) -> dict[str, Action]:
        seen.append(canonical_json(obs.to_dict()))
        for value in vars(obs).values():
            assert not isinstance(value, Scenario)
        return greedy_exit_policy(obs)

    for scenario in frozen.suite.scenarios:
        run_episode(scenario, recording_policy, seed=11)
    forms = {str(SEED), f"{SEED:x}", f"{SEED:X}"}
    blob = "\n".join(seen)
    assert seen and not any(f in blob for f in forms)
    # Nothing in the frozen artifacts carries the seed either.
    for data in all_files(frozen.root).values():
        assert not any(f.encode() in data for f in forms)
    assert "seed" not in json.loads((frozen.root / GENERATION_MANIFEST_NAME).read_text(encoding="utf-8"))
    for scenario in frozen.suite.scenarios:
        assert scenario.seed == 0


def test_observation_never_carries_generation_metadata(frozen: FrozenGeneratedSuite) -> None:
    observed: list[dict[str, Any]] = []

    def policy(obs: Observation) -> dict[str, Action]:
        observed.append(obs.to_dict())
        return stay_policy(obs)

    scenario = frozen.suite.scenarios[0]
    run_episode(scenario, policy, seed=3)
    text = canonical_json(observed)
    for secret in (frozen.generation.config_hash, frozen.generation.batch_id, scenario.scenario_id, scenario.content_hash(), "proxies"):
        assert secret not in text

