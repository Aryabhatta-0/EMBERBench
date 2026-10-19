"""Canonical V1 public scenario suite: manifest integrity and per-scenario diagnostics.

The diagnostic tests check only static, objective properties that define each
scenario's challenge. They never name a winning policy.
"""

import functools
import hashlib
import json
import re
import shutil
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from conftest import FIXTURE_DIR, FIXTURE_NAMES, fixture_scenario
from emberbench import (
    Action,
    FireModelName,
    Pos,
    Scenario,
    ScenarioFamily,
    Simulator,
    Suite,
    SuiteManifest,
    SuiteValidationError,
    load_scenario,
    load_suite,
    run_episode,
    save_scenario,
    verify_record,
)
from emberbench.grid import neighbours
from emberbench.suite import MANIFEST_NAME, SeedRole, Visibility

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"
SUITE_DOC = REPO_ROOT / "docs" / "scenario-suite.md"

# Pinned identity of suite 1.0.0. Any change to a scenario or to the manifest
# fails here. Such a change needs a new suite_version, not an in-place edit.
SUITE_VERSION = "1.0.0"
SUITE_HASH = "5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b"
PINNED_HASHES: dict[str, str] = {
    "v1-pub-01-single-exit-sparse": "d70b5c78b24e3020ebd7c3b3ddd7cd6ccc5ec6232c051cd2aaea2d007b84e59e",
    "v1-pub-02-single-exit-dense": "8fc36c0b598f123e59cfd102feb2ed48276b3ee9da78e62f72853105b2bd7042",
    "v1-pub-03-corridor-bottleneck": "0690eff35dba913fcdccef5d44eca0b360d4ece5b70be0e1616db61cc95e1b9a",
    "v1-pub-04-unequal-exits": "f5b630f390b69cf758df56907037e032455ab0dff06f2a615bdaead2ec50b64e",
    "v1-pub-05-central-ignition": "e1077d245b8c395403a1836578227479e2a32248a496fd301e68275186d0b937",
    "v1-pub-06-threatened-exit": "413bca35898f45a9b04446d513342e48d7389dff344520a41184531b593c3d34",
    "v1-pub-07-multiple-ignitions": "cab42b2e0ab0324575b55fd5feb17070e4b73b7002b2375e156beedc6ce4c954",
    "v1-pub-08-maze-deceptive-exit": "6aaf5079a9833ee862b3b07d7c300ef54b3bccba6e9f4f73e97e779d27a53445",
    "v1-pub-09-crowd-100-split-flow": "4cae3e2f63610053571e764305adec41f0820dc7ebd0dc658fc1760bbecf9964",
    "v1-pub-10-route-invalidation": "ee407099e5a8900e0d1d0fd7987e375aefda31d74bf31f74d6040e2f50e0ebef",
}
SCENARIO_IDS = sorted(PINNED_HASHES)
DEV_SEEDS = range(10)  # development seeds for smoke checks only, never scored seeds


@pytest.fixture(scope="module")
def suite() -> Suite:
    return load_suite(SUITE_DIR)


def scenario(scenario_id: str) -> Scenario:
    return load_scenario(SUITE_DIR / f"{scenario_id}.json")


DETERMINISTIC_IDS = [i for i in SCENARIO_IDS if scenario(i).fire.model is FireModelName.DETERMINISTIC]


def manifest_doc() -> dict[str, Any]:
    data = json.loads((SUITE_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def copy_suite(tmp_path: Path) -> Path:
    target = tmp_path / "suite"
    shutil.copytree(SUITE_DIR, target)
    return target


def write_manifest(directory: Path, data: dict[str, Any]) -> None:
    (directory / MANIFEST_NAME).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- static geometry helpers


def bfs(s: Scenario, sources: Iterable[Pos], blocked: frozenset[Pos]) -> dict[Pos, int]:
    """4-neighbour path length from the nearest source to every reachable cell."""
    dist = {p: 0 for p in sources}
    frontier = deque(sorted(dist))
    while frontier:
        cell = frontier.popleft()
        for nb in neighbours(cell, s.width, s.height):
            if nb not in dist and nb not in blocked:
                dist[nb] = dist[cell] + 1
                frontier.append(nb)
    return dist


def exit_distances(s: Scenario, blocked: frozenset[Pos] = frozenset()) -> dict[str, dict[Pos, int]]:
    """Per-exit static path distances around walls (and ``blocked``). People and fire are ignored."""
    return {e.id: bfs(s, [e.position], s.walls | blocked) for e in s.exits}


def nearest_exit(per_exit: dict[str, dict[Pos, int]], pos: Pos) -> str:
    return min((d[pos], exit_id) for exit_id, d in per_exit.items() if pos in d)[1]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


@functools.cache
def deterministic_ignition_steps(s: Scenario) -> dict[Pos, int]:
    """Step at which each cell starts burning under the deterministic model (0 for ignitions)."""
    assert s.fire.model is FireModelName.DETERMINISTIC
    blocked = s.walls if s.fire.exits_flammable else s.walls | frozenset(s.exit_positions)
    return {p: d * s.fire.spread_interval for p, d in bfs(s, s.ignitions, blocked).items()}


def solo_escape_step(s: Scenario, start: Pos, exit_ids: Iterable[str]) -> int | None:
    """Earliest step at which a person alone, starting at ``start``, can evacuate through one of ``exit_ids``.

    Time-expanded search over the frozen V1 phase order with deterministic fire:
    a move into a cell burning at the start of the step is fatal, an exit
    evacuates if it was not burning at the start of the step, and a person on
    a cell that ignites in the step dies. Other people are ignored, so this is
    an upper bound on what any policy can achieve for that person.
    """
    ignites = deterministic_ignition_steps(s)
    never = s.termination.max_steps + 1
    targets = {e.position for e in s.exits if e.id in set(exit_ids)}
    frontier = {start}
    for t in range(s.termination.max_steps):
        reached: set[Pos] = set()
        for cell in frontier:
            for nb in [cell, *neighbours(cell, s.width, s.height)]:
                if nb in s.walls or ignites.get(nb, never) <= t:
                    continue
                if nb in targets:
                    return t + 1
                if nb in s.exit_positions or ignites.get(nb, never) <= t + 1:
                    continue
                reached.add(nb)
        frontier = reached
        if not frontier:
            return None
    return None


def render_static(s: Scenario) -> str:
    grid = [["." for _ in range(s.width)] for _ in range(s.height)]
    for cells, glyph in (
        (s.obstacles, "#"),
        ((e.position for e in s.exits), "E"),
        (s.ignitions, "F"),
        ((p.position for p in s.people), "P"),
    ):
        for p in cells:
            grid[p.row][p.col] = glyph
    return "\n".join("".join(row) for row in grid)


# --------------------------------------------------------------------------- manifest and identity


def test_suite_identity_is_pinned(suite: Suite) -> None:
    m = suite.manifest
    assert m.suite_id == "emberbench-v1-public"
    assert m.suite_version == SUITE_VERSION
    assert m.visibility is Visibility.PUBLIC
    assert m.seed_role is SeedRole.DEVELOPMENT
    assert m.suite_hash() == SUITE_HASH


def test_manifest_lists_exactly_the_pinned_scenarios(suite: Suite) -> None:
    assert [e.scenario_id for e in suite.manifest.entries] == SCENARIO_IDS
    assert [s.scenario_id for s in suite.scenarios] == SCENARIO_IDS


def test_all_manifest_paths_exist() -> None:
    for entry in manifest_doc()["scenarios"]:
        assert (SUITE_DIR / entry["path"]).is_file(), entry["path"]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_parses_under_strict_schema(scenario_id: str) -> None:
    text = (SUITE_DIR / f"{scenario_id}.json").read_text(encoding="utf-8")
    s = Scenario.from_json(text)
    assert s.scenario_id == scenario_id
    assert s.schema_version == "1.0"


def test_ids_and_hashes_are_unique(suite: Suite) -> None:
    ids = [e.scenario_id for e in suite.manifest.entries]
    hashes = [e.content_hash for e in suite.manifest.entries]
    assert len(set(ids)) == len(ids)
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_hash_matches_manifest_and_pin(suite: Suite, scenario_id: str) -> None:
    s = scenario(scenario_id)
    assert s.content_hash() == suite.entry(scenario_id).content_hash == PINNED_HASHES[scenario_id]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_manifest_derived_fields_match_scenario(suite: Suite, scenario_id: str) -> None:
    s, e = scenario(scenario_id), suite.entry(scenario_id)
    assert e.path == f"{scenario_id}.json"
    assert e.schema_version == s.schema_version
    assert e.population == len(s.people)
    assert e.exit_count == len(s.exits)
    assert e.ignition_count == len(s.ignitions)
    assert e.fire_model is s.fire.model
    assert e.primary_family in e.families


def test_manifest_ordering_and_text_are_canonical(suite: Suite) -> None:
    text = (SUITE_DIR / MANIFEST_NAME).read_text(encoding="utf-8")
    assert text == suite.manifest.to_file_text()
    ids = [e.scenario_id for e in suite.manifest.entries]
    assert ids == sorted(ids)
    reparsed = SuiteManifest.from_dict(json.loads(text))
    assert reparsed == suite.manifest
    assert reparsed.suite_hash() == SUITE_HASH


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_file_round_trips_unchanged(scenario_id: str, tmp_path: Path) -> None:
    path = SUITE_DIR / f"{scenario_id}.json"
    s = load_scenario(path)
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(s.to_dict(), indent=2) + "\n"
    save_scenario(s, tmp_path / "copy.json")
    assert (tmp_path / "copy.json").read_text(encoding="utf-8") == text
    assert Scenario.from_json(s.to_json()) == s


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_every_configurable_rule_is_explicit(scenario_id: str) -> None:
    raw = json.loads((SUITE_DIR / f"{scenario_id}.json").read_text(encoding="utf-8"))
    fire_keys = {"model", "spread_interval", "exits_flammable"}
    if raw["fire"]["model"] == FireModelName.STOCHASTIC.value:
        fire_keys.add("spread_probability")
    assert set(raw["fire"]) == fire_keys
    assert set(raw["movement"]) == {"conflict_resolution", "enter_fire"}
    assert set(raw["termination"]) == {"max_steps"}
    assert raw["movement"]["enter_fire"] == "casualty"
    assert raw["description"].startswith("CANONICAL V1 PUBLIC SCENARIO. ")


def test_every_required_family_is_covered(suite: Suite) -> None:
    assert suite.manifest.families == frozenset(ScenarioFamily)
    primaries = Counter(e.primary_family for e in suite.manifest.entries)
    assert all(n == 1 for n in primaries.values()), "each scenario has its own primary challenge"


def test_seeds_are_development_defaults_only(suite: Suite) -> None:
    # Fixture/dev seeds are not hidden benchmark seeds. The suite stores no
    # seed schedule: only the single required per-scenario default.
    assert suite.manifest.seed_role is SeedRole.DEVELOPMENT
    assert all(s.seed == 0 for s in suite.scenarios)
    assert "seed" not in json.dumps(manifest_doc()["scenarios"])


# --------------------------------------------------------------------------- start positions and IDs


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_one_starts_on_a_wall_exit_or_fire(scenario_id: str) -> None:
    s = scenario(scenario_id)
    forbidden = s.walls | frozenset(s.exit_positions) | frozenset(s.ignitions)
    assert not [p.id for p in s.people if p.position in forbidden]
    assert not frozenset(s.ignitions) & frozenset(s.exit_positions)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_duplicate_people_exits_or_positions(scenario_id: str) -> None:
    s = scenario(scenario_id)
    for ids in ([p.id for p in s.people], [e.id for e in s.exits]):
        assert len(set(ids)) == len(ids)
    cells = [p.position for p in s.people] + [e.position for e in s.exits] + list(s.ignitions) + list(s.obstacles)
    assert len(set(cells)) == len(cells)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_ids_are_neutral(scenario_id: str) -> None:
    s = scenario(scenario_id)
    n = len(s.people)
    assert [p.id for p in s.people] == [f"P{i:03d}" for i in range(1, n + 1)]
    # The documented rule: rank of a salted SHA-256 of the start cell.
    by_hash = sorted(
        s.people,
        key=lambda p: hashlib.sha256(f"emberbench/v1/public/person-id/{s.scenario_id}/{p.position.row}/{p.position.col}".encode()).hexdigest(),
    )
    assert [p.id for p in by_hash] == [p.id for p in s.people]
    assert [e.id for e in s.exits] == [f"E{i}" for i in range(1, len(s.exits) + 1)]
    assert [e.id for e in sorted(s.exits, key=lambda e: e.position)] == [e.id for e in s.exits]
    if n >= 3:
        per_exit = exit_distances(s)
        nearest = [min(d[p.position] for d in per_exit.values()) for p in s.people]
        assert nearest not in (sorted(nearest), sorted(nearest, reverse=True)), "ID order follows exit distance"
        assert [p.id for p in sorted(s.people, key=lambda p: p.position)] != [p.id for p in s.people]


@pytest.mark.parametrize("scenario_id", DETERMINISTIC_IDS)
def test_no_person_is_doomed_by_construction(scenario_id: str) -> None:
    # Stochastic fire has no fixed arrival times; the route-invalidation
    # diagnostics cover that scenario instead.
    s = scenario(scenario_id)
    every_exit = [e.id for e in s.exits]
    doomed = [p.id for p in s.people if solo_escape_step(s, p.position, every_exit) is None]
    assert doomed == []


# --------------------------------------------------------------------------- separation from fixtures


def test_canonical_scenarios_are_not_test_fixtures() -> None:
    assert not SUITE_DIR.resolve().is_relative_to((REPO_ROOT / "tests").resolve())
    for entry in manifest_doc()["scenarios"]:
        path = (SUITE_DIR / entry["path"]).resolve()
        assert path.parent == SUITE_DIR.resolve()
        assert not path.is_relative_to(FIXTURE_DIR.resolve())
    assert not list(FIXTURE_DIR.glob("v1-pub-*"))


def test_fixtures_are_not_in_the_canonical_suite(suite: Suite) -> None:
    ids = {e.scenario_id for e in suite.manifest.entries}
    hashes = {e.content_hash for e in suite.manifest.entries}
    for name in FIXTURE_NAMES:
        f = fixture_scenario(name)
        assert f.description.startswith("TEST FIXTURE.")
        assert f.scenario_id not in ids
        assert f.content_hash() not in hashes
    assert not [s.scenario_id for s in suite.scenarios if s.description.startswith("TEST FIXTURE")]


def test_suite_directory_holds_only_listed_files(suite: Suite) -> None:
    present = sorted(p.name for p in SUITE_DIR.iterdir())
    assert present == sorted([MANIFEST_NAME, *(e.path for e in suite.manifest.entries)])


# --------------------------------------------------------------------------- loader rejects bad suites


def test_loader_rejects_a_tampered_hash(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    data = manifest_doc()
    data["scenarios"][0]["content_hash"] = "0" * 64
    write_manifest(d, data)
    with pytest.raises(SuiteValidationError, match="content_hash"):
        load_suite(d)


def test_loader_rejects_an_edited_scenario(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    s = load_scenario(d / f"{SCENARIO_IDS[0]}.json")
    doc = s.to_dict()
    doc["termination"]["max_steps"] += 1
    (d / f"{SCENARIO_IDS[0]}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SuiteValidationError, match="content_hash"):
        load_suite(d)


def test_loader_rejects_wrong_metadata(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    data = manifest_doc()
    data["scenarios"][1]["population"] += 1
    write_manifest(d, data)
    with pytest.raises(SuiteValidationError, match="population"):
        load_suite(d)


def test_loader_rejects_unsorted_entries(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    data = manifest_doc()
    data["scenarios"].reverse()
    write_manifest(d, data)
    with pytest.raises(SuiteValidationError, match="sorted"):
        load_suite(d)


def test_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    data = manifest_doc()
    data["seed_schedule"] = [1, 2, 3]
    write_manifest(d, data)
    with pytest.raises(SuiteValidationError, match="unknown keys"):
        load_suite(d)


def test_loader_rejects_missing_and_extra_files(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    shutil.copy(FIXTURE_DIR / f"{FIXTURE_NAMES[0]}.json", d / "extra.json")
    with pytest.raises(SuiteValidationError, match="not listed"):
        load_suite(d)
    (d / "extra.json").unlink()
    (d / f"{SCENARIO_IDS[-1]}.json").unlink()
    with pytest.raises(SuiteValidationError, match="does not exist"):
        load_suite(d)


def test_loader_rejects_paths_outside_the_suite(tmp_path: Path) -> None:
    d = copy_suite(tmp_path)
    data = manifest_doc()
    data["scenarios"][0]["path"] = f"../{SCENARIO_IDS[0]}.json"
    write_manifest(d, data)
    with pytest.raises(SuiteValidationError, match="path"):
        load_suite(d)


# --------------------------------------------------------------------------- documentation


def doc_layout(doc: str, scenario_id: str) -> str:
    match = re.search(rf"^### {re.escape(scenario_id)}\n.*?```text\n(.*?)```", doc, flags=re.S | re.M)
    assert match, f"no layout for {scenario_id}"
    return match.group(1).rstrip("\n")


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_documented_layout_and_hash_match_the_file(scenario_id: str) -> None:
    doc = SUITE_DOC.read_text(encoding="utf-8")
    s = scenario(scenario_id)
    assert doc_layout(doc, scenario_id) == render_static(s)
    assert f"`{PINNED_HASHES[scenario_id]}`" in doc
    assert f"`{SUITE_HASH}`" in doc


# --------------------------------------------------------------------------- smoke run


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_smoke_run_replays(scenario_id: str) -> None:
    # Everyone stays put: this only proves that the simulator accepts the
    # scenario and that the record replays. It is not a reference policy.
    s = scenario(scenario_id)
    record = run_episode(s, lambda obs: {p.id: Action.STAY for p in obs.active})
    verify_record(s, record)
    assert record.final_state.terminated


# --------------------------------------------------------------------------- per-scenario diagnostics


def open_cells(s: Scenario) -> int:
    return s.width * s.height - len(s.obstacles) - len(s.exits)


def test_single_exit_sparse_is_low_density() -> None:
    s = scenario("v1-pub-01-single-exit-sparse")
    assert len(s.exits) == 1 and len(s.ignitions) == 1
    assert len(s.people) / open_cells(s) <= 0.1
    to_exit = exit_distances(s)["E1"]
    # The fire starts farther from the exit than any person, so it is time
    # pressure, not a routing obstacle.
    assert to_exit[s.ignitions[0]] > max(to_exit[p.position] for p in s.people)


def test_single_exit_dense_is_throughput_bound() -> None:
    s = scenario("v1-pub-02-single-exit-dense")
    assert len(s.exits) == 1
    assert len(s.people) / open_cells(s) >= 0.4
    to_exit = exit_distances(s)["E1"]
    # One person per exit per step (R4): the last evacuation cannot happen
    # before step N, which is later than every person's walk.
    assert len(s.people) > max(to_exit[p.position] for p in s.people)


def test_corridor_bottleneck_has_one_wide_cut() -> None:
    s = scenario("v1-pub-03-corridor-bottleneck")
    assert len(s.exits) == 1 and len(s.people) > 1
    cut_cells = []
    for row in range(s.height):
        for col in range(s.width):
            c = Pos(row, col)
            if c in s.walls or c in s.exit_positions or any(p.position == c for p in s.people):
                continue
            reach = exit_distances(s, frozenset({c}))
            if all(p.position not in d for p in s.people for d in reach.values()):
                cut_cells.append(c)
    # Cut cells with at most two open neighbours form a one-cell-wide corridor.
    corridor = [c for c in cut_cells if len([nb for nb in neighbours(c, s.width, s.height) if nb not in s.walls]) <= 2]
    assert len(corridor) >= 5, "every route shares a one-wide corridor of at least five cells"
    # The fire is on the people's side of the corridor.
    behind = bfs(s, [p.position for p in s.people], s.walls | {corridor[0]})
    assert all(f in behind for f in s.ignitions)


def test_unequal_exits_have_clearly_different_distances() -> None:
    s = scenario("v1-pub-04-unequal-exits")
    assert len(s.exits) >= 2
    per_exit = exit_distances(s)
    for p in s.people:
        ds = sorted(d[p.position] for d in per_exit.values())
        assert all(b - a >= 4 for a, b in zip(ds, ds[1:])), p.id
    assert {nearest_exit(per_exit, p.position) for p in s.people} == {e.id for e in s.exits}


def test_central_ignition_is_central_and_surrounded() -> None:
    s = scenario("v1-pub-05-central-ignition")
    assert s.ignitions == (Pos(s.height // 2, s.width // 2),)
    f = s.ignitions[0]
    rows = [p.position.row for p in s.people]
    cols = [p.position.col for p in s.people]
    assert min(rows) < f.row < max(rows) and min(cols) < f.col < max(cols)
    assert any(e.position.col < f.col for e in s.exits) and any(e.position.col > f.col for e in s.exits)


def test_threatened_exit_can_burn_before_some_people_reach_it() -> None:
    s = scenario("v1-pub-06-threatened-exit")
    assert s.fire.exits_flammable and len(s.exits) >= 2
    per_exit = exit_distances(s)
    nearest = Counter(nearest_exit(per_exit, p.position) for p in s.people)
    threatened = nearest.most_common(1)[0][0]
    assert nearest[threatened] > len(s.people) / 2
    burns = deterministic_ignition_steps(s)
    position = {e.id: e.position for e in s.exits}
    others = [e.id for e in s.exits if e.id != threatened]
    assert burns[position[threatened]] < min(burns.get(position[o], s.termination.max_steps + 1) for o in others)
    assert burns[position[threatened]] < s.termination.max_steps
    committed = [p for p in s.people if nearest_exit(per_exit, p.position) == threatened]
    lost = [p for p in committed if solo_escape_step(s, p.position, [threatened]) is None]
    in_time = [p for p in committed if solo_escape_step(s, p.position, [threatened]) is not None]
    assert lost and in_time, "the nearest exit is safe for some people and a trap for others"
    assert all(solo_escape_step(s, p.position, others) is not None for p in lost)


def test_multiple_ignitions_are_separate_sources() -> None:
    s = scenario("v1-pub-07-multiple-ignitions")
    assert len(s.ignitions) >= 3
    pairs = [(a, b) for i, a in enumerate(s.ignitions) for b in s.ignitions[i + 1 :]]
    assert all(manhattan(a, b) >= 6 for a, b in pairs)


def test_maze_needs_detours_and_nearest_exit_is_deceptive() -> None:
    s = scenario("v1-pub-08-maze-deceptive-exit")
    per_exit = exit_distances(s)
    for p in s.people:
        for e in s.exits:
            assert per_exit[e.id][p.position] > manhattan(p.position, e.position), (p.id, e.id)
        straight_line = min((manhattan(p.position, e.position), e.id) for e in s.exits)[1]
        assert straight_line != nearest_exit(per_exit, p.position), p.id


def test_crowd_has_100_people_and_rewards_split_flow() -> None:
    s = scenario("v1-pub-09-crowd-100-split-flow")
    assert len(s.people) == 100
    assert len(s.exits) >= 2
    per_exit = exit_distances(s)
    load = Counter(nearest_exit(per_exit, p.position) for p in s.people)
    busiest, n = load.most_common(1)[0]
    assert n > len(s.people) / len(s.exits), "nearest-exit choice is unbalanced"
    # With one person per exit per step (R4), the queue at the busiest exit
    # is longer than the walk from some of its people to another exit.
    assert any(
        per_exit[other][p.position] < n
        for p in s.people
        if nearest_exit(per_exit, p.position) == busiest
        for other in per_exit
        if other != busiest
    )


def test_route_invalidation_cuts_the_short_route_at_an_uncertain_time() -> None:
    s = scenario("v1-pub-10-route-invalidation")
    assert s.fire.model is FireModelName.STOCHASTIC
    assert s.fire.spread_probability is not None and 0.0 < s.fire.spread_probability < 1.0
    assert not s.fire.exits_flammable
    base = exit_distances(s)
    flammable = s.walls | frozenset(s.exit_positions)
    fire_region = bfs(s, s.ignitions, flammable)
    # Cells that fire can reach and whose loss makes every person's route
    # longer while every person keeps some route to an exit.
    cuts = []
    for c in sorted(fire_region):
        if c in s.ignitions:
            continue
        after = exit_distances(s, frozenset({c}))
        if all(
            any(p.position in d for d in after.values())
            and min(d[p.position] for d in after.values() if p.position in d) > min(d[p.position] for d in base.values())
            for p in s.people
        ):
            cuts.append(c)
    assert cuts, "the fire can invalidate every person's shortest route"
    first_cut: set[int] = set()
    for seed in DEV_SEEDS:
        sim = Simulator(s, seed=seed)
        while not sim.done and not set(cuts) & sim.state.burning:
            sim.step({p.id: Action.STAY for p in sim.observe().active})
        assert set(cuts) & sim.state.burning, seed
        first_cut.add(sim.state.step)
    assert len(first_cut) >= 2, "the time of the cut depends on the seed"
