import copy
import json
from typing import Any

import pytest

from conftest import FIXTURE_DIR, FIXTURE_NAMES, fixture_scenario
from emberbench import Scenario, ScenarioValidationError, load_scenario, save_scenario
from emberbench.scenario import FireConfig, FireModelName


def base_doc() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((FIXTURE_DIR / "one_person_one_exit.json").read_text())
    return doc


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixtures_are_valid_and_marked_as_test_fixtures(name: str) -> None:
    scenario = fixture_scenario(name)
    assert scenario.scenario_id == name
    assert scenario.description.startswith("TEST FIXTURE.")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_round_trip_is_stable(name: str, tmp_path: Any) -> None:
    scenario = fixture_scenario(name)
    again = Scenario.from_dict(scenario.to_dict())
    assert again == scenario
    assert again.to_json() == scenario.to_json()
    assert Scenario.from_json(scenario.to_json()).content_hash() == scenario.content_hash()
    path = tmp_path / "s.json"
    save_scenario(scenario, path)
    assert load_scenario(path) == scenario


def test_input_order_does_not_change_identity() -> None:
    doc = base_doc()
    doc["obstacles"].reverse()
    shuffled = Scenario.from_dict(doc)
    assert shuffled == Scenario.from_dict(base_doc())
    assert shuffled.content_hash() == Scenario.from_dict(base_doc()).content_hash()


def test_hash_changes_with_any_input() -> None:
    a = Scenario.from_dict(base_doc())
    doc = base_doc()
    doc["seed"] = 2
    assert Scenario.from_dict(doc).content_hash() != a.content_hash()


def _mutate(path: str, value: Any) -> dict[str, Any]:
    doc = copy.deepcopy(base_doc())
    *parents, last = path.split(".")
    target: Any = doc
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if value is _DELETE:
        del target[last]
    elif isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return doc


_DELETE = object()

INVALID_CASES = [
    ("schema_version", "2.0", "unsupported version"),
    ("schema_version", _DELETE, "missing required keys"),
    ("seed", _DELETE, "missing required keys"),
    ("termination", _DELETE, "missing required keys"),
    ("bogus", 1, "unknown keys"),
    ("fire.bogus", 1, "unknown keys"),
    ("seed", -1, "seed"),
    ("seed", 2**64, "seed"),
    ("seed", True, "expected an integer"),
    ("seed", 1.0, "expected an integer"),
    ("scenario_id", "has space", "id must match"),
    ("grid.width", 0, "grid.width"),
    ("grid.height", "3", "expected an integer"),
    ("people", [], "at least one person"),
    ("exits", [], "at least one exit"),
    ("people.0.position", [1, 99], "outside"),
    ("people.0.position", [0, 0], "obstacle"),
    ("people.0.position", [1, 5], "cannot start on an exit"),
    ("people.0.position", [1], "position must be"),
    ("people.0.id", "", "id must match"),
    ("exits.0.position", [0, 3], "obstacle"),
    ("ignitions", [[0, 0]], "walls cannot burn"),
    ("ignitions", [[1, 1]], "ignition point"),
    ("ignitions", [[1, 3], [1, 3]], "duplicate position"),
    ("obstacles", [[0, 0], [0, 0]], "duplicate position"),
    ("fire.model", "magic", "must be one of"),
    ("fire.spread_interval", 0, "fire.spread_interval"),
    ("fire.exits_flammable", 1, "expected a boolean"),
    ("fire.spread_probability", 0.5, "only allowed when"),
    ("movement.enter_fire", "maybe", "must be one of"),
    ("movement.conflict_resolution", "random", "must be one of"),
    ("termination.max_steps", 0, "termination.max_steps"),
]


@pytest.mark.parametrize(("path", "value", "message"), INVALID_CASES)
def test_invalid_scenarios_are_rejected(path: str, value: Any, message: str) -> None:
    with pytest.raises(ScenarioValidationError, match=message):
        Scenario.from_dict(_mutate(path, value))


def test_duplicate_person_ids_rejected() -> None:
    doc = base_doc()
    doc["people"].append({"id": "P1", "position": [1, 2]})
    with pytest.raises(ScenarioValidationError, match="duplicate ids"):
        Scenario.from_dict(doc)


def test_two_people_on_one_cell_rejected() -> None:
    doc = base_doc()
    doc["people"].append({"id": "P2", "position": [1, 1]})
    with pytest.raises(ScenarioValidationError, match="duplicate position"):
        Scenario.from_dict(doc)


def test_stochastic_requires_valid_probability() -> None:
    for fire, message in [
        ({"model": "stochastic", "spread_interval": 1, "exits_flammable": True}, "required"),
        ({"model": "stochastic", "spread_interval": 1, "exits_flammable": True, "spread_probability": 1.5}, r"\[0, 1\]"),
        ({"model": "stochastic", "spread_interval": 1, "exits_flammable": True, "spread_probability": "x"}, "number"),
    ]:
        doc = base_doc()
        doc["fire"] = fire
        with pytest.raises(ScenarioValidationError, match=message):
            Scenario.from_dict(doc)


def test_ignition_on_exit_requires_flammable_exits() -> None:
    doc = base_doc()
    doc["ignitions"] = [[1, 5]]
    Scenario.from_dict(doc)  # allowed: exits_flammable is true
    doc["fire"]["exits_flammable"] = False
    with pytest.raises(ScenarioValidationError, match="exits_flammable"):
        Scenario.from_dict(doc)


def test_non_finite_and_malformed_json_rejected() -> None:
    with pytest.raises(ScenarioValidationError, match="non-finite"):
        Scenario.from_json('{"seed": NaN}')
    with pytest.raises(ScenarioValidationError, match="not valid JSON"):
        Scenario.from_json("{")
    with pytest.raises(ScenarioValidationError, match="expected an object"):
        Scenario.from_json("[]")


def test_python_construction_is_validated_too() -> None:
    s = fixture_scenario("one_person_one_exit")
    with pytest.raises(ScenarioValidationError, match="spread_probability"):
        Scenario(
            scenario_id="x",
            width=s.width,
            height=s.height,
            obstacles=s.obstacles,
            exits=s.exits,
            people=s.people,
            ignitions=(),
            seed=0,
            fire=FireConfig(model=FireModelName.STOCHASTIC),
        )
