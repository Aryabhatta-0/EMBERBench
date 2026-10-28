"""The frozen calibration panel, admission and difficulty rules (emberbench.calibration)."""

import inspect
from pathlib import Path

import pytest

from emberbench import SCORING_VERSION, agent_policy, baseline_factory, load_suite, run_episode, score_episode
from emberbench.calibration import (
    ADMISSION_RULES_VERSION,
    ADMITTED,
    BANDS,
    CALIBRATION_PANEL,
    CALIBRATION_PANEL_VERSION,
    CEILING_MIN,
    DIFFICULTY_RULES_VERSION,
    EASY_MIN_BEST,
    EXCLUSIONS,
    FLOOR,
    FLOOR_AGENT,
    HARD_MIN_BEST,
    MODERATE_MIN_BEST,
    SATURATED,
    WorldMeasurement,
    calibrate,
    calibration_rules,
    capacity_ceiling,
    difficulty_band,
    measure_world,
)
from emberbench.families import FireEvidence
from emberbench.scoring import OFFICIAL_CONSTANTS

PUBLIC = Path(__file__).resolve().parent.parent / "scenarios" / "public" / "v1"
NO_FIRE = FireEvidence(0.0, 0.0, 0.0, 0.0)


def world(scores: tuple[float, float, float, float], *, savable: int = 10, cap: int | None = None, ceiling: float | None = 100.0) -> WorldMeasurement:
    return WorldMeasurement(
        seed=1,
        world="0" * 64,
        savable=savable,
        capacity_savable=savable if cap is None else cap,
        capacity_ceiling=ceiling if savable else None,
        fire=NO_FIRE,
        panel_scores=tuple(zip(CALIBRATION_PANEL, scores, strict=True)),
    )


def test_versions_and_thresholds_are_the_pr33_values() -> None:
    assert (CALIBRATION_PANEL_VERSION, DIFFICULTY_RULES_VERSION, ADMISSION_RULES_VERSION) == ("1.0", "1.0", "1.0")
    assert CALIBRATION_PANEL == ("nearest-exit-greedy-v1", "static-shortest-path-v1", "fire-aware-shortest-path-v1", "congestion-aware-v1")
    assert FLOOR_AGENT == "random-legal-v1" and FLOOR_AGENT not in CALIBRATION_PANEL
    assert (EASY_MIN_BEST, MODERATE_MIN_BEST, HARD_MIN_BEST) == (80.0, 30.0, 10.0)
    assert (SATURATED, FLOOR, CEILING_MIN) == (95.0, 1.0, 50.0)
    assert BANDS == ("easy", "moderate", "hard", "stress")
    assert EXCLUSIONS == ("not_applicable", "capacity_artifact", "saturated", "floor", "no_separation")
    rules = calibration_rules()
    assert rules["calibration_panel"] == list(CALIBRATION_PANEL)
    assert rules["difficulty_rules_version"] == DIFFICULTY_RULES_VERSION


def test_difficulty_band_boundaries() -> None:
    assert [difficulty_band(b) for b in (100.0, 80.0, 79.99, 30.0, 29.99, 10.0, 9.99, 0.0)] == [
        "easy", "easy", "moderate", "moderate", "hard", "hard", "stress", "stress",
    ]  # fmt: skip


def test_admission_rules_in_order() -> None:
    assert calibrate([world((0, 0, 0, 0), savable=0)]).admission == "not_applicable"
    # The capacity filter comes first: a saturated panel on a capacity artifact is still a capacity artifact.
    assert calibrate([world((99, 99, 99, 99), ceiling=49.9)]).admission == "capacity_artifact"
    assert calibrate([world((99, 99, 99, 99), ceiling=50.0)]).admission == "saturated"
    assert calibrate([world((0.5, 0.9, 0.2, 0.0))]).admission == "floor"
    assert calibrate([world((40.0, 40.5, 40.2, 40.9))]).admission == "no_separation"
    c = calibrate([world((40.0, 60.0, 50.0, 45.0))])
    assert (c.admission, c.band, c.best, c.worst, c.spread) == (ADMITTED, "moderate", 60.0, 40.0, 20.0)
    assert 1.0 == float(OFFICIAL_CONSTANTS.practical_difference)


def test_calibration_uses_the_mean_over_applicable_worlds() -> None:
    c = calibrate([world((90, 50, 70, 60)), world((70, 10, 30, 20)), world((0, 0, 0, 0), savable=0)])
    assert c.panel[CALIBRATION_PANEL[0]] == 80.0 and c.best == 80.0 and c.band == "easy"
    capped = calibrate([world((90, 50, 70, 60), cap=8, ceiling=60.0), world((70, 10, 30, 20), ceiling=40.0)])
    assert capped.capacity_limited and capped.ceiling == 50.0 and capped.admission == ADMITTED


def test_capacity_ceiling_is_the_official_score_of_the_capacity_bound() -> None:
    assert capacity_ceiling(10, 0, 0) is None
    assert capacity_ceiling(10, 10, 10) == 100.0
    full, half = capacity_ceiling(20, 10, 10), capacity_ceiling(20, 10, 5)
    assert full is not None and half is not None and half < full


@pytest.mark.parametrize("name", ["v1-pub-01-single-exit-sparse", "v1-pub-06-threatened-exit"])
def test_panel_scores_equal_the_official_score_episode(name: str) -> None:
    scenario = load_suite(PUBLIC).get(name)
    m = measure_world(scenario, 5)
    for agent in CALIBRATION_PANEL:
        record = run_episode(scenario, agent_policy(baseline_factory(agent)()), seed=5)
        assert m.score(agent) == float(score_episode(scenario, record).score)
    assert m.savable >= m.capacity_savable >= 0
    assert SCORING_VERSION == "1.1"


def test_no_model_is_an_input_to_calibration() -> None:
    for fn in (measure_world, calibrate):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"model", "agent", "submission", "policy", "factory"}
    assert inspect.signature(measure_world).parameters["panel"].default == CALIBRATION_PANEL
