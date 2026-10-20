"""The frozen V1 family rules (family rules version 1.0): structural tags, fire-dependent tags, split flow, deceptive and narrow detection."""

import dataclasses
from typing import Any

import pytest
from conftest import layout

from emberbench import GeneratorConfig, ScenarioFamily, generate_suite
from emberbench.calibration import fire_evidence
from emberbench.families import (
    FAMILY_RULES_VERSION,
    FIRE_DEPENDENT_FAMILIES,
    PRIMARY_FAMILIES,
    SECONDARY_FAMILIES,
    STRUCTURAL_PRIMARY_FAMILIES,
    FireEvidence,
    descriptive_primary,
    family_rules,
    family_tags,
    is_deceptive,
    is_narrow_corridor,
    is_split_flow,
    primary_candidates,
    structural_tags,
)
from emberbench.generator import describe_families
from emberbench.scenario import FireConfig, FireModelName, TerminationConfig
from emberbench.structure import StructuralProfile, profile_structure

F = ScenarioFamily
NO_FIRE = FireEvidence(0.0, 0.0, 0.0, 0.0)

DECEPTIVE = [".....E......", "###########.", "....abcd...E", "F..........."]
DECEPTIVE_OPEN = [".....E......", "............", "....abcd...E", "F..........."]
NARROW = ["..........", ".abcd.....", ".efgh.....", "#####.####", "..........", "....E...F."]
WIDE_DOOR = ["..........", ".abcd.....", ".efgh.....", "####...###", "..........", "....E...F."]
DOOR_FEW = ["..........", ".a........", "......b...", "#####.####", "..........", "....E...F."]
SPLIT_LOW = ["E..........................E", "abcdefghij........klmnopqrst", "............F..............."]
SPLIT_HIGH = ["E.E.........................", "abcdefghijklmnopqrstu.......", "...........................F"]


def profile_of(rows: list[str]) -> StructuralProfile:
    return profile_structure(layout(rows))


def base() -> StructuralProfile:
    """A neutral two-exit profile with no tag of its own, to move one measure at a time."""
    return dataclasses.replace(
        profile_of(SPLIT_LOW),
        exit_count=2,
        population=10,
        crowd_density=0.05,
        ignition_count=1,
        passage_width=3,
        bottleneck_severity=1.0,
        obstacle_density=0.0,
        detour_ratio=1.0,
        deceptive_share=0.0,
        ignition_centrality=1.0,
        exit_gap_median=1.0,
        split_pressure=0.5,
    )


def tags(**changes: Any) -> frozenset[ScenarioFamily]:
    return structural_tags(dataclasses.replace(base(), **changes))


def test_the_family_sets_are_the_frozen_v1_sets() -> None:
    assert FAMILY_RULES_VERSION == "1.0"
    assert [f.value for f in PRIMARY_FAMILIES] == [
        "single_exit_low_density",
        "single_exit_high_density",
        "narrow_corridor_bottleneck",
        "split_flow_across_exits",
        "crowd_100",
        "maze_obstacle_routing",
        "deceptive_nearest_exit",
        "threatened_nearest_exit",
        "multiple_ignitions",
        "route_invalidation_replanning",
    ]
    assert set(SECONDARY_FAMILIES) == {F.CENTRAL_IGNITION, F.MULTIPLE_EXITS_UNEQUAL_DISTANCE}
    assert set(PRIMARY_FAMILIES) | set(SECONDARY_FAMILIES) == set(ScenarioFamily)
    assert FIRE_DEPENDENT_FAMILIES == {F.THREATENED_NEAREST_EXIT, F.ROUTE_INVALIDATION_REPLANNING}
    assert set(STRUCTURAL_PRIMARY_FAMILIES) == set(PRIMARY_FAMILIES) - FIRE_DEPENDENT_FAMILIES


def test_family_rules_record_the_pr33_thresholds() -> None:
    t = family_rules()["thresholds"]
    assert t == {
        "high_crowd_density": 0.15,
        "split_pressure_min_exclusive": 1.0,
        "split_flow_min_population": 20,
        "crowd_100_population": 100,
        "multiple_ignitions_min": 2,
        "narrow_passage_width": 1,
        "bottleneck_severity_min": 4.0,
        "maze_obstacle_density": 0.25,
        "maze_detour_ratio": 1.5,
        "deceptive_share_min": 0.25,
        "deceptive_route_excess": 2,
        "central_ignition_max": 0.34,
        "unequal_exit_gap_min": 4.0,
        "threatened_share_min": 0.2,
        "recoverable_invalidation_min": 0.2,
    }
    assert family_rules()["family_rules_version"] == FAMILY_RULES_VERSION


def test_the_neutral_base_has_only_the_multi_exit_fallback() -> None:
    assert tags() == {F.MULTIPLE_EXITS_UNEQUAL_DISTANCE}


def test_rule_single_exit_low_and_high_density() -> None:
    assert F.SINGLE_EXIT_LOW_DENSITY in tags(exit_count=1, crowd_density=0.149)
    assert F.SINGLE_EXIT_HIGH_DENSITY in tags(exit_count=1, crowd_density=0.15)
    assert not {F.SINGLE_EXIT_LOW_DENSITY, F.SINGLE_EXIT_HIGH_DENSITY} & tags(exit_count=2, crowd_density=0.5)


def test_rule_narrow_corridor_bottleneck() -> None:
    assert F.NARROW_CORRIDOR_BOTTLENECK in tags(passage_width=1, bottleneck_severity=4.0)
    assert F.NARROW_CORRIDOR_BOTTLENECK not in tags(passage_width=1, bottleneck_severity=3.99)
    assert F.NARROW_CORRIDOR_BOTTLENECK not in tags(passage_width=2, bottleneck_severity=50.0)


def test_rule_split_flow_across_exits() -> None:
    assert F.SPLIT_FLOW_ACROSS_EXITS in tags(population=20, split_pressure=1.01)
    assert F.SPLIT_FLOW_ACROSS_EXITS not in tags(population=20, split_pressure=1.0)
    assert F.SPLIT_FLOW_ACROSS_EXITS not in tags(population=19, split_pressure=9.0)
    assert F.SPLIT_FLOW_ACROSS_EXITS not in tags(exit_count=1, population=50, split_pressure=None)


def test_rule_crowd_100() -> None:
    assert F.CROWD_100 in tags(population=100)
    assert F.CROWD_100 not in tags(population=99)


def test_rule_maze_obstacle_routing() -> None:
    assert F.MAZE_OBSTACLE_ROUTING in tags(obstacle_density=0.25)
    assert F.MAZE_OBSTACLE_ROUTING in tags(detour_ratio=1.5)
    assert F.MAZE_OBSTACLE_ROUTING not in tags(obstacle_density=0.249, detour_ratio=1.49)


def test_rule_deceptive_nearest_exit() -> None:
    assert F.DECEPTIVE_NEAREST_EXIT in tags(deceptive_share=0.25)
    assert F.DECEPTIVE_NEAREST_EXIT not in tags(deceptive_share=0.24)
    assert F.DECEPTIVE_NEAREST_EXIT not in tags(exit_count=1, deceptive_share=1.0)


def test_rule_multiple_ignitions() -> None:
    assert F.MULTIPLE_IGNITIONS in tags(ignition_count=2)
    assert F.MULTIPLE_IGNITIONS not in tags(ignition_count=1)


def test_rule_threatened_nearest_exit_uses_the_mean_over_worlds() -> None:
    p = base()
    hit, miss = FireEvidence(0.3, 0.0, 0.0, 0.1), FireEvidence(0.05, 0.0, 0.0, 0.1)
    assert F.THREATENED_NEAREST_EXIT in family_tags(p, [hit])
    assert F.THREATENED_NEAREST_EXIT in family_tags(p, [hit, hit, miss])  # mean 0.217
    assert F.THREATENED_NEAREST_EXIT not in family_tags(p, [hit, miss, miss])  # mean 0.133
    assert F.THREATENED_NEAREST_EXIT not in family_tags(dataclasses.replace(p, exit_count=1), [hit])


def test_rule_route_invalidation_needs_recoverable_invalidation() -> None:
    p = base()
    assert F.ROUTE_INVALIDATION_REPLANNING in family_tags(p, [FireEvidence(0.0, 0.5, 0.2, 0.1)])
    assert F.ROUTE_INVALIDATION_REPLANNING not in family_tags(p, [FireEvidence(0.0, 0.9, 0.1, 0.1)])
    assert F.ROUTE_INVALIDATION_REPLANNING not in family_tags(p, [])


def test_secondary_tags_stay_secondary() -> None:
    t = tags(ignition_centrality=0.2, exit_gap_median=6.0)
    assert t == {F.CENTRAL_IGNITION, F.MULTIPLE_EXITS_UNEQUAL_DISTANCE}
    assert primary_candidates(t) == frozenset()
    assert primary_candidates(t | {F.CROWD_100}) == {F.CROWD_100}
    assert descriptive_primary(t | {F.CROWD_100}) is F.CROWD_100
    with pytest.raises(ValueError):
        descriptive_primary([])


def test_structural_tags_never_carry_a_fire_dependent_family() -> None:
    for rows in (DECEPTIVE, NARROW, SPLIT_HIGH, SPLIT_LOW):
        assert not structural_tags(profile_of(rows)) & FIRE_DEPENDENT_FAMILIES


def test_split_flow_fix_low_pressure_multi_exit_is_not_split_flow() -> None:
    p = profile_of(SPLIT_LOW)
    assert p.exit_count == 2 and p.population == 20
    assert p.split_pressure is not None and p.split_pressure <= 1.0
    assert not is_split_flow(p)
    assert structural_tags(p) == {F.MULTIPLE_EXITS_UNEQUAL_DISTANCE}


def test_split_flow_fix_pressured_medium_population_is_split_flow() -> None:
    p = profile_of(SPLIT_HIGH)
    assert p.population == 21 and p.split_pressure is not None and p.split_pressure > 1.0
    assert is_split_flow(p) and F.SPLIT_FLOW_ACROSS_EXITS in structural_tags(p)
    assert F.CROWD_100 not in structural_tags(p)


def test_deceptive_exit_detection() -> None:
    p = profile_of(DECEPTIVE)
    assert p.deceptive_share >= 0.25 and is_deceptive(p)
    q = profile_of(DECEPTIVE_OPEN)
    assert q.deceptive_share == 0.0 and not is_deceptive(q)


def test_narrow_corridor_detection_is_not_every_door() -> None:
    assert is_narrow_corridor(profile_of(NARROW))
    wide = profile_of(WIDE_DOOR)
    assert wide.passage_width > 1 and not is_narrow_corridor(wide)
    few = profile_of(DOOR_FEW)
    assert few.passage_width == 1 and few.bottleneck_severity < 4.0 and not is_narrow_corridor(few)


def test_deterministic_fire_tags_come_from_the_one_trajectory() -> None:
    s = layout(DECEPTIVE)
    assert s.fire.model is FireModelName.DETERMINISTIC
    assert fire_evidence(s, 0) == fire_evidence(s, 1) == fire_evidence(s, 2**63)


def test_stochastic_fire_evidence_depends_on_the_world() -> None:
    s = layout(
        SPLIT_HIGH, termination=TerminationConfig(max_steps=8), fire=FireConfig(model=FireModelName.STOCHASTIC, spread_interval=1, spread_probability=0.5)
    )
    values = {fire_evidence(s, seed) for seed in range(12)}
    assert len(values) > 1


def test_generator_labels_are_structural_only() -> None:
    from test_generator import BASE_CONFIG

    generated = generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), 4, batch_id="families-check")
    for scenario, entry in zip(generated.scenarios, generated.generation.entries, strict=True):
        primary, families = describe_families(scenario, entry.proxies)
        assert set(families) == structural_tags(profile_structure(scenario))
        assert not set(families) & FIRE_DEPENDENT_FAMILIES
        assert primary in families
