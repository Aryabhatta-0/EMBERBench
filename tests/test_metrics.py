"""Per-episode metrics: hand-computed fixtures, fail-closed input checks and reconciliation.

Every expected value below is worked out by hand from the layout and the
scripted actions. The comments show the arithmetic.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence

import pytest

from conftest import FIXTURE_NAMES, fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    METRICS_VERSION,
    CasualtyCause,
    FireConfig,
    MetricsInputError,
    MoveOutcome,
    PersonStatus,
    Pos,
    RunRecord,
    Scenario,
    Simulator,
    TerminationConfig,
    TerminationReason,
    compute_episode_metrics,
    replay,
    run_episode,
)
from emberbench.metrics import (
    ExitUtilization,
    FireExposureProxy,
    MovementMetrics,
    PathEfficiencyMetrics,
    PersonMetrics,
    ReplanningProxies,
    SafetyMetrics,
    TimeMetrics,
    static_exit_distances,
)
from emberbench.state import PersonState

NO_CAUSES = (("entered_fire", 0), ("fire_spread", 0))
NO_BLOCKS = tuple((o.value, 0) for o in MoveOutcome if o.value.startswith("blocked_"))
NO_REPLANNING = ReplanningProxies(
    blocked_events=0,
    blocks_with_next_decision=0,
    blocks_without_next_decision=0,
    moved_after_block=0,
    blocked_after_block=0,
    stayed_after_block=0,
    action_changed_after_block=0,
    max_consecutive_blocks=0,
    recovery_proxy_rate=None,
    repeated_block_proxy_rate=None,
    action_change_proxy_rate=None,
)
NO_FIRE_EVER = FireConfig(spread_interval=1000)  # the fire never spreads within max_steps


def blocks(**counts: int) -> tuple[tuple[str, int], ...]:
    return tuple((name, counts.get(name.removeprefix("blocked_"), 0)) for name, _ in NO_BLOCKS)


def run(scenario: Scenario, script: Sequence[Mapping[str, str]]) -> RunRecord:
    return replay(scenario, script)


# --------------------------------------------------------------------------- all evacuated, no fire


ALL_EVACUATED = ["a..E", "b..."]
ALL_EVACUATED_SCRIPT = [
    {"Pa": "east", "Pb": "east"},
    {"Pa": "east", "Pb": "east"},
    {"Pa": "east", "Pb": "east"},  # Pa reaches E1 (0,3) at step 3
    {"Pb": "north"},  # Pb reaches E1 at step 4
]


def test_all_evacuated_no_fire_every_section() -> None:
    scenario = layout(ALL_EVACUATED)
    m = compute_episode_metrics(scenario, run(scenario, ALL_EVACUATED_SCRIPT))

    assert m.metrics_version == METRICS_VERSION
    assert m.scenario_id == "test"
    assert m.scenario_hash == scenario.content_hash()
    assert m.seed == 0
    assert m.termination_reason is TerminationReason.ALL_RESOLVED
    assert m.safety == SafetyMetrics(
        population=2,
        evacuated=2,
        casualties=0,
        stranded=0,
        evacuated_rate=1.0,
        casualty_rate=0.0,
        stranded_rate=0.0,
        terminal_trapped_count=0,
        terminal_trapped_rate=0.0,
        casualties_by_cause=NO_CAUSES,
    )
    assert m.time == TimeMetrics(
        episode_steps=4,
        max_steps=100,
        evacuees=2,
        non_evacuees=0,
        evacuation_steps=(("Pa", 3), ("Pb", 4)),
        first_evacuation_step=3,
        last_evacuation_step=4,
        mean_evacuation_step=3.5,
        median_evacuation_step=3.5,
        person_time_to_safety_total=7,  # 3 + 4
        mean_person_time_to_safety=3.5,  # 7 / 2
    )
    assert m.movement == MovementMetrics(
        decisions=7,  # 2 + 2 + 2 + 1 active people per step
        stay_actions=0,
        movement_attempts=7,
        successful_moves=7,
        blocked_moves=0,
        blocked_by_outcome=NO_BLOCKS,
        congestion_blocks=0,
        stay_rate=0.0,
        blocked_rate=0.0,
        congestion_block_rate=0.0,
    )
    assert m.path_efficiency == PathEfficiencyMetrics(
        evacuees_measured=2,
        total_cells_moved=7,
        total_static_lower_bound=7,  # Pa 3, Pb 4
        aggregate_path_efficiency=1.0,
        mean_path_efficiency=1.0,
        mean_detour_ratio=1.0,
    )
    # No fire: s_0..s_4 hold 2, 2, 2, 1, 0 active people.
    assert m.fire_exposure == FireExposureProxy(
        active_person_states=7,
        person_states_with_fire=0,
        person_states_without_fire=7,
        distance_histogram=(),
        adjacent_person_states=0,
        min_fire_distance=None,
        exposure_total=0.0,
        mean_exposure=0.0,
    )
    assert m.exits == ExitUtilization(
        exit_count=1,
        exits_used=1,
        evacuations_per_exit=(("E1", 2),),
        exit_shares=(("E1", 1.0),),
        max_exit_share=1.0,
        exit_balance=None,  # undefined for a single exit
    )
    assert m.replanning == NO_REPLANNING
    assert m.people[0] == PersonMetrics(
        person_id="Pa",
        status=PersonStatus.EVACUATED,
        initial_position=Pos(0, 0),
        final_position=Pos(0, 3),
        resolution_step=3,
        exit_id="E1",
        casualty_cause=None,
        decisions=3,
        stay_actions=0,
        cells_moved=3,
        blocked_moves=0,
        max_consecutive_blocks=0,
        time_to_safety=3,
        static_exit_distance=3,
        path_efficiency=1.0,
        detour_ratio=1.0,
        min_fire_distance=None,
        fire_exposure=0.0,
    )
    assert [p.person_id for p in m.people] == ["Pa", "Pb"]
    assert m.people[1].static_exit_distance == 4
    assert m.people[1].time_to_safety == 4


# --------------------------------------------------------------------------- casualties


def test_casualty_by_fire_spread() -> None:
    # E1 (0,0), Pa (0,1), Pb (0,2), fire (0,4). Step 1: Pa evacuates; Pb steps
    # east onto (0,3), and the fire spreads onto (0,3) in the same step.
    scenario = layout(["Eab.F"])
    m = compute_episode_metrics(scenario, run(scenario, [{"Pa": "west", "Pb": "east"}]))

    assert m.termination_reason is TerminationReason.ALL_RESOLVED
    assert m.safety == SafetyMetrics(
        population=2,
        evacuated=1,
        casualties=1,
        stranded=0,
        evacuated_rate=0.5,
        casualty_rate=0.5,
        stranded_rate=0.0,
        terminal_trapped_count=0,
        terminal_trapped_rate=0.0,
        casualties_by_cause=(("entered_fire", 0), ("fire_spread", 1)),
    )
    assert m.time == TimeMetrics(
        episode_steps=1,
        max_steps=100,
        evacuees=1,
        non_evacuees=1,
        evacuation_steps=(("Pa", 1),),
        first_evacuation_step=1,
        last_evacuation_step=1,
        mean_evacuation_step=1.0,
        median_evacuation_step=1.0,
        person_time_to_safety_total=101,  # Pa 1 + Pb censored at max_steps 100
        mean_person_time_to_safety=50.5,
    )
    # Only s_0 has active people: Pa at distance 3, Pb at distance 2.
    assert m.fire_exposure == FireExposureProxy(
        active_person_states=2,
        person_states_with_fire=2,
        person_states_without_fire=0,
        distance_histogram=((2, 1), (3, 1)),
        adjacent_person_states=0,
        min_fire_distance=2,
        exposure_total=5 / 6,  # 1/3 + 1/2
        mean_exposure=5 / 12,  # (5/6) / 2
    )
    assert m.people[1] == PersonMetrics(
        person_id="Pb",
        status=PersonStatus.CASUALTY,
        initial_position=Pos(0, 2),
        final_position=Pos(0, 3),
        resolution_step=1,
        exit_id=None,
        casualty_cause=CasualtyCause.FIRE_SPREAD,
        decisions=1,
        stay_actions=0,
        cells_moved=1,
        blocked_moves=0,
        max_consecutive_blocks=0,
        time_to_safety=100,
        static_exit_distance=2,
        path_efficiency=None,  # evacuees only
        detour_ratio=None,
        min_fire_distance=2,
        fire_exposure=0.5,
    )
    # A casualty never enters the path-efficiency aggregate.
    assert m.path_efficiency == PathEfficiencyMetrics(
        evacuees_measured=1,
        total_cells_moved=1,
        total_static_lower_bound=1,
        aggregate_path_efficiency=1.0,
        mean_path_efficiency=1.0,
        mean_detour_ratio=1.0,
    )


def test_casualty_by_entering_fire() -> None:
    scenario = layout(["aF.E"])  # enter_fire defaults to "casualty"
    m = compute_episode_metrics(scenario, run(scenario, [{"Pa": "east"}]))
    assert m.safety.casualties == 1
    assert m.safety.casualty_rate == 1.0
    assert m.safety.casualties_by_cause == (("entered_fire", 1), ("fire_spread", 0))
    person = m.people[0]
    assert person.casualty_cause is CasualtyCause.ENTERED_FIRE
    assert person.cells_moved == 1  # the move into fire succeeds, then the person is lost
    assert person.min_fire_distance == 1
    assert person.fire_exposure == 1.0
    assert m.time.person_time_to_safety_total == 100
    assert m.time.mean_evacuation_step is None


# --------------------------------------------------------------------------- stranded, trapped, zero evacuation


def test_stranded_at_max_steps_with_trapped_person_and_zero_evacuations() -> None:
    # Pa (0,0) is walled in: trapped. Pb (1,1) can reach E1 (0,2) in 2 moves but stays.
    scenario = layout(["a#E", "#b."], termination=TerminationConfig(max_steps=2))
    script = [{"Pa": "east", "Pb": "stay"}] * 2
    m = compute_episode_metrics(scenario, run(scenario, script))

    assert m.termination_reason is TerminationReason.MAX_STEPS
    assert m.safety == SafetyMetrics(
        population=2,
        evacuated=0,
        casualties=0,
        stranded=2,
        evacuated_rate=0.0,
        casualty_rate=0.0,
        stranded_rate=1.0,
        terminal_trapped_count=1,
        terminal_trapped_rate=0.5,
        casualties_by_cause=NO_CAUSES,
    )
    assert m.time == TimeMetrics(
        episode_steps=2,
        max_steps=2,
        evacuees=0,
        non_evacuees=2,
        evacuation_steps=(),
        first_evacuation_step=None,
        last_evacuation_step=None,
        mean_evacuation_step=None,
        median_evacuation_step=None,
        person_time_to_safety_total=4,  # both censored at max_steps 2
        mean_person_time_to_safety=2.0,
    )
    assert m.movement == MovementMetrics(
        decisions=4,
        stay_actions=2,
        movement_attempts=2,
        successful_moves=0,
        blocked_moves=2,
        blocked_by_outcome=blocks(wall=2),
        congestion_blocks=0,
        stay_rate=0.5,
        blocked_rate=1.0,
        congestion_block_rate=0.0,
    )
    assert m.path_efficiency == PathEfficiencyMetrics(
        evacuees_measured=0,
        total_cells_moved=0,
        total_static_lower_bound=0,
        aggregate_path_efficiency=None,
        mean_path_efficiency=None,
        mean_detour_ratio=None,
    )
    assert m.exits == ExitUtilization(
        exit_count=1,
        exits_used=0,
        evacuations_per_exit=(("E1", 0),),
        exit_shares=None,
        max_exit_share=None,
        exit_balance=None,
    )
    # Pa is blocked at steps 1 and 2. Only the step-1 block has a next decision.
    assert m.replanning == ReplanningProxies(
        blocked_events=2,
        blocks_with_next_decision=1,
        blocks_without_next_decision=1,
        moved_after_block=0,
        blocked_after_block=1,
        stayed_after_block=0,
        action_changed_after_block=0,
        max_consecutive_blocks=2,
        recovery_proxy_rate=0.0,
        repeated_block_proxy_rate=1.0,
        action_change_proxy_rate=0.0,
    )
    assert m.fire_exposure.active_person_states == 6  # s_0, s_1, s_2 with 2 active each
    assert m.fire_exposure.person_states_without_fire == 6
    pa, pb = m.people
    assert pa.static_exit_distance is None
    assert pa.resolution_step is None
    assert pa.time_to_safety == 2
    assert pa.status is PersonStatus.ACTIVE
    assert pb.static_exit_distance == 2


# --------------------------------------------------------------------------- congestion


def test_bottleneck_conflict() -> None:
    # Pa (0,0) and Pb (0,2) both step into (0,1) above E1 (1,1). Pa < Pb wins.
    scenario = layout(["a.b", "#E#"])
    script = [{"Pa": "east", "Pb": "west"}, {"Pa": "south", "Pb": "west"}, {"Pb": "south"}]
    m = compute_episode_metrics(scenario, run(scenario, script))

    assert m.movement == MovementMetrics(
        decisions=5,
        stay_actions=0,
        movement_attempts=5,
        successful_moves=4,
        blocked_moves=1,
        blocked_by_outcome=blocks(conflict=1),
        congestion_blocks=1,
        stay_rate=0.0,
        blocked_rate=0.2,
        congestion_block_rate=0.2,
    )
    assert m.replanning == ReplanningProxies(
        blocked_events=1,
        blocks_with_next_decision=1,
        blocks_without_next_decision=0,
        moved_after_block=1,  # Pb repeats "west" at step 2 and moves
        blocked_after_block=0,
        stayed_after_block=0,
        action_changed_after_block=0,
        max_consecutive_blocks=1,
        recovery_proxy_rate=1.0,
        repeated_block_proxy_rate=0.0,
        action_change_proxy_rate=0.0,
    )
    assert m.time.evacuation_steps == (("Pa", 2), ("Pb", 3))
    assert m.time.median_evacuation_step == 2.5
    assert m.path_efficiency.aggregate_path_efficiency == 1.0  # 4 / 4: the blocked step costs time, not distance


def test_swap_cycle_and_occupied_blocks() -> None:
    # E1 (0,0), Pa (0,1), Pb (0,2).
    scenario = layout(["Eab"])
    script = [
        {"Pa": "east", "Pb": "west"},  # swap: both blocked_cycle
        {"Pa": "stay", "Pb": "west"},  # Pb: blocked_occupied by the staying Pa
        {"Pa": "west", "Pb": "west"},  # Pa evacuates, Pb follows into (0,1)
        {"Pb": "west"},  # Pb evacuates
    ]
    m = compute_episode_metrics(scenario, run(scenario, script))

    assert m.movement == MovementMetrics(
        decisions=7,
        stay_actions=1,
        movement_attempts=6,
        successful_moves=3,
        blocked_moves=3,
        blocked_by_outcome=blocks(occupied=1, cycle=2),
        congestion_blocks=3,
        stay_rate=1 / 7,
        blocked_rate=0.5,
        congestion_block_rate=0.5,
    )
    # Pa@1 -> stayed (action changed); Pb@1 -> blocked again; Pb@2 -> moved.
    assert m.replanning == ReplanningProxies(
        blocked_events=3,
        blocks_with_next_decision=3,
        blocks_without_next_decision=0,
        moved_after_block=1,
        blocked_after_block=1,
        stayed_after_block=1,
        action_changed_after_block=1,
        max_consecutive_blocks=2,
        recovery_proxy_rate=1 / 3,
        repeated_block_proxy_rate=1 / 3,
        action_change_proxy_rate=1 / 3,
    )
    pa, pb = m.people
    assert (pa.decisions, pa.stay_actions, pa.cells_moved, pa.blocked_moves, pa.max_consecutive_blocks) == (3, 1, 1, 1, 1)
    assert (pb.decisions, pb.stay_actions, pb.cells_moved, pb.blocked_moves, pb.max_consecutive_blocks) == (4, 0, 2, 2, 2)
    assert m.time.evacuation_steps == (("Pa", 3), ("Pb", 4))


# --------------------------------------------------------------------------- multiple exits


def test_multiple_exits_shares_and_balance() -> None:
    # E1 (0,0), Pa (0,1), Pb (0,3), Pc (0,5), E2 (0,6).
    scenario = layout(["Ea.b.cE"])
    script = [{"Pa": "west", "Pb": "west", "Pc": "east"}, {"Pb": "west"}, {"Pb": "west"}]
    m = compute_episode_metrics(scenario, run(scenario, script))

    # Shares 2/3 and 1/3: 1 - (4/9 + 1/9) = 4/9, normalized by 1 - 1/2 gives 8/9.
    assert m.exits == ExitUtilization(
        exit_count=2,
        exits_used=2,
        evacuations_per_exit=(("E1", 2), ("E2", 1)),
        exit_shares=(("E1", 2 / 3), ("E2", 1 / 3)),
        max_exit_share=2 / 3,
        exit_balance=8 / 9,
    )
    assert m.time == TimeMetrics(
        episode_steps=3,
        max_steps=100,
        evacuees=3,
        non_evacuees=0,
        evacuation_steps=(("Pa", 1), ("Pb", 3), ("Pc", 1)),
        first_evacuation_step=1,
        last_evacuation_step=3,
        mean_evacuation_step=5 / 3,
        median_evacuation_step=1.0,
        person_time_to_safety_total=5,
        mean_person_time_to_safety=5 / 3,
    )
    assert m.people[1].static_exit_distance == 3  # equidistant from both exits
    assert m.people[1].exit_id == "E1"


@pytest.mark.parametrize(
    ("rows", "script", "per_exit", "shares", "used", "balance"),
    [
        (["Ea..bE"], [{"Pa": "west", "Pb": "east"}], (("E1", 1), ("E2", 1)), (("E1", 0.5), ("E2", 0.5)), 2, 1.0),
        (
            ["Eab..E"],
            [{"Pa": "west", "Pb": "west"}, {"Pb": "west"}],
            (("E1", 2), ("E2", 0)),
            (("E1", 1.0), ("E2", 0.0)),
            1,
            0.0,
        ),
    ],
)
def test_exit_balance_extremes(
    rows: list[str],
    script: list[dict[str, str]],
    per_exit: tuple[tuple[str, int], ...],
    shares: tuple[tuple[str, float], ...],
    used: int,
    balance: float,
) -> None:
    scenario = layout(rows)
    m = compute_episode_metrics(scenario, run(scenario, script))
    assert m.exits.evacuations_per_exit == per_exit
    assert m.exits.exit_shares == shares
    assert m.exits.exits_used == used
    assert m.exits.exit_balance == balance


# --------------------------------------------------------------------------- detours


def test_detour_around_fire_lowers_path_efficiency() -> None:
    # Pa (0,0), fire (0,1), E1 (0,2). The static bound ignores fire: 2 cells.
    # Pa walks around the fire through row 1: 4 cells.
    scenario = layout(["aFE", "..."], fire=NO_FIRE_EVER)
    script = [{"Pa": "south"}, {"Pa": "east"}, {"Pa": "east"}, {"Pa": "north"}]
    m = compute_episode_metrics(scenario, run(scenario, script))

    assert m.path_efficiency == PathEfficiencyMetrics(
        evacuees_measured=1,
        total_cells_moved=4,
        total_static_lower_bound=2,
        aggregate_path_efficiency=0.5,
        mean_path_efficiency=0.5,
        mean_detour_ratio=2.0,
    )
    # Distances from (0,1): s_0 (0,0)=1, s_1 (1,0)=2, s_2 (1,1)=1, s_3 (1,2)=2, s_4 none active.
    assert m.fire_exposure == FireExposureProxy(
        active_person_states=4,
        person_states_with_fire=4,
        person_states_without_fire=0,
        distance_histogram=((1, 2), (2, 2)),
        adjacent_person_states=2,
        min_fire_distance=1,
        exposure_total=3.0,  # 1 + 1/2 + 1 + 1/2
        mean_exposure=0.75,
    )
    assert m.people[0].detour_ratio == 2.0
    assert m.people[0].path_efficiency == 0.5


def test_detour_around_walls_is_in_the_static_bound() -> None:
    # The wall at (0,1) makes the static bound 4, not the Manhattan distance 2.
    scenario = layout(["a#E", "..."])
    script = [{"Pa": "south"}, {"Pa": "east"}, {"Pa": "east"}, {"Pa": "north"}]
    m = compute_episode_metrics(scenario, run(scenario, script))
    assert m.people[0].static_exit_distance == 4
    assert m.people[0].cells_moved == 4
    assert m.path_efficiency.aggregate_path_efficiency == 1.0


def test_static_exit_distances_ignore_fire_and_people() -> None:
    scenario = layout(["aF.E", "#b#."])
    dist = static_exit_distances(scenario)
    assert dist[Pos(0, 0)] == 3  # straight through the fire cell
    assert dist[Pos(1, 1)] == 3  # (0,1) is burning and (0,0) holds Pa; both are ignored
    assert Pos(1, 0) not in dist  # a wall


# --------------------------------------------------------------------------- fail closed


def _complete_record() -> tuple[Scenario, RunRecord]:
    scenario = layout(ALL_EVACUATED)
    return scenario, run(scenario, ALL_EVACUATED_SCRIPT)


def test_rejects_scenario_hash_mismatch() -> None:
    _, record = _complete_record()
    other = layout(ALL_EVACUATED, termination=TerminationConfig(max_steps=99))
    with pytest.raises(MetricsInputError, match="scenario hash"):
        compute_episode_metrics(other, record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("simulator_version", "0.1.0", "simulator"),
        ("observation_version", "0.9", "observation version"),
        ("schema_version", "2.0", "schema version"),
        ("scenario_id", "someone_else", "scenario_id"),
        ("scenario_hash", "0" * 64, "scenario hash"),
    ],
)
def test_rejects_incompatible_header(field: str, value: str, message: str) -> None:
    scenario, record = _complete_record()
    with pytest.raises(MetricsInputError, match=message):
        compute_episode_metrics(scenario, dataclasses.replace(record, **{field: value}))  # type: ignore[arg-type]


def test_rejects_unterminated_and_empty_records() -> None:
    scenario, _ = _complete_record()
    with pytest.raises(MetricsInputError, match="termination"):
        compute_episode_metrics(scenario, run(scenario, ALL_EVACUATED_SCRIPT[:2]))
    with pytest.raises(MetricsInputError, match="no steps"):
        compute_episode_metrics(scenario, Simulator(scenario).record())


def test_rejects_wrong_argument_types() -> None:
    scenario, record = _complete_record()
    with pytest.raises(TypeError):
        compute_episode_metrics(record, record)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_episode_metrics(scenario, record.to_dict())  # type: ignore[arg-type]


def _tamper_step(index: int, **changes: object) -> Callable[[RunRecord], RunRecord]:
    def apply(record: RunRecord) -> RunRecord:
        steps = list(record.steps)
        steps[index] = dataclasses.replace(steps[index], **changes)  # type: ignore[arg-type]
        return dataclasses.replace(record, steps=tuple(steps))

    return apply


def _drop_evacuation(record: RunRecord) -> RunRecord:
    return _tamper_step(2, evacuations=())(record)


def _flip_outcome(record: RunRecord) -> RunRecord:
    step = record.steps[0]
    moves = (dataclasses.replace(step.movements[0], outcome=MoveOutcome.BLOCKED_WALL),) + step.movements[1:]
    return _tamper_step(0, movements=moves)(record)


def _resurrect(record: RunRecord) -> RunRecord:
    last = record.steps[-1]
    people = tuple(PersonState(p.id, p.position, PersonStatus.ACTIVE) for p in last.state.people)
    return _tamper_step(len(record.steps) - 1, state=dataclasses.replace(last.state, people=people))(record)


def _add_fire(record: RunRecord) -> RunRecord:
    last = record.steps[-1]
    burning = last.state.burning | {Pos(1, 0)}
    return _tamper_step(len(record.steps) - 1, state=dataclasses.replace(last.state, burning=burning))(record)


def _renumber(record: RunRecord) -> RunRecord:
    return _tamper_step(1, step=7)(record)


def _extra_step(record: RunRecord) -> RunRecord:
    last = record.steps[-1]
    extra = dataclasses.replace(last, step=last.step + 1, state=dataclasses.replace(last.state, step=last.step + 1))
    return dataclasses.replace(record, steps=record.steps + (extra,))


def _wrong_initial_state(record: RunRecord) -> RunRecord:
    return dataclasses.replace(record, initial_state=dataclasses.replace(record.initial_state, burning=frozenset({Pos(1, 3)})))


def _drop_action(record: RunRecord) -> RunRecord:
    return _tamper_step(0, actions=record.steps[0].actions[:1])(record)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_drop_evacuation, "state of Pa does not match its events"),
        (_flip_outcome, "final cell of Pa is inconsistent"),
        (_resurrect, "resolved person Pa changed"),
        (_add_fire, "burning cells do not equal"),
        (_renumber, "not consecutive"),
        (_extra_step, "past termination"),
        (_wrong_initial_state, "initial burning"),
        (_drop_action, "actions do not cover"),
    ],
)
def test_rejects_malformed_records(tamper: Callable[[RunRecord], RunRecord], message: str) -> None:
    scenario, record = _complete_record()
    with pytest.raises(MetricsInputError, match=message):
        compute_episode_metrics(scenario, tamper(record))


# --------------------------------------------------------------------------- purity and reconciliation


def test_inputs_are_not_mutated_and_results_are_deterministic() -> None:
    scenario, record = _complete_record()
    scenario_json, record_json = scenario.to_json(), record.to_json()
    first = compute_episode_metrics(scenario, record)
    second = compute_episode_metrics(scenario, RunRecord.from_json(record_json))
    assert scenario.to_json() == scenario_json
    assert record.to_json() == record_json
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.safety.evacuated = 0  # type: ignore[misc]


def test_no_composite_score_field() -> None:
    names = {f.name for f in dataclasses.fields(compute_episode_metrics(*_complete_record()))}
    assert not {"score", "ember_score", "composite"} & names


def _reconcile(scenario: Scenario, record: RunRecord) -> None:
    m = compute_episode_metrics(scenario, record)
    s, mv = m.safety, m.movement
    assert s.population == len(scenario.people)
    assert s.evacuated + s.casualties + s.stranded == s.population
    assert s.terminal_trapped_count <= s.stranded
    assert sum(n for _, n in s.casualties_by_cause) == s.casualties
    assert mv.decisions == sum(len(step.movements) for step in record.steps)
    assert mv.stay_actions + mv.successful_moves + mv.blocked_moves == mv.decisions
    assert sum(n for _, n in mv.blocked_by_outcome) == mv.blocked_moves
    assert m.time.evacuees == s.evacuated
    assert sum(n for _, n in m.exits.evacuations_per_exit) == s.evacuated
    assert m.time.person_time_to_safety_total == sum(p.time_to_safety for p in m.people)
    assert m.replanning.blocked_events == mv.blocked_moves
    fx = m.fire_exposure
    assert fx.person_states_with_fire + fx.person_states_without_fire == fx.active_person_states
    assert sum(n for _, n in fx.distance_histogram) == fx.person_states_with_fire
    assert fx.active_person_states == sum(len(st.active) for st in (record.initial_state, *(x.state for x in record.steps)))
    for p in m.people:
        if p.path_efficiency is not None:
            assert 0 < p.path_efficiency <= 1
    # The metrics serialise to JSON (no NaN or infinity) and round-trip records give equal metrics.
    from emberbench.canonical import canonical_json

    canonical_json(m.to_dict())
    assert compute_episode_metrics(scenario, RunRecord.from_json(record.to_json())) == m


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_metrics_reconcile(name: str) -> None:
    scenario = fixture_scenario(name)
    _reconcile(scenario, run_episode(scenario, greedy_exit_policy))
    _reconcile(scenario, run_episode(scenario, lambda obs: {p.id: "stay" for p in obs.active}))


def test_fixture_one_person_one_exit_exact() -> None:
    # Layout "#P...E#": the greedy policy walks 4 cells east and evacuates at step 4. No fire.
    scenario = fixture_scenario("one_person_one_exit")
    m = compute_episode_metrics(scenario, run_episode(scenario, greedy_exit_policy))
    person = m.people[0]
    assert (person.resolution_step, person.cells_moved, person.static_exit_distance) == (4, 4, 4)
    assert m.safety.evacuated_rate == 1.0
    assert m.time.person_time_to_safety_total == 4
    assert m.path_efficiency.aggregate_path_efficiency == 1.0
    assert m.fire_exposure.person_states_without_fire == 4  # s_0..s_3; s_4 has nobody active
    assert m.fire_exposure.exposure_total == 0.0
