"""Tests for the EMBER Score (``emberbench.scoring``): hand-calculated values, penalties, ceilings and aggregation."""

import ast
import dataclasses
import itertools
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import greedy_exit_policy, layout
from emberbench import (
    EVALUATION_RESULT_VERSION,
    OFFICIAL_CONSTANTS,
    SCORING_VERSION,
    Action,
    ComparisonOutcome,
    CongestionDiagnostics,
    EpisodeEvidence,
    EpisodeResult,
    EpisodeStatus,
    EvaluationManifest,
    EvaluationResult,
    FailureKind,
    Observation,
    OperationalRecord,
    Scenario,
    ScoringInputError,
    Suite,
    TerminationConfig,
    agent_policy,
    baseline_factory,
    build_evaluation_manifest,
    compare_scores,
    compute_episode_metrics,
    efficiency_view,
    episode_evidence,
    format_summary,
    inference_config_digest,
    leaderboard_order,
    load_suite,
    run_episode,
    score_episode,
    score_result,
    score_results,
)
from emberbench.canonical import canonical_json
from emberbench.movement import MoveOutcome, resolve_movements
from emberbench.oracle import privileged_escape_bound
from emberbench.simulator import Policy
from emberbench.scoring import (
    HAZARD_RADIUS,
    QUALITY_COMPONENTS,
    SAFETY_COMPONENTS,
    ScoringConstants,
    _references,
    failed_episode,
    quality_components,
    safety_components,
    score_evidence,
    severe_failures,
)

SUITE_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "public" / "v1"
# pub-08 belongs to two families (primary maze_obstacle_routing); pub-10 has stochastic fire.
IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-08-maze-deceptive-exit", "v1-pub-10-route-invalidation")
SEEDS = (11, 7, 3, 5, 9)
RESAMPLES = 200
F = Fraction
BETA = float(OFFICIAL_CONSTANTS.quality_exponent)  # ln 0.6 / ln 0.5


# --------------------------------------------------------------------------- helpers


def evidence(
    population: int = 100,
    *,
    max_evacuated: int | None = None,
    casualties: int = 0,
    stranded: int = 0,
    unavoidable_casualties: int | None = None,
    time_bound: int = 10,
    free_bound: int | None = None,
    time: int | None = None,
    attempts: int = 1000,
    blocks: int = 0,
    moves_bound: int = 10,
    moves: int | None = None,
    danger: int = 0,
    danger_bound: int = 0,
    replan: int = 0,
    replan_bound: int = 0,
) -> EpisodeEvidence:
    """Hand-made evidence. Per-evacuee values are multiplied by the number of evacuees.

    ``population - max_evacuated`` people are unsavable; by default the first
    ``min(unsavable, casualties)`` casualties are unsavable and the rest of the
    unsavable people are stranded. ``time_bound`` is the cut-capacity bound
    (the Time reference); ``free_bound`` (default: the same) is the free-flow
    and exit-capacity bound below it.
    """
    evac = population - casualties - stranded
    unsavable = 0 if max_evacuated is None else population - max_evacuated
    u_cas = min(unsavable, casualties) if unavoidable_casualties is None else unavoidable_casualties
    dead = [f"c{i:04d}" for i in range(casualties)]
    inside = [f"s{i:04d}" for i in range(stranded)]
    return EpisodeEvidence(
        scenario_id="synthetic",
        scenario_hash="0" * 64,
        seed=0,
        record_digest="0" * 64,
        population=population,
        max_evacuated=population if max_evacuated is None else max_evacuated,
        evacuated=evac,
        casualties=casualties,
        stranded=stranded,
        avoidable_casualty_ids=tuple(dead[u_cas:]),
        avoidable_stranded_ids=tuple(inside[unsavable - u_cas :]),
        unavoidable_casualty_ids=tuple(dead[:u_cas]),
        unavoidable_stranded_ids=tuple(inside[: unsavable - u_cas]),
        evacuee_time=evac * (time_bound if time is None else time),
        evacuee_time_bound=evac * (time_bound if free_bound is None else free_bound),
        evacuee_time_queue_bound=evac * (time_bound if free_bound is None else free_bound),
        evacuee_time_cut_bound=evac * time_bound,
        movement_attempts=attempts,
        blocked_occupied=blocks,
        blocked_conflict=0,
        blocked_cycle=0,
        path_cells_moved=evac * (moves_bound if moves is None else moves),
        path_moves_bound=evac * moves_bound,
        path_static_bound=evac * moves_bound,
        evacuee_danger_states=evac * danger,
        evacuee_danger_bound=evac * danger_bound,
        invalidation_events=1 if replan else 0,
        unrecoverable_invalidations=0,
        replanning_remaining=replan,
        replanning_remaining_bound=replan_bound,
    )


def points(ev: EpisodeEvidence, constants: ScoringConstants = OFFICIAL_CONSTANTS) -> float:
    return float(score_evidence(0, ev, constants).score)


def close(actual: Fraction | float | None, expected: float, tol: float = 1e-9) -> bool:
    return actual is not None and math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol)


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def delayed_policy(obs: Observation) -> dict[str, Action]:
    return stay_policy(obs) if obs.step < 3 else greedy_exit_policy(obs)


def subset(suite: Suite, ids: tuple[str, ...]) -> Suite:
    entries = tuple(e for e in suite.manifest.entries if e.scenario_id in ids)
    return Suite(suite.root, dataclasses.replace(suite.manifest, entries=entries), tuple(s for s in suite.scenarios if s.scenario_id in ids))


def run_result(suite: Suite, manifest: EvaluationManifest, model_id: str, policy: Policy, skip: frozenset[int] = frozenset()) -> EvaluationResult:
    episodes = []
    for spec in manifest.episodes:
        if spec.index in skip:
            continue
        scenario = suite.get(spec.scenario_id)
        record = run_episode(scenario, policy, seed=spec.seed)
        episodes.append(EpisodeResult(spec, record, compute_episode_metrics(scenario, record)))
    return EvaluationResult(EVALUATION_RESULT_VERSION, model_id, manifest, tuple(episodes))


@pytest.fixture(scope="module")
def public() -> Suite:
    return load_suite(SUITE_DIR)


@pytest.fixture(scope="module")
def small(public: Suite) -> Suite:
    return subset(public, IDS)


@pytest.fixture(scope="module")
def manifest(small: Suite) -> EvaluationManifest:
    return build_evaluation_manifest(small, SEEDS)


@pytest.fixture(scope="module")
def greedy(small: Suite, manifest: EvaluationManifest) -> EvaluationResult:
    return run_result(small, manifest, "greedy", greedy_exit_policy)


@pytest.fixture(scope="module")
def delayed(small: Suite, manifest: EvaluationManifest) -> EvaluationResult:
    return run_result(small, manifest, "delayed", delayed_policy)


@pytest.fixture(scope="module")
def stay(small: Suite, manifest: EvaluationManifest) -> EvaluationResult:
    return run_result(small, manifest, "stay", stay_policy)


# --------------------------------------------------------------------------- constants and version


def test_version_and_official_constants() -> None:
    assert SCORING_VERSION == "1.1"
    c = OFFICIAL_CONSTANTS
    assert c.weights() == {"time": F(3, 8), "hazard": F(1, 4), "path": F(1, 8), "replanning": F(1, 4)}
    assert QUALITY_COMPONENTS == ("time", "hazard", "path", "replanning")  # no action-log component
    assert str(c.casualty_incident_factor) == "0.6" and str(c.casualty_rate_exponent) == "4"
    assert str(c.stranding_incident_factor) == "0.8" and str(c.stranding_rate_exponent) == "2"
    assert str(c.evacuation_exponent) == "2" and str(c.hazard_exponent) == "1"
    assert c.catastrophic_loss_rate == F(1, 2) and c.practical_difference == F(1)
    assert HAZARD_RADIUS == 2
    assert c.severe_time == F(1, 2) and c.severe_hazard_rate == F(1, 4)
    assert c.severe_path == F(1, 2) and c.severe_replanning == F(1, 2)
    assert str(c.severe_factor) == "0.8" and str(c.severe_exponent) == "1"
    thresholds = {k: float(v) for k, v in c.severe_thresholds().items()}
    assert thresholds == {"time": 0.5, "hazard": 0.75, "path": 0.5, "replanning": 0.5}
    for removed in ("weight_coordination", "congestion_exponent", "severe_block_rate"):
        assert not hasattr(c, removed)
    # beta is derived, not tuned: the one-casualty factor over the lowest threshold.
    assert math.isclose(BETA, math.log(0.6) / math.log(0.5), rel_tol=1e-15)
    assert c.to_dict()["quality_exponent"].startswith("0.7369655941662")
    assert len(c.digest()) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"weight_time": F(1, 2)},  # weights no longer sum to 1
        {"casualty_incident_factor": Decimal("0")},
        {"stranding_incident_factor": Decimal("1.5")},
        {"catastrophic_loss_rate": F(0)},
        {"evacuation_exponent": Decimal("-1")},
        {"severe_time": F(1)},
        {"severe_hazard_rate": F(1)},
        {"severe_factor": Decimal("0")},
    ],
)
def test_invalid_constants_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(OFFICIAL_CONSTANTS, **change)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- hand-calculated episode scores


def test_perfect_episode_scores_100() -> None:
    s = score_evidence(0, evidence())
    assert s.status is EpisodeStatus.SCORED and s.score == 100
    assert all(v == 1 for name, v in s.components if v is not None)
    assert s.component("replanning") is None  # no invalidation: undefined, not a fabricated 1


def test_time_penalty_hand_calculated() -> None:
    # Time 1/2 is at the threshold, not severe; replanning undefined, so the weights renormalize over 6/8.
    s = score_evidence(0, evidence(time_bound=10, time=20))
    assert s.component("time") == F(1, 2) and s.severe == () and s.severity == 1
    assert close(s.score, 100 * 0.5 ** (1 / 2 * BETA))
    assert 100 - s.score >= 10  # Time is not irrelevant


def test_severe_delay_is_gated() -> None:
    # Time 0.4 is severe: factor 0.8 times (0.4 / 0.5) on top of the quality term.
    s = score_evidence(0, evidence(time_bound=10, time=25))
    assert s.severe == ("time",) and close(s.severity, 0.8 * 0.8, 1e-12)
    assert close(s.score, 100 * 0.4 ** (1 / 2 * BETA) * 0.64)
    assert s.ceiling is not None and close(s.ceiling, 64, 1e-9)
    # Mild delay beats extreme delay by a wide margin; extreme delay keeps almost nothing.
    assert points(evidence(time=12)) - points(evidence(time=100)) > 70
    assert points(evidence(time=100)) < 10


def test_congestion_blocks_are_diagnostics_not_score_inputs() -> None:
    # A blocked move and STAY give the same trajectory, so the block count never changes the score.
    for base in (evidence(), evidence(time=25), evidence(casualties=1, time=13), evidence(stranded=2, danger=3, danger_bound=1)):
        reference = score_evidence(0, base)
        for blocks in (0, 1, 250, 999):
            for attempts in (blocks, 1000, 5000):
                if blocks > attempts:
                    continue
                varied = dataclasses.replace(
                    base,
                    movement_attempts=attempts,
                    blocked_occupied=blocks,
                    contested_cell_steps=3 * blocks,
                    hotspot_cells=blocks // 10,
                    peak_contested_cells=blocks // 20,
                    exit_active_steps=attempts,
                )
                assert dataclasses.replace(score_evidence(0, varied), evidence=base) == reference
        assert not any(name in ("coordination", "congestion") for name, _ in reference.components)


def test_exposure_penalty_without_deaths() -> None:
    # Each evacuee spends 4 of 10 states in the danger zone where 1 was unavoidable: H = 1 - 3/10.
    s = score_evidence(0, evidence(danger=4, danger_bound=1))
    assert s.component("hazard") == F(7, 10) and s.severe == ("hazard",)
    assert s.component("casualty") == 1 and s.safety == 1
    assert close(s.score, 100 * 0.7 ** (BETA / 3) * 0.8 * (0.7 / 0.75))
    assert s.score < 80  # lucky unsafe behaviour is never excellent
    # Mild avoidable exposure costs less, without a gate.
    mild = score_evidence(0, evidence(danger=2, danger_bound=1))
    assert mild.severe == () and close(mild.score, 100 * 0.9 ** (BETA / 3))
    # Unavoidable danger costs nothing.
    assert score_evidence(0, evidence(danger=3, danger_bound=3)).score == 100


def test_path_penalty_hand_calculated() -> None:
    s = score_evidence(0, evidence(moves_bound=10, moves=16))
    assert s.component("path") == F(10, 16) and s.severe == ()
    assert close(s.score, 100 * (10 / 16) ** (BETA / 6))
    # Moves above twice the privileged minimum are severe.
    assert score_evidence(0, evidence(moves_bound=10, moves=25)).severe == ("path",)


def test_path_does_not_charge_necessary_fire_detours() -> None:
    # The path bound is the fire-aware privileged minimum: a forced detour is in the bound, not a penalty.
    ev = evidence(moves_bound=14, moves=14)
    assert quality_components(ev)["path"] == 1 and score_evidence(0, ev).score == 100


def test_replanning_penalty_hand_calculated() -> None:
    # Replanning 1/4 with every weight in use (severe): 100 * 0.25^(beta / 4) * 0.8 * (0.25 / 0.5).
    s = score_evidence(0, evidence(replan=40, replan_bound=10))
    assert s.component("replanning") == F(1, 4) and s.severe == ("replanning",)
    assert close(s.score, 100 * 0.25 ** (BETA / 4) * 0.4)
    # Failure to replan costs on its own, before any loss of life that it causes is charged to Safety.
    assert s.score < 80


def test_one_casualty_penalty_hand_calculated() -> None:
    # 99 of 100 evacuated: E = 0.99^2; casualty = 0.6^log2(2) * 0.99^4 = 0.6 * 0.99^4.
    s = score_evidence(0, evidence(casualties=1))
    assert close(s.component("evacuation"), 0.99**2, 1e-12)
    assert close(s.component("casualty"), 0.6 * 0.99**4, 1e-12)
    assert close(s.score, 100 * 0.6 * 0.99**6)
    assert s.score <= 60  # the incident ceiling
    assert s.safety_ceiling is not None and close(s.safety_ceiling, 100 * 0.6 * 0.99**6, 1e-9)


def test_casualties_are_nonlinear_and_progressive() -> None:
    scores = [points(evidence(casualties=k)) for k in range(0, 11)]
    assert all(a > b for a, b in itertools.pairwise(scores))
    # Each doubling of (1 + count) costs another incident factor on top of the rate term.
    for k in (1, 3, 7):
        p_k = safety_components(evidence(casualties=k))["casualty"]
        p_next = safety_components(evidence(casualties=2 * k + 1))["casualty"]
        assert p_next <= F(6, 10) * p_k
    # A casualty hurts much more than an equivalent success helps: losing 1 of 100 costs > 40 points,
    # while the last person evacuated adds at most 2 points through E.
    assert 100 - scores[1] > 40


def test_casualty_counts_in_small_groups_weigh_more() -> None:
    assert points(evidence(4, casualties=1)) < points(evidence(100, casualties=1))
    assert close(points(evidence(4, casualties=1)), 100 * 0.6 * 0.75**4 * 0.75**2)


def test_stranding_penalized_independently() -> None:
    s = score_evidence(0, evidence(stranded=1))
    assert close(s.component("stranding"), 0.8 * 0.99**2, 1e-12)
    assert s.component("casualty") == 1
    assert close(s.score, 100 * 0.99**2 * 0.8 * 0.99**2)
    # A casualty is worse than a stranding; both are worse than nothing.
    assert points(evidence(casualties=1)) < points(evidence(stranded=1)) < 100


def test_incomplete_evacuation_penalized() -> None:
    # Two people nobody can save are not charged.
    unavoidable = evidence(100, max_evacuated=98, casualties=2)
    assert unavoidable.avoidable_casualties == 0 and unavoidable.unavoidable_losses == 2
    assert score_evidence(0, unavoidable).score == 100
    # A savable person lost on top is charged.
    assert points(evidence(100, max_evacuated=98, casualties=3)) < 60
    # E falls with every savable person not evacuated, independently of the incident factors.
    e = [safety_components(evidence(100, stranded=k))["evacuation"] for k in (0, 1, 5)]
    assert e[0] == 1 and e[0] > e[1] > e[2]


# --------------------------------------------------------------------------- person-level attribution


def test_savable_casualty_with_unsavable_stranded() -> None:
    # P1 savable dies, P2 unsavable is stranded: the casualty is avoidable, the stranding is not.
    ev = evidence(100, max_evacuated=99, casualties=1, stranded=1, unavoidable_casualties=0)
    assert ev.avoidable_casualty_ids == ("c0000",) and ev.unavoidable_stranded_ids == ("s0000",)
    assert (ev.avoidable_casualties, ev.avoidable_stranded, ev.unavoidable_losses) == (1, 0, 1)
    s = score_evidence(0, ev)
    assert s.component("casualty") != 1 and s.component("stranding") == 1
    assert s.score <= 60


def test_unsavable_casualty_with_savable_stranded() -> None:
    ev = evidence(100, max_evacuated=99, casualties=1, stranded=1, unavoidable_casualties=1)
    assert ev.unavoidable_casualty_ids == ("c0000",) and ev.avoidable_stranded_ids == ("s0000",)
    assert (ev.avoidable_casualties, ev.avoidable_stranded) == (0, 1)
    s = score_evidence(0, ev)
    assert s.component("casualty") == 1 and s.component("stranding") != 1
    # One stranded savable person is less severe than one savable casualty.
    assert s.score > points(evidence(100, max_evacuated=99, casualties=1, stranded=1, unavoidable_casualties=0))


def test_mixed_identity_losses_are_never_offset() -> None:
    # 3 unsavable people (2 dead, 1 stranded) and 2 savable casualties + 1 savable stranded.
    ev = evidence(100, max_evacuated=97, casualties=4, stranded=2, unavoidable_casualties=2)
    assert (ev.avoidable_casualties, ev.avoidable_stranded, ev.unavoidable_losses) == (2, 1, 3)
    # Exactly the score of the same avoidable losses with no unsavable people present.
    same = evidence(97, casualties=2, stranded=1)
    assert score_evidence(0, ev).score == score_evidence(0, same).score


@pytest.mark.parametrize(
    "change",
    [
        {"avoidable_casualty_ids": ("b", "a")},  # not sorted
        {"avoidable_casualty_ids": ("s0000",), "unavoidable_stranded_ids": ("s0000",)},  # not disjoint
        {"avoidable_casualty_ids": ("x", "y")},  # does not match the casualty count
        {"max_evacuated": 100},  # does not match the unsavable identities
    ],
)
def test_evidence_identities_must_be_consistent(change: dict[str, object]) -> None:
    base = evidence(100, max_evacuated=99, casualties=1, stranded=1, unavoidable_casualties=0)
    with pytest.raises(ScoringInputError):
        dataclasses.replace(base, **change)  # type: ignore[arg-type]


def test_real_episode_person_level_attribution() -> None:
    # Pa can outrun the fire to the exit but stays and dies; Pb is walled in and can never escape.
    scenario = layout(["Fa...E", "######", "#b####"], termination=TerminationConfig(max_steps=12))
    record = run_episode(scenario, stay_policy, seed=0)
    ev = episode_evidence(scenario, record)
    assert ev.max_evacuated == 1
    assert ev.avoidable_casualty_ids == ("Pa",) and ev.unavoidable_stranded_ids == ("Pb",)
    assert ev.avoidable_stranded_ids == () and ev.unavoidable_casualty_ids == ()
    # The draft's aggregate rule (unavoidable losses offset casualties first) would have hidden Pa's death.
    assert max(0, ev.casualties - (ev.population - ev.max_evacuated)) == 0
    assert score_evidence(0, ev).status is EpisodeStatus.CATASTROPHIC


def test_unsafe_fast_cannot_compensate() -> None:
    # R12: no avoidable casualty and no severe failure always beats perfect quality with one casualty,
    # even with every quality component exactly at its severe threshold, for any population.
    for n in (2, 4, 10, 100, 1000):
        corner = evidence(n, time=20, danger=5, danger_bound=0, moves=20, replan=20, replan_bound=10)
        s = score_evidence(0, corner)
        assert s.severe == () and s.safety == 1
        assert s.score > points(evidence(n, casualties=1))
        assert s.score > 60
    # The same holds with equal stranding.
    assert points(evidence(100, stranded=1, time=20)) > points(evidence(100, stranded=1, casualties=1))
    # A severe failure is allowed to fall below one casualty: it is a failure too.
    assert points(evidence(time=40)) < points(evidence(casualties=1))


def test_realized_bottleneck_cannot_compensate() -> None:
    # A bottleneck is charged by the delay it causes, over the cut-capacity bound (Time), not by blocked actions.
    # R8: a bottleneck that doubles evacuee person-time over the capacity bound costs >= 10 points.
    assert 100 - points(evidence(time=20)) >= 10 and severe_failures(evidence(time=20)) == {}
    # R13: a severe realized bottleneck (Time 0.4), everything else perfect, is never excellent and loses >= 20.
    assert points(evidence(time=25)) < 80 and 100 - points(evidence(time=25)) >= 20
    # A mild realized delay (Time 0.9) stays far above a severe one (Time 0.4).
    assert points(evidence(time_bound=9, time=10)) - points(evidence(time_bound=10, time=25)) > 40
    # A non-compensatory mean: one very bad quality component caps the score.
    assert points(evidence(time=100)) <= 60
    # Several severe failures compound.
    two = score_evidence(0, evidence(time=40, moves=25))
    assert two.severe == ("time", "path") and two.score < points(evidence(time=40))


def test_unavoidable_queueing_is_not_charged() -> None:
    # Person-time above the free-flow and exit bounds, but at the cut-capacity bound, costs nothing.
    s = score_evidence(0, evidence(time_bound=15, free_bound=14, time=15))
    assert s.component("time") == 1 and s.score == 100
    # The same person-time against a higher capacity bound is avoidable delay.
    assert score_evidence(0, evidence(time_bound=12, free_bound=12, time=15)).component("time") == F(12, 15)


def test_no_component_duplicates_time() -> None:
    # Delay is charged once, by Time: changing only the person-time changes only the Time component,
    # and no component reads the block count (a block-based term would be a second delay charge or none).
    base = quality_components(evidence(moves=12, danger=2, danger_bound=1, replan=4, replan_bound=2))
    slow = quality_components(evidence(time=25, moves=12, danger=2, danger_bound=1, replan=4, replan_bound=2))
    assert {k for k in QUALITY_COMPONENTS if base[k] != slow[k]} <= {"time", "hazard"}
    assert slow["time"] == F(10, 25) and slow["hazard"] is not None and slow["hazard"] > base["hazard"]  # type: ignore[operator]
    blocked = quality_components(evidence(blocks=999, moves=12, danger=2, danger_bound=1, replan=4, replan_bound=2))
    assert blocked == base


def test_dead_person_blocks_are_not_double_charged() -> None:
    # Blocked moves of people who later die are not charged again: the death is in Safety, the blocks nowhere.
    for cas in (1, 3):
        assert points(evidence(casualties=cas, blocks=900)) == points(evidence(casualties=cas))
    # Time, Path and Hazard are over evacuees only: casualties do not change them.
    q = quality_components(evidence(casualties=3, time=13, moves=12, danger=2, danger_bound=1))
    assert q == quality_components(evidence(casualties=0, time=13, moves=12, danger=2, danger_bound=1))


def test_catastrophic_gate_and_ceilings() -> None:
    s = score_evidence(0, evidence(casualties=30, stranded=20))
    assert s.status is EpisodeStatus.CATASTROPHIC and s.score == 0 and s.safety == 0
    # No score above 60 after any avoidable casualty, for any population.
    for n in (2, 4, 10, 100, 1000):
        assert points(evidence(n, casualties=1)) <= 60
    # No 95+ after one stranded person either.
    for n in (2, 10, 100, 1000):
        assert points(evidence(n, stranded=1)) < 95


def test_scores_and_components_are_bounded() -> None:
    for cas, st, time, blocks, moves, danger, replan in itertools.product((0, 1, 7), (0, 2), (10, 13, 90), (0, 999), (10, 50), (0, 9), (0, 5)):
        ev = evidence(20, casualties=cas, stranded=st, time=time, blocks=blocks, moves=moves, danger=danger, replan=replan, replan_bound=min(replan, 3))
        s = score_evidence(0, ev)
        assert 0 <= s.score <= 100
        for _, v in s.components:
            assert v is None or 0 <= v <= 1
        assert s.safety is not None and 0 <= s.safety <= 1
        assert s.quality is not None and 0 <= s.quality <= 1


def test_quality_component_definitions() -> None:
    q = quality_components(evidence(population=10, casualties=10))
    assert set(q) == set(QUALITY_COMPONENTS)
    assert q["time"] is None and q["path"] is None and q["hazard"] is None  # nobody evacuated


def test_not_applicable_and_failed_episodes() -> None:
    na = score_evidence(0, evidence(10, max_evacuated=0, casualties=10))
    assert na.status is EpisodeStatus.NOT_APPLICABLE and na.safety is None
    failed = failed_episode(3, "x", 1, FailureKind.TIMEOUT)
    assert failed.status is EpisodeStatus.FAILED and failed.score == 0 and failed.safety == 0
    assert failed.to_dict()["failure"] == "timeout"
    assert [n for n, _ in failed.components] == [*SAFETY_COMPONENTS, *QUALITY_COMPONENTS]


# --------------------------------------------------------------------------- evidence from real episodes


def corridor(**kwargs: object) -> Scenario:
    return layout(["a.....E"], termination=TerminationConfig(max_steps=20), **kwargs)  # type: ignore[arg-type]


def test_real_episode_perfect_and_delayed() -> None:
    scenario = corridor()
    record = run_episode(scenario, greedy_exit_policy, seed=0)
    s = score_episode(scenario, record)
    assert s.score == 100 and s.component("replanning") is None
    delayed = score_episode(scenario, run_episode(scenario, delayed_policy, seed=0))
    # Six moves; three steps of waiting: Time = 6/9, the only imperfect component.
    assert delayed.component("time") == F(6, 9) and delayed.severe == ()
    assert close(delayed.score, 100 * (6 / 9) ** (1 / 2 * BETA))


def test_real_episode_time_bound_counts_exit_capacity() -> None:
    # Two people one step from the same exit: at most one leaves per step, so the bound is 1 + 2, not 1 + 1.
    scenario = layout([".a", "bE"], termination=TerminationConfig(max_steps=10))
    ev = episode_evidence(scenario, run_episode(scenario, greedy_exit_policy, seed=0))
    assert ev.evacuated == 2 and ev.evacuee_time_bound == 1 + 1 and ev.evacuee_time_queue_bound == 1 + 2
    assert ev.evacuee_time_cut_bound == 3 and quality_components(ev)["time"] == F(3, ev.evacuee_time)


DOOR = ["a#..E", "b...E", "c#..E"]  # three people behind a one-cell door; three exits beyond it


def east_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.EAST for p in obs.active}


def test_real_episode_unavoidable_door_queue_is_not_charged() -> None:
    # Free flow 4 + 5 + 5 = 14 and three exits, but one person per step crosses the door:
    # entries at steps 1, 2, 3 plus 3 more steps each give 15.
    scenario = layout(DOOR, termination=TerminationConfig(max_steps=20))
    record = run_episode(scenario, greedy_exit_policy, seed=0)
    ev = episode_evidence(scenario, record)
    assert (ev.evacuee_time_bound, ev.evacuee_time_queue_bound, ev.evacuee_time_cut_bound) == (14, 14, 15)
    assert ev.evacuee_time == 15 and ev.congestion_blocks >= 1  # mild, unavoidable congestion
    s = score_episode(scenario, record)
    assert s.component("time") == 1 and s.severe == () and s.score == 100


def test_real_episode_avoidable_queue_is_charged() -> None:
    # Two exits; everyone queues at the east exit. The capacity bound is 2 + 3 + 3 + 2 = 10; the queue gives 14.
    scenario = layout(["E.abcd.E"], termination=TerminationConfig(max_steps=20))
    ev = episode_evidence(scenario, run_episode(scenario, east_policy, seed=0))
    assert ev.evacuee_time_cut_bound == 10 and ev.evacuee_time == 14
    s = score_evidence(0, ev)
    assert s.component("time") == F(10, 14) and s.score < 100 - 10
    assert score_episode(scenario, run_episode(scenario, greedy_exit_policy, seed=0)).score == 100


CONGESTION_OUTCOMES = frozenset({MoveOutcome.BLOCKED_OCCUPIED, MoveOutcome.BLOCKED_CONFLICT, MoveOutcome.BLOCKED_CYCLE})


def block_scrub(inner: Policy) -> Policy:
    """The inner policy, but every move the public resolver would block on a person becomes STAY."""

    def policy(obs: Observation) -> dict[str, Action]:
        acts = dict(inner(obs))
        positions = {p.id: p.position for p in obs.active}
        resolved = resolve_movements(
            positions,
            acts,
            width=obs.map.width,
            height=obs.map.height,
            walls=obs.map.walls,
            burning=obs.burning,
            enter_fire=obs.rules.enter_fire,
        )
        for m in resolved:
            if m.outcome in CONGESTION_OUTCOMES:
                acts[m.person_id] = Action.STAY
        return acts

    return policy


@pytest.mark.parametrize("scenario_id", ["v1-pub-02-single-exit-dense", "v1-pub-03-corridor-bottleneck"])
def test_block_scrub_keeps_trajectory_and_score(public: Suite, scenario_id: str) -> None:
    # Trajectory invariance: identical authenticated state trajectories give identical scores.
    scenario = public.get(scenario_id)
    for seed in (0, 1):
        base = run_episode(scenario, agent_policy(baseline_factory("congestion-aware-v1")()), seed=seed)
        scrub = run_episode(scenario, block_scrub(agent_policy(baseline_factory("congestion-aware-v1")())), seed=seed)
        # Identical observations-driving states: positions, statuses, fire, evacuations, casualties, every step.
        assert [canonical_json(x.state.to_dict()) for x in base.steps] == [canonical_json(x.state.to_dict()) for x in scrub.steps]
        a, b = episode_evidence(scenario, base), episode_evidence(scenario, scrub)
        assert a.congestion_blocks > 0 and b.congestion_blocks == 0  # only the action-log labels differ
        assert (a.evacuated, a.casualties, a.stranded, a.evacuee_time, a.path_cells_moved, a.evacuee_danger_states) == (
            b.evacuated,
            b.casualties,
            b.stranded,
            b.evacuee_time,
            b.path_cells_moved,
            b.evacuee_danger_states,
        )
        sa, sb = score_evidence(0, a), score_evidence(0, b)
        assert sa.score == sb.score and sa.components == sb.components and sa.severe == sb.severe


def test_real_episode_nobody_can_escape_is_not_applicable() -> None:
    scenario = layout(["Fa......E"], termination=TerminationConfig(max_steps=3))
    s = score_episode(scenario, run_episode(scenario, greedy_exit_policy, seed=0))
    assert s.status is EpisodeStatus.NOT_APPLICABLE


def test_references_match_the_privileged_oracle(public: Suite) -> None:
    for scenario in public.scenarios:
        refs = _references(scenario, 0)
        bound = privileged_escape_bound(scenario, 0)
        earliest = dict(bound.earliest_evacuation)
        for person in scenario.people:
            mine = refs.earliest[0][refs.index[person.position]]
            theirs = earliest[person.id]
            assert (None if mine == 1 << 62 else mine) == theirs, (scenario.scenario_id, person.id)
            if theirs is not None:
                assert refs.min_moves[person.position] <= theirs
                assert person.position in refs.min_danger


def test_evidence_rejects_a_tampered_record(small: Suite, greedy: EvaluationResult) -> None:
    e = greedy.episodes[0]
    scenario = small.get(e.episode.scenario_id)
    tampered = dataclasses.replace(e.record, seed=e.record.seed + 1)
    with pytest.raises(Exception):
        episode_evidence(scenario, tampered)


# --------------------------------------------------------------------------- results and aggregation


def test_hierarchical_aggregation_hand_check(small: Suite, greedy: EvaluationResult) -> None:
    report = score_result(greedy, small, resamples=RESAMPLES)
    by_index = dict(report.episode_scores)
    by_scenario: dict[str, list[Fraction]] = {}
    for spec in greedy.manifest.episodes:
        if spec.index in by_index:
            by_scenario.setdefault(spec.scenario_id, []).append(by_index[spec.index])
    means = {k: sum(v, F(0)) / len(v) for k, v in by_scenario.items()}
    # Each public scenario has a different primary family, so the headline is the mean of scenario means.
    assert report.score == sum(means.values(), F(0)) / len(means)
    assert {g.name: g.score for g in report.scenarios} == means
    assert report.family("maze_obstacle_routing").score == means["v1-pub-08-maze-deceptive-exit"]
    assert [g.name for g in report.families] == ["single_exit_low_density", "maze_obstacle_routing", "route_invalidation_replanning"]
    averages = dict(report.averages)
    assert averages["family_balanced"] == report.score
    assert averages["micro"] == sum(by_index.values(), F(0)) / len(by_index)
    assert report.ci is not None and report.ci.lower <= report.score <= report.ci.upper
    assert not report.official and report.pack_digest is None


def test_report_is_deterministic(small: Suite, greedy: EvaluationResult) -> None:
    a = score_result(greedy, small, resamples=RESAMPLES)
    b = score_result(greedy, small, resamples=RESAMPLES)
    assert a.digest() == b.digest() and a.to_json() == b.to_json()


def test_missing_and_declared_failures_score_zero(small: Suite, greedy: EvaluationResult) -> None:
    partial = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, greedy.episodes[1:-1])
    last = greedy.manifest.episodes[-1].index
    report = score_result(partial, small, failures={last: FailureKind.TIMEOUT}, resamples=RESAMPLES)
    by_index = dict(report.episode_scores)
    assert by_index[0] == 0 and by_index[last] == 0
    failed = dict(report.robustness.failed_episode_trials)
    assert failed == {"missing": 1, "invalid_output": 0, "model_error": 0, "timeout": 1, "refusal": 0}
    full = score_result(greedy, small, resamples=RESAMPLES)
    assert report.score is not None and full.score is not None and report.score < full.score
    statuses = {e.index: e.status for e in report.episodes}
    assert statuses[0] is EpisodeStatus.FAILED


def test_invalid_output_scores_zero(small: Suite, greedy: EvaluationResult) -> None:
    partial = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, greedy.episodes[1:])
    report = score_result(partial, small, failures={0: FailureKind.INVALID_OUTPUT}, resamples=RESAMPLES)
    assert dict(report.episode_scores)[0] == 0
    assert dict(report.robustness.failed_episode_trials)["invalid_output"] == 1


@pytest.mark.parametrize(
    "failures",
    [
        {0: FailureKind.TIMEOUT},  # episode 0 has a committed result
        {999: FailureKind.TIMEOUT},  # not scheduled
        {1: "timeout"},  # not a FailureKind
    ],
)
def test_invalid_failure_declarations_rejected(small: Suite, greedy: EvaluationResult, failures: dict[int, object]) -> None:
    partial = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, (greedy.episodes[0],))
    with pytest.raises(ScoringInputError):
        score_result(partial, small, failures=failures, resamples=RESAMPLES)  # type: ignore[arg-type]


def test_tampered_metrics_fail_closed(small: Suite, greedy: EvaluationResult) -> None:
    e = greedy.episodes[0]
    wrong = dataclasses.replace(e.metrics, seed=e.metrics.seed + 1)
    bad = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, (dataclasses.replace(e, metrics=wrong), *greedy.episodes[1:]))
    with pytest.raises(ScoringInputError):
        score_result(bad, small, resamples=RESAMPLES)


def test_trials_average_and_must_match(small: Suite, greedy: EvaluationResult, delayed: EvaluationResult) -> None:
    one = score_results([greedy], small, resamples=RESAMPLES)
    two = score_results([greedy, greedy], small, resamples=RESAMPLES)
    assert one.score == two.score and two.trials == 2 and len(two.episodes) == 2 * len(one.episodes)
    # A trial that failed entirely halves every episode mean.
    empty = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, ())
    half = score_results([greedy, empty], small, resamples=RESAMPLES)
    assert one.score is not None and half.score == one.score / 2
    with pytest.raises(ScoringInputError):
        score_results([greedy, delayed], small, resamples=RESAMPLES)  # different models
    with pytest.raises(ScoringInputError):
        score_results([], small, resamples=RESAMPLES)


# --------------------------------------------------------------------------- comparison and leaderboard


def test_paired_comparison_outcomes(small: Suite, greedy: EvaluationResult, stay: EvaluationResult) -> None:
    g = score_result(greedy, small, resamples=RESAMPLES)
    s = score_result(stay, small, resamples=RESAMPLES)
    assert compare_scores(g, g, resamples=RESAMPLES).outcome is ComparisonOutcome.EXACT_TIE
    better = compare_scores(g, s, resamples=RESAMPLES)
    assert better.outcome is ComparisonOutcome.A_BETTER and better.delta == g.score
    assert better.ci is not None and better.ci.lower > 0
    assert compare_scores(s, g, resamples=RESAMPLES).outcome is ComparisonOutcome.B_BETTER


def test_small_differences_are_inconclusive(small: Suite, greedy: EvaluationResult, delayed: EvaluationResult) -> None:
    huge_margin = dataclasses.replace(OFFICIAL_CONSTANTS, practical_difference=F(1000))
    g = score_result(greedy, small, resamples=RESAMPLES, constants=huge_margin)
    d = score_result(delayed, small, resamples=RESAMPLES, constants=huge_margin)
    assert compare_scores(g, d, resamples=RESAMPLES).outcome is ComparisonOutcome.INCONCLUSIVE
    official = score_result(greedy, small, resamples=RESAMPLES)
    with pytest.raises(ScoringInputError):
        compare_scores(official, d, resamples=RESAMPLES)  # different constants


def test_too_few_episodes_is_inconclusive(public: Suite) -> None:
    tiny = subset(public, (IDS[0],))
    m = build_evaluation_manifest(tiny, (1, 2))
    g = score_result(run_result(tiny, m, "g", greedy_exit_policy), tiny, resamples=RESAMPLES)
    d = score_result(run_result(tiny, m, "d", delayed_policy), tiny, resamples=RESAMPLES)
    assert g.ci is None
    assert compare_scores(g, d, resamples=RESAMPLES).outcome is ComparisonOutcome.INCONCLUSIVE


def test_leaderboard_safety_dominates_ties(small: Suite, greedy: EvaluationResult, delayed: EvaluationResult, stay: EvaluationResult) -> None:
    reports = [score_result(r, small, resamples=RESAMPLES) for r in (stay, delayed, greedy)]
    ordered = leaderboard_order(reports)
    assert ordered[-1].model_id == "stay" and ordered[0].score == max(r.score for r in reports if r.score is not None)
    g, d = reports[2], reports[1]
    # Same displayed score: the safer report ranks first, whatever its model ID.
    tie_g = dataclasses.replace(g, model_id="z-safer", score=F(500, 10), safety=F(9, 10))
    tie_d = dataclasses.replace(d, model_id="a-less-safe", score=F(5004, 100), safety=F(8, 10))
    assert [r.model_id for r in leaderboard_order([tie_d, tie_g])] == ["z-safer", "a-less-safe"]


def test_congestion_diagnostics_are_reported_not_scored(small: Suite, greedy: EvaluationResult) -> None:
    report = score_result(greedy, small, resamples=RESAMPLES)
    dg = report.diagnostics
    assert isinstance(dg, CongestionDiagnostics) and dg.movement_attempts > 0
    assert dg.congestion_block_rate is not None and 0 <= dg.congestion_block_rate <= 1
    assert dg.exit_throughput is not None and 0 < dg.exit_throughput <= 1
    assert set(report.to_dict()["diagnostics"]) >= {"congestion_block_rate", "contested_cell_steps", "hotspot_cells", "exit_throughput"}
    assert "Congestion diagnostics (not scored)" in format_summary(report)


def test_summary_is_points_not_percent(small: Suite, greedy: EvaluationResult) -> None:
    text = format_summary(score_result(greedy, small, resamples=RESAMPLES))
    assert "benchmark-performance points under EMBER scoring version 1.1" in text
    assert "not a percentage of people evacuated" in text
    assert "UNOFFICIAL" in text


def test_efficiency_view_is_separate(small: Suite, greedy: EvaluationResult) -> None:
    report = score_result(greedy, small, resamples=RESAMPLES)
    view = efficiency_view(report, F(2), "USD")
    assert view.score == report.score and report.score is not None and view.points_per_unit == report.score / 2
    assert "cost" not in report.to_dict()
    with pytest.raises(ValueError):
        efficiency_view(report, F(-1), "USD")


def test_operational_record_is_separate_and_validated() -> None:
    config = {"model": "example-model", "snapshot": "2026-10-01", "temperature": 0, "top_p": 1, "max_tokens": 64, "adapter": "1.0"}
    digest = inference_config_digest(config)
    assert digest == inference_config_digest(dict(reversed(list(config.items()))))  # canonical: key order is irrelevant
    record = OperationalRecord("example-model", "example-provider", digest, 3, 30, 12.5, 1000, 200, F(3, 100), "USD", 2, 1)
    assert record.to_dict()["cost"] == "3/100" and len(record.digest()) == 64
    # The score never reads operational data: score_results has no parameter for it.
    assert "operational" not in score_results.__code__.co_varnames
    bad: list[dict[str, object]] = [
        {"model_id": ""},
        {"inference_config_digest": "x" * 64},
        {"retries": -1},
        {"trials": 0},
        {"latency_seconds": float("nan")},
        {"cost_unit": None},
    ]
    for change in bad:
        with pytest.raises(ScoringInputError):
            dataclasses.replace(record, **change)  # type: ignore[arg-type]


def test_scoring_does_not_depend_on_visualization() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "emberbench"
    for name in ("scoring.py", "environment_pack.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not imported & {"audit", "visualization", "emberbench.audit", "emberbench.visualization"}, name
