"""Tests for the EMBER Score (``emberbench.scoring``): hand-calculated values, penalties, ceilings and aggregation."""

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
    EpisodeEvidence,
    EpisodeResult,
    EpisodeStatus,
    EvaluationManifest,
    EvaluationResult,
    FailureKind,
    Observation,
    Scenario,
    ScoringInputError,
    Suite,
    TerminationConfig,
    build_evaluation_manifest,
    compare_scores,
    compute_episode_metrics,
    efficiency_view,
    episode_evidence,
    format_summary,
    leaderboard_order,
    load_suite,
    run_episode,
    score_episode,
    score_result,
    score_results,
)
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
)

SUITE_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "public" / "v1"
# pub-08 belongs to two families (primary maze_obstacle_routing); pub-10 has stochastic fire.
IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-08-maze-deceptive-exit", "v1-pub-10-route-invalidation")
SEEDS = (11, 7, 3, 5, 9)
RESAMPLES = 200
F = Fraction


# --------------------------------------------------------------------------- helpers


def evidence(
    population: int = 100,
    *,
    max_evacuated: int | None = None,
    casualties: int = 0,
    stranded: int = 0,
    time_bound: int = 10,
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
    """Hand-made evidence. Per-evacuee values are multiplied by the number of evacuees."""
    evac = population - casualties - stranded
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
        evacuee_time=evac * (time_bound if time is None else time),
        evacuee_time_bound=evac * time_bound,
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
    assert SCORING_VERSION == "1.0"
    c = OFFICIAL_CONSTANTS
    assert c.weights() == {"time": F(3, 10), "coordination": F(1, 5), "hazard": F(1, 5), "path": F(1, 10), "replanning": F(1, 5)}
    assert str(c.casualty_incident_factor) == "0.6" and str(c.casualty_rate_exponent) == "4"
    assert str(c.stranding_incident_factor) == "0.8" and str(c.stranding_rate_exponent) == "2"
    assert str(c.evacuation_exponent) == "2" and str(c.congestion_exponent) == "2" and str(c.hazard_exponent) == "1"
    assert c.catastrophic_loss_rate == F(1, 2) and c.practical_difference == F(1)
    assert HAZARD_RADIUS == 2
    assert len(c.digest()) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"weight_time": F(1, 2)},  # weights no longer sum to 1
        {"casualty_incident_factor": Decimal("0")},
        {"stranding_incident_factor": Decimal("1.5")},
        {"catastrophic_loss_rate": F(0)},
        {"evacuation_exponent": Decimal("-1")},
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
    # Time 1/2; replanning undefined, so the weights renormalize over 0.8: 100 * 0.5^(0.3/0.8).
    s = score_evidence(0, evidence(time_bound=10, time=20))
    assert s.component("time") == F(1, 2)
    assert close(s.score, 100 * 0.5 ** (3 / 8))


def test_congestion_penalty_is_nonlinear() -> None:
    # Congestion block rate 0.3: Coordination = 0.7^2 = 0.49; 100 * 0.49^(0.2/0.8).
    s = score_evidence(0, evidence(blocks=300))
    assert s.component("coordination") == F(49, 100)
    assert close(s.score, 100 * 0.49 ** (1 / 4))
    # Doubling the block rate more than doubles the loss.
    loss = [100 - points(evidence(blocks=b)) for b in (150, 300, 600)]
    assert loss[1] > 2 * loss[0] * 0.95 and loss[2] > 2 * loss[1]


def test_exposure_penalty_without_deaths() -> None:
    # Each evacuee spends 4 of 10 states in the danger zone where 1 was unavoidable: H = 1 - 3/10.
    s = score_evidence(0, evidence(danger=4, danger_bound=1))
    assert s.component("hazard") == F(7, 10)
    assert s.component("casualty") == 1 and s.safety == 1
    assert close(s.score, 100 * 0.7 ** (1 / 4))
    # Unavoidable danger costs nothing.
    assert score_evidence(0, evidence(danger=3, danger_bound=3)).score == 100


def test_path_penalty_hand_calculated() -> None:
    s = score_evidence(0, evidence(moves_bound=10, moves=16))
    assert s.component("path") == F(10, 16)
    assert close(s.score, 100 * (10 / 16) ** (1 / 8))


def test_replanning_penalty_hand_calculated() -> None:
    # Replanning 1/4 with every weight in use: 100 * 0.25^0.2.
    s = score_evidence(0, evidence(replan=40, replan_bound=10))
    assert s.component("replanning") == F(1, 4)
    assert close(s.score, 100 * 0.25**0.2)
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
    # Two people nobody can save are not charged: unavoidable losses go to casualties first.
    unavoidable = evidence(100, max_evacuated=98, casualties=2)
    assert unavoidable.avoidable_casualties == 0
    assert score_evidence(0, unavoidable).score == 100
    # One more loss than unavoidable is charged.
    assert points(evidence(100, max_evacuated=98, casualties=3)) < 60


def test_unsafe_fast_cannot_compensate() -> None:
    fast_unsafe = points(evidence(casualties=1))  # perfect quality, one casualty
    safe_slow = points(evidence(time=40))  # 4x slower than possible, nobody lost
    assert safe_slow > fast_unsafe


def test_bottleneck_cannot_compensate() -> None:
    # Perfect safety and speed cannot hide a congestion block rate of 0.6.
    assert points(evidence(blocks=600)) < 80
    # A non-compensatory mean: one very bad quality component caps the score.
    assert points(evidence(time=100)) <= 60


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
    assert delayed.component("time") == F(6, 9)
    assert close(delayed.score, 100 * (6 / 9) ** (3 / 8))


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
    assert failed == {"missing": 1, "invalid_output": 0, "model_error": 0, "timeout": 1}
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


def test_summary_is_points_not_percent(small: Suite, greedy: EvaluationResult) -> None:
    text = format_summary(score_result(greedy, small, resamples=RESAMPLES))
    assert "benchmark-performance points under EMBER scoring version 1.0" in text
    assert "not a percentage of people evacuated" in text
    assert "UNOFFICIAL" in text


def test_efficiency_view_is_separate(small: Suite, greedy: EvaluationResult) -> None:
    report = score_result(greedy, small, resamples=RESAMPLES)
    view = efficiency_view(report, F(2), "USD")
    assert view.score == report.score and report.score is not None and view.points_per_unit == report.score / 2
    assert "cost" not in report.to_dict()
    with pytest.raises(ValueError):
        efficiency_view(report, F(-1), "USD")
