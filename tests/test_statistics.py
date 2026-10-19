"""Aggregate and paired statistics: exact formulas, fail-closed input, missing data, pairing and determinism."""

import dataclasses
import hashlib
import math
import os
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import greedy_exit_policy
from emberbench import (
    METRICS_VERSION,
    OBSERVATION_VERSION,
    SCHEMA_VERSION,
    SIMULATOR_VERSION,
    STATISTICS_VERSION,
    Action,
    AggregateStatistics,
    EpisodeResult,
    EvaluationError,
    EvaluationManifest,
    EvaluationResult,
    MetricDirection,
    MetricKind,
    Observation,
    PairedStatistics,
    ScenarioFamily,
    StatisticsInputError,
    Suite,
    aggregate_result,
    build_evaluation_manifest,
    compare_results,
    compute_episode_metrics,
    load_suite,
    run_episode,
)
from emberbench.evaluation import EVALUATION_RESULT_VERSION
from emberbench.simulator import Policy
from emberbench.statistics import (
    METRIC_DEFINITIONS,
    METRIC_NAMES,
    MIN_BOOTSTRAP_SAMPLE_SIZE,
    BootstrapStream,
    DescriptiveStatistics,
    bootstrap_mean_ci,
    bootstrap_seed,
    describe,
    metric_definition,
    quantile,
)
from test_suite import PINNED_HASHES, SUITE_DIR, SUITE_HASH

ROOT = Path(__file__).resolve().parents[1]

# pub-08 belongs to two families; pub-10 has stochastic fire.
IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-08-maze-deceptive-exit", "v1-pub-10-route-invalidation")
MULTI_FAMILY_ID = "v1-pub-08-maze-deceptive-exit"
SEEDS = (11, 7, 3, 5, 9)
RESAMPLES = 200  # small, to keep the tests fast
# Episodes the partial result does not contain: one of pub-01, two of pub-10.
SKIPPED = frozenset({1, 10, 13})

F = Fraction


def stay_policy(obs: Observation) -> dict[str, Action]:
    return {p.id: Action.STAY for p in obs.active}


def delayed_policy(obs: Observation) -> dict[str, Action]:
    """Stay for three steps, then head for the nearest exit: a second model with different outcomes."""
    return stay_policy(obs) if obs.step < 3 else greedy_exit_policy(obs)


def subset(suite: Suite, ids: tuple[str, ...]) -> Suite:
    entries = tuple(e for e in suite.manifest.entries if e.scenario_id in ids)
    return Suite(
        root=suite.root,
        manifest=dataclasses.replace(suite.manifest, entries=entries),
        scenarios=tuple(s for s in suite.scenarios if s.scenario_id in ids),
    )


def run_result(
    suite: Suite, manifest: EvaluationManifest, model_id: str, policy: Policy, skip: frozenset[int] = frozenset()
) -> EvaluationResult:
    """An in-memory result: the same episodes ``run_evaluation`` would produce, without the file."""
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


# A one-scenario suite with five seeds keeps the many-call tests cheap.
@pytest.fixture(scope="module")
def tiny(public: Suite) -> Suite:
    return subset(public, (IDS[0],))


@pytest.fixture(scope="module")
def tiny_manifest(tiny: Suite) -> EvaluationManifest:
    return build_evaluation_manifest(tiny, (1, 2, 3, 4, 5))


@pytest.fixture(scope="module")
def tiny_greedy(tiny: Suite, tiny_manifest: EvaluationManifest) -> EvaluationResult:
    return run_result(tiny, tiny_manifest, "greedy", greedy_exit_policy)


@pytest.fixture(scope="module")
def tiny_stay(tiny: Suite, tiny_manifest: EvaluationManifest) -> EvaluationResult:
    return run_result(tiny, tiny_manifest, "stay", stay_policy)


@pytest.fixture(scope="module")
def partial(greedy: EvaluationResult) -> EvaluationResult:
    episodes = tuple(e for e in greedy.episodes if e.episode.index not in SKIPPED)
    return EvaluationResult(EVALUATION_RESULT_VERSION, "greedy-partial", greedy.manifest, episodes)


@pytest.fixture(scope="module")
def agg(greedy: EvaluationResult, small: Suite) -> AggregateStatistics:
    return aggregate_result(greedy, small, resamples=RESAMPLES)


@pytest.fixture(scope="module")
def agg_partial(partial: EvaluationResult, small: Suite) -> AggregateStatistics:
    return aggregate_result(partial, small, resamples=RESAMPLES)


@pytest.fixture(scope="module")
def paired_a(greedy: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", greedy.manifest, tuple(e for e in greedy.episodes if e.episode.index != 2))


@pytest.fixture(scope="module")
def paired_b(delayed: EvaluationResult) -> EvaluationResult:
    episodes = tuple(e for e in delayed.episodes if e.episode.index not in {4, 6})
    return EvaluationResult(EVALUATION_RESULT_VERSION, "delayed", delayed.manifest, episodes)


@pytest.fixture(scope="module")
def paired(paired_a: EvaluationResult, paired_b: EvaluationResult, small: Suite) -> PairedStatistics:
    return compare_results(paired_a, paired_b, small, resamples=RESAMPLES)


def exact(result: EvaluationResult, metric: str) -> dict[int, Fraction | None]:
    """The metric of every episode in ``result``, rebuilt from its stored float (exact for these values)."""
    d = metric_definition(metric)
    return {e.episode.index: d.exact(e.metrics) for e in result.episodes}


# --------------------------------------------------------------------------- descriptive statistics


def test_hand_computed_even_sample() -> None:
    s = describe([F(4), F(1), F(3), F(2)])
    assert s.n == 4
    assert s.mean == F(5, 2)
    assert s.median == F(5, 2)
    # Σ(x - 5/2)² = 9/4 + 1/4 + 1/4 + 9/4 = 5; divided by n - 1 = 3.
    assert s.variance == F(5, 3)
    assert s.std == math.sqrt(5 / 3)
    assert (s.minimum, s.maximum) == (F(1), F(4))
    # Type 7: h = 3p. p25 -> 1 + 0.75; p75 -> 3 + 0.25; p90 -> 3 + 0.7; p95 -> 3 + 0.85.
    assert dict(s.quantiles) == {"p25": F(7, 4), "p50": F(5, 2), "p75": F(13, 4), "p90": F(37, 10), "p95": F(77, 20)}


def test_hand_computed_odd_sample_with_fractions() -> None:
    s = describe([F(1, 3), F(0), F(1)])
    assert s.mean == F(4, 9)
    assert s.median == F(1, 3)
    # (4/9)² + (1/9)² + (5/9)² = 42/81, divided by 2.
    assert s.variance == F(21, 81)
    assert dict(s.quantiles)["p25"] == F(1, 6)
    assert dict(s.quantiles)["p95"] == F(1, 3) + F(9, 10) * F(2, 3)


def test_empty_sample_is_undefined_not_zero() -> None:
    s = describe([])
    assert s == DescriptiveStatistics(0, None, None, None, None, None, None, tuple((label, None) for label, _ in s.quantiles))
    assert s.to_dict()["mean"] is None
    assert quantile([], F(1, 2)) is None


def test_single_value_has_no_variance() -> None:
    s = describe([F(3, 4)])
    assert (s.n, s.mean, s.median, s.minimum, s.maximum) == (1, F(3, 4), F(3, 4), F(3, 4), F(3, 4))
    assert s.variance is None and s.std is None
    assert all(q == F(3, 4) for _, q in s.quantiles)


def test_quantile_level_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        quantile([F(1)], F(3, 2))


def test_order_of_input_does_not_matter() -> None:
    values = [F(i % 5, 3) for i in range(11)]
    assert describe(values) == describe(list(reversed(values)))


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_stream_is_the_specified_shake_256_stream() -> None:
    seed = hashlib.sha256(b"seed").digest()
    block = hashlib.shake_256(b"emberbench/statistics/bootstrap/v1\x00" + seed + (0).to_bytes(8, "big")).digest(24)
    stream = BootstrapStream(seed)
    assert [stream.next_u64() for _ in range(3)] == [int.from_bytes(block[i : i + 8], "big") for i in (0, 8, 16)]


def test_bootstrap_ci_is_deterministic_and_seed_dependent() -> None:
    values = [F(i % 7, 12) for i in range(20)]
    seed = bootstrap_seed({"x": 1})
    first = bootstrap_mean_ci(values, seed, RESAMPLES)
    assert first is not None
    assert first == bootstrap_mean_ci(values, seed, RESAMPLES)
    assert min(values) <= first.lower <= describe(values).mean <= first.upper <= max(values)  # type: ignore[operator]
    assert first != bootstrap_mean_ci(values, bootstrap_seed({"x": 2}), RESAMPLES)
    assert first.to_dict()["seed"] == seed.hex()


def test_bootstrap_ci_needs_a_minimum_sample() -> None:
    seed = bootstrap_seed({})
    assert bootstrap_mean_ci([F(1)] * (MIN_BOOTSTRAP_SAMPLE_SIZE - 1), seed, RESAMPLES) is None
    constant = bootstrap_mean_ci([F(2, 3)] * MIN_BOOTSTRAP_SAMPLE_SIZE, seed, RESAMPLES)
    assert constant is not None and constant.lower == constant.upper == F(2, 3)


@pytest.mark.parametrize("resamples", [0, 99, True, 1.5, 10**7])
def test_invalid_resample_counts_are_rejected(resamples: object, greedy: EvaluationResult, small: Suite) -> None:
    with pytest.raises((TypeError, ValueError)):
        aggregate_result(greedy, small, resamples=resamples)  # type: ignore[arg-type]


def test_bootstrap_does_not_touch_global_random_state() -> None:
    import random

    random.seed(1234)
    before = random.getstate()
    bootstrap_mean_ci([F(i) for i in range(10)], bootstrap_seed({}), RESAMPLES)
    assert random.getstate() == before


# --------------------------------------------------------------------------- metric definitions


def test_metric_set_and_directions_are_pinned() -> None:
    directions = {d.name: (d.kind, d.direction) for d in METRIC_DEFINITIONS}
    O, B, P, D = MetricKind.OUTCOME, MetricKind.BEHAVIOUR, MetricKind.PROXY, MetricKind.DESCRIPTIVE
    H, L, N = MetricDirection.HIGHER_IS_BETTER, MetricDirection.LOWER_IS_BETTER, MetricDirection.DESCRIPTIVE
    assert directions == {
        "evacuated_rate": (O, H),
        "casualty_rate": (O, L),
        "stranded_rate": (O, L),
        "terminal_trapped_rate": (O, L),
        "mean_person_time_to_safety": (O, L),
        "blocked_rate": (B, L),
        "congestion_block_rate": (B, L),
        "aggregate_path_efficiency": (P, H),
        "mean_exposure": (P, L),
        "exits_used": (D, N),
        "max_exit_share": (D, N),
        "exit_balance": (D, N),
    }
    # Only descriptive metrics lack a direction, and every descriptive-direction metric is labelled descriptive.
    assert all((d.kind is D) == (d.direction is N) for d in METRIC_DEFINITIONS)


def test_exact_values_match_the_stored_floats(
    greedy: EvaluationResult, delayed: EvaluationResult, tiny_stay: EvaluationResult
) -> None:
    for result in (greedy, delayed, tiny_stay):
        for e in result.episodes:
            for d in METRIC_DEFINITIONS:
                value, stored = d.exact(e.metrics), d.stored(e.metrics)
                assert (value is None) == (stored is None)
                if value is not None:
                    assert float(value) == stored


# --------------------------------------------------------------------------- aggregate


def test_complete_aggregate_matches_direct_computation(agg: AggregateStatistics, greedy: EvaluationResult) -> None:
    assert agg.statistics_version == STATISTICS_VERSION == "1.0"
    assert agg.missing_episodes == ()
    for name in METRIC_NAMES:
        values = [v for v in exact(greedy, name).values() if v is not None]
        summary = agg.overall.metric(name)
        assert summary.statistics == describe(values)
        assert (summary.scheduled, summary.observed, summary.missing) == (15, 15, 0)
        assert summary.undefined == 15 - len(values)


def test_robustness_counts(agg: AggregateStatistics, greedy: EvaluationResult) -> None:
    r = agg.overall.robustness
    ms = [e.metrics for e in greedy.episodes]
    assert (r.scheduled, r.observed, r.missing) == (15, 15, 0)
    assert r.all_resolved + r.max_steps == r.observed
    assert r.all_resolved == sum(1 for m in ms if m.termination_reason.value == "all_resolved")
    assert r.with_casualties == sum(1 for m in ms if m.safety.casualties)
    assert r.with_stranded == sum(1 for m in ms if m.safety.stranded)
    assert r.zero_evacuation == sum(1 for m in ms if not m.safety.evacuated)


def test_partial_result_reports_missing_episodes_without_imputation(agg_partial: AggregateStatistics, partial: EvaluationResult) -> None:
    assert [k.index for k in agg_partial.missing_episodes] == sorted(SKIPPED)
    assert agg_partial.to_dict()["observed_episodes"] == 15 - len(SKIPPED)
    assert agg_partial.overall.missing_indices == tuple(sorted(SKIPPED))
    assert agg_partial.overall.robustness.missing == len(SKIPPED)
    assert agg_partial.scenario(IDS[0]).missing_indices == (1,)
    assert agg_partial.scenario(IDS[1]).missing_indices == ()
    assert agg_partial.scenario(IDS[2]).missing_indices == (10, 13)
    for summary in agg_partial.overall.metrics:
        assert summary.missing == len(SKIPPED)
        assert summary.statistics.n == summary.observed - summary.undefined == 12 - summary.undefined
    # Statistics use the observed episodes only: no zero, mean or worst-case fill-in.
    values = [v for v in exact(partial, "evacuated_rate").values() if v is not None]
    assert agg_partial.overall.metric("evacuated_rate").statistics == describe(values)


def test_missing_is_not_counted_as_failure(agg_partial: AggregateStatistics) -> None:
    r = agg_partial.overall.robustness
    assert r.all_resolved + r.max_steps == r.observed == r.scheduled - r.missing


def test_undefined_metrics_are_counted_not_filled(tiny_stay: EvaluationResult, tiny: Suite) -> None:
    s = aggregate_result(tiny_stay, tiny, resamples=RESAMPLES)
    # Nobody moves, so blocked_rate has no attempts and nobody evacuates, so exit shares are undefined.
    for name in ("blocked_rate", "congestion_block_rate", "aggregate_path_efficiency", "max_exit_share", "exit_balance"):
        summary = s.overall.metric(name)
        assert summary.undefined == summary.observed == 5
        assert summary.statistics.n == 0 and summary.statistics.mean is None and summary.mean_ci is None
    assert s.overall.robustness.zero_evacuation == 5


def test_overall_uses_episodes_not_scenario_means(agg_partial: AggregateStatistics, partial: EvaluationResult) -> None:
    name = "mean_person_time_to_safety"
    values = exact(partial, name)
    overall = agg_partial.overall.metric(name).statistics.mean
    assert overall == sum(v for v in values.values() if v is not None) / len(values)
    scenario_means = [m for g in agg_partial.scenarios if (m := g.metric(name).statistics.mean) is not None]
    assert len(scenario_means) == len(agg_partial.scenarios)
    assert overall != sum(scenario_means) / len(scenario_means)  # unequal group sizes after the skipped episodes


def test_scenario_and_family_breakdown(agg: AggregateStatistics, small: Suite, manifest: EvaluationManifest) -> None:
    assert [g.name for g in agg.scenarios] == list(IDS)  # suite order
    expected_families = [f.value for f in ScenarioFamily if any(f in e.families for e in small.manifest.entries)]
    assert [g.name for g in agg.families] == expected_families  # ScenarioFamily order
    for g in agg.scenarios:
        assert g.episode_indices == tuple(s.index for s in manifest.episodes if s.scenario_id == g.name)
    multi = agg.scenario(MULTI_FAMILY_ID).episode_indices
    families = small.manifest.entries[1].families
    assert len(families) == 2
    for family in families:
        assert agg.family(family).episode_indices == multi
        for name in METRIC_NAMES:
            assert agg.family(family).metric(name).statistics == agg.scenario(MULTI_FAMILY_ID).metric(name).statistics


def test_confidence_intervals_follow_sample_size(agg: AggregateStatistics) -> None:
    ci = agg.overall.metric("mean_person_time_to_safety").mean_ci
    assert ci is not None and ci.resamples == RESAMPLES and ci.statistic == "mean"
    mean = agg.overall.metric("mean_person_time_to_safety").statistics.mean
    assert mean is not None and ci.lower <= mean <= ci.upper
    # Five seeds per scenario: just enough for an interval.
    assert all(g.metric("evacuated_rate").mean_ci is not None for g in agg.scenarios)


def test_aggregate_is_deterministic_and_canonical(agg: AggregateStatistics, greedy: EvaluationResult, small: Suite) -> None:
    again = aggregate_result(greedy, small, resamples=RESAMPLES)
    assert again == agg
    assert again.to_json() == agg.to_json()
    assert again.digest() == agg.digest()


def test_digest_changes_with_data_manifest_label_and_resamples(
    tiny: Suite, tiny_greedy: EvaluationResult, tiny_stay: EvaluationResult
) -> None:
    reseeded = run_result(tiny, build_evaluation_manifest(tiny, (1, 2, 3, 4, 6)), "greedy", greedy_exit_policy)
    partial = EvaluationResult(EVALUATION_RESULT_VERSION, "greedy", tiny_greedy.manifest, tiny_greedy.episodes[1:])
    relabelled = dataclasses.replace(tiny_greedy, model_id="greedy-2")
    base = aggregate_result(tiny_greedy, tiny, resamples=RESAMPLES).digest()
    digests = {
        base,
        aggregate_result(tiny_stay, tiny, resamples=RESAMPLES).digest(),
        aggregate_result(reseeded, tiny, resamples=RESAMPLES).digest(),
        aggregate_result(partial, tiny, resamples=RESAMPLES).digest(),
        aggregate_result(relabelled, tiny, resamples=RESAMPLES).digest(),
        aggregate_result(tiny_greedy, tiny, resamples=RESAMPLES + 1).digest(),
    }
    assert len(digests) == 6
    assert aggregate_result(tiny_greedy, tiny, resamples=RESAMPLES).digest() == base


def test_no_score_ranking_or_winner_in_aggregate(agg: AggregateStatistics) -> None:
    text = agg.to_json().lower()
    for word in ("score", "rank", "winner", "leaderboard"):
        assert word not in text


# --------------------------------------------------------------------------- fail-closed input


def _one_episode(result: EvaluationResult, index: int, **changes: object) -> EvaluationResult:
    e = next(e for e in result.episodes if e.episode.index == index)
    return EvaluationResult(EVALUATION_RESULT_VERSION, result.model_id, result.manifest, (dataclasses.replace(e, **changes),))  # type: ignore[arg-type]


def test_tampered_run_record_is_rejected(greedy: EvaluationResult, small: Suite) -> None:
    e = greedy.episodes[11]  # pub-10: stochastic fire
    last = e.record.steps[-1]
    forged = dataclasses.replace(last, state=dataclasses.replace(last.state, burning=frozenset()))
    record = dataclasses.replace(e.record, steps=e.record.steps[:-1] + (forged,))
    with pytest.raises(StatisticsInputError, match="authentication"):
        aggregate_result(_one_episode(greedy, 11, record=record), small, resamples=RESAMPLES)


def test_genuine_record_with_wrong_metrics_is_rejected(greedy: EvaluationResult, delayed: EvaluationResult, small: Suite) -> None:
    # The delayed record is genuine and belongs to the same episode, but the stored metrics are greedy's.
    with pytest.raises(StatisticsInputError, match="stored metrics differ"):
        aggregate_result(_one_episode(greedy, 0, record=delayed.episodes[0].record), small, resamples=RESAMPLES)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda m: dataclasses.replace(m, safety=dataclasses.replace(m.safety, evacuated_rate=0.5)),
        lambda m: dataclasses.replace(m, exits=dataclasses.replace(m.exits, max_exit_share=0.25)),
        lambda m: dataclasses.replace(m, seed=m.seed + 1),
    ],
    ids=["rate", "exit-share", "header"],
)
def test_tampered_stored_metrics_are_rejected(
    tamper: Callable[..., object], greedy: EvaluationResult, small: Suite
) -> None:
    e = greedy.episodes[0]
    with pytest.raises(StatisticsInputError, match="stored metrics differ"):
        aggregate_result(_one_episode(greedy, 0, metrics=tamper(e.metrics)), small, resamples=RESAMPLES)


def test_episode_mutated_after_construction_is_rejected(greedy: EvaluationResult, small: Suite) -> None:
    result = _one_episode(greedy, 0)
    spec = result.episodes[0].episode
    object.__setattr__(result.episodes[0], "episode", dataclasses.replace(spec, seed=spec.seed + 1))
    with pytest.raises(StatisticsInputError):
        aggregate_result(result, small, resamples=RESAMPLES)


def test_result_for_another_suite_is_rejected(greedy: EvaluationResult, public: Suite) -> None:
    with pytest.raises(StatisticsInputError, match="suite_hash"):
        aggregate_result(greedy, public, resamples=RESAMPLES)


def test_non_results_are_rejected(small: Suite) -> None:
    with pytest.raises(TypeError):
        aggregate_result("greedy.json", small)  # type: ignore[arg-type]


def test_paired_comparison_authenticates_both_sides(tiny_greedy: EvaluationResult, tiny_stay: EvaluationResult, tiny: Suite) -> None:
    bad = _one_episode(tiny_stay, 0, metrics=tiny_greedy.episodes[0].metrics)
    with pytest.raises(StatisticsInputError):
        compare_results(tiny_greedy, bad, tiny, resamples=RESAMPLES)
    with pytest.raises(StatisticsInputError):
        compare_results(bad, tiny_greedy, tiny, resamples=RESAMPLES)


# --------------------------------------------------------------------------- paired comparison


def test_paired_join_is_exact(paired: PairedStatistics) -> None:
    assert [k.index for k in paired.missing_from_a] == [2]
    assert [k.index for k in paired.missing_from_b] == [4, 6]
    assert [k.index for k in paired.paired] == [i for i in range(15) if i not in {2, 4, 6}]
    o = paired.overall
    assert (o.missing_from_a, o.missing_from_b) == ((2,), (4, 6))
    for m in o.metrics:
        assert m.paired == 12 and (m.missing_from_a, m.missing_from_b) == (1, 2)
        assert {i for i, _ in m.deltas} <= set(o.paired_indices)
        assert m.statistics.n == len(m.deltas) == m.paired - m.undefined
    s = paired.scenario(IDS[1])
    assert (s.paired_indices, s.missing_from_a, s.missing_from_b) == ((5, 7, 8, 9), (), (6,))


def test_delta_is_a_minus_b(paired: PairedStatistics, paired_a: EvaluationResult, paired_b: EvaluationResult) -> None:
    for name in ("evacuated_rate", "mean_person_time_to_safety", "mean_exposure"):
        a, b = exact(paired_a, name), exact(paired_b, name)
        m = paired.overall.metric(name)
        assert m.deltas == tuple((i, a[i] - b[i]) for i in paired.overall.paired_indices)  # type: ignore[operator]
        assert m.statistics == describe([d for _, d in m.deltas])


def test_win_loss_tie_follow_each_metric_direction(paired: PairedStatistics) -> None:
    for m in paired.overall.metrics:
        assert m.delta_positive + m.delta_negative + m.delta_zero == len(m.deltas)
        if m.direction is MetricDirection.HIGHER_IS_BETTER:
            assert (m.a_better, m.b_better, m.ties) == (m.delta_positive, m.delta_negative, m.delta_zero)
        elif m.direction is MetricDirection.LOWER_IS_BETTER:
            assert (m.a_better, m.b_better, m.ties) == (m.delta_negative, m.delta_positive, m.delta_zero)
        else:
            assert (m.a_better, m.b_better, m.ties) == (None, None, None)
    # A three-step delayed start makes evacuation slower on at least one paired episode.
    slower = paired.overall.metric("mean_person_time_to_safety")
    assert slower.a_better is not None and slower.a_better > 0


def test_known_signs_and_swapped_models(tiny_greedy: EvaluationResult, tiny_stay: EvaluationResult, tiny: Suite) -> None:
    ab = compare_results(tiny_greedy, tiny_stay, tiny, resamples=RESAMPLES)
    ba = compare_results(tiny_stay, tiny_greedy, tiny, resamples=RESAMPLES)
    # Greedy evacuates everybody and staying evacuates nobody: A - B = 1 on every episode.
    evac = ab.overall.metric("evacuated_rate")
    assert [d for _, d in evac.deltas] == [F(1)] * 5
    assert (evac.a_better, evac.b_better, evac.ties) == (5, 0, 0)
    # Lower is better: a negative delta is a win for A.
    slower = ab.overall.metric("mean_person_time_to_safety")
    assert all(d < 0 for _, d in slower.deltas) and (slower.a_better, slower.b_better) == (5, 0)
    casualty = ab.overall.metric("casualty_rate")
    assert all(d <= 0 for _, d in casualty.deltas) and casualty.b_better == 0
    for x, y in zip(ab.overall.metrics, ba.overall.metrics, strict=True):
        assert tuple((i, -d) for i, d in x.deltas) == y.deltas
        assert (x.a_better, x.b_better, x.ties) == (y.b_better, y.a_better, y.ties)
        assert (x.delta_positive, x.delta_negative) == (y.delta_negative, y.delta_positive)
    assert ab.digest() != ba.digest()


def test_descriptive_metrics_have_no_winner(paired: PairedStatistics) -> None:
    for group in (paired.overall, *paired.scenarios, *paired.families):
        for name in ("exits_used", "max_exit_share", "exit_balance"):
            m = group.metric(name)
            assert m.direction is MetricDirection.DESCRIPTIVE
            assert m.a_better is None and m.b_better is None and m.ties is None


def test_paired_ci_resamples_deltas(paired: PairedStatistics) -> None:
    m = paired.overall.metric("evacuated_rate")
    ci = m.mean_delta_ci
    assert ci is not None and ci.statistic == "mean_delta"
    mean = m.statistics.mean
    assert mean is not None
    assert min(d for _, d in m.deltas) <= ci.lower <= mean <= ci.upper <= max(d for _, d in m.deltas)


def test_identical_models_have_zero_deltas(tiny_greedy: EvaluationResult, tiny: Suite) -> None:
    twin = dataclasses.replace(tiny_greedy, model_id="greedy-twin")
    c = compare_results(tiny_greedy, twin, tiny, resamples=RESAMPLES)
    for m in c.overall.metrics:
        assert all(d == 0 for _, d in m.deltas)
        if m.direction is not MetricDirection.DESCRIPTIVE:
            assert m.a_better == m.b_better == 0 and m.ties == len(m.deltas)


def test_different_manifests_are_not_compared(tiny_greedy: EvaluationResult, tiny: Suite) -> None:
    other = run_result(tiny, build_evaluation_manifest(tiny, (1, 2, 3, 4, 6)), "other", greedy_exit_policy)
    with pytest.raises(EvaluationError, match="incompatible evaluation manifests"):
        compare_results(tiny_greedy, other, tiny, resamples=RESAMPLES)


def test_no_cross_metric_winner(paired: PairedStatistics) -> None:
    doc = paired.to_dict()
    assert not {"winner", "score", "rank", "overall_winner"} & set(doc)
    assert not {"winner", "score"} & set(doc["overall"])


# --------------------------------------------------------------------------- determinism and versions


def golden_digests() -> list[tuple[str, str]]:
    """Statistics digests for a tiny fixed input. Called in-process and from fresh interpreters."""
    suite = subset(load_suite(SUITE_DIR), (IDS[0],))
    manifest = build_evaluation_manifest(suite, (1, 2, 3, 4, 5))
    a = run_result(suite, manifest, "greedy", greedy_exit_policy)
    b = run_result(suite, manifest, "stay", stay_policy, skip=frozenset({3}))
    return [
        ("aggregate", aggregate_result(a, suite, resamples=RESAMPLES).digest()),
        ("paired", compare_results(a, b, suite, resamples=RESAMPLES).digest()),
    ]


# Pinned for STATISTICS_VERSION 1.0. A change here means the statistics output
# changed: bump STATISTICS_VERSION and update the digests on purpose.
GOLDEN_STATISTICS_DIGESTS = {
    "aggregate": "ee5e6b8f179a90aa30d65380297404d90df0a9c91a70c569e9e7a07f6e8abe31",
    "paired": "f7a2726b2d6f2177266ec877a59e3f541184b65bb03e596f2001e96c95339d13",
}

_SUBPROCESS_SCRIPT = """
import sys
sys.path[:0] = [{src!r}, {tests!r}]
from test_statistics import golden_digests
for name, digest in golden_digests():
    print(name, digest)
"""


def test_statistics_identical_across_processes_and_hash_seeds() -> None:
    script = _SUBPROCESS_SCRIPT.format(src=str(ROOT / "src"), tests=str(ROOT / "tests"))
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout.replace("\r\n", "\n"))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0] == "".join(f"{name} {digest}\n" for name, digest in golden_digests())
    for line in outputs[0].splitlines():
        name, digest = line.split()
        assert digest == GOLDEN_STATISTICS_DIGESTS[name], f"{name}: {digest}"


def test_versions_and_public_hashes_are_unchanged(public: Suite) -> None:
    assert (SIMULATOR_VERSION, OBSERVATION_VERSION, SCHEMA_VERSION, METRICS_VERSION) == ("0.2.0", "1.0", "1.0", "1.0")
    assert EVALUATION_RESULT_VERSION == "1.0"
    assert public.manifest.suite_hash() == SUITE_HASH
    assert {s.scenario_id: s.content_hash() for s in public.scenarios} == PINNED_HASHES
