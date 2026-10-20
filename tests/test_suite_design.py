"""Evaluation-suite design helper (tools/analyze_suite_design.py): world identity, selection, difficulty, leakage."""

import ast
import random
from pathlib import Path

import pytest
from conftest import layout

from analyze_suite_design import (
    DEVELOPMENT_SEED_DOMAIN,
    PUBLIC_SUITE_DIR,
    REFERENCE_PANEL,
    Candidate,
    build_unique_world_manifest,
    dedupe_schedule,
    development_seed,
    difficulty_band,
    fire_trajectory_digest,
    profile_scenario,
    profile_world,
    provisional_score,
    run_outcome,
    select_distinct_seeds,
    select_suite,
    world_id,
)
from emberbench import (
    BASELINE_IDS,
    FireConfig,
    FireModelName,
    ScenarioFamily,
    TerminationConfig,
    agent_policy,
    baseline_factory,
    load_suite,
    run_episode,
)
from emberbench.leakguard import forbidden_reason
from emberbench.oracle import fire_trajectory
from test_suite import SUITE_HASH

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "analyze_suite_design.py"
SUITE = load_suite(PUBLIC_SUITE_DIR)
SCENARIOS = {s.scenario_id: s for s in SUITE.scenarios}
SPARSE = SCENARIOS["v1-pub-01-single-exit-sparse"]
STOCHASTIC = SCENARIOS["v1-pub-10-route-invalidation"]

# A stochastic scenario whose fire can never spread: every seed gives the same fire world.
SEALED = layout(
    [
        "#######",
        "#a...#E",
        "#....##",
        "#.##..#",
        "##F#..#",
        "#######",
    ],
    scenario_id="sealed-stochastic",
    fire=FireConfig(model=FireModelName.STOCHASTIC, spread_interval=1, spread_probability=0.5),
    termination=TerminationConfig(max_steps=30),
)


# --------------------------------------------------------------------------- unique worlds


def test_deterministic_scenario_is_one_world_for_every_seed() -> None:
    seeds = list(range(25))
    dedup = dedupe_schedule(SCENARIOS, [(SPARSE.scenario_id, s) for s in seeds])
    assert len(dedup.episodes) == 25
    assert len(dedup.unique) == 1
    assert dedup.duplicate_count == 24
    assert select_distinct_seeds(SPARSE, seeds, 8) == (0,)


def test_repeated_deterministic_episodes_are_not_double_counted_in_a_suite_schedule() -> None:
    schedule = [(sid, seed) for sid in sorted(SCENARIOS) for seed in (11, 12, 13)]
    dedup = dedupe_schedule(SCENARIOS, schedule)
    per = dedup.worlds_per_scenario()
    assert all(per[sid] == 1 for sid, s in SCENARIOS.items() if s.fire.model.value == "deterministic")
    assert per[STOCHASTIC.scenario_id] == 3
    assert len(dedup.unique) == 9 + 3


def test_unique_world_manifest_schedules_each_world_once() -> None:
    def candidates(scenario_id: str) -> list[int]:
        return [development_seed(f"test/{scenario_id}/{i}") for i in range(12)]

    manifest = build_unique_world_manifest(SUITE, candidates, stochastic_worlds=3)
    schedule = [(e.scenario_id, e.seed) for e in manifest.episodes]
    dedup = dedupe_schedule(SCENARIOS, schedule)
    assert dedup.duplicate_count == 0
    assert len(manifest.episodes) == 9 + 3
    assert manifest == build_unique_world_manifest(SUITE, candidates, stochastic_worlds=3)
    # The cross-product builder would schedule 9 x 3 duplicate deterministic episodes.
    crossed = dedupe_schedule(SCENARIOS, [(sid, s) for sid in SCENARIOS for s in (1, 2, 3)])
    assert crossed.duplicate_count == 9 * 2
    with pytest.raises(ValueError):
        build_unique_world_manifest(SUITE, lambda sid: [5, 5, 5], stochastic_worlds=3)


def test_equal_fire_trajectories_are_detected_across_seeds() -> None:
    assert SEALED.fire.model.value == "stochastic"
    digests = {fire_trajectory_digest(SEALED, s) for s in range(40)}
    assert len(digests) == 1
    assert len({world_id(SEALED, s) for s in range(40)}) == 1
    assert select_distinct_seeds(SEALED, range(40), 4) == (0,)


def test_world_id_follows_the_fire_trajectory_not_the_seed() -> None:
    seeds = list(range(30))
    fires = {s: fire_trajectory(STOCHASTIC, s) for s in seeds} | {("sealed", s): fire_trajectory(SEALED, s) for s in seeds[:5]}
    worlds = {s: world_id(STOCHASTIC, s) for s in seeds} | {("sealed", s): world_id(SEALED, s) for s in seeds[:5]}
    keys = list(fires)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            same_scenario = isinstance(a, int) == isinstance(b, int)
            same_fire = same_scenario and fires[a] == fires[b]
            assert (worlds[a] == worlds[b]) == same_fire
    chosen = select_distinct_seeds(STOCHASTIC, seeds, 5)
    assert len(chosen) == 5
    assert len({fire_trajectory_digest(STOCHASTIC, s) for s in chosen}) == 5


def test_world_ids_are_deterministic_and_content_addressed() -> None:
    first = [world_id(s, 7) for s in SUITE.scenarios]
    again = [world_id(s, 7) for s in SUITE.scenarios]
    assert first == again
    assert all(len(w) == 64 and int(w, 16) >= 0 for w in first)
    assert len(set(first)) == len(first)  # different scenarios are different worlds


@pytest.mark.parametrize("seed", [3, 4])
def test_model_identity_does_not_change_the_world(seed: int) -> None:
    # The world ID takes no model argument; here we check the premise behind that:
    # every model sees exactly the precomputed exogenous fire of the world.
    trajectory = fire_trajectory(STOCHASTIC, seed)
    for bid in ("random-legal-v1", "nearest-exit-greedy-v1", "congestion-aware-v1"):
        record = run_episode(STOCHASTIC, agent_policy(baseline_factory(bid)()), seed=seed)
        burning = (record.initial_state.burning, *(step.state.burning for step in record.steps))
        assert burning == trajectory[: len(burning)]


# --------------------------------------------------------------------------- selection and difficulty


def _candidates() -> list[Candidate]:
    F = ScenarioFamily
    fams = [F.CROWD_100, F.MAZE_OBSTACLE_ROUTING, F.SINGLE_EXIT_LOW_DENSITY]
    out = []
    for i in range(60):
        tags = frozenset({fams[i % 3], fams[(i // 3) % 3]})
        band = ("easy", "moderate", "hard", "stress")[i % 4]
        out.append(Candidate(f"gen-{i:04d}", f"{i:064x}", tags, band, i % 5 == 0))
    return out


def test_family_stratification_is_deterministic_and_order_independent() -> None:
    quotas = {f: {"easy": 1, "moderate": 2, "hard": 2, "stress": 1} for f in (ScenarioFamily.CROWD_100, ScenarioFamily.MAZE_OBSTACLE_ROUTING, ScenarioFamily.SINGLE_EXIT_LOW_DENSITY)}
    cands = _candidates()
    a = select_suite(cands, quotas, "key-a")
    shuffled = list(cands)
    random.Random(1).shuffle(shuffled)
    b = select_suite(shuffled, quotas, "key-a")
    assert a == b
    hashes = [c.scenario_hash for _, c in a.chosen]
    assert len(hashes) == len(set(hashes))  # no scenario counts in two families
    for fam, c in a.chosen:
        assert fam in c.tags
    assert sum(sum(q.values()) for q in quotas.values()) == len(a.chosen) + sum(n for *_, n in a.shortfall)
    assert select_suite(cands, quotas, "key-b") != a  # the private key decides which candidates are taken


def test_selection_reports_shortfall_instead_of_borrowing_another_band() -> None:
    cands = [c for c in _candidates() if c.band != "stress"]
    sel = select_suite(cands, {ScenarioFamily.CROWD_100: {"stress": 2, "easy": 1}}, "k")
    assert sel.shortfall == ((ScenarioFamily.CROWD_100, "stress", 2),)
    assert [c.band for _, c in sel.chosen] == ["easy"]
    with pytest.raises(ValueError):
        select_suite([cands[0], cands[0]], {ScenarioFamily.CROWD_100: {"easy": 1}}, "k")


def test_difficulty_band_rules() -> None:
    def panel(*v: float) -> dict[str, float]:
        return dict(zip(REFERENCE_PANEL, v, strict=True))

    assert difficulty_band(panel(100, 95, 90, 100)) == "easy"
    assert difficulty_band(panel(0, 100, 100, 100)) == "moderate"
    assert difficulty_band(panel(10, 10, 40, 89.9)) == "hard"
    assert difficulty_band(panel(0, 5, 10, 29.9)) == "stress"


def test_difficulty_assignment_is_reproducible_on_public_scenarios() -> None:
    def band(scenario_id: str) -> str:
        s = SCENARIOS[scenario_id]
        wp = profile_world(s, 0)
        scores = {}
        for bid in BASELINE_IDS:
            o = run_outcome(s, 0, bid, baseline_factory(bid), wp.escapable)
            assert o.score is not None
            scores[bid] = o.score
        return difficulty_band(scores)

    assert band("v1-pub-01-single-exit-sparse") == "easy"
    assert band("v1-pub-08-maze-deceptive-exit") == "moderate"
    assert band("v1-pub-08-maze-deceptive-exit") == "moderate"  # same result on a second run
    assert band("v1-pub-02-single-exit-dense") == "hard"


def test_structural_profile_measures_known_public_challenges() -> None:
    corridor = profile_scenario(SCENARIOS["v1-pub-03-corridor-bottleneck"])
    assert corridor.passage_width == 1 and corridor.min_cut == 1 and corridor.bottleneck_severity == 8
    crowd = profile_scenario(SCENARIOS["v1-pub-09-crowd-100-split-flow"])
    assert crowd.min_cut == 3 and crowd.split_pressure is not None and crowd.split_pressure > 1
    maze = profile_scenario(SCENARIOS["v1-pub-08-maze-deceptive-exit"])
    assert maze.deceptive_share == 1.0 and maze.detour_ratio > 1.5
    threatened = profile_world(SCENARIOS["v1-pub-06-threatened-exit"], 0)
    assert threatened.threatened_share > 0 and threatened.escapable == threatened.population


def test_provisional_score_charges_only_avoidable_losses() -> None:
    assert provisional_score(10, 0, 0, 10, 0) is None
    assert provisional_score(10, 10, 10, 0, 0) == 100.0
    assert provisional_score(10, 8, 8, 2, 0) == 100.0  # two people nobody could save
    assert provisional_score(10, 10, 5, 5, 0) == 0.0  # catastrophic gate
    one = provisional_score(10, 10, 9, 1, 0)
    stranded = provisional_score(10, 10, 9, 0, 1)
    assert one is not None and stranded is not None and 0 < one < stranded < 100


# --------------------------------------------------------------------------- leakage and pins


def test_development_seeds_are_public_and_never_private() -> None:
    assert development_seed("x") == development_seed("x") != development_seed("y")
    assert DEVELOPMENT_SEED_DOMAIN.endswith("/development/")
    # The tool embeds no seed literal: every seed is derived from a public label.
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    big = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, int) and n.value > 1_000_000]
    assert big == []


def test_new_research_paths_pass_the_leak_guard() -> None:
    for path in (
        "tools/analyze_suite_design.py",
        "tests/test_suite_design.py",
        "docs/evaluation-suite-design.md",
        "docs/difficulty-calibration.md",
        "docs/suite-design-results.md",
        "docs/figures/suite-ci-width.svg",
    ):
        assert forbidden_reason(path) is None, path
    # The name the task first suggested would be rejected by the leak guard.
    assert forbidden_reason("docs/heldout-suite-design.md") is not None


def test_public_suite_hashes_are_unchanged() -> None:
    assert SUITE.manifest.suite_hash() == SUITE_HASH
