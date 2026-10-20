"""Evaluation-suite design: environment-world identity, the unique-world manifest and the design tool.

The tool is ``tools/analyze_suite_design.py``. It uses only the official EMBER
Score (``emberbench.scoring``); these tests pin its world identity, capacity
bound, calibration rules, selection rules and leakage guarantees.
"""

import ast
import inspect
import random
import subprocess
from pathlib import Path

import pytest
from conftest import layout

from analyze_suite_design import (
    DEVELOPMENT_SEED_DOMAIN,
    FINAL_FAMILIES,
    PUBLIC_SUITE_DIR,
    Candidate,
    PanelRun,
    WorldRecord,
    allocate,
    dedupe_schedule,
    development_seed,
    distinct_world_seeds,
    final_design,
    official_run,
    profile_scenario,
    select_suite,
)
from emberbench.calibration import (
    ADMITTED,
    CALIBRATION_PANEL,
    CALIBRATION_PANEL_VERSION,
    EASY_MIN_BEST,
    FLOOR_AGENT,
    HARD_MIN_BEST,
    MODERATE_MIN_BEST,
    calibrate,
    capacity_ceiling,
    capacity_savable,
    difficulty_band,
    _glover,
)
from emberbench.families import FireEvidence as FireProfile
from emberbench import (
    BASELINE_IDS,
    ENVIRONMENT_WORLD_VERSION,
    EvaluationError,
    FireConfig,
    FireModelName,
    ScenarioFamily,
    TerminationConfig,
    agent_policy,
    baseline_factory,
    build_evaluation_manifest,
    build_unique_world_manifest,
    environment_world_digest,
    fire_world_digest,
    freeze_environment_pack,
    load_environment_pack,
    load_suite,
    run_episode,
    score_episode,
)
from emberbench.evaluation import verify_manifest
from emberbench.leakguard import forbidden_reason
from emberbench.oracle import fire_trajectory, privileged_escape_bound
from test_suite import SUITE_HASH

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "analyze_suite_design.py"
SUITE = load_suite(PUBLIC_SUITE_DIR)
SCENARIOS = {s.scenario_id: s for s in SUITE.scenarios}
SPARSE = SCENARIOS["v1-pub-01-single-exit-sparse"]
STOCHASTIC = SCENARIOS["v1-pub-10-route-invalidation"]
DETERMINISTIC_IDS = sorted(sid for sid, s in SCENARIOS.items() if s.fire.model == FireModelName.DETERMINISTIC)
STOCHASTIC_IDS = sorted(sid for sid in SCENARIOS if sid not in DETERMINISTIC_IDS)

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

# Three people next to one exit that burns at step 2. Alone, each can escape (M* = 3); one exit
# cell lets at most one person out per step, so at most two can leave before it burns (M_cap = 2).
CROWDED_EXIT = layout(
    [
        "#####",
        "#abF#",
        "#cE.#",
        "#####",
    ],
    scenario_id="crowded-exit",
    fire=FireConfig(model=FireModelName.DETERMINISTIC, spread_interval=1),
    termination=TerminationConfig(max_steps=20),
)


def _seeds(label: str, n: int) -> list[int]:
    return [development_seed(f"test/{label}/{i}") for i in range(n)]


# --------------------------------------------------------------------------- environment-world identity


def test_deterministic_fire_is_one_world_for_every_seed() -> None:
    seeds = _seeds("deterministic", 100)
    worlds = {environment_world_digest(SPARSE, s) for s in seeds}
    assert len(worlds) == 1
    # PR #32's seed-bound fire_world_digest is unchanged: it still separates the seeds.
    assert len({fire_world_digest(SPARSE, s) for s in seeds[:10]}) == 10
    dedup = dedupe_schedule(SCENARIOS, [(SPARSE.scenario_id, s) for s in seeds])
    assert (len(dedup.episodes), len(dedup.unique), dedup.duplicate_count) == (100, 1, 99)
    assert distinct_world_seeds(SPARSE, seeds, 8) == (seeds[0],)


def test_stochastic_duplicate_fire_worlds_are_one_world() -> None:
    assert SEALED.fire.model == FireModelName.STOCHASTIC
    seeds = list(range(40))
    assert len({environment_world_digest(SEALED, s) for s in seeds}) == 1
    assert distinct_world_seeds(SEALED, seeds, 4) == (0,)


def test_world_digest_follows_the_fire_trajectory_not_the_seed() -> None:
    seeds = list(range(30))
    keys: list[tuple[str, int]] = [("stochastic", s) for s in seeds] + [("sealed", s) for s in seeds[:5]]
    scenario = {"stochastic": STOCHASTIC, "sealed": SEALED}
    fires = {k: fire_trajectory(scenario[k[0]], k[1]) for k in keys}
    worlds = {k: environment_world_digest(scenario[k[0]], k[1]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            same_fire = a[0] == b[0] and fires[a] == fires[b]
            assert (worlds[a] == worlds[b]) == same_fire
    chosen = distinct_world_seeds(STOCHASTIC, seeds, 5)
    assert len(chosen) == 5 and len({environment_world_digest(STOCHASTIC, s) for s in chosen}) == 5


def test_world_digests_are_deterministic_and_separate_scenarios() -> None:
    first = [environment_world_digest(s, 7) for s in SUITE.scenarios]
    assert first == [environment_world_digest(s, 7) for s in SUITE.scenarios]
    assert all(len(w) == 64 and int(w, 16) >= 0 for w in first)
    assert len(set(first)) == len(first)
    assert ENVIRONMENT_WORLD_VERSION == "1.0"


@pytest.mark.parametrize("seed", [3, 4])
def test_model_identity_does_not_change_the_world(seed: int) -> None:
    # The digest takes no model argument; this checks the premise: every policy sees the same exogenous fire.
    trajectory = fire_trajectory(STOCHASTIC, seed)
    for bid in ("random-legal-v1", "nearest-exit-greedy-v1", "congestion-aware-v1"):
        record = run_episode(STOCHASTIC, agent_policy(baseline_factory(bid)()), seed=seed)
        burning = (record.initial_state.burning, *(step.state.burning for step in record.steps))
        assert burning == trajectory[: len(burning)]


# --------------------------------------------------------------------------- the unique-world manifest


def test_unique_world_manifest_schedules_each_world_once(tmp_path: Path) -> None:
    seeds = _seeds("manifest", 12)
    manifest = build_unique_world_manifest(SUITE, seeds, stochastic_worlds=3)
    verify_manifest(manifest, SUITE)
    assert manifest == build_unique_world_manifest(SUITE, seeds, stochastic_worlds=3)
    assert [e.index for e in manifest.episodes] == list(range(len(manifest.episodes)))
    assert len(manifest.episodes) == len(DETERMINISTIC_IDS) + 3 * len(STOCHASTIC_IDS)
    dedup = dedupe_schedule(SCENARIOS, [(e.scenario_id, e.seed) for e in manifest.episodes])
    assert dedup.duplicate_count == 0
    for sid in DETERMINISTIC_IDS:
        assert [e.seed for e in manifest.episodes if e.scenario_id == sid] == [seeds[0]]
    for sid in STOCHASTIC_IDS:
        assert [e.seed for e in manifest.episodes if e.scenario_id == sid] == list(distinct_world_seeds(SCENARIOS[sid], seeds, 3))
    # The nominal builder is unchanged: the full scenario x seed product, most of it repeated worlds.
    nominal = build_evaluation_manifest(SUITE, seeds)
    assert len(nominal.episodes) == len(SCENARIOS) * len(seeds)
    assert len(dedupe_schedule(SCENARIOS, [(e.scenario_id, e.seed) for e in nominal.episodes]).unique) <= len(DETERMINISTIC_IDS) + 12 * len(STOCHASTIC_IDS)
    # The manifest freezes into a verifiable environment pack.
    pack = freeze_environment_pack(SUITE, manifest, tmp_path / "pack", pack_id="test-unique-worlds")
    assert load_environment_pack(tmp_path / "pack").manifest == pack.manifest == manifest


def test_unique_world_manifest_fails_closed() -> None:
    seeds = _seeds("short", 2)
    with pytest.raises(EvaluationError):
        build_unique_world_manifest(SUITE, seeds, stochastic_worlds=3)  # only two candidate worlds
    for bad in (0, -1):
        with pytest.raises(EvaluationError):
            build_unique_world_manifest(SUITE, seeds, stochastic_worlds=bad)
    with pytest.raises(EvaluationError):
        build_unique_world_manifest(SUITE, [], stochastic_worlds=1)


def test_world_builders_and_selection_take_no_model_input() -> None:
    for fn in (environment_world_digest, build_unique_world_manifest, select_suite, calibrate, dedupe_schedule):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"model", "model_id", "agent", "policy", "result", "results"}, fn.__name__


# --------------------------------------------------------------------------- capacity bound


def test_capacity_bound_is_below_single_person_savability_when_an_exit_is_shared() -> None:
    savable, cap = capacity_savable(CROWDED_EXIT, 0)
    assert savable == privileged_escape_bound(CROWDED_EXIT, 0).max_evacuated == 3
    assert cap == 2
    for bid in BASELINE_IDS:
        run, _ = official_run(CROWDED_EXIT, 0, bid, baseline_factory(bid))
        assert run.evacuated <= cap
    # Without the bound, the best possible score would look like an avoidable loss of one person.
    ceiling = capacity_ceiling(3, savable, cap)
    assert ceiling is not None and ceiling < 100.0


def test_capacity_bound_is_exact_on_an_uncontested_world() -> None:
    savable, cap = capacity_savable(SPARSE, 0)
    assert cap == savable == privileged_escape_bound(SPARSE, 0).max_evacuated


def test_glover_greedy_is_optimal_on_small_cases() -> None:
    assert _glover([], [0, 1, 1]) == 0
    assert _glover([(1, 1), (1, 1), (1, 1)], [0, 1]) == 1
    assert _glover([(1, 2), (1, 1), (2, 3)], [0, 1, 1, 1]) == 3
    assert _glover([(1, 3), (1, 3), (1, 3)], [0, 1, 0, 1]) == 2
    assert _glover([(2, 2), (1, 3)], [0, 0, 2, 0]) == 2


def test_capacity_ceiling_uses_the_official_score() -> None:
    assert capacity_ceiling(10, 0, 0) is None
    assert capacity_ceiling(10, 8, 8) == 100.0
    one = capacity_ceiling(10, 10, 9)
    two = capacity_ceiling(10, 10, 8)
    assert one is not None and two is not None and 0 < two < one < 100
    assert capacity_ceiling(10, 10, 5) == 0.0  # the official catastrophic gate at 50% avoidable loss


# --------------------------------------------------------------------------- official scoring and calibration


def test_panel_runs_are_official_scores() -> None:
    assert len(CALIBRATION_PANEL) == 4 and FLOOR_AGENT not in CALIBRATION_PANEL
    assert set(CALIBRATION_PANEL) <= set(BASELINE_IDS) and CALIBRATION_PANEL_VERSION == "1.0"
    for bid in CALIBRATION_PANEL:
        run, scored = official_run(STOCHASTIC, 5, bid, baseline_factory(bid))
        again = score_episode(STOCHASTIC, run_episode(STOCHASTIC, agent_policy(baseline_factory(bid)()), seed=5))
        assert run.score == float(scored.score) == float(again.score)
        assert run.status == again.status.value


def test_tool_has_no_provisional_or_private_score() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "provisional" not in source.lower()
    assert "from emberbench.scoring import" in source
    for doc in ("evaluation-suite-design.md", "difficulty-calibration.md", "suite-design-results.md"):
        assert "provisional" not in (REPO_ROOT / "docs" / doc).read_text(encoding="utf-8").lower(), doc


def test_difficulty_band_thresholds() -> None:
    assert (EASY_MIN_BEST, MODERATE_MIN_BEST, HARD_MIN_BEST) == (80.0, 30.0, 10.0)
    assert [difficulty_band(b) for b in (100, 80, 79.9, 30, 29.9, 10, 9.9, 0)] == [
        "easy", "easy", "moderate", "moderate", "hard", "hard", "stress", "stress",
    ]


def _world(scores: tuple[float, float, float, float], savable: int = 10, cap: int | None = None) -> WorldRecord:
    cap = savable if cap is None else cap
    fire = FireProfile(0.0, 0.0, 0.0, 0.0)
    runs = tuple(PanelRun(a, s, "scored", None, None, 0, 0, (), 1, 0.0) for a, s in zip(CALIBRATION_PANEL, scores, strict=True))
    return WorldRecord(0, "0" * 64, savable, cap, capacity_ceiling(10, savable, cap), fire, runs)


def test_calibration_admission_rules_in_order() -> None:
    assert calibrate([_world((50, 50, 50, 50), savable=0)]).admission == "not_applicable"
    assert calibrate([_world((10, 20, 30, 40), cap=5)]).admission == "capacity_artifact"
    assert calibrate([_world((95, 99, 100, 100))]).admission == "saturated"
    assert calibrate([_world((0, 0.5, 0, 0.9))]).admission == "floor"
    assert calibrate([_world((60, 60.5, 60, 60.9))]).admission == "no_separation"
    c = calibrate([_world((10, 20, 30, 40)), _world((20, 30, 40, 50))])
    assert (c.admission, c.band, c.best, c.worst) == (ADMITTED, "moderate", 45.0, 15.0)
    # The privileged capacity bound is a bound only: a small shortfall does not change the band.
    assert calibrate([_world((0, 100, 90, 100), cap=9)]).band == "easy"


# --------------------------------------------------------------------------- selection and allocation


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
    fams = (ScenarioFamily.CROWD_100, ScenarioFamily.MAZE_OBSTACLE_ROUTING, ScenarioFamily.SINGLE_EXIT_LOW_DENSITY)
    quotas = {f: {"easy": 1, "moderate": 2, "hard": 2, "stress": 1} for f in fams}
    cands = _candidates()
    a = select_suite(cands, quotas, "key-a")
    shuffled = list(cands)
    random.Random(1).shuffle(shuffled)
    assert select_suite(shuffled, quotas, "key-a") == a
    hashes = [c.scenario_hash for _, c in a.chosen]
    assert len(hashes) == len(set(hashes))  # no scenario counts in two families
    assert all(fam in c.tags for fam, c in a.chosen)
    assert sum(sum(q.values()) for q in quotas.values()) == len(a.chosen) + sum(n for *_, n in a.shortfall)
    assert select_suite(cands, quotas, "key-b") != a  # the private key decides which candidates are taken


def test_selection_reports_shortfall_instead_of_borrowing_another_band() -> None:
    cands = [c for c in _candidates() if c.band != "stress"]
    sel = select_suite(cands, {ScenarioFamily.CROWD_100: {"stress": 2, "easy": 1}}, "k")
    assert sel.shortfall == ((ScenarioFamily.CROWD_100, "stress", 2),)
    assert [c.band for _, c in sel.chosen] == ["easy"]
    with pytest.raises(ValueError):
        select_suite([cands[0], cands[0]], {ScenarioFamily.CROWD_100: {"easy": 1}}, "k")


def test_band_allocation_and_final_design() -> None:
    shares = (("easy", 2 / 16), ("moderate", 4 / 16), ("hard", 6 / 16), ("stress", 4 / 16))
    assert allocate(16, shares) == (("easy", 2), ("moderate", 4), ("hard", 6), ("stress", 4))
    for n in (10, 12, 16, 20, 24):
        assert sum(k for _, k in allocate(n, shares)) == n
    design = final_design()
    assert sum(k for _, k in design.bands) == design.per_family
    assert design.worlds_per_family == design.per_family + design.stochastic_per_family * (design.stochastic_worlds - 1)
    assert len(FINAL_FAMILIES) == len(set(FINAL_FAMILIES))


def test_structural_profile_measures_known_public_challenges() -> None:
    corridor = profile_scenario(SCENARIOS["v1-pub-03-corridor-bottleneck"])
    assert corridor.passage_width == 1 and corridor.min_cut == 1 and corridor.bottleneck_severity == 8
    crowd = profile_scenario(SCENARIOS["v1-pub-09-crowd-100-split-flow"])
    assert crowd.min_cut == 3 and crowd.split_pressure is not None and crowd.split_pressure > 1
    maze = profile_scenario(SCENARIOS["v1-pub-08-maze-deceptive-exit"])
    assert maze.deceptive_share == 1.0 and maze.detour_ratio > 1.5


# --------------------------------------------------------------------------- leakage and pins


def test_development_seeds_are_public_and_never_private() -> None:
    assert development_seed("x") == development_seed("x") != development_seed("y")
    assert DEVELOPMENT_SEED_DOMAIN.endswith("/development/")
    # Neither the tool nor the world module embeds a seed literal.
    for path in (TOOL, REPO_ROOT / "src" / "emberbench" / "worlds.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        big = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, int) and n.value > 1_000_000]
        assert big == [], path


def test_new_research_paths_pass_the_leak_guard() -> None:
    for path in (
        "src/emberbench/worlds.py",
        "tools/analyze_suite_design.py",
        "tests/test_suite_design.py",
        "docs/evaluation-suite-design.md",
        "docs/difficulty-calibration.md",
        "docs/suite-design-results.md",
        "docs/figures/suite-ci-width.svg",
    ):
        assert forbidden_reason(path) is None, path
    assert forbidden_reason("docs/heldout-suite-design.md") is not None


def test_no_study_cache_or_generated_suite_is_committed() -> None:
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for name in tracked:
        stem = Path(name).name
        assert not stem.startswith(("pool-", "surrogates-", "trials-")), name
        assert not (name.startswith(("tools/", "docs/")) and name.endswith(".json")), name


def test_public_suite_hashes_are_unchanged() -> None:
    assert SUITE.manifest.suite_hash() == SUITE_HASH
