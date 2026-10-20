"""Evaluator-side suite composition: design quotas, recipes, deterministic selection, top-up and the freeze record."""

import dataclasses
import inspect
import json
import random
from pathlib import Path

import pytest
from conftest import layout

from emberbench import ScenarioFamily, load_suite
from emberbench.calibration import CALIBRATION_PANEL, WorldMeasurement
from emberbench.composition import (
    INITIAL_POOL_SIZE,
    POOL_CAP,
    SUITE_FREEZE_RECORD_NAME,
    TOP_UP_BATCH_SIZE,
    V1_DESIGN,
    V1_INITIAL_PLAN,
    V1_RECIPES,
    V1_TOP_UP,
    CandidateOrigin,
    CompositionResult,
    MeasuredCandidate,
    QuotaCell,
    SuiteDesign,
    batch_seed,
    commit,
    compose_pool,
    compose_suite,
    eligible,
    freeze_composed_suite,
    generate_batch,
    load_composed_suite,
    measure_candidate,
    recipe_by_id,
    rule_set,
    schedule_composed_worlds,
    select_suite,
    top_up_recipe,
)
from emberbench.errors import GenerationError
from emberbench.families import FIRE_DEPENDENT_FAMILIES, PRIMARY_FAMILIES, STRUCTURAL_PRIMARY_FAMILIES, FireEvidence, structural_tags
from emberbench.leakguard import forbidden_reason
from emberbench.scenario import FireConfig, FireModelName, Scenario
from emberbench.structure import measure_proxies, profile_structure

F = ScenarioFamily
LOW = F.SINGLE_EXIT_LOW_DENSITY
SPARSE = ["E...........", "............", "...a........", "............", "......b....F", "............"]
DECEPTIVE = [".....E......", "###########.", "....abcd...E", "F..........."]
PUBLIC = Path(__file__).resolve().parent.parent / "scenarios" / "public" / "v1"
EVAL_SEEDS = [10**15 + 7919 * i for i in range(16)]
KEY = "test-selection-key"


def candidate(rows: list[str], i: int, best: float, *, stochastic: bool = False, ceiling: float = 100.0) -> MeasuredCandidate:
    s = layout(rows, scenario_id=f"c{i:04d}")
    if stochastic:
        s = dataclasses.replace(s, fire=FireConfig(model=FireModelName.STOCHASTIC, spread_interval=2, spread_probability=0.5))
    scores = (best, best - 5.0, best - 10.0, best - 20.0)
    w = WorldMeasurement(1, "0" * 64, 2, 2, ceiling, FireEvidence(0.0, 0.0, 0.0, 0.0), tuple(zip(CALIBRATION_PANEL, scores, strict=True)))
    return MeasuredCandidate(CandidateOrigin("unit", "1.0", 0, "unit", i), s, measure_proxies(s), (0,), (w,), None, 0.0)


def chosen_hashes(sel_chosen: tuple[tuple[QuotaCell, MeasuredCandidate], ...]) -> list[tuple[QuotaCell, str]]:
    return [(cell, c.scenario_hash) for cell, c in sel_chosen]


# --------------------------------------------------------------------------- design


def test_v1_design_is_the_frozen_pr33_design() -> None:
    d = V1_DESIGN
    assert d.families == PRIMARY_FAMILIES and len(d.families) == 10
    assert dict(d.bands) == {"easy": 2, "moderate": 4, "hard": 6, "stress": 4}
    assert (d.per_family, d.scenario_count, d.world_count) == (16, 160, 240)
    assert (d.stochastic_per_family, d.stochastic_worlds) == (4, 3)
    slots = d.slots()
    assert sum(1 for _, st in slots if st) == 4 and sum(1 for _, st in slots if not st) == 12
    stochastic_by_band = {b: sum(1 for bb, st in slots if st and bb == b) for b in ("easy", "moderate", "hard", "stress")}
    assert stochastic_by_band == {"easy": 0, "moderate": 1, "hard": 2, "stress": 1}
    q = d.quotas()
    assert len(q) == 70 and sum(q.values()) == 160
    assert all(sum(n for (f, b, _), n in q.items() if f is fam and b == band) == dict(d.bands)[band] for fam in d.families for band in dict(d.bands))


def test_suite_design_rejects_invalid_designs() -> None:
    with pytest.raises(ValueError):
        SuiteDesign((LOW,), (("moderate", 1), ("easy", 1), ("hard", 0), ("stress", 0)), 0, 1)
    with pytest.raises(ValueError):
        SuiteDesign((F.CENTRAL_IGNITION,), (("easy", 1), ("moderate", 0), ("hard", 0), ("stress", 0)), 0, 1)
    with pytest.raises(ValueError):
        SuiteDesign((LOW,), (("easy", 1), ("moderate", 0), ("hard", 0), ("stress", 0)), 2, 1)


# --------------------------------------------------------------------------- recipes


def test_recipe_catalog_covers_every_family_and_fire_model() -> None:
    ids = [r.recipe_id for r in V1_RECIPES]
    assert len(ids) == len(set(ids))
    for required in ("dense-single-stochastic", "split-flow-medium", "maze-crowded", "maze-stochastic", "crowd-stochastic",
                     "deceptive-exit", "exit-threat", "exit-threat-stochastic", "route-invalidation", "route-invalidation-stochastic"):  # fmt: skip
        assert required in ids
    for family in PRIMARY_FAMILIES:
        for stochastic in (False, True):
            r = top_up_recipe((family, "hard", stochastic))
            assert family in r.targets and r.stochastic == stochastic
    assert sum(n for _, n in V1_INITIAL_PLAN) == INITIAL_POOL_SIZE == 4000
    assert {rid for rid, _ in V1_INITIAL_PLAN} == set(ids)
    assert (TOP_UP_BATCH_SIZE, POOL_CAP) == (200, 10_000)


def test_top_up_table_has_one_dedicated_recipe_per_family_and_fire_model() -> None:
    assert len(V1_TOP_UP) == 20
    assert V1_TOP_UP[(F.SPLIT_FLOW_ACROSS_EXITS, False)] == "split-flow-medium"
    assert V1_TOP_UP[(F.MAZE_OBSTACLE_ROUTING, False)] == "maze-crowded"
    assert V1_TOP_UP[(F.ROUTE_INVALIDATION_REPLANNING, True)] == "route-invalidation-stochastic"
    for (family, stochastic), rid in V1_TOP_UP.items():
        r = recipe_by_id(rid)
        assert family in r.targets and r.stochastic == stochastic
        for band in ("easy", "stress"):
            assert top_up_recipe((family, band, stochastic)) is r
    assert rule_set()["selection_rules"]["top_up"]["recipes"] == sorted([f.value, st, rid] for (f, st), rid in V1_TOP_UP.items())


def test_recipes_define_distributions_not_scenarios() -> None:
    for r in V1_RECIPES:
        assert "scenario_count" not in r.config and "seed" not in json.dumps(r.config)
        assert r.config_for(3).scenario_count == 3


def test_split_flow_recipe_makes_medium_population_split_flow() -> None:
    b = generate_batch(recipe_by_id("split-flow-medium"), 11, 0, 6)
    for scenario, proxies in b.scenarios:
        assert len(scenario.people) < 100
        assert F.SPLIT_FLOW_ACROSS_EXITS in structural_tags(profile_structure(scenario, proxies))


@pytest.mark.parametrize("family", STRUCTURAL_PRIMARY_FAMILIES, ids=lambda f: f.value)
@pytest.mark.parametrize("stochastic", [False, True], ids=["deterministic", "stochastic"])
def test_targeted_recipes_hit_their_structural_family(family: ScenarioFamily, stochastic: bool) -> None:
    r = top_up_recipe((family, "easy", stochastic))
    b = generate_batch(r, 23, 0, 6)
    assert all((s.fire.model is FireModelName.STOCHASTIC) == stochastic for s, _ in b.scenarios)
    assert any(family in structural_tags(profile_structure(s, p)) for s, p in b.scenarios)


def test_fire_dependent_families_have_targeted_recipes_but_no_generator_label() -> None:
    for family in FIRE_DEPENDENT_FAMILIES:
        for stochastic in (False, True):
            r = top_up_recipe((family, "easy", stochastic))
            b = generate_batch(r, 5, 0, 4)
            assert all(not structural_tags(profile_structure(s, p)) & FIRE_DEPENDENT_FAMILIES for s, p in b.scenarios)


def test_recipe_batches_are_deterministic_and_seed_separated() -> None:
    r = recipe_by_id("open-sparse")
    a, again = generate_batch(r, 7, 0, 3), generate_batch(r, 7, 0, 3)
    assert a.generation_hash == again.generation_hash and a.scenarios == again.scenarios
    assert generate_batch(r, 7, 1, 3).generation_hash != a.generation_hash
    assert generate_batch(r, 8, 0, 3).generation_hash != a.generation_hash
    assert batch_seed(7, r, 0) != batch_seed(7, r, 1) != batch_seed(7, recipe_by_id("maze"), 1)
    assert a.seed_commitment == commit("generator-seed", batch_seed(7, r, 0))
    assert str(batch_seed(7, r, 0)) not in json.dumps(a.to_dict())


# --------------------------------------------------------------------------- selection


def test_selection_is_deterministic_and_order_independent() -> None:
    pool = [candidate(SPARSE, i, 90.0) for i in range(6)] + [candidate(SPARSE, 10 + i, 50.0) for i in range(3)]
    quotas = {(LOW, "easy", False): 2, (LOW, "moderate", False): 1}
    a = select_suite(pool, quotas, KEY)
    shuffled = pool[:]
    random.Random(3).shuffle(shuffled)
    assert chosen_hashes(select_suite(shuffled, quotas, KEY).chosen) == chosen_hashes(a.chosen)
    assert a.complete and len(a.chosen) == 3
    picks = {frozenset(h for _, h in chosen_hashes(select_suite(pool, quotas, f"key-{k}").chosen)) for k in range(8)}
    assert len(picks) > 1  # the private key, not the pool order, decides


def test_selection_never_borrows_a_band_or_a_fire_model() -> None:
    pool = [candidate(SPARSE, i, 50.0) for i in range(3)] + [candidate(SPARSE, 10, 90.0, stochastic=True)]
    sel = select_suite(pool, {(LOW, "easy", False): 1, (LOW, "moderate", False): 1}, KEY)
    assert sel.shortfall == (((LOW, "easy", False), 1),)
    assert [cell for cell, _ in sel.chosen] == [(LOW, "moderate", False)]


def test_capacity_artifacts_are_never_selected() -> None:
    artifact = candidate(SPARSE, 0, 90.0, ceiling=40.0)
    assert artifact.admission == "capacity_artifact" and not eligible(artifact)
    sel = select_suite([artifact], {(LOW, "easy", False): 1}, KEY)
    assert not sel.chosen and sel.shortfall == (((LOW, "easy", False), 1),)


def test_each_candidate_gets_exactly_one_primary_family() -> None:
    c = candidate(DECEPTIVE, 0, 90.0)
    assert {F.DECEPTIVE_NEAREST_EXIT, F.MAZE_OBSTACLE_ROUTING} <= c.tags
    sel = select_suite([c], {(F.MAZE_OBSTACLE_ROUTING, "easy", False): 1, (F.DECEPTIVE_NEAREST_EXIT, "easy", False): 1}, KEY)
    # Equal scarcity: the tie goes to the family name, so the result is fixed.
    assert [cell for cell, _ in sel.chosen] == [(F.DECEPTIVE_NEAREST_EXIT, "easy", False)]
    assert sel.shortfall == (((F.MAZE_OBSTACLE_ROUTING, "easy", False), 1),)


def test_top_up_keeps_the_previous_selection() -> None:
    first = [candidate(SPARSE, i, 90.0) for i in range(2)]
    quotas = {(LOW, "easy", False): 3}
    before = select_suite(first, quotas, KEY)
    assert before.shortfall == (((LOW, "easy", False), 1),)
    more = first + [candidate(SPARSE, 10 + i, 90.0) for i in range(5)]
    after = select_suite(more, quotas, KEY, previous=before)
    assert after.complete and chosen_hashes(after.chosen)[:2] == chosen_hashes(before.chosen)


def test_no_model_is_an_input_to_composition() -> None:
    for fn in (compose_pool, select_suite, measure_candidate, compose_suite, generate_batch):
        assert not set(inspect.signature(fn).parameters) & {"model", "agent", "submission", "policy", "factory", "scores"}


# --------------------------------------------------------------------------- the pool workflow and the freeze record

# One Easy deterministic and one Easy stochastic slot (the even-spread rule puts the stochastic slot last).
TINY = SuiteDesign((LOW,), (("easy", 2), ("moderate", 0), ("hard", 0), ("stress", 0)), stochastic_per_family=1, stochastic_worlds=2)
TINY_PLAN = (("open-sparse", 6), ("open-sparse-stochastic", 6))


@pytest.fixture(scope="module")
def pool() -> tuple[CompositionResult, str]:
    excluded = generate_batch(recipe_by_id("open-sparse"), 99, 0, 6).scenarios[0][0].content_hash()  # the first planned candidate
    public = [s.content_hash() for s in load_suite(PUBLIC).scenarios]
    result = compose_pool(
        pool_seed=99, evaluation_seeds=EVAL_SEEDS, selection_key=KEY, design=TINY, initial_plan=TINY_PLAN,
        top_up_batch=6, pool_cap=48, exclude_hashes=[excluded, *public],
    )  # fmt: skip
    return result, excluded


def test_pool_composition_fills_the_tiny_design(pool: tuple[CompositionResult, str]) -> None:
    result, excluded = pool
    assert result.selection.complete
    assert result.dropped.get("excluded_hash") == 1 and excluded not in {c.scenario_hash for c in result.candidates}
    cells = sorted(cell for cell, _ in result.selection.chosen)
    assert cells == sorted(TINY.quotas())
    for cell, c in result.selection.chosen:
        assert c.band == cell[1] and c.stochastic == cell[2] and cell[0] in c.tags
        assert len(c.worlds) == (2 if c.stochastic else 1) and len({w.world for w in c.worlds}) == len(c.worlds)


def test_composition_is_reproducible(pool: tuple[CompositionResult, str]) -> None:
    result, _ = pool
    a = compose_suite(result, suite_id="tiny-suite", selection_key=KEY)
    b = compose_suite(result, suite_id="tiny-suite", selection_key=KEY)
    assert a.record.digest() == b.record.digest() and a.manifest.suite_hash() == b.manifest.suite_hash()
    with pytest.raises(GenerationError, match="commitment"):
        compose_suite(result, suite_id="tiny-suite", selection_key="another-key")


def test_freeze_round_trip_and_world_schedule(pool: tuple[CompositionResult, str], tmp_path: Path) -> None:
    result, _ = pool
    composed = compose_suite(result, suite_id="tiny-suite", selection_key=KEY)
    frozen = freeze_composed_suite(composed, tmp_path / "frozen")
    assert (tmp_path / "frozen" / SUITE_FREEZE_RECORD_NAME).is_file()
    assert [s.scenario_id for s in frozen.suite.scenarios] == ["gen-0000", "gen-0001"]
    manifest = schedule_composed_worlds(frozen, EVAL_SEEDS)
    assert len(manifest.episodes) == TINY.world_count == 3
    with pytest.raises(GenerationError, match="commitment"):
        schedule_composed_worlds(frozen, EVAL_SEEDS[::-1])
    with pytest.raises(GenerationError, match="not empty"):
        freeze_composed_suite(composed, tmp_path / "frozen")


def test_freeze_record_contents(pool: tuple[CompositionResult, str]) -> None:
    result, _ = pool
    record = compose_suite(result, suite_id="tiny-suite", selection_key=KEY).record
    data = record.to_dict()
    versions = data["rule_set"]["versions"]
    for name in ("generator", "generator_config", "recipe_catalog", "family_rules", "selection_rules",
                 "calibration_panel_version", "difficulty_rules_version", "admission_rules_version"):  # fmt: skip
        assert name in versions
    assert data["batches"] and all("generator_seed_commitment" in b and "generation_hash" in b for b in data["batches"])
    assert set(data["commitments"]) == {"pool_seed", "evaluation_seeds", "selection_key"}
    for entry in data["selected"]:
        assert entry["primary_family"] in entry["tags"] and entry["band"] in ("easy", "moderate")
        assert all(len(w["environment_world_digest"]) == 64 for w in entry["worlds"])
    text = record.to_file_text()
    assert KEY not in text and not any(str(s) in text for s in EVAL_SEEDS) and '"99"' not in text
    public = record.public_commitment()
    assert set(public) >= {"suite_hash", "freeze_record_digest", "rule_set_digest", "cell_counts"}
    assert "selected" not in public and "batches" not in public
    assert data["rule_set"] == rule_set(TINY, V1_RECIPES)


def test_tampered_freeze_is_rejected(pool: tuple[CompositionResult, str], tmp_path: Path) -> None:
    result, _ = pool
    frozen = freeze_composed_suite(compose_suite(result, suite_id="tiny-suite", selection_key=KEY), tmp_path / "f")
    path = frozen.root / SUITE_FREEZE_RECORD_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["selected"][0]["primary_family"] = "crowd_100"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GenerationError):
        load_composed_suite(frozen.root)


def test_relabeling_keeps_every_world(pool: tuple[CompositionResult, str]) -> None:
    result, _ = pool
    composed = compose_suite(result, suite_id="tiny-suite", selection_key=KEY)
    for s, entry in zip(composed.scenarios, composed.record.selected, strict=True):
        assert isinstance(s, Scenario) and s.content_hash() == entry.scenario_hash != entry.candidate_hash


def test_the_freeze_record_is_a_forbidden_repository_file() -> None:
    assert forbidden_reason(f"scenarios/public/{SUITE_FREEZE_RECORD_NAME}") is not None
    assert forbidden_reason(f"docs/{SUITE_FREEZE_RECORD_NAME}") is not None


def test_development_pool_filled_every_cell_under_the_current_rules() -> None:
    """The committed coverage evidence (tools/validate_generator_coverage.py, development seeds) matches the frozen rules."""
    from emberbench.composition import rule_set_digest

    data = json.loads((Path(__file__).parent / "fixtures" / "generator-coverage-summary.json").read_text(encoding="utf-8"))
    assert data["rule_set_digest"] == rule_set_digest(), "rules changed: rerun tools/validate_generator_coverage.py"
    cells = {(c["family"], c["band"], c["stochastic"]): c for c in data["cells"]}
    assert set(cells) == {(f.value, b, st) for f, b, st in V1_DESIGN.quotas()}
    assert all(c["quota"] == V1_DESIGN.quotas()[(F(c["family"]), c["band"], c["stochastic"])] for c in data["cells"])
    assert all(c["selected"] == c["quota"] <= c["eligible"] for c in data["cells"])
    assert {(c["family"], c["band"]) for c in data["cells"]} == {(f.value, b) for f in PRIMARY_FAMILIES for b in ("easy", "moderate", "hard", "stress")}
    assert (data["selected"], data["shortfall_slots"], data["candidates"]) == (160, 0, 4000)
