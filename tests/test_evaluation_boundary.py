"""The open task remains usable; official construction cannot fall back to it."""
import zipfile
from pathlib import Path
import pytest
import emberbench
from emberbench import composition
from emberbench.errors import GenerationError
from emberbench.leakguard import check_public_payload, check_distribution


def test_public_task_and_scoring_remain_open() -> None:
    assert callable(emberbench.generate_suite)
    assert callable(emberbench.score_result)
    assert callable(emberbench.verify_evaluation_provenance)
    assert emberbench.SCORING_VERSION == "1.0"


@pytest.mark.parametrize("name", ["compose_suite", "compose_pool", "freeze_composed_suite", "load_composed_suite", "schedule_composed_worlds"])
def test_legacy_official_construction_fails_closed(name: str) -> None:
    with pytest.raises(GenerationError, match="evaluator-private"):
        getattr(composition, name)()


def test_no_public_composition_defaults() -> None:
    assert not hasattr(composition, "V1_DESIGN")
    assert not hasattr(composition, "V1_RECIPES")
    assert not hasattr(emberbench, "SuiteDesign")
    assert len(composition.rule_set_digest()) == 64


def test_known_recipe_under_innocent_filename_is_rejected() -> None:
    assert check_public_payload("src/emberbench/helper.py", b"V1_DESIGN = {}")
    assert check_public_payload("docs/help.md", b'parameters: {"candidate_pool_size": 12}')
    assert not check_public_payload("docs/scoring.md", b"Public component weights and scoring gates remain transparent.")
    assert not check_public_payload("src/emberbench/test.py", b'example = "V1_DESIGN = {}"')


def test_wheel_content_cannot_bypass_path_allowlist(tmp_path: Path) -> None:
    wheel = tmp_path / "emberbench-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("emberbench/helper.py", "V1_RECIPES = ()")
    assert any("construction" in v.reason for v in check_distribution(wheel))
