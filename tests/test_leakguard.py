"""Repository and package leak guard: only public assets in the checkout, the wheel and the sdist."""

import tarfile
import zipfile
from pathlib import Path

import pytest

from emberbench.leakguard import (
    PUBLIC_ASSET_ROOTS,
    check_distribution,
    check_repository,
    check_repository_paths,
    check_sdist_members,
    check_wheel_members,
    forbidden_reason,
    repository_paths,
)
from test_suite import REPO_ROOT

FORBIDDEN_PATHS = [
    "hidden/v1/manifest.json",
    "Hidden/readme.md",
    "scenarios/heldout/a.json",
    "scenarios/held_out/a.json",
    "tests/fixtures/held-out/a.json",
    "scored/results.csv",
    "docs/scored-run.md",
    "private-seeds.txt",
    "tools/private_seeds_2026.json",
    "docs/secret-manifest.json",
    "docs/Secret_Manifest_v2.md",
    "production-eval/run.json",
    "src/emberbench/production_eval.py",
    "tests/fixtures/generation.json",
    "tests/fixtures/evaluation-manifest.json",
    "docs/provenance.json",
    "tests/fixtures/run/result.json",
    "tests/fixtures/suite-freeze.json",
    "scenarios/public/v1/suite-freeze.json",
    "tools/evaluation-run/notes.md",
    "tools/batch.seed",
    "docs/signing.key",
    "tests/fixtures/cert.pem",
    "src/emberbench/private/__init__.py",
]

ALLOWED_PATHS = [
    "README.md",
    "pyproject.toml",
    ".github/workflows/tests.yml",
    "docs/integrity.md",
    "scenarios/public/v1/manifest.json",
    "tests/fixtures/scenarios/narrow_corridor.json",
    "tests/test_hidden_names.py",  # a component must *be* a forbidden name, not merely mention one
    "src/emberbench/integrity.py",
    "tools/check_public_assets.py",
    "tools/validate_generator_coverage.py",
    "docs/generator-coverage-results.md",
    "src/emberbench/composition.py",
]


@pytest.mark.parametrize("path", FORBIDDEN_PATHS)
def test_private_evaluation_paths_are_rejected(path: str) -> None:
    assert forbidden_reason(path) is not None
    assert [v.path for v in check_repository_paths([path])] == [path]


@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_public_paths_are_allowed(path: str) -> None:
    assert forbidden_reason(path) is None
    assert check_repository_paths([path]) == []


@pytest.mark.parametrize(
    "path,reason",
    [
        ("data/readme.md", "top-level directory"),
        ("notes.txt", "top-level file"),
        ("docs/example.json", "benchmark asset files"),
        ("src/emberbench/defaults.json", "benchmark asset files"),
        ("scenarios/v2/a.md", "only scenarios/public/"),
        ("/etc/passwd", "relative path"),
        ("docs/../../x.md", "relative path"),
    ],
)
def test_allowlist_rejects_everything_else(path: str, reason: str) -> None:
    (violation,) = check_repository_paths([path])
    assert reason in violation.reason


def test_asset_roots_are_public() -> None:
    assert PUBLIC_ASSET_ROOTS == ("scenarios/public/", "tests/fixtures/")
    for root in PUBLIC_ASSET_ROOTS:
        assert forbidden_reason(root + "x.json") is None


def test_repository_holds_only_public_assets() -> None:
    paths = repository_paths(REPO_ROOT)
    assert "src/emberbench/__init__.py" in paths and "scenarios/public/v1/manifest.json" in paths
    assert check_repository(REPO_ROOT) == []


def test_wheel_and_sdist_member_rules() -> None:
    wheel = ["emberbench/__init__.py", "emberbench/py.typed", "emberbench-0.1.0.dist-info/METADATA", "emberbench-0.1.0.dist-info/RECORD"]
    assert check_wheel_members(wheel) == []
    for planted in (
        "emberbench/heldout/suite.json",
        "emberbench/defaults.json",
        "emberbench/hidden_suite.py",
        "hidden/x.py",
        "emberbench/sub/module.py",
        "emberbench-0.1.0.dist-info/extra/generation.json",
    ):
        assert check_wheel_members([*wheel, planted]), planted
    sdist = ["emberbench-0.1.0/PKG-INFO", "emberbench-0.1.0/pyproject.toml", "emberbench-0.1.0/src/emberbench/__init__.py"]
    assert check_sdist_members(sdist) == []
    for planted in ("emberbench-0.1.0/scored/a.json", "emberbench-0.1.0/data/x.json", "other/x.py", "emberbench-0.1.0/private-seeds.txt"):
        assert check_sdist_members([*sdist, planted]), planted


def test_built_distributions_hold_no_heldout_assets(built_distributions: tuple[Path, Path]) -> None:
    wheel, sdist = built_distributions
    assert check_distribution(wheel) == []
    assert check_distribution(sdist) == []
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    assert "emberbench/integrity.py" in names and "emberbench/py.typed" in names
    assert not [n for n in names if n.endswith(".json")]
    with tarfile.open(sdist, "r:gz") as t:
        members = [m.name for m in t.getmembers() if not m.isdir()]
    assert any(m.endswith("src/emberbench/integrity.py") for m in members)
    # Positive control: the same wheel with a planted held-out file fails.
    assert check_wheel_members([*names, "emberbench/heldout/manifest.json"])
