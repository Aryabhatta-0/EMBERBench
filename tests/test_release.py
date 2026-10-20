"""Release hardening: one package version, pinned tools, frozen digests, portable outputs, package metadata.

See docs/versioning.md and docs/release-checklist.md.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

import emberbench
from emberbench import (
    BENCHMARK_REVISION,
    RunRecord,
    build_evaluation_manifest,
    load_evaluation_manifest,
    load_evaluation_result,
    load_scenario,
    load_suite,
    run_episode,
    run_evaluation,
)
from emberbench._version import __version__ as source_version
from emberbench.canonical import sha256_hex
from emberbench.cli import main as cli_main
from emberbench.cli import version_info
from emberbench.generator import GeneratorConfig, freeze_generated_suite, generate_suite, load_generated_suite
from conftest import greedy_exit_policy
from release_digests import REGISTRY_PATH, load_registry
from test_baselines import GOLDEN_CONFIG_DIGESTS, GOLDEN_RESULT_DIGESTS
from test_determinism import GOLDEN_DIGESTS
from test_evaluation import PUBLIC_DEV_MANIFEST_HASH, subset
from test_generator import BASE_CONFIG, BATCH, GOLDEN_CONFIG_HASH, GOLDEN_GENERATION_HASH, GOLDEN_SUITE_HASH, SEED
from test_statistics import GOLDEN_STATISTICS_DIGESTS
from test_stochastic_hazard import GOLDEN as GOLDEN_FIRE_TRACES
from test_suite import PINNED_HASHES, REPO_ROOT, SUITE_DIR, SUITE_HASH

TOOLS = REPO_ROOT / "tools"
DOCS = REPO_ROOT / "docs"
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
REGISTRY = load_registry()
SPARSE = "v1-pub-01-single-exit-sparse"

# Version constants that are not a separate line of `emberbench version`, with the reason.
NOT_REPORTED = {"GENERATED_SUITE_VERSION": "moves with GENERATOR_VERSION; written into every generated suite manifest"}


def _tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_release_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _version_constants() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted((REPO_ROOT / "src" / "emberbench").glob("*.py")):
        for name, value in re.findall(r'^([A-Z][A-Z_]*_VERSION|BENCHMARK_REVISION) = "([^"]+)"$', path.read_text(encoding="utf-8"), re.M):
            assert name not in found, f"{name} is defined twice"
            found[name] = value
    return found


def _crlf_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    for path in destination.rglob("*.json"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    return destination


def _cli(*args: str) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return cli_main(list(args))


# --------------------------------------------------------------------------- package version


def test_package_version_has_one_source() -> None:
    project = PYPROJECT["project"]
    assert "version" not in project and project["dynamic"] == ["version"]
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "emberbench._version.__version__"}
    assert emberbench.__version__ == source_version
    assert re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)((a|b|rc)(0|[1-9][0-9]*))?", source_version)
    # No other module reads or writes the package version, so no artifact can hold it.
    users = sorted(p.name for p in (REPO_ROOT / "src" / "emberbench").glob("*.py") if "__version__" in p.read_text(encoding="utf-8"))
    assert users == ["__init__.py", "_version.py", "cli.py"]


def test_version_output_keeps_package_version_and_benchmark_revision_apart() -> None:
    info = version_info()
    assert info["package_version"] == source_version
    assert info["benchmark_revision"] == BENCHMARK_REVISION == "1"
    assert list(info)[0] == "package_version"


def test_every_component_version_is_reported_and_documented() -> None:
    constants = _version_constants()
    assert "SIMULATOR_VERSION" in constants and "BENCHMARK_REVISION" in constants
    reported = version_info()
    policy = (DOCS / "versioning.md").read_text(encoding="utf-8")
    for name, value in constants.items():
        if name not in NOT_REPORTED:
            assert reported.get(name.lower()) == value, f"{name} is missing from `emberbench version`"
        assert f"| `{name}` | `{value}` |" in policy, f"{name} = {value!r} is missing from docs/versioning.md"
    assert set(NOT_REPORTED) <= set(constants)


def test_changelog_records_the_v1_milestones() -> None:
    changelog = (DOCS / "changelog.md").read_text(encoding="utf-8")
    assert f"package `{source_version}`, benchmark revision `{BENCHMARK_REVISION}`" in changelog
    for pr in (2, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28):
        assert f"PR #{pr})" in changelog or f"PR #{pr}," in changelog, pr
    for name, value in _version_constants().items():
        if name != "BENCHMARK_REVISION":
            assert f"`{name}` `{value}`" in changelog, name


def test_release_checklist_covers_the_release_gates() -> None:
    checklist = (DOCS / "release-checklist.md").read_text(encoding="utf-8").lower()
    for gate in (
        "clean `main`",
        "ci",
        "component",
        "digest registry",
        "0 violation",
        "check_release.py",
        "changelog",
        "held-out",
        "tag",
        "rollback",
        "benchmark_revision",
    ):
        assert gate in checklist, gate


# --------------------------------------------------------------------------- development tool constraints


def _check_constraints(constraints: Path, pyproject: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOLS / "check_dev_constraints.py"), "--constraints", str(constraints), "--pyproject", str(pyproject)]
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def test_dev_constraints_pin_every_declared_tool() -> None:
    proc = _check_constraints(TOOLS / "dev-constraints.txt", REPO_ROOT / "pyproject.toml")
    assert proc.returncode == 0, proc.stderr
    assert PYPROJECT["project"]["dependencies"] == []  # runtime dependencies stay zero


@pytest.mark.parametrize(
    "dev, pins, problem",
    [
        ('"pytest>=8"', "pip==1\npytest==8.0\nsetuptools==70\n", "pytest>=8 has no upper bound"),
        ('"pytest>=8,<9"', "pip==1\nsetuptools==70\n", "pytest is not pinned"),
        ('"pytest>=8,<9"', "pip==1\npytest==9.0\nsetuptools==70\n", "pin pytest==9.0 is outside"),
        ('"pytest>=8,<9"', "pytest==8.0\nsetuptools==70\n", "pip is not pinned"),
        ('"pytest>=8,<9"', "pip==1\npytest>=8.0\nsetuptools==70\n", "is not 'name==version'"),
        ('"pytest>=8,<9"', "pip==1\npytest==8.0\npytest==8.1\nsetuptools==70\n", "pytest is pinned twice"),
    ],
)
def test_dev_constraints_reject_loose_or_missing_pins(dev: str, pins: str, problem: str, tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[build-system]\nrequires = ["setuptools>=69,<85"]\n[project]\nname = "x"\n[project.optional-dependencies]\ndev = [{dev}]\n',
        encoding="utf-8",
    )
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(pins, encoding="utf-8")
    proc = _check_constraints(constraints, pyproject)
    assert proc.returncode == 1 and problem in proc.stderr, proc.stderr


def test_ci_installs_only_pinned_tools_and_keeps_the_release_gates() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "PIP_CONSTRAINT: ${{ github.workspace }}/tools/dev-constraints.txt" in workflow
    assert "PIP_BUILD_CONSTRAINT: ${{ github.workspace }}/tools/dev-constraints.txt" in workflow
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow
    assert 'python-version: ["3.12", "3.13"]' in workflow
    for command in (
        "python tools/check_dev_constraints.py --installed",
        "python -m mypy --strict",
        "python -m pytest -q",
        "python -m build --no-isolation --outdir dist .",
        "python tools/check_public_assets.py --dist dist",
        "python tools/check_release.py --dist dist --wheelhouse wheelhouse",
    ):
        assert f"- run: {command}" in workflow, command
    # Every pip install in CI resolves through the constraints, never by a bare "pip install -U".
    assert "-U" not in workflow and "--upgrade" not in workflow


# --------------------------------------------------------------------------- regression digest registry


def test_registry_is_compact_and_well_formed() -> None:
    raw = REGISTRY_PATH.read_bytes()
    assert len(raw) < 16_000 and b"\r" not in raw
    assert list(REGISTRY) == sorted(REGISTRY) and 10 <= len(REGISTRY) <= 40
    for name, entry in REGISTRY.items():
        assert set(entry) == {"digest", "owner", "mirrors", "what"}, name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["digest"]), name
        assert entry["owner"] and entry["what"], name


def test_registry_mirrors_the_test_pins() -> None:
    """The registry and the pins in the component tests can only change together."""
    digest = {name: entry["digest"] for name, entry in REGISTRY.items()}
    mirrored = {
        "suite.public-v1.suite_hash": SUITE_HASH,
        "evaluation.public-v1-dev-seeds.manifest_hash": PUBLIC_DEV_MANIFEST_HASH,
        "hazard.stochastic-golden-traces": sha256_hex(GOLDEN_FIRE_TRACES),
        "generator.test-config.config_hash": GOLDEN_CONFIG_HASH,
        "generator.test-config.generation_hash": GOLDEN_GENERATION_HASH,
        "generator.test-config.suite_hash": GOLDEN_SUITE_HASH,
        "statistics.aggregate": GOLDEN_STATISTICS_DIGESTS["aggregate"],
        "statistics.paired": GOLDEN_STATISTICS_DIGESTS["paired"],
    }
    for name, value in digest.items():
        if name.startswith("scenario."):
            mirrored[name] = PINNED_HASHES[name.split(".")[1]]
        elif name.startswith("simulator.golden-record."):
            mirrored[name] = GOLDEN_DIGESTS[name.rsplit(".", 1)[1]]
        elif name.startswith("baseline.") and name.endswith(".config_digest"):
            mirrored[name] = GOLDEN_CONFIG_DIGESTS[name.split(".")[1]]
        elif name.startswith("baseline.") and name.endswith(".result_digest"):
            mirrored[name] = GOLDEN_RESULT_DIGESTS[name.split(".")[1]]
    registry_only = {"integrity.test-run.provenance_digest", "cli.evaluation-manifest.file_sha256"}
    assert set(digest) == set(mirrored) | registry_only
    drift = {name: (digest[name], value) for name, value in mirrored.items() if digest[name] != value}
    assert not drift, f"registry entry != test pin: {drift}"
    for name in registry_only:
        assert REGISTRY[name]["mirrors"] == "this registry only"


_FRESH_SCRIPT = """
import json, sys
sys.path[:0] = [{src!r}, {tests!r}]
from release_digests import compute_registry_digests
sys.stdout.write(json.dumps(compute_registry_digests(), sort_keys=True) + "\\n")
"""

# The second environment changes everything that must not matter.
_FRESH_ENVIRONMENTS = (
    {"PYTHONHASHSEED": "0"},
    {"PYTHONHASHSEED": "4242", "LC_ALL": "tr_TR.UTF-8", "LANG": "tr_TR.UTF-8", "TZ": "Pacific/Chatham", "PYTHONIOENCODING": "latin-1"},
)


def test_registry_digests_in_fresh_interpreters(tmp_path: Path) -> None:
    script = _FRESH_SCRIPT.format(src=str(REPO_ROOT / "src"), tests=str(REPO_ROOT / "tests"))
    expected = {name: entry["digest"] for name, entry in REGISTRY.items()}
    cwd = tmp_path / "fresh cwd with spaces"
    cwd.mkdir()
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LC_ALL", "LANG", "TZ", "PYTHONIOENCODING")}
    for changes in _FRESH_ENVIRONMENTS:
        proc = subprocess.run([sys.executable, "-c", script], cwd=cwd, env={**base, **changes}, capture_output=True, timeout=600)
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
        actual = json.loads(proc.stdout)
        drift = {name: {"registry": expected.get(name), "computed": actual.get(name)} for name in sorted(set(expected) | set(actual)) if expected.get(name) != actual.get(name)}
        assert not drift, f"contract drift under {changes}: {json.dumps(drift, indent=2)}"


# --------------------------------------------------------------------------- newline and path portability


def test_crlf_inputs_load_to_the_same_canonical_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dir with spaces"
    public = load_suite(SUITE_DIR)
    crlf_suite = load_suite(_crlf_copy(SUITE_DIR, root / "public suite"))
    assert crlf_suite.manifest.suite_hash() == SUITE_HASH
    assert {s.scenario_id: s.content_hash() for s in crlf_suite.scenarios} == PINNED_HASHES
    assert load_scenario(root / "public suite" / f"{SPARSE}.json").content_hash() == PINNED_HASHES[SPARSE]

    manifest = build_evaluation_manifest(public, [0, 1, 2])
    (root / "manifest crlf.json").write_bytes(manifest.to_file_text().encode("ascii").replace(b"\n", b"\r\n"))
    reloaded = load_evaluation_manifest(root / "manifest crlf.json")
    assert reloaded.manifest_hash() == PUBLIC_DEV_MANIFEST_HASH and "\r" not in reloaded.to_file_text()

    record = run_episode(public.get(SPARSE), greedy_exit_policy, seed=0)
    assert RunRecord.from_json(record.to_json().replace("\n", "\r\n")).digest() == record.digest()

    small = subset(public, (SPARSE,))
    small_manifest = build_evaluation_manifest(small, [3])
    result = run_evaluation(small, small_manifest, "greedy", lambda: greedy_exit_policy, root / "result.json")
    (root / "result crlf.json").write_bytes((root / "result.json").read_bytes().replace(b"\n", b"\r\n"))
    again = load_evaluation_result(root / "result crlf.json", small)
    assert again.to_file_text() == result.to_file_text() and "\r" not in again.to_file_text()

    frozen = freeze_generated_suite(generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH), root / "frozen")
    crlf_frozen = load_generated_suite(_crlf_copy(frozen.root, root / "frozen crlf"))
    assert crlf_frozen.suite.manifest.suite_hash() == GOLDEN_SUITE_HASH
    assert crlf_frozen.generation.generation_hash() == GOLDEN_GENERATION_HASH


def test_cli_artifact_is_independent_of_cwd_path_spelling_and_newlines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = REGISTRY["cli.evaluation-manifest.file_sha256"]["digest"]
    crlf_suite = _crlf_copy(SUITE_DIR, tmp_path / "a b" / "public suite")
    elsewhere = tmp_path / "another cwd"
    elsewhere.mkdir()
    outputs: list[bytes] = []
    for suite, cwd in ((SUITE_DIR, REPO_ROOT), (crlf_suite, elsewhere)):
        monkeypatch.chdir(cwd)
        out = tmp_path / f"out {len(outputs)}" / "manifest.json"
        out.parent.mkdir()
        assert _cli("evaluation", "manifest", "--suite", os.path.relpath(suite, cwd), "--seed", "0", "--seed", "1", "--seed", "2", "--out", str(out)) == 0
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]
    assert hashlib.sha256(outputs[0]).hexdigest() == pinned
    assert b"\r" not in outputs[0] and str(tmp_path).encode() not in outputs[0] and str(REPO_ROOT).encode() not in outputs[0]


# --------------------------------------------------------------------------- package metadata


def test_project_metadata() -> None:
    project = PYPROJECT["project"]
    assert project["name"] == "emberbench"
    assert project["description"].startswith("A deterministic benchmark")
    assert project["readme"] == "README.md" and project["requires-python"] == ">=3.12"
    assert project["scripts"] == {"emberbench": "emberbench.cli:main"}
    assert PYPROJECT["tool"]["setuptools"]["package-data"] == {"emberbench": ["py.typed"]}
    assert (REPO_ROOT / "src" / "emberbench" / "py.typed").is_file()


def test_built_wheel_and_sdist_metadata(built_distributions: tuple[Path, Path]) -> None:
    wheel, sdist = built_distributions
    assert wheel.name == f"emberbench-{source_version}-py3-none-any.whl"
    assert sdist.name == f"emberbench-{source_version}.tar.gz"
    problems: list[str] = _tool("check_release").check_metadata(wheel, sdist, False)
    assert problems == []
    with zipfile.ZipFile(wheel) as z:
        assert not [n for n in z.namelist() if n.startswith(("tests/", "tools/", "scenarios/", "docs/"))]


def test_release_check_compares_normalized_members(tmp_path: Path) -> None:
    tool = _tool("check_release")
    paths = []
    for n, content in enumerate((b"a", b"a", b"b")):
        path = tmp_path / f"w{n}.whl"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(zipfile.ZipInfo("emberbench/x.py", date_time=(1980 + n, 1, 1, 0, 0, 0)), content)
        paths.append(path)
    first, same, different = (tool.wheel_members(p) for p in paths)
    tool.compare("same", first, same)  # timestamps differ, members and content do not
    assert paths[0].read_bytes() != paths[1].read_bytes()
    with pytest.raises(tool.ReleaseCheckError, match=r"content differs \['emberbench/x.py'\]"):
        tool.compare("different", first, different)


@pytest.mark.skipif(os.environ.get("EMBERBENCH_RELEASE_SMOKE") != "1", reason="the CI packaging job runs tools/check_release.py; set EMBERBENCH_RELEASE_SMOKE=1 to run it here")
def test_release_smoke(tmp_path: Path) -> None:
    proc = subprocess.run([sys.executable, str(TOOLS / "check_release.py"), "--dist", str(tmp_path / "dist")], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
