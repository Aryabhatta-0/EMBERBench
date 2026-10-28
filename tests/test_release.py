"""Release hardening: one package version, pinned tools, frozen digests, portable outputs, package metadata.

See docs/versioning.md and docs/release-checklist.md.
"""

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import pkgutil
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
    benchmark_versions,
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
from release_digests import BYTE_PINNED, PUBLIC_SUITE_ENV, REGISTRY_ONLY, REGISTRY_PATH, SPEC, load_registry, registry_drift, registry_text
from test_baselines import GOLDEN_CONFIG_DIGESTS, GOLDEN_RESULT_DIGESTS
from test_determinism import GOLDEN_DIGESTS
from test_evaluation import PUBLIC_DEV_MANIFEST_HASH, subset
from test_generator import BASE_CONFIG, BATCH, GOLDEN_CONFIG_HASH, GOLDEN_GENERATION_HASH, GOLDEN_SUITE_HASH, SEED
from test_replay_audit import GOLDEN_DIGESTS as GOLDEN_REPLAY_DIGESTS
from test_statistics import GOLDEN_STATISTICS_DIGESTS
from test_stochastic_hazard import GOLDEN as GOLDEN_FIRE_TRACES
from test_suite import PINNED_HASHES, REPO_ROOT, SUITE_DIR, SUITE_HASH

TOOLS = REPO_ROOT / "tools"
DOCS = REPO_ROOT / "docs"
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
REGISTRY = load_registry()
SPARSE = "v1-pub-01-single-exit-sparse"



def _tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_release_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK_RELEASE = _tool("check_release")


def _version_constants() -> dict[str, str]:
    """Every ``*_VERSION`` string constant and ``BENCHMARK_REVISION`` of every emberbench module, by import."""
    found: dict[str, str] = {}
    for info in pkgutil.iter_modules(emberbench.__path__):
        if info.name == "__main__":
            continue
        module = importlib.import_module(f"emberbench.{info.name}")
        for name, value in vars(module).items():
            if isinstance(value, str) and (name == "BENCHMARK_REVISION" or re.fullmatch(r"[A-Z][A-Z0-9_]*_VERSION", name)):
                assert found.setdefault(name, value) == value, f"{name} has two values"
    for name, value in vars(emberbench).items():
        if isinstance(value, str) and name in found:
            assert value == found[name], name
    return found


def _reported() -> dict[str, str]:
    return {key: value for key, value in version_info().items() if key != "package_version"}


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
    assert info["package_version"] == source_version == "1.0.0rc1"  # the V1 release candidate (docs/versioning.md, section 1)
    assert info["benchmark_revision"] == BENCHMARK_REVISION == "3"
    assert list(info)[0] == "package_version"


def test_version_output_contains_every_benchmark_version_by_construction() -> None:
    """Provenance and `emberbench version` cannot disagree: the CLI builds its benchmark part from benchmark_versions()."""
    info = version_info()
    for key, value in benchmark_versions().items():
        assert info[CHECK_RELEASE.BENCHMARK_VERSION_KEYS.get(key, f"{key}_version")] == value, key
    assert len(benchmark_versions()) == 28
    tooling = {k for k in info if k != "package_version"} - {CHECK_RELEASE.BENCHMARK_VERSION_KEYS.get(k, f"{k}_version") for k in benchmark_versions()}
    assert tooling == {"cli_version", "baselines_version", "integrity_protocol_version", "replay_audit_version", "animation_format_version", "run_record_version"}


def test_every_version_constant_has_an_owner_and_a_reported_key() -> None:
    """A new *_VERSION constant in any module fails here until it is reported (or listed as a suite label)."""
    constants = _version_constants()
    assert "SIMULATOR_VERSION" in constants and constants["BENCHMARK_REVISION"] == BENCHMARK_REVISION
    reported = _reported()
    for name, value in constants.items():
        if name in CHECK_RELEASE.UNREPORTED_LABELS:
            assert name.lower() not in reported, f"{name} is reported; remove it from UNREPORTED_LABELS"
        else:
            assert reported.get(name.lower()) == value, f"{name} = {value!r} is missing from `emberbench version`"
    # Every reported key and every benchmark_versions() entry has an owning constant with the same value.
    assert {key.upper() for key in reported} <= set(constants)
    for key, value in benchmark_versions().items():
        constant = CHECK_RELEASE.BENCHMARK_VERSION_KEYS.get(key, f"{key}_version").upper()
        assert constants.get(constant) == value, f"benchmark_versions()[{key!r}] has no owning constant {constant}"
    assert set(CHECK_RELEASE.UNREPORTED_LABELS) <= set(constants)


def test_version_contracts_match_the_docs_changelog_checklist_and_provenance() -> None:
    """The same check that tools/check_release.py runs against the installed wheel, here against the source."""
    assert CHECK_RELEASE.version_contract_problems(version_info(), benchmark_versions()) == []
    constants = _version_constants()
    table = CHECK_RELEASE.versioning_table()
    for name, value in constants.items():
        assert table.get(name, ("", ""))[0] == value, f"{name} = {value!r} is missing from docs/versioning.md section 3"
    assert set(table) == set(constants)
    checklist = (DOCS / "release-checklist.md").read_text(encoding="utf-8")
    changelog = (DOCS / "changelog.md").read_text(encoding="utf-8")
    for name in CHECK_RELEASE.UNREPORTED_LABELS:
        assert f"`{name}`" in checklist and f"`{name}` `{constants[name]}`" in changelog, name


def test_version_contract_check_reports_each_kind_of_drift() -> None:
    info = version_info()
    provenance = benchmark_versions()
    for changed, expected in (
        ({**info, "scoring_version": "9.9"}, "docs/versioning.md section 3: row for SCORING_VERSION"),
        ({**info, "brand_new_version": "1.0"}, "the version review does not name `BRAND_NEW_VERSION`"),
        ({**info, "package_version": "9.9.9"}, "package_version is '9.9.9'"),
    ):
        assert any(expected in p for p in CHECK_RELEASE.version_contract_problems(changed, provenance)), expected
    missing = {k: v for k, v in info.items() if k != "environment_world_version"}
    assert any("is not reported by `emberbench version`" in p for p in CHECK_RELEASE.version_contract_problems(missing, provenance))


def test_no_doc_states_a_stale_benchmark_revision(tmp_path: Path) -> None:
    assert CHECK_RELEASE.stale_revision_problems(BENCHMARK_REVISION) == []
    assert CHECK_RELEASE.stale_revision_problems("4"), "the stale-revision scan finds nothing"


def test_changelog_records_the_v1_milestones() -> None:
    changelog = (DOCS / "changelog.md").read_text(encoding="utf-8")
    assert f"package `{source_version}`, benchmark revision `{BENCHMARK_REVISION}`" in changelog
    for pr in (2, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34):
        assert f"PR #{pr})" in changelog or f"PR #{pr}," in changelog or f"PR #{pr}):" in changelog, pr
    for milestone in ("EMBER Score", "EnvironmentPack", "unique-world", "Generator 1.1", "10-family", "Benchmark revision 2", "Benchmark revision 3", "Coordination is not a score component"):
        assert milestone in changelog, milestone


def test_release_checklist_covers_the_release_gates() -> None:
    checklist = (DOCS / "release-checklist.md").read_text(encoding="utf-8")
    for command in (
        "git status --porcelain",
        "python -m pip install -c tools/dev-constraints.txt -e \".[dev]\"",
        "python tools/check_dev_constraints.py --installed",
        "python -m mypy --strict",
        "python -m pytest -q",
        "EMBERBENCH_RELEASE_SMOKE=1 python -m pytest -q tests/test_release.py",
        "EMBERBENCH_FULL_PACK=1 python -m pytest -q tests/test_evaluation_pack.py",
        "python tests/release_digests.py --check",
        "python tools/check_release.py --contracts-only",
        "python tools/check_public_assets.py --untracked",
        "python tools/check_release.py --dist dist --wheelhouse wheelhouse --require-license --require-clean",
        "SOURCE_DATE_EPOCH=",
        "emberbench\" version --json",
        "-m emberbench version --json",
        "git tag -a v<version>",
        "git diff <previous-tag> -- tests/fixtures/release/digest_registry.json",
    ):
        assert command in checklist, command
    lowered = checklist.lower()
    for gate in ("license", "0 violations", "changelog", "held-out", "rollback", "reproducibility boundary", "byte-identical wheels"):
        assert gate in lowered, gate
    assert "run tests" not in lowered


def test_license_is_an_explicit_release_blocker(built_distributions: tuple[Path, Path]) -> None:
    """No license is chosen for the maintainers: --require-license fails until one is configured."""
    wheel, sdist = built_distributions
    problems = CHECK_RELEASE.check_metadata(wheel, sdist, True)
    if CHECK_RELEASE.license_configured():
        assert problems == []
    else:
        assert not any(REPO_ROOT.glob("LICENSE*")) and "license" not in PYPROJECT["project"]
        assert len(problems) == 1 and "LICENSE file" in problems[0] and "SPDX" in problems[0], problems


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


def test_installed_check_follows_the_tool_closure_only(tmp_path: Path) -> None:
    """Runner-preinstalled packages that no declared tool needs are ignored; the tools' dependencies are not."""
    tool = _tool("check_dev_constraints")

    def dist(name: str, version: str, *requires: str) -> importlib.metadata.Distribution:
        info = tmp_path / f"{name}-{version}.dist-info"
        info.mkdir()
        lines = [f"Name: {name}", f"Version: {version}", *(f"Requires-Dist: {r}" for r in requires)]
        (info / "METADATA").write_text("Metadata-Version: 2.1\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return importlib.metadata.PathDistribution(info)

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[build-system]\nrequires = ["setuptools>=69,<85"]\n[project]\nname = "x"\n[project.optional-dependencies]\ndev = ["pytest>=8,<10"]\n', encoding="utf-8")
    installed = {
        "pytest": dist("pytest", "9.1.1", "iniconfig>=1", 'colorama; sys_platform == "no-such-platform"', 'extra-only; extra == "dev"'),
        "iniconfig": dist("iniconfig", "2.0"),
        "setuptools": dist("setuptools", "84.2.0"),
        "pip": dist("pip", "26.2.1"),
        "pipx": dist("pipx", "1.0", "click"),
        "click": dist("click", "8.0"),
    }
    assert tool.tool_closure(pyproject, installed) == ["iniconfig", "pip", "pytest", "setuptools"]


def test_ci_installs_only_pinned_tools_and_keeps_the_release_gates() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "PIP_CONSTRAINT: ${{ github.workspace }}/tools/dev-constraints.txt" in workflow
    # Only on the editable install: pip rejects a build constraint with --no-build-isolation (the wheel tests use it).
    assert workflow.count("PIP_BUILD_CONSTRAINT: ${{ github.workspace }}/tools/dev-constraints.txt") == 1
    assert '- run: python -m pip install -e ".[dev]"\n        env:\n          PIP_BUILD_CONSTRAINT:' in workflow
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow
    assert 'python-version: ["3.12", "3.13"]' in workflow
    for command in (
        "python tools/check_dev_constraints.py --installed",
        "python -m mypy --strict",
        "python -m pytest -q",
        "python -m build --no-isolation --outdir dist .",
        "python tools/check_public_assets.py --dist dist",
        "SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) python tools/check_release.py --dist dist --wheelhouse wheelhouse",
        "python -m pip install build setuptools packaging pytest",
    ):
        assert f"- run: {command}" in workflow, command
    # The expensive release checks run in one job, not in every matrix cell.
    assert workflow.count("tools/check_release.py") == 1 and workflow.index("tools/check_release.py") > workflow.index("  packaging:")
    # Every pip install in CI resolves through the constraints, never by a bare "pip install -U".
    assert "-U" not in workflow and "--upgrade" not in workflow
    # A plain YAML scalar cannot hold ": " (for example "--only-binary=:all: --dest"): GitHub rejects the whole file.
    for line in workflow.splitlines():
        value = line.strip().removeprefix("- ")
        if value.startswith("run: ") and not value.startswith(("run: '", 'run: "', "run: |")):
            assert ": " not in value[len("run: ") :].split(" #", 1)[0], f"quote this run step: {line.strip()}"


# --------------------------------------------------------------------------- regression digest registry


def test_registry_is_compact_well_formed_and_generated_from_spec() -> None:
    raw = REGISTRY_PATH.read_bytes()
    assert len(raw) < 24_000 and b"\r" not in raw
    assert list(REGISTRY) == sorted(REGISTRY) == sorted(SPEC) and 20 <= len(REGISTRY) <= 40
    for name, entry in REGISTRY.items():
        assert set(entry) == {"digest", "owner", "mirrors", "what"}, name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["digest"]), name
        assert entry["owner"] and entry["what"], name
    # The file is exactly what `python tests/release_digests.py --write` produces.
    assert raw == registry_text({name: entry["digest"] for name, entry in REGISTRY.items()}).encode("ascii")


def test_registry_holds_no_host_private_or_hidden_material() -> None:
    text = REGISTRY_PATH.read_text(encoding="ascii").lower()
    for forbidden in (str(REPO_ROOT).lower(), "/home/", "/users/", "c:\\\\", "/tmp/", "heldout", "hidden-suite", "private", "selection_key", "seed="):
        assert forbidden not in text, forbidden


def test_registry_mirrors_the_test_pins() -> None:
    """The registry and the pins in the component tests can only change together."""
    digest = {name: entry["digest"] for name, entry in REGISTRY.items()}
    from emberbench.composition import rule_set_digest
    replay = GOLDEN_REPLAY_DIGESTS["public:route-invalidation:fire-aware:s1"]
    mirrored = {
        "suite.public-v1.suite_hash": SUITE_HASH,
        "evaluation.public-v1-dev-seeds.manifest_hash": PUBLIC_DEV_MANIFEST_HASH,
        "hazard.stochastic-golden-traces": sha256_hex(GOLDEN_FIRE_TRACES),
        "generator.test-config.config_hash": GOLDEN_CONFIG_HASH,
        "generator.test-config.generation_hash": GOLDEN_GENERATION_HASH,
        "generator.test-config.suite_hash": GOLDEN_SUITE_HASH,
        "statistics.aggregate": GOLDEN_STATISTICS_DIGESTS["aggregate"],
        "statistics.paired": GOLDEN_STATISTICS_DIGESTS["paired"],
        "replay.route-invalidation-fire-aware-s1.audit_digest": replay[1],
        "replay.route-invalidation-fire-aware-s1.animation_sha256": replay[2],
        "rules.composition.rule_set_digest": rule_set_digest(),
    }
    for name in digest:
        if name.startswith("scenario."):
            mirrored[name] = PINNED_HASHES[name.split(".")[1]]
        elif name.startswith("simulator.golden-record."):
            mirrored[name] = GOLDEN_DIGESTS[name.rsplit(".", 1)[1]]
        elif name.startswith("baseline.") and name.endswith(".config_digest"):
            mirrored[name] = GOLDEN_CONFIG_DIGESTS[name.split(".")[1]]
        elif name.startswith("baseline.") and name.endswith(".result_digest"):
            mirrored[name] = GOLDEN_RESULT_DIGESTS[name.split(".")[1]]
    registry_only = {name for name, entry in REGISTRY.items() if entry["mirrors"] == REGISTRY_ONLY}
    assert set(digest) == set(mirrored) | registry_only and not set(mirrored) & registry_only
    drift = registry_drift({name: digest[name] for name in mirrored}, mirrored)
    assert not drift, f"registry entry != test pin: {json.dumps(drift, indent=2)}"


def test_registry_drift_is_reported_as_a_diff() -> None:
    drift = registry_drift({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"})
    assert drift == {"b": {"registry": "2", "computed": "3"}, "c": {"registry": None, "computed": "4"}}


_FRESH_SCRIPT = """
import json, sys
sys.path[:0] = [{src!r}, {tests!r}]
from release_digests import compute_registry_digests
sys.stdout.write(json.dumps(compute_registry_digests(), sort_keys=True) + "\\n")
"""

# The second environment changes everything that must not matter, and reads a CRLF copy of the public suite.
_FRESH_ENVIRONMENTS = (
    {"PYTHONHASHSEED": "0"},
    {"PYTHONHASHSEED": "4242", "LC_ALL": "tr_TR.UTF-8", "LANG": "tr_TR.UTF-8", "TZ": "Pacific/Chatham", "PYTHONIOENCODING": "latin-1", PUBLIC_SUITE_ENV: ""},
)


def test_registry_digests_in_fresh_interpreters(tmp_path: Path) -> None:
    script = _FRESH_SCRIPT.format(src=str(REPO_ROOT / "src"), tests=str(REPO_ROOT / "tests"))
    expected = {name: entry["digest"] for name, entry in REGISTRY.items()}
    cwd = tmp_path / "fresh cwd with spaces"
    cwd.mkdir()
    crlf_suite = _crlf_copy(SUITE_DIR, tmp_path / "crlf copy" / "public suite")
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LC_ALL", "LANG", "TZ", "PYTHONIOENCODING", PUBLIC_SUITE_ENV)}
    for changes in _FRESH_ENVIRONMENTS:
        env = {**base, **changes}
        if PUBLIC_SUITE_ENV in env:
            env[PUBLIC_SUITE_ENV] = str(crlf_suite)
        proc = subprocess.run([sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, timeout=900)
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
        drift = registry_drift(expected, json.loads(proc.stdout))
        if PUBLIC_SUITE_ENV in env:
            # A pack pins exact file bytes by design; every other contract is independent of line endings.
            assert set(drift) == BYTE_PINNED, f"contract drift under {changes}: {json.dumps(drift, indent=2)}"
        else:
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
    assert CHECK_RELEASE.check_metadata(wheel, sdist, False) == []
    assert CHECK_RELEASE.content_problems(wheel, sdist) == []
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    assert not [n for n in names if n.startswith(("tests/", "tools/", "scenarios/", "docs/"))]
    # The wheel is the package code only: every module of src/emberbench and py.typed.
    sources = sorted(f"emberbench/{p.name}" for p in (REPO_ROOT / "src" / "emberbench").iterdir() if p.suffix == ".py" or p.name == "py.typed")
    assert sorted(n for n in names if n.startswith("emberbench/")) == sources


def test_release_check_rejects_unintended_package_data(tmp_path: Path, built_distributions: tuple[Path, Path]) -> None:
    """Generated research outputs (for example a 14 MB evaluation-pack run) never become package data."""
    wheel, sdist = built_distributions
    planted = tmp_path / wheel.name
    shutil.copyfile(wheel, planted)
    with zipfile.ZipFile(planted, "a") as z:
        z.writestr("emberbench/data/pack-results.json", b"0" * (CHECK_RELEASE.MEMBER_MAX_BYTES + 1))
    problems = CHECK_RELEASE.content_problems(planted, sdist)
    assert any("not package code" in p for p in problems) and any("bytes (limit" in p for p in problems), problems


def test_release_check_freezes_source_date_epoch(built_distributions: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")
    assert CHECK_RELEASE.source_date_epoch() == "1767225600"
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "yesterday")
    with pytest.raises(CHECK_RELEASE.ReleaseCheckError):
        CHECK_RELEASE.source_date_epoch()
    # The shared test build sets no SOURCE_DATE_EPOCH, so its member times are the build time, not a frozen epoch.
    wheel, _ = built_distributions
    assert CHECK_RELEASE.wheel_time_problems(wheel, "0")
    # Zip times have two-second resolution: an odd epoch is stored rounded down; before 1980 it is clamped.
    assert CHECK_RELEASE.zip_time("1792523845") == (2026, 10, 20, 19, 17, 24)
    assert CHECK_RELEASE.zip_time("1792523844") == (2026, 10, 20, 19, 17, 24)
    assert CHECK_RELEASE.zip_time("0") == (1980, 1, 1, 0, 0, 0)


def test_release_check_compares_normalized_members(tmp_path: Path) -> None:
    tool = CHECK_RELEASE
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
    # A relative --dist, as CI passes it, from a working directory outside the checkout.
    proc = subprocess.run([sys.executable, str(TOOLS / "check_release.py"), "--dist", "dist"], cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
