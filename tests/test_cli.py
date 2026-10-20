"""The ``emberbench`` command-line interface: commands, exit codes, file safety and determinism."""

import contextlib
import dataclasses
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import cli_agents
import emberbench
from emberbench import (
    GeneratorConfig,
    RunRecord,
    agent_policy_factory,
    aggregate_result,
    build_evaluation_manifest,
    compare_results,
    freeze_generated_suite,
    generate_suite,
    load_evaluation_manifest,
    load_evaluation_result,
    load_scenario,
    load_suite,
    run_evaluation,
    verify_record,
)
from emberbench import cli
from emberbench.canonical import canonical_json, sha256_hex
from emberbench.cli import CLI_VERSION, ExitCode, main
from emberbench.statistics import DEFAULT_BOOTSTRAP_RESAMPLES

from test_evaluation import PUBLIC_DEV_MANIFEST_HASH
from test_generator import BASE_CONFIG, BATCH, SEED
from test_suite import PINNED_HASHES, SUITE_DIR, SUITE_HASH

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
SCN01 = SUITE_DIR / "v1-pub-01-single-exit-sparse.json"
SCN10 = SUITE_DIR / "v1-pub-10-route-invalidation.json"
SMALL_IDS = ("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation")
SEEDS = ("11", "7")
RESAMPLES = "200"  # small, to keep the tests fast


@dataclasses.dataclass(frozen=True)
class Outcome:
    code: int
    out: str
    err: str

    def json(self) -> dict[str, Any]:
        assert self.out.endswith("\n") and self.out.count("\n") == 1, self.out
        data = json.loads(self.out)
        assert isinstance(data, dict)
        assert self.out == canonical_json(data) + "\n"
        return data

    def error(self) -> dict[str, Any]:
        """The JSON-mode error object on stderr. Stdout must stay empty on failure."""
        assert self.out == ""
        assert self.err.count("\n") == 1, self.err
        data = json.loads(self.err)
        assert set(data) == {"error"} and set(data["error"]) == {"exit_code", "kind", "message"}
        assert data["error"]["exit_code"] == self.code
        error: dict[str, Any] = data["error"]
        return error


def invoke(*args: str | Path) -> Outcome:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in args])
    return Outcome(code, out.getvalue(), err.getvalue())


def run_cli(*args: str | Path, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    """``python -m emberbench`` in a fresh interpreter."""
    full = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHASHSEED")}
    full["PYTHONPATH"] = str(ROOT / "src")
    full.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "emberbench", *(str(a) for a in args)], cwd=cwd, env=full, capture_output=True, timeout=600
    )


def write_subset_suite(dest: Path, ids: tuple[str, ...]) -> Path:
    public = load_suite(SUITE_DIR)
    entries = tuple(e for e in public.manifest.entries if e.scenario_id in ids)
    dest.mkdir()
    for e in entries:
        shutil.copyfile(SUITE_DIR / e.path, dest / e.path)
    (dest / "manifest.json").write_bytes(dataclasses.replace(public.manifest, entries=entries).to_file_text().encode("ascii"))
    return dest


def rewrite_json(path: Path, data: Any) -> None:
    path.write_bytes((canonical_json(data) + "\n").encode("ascii"))


def no_temp_files(directory: Path) -> bool:
    return not any(p.name.endswith(".tmp") for p in directory.iterdir())


# --------------------------------------------------------------------------- shared artifacts


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def small(work: Path) -> Path:
    return write_subset_suite(work / "small", SMALL_IDS)


@pytest.fixture(scope="module")
def manifest_file(small: Path, work: Path) -> Path:
    path = work / "manifest.json"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", SEEDS[0], "--seed", SEEDS[1], "--out", path).code == 0
    return path


def _evaluate(small: Path, manifest_file: Path, out: Path, model_id: str, agent: str) -> Path:
    o = invoke("evaluation", "run", "--suite", small, "--manifest", manifest_file, "--model-id", model_id, "--agent", agent, "--out", out)
    assert o.code == 0, o.err
    return out


@pytest.fixture(scope="module")
def greedy_result(small: Path, manifest_file: Path, work: Path) -> Path:
    return _evaluate(small, manifest_file, work / "greedy.json", "greedy", "cli_agents:greedy")


@pytest.fixture(scope="module")
def stay_result(small: Path, manifest_file: Path, work: Path) -> Path:
    return _evaluate(small, manifest_file, work / "stay.json", "stay", "cli_agents:stay")


@pytest.fixture(scope="module")
def record_file(work: Path) -> Path:
    path = work / "record.json"
    assert invoke("episode", "run", "--scenario", SCN10, "--seed", "7", "--agent", "cli_agents:greedy", "--out", path).code == 0
    return path


@pytest.fixture(scope="module")
def frozen_dir(work: Path) -> Path:
    root = work / "frozen"
    freeze_generated_suite(generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH), root)
    return root


# --------------------------------------------------------------------------- entry points, versions, exit codes


def test_exit_codes_are_pinned() -> None:
    assert {c.name: int(c) for c in ExitCode} == {
        "OK": 0,
        "INTERNAL": 1,
        "USAGE": 2,
        "VALIDATION": 3,
        "AGENT": 4,
        "MISMATCH": 5,
        "OUTPUT": 6,
        "INTERRUPTED": 130,
    }


def test_version_json_lists_every_contract_version() -> None:
    o = invoke("version", "--json")
    assert o.code == 0 and o.err == ""
    assert o.json() == {
        "command": "version",
        "cli_version": "1.0",
        "simulator_version": emberbench.SIMULATOR_VERSION,
        "schema_version": emberbench.SCHEMA_VERSION,
        "observation_version": emberbench.OBSERVATION_VERSION,
        "action_schema_version": emberbench.ACTION_SCHEMA_VERSION,
        "metrics_version": emberbench.METRICS_VERSION,
        "statistics_version": emberbench.STATISTICS_VERSION,
        "generator_version": emberbench.GENERATOR_VERSION,
        "generator_config_version": emberbench.GENERATOR_CONFIG_VERSION,
        "generation_manifest_version": emberbench.GENERATION_MANIFEST_VERSION,
        "evaluation_manifest_version": emberbench.EVALUATION_MANIFEST_VERSION,
        "evaluation_result_version": emberbench.EVALUATION_RESULT_VERSION,
        "crowd_diagnostics_version": emberbench.CROWD_DIAGNOSTICS_VERSION,
        "baselines_version": emberbench.BASELINES_VERSION,
        "integrity_protocol_version": emberbench.INTEGRITY_PROTOCOL_VERSION,
        "scoring_version": emberbench.SCORING_VERSION,
        "environment_pack_version": emberbench.ENVIRONMENT_PACK_VERSION,
        "replay_audit_version": emberbench.REPLAY_AUDIT_VERSION,
        "animation_format_version": emberbench.ANIMATION_FORMAT_VERSION,
        "benchmark_revision": emberbench.BENCHMARK_REVISION,
    }


def test_the_cli_changes_no_library_version() -> None:
    assert CLI_VERSION == "1.0"
    assert (
        emberbench.SIMULATOR_VERSION,
        emberbench.OBSERVATION_VERSION,
        emberbench.ACTION_SCHEMA_VERSION,
        emberbench.SCHEMA_VERSION,
        emberbench.METRICS_VERSION,
        emberbench.STATISTICS_VERSION,
        emberbench.GENERATOR_VERSION,
        emberbench.EVALUATION_MANIFEST_VERSION,
        emberbench.EVALUATION_RESULT_VERSION,
        emberbench.BASELINES_VERSION,
        emberbench.INTEGRITY_PROTOCOL_VERSION,
        emberbench.BENCHMARK_REVISION,
    ) == ("0.2.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.1.0", "1.0", "1.0", "1.0", "1.0", "2")


def test_version_human_mode() -> None:
    o = invoke("version")
    assert o.code == 0
    assert "cli_version: 1.0\n" in o.out and "{" not in o.out


def test_both_entry_points_run_the_same_main(tmp_path: Path) -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"emberbench": "emberbench.cli:main"}
    assert importlib.import_module("emberbench.__main__").main is main
    proc = run_cli("version", "--json", cwd=tmp_path)
    assert proc.returncode == 0 and proc.stderr == b""
    assert proc.stdout.decode("ascii").replace("\r\n", "\n") == invoke("version", "--json").out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["scenario"],
        ["scenario", "validate"],
        ["version", "--nope"],
        ["evaluation", "manifest", "--suite", "x", "--out", "y"],  # no --seed
        ["evaluation", "manifest", "--suite", "x", "--generated-suite", "y", "--seed", "1", "--out", "z"],
        ["evaluation", "manifest", "--suite", "x", "--seed", "-1", "--out", "y"],
        ["evaluation", "manifest", "--suite", "x", "--seed", "1_0", "--out", "y"],
        ["evaluation", "manifest", "--suite", "x", "--seed", "0x10", "--out", "y"],
        ["evaluation", "manifest", "--suite", "x", "--seed", str(2**64), "--out", "y"],
        ["evaluation", "manifest", "--suite", "x", "--seed", "3", "--seed", "3", "--out", "y"],
        ["evaluation", "manifest", "--sui", "x", "--seed", "3", "--out", "y"],  # no abbreviations
        ["statistics", "aggregate", "--suite", "x", "--result", "r", "--out", "o", "--resamples", "0"],
        ["statistics", "aggregate", "--suite", "x", "--result", "r", "--out", "o", "--resamples", "many"],
        ["evaluation", "run", "--suite", "x", "--manifest", "m", "--model-id", " padded", "--agent", "a:b", "--out", "o"],
        ["episode", "run", "--scenario", str(SCN01), "--seed", "1", "--agent", "no_colon", "--out", "o"],
        ["episode", "run", "--scenario", str(SCN01), "--seed", "1", "--agent", "a-b:c", "--out", "o"],
        ["episode", "run", "--scenario", str(SCN01), "--seed", "1", "--agent", "a:b:c", "--out", "o"],
    ],
)
def test_malformed_command_lines_exit_2(argv: list[str], tmp_path: Path) -> None:
    with contextlib.chdir(tmp_path):
        o = invoke(*argv)
        assert o.code == ExitCode.USAGE
        assert o.out == "" and o.err and "Traceback" not in o.err
        assert not any(tmp_path.iterdir())
        assert invoke(*argv) == o  # deterministic stderr


def test_help_exits_0() -> None:
    assert invoke("--help").code == 0
    assert invoke("evaluation", "manifest", "--help").code == 0


def test_json_mode_errors_are_one_json_line_on_stderr() -> None:
    o = invoke("evaluation", "manifest", "--suite", "x", "--seed", "3", "--seed", "3", "--out", "y", "--json")
    assert o.code == 2
    assert o.error() == {"exit_code": 2, "kind": "usage", "message": o.error()["message"]}
    assert "duplicate seeds [3]" in o.error()["message"]


def test_internal_errors_hide_the_traceback_unless_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(args: Any) -> dict[str, Any]:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "_cmd_version", broken)
    o = invoke("version")
    assert o.code == ExitCode.INTERNAL
    assert o.err == "emberbench: error: internal error: RuntimeError: bug (rerun with --debug for a traceback)\n"
    debug = invoke("version", "--debug")
    assert debug.code == ExitCode.INTERNAL and "Traceback (most recent call last)" in debug.err
    assert invoke("version", "--json").error()["kind"] == "internal"


def test_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(args: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_cmd_version", interrupted)
    assert invoke("version").code == 130


# --------------------------------------------------------------------------- validation commands


def test_scenario_validate() -> None:
    o = invoke("scenario", "validate", SCN01, "--json")
    assert o.code == 0 and o.err == ""
    assert o.json() == {
        "command": "scenario validate",
        "valid": True,
        "scenario_id": "v1-pub-01-single-exit-sparse",
        "scenario_hash": PINNED_HASHES["v1-pub-01-single-exit-sparse"],
        "schema_version": "1.0",
    }


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"schema_version": "1.0"}', b"\xff\xfe", b"[]", b'{"x": NaN}'],
    ids=["bad-json", "missing-keys", "not-utf8", "not-object", "nan"],
)
def test_scenario_validate_fails_closed_without_touching_input(content: bytes, tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)
    o = invoke("scenario", "validate", path)
    assert o.code == ExitCode.VALIDATION and o.out == "" and "Traceback" not in o.err
    assert path.read_bytes() == content
    assert invoke("scenario", "validate", path) == o
    assert invoke("scenario", "validate", path, "--json").error()["kind"] == "validation"


def test_scenario_validate_missing_or_directory(tmp_path: Path) -> None:
    assert invoke("scenario", "validate", tmp_path / "absent.json").code == ExitCode.VALIDATION
    assert invoke("scenario", "validate", tmp_path).code == ExitCode.VALIDATION


def test_scenario_validate_rejects_a_tampered_scenario(tmp_path: Path) -> None:
    data = json.loads(SCN01.read_text(encoding="utf-8"))
    data["width"] = -1
    path = tmp_path / "s.json"
    rewrite_json(path, data)
    assert invoke("scenario", "validate", path).code == ExitCode.VALIDATION


def test_suite_validate_reports_pinned_hashes() -> None:
    o = invoke("suite", "validate", SUITE_DIR, "--json")
    assert o.code == 0
    data = o.json()
    assert data["suite_hash"] == SUITE_HASH
    assert data["scenario_count"] == len(PINNED_HASHES)
    assert {s["scenario_id"]: s["content_hash"] for s in data["scenarios"]} == PINNED_HASHES
    assert str(SUITE_DIR) not in o.out  # no paths in results


def test_suite_validate_fails_closed(tmp_path: Path) -> None:
    suite = write_subset_suite(tmp_path / "s", SMALL_IDS)
    assert invoke("suite", "validate", suite).code == 0
    (suite / "extra.json").write_text("{}", encoding="utf-8")
    assert invoke("suite", "validate", suite).code == ExitCode.VALIDATION
    (suite / "extra.json").unlink()
    scenario = suite / SCN01.name
    scenario.write_bytes(scenario.read_bytes().replace(b'"max_steps": 60', b'"max_steps": 61'))
    before = {p.name: p.read_bytes() for p in suite.iterdir()}
    assert invoke("suite", "validate", suite).code == ExitCode.VALIDATION
    assert {p.name: p.read_bytes() for p in suite.iterdir()} == before
    assert invoke("suite", "validate", tmp_path / "absent").code == ExitCode.VALIDATION
    assert invoke("suite", "validate", SCN01).code == ExitCode.VALIDATION


def test_generated_validate(frozen_dir: Path) -> None:
    o = invoke("generated", "validate", frozen_dir, "--json")
    assert o.code == 0, o.err
    data = o.json()
    assert data["batch_id"] == BATCH and data["generator_version"] == emberbench.GENERATOR_VERSION
    assert data["scenario_count"] == BASE_CONFIG["scenario_count"]
    assert str(SEED) not in o.out and format(SEED, "x") not in o.out  # the private seed is never stored or shown


def test_generated_validate_fails_closed(frozen_dir: Path, tmp_path: Path) -> None:
    assert invoke("generated", "validate", SUITE_DIR).code == ExitCode.VALIDATION
    copy = tmp_path / "frozen"
    shutil.copytree(frozen_dir, copy)
    (copy / "stray.txt").write_text("x", encoding="utf-8")
    assert invoke("generated", "validate", copy).code == ExitCode.VALIDATION


# --------------------------------------------------------------------------- evaluation manifest


def test_manifest_matches_the_golden_public_hash(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    o = invoke("evaluation", "manifest", "--suite", SUITE_DIR, "--seed", "0", "--seed", "1", "--seed", "2", "--out", out, "--json")
    assert o.code == 0, o.err
    assert o.json() == {
        "command": "evaluation manifest",
        "manifest_version": "1.0",
        "manifest_hash": PUBLIC_DEV_MANIFEST_HASH,
        "suite_id": "emberbench-v1-public",
        "suite_version": "1.0.0",
        "suite_hash": SUITE_HASH,
        "episode_count": 3 * len(PINNED_HASHES),
        "seed_count": 3,
    }
    library = build_evaluation_manifest(load_suite(SUITE_DIR), [0, 1, 2])
    assert out.read_bytes() == library.to_file_text().encode("ascii")
    assert load_evaluation_manifest(out) == library


def test_manifest_keeps_seed_order_and_ignores_option_order(small: Path, tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "11", "--seed", "7", "--out", a).code == 0
    assert invoke("evaluation", "manifest", "--out", b, "--seed", "11", "--suite", small, "--seed", "7").code == 0
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "7", "--seed", "11", "--out", c).code == 0
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    assert [e.seed for e in load_evaluation_manifest(a).episodes] == [11, 7, 11, 7]
    assert [e.seed for e in load_evaluation_manifest(c).episodes] == [7, 11, 7, 11]


def test_manifest_output_does_not_echo_seeds(small: Path, tmp_path: Path) -> None:
    o = invoke("evaluation", "manifest", "--suite", small, "--seed", "987654321", "--out", tmp_path / "m.json")
    assert o.code == 0 and "987654321" not in o.out + o.err


def test_manifest_from_a_generated_suite(frozen_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    o = invoke("evaluation", "manifest", "--generated-suite", frozen_dir, "--seed", "5", "--out", out, "--json")
    assert o.code == 0, o.err
    assert o.json()["suite_id"] == BATCH
    assert invoke("evaluation", "manifest", "--generated-suite", SUITE_DIR, "--seed", "5", "--out", tmp_path / "x.json").code == 3


def test_manifest_refuses_to_overwrite(small: Path, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    out.write_bytes(b"keep")
    o = invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", out, "--json")
    assert o.code == ExitCode.OUTPUT and o.error()["kind"] == "output"
    assert out.read_bytes() == b"keep"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", out, "--force").code == 0
    assert load_evaluation_manifest(out).episodes[0].seed == 1
    assert no_temp_files(tmp_path)


def test_outputs_never_create_directories(small: Path, tmp_path: Path) -> None:
    out = tmp_path / "missing" / "m.json"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", out).code == ExitCode.OUTPUT
    assert not out.parent.exists()
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", tmp_path).code == ExitCode.OUTPUT


def test_invalid_suite_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    assert invoke("evaluation", "manifest", "--suite", tmp_path / "absent", "--seed", "1", "--out", out).code == 3
    assert not out.exists() and no_temp_files(tmp_path)


# --------------------------------------------------------------------------- atomic output


def test_write_output_is_atomic_and_never_replaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "a.json"
    cli.write_output(target, b"one\n", force=False)
    assert target.read_bytes() == b"one\n"
    # A file that appears after the pre-check is still never replaced.
    with pytest.raises(cli.CliError) as info:
        cli.write_output(target, b"two\n", force=False)
    assert info.value.code == ExitCode.OUTPUT and target.read_bytes() == b"one\n"
    cli.write_output(target, b"two\n", force=True)
    assert target.read_bytes() == b"two\n"

    def no_links(src: str, dst: str) -> None:
        raise OSError("hard links are not supported")

    monkeypatch.setattr(os, "link", no_links)
    cli.write_output(tmp_path / "b.json", b"three\n", force=False)
    assert (tmp_path / "b.json").read_bytes() == b"three\n"
    with pytest.raises(cli.CliError):
        cli.write_output(target, b"four\n", force=False)
    assert target.read_bytes() == b"two\n"
    assert no_temp_files(tmp_path)


def test_a_failed_write_leaves_no_partial_artifact(small: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def disk_full(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", disk_full)
    out = tmp_path / "m.json"
    o = invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", out)
    assert o.code == ExitCode.OUTPUT
    assert o.err == "emberbench: error: cannot write output: No space left on device\n"
    assert list(tmp_path.iterdir()) == []


# Hard links unsupported: the no-force path must fall back to an exclusive create, never check-then-replace.


def _no_hard_links(monkeypatch: pytest.MonkeyPatch, before_fail: Any = None) -> list[str]:
    """Make ``os.link`` fail as on a filesystem without hard links. ``before_fail`` runs first (a racing writer)."""
    calls: list[str] = []

    def link(src: str, dst: str) -> None:
        calls.append(str(dst))
        if before_fail is not None:
            before_fail(Path(dst))
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "link", link)
    return calls


def _forbid_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    def replace(src: Any, dst: Any) -> None:
        raise AssertionError("os.replace must not be used without --force")

    monkeypatch.setattr(os, "replace", replace)


def test_fallback_writes_an_absent_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_hard_links(monkeypatch)
    _forbid_replace(monkeypatch)
    target = tmp_path / "a.json"
    cli.write_output(target, b"data\n", force=False)
    assert calls == [str(target)]
    assert target.read_bytes() == b"data\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_fallback_never_changes_an_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_hard_links(monkeypatch)
    _forbid_replace(monkeypatch)
    target = tmp_path / "a.json"
    target.write_bytes(b"keep")
    with pytest.raises(cli.CliError) as info:
        cli.write_output(target, b"data\n", force=False)
    assert info.value.code == ExitCode.OUTPUT
    assert info.value.message == f"{target}: already exists; pass --force to replace it"
    assert target.read_bytes() == b"keep"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_fallback_preserves_a_destination_that_appears_concurrently(small: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def racing_writer(dst: Path) -> None:
        dst.write_bytes(b"adversary")

    _no_hard_links(monkeypatch, racing_writer)
    _forbid_replace(monkeypatch)
    out = tmp_path / "m.json"
    o = invoke("evaluation", "manifest", "--suite", small, "--seed", "1", "--out", out, "--json")
    assert o.code == ExitCode.OUTPUT
    assert o.error() == {"exit_code": 6, "kind": "output", "message": f"{out}: already exists; pass --force to replace it"}
    assert out.read_bytes() == b"adversary"
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_a_failed_fallback_write_removes_only_its_own_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_hard_links(monkeypatch)
    real_fsync = os.fsync
    calls = 0

    def fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:  # the first fsync is the temporary file, the second the exclusively created output
            raise OSError(28, "No space left on device")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    target = tmp_path / "a.json"
    with pytest.raises(OSError):
        cli.write_output(target, b"data\n", force=False)
    assert calls == 2
    assert list(tmp_path.iterdir()) == []


def test_cleanup_never_removes_a_file_that_replaced_ours(tmp_path: Path) -> None:
    target = tmp_path / "a.json"
    target.write_bytes(b"ours")
    created = os.stat(target)
    other = tmp_path / "other"
    other.write_bytes(b"adversary")
    os.replace(other, target)
    cli._remove_if_created(target, created)
    assert target.read_bytes() == b"adversary"
    cli._remove_if_created(target, os.stat(target))
    assert not target.exists()
    cli._remove_if_created(target, created)  # already gone: no error


def test_force_still_replaces_without_hard_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _no_hard_links(monkeypatch)
    target = tmp_path / "a.json"
    target.write_bytes(b"old")
    cli.write_output(target, b"new\n", force=True)
    assert calls == []  # --force renames; it never links or creates exclusively
    assert target.read_bytes() == b"new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


# --------------------------------------------------------------------------- single episodes and replay


def test_episode_run_writes_an_authentic_canonical_record(record_file: Path) -> None:
    text = record_file.read_bytes().decode("ascii")
    record = RunRecord.from_json(text)
    assert text == record.to_json() + "\n"
    assert record.seed == 7 and record.scenario_id == "v1-pub-10-route-invalidation"
    verify_record(load_scenario(SCN10), record)


def test_episode_run_reports_and_is_repeatable(record_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    o = invoke("episode", "run", "--seed", "7", "--out", out, "--agent", "cli_agents:greedy", "--scenario", SCN10, "--json")
    assert o.code == 0, o.err
    record = RunRecord.from_json(out.read_text(encoding="ascii"))
    assert o.json() == {
        "command": "episode run",
        "scenario_id": record.scenario_id,
        "scenario_hash": PINNED_HASHES[record.scenario_id],
        "simulator_version": emberbench.SIMULATOR_VERSION,
        "observation_version": emberbench.OBSERVATION_VERSION,
        "record_digest": record.digest(),
        "steps": len(record.steps),
        "termination_reason": record.final_state.termination_reason.value if record.final_state.termination_reason else None,
    }
    assert out.read_bytes() == record_file.read_bytes()


def test_episode_run_supports_nested_factories(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert invoke("episode", "run", "--scenario", SCN01, "--seed", "3", "--agent", "cli_agents:Namespace.greedy", "--out", out).code == 0


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ("cli_agents_absent:make", "module 'cli_agents_absent' not found"),
        ("cli_agents:absent", "has no attribute 'absent'"),
        ("cli_agents:NOT_CALLABLE", "int is not callable"),
        ("cli_agents:factory_raises", "raised ValueError: cannot build"),
        ("cli_agents:not_an_agent", "returned object, not an agent"),
        ("cli_agents:raises", "agent raised RuntimeError at step 0: model exploded"),
        ("cli_agents:bad_output", "model output rejected"),
    ],
)
def test_agent_failures_exit_4_and_write_nothing(agent: str, message: str, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    o = invoke("episode", "run", "--scenario", SCN01, "--seed", "1", "--agent", agent, "--out", out)
    assert o.code == ExitCode.AGENT, o.err
    assert message in o.err and "Traceback" not in o.err
    assert list(tmp_path.iterdir()) == []
    assert invoke("episode", "run", "--scenario", SCN01, "--seed", "1", "--agent", agent, "--out", out, "--json").error()["kind"] == "agent"


def test_model_exceptions_keep_their_cause_under_debug(tmp_path: Path) -> None:
    o = invoke("episode", "run", "--scenario", SCN01, "--seed", "1", "--agent", "cli_agents:raises", "--out", tmp_path / "r", "--debug")
    assert o.code == ExitCode.AGENT
    assert "RuntimeError: model exploded" in o.err and "Traceback" in o.err


def test_agent_import_failures_and_explicit_agent_path(tmp_path: Path) -> None:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "cli_broken_agent.py").write_text("raise ImportError('broken dependency')\n", encoding="utf-8")
    (agents / "cli_exit_agent.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    (agents / "cli_path_agent.py").write_text("from cli_agents import greedy as make\n", encoding="utf-8")
    run = ("episode", "run", "--scenario", SCN01, "--seed", "1")
    o = invoke(*run, "--agent", "cli_broken_agent:make", "--agent-path", agents, "--out", tmp_path / "a.json")
    assert o.code == ExitCode.AGENT and "ImportError: broken dependency" in o.err
    o = invoke(*run, "--agent", "cli_exit_agent:make", "--agent-path", agents, "--out", tmp_path / "a.json")
    assert o.code == ExitCode.AGENT and "SystemExit" in o.err
    # Without --agent-path the module is not importable: nothing is discovered implicitly.
    assert invoke(*run, "--agent", "cli_path_agent:make", "--out", tmp_path / "a.json").code == ExitCode.AGENT
    sys.modules.pop("cli_path_agent", None)
    assert invoke(*run, "--agent", "cli_path_agent:make", "--agent-path", agents, "--out", tmp_path / "a.json").code == 0
    assert str(agents) not in sys.path  # restored after the command
    assert invoke(*run, "--agent", "cli_agents:greedy", "--agent-path", tmp_path / "absent", "--out", tmp_path / "b.json").code == 2


def test_episode_run_validates_inputs_before_importing_the_agent(tmp_path: Path) -> None:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "cli_marker_agent.py").write_text("raise ImportError('must not be imported')\n", encoding="utf-8")
    o = invoke("episode", "run", "--scenario", tmp_path / "absent.json", "--seed", "1", "--agent", "cli_marker_agent:f", "--agent-path", agents, "--out", tmp_path / "r.json")
    assert o.code == ExitCode.VALIDATION and "must not be imported" not in o.err


def test_the_cwd_is_never_an_implicit_agent_source(tmp_path: Path) -> None:
    (tmp_path / "cwd_agent.py").write_text("from cli_agents import greedy as make\n", encoding="utf-8")
    common = ("episode", "run", "--scenario", SCN01, "--seed", "1", "--agent", "cwd_agent:make")
    env = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(TESTS)])}
    implicit = run_cli(*common, "--out", "a.json", cwd=tmp_path, env=env)
    assert implicit.returncode == ExitCode.AGENT, implicit.stderr
    explicit = run_cli(*common, "--agent-path", ".", "--out", "b.json", cwd=tmp_path, env=env)
    assert explicit.returncode == 0, explicit.stderr


def test_the_same_agent_instance_twice_is_rejected(small: Path, manifest_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    o = invoke("evaluation", "run", "--suite", small, "--manifest", manifest_file, "--model-id", "m", "--agent", "cli_agents:same_instance", "--out", out)
    assert o.code == ExitCode.AGENT and "earlier episode" in o.err


def test_episode_run_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    out.write_bytes(b"keep")
    run = ("episode", "run", "--scenario", SCN01, "--seed", "1", "--agent", "cli_agents:greedy", "--out", out)
    assert invoke(*run).code == ExitCode.OUTPUT and out.read_bytes() == b"keep"
    assert invoke(*run, "--force").code == 0 and out.read_bytes() != b"keep"
    copy = tmp_path / "s.json"
    shutil.copyfile(SCN01, copy)
    o = invoke("episode", "run", "--scenario", copy, "--seed", "1", "--agent", "cli_agents:greedy", "--out", copy, "--force")
    assert o.code == ExitCode.OUTPUT and "input" in o.err and copy.read_bytes() == SCN01.read_bytes()


def test_replay_verify(record_file: Path) -> None:
    o = invoke("replay", "verify", "--scenario", SCN10, "--record", record_file, "--json")
    assert o.code == 0, o.err
    record = RunRecord.from_json(record_file.read_text(encoding="ascii"))
    assert o.json() == {
        "command": "replay verify",
        "verified": True,
        "scenario_id": record.scenario_id,
        "scenario_hash": record.scenario_hash,
        "record_digest": record.digest(),
        "simulator_version": record.simulator_version,
        "observation_version": record.observation_version,
        "steps": len(record.steps),
        "termination_reason": record.final_state.termination_reason.value if record.final_state.termination_reason else None,
    }


def _tampered(record_file: Path, tmp_path: Path, mutate: Any) -> Path:
    data = json.loads(record_file.read_text(encoding="ascii"))
    mutate(data)
    path = tmp_path / "t.json"
    rewrite_json(path, data)
    return path


def _first_action(data: dict[str, Any]) -> None:
    step = data["steps"][0]
    person = sorted(step["actions"])[0]
    step["actions"][person] = "stay" if step["actions"][person] != "stay" else "north"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_first_action, ExitCode.MISMATCH),
        (lambda d: d["steps"].pop(), ExitCode.VALIDATION),  # a replayable prefix, but not a complete episode
        (lambda d: d.update(seed=d["seed"] + 1), ExitCode.MISMATCH),
        (lambda d: d.update(simulator_version="0.1.0"), ExitCode.MISMATCH),
        (lambda d: d.update(extra=1), ExitCode.VALIDATION),
        (lambda d: d.pop("steps"), ExitCode.VALIDATION),
    ],
    ids=["action", "truncated", "seed", "simulator-version", "unknown-key", "missing-key"],
)
def test_replay_verify_rejects_tampering(record_file: Path, tmp_path: Path, mutate: Any, code: ExitCode) -> None:
    path = _tampered(record_file, tmp_path, mutate)
    before = path.read_bytes()
    o = invoke("replay", "verify", "--scenario", SCN10, "--record", path, "--json")
    assert o.code == code, o.err
    assert o.error()["kind"] == ("mismatch" if code == ExitCode.MISMATCH else "validation")
    assert path.read_bytes() == before


def test_replay_verify_against_the_wrong_scenario(record_file: Path, tmp_path: Path) -> None:
    assert invoke("replay", "verify", "--scenario", SCN01, "--record", record_file).code == ExitCode.MISMATCH
    (tmp_path / "bad.json").write_bytes(b"{")
    assert invoke("replay", "verify", "--scenario", SCN10, "--record", tmp_path / "bad.json").code == ExitCode.VALIDATION
    assert invoke("replay", "verify", "--scenario", SCN10, "--record", tmp_path / "absent.json").code == ExitCode.VALIDATION


# --------------------------------------------------------------------------- evaluation run and verify


def test_evaluation_run_matches_the_library_runner(small: Path, manifest_file: Path, greedy_result: Path, tmp_path: Path) -> None:
    library = tmp_path / "library.json"
    run_evaluation(load_suite(small), load_evaluation_manifest(manifest_file), "greedy", agent_policy_factory(cli_agents.greedy), library)
    assert greedy_result.read_bytes() == library.read_bytes()


def test_evaluation_run_reports(small: Path, manifest_file: Path, tmp_path: Path) -> None:
    o = invoke("evaluation", "run", "--suite", small, "--manifest", manifest_file, "--model-id", "greedy", "--agent", "cli_agents:greedy", "--out", tmp_path / "r.json", "--json")
    assert o.code == 0, o.err
    result = load_evaluation_result(tmp_path / "r.json", load_suite(small))
    assert o.json() == {
        "command": "evaluation run",
        "model_id": "greedy",
        "result_version": "1.0",
        "manifest_hash": load_evaluation_manifest(manifest_file).manifest_hash(),
        "suite_id": result.manifest.suite_id,
        "suite_version": result.manifest.suite_version,
        "suite_hash": result.manifest.suite_hash,
        "result_digest": sha256_hex(result.to_dict()),
        "scheduled_episodes": 4,
        "completed_episodes": 4,
        "complete": True,
    }


def test_evaluation_run_resume(small: Path, manifest_file: Path, greedy_result: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    run = ("evaluation", "run", "--suite", small, "--manifest", manifest_file, "--model-id", "greedy", "--out", out)
    crashed = invoke(*run, "--agent", "cli_agents:crash_on_wide_maps")
    assert crashed.code == ExitCode.AGENT and "RuntimeError at step 0: wide map" in crashed.err
    partial = load_evaluation_result(out, load_suite(small))
    assert partial.completed_indices == {0, 1}  # the checkpoint of run_evaluation, kept for --resume

    before = out.read_bytes()
    refused = invoke(*run, "--agent", "cli_agents:greedy")
    assert refused.code == ExitCode.OUTPUT and "--resume" in refused.err and out.read_bytes() == before
    other = invoke("evaluation", "run", "--suite", small, "--manifest", manifest_file, "--model-id", "other", "--out", out, "--agent", "cli_agents:greedy", "--resume")
    assert other.code == ExitCode.OUTPUT and out.read_bytes() == before

    resumed = invoke(*run, "--agent", "cli_agents:greedy", "--resume", "--json")
    assert resumed.code == 0, resumed.err
    assert resumed.json()["complete"] is True
    assert out.read_bytes() == greedy_result.read_bytes()


def test_evaluation_run_rejects_a_foreign_manifest(small: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    assert invoke("evaluation", "manifest", "--suite", SUITE_DIR, "--seed", "1", "--out", manifest).code == 0
    o = invoke("evaluation", "run", "--suite", small, "--manifest", manifest, "--model-id", "m", "--agent", "cli_agents:greedy", "--out", tmp_path / "r.json")
    assert o.code == ExitCode.VALIDATION and not (tmp_path / "r.json").exists()


def test_evaluation_verify(small: Path, manifest_file: Path, greedy_result: Path) -> None:
    o = invoke("evaluation", "verify", "--suite", small, "--result", greedy_result, "--manifest", manifest_file, "--require-complete", "--json")
    assert o.code == 0, o.err
    data = o.json()
    assert data["verified"] is True and data["complete"] is True and data["model_id"] == "greedy"
    assert data["result_digest"] == sha256_hex(json.loads(greedy_result.read_text(encoding="ascii")))


def test_evaluation_verify_mismatches(small: Path, manifest_file: Path, greedy_result: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "7", "--seed", "11", "--out", other).code == 0
    o = invoke("evaluation", "verify", "--suite", small, "--result", greedy_result, "--manifest", other, "--json")
    assert o.code == ExitCode.MISMATCH and o.error()["kind"] == "mismatch"
    partial = tmp_path / "partial.json"
    data = json.loads(greedy_result.read_text(encoding="ascii"))
    data["episodes"] = data["episodes"][:2]
    rewrite_json(partial, data)
    assert invoke("evaluation", "verify", "--suite", small, "--result", partial).code == 0
    assert invoke("evaluation", "verify", "--suite", small, "--result", partial, "--require-complete").code == ExitCode.MISMATCH


def _retouch_record(data: dict[str, Any]) -> None:
    """Change one recorded action and re-seal the digest, so only a replay can tell."""
    record = data["episodes"][0]["record"]
    _first_action(record)
    data["episodes"][0]["record_digest"] = sha256_hex(record)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_retouch_record, ExitCode.VALIDATION),  # the metrics stage rejects the inconsistent trajectory first
        (lambda d: d["episodes"][0]["metrics"].update(evacuated=999), ExitCode.VALIDATION),
        (lambda d: d["episodes"][0].update(record_digest="0" * 64), ExitCode.VALIDATION),
        (lambda d: d["episodes"].append(d["episodes"][0]), ExitCode.VALIDATION),
        (lambda d: d.update(result_version="9.9"), ExitCode.VALIDATION),
        (lambda d: d.update(suite_hash="0" * 64), ExitCode.VALIDATION),
        (lambda d: d["manifest"]["episodes"][0].update(seed=12345), ExitCode.VALIDATION),
    ],
    ids=["replayed-action", "metrics", "digest", "duplicate", "result-version", "suite-hash", "manifest-seed"],
)
def test_evaluation_verify_never_drops_a_bad_episode(small: Path, greedy_result: Path, tmp_path: Path, mutate: Any, code: ExitCode) -> None:
    path = tmp_path / "t.json"
    data = json.loads(greedy_result.read_text(encoding="ascii"))
    mutate(data)
    rewrite_json(path, data)
    o = invoke("evaluation", "verify", "--suite", small, "--result", path)
    assert o.code == code, o.err


# --------------------------------------------------------------------------- statistics


def test_statistics_aggregate_writes_library_bytes(small: Path, greedy_result: Path, tmp_path: Path) -> None:
    out = tmp_path / "stats.json"
    o = invoke("statistics", "aggregate", "--suite", small, "--result", greedy_result, "--out", out, "--resamples", RESAMPLES, "--json")
    assert o.code == 0, o.err
    suite = load_suite(small)
    stats = aggregate_result(load_evaluation_result(greedy_result, suite), suite, resamples=int(RESAMPLES))
    assert out.read_bytes() == (stats.to_json() + "\n").encode("ascii")
    assert o.json() == {
        "command": "statistics aggregate",
        "statistics_version": "1.0",
        "digest": stats.digest(),
        "model_id": "greedy",
        "manifest_hash": stats.manifest.manifest_hash(),
        "resamples": int(RESAMPLES),
    }


def test_statistics_default_resamples_is_the_library_default(small: Path, greedy_result: Path, tmp_path: Path) -> None:
    o = invoke("statistics", "aggregate", "--suite", small, "--result", greedy_result, "--out", tmp_path / "s.json", "--json")
    assert o.code == 0, o.err
    assert o.json()["resamples"] == DEFAULT_BOOTSTRAP_RESAMPLES


def test_statistics_compare_writes_library_bytes(small: Path, greedy_result: Path, stay_result: Path, tmp_path: Path) -> None:
    out = tmp_path / "paired.json"
    o = invoke("statistics", "compare", "--suite", small, "--a", greedy_result, "--b", stay_result, "--out", out, "--resamples", RESAMPLES, "--json")
    assert o.code == 0, o.err
    suite = load_suite(small)
    paired = compare_results(
        load_evaluation_result(greedy_result, suite), load_evaluation_result(stay_result, suite), suite, resamples=int(RESAMPLES)
    )
    assert out.read_bytes() == (paired.to_json() + "\n").encode("ascii")
    data = o.json()
    assert (data["model_a"], data["model_b"], data["paired_count"], data["digest"]) == ("greedy", "stay", 4, paired.digest())


def test_statistics_fail_closed(small: Path, greedy_result: Path, stay_result: Path, tmp_path: Path) -> None:
    out = tmp_path / "s.json"
    tampered = tmp_path / "t.json"
    data = json.loads(greedy_result.read_text(encoding="ascii"))
    _retouch_record(data)
    rewrite_json(tampered, data)
    agg = ("statistics", "aggregate", "--suite", small, "--out", out, "--resamples", RESAMPLES)
    assert invoke(*agg, "--result", tampered).code == ExitCode.VALIDATION
    assert invoke(*agg, "--result", tmp_path / "absent.json").code == ExitCode.VALIDATION
    assert invoke("statistics", "aggregate", "--suite", SUITE_DIR, "--result", greedy_result, "--out", out).code == ExitCode.VALIDATION
    other_manifest = tmp_path / "m.json"
    assert invoke("evaluation", "manifest", "--suite", small, "--seed", "3", "--out", other_manifest).code == 0
    other = _evaluate(small, other_manifest, tmp_path / "other.json", "stay", "cli_agents:stay")
    cmp = ("statistics", "compare", "--suite", small, "--out", out, "--resamples", RESAMPLES)
    assert invoke(*cmp, "--a", greedy_result, "--b", other).code == ExitCode.VALIDATION
    assert not out.exists() and no_temp_files(tmp_path)
    out.write_bytes(b"keep")
    assert invoke(*cmp, "--a", greedy_result, "--b", stay_result).code == ExitCode.OUTPUT
    assert out.read_bytes() == b"keep"
    o = invoke("statistics", "aggregate", "--suite", small, "--result", tampered, "--out", tampered, "--force")
    assert o.code == ExitCode.OUTPUT and "input" in o.err


# --------------------------------------------------------------------------- determinism, environment and network


def _pipeline(base: Path, cwd: Path, env: dict[str, str] | None = None) -> dict[str, bytes]:
    """Run every artifact-producing command in a fresh interpreter; return the artifacts and the JSON stdout."""
    base.mkdir()
    agent = ("--agent-path", TESTS)
    steps: list[tuple[str, tuple[str | Path, ...]]] = [
        ("manifest", ("evaluation", "manifest", "--suite", SUITE_DIR, "--seed", "11", "--seed", "7", "--out", base / "manifest.json")),
        ("record", ("episode", "run", "--scenario", SCN10, "--seed", "7", "--agent", "cli_agents:greedy", *agent, "--out", base / "record.json")),
        ("replay", ("replay", "verify", "--scenario", SCN10, "--record", base / "record.json")),
    ]
    outputs: dict[str, bytes] = {}
    for name, args in steps:
        proc = run_cli(*args, "--json", cwd=cwd, env=env)
        assert proc.returncode == 0, proc.stderr
        assert proc.stderr == b""
        outputs[f"{name}.stdout"] = proc.stdout.replace(b"\r\n", b"\n")
    for path in sorted(base.iterdir()):
        outputs[path.name] = path.read_bytes()
    return outputs


def test_outputs_are_independent_of_hash_seed_and_cwd(tmp_path: Path) -> None:
    (tmp_path / "cwd-a").mkdir()
    (tmp_path / "cwd-b").mkdir()
    first = _pipeline(tmp_path / "a", tmp_path / "cwd-a", {"PYTHONHASHSEED": "0"})
    second = _pipeline(tmp_path / "b", tmp_path / "cwd-b", {"PYTHONHASHSEED": "12345", "EMBERBENCH_SEED": "1", "HOME": str(tmp_path)})
    assert first == second
    assert first["manifest.json"] == build_evaluation_manifest(load_suite(SUITE_DIR), [11, 7]).to_file_text().encode("ascii")


def test_repeated_commands_give_identical_bytes(small: Path, greedy_result: Path, tmp_path: Path) -> None:
    outs = []
    for name in ("a.json", "b.json"):
        o = invoke("statistics", "aggregate", "--suite", small, "--result", greedy_result, "--out", tmp_path / name, "--resamples", RESAMPLES, "--json")
        assert o.code == 0
        outs.append((o.out, (tmp_path / name).read_bytes()))
    assert outs[0] == outs[1]


def test_json_stdout_has_no_ansi_or_paths(small: Path, greedy_result: Path, record_file: Path) -> None:
    for args in (
        ("version",),
        ("suite", "validate", small),
        ("scenario", "validate", SCN01),
        ("replay", "verify", "--scenario", SCN10, "--record", record_file),
        ("evaluation", "verify", "--suite", small, "--result", greedy_result),
    ):
        o = invoke(*args, "--json")
        assert o.code == 0 and o.err == ""
        o.json()
        assert "\x1b" not in o.out
        assert str(small) not in o.out and str(record_file) not in o.out and str(greedy_result) not in o.out


_NETWORK_GUARD = """
import sys

def guard(event, args):
    if event.startswith(("socket.", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.")):
        raise RuntimeError("network access attempted: " + event)

sys.addaudithook(guard)
from emberbench.cli import main
codes = []
for argv in COMMANDS:
    codes.append(main(argv))
print(codes)
"""


def test_cli_makes_no_network_request(tmp_path: Path) -> None:
    suite = write_subset_suite(tmp_path / "suite", SMALL_IDS)
    m, r, rec = tmp_path / "m.json", tmp_path / "r.json", tmp_path / "rec.json"
    commands = [
        ["version", "--json"],
        ["scenario", "validate", str(SCN01)],
        ["suite", "validate", str(suite)],
        ["evaluation", "manifest", "--suite", str(suite), "--seed", "11", "--out", str(m)],
        ["episode", "run", "--scenario", str(SCN01), "--seed", "1", "--agent", "cli_agents:greedy", "--out", str(rec)],
        ["replay", "verify", "--scenario", str(SCN01), "--record", str(rec)],
        ["evaluation", "run", "--suite", str(suite), "--manifest", str(m), "--model-id", "g", "--agent", "cli_agents:greedy", "--out", str(r)],
        ["evaluation", "verify", "--suite", str(suite), "--result", str(r)],
        ["statistics", "aggregate", "--suite", str(suite), "--result", str(r), "--out", str(tmp_path / "s.json"), "--resamples", "100"],
        ["statistics", "compare", "--suite", str(suite), "--a", str(r), "--b", str(r), "--out", str(tmp_path / "p.json"), "--resamples", "100"],
    ]
    script = f"COMMANDS = {commands!r}\n" + _NETWORK_GUARD
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(TESTS)])
    proc = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.decode().splitlines()[-1] == str([0] * len(commands))


def test_cli_source_imports_no_network_or_config_modules() -> None:
    source = (ROOT / "src" / "emberbench" / "cli.py").read_text(encoding="utf-8")
    for banned in ("socket", "urllib", "http", "requests", "dotenv", "configparser", "os.environ", "getenv", "expanduser", "Path.home"):
        assert banned not in source, banned


# --------------------------------------------------------------------------- packaging


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("setuptools")
    # Build from a clean copy, so no build/ or egg-info directory in the checkout can leak into the wheel.
    source = tmp_path_factory.mktemp("source")
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(ROOT / name, source / name)
    dist = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index", "-w", str(dist), str(source)],
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0 and b"No module named pip" in proc.stderr:
        pytest.skip("pip is not available to build a wheel")
    assert proc.returncode == 0, proc.stderr
    (path,) = dist.glob("emberbench-*.whl")
    return path


def test_wheel_contains_only_the_package(wheel: Path) -> None:
    import zipfile

    names = zipfile.ZipFile(wheel).namelist()
    assert "emberbench/cli.py" in names and "emberbench/__main__.py" in names and "emberbench/py.typed" in names
    assert not [n for n in names if n.endswith(".json") or n.startswith(("scenarios/", "tests/"))]
    entry_points = next(n for n in names if n.endswith("entry_points.txt"))
    assert "emberbench = emberbench.cli:main" in zipfile.ZipFile(wheel).read(entry_points).decode()


def test_wheel_console_script_in_a_clean_environment(wheel: Path, tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True, timeout=600)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    install = subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], env=env, capture_output=True, timeout=600)
    assert install.returncode == 0, install.stderr
    script = bindir / ("emberbench.exe" if os.name == "nt" else "emberbench")

    def run(*args: str | Path) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run([str(script), *(str(a) for a in args)], cwd=tmp_path, env=env, capture_output=True, timeout=600)

    version = run("version", "--json")
    assert version.returncode == 0, version.stderr
    assert json.loads(version.stdout)["cli_version"] == CLI_VERSION
    assert run("scenario", "validate", SCN01, "--json").returncode == 0
    suite = run("suite", "validate", SUITE_DIR, "--json")
    assert suite.returncode == 0 and json.loads(suite.stdout)["suite_hash"] == SUITE_HASH
    where = subprocess.run([str(python), "-c", "import emberbench; print(emberbench.__file__)"], cwd=tmp_path, env=env, capture_output=True, timeout=600)
    assert str(venv.resolve()).lower() in where.stdout.decode().strip().lower()  # the wheel, not the source tree


@pytest.fixture(autouse=True)
def _restore_sys_modules() -> Iterator[None]:
    """Drop agent modules that a test imported from a temporary directory."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("cli_") and name != "cli_agents":
            del sys.modules[name]
