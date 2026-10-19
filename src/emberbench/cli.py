"""The ``emberbench`` command-line interface: a thin, deterministic, provider-free front end.

Every command calls the existing library API. The CLI adds argument parsing,
fail-closed input loading, no-overwrite atomic output and stable exit codes.
It never generates seeds, reads ambient configuration, contacts a network or
imports code that the operator did not name with ``--agent``.

``emberbench ...`` (the console script) and ``python -m emberbench ...`` run
the same :func:`main`. See ``docs/cli.md``.
"""

import argparse
import contextlib
import importlib
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any, NoReturn, TextIO

from . import SIMULATOR_VERSION
from .agent import ACTION_SCHEMA_VERSION, ActionOutput, Agent, AgentFactory, agent_policy, agent_policy_factory
from .canonical import canonical_json, sha256_hex
from .crowd import CROWD_DIAGNOSTICS_VERSION
from .errors import EmberBenchError, ModelOutputError, ReplayMismatchError, SimulationError
from .evaluation import (
    EVALUATION_MANIFEST_VERSION,
    EVALUATION_RESULT_VERSION,
    EvaluationManifest,
    EvaluationResult,
    build_evaluation_manifest,
    check_model_id,
    load_evaluation_manifest,
    load_evaluation_result,
    run_evaluation,
)
from .events import RunRecord
from .generator import GENERATION_MANIFEST_VERSION, GENERATOR_CONFIG_VERSION, GENERATOR_VERSION, load_generated_suite
from .metrics import METRICS_VERSION, validate_run_record
from .observation import OBSERVATION_VERSION, Observation
from .replay import verify_record
from .rng import MAX_SEED
from .scenario import SCHEMA_VERSION, Scenario, load_scenario
from .simulator import run_episode
from .statistics import STATISTICS_VERSION, aggregate_result, check_resamples, compare_results
from .suite import Suite, load_suite

CLI_VERSION = "1.0"
PROG = "emberbench"


class ExitCode(IntEnum):
    """Stable process exit codes. They are part of the CLI contract (``docs/cli.md``)."""

    OK = 0
    INTERNAL = 1  # unexpected failure: a bug in EMBERBench or the CLI
    USAGE = 2  # malformed command line or argument value
    VALIDATION = 3  # an input file is missing, malformed, tampered or fails a library check
    AGENT = 4  # the agent failed to load, to build, to act, or returned rejected output
    MISMATCH = 5  # a replay or an evaluation artifact does not match what it must reproduce
    OUTPUT = 6  # an output path conflicts with an existing file, or cannot be written
    INTERRUPTED = 130  # KeyboardInterrupt


_KIND = {
    ExitCode.INTERNAL: "internal",
    ExitCode.USAGE: "usage",
    ExitCode.VALIDATION: "validation",
    ExitCode.AGENT: "agent",
    ExitCode.MISMATCH: "mismatch",
    ExitCode.OUTPUT: "output",
    ExitCode.INTERRUPTED: "interrupted",
}


class CliError(Exception):
    """A failure with a known exit code and a deterministic message."""

    def __init__(self, code: ExitCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AgentExecutionError(Exception):
    """The operator's agent code raised, or broke the AgentFactory contract.

    Deliberately not an :class:`EmberBenchError`, so no library handler can
    absorb it on its way out of ``run_episode`` or ``run_evaluation``.
    """


def _usage(message: str) -> CliError:
    return CliError(ExitCode.USAGE, message)


def _invalid(message: str) -> CliError:
    return CliError(ExitCode.VALIDATION, message)


def _conflict(message: str) -> CliError:
    return CliError(ExitCode.OUTPUT, message)


def _agent_failure(message: str) -> CliError:
    return CliError(ExitCode.AGENT, message)


def _mismatch(message: str) -> CliError:
    return CliError(ExitCode.MISMATCH, message)


# --------------------------------------------------------------------------- version


def version_info() -> dict[str, str]:
    """Every contract version this build implements. Nothing here depends on the environment."""
    return {
        "cli_version": CLI_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "metrics_version": METRICS_VERSION,
        "statistics_version": STATISTICS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_config_version": GENERATOR_CONFIG_VERSION,
        "generation_manifest_version": GENERATION_MANIFEST_VERSION,
        "evaluation_manifest_version": EVALUATION_MANIFEST_VERSION,
        "evaluation_result_version": EVALUATION_RESULT_VERSION,
        "crowd_diagnostics_version": CROWD_DIAGNOSTICS_VERSION,
    }


# --------------------------------------------------------------------------- input loading (fail closed)


def _read_failure(path: str, exc: OSError) -> CliError:
    if isinstance(exc, FileNotFoundError):
        return _invalid(f"{path}: file not found")
    return _invalid(f"{path}: cannot read: {exc.strerror or type(exc).__name__}")


def _read_scenario(path: str) -> Scenario:
    try:
        return load_scenario(path)
    except OSError as exc:
        raise _read_failure(path, exc) from exc
    except UnicodeDecodeError as exc:
        raise _invalid(f"{path}: not UTF-8 text") from exc


def _read_suite(args: argparse.Namespace) -> Suite:
    """The suite named by ``--suite DIR`` or by ``--generated-suite DIR`` (a frozen generated suite)."""
    try:
        if args.generated_suite is not None:
            return load_generated_suite(args.generated_suite).suite
        return load_suite(args.suite)
    except OSError as exc:
        raise _read_failure(args.generated_suite or args.suite, exc) from exc
    except UnicodeDecodeError as exc:
        raise _invalid(f"{args.generated_suite or args.suite}: not UTF-8 text") from exc


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name!r} is not allowed")


def _read_json(path: str) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise _read_failure(path, exc) from exc
    except UnicodeDecodeError as exc:
        raise _invalid(f"{path}: not UTF-8 text") from exc
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise _invalid(f"{path}: not valid JSON: {exc}") from exc


def _read_record(path: str) -> RunRecord:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise _invalid(f"{path}: expected a run record object, got {type(data).__name__}")
    try:
        record = RunRecord.from_dict(data)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, EmberBenchError) as exc:
        raise _invalid(f"{path}: malformed run record: {exc!r}") from exc
    # from_dict ignores unknown keys; a record must round-trip exactly or it is not the record that was written.
    if canonical_json(record.to_dict()) != canonical_json(data):
        raise _invalid(f"{path}: run record is not in canonical form")
    return record


def _read_manifest(path: str) -> EvaluationManifest:
    try:
        return load_evaluation_manifest(path)
    except OSError as exc:
        raise _read_failure(path, exc) from exc
    except UnicodeDecodeError as exc:
        raise _invalid(f"{path}: not UTF-8 text") from exc


def _read_result(path: str, suite: Suite) -> EvaluationResult:
    try:
        return load_evaluation_result(path, suite)
    except OSError as exc:
        raise _read_failure(path, exc) from exc
    except UnicodeDecodeError as exc:
        raise _invalid(f"{path}: not UTF-8 text") from exc


# --------------------------------------------------------------------------- output (no overwrite, atomic)


def _check_output(path: str, *, force: bool, inputs: Sequence[str] = ()) -> Path:
    """Refuse an output path before any work is done. Parent directories are never created."""
    out = Path(path)
    if not out.parent.is_dir():
        raise _conflict(f"{path}: parent directory does not exist (it is never created)")
    if out.is_dir():
        raise _conflict(f"{path}: is a directory")
    if out.exists():
        for source in inputs:
            with contextlib.suppress(OSError):
                if os.path.samefile(out, source):
                    raise _conflict(f"{path}: is also an input file; an input is never rewritten")
        if not force:
            raise _conflict(f"{path}: already exists; pass --force to replace it")
    return out


def _exists_error(path: Path) -> CliError:
    return _conflict(f"{path}: already exists; pass --force to replace it")


def _remove_if_created(path: Path, created: os.stat_result) -> None:
    """Unlink ``path`` only if it is still the file this process created (same device and inode)."""
    with contextlib.suppress(FileNotFoundError):
        current = os.stat(path)
        if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
            os.unlink(path)


def _write_exclusive(path: Path, data: bytes) -> None:
    """Create ``path`` with ``O_EXCL`` and write ``data`` into it. An existing file is never opened.

    The fallback for filesystems without hard links. If the write fails, the
    file this call created is removed; a file that replaced it is left alone.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o666)
    except FileExistsError:
        raise _exists_error(path) from None
    try:
        created = os.fstat(fd)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # nothing else can have replaced it yet: it is still open and was just made
        raise
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _remove_if_created(path, created)
        raise


def write_output(path: Path, data: bytes, *, force: bool) -> None:
    """Write ``data`` to a temporary file beside ``path``, then move it into place.

    With ``force`` the temporary file is renamed over ``path``. Without
    ``force`` it is hard-linked to ``path``, which fails if ``path`` exists,
    so an existing file is never replaced, even one that appears while this
    runs. On a filesystem without hard links, ``path`` is created exclusively
    (``O_EXCL``) and written; there is no check-then-replace step. A failure
    at any point leaves no partial ``path`` and no temporary file.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if force:
            os.replace(tmp, path)
            return
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise _exists_error(path) from None
        except OSError:
            _write_exclusive(path, data)  # no hard links here
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _artifact_bytes(canonical: str) -> bytes:
    """Canonical JSON plus one LF, the file form ``save_evaluation_result`` uses."""
    return (canonical + "\n").encode("ascii")


# --------------------------------------------------------------------------- explicit agent loading

_AGENT_REFERENCE = re.compile(r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):(?P<attr>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$", re.ASCII)


def parse_agent_reference(reference: str) -> tuple[str, str]:
    """Split ``package.module:factory`` into its module and attribute paths. Anything else is a usage error."""
    match = _AGENT_REFERENCE.match(reference)
    if match is None:
        raise _usage(f"--agent: expected MODULE:ATTRIBUTE (for example my_pkg.agents:make_agent), got {reference!r}")
    return match["module"], match["attr"]


def _add_agent_paths(paths: Sequence[str]) -> None:
    resolved = []
    for p in paths:
        if not Path(p).is_dir():
            raise _usage(f"--agent-path: {p}: not a directory")
        resolved.append(str(Path(p).resolve()))
    sys.path[:0] = resolved


def load_agent_factory(reference: str) -> Callable[[], object]:
    """Import exactly the module named in ``reference`` and return the named zero-argument factory.

    No entry points, plugins, fallbacks or network locations are consulted:
    the module is found on ``sys.path`` as it stands (plus any ``--agent-path``).
    """
    module_name, attr_path = parse_agent_reference(reference)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            raise _agent_failure(f"--agent {reference}: module {module_name!r} not found; install it or pass --agent-path DIR") from exc
        raise _agent_failure(f"--agent {reference}: importing {module_name!r} failed: {type(exc).__name__}: {exc}") from exc
    except (Exception, SystemExit) as exc:
        raise _agent_failure(f"--agent {reference}: importing {module_name!r} failed: {type(exc).__name__}: {exc}") from exc
    obj: object = module
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise _agent_failure(f"--agent {reference}: {module_name!r} has no attribute {attr_path!r}") from exc
    if not callable(obj):
        raise _agent_failure(f"--agent {reference}: {type(obj).__name__} is not callable; expected a zero-argument agent factory")
    return obj


class _GuardedAgent:
    """Forward ``act`` unchanged and label any exception the model raises as an agent failure."""

    __slots__ = ("_act",)

    def __init__(self, act: Callable[[Observation], ActionOutput]) -> None:
        self._act = act

    def act(self, observation: Observation) -> ActionOutput:
        try:
            return self._act(observation)
        except Exception as exc:
            raise AgentExecutionError(f"agent raised {type(exc).__name__} at step {observation.step}: {exc}") from exc


def guarded_factory(factory: Callable[[], object], reference: str) -> AgentFactory:
    """Enforce the AgentFactory contract: every call returns a new object with ``act(observation)``."""
    seen: dict[int, object] = {}  # strong references keep each id unique

    def build() -> Agent:
        try:
            agent = factory()
        except Exception as exc:
            raise AgentExecutionError(f"agent factory {reference} raised {type(exc).__name__}: {exc}") from exc
        act = getattr(agent, "act", None)
        if not isinstance(agent, Agent) or not callable(act):
            raise AgentExecutionError(
                f"agent factory {reference} returned {type(agent).__name__}, not an agent with an act(observation) method"
            )
        if id(agent) in seen:
            raise AgentExecutionError(f"agent factory {reference} returned an agent of an earlier episode; it must build a new one")
        seen[id(agent)] = agent
        return _GuardedAgent(act)

    return build


def _agent_from_args(args: argparse.Namespace) -> AgentFactory:
    parse_agent_reference(args.agent)  # a malformed reference is a usage error, checked before any import
    _add_agent_paths(args.agent_path)
    return guarded_factory(load_agent_factory(args.agent), args.agent)


# --------------------------------------------------------------------------- argument values

_DECIMAL = re.compile(r"^[0-9]+$", re.ASCII)


def _seed_value(text: str) -> int:
    if not _DECIMAL.match(text):
        raise argparse.ArgumentTypeError(f"expected a decimal integer in [0, 2**64 - 1], got {text!r}")
    value = int(text)
    if value > MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be in [0, 2**64 - 1], got {text}")
    return value


def _model_id_value(text: str) -> str:
    try:
        return check_model_id(text)
    except EmberBenchError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _resamples_value(text: str) -> int:
    if not _DECIMAL.match(text):
        raise argparse.ArgumentTypeError(f"expected a decimal integer, got {text!r}")
    try:
        return check_resamples(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _resamples_kwargs(args: argparse.Namespace) -> dict[str, int]:
    """Pass ``resamples`` only when given, so the default is always the library default."""
    return {} if args.resamples is None else {"resamples": args.resamples}


# --------------------------------------------------------------------------- commands


def _cmd_version(args: argparse.Namespace) -> dict[str, Any]:
    return {"command": "version", **version_info()}


def _cmd_scenario_validate(args: argparse.Namespace) -> dict[str, Any]:
    s = _read_scenario(args.file)
    return {
        "command": "scenario validate",
        "valid": True,
        "scenario_id": s.scenario_id,
        "scenario_hash": s.content_hash(),
        "schema_version": s.schema_version,
    }


def _suite_summary(suite: Suite) -> dict[str, Any]:
    m = suite.manifest
    return {
        "suite_id": m.suite_id,
        "suite_version": m.suite_version,
        "suite_hash": m.suite_hash(),
        "visibility": m.visibility.value,
        "seed_role": m.seed_role.value,
        "schema_version": m.schema_version,
        "scenario_count": len(m.entries),
        "scenarios": [{"scenario_id": e.scenario_id, "content_hash": e.content_hash} for e in m.entries],
    }


def _cmd_suite_validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        suite = load_suite(args.dir)
    except OSError as exc:
        raise _read_failure(args.dir, exc) from exc
    return {"command": "suite validate", "valid": True, **_suite_summary(suite)}


def _cmd_generated_validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        frozen = load_generated_suite(args.dir)
    except OSError as exc:
        raise _read_failure(args.dir, exc) from exc
    g = frozen.generation
    return {
        "command": "generated validate",
        "valid": True,
        "generation_manifest_version": g.generation_manifest_version,
        "generator_version": g.generator_version,
        "batch_id": g.batch_id,
        "config_hash": g.config_hash,
        **_suite_summary(frozen.suite),
    }


def _cmd_evaluation_manifest(args: argparse.Namespace) -> dict[str, Any]:
    seeds: list[int] = args.seed
    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise _usage(f"--seed: duplicate seeds {duplicates}")
    out = _check_output(args.out, force=args.force)
    suite = _read_suite(args)
    manifest = build_evaluation_manifest(suite, seeds)  # explicit order, no generated seeds
    write_output(out, manifest.to_file_text().encode("ascii"), force=args.force)
    return {
        "command": "evaluation manifest",
        "manifest_version": manifest.manifest_version,
        "manifest_hash": manifest.manifest_hash(),
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "suite_hash": manifest.suite_hash,
        "episode_count": len(manifest.episodes),
        "seed_count": len(seeds),
    }


def _cmd_episode_run(args: argparse.Namespace) -> dict[str, Any]:
    out = _check_output(args.out, force=args.force, inputs=[args.scenario])
    scenario = _read_scenario(args.scenario)
    build = _agent_from_args(args)
    # The one simulator path: run_episode with a library-validated agent policy.
    record = run_episode(scenario, agent_policy(build()), seed=args.seed)
    verify_record(scenario, record)
    write_output(out, _artifact_bytes(record.to_json()), force=args.force)
    reason = record.final_state.termination_reason
    return {
        "command": "episode run",
        "scenario_id": record.scenario_id,
        "scenario_hash": record.scenario_hash,
        "simulator_version": record.simulator_version,
        "observation_version": record.observation_version,
        "record_digest": record.digest(),
        "steps": len(record.steps),
        "termination_reason": None if reason is None else reason.value,
    }


def _cmd_replay_verify(args: argparse.Namespace) -> dict[str, Any]:
    scenario = _read_scenario(args.scenario)
    record = _read_record(args.record)
    verify_record(scenario, record)  # a trajectory the simulator does not reproduce: exit 5
    validate_run_record(scenario, record)  # a complete, well-formed episode (a replayable prefix is not): exit 3
    reason = record.final_state.termination_reason
    return {
        "command": "replay verify",
        "verified": True,
        "scenario_id": record.scenario_id,
        "scenario_hash": record.scenario_hash,
        "record_digest": record.digest(),
        "simulator_version": record.simulator_version,
        "observation_version": record.observation_version,
        "steps": len(record.steps),
        "termination_reason": None if reason is None else reason.value,
    }


def _result_summary(result: EvaluationResult) -> dict[str, Any]:
    return {
        "model_id": result.model_id,
        "result_version": result.result_version,
        "manifest_hash": result.manifest_hash,
        "suite_id": result.manifest.suite_id,
        "suite_version": result.manifest.suite_version,
        "suite_hash": result.manifest.suite_hash,
        "result_digest": sha256_hex(result.to_dict()),
        "scheduled_episodes": len(result.manifest.episodes),
        "completed_episodes": len(result.episodes),
        "complete": result.is_complete,
    }


def _cmd_evaluation_run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    if not out.parent.is_dir():
        raise _conflict(f"{args.out}: parent directory does not exist (it is never created)")
    if out.is_dir():
        raise _conflict(f"{args.out}: is a directory")
    if out.exists() and not args.resume:
        raise _conflict(f"{args.out}: already exists; pass --resume to continue it (it is never overwritten)")
    _check_output(args.out, force=True, inputs=[args.manifest])
    suite = _read_suite(args)
    manifest = _read_manifest(args.manifest)
    if out.exists():
        # run_evaluation applies the same check; doing it first gives a conflict its own exit code.
        existing = _read_result(args.out, suite)
        if existing.manifest_hash != manifest.manifest_hash() or existing.model_id != args.model_id:
            raise _conflict(f"{args.out}: holds an artifact of another manifest or model ID; it is never overwritten")
    build = _agent_from_args(args)
    result = run_evaluation(suite, manifest, args.model_id, agent_policy_factory(build), out, resume=args.resume)
    return {"command": "evaluation run", **_result_summary(result)}


def _cmd_evaluation_verify(args: argparse.Namespace) -> dict[str, Any]:
    suite = _read_suite(args)
    result = _read_result(args.result, suite)
    if args.manifest is not None:
        expected = _read_manifest(args.manifest)
        if result.manifest_hash != expected.manifest_hash():
            raise _mismatch(f"{args.result}: embedded manifest {result.manifest_hash} is not {expected.manifest_hash()}")
    if args.require_complete and not result.is_complete:
        raise _mismatch(f"{args.result}: {len(result.episodes)} of {len(result.manifest.episodes)} episodes completed")
    return {"command": "evaluation verify", "verified": True, **_result_summary(result)}


def _cmd_statistics_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    out = _check_output(args.out, force=args.force, inputs=[args.result])
    suite = _read_suite(args)
    result = _read_result(args.result, suite)
    stats = aggregate_result(result, suite, **_resamples_kwargs(args))
    write_output(out, _artifact_bytes(stats.to_json()), force=args.force)
    return {
        "command": "statistics aggregate",
        "statistics_version": stats.statistics_version,
        "digest": stats.digest(),
        "model_id": stats.model_id,
        "manifest_hash": result.manifest_hash,
        "resamples": stats.resamples,
    }


def _cmd_statistics_compare(args: argparse.Namespace) -> dict[str, Any]:
    out = _check_output(args.out, force=args.force, inputs=[args.a, args.b])
    suite = _read_suite(args)
    a = _read_result(args.a, suite)
    b = _read_result(args.b, suite)
    paired = compare_results(a, b, suite, **_resamples_kwargs(args))
    write_output(out, _artifact_bytes(paired.to_json()), force=args.force)
    return {
        "command": "statistics compare",
        "statistics_version": paired.statistics_version,
        "digest": paired.digest(),
        "model_a": paired.model_a,
        "model_b": paired.model_b,
        "manifest_hash": a.manifest_hash,
        "paired_count": len(paired.paired),
        "resamples": paired.resamples,
    }


# --------------------------------------------------------------------------- parser


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser whose usage errors are deterministic and never call ``sys.exit`` directly."""

    def error(self, message: str) -> NoReturn:
        raise CliError(ExitCode.USAGE, f"{message} (see '{self.prog} --help')")


def _parser_options() -> dict[str, Any]:
    options: dict[str, Any] = {"allow_abbrev": False}
    if sys.version_info >= (3, 14):
        options["color"] = False  # never ANSI escapes, whatever the terminal or environment says
    return options


def build_parser() -> argparse.ArgumentParser:
    common = _Parser(add_help=False, **_parser_options())
    common.add_argument("--json", action="store_true", help="print only the result as one canonical JSON line on stdout")
    common.add_argument("--debug", action="store_true", help="print a Python traceback on failure")

    def suite_options(p: argparse.ArgumentParser) -> None:
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--suite", metavar="DIR", help="a suite directory (manifest.json plus scenario files)")
        group.add_argument("--generated-suite", metavar="DIR", help="a frozen generated suite (generation.json plus suite/)")

    def agent_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--agent", required=True, metavar="MODULE:FACTORY", help="explicit zero-argument AgentFactory to import")
        p.add_argument(
            "--agent-path", action="append", default=[], metavar="DIR", help="prepend DIR to the import path for --agent (repeatable)"
        )

    def output_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, metavar="FILE", help="output file; its parent directory must exist")
        p.add_argument("--force", action="store_true", help="replace an existing output file")

    def resamples_option(p: argparse.ArgumentParser) -> None:
        p.add_argument("--resamples", type=_resamples_value, default=None, metavar="N", help="bootstrap resamples (default: library default)")

    parser = _Parser(prog=PROG, description="Deterministic, provider-free EMBERBench command-line interface.", **_parser_options())
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    def group(name: str, help_text: str) -> Any:
        p = commands.add_parser(name, help=help_text, description=help_text, **_parser_options())
        return p.add_subparsers(dest="subcommand", metavar="SUBCOMMAND", parser_class=_Parser, required=True)

    def leaf(subs: Any, name: str, help_text: str, handler: Callable[[argparse.Namespace], dict[str, Any]]) -> argparse.ArgumentParser:
        p: argparse.ArgumentParser = subs.add_parser(name, parents=[common], help=help_text, description=help_text, **_parser_options())
        p.set_defaults(handler=handler)
        return p

    p: argparse.ArgumentParser = commands.add_parser("version", parents=[common], help="print contract versions", **_parser_options())
    p.set_defaults(handler=_cmd_version)

    scenario = group("scenario", "scenario files")
    p = leaf(scenario, "validate", "validate one scenario file", _cmd_scenario_validate)
    p.add_argument("file", metavar="FILE")

    suite = group("suite", "scenario suites")
    p = leaf(suite, "validate", "validate a suite directory against its manifest", _cmd_suite_validate)
    p.add_argument("dir", metavar="DIR")

    generated = group("generated", "frozen generated suites")
    p = leaf(generated, "validate", "validate a frozen generated suite against its freeze record", _cmd_generated_validate)
    p.add_argument("dir", metavar="DIR")

    evaluation = group("evaluation", "evaluation manifests and results")
    p = leaf(evaluation, "manifest", "build an evaluation manifest from a suite and an ordered seed list", _cmd_evaluation_manifest)
    suite_options(p)
    p.add_argument("--seed", action="append", required=True, type=_seed_value, metavar="N", help="evaluator seed (repeatable, order kept)")
    output_options(p)
    p = leaf(evaluation, "run", "run every manifest episode with a local agent (resumable)", _cmd_evaluation_run)
    suite_options(p)
    p.add_argument("--manifest", required=True, metavar="FILE")
    p.add_argument("--model-id", required=True, type=_model_id_value, metavar="ID", help="free-text model label stored in the result")
    agent_options(p)
    p.add_argument("--out", required=True, metavar="FILE", help="result artifact; its parent directory must exist")
    p.add_argument("--resume", action="store_true", help="continue an existing artifact of the same manifest and model ID")
    p = leaf(evaluation, "verify", "authenticate an evaluation result against a suite", _cmd_evaluation_verify)
    suite_options(p)
    p.add_argument("--result", required=True, metavar="FILE")
    p.add_argument("--manifest", metavar="FILE", help="also require this exact evaluation manifest")
    p.add_argument("--require-complete", action="store_true", help="also require every scheduled episode")

    episode = group("episode", "single episodes")
    p = leaf(episode, "run", "run one scenario with one seed and a local agent", _cmd_episode_run)
    p.add_argument("--scenario", required=True, metavar="FILE")
    p.add_argument("--seed", required=True, type=_seed_value, metavar="N", help="evaluator seed")
    agent_options(p)
    output_options(p)

    replay = group("replay", "run records")
    p = leaf(replay, "verify", "authenticate a run record by replay", _cmd_replay_verify)
    p.add_argument("--scenario", required=True, metavar="FILE")
    p.add_argument("--record", required=True, metavar="FILE")

    statistics = group("statistics", "statistics artifacts")
    p = leaf(statistics, "aggregate", "summarize one authenticated result", _cmd_statistics_aggregate)
    suite_options(p)
    p.add_argument("--result", required=True, metavar="FILE")
    resamples_option(p)
    output_options(p)
    p = leaf(statistics, "compare", "compare two authenticated results of one manifest (A - B)", _cmd_statistics_compare)
    suite_options(p)
    p.add_argument("--a", required=True, metavar="FILE", help="result A")
    p.add_argument("--b", required=True, metavar="FILE", help="result B")
    resamples_option(p)
    output_options(p)
    return parser


# --------------------------------------------------------------------------- output and errors


def _write(stream: TextIO, text: str) -> None:
    try:
        stream.write(text)
    except UnicodeEncodeError:
        stream.write(text.encode("ascii", "backslashreplace").decode("ascii"))
    stream.flush()


def _human(payload: dict[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend("  - " + " ".join(f"{k}={v}" for k, v in item.items()) for item in value)
        elif isinstance(value, bool) or value is None:
            lines.append(f"{key}: {json.dumps(value)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _causes(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify(exc: BaseException) -> tuple[ExitCode, str]:
    """Map an exception to its stable exit code and a one-line message."""
    if isinstance(exc, CliError):
        return exc.code, exc.message
    if isinstance(exc, KeyboardInterrupt):
        return ExitCode.INTERRUPTED, "interrupted"
    if isinstance(exc, AgentExecutionError):
        return ExitCode.AGENT, str(exc)
    if isinstance(exc, ModelOutputError):
        return ExitCode.AGENT, f"model output rejected: {exc}"
    if any(isinstance(e, ReplayMismatchError) for e in _causes(exc)):
        return ExitCode.MISMATCH, str(exc)
    if isinstance(exc, SimulationError):
        return ExitCode.INTERNAL, f"internal error: {type(exc).__name__}: {exc}"
    if isinstance(exc, EmberBenchError):
        return ExitCode.VALIDATION, str(exc)
    if isinstance(exc, OSError):
        return ExitCode.OUTPUT, f"cannot write output: {exc.strerror or type(exc).__name__}"
    return ExitCode.INTERNAL, f"internal error: {type(exc).__name__}: {exc}"


def _report_failure(exc: BaseException, *, as_json: bool, debug: bool) -> int:
    code, message = classify(exc)
    if as_json:
        _write(sys.stderr, canonical_json({"error": {"exit_code": int(code), "kind": _KIND[code], "message": message}}) + "\n")
    else:
        hint = "" if debug or code is not ExitCode.INTERNAL else " (rerun with --debug for a traceback)"
        _write(sys.stderr, f"{PROG}: error: {message}{hint}\n")
    if debug:
        _write(sys.stderr, "".join(traceback.format_exception(exc)))
    return int(code)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code. Nothing is read from the environment or the cwd."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args_list
    debug = "--debug" in args_list
    saved_path = list(sys.path)
    try:
        parser = build_parser()
        try:
            args = parser.parse_args(args_list)
        except SystemExit as exc:  # --help prints and exits 0; usage errors raise CliError instead
            return exc.code if isinstance(exc.code, int) else int(ExitCode.OK if exc.code is None else ExitCode.USAGE)
        if getattr(args, "handler", None) is None:
            _write(sys.stderr, parser.format_usage())
            return int(ExitCode.USAGE)
        as_json, debug = args.json, args.debug
        payload = args.handler(args)
        _write(sys.stdout, canonical_json(payload) + "\n" if as_json else _human(payload))
        return int(ExitCode.OK)
    except (Exception, KeyboardInterrupt) as exc:
        return _report_failure(exc, as_json=as_json, debug=debug)
    finally:
        sys.path[:] = saved_path
