"""Structural leak guard for the repository and the built distributions.

The repository and the package may contain public development assets only
(see :data:`emberbench.integrity.ASSET_POLICIES`). This module checks paths and known construction-content markers: every path must sit under an allowlisted root,
benchmark asset files (JSON) only under :data:`PUBLIC_ASSET_ROOTS`, and no
path component may look like private evaluation material (``hidden/``,
``heldout/``, ``scored/``, ``private-seeds*``, ``secret-manifest*``,
``production-eval*``, an evaluator workspace file, a key file, ...).

The checks are deterministic and need no network. ``tools/check_public_assets.py``
runs them on the checkout and on built wheels and sdists.
"""

import ast
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Top-level repository entries. Anything else in a checkout is a violation.
REPOSITORY_TOP_LEVEL_FILES = frozenset({".gitattributes", ".gitignore", "LICENSE", "README.md", "pyproject.toml"})
REPOSITORY_TOP_LEVEL_DIRS = frozenset({".github", "docs", "scenarios", "src", "tests", "tools"})

# The only directories that may hold benchmark asset files.
PUBLIC_ASSET_ROOTS: tuple[str, ...] = ("scenarios/public/", "tests/fixtures/")
ASSET_SUFFIXES = frozenset({".json"})

# Path components (lowercased, "_" read as "-") that name private evaluation material.
FORBIDDEN_NAMES = frozenset({"hidden", "heldout", "held-out", "scored", "private", "secret", "secrets"})
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "hidden-",
    "heldout-",
    "held-out-",
    "scored-",
    "private-seed",
    "private-eval",
    "secret-manifest",
    "production-eval",
    "evaluation-run",
)
# Evaluator workspace and freeze-record file names (see emberbench.integrity and emberbench.generator).
FORBIDDEN_FILE_NAMES = frozenset(
    {
        "generation.json",
        "evaluation-manifest.json",
        "submission-roster.json",
        "identity.json",
        "provenance.json",
        "result.json",
        "suite-freeze.json",
    }
)
FORBIDDEN_SUFFIXES = frozenset({".seed", ".seeds", ".key", ".pem", ".p12", ".pfx"})

# The wheel may hold the package modules and its metadata, nothing else.
WHEEL_PACKAGE_DIR = "emberbench/"
WHEEL_PACKAGE_SUFFIXES = frozenset({".py"})
WHEEL_PACKAGE_FILES = frozenset({"py.typed"})
# Files setuptools adds to an sdist next to the repository files.
SDIST_GENERATED_FILES = frozenset({"PKG-INFO", "setup.cfg"})
SDIST_GENERATED_DIRS: tuple[str, ...] = ("src/emberbench.egg-info/",)


@dataclass(frozen=True)
class LeakViolation:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _normalized(part: str) -> str:
    return part.lower().replace("_", "-")


def forbidden_reason(path: str) -> str | None:
    """Why ``path`` (POSIX, relative) looks like private evaluation material, or ``None``."""
    parts = PurePosixPath(path).parts
    for part in parts:
        name = _normalized(part)
        stem = name.split(".", 1)[0]
        if stem in FORBIDDEN_NAMES:
            return f"component {part!r} names private evaluation material"
        if name.startswith(FORBIDDEN_PREFIXES):
            return f"component {part!r} names private evaluation material"
    name = _normalized(parts[-1]) if parts else ""
    if name in FORBIDDEN_FILE_NAMES:
        return f"{parts[-1]!r} is an evaluator workspace or freeze-record file"
    if PurePosixPath(name).suffix in FORBIDDEN_SUFFIXES:
        return f"{parts[-1]!r} has a seed or key file type"
    return None


def check_repository_paths(paths: Iterable[str]) -> list[LeakViolation]:
    """Check relative POSIX paths of a checkout against the allowlist and the forbidden patterns."""
    violations: list[LeakViolation] = []
    for path in sorted(set(paths)):
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            violations.append(LeakViolation(path, "not a relative path inside the repository"))
            continue
        reason = forbidden_reason(path)
        if reason is not None:
            violations.append(LeakViolation(path, reason))
            continue
        top = pure.parts[0]
        if len(pure.parts) == 1:
            if top not in REPOSITORY_TOP_LEVEL_FILES:
                violations.append(LeakViolation(path, "top-level file is not on the public allowlist"))
            continue
        if top not in REPOSITORY_TOP_LEVEL_DIRS:
            violations.append(LeakViolation(path, f"top-level directory {top!r} is not on the public allowlist"))
            continue
        if top == "scenarios" and not path.startswith("scenarios/public/"):
            violations.append(LeakViolation(path, "only scenarios/public/ may hold scenarios"))
            continue
        if pure.suffix.lower() in ASSET_SUFFIXES and not path.startswith(PUBLIC_ASSET_ROOTS):
            violations.append(LeakViolation(path, f"benchmark asset files may live only under {list(PUBLIC_ASSET_ROOTS)}"))
    return violations


def repository_paths(root: str | Path, *, include_untracked: bool = False) -> list[str]:
    """Files of the git checkout at ``root``: tracked ones, plus untracked, not ignored ones if asked.

    ``include_untracked`` catches a held-out file before it is committed, but
    also reports local tool files. Raises :class:`RuntimeError` if ``root`` is
    not a git checkout.
    """
    command = ["git", "ls-files", "-z", "--cached"] + (["--others", "--exclude-standard"] if include_untracked else [])
    try:
        out = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{root}: cannot list repository files with git: {exc}") from exc
    paths = sorted({p for p in out.decode("utf-8").split("\0") if p})
    # A file deleted in the working tree but still in the index is not shipped.
    return [p for p in paths if (Path(root) / p).exists()]


PRIVATE_DEFINITIONS = frozenset({
    "INITIAL_POOL_SIZE", "TOP_UP_BATCH_SIZE", "POOL_CAP", "V1_DESIGN",
    "V1_RECIPES", "V1_INITIAL_PLAN", "V1_TOP_UP", "_catalog", "_v1_top_up",
    "top_up_recipe", "select_suite", "generate_batch",
})
PRIVATE_JSON_FIELDS = re.compile(
    r'"(?:candidate_pool_size|pool_cap|top_up_batch_size|stochastic_per_family|private_mixture|official_family_thresholds)"\s*:'
)


def check_public_payload(path: str, payload: bytes) -> list[LeakViolation]:
    """Reject known construction recipes under otherwise innocuous public paths.

    This is a regression tripwire, not proof against arbitrary confidential text.
    Human publication review and the private-workspace boundary remain required.
    """
    suffix = PurePosixPath(path).suffix.lower()
    if suffix not in {".py", ".md", ".json", ".txt"}:
        return []
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [LeakViolation(path, "unreviewable non-UTF-8 public text")]
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [LeakViolation(path, "unreviewable Python source")]
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        names.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        if names & PRIVATE_DEFINITIONS:
            return [LeakViolation(path, "retired evaluation-construction definition in public source")]
    elif suffix in {".md", ".json"}:
        if any(re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text) for name in PRIVATE_DEFINITIONS) or PRIVATE_JSON_FIELDS.search(text):
            return [LeakViolation(path, "evaluation-construction recipe in public content")]
    return []


def check_repository(root: str | Path, *, include_untracked: bool = False) -> list[LeakViolation]:
    paths = repository_paths(root, include_untracked=include_untracked)
    violations = check_repository_paths(paths)
    for name in paths:
        path = Path(root) / name
        if path.is_symlink():
            violations.append(LeakViolation(name, "symlink crosses the publication boundary"))
        elif path.is_file():
            violations.extend(check_public_payload(name, path.read_bytes()))
    return violations


def check_wheel_members(names: Iterable[str]) -> list[LeakViolation]:
    violations: list[LeakViolation] = []
    for name in sorted(set(names)):
        if name.endswith("/"):
            continue
        pure = PurePosixPath(name)
        if pure.parts and pure.parts[0].endswith(".dist-info"):
            if len(pure.parts) == 2 or (len(pure.parts) == 3 and pure.parts[1] == "licenses"):
                continue
            violations.append(LeakViolation(name, "unexpected nested dist-info file"))
            continue
        reason = forbidden_reason(name)
        if reason is not None:
            violations.append(LeakViolation(name, reason))
        elif not name.startswith(WHEEL_PACKAGE_DIR) or len(pure.parts) != 2:
            violations.append(LeakViolation(name, "the wheel may hold only top-level emberbench modules and metadata"))
        elif pure.suffix not in WHEEL_PACKAGE_SUFFIXES and pure.name not in WHEEL_PACKAGE_FILES:
            violations.append(LeakViolation(name, "the wheel may hold only Python modules and py.typed"))
    return violations


def check_sdist_members(names: Iterable[str]) -> list[LeakViolation]:
    """Check sdist file members: ``emberbench-<version>/`` plus repository paths and setuptools metadata."""
    violations: list[LeakViolation] = []
    relative: list[str] = []
    for name in sorted(set(names)):
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or not parts[0].startswith("emberbench-"):
            violations.append(LeakViolation(name, "sdist member outside the emberbench-<version>/ root"))
            continue
        rel = PurePosixPath(*parts[1:]).as_posix()
        if rel in SDIST_GENERATED_FILES:
            continue
        if rel.startswith(SDIST_GENERATED_DIRS):
            reason = forbidden_reason(rel)
            if reason is not None:
                violations.append(LeakViolation(name, reason))
            continue
        relative.append(rel)
    violations += [LeakViolation(f"<sdist>/{v.path}", v.reason) for v in check_repository_paths(relative)]
    return violations


def check_distribution(path: str | Path) -> list[LeakViolation]:
    """Check a wheel or sdist by member paths and known construction payloads."""
    p = Path(path)
    if p.suffix == ".whl":
        with zipfile.ZipFile(p) as z:
            violations = check_wheel_members(z.namelist())
            for name in z.namelist():
                if not name.endswith("/"):
                    violations.extend(check_public_payload(name, z.read(name)))
            return violations
    if p.name.endswith(".tar.gz"):
        with tarfile.open(p, "r:gz") as t:
            members = [m for m in t.getmembers() if not m.isdir()]
            violations = check_sdist_members(m.name for m in members)
            for member in members:
                if not member.isfile():
                    violations.append(LeakViolation(member.name, "non-regular archive member"))
                    continue
                stream = t.extractfile(member)
                if stream is not None:
                    violations.extend(check_public_payload(member.name, stream.read()))
            return violations
    return [LeakViolation(str(p), "not a wheel or sdist")]
