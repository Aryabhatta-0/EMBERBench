"""Structural leak guard for the repository and the built distributions.

The repository and the package may contain public development assets only
(see :data:`emberbench.integrity.ASSET_POLICIES`). This module checks file
*paths*, not file contents: every path must sit under an allowlisted root,
benchmark asset files (JSON) only under :data:`PUBLIC_ASSET_ROOTS`, and no
path component may look like private evaluation material (``hidden/``,
``heldout/``, ``scored/``, ``private-seeds*``, ``secret-manifest*``,
``production-eval*``, an evaluator workspace file, a key file, ...).

The checks are deterministic and need no network. ``tools/check_public_assets.py``
runs them on the checkout and on built wheels and sdists.
"""

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


def check_repository(root: str | Path, *, include_untracked: bool = False) -> list[LeakViolation]:
    return check_repository_paths(repository_paths(root, include_untracked=include_untracked))


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
    """Check one built ``.whl`` or ``.tar.gz`` by its member paths."""
    p = Path(path)
    if p.suffix == ".whl":
        with zipfile.ZipFile(p) as z:
            return check_wheel_members(z.namelist())
    if p.name.endswith(".tar.gz"):
        with tarfile.open(p, "r:gz") as t:
            names = [m.name for m in t.getmembers() if not m.isdir()]
        return check_sdist_members(names)
    return [LeakViolation(str(p), "not a wheel or sdist")]
