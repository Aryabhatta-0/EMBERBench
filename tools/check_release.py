"""Release gate: version contracts, reproducible builds, metadata, clean install, registry and leak guard.

Usage::

    python tools/check_release.py --dist dist [--wheelhouse DIR] [--require-license] [--require-clean]
    python tools/check_release.py --contracts-only

Needs ``build``, ``setuptools``, ``packaging`` and ``pytest`` (pinned in
tools/dev-constraints.txt). It never imports EMBERBench from the checkout:
everything about the package is asked of the wheel it builds, installed in a
clean virtual environment. Every step is fatal on failure and says what to
fix:

0. Source checks, without building: the working tree is clean (warning, or
   fatal with ``--require-clean``), the public V1 suite is complete, and no
   README or doc names a stale benchmark revision.
1. Copy the checkout's files (tracked, plus untracked and not ignored) into
   two fresh source trees.
2. Build each tree with ``python -m build --no-isolation`` (sdist, then the
   wheel from the sdist) under one frozen ``SOURCE_DATE_EPOCH`` and no
   package index (offline).
3. Compare the two builds: identical member names and member content digests
   for the wheel and the sdist, and byte-identical wheels. Every wheel
   member carries the ``SOURCE_DATE_EPOCH`` time. Byte identity of the
   sdists is reported, not required (docs/release-checklist.md,
   "Reproducibility boundary").
4. Build a wheel straight from the source tree and require the same members
   and content as the wheel built from the sdist.
5. With ``--wheelhouse``: build the wheel from the sdist again in an isolated
   build environment that may install only from that directory. Put exactly
   the pinned build backend there; an unexpected build dependency fails.
6. Check the wheel and sdist metadata (name, version, summary, Python
   requirement, README, no runtime dependencies, console script, py.typed),
   the license (warning, or fatal with ``--require-license``), the wheel
   member allowlist and the archive size limits.
7. Install the wheel with no index into a new virtual environment under a
   path with spaces, import the package from it (not from the checkout), run
   the console script and ``python -m emberbench version --json``, and check
   every reported version against docs/versioning.md, the changelog, the
   release checklist and docs/integrity.md.
8. Validate the public suite by an explicit path and write the CLI manifest
   artifact with the installed wheel; both must match the digest registry.
9. Recompute the whole regression digest registry with the installed wheel.
10. Run the public-asset leak guard (tools/check_public_assets.py) with the
    installed wheel against the repository and both archives.

``--contracts-only`` runs step 0 and the version-contract checks of step 7
against ``python -m emberbench version --json`` of the current interpreter,
without building. The first build is copied to ``--dist``.
"""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
REGISTRY = REPO_ROOT / "tests" / "fixtures" / "release" / "digest_registry.json"
SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"
# Used only when SOURCE_DATE_EPOCH is unset and the checkout has no commit time (for example an exported tree).
FALLBACK_SOURCE_DATE_EPOCH = "1767225600"  # 2026-01-01T00:00:00Z
# A zip archive cannot hold a time before 1980-01-01.
ZIP_EPOCH_FLOOR = 315532800

# Version constants that are a `suite_version` label, not a separate line of `emberbench version`, with the reason.
UNREPORTED_LABELS = {
    "GENERATED_SUITE_VERSION": "suite_version label of generated suites; content identity is the suite hash (GENERATOR_VERSION)",
    "COMPOSED_SUITE_VERSION": "suite_version label of composed suites; content identity is owned by the composition versions",
}
# benchmark_versions() keys whose `emberbench version` key is not "<key>_version" (CLI 1.0 names).
BENCHMARK_VERSION_KEYS = {"benchmark_revision": "benchmark_revision", "scenario_schema": "schema_version"}

# Size limits: the wheel holds code only; the sdist holds code, tests and docs. Generated research
# outputs (for example evaluation-pack runs) are never package data.
WHEEL_MAX_BYTES = 1_000_000
SDIST_MAX_UNPACKED_BYTES = 4_000_000
MEMBER_MAX_BYTES = 512_000
WHEEL_PACKAGE_SUFFIXES = (".py", "/py.typed")
WHEEL_DIST_INFO_FILES = ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")

LICENSE_TODO = (
    "no license is configured. This is a maintainer decision and a release blocker. Once a license is chosen, add: "
    "(1) a LICENSE file at the repository root with the license text; "
    "(2) in pyproject.toml [project]: license = \"<SPDX expression>\" and license-files = [\"LICENSE\"]; "
    "(3) the license in docs/release-checklist.md and docs/changelog.md. "
    "The wheel must then carry License-Expression and License-File metadata and the LICENSE file"
)


class ReleaseCheckError(Exception):
    pass


def step(message: str) -> None:
    print(f"== {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ReleaseCheckError(f"{' '.join(command)} exited with {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc


def source_files() -> list[str]:
    out = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT).stdout
    return sorted(p for p in set(out.split("\0")) if p and (REPO_ROOT / p).is_file())


def copy_source(dest: Path) -> Path:
    for rel in source_files():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, target)
    return dest


def source_date_epoch() -> str:
    """``SOURCE_DATE_EPOCH`` if set, else the time of the checked-out commit."""
    value = os.environ.get("SOURCE_DATE_EPOCH", "")
    if value:
        if not value.isdigit():
            raise ReleaseCheckError(f"SOURCE_DATE_EPOCH={value!r} is not a non-negative integer")
        return value
    proc = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=REPO_ROOT, capture_output=True, text=True)
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value.isdigit() else FALLBACK_SOURCE_DATE_EPOCH


def build_env(epoch: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env.update({"SOURCE_DATE_EPOCH": epoch, "PIP_NO_INDEX": "1", "PYTHONHASHSEED": "0"})
    return env


def build(source: Path, outdir: Path, env: dict[str, str], *flags: str) -> None:
    run([sys.executable, "-m", "build", "--no-isolation", *flags, "--outdir", str(outdir), str(source)], env=env)


def only(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise ReleaseCheckError(f"{directory}: expected one {pattern}, found {[p.name for p in found]}")
    return found[0]


def wheel_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {name: hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist()) if not name.endswith("/")}


def sdist_members(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as t:
        for member in t.getmembers():
            if member.isfile():
                data = t.extractfile(member)
                assert data is not None
                out[member.name] = hashlib.sha256(data.read()).hexdigest()
    return dict(sorted(out.items()))


def compare(label: str, a: dict[str, str], b: dict[str, str]) -> None:
    if a == b:
        return
    names = sorted(set(a) ^ set(b))
    changed = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    raise ReleaseCheckError(f"{label} differ: members only in one build {names}, content differs {changed}")


def same_bytes(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def wheel_time_problems(wheel: Path, epoch: str) -> list[str]:
    expected = time.gmtime(max(int(epoch), ZIP_EPOCH_FLOOR))[:6]
    with zipfile.ZipFile(wheel) as z:
        wrong = sorted(i.filename for i in z.infolist() if i.date_time != expected)
    return [f"wheel members {wrong[:5]} do not carry SOURCE_DATE_EPOCH={epoch} ({expected})"] if wrong else []


def project() -> tuple[dict[str, object], str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tree = ast.parse((REPO_ROOT / "src" / "emberbench" / "_version.py").read_text(encoding="utf-8"))
    version = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    return data["project"], version


# --------------------------------------------------------------------------- source checks


def dirty_paths() -> list[str]:
    out = run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO_ROOT).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def public_asset_problems() -> list[str]:
    manifest_path = SUITE_DIR / "manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path.relative_to(REPO_ROOT)} is missing: the public V1 suite must be in the repository"]
    entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("scenarios") or []
    if not entries:
        return ["the public suite manifest lists no scenario"]
    return [f"public suite file {(SUITE_DIR / e['path']).relative_to(REPO_ROOT)} is missing" for e in entries if not (SUITE_DIR / e["path"]).is_file()]


_REVISION_CLAIMS = (
    re.compile(r'BENCHMARK_REVISION\s*=\s*"(\d+)"'),
    re.compile(r'"benchmark_revision"\s*:\s*"(\d+)"'),
    re.compile(r"benchmark revision `(\d+)`", re.IGNORECASE),
    re.compile(r"benchmark revision \| `(\d+)`", re.IGNORECASE),
    re.compile(r"`(\d+)` \(`BENCHMARK_REVISION`\)"),
)


def stale_revision_problems(revision: str) -> list[str]:
    """Every README or docs line that states a benchmark revision other than ``revision``."""
    problems = []
    for path in [REPO_ROOT / "README.md", *sorted(DOCS.glob("*.md"))]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in _REVISION_CLAIMS:
                for match in pattern.finditer(line):
                    if match.group(1) != revision:
                        problems.append(f"{path.relative_to(REPO_ROOT)}:{number}: states benchmark revision {match.group(1)}, current is {revision}")
    return problems


def _cli_key(benchmark_key: str) -> str:
    return BENCHMARK_VERSION_KEYS.get(benchmark_key, f"{benchmark_key}_version")


def versioning_table() -> dict[str, tuple[str, str]]:
    """``{constant: (value, "yes" | "no")}`` from the component table of docs/versioning.md."""
    text = (DOCS / "versioning.md").read_text(encoding="utf-8")
    return {name: (value, prov) for name, value, prov in re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \| `([^`]+)` \| (yes|no) \|", text, re.M)}


def version_contract_problems(reported: dict[str, str], provenance: dict[str, str]) -> list[str]:
    """Check what ``emberbench version --json`` reports against the source, the docs and provenance.

    ``reported`` is the version output without ``command``; ``provenance`` is
    ``benchmark_versions()`` of the same build. Each problem says what to add
    or change.
    """
    _, source_version = project()
    problems: list[str] = []
    if reported.get("package_version") != source_version:
        problems.append(f"package_version is {reported.get('package_version')!r}, src/emberbench/_version.py says {source_version!r}")
    revision = reported.get("benchmark_revision", "")
    if provenance.get("benchmark_revision") != revision:
        problems.append(f"benchmark_revision {revision!r} differs from benchmark_versions() {provenance.get('benchmark_revision')!r}")
    for key, value in provenance.items():
        if reported.get(_cli_key(key)) != value:
            problems.append(f"benchmark_versions()[{key!r}] = {value!r} is not reported by `emberbench version` as {_cli_key(key)!r}")
    in_provenance = {_cli_key(k).upper() for k in provenance}
    versions = {key.upper(): value for key, value in reported.items() if key != "package_version"}

    table = versioning_table()
    changelog = (DOCS / "changelog.md").read_text(encoding="utf-8")
    checklist = (DOCS / "release-checklist.md").read_text(encoding="utf-8")
    integrity = (DOCS / "integrity.md").read_text(encoding="utf-8")
    for name, value in versions.items():
        prov = "yes" if name in in_provenance else "no"
        if table.get(name) != (value, prov):
            problems.append(f"docs/versioning.md section 3: row for {name} is {table.get(name)}, expected | `{name}` | `{value}` | {prov} |")
        if name != "BENCHMARK_REVISION" and f"`{name}` `{value}`" not in changelog:
            problems.append(f"docs/changelog.md: the current entry does not list `{name}` `{value}`")
        if f"`{name}`" not in checklist:
            problems.append(f"docs/release-checklist.md: the version review does not name `{name}`")
        if prov == "yes" and f"`{name}`" not in integrity:
            problems.append(f"docs/integrity.md section 12: provenance pins {name}, but the revision policy does not name it")
    for name in sorted(set(table) - set(versions) - set(UNREPORTED_LABELS)):
        problems.append(f"docs/versioning.md lists {name}, which `emberbench version` does not report")
    header = f"package `{source_version}`, benchmark revision `{revision}`"
    if header not in changelog:
        problems.append(f"docs/changelog.md: no entry headed with ({header})")
    problems += stale_revision_problems(revision)
    return problems


# --------------------------------------------------------------------------- metadata


def license_configured() -> bool:
    proj, _ = project()
    return "license" in proj and any(REPO_ROOT.glob("LICENSE*"))


def check_metadata(wheel: Path, sdist: Path, require_license: bool) -> list[str]:
    proj, version = project()
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta_name = only_name(names, ".dist-info/METADATA")
        meta = email.parser.Parser().parsestr(z.read(meta_name).decode("utf-8"))
        entry_points = z.read(only_name(names, ".dist-info/entry_points.txt")).decode("utf-8")
    with tarfile.open(sdist, "r:gz") as t:
        pkg_info_member = next(m for m in t.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO"))
        handle = t.extractfile(pkg_info_member)
        assert handle is not None
        pkg_info = email.parser.Parser().parsestr(handle.read().decode("utf-8"))
        sdist_names = [m.name for m in t.getmembers() if m.isfile()]
    problems: list[str] = []
    expected = {
        "Name": "emberbench",
        "Version": version,
        "Summary": proj["description"],
        "Requires-Python": proj["requires-python"],
        "Description-Content-Type": "text/markdown",
    }
    for field, value in expected.items():
        for label, m in (("wheel", meta), ("sdist", pkg_info)):
            if m.get(field) != value:
                problems.append(f"{label} metadata {field} is {m.get(field)!r}, expected {value!r}")
    if not str(meta.get_payload() or "").lstrip().startswith("# EMBERBench"):
        problems.append("wheel metadata does not carry README.md as the long description")
    runtime = [r for r in meta.get_all("Requires-Dist") or [] if "extra ==" not in r]
    if runtime:
        problems.append(f"runtime dependencies must be empty, found {runtime}")
    if "[console_scripts]\nemberbench = emberbench.cli:main" not in entry_points.replace("\r\n", "\n"):
        problems.append("console script 'emberbench = emberbench.cli:main' is missing")
    if "emberbench/py.typed" not in names:
        problems.append("emberbench/py.typed is missing from the wheel")
    if wheel.name != f"emberbench-{version}-py3-none-any.whl":
        problems.append(f"wheel name {wheel.name} is not a pure-Python wheel for version {version}")

    licence = meta.get("License-Expression") and meta.get_all("License-File")
    shipped = any("/licenses/" in n for n in names) and any(Path(n).name.startswith("LICENSE") for n in sdist_names)
    if not (licence and shipped and license_configured()):
        if require_license:
            problems.append(f"license: {LICENSE_TODO}")
        else:
            print(f"   WARNING: license: {LICENSE_TODO}. --require-license makes this fatal")
    return problems


def only_name(names: list[str], suffix: str) -> str:
    found = [n for n in names if n.endswith(suffix) and n.count("/") == 1]
    if len(found) != 1:
        raise ReleaseCheckError(f"expected one {suffix}, found {found}")
    return found[0]


def content_problems(wheel: Path, sdist: Path) -> list[str]:
    """The wheel holds only the package code and its metadata; neither archive holds a large file."""
    _, version = project()
    dist_info = f"emberbench-{version}.dist-info/"
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            name = info.filename
            allowed = (name.startswith("emberbench/") and name.endswith(WHEEL_PACKAGE_SUFFIXES)) or (
                name.startswith(dist_info) and (name[len(dist_info) :] in WHEEL_DIST_INFO_FILES or name.startswith(f"{dist_info}licenses/"))
            )
            if not allowed:
                problems.append(f"wheel member {name} is not package code or metadata; only intended package assets may be packaged")
            if info.file_size > MEMBER_MAX_BYTES:
                problems.append(f"wheel member {name} is {info.file_size} bytes (limit {MEMBER_MAX_BYTES})")
    if wheel.stat().st_size > WHEEL_MAX_BYTES:
        problems.append(f"{wheel.name} is {wheel.stat().st_size} bytes (limit {WHEEL_MAX_BYTES})")
    with tarfile.open(sdist, "r:gz") as t:
        files = [m for m in t.getmembers() if m.isfile()]
    for member in files:
        if member.size > MEMBER_MAX_BYTES:
            problems.append(f"sdist member {member.name} is {member.size} bytes (limit {MEMBER_MAX_BYTES}); generated outputs stay out of the package")
    total = sum(m.size for m in files)
    if total > SDIST_MAX_UNPACKED_BYTES:
        problems.append(f"{sdist.name} unpacks to {total} bytes (limit {SDIST_MAX_UNPACKED_BYTES})")
    return problems


# --------------------------------------------------------------------------- clean install


def clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV") and not k.startswith("PIP_")}
    env["PIP_NO_INDEX"] = "1"
    return env


def installed_versions(python: Path, cwd: Path, env: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """``(emberbench version --json without "command", benchmark_versions())`` of the installed wheel."""
    reported = json.loads(run([str(python), "-I", "-m", "emberbench", "version", "--json"], cwd=cwd, env=env).stdout)
    reported.pop("command", None)
    script = "import json; from emberbench import benchmark_versions; print(json.dumps(benchmark_versions()))"
    provenance = json.loads(run([str(python), "-I", "-c", script], cwd=cwd, env=env).stdout)
    return reported, provenance


def verify_registry_with(python_purelib: str, cwd: Path, env: dict[str, str]) -> None:
    """Recompute the registry with this interpreter's test tools and the emberbench at ``python_purelib``."""
    if importlib.util.find_spec("pytest") is None:
        raise ReleaseCheckError("the registry check imports the test helpers and needs pytest: python -m pip install -c tools/dev-constraints.txt pytest")
    tests = str(REPO_ROOT / "tests")
    script = (
        "import sys\n"
        f"sys.path[:0] = [{tests!r}, {python_purelib!r}]\n"
        "import emberbench\n"
        "from pathlib import Path\n"
        f"if not Path(emberbench.__file__).resolve().is_relative_to(Path({python_purelib!r}).resolve()):\n"
        "    sys.exit(f'emberbench imported from {emberbench.__file__}, not from the clean environment')\n"
        "import release_digests\n"
        "sys.exit(release_digests.main(['--check']))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("   " + proc.stdout.strip())
    if proc.returncode != 0:
        raise ReleaseCheckError(
            f"regression digest registry drift with the installed wheel:\n{proc.stderr}\n"
            "Fix the code, or for a deliberate change bump the owner version, run 'python tests/release_digests.py --write' "
            "and add a changelog entry (docs/versioning.md, section 4)."
        )


def smoke_install(wheel: Path, dist: Path, work: Path) -> None:
    _, version = project()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["entries"]
    venv = work / "clean venv"
    run([sys.executable, "-m", "venv", str(venv)])
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    script = bindir / ("emberbench.exe" if os.name == "nt" else "emberbench")
    env = clean_env()
    run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], env=env)
    cwd = work / "outside the checkout"
    cwd.mkdir()

    where = run([str(python), "-I", "-c", "import emberbench; print(emberbench.__file__); print(emberbench.__version__)"], cwd=cwd, env=env)
    location, runtime_version = where.stdout.splitlines()
    if not Path(location).resolve().is_relative_to(venv.resolve()):
        raise ReleaseCheckError(f"emberbench imported from {location}, not from the clean environment")
    if runtime_version != version:
        raise ReleaseCheckError(f"installed emberbench.__version__ is {runtime_version}, expected {version}")

    step("installed `emberbench version --json`: every version against the docs, the changelog and provenance")
    by_script = json.loads(run([str(script), "version", "--json"], cwd=cwd, env=env).stdout)
    reported, provenance = installed_versions(python, cwd, env)
    if {k: v for k, v in by_script.items() if k != "command"} != reported:
        raise ReleaseCheckError(f"the console script reported {by_script}, python -m emberbench reported {reported}")
    problems = version_contract_problems(reported, provenance)
    if problems:
        raise ReleaseCheckError("version contracts are inconsistent:\n  " + "\n  ".join(problems))
    print(f"   {len(reported)} versions consistent; package {reported['package_version']}, benchmark revision {reported['benchmark_revision']}")

    step("public suite and CLI artifact from the installed wheel")
    # Scenario assets are intentionally not in the wheel: pass the public suite by an explicit path.
    suite_copy = work / "public suite"
    shutil.copytree(SUITE_DIR, suite_copy)
    suite = json.loads(run([str(script), "suite", "validate", str(suite_copy), "--json"], cwd=cwd, env=env).stdout)
    if suite.get("suite_hash") != registry["suite.public-v1.suite_hash"]["digest"]:
        raise ReleaseCheckError(f"public suite hash from the installed wheel is {suite.get('suite_hash')}, the registry says {registry['suite.public-v1.suite_hash']['digest']}")
    out = work / "cli artifacts" / "manifest.json"
    out.parent.mkdir()
    run([str(script), "evaluation", "manifest", "--suite", str(suite_copy), "--seed", "0", "--seed", "1", "--seed", "2", "--out", str(out)], cwd=cwd, env=env)
    if hashlib.sha256(out.read_bytes()).hexdigest() != registry["cli.evaluation-manifest.file_sha256"]["digest"]:
        raise ReleaseCheckError("the CLI manifest artifact from the installed wheel does not match the registry")

    step("regression digest registry recomputed with the installed wheel")
    purelib = run([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], cwd=cwd, env=env).stdout.strip()
    verify_registry_with(purelib, cwd, env)

    step("leak guard with the installed wheel: repository, wheel and sdist")
    leak = subprocess.run([str(python), str(REPO_ROOT / "tools" / "check_public_assets.py"), "--dist", str(dist)], cwd=cwd, env=env, capture_output=True, text=True)
    print("   " + leak.stdout.strip())
    if leak.returncode != 0:
        raise ReleaseCheckError(f"leak guard failed:\n{leak.stdout}\n{leak.stderr}")


# --------------------------------------------------------------------------- main


def source_checks(require_clean: bool) -> None:
    step("source checks: clean tree, public assets")
    dirty = dirty_paths()
    if dirty:
        message = f"the working tree is not clean ({len(dirty)} paths, for example {dirty[:5]}); a release is built from a clean, committed tree"
        if require_clean:
            raise ReleaseCheckError(message)
        print(f"   WARNING: {message}. --require-clean makes this fatal")
    problems = public_asset_problems()
    if problems:
        raise ReleaseCheckError("public assets are incomplete:\n  " + "\n  ".join(problems))


def contracts_only() -> int:
    """The version-contract and docs checks against the current interpreter's emberbench, without building."""
    try:
        source_checks(require_clean=False)
        step("version contracts of `python -m emberbench version --json` (current interpreter)")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        reported = json.loads(run([sys.executable, "-m", "emberbench", "version", "--json"], cwd=REPO_ROOT, env=env).stdout)
        reported.pop("command", None)
        script = "import json; from emberbench import benchmark_versions; print(json.dumps(benchmark_versions()))"
        provenance = json.loads(run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env).stdout)
        problems = version_contract_problems(reported, provenance)
        if problems:
            raise ReleaseCheckError("version contracts are inconsistent:\n  " + "\n  ".join(problems))
    except ReleaseCheckError as exc:
        print(f"release check FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"version contracts consistent: {len(reported)} versions")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dist", type=Path, help="empty or new directory for the checked wheel and sdist")
    parser.add_argument("--wheelhouse", type=Path, help="directory holding only the pinned build backend wheel")
    parser.add_argument("--require-license", action="store_true", help="fail when no license is configured (required for a release)")
    parser.add_argument("--require-clean", action="store_true", help="fail when the working tree has uncommitted changes (required for a release)")
    parser.add_argument("--contracts-only", action="store_true", help="check version contracts and docs only, without building")
    args = parser.parse_args(argv)
    if args.contracts_only:
        return contracts_only()
    if args.dist is None:
        parser.error("--dist is required unless --contracts-only is given")
    if args.dist.exists() and any(args.dist.iterdir()):
        print(f"{args.dist}: not empty", file=sys.stderr)
        return 2
    try:
        epoch = source_date_epoch()
        env = build_env(epoch)
        source_checks(args.require_clean)
        with tempfile.TemporaryDirectory(prefix="emb rel ") as tmp:
            work = Path(tmp)
            step(f"two clean builds (SOURCE_DATE_EPOCH={epoch}, no package index)")
            builds = []
            for n in (1, 2):
                out = work / f"dist {n}"
                build(copy_source(work / f"src {n}"), out, env)
                builds.append((only(out, "*.whl"), only(out, "*.tar.gz")))
            (wheel_a, sdist_a), (wheel_b, sdist_b) = builds
            compare("wheel members", wheel_members(wheel_a), wheel_members(wheel_b))
            compare("sdist members", sdist_members(sdist_a), sdist_members(sdist_b))
            print(f"   normalized members identical: {wheel_a.name} ({len(wheel_members(wheel_a))} files), {sdist_a.name} ({len(sdist_members(sdist_a))} files)")
            if not same_bytes(wheel_a, wheel_b):
                raise ReleaseCheckError("the two wheels have identical members but are not byte-identical")
            problems = wheel_time_problems(wheel_a, epoch)
            if problems:
                raise ReleaseCheckError("\n".join(problems))
            print(f"   wheels byte-identical: True (sha256 {hashlib.sha256(wheel_a.read_bytes()).hexdigest()}); sdists byte-identical (reported, not required): {same_bytes(sdist_a, sdist_b)}")

            step("wheel from the source tree equals the wheel from the sdist")
            direct = work / "dist direct"
            build(work / "src 1", direct, env, "--wheel")
            compare("direct and sdist wheels", wheel_members(only(direct, "*.whl")), wheel_members(wheel_a))

            if args.wheelhouse is not None:
                step(f"offline isolated build from the sdist with only {sorted(p.name for p in args.wheelhouse.iterdir())}")
                isolated = work / "dist isolated"
                iso_env = {**env, "PIP_FIND_LINKS": str(args.wheelhouse.resolve())}
                run([sys.executable, "-m", "build", "--wheel", "--outdir", str(isolated), str(sdist_a)], env=iso_env)
                compare("isolated and sdist wheels", wheel_members(only(isolated, "*.whl")), wheel_members(wheel_a))

            step("package metadata, license, wheel contents and archive sizes")
            problems = check_metadata(wheel_a, sdist_a, args.require_license) + content_problems(wheel_a, sdist_a)
            if problems:
                raise ReleaseCheckError("\n".join(problems))

            args.dist.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel_a, args.dist / wheel_a.name)
            shutil.copy2(sdist_a, args.dist / sdist_a.name)

            step("clean virtual environment install (path with spaces), console script and python -m emberbench")
            smoke_install(args.dist / wheel_a.name, args.dist, work)
    except ReleaseCheckError as exc:
        print(f"release check FAILED: {exc}", file=sys.stderr)
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
