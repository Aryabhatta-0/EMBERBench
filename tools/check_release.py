"""Release packaging smoke: reproducible builds, metadata, clean install and leak guard.

Usage::

    python tools/check_release.py --dist dist [--wheelhouse DIR] [--require-license]

Needs ``build``, ``setuptools`` and ``packaging`` (pinned in
tools/dev-constraints.txt). It does not need EMBERBench installed and never
imports it from the checkout. Steps, each fatal on failure:

1. Copy the checkout's files (tracked, plus untracked and not ignored) into
   two fresh source trees.
2. Build each tree with ``python -m build --no-isolation`` (sdist, then the
   wheel from the sdist) under one ``SOURCE_DATE_EPOCH`` and no package index.
3. Compare the two builds: identical member names and member content digests
   for the wheel and the sdist, and byte-identical wheels. Byte identity of
   the sdists is reported, not required (the reproducibility boundary is in
   docs/release-checklist.md).
4. Build a wheel straight from the source tree and require the same members
   and content as the wheel built from the sdist.
5. With ``--wheelhouse``: build the wheel from the sdist again in an isolated
   build environment that may install only from that directory. Put exactly
   the pinned build backend there; an unexpected build dependency fails.
6. Check the wheel and sdist metadata (name, version, summary, Python
   requirement, README, no runtime dependencies, console script, py.typed).
7. Install the wheel with no index into a new virtual environment under a
   path with spaces, import the package from it (not from the checkout), run
   the console script and ``python -m emberbench``, validate the public suite
   by an explicit path, and write the CLI manifest artifact.
8. Run the public-asset leak guard (tools/check_public_assets.py) with the
   installed wheel against the repository and both archives.

The first build is copied to ``--dist``.
"""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "tests" / "fixtures" / "release" / "digest_registry.json"
SUITE_DIR = REPO_ROOT / "scenarios" / "public" / "v1"
# Used only when the checkout has no commit time (for example an exported tree).
FALLBACK_SOURCE_DATE_EPOCH = "1767225600"  # 2026-01-01T00:00:00Z


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
    if not (meta.get_payload() or "").lstrip().startswith("# EMBERBench"):
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
    licence = meta.get("License-Expression") or meta.get("License") or meta.get_all("License-File")
    if not licence:
        message = "no license metadata (License-Expression, License or License-File): a maintainer decision"
        if require_license:
            problems.append(message)
        else:
            print(f"   WARNING: {message}; --require-license makes this fatal")
    return problems


def only_name(names: list[str], suffix: str) -> str:
    found = [n for n in names if n.endswith(suffix) and n.count("/") == 1]
    if len(found) != 1:
        raise ReleaseCheckError(f"expected one {suffix}, found {found}")
    return found[0]


def smoke_install(wheel: Path, dist: Path, work: Path) -> None:
    _, version = project()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["entries"]
    venv = work / "clean venv"
    run([sys.executable, "-m", "venv", str(venv)])
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    script = bindir / ("emberbench.exe" if os.name == "nt" else "emberbench")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV") and not k.startswith("PIP_")}
    env["PIP_NO_INDEX"] = "1"
    run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], env=env)
    cwd = work / "outside the checkout"
    cwd.mkdir()

    where = run([str(python), "-I", "-c", "import emberbench; print(emberbench.__file__); print(emberbench.__version__)"], cwd=cwd, env=env)
    location, runtime_version = where.stdout.splitlines()
    if not Path(location).resolve().is_relative_to(venv.resolve()):
        raise ReleaseCheckError(f"emberbench imported from {location}, not from the clean environment")
    if runtime_version != version:
        raise ReleaseCheckError(f"installed emberbench.__version__ is {runtime_version}, expected {version}")

    by_script = json.loads(run([str(script), "version", "--json"], cwd=cwd, env=env).stdout)
    by_module = json.loads(run([str(python), "-I", "-m", "emberbench", "version", "--json"], cwd=cwd, env=env).stdout)
    if by_script != by_module or by_script.get("package_version") != version or by_script.get("benchmark_revision") != "1":
        raise ReleaseCheckError(f"emberbench version --json reported {by_script} (module: {by_module})")

    # Scenario assets are intentionally not in the wheel: pass the public suite by an explicit path.
    suite_copy = work / "public suite"
    shutil.copytree(SUITE_DIR, suite_copy)
    suite = json.loads(run([str(script), "suite", "validate", str(suite_copy), "--json"], cwd=cwd, env=env).stdout)
    if suite.get("suite_hash") != registry["suite.public-v1.suite_hash"]["digest"]:
        raise ReleaseCheckError(f"public suite hash from the installed wheel is {suite.get('suite_hash')}")
    out = work / "cli artifacts" / "manifest.json"
    out.parent.mkdir()
    run([str(script), "evaluation", "manifest", "--suite", str(suite_copy), "--seed", "0", "--seed", "1", "--seed", "2", "--out", str(out)], cwd=cwd, env=env)
    if hashlib.sha256(out.read_bytes()).hexdigest() != registry["cli.evaluation-manifest.file_sha256"]["digest"]:
        raise ReleaseCheckError("the CLI manifest artifact from the installed wheel does not match the registry")

    leak = subprocess.run([str(python), str(REPO_ROOT / "tools" / "check_public_assets.py"), "--dist", str(dist)], cwd=cwd, env=env, capture_output=True, text=True)
    print("   " + leak.stdout.strip())
    if leak.returncode != 0:
        raise ReleaseCheckError(f"leak guard failed:\n{leak.stderr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dist", type=Path, required=True, help="empty or new directory for the checked wheel and sdist")
    parser.add_argument("--wheelhouse", type=Path, help="directory holding only the pinned build backend wheel")
    parser.add_argument("--require-license", action="store_true", help="fail when the package has no license metadata")
    args = parser.parse_args(argv)
    if args.dist.exists() and any(args.dist.iterdir()):
        print(f"{args.dist}: not empty", file=sys.stderr)
        return 2
    epoch = source_date_epoch()
    env = build_env(epoch)
    try:
        with tempfile.TemporaryDirectory(prefix="emberbench release ") as tmp:
            work = Path(tmp)
            step(f"two clean builds (SOURCE_DATE_EPOCH={epoch}, no package index)")
            builds = []
            for n in (1, 2):
                out = work / f"dist {n}"
                build(copy_source(work / f"source {n}"), out, env)
                builds.append((only(out, "*.whl"), only(out, "*.tar.gz")))
            (wheel_a, sdist_a), (wheel_b, sdist_b) = builds
            compare("wheel members", wheel_members(wheel_a), wheel_members(wheel_b))
            compare("sdist members", sdist_members(sdist_a), sdist_members(sdist_b))
            print(f"   normalized members identical: {wheel_a.name} ({len(wheel_members(wheel_a))} files), {sdist_a.name} ({len(sdist_members(sdist_a))} files)")
            if not same_bytes(wheel_a, wheel_b):
                raise ReleaseCheckError("the two wheels have identical members but are not byte-identical")
            print(f"   wheels byte-identical: True; sdists byte-identical (reported, not required): {same_bytes(sdist_a, sdist_b)}")

            step("wheel from the source tree equals the wheel from the sdist")
            direct = work / "dist direct"
            build(work / "source 1", direct, env, "--wheel")
            compare("direct and sdist wheels", wheel_members(only(direct, "*.whl")), wheel_members(wheel_a))

            if args.wheelhouse is not None:
                step(f"isolated build from the sdist with only {sorted(p.name for p in args.wheelhouse.iterdir())}")
                isolated = work / "dist isolated"
                iso_env = {**env, "PIP_FIND_LINKS": str(args.wheelhouse.resolve())}
                run([sys.executable, "-m", "build", "--wheel", "--outdir", str(isolated), str(sdist_a)], env=iso_env)
                compare("isolated and sdist wheels", wheel_members(only(isolated, "*.whl")), wheel_members(wheel_a))

            step("package metadata")
            problems = check_metadata(wheel_a, sdist_a, args.require_license)
            if problems:
                raise ReleaseCheckError("\n".join(problems))

            args.dist.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel_a, args.dist / wheel_a.name)
            shutil.copy2(sdist_a, args.dist / sdist_a.name)

            step("clean virtual environment install, console script, explicit-path suite and leak guard")
            smoke_install(args.dist / wheel_a.name, args.dist, work)
    except ReleaseCheckError as exc:
        print(f"release check FAILED: {exc}", file=sys.stderr)
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
