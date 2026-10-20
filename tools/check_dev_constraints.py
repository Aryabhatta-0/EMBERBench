"""Check that the development tool pins in tools/dev-constraints.txt are complete and consistent.

Usage::

    python tools/check_dev_constraints.py              # the files only
    python tools/check_dev_constraints.py --installed  # also the current environment (CI)

The files check needs no network and no installed tools except ``packaging``
(a dependency of ``pytest`` and ``build``). It fails if

- a requirement of the ``dev`` extra or of ``[build-system] requires`` has no
  upper bound, has no exact pin, or its pin is outside its range;
- ``pip`` is not pinned;
- a line is not ``name==version`` or a package is pinned twice.

With ``--installed`` it also walks the installed dependency closure of the
declared tools (the ``dev`` extra, ``[build-system] requires`` and ``pip``,
following ``Requires-Dist`` with its environment markers) and fails if any
distribution in it is not pinned or its version is not the pinned one. That
catches a new transitive dependency on any CI platform. Packages that the
Python installation brings and no declared tool needs (for example a hosted
runner's ``pipx``) are outside the closure and are not checked. See
docs/release-checklist.md, section "Development tool constraints".
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.+!-]*)$")
ALWAYS_PINNED = ("pip",)
NEVER_PINNED = ("emberbench",)


def read_pins(path: Path) -> tuple[dict[str, Version], list[str]]:
    pins: dict[str, Version] = {}
    errors: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.match(line)
        if match is None:
            errors.append(f"{path.name}:{number}: {raw!r} is not 'name==version'")
            continue
        name = canonicalize_name(match.group(1))
        if name in pins:
            errors.append(f"{path.name}:{number}: {name} is pinned twice")
        pins[name] = Version(match.group(2))
    return pins, errors


def declared_requirements(pyproject: Path) -> list[tuple[str, Requirement]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    declared = [("[build-system] requires", Requirement(r)) for r in data["build-system"]["requires"]]
    declared += [("[project.optional-dependencies] dev", Requirement(r)) for r in data["project"]["optional-dependencies"]["dev"]]
    return declared


def check_files(constraints: Path, pyproject: Path) -> tuple[dict[str, Version], list[str]]:
    pins, errors = read_pins(constraints)
    for where, req in declared_requirements(pyproject):
        name = canonicalize_name(req.name)
        if not any(spec.operator in ("<", "<=", "==", "~=") for spec in req.specifier):
            errors.append(f"{where}: {req} has no upper bound")
        if name not in pins:
            errors.append(f"{where}: {req.name} is not pinned in {constraints.name}")
        elif not req.specifier.contains(pins[name], prereleases=True):
            errors.append(f"{where}: pin {name}=={pins[name]} is outside {req}")
    for always in ALWAYS_PINNED:
        if always not in pins:
            errors.append(f"{constraints.name}: {always} is not pinned")
    for never in NEVER_PINNED:
        if never in pins:
            errors.append(f"{constraints.name}: {never} must not be pinned")
    return pins, errors


def _installed() -> dict[str, importlib.metadata.Distribution]:
    out: dict[str, importlib.metadata.Distribution] = {}
    for dist in importlib.metadata.distributions():
        out.setdefault(canonicalize_name(dist.metadata["Name"]), dist)
    return out


def tool_closure(pyproject: Path, installed: dict[str, importlib.metadata.Distribution]) -> list[str]:
    """Installed distributions that the declared tools need, transitively, under this environment's markers."""
    pending = [canonicalize_name(req.name) for where, req in declared_requirements(pyproject) if req.marker is None or req.marker.evaluate({"extra": ""})]
    pending += [canonicalize_name(name) for name in ALWAYS_PINNED]
    closure: set[str] = set()
    while pending:
        name = pending.pop()
        if name in closure or name not in installed:
            continue  # a declared tool that this job does not install is not checked
        closure.add(name)
        for raw in installed[name].requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(canonicalize_name(req.name))
    return sorted(closure - set(NEVER_PINNED))


def check_installed(pins: dict[str, Version], pyproject: Path) -> list[str]:
    errors: list[str] = []
    installed = _installed()
    for name in tool_closure(pyproject, installed):
        version = installed[name].version
        if name not in pins:
            errors.append(f"installed {name}=={version} is not pinned")
        elif Version(version) != pins[name]:
            errors.append(f"installed {name}=={version}, pinned {pins[name]}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--constraints", type=Path, default=REPO_ROOT / "tools" / "dev-constraints.txt")
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml")
    parser.add_argument("--installed", action="store_true", help="also check the installed tools and their dependencies")
    args = parser.parse_args(argv)
    pins, errors = check_files(args.constraints, args.pyproject)
    if args.installed:
        errors += check_installed(pins, args.pyproject)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"checked {len(pins)} pin(s){' and the installed environment' if args.installed else ''}: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
