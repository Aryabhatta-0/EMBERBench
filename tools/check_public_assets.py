"""Fail if the checkout or a built distribution holds anything but public assets.

Usage::

    python tools/check_public_assets.py                 # check the tracked repository files
    python tools/check_public_assets.py --untracked     # also check untracked, not ignored files
    python tools/check_public_assets.py --dist dist     # also check every wheel and sdist in dist/

Exit status 1 lists every violation. See docs/integrity.md, section "Leak guard".
"""

import argparse
import sys
from pathlib import Path

from emberbench.leakguard import LeakViolation, check_distribution, check_repository

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dist", type=Path, help="directory with built .whl and .tar.gz files to check")
    parser.add_argument("--untracked", action="store_true", help="also check untracked, not ignored files")
    args = parser.parse_args(argv)

    violations: list[LeakViolation] = check_repository(REPO_ROOT, include_untracked=args.untracked)
    checked = ["repository"]
    if args.dist is not None:
        archives = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
        if not any(a.suffix == ".whl" for a in archives) or not any(a.name.endswith(".tar.gz") for a in archives):
            print(f"{args.dist}: expected at least one wheel and one sdist", file=sys.stderr)
            return 1
        for archive in archives:
            violations += check_distribution(archive)
            checked.append(archive.name)
    for v in violations:
        print(v, file=sys.stderr)
    print(f"checked {', '.join(checked)}: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
