"""One-command demo: the public pack -> official results and a cost sheet.

1. Builds and runs the public evaluation pack with the reference baselines
   (``run_public_benchmark.py``): ``pack/index.json`` and ``pack/REPORT.md``.
2. Runs ``emberbench run`` on that frozen pack for one baseline and, with
   ``--model``, for one remote model (key in ``--key-env``, spend capped by
   ``--budget-usd``): a run record each under ``runs/``.
3. Writes the cost sheet of those run records (``COST.md``, ``cost.csv``).

Public development data only; it is not a hidden-suite leaderboard.

Usage::

    python tools/experiments/v1/demo.py --out DIR [--protocol smoke] [--model typesafe/jev-1.14 --budget-usd 1]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import cost_table

HERE = Path(__file__).resolve().parent
DEMO_BASELINE = "nearest-exit-greedy-v1"


def _step(title: str, cmd: Sequence[str]) -> None:
    print(f"== {title}", flush=True)
    code = subprocess.call([sys.executable, *cmd])
    if code != 0:
        raise SystemExit(f"{title}: exit {code}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="a new output directory, outside the repository")
    parser.add_argument("--protocol", choices=("public-development", "smoke"), default="public-development")
    parser.add_argument("--model", help="also run this remote model (OpenRouter decisions API by default)")
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY", help="environment variable with the API key (default OPENROUTER_API_KEY)")
    parser.add_argument("--budget-usd", type=float, default=1.0, help="spend cap of the model run (default 1.0; the run can be resumed)")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out}: not empty; give a new directory")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    pack = out / "pack"
    _step("public pack + baselines", [str(HERE / "run_public_benchmark.py"), "--protocol", args.protocol, "--out", str(pack)])
    suite = str(pack / "environment-pack")
    runs = [out / "runs" / DEMO_BASELINE]
    _step(f"emberbench run --baseline {DEMO_BASELINE}", ["-m", "emberbench", "run", "--suite", suite, "--baseline", DEMO_BASELINE, "--out", str(runs[0])])
    if args.model:
        runs.append(out / "runs" / args.model.replace("/", "_"))
        _step(
            f"emberbench run --model {args.model}",
            ["-m", "emberbench", "run", "--suite", suite, "--model", args.model, "--key-env", args.key_env,
             "--budget-usd", str(args.budget_usd), "--out", str(runs[-1])],
        )  # fmt: skip
    rows = cost_table.load_rows(runs)
    cost_table.write(rows, out)
    print(cost_table.markdown(rows), end="")
    print(f"\nresults: {pack / 'REPORT.md'}\ncost sheet: {out / 'COST.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
