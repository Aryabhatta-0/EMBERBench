"""Run every offline reference agent of the evaluation pack on any frozen environment pack, and score it.

The agents are the registered baselines plus the pack agents
(``tools/experiments/v1/pack_agents.py``): random, always-stay, greedy, the
shortest-path ladder, congestion-aware, forecast and cooperative WHCA. No agent
needs a network. Results are authenticated by replay and scored with the
official ``score_result`` against the pack. A rerun resumes and must give the
same bytes (deterministic baselines reproduce exactly).

Usage (``--out`` outside the repository)::

    python tools/experiments/validity/run_baselines.py --pack ../pubval/environment-pack --out ../pubval/baselines
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from emberbench import agent_policy_factory, load_environment_pack, load_evaluation_result, run_evaluation, score_result  # noqa: E402

import pack_agents  # noqa: E402
import pack_index  # noqa: E402


def run(pack_dir: Path, out: Path, *, resamples: int = 2000, only: set[str] | None = None) -> dict[str, str]:
    pack = load_environment_pack(pack_dir)
    suite, manifest = pack.suite, pack.manifest
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    digests: dict[str, str] = {}
    for spec in pack_agents.pack_agent_specs():
        if only and spec.model_id not in only:
            continue
        slug = pack_index.model_slug(spec.model_id)
        path = out / "results" / f"{slug}.json"
        run_evaluation(suite, manifest, spec.model_id, agent_policy_factory(spec.factory), path, resume=True)
        result = load_evaluation_result(path, suite)
        report = score_result(result, pack, resamples=resamples)
        (out / "scores" / f"{slug}.json").write_text(report.to_json() + "\n", encoding="ascii", newline="\n")
        digests[spec.model_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{spec.model_id:<36} score {float(report.score or 0):6.2f}  result sha256 {digests[spec.model_id][:16]}", flush=True)
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--only", action="append", help="run only this model ID; repeatable")
    args = parser.parse_args(argv)
    run(args.pack, args.out, resamples=args.resamples, only=set(args.only) if args.only else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
