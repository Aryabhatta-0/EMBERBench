"""Build the public-family validation set: ~120 seeded generated scenarios as one environment pack.

**PUBLIC DEVELOPMENT VALIDATION SET — NOT A HIDDEN SUITE.** Every config and
seed below is public. The set exists for robustness analysis (seed, family,
population and hazard variance; baseline ordering), not for a leaderboard.
Models may tune on it.

The set is reproducible: the same code, configs and seeds give a byte-identical
environment pack (the digest is printed and recorded in ``validation.json``).

Usage (``--out`` outside the repository)::

    python tools/experiments/validity/build_public_validation.py --out ../pubval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from emberbench import GeneratorConfig, build_unique_world_manifest, freeze_environment_pack, freeze_generated_suite, generate_suite  # noqa: E402

from compose import compose_suite  # noqa: E402

SET_ID = "emberbench-public-validation-v1"
LABEL = "PUBLIC DEVELOPMENT VALIDATION SET - NOT A HIDDEN SUITE"
HORIZON = 100  # the canonical benchmark horizon of the public suite
CANDIDATE_SEEDS = tuple(range(16))  # public evaluation candidates; one world per scenario
PART_SEEDS = {  # public generator seeds (published; never reuse as private material)
    "open": 4101,
    "multi_exit": 4102,
    "corridor": 4103,
    "maze": 4104,
    "stochastic": 4105,
    "crowd": 4106,
}


def _range(lo: float, hi: float) -> dict[str, float]:
    return {"min": lo, "max": hi}


def _config(
    *, count: int, width: int, height: int, population: tuple[int, int], exits: tuple[int, int], ignitions: tuple[int, int],
    partitions: tuple[int, int] = (0, 1), door_width: tuple[int, int] = (1, 2), scatter: tuple[float, float] = (0.0, 0.05),
    fire: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cells = width * height
    constraints: dict[str, Any] = {
        "route_length": _range(1, HORIZON),
        "obstacle_density": _range(0.0, 0.6),
        "crowd_density": _range(0.0, 0.6),
        "exit_separation": _range(2, width + height),
        "passage_width": _range(1, max(width, height)),
        "route_redundancy": _range(1, 4),
        "ignition_person_distance": _range(2, cells),
        "ignition_exit_distance": _range(3, cells),
        **(extra or {}),
    }
    return {
        "config_version": "1.1",
        "scenario_count": count,
        "grid": {"width": width, "height": height},
        "population": _range(*population),
        "exit_count": _range(*exits),
        "ignition_count": _range(*ignitions),
        "layout": {
            "partition_count": _range(*partitions),
            "doors_per_partition": _range(1, 2),
            "door_width": _range(*door_width),
            "scatter_density": _range(*scatter),
        },
        "fire": fire or {"model": "deterministic", "spread_interval": _range(1, 3), "exits_flammable": True},
        "movement": {"enter_fire": "casualty"},
        "max_steps": HORIZON,
        "max_attempts": 2000,
        "constraints": constraints,
    }


PARTS: dict[str, dict[str, Any]] = {
    "open": _config(count=24, width=20, height=14, population=(4, 40), exits=(1, 1), ignitions=(1, 1)),
    "multi_exit": _config(count=24, width=22, height=16, population=(8, 30), exits=(2, 3), ignitions=(1, 2), partitions=(0, 2)),
    "corridor": _config(
        count=20, width=24, height=12, population=(8, 24), exits=(1, 2), ignitions=(1, 1), partitions=(1, 3), door_width=(1, 1),
        extra={"passage_width": _range(1, 1)},
    ),
    "maze": _config(count=20, width=20, height=20, population=(5, 15), exits=(2, 3), ignitions=(1, 2), partitions=(0, 2), scatter=(0.18, 0.3)),
    "stochastic": _config(
        count=24, width=18, height=14, population=(6, 20), exits=(1, 3), ignitions=(1, 2), partitions=(0, 2),
        fire={"model": "stochastic", "spread_interval": _range(1, 2), "exits_flammable": True, "spread_probability_choices": [0.5, 0.75]},
    ),
    "crowd": _config(count=6, width=30, height=24, population=(100, 110), exits=(2, 3), ignitions=(1, 2), partitions=(0, 1)),
}


def build(out: Path) -> dict[str, Any]:
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out}: exists and is not empty")
    parts = []
    for name, data in PARTS.items():
        config = GeneratorConfig.from_dict(data)
        generated = generate_suite(config, PART_SEEDS[name], batch_id=f"pubval-{name.replace('_', '-')}")
        parts.append(freeze_generated_suite(generated, out / "parts" / name).suite)
        print(f"part     {name}: {len(generated.scenarios)} scenarios", flush=True)
    suite, sources = compose_suite(parts, out / "suite", suite_id=SET_ID, id_prefix="pv")
    manifest = build_unique_world_manifest(suite, CANDIDATE_SEEDS, stochastic_worlds=1)
    pack = freeze_environment_pack(suite, manifest, out / "environment-pack", pack_id=SET_ID)
    names = list(PARTS)
    record = {
        "label": LABEL,
        "set_id": SET_ID,
        "pack_digest": pack.digest,
        "episodes": len(manifest.episodes),
        "part_seeds": PART_SEEDS,
        "configs": PARTS,
        "scenarios": [
            {"scenario_id": e.scenario_id, "part": names[p], "source_id": src, "primary_family": e.primary_family.value,
             "families": [f.value for f in e.families], "population": e.population, "fire_model": e.fire_model.value}
            for e, (p, src) in zip(suite.manifest.entries, sources, strict=True)
        ],
    }  # fmt: skip
    (out / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii", newline="\n")
    print(f"pack     {SET_ID} sha256:{pack.digest}: {len(manifest.episodes)} episodes", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
