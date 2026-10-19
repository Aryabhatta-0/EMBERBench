"""Manual timing of large crowds. Not run by CI and never a pass/fail gate.

Usage::

    python tools/profile_crowd.py            # 50, 100 and 200 people
    python tools/profile_crowd.py 400 800    # other sizes

For each size it runs the greedy test baseline on the synthetic stress room
from ``tests/conftest.py`` and prints wall-clock seconds for the episode, the
replay check and the crowd diagnostics. Timings depend on the machine, so
compare them only on one machine.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from conftest import crowd_stress_scenario, greedy_exit_policy  # noqa: E402
from emberbench import compute_crowd_diagnostics, run_episode, verify_record  # noqa: E402


def main(sizes: list[int]) -> None:
    print(f"{'people':>7} {'steps':>6} {'evacuated':>9} {'congestion':>10} {'run_s':>7} {'verify_s':>8} {'crowd_s':>7}")
    for population in sizes:
        scenario = crowd_stress_scenario(population, max_steps=max(400, 4 * population))
        t0 = time.perf_counter()
        record = run_episode(scenario, greedy_exit_policy)
        t1 = time.perf_counter()
        verify_record(scenario, record)
        t2 = time.perf_counter()
        diag = compute_crowd_diagnostics(scenario, record)  # includes its own replay check
        t3 = time.perf_counter()
        print(
            f"{population:>7} {diag.episode_steps:>6} {diag.exits.total_evacuations:>9} "
            f"{diag.totals.congestion_blocks:>10} {t1 - t0:>7.2f} {t2 - t1:>8.2f} {t3 - t2:>7.2f}"
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [50, 100, 200])
