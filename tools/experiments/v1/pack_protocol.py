"""The committed public development protocol of the V1 evaluation pack.

**PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD.** The
candidate seeds below are public development seeds chosen for this pack. They
are not an evaluator's scored seed schedule, the suite is the public V1 suite,
and no number this pack produces is a hidden-suite score.

Every value that decides a pack result is pinned here. The schedule is
``build_unique_world_manifest(suite, CANDIDATE_SEEDS, stochastic_worlds=...)``:
one episode per unique environment world (docs/evaluation-suite-design.md,
section 3). Deterministic-fire scenarios are scheduled once, whatever the seed;
the stochastic scenario gets ``stochastic_worlds`` pairwise-distinct fire
worlds. The runner refuses to run when the canonical public suite or the
unique-world manifest does not match the pinned hashes.
"""

from dataclasses import dataclass

from pack_agents import ALWAYS_STAY_ID, COOPERATIVE_ID, FORECAST_ID

PACK_VERSION = "2"
RUN_LABEL = "PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD"

# Provenance: every version that defines what a score means, as `benchmark_versions()` returns it on
# the benchmark this pack was generated on (benchmark revision 3). The runner and the verifier fail
# closed when any key or value differs: a new benchmark revision needs a new pack run, never a mix.
BENCHMARK_REVISION = "3"
BENCHMARK_VERSIONS: dict[str, str] = {
    "accounting": "1.0",
    "adapter": "r1",
    "action_schema": "1.0",
    "admission_rules": "1.0",
    "benchmark_revision": BENCHMARK_REVISION,
    "calibration_panel": "1.0",
    "composition": "1.0",
    "crowd_diagnostics": "1.0",
    "difficulty_rules": "1.0",
    "environment_pack": "1.0",
    "environment_world": "1.0",
    "evaluation_manifest": "1.0",
    "evaluation_result": "1.0",
    "family_rules": "1.0",
    "generation_manifest": "1.1",
    "generator": "1.1.0",
    "generator_config": "1.1",
    "harness": "1.0",
    "metrics": "1.0",
    "observation": "1.0",
    "recipe_catalog": "1.0",
    "scenario_schema": "1.0",
    "scoring": "1.0",
    "selection_rules": "1.0",
    "simulator": "0.2.0",
    "statistics": "1.0",
    "commitment": "1.0",
    "suite_freeze_record": "1.0",
}

# The canonical public suite, relative to the repository root.
PUBLIC_SUITE_DIR = "scenarios/public/v1"
PUBLIC_SUITE_HASH = "5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b"


@dataclass(frozen=True)
class Protocol:
    """One pack protocol: which suite scenarios, which candidate seeds, which pins."""

    name: str
    scenario_ids: tuple[str, ...] | None  # None: every scenario of the canonical public suite
    candidate_seeds: tuple[int, ...]  # the ordered candidate list of build_unique_world_manifest
    stochastic_worlds: int  # distinct fire worlds per stochastic-fire scenario
    resamples: int  # bootstrap resamples of every interval
    suite_hash: str
    manifest_hash: str


# The public development candidate seeds. The nine deterministic scenarios use
# the first candidate (every seed gives the same world); v1-pub-10, the only
# stochastic scenario, keeps the first 10 candidates that give a new fire
# world. The 6 extra candidates are spares for world collisions; none is
# needed under the current suite (seeds 0..15 give 16 distinct worlds).
CANDIDATE_SEEDS: tuple[int, ...] = tuple(range(16))

PUBLIC = Protocol(
    name="public-development",
    scenario_ids=None,
    candidate_seeds=CANDIDATE_SEEDS,
    stochastic_worlds=10,
    resamples=2000,  # the scoring and statistics library default
    suite_hash=PUBLIC_SUITE_HASH,
    manifest_hash="3290930556bee1cf0a8232fd9cf2a118b40d4560d79ef18fc23d79f76139b82c",
)

# A tiny CI smoke subset: two small scenarios (one deterministic, one
# stochastic), 4 stochastic worlds, so 5 unique worlds and a defined interval
# (5 is the minimum bootstrap sample). It exercises the whole pipeline; its
# numbers mean nothing.
SMOKE = Protocol(
    name="smoke",
    scenario_ids=("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation"),
    candidate_seeds=(0, 1, 2, 3, 4, 5),
    stochastic_worlds=4,
    resamples=100,
    suite_hash="7b8e205b3bfe3faf7eee5dce1f32cd7aecec0b886d7d1b771029996c14c6052c",
    manifest_hash="a21a0152619d1e65c6c8825e46e307e1d218c967c2550d4c426216a960d0e7a0",
)

PROTOCOLS = {p.name: p for p in (PUBLIC, SMOKE)}

# Paired comparisons (A - B) with the official comparison API. The required
# pairs come first; then each pack agent against every other evaluated model
# and the ladder of registered baselines. No pair involves a privileged
# reference.
REQUIRED_PAIRS = (
    ("congestion-aware-v1", "fire-aware-shortest-path-v1"),
    ("congestion-aware-v1", "static-shortest-path-v1"),
    (COOPERATIVE_ID, "congestion-aware-v1"),
    (FORECAST_ID, "congestion-aware-v1"),
)
LADDER = (
    ("random-legal-v1", ALWAYS_STAY_ID),
    ("nearest-exit-greedy-v1", "random-legal-v1"),
    ("static-shortest-path-v1", "nearest-exit-greedy-v1"),
    ("fire-aware-shortest-path-v1", "static-shortest-path-v1"),
)


def paired_comparisons(model_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []

    def add(a: str, b: str) -> None:
        if a in model_ids and b in model_ids and a != b and (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))

    for a, b in REQUIRED_PAIRS:
        add(a, b)
    for pack_id in (COOPERATIVE_ID, FORECAST_ID):
        for other in model_ids:
            add(pack_id, other)
    for a, b in LADDER:
        add(a, b)
    return tuple(pairs)
