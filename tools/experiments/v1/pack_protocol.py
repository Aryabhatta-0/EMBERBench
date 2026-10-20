"""The committed public development protocol of the V1 evaluation pack.

**PUBLIC DEVELOPMENT EVALUATION -- NOT HELD-OUT SCORE.** The seeds below are
public development seeds chosen for this pack. They are not an evaluator's
scored seed schedule, and no number this pack produces is a held-out score.

Every value that decides a pack result is pinned here. The runner refuses to
run when the canonical public suite or the evaluation manifest built from
these seeds does not match the pinned hashes.
"""

from dataclasses import dataclass

from pack_agents import ALWAYS_STAY_ID, COOPERATIVE_ID, FORECAST_ID

PACK_VERSION = "1"
RUN_LABEL = "PUBLIC DEVELOPMENT EVALUATION — NOT HELD-OUT SCORE"

# The canonical public suite, relative to the repository root.
PUBLIC_SUITE_DIR = "scenarios/public/v1"


@dataclass(frozen=True)
class Protocol:
    """One pack protocol: which suite scenarios, which seeds, which pins."""

    name: str
    scenario_ids: tuple[str, ...] | None  # None: every scenario of the canonical public suite
    seeds: tuple[int, ...]
    resamples: int
    suite_hash: str
    manifest_hash: str


# The development seed schedule: every public scenario under every seed, in
# suite order, then in this order. Only v1-pub-10 has stochastic fire, so the
# other nine scenarios give the same episode under every seed (see REPORT.md,
# "Limitations").
DEVELOPMENT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

PUBLIC = Protocol(
    name="public-development",
    scenario_ids=None,
    seeds=DEVELOPMENT_SEEDS,
    resamples=2000,  # the statistics library default
    suite_hash="5274d103928e6abbdf8bf761f4cc608eeefc55a802d876a21e878cce9838065b",
    manifest_hash="0280777fbbcbdd2dfb9c878394d22a846e8e4dfff41c7f33fd2a73de7659b604",
)

# A tiny CI smoke subset: two small scenarios (one deterministic, one
# stochastic), three seeds, the minimum bootstrap size. It exercises the whole
# pipeline; its numbers mean nothing.
SMOKE = Protocol(
    name="smoke",
    scenario_ids=("v1-pub-01-single-exit-sparse", "v1-pub-10-route-invalidation"),
    seeds=(0, 1, 2),
    resamples=100,
    suite_hash="7b8e205b3bfe3faf7eee5dce1f32cd7aecec0b886d7d1b771029996c14c6052c",
    manifest_hash="ae210bd5f8a302b1f350b76709a94f25ef0196ab59b196a8517e71490c03c0cd",
)

PROTOCOLS = {p.name: p for p in (PUBLIC, SMOKE)}

# Paired comparisons (A - B). Each pack agent against every other evaluated
# model, plus the ladder of registered baselines. No pair involves the
# privileged bound.
LADDER = (
    ("random-legal-v1", ALWAYS_STAY_ID),
    ("nearest-exit-greedy-v1", "random-legal-v1"),
    ("static-shortest-path-v1", "nearest-exit-greedy-v1"),
    ("fire-aware-shortest-path-v1", "static-shortest-path-v1"),
    ("congestion-aware-v1", "fire-aware-shortest-path-v1"),
)


def paired_comparisons(model_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for pack_id in (COOPERATIVE_ID, FORECAST_ID):
        for other in model_ids:
            if other != pack_id and (other, pack_id) not in pairs:
                pairs.append((pack_id, other))
    for pair in LADDER:
        if pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)
