"""Unique environment worlds (environment world version 1.0): the scientific identity of an evaluation world.

An :class:`~emberbench.EnvironmentPack` binds exact scheduled episodes, seed
included: :func:`~emberbench.fire_world_digest` hashes the seed, so two
scheduled episodes always have different pack digests. That is the right
identity for a frozen schedule. It is the wrong unit for counting evidence.

In V1 the evaluator seed reaches a run only through the simulator's fire
stream, the fire ignores people and actions, and the observation never
contains the seed or any RNG state. A run is therefore fully determined by the
simulator and observation contracts, the scenario bytes, the exogenous fire
trajectory and the model. Two seeds that give the same burning-set trajectory
for the same scenario give the same environment to every model, and every
official score input is a function of that trajectory (the EMBER Score is
trajectory-invariant). They are **one environment world**.

:func:`environment_world_digest` is that identity. It hashes:

- ``ENVIRONMENT_WORLD_VERSION``, ``SIMULATOR_VERSION`` and ``OBSERVATION_VERSION``;
- the scenario content hash;
- the burning set at every snapshot ``0..max_steps``.

It never hashes the seed. A deterministic-fire scenario has exactly one world,
whatever seeds are scheduled.

:func:`build_unique_world_manifest` schedules one episode per unique world. It
is a separate, explicit helper; :func:`~emberbench.build_evaluation_manifest`
keeps its full scenario x seed product. The definition is in
``docs/evaluation-suite-design.md`` section 3.
"""

from collections.abc import Sequence
from dataclasses import replace

from . import SIMULATOR_VERSION
from .canonical import sha256_hex
from .errors import EvaluationError
from .evaluation import EpisodeSpec, EvaluationManifest, build_evaluation_manifest, verify_manifest
from .observation import OBSERVATION_VERSION
from .oracle import fire_trajectory
from .scenario import FireModelName, Scenario
from .suite import Suite

ENVIRONMENT_WORLD_VERSION = "1.0"


def environment_world_digest(scenario: Scenario, seed: int) -> str:
    """The model- and seed-independent identity of the environment world of ``(scenario, seed)``.

    Equal for two seeds exactly when they give the same burning-set trajectory
    for the same scenario bytes, under the same simulator and observation
    versions. Unlike :func:`~emberbench.fire_world_digest`, the seed is not an
    input of the hash.
    """
    trajectory = fire_trajectory(scenario, seed)
    return sha256_hex(
        {
            "environment_world_version": ENVIRONMENT_WORLD_VERSION,
            "simulator_version": SIMULATOR_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "scenario_hash": scenario.content_hash(),
            "burning": [[[p.row, p.col] for p in sorted(burning)] for burning in trajectory],
        }
    )


def build_unique_world_manifest(suite: Suite, candidate_seeds: Sequence[int], *, stochastic_worlds: int) -> EvaluationManifest:
    """An evaluation manifest with exactly one episode per unique environment world.

    ``candidate_seeds`` is the evaluator's private, ordered candidate list
    (validated like the seeds of :func:`~emberbench.build_evaluation_manifest`:
    valid 64-bit seeds, no duplicates). For every suite scenario, in suite
    manifest order:

    - a deterministic-fire scenario gets one episode, with the first candidate
      seed (every seed gives the same world);
    - a stochastic-fire scenario walks the candidates in order and keeps a seed
      only when its world is new for that scenario, until it has
      ``stochastic_worlds`` pairwise-distinct worlds. Fewer distinct worlds
      among the candidates raise :class:`~emberbench.EvaluationError`.

    The result depends only on the suite and the candidate list (no model,
    clock or environment input), so the same inputs give the identical
    manifest. It passes :func:`~emberbench.verify_manifest`.
    """
    if not isinstance(stochastic_worlds, int) or isinstance(stochastic_worlds, bool) or stochastic_worlds < 1:
        raise EvaluationError("stochastic_worlds must be a positive integer")
    template = build_evaluation_manifest(suite, candidate_seeds)  # validates the seeds and the suite
    seeds = [spec.seed for spec in template.episodes[: len(candidate_seeds)]]
    episodes: list[EpisodeSpec] = []
    for entry in suite.manifest.entries:
        scenario = suite.get(entry.scenario_id)
        if scenario.fire.model is FireModelName.DETERMINISTIC:
            chosen = seeds[:1]
        else:
            chosen, seen = [], set()
            for seed in seeds:
                world = environment_world_digest(scenario, seed)
                if world not in seen:
                    seen.add(world)
                    chosen.append(seed)
                    if len(chosen) == stochastic_worlds:
                        break
            if len(chosen) < stochastic_worlds:
                raise EvaluationError(
                    f"{entry.scenario_id}: only {len(chosen)} distinct fire worlds among {len(seeds)} candidate seeds, {stochastic_worlds} required"
                )
        for seed in chosen:
            episodes.append(EpisodeSpec(len(episodes), entry.scenario_id, entry.content_hash, seed))
    manifest = replace(template, episodes=tuple(episodes))
    verify_manifest(manifest, suite)
    return manifest
