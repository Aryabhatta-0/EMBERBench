"""PRIVILEGED, evaluator-side upper-bound analysis. This is not a model.

Everything here reads information that no evaluated model may see: the
:class:`~emberbench.scenario.Scenario`, the evaluator seed and therefore the
whole future fire trajectory. It is a reference analysis for interpreting
results, never a competitor. It does not implement the
:class:`~emberbench.agent.Agent` interface, it is not in the baseline
registry, and it produces no actions and no ``EvaluationResult``.

**The relaxed escape bound.** In V1 the fire never sees people or actions
(``emberbench.fire``), so under a fixed scenario and seed the burning set at
every step is the same for every policy. For each person alone, ignoring
every other person, a time-expanded breadth-first search over
``(cell, step)`` finds the earliest step at which that person can reach a
usable exit while never standing on a cell that is burning after the fire
phase. The search uses the simulator phase order exactly: a move into a cell
that burns at the start of the step is never useful (it is refused under
``blocked`` and fatal under ``casualty``), an exit that is not burning at the
start of the step evacuates at once, and a person on a cell that burns after
fire spread is a casualty.

Other people can only block moves (V1 rules R1-R4). Dropping them is a
relaxation, so for every policy and every person:

- the person is evacuated only if the bound says the person can escape, and
- the evacuation step is at least the bound's earliest step.

Hence ``max_evacuated`` is an upper bound on ``safety.evacuated`` and
``min_person_time_to_safety_total`` is a lower bound on
``time.person_time_to_safety_total`` for every policy. The bound is usually
not attainable when people compete for exits. It says nothing about
casualties versus stranded people. See ``docs/baselines.md``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .baselines import BASELINES_VERSION
from .evaluation import EvaluationManifest, resolve_scenario, verify_manifest
from .fire import FireTerrain, build_fire_model
from .grid import Pos, neighbours
from .rng import SplitMix64, derive_stream_seed, validate_seed
from .scenario import Scenario
from .simulator import FIRE_STREAM
from .suite import Suite

# Marks every object in this module. Results are reference analysis only.
PRIVILEGED = True


def fire_trajectory(scenario: Scenario, seed: int) -> tuple[frozenset[Pos], ...]:
    """The burning set at the start of every step 0..``max_steps`` (privileged: uses the seed).

    Entry ``t`` is the set a policy observes at step ``t``. Entry
    ``max_steps`` is the set after the last fire phase.
    """
    terrain = FireTerrain.from_scenario(scenario)
    model = build_fire_model(scenario.fire)
    rng = SplitMix64(derive_stream_seed(validate_seed(seed), FIRE_STREAM))
    burning = frozenset(scenario.ignitions)
    trajectory = [burning]
    for step_no in range(1, scenario.termination.max_steps + 1):
        burning = burning | model.spread(terrain, burning, step_no, rng)
        trajectory.append(burning)
    return tuple(trajectory)


def _earliest_escape(scenario: Scenario, start: Pos, trajectory: Sequence[frozenset[Pos]]) -> int | None:
    """Earliest evacuation step (1-based, as in run records) for one person alone, or ``None``."""
    width, height = scenario.width, scenario.height
    walls = scenario.walls
    exits = frozenset(scenario.exit_positions)
    if start in trajectory[0]:
        return None
    frontier = {start}
    for t in range(scenario.termination.max_steps):
        now, after = trajectory[t], trajectory[t + 1]
        reached: set[Pos] = set()
        for cell in frontier:
            for nxt in (cell, *neighbours(cell, width, height)):
                if nxt in walls or nxt in now:
                    continue
                if nxt in exits:
                    return t + 1
                if nxt not in after:
                    reached.add(nxt)
        if not reached:
            return None
        frontier = reached
    return None


@dataclass(frozen=True)
class EscapeBound:
    """The privileged relaxed escape bound of one episode (scenario + evaluator seed)."""

    scenario_id: str
    seed: int
    population: int
    max_steps: int
    earliest_evacuation: tuple[tuple[str, int | None], ...]  # (person ID, step or None), sorted by ID
    privileged: bool = True

    @property
    def max_evacuated(self) -> int:
        return sum(1 for _, step in self.earliest_evacuation if step is not None)

    @property
    def min_person_time_to_safety_total(self) -> int:
        return sum(self.max_steps if step is None else step for _, step in self.earliest_evacuation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baselines_version": BASELINES_VERSION,
            "privileged": self.privileged,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "population": self.population,
            "max_steps": self.max_steps,
            "max_evacuated": self.max_evacuated,
            "min_person_time_to_safety_total": self.min_person_time_to_safety_total,
            "earliest_evacuation": [[pid, step] for pid, step in self.earliest_evacuation],
        }


def privileged_escape_bound(scenario: Scenario, seed: int) -> EscapeBound:
    """Compute the relaxed escape bound of ``scenario`` under the evaluator ``seed``. Privileged."""
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    trajectory = fire_trajectory(scenario, seed)
    people = sorted(scenario.people, key=lambda p: p.id)
    return EscapeBound(
        scenario_id=scenario.scenario_id,
        seed=seed,
        population=len(people),
        max_steps=scenario.termination.max_steps,
        earliest_evacuation=tuple((p.id, _earliest_escape(scenario, p.position, trajectory)) for p in people),
    )


def privileged_manifest_bounds(suite: Suite, manifest: EvaluationManifest) -> tuple[EscapeBound, ...]:
    """One :class:`EscapeBound` per manifest episode, in manifest order. Privileged."""
    verify_manifest(manifest, suite)
    return tuple(privileged_escape_bound(resolve_scenario(suite, spec), spec.seed) for spec in manifest.episodes)
