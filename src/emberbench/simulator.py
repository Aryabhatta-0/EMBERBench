"""The discrete-time evacuation simulator.

Each call to :meth:`Simulator.step` runs these phases, in this fixed order:

1. Observation: the policy sees ``observe()``, the model-visible view of the state
   before the step. It never contains the seed or RNG state.
2. Actions: the submitted mapping is validated (exactly one action per active person).
3. Movement validation: boundary, wall and (policy-dependent) fire checks.
4. Collision and congestion resolution: occupied cells, lowest-ID conflicts, chains and cycles.
5. Person movement: resolved moves are applied all at once.
6. Evacuation: active people on a non-burning exit leave the grid.
7. Fire propagation: the fire model ignites new cells using the fire RNG stream.
8. Casualties and trapping: active people on burning cells become casualties.
   The trapped set is recomputed after that.
9. Termination check: ``all_resolved`` wins over ``max_steps``.

Phases 2-4 raise or compute only. Nothing changes until phase 5, so a
rejected action set leaves the simulator untouched.
"""

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from . import SIMULATOR_VERSION
from .errors import SimulationError
from .events import Casualty, CasualtyCause, Evacuation, RunRecord, StepRecord
from .fire import FireTerrain, build_fire_model
from .grid import Pos, neighbours
from .movement import parse_actions, resolve_movements
from .observation import OBSERVATION_VERSION, MapView, Observation, RulesView
from .rng import SplitMix64, derive_stream_seed, validate_seed
from .scenario import Scenario
from .state import PersonState, PersonStatus, SimState, TerminationReason

FIRE_STREAM = "fire"

Policy = Callable[[Observation], Mapping[str, Any]]


def compute_trapped(scenario: Scenario, people: tuple[PersonState, ...], burning: frozenset[Pos]) -> frozenset[str]:
    """Active people with no path to a usable (non-burning) exit.

    A path may cross any cell that is not a wall and not burning. Other
    people are ignored, because congestion is temporary and walls and fire
    are not. This is an evaluator diagnostic only. It is not part of the
    observation, and being trapped does not end a person's run.
    """
    walls = scenario.walls
    frontier = deque(sorted(p for p in scenario.exit_positions if p not in burning))
    reachable: set[Pos] = set(frontier)
    while frontier:
        cell = frontier.popleft()
        for nb in neighbours(cell, scenario.width, scenario.height):
            if nb not in reachable and nb not in walls and nb not in burning:
                reachable.add(nb)
                frontier.append(nb)
    return frozenset(p.id for p in people if p.status is PersonStatus.ACTIVE and p.position not in reachable)


class Simulator:
    """Owns one run of one scenario under one seed.

    All randomness comes from simulator-owned streams derived from the seed.
    Two simulators built from equal scenarios and seeds, and stepped with
    equal actions, produce byte-identical records.

    The simulator is evaluator-side. A harness gives a decision model only
    :meth:`observe`, never the simulator, its ``seed``, ``state``,
    ``scenario`` or ``record()``.
    """

    def __init__(self, scenario: Scenario, *, seed: int | None = None) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        self._scenario = scenario
        self._seed = validate_seed(scenario.seed if seed is None else seed)
        self._terrain = FireTerrain.from_scenario(scenario)
        self._fire_model = build_fire_model(scenario.fire)
        self._fire_rng = SplitMix64(derive_stream_seed(self._seed, FIRE_STREAM))
        self._exit_ids = scenario.exit_positions
        self._scenario_hash = scenario.content_hash()
        self._map = MapView.from_scenario(scenario)
        self._rules = RulesView.from_scenario(scenario)

        people = tuple(PersonState(p.id, p.position, PersonStatus.ACTIVE) for p in scenario.people)
        burning = frozenset(scenario.ignitions)
        self._initial = SimState(
            step=0,
            people=people,
            burning=burning,
            trapped=compute_trapped(scenario, people, burning),
            rng_state=self._rng_state(),
        )
        self._state = self._initial
        self._steps: list[StepRecord] = []

    # ------------------------------------------------------------------ accessors

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def state(self) -> SimState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state.terminated

    def observe(self) -> Observation:
        """Phase 1: the model-visible view of the current state (see ``observation.py``)."""
        return Observation.from_state(self._map, self._rules, self._state)

    def record(self) -> RunRecord:
        return RunRecord(
            simulator_version=SIMULATOR_VERSION,
            observation_version=OBSERVATION_VERSION,
            schema_version=self._scenario.schema_version,
            scenario_id=self._scenario.scenario_id,
            scenario_hash=self._scenario_hash,
            seed=self._seed,
            initial_state=self._initial,
            steps=tuple(self._steps),
        )

    def _rng_state(self) -> tuple[tuple[str, int], ...]:
        return ((FIRE_STREAM, self._fire_rng.state),)

    # ------------------------------------------------------------------ stepping

    def step(self, actions: Mapping[str, Any]) -> StepRecord:
        if self.done:
            raise SimulationError(f"simulation already terminated ({self._state.termination_reason})")
        scenario = self._scenario
        before = self._state
        step_no = before.step + 1

        # Phase 2: validate the submitted action set.
        parsed = parse_actions(actions, before.active_ids)

        # Phases 3-4: movement validation and collision resolution.
        positions = {p.id: p.position for p in before.active}
        movements = resolve_movements(
            positions,
            parsed,
            width=scenario.width,
            height=scenario.height,
            walls=scenario.walls,
            burning=before.burning,
            enter_fire=scenario.movement.enter_fire,
        )

        # Phase 5: apply movement simultaneously.
        final = {m.person_id: m.final for m in movements}
        if len(set(final.values())) != len(final):
            raise SimulationError("internal invariant violated: two people resolved to the same cell")
        people = {p.id: p for p in before.people}
        for pid, pos in sorted(final.items()):
            people[pid] = PersonState(pid, pos, PersonStatus.ACTIVE)

        # Phase 6: evacuation through non-burning exits.
        evacuations = []
        for pid in sorted(final):
            pos = final[pid]
            if pos in self._exit_ids and pos not in before.burning:
                people[pid] = PersonState(pid, pos, PersonStatus.EVACUATED)
                evacuations.append(Evacuation(pid, self._exit_ids[pos], pos))

        # Phase 7: fire propagation.
        ignited = self._fire_model.spread(self._terrain, before.burning, step_no, self._fire_rng)
        if ignited & before.burning or any(not self._terrain.is_flammable(c) for c in ignited):
            raise SimulationError("fire model violated its contract (re-ignited or non-flammable cell)")
        burning = before.burning | ignited

        # Phase 8: casualties, then the trapped diagnostic.
        casualties = []
        for pid in sorted(people):
            person = people[pid]
            if person.status is PersonStatus.ACTIVE and person.position in burning:
                cause = CasualtyCause.ENTERED_FIRE if person.position in before.burning else CasualtyCause.FIRE_SPREAD
                people[pid] = PersonState(pid, person.position, PersonStatus.CASUALTY)
                casualties.append(Casualty(pid, person.position, cause))
        people_tuple = tuple(people[pid] for pid in sorted(people))
        trapped = compute_trapped(scenario, people_tuple, burning)

        # Phase 9: termination.
        reason: TerminationReason | None = None
        if not any(p.status is PersonStatus.ACTIVE for p in people_tuple):
            reason = TerminationReason.ALL_RESOLVED
        elif step_no >= scenario.termination.max_steps:
            reason = TerminationReason.MAX_STEPS

        self._state = SimState(
            step=step_no,
            people=people_tuple,
            burning=burning,
            trapped=trapped,
            rng_state=self._rng_state(),
            terminated=reason is not None,
            termination_reason=reason,
        )
        record = StepRecord(
            step=step_no,
            actions=tuple(sorted(parsed.items())),
            movements=movements,
            ignited=tuple(sorted(ignited)),
            evacuations=tuple(evacuations),
            casualties=tuple(casualties),
            state=self._state,
        )
        self._steps.append(record)
        return record


def run_episode(scenario: Scenario, policy: Policy, *, seed: int | None = None) -> RunRecord:
    """Run ``policy`` until termination. The policy maps an Observation to an action mapping."""
    sim = Simulator(scenario, seed=seed)
    while not sim.done:
        sim.step(policy(sim.observe()))
    return sim.record()
