# Simulation contract (simulator version 0.1.0)

This document is normative. If the code and this document disagree, it is a bug.
The V1 fairness contract, the model-visible observation and the rationale for
each frozen rule are in [v1-fairness-spec.md](v1-fairness-spec.md).
A change to any rule that changes output for the same inputs must bump
`SIMULATOR_VERSION` in `src/emberbench/__init__.py`.

## Coordinates and neighbourhood

- A position is `[row, col]`, and `[0, 0]` is the top-left cell.
- `north` is row − 1, `south` is row + 1, `west` is col − 1, `east` is col + 1.
- The neighbourhood is von Neumann (4-connected). Diagonal moves and diagonal fire spread do not exist.
- Canonical order is row-major (sort by row, then by column). Person and exit IDs sort as plain strings by code point, so `"P10" < "P2"`.

## State

A `SimState` snapshot contains: `step`, every person (`id`, `position`, `status`),
the burning cells, the trapped set, the RNG stream positions, `terminated` and
`termination_reason`. Snapshots are frozen dataclasses. Two equal snapshots
serialise to identical canonical JSON. A `SimState` is evaluator-private. A
decision model sees only the `Observation` (see
[v1-fairness-spec.md](v1-fairness-spec.md#4-observation-contract-version-10)).

The person status is one of `active`, `evacuated` or `casualty`. A resolved
person stays in the record with its last position, but it does not occupy a
cell any more and must not receive actions.

At the start of every step, no active person stands on an exit or on a burning
cell.

## Phase order

Each `Simulator.step(actions)` runs these phases in this order:

| # | Phase | Rule |
|---|-------|------|
| 1 | Observation | `observe()` returns the pre-step model-visible `Observation`: step, static map, people, burning cells and allowed actions. It never includes the seed, RNG state, hidden scenario metadata or the trapped set. |
| 2 | Actions | The mapping must have exactly one entry per **active** person. Values are `stay`/`north`/`south`/`east`/`west` (or `Action`). Anything else raises `InvalidActionError`, and the state does not change. |
| 3 | Movement validation | `stay` gives `stayed`. A move off the grid gives `blocked_boundary`. A move into an obstacle gives `blocked_wall`. A move into a burning cell gives `blocked_fire` if `movement.enter_fire = "blocked"`. Otherwise the move continues. |
| 4 | Collision resolution | See [Collisions](#collisions). |
| 5 | Movement | All moves resolved as `moved` happen at once. |
| 6 | Evacuation | Every active person on an exit that was **not burning at the start of the step** is `evacuated` and recorded with the exit ID. |
| 7 | Fire propagation | The fire model returns newly ignited cells for step number `t` (1-based). |
| 8 | Casualties and trapping | Every still-active person on a burning cell becomes a `casualty`. The cause is `entered_fire` if the cell was burning before phase 7, else `fire_spread`. Then the trapped set is recomputed. |
| 9 | Termination | If no person is active, the reason is `all_resolved`. Else, if `t >= termination.max_steps`, the reason is `max_steps`. `all_resolved` wins when both hold. A terminated simulator raises `SimulationError` on `step()`. |

Consequences of the order:

- Evacuation (6) comes before fire spread (7). A person who reaches an exit in the
  same step that the fire reaches it gets out.
- Casualties (8) come after fire spread (7). A person who does not move away from a
  cell that ignites this step dies.

## Collisions

Collision resolution uses these rules in this order. Each rule only looks at movers
that are still unresolved.

1. **Occupied by a non-mover.** If the target holds a person who is not trying to
   move (`stay`, or blocked in phase 3), the move gives `blocked_occupied`.
2. **Same target.** If two or more movers target one cell, the **lowest ID wins**.
   Every other mover gives `blocked_conflict`. This is decided on intent only. A loser
   is not re-awarded the cell if the winner fails later.
3. **Chains.** A winner moves if its target is empty, or if the target's occupant
   moves out successfully in the same step. People can follow each other in one
   step (`a b . E`: both move east). If the chain ends at a person who does not
   move, every mover in the chain gives `blocked_occupied`.
4. **Cycles.** Swaps (`A <-> B`) and longer rotations are never allowed. Every member
   gives `blocked_cycle`.

A blocked person stays in its cell. The outcome does not depend on the order in
which actions are submitted.

**Exits** are normal cells for movement. Each exit cell therefore lets at most one
person through per step: two people who target the same exit are a normal conflict.
Because people leave in phase 6, an exit is always empty at the start of the next step.

**Entering fire.** `movement.enter_fire` is either `blocked` or `casualty`. With
`blocked`, the move into fire is refused (`blocked_fire`). With `casualty`, the
person moves and dies in phase 8 (`entered_fire`). This includes a burning exit:
a burning exit never evacuates anyone.

## Fire

The fire model interface is in `src/emberbench/fire.py`:

```python
class FireModel(Protocol):
    name: str  # read-only
    def spread(self, terrain: FireTerrain, burning: frozenset[Pos], step: int, rng: SplitMix64) -> frozenset[Pos]: ...
```

Contract:

- The model returns only cells that are not burning and that are flammable. The
  simulator checks this and raises `SimulationError` on a violation.
- **Walls never burn, and fire never passes through them.** Spread is only into
  flammable 4-neighbours of burning cells.
- **Exits** burn only if `fire.exits_flammable = true`. A burning exit stays
  burning and cannot evacuate anyone. With `false`, exits never ignite. Validation
  then rejects an ignition point on an exit.
- Cells burn forever (no burn-out in V1). People do not block fire.
- Spread happens only on steps where `step % fire.spread_interval == 0`.
- Randomness comes only from the passed `rng`, which is the simulator's `fire`
  stream. The model draws in canonical order.

V1 models:

- `deterministic`: on a spread step, every flammable neighbour of a burning cell ignites.
- `stochastic`: on a spread step, each (candidate cell, burning neighbour) edge
  takes one Bernoulli(`spread_probability`) draw. The order is candidates
  row-major, then sources row-major. A candidate ignites if any of its edges
  succeeds. All draws are always taken, so the stream position depends only
  on the geometry. With `p = 1` the result equals `deterministic`. With
  `p = 0` there is no spread.

## Randomness

- The only generator is `emberbench.rng.SplitMix64`. It is fully specified in
  code, and a test checks it against the reference output.
- Stream seeds are `sha256("emberbench/v1/{seed}/{label}")[:8]` (big-endian).
  V1 has one stream, `fire`. A new subsystem gets a new label, so it does not
  shift the existing streams.
- `bernoulli(p)` always consumes exactly one draw and compares a 53-bit dyadic
  float with `p`. It uses no other floating-point arithmetic.
- The stream positions are part of `SimState` (for audit and resume). They are
  not part of `Observation`.
- The fire stream is exogenous: the fire model never sees people or actions.
  For the same scenario and seed, every action sequence gives the same fire
  trajectory on the steps that the runs share.

## Trapping

`trapped` is the set of active people with no path to a non-burning exit. A
path may cross any cell that is not a wall and not burning. Other people are
ignored. This is an evaluator diagnostic for later metrics. It is not in the
`Observation`. It does not change the person's status or end the run.

## Replay

`RunRecord` stores the header (`simulator_version`, `schema_version`,
`scenario_id`, `scenario_hash`, `seed`), the initial state, and for each step
the actions, resolved movements, ignitions, evacuations, casualties and the
post-step state. `verify_record(scenario, record)` checks the version and the
scenario hash. It then replays the recorded actions and requires a byte-identical
canonical encoding. It reports the first step that diverges.
