# Scenario schema, version `1.0`

A scenario is one JSON object. **All keys are required** except `description`.
Unknown keys are rejected at every level, so a typo cannot silently become a
default. Numbers must be finite. Integers must be JSON integers (`1`, not
`1.0` and not `true`).

```json
{
  "schema_version": "1.0",
  "scenario_id": "one_person_one_exit",
  "description": "optional free text",
  "grid": {"width": 7, "height": 3},
  "obstacles": [[0, 0], [0, 1]],
  "exits": [{"id": "E1", "position": [1, 5]}],
  "people": [{"id": "P1", "position": [1, 1]}],
  "ignitions": [[1, 3]],
  "fire": {"model": "deterministic", "spread_interval": 1, "exits_flammable": true},
  "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "casualty"},
  "termination": {"max_steps": 50},
  "seed": 1
}
```

| Field | Type | Rules |
| --- | --- | --- |
| `schema_version` | string | Must be `"1.0"`. |
| `scenario_id` | id | Matches `^[A-Za-z0-9_-]{1,64}$`. |
| `grid.width`, `grid.height` | int | 1 to 1024. |
| `obstacles` | list of `[row, col]` | In bounds, no duplicates. These are walls: impassable and non-flammable. |
| `exits` | list of `{id, position}` | At least one. Unique IDs and positions. Not on an obstacle. |
| `people` | list of `{id, position}` | At least one. Unique IDs and positions. Not on an obstacle, exit or ignition point. |
| `ignitions` | list of `[row, col]` | Cells burning at step 0. May be empty. In bounds, no duplicates, not on an obstacle. On an exit only if `fire.exits_flammable` is true. |
| `fire.model` | `"deterministic"` \| `"stochastic"` | See the simulation contract. |
| `fire.spread_interval` | int ≥ 1 | Fire spreads on steps where `step % interval == 0`. |
| `fire.exits_flammable` | bool | Whether exits can catch fire and become unusable. |
| `fire.spread_probability` | number in [0, 1] | **Required** for `stochastic`. **Forbidden** for `deterministic`. |
| `movement.conflict_resolution` | `"lowest_id_wins"` | The only V1 rule. It is stated explicitly. |
| `movement.enter_fire` | `"blocked"` \| `"casualty"` | What happens when someone moves into a burning cell. |
| `termination.max_steps` | int, 1 to 100000 | Step limit. |
| `seed` | int, 0 to 2^64 − 1 | Default run seed. `Simulator(scenario, seed=...)` can override it, and the record stores the seed used. |

## Canonical form and identity

After parsing, the following lists are sorted: obstacles and ignitions by
position, and exits and people by ID. `Scenario.to_json()` emits canonical
JSON (sorted keys, no whitespace). `Scenario.content_hash()` is its SHA-256.
Input order therefore never changes a scenario's identity. Run records store
the hash, so a replay against a different scenario is rejected.

## Fixtures

`tests/fixtures/scenarios/*.json` are small **test fixtures**. Each description
starts with `TEST FIXTURE.` and ends with an ASCII layout. They are not
benchmark scoring scenarios, and they are not the hidden evaluation set.

| Fixture | Purpose |
| --- | --- |
| `one_person_one_exit` | Basic movement and evacuation, no fire. |
| `blocked_shortest_route` | The shortest route crosses fire. The safe route is longer. Exits are not flammable. |
| `multiple_exits` | Three people, two exits, fire in the middle. |
| `narrow_corridor` | A one-wide queue, chain moves and exit throughput. |
| `fire_threatening_exit` | Fire burns the near exit at step 1. The far exit stays usable. |
| `stochastic_room` | Seeded stochastic spread with the `blocked` fire-entry policy. |
