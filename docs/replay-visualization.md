# Replay audit and visualization (replay audit version 1.0, animation format version 1.0)

This document defines the read-only replay audit and the two views built
from it: canonical text frames and a portable JSON animation export.

- `emberbench.build_replay_audit(scenario, record, *, crowd=False) -> ReplayAudit`
  is in `src/emberbench/audit.py`.
- `render_frame`, `build_animation`, `animation_json`, `export_animation`
  and `verify_animation` are in `src/emberbench/visualization.py`.
- The tests are in `tests/test_replay_audit.py`.

**Visualization never affects benchmark results.** The audit and every view
only read an authenticated `RunRecord`. They do not run the simulator rules
again, they do not change a record, and no metric, statistic, baseline,
integrity check or CLI command reads them.

## 1. Read-only guarantee

- **Authenticated input only.** `build_replay_audit` first calls
  `validate_run_record`, the same three stages that the metrics, the crowd
  diagnostics and `emberbench replay verify` use: version and scenario
  binding, structural invariants, and a full simulator replay that must
  reproduce the record byte for byte. Any failure raises
  `ReplayAuditError` (a subclass of `MetricsInputError`). There is no
  function that audits or renders an unchecked record.
- **Complete episodes only.** As for the metrics, the record must reach
  termination. A replayable prefix is rejected. The shortest valid episode
  has one step.
- **No inference.** Every event in the audit is an event in the record.
  Every count is a count of recorded values. There is no fire prediction,
  no path planning and no re-simulation of a trajectory.
- **No score.** The congestion overlay copies raw counts. There is no crowd
  score and no ratio.
- **No environment.** The artifacts hold no timestamps, no file paths, no
  host or user names and no interpreter details.

`verify_replay_audit(scenario, record, audit)` and
`verify_animation(scenario, record, animation)` rebuild the artifact from
the authenticated record and require byte-identical canonical JSON. They
fail closed on a wrong scenario, a wrong hash, an incompatible version, a
structurally invalid record, a replay mismatch, or an edited artifact.

## 2. Audit model

`ReplayAudit.to_dict()` has exactly these keys:

| Key | Meaning |
| --- | --- |
| `replay_audit_version` | `"1.0"` (`REPLAY_AUDIT_VERSION`) |
| `simulator_version`, `observation_version`, `schema_version` | copied from the record header |
| `scenario_id`, `scenario_hash`, `seed` | copied from the record header |
| `record_digest` | `RunRecord.digest()` of the audited record |
| `map` | `width`, `height`, `walls` (row-major), `exits` (`{id, position}`, exit-ID order) |
| `episode_steps` | number of recorded steps `N` |
| `termination_reason` | `all_resolved` or `max_steps` |
| `timeline` | `N + 1` frames, see [section 3](#3-timeline-and-frame-semantics) |
| `people` | one path per person, person-ID order, see [section 4](#4-person-paths) |
| `fire` | `initial_burning` and `N` fire deltas, see [section 5](#5-fire-timeline) |
| `crowd` | the congestion overlay, or `null` unless `crowd=True`, see [section 6](#6-congestion-overlay) |

Cells are `[row, col]` lists. `ReplayAudit.to_json()` is canonical JSON
(sorted keys, no whitespace, ASCII). `ReplayAudit.digest()` is its SHA-256.

## 3. Timeline and frame semantics

Frame `k` describes the state **after** step `k`. Frame 0 is the initial
state. `audit.frame(k)` returns frame `k` for `0 <= k <= N`; other values
raise `ValueError` (or `TypeError` for a non-integer). Lookup is constant
time: `audit.timeline[k].step == k`.

| Field | Frame 0 | Frame `k >= 1` |
| --- | --- | --- |
| `movements` | `[]` | the recorded `MoveResolution` of every person active at decision time: `person_id`, `action` (the attempted action), `origin`, `target`, `final`, `outcome` |
| `newly_ignited` | `[]` | `StepRecord.ignited` |
| `evacuations` | `[]` | `StepRecord.evacuations` (`person_id`, `exit_id`, `position`) |
| `casualties` | `[]` | `StepRecord.casualties` (`person_id`, `position`, `cause`) |
| `people` | every person after the step: `id`, `position`, `status` | same |
| `burning_count` | number of burning cells | same |
| `active_count` | active people | same |
| `evacuated_total`, `casualty_total` | cumulative, up to and including the step | same |
| `trapped` | the recorded trapped diagnostic, sorted | same |
| `terminated`, `termination_reason` | `false`, `null` | set only on frame `N` |

A resolved person keeps the last cell occupied (the exit, or the burning
cell), as in `SimState`.

## 4. Person paths

`audit.person(person_id)` returns a `PersonPath`:

| Field | Meaning |
| --- | --- |
| `initial_position` | position at step 0 |
| `steps` | one entry for each step in which the person was active at decision time: `step`, `action`, `outcome`, `origin`, `target`, `position` (the cell after movement). Steps run from 1 to `len(steps)` with no gaps. |
| `final_status` | `evacuated`, `casualty`, or `active` |
| `final_position` | the person's position in the final state |
| `resolution_step` | the evacuation or casualty step; `null` if unresolved |
| `exit_id` | the exit used; set only for an evacuation |
| `casualty_cause` | `entered_fire` or `fire_spread`; set only for a casualty |
| `trapped_at_end` | `true` only for an unresolved person in the final trapped set |

`final_status == "active"` means **unresolved**: the episode ended at
`max_steps` with the person still on the grid. `PersonPath.unresolved` is
the same test. For a resolved person the last entry of `steps` is the
resolution step.

`PersonPath.position_at(k)` returns the position after step `k`, read from
the path. The tests check that it equals `SimState.person(id).position` for
every person and every step, and that each path entry equals the person's
recorded `MoveResolution`. The paths come from one pass over the recorded
movements; nothing is simulated again.

## 5. Fire timeline

`fire.initial_burning` is the burning set at step 0 (the scenario
ignitions). `fire.steps[k - 1]` (or `audit.fire_delta(k)`) is the delta for
step `k`:

| Field | Meaning |
| --- | --- |
| `burning_before` | cells burning before step `k` (the state after step `k - 1`) |
| `newly_ignited` | cells ignited in step `k` (`StepRecord.ignited`) |
| `burning_after` | cells burning after step `k` (`StepRecord.state.burning`) |

Every delta satisfies `burning_after == burning_before | newly_ignited`, and
`burning_before` and `newly_ignited` are disjoint. `audit.burning_at(k)`
returns the burning set after step `k`. Stochastic fire is audited the same
way: the deltas are the recorded ignitions of that seed. There is no delta
after step `N` and no prediction. All cell lists are in row-major order.

## 6. Congestion overlay

`build_replay_audit(..., crowd=True)` calls `compute_crowd_diagnostics` on
the same inputs and attaches an overlay. It uses the definitions in
[crowd-diagnostics.md](crowd-diagnostics.md). It does not define a second
notion of congestion. The overlay is optional, and it never changes any
other audit field.

| Field | Meaning |
| --- | --- |
| `crowd_diagnostics_version` | the crowd diagnostics version |
| `crowd_diagnostics_digest` | `CrowdDiagnostics.digest()` of the diagnostics used |
| `steps[k - 1].blocked` | every recorded blocked move of step `k`: `person_id`, `origin`, `target`, `reason` (the recorded `blocked_*` outcome) |
| `steps[k - 1].contested_cells` | target cells with two or more contenders in step `k`, row-major; each lists the contenders in person-ID order with their recorded outcome |
| `steps[k - 1].congestion_blocks` | `blocked_occupied + blocked_conflict + blocked_cycle` in step `k` |
| `steps[k - 1].evacuations_by_exit` | evacuations per exit in step `k` |
| `hotspots` | the `CellCongestion` entries with at least one congestion block, row-major |
| `exit_throughput` | `CrowdDiagnostics.exits` (`ExitLoadSignals`) |

A contender is a movement attempt that passed the static boundary, wall and
fire checks (the crowd diagnostics definition). The overlay build checks
that the number of blocked moves, contested cells, the largest contender
group and the congestion blocks of every step equal the matching
`StepCrowdSignals` values. A difference raises `ReplayAuditError`.

## 7. Canonical text renderer

`render_frame(audit, step, *, events=False, header=False)` returns one
ASCII frame. Rows are joined with `\n`. There is no trailing newline and no
colour. `render_initial(audit)` is frame 0 and `render_final(audit)` is
frame `N`.

### Legend

The legend is in precedence order. When several entries apply to one cell,
the first one wins.

| Glyph | Meaning | Layer |
| --- | --- | --- |
| `#` | wall | base |
| `X` | casualty in this step | event |
| `V` | evacuation in this step | event |
| `f` | newly ignited in this step | event |
| `F` | burning cell | base |
| `P` | active person | base |
| `E` | exit that is not burning | base |
| `.` | empty cell | base |

- Event glyphs appear only with `events=True`. Frame 0 has no events.
- Without events, a frame equals `emberbench.render(scenario, state)` for
  the same state. The tests check this for every step of every audited
  episode.
- A burning exit shows `F`. A casualty cell is always burning, so it shows
  `F`, or `X` in the casualty step with events.
- With `header=True` the first line is
  `step k/N active=a evacuated=e casualties=c burning=b`, followed by
  ` terminated=<reason>` on the final frame.

`colorize(text)` adds ANSI colour for a human terminal. Its output is never
canonical and never enters an artifact.

Example: one person enters a fire cell while another evacuates, and the
fire spreads in the same step (`#PF.PE#` at step 0):

```
step 1/1 active=0 evacuated=1 casualties=1 burning=3 terminated=all_resolved
#######
#fXf.V#
#######
```

## 8. Portable animation schema

`build_animation(audit)` returns a provider-neutral document. It holds data
only: no HTML, no script, no styling. `animation_json(audit)` is its
canonical JSON, and `export_animation(scenario, record, *, crowd=False)`
authenticates, audits and serialises in one call.

Top-level keys:

| Key | Meaning |
| --- | --- |
| `animation_format_version` | `"1.0"` (`ANIMATION_FORMAT_VERSION`), separate from the audit version |
| `replay_audit_version` | the audit version the document was built from |
| `source` | `simulator_version`, `observation_version`, `schema_version`, `scenario_id`, `scenario_hash`, `seed`, `record_digest`, `audit_digest` |
| `map` | as in the audit |
| `people` | every person ID, sorted |
| `episode_steps`, `termination_reason` | as in the audit |
| `frames` | `N + 1` frames, step order |
| `congestion` | `null`, or `hotspots` and `exit_throughput` from the overlay |

Each frame has these keys:

| Key | Meaning |
| --- | --- |
| `step` | `0..N` |
| `people` | every person: `id`, `position`, `status` (person-ID order) |
| `burning` | every burning cell after the step (row-major) |
| `newly_ignited` | cells ignited in the step |
| `movements` | movement events: `person_id`, `action`, `origin`, `target`, `final`, `outcome` |
| `evacuations` | evacuation events |
| `casualties` | casualty events |
| `congestion` | `null` without the overlay; otherwise `blocked` and `contested_cells` for the step (empty at step 0) |
| `terminated`, `termination_reason` | as in the timeline |

A viewer can draw any frame from `map` and one frame alone.

## 9. Determinism

- The same record gives byte-identical audit JSON, animation JSON and text
  frames, also after a `RunRecord` JSON round trip.
- People are in person-ID order, cells in row-major order, exits in exit-ID
  order and events in the recorded (person-ID) order.
- The tests pin golden digests of the audit, the animation and the text
  frames, and check them in subprocesses with `PYTHONHASHSEED` 0, 1 and
  12345.

## 10. Cost

The audit is built in one pass over the steps. Its size is linear in the
record: frames hold every person, and fire deltas hold the burning sets
before and after each step. The tests audit a 100-person crowd run and a
long stochastic run. There is no wall-clock gate.

## 11. Versions

`REPLAY_AUDIT_VERSION` and `ANIMATION_FORMAT_VERSION` change only when a
field of these artifacts changes. This feature changes no simulator,
observation, action schema, scenario schema, metric, statistic, generator,
baseline, integrity protocol or CLI version, and no suite hash.
