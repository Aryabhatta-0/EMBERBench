"""Replay audit (``audit.py``), text frames and the animation export (``visualization.py``).

The audit only reads an authenticated record. These tests check that every
frame, path and fire delta reconciles exactly with the ``StepRecord`` and
``SimState`` values, that the congestion layer reconciles with
``compute_crowd_diagnostics``, that the output is byte-stable, and that any
unauthentic input fails closed. There are no image snapshots: frames are
plain text and the exports are pinned by digest.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from conftest import FIXTURE_NAMES, crowd_stress_scenario, fixture_scenario, greedy_exit_policy, layout
from emberbench import (
    ANIMATION_FORMAT_VERSION,
    REPLAY_AUDIT_VERSION,
    Action,
    CasualtyCause,
    MetricsInputError,
    MoveOutcome,
    Observation,
    PersonStatus,
    Pos,
    ReplayAudit,
    ReplayAuditError,
    RunRecord,
    Scenario,
    TerminationConfig,
    TerminationReason,
    agent_policy,
    animation_json,
    baseline_factory,
    build_animation,
    build_replay_audit,
    compute_crowd_diagnostics,
    export_animation,
    load_scenario,
    render,
    render_frame,
    run_episode,
    verify_animation,
    verify_replay_audit,
)
from emberbench.canonical import canonical_json
from emberbench.events import Casualty
from emberbench.visualization import LEGEND, colorize, frame_header, render_final, render_initial, render_legend

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "scenarios" / "public" / "v1"
E, W, STAY = Action.EAST, Action.WEST, Action.STAY
Policy = Callable[[Observation], Mapping[str, Any]]


def public(scenario_id: str) -> Scenario:
    return load_scenario(SUITE_DIR / f"{scenario_id}.json")


def always(action: Action) -> Policy:
    return lambda obs: {p.id: action for p in obs.active}


def scripted(script: Mapping[int, Mapping[str, Action]], then: Policy = greedy_exit_policy) -> Policy:
    """Fixed actions for the listed steps (observation step numbers), then ``then``."""

    def policy(obs: Observation) -> Mapping[str, Any]:
        return script[obs.step] if obs.step in script else then(obs)

    return policy


def baseline(baseline_id: str) -> Policy:
    return agent_policy(baseline_factory(baseline_id)())


# --------------------------------------------------------------------------- hand-built episodes

# Pa enters the fire (entered_fire), Pb evacuates, and the fire spreads, all in step 1.
OVERLAP = layout(["#######", "#aF.bE#", "#######"], scenario_id="overlap", termination=TerminationConfig(max_steps=5))
# One step: Pa steps onto the exit.
SHORT = layout(["######", "#aE.F#", "######"], scenario_id="short")
# Fire walls Pa off, then spreads onto Pa's cell in step 2 (fire_spread).
SPREAD = layout(["#######", "#a.F.E#", "#######"], scenario_id="spread", termination=TerminationConfig(max_steps=5))
# The same room cut at step 1: Pa is unresolved and trapped.
STRANDED = dataclasses.replace(SPREAD, scenario_id="stranded", termination=TerminationConfig(max_steps=1))
# Pa and Pb both target (1, 2) in step 1. Pa wins (lower ID), Pb gets blocked_conflict.
CONFLICT = layout(["#######", "#a.b..#", "##E####", "#....F#", "#######"], scenario_id="conflict")
CONFLICT_SCRIPT = {0: {"Pa": E, "Pb": W}}


def _episodes() -> dict[str, tuple[Scenario, RunRecord]]:
    cases: dict[str, tuple[Scenario, Policy, int | None]] = {
        "overlap": (OVERLAP, always(E), None),
        "short": (SHORT, greedy_exit_policy, None),
        "spread": (SPREAD, always(STAY), None),
        "stranded": (STRANDED, always(STAY), None),
        "conflict": (CONFLICT, scripted(CONFLICT_SCRIPT), None),
    }
    for name in FIXTURE_NAMES:
        cases[f"fixture:{name}"] = (fixture_scenario(name), greedy_exit_policy, None)
    cases["public:route-invalidation:fire-aware:s1"] = (public("v1-pub-10-route-invalidation"), baseline("fire-aware-shortest-path-v1"), 1)
    cases["public:route-invalidation:stay:s7"] = (public("v1-pub-10-route-invalidation"), always(STAY), 7)
    cases["public:crowd-100:greedy"] = (public("v1-pub-09-crowd-100-split-flow"), greedy_exit_policy, None)
    cases["public:bottleneck:congestion-aware"] = (public("v1-pub-03-corridor-bottleneck"), baseline("congestion-aware-v1"), None)
    cases["stress-100"] = (crowd_stress_scenario(100), greedy_exit_policy, None)
    return {name: (s, run_episode(s, policy, seed=seed)) for name, (s, policy, seed) in cases.items()}


EPISODES = _episodes()
EPISODE_NAMES = sorted(EPISODES)


@pytest.fixture(scope="module", params=EPISODE_NAMES)
def episode(request: pytest.FixtureRequest) -> tuple[Scenario, RunRecord, ReplayAudit]:
    scenario, record = EPISODES[request.param]
    return scenario, record, build_replay_audit(scenario, record, crowd=True)


def _states(record: RunRecord) -> list[Any]:
    return [record.initial_state] + [s.state for s in record.steps]


# --------------------------------------------------------------------------- header and lookup


def test_header_binds_the_audit_to_the_record() -> None:
    scenario, record = EPISODES["fixture:stochastic_room"]
    audit = build_replay_audit(scenario, record)
    assert audit.replay_audit_version == REPLAY_AUDIT_VERSION == "1.0"
    assert (audit.simulator_version, audit.observation_version, audit.schema_version) == (
        record.simulator_version,
        record.observation_version,
        record.schema_version,
    )
    assert (audit.scenario_id, audit.scenario_hash, audit.seed) == (record.scenario_id, scenario.content_hash(), record.seed)
    assert audit.record_digest == record.digest()
    assert audit.episode_steps == len(record.steps)
    assert audit.termination_reason is record.final_state.termination_reason
    assert audit.crowd is None
    assert audit.map.walls == tuple(sorted(scenario.walls))
    assert [eid for eid, _ in audit.map.exits] == sorted(e.id for e in scenario.exits)


def test_lookup_by_step_is_deterministic_and_bounded() -> None:
    scenario, record = EPISODES["conflict"]
    audit = build_replay_audit(scenario, record)
    for k in range(audit.episode_steps + 1):
        assert audit.frame(k).step == k
        assert audit.frame(k) is audit.timeline[k]
    assert audit.initial_frame is audit.frame(0)
    assert audit.final_frame is audit.frame(audit.episode_steps)
    for bad in (-1, audit.episode_steps + 1):
        with pytest.raises(ValueError):
            audit.frame(bad)
    for wrong in (True, 1.0, "1"):
        with pytest.raises(TypeError):
            audit.frame(wrong)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        audit.fire_delta(0)
    assert audit.step_congestion(1) is None  # no crowd layer requested
    with pytest.raises(KeyError):
        audit.person("nobody")


# --------------------------------------------------------------------------- timeline


def test_timeline_reconciles_with_the_record(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    _, record, audit = episode
    states = _states(record)
    assert len(audit.timeline) == len(record.steps) + 1
    initial = audit.frame(0)
    assert (initial.movements, initial.newly_ignited, initial.evacuations, initial.casualties) == ((), (), (), ())
    for k, frame in enumerate(audit.timeline):
        state = states[k]
        assert frame.step == state.step == k
        assert frame.people == state.people
        assert frame.burning_count == len(state.burning)
        assert frame.active_count == len(state.active)
        assert frame.evacuated_total == len(state.ids_with_status(PersonStatus.EVACUATED))
        assert frame.casualty_total == len(state.ids_with_status(PersonStatus.CASUALTY))
        assert frame.trapped == tuple(sorted(state.trapped))
        assert (frame.terminated, frame.termination_reason) == (state.terminated, state.termination_reason)
        if k:
            step = record.steps[k - 1]
            assert frame.movements == step.movements
            assert frame.newly_ignited == step.ignited
            assert frame.evacuations == step.evacuations
            assert frame.casualties == step.casualties
            assert [m.action for m in frame.movements] == [a for _, a in step.actions]
    # Only the last frame is terminal.
    assert [f.terminated for f in audit.timeline] == [False] * audit.episode_steps + [True]


def test_no_event_is_inferred(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    _, record, audit = episode
    final = record.final_state
    assert sum(len(f.evacuations) for f in audit.timeline) == len(final.ids_with_status(PersonStatus.EVACUATED))
    assert sum(len(f.casualties) for f in audit.timeline) == len(final.ids_with_status(PersonStatus.CASUALTY))
    assert sum(len(f.movements) for f in audit.timeline) == sum(len(s.movements) for s in record.steps)
    assert sum(len(f.newly_ignited) for f in audit.timeline) == len(final.burning) - len(record.initial_state.burning)


# --------------------------------------------------------------------------- person paths


def test_person_paths_reconcile_with_step_records_and_states(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    _, record, audit = episode
    states = _states(record)
    final = record.final_state
    assert [p.person_id for p in audit.people] == [p.id for p in final.people]
    for path in audit.people:
        pid = path.person_id
        assert path.initial_position == record.initial_state.person(pid).position
        # One entry per step in which the person was active at decision time, without gaps.
        active_steps = [s.step for s in record.steps if pid in s.action_map]
        assert [ps.step for ps in path.steps] == active_steps == list(range(1, len(path.steps) + 1))
        for ps in path.steps:
            m = next(m for m in record.steps[ps.step - 1].movements if m.person_id == pid)
            assert (ps.action, ps.outcome, ps.origin, ps.target, ps.position) == (m.action, m.outcome, m.origin, m.target, m.final)
        for k in range(audit.episode_steps + 1):
            assert path.position_at(k) == states[k].person(pid).position
        person = final.person(pid)
        assert (path.final_status, path.final_position) == (person.status, person.position)
        evac = [(s.step, e) for s in record.steps for e in s.evacuations if e.person_id == pid]
        dead = [(s.step, c) for s in record.steps for c in s.casualties if c.person_id == pid]
        assert len(evac) + len(dead) <= 1
        if evac:
            assert path.final_status is PersonStatus.EVACUATED
            assert (path.resolution_step, path.exit_id, path.casualty_cause) == (evac[0][0], evac[0][1].exit_id, None)
            assert path.steps[-1].step == path.resolution_step
        elif dead:
            assert path.final_status is PersonStatus.CASUALTY
            assert (path.resolution_step, path.exit_id, path.casualty_cause) == (dead[0][0], None, dead[0][1].cause)
            assert path.steps[-1].step == path.resolution_step
        else:
            assert path.unresolved and path.final_status is PersonStatus.ACTIVE
            assert (path.resolution_step, path.exit_id, path.casualty_cause) == (None, None, None)
            assert len(path.steps) == audit.episode_steps
        assert path.trapped_at_end == (path.unresolved and pid in final.trapped)
        assert audit.person(pid) is path


def test_evacuation_path() -> None:
    scenario, record = EPISODES["short"]
    audit = build_replay_audit(scenario, record)
    assert audit.episode_steps == 1 and audit.termination_reason is TerminationReason.ALL_RESOLVED
    path = audit.person("Pa")
    assert (path.final_status, path.resolution_step, path.exit_id) == (PersonStatus.EVACUATED, 1, "E1")
    assert [(s.action, s.outcome) for s in path.steps] == [(E, MoveOutcome.MOVED)]
    assert path.position_at(1) == path.final_position == Pos(1, 2)


def test_casualty_by_entering_fire_and_by_spread() -> None:
    entered = build_replay_audit(*EPISODES["overlap"]).person("Pa")
    assert (entered.final_status, entered.resolution_step, entered.casualty_cause) == (PersonStatus.CASUALTY, 1, CasualtyCause.ENTERED_FIRE)
    assert entered.final_position == Pos(1, 2) and entered.steps[0].outcome is MoveOutcome.MOVED
    spread_audit = build_replay_audit(*EPISODES["spread"])
    spread = spread_audit.person("Pa")
    assert (spread.final_status, spread.resolution_step, spread.casualty_cause) == (PersonStatus.CASUALTY, 2, CasualtyCause.FIRE_SPREAD)
    assert spread.final_position == spread.initial_position == Pos(1, 1)
    assert spread_audit.frame(2).casualties == (Casualty("Pa", Pos(1, 1), CasualtyCause.FIRE_SPREAD),)


def test_stranded_person_is_unresolved() -> None:
    audit = build_replay_audit(*EPISODES["stranded"])
    assert audit.termination_reason is TerminationReason.MAX_STEPS and audit.episode_steps == 1
    path = audit.person("Pa")
    assert path.unresolved and path.trapped_at_end
    assert (path.resolution_step, path.exit_id, path.casualty_cause) == (None, None, None)
    assert [(s.action, s.outcome) for s in path.steps] == [(STAY, MoveOutcome.STAYED)]
    assert audit.final_frame.trapped == ("Pa",)


# --------------------------------------------------------------------------- fire


def test_fire_deltas_reconcile_exactly(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    scenario, record, audit = episode
    states = _states(record)
    assert audit.initial_burning == record.initial_state.burning == frozenset(scenario.ignitions)
    assert audit.burning_at(0) == audit.initial_burning
    assert [d.step for d in audit.fire] == list(range(1, audit.episode_steps + 1))
    for k in range(1, audit.episode_steps + 1):
        delta = audit.fire_delta(k)
        assert delta.burning_before == states[k - 1].burning
        assert delta.newly_ignited == record.steps[k - 1].ignited == tuple(sorted(delta.newly_ignited))
        assert delta.burning_after == states[k].burning == audit.burning_at(k)
        assert not delta.burning_before & set(delta.newly_ignited)
        assert delta.burning_after == delta.burning_before | set(delta.newly_ignited)
        if k > 1:
            assert delta.burning_before == audit.fire_delta(k - 1).burning_after


def test_stochastic_fire_is_audited_not_predicted() -> None:
    scenario, record = EPISODES["public:route-invalidation:stay:s7"]
    audit = build_replay_audit(scenario, record)
    # The audit holds exactly the recorded ignitions and nothing after the last step.
    assert len(audit.fire) == len(record.steps)
    assert [d.newly_ignited for d in audit.fire] == [s.ignited for s in record.steps]
    other = run_episode(scenario, always(STAY), seed=8)
    assert build_replay_audit(scenario, other).digest() != audit.digest()


# --------------------------------------------------------------------------- congestion overlay


def test_crowd_overlay_reconciles_with_crowd_diagnostics(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    scenario, record, audit = episode
    diag = compute_crowd_diagnostics(scenario, record)
    overlay = audit.crowd
    assert overlay is not None
    assert overlay.crowd_diagnostics_version == diag.crowd_diagnostics_version
    assert overlay.crowd_diagnostics_digest == diag.digest()
    assert overlay.exit_throughput == diag.exits
    assert overlay.hotspots == tuple(c for c in diag.cells if c.congestion_blocks)
    assert len(overlay.hotspots) == diag.totals.hotspot_cells
    for step, signals, markers in zip(record.steps, diag.steps, overlay.steps, strict=True):
        assert markers.step == step.step == audit.step_congestion(step.step).step  # type: ignore[union-attr]
        blocked = [m for m in step.movements if m.outcome not in (MoveOutcome.STAYED, MoveOutcome.MOVED)]
        assert [(b.person_id, b.origin, b.target, b.reason) for b in markers.blocked] == [
            (m.person_id, m.origin, m.target, m.outcome) for m in blocked
        ]
        assert len(markers.contested_cells) == signals.contested_cells
        assert max((len(c.contenders) for c in markers.contested_cells), default=0) == (
            signals.max_contenders if signals.contested_cells else 0
        )
        assert markers.congestion_blocks == signals.congestion_blocks
        assert markers.evacuations_by_exit == signals.evacuations_by_exit
        cells = [c.cell for c in markers.contested_cells]
        assert cells == sorted(cells)
        for c in markers.contested_cells:
            # At most one contender enters a contested cell; every other one is blocked by congestion.
            outcomes = [o for _, o in c.contenders]
            assert outcomes.count(MoveOutcome.MOVED) <= 1
            assert set(outcomes) <= {MoveOutcome.MOVED, MoveOutcome.BLOCKED_OCCUPIED, MoveOutcome.BLOCKED_CONFLICT, MoveOutcome.BLOCKED_CYCLE}


def test_conflict_markers() -> None:
    audit = build_replay_audit(*EPISODES["conflict"], crowd=True)
    markers = audit.step_congestion(1)
    assert markers is not None
    assert [c.to_dict() for c in markers.contested_cells] == [
        {"cell": [1, 2], "contenders": [{"person_id": "Pa", "outcome": "moved"}, {"person_id": "Pb", "outcome": "blocked_conflict"}]}
    ]
    assert [b.to_dict() for b in markers.blocked] == [{"person_id": "Pb", "origin": [1, 3], "target": [1, 2], "reason": "blocked_conflict"}]
    assert markers.congestion_blocks == 1
    assert audit.step_congestion(0) is None
    assert [c.cell for c in audit.crowd.hotspots] == [Pos(1, 2)]  # type: ignore[union-attr]


def test_crowd_layer_is_optional_and_does_not_change_the_base_audit() -> None:
    scenario, record = EPISODES["public:crowd-100:greedy"]
    base = build_replay_audit(scenario, record).to_dict()
    with_crowd = build_replay_audit(scenario, record, crowd=True).to_dict()
    assert base["crowd"] is None and with_crowd["crowd"] is not None
    assert {k: v for k, v in with_crowd.items() if k != "crowd"} == {k: v for k, v in base.items() if k != "crowd"}


def test_no_crowd_score_in_the_overlay() -> None:
    text = build_replay_audit(*EPISODES["public:crowd-100:greedy"], crowd=True).to_json()
    assert "score" not in text and "ratio" not in text


# --------------------------------------------------------------------------- text frames


def test_base_frames_equal_the_state_renderer(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    scenario, record, audit = episode
    for k, state in enumerate(_states(record)):
        assert render_frame(audit, k) == render(scenario, state)


def test_overlap_frames_follow_the_fixed_precedence() -> None:
    audit = build_replay_audit(*EPISODES["overlap"])
    assert render_initial(audit) == "#######\n#PF.PE#\n#######"
    assert render_frame(audit, 1) == "#######\n#FFF.E#\n#######"
    # Casualty X over fire, evacuation V over exit, newly ignited f over F.
    assert render_frame(audit, 1, events=True) == "#######\n#fXf.V#\n#######"
    assert render_final(audit, events=True, header=True) == (
        "step 1/1 active=0 evacuated=1 casualties=1 burning=3 terminated=all_resolved\n#######\n#fXf.V#\n#######"
    )
    assert render_frame(audit, 0, events=True) == render_initial(audit)  # step 0 has no events


def test_burning_exit_shows_fire() -> None:
    scenario, record = EPISODES["fixture:fire_threatening_exit"]
    audit = build_replay_audit(scenario, record)
    (exit_id, pos), *_ = [(eid, p) for eid, p in audit.map.exits if p in audit.burning_at(audit.episode_steps)]
    assert render_final(audit).split("\n")[pos.row][pos.col] == "F"
    assert render_initial(audit).split("\n")[pos.row][pos.col] == "E"


def test_legend_precedence_and_plain_ascii() -> None:
    assert [g for g, _ in LEGEND] == ["#", "X", "V", "f", "F", "P", "E", "."]
    assert len({g for g, _ in LEGEND}) == len(LEGEND)
    assert render_legend().split("\n")[0] == "# wall"
    for name in EPISODE_NAMES:
        audit = build_replay_audit(*EPISODES[name])
        for k in (0, audit.episode_steps):
            text = render_frame(audit, k, events=True, header=True)
            assert text.isascii() and "\x1b" not in text and not text.endswith("\n")
            grid = text.split("\n")[1:]
            assert len(grid) == audit.map.height and {len(r) for r in grid} == {audit.map.width}
            assert set("".join(grid)) <= {g for g, _ in LEGEND}


def test_colour_output_is_separate_from_canonical_frames() -> None:
    audit = build_replay_audit(*EPISODES["overlap"])
    plain = render_frame(audit, 1, events=True, header=True)
    coloured = colorize(plain)
    assert "\x1b[" in coloured and coloured != plain
    assert coloured.split("\n")[0] == frame_header(audit, 1)
    assert render_frame(audit, 1, events=True, header=True) == plain  # colour never feeds back
    assert "\x1b" not in animation_json(audit) and "\x1b" not in audit.to_json()


# --------------------------------------------------------------------------- animation export


def test_animation_frames_carry_every_recorded_event(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    _, record, audit = episode
    doc = build_animation(audit)
    assert doc["animation_format_version"] == ANIMATION_FORMAT_VERSION == "1.0"
    assert doc["replay_audit_version"] == REPLAY_AUDIT_VERSION
    assert doc["source"]["record_digest"] == record.digest() and doc["source"]["audit_digest"] == audit.digest()
    assert doc["people"] == [p.person_id for p in audit.people]
    assert [f["step"] for f in doc["frames"]] == list(range(audit.episode_steps + 1))
    for frame, state in zip(doc["frames"], _states(record), strict=True):
        assert frame["people"] == [p.to_dict() for p in state.people]
        assert frame["burning"] == [p.to_list() for p in sorted(state.burning)]
    for frame, step in zip(doc["frames"][1:], record.steps, strict=True):
        assert frame["newly_ignited"] == [p.to_list() for p in step.ignited]
        assert frame["movements"] == [m.to_dict() for m in step.movements]
        assert frame["evacuations"] == [e.to_dict() for e in step.evacuations]
        assert frame["casualties"] == [c.to_dict() for c in step.casualties]
        assert frame["congestion"]["blocked"] == [b.to_dict() for b in audit.step_congestion(step.step).blocked]  # type: ignore[union-attr]
    assert doc["frames"][0]["congestion"] == {"blocked": [], "contested_cells": []}
    assert doc["frames"][-1]["termination_reason"] == audit.termination_reason.value


ANIMATION_KEYS = {
    "animation_format_version",
    "replay_audit_version",
    "source",
    "map",
    "people",
    "episode_steps",
    "termination_reason",
    "frames",
    "congestion",
}
FRAME_KEYS = {
    "step",
    "people",
    "burning",
    "newly_ignited",
    "movements",
    "evacuations",
    "casualties",
    "congestion",
    "terminated",
    "termination_reason",
}
AUDIT_KEYS = {
    "replay_audit_version",
    "simulator_version",
    "observation_version",
    "schema_version",
    "scenario_id",
    "scenario_hash",
    "seed",
    "record_digest",
    "map",
    "episode_steps",
    "termination_reason",
    "timeline",
    "people",
    "fire",
    "crowd",
}


def test_schemas_have_a_fixed_key_set() -> None:
    scenario, record = EPISODES["conflict"]
    for crowd in (False, True):
        audit = build_replay_audit(scenario, record, crowd=crowd)
        assert set(audit.to_dict()) == AUDIT_KEYS
        doc = build_animation(audit)
        assert set(doc) == ANIMATION_KEYS
        assert {k for f in doc["frames"] for k in f} == FRAME_KEYS
        assert (doc["congestion"] is None) == (not crowd)
        assert all((f["congestion"] is None) == (not crowd) for f in doc["frames"])


def test_no_markup_timestamps_paths_or_environment() -> None:
    scenario, record = EPISODES["public:crowd-100:greedy"]
    audit = build_replay_audit(scenario, record, crowd=True)
    banned_words = {"time", "timestamp", "date", "created", "path", "dir", "file", "host", "hostname", "env", "environment"}
    banned_words |= {"user", "cwd", "platform", "python", "machine", "pid", "duration", "elapsed"}
    for text in (audit.to_json(), animation_json(audit)):
        for key in _keys(json.loads(text)):
            assert not banned_words & set(key.split("_")), key
        for needle in ("<", "script", str(ROOT), os.getcwd(), "\\\\", os.environ.get("USERNAME", "\x00"), sys.version.split()[0]):
            assert needle not in text, needle


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _keys(v)}
    return set()


# --------------------------------------------------------------------------- determinism


def test_same_record_gives_byte_identical_artifacts(episode: tuple[Scenario, RunRecord, ReplayAudit]) -> None:
    scenario, record, audit = episode
    reloaded = RunRecord.from_json(record.to_json())
    again = build_replay_audit(scenario, reloaded, crowd=True)
    assert again.to_json() == audit.to_json()
    assert animation_json(again) == animation_json(audit)
    assert audit.to_json() == canonical_json(json.loads(audit.to_json()))
    for k in range(audit.episode_steps + 1):
        assert render_frame(again, k, events=True, header=True) == render_frame(audit, k, events=True, header=True)


def test_canonical_ordering() -> None:
    audit = build_replay_audit(*EPISODES["stress-100"], crowd=True)
    ids = [p.person_id for p in audit.people]
    assert ids == sorted(ids) and len(ids) == 100
    for frame in audit.timeline:
        assert [p.id for p in frame.people] == ids
        assert [m.person_id for m in frame.movements] == sorted(m.person_id for m in frame.movements)
        assert [e.person_id for e in frame.evacuations] == sorted(e.person_id for e in frame.evacuations)
        assert [c.person_id for c in frame.casualties] == sorted(c.person_id for c in frame.casualties)
        assert list(frame.newly_ignited) == sorted(frame.newly_ignited)
    doc = build_animation(audit)
    for frame in doc["frames"]:
        assert frame["burning"] == sorted(frame["burning"])


def _frames_digest(audit: ReplayAudit) -> str:
    text = "\n\n".join(render_frame(audit, k, events=True, header=True) for k in range(audit.episode_steps + 1))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


GOLDEN_CASES = ("conflict", "fixture:stochastic_room", "public:crowd-100:greedy", "public:route-invalidation:fire-aware:s1")

# (audit digest without crowd, audit digest with crowd, animation digest with crowd, text frames digest).
# They must be equal on every platform, Python version and hash seed.
GOLDEN_DIGESTS: dict[str, tuple[str, str, str, str]] = {
    "conflict": (
        "041ae3587a6579e262e95f14711c1f45d67a8920b746755a8351834e0a6bb2de",
        "0368ae4a0519af3f140fc325385bcd3ebe8ec72ecd92a17d13d214fc010bb47c",
        "ca3234b14b89120aef79cdf99b6d9e5bec5d93e69db30bccacddc62a4ee2dc40",
        "7d9a35f8f316fbd47917b0ba1b11a1c4e813770a898869f8e8beab118209768c",
    ),
    "fixture:stochastic_room": (
        "20f9693b3b87f0e19d6cb73ece5f832219089ac1149d5a18bdc0a2a8617f0697",
        "ed800d3935c97ca1584147b115dd855321ddab35f1aefd6bb86bb9e3f7254055",
        "acd167039a7b6ad870a922d1a1dc83c7812625259a8f27f88081ea9b2486e6e2",
        "f0f30f7d5835dad4fee91764faa18f99fe85f2126320acb5290d66abed81b717",
    ),
    "public:crowd-100:greedy": (
        "685adca15d2b2619eeb30ae81362fc9a186ad1f68fa9581043789be37f0212f4",
        "5cde5dddfb925f6a6c4e99ce0ac8a18eb7432b1073560fdb76991da312467f3b",
        "7f8a0a8453118e5a802259560c5b4e3f690aa3e9ce4b87db1c418bbe743fe652",
        "bd85984d4c691c51617e873c51e01e4a419da5bd176c25f9318bd57f5f64cd2d",
    ),
    "public:route-invalidation:fire-aware:s1": (
        "69482ae8263e4ac9f4030b787f28293bd85b59082d6470c4967aa1373279f329",
        "34a6a0a713ec8aa378a7cf8516eec79d2f55a1f7955bbf684f8dba19dc391a68",
        "7954b3d4b2dc02ed6b4e7ea28a99ee19c2b7b2a7f5d43fb8e38a8e61d9376c7a",
        "6cb1b4060fe361121ffa50f012fc14ab202be4f8cda27a79caef604fe50faf2f",
    ),
}


def _golden_row(name: str) -> tuple[str, str, str, str]:
    scenario, record = EPISODES[name]
    base = build_replay_audit(scenario, record)
    full = build_replay_audit(scenario, record, crowd=True)
    return base.digest(), full.digest(), hashlib.sha256(animation_json(full).encode("ascii")).hexdigest(), _frames_digest(base)


@pytest.mark.parametrize("name", GOLDEN_CASES)
def test_golden_digests(name: str) -> None:
    assert _golden_row(name) == GOLDEN_DIGESTS[name], name


_SUBPROCESS_SCRIPT = """
import sys
sys.path[:0] = [{src!r}, {tests!r}]
from test_replay_audit import GOLDEN_CASES, _golden_row
for name in GOLDEN_CASES:
    print(name, *_golden_row(name))
"""


def test_artifacts_identical_across_processes_and_hash_seeds() -> None:
    script = _SUBPROCESS_SCRIPT.format(src=str(ROOT / "src"), tests=str(ROOT / "tests"))
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout.replace("\r\n", "\n"))
    assert outputs[0] == outputs[1] == outputs[2]
    for line in outputs[0].splitlines():
        name, *digests = line.split()
        assert tuple(digests) == GOLDEN_DIGESTS[name], name


# --------------------------------------------------------------------------- generated scenario and scale


def test_generated_scenario() -> None:
    from emberbench import GeneratorConfig, generate_suite
    from test_generator import BASE_CONFIG, BATCH, SEED

    suite = generate_suite(GeneratorConfig.from_dict(BASE_CONFIG), SEED, batch_id=BATCH)
    for scenario in suite.scenarios[:2]:
        record = run_episode(scenario, greedy_exit_policy)
        audit = build_replay_audit(scenario, record, crowd=True)
        assert audit.final_frame.people == record.final_state.people
        assert audit.burning_at(audit.episode_steps) == record.final_state.burning
        verify_replay_audit(scenario, record, audit.to_json())
        verify_animation(scenario, record, animation_json(audit))


def test_large_and_long_episodes() -> None:
    # 100 people and a long stochastic run. No wall-clock gate: the build is one pass over the steps.
    for name in ("stress-100", "public:crowd-100:greedy", "public:route-invalidation:stay:s7"):
        scenario, record = EPISODES[name]
        audit = build_replay_audit(scenario, record, crowd=True)
        assert len(audit.timeline) == len(record.steps) + 1
        assert sum(len(p.steps) for p in audit.people) == sum(len(s.movements) for s in record.steps)
    long_record = EPISODES["public:route-invalidation:stay:s7"][1]
    assert len(long_record.steps) >= 20


# --------------------------------------------------------------------------- fail closed


def _replace_step(record: RunRecord, index: int, **changes: Any) -> RunRecord:
    steps = list(record.steps)
    steps[index] = dataclasses.replace(steps[index], **changes)
    return dataclasses.replace(record, steps=tuple(steps))


def _tampered() -> dict[str, tuple[Scenario, RunRecord]]:
    scenario, record = EPISODES["fixture:stochastic_room"]
    other = fixture_scenario("narrow_corridor")
    step2 = record.steps[1]
    cases = {
        "wrong scenario": (other, record),
        "wrong scenario hash": (scenario, dataclasses.replace(record, scenario_hash="0" * 64)),
        "wrong scenario id": (scenario, dataclasses.replace(record, scenario_id="other")),
        "simulator version": (scenario, dataclasses.replace(record, simulator_version="0.0.1")),
        "observation version": (scenario, dataclasses.replace(record, observation_version="0")),
        "schema version": (scenario, dataclasses.replace(record, schema_version="0.9")),
        "truncated (no termination)": (scenario, dataclasses.replace(record, steps=record.steps[:-1])),
        "no steps": (scenario, dataclasses.replace(record, steps=())),
        "seed (replay mismatch)": (scenario, dataclasses.replace(record, seed=record.seed + 1)),
        "rng state (replay mismatch)": (
            scenario,
            _replace_step(record, 1, state=dataclasses.replace(step2.state, rng_state=(("fire", 1),))),
        ),
        "dropped movement": (scenario, _replace_step(record, 1, movements=step2.movements[1:])),
        "trapped set": (scenario, _replace_step(record, 1, state=dataclasses.replace(step2.state, trapped=frozenset({"ghost"})))),
        "extra ignition": (scenario, _replace_step(record, 1, ignited=step2.ignited + (Pos(0, 0),))),
    }
    tampered_json = json.loads(record.to_json())
    tampered_json["steps"][0]["state"]["people"][0]["position"] = [0, 0]
    cases["edited JSON"] = (scenario, RunRecord.from_dict(tampered_json))
    return cases


TAMPERED = _tampered()


@pytest.mark.parametrize("case", sorted(TAMPERED))
def test_unauthentic_records_fail_closed(case: str) -> None:
    scenario, record = TAMPERED[case]
    for crowd in (False, True):
        with pytest.raises(ReplayAuditError):
            build_replay_audit(scenario, record, crowd=crowd)
    with pytest.raises(ReplayAuditError):
        export_animation(scenario, record)


def test_audit_error_is_a_metrics_input_error() -> None:
    assert issubclass(ReplayAuditError, MetricsInputError)
    scenario, record = EPISODES["short"]
    with pytest.raises(TypeError):
        build_replay_audit(record, scenario)  # type: ignore[arg-type]


def test_verify_rejects_tampered_artifacts() -> None:
    scenario, record = EPISODES["conflict"]
    audit = build_replay_audit(scenario, record, crowd=True)
    assert verify_replay_audit(scenario, record, audit).to_json() == audit.to_json()
    assert verify_replay_audit(scenario, record, audit.to_json()).to_json() == audit.to_json()
    verify_animation(scenario, record, animation_json(audit))
    verify_animation(scenario, record, export_animation(scenario, record))

    edited = json.loads(audit.to_json())
    edited["timeline"][1]["people"][0]["position"] = [0, 0]
    old = json.loads(audit.to_json())
    old["replay_audit_version"] = "0.9"
    stripped = json.loads(audit.to_json())
    stripped["crowd"]["hotspots"] = []
    for bad in (json.dumps(edited), json.dumps(old), json.dumps(stripped), "{not json", "[]"):
        with pytest.raises(ReplayAuditError):
            verify_replay_audit(scenario, record, bad)

    doc = build_animation(audit)
    doc["frames"][1]["casualties"] = [{"person_id": "Pa", "position": [1, 2], "cause": "entered_fire"}]
    newer = build_animation(audit)
    newer["animation_format_version"] = "2.0"
    for bad_doc in (doc, newer):
        with pytest.raises(ReplayAuditError):
            verify_animation(scenario, record, json.dumps(bad_doc))
    other_scenario, other_record = EPISODES["short"]
    with pytest.raises(ReplayAuditError):
        verify_animation(scenario, record, export_animation(other_scenario, other_record))
    with pytest.raises(ReplayAuditError):
        verify_replay_audit(other_scenario, record, audit)
