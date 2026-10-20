"""Deterministic text frames and a portable JSON animation export, built from a :class:`ReplayAudit`.

Everything here reads an audit that :func:`emberbench.audit.build_replay_audit`
produced from an authenticated record. Nothing here runs the simulator,
changes a record or feeds a score. Visualization never affects benchmark
results. The definitions are in ``docs/replay-visualization.md``.

Canonical text frames are plain ASCII with no colour codes. :func:`colorize`
adds ANSI colour for a human terminal only; its output is never canonical.
"""

from collections.abc import Mapping
from typing import Any

from .audit import ReplayAudit, _as_mapping, build_replay_audit
from .canonical import canonical_json, sha256_hex
from .errors import ReplayAuditError
from .events import RunRecord
from .grid import Pos
from .scenario import Scenario
from .state import PersonStatus

# Bump this whenever the layout or the meaning of any animation field changes.
ANIMATION_FORMAT_VERSION = "1.0"

GLYPH_WALL = "#"
GLYPH_CASUALTY = "X"  # event layer: a casualty in this step
GLYPH_EVACUATION = "V"  # event layer: an evacuation in this step
GLYPH_IGNITED = "f"  # event layer: a cell that ignited in this step
GLYPH_FIRE = "F"
GLYPH_PERSON = "P"  # an active person
GLYPH_EXIT = "E"  # an exit that is not burning
GLYPH_EMPTY = "."

# The canonical legend in precedence order: when several apply to one cell,
# the first one wins. The event glyphs appear only with ``events=True``.
# Without events a frame equals ``emberbench.state.render`` for the same state.
LEGEND: tuple[tuple[str, str], ...] = (
    (GLYPH_WALL, "wall"),
    (GLYPH_CASUALTY, "casualty in this step (event layer)"),
    (GLYPH_EVACUATION, "evacuation in this step (event layer)"),
    (GLYPH_IGNITED, "newly ignited in this step (event layer)"),
    (GLYPH_FIRE, "burning cell"),
    (GLYPH_PERSON, "active person"),
    (GLYPH_EXIT, "exit (not burning)"),
    (GLYPH_EMPTY, "empty cell"),
)


# --------------------------------------------------------------------------- text frames


def render_frame(audit: ReplayAudit, step: int, *, events: bool = False, header: bool = False) -> str:
    """The canonical text frame after ``step`` (0 is the initial state).

    Rows are joined with ``\\n`` and there is no trailing newline. With
    ``events=True`` the casualties, evacuations and ignitions of this step
    are drawn over the base layer. With ``header=True`` one summary line
    comes first.
    """
    frame = audit.frame(step)
    width, height = audit.map.width, audit.map.height
    grid = [[GLYPH_EMPTY] * width for _ in range(height)]

    # Paint from the lowest precedence to the highest, so later layers win.
    layers: list[tuple[str, list[Pos]]] = [
        (GLYPH_EXIT, [pos for _, pos in audit.map.exits]),
        (GLYPH_PERSON, [p.position for p in frame.people if p.status is PersonStatus.ACTIVE]),
        (GLYPH_FIRE, sorted(audit.burning_at(step))),
    ]
    if events:
        layers += [
            (GLYPH_IGNITED, list(frame.newly_ignited)),
            (GLYPH_EVACUATION, [e.position for e in frame.evacuations]),
            (GLYPH_CASUALTY, [c.position for c in frame.casualties]),
        ]
    layers.append((GLYPH_WALL, list(audit.map.walls)))
    for glyph, cells in layers:
        for pos in cells:
            grid[pos.row][pos.col] = glyph

    lines = ["".join(row) for row in grid]
    if header:
        lines.insert(0, frame_header(audit, step))
    return "\n".join(lines)


def frame_header(audit: ReplayAudit, step: int) -> str:
    """One summary line for ``step``, for example ``step 3/12 active=4 evacuated=1 casualties=0 burning=7``."""
    f = audit.frame(step)
    line = (
        f"step {f.step}/{audit.episode_steps} active={f.active_count} evacuated={f.evacuated_total}"
        f" casualties={f.casualty_total} burning={f.burning_count}"
    )
    if f.termination_reason is not None:
        line += f" terminated={f.termination_reason.value}"
    return line


def render_initial(audit: ReplayAudit, *, header: bool = False) -> str:
    return render_frame(audit, 0, header=header)


def render_final(audit: ReplayAudit, *, events: bool = False, header: bool = False) -> str:
    return render_frame(audit, audit.episode_steps, events=events, header=header)


def render_legend() -> str:
    """The legend as text, one ``glyph meaning`` line per entry in precedence order."""
    return "\n".join(f"{glyph} {meaning}" for glyph, meaning in LEGEND)


_ANSI: Mapping[str, str] = {
    GLYPH_WALL: "\x1b[90m",
    GLYPH_CASUALTY: "\x1b[1;35m",
    GLYPH_EVACUATION: "\x1b[1;32m",
    GLYPH_IGNITED: "\x1b[93m",
    GLYPH_FIRE: "\x1b[31m",
    GLYPH_PERSON: "\x1b[1;36m",
    GLYPH_EXIT: "\x1b[32m",
}
_ANSI_RESET = "\x1b[0m"


def colorize(frame_text: str) -> str:
    """Add ANSI colour to a grid frame for a human terminal. Never use the result as an artifact.

    A header line, if present, is left as it is.
    """
    lines = frame_text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i == 0 and line.startswith("step "):
            out.append(line)
            continue
        out.append("".join(f"{_ANSI[ch]}{ch}{_ANSI_RESET}" if ch in _ANSI else ch for ch in line))
    return "\n".join(out)


# --------------------------------------------------------------------------- animation export


def build_animation(audit: ReplayAudit) -> dict[str, Any]:
    """The provider-neutral animation document for ``audit``. Serialise it with :func:`animation_json`.

    The document holds data only: no markup, no script, no timestamps, no
    paths and no environment values. ``congestion`` keys are ``None`` unless
    the audit carries the crowd overlay.
    """
    frames = []
    for frame in audit.timeline:
        congestion = audit.step_congestion(frame.step)
        frames.append(
            {
                "step": frame.step,
                "people": [p.to_dict() for p in frame.people],
                "burning": [p.to_list() for p in sorted(audit.burning_at(frame.step))],
                "newly_ignited": [p.to_list() for p in frame.newly_ignited],
                "movements": [m.to_dict() for m in frame.movements],
                "evacuations": [e.to_dict() for e in frame.evacuations],
                "casualties": [c.to_dict() for c in frame.casualties],
                "congestion": None
                if audit.crowd is None
                else {
                    "blocked": [] if congestion is None else [b.to_dict() for b in congestion.blocked],
                    "contested_cells": [] if congestion is None else [c.to_dict() for c in congestion.contested_cells],
                },
                "terminated": frame.terminated,
                "termination_reason": None if frame.termination_reason is None else frame.termination_reason.value,
            }
        )
    return {
        "animation_format_version": ANIMATION_FORMAT_VERSION,
        "replay_audit_version": audit.replay_audit_version,
        "source": {
            "simulator_version": audit.simulator_version,
            "observation_version": audit.observation_version,
            "schema_version": audit.schema_version,
            "scenario_id": audit.scenario_id,
            "scenario_hash": audit.scenario_hash,
            "seed": audit.seed,
            "record_digest": audit.record_digest,
            "audit_digest": audit.digest(),
        },
        "map": audit.map.to_dict(),
        "people": [p.person_id for p in audit.people],
        "episode_steps": audit.episode_steps,
        "termination_reason": audit.termination_reason.value,
        "frames": frames,
        "congestion": None
        if audit.crowd is None
        else {
            "hotspots": [c.to_dict() for c in audit.crowd.hotspots],
            "exit_throughput": audit.crowd.exit_throughput.to_dict(),
        },
    }


def animation_json(audit: ReplayAudit) -> str:
    """Canonical JSON of the animation document: sorted keys, no whitespace, ASCII only."""
    return canonical_json(build_animation(audit))


def animation_digest(audit: ReplayAudit) -> str:
    return sha256_hex(build_animation(audit))


def export_animation(scenario: Scenario, record: RunRecord, *, crowd: bool = False) -> str:
    """Authenticate ``record``, build its audit and return the canonical animation JSON."""
    return animation_json(build_replay_audit(scenario, record, crowd=crowd))


def verify_animation(scenario: Scenario, record: RunRecord, animation: str | Mapping[str, Any]) -> None:
    """Rebuild the animation from the authenticated record and require byte-identical canonical JSON."""
    data = _as_mapping(animation, "animation export")
    version = data.get("animation_format_version")
    if version != ANIMATION_FORMAT_VERSION:
        raise ReplayAuditError(f"animation format version {version!r} is not supported; this is {ANIMATION_FORMAT_VERSION!r}")
    expected = export_animation(scenario, record, crowd=data.get("congestion") is not None)
    try:
        given = canonical_json(data)
    except (TypeError, ValueError) as exc:
        raise ReplayAuditError(f"animation export is not canonical JSON data: {exc}") from exc
    if given != expected:
        raise ReplayAuditError("animation export does not match the authenticated record")
