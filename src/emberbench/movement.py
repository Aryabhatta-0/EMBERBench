"""Action parsing, movement validation and collision resolution.

These are simulator phases 2-4. The rules are listed in order. The full
contract is in ``docs/simulation-contract.md``.

1. **Action set.** The submitted mapping must have exactly one action for
   every active person, and nothing else. A missing ID, an unknown or
   inactive ID, or a value that is not an :class:`Action` raises
   :class:`InvalidActionError`. Nothing is mutated before this check passes.
2. **Validation.** ``STAY`` gives ``STAYED``. A move off the grid gives
   ``BLOCKED_BOUNDARY``. A move into an obstacle gives ``BLOCKED_WALL``. A
   move into a burning cell gives ``BLOCKED_FIRE`` when
   ``movement.enter_fire`` is ``blocked``. Under ``casualty`` the move is
   allowed, and the person becomes a casualty in phase 8.
3. **Occupied by a non-mover.** If the target cell holds a person who does
   not try to move (``STAY`` or blocked in step 2), every mover into that
   cell gets ``BLOCKED_OCCUPIED``.
4. **Conflict.** If several movers target the same cell, the mover with the
   lowest person ID wins. IDs compare as plain strings, by code point, so
   ``"P10" < "P2"``. All other movers get ``BLOCKED_CONFLICT``. This decision
   uses intent only and is never revisited. A loser stays blocked even when
   the winner fails later in rule 5.
5. **Chains and cycles.** A winner moves into an empty cell, or into a cell
   whose occupant moves out successfully. Following another person in the
   same step is allowed. If the chain ends at a person who stays, the move
   fails with ``BLOCKED_OCCUPIED``. A closed chain (a swap ``A<->B`` or any
   longer rotation) is never allowed: every member gets ``BLOCKED_CYCLE``.
   An outside mover that targets a cycle cell always meets a cycle member in
   rule 4. If the outsider has the lower ID, it wins, and that cycle member
   becomes a non-mover. The rotation is then broken, and the chain resolves
   as ``BLOCKED_OCCUPIED``.

A blocked person stays in its cell. The result does not depend on the
insertion order of the action mapping.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import InvalidActionError
from .grid import Action, Pos, in_bounds
from .scenario import EnterFirePolicy


class MoveOutcome(StrEnum):
    STAYED = "stayed"
    MOVED = "moved"
    BLOCKED_BOUNDARY = "blocked_boundary"
    BLOCKED_WALL = "blocked_wall"
    BLOCKED_FIRE = "blocked_fire"
    BLOCKED_OCCUPIED = "blocked_occupied"
    BLOCKED_CONFLICT = "blocked_conflict"
    BLOCKED_CYCLE = "blocked_cycle"


@dataclass(frozen=True)
class MoveResolution:
    person_id: str
    action: Action
    origin: Pos
    target: Pos  # intended cell (equal to origin for STAY)
    final: Pos
    outcome: MoveOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "action": self.action.value,
            "origin": self.origin.to_list(),
            "target": self.target.to_list(),
            "final": self.final.to_list(),
            "outcome": self.outcome.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MoveResolution":
        return cls(
            person_id=d["person_id"],
            action=Action(d["action"]),
            origin=Pos(*d["origin"]),
            target=Pos(*d["target"]),
            final=Pos(*d["final"]),
            outcome=MoveOutcome(d["outcome"]),
        )


def parse_actions(actions: Mapping[Any, Any], active_ids: frozenset[str]) -> dict[str, Action]:
    """Validate a submitted action mapping (phase 2). Raises :class:`InvalidActionError`."""
    if not isinstance(actions, Mapping):
        raise InvalidActionError(f"actions must be a mapping of person id -> action, got {type(actions).__name__}")
    bad_keys = sorted(repr(k) for k in actions if not isinstance(k, str))
    if bad_keys:
        raise InvalidActionError(f"action keys must be person id strings, got {bad_keys}")
    submitted = frozenset(actions)
    missing = sorted(active_ids - submitted)
    unknown = sorted(submitted - active_ids)
    if missing or unknown:
        raise InvalidActionError(f"action set must cover exactly the active people; missing={missing} unexpected={unknown}")
    parsed: dict[str, Action] = {}
    for pid in sorted(submitted):
        raw = actions[pid]
        if isinstance(raw, Action):
            parsed[pid] = raw
        elif isinstance(raw, str) and raw in Action._value2member_map_:
            parsed[pid] = Action(raw)
        else:
            allowed = [a.value for a in Action]
            raise InvalidActionError(f"invalid action {raw!r} for person {pid!r}; allowed: {allowed}")
    return parsed


def resolve_movements(
    positions: Mapping[str, Pos],
    actions: Mapping[str, Action],
    *,
    width: int,
    height: int,
    walls: frozenset[Pos],
    burning: frozenset[Pos],
    enter_fire: EnterFirePolicy,
) -> tuple[MoveResolution, ...]:
    """Resolve one timestep of movement (phases 3-4). The result is sorted by person ID."""
    ids = sorted(positions)
    occupant: dict[Pos, str] = {positions[pid]: pid for pid in ids}
    target: dict[str, Pos] = {}
    outcome: dict[str, MoveOutcome] = {}
    intent: dict[str, Pos] = {}

    # Rule 2: per-person validation.
    for pid in ids:
        action = actions[pid]
        origin = positions[pid]
        dest = origin.offset(*action.delta)
        target[pid] = dest
        if action is Action.STAY:
            outcome[pid] = MoveOutcome.STAYED
        elif not in_bounds(dest, width, height):
            outcome[pid] = MoveOutcome.BLOCKED_BOUNDARY
        elif dest in walls:
            outcome[pid] = MoveOutcome.BLOCKED_WALL
        elif dest in burning and enter_fire is EnterFirePolicy.BLOCKED:
            outcome[pid] = MoveOutcome.BLOCKED_FIRE
        else:
            intent[pid] = dest

    # Rule 3: target occupied by someone who is not trying to move.
    for pid in sorted(intent):
        holder = occupant.get(intent[pid])
        if holder is not None and holder not in intent:
            outcome[pid] = MoveOutcome.BLOCKED_OCCUPIED
    intent = {pid: dest for pid, dest in intent.items() if pid not in outcome}

    # Rule 4: lowest ID wins each contested cell.
    contenders: dict[Pos, list[str]] = {}
    for pid in sorted(intent):
        contenders.setdefault(intent[pid], []).append(pid)
    for _cell, group in sorted(contenders.items()):
        for loser in group[1:]:
            outcome[loser] = MoveOutcome.BLOCKED_CONFLICT
    intent = {pid: dest for pid, dest in intent.items() if pid not in outcome}

    # Rule 5: follow chains of movers; cycles are blocked.
    success: dict[str, bool] = {}
    in_cycle: set[str] = set()
    for start in sorted(intent):
        if start in success:
            continue
        path: list[str] = []
        index: dict[str, int] = {}
        current = start
        while True:
            if current in success:
                verdict = success[current]
                break
            if current in index:
                for member in path[index[current] :]:
                    in_cycle.add(member)
                    success[member] = False
                verdict = False
                break
            index[current] = len(path)
            path.append(current)
            holder = occupant.get(intent[current])
            if holder is None:
                verdict = True
                break
            if holder not in intent:
                verdict = False
                break
            current = holder
        for member in path:
            success.setdefault(member, verdict)

    for pid in intent:
        if success[pid]:
            outcome[pid] = MoveOutcome.MOVED
        elif pid in in_cycle:
            outcome[pid] = MoveOutcome.BLOCKED_CYCLE
        else:
            outcome[pid] = MoveOutcome.BLOCKED_OCCUPIED

    return tuple(
        MoveResolution(
            person_id=pid,
            action=actions[pid],
            origin=positions[pid],
            target=target[pid],
            final=target[pid] if outcome[pid] is MoveOutcome.MOVED else positions[pid],
            outcome=outcome[pid],
        )
        for pid in ids
    )
