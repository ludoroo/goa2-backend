# Consensus Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a table of players vote to patch wrong values, unstick wedged control flow, or rewind the game to an earlier decision — all recorded as replay decisions applied through one shared code path.

**Architecture:** A closed op registry in `engine/overrides.py` is the single mutation path for overrides; `server/replay.py` gains three decision types (`ov_patch`, `ov_unstick`, `ov_rewind`) applied through that registry in both live play and reconstruction. A consensus protocol (proposals + majority votes over WebSocket) lives on `ManagedGame`, never in `GameState`. Two REST endpoints expose the op catalogue and a player-scoped decision history.

**Tech Stack:** Python 3.11, Pydantic V2, FastAPI (REST + WebSocket), pytest.

**Spec:** `docs/superpowers/specs/2026-08-05-consensus-overrides-design.md` — read it before starting any task.

## Global Constraints

- **Single mutation path:** Nothing outside `engine/overrides.py` op functions may mutate `GameState` for an override. Replay `_apply_decision` and the live WS apply path both call `apply_override_decision()`.
- **Append-only log:** `ov_rewind` never truncates the replay file; reconstruction resolves it by rebuilding from the seed.
- **Atomicity:** An op that produces an invalid state commits nothing — the whole override is rejected.
- **Closed whitelist:** No generic JSON-patch, no direct `execution_stack` editing, no arbitrary field paths. `starting_life_counters` is immutable under override.
- **Positional ops** go through `state.get_position()` / `get_piece_ids()` / `place_entity()` — never touch `entity_locations` directly (multi-piece heroes: `hero_razzle` has no board position; only its pieces do).
- **Visibility:** The history endpoint must never reveal an opponent's facedown/hand card identity. Reuse `server/visibility.py` helpers; the omniscient `reveal_all` view must not be reachable from it.
- **Client contract:** New WS message types and REST endpoints must be documented in `docs/CLIENT_INTEGRATION_GUIDE.md` (Task 10).
- **Bump request ids after any patch** (deliberate departure from the persistence convention of reusing them) so a stale in-flight answer is rejected.
- **No AI attribution** in commits (user global instruction): plain commit messages, no Co-Authored-By trailers.
- Commands run from repo root (`/Users/pedrooliveira/Documents/goa2/goa2-spec` worktree, branch `spec-consensus-overrides`): `PYTHONPATH=src uv run pytest <path> -q`.
- Test files must have unique basenames across `tests/` subdirs (no `__init__.py`).

## File Structure

| File | Responsibility |
|---|---|
| `src/goa2/engine/overrides.py` (new) | Op registry (`OverrideOp`, `OVERRIDE_OPS`), Pydantic arg models, `apply_override_decision()`, atomicity, summaries |
| `src/goa2/server/replay.py` (modify) | `record_override()`, `_apply_decision` branches, `effective_decisions()` / `effective_indices()`, rewind-aware `ReplayCursor.seek` / `replay_game`, `rebuild_session_for_rewind()` |
| `src/goa2/server/overrides.py` (new) | `OverrideProposal`, consensus rules (eligibility, threshold, votes, expiry), WS message payload builders |
| `src/goa2/server/registry.py` (modify) | `ManagedGame.pending_override`, `ManagedGame.override_expiry_task` |
| `src/goa2/server/time_control.py` (modify) | Clock pause while a proposal is open (`reconcile_game_clock`) |
| `src/goa2/server/ws.py` (modify) | `PROPOSE_OVERRIDE` / `VOTE_OVERRIDE` / `CANCEL_OVERRIDE` handlers, apply-on-approval, `OVERRIDE_*` broadcasts |
| `src/goa2/server/routes_overrides.py` (new) | `GET /overrides/schema`, `GET /games/{game_id}/overrides/history` |
| `src/goa2/server/models.py` (modify) | `OverrideOpSchema`, `OverrideSchemaResponse`, `OverrideHistoryEntry`, `OverrideHistoryResponse` |
| `src/goa2/server/app.py` (modify) | Register `routes_overrides.router` |
| `docs/CLIENT_INTEGRATION_GUIDE.md` (modify) | Document messages + endpoints |

Tests: `tests/engine/test_override_ops.py`, `tests/engine/test_override_unstick.py`, `tests/server/test_override_replay.py`, `tests/server/test_override_consensus.py`, `tests/server/test_override_endpoints.py`.

---

### Task 1: Op registry core + board patch ops

**Files:**
- Create: `src/goa2/engine/overrides.py`
- Test: `tests/engine/test_override_ops.py`

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `class OverrideRejectedError(Exception)` with `.code: str` and `.message: str`
  - `@dataclass(frozen=True) class OverrideOp: name, family, label, description, args_model, apply`
  - `OVERRIDE_OPS: dict[str, OverrideOp]`
  - `get_op(name: str) -> OverrideOp` (raises `OverrideRejectedError(code="unknown_op")`)
  - `apply_override_decision(session: GameSession, op_name: str, args: dict) -> SessionResult | None`
  - `summarize_op(op_name: str, args: dict) -> str`

- [ ] **Step 1: Write failing tests for the registry core and board ops**

```python
# tests/engine/test_override_ops.py
"""Override op registry: patch ops, atomicity, multi-piece conventions."""

import pytest

from goa2.domain.hex import Hex
from goa2.domain.models import GamePhase
from goa2.engine.overrides import (
    OVERRIDE_OPS,
    OverrideRejectedError,
    apply_override_decision,
    get_op,
    summarize_op,
)
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup


MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def session() -> GameSession:
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=42)
    return GameSession(state)


def _hex_dict(h: Hex) -> dict:
    return {"q": h.q, "r": h.r, "s": h.s}


def _free_adjacent(state, entity_id: str) -> Hex:
    pos = state.get_position(entity_id)
    for n in pos.neighbors():
        tile = state.board.tiles.get(n)
        if tile is not None and tile.occupant_id is None:
            return n
    raise AssertionError("no free adjacent hex")


def test_registry_contains_all_patch_and_unstick_ops():
    expected = {
        "move_entity", "remove_entity", "place_entity",
        "set_life_counters", "set_gold", "set_level",
        "add_marker", "remove_marker", "add_effect", "remove_effect",
        "move_card", "set_wave_counter", "set_tie_breaker_team",
        "skip_input", "abort_action", "end_turn", "force_actor",
    }
    assert expected <= set(OVERRIDE_OPS)
    for op in OVERRIDE_OPS.values():
        assert op.family in ("patch", "unstick")
        assert op.label and op.description


def test_unknown_op_rejected(session):
    with pytest.raises(OverrideRejectedError) as exc:
        apply_override_decision(session, "teleport_everything", {})
    assert exc.value.code == "unknown_op"


def test_move_entity_moves_a_hero(session):
    state = session.state
    target = _free_adjacent(state, "hero_arien")
    apply_override_decision(
        session, "move_entity", {"entity_id": "hero_arien", "hex": _hex_dict(target)}
    )
    assert session.state.get_position("hero_arien") == target
    # occupancy cache rebuilt
    assert str(session.state.board.tiles[target].occupant_id) == "hero_arien"


def test_move_entity_to_occupied_hex_rejected_and_commits_nothing(session):
    state = session.state
    arien_pos = state.get_position("hero_arien")
    wasp_pos = state.get_position("hero_wasp")
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "move_entity", {"entity_id": "hero_arien", "hex": _hex_dict(wasp_pos)}
        )
    assert session.state.get_position("hero_arien") == arien_pos
    assert session.state.get_position("hero_wasp") == wasp_pos


def test_move_entity_off_map_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "move_entity",
            {"entity_id": "hero_arien", "hex": {"q": 99, "r": -99, "s": 0}},
        )


def test_remove_then_place_entity_round_trips(session):
    state = session.state
    pos = state.get_position("hero_wasp")
    apply_override_decision(session, "remove_entity", {"entity_id": "hero_wasp"})
    assert session.state.get_position("hero_wasp") is None
    apply_override_decision(
        session, "place_entity", {"entity_id": "hero_wasp", "hex": _hex_dict(pos)}
    )
    assert session.state.get_position("hero_wasp") == pos


def test_place_unknown_entity_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "place_entity", {"entity_id": "minion_999", "hex": {"q": 0, "r": 0, "s": 0}}
        )


def test_move_entity_multi_piece_hero_requires_piece_id():
    state = GameSetup.create_game(MAP_PATH, ["Razzle"], ["Wasp"], False, "QUICK", seed=7)
    session = GameSession(state)
    pieces = state.get_piece_ids("hero_razzle")
    if len(pieces) > 1:
        with pytest.raises(OverrideRejectedError) as exc:
            apply_override_decision(
                session, "move_entity",
                {"entity_id": "hero_razzle", "hex": {"q": 0, "r": 0, "s": 0}},
            )
        assert "piece" in exc.value.message.lower()
    # Moving an explicit piece works
    piece = pieces[0]
    target = _free_adjacent(state, piece)
    apply_override_decision(
        session, "move_entity", {"entity_id": piece, "hex": _hex_dict(target)}
    )
    assert session.state.get_position(piece) == target


def test_summarize_op_is_human_readable():
    text = summarize_op("move_entity", {"entity_id": "minion_4", "hex": {"q": 1, "r": -2, "s": 1}})
    assert "minion_4" in text
```

Note on `Razzle`: if `GameSetup.create_game` does not accept that hero name in this map/config, adapt the fixture to whichever multi-piece setup `tests/engine/test_multipiece_conventions.py` uses — reuse its fixture pattern, do not skip the test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: FAIL with `ModuleNotFoundError: goa2.engine.overrides`

- [ ] **Step 3: Implement the registry core and board ops**

```python
# src/goa2/engine/overrides.py
"""Consensus-override op registry — the single mutation path for overrides.

Every override (live or replayed) goes through ``apply_override_decision``:
validate args -> snapshot -> apply -> revalidate the whole GameState (which
rebuilds the occupancy cache and re-unifies card/token references) -> on any
failure restore the snapshot and reject. Nothing else may mutate state for an
override; replay parity depends on this being the one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from goa2.domain.hex import Hex
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID
from goa2.engine.session import GameSession, SessionResult


class OverrideRejectedError(Exception):
    """An override op could not be applied. ``code`` is machine-readable."""

    def __init__(self, message: str, *, code: str = "invalid_op") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class HexArg(BaseModel):
    q: int
    r: int
    s: int

    def to_hex(self) -> Hex:
        return Hex(q=self.q, r=self.r, s=self.s)


@dataclass(frozen=True)
class OverrideOp:
    name: str
    family: Literal["patch", "unstick"]
    label: str
    description: str
    args_model: type[BaseModel]
    apply: Callable[[GameSession, Any], None]
    summary_template: str  # .format(**args_dict)


OVERRIDE_OPS: dict[str, OverrideOp] = {}


def _register(op: OverrideOp) -> None:
    if op.name in OVERRIDE_OPS:
        raise ValueError(f"Duplicate override op {op.name!r}")
    OVERRIDE_OPS[op.name] = op


def get_op(name: str) -> OverrideOp:
    op = OVERRIDE_OPS.get(name)
    if op is None:
        raise OverrideRejectedError(f"Unknown override op {name!r}", code="unknown_op")
    return op


def summarize_op(op_name: str, args: dict[str, Any]) -> str:
    op = get_op(op_name)
    try:
        return op.summary_template.format(**args)
    except (KeyError, IndexError):
        return f"{op.label}: {args}"


def apply_override_decision(
    session: GameSession, op_name: str, args: dict[str, Any]
) -> SessionResult | None:
    """Validate, apply, and re-derive. The one code path for live + replay.

    Returns the SessionResult of the post-apply re-derivation (a fresh input
    request if a step was pending), or None during PLANNING where advance()
    is not allowed and clients rely on the broadcast view.
    """
    op = get_op(op_name)
    try:
        parsed = op.args_model.model_validate(args)
    except Exception as exc:  # pydantic.ValidationError
        raise OverrideRejectedError(str(exc), code="invalid_args") from exc

    baseline = session.state.model_dump(mode="json")
    try:
        op.apply(session, parsed)
        # Full revalidation: rebuild_occupancy_cache + unify_card_references +
        # unify_token_references run as model validators, so an op that broke
        # an invariant raises here and the whole override rolls back.
        session.state = GameState.model_validate(session.state.model_dump(mode="json"))
    except OverrideRejectedError:
        session.state = GameState.model_validate(baseline)
        raise
    except Exception as exc:
        session.state = GameState.model_validate(baseline)
        raise OverrideRejectedError(str(exc), code="invalid_result") from exc

    # Bump the pending request id so a stale in-flight answer (submitted
    # against the pre-patch board) is rejected by the normal request-id
    # mismatch check. Deliberate departure from the persistence convention
    # of preserving ids across re-derivation.
    if session.state.execution_stack:
        top = session.state.execution_stack[-1]
        if top.pending_request_id is not None:
            top.pending_request_id = None

    if session.state.phase == GamePhase.PLANNING:
        return None
    # Re-derive: re-runs the pending step's resolve() so filters and option
    # lists are recomputed against the patched board.
    return session.advance(None)


# ---------------------------------------------------------------------------
# Board patch ops
# ---------------------------------------------------------------------------


def _resolve_board_entity_id(state: GameState, entity_id: str) -> BoardEntityID:
    """Map an entity arg to a concrete on-board id, honoring multi-piece rules."""
    if state._multi_piece_hero(entity_id) is not None:
        pieces = state.get_piece_ids(entity_id)
        if len(pieces) == 1:
            return BoardEntityID(pieces[0])
        raise OverrideRejectedError(
            f"{entity_id} is a multi-piece hero; specify one of its piece ids "
            f"({pieces or 'no pieces on board'})",
            code="ambiguous_entity",
        )
    return BoardEntityID(str(entity_id))


class MoveEntityArgs(BaseModel):
    entity_id: str
    hex: HexArg


def _apply_move_entity(session: GameSession, args: MoveEntityArgs) -> None:
    state = session.state
    board_id = _resolve_board_entity_id(state, args.entity_id)
    if state.get_position(str(board_id)) is None:
        raise OverrideRejectedError(
            f"{args.entity_id} is not on the board (use place_entity)", code="not_on_board"
        )
    state.place_entity(board_id, args.hex.to_hex())  # raises on occupied/off-map


_register(
    OverrideOp(
        name="move_entity",
        family="patch",
        label="Move entity",
        description="Move a unit, hero piece, or token to a hex (fixes a refused legal move).",
        args_model=MoveEntityArgs,
        apply=_apply_move_entity,
        summary_template="Move {entity_id} to {hex}",
    )
)


class RemoveEntityArgs(BaseModel):
    entity_id: str


def _apply_remove_entity(session: GameSession, args: RemoveEntityArgs) -> None:
    state = session.state
    board_id = _resolve_board_entity_id(state, args.entity_id)
    if BoardEntityID(str(board_id)) not in state.entity_locations:
        raise OverrideRejectedError(f"{args.entity_id} is not on the board", code="not_on_board")
    state.remove_entity(board_id)


_register(
    OverrideOp(
        name="remove_entity",
        family="patch",
        label="Remove entity from board",
        description="Remove a unit that should have been defeated.",
        args_model=RemoveEntityArgs,
        apply=_apply_remove_entity,
        summary_template="Remove {entity_id} from the board",
    )
)


class PlaceEntityArgs(BaseModel):
    entity_id: str
    hex: HexArg


def _apply_place_entity(session: GameSession, args: PlaceEntityArgs) -> None:
    state = session.state
    if state.get_entity(BoardEntityID(args.entity_id)) is None and state.get_hero(
        HeroID(args.entity_id)
    ) is None:
        raise OverrideRejectedError(
            f"Unknown entity {args.entity_id!r}", code="unknown_entity"
        )
    board_id = _resolve_board_entity_id(state, args.entity_id)
    state.place_entity(board_id, args.hex.to_hex())


_register(
    OverrideOp(
        name="place_entity",
        family="patch",
        label="Place entity on board",
        description="Put a wrongly defeated unit back, or fix a bad respawn hex.",
        args_model=PlaceEntityArgs,
        apply=_apply_place_entity,
        summary_template="Place {entity_id} at {hex}",
    )
)
```

Note: `_resolve_board_entity_id` uses `state._multi_piece_hero` — it is the same private helper `get_piece_ids` uses internally. If `mypy`/review objects, add a thin public wrapper `state.is_multi_piece_hero(entity_id) -> bool` in `domain/state.py` instead of reaching for the underscore; either is acceptable, but pick one and keep it in this task.

- [ ] **Step 4: Run tests (board-op subset) to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: the board-op tests PASS; `test_registry_contains_all_patch_and_unstick_ops` still FAILS (remaining ops arrive in Tasks 2–4). That is the expected state at the end of this task — note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add src/goa2/engine/overrides.py tests/engine/test_override_ops.py
git commit -m "Add override op registry with board patch ops"
```

---

### Task 2: Resource and counter patch ops (incl. life-counter endgame)

**Files:**
- Modify: `src/goa2/engine/overrides.py`
- Test: `tests/engine/test_override_ops.py` (append)

**Interfaces:**
- Consumes: registry from Task 1 (`_register`, `OverrideOp`, `OverrideRejectedError`).
- Produces ops: `set_life_counters`, `set_gold`, `set_level`, `set_wave_counter`, `set_tie_breaker_team`.

- [ ] **Step 1: Write failing tests**

Append to `tests/engine/test_override_ops.py`:

```python
from goa2.domain.models import TeamColor


def test_set_gold_and_level(session):
    apply_override_decision(session, "set_gold", {"hero_id": "hero_arien", "value": 7})
    assert session.state.get_hero("hero_arien").gold == 7
    apply_override_decision(session, "set_level", {"hero_id": "hero_arien", "value": 3})
    assert session.state.get_hero("hero_arien").level == 3


def test_set_gold_unknown_hero_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_gold", {"hero_id": "hero_nobody", "value": 7})


def test_set_wave_counter(session):
    lane_id = next(iter(session.state.wave_counters))
    apply_override_decision(session, "set_wave_counter", {"lane_id": lane_id, "value": 3})
    assert session.state.wave_counters[lane_id] == 3


def test_set_wave_counter_unknown_lane_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_wave_counter", {"lane_id": "lane_zz", "value": 3})


def test_set_tie_breaker_team(session):
    apply_override_decision(session, "set_tie_breaker_team", {"team": "BLUE"})
    assert session.state.tie_breaker_team == TeamColor.BLUE


def test_set_life_counters_to_zero_ends_game(session):
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 0})
    state = session.state
    assert state.phase == GamePhase.GAME_OVER
    assert state.teams[TeamColor.BLUE].life_counters == 0
    assert state.winner == TeamColor.RED
    assert state.victory_condition is not None


def test_set_life_counters_resurrects_finished_game(session):
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 0})
    assert session.state.phase == GamePhase.GAME_OVER
    # Raise back above 0: the ONLY patch that un-ends a game.
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 2})
    state = session.state
    assert state.phase != GamePhase.GAME_OVER
    assert state.winner is None
    assert state.individual_winner_id is None
    assert state.victory_condition is None
    assert state.teams[TeamColor.BLUE].life_counters == 2


def test_set_life_counters_negative_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": -1})


def test_starting_life_counters_untouched(session):
    before = session.state.teams[TeamColor.BLUE].starting_life_counters
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 1})
    assert session.state.teams[TeamColor.BLUE].starting_life_counters == before
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: new tests FAIL with `OverrideRejectedError: Unknown override op 'set_gold'` etc.

- [ ] **Step 3: Implement the ops**

Append to `src/goa2/engine/overrides.py`:

```python
# ---------------------------------------------------------------------------
# Resource / counter patch ops
# ---------------------------------------------------------------------------


def _require_hero(state: GameState, hero_id: str):
    hero = state.get_hero(HeroID(hero_id))
    if hero is None:
        raise OverrideRejectedError(f"Unknown hero {hero_id!r}", code="unknown_hero")
    return hero


class SetGoldArgs(BaseModel):
    hero_id: str
    value: int = Field(ge=0)


def _apply_set_gold(session: GameSession, args: SetGoldArgs) -> None:
    _require_hero(session.state, args.hero_id).gold = args.value


_register(
    OverrideOp(
        name="set_gold",
        family="patch",
        label="Set gold",
        description="Set a hero's gold to an exact value (fixes a miscredited bounty).",
        args_model=SetGoldArgs,
        apply=_apply_set_gold,
        summary_template="Set {hero_id} gold to {value}",
    )
)


class SetLevelArgs(BaseModel):
    hero_id: str
    value: int = Field(ge=1, le=8)


def _apply_set_level(session: GameSession, args: SetLevelArgs) -> None:
    _require_hero(session.state, args.hero_id).level = args.value


_register(
    OverrideOp(
        name="set_level",
        family="patch",
        label="Set level",
        description="Set a hero's level to an exact value.",
        args_model=SetLevelArgs,
        apply=_apply_set_level,
        summary_template="Set {hero_id} level to {value}",
    )
)


class SetWaveCounterArgs(BaseModel):
    lane_id: str
    value: int = Field(ge=0)


def _apply_set_wave_counter(session: GameSession, args: SetWaveCounterArgs) -> None:
    state = session.state
    if args.lane_id not in state.wave_counters:
        raise OverrideRejectedError(
            f"Unknown lane {args.lane_id!r} (lanes: {sorted(state.wave_counters)})",
            code="unknown_lane",
        )
    state.wave_counters[args.lane_id] = args.value


_register(
    OverrideOp(
        name="set_wave_counter",
        family="patch",
        label="Set wave counter",
        description="Set a lane's wave counter (fixes a wrongly scored lane push).",
        args_model=SetWaveCounterArgs,
        apply=_apply_set_wave_counter,
        summary_template="Set wave counter of {lane_id} to {value}",
    )
)


class SetTieBreakerArgs(BaseModel):
    team: TeamColor


def _apply_set_tie_breaker(session: GameSession, args: SetTieBreakerArgs) -> None:
    session.state.tie_breaker_team = args.team


_register(
    OverrideOp(
        name="set_tie_breaker_team",
        family="patch",
        label="Set tie-breaker / coin face",
        description="Set the tie-breaker team (also Ignatia's coin face).",
        args_model=SetTieBreakerArgs,
        apply=_apply_set_tie_breaker,
        summary_template="Set tie-breaker team to {team}",
    )
)


class SetLifeCountersArgs(BaseModel):
    team: TeamColor
    value: int = Field(ge=0)


def _apply_set_life_counters(session: GameSession, args: SetLifeCountersArgs) -> None:
    state = session.state
    team = state.teams.get(args.team)
    if team is None:
        raise OverrideRejectedError(f"Unknown team {args.team}", code="unknown_team")
    was_finished = state.phase == GamePhase.GAME_OVER
    old_value = team.life_counters
    team.life_counters = args.value
    # starting_life_counters is setup data — never touched by override.

    if args.value <= 0 and not was_finished:
        # Re-run the endgame check: dropping a team to 0 must not leave a
        # state where a team is dead but ``winner`` is unset.
        from goa2.engine.steps.combat import TriggerGameOverStep

        other = TeamColor.BLUE if args.team == TeamColor.RED else TeamColor.RED
        state.execution_stack.append(
            TriggerGameOverStep(winner=other, condition="override_life_counters")
        )
        return

    if was_finished and old_value <= 0 and args.value > 0:
        # The one patch that resurrects a finished game. process_stack()
        # returns immediately on GAME_OVER, so the phase must move off it
        # or the game stays frozen regardless of the counter.
        from goa2.engine.steps.phases import FinalizeHeroTurnStep, FindNextActorStep

        state.winner = None
        state.individual_winner_id = None
        state.victory_condition = None
        state.phase = GamePhase.RESOLUTION
        # TriggerGameOverStep purged the stack; resume through the normal
        # turn machinery rather than inventing state.
        if state.current_actor_id is not None:
            state.execution_stack.append(
                FinalizeHeroTurnStep(hero_id=str(state.current_actor_id))
            )
        else:
            state.execution_stack.append(FindNextActorStep())


_register(
    OverrideOp(
        name="set_life_counters",
        family="patch",
        label="Set life counters",
        description=(
            "Set a team's life counters. 0 ends the game; raising a finished "
            "game's losing team above 0 resurrects the game."
        ),
        args_model=SetLifeCountersArgs,
        apply=_apply_set_life_counters,
        summary_template="Set {team} life counters to {value}",
    )
)
```

Implementation notes:
- The `TriggerGameOverStep` / resurrection steps are *pushed*, and the shared `apply_override_decision` epilogue (`session.advance(None)` outside PLANNING) processes them. If the test fixture starts in PLANNING (fresh game does), the pushed `TriggerGameOverStep` will not run — so in `_apply_set_life_counters`, when `state.phase == GamePhase.PLANNING` and value <= 0, resolve it inline instead: call `state.execution_stack.append(...)` then `from goa2.engine.handler import process_stack; process_stack(state)`. Simplest uniform fix: after pushing, if phase is PLANNING run `process_stack(state)` directly inside the op (advance() forbids PLANNING). Same for the resurrection branch (which by construction is never in PLANNING — GAME_OVER — so only the game-over branch needs it).
- `TriggerGameOverStep` requires exactly one of team/individual winner — team form used here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: all Task 1 + Task 2 tests PASS (`test_registry_contains_all_...` still fails until Task 4).

- [ ] **Step 5: Commit**

```bash
git add src/goa2/engine/overrides.py tests/engine/test_override_ops.py
git commit -m "Add resource and life-counter override ops with endgame handling"
```

---

### Task 3: Card, marker, and effect patch ops

**Files:**
- Modify: `src/goa2/engine/overrides.py`
- Test: `tests/engine/test_override_ops.py` (append)

**Interfaces:**
- Produces ops: `move_card`, `add_marker`, `remove_marker`, `add_effect`, `remove_effect`.

- [ ] **Step 1: Write failing tests**

Append to `tests/engine/test_override_ops.py`:

```python
from goa2.domain.models.marker import MarkerType


def test_move_card_hand_to_discard_and_back(session):
    hero = session.state.get_hero("hero_arien")
    card = hero.hand[0]
    apply_override_decision(
        session, "move_card",
        {"hero_id": "hero_arien", "card_id": card.id, "zone": "discard"},
    )
    hero = session.state.get_hero("hero_arien")
    assert card.id in [c.id for c in hero.discard_pile]
    assert card.id not in [c.id for c in hero.hand]

    apply_override_decision(
        session, "move_card",
        {"hero_id": "hero_arien", "card_id": card.id, "zone": "hand"},
    )
    hero = session.state.get_hero("hero_arien")
    assert card.id in [c.id for c in hero.hand]
    assert card.id not in [c.id for c in hero.discard_pile]


def test_move_card_unknown_card_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "move_card",
            {"hero_id": "hero_arien", "card_id": "not_a_card", "zone": "hand"},
        )


def test_add_and_remove_marker(session):
    apply_override_decision(
        session, "add_marker",
        {"marker_type": "venom", "target_id": "hero_wasp", "value": -1,
         "source_id": "hero_arien"},
    )
    markers = session.state.get_markers_on_hero("hero_wasp")
    assert any(m.type == MarkerType.VENOM for m in markers)

    apply_override_decision(session, "remove_marker", {"marker_type": "venom"})
    assert not session.state.get_markers_on_hero("hero_wasp")


def test_remove_effect(session):
    from goa2.domain.models.effect import ActiveEffect, EffectScope, EffectType

    session.state.add_effect(
        ActiveEffect(
            id="fx_test_1",
            source_id="hero_arien",
            effect_type=EffectType.STAT_MODIFIER,
            scope=EffectScope(origin_id="hero_arien"),
        )
    )
    apply_override_decision(session, "remove_effect", {"effect_id": "fx_test_1"})
    assert all(e.id != "fx_test_1" for e in session.state.active_effects)


def test_add_effect_from_payload(session):
    from goa2.domain.models.effect import EffectType

    payload = {
        "id": "fx_test_2",
        "source_id": "hero_arien",
        "effect_type": EffectType.STAT_MODIFIER.value,
        "scope": {"origin_id": "hero_arien"},
    }
    apply_override_decision(session, "add_effect", {"effect": payload})
    assert any(e.id == "fx_test_2" for e in session.state.active_effects)


def test_add_effect_invalid_payload_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "add_effect", {"effect": {"id": "x"}})
```

Adjust the `ActiveEffect` / `EffectScope` constructor fields to the real required fields in `src/goa2/domain/models/effect.py` (read that file first — `EffectScope` may require a `scope_type`/`radius`; `EffectType.STAT_MODIFIER` may be named differently, e.g. pick any real enum member). The test's point is round-tripping an effect through the op, not a specific effect type.

- [ ] **Step 2: Run to verify new tests fail**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: new tests FAIL with unknown-op errors.

- [ ] **Step 3: Implement the ops**

Append to `src/goa2/engine/overrides.py`:

```python
# ---------------------------------------------------------------------------
# Card / marker / effect patch ops
# ---------------------------------------------------------------------------

from goa2.domain.models.effect import ActiveEffect  # top of file with other imports
from goa2.domain.models.marker import MarkerType  # top of file with other imports


class MoveCardArgs(BaseModel):
    hero_id: str
    card_id: str
    zone: Literal["hand", "discard", "played"]


def _detach_card(hero, card_id: str):
    """Remove the card from every live zone it occupies; return the Card."""
    found = None
    for i, c in enumerate(hero.hand):
        if c.id == card_id:
            found = hero.hand.pop(i)
            break
    if found is None:
        for i, c in enumerate(hero.discard_pile):
            if c.id == card_id:
                found = hero.discard_pile.pop(i)
                break
    if found is None:
        for i, c in enumerate(hero.played_cards):
            if c is not None and c.id == card_id:
                found = c
                hero.played_cards[i] = None
                break
    if found is None and hero.current_turn_card and hero.current_turn_card.id == card_id:
        found = hero.current_turn_card
        hero.current_turn_card = None
    if found is None and hero.extra_turn_card and hero.extra_turn_card.id == card_id:
        found = hero.extra_turn_card
        hero.extra_turn_card = None
    return found


def _apply_move_card(session: GameSession, args: MoveCardArgs) -> None:
    hero = _require_hero(session.state, args.hero_id)
    card = _detach_card(hero, args.card_id)
    if card is None:
        raise OverrideRejectedError(
            f"Card {args.card_id!r} not found in a movable zone of {args.hero_id} "
            "(hand / discard / played / current / extra)",
            code="unknown_card",
        )
    if args.zone == "hand":
        hero.hand.append(card)
    elif args.zone == "discard":
        hero.discard_pile.append(card)
    else:  # played — fill the first empty slot, else append
        for i, slot in enumerate(hero.played_cards):
            if slot is None:
                hero.played_cards[i] = card
                break
        else:
            hero.played_cards.append(card)


_register(
    OverrideOp(
        name="move_card",
        family="patch",
        label="Move card between zones",
        description="Move a card stuck in the wrong zone to hand, discard, or played.",
        args_model=MoveCardArgs,
        apply=_apply_move_card,
        summary_template="Move card {card_id} of {hero_id} to {zone}",
    )
)


class AddMarkerArgs(BaseModel):
    marker_type: MarkerType
    target_id: str
    value: int = 0
    source_id: str


def _apply_add_marker(session: GameSession, args: AddMarkerArgs) -> None:
    state = session.state
    if state.get_hero(HeroID(args.target_id)) is None and state.get_entity(
        BoardEntityID(args.target_id)
    ) is None:
        raise OverrideRejectedError(
            f"Unknown marker target {args.target_id!r}", code="unknown_entity"
        )
    state.place_marker(args.marker_type, args.target_id, args.value, args.source_id)


_register(
    OverrideOp(
        name="add_marker",
        family="patch",
        label="Place marker",
        description="Place a singleton marker (venom / poison / bounty) on a target.",
        args_model=AddMarkerArgs,
        apply=_apply_add_marker,
        summary_template="Place {marker_type} marker on {target_id}",
    )
)


class RemoveMarkerArgs(BaseModel):
    marker_type: MarkerType


def _apply_remove_marker(session: GameSession, args: RemoveMarkerArgs) -> None:
    session.state.remove_marker(args.marker_type)


_register(
    OverrideOp(
        name="remove_marker",
        family="patch",
        label="Return marker to supply",
        description="Return a marker to the supply (fixes wrong attribution).",
        args_model=RemoveMarkerArgs,
        apply=_apply_remove_marker,
        summary_template="Return {marker_type} marker to supply",
    )
)


class AddEffectArgs(BaseModel):
    effect: dict


def _apply_add_effect(session: GameSession, args: AddEffectArgs) -> None:
    try:
        effect = ActiveEffect.model_validate(args.effect)
    except Exception as exc:
        raise OverrideRejectedError(
            f"Invalid effect payload: {exc}", code="invalid_args"
        ) from exc
    if any(e.id == effect.id for e in session.state.active_effects):
        raise OverrideRejectedError(
            f"Effect id {effect.id!r} already active", code="duplicate_effect"
        )
    session.state.add_effect(effect)


_register(
    OverrideOp(
        name="add_effect",
        family="patch",
        label="Add active effect",
        description="Re-instate a buff/debuff that expired early (full ActiveEffect payload).",
        args_model=AddEffectArgs,
        apply=_apply_add_effect,
        summary_template="Add effect {effect}",
    )
)


class RemoveEffectArgs(BaseModel):
    effect_id: str


def _apply_remove_effect(session: GameSession, args: RemoveEffectArgs) -> None:
    state = session.state
    before = len(state.active_effects)
    state.active_effects = [e for e in state.active_effects if e.id != args.effect_id]
    if len(state.active_effects) == before:
        raise OverrideRejectedError(
            f"No active effect with id {args.effect_id!r}", code="unknown_effect"
        )


_register(
    OverrideOp(
        name="remove_effect",
        family="patch",
        label="Remove active effect",
        description="End a spurious buff/debuff immediately.",
        args_model=RemoveEffectArgs,
        apply=_apply_remove_effect,
        summary_template="Remove effect {effect_id}",
    )
)
```

Check the real `Hero` model field names in `src/goa2/domain/models/` before writing `_detach_card` (hand / discard_pile / played_cards / current_turn_card / extra_turn_card are confirmed from `server/visibility.py`). The card-identity unification validator in `GameState` runs in the revalidation round-trip and will reconcile the moved card object with `hero.deck` (the master list); if it rejects a zone move, surface that as the atomic rejection it is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py -q`
Expected: PASS except `test_registry_contains_all_patch_and_unstick_ops` (unstick ops in Task 4).

- [ ] **Step 5: Commit**

```bash
git add src/goa2/engine/overrides.py tests/engine/test_override_ops.py
git commit -m "Add card, marker, and effect override ops"
```

---

### Task 4: Unstick ops + pending-input interaction

**Files:**
- Modify: `src/goa2/engine/overrides.py`
- Test: `tests/engine/test_override_unstick.py` (new)

**Interfaces:**
- Produces ops: `skip_input`, `abort_action`, `end_turn`, `force_actor`.
- Verifies the Task 1 epilogue behaviors: stale option list re-derived, stale answer rejected via bumped request id.

- [ ] **Step 1: Write failing tests**

```python
# tests/engine/test_override_unstick.py
"""Unstick override ops + patch/pending-input interaction."""

import pytest

from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase
from goa2.engine.overrides import OverrideRejectedError, apply_override_decision
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def mid_resolution_session() -> GameSession:
    """A 1v1 game advanced into RESOLUTION with a pending input request."""
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=42)
    session = GameSession(state)
    for hero_id in ("hero_arien", "hero_wasp"):
        hero = state.get_hero(hero_id)
        session.commit_card(hero_id, hero.hand[0])
    result = session.advance(None) if state.phase != GamePhase.PLANNING else session._check_after_planning()
    # Drive until an input request is pending
    while result.input_request is None and state.phase == GamePhase.RESOLUTION:
        result = session.advance(None)
    assert result.input_request is not None
    session._last_request = result.input_request  # test-side stash
    return session


def test_patch_bumps_pending_request_id(mid_resolution_session):
    session = mid_resolution_session
    old_request = session._last_request
    result = apply_override_decision(
        session, "set_gold", {"hero_id": "hero_arien", "value": 9}
    )
    assert result is not None and result.input_request is not None
    # Re-derived request has a NEW id: a stale in-flight answer must be rejected.
    assert result.input_request.id != old_request.id
    with pytest.raises(ValueError):
        session.advance(InputResponse(request_id=old_request.id, selection="anything"))


def test_skip_input_answers_pending_request(mid_resolution_session):
    session = mid_resolution_session
    result = apply_override_decision(session, "skip_input", {})
    # The wedged request was answered with SKIP; play moved on (a new request
    # or a completed action, but not the same request id).
    if result is not None and result.input_request is not None:
        assert result.input_request.id != session._last_request.id


def test_abort_action_unwinds_to_turn_end(mid_resolution_session):
    session = mid_resolution_session
    result = apply_override_decision(session, "abort_action", {})
    assert session.state.phase in (GamePhase.RESOLUTION, GamePhase.PLANNING, GamePhase.CLEANUP)
    # The wedged step is gone from the stack.
    assert all(
        s.pending_request_id != session._last_request.id
        for s in session.state.execution_stack
    )


def test_end_turn_forces_turn_end(mid_resolution_session):
    session = mid_resolution_session
    actor = str(session.state.current_actor_id)
    result = apply_override_decision(session, "end_turn", {})
    hero = session.state.get_hero(actor)
    assert hero.current_turn_card is None  # finalized


def test_force_actor_sets_current_actor(mid_resolution_session):
    session = mid_resolution_session
    apply_override_decision(session, "force_actor", {"hero_id": "hero_wasp"})
    assert str(session.state.current_actor_id) == "hero_wasp"


def test_force_actor_unknown_hero_rejected(mid_resolution_session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(mid_resolution_session, "force_actor", {"hero_id": "hero_x"})


def test_skip_input_without_pending_request_rejected():
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=1)
    session = GameSession(state)  # PLANNING, nothing pending
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "skip_input", {})
```

Fixture note: the exact drive-to-input sequence depends on the first committed cards; if `hero.hand[0]` resolves without input, iterate hand cards or use the `tests/engine/effects/` builders (`goa2-card-effect-tests` skill) to construct a deterministic pending `SelectStep`. The assertions are what matter: a pending request exists before the override.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_unstick.py -q`
Expected: FAIL with unknown-op errors.

- [ ] **Step 3: Implement unstick ops**

Append to `src/goa2/engine/overrides.py`:

```python
# ---------------------------------------------------------------------------
# Unstick ops — repair control flow, not values
# ---------------------------------------------------------------------------


class NoArgs(BaseModel):
    pass


def _pending_step(state: GameState):
    if state.execution_stack:
        top = state.execution_stack[-1]
        if top.pending_request_id is not None:
            return top
    return None


def _apply_skip_input(session: GameSession, args: NoArgs) -> None:
    from goa2.engine.handler import submit_input

    step = _pending_step(session.state)
    if step is None:
        raise OverrideRejectedError("No pending input request to skip", code="nothing_pending")
    # The existing "SKIP" sentinel; the shared epilogue's advance() processes it.
    submit_input(session.state, {"selection": "SKIP"})


_register(
    OverrideOp(
        name="skip_input",
        family="unstick",
        label="Skip the pending input",
        description="Answer the wedged input request with the SKIP sentinel.",
        args_model=NoArgs,
        apply=_apply_skip_input,
        summary_template="Skip the pending input request",
    )
)


def _apply_abort_action(session: GameSession, args: NoArgs) -> None:
    from goa2.engine.handler import _clear_after_abort

    state = session.state
    step = _pending_step(state)
    if step is not None:
        # _clear_after_abort assumes the failing step was already popped.
        state.execution_stack.pop()
    _clear_after_abort(state)


_register(
    OverrideOp(
        name="abort_action",
        family="unstick",
        label="Abort the current action",
        description="Unwind the wedged action through the normal abort path "
        "(defense/reaction sequences unwind correctly).",
        args_model=NoArgs,
        apply=_apply_abort_action,
        summary_template="Abort the current action",
    )
)


def _apply_end_turn(session: GameSession, args: NoArgs) -> None:
    from goa2.engine.steps.phases import FinalizeHeroTurnStep, FindNextActorStep

    state = session.state
    if state.phase != GamePhase.RESOLUTION:
        raise OverrideRejectedError(
            f"end_turn only applies during RESOLUTION (phase is {state.phase})",
            code="wrong_phase",
        )
    actor = state.current_actor_id
    state.execution_stack.clear()
    if actor is not None:
        # FinalizeHeroTurnStep clears context and chains FindNextActorStep itself.
        state.execution_stack.append(FinalizeHeroTurnStep(hero_id=str(actor)))
    else:
        state.execution_context.clear()
        state.execution_stack.append(FindNextActorStep())


_register(
    OverrideOp(
        name="end_turn",
        family="unstick",
        label="Force end of hero turn",
        description="Discard the wedged stack and finalize the current hero's turn.",
        args_model=NoArgs,
        apply=_apply_end_turn,
        summary_template="Force the current hero turn to end",
    )
)


class ForceActorArgs(BaseModel):
    hero_id: str


def _apply_force_actor(session: GameSession, args: ForceActorArgs) -> None:
    state = session.state
    _require_hero(state, args.hero_id)
    state.current_actor_id = HeroID(args.hero_id)
    state.resolution_owner_id = HeroID(args.hero_id)


_register(
    OverrideOp(
        name="force_actor",
        family="unstick",
        label="Fix the current actor",
        description="Set current_actor_id when turn order itself went wrong.",
        args_model=ForceActorArgs,
        apply=_apply_force_actor,
        summary_template="Set the current actor to {hero_id}",
    )
)
```

Note on the epilogue interaction: `skip_input` sets `pending_input` on the top step and clears its `pending_request_id`; the shared epilogue in `apply_override_decision` then finds no `pending_request_id` to bump and `advance(None)` processes the SKIP. `abort_action`/`end_turn` leave a clean stack for the same `advance(None)`.

- [ ] **Step 4: Run all engine override tests**

Run: `PYTHONPATH=src uv run pytest tests/engine/test_override_ops.py tests/engine/test_override_unstick.py -q`
Expected: ALL PASS, including `test_registry_contains_all_patch_and_unstick_ops`.

- [ ] **Step 5: Resolve()-idempotence spot-audit (spec requirement, not a footnote)**

The re-derive path re-runs `resolve()` on a step that already issued a request. Persistence already leans on this globally, but verify: grep `src/goa2/engine/steps/` for steps that mutate `self` in `resolve()` before returning `requires_input` (`grep -n "self\." src/goa2/engine/steps/*.py` around `requires_input=True` returns). For each hit, confirm the mutation is either overwritten on re-run or guarded. Record findings in the commit message body; if a genuine double-count exists, guard it in that step (separate commit).

- [ ] **Step 6: Commit**

```bash
git add src/goa2/engine/overrides.py tests/engine/test_override_unstick.py
git commit -m "Add unstick override ops and request-id bump on patch"
```

---

### Task 5: Replay integration — records, reconstruction, rewind

**Files:**
- Modify: `src/goa2/server/replay.py`
- Test: `tests/server/test_override_replay.py` (new)

**Interfaces:**
- Consumes: `apply_override_decision` (Task 1), op names (Tasks 1–4).
- Produces:
  - `ReplayRecorder.record_override(self, record: dict[str, Any]) -> None`
  - `effective_indices(decisions: list[dict], upto: int | None = None) -> list[int]`
  - `effective_decisions(decisions: list[dict], upto: int | None = None) -> list[dict]`
  - `rebuild_session_for_rewind(path: str, target_index: int) -> GameSession`
  - `_apply_decision` handles `ov_patch` / `ov_unstick` (raises on `ov_rewind` — cursor-level record)
  - `ReplayCursor.seek` and `replay_game` handle `ov_rewind` in the driving loop.

- [ ] **Step 1: Write failing tests**

```python
# tests/server/test_override_replay.py
"""Replay parity and rewind semantics for override records."""

import json

import pytest

from goa2.engine.overrides import apply_override_decision
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server.replay import (
    ReplayCursor,
    ReplayRecorder,
    _apply_decision,
    effective_decisions,
    effective_indices,
    load_replay,
    rebuild_session_for_rewind,
    replay_game,
)

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


def _fresh_session(seed=42) -> GameSession:
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=seed)
    return GameSession(state)


def _write_replay(tmp_path, decisions, seed=42) -> str:
    rec = ReplayRecorder("g1", replay_dir=str(tmp_path))
    rec.record_setup(
        map_name="forgotten_island", red_heroes=["Arien"], blue_heroes=["Wasp"],
        game_type="QUICK", cheats=False, seed=seed,
    )
    for d in decisions:
        rec._append(d)
    return str(rec.path)


# ---- effective_indices / effective_decisions ----

def test_effective_indices_no_rewind():
    ds = [{"type": "pass"}, {"type": "pass"}]
    assert effective_indices(ds) == [0, 1]


def test_effective_indices_simple_rewind():
    ds = [
        {"type": "pass", "hero": "a"},          # 0
        {"type": "pass", "hero": "b"},          # 1
        {"type": "ov_rewind", "to": 1},          # 2  -> keep only decision 0
        {"type": "pass", "hero": "c"},          # 3
    ]
    assert effective_indices(ds) == [0, 3]
    assert [d["hero"] for d in effective_decisions(ds)] == ["a", "c"]


def test_effective_indices_nested_rewind():
    ds = [
        {"type": "pass", "hero": "a"},          # 0
        {"type": "ov_rewind", "to": 0},          # 1  -> back to start
        {"type": "pass", "hero": "b"},          # 2
        {"type": "ov_rewind", "to": 3},          # 3  -> replay 0..2 effectively = [b]... 
        {"type": "pass", "hero": "c"},          # 4
    ]
    # effective(3) resolves the inner rewind: [b]; so final = [b, c]
    assert [d["hero"] for d in effective_decisions(ds)] == ["b", "c"]


# ---- replay parity (the load-bearing test) ----

def test_override_patch_replay_parity(tmp_path):
    live = _fresh_session()
    result = apply_override_decision(live, "set_gold", {"hero_id": "hero_arien", "value": 9})
    record = {
        "type": "ov_patch", "r": live.state.round, "t": live.state.turn,
        "hero": "hero_arien", "op": "set_gold",
        "args": {"hero_id": "hero_arien", "value": 9},
        "voters": ["hero_arien", "hero_wasp"],
    }
    path = _write_replay(tmp_path, [record])
    replayed = replay_game(path)
    assert (
        replayed.state.model_dump(mode="json", exclude={"clock", "time_control"})
        == live.state.model_dump(mode="json", exclude={"clock", "time_control"})
    )


def test_voters_field_ignored_by_reconstruction(tmp_path):
    record = {
        "type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
        "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 5},
        "voters": [],
    }
    path = _write_replay(tmp_path, [record])
    replayed = replay_game(path)
    assert replayed.state.get_hero("hero_arien").gold == 5


def test_cheat_gold_legacy_branch_still_loads(tmp_path):
    record = {"type": "cheat_gold", "r": 1, "t": 1, "hero": "hero_arien", "amount": 3}
    path = _write_replay(tmp_path, [record])
    start_gold = _fresh_session().state.get_hero("hero_arien").gold
    replayed = replay_game(path)
    assert replayed.state.get_hero("hero_arien").gold == start_gold + 3


# ---- rewind determinism ----

def test_rewind_record_rebuilds_from_seed(tmp_path):
    ds = [
        {"type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
         "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 9}},
        {"type": "ov_rewind", "r": 1, "t": 1, "hero": "hero_arien", "to": 0,
         "voters": []},
        {"type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
         "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 4}},
    ]
    path = _write_replay(tmp_path, ds)
    replayed = replay_game(path)
    # The gold=9 segment is dead; only gold=4 applies.
    assert replayed.state.get_hero("hero_arien").gold == 4


def test_cursor_seek_through_rewind_and_back(tmp_path):
    ds = [
        {"type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
         "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 9}},
        {"type": "ov_rewind", "r": 1, "t": 1, "hero": "hero_arien", "to": 0},
        {"type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
         "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 4}},
    ]
    path = _write_replay(tmp_path, ds)
    setup, decisions = load_replay(path)
    cursor = ReplayCursor(setup, decisions)
    s1 = cursor.seek(1)
    assert s1.state.get_hero("hero_arien").gold == 9
    s2 = cursor.seek(2)  # crosses the rewind record
    assert s2.state.get_hero("hero_arien").gold != 9
    s3 = cursor.seek(3)
    assert s3.state.get_hero("hero_arien").gold == 4
    # Backward seek still rebuilds correctly
    s1b = cursor.seek(1)
    assert s1b.state.get_hero("hero_arien").gold == 9


def test_rebuild_session_for_rewind(tmp_path):
    ds = [
        {"type": "ov_patch", "r": 1, "t": 1, "hero": "hero_arien",
         "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 9}},
    ]
    path = _write_replay(tmp_path, ds)
    session = rebuild_session_for_rewind(path, 0)
    assert session.state.get_hero("hero_arien").gold == _fresh_session().state.get_hero("hero_arien").gold
    with pytest.raises(ValueError):
        rebuild_session_for_rewind(path, 5)


def test_record_override_appends_with_ts(tmp_path):
    rec = ReplayRecorder("g2", replay_dir=str(tmp_path))
    rec.record_setup(map_name="forgotten_island", red_heroes=["Arien"],
                     blue_heroes=["Wasp"], game_type="QUICK", cheats=False, seed=1)
    rec.record_override({"type": "ov_unstick", "r": 2, "t": 1,
                         "hero": "hero_arien", "op": "abort_action", "args": {},
                         "voters": ["hero_arien"]})
    lines = [json.loads(l) for l in open(rec.path) if l.strip()]
    assert lines[-1]["type"] == "ov_unstick"
    assert "ts" in lines[-1]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_replay.py -q`
Expected: FAIL with `ImportError: cannot import name 'effective_decisions'`.

- [ ] **Step 3: Implement replay support**

In `src/goa2/server/replay.py`:

1. Recorder method (after `record_timer_timeout`):

```python
    def record_override(self, record: dict[str, Any]) -> None:
        """Append a consensus-override decision (ov_patch / ov_unstick / ov_rewind).

        The caller builds the full record (type, r, t, hero, op/args or to,
        voters). ``voters`` is auditability only; reconstruction ignores it.
        """
        if record.get("type") not in {"ov_patch", "ov_unstick", "ov_rewind"}:
            raise ValueError(f"Not an override record: {record.get('type')!r}")
        self._append(record)
```

2. Effective-index resolution (near `index_for_round_turn`):

```python
def effective_indices(decisions: list[dict[str, Any]], upto: int | None = None) -> list[int]:
    """Raw indices of decisions still live after resolving ov_rewind records.

    An ov_rewind at raw index i with to=N replaces the live list with the
    effective resolution of the first N raw records (N < i, so this recursion
    terminates). Rewind records themselves are never live — they are cursor
    moves, not decisions.
    """
    end = len(decisions) if upto is None else upto
    live: list[int] = []
    for i in range(end):
        d = decisions[i]
        if d.get("type") == "ov_rewind":
            live = effective_indices(decisions, int(d["to"]))
        else:
            live.append(i)
    return live


def effective_decisions(
    decisions: list[dict[str, Any]], upto: int | None = None
) -> list[dict[str, Any]]:
    return [decisions[i] for i in effective_indices(decisions, upto)]
```

3. `_apply_decision` branches (before the final `else`):

```python
    elif kind in ("ov_patch", "ov_unstick"):
        from goa2.engine.overrides import apply_override_decision

        apply_override_decision(session, decision["op"], decision.get("args", {}))
    elif kind == "ov_rewind":
        # A rewind changes the *cursor*, not the session; the driving loop
        # (ReplayCursor.seek / replay_game) owns the cursor and must handle it.
        raise ValueError("ov_rewind must be handled by the replay driving loop")
```

4. Rewind-aware driving loops. In `ReplayCursor.seek`, replace the `while` body:

```python
        while self.cursor < target:
            decision = self.decisions[self.cursor]
            if decision.get("type") == "ov_rewind":
                self.session = build_session_from_setup(self.setup)
                for d in effective_decisions(self.decisions, int(decision["to"])):
                    _apply_decision(self.session, d)
            else:
                _apply_decision(self.session, decision)
            self.cursor += 1
        return self.session
```

In `replay_game`, replace `_apply_decision(session, decision)` in the loop with the same pattern:

```python
        if decision.get("type") == "ov_rewind":
            session = build_session_from_setup(setup)
            for d in effective_decisions(decisions, int(decision["to"])):
                _apply_decision(session, d)
        else:
            _apply_decision(session, decision)
```

(`session` reassignment means the final `return session` must return the rebound variable — it already does.)

5. Live-rewind builder (after `replay_game`):

```python
def rebuild_session_for_rewind(path: str, target_index: int) -> GameSession:
    """Reconstruct a session positioned after `target_index` raw decisions.

    Used by the live rewind apply path: the returned session REPLACES
    ManagedGame.session. Prior ov_rewind records inside the prefix are honored.
    """
    setup, decisions = load_replay(path)
    if not 0 <= target_index <= len(decisions):
        raise ValueError(
            f"Rewind target {target_index} out of range 0..{len(decisions)}"
        )
    session = build_session_from_setup(setup)
    for d in effective_decisions(decisions, target_index):
        _apply_decision(session, d)
    return session
```

6. Update the module docstring's format listing with the three new record shapes (copy the JSON examples from the spec's "Overrides are replay records" section).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_replay.py tests/server/test_replay.py tests/server/test_replays_api.py -q`
Expected: ALL PASS (existing replay tests must not regress).

- [ ] **Step 5: Commit**

```bash
git add src/goa2/server/replay.py tests/server/test_override_replay.py
git commit -m "Record and reconstruct override decisions, incl. append-only rewind"
```

---

### Task 6: Consensus protocol — proposals, votes, expiry, clock pause

**Files:**
- Create: `src/goa2/server/overrides.py`
- Modify: `src/goa2/server/registry.py` (ManagedGame fields)
- Modify: `src/goa2/server/time_control.py` (`reconcile_game_clock` pause branch)
- Test: `tests/server/test_override_consensus.py` (new)

**Interfaces:**
- Produces:
  - `OVERRIDE_PROPOSAL_TIMEOUT_SECONDS()` → int from `GOA2_OVERRIDE_TIMEOUT_SECONDS` env (default 120)
  - `@dataclass class OverrideProposal` with fields `id, proposer_hero_id, family, op, args, to, summary, eligible_voters, votes, created_at, expires_at` and methods `threshold() -> int`, `tally() -> dict`, `outcome() -> str | None`
  - `connected_hero_ids(game) -> list[str]`
  - `create_proposal(game, proposer_hero_id, data) -> OverrideProposal` (raises `ValueError` on invalid)
  - `register_vote(proposal, hero_id, approve: bool) -> None` (raises `ValueError`)
  - `proposed_msg(proposal) / updated_msg(proposal) / resolved_msg(proposal, outcome, reason=None) -> dict`
- `ManagedGame` gains: `pending_override: "OverrideProposal | None" = None`, `override_expiry_task: asyncio.Task[None] | None = None`.

- [ ] **Step 1: Write failing tests**

```python
# tests/server/test_override_consensus.py
"""Consensus rules: threshold snapshotting, votes, expiry-as-rejection."""

import time
from types import SimpleNamespace

import pytest

from goa2.server.overrides import (
    OverrideProposal,
    connected_hero_ids,
    create_proposal,
    register_vote,
)


def _fake_game(connected: list[str]):
    """Minimal ManagedGame stand-in: tokens + live ws connections."""
    player_tokens = {f"tok_{h}": h for h in connected + ["hero_offline"]}
    ws_connections = {f"tok_{h}": object() for h in connected}
    recorder = SimpleNamespace(path="/nonexistent.jsonl")
    return SimpleNamespace(
        player_tokens=player_tokens,
        ws_connections=ws_connections,
        pending_override=None,
        replay_recorder=recorder,
        session=None,
    )


def test_connected_hero_ids_only_live_connections():
    game = _fake_game(["hero_a", "hero_b"])
    assert sorted(connected_hero_ids(game)) == ["hero_a", "hero_b"]


def test_create_proposal_snapshots_voters_and_auto_yes():
    game = _fake_game(["hero_a", "hero_b", "hero_c"])
    p = create_proposal(game, "hero_a", {
        "family": "patch", "op": "set_gold",
        "args": {"hero_id": "hero_a", "value": 5},
    })
    assert sorted(p.eligible_voters) == ["hero_a", "hero_b", "hero_c"]
    assert p.votes == {"hero_a": True}          # proposer auto-counts yes
    assert p.threshold() == 2                    # majority of 3
    assert p.outcome() is None                   # not decided yet
    assert p.summary                             # server-rendered
    assert p.expires_at > time.time()


def test_two_player_game_requires_both():
    game = _fake_game(["hero_a", "hero_b"])
    p = create_proposal(game, "hero_a", {
        "family": "unstick", "op": "abort_action", "args": {},
    })
    assert p.threshold() == 2
    assert p.outcome() is None
    register_vote(p, "hero_b", True)
    assert p.outcome() == "applied"


def test_majority_no_rejects_early():
    game = _fake_game(["hero_a", "hero_b", "hero_c"])
    p = create_proposal(game, "hero_a", {
        "family": "unstick", "op": "abort_action", "args": {},
    })
    register_vote(p, "hero_b", False)
    register_vote(p, "hero_c", False)
    assert p.outcome() == "rejected"


def test_vote_from_non_eligible_rejected():
    game = _fake_game(["hero_a", "hero_b"])
    p = create_proposal(game, "hero_a", {
        "family": "unstick", "op": "abort_action", "args": {},
    })
    with pytest.raises(ValueError):
        register_vote(p, "hero_offline", True)


def test_duplicate_vote_updates_not_duplicates():
    game = _fake_game(["hero_a", "hero_b", "hero_c"])
    p = create_proposal(game, "hero_a", {
        "family": "unstick", "op": "abort_action", "args": {},
    })
    register_vote(p, "hero_b", False)
    register_vote(p, "hero_b", True)
    assert p.votes["hero_b"] is True
    assert p.outcome() == "applied"


def test_unknown_op_rejected_at_proposal_time():
    game = _fake_game(["hero_a", "hero_b"])
    with pytest.raises(ValueError):
        create_proposal(game, "hero_a", {"family": "patch", "op": "nope", "args": {}})


def test_invalid_args_rejected_at_proposal_time():
    game = _fake_game(["hero_a", "hero_b"])
    with pytest.raises(ValueError):
        create_proposal(game, "hero_a", {
            "family": "patch", "op": "set_gold", "args": {"hero_id": "x", "value": -3},
        })


def test_one_open_proposal_at_a_time():
    game = _fake_game(["hero_a", "hero_b"])
    game.pending_override = object()
    with pytest.raises(ValueError):
        create_proposal(game, "hero_a", {
            "family": "unstick", "op": "abort_action", "args": {},
        })


def test_rewind_family_needs_no_op():
    game = _fake_game(["hero_a", "hero_b"])
    # 'to' range validation against the replay log happens in the ws handler
    # where the recorder path is real; here only shape validation applies.
    p = create_proposal(game, "hero_a", {"family": "rewind", "to": 3})
    assert p.family == "rewind" and p.op is None and p.to == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_consensus.py -q`
Expected: FAIL with `ModuleNotFoundError: goa2.server.overrides`.

- [ ] **Step 3: Implement `server/overrides.py`**

```python
# src/goa2/server/overrides.py
"""Consensus-override proposal lifecycle.

A proposal is coordination, not game state: it lives on ManagedGame, is never
saved or broadcast in views, and dies on server restart. Only the outcome is
recorded (as a replay decision, by the ws apply path).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from goa2.engine.overrides import OverrideRejectedError, get_op, summarize_op

if TYPE_CHECKING:
    from goa2.server.registry import ManagedGame


def OVERRIDE_PROPOSAL_TIMEOUT_SECONDS() -> int:
    try:
        return int(os.environ.get("GOA2_OVERRIDE_TIMEOUT_SECONDS", 120))
    except ValueError:
        return 120


@dataclass
class OverrideProposal:
    id: str
    proposer_hero_id: str
    family: str                      # "patch" | "unstick" | "rewind"
    op: str | None                   # None for rewind
    args: dict[str, Any]
    to: int | None                   # rewind target decision index
    summary: str
    eligible_voters: list[str]       # snapshotted at proposal time
    votes: dict[str, bool]
    created_at: float
    expires_at: float

    def threshold(self) -> int:
        """Strictly more than half of the snapshotted voters."""
        return len(self.eligible_voters) // 2 + 1

    def tally(self) -> dict[str, list[str]]:
        return {
            "yes": sorted(h for h, v in self.votes.items() if v),
            "no": sorted(h for h, v in self.votes.items() if not v),
        }

    def outcome(self) -> str | None:
        """'applied' | 'rejected' when decided, None while still open."""
        yes = sum(1 for v in self.votes.values() if v)
        no = sum(1 for v in self.votes.values() if not v)
        if yes >= self.threshold():
            return "applied"
        if yes + (len(self.eligible_voters) - yes - no) < self.threshold():
            return "rejected"  # threshold unreachable
        return None


def connected_hero_ids(game: "ManagedGame") -> list[str]:
    """Heroes with a live player websocket right now. Spectators never count."""
    return sorted(
        {
            game.player_tokens[token]
            for token in game.ws_connections
            if token in game.player_tokens
        }
    )


def create_proposal(
    game: "ManagedGame", proposer_hero_id: str, data: dict[str, Any]
) -> OverrideProposal:
    if game.pending_override is not None:
        raise ValueError("Another override proposal is already open")

    family = data.get("family")
    if family not in ("patch", "unstick", "rewind"):
        raise ValueError("family must be patch, unstick, or rewind")

    op_name: str | None = None
    args: dict[str, Any] = {}
    to: int | None = None
    if family == "rewind":
        raw_to = data.get("to")
        if not isinstance(raw_to, int) or raw_to < 0:
            raise ValueError("rewind requires a non-negative integer 'to'")
        to = raw_to
        summary = f"Rewind the game to decision {to}"
    else:
        op_name = data.get("op", "")
        try:
            op = get_op(op_name)
        except OverrideRejectedError as exc:
            raise ValueError(exc.message) from exc
        if op.family != family:
            raise ValueError(f"op {op_name!r} belongs to family {op.family!r}")
        args = data.get("args") or {}
        try:
            op.args_model.model_validate(args)
        except Exception as exc:
            raise ValueError(f"Invalid args for {op_name}: {exc}") from exc
        summary = summarize_op(op_name, args)

    eligible = connected_hero_ids(game)
    if proposer_hero_id not in eligible:
        raise ValueError("Proposer must be a connected player")

    now = time.time()
    return OverrideProposal(
        id=uuid.uuid4().hex[:12],
        proposer_hero_id=proposer_hero_id,
        family=family,
        op=op_name,
        args=args,
        to=to,
        summary=summary,
        eligible_voters=eligible,
        votes={proposer_hero_id: True},
        created_at=now,
        expires_at=now + OVERRIDE_PROPOSAL_TIMEOUT_SECONDS(),
    )


def register_vote(proposal: OverrideProposal, hero_id: str, approve: bool) -> None:
    if hero_id not in proposal.eligible_voters:
        raise ValueError("Only players connected at proposal time may vote")
    proposal.votes[hero_id] = approve


# ---- WS payload builders -------------------------------------------------


def proposed_msg(proposal: OverrideProposal) -> dict[str, Any]:
    return {
        "type": "OVERRIDE_PROPOSED",
        "proposal_id": proposal.id,
        "proposer_hero_id": proposal.proposer_hero_id,
        "family": proposal.family,
        "op": proposal.op,
        "args": proposal.args,
        "to": proposal.to,
        "summary": proposal.summary,
        "eligible_voters": proposal.eligible_voters,
        "threshold": proposal.threshold(),
        "tally": proposal.tally(),
        "expires_at": proposal.expires_at,
    }


def updated_msg(proposal: OverrideProposal) -> dict[str, Any]:
    return {
        "type": "OVERRIDE_UPDATED",
        "proposal_id": proposal.id,
        "tally": proposal.tally(),
    }


def resolved_msg(
    proposal: OverrideProposal,
    outcome: str,
    reason: dict[str, str] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "OVERRIDE_RESOLVED",
        "proposal_id": proposal.id,
        "outcome": outcome,  # applied | rejected | expired | cancelled
        "tally": proposal.tally(),
    }
    if reason is not None:
        msg["reason"] = reason  # {"code": ..., "message": ...}
    return msg
```

- [ ] **Step 4: Add ManagedGame fields**

In `src/goa2/server/registry.py`, add to the `ManagedGame` dataclass (after `timer_task`):

```python
    # Consensus-override negotiation state. Coordination, not game state:
    # never saved, never in views; unresolved proposals die on restart.
    pending_override: Any | None = None  # OverrideProposal (server/overrides.py)
    override_expiry_task: asyncio.Task[None] | None = None
```

(`Any` avoids a registry↔overrides import cycle; `overrides.py` already type-checks it via TYPE_CHECKING.) Also cancel the task in `GameRegistry.remove()` alongside `timer_task`:

```python
        if game is not None and game.override_expiry_task is not None:
            game.override_expiry_task.cancel()
```

- [ ] **Step 5: Clock pause while a proposal is open**

In `src/goa2/server/time_control.py`, in `reconcile_game_clock`, immediately after the `GAME_OVER` branch (`finish_game_clock` return) and before the turn-change accounting:

```python
    if getattr(game, "pending_override", None) is not None:
        # A 120s override negotiation is not the active player's doing:
        # pause every personal clock while the proposal is open. Deliberate
        # departure from the rollback() precedent of never refunding time.
        activate_clocks(clock, None, request_id=None, now_ms=at_ms)
        return
```

With no active clocks, `apply_due_timeouts` fires nothing and `schedule_deadline` schedules nothing; resolution re-runs `reconcile_game_clock` and play resumes where it left off. Add a test to `tests/server/test_override_consensus.py`:

```python
def test_clock_pauses_while_proposal_open():
    """reconcile_game_clock deactivates all clocks when pending_override is set."""
    # Build a timed game the same way tests/server/test_server_time_control.py
    # does (reuse its ManagedGame/clock fixture pattern), assert
    # clock.active_kind is not None, set game.pending_override = object(),
    # call reconcile_game_clock(game, now_ms()), assert clock.active_kind is None,
    # clear pending_override, reconcile again, assert clocks reactivate.
```

Write it concretely by copying the fixture from `tests/server/test_server_time_control.py` (it constructs a ManagedGame with a `TimeControlConfig`); the assertion sequence above is the entire test body.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_consensus.py tests/server/test_server_time_control.py -q`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add src/goa2/server/overrides.py src/goa2/server/registry.py src/goa2/server/time_control.py tests/server/test_override_consensus.py
git commit -m "Add override consensus proposals with snapshot threshold and clock pause"
```

---

### Task 7: WebSocket wiring — propose/vote/cancel, apply on approval, live rewind

**Files:**
- Modify: `src/goa2/server/ws.py`
- Test: `tests/server/test_override_consensus.py` (append WS integration tests)

**Interfaces:**
- Consumes: Task 6 (`create_proposal`, `register_vote`, `*_msg`, `OVERRIDE_PROPOSAL_TIMEOUT_SECONDS`), Task 5 (`rebuild_session_for_rewind`, `record_override`, `load_replay`), Task 1 (`apply_override_decision`, `OverrideRejectedError`).
- Produces WS message handling: inbound `PROPOSE_OVERRIDE`, `VOTE_OVERRIDE`, `CANCEL_OVERRIDE`; broadcast `OVERRIDE_PROPOSED`, `OVERRIDE_UPDATED`, `OVERRIDE_RESOLVED`.

- [ ] **Step 1: Write failing WS integration tests**

Append to `tests/server/test_override_consensus.py` (reuse the `client`/`game_data`/`_token_for` fixture pattern from `tests/server/test_server_ws.py` — copy those three fixtures in, renamed if pytest complains about duplicates across files):

```python
# --- WS integration -------------------------------------------------------

import os
from fastapi.testclient import TestClient
from goa2.server.app import create_app


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("GOA2_SAVE_DIR", None)


@pytest.fixture
def game_data(client):
    resp = client.post("/games", json={
        "map_name": "forgotten_island",
        "red_heroes": ["Arien"], "blue_heroes": ["Wasp"],
    })
    return resp.json()


def _token_for(game_data, hero_id):
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(hero_id)


def _drain_until(ws, msg_type):
    for _ in range(20):
        msg = ws.receive_json()
        if msg["type"] == msg_type:
            return msg
    raise AssertionError(f"never received {msg_type}")


def test_propose_vote_apply_full_cycle(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    t_w = _token_for(game_data, "hero_wasp")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a, \
         client.websocket_connect(f"/games/{gid}/ws?token={t_w}") as ws_w:
        ws_a.receive_json()  # initial STATE_UPDATE
        ws_w.receive_json()
        ws_a.send_json({
            "type": "PROPOSE_OVERRIDE", "family": "patch",
            "op": "set_gold", "args": {"hero_id": "hero_arien", "value": 9},
        })
        proposed = _drain_until(ws_a, "OVERRIDE_PROPOSED")
        assert proposed["threshold"] == 2          # 2 connected -> both must agree
        assert proposed["tally"]["yes"] == ["hero_arien"]
        assert proposed["summary"]
        pid = proposed["proposal_id"]
        _drain_until(ws_w, "OVERRIDE_PROPOSED")

        ws_w.send_json({"type": "VOTE_OVERRIDE", "proposal_id": pid, "approve": True})
        resolved = _drain_until(ws_w, "OVERRIDE_RESOLVED")
        assert resolved["outcome"] == "applied"
        # Every client then receives a STATE_UPDATE with the patched value.
        update = _drain_until(ws_w, "STATE_UPDATE")
        arien = next(
            h for h in update["view"]["teams"]["RED"]["heroes"]
            if h["id"] == "hero_arien"
        )
        assert arien["gold"] == 9


def test_second_proposal_while_open_rejected(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a:
        ws_a.receive_json()
        ws_a.send_json({"type": "PROPOSE_OVERRIDE", "family": "unstick",
                        "op": "abort_action", "args": {}})
        _drain_until(ws_a, "OVERRIDE_PROPOSED")
        ws_a.send_json({"type": "PROPOSE_OVERRIDE", "family": "unstick",
                        "op": "abort_action", "args": {}})
        err = _drain_until(ws_a, "ERROR")
        assert "already open" in err["detail"]


def test_cancel_by_proposer_only(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    t_w = _token_for(game_data, "hero_wasp")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a, \
         client.websocket_connect(f"/games/{gid}/ws?token={t_w}") as ws_w:
        ws_a.receive_json(); ws_w.receive_json()
        ws_a.send_json({"type": "PROPOSE_OVERRIDE", "family": "unstick",
                        "op": "abort_action", "args": {}})
        pid = _drain_until(ws_a, "OVERRIDE_PROPOSED")["proposal_id"]
        _drain_until(ws_w, "OVERRIDE_PROPOSED")
        ws_w.send_json({"type": "CANCEL_OVERRIDE", "proposal_id": pid})
        err = _drain_until(ws_w, "ERROR")
        assert "proposer" in err["detail"].lower()
        ws_a.send_json({"type": "CANCEL_OVERRIDE", "proposal_id": pid})
        resolved = _drain_until(ws_a, "OVERRIDE_RESOLVED")
        assert resolved["outcome"] == "cancelled"


def test_spectator_cannot_propose_or_vote(client, game_data):
    gid = game_data["game_id"]
    with client.websocket_connect(
        f"/games/{gid}/ws?token={game_data['spectator_token']}"
    ) as ws:
        ws.receive_json()
        ws.send_json({"type": "PROPOSE_OVERRIDE", "family": "unstick",
                      "op": "abort_action", "args": {}})
        err = ws.receive_json()
        assert err["type"] == "ERROR"  # existing spectator guard


def test_rejected_patch_reports_structured_reason(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    t_w = _token_for(game_data, "hero_wasp")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a, \
         client.websocket_connect(f"/games/{gid}/ws?token={t_w}") as ws_w:
        ws_a.receive_json(); ws_w.receive_json()
        # Move Arien onto Wasp's hex: passes arg validation, fails at apply.
        wasp_hex = None
        # fetch positions from the initial view via GET_VIEW
        ws_a.send_json({"type": "GET_VIEW"})
        view = _drain_until(ws_a, "STATE_UPDATE")["view"]
        for team in view["teams"].values():
            for h in team["heroes"]:
                if h["id"] == "hero_wasp":
                    wasp_hex = h["position"]
        ws_a.send_json({
            "type": "PROPOSE_OVERRIDE", "family": "patch", "op": "move_entity",
            "args": {"entity_id": "hero_arien", "hex": wasp_hex},
        })
        pid = _drain_until(ws_a, "OVERRIDE_PROPOSED")["proposal_id"]
        _drain_until(ws_w, "OVERRIDE_PROPOSED")
        ws_w.send_json({"type": "VOTE_OVERRIDE", "proposal_id": pid, "approve": True})
        resolved = _drain_until(ws_w, "OVERRIDE_RESOLVED")
        assert resolved["outcome"] == "rejected"
        assert resolved["reason"]["code"]      # machine-readable
        assert resolved["reason"]["message"]   # human-readable


def test_rewind_proposal_replaces_session(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    t_w = _token_for(game_data, "hero_wasp")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a, \
         client.websocket_connect(f"/games/{gid}/ws?token={t_w}") as ws_w:
        ws_a.receive_json(); ws_w.receive_json()
        # Make one recorded decision: Arien passes.
        ws_a.send_json({"type": "PASS_TURN"})
        _drain_until(ws_a, "ACTION_RESULT")
        _drain_until(ws_w, "STATE_UPDATE")
        # Rewind to before it.
        ws_a.send_json({"type": "PROPOSE_OVERRIDE", "family": "rewind", "to": 0})
        pid = _drain_until(ws_a, "OVERRIDE_PROPOSED")["proposal_id"]
        _drain_until(ws_w, "OVERRIDE_PROPOSED")
        ws_w.send_json({"type": "VOTE_OVERRIDE", "proposal_id": pid, "approve": True})
        resolved = _drain_until(ws_w, "OVERRIDE_RESOLVED")
        assert resolved["outcome"] == "applied"
        update = _drain_until(ws_w, "STATE_UPDATE")
        # Arien's pass was undone: back in PLANNING with a full hand.
        assert update["view"]["phase"] == "PLANNING"


def test_view_payload_has_no_override_state(client, game_data):
    gid = game_data["game_id"]
    t_a = _token_for(game_data, "hero_arien")
    with client.websocket_connect(f"/games/{gid}/ws?token={t_a}") as ws_a:
        initial = ws_a.receive_json()
        assert "override" not in str(initial["view"]).lower() or True
        # The real assertion: build_view output gained no new keys.
        assert "pending_override" not in initial["view"]
```

Adjust view-shape assertions (`h["gold"]`, `h["position"]`) to the real `build_view` hero-view keys — read `_build_hero_view` in `domain/views.py` first and use its actual field names.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_consensus.py -q`
Expected: WS tests FAIL (`ERROR: Unknown message type: PROPOSE_OVERRIDE`).

- [ ] **Step 3: Implement the WS handlers**

In `src/goa2/server/ws.py`:

1. Imports:

```python
from goa2.engine.overrides import OverrideRejectedError, apply_override_decision
from goa2.server import overrides as ov
from goa2.server.replay import load_replay, rebuild_session_for_rewind
from goa2.server.time_control import (  # extend existing import
    finalize_timed_mutation, now_ms, prepare_timed_mutation, reconcile_game_clock,
)
```

2. Handlers (place after `_handle_cheats_gold`). Do **not** add the new message types to `MUTATION_MESSAGE_TYPES` — a vote alone mutates nothing; the apply step runs its own prepare/finalize. The dispatch for these three types gets its own branch in the main loop **before** the generic mutation machinery, mirroring the `PING` branch's lock usage:

```python
async def _resolve_override(
    game: ManagedGame,
    registry: GameRegistry,
    outcome: str,
    reason: dict[str, str] | None = None,
) -> CapturedBroadcast:
    """Resolve the open proposal (caller holds game.lock). Returns broadcasts."""
    proposal = game.pending_override
    assert proposal is not None
    game.pending_override = None
    if game.override_expiry_task is not None:
        game.override_expiry_task.cancel()
        game.override_expiry_task = None

    messages: CapturedBroadcast = []
    events: list[dict[str, Any]] | None = None

    if outcome == "applied":
        prepare_timed_mutation(game, registry=registry)
        rec_round, rec_turn = game.session.state.round, game.session.state.turn
        try:
            if proposal.family == "rewind":
                new_session = rebuild_session_for_rewind(
                    str(game.replay_recorder.path), proposal.to
                )
                game.session = new_session
                game.last_result = (
                    new_session.advance(None)
                    if new_session.state.phase
                    not in (GamePhase.PLANNING, GamePhase.GAME_OVER)
                    else None
                )
            else:
                result = apply_override_decision(
                    game.session, proposal.op, proposal.args
                )
                if result is not None:
                    game.last_result = result
                    events = [ev.model_dump() for ev in result.events]
        except (OverrideRejectedError, ValueError) as exc:
            outcome = "rejected"
            code = getattr(exc, "code", "invalid_op")
            reason = {"code": code, "message": str(exc)}
        else:
            record: dict[str, Any] = {
                "type": f"ov_{'rewind' if proposal.family == 'rewind' else proposal.family}",
                "r": rec_round,
                "t": rec_turn,
                "hero": proposal.proposer_hero_id,
                "voters": proposal.tally()["yes"],
            }
            if proposal.family == "rewind":
                record["to"] = proposal.to
            else:
                record["op"] = proposal.op
                record["args"] = proposal.args
            if game.replay_recorder:
                game.replay_recorder.record_override(record)
        finalize_timed_mutation(game, registry)  # reconcile + save + reschedule
        if outcome == "applied":
            messages.extend(_capture_broadcast(game, events))
    else:
        # No state mutation; just un-pause the clocks.
        reconcile_game_clock(game, now_ms())
        registry.save_game(game.game_id)

    resolved = ov.resolved_msg(proposal, outcome, reason)
    for token, ws_conn in list(game.ws_connections.items()):
        messages.append((token, ws_conn, dict(resolved)))
    for ws_conn in list(game.spectator_ws_connections.values()):
        messages.append((None, ws_conn, dict(resolved)))
    return messages


def _override_broadcast_to_all(game: ManagedGame, msg: dict[str, Any]) -> CapturedBroadcast:
    messages: CapturedBroadcast = [
        (token, ws_conn, dict(msg)) for token, ws_conn in list(game.ws_connections.items())
    ]
    messages.extend(
        (None, ws_conn, dict(msg)) for ws_conn in list(game.spectator_ws_connections.values())
    )
    return messages


async def _expire_override(game: ManagedGame, registry: GameRegistry, proposal_id: str) -> None:
    """Background task: expiry is a rejection (nobody actively agreed)."""
    proposal = game.pending_override
    if proposal is None:
        return
    delay = max(0.0, proposal.expires_at - time.time())
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    async with game.outbound_lock:
        async with game.lock:
            current = game.pending_override
            if current is None or current.id != proposal_id:
                return
            messages = await _resolve_override(game, registry, "expired")
        await _send_captured_broadcast(game, messages)
```

(add `import asyncio` and `from goa2.domain.models import GamePhase` is already imported; `time` is already imported.)

3. Dispatch branch in `game_ws`'s loop, after the `PING` branch and before the generic `try` (spectators are already screened out by the existing `is_spectator` guard above):

```python
            if msg_type in ("PROPOSE_OVERRIDE", "VOTE_OVERRIDE", "CANCEL_OVERRIDE"):
                try:
                    async with game.outbound_lock:
                        async with game.lock:
                            messages: CapturedBroadcast = []
                            if msg_type == "PROPOSE_OVERRIDE":
                                proposal = ov.create_proposal(game, hero_id, data)
                                if proposal.family == "rewind":
                                    _, decisions = load_replay(
                                        str(game.replay_recorder.path)
                                    )
                                    if not 0 <= proposal.to <= len(decisions):
                                        raise ValueError(
                                            f"Rewind target {proposal.to} out of "
                                            f"range 0..{len(decisions)}"
                                        )
                                game.pending_override = proposal
                                reconcile_game_clock(game, now_ms())  # pause
                                game.override_expiry_task = asyncio.create_task(
                                    _expire_override(game, registry, proposal.id)
                                )
                                messages = _override_broadcast_to_all(
                                    game, ov.proposed_msg(proposal)
                                )
                                # A 2-player proposer-only majority is impossible;
                                # threshold >= 2 always needs another vote, so no
                                # immediate-resolve check is required here — but
                                # guard anyway for a 1-connected-player game:
                                if proposal.outcome() == "applied":
                                    messages.extend(
                                        await _resolve_override(game, registry, "applied")
                                    )
                            elif msg_type == "VOTE_OVERRIDE":
                                proposal = game.pending_override
                                if proposal is None or proposal.id != data.get("proposal_id"):
                                    raise ValueError("No matching open proposal")
                                ov.register_vote(proposal, hero_id, bool(data.get("approve")))
                                outcome = proposal.outcome()
                                if outcome is None:
                                    messages = _override_broadcast_to_all(
                                        game, ov.updated_msg(proposal)
                                    )
                                else:
                                    messages = await _resolve_override(
                                        game, registry, outcome
                                    )
                            else:  # CANCEL_OVERRIDE
                                proposal = game.pending_override
                                if proposal is None or proposal.id != data.get("proposal_id"):
                                    raise ValueError("No matching open proposal")
                                if hero_id != proposal.proposer_hero_id:
                                    raise ValueError(
                                        "Only the proposer may cancel an override proposal"
                                    )
                                messages = await _resolve_override(
                                    game, registry, "cancelled"
                                )
                        await _send_captured_broadcast(game, messages)
                except ValueError as exc:
                    if game.game_logger:
                        game.game_logger.log_error(str(exc), hero_id)
                    async with game.outbound_lock:
                        await websocket.send_json({"type": "ERROR", "detail": str(exc)})
                continue
```

Ordering inside `_resolve_override` follows the spec: apply via op registry → append replay record → `registry.save_game()` (inside `finalize_timed_mutation`) → broadcast. The `OVERRIDE_RESOLVED` message is appended after the STATE_UPDATE captures so clients see outcome + fresh state in one flush.

- [ ] **Step 4: Run WS + full server tests**

Run: `PYTHONPATH=src uv run pytest tests/server/ -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/goa2/server/ws.py tests/server/test_override_consensus.py
git commit -m "Wire override consensus over WebSocket with apply-on-approval"
```

---

### Task 8: `GET /overrides/schema` endpoint

**Files:**
- Create: `src/goa2/server/routes_overrides.py`
- Modify: `src/goa2/server/models.py`, `src/goa2/server/app.py`
- Test: `tests/server/test_override_endpoints.py` (new)

**Interfaces:**
- Produces: `GET /overrides/schema` → `OverrideSchemaResponse`; router `overrides_router` registered in `app.py`.

- [ ] **Step 1: Write failing tests**

```python
# tests/server/test_override_endpoints.py
"""REST endpoints: op schema catalogue + player-scoped history."""

import os

import pytest
from fastapi.testclient import TestClient

from goa2.engine.overrides import OVERRIDE_OPS
from goa2.server.app import create_app


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("GOA2_SAVE_DIR", None)


def test_schema_lists_every_registered_op(client):
    resp = client.get("/overrides/schema")
    assert resp.status_code == 200
    ops = {o["name"]: o for o in resp.json()["ops"]}
    # Schema completeness: every registered op appears with a valid arg schema.
    assert set(ops) == set(OVERRIDE_OPS)
    for name, op in ops.items():
        assert op["family"] in ("patch", "unstick")
        assert op["label"] and op["description"]
        assert isinstance(op["args_schema"], dict)
        assert op["args_schema"].get("type") == "object"


def test_schema_is_static_and_unauthenticated(client):
    # Game-independent; clients fetch once and cache.
    assert client.get("/overrides/schema").json() == client.get("/overrides/schema").json()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_endpoints.py -q`
Expected: FAIL with 404.

- [ ] **Step 3: Implement models + endpoint**

Append to `src/goa2/server/models.py`:

```python
class OverrideOpSchema(BaseModel):
    name: str
    family: str  # "patch" | "unstick"
    label: str
    description: str
    args_schema: dict  # JSON Schema for the op's args


class OverrideSchemaResponse(BaseModel):
    ops: list[OverrideOpSchema]


class OverrideHistoryEntry(BaseModel):
    index: int          # raw decision index (rewind targets use this space)
    type: str           # replay record type (commit / input / ov_patch / ...)
    round: int | None = None
    turn: int | None = None
    hero_id: str | None = None
    label: str          # human-readable, viewer-scoped
    superseded: bool = False  # dead segment behind a rewind


class OverrideHistoryResponse(BaseModel):
    total: int
    decisions: list[OverrideHistoryEntry]
```

Create `src/goa2/server/routes_overrides.py`:

```python
"""Consensus-override REST endpoints (schema catalogue + decision history)."""

from __future__ import annotations

from fastapi import APIRouter

from goa2.engine.overrides import OVERRIDE_OPS
from goa2.server.models import OverrideOpSchema, OverrideSchemaResponse

router = APIRouter(tags=["overrides"])


@router.get("/overrides/schema", response_model=OverrideSchemaResponse)
async def get_override_schema() -> OverrideSchemaResponse:
    """The op catalogue, auto-derived from the registry.

    Static and game-independent (like /heroes): clients fetch once and cache.
    A hand-written catalogue would drift the first time an op is added.
    """
    return OverrideSchemaResponse(
        ops=[
            OverrideOpSchema(
                name=op.name,
                family=op.family,
                label=op.label,
                description=op.description,
                args_schema=op.args_model.model_json_schema(),
            )
            for op in sorted(OVERRIDE_OPS.values(), key=lambda o: (o.family, o.name))
        ]
    )
```

In `src/goa2/server/app.py`, import and register next to the other routers:

```python
    from goa2.server.routes_overrides import router as overrides_router
    app.include_router(overrides_router)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_endpoints.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/goa2/server/routes_overrides.py src/goa2/server/models.py src/goa2/server/app.py tests/server/test_override_endpoints.py
git commit -m "Add auto-derived override op schema endpoint"
```

---

### Task 9: `GET /games/{game_id}/overrides/history` — player-scoped decision history

**Files:**
- Modify: `src/goa2/server/routes_overrides.py`, `src/goa2/server/visibility.py` (export helper)
- Test: `tests/server/test_override_endpoints.py` (append)

**Interfaces:**
- Consumes: `effective_indices`, `load_replay` (Task 5); `_card_visible_to`, `_card_location` from `server/visibility.py`; `PlayerDep`/`RegistryDep` from `server/auth.py`; `summarize_op` (Task 1); `OverrideHistoryResponse` (Task 8).
- Produces: `GET /games/{game_id}/overrides/history` → `OverrideHistoryResponse`, bearer-authenticated, viewer-scoped.

- [ ] **Step 1: Write failing tests**

Append to `tests/server/test_override_endpoints.py`:

```python
@pytest.fixture
def game_data(client):
    resp = client.post("/games", json={
        "map_name": "forgotten_island",
        "red_heroes": ["Arien"], "blue_heroes": ["Wasp"],
    })
    return resp.json()


def _token_for(game_data, hero_id):
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(hero_id)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _commit_first_card(client, game_data, hero_id):
    token = _token_for(game_data, hero_id)
    gid = game_data["game_id"]
    view = client.get(f"/games/{gid}", headers=_auth(token)).json()["view"]
    for team in view["teams"].values():
        for h in team["heroes"]:
            if h["id"] == hero_id:
                card_id = h["hand"][0]["id"]
    client.post(f"/games/{gid}/commit", json={"card_id": card_id}, headers=_auth(token))
    return card_id


def test_history_requires_auth(client, game_data):
    gid = game_data["game_id"]
    assert client.get(f"/games/{gid}/overrides/history").status_code == 401


def test_history_masks_opponent_facedown_commit(client, game_data):
    gid = game_data["game_id"]
    arien_card = _commit_first_card(client, game_data, "hero_arien")

    # Wasp must NOT learn the identity of Arien's facedown commit.
    wasp = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(_token_for(game_data, "hero_wasp")),
    ).json()
    commit_rows = [d for d in wasp["decisions"] if d["type"] == "commit"]
    assert commit_rows, "commit decision missing from history"
    assert arien_card not in commit_rows[0]["label"]
    assert "a card" in commit_rows[0]["label"]

    # Arien sees their own card.
    arien = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(_token_for(game_data, "hero_arien")),
    ).json()
    own_rows = [d for d in arien["decisions"] if d["type"] == "commit"]
    assert arien_card in own_rows[0]["label"] or "a card" not in own_rows[0]["label"]


def test_history_spectator_gets_fully_masked_form(client, game_data):
    gid = game_data["game_id"]
    arien_card = _commit_first_card(client, game_data, "hero_arien")
    spec = client.get(
        f"/games/{gid}/overrides/history",
        headers=_auth(game_data["spectator_token"]),
    ).json()
    for row in spec["decisions"]:
        assert arien_card not in row["label"]


def test_history_marks_superseded_segments(client, game_data, tmp_path):
    """Records behind a rewind carry superseded=True."""
    gid = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")
    # Fabricate the scenario at the replay-log level: one decision + a rewind.
    from goa2.server.replay import create_replay_recorder
    rec = create_replay_recorder(gid)
    rec.record_pass("hero_arien", 1, 1)
    rec.record_override({"type": "ov_rewind", "r": 1, "t": 1,
                         "hero": "hero_arien", "to": 0, "voters": []})
    hist = client.get(
        f"/games/{gid}/overrides/history", headers=_auth(token)
    ).json()
    rows = {d["index"]: d for d in hist["decisions"]}
    pass_row = next(d for d in hist["decisions"] if d["type"] == "pass")
    rewind_row = next(d for d in hist["decisions"] if d["type"] == "ov_rewind")
    assert pass_row["superseded"] is True
    assert rewind_row["superseded"] is False
    assert "rewound" in rewind_row["label"].lower() or "rewind" in rewind_row["label"].lower()
```

Adjust `_commit_first_card` REST paths/shapes to the real routes in `routes_games.py` (`GET /games/{id}` view shape, commit endpoint path/body) — read those handlers first.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_endpoints.py -q`
Expected: new tests FAIL with 404.

- [ ] **Step 3: Implement the history endpoint**

Append to `src/goa2/server/routes_overrides.py`:

```python
from typing import Any

from fastapi import HTTPException

from goa2.domain.state import GameState
from goa2.server.auth import PlayerDep, RegistryDep
from goa2.server.errors import GameNotFoundError
from goa2.server.models import OverrideHistoryEntry, OverrideHistoryResponse
from goa2.server.replay import effective_indices, load_replay
from goa2.server.visibility import _card_location, _card_visible_to


def _card_label(state: GameState, card_id: str, for_hero_id: str | None) -> str:
    """Card name if the viewer is entitled to it NOW, else an anonymous form.

    Identity is masked with the same visibility rule the view uses; a card
    committed facedown reads "a card" until it is public. The omniscient
    replay-debugger view (reveal_all) is never reachable from here.
    """
    if _card_visible_to(state, card_id, for_hero_id) is False:
        return "a card"
    located = _card_location(state, card_id)
    return located[2].name if located else card_id


def _decision_label(
    d: dict[str, Any], state: GameState, for_hero_id: str | None
) -> str:
    kind = d.get("type", "?")
    hero = d.get("hero", "?")
    if kind == "commit":
        return f"{hero} committed {_card_label(state, d.get('card', ''), for_hero_id)}"
    if kind == "pass":
        return f"{hero} passed"
    if kind == "uncommit":
        return f"{hero} took back a committed card"
    if kind == "finish_planning":
        return f"{hero} finished planning"
    if kind == "input":
        sel = d.get("sel")
        if isinstance(sel, str) and _card_visible_to(state, sel, for_hero_id) is False:
            sel = "a hidden card"
        return f"{hero} chose {sel!r}"
    if kind == "rollback":
        return f"{hero} rolled back their action"
    if kind == "cheat_gold":
        return f"{hero} gained {d.get('amount')} gold (cheat)"
    if kind == "timer_timeout":
        return f"Automatic decision for {hero} (timer expired)"
    if kind in ("ov_patch", "ov_unstick"):
        from goa2.engine.overrides import summarize_op

        try:
            return f"Override: {summarize_op(d.get('op', ''), d.get('args', {}))}"
        except Exception:
            return f"Override: {d.get('op')}"
    if kind == "ov_rewind":
        return f"The table rewound the game to decision {d.get('to')}"
    return kind


@router.get(
    "/games/{game_id}/overrides/history",
    response_model=OverrideHistoryResponse,
)
async def get_override_history(
    game_id: str, player: PlayerDep, registry: RegistryDep
) -> OverrideHistoryResponse:
    """Player-scoped decision list so a rewind target index means something.

    Card identity is masked with the view's visibility rule: spectators get
    the fully-masked form; a player never sees an opponent's facedown commit.
    """
    try:
        game = registry.get(game_id)
    except GameNotFoundError:
        raise HTTPException(status_code=404, detail="Game not found") from None
    recorder = game.replay_recorder
    if recorder is None or not recorder.path.is_file():
        return OverrideHistoryResponse(total=0, decisions=[])

    _, decisions = load_replay(str(recorder.path))
    live = set(effective_indices(decisions))
    state = game.session.state
    viewer = player.hero_id if not player.is_spectator else None

    entries = [
        OverrideHistoryEntry(
            index=i,
            type=str(d.get("type", "?")),
            round=d.get("r"),
            turn=d.get("t"),
            hero_id=d.get("hero"),
            label=_decision_label(d, state, viewer),
            superseded=(i not in live and d.get("type") != "ov_rewind"),
        )
        for i, d in enumerate(decisions)
    ]
    return OverrideHistoryResponse(total=len(decisions), decisions=entries)
```

If importing `_card_location` / `_card_visible_to` (underscore names) from `visibility.py` bothers review, rename them to public (`card_location`, `card_visible_to`) in `visibility.py` with alias assignments for the old names — do it in this task.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/server/test_override_endpoints.py tests/server/test_visibility.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/goa2/server/routes_overrides.py src/goa2/server/visibility.py src/goa2/server/models.py tests/server/test_override_endpoints.py
git commit -m "Add player-scoped override history endpoint with rewind markers"
```

---

### Task 10: Client integration guide + full-suite verification

**Files:**
- Modify: `docs/CLIENT_INTEGRATION_GUIDE.md`

- [ ] **Step 1: Document the feature in the client guide**

Add a "Consensus Overrides" section covering, with concrete JSON examples matching the Task 6/7 payload builders exactly:
- The three inbound WS messages (`PROPOSE_OVERRIDE` with `family`/`op`/`args` or `family:"rewind"`/`to`; `VOTE_OVERRIDE` with `proposal_id`/`approve`; `CANCEL_OVERRIDE` with `proposal_id`) and who may send them (players only; proposer-only cancel; spectators receive broadcasts but cannot vote).
- The three broadcasts (`OVERRIDE_PROPOSED` full payload incl. `summary`, `eligible_voters`, `threshold`, `tally`, absolute `expires_at`; `OVERRIDE_UPDATED` tally; `OVERRIDE_RESOLVED` outcome ∈ `applied|rejected|expired|cancelled` + structured `reason` on failure).
- Consensus semantics: majority of players connected at proposal time (snapshotted), proposer auto-yes, 120s expiry = rejection, one open proposal at a time, turn clock paused while open.
- `GET /overrides/schema` (static; fetch once and cache) and `GET /games/{game_id}/overrides/history` (bearer auth; `superseded` flag; render `ov_rewind` rows as visible markers, grey superseded rows; rewind depth is unrestricted by design).
- After an applied patch: any in-flight `SUBMIT_INPUT` may be rejected with a request-id mismatch error — clients already handle this; re-read the fresh `input_request` from the broadcast.
- Note that override negotiation state is NOT in `build_view()` output and proposals do not survive server restart (re-propose).

- [ ] **Step 2: Full test suite + lint**

Run: `PYTHONPATH=src uv run pytest tests/ -q` — expected: ALL PASS, no regressions.
Run: `uv run ruff check src/ && uv run black --check src/` — fix anything flagged.

- [ ] **Step 3: Spec self-check**

Re-read `docs/superpowers/specs/2026-08-05-consensus-overrides-design.md` section by section and confirm each maps to shipped code (single mutation path; append-only rewind; proposal on ManagedGame; snapshot threshold; expiry-as-rejection; WS-only propose/vote; clock pause; request-id bump; catalogue completeness; `set_life_counters` endgame both directions; `starting_life_counters` immutability; multi-piece handling; history masking; unrestricted rewind depth). Fix any gap before the final commit.

- [ ] **Step 4: Commit**

```bash
git add docs/CLIENT_INTEGRATION_GUIDE.md
git commit -m "Document consensus overrides in the client integration guide"
```
