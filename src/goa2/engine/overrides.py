"""Consensus-override op registry — the single mutation path for overrides.

Every override (live or replayed) goes through ``apply_override_decision``:
validate args -> snapshot -> apply -> revalidate the whole GameState (which
rebuilds the occupancy cache and re-unifies card/token references) -> on any
failure restore the snapshot and reject. Nothing else may mutate state for an
override; replay parity depends on this being the one code path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from goa2.domain.hex import Hex
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.models.effect import ActiveEffect
from goa2.domain.models.marker import MarkerType
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
    if (
        state.get_entity(BoardEntityID(args.entity_id)) is None
        and state.get_hero(HeroID(args.entity_id)) is None
    ):
        raise OverrideRejectedError(f"Unknown entity {args.entity_id!r}", code="unknown_entity")
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
        if state.phase == GamePhase.PLANNING:
            # advance() forbids PLANNING; resolve the game-over inline.
            from goa2.engine.handler import process_stack

            process_stack(state)
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
            state.execution_stack.append(FinalizeHeroTurnStep(hero_id=str(state.current_actor_id)))
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


# ---------------------------------------------------------------------------
# Card / marker / effect patch ops
# ---------------------------------------------------------------------------


class MoveCardArgs(BaseModel):
    hero_id: str
    card_id: str
    zone: Literal["hand", "discard", "played"]


def _detach_card(hero: Any, card_id: str) -> Any:
    """Remove the card from every live zone it occupies; return the Card."""
    for i, c in enumerate(hero.hand):
        if c.id == card_id:
            return hero.hand.pop(i)
    for i, c in enumerate(hero.discard_pile):
        if c.id == card_id:
            return hero.discard_pile.pop(i)
    for i, c in enumerate(hero.played_cards):
        if c is not None and c.id == card_id:
            hero.played_cards[i] = None
            return c
    if hero.current_turn_card and hero.current_turn_card.id == card_id:
        card = hero.current_turn_card
        hero.current_turn_card = None
        return card
    if hero.extra_turn_card and hero.extra_turn_card.id == card_id:
        card = hero.extra_turn_card
        hero.extra_turn_card = None
        return card
    return None


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
    if (
        state.get_hero(HeroID(args.target_id)) is None
        and state.get_entity(BoardEntityID(args.target_id)) is None
    ):
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
        raise OverrideRejectedError(f"Invalid effect payload: {exc}", code="invalid_args") from exc
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


# ---------------------------------------------------------------------------
# Unstick ops — repair control flow, not values
# ---------------------------------------------------------------------------


class NoArgs(BaseModel):
    pass


def _pending_step(state: GameState) -> Any:
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
        description=(
            "Unwind the wedged action through the normal abort path "
            "(defense/reaction sequences unwind correctly)."
        ),
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
