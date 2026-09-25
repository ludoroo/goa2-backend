from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import goa2.engine.step_types as _step_types  # noqa: F401 — patches model annotations
from goa2.domain.events import GameEvent
from goa2.domain.input import InputRequest, InputResponse
from goa2.domain.models import GamePhase
from goa2.domain.models.effect import EffectType
from goa2.domain.state import GameState
from goa2.engine.steps import FinishedExpiringEffectStep, GameStep, StepResult

logger = logging.getLogger(__name__)


@dataclass
class StackResult:
    """Bundles the result of processing the execution stack."""

    input_request: InputRequest | None = None
    events: list[GameEvent] = field(default_factory=list)


def submit_input(state: GameState, response: InputResponse | dict) -> None:
    """Validate and apply player input to the pending step on the stack."""
    if not state.execution_stack:
        raise ValueError("No pending step to receive input")

    current_step = state.execution_stack[-1]

    if isinstance(response, InputResponse):
        expected_request_id = current_step.pending_request_id
        if expected_request_id is None:
            raise ValueError("The pending step has not issued an input request")
        if response.request_id != expected_request_id:
            raise ValueError(
                f"Input response request_id {response.request_id!r} does not match "
                f"pending request {expected_request_id!r}"
            )
        current_step.pending_input = {"selection": response.selection}
    else:
        current_step.pending_input = response
    current_step.pending_request_id = None


def process_stack(
    state: GameState,
    *,
    stop_before_step: Callable[[GameState, GameStep], bool] | None = None,
) -> StackResult:
    """Process the stack, optionally stopping before an internal boundary step."""
    safety_counter = 0
    MAX_STEPS = 1000
    collected_events: list[GameEvent] = []

    if state.phase == GamePhase.GAME_OVER:
        return StackResult()

    while state.execution_stack:
        if stop_before_step is not None and stop_before_step(state, state.execution_stack[-1]):
            return StackResult(events=collected_events)
        safety_counter += 1
        if safety_counter > MAX_STEPS:
            raise RuntimeError("Infinite Loop detected in Engine Resolution Stack")

        current_step: GameStep = state.execution_stack.pop()

        # Centralized skip check — steps no longer need to call should_skip() themselves
        if current_step.should_skip(state.execution_context):
            continue

        result: StepResult = current_step.resolve(state, state.execution_context)

        # Collect events even from aborted steps
        collected_events.extend(result.events)

        if result.abort_action:
            _clear_after_abort(state)
            continue

        if result.requires_input:
            state.execution_stack.append(current_step)
            request = result.input_request
            if request is not None:
                if current_step.pending_request_id is None:
                    current_step.pending_request_id = request.id
                else:
                    request.id = current_step.pending_request_id
            controller_id = _action_controller(state, request.player_id) if request else None
            if request is not None and controller_id is not None:
                request.context["controlled_hero_id"] = request.player_id
                request.player_id = controller_id
            return StackResult(input_request=request, events=collected_events)

        if not result.is_finished:
            state.execution_stack.append(current_step)

        if result.new_steps:
            state.execution_stack.extend(reversed(result.new_steps))

    return StackResult(events=collected_events)


def _clear_after_abort(state: GameState):
    """Clear skipped work after a mandatory step aborts.

    If the abort happened during a defense/reaction sequence before combat has
    resolved, keep the actor-restore step immediately before ResolveCombatStep
    so the attack still resolves with the selected defense value.
    """
    if _clear_to_pending_combat(state):
        return
    if _clear_to_finalize(state):
        # Stopped at another hero's nested action, whose RestoreActionContextStep
        # unwinds its own level of the context stack when it resolves.
        return
    from goa2.engine.steps.phases import restore_action_context

    while state.execution_context.get("action_context_stack"):
        restore_action_context(state.execution_context)


def _clear_to_pending_combat(state: GameState) -> bool:
    """Skip remaining pre-combat reaction work while preserving combat.

    Defense text runs with ``current_actor_id`` temporarily set to the defender.
    The stack contains a ``SetActorStep(actor_key="_pre_defense_actor")`` just
    above ``ResolveCombatStep`` to restore the attacker. Stop there when it is
    present; otherwise stop at ``ResolveCombatStep`` itself.
    """
    if not state.execution_context.get("defense_card_id") or not state.execution_context.get(
        "is_primary_defense"
    ):
        return False

    from goa2.engine.steps import ConfirmResolutionStep, FinalizeHeroTurnStep
    from goa2.engine.steps.combat import ResolveCombatStep
    from goa2.engine.steps.utility import SetActorStep

    combat_index: int | None = None
    for index in range(len(state.execution_stack) - 1, -1, -1):
        step = state.execution_stack[index]
        if (
            isinstance(
                step, (ConfirmResolutionStep, FinalizeHeroTurnStep, FinishedExpiringEffectStep)
            )
            or step.survives_action_abort
        ):
            break
        if isinstance(step, ResolveCombatStep):
            combat_index = index
            break

    if combat_index is None:
        return False

    target_index = combat_index
    if combat_index + 1 < len(state.execution_stack):
        maybe_restore = state.execution_stack[combat_index + 1]
        if (
            isinstance(maybe_restore, SetActorStep)
            and maybe_restore.actor_key == "_pre_defense_actor"
        ):
            target_index = combat_index + 1

    while len(state.execution_stack) - 1 > target_index:
        step = _discard_aborted_step(state)
        logger.debug("Skipped step before pending combat: %s", step.type)
    return True


def _discard_aborted_step(state: GameState) -> GameStep:
    """Discard work while still unwinding any nested action it would restore."""
    from goa2.engine.steps.phases import RestoreActionContextStep, restore_action_context

    step = state.execution_stack.pop()
    if isinstance(step, RestoreActionContextStep):
        restore_action_context(state.execution_context)
    return step


def _clear_to_finalize(state: GameState) -> bool:
    """
    Clears all steps from the stack until ConfirmResolutionStep or FinalizeHeroTurnStep is found.
    Stops at ConfirmResolutionStep so the player can review the abort and optionally rollback.

    Returns True when it stopped at another hero's nested action, i.e. only that
    nested action was cancelled and the outer hero's action resumes.
    """
    from goa2.engine.steps import ConfirmResolutionStep, FinalizeHeroTurnStep
    from goa2.engine.steps.phases import RestoreActionContextStep

    while state.execution_stack:
        step = state.execution_stack[-1]
        if isinstance(step, RestoreActionContextStep) and step.other_hero_action:
            return True
        if (
            isinstance(
                step, (ConfirmResolutionStep, FinalizeHeroTurnStep, FinishedExpiringEffectStep)
            )
            or step.survives_action_abort
        ):
            break
        _discard_aborted_step(state)
        logger.debug("Skipped step: %s", step.type)
    return False


def _action_controller(state: GameState, player_id: str) -> str | None:
    """The Ultimate Trick (Hanu): if `player_id` is resolving their own turn on
    the card an active CONTROL_NEXT_ACTION effect targets, return the
    controlling hero's id. The remap changes only who answers — options/legality
    were already computed relative to the actor.

    Control covers that hero's whole turn window, including nested actions their
    card spawns (NebKher performing a card in another slot, Gydion casting a
    spell) and prompts outside any card (respawn placement, passive confirms) —
    hence the turn card, not the card currently being performed. It requires
    ``resolution_owner_id`` to be them as well: an action forced on them during
    someone else's turn (Whisper making the defender move on their defense card)
    is not their controlled action, so they answer it themselves.
    """
    if state.current_actor_id is None or player_id != str(state.current_actor_id):
        return None
    if state.resolution_owner_id is None or player_id != str(state.resolution_owner_id):
        return None
    hero = state.get_hero(state.current_actor_id)
    if hero is None or hero.current_turn_card is None:
        return None
    for effect in state.active_effects:
        if (
            effect.effect_type == EffectType.CONTROL_NEXT_ACTION
            and effect.is_active
            and effect.scope.origin_id == player_id
            and effect.controlled_card_id == hero.current_turn_card.id
        ):
            return effect.source_id
    return None


def push_steps(state: GameState, steps: list[GameStep]):
    """Helper to push new steps. Note: Stack is LIFO, so we extend in REVERSE order if we want them to execute 1, 2, 3."""
    # If we want Step 1 to run first, it must be at the TOP (end of list).
    # So if we have [S1, S2, S3], we should push S3, then S2, then S1.
    state.execution_stack.extend(reversed(steps))
