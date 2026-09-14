"""Turn and phase orchestration steps."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from goa2.domain.events import GameEvent, GameEventType, _hex_dict
from goa2.domain.models import ActionType, GamePhase, StepType
from goa2.domain.models.effect import DurationType
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.effect_manager import EffectManager
from goa2.engine.steps.base import GameStep, StepResult

logger = logging.getLogger(__name__)

ACTION_CONTEXT_KEYS = (
    "current_action_type",
    "current_card_id",
    "performing_card_id",
    "performing_card_owner_id",
    "reperforming_card_id",
)
ACTION_CONTEXT_STACK_KEY = "action_context_stack"


def push_action_context(
    context: dict[str, Any],
    *,
    action_type: ActionType,
    card_id: str,
    card_owner_id: str,
) -> None:
    stack = context.setdefault(ACTION_CONTEXT_STACK_KEY, [])
    stack.append(
        {key: {"present": key in context, "value": context.get(key)} for key in ACTION_CONTEXT_KEYS}
    )
    context["current_action_type"] = action_type
    context["current_card_id"] = card_id
    context["performing_card_id"] = card_id
    context["performing_card_owner_id"] = card_owner_id
    context["reperforming_card_id"] = card_id


def restore_action_context(context: dict[str, Any]) -> None:
    stack = context.get(ACTION_CONTEXT_STACK_KEY, [])
    if not stack:
        context.pop(ACTION_CONTEXT_STACK_KEY, None)
        return
    snapshot = stack.pop()
    for key in ACTION_CONTEXT_KEYS:
        saved = snapshot[key]
        if saved["present"]:
            context[key] = saved["value"]
        else:
            context.pop(key, None)
    if not stack:
        context.pop(ACTION_CONTEXT_STACK_KEY, None)


class FindNextActorStep(GameStep):
    """
    Triggers the Phase engine to identify the next active player.
    Used to chain turns together.
    """

    type: StepType = StepType.FIND_NEXT_ACTOR
    survives_action_abort: ClassVar[bool] = True

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        # Import internally to avoid circular dependency (steps <-> phases)
        from goa2.engine.phases import resolve_next_action

        logger.debug("   [LOOP] Finding next actor...")
        resolve_next_action(state)
        return StepResult(is_finished=True)


class FinalizeHeroTurnStep(GameStep):
    """
    Finalizes a hero's turn by moving their current card to the resolved dashboard.
    Activates any effects created by this card and clears the actor context.
    """

    type: StepType = StepType.FINALIZE_HERO_TURN
    hero_id: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import CheckLanePushStep, ReturnMinionToZoneStep

        hero = state.get_hero(HeroID(self.hero_id))
        if hero and hero.current_turn_card:
            card_id = hero.current_turn_card.id
            logger.debug(f"   [LOGIC] Finalizing turn for {self.hero_id}. Card moved to Resolved.")
            hero.resolve_current_card()

            # Activate all effects created by this card
            EffectManager.activate_effects_by_card(state, card_id)

        # Reset passive usage counters for all cards (they reset each turn)
        if hero:
            for card in hero.played_cards:
                if card and card.passive_uses_this_turn > 0:
                    card.passive_uses_this_turn = 0
            # Also reset ultimate card if present
            if hero.ultimate_card and hero.ultimate_card.passive_uses_this_turn > 0:
                hero.ultimate_card.passive_uses_this_turn = 0
            for effect in state.active_effects:
                if str(effect.source_id) == str(hero.id) and effect.passive_uses_this_turn > 0:
                    effect.passive_uses_this_turn = 0

        # A revealed guess card stays on the table until play moves past it.
        # Clearing on the guesser's own turn end is too early: that runs in the
        # same process_stack pass as the reveal, so the client would never get
        # a view containing it. Clear once a *different* hero finishes a turn.
        if state.card_guess and state.card_guess.get("guesser_id") != str(self.hero_id):
            state.card_guess = None
        if state.card_reveal and state.card_reveal.get("revealer_id") != str(self.hero_id):
            state.card_reveal = None

        # Clear transient context for the next actor
        context.clear()
        state.acting_piece_id = None
        state.current_actor_id = None
        state.resolution_owner_id = None

        return StepResult(
            is_finished=True,
            new_steps=[
                ReturnMinionToZoneStep(),
                CheckLanePushStep(),
                FindNextActorStep(),
            ],
            events=[
                GameEvent(
                    event_type=GameEventType.TURN_ENDED,
                    actor_id=self.hero_id,
                )
            ],
        )


class EndPhaseCleanupStep(GameStep):
    """
    Handles the non-combat cleanup of End Phase:
    Retrieve Cards, Clear Tokens, Level Up, Round Reset.
    """

    type: StepType = StepType.END_PHASE_CLEANUP
    survives_action_abort: ClassVar[bool] = True

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.phases import record_position_snapshot
        from goa2.engine.steps.cards import ResolveUpgradesStep

        logger.debug("   [CLEANUP] Processing End Phase Cleanup...")
        from goa2.engine.effect_manager import EffectManager

        # Expire THIS_ROUND items
        EffectManager.expire_effects(state, DurationType.THIS_ROUND)

        # Return all markers to supply (per board game rules)
        state.return_all_markers()
        logger.debug("   [CLEANUP] All markers returned to supply")

        # Cleanup stale items (lazy expiration for cards leaving play)
        EffectManager.cleanup_stale_effects(state)

        self._retrieve_cards(state)
        token_events = self._clear_tokens(state)
        token_events.extend(self._level_up(state))

        if state.pending_upgrades:
            logger.debug("   [PHASE] Level Up Phase started.")
            return StepResult(
                is_finished=True, new_steps=[ResolveUpgradesStep()], events=token_events
            )

        state.round += 1
        state.turn = 1
        record_position_snapshot(state)
        state.phase = GamePhase.PLANNING
        logger.debug(f"   [ROUND START] Round {state.round}, Turn {state.turn}")

        return StepResult(is_finished=True, events=token_events)

    def _retrieve_cards(self, state: GameState):
        for team in state.teams.values():
            for hero in team.heroes:
                # Expire (not merely deactivate) effects tied to each card before
                # retrieval. expire_by_card removes the effect from active_effects
                # AND clears the card's is_active flag, so cards don't return to
                # hand still marked active. (deactivate_effects_by_card only flips
                # effect.is_active and left both the effect lingering and the card
                # flag stuck True.) Token-bound PASSIVE effects have no
                # source_card_id and are untouched here — they persist via
                # cleanup_stale_effects. Any THIS_ROUND / THIS_TURN finishing steps
                # have already fired earlier in the End Phase, so the
                # premature-removal semantics of expire_by_card are correct.
                for card in hero.played_cards:
                    if card:
                        EffectManager.expire_by_card(state, card.id)
                if hero.current_turn_card:
                    EffectManager.expire_by_card(state, hero.current_turn_card.id)
                hero.retrieve_cards()

    def _clear_tokens(self, state: GameState) -> list[GameEvent]:
        from goa2.engine.steps.markers import _remove_token_from_board

        events: list[GameEvent] = []
        for token_list in state.token_pool.values():
            for token in token_list:
                if token.persists_end_of_round:
                    continue
                from_hex, removed_effects = _remove_token_from_board(state, str(token.id))
                if from_hex:
                    events.append(
                        GameEvent(
                            event_type=GameEventType.TOKEN_REMOVED,
                            actor_id=None,
                            target_id=str(token.id),
                            from_hex=_hex_dict(from_hex),
                            metadata={"effects_removed": removed_effects},
                        )
                    )
        return events

    def _level_up(self, state: GameState) -> list[GameEvent]:
        """
        Calculates gold spending and level increments.
        Rule 3.1: Costs follow cumulative table. Mandatory purchase.

        Note: Level 8 unlocks the ultimate card automatically (no upgrade choice).
        Only levels 2-7 count as pending upgrades requiring card selection.

        Returns any events emitted by `on_ultimate_unlocked` hooks.
        """
        LEVEL_COSTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
        any_level_ups = False
        unlock_events: list[GameEvent] = []

        for team in state.teams.values():
            for hero in team.heroes:
                upgrades_this_round = 0
                unlocked_ultimate = False
                while hero.level < 8:
                    next_level = hero.level + 1
                    cost = LEVEL_COSTS[next_level]
                    if hero.gold >= cost:
                        hero.gold -= cost
                        hero.level = next_level
                        any_level_ups = True

                        if next_level == 8:
                            # Level 8: Ultimate unlocks automatically (no upgrade choice)
                            unlocked_ultimate = True
                            if hero.ultimate_card:
                                logger.debug(
                                    f"   [ULTIMATE UNLOCKED] {hero.id} reached Level 8! "
                                    f"'{hero.ultimate_card.name}' is now active!"
                                )
                                # Call on_ultimate_unlocked if the effect supports it
                                if hero.ultimate_card.effect_id:
                                    from goa2.engine.effects import CardEffectRegistry

                                    ult_effect = CardEffectRegistry.get(
                                        hero.ultimate_card.effect_id
                                    )
                                    if ult_effect and hasattr(ult_effect, "on_ultimate_unlocked"):
                                        # Hooks may return GameEvents (or None).
                                        emitted = ult_effect.on_ultimate_unlocked(state, hero)
                                        if emitted:
                                            unlock_events.extend(emitted)
                            else:
                                logger.debug(f"   [LEVEL] {hero.id} reached Level 8!")
                        else:
                            # Levels 2-7: Count as pending upgrade (requires card choice)
                            upgrades_this_round += 1
                            logger.debug(f"   [LEVEL] {hero.id} reached Level {hero.level}!")
                    else:
                        break

                if upgrades_this_round > 0:
                    state.pending_upgrades[hero.id] = upgrades_this_round
                    state.upgrade_reveal_snapshot[hero.id] = dict(hero.items)
                elif not unlocked_ultimate:
                    # Pity Coin: Players who did not Level Up gain 1 Gold.
                    # (Don't give pity coin if they unlocked ultimate)
                    hero.gold += 1
                    logger.debug(
                        f"   [ECONOMY] {hero.id} did not level up. Gains 1 Pity Gold. (Gold: {hero.gold})"
                    )

        if any_level_ups:
            state.phase = GamePhase.LEVEL_UP

        return unlock_events


class EndPhaseStep(GameStep):
    """
    Entry point for End Phase.
    Expires THIS_ROUND effects (with finishing steps), then queues
    MinionBattleStep, CheckLanePushStep, and EndPhaseCleanupStep.
    """

    type: StepType = StepType.END_PHASE
    survives_action_abort: ClassVar[bool] = True

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import CheckLanePushStep, MinionBattleStep
        from goa2.engine.steps.effects import FinishedExpiringEffectStep
        from goa2.engine.steps.utility import SetActorStep

        logger.debug("   [ROUND END] Processing End Phase (Battle)...")

        # Expire THIS_ROUND effects and collect finishing steps
        finishing = EffectManager.expire_effects(state, DurationType.THIS_ROUND)

        new_steps: list[GameStep] = []

        # Inject finishing steps (with SetActorStep wrappers) before battle
        for source_id, steps in finishing:
            new_steps.append(SetActorStep(actor_id=source_id))
            new_steps.extend(steps)
            new_steps.append(FinishedExpiringEffectStep())

        # Minion battle now computed lazily (after finishing steps execute)
        new_steps.append(MinionBattleStep())

        # CheckLanePushStep handles the case where one team already has 0 minions
        new_steps.append(CheckLanePushStep())

        new_steps.append(EndPhaseCleanupStep())

        return StepResult(is_finished=True, new_steps=new_steps)


class AdvanceTurnStep(GameStep):
    """
    Handles turn advancement after finishing steps have executed.
    Encapsulates the logic from end_turn() so it can be deferred
    when finishing steps need to run first.
    """

    type: StepType = StepType.ADVANCE_TURN
    survives_action_abort: ClassVar[bool] = True

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.phases import (
            _check_phase_transition,
            record_position_snapshot,
            start_end_phase,
        )

        if state.turn < 4:
            state.turn += 1
            record_position_snapshot(state)
            state.phase = GamePhase.PLANNING
            logger.debug(f"   [Turn] Start of Turn {state.turn}. Phase: PLANNING")
            auto_passed = False
            for team in state.teams.values():
                for hero in team.heroes:
                    if len(hero.hand) == 0:
                        state.pending_inputs[hero.id] = None
                        logger.debug(f"   [Planning] {hero.id} auto-passed (empty hand).")
                        auto_passed = True
            if auto_passed:
                _check_phase_transition(state)
        else:
            start_end_phase(state)
        return StepResult(is_finished=True)


class RestoreActionTypeStep(GameStep):
    """
    Restores the previous action type from the stack after defense resolution.

    Used after defense effects complete to restore the original action type
    (e.g., ATTACK) so that any subsequent effects are correctly attributed.
    """

    type: StepType = StepType.RESTORE_ACTION_TYPE

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        stack = context.get("action_type_stack", [])
        if stack:
            previous_type = stack.pop()
            context["current_action_type"] = previous_type
            logger.debug(f"   [CONTEXT] Restored action type to {previous_type.name}")
        return StepResult(is_finished=True)


class RestoreActionContextStep(GameStep):
    """Restore the outer card/action source after a nested performed action."""

    type: StepType = StepType.RESTORE_ACTION_CONTEXT

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        restore_action_context(context)
        return StepResult(is_finished=True)
