"""Card resolution, discard, retrieval, upgrades, and economy steps."""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import Field, model_validator

from goa2.domain.events import GameEvent, GameEventType
from goa2.domain.input import InputRequestType, create_input_request
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardContainerType,
    CardState,
    CardTier,
    GamePhase,
    Hero,
    StepType,
    TargetType,
    TeamColor,
    TokenType,
)
from goa2.domain.models.effect import EffectType
from goa2.domain.models.enums import PassiveTrigger, StatType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID, UnitID
from goa2.engine import rules
from goa2.engine.filters_hex import RangeFilter
from goa2.engine.filters_units import ExcludeIdentityFilter, ImmunityFilter, UnitTypeFilter
from goa2.engine.stats import get_computed_stat
from goa2.engine.steps.base import GameStep, StepResult

logger = logging.getLogger(__name__)


def action_passes_initial_target_gate(
    state: GameState, hero: Hero, card: Card, act_type: ActionType, *, is_primary: bool
) -> bool:
    """Whether the action's initial mandatory target gate can succeed.

    Returns False only when the first top-level mandatory target gate (a
    SelectStep, or an AttackSequenceStep that would spawn one) has no valid
    candidate in the current state. Used to prune guaranteed no-op options from
    the CHOOSE_ACTION menu.

    Conservative by design: anything unexpected returns True (never hide a legal
    move). Does not recurse into CreateEffectStep.finishing_steps (deferred
    end-of-turn targets like Silverarrow's warning_shot stay available).
    """
    from goa2.engine.effects import CardEffectRegistry
    from goa2.engine.filters_units import TeamFilter
    from goa2.engine.steps.combat import AttackSequenceStep
    from goa2.engine.steps.selection import SelectStep

    try:
        # The acting piece is chosen after the action menu. Until then, target
        # legality and per-piece action restrictions cannot be evaluated from a
        # single origin safely, so never prune a multi-piece hero here.
        if hero.is_multi_piece and not state.acting_piece_id:
            return True

        # Source the top-level steps this action would expand into.
        steps: list[GameStep]
        effect_id = card.current_effect_id if is_primary else None
        effect = CardEffectRegistry.get(effect_id) if effect_id else None
        if effect is not None:
            steps = list(effect.get_steps(state, hero, card))
        elif effect_id is not None:
            # Scripted card whose effect isn't registered: we can't know its
            # step shape, so never prune (conservative).
            return True
        elif act_type == ActionType.ATTACK:
            # Standard / basic / secondary ATTACK: a bare AttackSequenceStep.
            base_rng = card.get_base_stat_value(StatType.RANGE) or 1
            eff_rng = get_computed_stat(state, UnitID(hero.id), StatType.RANGE, base_rng)
            steps = [AttackSequenceStep(damage=0, range_val=eff_rng)]
        else:
            # SKILL with no registered effect does nothing meaningful; leave it.
            return True

        # Walk to the first unconditional gating step.
        for step in steps:
            if step.should_skip(state.execution_context):
                continue
            if isinstance(step, SelectStep):
                if not step.is_mandatory:
                    return True  # optional target -> action still playable
                if step.target_type == TargetType.NUMBER:
                    return True  # mode choice, not a target gate
                return step.has_valid_candidate(state, state.execution_context)
            if isinstance(step, AttackSequenceStep):
                if step.target_id_key or not step.is_mandatory:
                    return True  # target pre-selected, or optional
                probe = SelectStep(
                    target_type=TargetType.UNIT,
                    prompt="",
                    filters=[
                        RangeFilter(max_range=step.range_val),
                        TeamFilter(relation="ENEMY"),
                        *step.target_filters,
                    ],
                )
                return probe.has_valid_candidate(state, state.execution_context)
            # Any other leading step (aura, self, setup, movement, etc.) means
            # there is no target gate up front -> not prunable.
            return True
        return True
    except Exception:  # pragma: no cover - never hide a legal move on error
        logger.exception("initial target gate probe failed; leaving option available")
        return True


def action_may_change_targets_before_card_text(
    state: GameState, hero: Hero, act_type: ActionType, *, is_primary: bool
) -> bool:
    """Whether a hook before card text could change target availability.

    The menu is built before these hooks resolve. If one is eligible, probing
    the current board could hide an action that becomes targetable after the
    hook, so pruning must remain conservative.
    """
    from goa2.engine.effects import CardEffectRegistry

    if is_primary and any(
        effect.effect_type == EffectType.PRE_ACTION_MOVEMENT
        and effect.source_id == hero.id
        and effect.is_active
        for effect in state.active_effects
    ):
        return True

    triggers = {PassiveTrigger.BEFORE_ACTION}
    if act_type == ActionType.ATTACK:
        triggers.add(PassiveTrigger.BEFORE_ATTACK)
    elif act_type == ActionType.SKILL:
        triggers.add(PassiveTrigger.BEFORE_SKILL)

    passive_cards = [
        card
        for card in hero.played_cards
        if card is not None and card.state == CardState.RESOLVED and not card.is_facedown
    ]
    if hero.level >= 8 and hero.ultimate_card:
        passive_cards.append(hero.ultimate_card)

    for passive_card in passive_cards:
        effect_id = passive_card.current_effect_id
        effect = CardEffectRegistry.get(effect_id) if effect_id else None
        if not effect:
            continue
        for config in effect.get_passive_configs():
            if config.trigger not in triggers:
                continue
            if (
                config.uses_per_turn <= 0
                or passive_card.passive_uses_this_turn < config.uses_per_turn
            ):
                return True
    return False


class SetCardInitiativeStep(GameStep):
    """Hanu's Hurry Up!: set the printed Initiative of a target hero's unresolved
    current_turn_card to ``value`` (11), overriding only the BASE value so items
    and other Initiative modifiers still stack via ``get_computed_stat``. Records
    the original and schedules an end-of-turn restore (THIS_TURN DELAYED_TRIGGER)
    so the printed value returns "once it is resolved or otherwise changes
    state".
    """

    type: StepType = StepType.SET_CARD_INITIATIVE
    hero_key: str
    value: int = 11

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        hero_id = context.get(self.hero_key)
        if not hero_id:
            return StepResult(is_finished=True)
        hero = state.get_hero(HeroID(str(hero_id)))
        if not hero or hero.current_turn_card is None:
            return StepResult(is_finished=True)

        card = hero.current_turn_card
        if card.state != CardState.UNRESOLVED:
            return StepResult(is_finished=True)

        original = card.initiative
        card.initiative = self.value

        from goa2.domain.models import DurationType
        from goa2.domain.models.effect import EffectScope, EffectType, Shape
        from goa2.engine.effect_manager import EffectManager

        EffectManager.create_effect(
            state=state,
            source_id=str(state.current_actor_id) if state.current_actor_id else "system",
            effect_type=EffectType.DELAYED_TRIGGER,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            is_active=True,
            finishing_steps=[
                RestoreCardInitiativeStep(card_id=card.id, original_initiative=original)
            ],
        )

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.EFFECT_CREATED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(hero_id),
                    metadata={
                        "effect": "initiative_override",
                        "card_id": card.id,
                        "value": self.value,
                    },
                )
            ],
        )


class RestoreCardInitiativeStep(GameStep):
    """Restores a card's printed Initiative to its original value (end-of-turn
    payload scheduled by ``SetCardInitiativeStep``)."""

    type: StepType = StepType.RESTORE_CARD_INITIATIVE
    card_id: str
    original_initiative: int

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        card = state.get_card_by_id(self.card_id)
        if card is not None:
            card.initiative = self.original_initiative
        return StepResult(is_finished=True)


class DiscardCardStep(GameStep):
    """
    Forces a specific card to be discarded.
    """

    type: StepType = StepType.DISCARD_CARD
    card_id: str | None = None
    card_key: str | None = None
    hero_id: str | None = None
    hero_key: str | None = None
    source: CardContainerType = CardContainerType.HAND  # HAND or PLAYED

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.effects import CheckPassiveAbilitiesStep

        if self.should_skip(context):
            return StepResult(is_finished=True)

        # Resolve Hero
        h_id = self.hero_id
        if not h_id and self.hero_key:
            h_id = context.get(self.hero_key)

        if not h_id:
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(h_id)))
        if not hero:
            return StepResult(is_finished=True)
        owner_id = HeroID(str(hero.id))

        # Resolve Card
        c_id = self.card_id
        if not c_id and self.card_key:
            c_id = context.get(self.card_key)

        if not c_id:
            return StepResult(is_finished=True)

        # Find card in the specified source container.
        def _find(container: CardContainerType):
            if container == CardContainerType.HAND:
                return next((c for c in hero.hand if c.id == c_id), None)
            if container == CardContainerType.PLAYED:
                # The "played" area includes the current_turn_card (a card created
                # this turn that has not yet resolved into played_cards). A
                # discard-shield can be the current_turn_card during its own turn,
                # so it must be findable here for the forced-discard redirect.
                found = next((c for c in hero.played_cards if c is not None and c.id == c_id), None)
                if (
                    found is None
                    and hero.current_turn_card is not None
                    and hero.current_turn_card.id == c_id
                ):
                    found = hero.current_turn_card
                return found
            return None

        actual_source = self.source
        target_card = _find(self.source)
        if not target_card:
            # Auto-detect across hand/played: a forced hand-discard may be
            # redirected onto a played discard-shield card (Mrak), so the chosen
            # card can live in the other container.
            for alt in (CardContainerType.HAND, CardContainerType.PLAYED):
                if alt == self.source:
                    continue
                target_card = _find(alt)
                if target_card:
                    actual_source = alt
                    break

        if not target_card:
            logger.debug(f"   [DISCARD] Card {c_id} not found in {h_id}'s hand/played.")
            return StepResult(is_finished=True)

        logger.debug(f"   [DISCARD] {h_id} discards {target_card.name}")

        # Determine the discard source BEFORE discarding (discard_card mutates the
        # card state to DISCARD). A card found in the played area that is not yet
        # RESOLVED is the current_turn_card — still being played this turn — so it
        # is reported as "current_turn", NOT PLAYED. Resolved-card-discard passives
        # (Garrus's Battle Fury) gate on PLAYED meaning a *resolved* card.
        if actual_source == CardContainerType.HAND:
            discard_source = CardContainerType.HAND.value
        elif target_card.state == CardState.RESOLVED:
            discard_source = CardContainerType.PLAYED.value
        else:
            discard_source = "current_turn"

        hero.discard_card(target_card, from_hand=(actual_source == CardContainerType.HAND))
        # The discard pile is public regardless of the card's previous face.
        # Record only after the lifecycle transition succeeds.
        state.record_public_revealed_card(owner_id, str(target_card.id))

        # Record in the turn-scoped discard log (cleared at end_turn); read by
        # "retrieve all cards discarded this turn" effects (Emmitt).
        state.turn_discard_log.setdefault(owner_id, []).append(target_card.id)

        # Changing a card's state cancels its active effect (premature end, so
        # finishing_steps do not run). Harmless no-op for hand cards.
        from goa2.engine.effect_manager import EffectManager

        EffectManager.expire_by_card(state, target_card.id)

        # Fire AFTER_CARD_DISCARD passive trigger for every discard.
        # Passives that only care about specific sources (e.g. Battle Fury, which
        # only triggers on discards of resolved cards) filter via discard_source.
        from goa2.domain.models.enums import PassiveTrigger

        context["discarded_card_id"] = target_card.id
        context["discarded_card_owner_id"] = str(owner_id)
        context["discard_source"] = discard_source
        return StepResult(
            is_finished=True,
            new_steps=[
                CheckPassiveAbilitiesStep(
                    trigger=PassiveTrigger.AFTER_CARD_DISCARD.value,
                    hero_id=str(h_id),
                )
            ],
        )


class ResolvePreActionDiscardStep(GameStep):
    """
    Checks for active PRE_ACTION_DISCARD effects (Trinkets - Disruptor family)
    that affect a hero about to perform a primary action.

    For each matching effect (one per resolve, re-checking state in between):
    - If the hero has cards: expire (consume) the effect and force a discard.
      Removing it untaps the source card, whose is_active flag tracks effect
      existence.
    - If the hero has no cards and the effect is discard_or_defeat: defeat
      the hero (the effect stays active — only a discard consumes it).
    - Otherwise ("if able"): nothing happens and the effect stays active.

    If the hero leaves the board mid-resolution (defeated by a disruptor),
    the remaining action is aborted.
    """

    type: StepType = StepType.RESOLVE_PRE_ACTION_DISCARD
    hero_id: str | None = None
    hero_key: str | None = None
    # When the victim is defeated by a disruptor (no cards to discard), abort the
    # current action. Correct on the victim's OWN turn; must be False when the
    # victim is a defender (aborting would wrongly cancel the attacker's turn).
    abort_on_defeat: bool = True
    processed_effect_ids: list[str] = Field(default_factory=list)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.domain.models.effect import EffectType
        from goa2.engine.effect_manager import EffectManager
        from goa2.engine.rules import unit_ignores_effect_due_to_immunity
        from goa2.engine.stats import _is_effect_active, is_unit_in_effect_scope
        from goa2.engine.steps.combat import DefeatUnitStep

        if self.should_skip(context):
            return StepResult(is_finished=True)

        hero_id = self.hero_id
        if not hero_id and self.hero_key:
            hero_id = context.get(self.hero_key)
        if not hero_id:
            return StepResult(is_finished=True)
        hero_id = str(hero_id)

        hero = state.get_hero(HeroID(hero_id))
        if not hero:
            return StepResult(is_finished=True)

        if not state.has_board_presence(hero_id):
            # Hero left the board (defeated by a previous disruptor trigger).
            if self.abort_on_defeat:
                logger.debug(f"   [DISRUPTOR] {hero_id} left the board. Aborting action.")
                return StepResult(is_finished=True, abort_action=True)
            return StepResult(is_finished=True)

        board_actor_id = state.resolve_board_actor(hero_id)

        for effect in state.active_effects:
            if effect.effect_type != EffectType.PRE_ACTION_DISCARD:
                continue
            if effect.id in self.processed_effect_ids:
                continue
            if not _is_effect_active(effect, state):
                continue
            if not is_unit_in_effect_scope(effect, board_actor_id, state):
                continue
            if unit_ignores_effect_due_to_immunity(effect, board_actor_id, state):
                continue

            self.processed_effect_ids.append(effect.id)

            if hero.hand:
                # A discard will definitely happen — the disruptor is spent.
                # Expire (remove) it rather than just deactivating: the card's
                # is_active flag tracks effect existence, so a merely-deactivated
                # effect would leave the Trinkets card marked active (tapped) for
                # the rest of the round. Removal untaps the card and also stops
                # the disruptor from re-firing regardless of card linkage.
                EffectManager.expire_effect_by_id(state, effect.id)
                context["pre_action_discard_victim"] = hero_id
                logger.debug(f"   [DISRUPTOR] {hero_id} must discard before primary action.")
                return StepResult(
                    is_finished=False,
                    new_steps=[ForceDiscardStep(victim_key="pre_action_discard_victim")],
                )

            if effect.discard_or_defeat:
                logger.debug(f"   [DISRUPTOR] {hero_id} has no cards to discard! DEFEATED!")
                # Credit the defeat to the hero who created the disruptor
                # (effect.source_id == Trinkets), NOT current_actor_id — the
                # current actor here is the victim taking their own action.
                return StepResult(
                    is_finished=False,
                    new_steps=[
                        DefeatUnitStep(victim_id=hero_id, killer_id=effect.source_id),
                    ],
                )

            logger.debug(f"   [DISRUPTOR] {hero_id} has no cards to discard (Safe).")

        return StepResult(is_finished=True)


class ForceDiscardStep(GameStep):
    """
    Checks if a victim has cards.
    If YES: Spawns a SelectStep (for victim to choose) + DiscardCardStep.
    If NO: Completes successfully (no penalty).

    Victim resolves from a non-empty ``victim_id`` literal, else from
    ``victim_key`` in context. At least one must be non-empty. The resolved
    victim is snapshotted into the emitted child steps as literals so a later
    shared-context write cannot re-route the discard prompt.
    """

    type: StepType = StepType.FORCE_DISCARD
    victim_id: str | None = None
    victim_key: str | None = None
    card_is_basic: bool | None = None
    immunity_source_id: str | None = None

    @model_validator(mode="after")
    def _require_victim_source(self) -> ForceDiscardStep:
        if not self.victim_id and not self.victim_key:
            raise ValueError("ForceDiscardStep requires a non-empty victim_id or victim_key")
        return self

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.selection import SelectStep

        # Non-empty literal wins; empty literal falls back to key lookup.
        resolved_victim: str | None = self.victim_id or None
        if not resolved_victim and self.victim_key:
            ctx_val = context.get(self.victim_key)
            resolved_victim = str(ctx_val) if ctx_val else None
        if not resolved_victim:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(resolved_victim)))
        if not victim:
            return StepResult(is_finished=True)
        if self.immunity_source_id and rules.is_immune_to_actor(
            victim, state, actor_id=self.immunity_source_id
        ):
            logger.debug(
                "   [EFFECT] %s is immune to forced discard from %s.",
                resolved_victim,
                self.immunity_source_id,
            )
            return StepResult(is_finished=True)

        eligible_hand = [
            card
            for card in victim.hand
            if self.card_is_basic is None or card.is_basic == self.card_is_basic
        ]
        if not eligible_hand:
            logger.debug(f"   [EFFECT] {resolved_victim} has no matching cards to discard (Safe).")
            return StepResult(is_finished=True)

        # Mrak's discard-shield: a forced HAND discard may be redirected onto a
        # played shield card instead. Offer it alongside the hand.
        from goa2.engine.effects import get_active_shield_cards

        shield_cards = get_active_shield_cards(state, victim)
        if shield_cards:
            allowed_ids = None
            if self.card_is_basic is not None:
                allowed_ids = [card.id for card in eligible_hand]
                allowed_ids.extend(card.id for card in shield_cards)
            select = SelectStep(
                target_type=TargetType.CARD,
                prompt=f"{resolved_victim}, select a card to discard.",
                output_key="card_to_discard",
                card_containers=[CardContainerType.HAND, CardContainerType.PLAYED],
                restrict_played_to_shields=True,
                allowed_card_ids=allowed_ids,
                context_hero_id=resolved_victim,
                override_player_id=resolved_victim,
                is_mandatory=True,
            )
        else:
            select = SelectStep(
                target_type=TargetType.CARD,
                prompt=f"{resolved_victim}, select a card to discard.",
                output_key="card_to_discard",
                card_container=CardContainerType.HAND,
                card_is_basic=self.card_is_basic,
                context_hero_id=resolved_victim,  # Look at victim's hand
                override_player_id=resolved_victim,  # Victim chooses
                is_mandatory=True,
            )

        # Has cards -> Force Discard
        return StepResult(
            is_finished=True,
            new_steps=[
                select,
                DiscardCardStep(card_key="card_to_discard", hero_id=resolved_victim),
            ],
        )


class ForceDiscardByColorStep(GameStep):
    """
    Forces a hero to discard a card matching a named color, if able.

    The victim chooses which matching card to discard. If their hand has no
    card of that color, the step completes without penalty.
    """

    type: StepType = StepType.FORCE_DISCARD_BY_COLOR
    victim_key: str
    color: CardColor | None = None
    color_key: str | None = None
    output_key: str = "card_to_discard"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.selection import SelectStep

        victim_id = context.get(self.victim_key)
        if not victim_id:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(victim_id)))
        if not victim:
            return StepResult(is_finished=True)

        color = self.color
        if color is None and self.color_key:
            color_val = context.get(self.color_key)
            if color_val:
                color = CardColor(str(color_val))

        if color is None:
            return StepResult(is_finished=True)

        if not any(c.color == color for c in victim.hand):
            logger.debug(f"   [EFFECT] {victim_id} has no {color.value} card to discard.")
            return StepResult(is_finished=True)

        return StepResult(
            is_finished=True,
            new_steps=[
                SelectStep(
                    target_type=TargetType.CARD,
                    prompt=f"{victim_id}, select a {color.value} card to discard.",
                    output_key=self.output_key,
                    card_container=CardContainerType.HAND,
                    context_hero_id_key=self.victim_key,
                    override_player_id_key=self.victim_key,
                    card_colors=[color],
                    is_mandatory=True,
                ),
                DiscardCardStep(card_key=self.output_key, hero_key=self.victim_key),
            ],
        )


class ForceDiscardOrDefeatStep(GameStep):
    """
    Checks if a victim has cards.
    If YES: Spawns a SelectStep (for victim to choose) + DiscardCardStep.
    If NO: Spawns DefeatUnitStep (the penalty for not discarding).

    Kill attribution: by default the defeat is credited to the current actor,
    which is correct for every caller that runs during its own action chain.
    Callers where the actor is NOT the source (e.g. an effect that fires on the
    victim's own turn) can override via killer_id / killer_key.
    """

    type: StepType = StepType.FORCE_DISCARD_OR_DEFEAT
    victim_key: str
    killer_id: str | None = None  # Literal override for kill credit
    killer_key: str | None = None  # Context key override for kill credit
    immunity_source_id: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import DefeatUnitStep
        from goa2.engine.steps.selection import SelectStep

        victim_id = context.get(self.victim_key)
        if not victim_id:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(victim_id)))
        if not victim:
            return StepResult(is_finished=True)
        if self.immunity_source_id and rules.is_immune_to_actor(
            victim, state, actor_id=self.immunity_source_id
        ):
            logger.debug(
                "   [EFFECT] %s is immune to discard-or-defeat from %s.",
                victim_id,
                self.immunity_source_id,
            )
            return StepResult(is_finished=True)

        victim_entity_id = BoardEntityID(str(victim_id))
        is_owner_level_multipiece_victim = (
            str(victim.id) == str(victim_id)
            and victim.is_multi_piece
            and state.has_board_presence(str(victim.id))
        )
        if victim_entity_id not in state.entity_locations and not is_owner_level_multipiece_victim:
            logger.debug(
                "   [EFFECT] Skipping discard-or-defeat for off-board victim %s.", victim_id
            )
            return StepResult(is_finished=True)

        if not victim.hand:
            logger.debug(f"   [EFFECT] {victim_id} has no cards to discard! DEFEATED!")
            # Credit the kill. Prefer an explicit override; otherwise fall back to
            # the current actor, which is the source for every caller that runs
            # during its own action chain. (An effect that fires on the victim's
            # own turn must override, since current_actor would be the victim.)
            killer_id = self.killer_id
            if not killer_id and self.killer_key:
                ctx_killer = context.get(self.killer_key)
                killer_id = str(ctx_killer) if ctx_killer else None
            if not killer_id:
                killer_id = str(state.current_actor_id) if state.current_actor_id else None
            return StepResult(
                is_finished=True,
                new_steps=[DefeatUnitStep(victim_id=str(victim_id), killer_id=killer_id)],
            )

        # Has cards -> Force Discard
        return StepResult(
            is_finished=True,
            new_steps=[
                SelectStep(
                    target_type=TargetType.CARD,
                    prompt=f"{victim_id}, select a card to discard (or be Defeated).",
                    output_key="card_to_discard",
                    card_container=CardContainerType.HAND,
                    context_hero_id_key=self.victim_key,  # Look at victim's hand
                    override_player_id_key=self.victim_key,  # Victim chooses
                    is_mandatory=True,
                ),
                DiscardCardStep(card_key="card_to_discard", hero_key=self.victim_key),
            ],
        )


class ResolveCardTextStep(GameStep):
    """
    Placeholder for executing the specific Python script/logic associated with a card's text.
    In a full implementation, this would look up a registry using `card.effect_id`
    and execute the specific function/class for that card.
    """

    type: StepType = StepType.RESOLVE_CARD_TEXT
    card_id: str
    hero_id: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import AttackSequenceStep
        from goa2.engine.steps.movement import MoveSequenceStep
        from goa2.engine.steps.utility import LogMessageStep

        hero = state.get_hero(HeroID(self.hero_id))
        if not hero or not hero.current_turn_card:
            return StepResult(is_finished=True)

        card = hero.current_turn_card

        # Set card ID in context for effect creation
        context["current_card_id"] = card.id

        logger.debug(
            f"   [SCRIPT] Executing logic for '{card.name}' (Effect: {card.current_effect_id})"
        )

        from goa2.engine.effects import CardEffectRegistry

        if card.current_effect_id is None:
            return StepResult(is_finished=True)

        effect = CardEffectRegistry.get_for_card(card)

        if effect:
            # We must use a different variable name here or not declare `new_steps` again below
            effect_steps = effect.get_steps(state, hero, card)
            return StepResult(is_finished=True, new_steps=effect_steps)

        # Fallback to standard primary primitives if no specific script found
        if not card.current_primary_action:
            logger.debug("            > No custom script found and no primary action.")
            return StepResult(is_finished=True)

        logger.debug(
            f"            > No custom script found. Using standard {card.current_primary_action.name} logic."
        )

        # Declared here for the first time in this scope path
        steps_list: list[GameStep] = []

        if card.current_primary_action == ActionType.MOVEMENT:
            # MOVEMENT: Compute Total
            base_val = card.get_base_stat_value(StatType.MOVEMENT)
            total_val = get_computed_stat(state, UnitID(self.hero_id), StatType.MOVEMENT, base_val)
            steps_list.append(MoveSequenceStep(unit_id=self.hero_id, range_val=total_val))

        elif card.current_primary_action == ActionType.ATTACK:
            # ATTACK: Compute Damage & Range
            base_dmg = card.get_base_stat_value(StatType.ATTACK)
            total_dmg = get_computed_stat(state, UnitID(self.hero_id), StatType.ATTACK, base_dmg)

            base_rng = card.get_base_stat_value(StatType.RANGE)
            # Default Range is 1 if not specified (and get_base_stat_value returns 0 if None)
            if base_rng == 0:
                base_rng = 1
            total_rng = get_computed_stat(state, UnitID(self.hero_id), StatType.RANGE, base_rng)

            steps_list.append(AttackSequenceStep(damage=total_dmg, range_val=total_rng))

        elif card.current_primary_action == ActionType.DEFENSE:
            steps_list.append(LogMessageStep(message=f"{self.hero_id} Defends (Primary)."))
        elif card.current_primary_action == ActionType.SKILL:
            logger.debug(f"            > Skill '{card.name}' has no registered effect!")
            steps_list.append(LogMessageStep(message=f"Skill '{card.name}' did nothing."))

        return StepResult(is_finished=True, new_steps=steps_list)


class ResolveCardStep(GameStep):
    """
    Analyzes the active card and prompts the user to choose an Action.
    Spawns the appropriate logic steps based on the choice.
    """

    type: StepType = StepType.RESOLVE_CARD
    hero_id: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.steps.combat import AttackSequenceStep
        from goa2.engine.steps.effects import CheckPassiveAbilitiesStep
        from goa2.engine.steps.markers import RemoveTokenStep
        from goa2.engine.steps.movement import (
            FastTravelSequenceStep,
            MoveSequenceStep,
            ResolvePreActionMovementStep,
        )
        from goa2.engine.steps.selection import MultiSelectStep
        from goa2.engine.steps.utility import ForEachStep, LogMessageStep, SetContextFlagStep

        hero = state.get_hero(HeroID(self.hero_id))
        if not hero or not hero.current_turn_card:
            return StepResult(is_finished=True)

        # If hero is off-board (didn't respawn), skip action
        if not state.has_board_presence(self.hero_id):
            return StepResult(is_finished=True)

        card = hero.current_turn_card

        context["current_card_id"] = card.id
        options = []

        from goa2.engine.rules import get_safe_zones_for_fast_travel

        def is_action_available(act_type: ActionType, *, is_primary: bool) -> bool:
            # 1. Check Global/Effect Validation (e.g. Spell Break prevention)
            # We pass the 'card' object in context so validation can check exceptions (color).
            val_res = state.validator.can_perform_action(
                state, self.hero_id, act_type, context={"card": card}
            )
            if not val_res.allowed:
                return False

            if act_type == ActionType.FAST_TRAVEL:
                hero_positions = state.get_positions(self.hero_id)
                if not hero_positions:
                    return False
                zone_ids = {
                    z for z in (state.board.get_zone_for_hex(loc) for loc in hero_positions) if z
                }
                if not zone_ids:
                    return False

                if not hero:
                    return False

                # Ensure team is present
                team = getattr(hero, "team", None)
                if not team:
                    return False

                safe = [
                    z for zid in zone_ids for z in get_safe_zones_for_fast_travel(state, team, zid)
                ]
                if not safe:
                    return False

            # 2. Prune target-gated actions with no legal target (ATTACK/SKILL):
            # keeps a guaranteed no-op out of the menu so the AI never expands it.
            return (
                act_type not in (ActionType.ATTACK, ActionType.SKILL)
                or action_may_change_targets_before_card_text(
                    state, hero, act_type, is_primary=is_primary
                )
                or action_passes_initial_target_gate(
                    state, hero, card, act_type, is_primary=is_primary
                )
            )

        # Helper to compute option values
        def compute_option(act_type: ActionType, base_val: int | None) -> tuple[int, str]:
            # Default
            final_val = base_val or 0
            text_val = str(final_val) if base_val is not None else "-"

            # Map Action to Stat
            stat_type = None
            if act_type == ActionType.MOVEMENT:
                stat_type = StatType.MOVEMENT
            elif act_type == ActionType.ATTACK:
                stat_type = StatType.ATTACK
            elif act_type == ActionType.DEFENSE or act_type == ActionType.DEFENSE_SKILL:
                stat_type = StatType.DEFENSE

            if stat_type:
                final_val = get_computed_stat(state, UnitID(self.hero_id), stat_type, base_val or 0)
                text_val = str(final_val)

            return final_val, text_val

        def build_option(
            act_type: ActionType, base_val: int | None, label: str, *, is_primary: bool
        ) -> dict[str, Any] | None:
            """Build one CHOOSE_ACTION option, or None if it shouldn't be offered.

            - DEFENSE: never an active action on your turn -> omitted.
            - DEFENSE_SKILL: shown as a SKILL option (primary only).
            - is_action_available: prevention/FAST_TRAVEL gate + no-target prune.
            """
            if act_type == ActionType.DEFENSE:
                return None
            display_type = ActionType.SKILL if act_type == ActionType.DEFENSE_SKILL else act_type

            if not is_action_available(display_type, is_primary=is_primary):
                return None

            c_val, c_text = compute_option(display_type, base_val)
            return {
                "id": display_type.name,
                "type": display_type,
                "value": c_val,
                "text": f"{label}: {display_type.name} ({c_text})",
            }

        # Primary: DEFENSE omitted, DEFENSE_SKILL remapped to SKILL.
        primary_action = card.current_primary_action
        if primary_action:
            primary_opt = build_option(
                primary_action, card.current_primary_action_value, "Primary", is_primary=True
            )
            if primary_opt:
                options.append(primary_opt)

        # Secondaries: DEFENSE omitted.
        for action_type, val in card.current_secondary_actions.items():
            secondary_opt = build_option(action_type, val, "Secondary", is_primary=False)
            if secondary_opt:
                options.append(secondary_opt)

        if self.pending_input:
            choice_id = self.pending_input.get("selection")
            selected_opt = next((o for o in options if o["id"] == choice_id), None)

            if selected_opt:
                # Type safe access
                act_type = cast(ActionType, selected_opt["type"])
                val = cast(int, selected_opt["value"])
                # Determine if primary by checking the card itself
                is_primary = act_type == primary_action
                # DEFENSE_SKILL played as SKILL still uses primary effect
                if (
                    card.current_primary_action == ActionType.DEFENSE_SKILL
                    and act_type == ActionType.SKILL
                ):
                    is_primary = True

                logger.debug(f"   [CHOICE] Player selected {choice_id} ({act_type.name})")

                # Track current action type for effect origin tracking
                context["current_action_type"] = act_type

                # NOTE: Renamed local variable to avoid shadowing re-declaration if any
                steps_list: list[GameStep] = []

                from goa2.engine.steps.pieces import ChooseActingPieceStep

                steps_list.append(ChooseActingPieceStep(hero_id=self.hero_id))

                # Check for BEFORE_* passive abilities based on action type.
                # BEFORE_ACTION always fires — primary, secondary, or HOLD —
                # in addition to any specific BEFORE_ATTACK/MOVEMENT/SKILL.
                from goa2.domain.models.enums import PassiveTrigger

                steps_list.append(
                    CheckPassiveAbilitiesStep(trigger=PassiveTrigger.BEFORE_ACTION.value)
                )

                specific_trigger = None
                if act_type == ActionType.ATTACK:
                    specific_trigger = PassiveTrigger.BEFORE_ATTACK
                elif act_type == ActionType.MOVEMENT:
                    specific_trigger = PassiveTrigger.BEFORE_MOVEMENT
                elif act_type == ActionType.SKILL:
                    specific_trigger = PassiveTrigger.BEFORE_SKILL

                if specific_trigger:
                    steps_list.append(CheckPassiveAbilitiesStep(trigger=specific_trigger.value))

                if is_primary:
                    steps_list.append(ResolvePreActionMovementStep(hero_id=self.hero_id))
                    steps_list.append(ResolvePreActionDiscardStep(hero_id=self.hero_id))
                    steps_list.append(ResolveCardTextStep(card_id=card.id, hero_id=self.hero_id))
                else:
                    # Secondary: Standard Primitives
                    if act_type == ActionType.MOVEMENT:
                        move_base = card.current_secondary_actions.get(act_type, val)
                        steps_list.append(
                            MoveSequenceStep(
                                unit_id=self.hero_id,
                                range_val=move_base,
                                range_stat_type=StatType.MOVEMENT,
                            )
                        )

                    elif act_type == ActionType.FAST_TRAVEL:
                        steps_list.append(FastTravelSequenceStep(unit_id=self.hero_id))

                    elif act_type == ActionType.ATTACK:
                        attack_base = card.current_secondary_actions.get(act_type, val)
                        base_rng = card.get_base_stat_value(StatType.RANGE)
                        if base_rng == 0:
                            base_rng = 1

                        steps_list.append(
                            AttackSequenceStep(
                                damage=attack_base,
                                range_val=base_rng,
                                damage_stat_type=StatType.ATTACK,
                                range_stat_type=StatType.RANGE,
                            )
                        )

                    elif act_type == ActionType.CLEAR:
                        # Gate on board presence, not the owner position: a
                        # multi-piece hero has no owner-level position, and the
                        # RangeFilter keys off the bound acting piece at
                        # execution, so build the real selection whenever any
                        # piece is on the board.
                        if not state.has_board_presence(self.hero_id):
                            steps_list.append(
                                LogMessageStep(
                                    message=f"{self.hero_id} attempted clear but is not on board."
                                )
                            )
                        else:
                            steps_list.extend(
                                [
                                    MultiSelectStep(
                                        min_selections=0,
                                        max_selections=6,
                                        filters=[
                                            UnitTypeFilter(unit_type="TOKEN"),
                                            RangeFilter(max_range=1),
                                            ImmunityFilter(),
                                        ],
                                        output_key="clear_targets",
                                        target_type=TargetType.UNIT_OR_TOKEN,
                                        prompt="Select tokens to clear.",
                                    ),
                                    ForEachStep(
                                        list_key="clear_targets",
                                        item_key="target_id",
                                        steps_template=[RemoveTokenStep(token_key="target_id")],
                                    ),
                                ]
                            )
                    elif act_type == ActionType.HOLD:
                        steps_list.append(LogMessageStep(message=f"{self.hero_id} Holds."))

                    elif act_type == ActionType.DEFENSE:
                        # Should not happen as action, but valid in enum
                        steps_list.append(
                            LogMessageStep(message=f"{self.hero_id} Defends (Active).")
                        )

                # Add AFTER_ATTACK passive check for ALL attack actions
                if act_type == ActionType.ATTACK:
                    # Store attack info so passives can rebuild the effect
                    if is_primary and card.current_effect_id:
                        steps_list.append(
                            SetContextFlagStep(
                                key="attack_effect_id",
                                value=card.current_effect_id,
                            )
                        )
                        steps_list.append(SetContextFlagStep(key="attack_card_id", value=card.id))
                    steps_list.append(
                        CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_ATTACK.value)
                    )

                # Add AFTER_MOVEMENT passive check for ALL movement actions
                if act_type == ActionType.MOVEMENT:
                    steps_list.append(
                        CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_MOVEMENT.value)
                    )

                # Add AFTER_BASIC_SKILL passive check for Gold/Silver SKILL cards
                if act_type == ActionType.SKILL and card.current_color in (
                    CardColor.GOLD,
                    CardColor.SILVER,
                ):
                    steps_list.append(
                        CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_BASIC_SKILL.value)
                    )

                # Add AFTER_BASIC_ACTION passive check for basic card actions
                if card.is_basic and act_type in (
                    ActionType.ATTACK,
                    ActionType.MOVEMENT,
                    ActionType.SKILL,
                ):
                    steps_list.append(
                        SetContextFlagStep(key="basic_action_type", value=act_type.value)
                    )
                    steps_list.append(SetContextFlagStep(key="basic_action_value", value=val))
                    # Store range for attack repeats
                    if act_type == ActionType.ATTACK:
                        base_rng_ba = card.get_base_stat_value(StatType.RANGE)
                        if base_rng_ba == 0:
                            base_rng_ba = 1
                        total_rng_ba = get_computed_stat(
                            state,
                            UnitID(self.hero_id),
                            StatType.RANGE,
                            base_rng_ba,
                        )
                        steps_list.append(
                            SetContextFlagStep(key="basic_action_range", value=total_rng_ba)
                        )
                    # Store effect info for primary actions so passives can
                    # rebuild the full effect sequence (e.g. Blink Strike)
                    if is_primary and card.current_effect_id:
                        steps_list.append(
                            SetContextFlagStep(
                                key="basic_action_effect_id",
                                value=card.current_effect_id,
                            )
                        )
                        steps_list.append(
                            SetContextFlagStep(key="basic_action_card_id", value=card.id)
                        )
                    steps_list.append(
                        CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_BASIC_ACTION.value)
                    )

                # Fires only after a PRIMARY action (Cutter - Legend of the
                # Skies). Secondary actions must not trigger it.
                if is_primary:
                    steps_list.append(
                        CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_PRIMARY_ACTION.value)
                    )

                # Fires after the card's action fully resolves, primary or
                # secondary (Wuk - March of Nature). Must be last so it runs
                # after the AFTER_* checks above.
                steps_list.append(
                    CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_RESOLVE_CARD.value)
                )

                return StepResult(is_finished=True, new_steps=steps_list)

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.CHOOSE_ACTION,
                player_id=self.hero_id,
                prompt=f"Choose action for card {card.name}",
                options=options,
            ),
        )


class SwapCardStep(GameStep):
    """
    Swaps the Hero's current turn card with another card (specified by ID or key).
    """

    type: StepType = StepType.SWAP_CARD
    target_card_id: str | None = None
    target_card_key: str | None = None  # Key in context to find ID
    source_card_key: str | None = None  # Optional context key for the first card
    context_hero_id_key: str | None = None  # Key in context to find Hero ID

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        hero_id = state.current_actor_id

        # Override hero if key provided
        if self.context_hero_id_key:
            h_val = context.get(self.context_hero_id_key)
            if h_val:
                hero_id = HeroID(str(h_val))

        if not hero_id:
            return StepResult(is_finished=True)

        hero = state.get_hero(hero_id)
        if not hero:
            return StepResult(is_finished=True)

        # Find target card ID
        t_id = self.target_card_id
        if not t_id and self.target_card_key:
            t_id = context.get(self.target_card_key)

        if not t_id:
            logger.debug("   [SWAP] No target card specified for swap.")
            return StepResult(is_finished=True)

        def find_card(card_id: str) -> Card | None:
            for collection in (hero.hand, hero.discard_pile, hero.played_cards):
                card = next((c for c in collection if c is not None and c.id == card_id), None)
                if card:
                    return card
            if hero.current_turn_card and hero.current_turn_card.id == card_id:
                return hero.current_turn_card
            return None

        source_card = hero.current_turn_card
        if self.source_card_key:
            source_id = context.get(self.source_card_key)
            source_card = find_card(str(source_id)) if source_id else None
        target_card = find_card(str(t_id))

        if not source_card or not target_card or source_card.id == target_card.id:
            logger.debug(f"   [SWAP] Target card {t_id} not found in {hero_id}'s possession.")
            return StepResult(is_finished=True)

        logger.debug(f"   [SWAP] Swapping {hero.id}'s {source_card.name} with {target_card.name}")
        # Both cards change state in a swap, which cancels their active effects
        # (premature end, so finishing_steps do not run). Capture ids first.
        swapped_out_id = source_card.id
        swapped_in_id = target_card.id
        hero.swap_cards(source_card, target_card)

        from goa2.engine.effect_manager import EffectManager

        EffectManager.expire_by_card(state, swapped_out_id)
        EffectManager.expire_by_card(state, swapped_in_id)

        # NOTE: After swapping, the "current_turn_card" has changed!
        # This might affect subsequent steps if they rely on "current_card_id" in context.
        # But usually context has the old ID if it was set earlier.
        # Ideally, we should update context if necessary, but "current_card_id" is usually set once at start of ResolveCardText.

        return StepResult(is_finished=True)


class SwapWithDeckCardStep(GameStep):
    """Exchange a card that is in play with a card sitting in the owner's deck.

    The incoming deck card inherits the outgoing card's exact place (hand slot,
    discard pile, resolved slot, or the current turn card) along with its
    state/facedown/played_this_round flags. The outgoing card returns to the
    deck faceup. Bushido's rider (``facedown_if_from_discard_or_resolved``)
    forces the incoming card facedown when it lands in the discard pile or a
    resolved slot.

    ``hero.deck`` is the master card list, so the outgoing card never leaves it;
    only its state and container membership change.
    """

    type: StepType = StepType.SWAP_WITH_DECK_CARD

    hero_id: str | None = None
    hero_key: str | None = None
    outgoing_card_id: str | None = None
    outgoing_card_key: str | None = None
    incoming_card_key: str = "deck_swap_card"
    facedown_if_from_discard_or_resolved: bool = False

    def _resolve_hero(self, state: GameState, context: dict[str, Any]) -> Hero | None:
        hero_id: Any = self.hero_id
        if self.hero_key:
            hero_id = context.get(self.hero_key) or hero_id
        if not hero_id:
            hero_id = state.current_actor_id
        if not hero_id:
            return None
        return state.get_hero(HeroID(str(hero_id)))

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        hero = self._resolve_hero(state, context)
        if not hero:
            return StepResult(is_finished=True)

        outgoing_id = self.outgoing_card_id
        if self.outgoing_card_key:
            outgoing_id = context.get(self.outgoing_card_key) or outgoing_id
        incoming_id = context.get(self.incoming_card_key)
        if not outgoing_id or not incoming_id or str(outgoing_id) == str(incoming_id):
            return StepResult(is_finished=True)

        incoming = next((c for c in hero.deck if c.id == str(incoming_id)), None)
        if not incoming or incoming.state != CardState.DECK:
            logger.debug(f"   [DECK SWAP] {incoming_id!r} is not a card in {hero.id}'s deck.")
            return StepResult(is_finished=True)

        outgoing = next((c for c in hero.deck if c.id == str(outgoing_id)), None)
        if not outgoing or not self._install_in_place(hero, outgoing, incoming):
            logger.debug(f"   [DECK SWAP] {outgoing_id!r} is not in play for {hero.id}.")
            return StepResult(is_finished=True)

        from goa2.engine.effect_manager import EffectManager

        EffectManager.expire_by_card(state, outgoing.id)
        EffectManager.expire_by_card(state, incoming.id)

        logger.debug(f"   [DECK SWAP] {hero.id}: {outgoing.name} → deck, {incoming.name} → play")
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.DECK_CARD_SWAPPED,
                    actor_id=str(hero.id),
                    metadata={
                        "outgoing_card_id": outgoing.id,
                        "incoming_card_id": incoming.id,
                        "incoming_card_state": incoming.state.value,
                        "incoming_is_facedown": incoming.is_facedown,
                    },
                )
            ],
        )

    def _install_in_place(self, hero: Hero, outgoing: Card, incoming: Card) -> bool:
        """Put `incoming` exactly where `outgoing` was; send `outgoing` to the deck.

        Returns False (and changes nothing) when the outgoing card is not in play.
        """
        target_state = outgoing.state
        target_played_this_round = outgoing.played_this_round
        facedown = outgoing.is_facedown

        if outgoing in hero.hand:
            hero.hand[hero.hand.index(outgoing)] = incoming
        elif outgoing in hero.discard_pile:
            hero.discard_pile[hero.discard_pile.index(outgoing)] = incoming
            facedown = facedown or self.facedown_if_from_discard_or_resolved
        elif hero.current_turn_card is not None and hero.current_turn_card.id == outgoing.id:
            hero.current_turn_card = incoming
        else:
            slot = next(
                (
                    i
                    for i, c in enumerate(hero.played_cards)
                    if c is not None and c.id == outgoing.id
                ),
                None,
            )
            if slot is None:
                return False
            hero.played_cards[slot] = incoming
            facedown = facedown or self.facedown_if_from_discard_or_resolved

        incoming.state = target_state
        incoming.is_facedown = facedown
        incoming.played_this_round = target_played_this_round

        outgoing.state = CardState.DECK
        outgoing.is_facedown = False
        outgoing.played_this_round = False
        outgoing.is_active = False
        return True


class SwapItemCardStep(GameStep):
    """Exchange a hero's equipped item card for an eligible card in place."""

    type: StepType = StepType.SWAP_ITEM_CARD
    hero_key: str
    item_card_key: str
    target_card_key: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        hero_id = context.get(self.hero_key)
        item_id = context.get(self.item_card_key)
        target_id = context.get(self.target_card_key)
        if not hero_id or not item_id or not target_id or item_id == target_id:
            return StepResult(is_finished=True)
        hero = state.get_hero(HeroID(str(hero_id)))
        if not hero:
            return StepResult(is_finished=True)

        item_card = next((card for card in hero.deck if card.id == str(item_id)), None)
        target_card = next((card for card in hero.deck if card.id == str(target_id)), None)
        if (
            not item_card
            or not target_card
            or item_card.state != CardState.ITEM
            or target_card.state == CardState.ITEM
            or item_card.item is None
            or target_card.item is None
            or item_card.tier != target_card.tier
            or item_card.color != target_card.color
        ):
            return StepResult(is_finished=True)

        original_state = target_card.state
        original_facedown = target_card.is_facedown
        original_played_this_round = target_card.played_this_round
        if target_card in hero.hand:
            hero.hand[hero.hand.index(target_card)] = item_card
        elif target_card in hero.discard_pile:
            hero.discard_pile[hero.discard_pile.index(target_card)] = item_card
        elif target_card == hero.current_turn_card:
            hero.current_turn_card = item_card
        else:
            played_index = next(
                (i for i, card in enumerate(hero.played_cards) if card == target_card), None
            )
            if played_index is not None:
                hero.played_cards[played_index] = item_card
            elif original_state != CardState.DECK:
                return StepResult(is_finished=True)

        item_card.state = original_state
        item_card.is_facedown = original_facedown
        item_card.played_this_round = original_played_this_round
        target_card.state = CardState.ITEM
        target_card.is_facedown = False
        target_card.played_this_round = False
        hero.items[item_card.item] = hero.items.get(item_card.item, 0) - 1
        hero.items[target_card.item] = hero.items.get(target_card.item, 0) + 1

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.ITEM_GAINED,
                    actor_id=str(hero.id),
                    metadata={
                        "stat_type": target_card.item.value,
                        "amount": 1,
                        "source_card_id": target_card.id,
                    },
                )
            ],
        )


class SwapResolvedCardsStep(GameStep):
    """
    Swaps the slot positions of two RESOLVED cards of one hero, WITHOUT
    canceling active effects (NebKher's Diabolical Laughter — "Swap two
    resolved cards of an enemy hero in radius, without canceling active
    effects").

    Contrast with SwapCardStep, which swaps the current turn card with
    another card and intentionally expires both cards' active effects.
    Effects bind by card id (``source_card_id``), so a pure slot reorder
    leaves them untouched; previous-turn-slot lookups see the new order.
    No-ops if either card is missing from the hero's played slots or is
    not RESOLVED.
    """

    type: StepType = StepType.SWAP_RESOLVED_CARDS
    hero_id: str | None = None
    hero_key: str | None = None  # Context key holding the hero ID
    card_a_key: str = "swap_card_a"
    card_b_key: str = "swap_card_b"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        h_id = self.hero_id
        if not h_id and self.hero_key:
            h_val = context.get(self.hero_key)
            if h_val:
                h_id = str(h_val)
        if not h_id:
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(h_id)))
        if not hero:
            return StepResult(is_finished=True)

        card_a_id = context.get(self.card_a_key)
        card_b_id = context.get(self.card_b_key)
        if not card_a_id or not card_b_id or card_a_id == card_b_id:
            return StepResult(is_finished=True)

        def find_resolved(card_id: str) -> Card | None:
            for c in hero.played_cards:
                if c is not None and c.id == card_id and c.state == CardState.RESOLVED:
                    return c
            return None

        card_a = find_resolved(str(card_a_id))
        card_b = find_resolved(str(card_b_id))
        if not card_a or not card_b:
            logger.debug(
                "   [SWAP RESOLVED] Cards %s/%s not both resolved on %s; skipping.",
                card_a_id,
                card_b_id,
                h_id,
            )
            return StepResult(is_finished=True)

        hero.swap_cards(card_a, card_b)
        logger.debug(
            "   [SWAP RESOLVED] %s's resolved cards %s and %s traded slots.",
            h_id,
            card_a.name,
            card_b.name,
        )
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.RESOLVED_CARDS_SWAPPED,
                    actor_id=str(state.current_actor_id) if state.current_actor_id else None,
                    target_id=str(h_id),
                    metadata={"card_a_id": card_a.id, "card_b_id": card_b.id},
                )
            ],
        )


class RetrieveCardStep(GameStep):
    """
    Retrieves a card from discard pile back to hand.
    Uses context[card_key] for the card ID.
    If hero_key is set, looks up the hero ID from context; otherwise uses
    current_actor_id.
    """

    type: StepType = StepType.RETRIEVE_CARD
    card_key: str = ""
    hero_key: str | None = None
    retrieve_all_discarded: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        # Determine which hero retrieves the card
        if self.hero_key:
            hero_id_str = context.get(self.hero_key)
            if not hero_id_str:
                return StepResult(is_finished=True)
            actor_id = hero_id_str
        else:
            actor_id = state.current_actor_id
            if not actor_id:
                return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(actor_id)))
        if not hero:
            return StepResult(is_finished=True)

        if self.retrieve_all_discarded:
            cards = list(hero.discard_pile)
            for card in cards:
                hero.return_card_to_hand(card)
                from goa2.engine.effect_manager import EffectManager

                EffectManager.expire_by_card(state, card.id)
            return StepResult(
                is_finished=True,
                events=[
                    GameEvent(
                        event_type=GameEventType.CARD_RETRIEVED,
                        actor_id=str(actor_id),
                        metadata={"card_id": card.id, "card_name": card.name},
                    )
                    for card in cards
                ],
            )

        card_id = context.get(self.card_key)
        if not card_id:
            return StepResult(is_finished=True)

        # Find card in played_cards or discard_pile
        target_card = next(
            (c for c in hero.played_cards if c is not None and c.id == card_id),
            None,
        )
        source = "played"
        if not target_card:
            target_card = next((c for c in hero.discard_pile if c.id == card_id), None)
            source = "discard"
        # The card being resolved this turn lives in current_turn_card, not yet
        # in played_cards, so "retrieve this card" (Brynn - Peak Precision) needs
        # this fallback. return_card_to_hand clears current_turn_card.
        if not target_card and hero.current_turn_card and hero.current_turn_card.id == card_id:
            target_card = hero.current_turn_card
            source = "current_turn"
        if not target_card:
            logger.debug(
                f"   [RETRIEVE] Card {card_id} not found in {actor_id}'s played or discard."
            )
            return StepResult(is_finished=True)

        hero.return_card_to_hand(target_card)
        logger.debug(f"   [RETRIEVE] {actor_id} retrieved {target_card.name} from {source}.")

        # Returning a card to hand changes its state, cancelling its active
        # effect (premature end, so finishing_steps do not run).
        from goa2.engine.effect_manager import EffectManager

        EffectManager.expire_by_card(state, target_card.id)

        event = GameEvent(
            event_type=GameEventType.CARD_RETRIEVED,
            actor_id=str(actor_id),
            metadata={"card_id": card_id, "card_name": target_card.name},
        )
        return StepResult(is_finished=True, events=[event])


class RetrieveUnresolvedCardStep(GameStep):
    """
    Emmitt's Alternative Timelines: after revealing two cards, the hero MUST
    retrieve one of their two unresolved cards to hand; the other remains as
    the turn card. Runs at the start of Resolution, before any hero acts.
    """

    type: StepType = StepType.RETRIEVE_UNRESOLVED_CARD
    hero_id: str

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        hero = state.get_hero(HeroID(self.hero_id))
        if not hero or hero.extra_turn_card is None or hero.current_turn_card is None:
            return StepResult(is_finished=True)

        first, second = hero.current_turn_card, hero.extra_turn_card

        if self.pending_input:
            chosen_id = self.pending_input.get("selection")
            if chosen_id in (first.id, second.id):
                retrieved = first if chosen_id == first.id else second
                kept = second if retrieved is first else first

                hero.extra_turn_card = None
                hero.return_card_to_hand(retrieved)
                # return_card_to_hand clears current_turn_card when it points at
                # the retrieved card; the kept card is the turn card either way.
                hero.current_turn_card = kept

                logger.info(
                    "   [TIMELINES] %s retrieves %s, keeps %s.",
                    self.hero_id,
                    retrieved.name,
                    kept.name,
                )
                return StepResult(
                    is_finished=True,
                    events=[
                        GameEvent(
                            event_type=GameEventType.CARD_RETRIEVED,
                            actor_id=self.hero_id,
                            metadata={
                                "card_id": retrieved.id,
                                "card_name": retrieved.name,
                                "kept_card_id": kept.id,
                                "source": "alternative_timelines",
                            },
                        )
                    ],
                )
            logger.warning("   [TIMELINES] Invalid retrieve choice %s. Re-prompting.", chosen_id)
            self.pending_input = None

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.SELECT_CARD,
                player_id=self.hero_id,
                prompt="Alternative Timelines: retrieve one of your unresolved cards.",
                options=[
                    {"id": first.id, "text": f"{first.name} (initiative {first.initiative})"},
                    {"id": second.id, "text": f"{second.name} (initiative {second.initiative})"},
                ],
            ),
        )


class CountCardsStep(GameStep):
    """
    Counts cards in a hero's container (hand, discard, deck, played)
    and stores the count in context[output_key].
    """

    type: StepType = StepType.COUNT_CARDS
    hero_id: str | None = None
    hero_key: str | None = None
    card_container: CardContainerType = CardContainerType.DISCARD
    output_key: str = "card_count"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            context[self.output_key] = 0
            return StepResult(is_finished=True)

        h_id = self.hero_id
        if not h_id and self.hero_key:
            h_id = context.get(self.hero_key)
        if not h_id:
            h_id = state.current_actor_id

        if not h_id:
            context[self.output_key] = 0
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(h_id)))
        if not hero:
            context[self.output_key] = 0
            return StepResult(is_finished=True)

        if self.card_container == CardContainerType.HAND:
            count = len(hero.hand)
        elif self.card_container == CardContainerType.DISCARD:
            count = len(hero.discard_pile)
        elif self.card_container == CardContainerType.DECK:
            count = len(hero.deck)
        elif self.card_container == CardContainerType.PLAYED:
            count = len([c for c in hero.played_cards if c is not None])
        else:
            count = 0

        context[self.output_key] = count
        logger.debug(f"   [COUNT_CARDS] {h_id} {self.card_container.value}: {count}")
        return StepResult(is_finished=True)


class GainCoinsStep(GameStep):
    """Grants gold to a hero identified by a context key."""

    type: StepType = StepType.GAIN_COINS
    hero_key: str  # context key → hero ID
    amount: int = 0  # static amount
    amount_key: str = ""  # context key → dynamic amount (overrides static)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)
        hero_id = context.get(self.hero_key)
        if not hero_id:
            return StepResult(is_finished=True)
        hero = state.get_hero(HeroID(str(hero_id)))
        if not hero:
            return StepResult(is_finished=True)
        coins = context.get(self.amount_key, self.amount) if self.amount_key else self.amount
        hero.gold += coins
        logger.debug(f"   [COINS] {hero_id} gains {coins} gold")
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.GOLD_GAINED,
                    actor_id=str(hero_id),
                    metadata={"amount": coins, "reason": "effect"},
                )
            ],
        )


class LoseCoinsStep(GameStep):
    """Removes coins from a hero without transferring them to the actor."""

    type: StepType = StepType.LOSE_COINS
    victim_key: str  # context key -> hero or hero-piece ID
    amount: int = 1
    amount_key: str = ""
    output_key: str = ""  # actual amount lost, when requested

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        victim_unit_id = context.get(self.victim_key)
        if not victim_unit_id:
            return StepResult(is_finished=True)
        victim = state.get_hero(HeroID(str(victim_unit_id)))
        if victim is None:
            return StepResult(is_finished=True)

        requested = context.get(self.amount_key, self.amount) if self.amount_key else self.amount
        actual = min(max(int(requested), 0), victim.gold)
        victim.gold -= actual
        if self.output_key:
            context[self.output_key] = actual

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.GOLD_LOST,
                    actor_id=(str(state.current_actor_id) if state.current_actor_id else None),
                    target_id=str(victim_unit_id),
                    metadata={
                        "amount": actual,
                        "owner_id": str(victim.id),
                        "reason": "effect",
                    },
                )
            ],
        )


class RevealHandCardStep(GameStep):
    """Publicly reveal one selected hand card and expose its numeric tier.

    The real card remains in its private hand until later steps move it. The
    public table snapshot survives turn finalization/reconnects, while normal
    player-scoped hand masking continues to hide every other card.
    """

    type: StepType = StepType.REVEAL_HAND_CARD
    owner_key: str  # context key -> selected hero or hero-piece ID
    card_key: str  # context key -> selected hand card ID
    tier_value_key: str = "revealed_tier_value"

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        target_unit_id = context.get(self.owner_key)
        card_id = context.get(self.card_key)
        if not target_unit_id or not card_id:
            return StepResult(is_finished=True)

        owner = state.get_hero(HeroID(str(target_unit_id)))
        if owner is None:
            return StepResult(is_finished=True)
        target_card = next((candidate for candidate in owner.hand if candidate.id == card_id), None)
        if target_card is None:
            return StepResult(is_finished=True)

        tier_values = {
            CardTier.UNTIERED: 0,
            CardTier.I: 1,
            CardTier.II: 2,
            CardTier.III: 3,
            CardTier.IV: 4,
        }
        tier_value = tier_values[target_card.tier]
        context[self.tier_value_key] = tier_value
        context["rollback_reanchor_pending"] = True

        revealer_id = str(state.current_actor_id) if state.current_actor_id else None
        state.card_reveal = {
            "revealer_id": revealer_id,
            "target_unit_id": str(target_unit_id),
            "owner_id": str(owner.id),
            "card_id": target_card.id,
            "tier_value": tier_value,
        }
        state.record_public_revealed_card(owner.id, str(target_card.id))

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.CARD_REVEALED,
                    actor_id=revealer_id,
                    target_id=str(target_unit_id),
                    metadata={
                        "owner_id": str(owner.id),
                        "card_id": target_card.id,
                        "card_name": target_card.name,
                        "card_color": target_card.color.value if target_card.color else None,
                        "card_tier": target_card.tier.value,
                        "tier_value": tier_value,
                    },
                )
            ],
        )


class CheckSoloWinStep(GameStep):
    """Resolve Cutter's alternate victory after A Fistful of Coins gains coins."""

    type: StepType = StepType.CHECK_SOLO_WIN
    hero_key: str = ""  # context key → hero ID (falls back to current actor)
    threshold: int = 13

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)
        hero_id = context.get(self.hero_key) if self.hero_key else None
        hero_id = hero_id or state.current_actor_id
        if not hero_id:
            return StepResult(is_finished=True)
        hero = state.get_hero(HeroID(str(hero_id)))
        if not hero:
            return StepResult(is_finished=True)
        if hero.gold >= self.threshold:
            logger.debug(
                "   [SOLO WIN] %s reached %s coins (>= %s)",
                hero_id,
                hero.gold,
                self.threshold,
            )
            from goa2.engine.steps.combat import TriggerGameOverStep

            return StepResult(
                is_finished=True,
                new_steps=[
                    TriggerGameOverStep(
                        individual_winner_id=HeroID(str(hero_id)),
                        condition="A_FISTFUL_OF_COINS",
                    )
                ],
            )
        return StepResult(is_finished=True)


class GainItemStep(GameStep):
    """Grants a stat item to a hero identified by a context key."""

    type: StepType = StepType.GAIN_ITEM
    hero_key: str  # context key → hero ID
    stat_type: StatType  # which stat to boost
    amount: int = 1

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)
        hero_id = context.get(self.hero_key)
        if not hero_id:
            return StepResult(is_finished=True)
        hero = state.get_hero(HeroID(str(hero_id)))
        if not hero:
            return StepResult(is_finished=True)
        hero.items[self.stat_type] = hero.items.get(self.stat_type, 0) + self.amount
        logger.debug(f"   [ITEM] {hero_id} gains +{self.amount} {self.stat_type.name} item")
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.ITEM_GAINED,
                    actor_id=str(hero_id),
                    metadata={
                        "stat_type": self.stat_type.value,
                        "amount": self.amount,
                    },
                )
            ],
        )


class StealCoinsStep(GameStep):
    """Takes coins from an enemy hero and gives them to the current actor."""

    type: StepType = StepType.STEAL_COINS
    victim_key: str  # context key → enemy hero ID
    amount: int = 1  # static amount to steal
    amount_key: str = ""  # context key → dynamic amount (overrides static)
    output_key: str = ""  # if set, stores True in context when coins were stolen

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        victim_id = context.get(self.victim_key)
        if not victim_id:
            return StepResult(is_finished=True)

        victim = state.get_hero(HeroID(str(victim_id)))
        if not victim:
            return StepResult(is_finished=True)

        actor_id = context.get("current_actor_id") or state.current_actor_id
        actor = state.get_hero(HeroID(str(actor_id)))
        if not actor:
            return StepResult(is_finished=True)

        coins_requested = (
            context.get(self.amount_key, self.amount) if self.amount_key else self.amount
        )
        actual_stolen = min(coins_requested, victim.gold)

        if actual_stolen <= 0:
            return StepResult(is_finished=True)

        victim.gold -= actual_stolen
        actor.gold += actual_stolen
        if self.output_key:
            context[self.output_key] = True
        logger.debug(f"   [STEAL] {actor_id} steals {actual_stolen} coin(s) from {victim_id}")
        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.GOLD_GAINED,
                    actor_id=str(actor_id),
                    target_id=str(victim_id),
                    metadata={"amount": actual_stolen, "reason": "steal"},
                ),
            ],
        )


class PerformPrimaryActionStep(GameStep):
    """
    Looks up a card from context, computes its stats, calls its effect's
    build_steps(), and pushes the resulting steps onto the stack.

    Used by Ursafar's Angry Roar, Instinctive Reaction, Evolutionary Response.
    """

    type: StepType = StepType.PERFORM_PRIMARY_ACTION
    card_key: str = "selected_card"
    hero_id: str | None = None
    exclude_target_key: str | None = None

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        card_id = context.get(self.card_key)
        if not card_id:
            return StepResult(is_finished=True)

        actor_id = self.hero_id or (str(state.current_actor_id) if state.current_actor_id else None)
        if not actor_id:
            return StepResult(is_finished=True)

        hero = state.get_hero(HeroID(str(actor_id)))
        if not hero:
            return StepResult(is_finished=True)

        # Find the card anywhere on the hero. The current turn card is checked
        # first because a passive that re-performs the just-played basic card
        # (e.g. Bullet Time) fires while that card is still current_turn_card —
        # it only moves into played_cards in FinalizeHeroTurnStep.
        card = None
        search_cards = [
            hero.current_turn_card,
            *hero.played_cards,
            *hero.discard_pile,
            *hero.hand,
            *hero.deck,
            *hero.spells,
        ]
        for c in search_cards:
            if c is not None and c.id == card_id:
                card = c
                break

        if not card or not card.current_effect_id:
            logger.debug(f"   [PERFORM] Card {card_id} not found or has no effect.")
            return StepResult(is_finished=True)

        # Prevention effects key off the card the action is performed ON, not the
        # card that granted the re-performance. Widget's gold re-performs a Skill
        # action on a resolved skill card, so Arien's Spell Break ("cannot perform
        # skill actions, except on gold cards") stops it — the skill card is the
        # one being acted on, and it is not gold.
        from goa2.engine.rules import can_perform_card_primary

        if not can_perform_card_primary(state, str(actor_id), card):
            logger.debug(
                f"   [PERFORM] {actor_id} may not perform the primary action of {card.name}."
            )
            return StepResult(is_finished=True)

        from goa2.engine.effects import CardEffectRegistry
        from goa2.engine.stats import compute_card_stats

        effect = CardEffectRegistry.get_for_card(card)
        if not effect:
            logger.debug(f"   [PERFORM] No effect registered for {card.current_effect_id}.")
            return StepResult(is_finished=True)

        stats = compute_card_stats(state, UnitID(str(actor_id)), card)
        # Signal that this is a re-performance of an already-resolved action (via
        # Bullet Time, Reload, etc.). Some effects gate behaviour on this — e.g.
        # Bounce only grants its "may repeat once" when re-performed. The card's
        # own state lags (it stays current_turn_card until FinalizeHeroTurnStep),
        # so build_steps can't rely on card.state alone.
        context["reperforming_card_id"] = card.id
        steps = effect.get_steps_with_stats(state, hero, card, stats)
        if self.exclude_target_key:
            self._inject_exclusion_filter(steps, self.exclude_target_key)

        logger.debug(f"   [PERFORM] Performing primary action of {card.name} ({len(steps)} steps)")
        return StepResult(is_finished=True, new_steps=steps)

    @classmethod
    def _inject_exclusion_filter(cls, steps: list[GameStep], exclude_key: str) -> None:
        from goa2.engine.steps.combat import AttackSequenceStep
        from goa2.engine.steps.effects import CreateEffectStep
        from goa2.engine.steps.selection import MultiSelectStep, SelectStep
        from goa2.engine.steps.utility import ForEachStep, MayRepeatNTimesStep

        exclusion = ExcludeIdentityFilter(exclude_self=False, exclude_keys=[exclude_key])

        for step in steps:
            if isinstance(step, (SelectStep, MultiSelectStep)) and step.target_type in (
                TargetType.UNIT,
                TargetType.UNIT_OR_TOKEN,
            ):
                step.filters.append(exclusion.model_copy(deep=True))
            elif isinstance(step, AttackSequenceStep):
                step.target_filters.append(exclusion.model_copy(deep=True))
            elif isinstance(step, (MayRepeatNTimesStep, ForEachStep)):
                cls._inject_exclusion_filter(step.steps_template, exclude_key)
            elif isinstance(step, CreateEffectStep):
                cls._inject_exclusion_filter(step.finishing_steps, exclude_key)
            elif isinstance(step, PerformPrimaryActionStep) and not step.exclude_target_key:
                # A nested copy (e.g. Reload performing another card's primary
                # action) builds its steps at runtime, so push the exclusion down
                # via the step's own key — it will inject it when it resolves.
                step.exclude_target_key = exclude_key


class PerformCardActionStep(GameStep):
    """
    Performs ANY action of an arbitrary card as the acting hero — the normal
    card-resolution menu minus defense (NebKher's Mind Grip: "Perform an
    action on the card in the previous turn slot of an enemy hero").

    Differences from PerformPrimaryActionStep:
    - Offers a chooser over the card's printed options (primary unless
      DEFENSE, secondaries minus DEFENSE — HOLD is always present, so at
      least one action always exists; DEFENSE_SKILL primaries are offered as
      SKILL).
    - Values come from THAT card, computed with the PERFORMER as actor.
    - Optional substitution flags for copied effects: ``token_type_override``
      (all token placements place that type instead — Mind Grip places
      Illusions) and ``skip_markers`` (marker steps are skipped, the rest of
      the effect continues). The flags are context-scoped around the copied
      steps so they reach nested templates and runtime-built sub-steps.
    """

    type: StepType = StepType.PERFORM_CARD_ACTION
    card_key: str = "selected_card"
    card_owner_key: str | None = None  # context key: whose card list to search
    hero_id: str | None = None  # performer (default: current actor)
    # Instead of card_key, use the OWNER's previous turn slot (Mind Grip).
    # Slot index = performer.resolved_turn_count - 1 (repo turn-index
    # convention; see HasPreviousSlotCardFilter).
    previous_slot: bool = False
    token_type_override: TokenType | None = None
    skip_markers: bool = False
    suppress_after_resolve_card: bool = False

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.stats import get_computed_stat

        if self.should_skip(context):
            return StepResult(is_finished=True)

        performer_id = self.hero_id or (
            str(state.current_actor_id) if state.current_actor_id else None
        )
        if not performer_id:
            return StepResult(is_finished=True)
        performer = state.get_hero(HeroID(performer_id))
        if not performer:
            return StepResult(is_finished=True)

        owner = performer
        if self.card_owner_key:
            owner_val = context.get(self.card_owner_key)
            if owner_val:
                owner = state.get_hero(HeroID(str(owner_val))) or performer

        card: Card | None = None
        if self.previous_slot:
            prev_index = performer.resolved_turn_count - 1
            if 0 <= prev_index < len(owner.played_cards):
                card = owner.played_cards[prev_index]
            if not card:
                logger.debug(f"   [PERFORM ANY] {owner.id} has no previous-slot card.")
                return StepResult(is_finished=True)
        else:
            card_id = context.get(self.card_key)
            if not card_id:
                return StepResult(is_finished=True)
            for c in [
                owner.current_turn_card,
                *owner.played_cards,
                *owner.discard_pile,
                *owner.hand,
                *owner.deck,
                *owner.spells,
            ]:
                if c is not None and c.id == card_id:
                    card = c
                    break
            if not card:
                logger.debug(f"   [PERFORM ANY] Card {card_id} not found on {owner.id}.")
                return StepResult(is_finished=True)

        def is_action_available(act_type: ActionType, *, is_primary: bool) -> bool:
            val_res = state.validator.can_perform_action(
                state, performer_id, act_type, context={"card": card}
            )
            if not val_res.allowed:
                return False
            return act_type not in (
                ActionType.ATTACK,
                ActionType.SKILL,
            ) or action_passes_initial_target_gate(
                state, performer, card, act_type, is_primary=is_primary
            )

        def compute_option(act_type: ActionType, base_val: int | None) -> tuple[int, str]:
            final_val = base_val or 0
            text_val = str(final_val) if base_val is not None else "-"
            stat_type = None
            if act_type == ActionType.MOVEMENT:
                stat_type = StatType.MOVEMENT
            elif act_type == ActionType.ATTACK:
                stat_type = StatType.ATTACK
            if stat_type:
                final_val = get_computed_stat(state, UnitID(performer_id), stat_type, base_val or 0)
                text_val = str(final_val)
            return final_val, text_val

        def build_option(
            act_type: ActionType, base_val: int | None, label: str, *, is_primary: bool
        ) -> dict[str, Any] | None:
            if act_type == ActionType.DEFENSE:
                return None
            display_type = ActionType.SKILL if act_type == ActionType.DEFENSE_SKILL else act_type
            if not is_action_available(display_type, is_primary=is_primary):
                return None
            c_val, c_text = compute_option(display_type, base_val)
            return {
                "id": display_type.name,
                "type": display_type,
                "value": c_val,
                "text": f"{label}: {display_type.name} ({c_text})",
            }

        # Build the normal resolution menu minus defense.
        options: list[dict[str, Any]] = []
        primary_action = card.current_primary_action
        if primary_action:
            primary_opt = build_option(
                primary_action, card.current_primary_action_value, "Primary", is_primary=True
            )
            if primary_opt:
                options.append(primary_opt)

        for action_type, val in card.current_secondary_actions.items():
            secondary_opt = build_option(action_type, val, "Secondary", is_primary=False)
            if secondary_opt:
                options.append(secondary_opt)

        def request_action() -> StepResult:
            return StepResult(
                requires_input=True,
                input_request=create_input_request(
                    request_type=InputRequestType.CHOOSE_ACTION,
                    player_id=performer_id,
                    prompt=f"Choose an action to perform on {card.name}",
                    options=options,
                ),
            )

        if self.pending_input:
            choice_id = self.pending_input.get("selection")
            selected_opt = next((o for o in options if o["id"] == choice_id), None)
            if not selected_opt:
                logger.debug(
                    "   [PERFORM ANY] Rejected invalid action %r; re-requesting.", choice_id
                )
                self.pending_input = None
                return request_action()

            act_type = cast(ActionType, selected_opt["type"])
            val = cast(int, selected_opt["value"])
            is_primary = act_type == primary_action or (
                primary_action == ActionType.DEFENSE_SKILL and act_type == ActionType.SKILL
            )

            from goa2.engine.steps.phases import push_action_context

            push_action_context(
                context,
                action_type=act_type,
                card_id=card.id,
                card_owner_id=str(owner.id),
            )

            action_steps = self._build_action_steps(
                state, context, performer_id, performer, card, act_type, val, is_primary
            )
            lifecycle_steps = self._build_lifecycle_steps(
                state,
                performer_id,
                card,
                act_type,
                val,
                is_primary,
                action_steps,
            )
            from goa2.engine.steps.phases import RestoreActionContextStep

            return StepResult(
                is_finished=True,
                new_steps=[
                    *self._wrap_with_substitution_flags(lifecycle_steps),
                    RestoreActionContextStep(),
                ],
            )

        return request_action()

    def _build_action_steps(
        self,
        state: GameState,
        context: dict[str, Any],
        performer_id: str,
        performer: Hero,
        card: Card,
        act_type: ActionType,
        val: int,
        is_primary: bool,
    ) -> list[GameStep]:
        from goa2.engine.stats import compute_card_stats
        from goa2.engine.steps.combat import AttackSequenceStep
        from goa2.engine.steps.movement import FastTravelSequenceStep, MoveSequenceStep
        from goa2.engine.steps.utility import LogMessageStep

        if is_primary:
            context["reperforming_card_id"] = card.id
            stats = compute_card_stats(state, UnitID(performer_id), card)
            if card.current_effect_id:
                from goa2.engine.effects import CardEffectRegistry

                effect = CardEffectRegistry.get_for_card(card)
                if effect:
                    return effect.get_steps_with_stats(state, performer, card, stats)
            # No registered effect — fall back to bare primitives.
            if act_type == ActionType.ATTACK:
                return [AttackSequenceStep(damage=stats.primary_value, range_val=stats.range or 1)]
            if act_type == ActionType.MOVEMENT:
                return [MoveSequenceStep(unit_id=performer_id, range_val=stats.primary_value)]
            return []

        # Secondary primitives, mirroring ResolveCardStep.
        if act_type == ActionType.MOVEMENT:
            move_base = card.current_secondary_actions.get(act_type, val)
            return [
                MoveSequenceStep(
                    unit_id=performer_id,
                    range_val=move_base,
                    range_stat_type=StatType.MOVEMENT,
                )
            ]
        if act_type == ActionType.FAST_TRAVEL:
            return [FastTravelSequenceStep(unit_id=performer_id)]
        if act_type == ActionType.ATTACK:
            attack_base = card.current_secondary_actions.get(act_type, val)
            base_rng = card.get_base_stat_value(StatType.RANGE)
            if base_rng == 0:
                base_rng = 1
            return [
                AttackSequenceStep(
                    damage=attack_base,
                    range_val=base_rng,
                    damage_stat_type=StatType.ATTACK,
                    range_stat_type=StatType.RANGE,
                )
            ]
        if act_type == ActionType.CLEAR:
            from goa2.engine.steps.markers import RemoveTokenStep
            from goa2.engine.steps.selection import MultiSelectStep
            from goa2.engine.steps.utility import ForEachStep

            if not state.has_board_presence(performer_id):
                return [
                    LogMessageStep(message=f"{performer_id} attempted clear but is not on board.")
                ]
            return [
                MultiSelectStep(
                    min_selections=0,
                    max_selections=6,
                    filters=[
                        UnitTypeFilter(unit_type="TOKEN"),
                        RangeFilter(max_range=1),
                        ImmunityFilter(),
                    ],
                    output_key="clear_targets",
                    target_type=TargetType.UNIT_OR_TOKEN,
                    prompt="Select tokens to clear.",
                ),
                ForEachStep(
                    list_key="clear_targets",
                    item_key="target_id",
                    steps_template=[RemoveTokenStep(token_key="target_id")],
                ),
            ]
        # HOLD (or anything unhandled): nothing to do.
        return [LogMessageStep(message=f"{performer_id} performs {act_type.name}.")]

    def _build_lifecycle_steps(
        self,
        state: GameState,
        performer_id: str,
        card: Card,
        act_type: ActionType,
        val: int,
        is_primary: bool,
        action_steps: list[GameStep],
    ) -> list[GameStep]:
        from goa2.domain.models.enums import PassiveTrigger
        from goa2.engine.steps.effects import CheckPassiveAbilitiesStep
        from goa2.engine.steps.movement import ResolvePreActionMovementStep
        from goa2.engine.steps.pieces import ChooseActingPieceStep
        from goa2.engine.steps.utility import SetContextFlagStep

        steps: list[GameStep] = [
            ChooseActingPieceStep(hero_id=performer_id),
            CheckPassiveAbilitiesStep(trigger=PassiveTrigger.BEFORE_ACTION.value),
        ]
        specific_before = {
            ActionType.ATTACK: PassiveTrigger.BEFORE_ATTACK,
            ActionType.MOVEMENT: PassiveTrigger.BEFORE_MOVEMENT,
            ActionType.SKILL: PassiveTrigger.BEFORE_SKILL,
        }.get(act_type)
        if specific_before is not None:
            steps.append(CheckPassiveAbilitiesStep(trigger=specific_before.value))
        if is_primary:
            steps.extend(
                [
                    ResolvePreActionMovementStep(hero_id=performer_id),
                    ResolvePreActionDiscardStep(hero_id=performer_id),
                ]
            )
        steps.extend(action_steps)

        specific_after = {
            ActionType.ATTACK: PassiveTrigger.AFTER_ATTACK,
            ActionType.MOVEMENT: PassiveTrigger.AFTER_MOVEMENT,
        }.get(act_type)
        if act_type == ActionType.ATTACK and is_primary and card.current_effect_id:
            steps.extend(
                [
                    SetContextFlagStep(key="attack_effect_id", value=card.current_effect_id),
                    SetContextFlagStep(key="attack_card_id", value=card.id),
                ]
            )
        if specific_after is not None:
            steps.append(CheckPassiveAbilitiesStep(trigger=specific_after.value))
        if act_type == ActionType.SKILL and card.current_color in (
            CardColor.GOLD,
            CardColor.SILVER,
        ):
            steps.append(CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_BASIC_SKILL.value))
        if card.is_basic and act_type in (
            ActionType.ATTACK,
            ActionType.MOVEMENT,
            ActionType.SKILL,
        ):
            steps.extend(
                [
                    SetContextFlagStep(key="basic_action_type", value=act_type.value),
                    SetContextFlagStep(key="basic_action_value", value=val),
                ]
            )
            if act_type == ActionType.ATTACK:
                base_range = card.get_base_stat_value(StatType.RANGE) or 1
                steps.append(
                    SetContextFlagStep(
                        key="basic_action_range",
                        value=get_computed_stat(
                            state,
                            UnitID(performer_id),
                            StatType.RANGE,
                            base_range,
                        ),
                    )
                )
            if is_primary and card.current_effect_id:
                steps.extend(
                    [
                        SetContextFlagStep(
                            key="basic_action_effect_id",
                            value=card.current_effect_id,
                        ),
                        SetContextFlagStep(key="basic_action_card_id", value=card.id),
                    ]
                )
            steps.append(CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_BASIC_ACTION.value))
        if is_primary:
            steps.append(
                CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_PRIMARY_ACTION.value)
            )
        if not self.suppress_after_resolve_card:
            steps.append(CheckPassiveAbilitiesStep(trigger=PassiveTrigger.AFTER_RESOLVE_CARD.value))
        return steps

    def _wrap_with_substitution_flags(self, steps: list[GameStep]) -> list[GameStep]:
        """Bracket the copied action with the substitution context flags so
        every token placement / marker step resolving inside it — including
        nested templates and runtime-built sub-steps — sees them."""
        if not self.token_type_override and not self.skip_markers:
            return steps

        from goa2.engine.steps.markers import SKIP_MARKERS_KEY, TOKEN_TYPE_OVERRIDE_KEY
        from goa2.engine.steps.utility import SetContextFlagStep

        pre: list[GameStep] = []
        post: list[GameStep] = []
        if self.token_type_override:
            pre.append(
                SetContextFlagStep(
                    key=TOKEN_TYPE_OVERRIDE_KEY, value=self.token_type_override.value
                )
            )
            post.append(SetContextFlagStep(key=TOKEN_TYPE_OVERRIDE_KEY, value=None))
        if self.skip_markers:
            pre.append(SetContextFlagStep(key=SKIP_MARKERS_KEY, value=True))
            post.append(SetContextFlagStep(key=SKIP_MARKERS_KEY, value=None))
        return [*pre, *steps, *post]


class ConvertCardToItemStep(GameStep):
    """Converts a selected card into a permanent item for its owner hero.

    Reads a card ID from context[card_key], finds it in the hero's deck,
    increments hero.items[card.item], and sets card.state = CardState.ITEM.
    """

    type: StepType = StepType.CONVERT_CARD_TO_ITEM
    card_key: str  # context key → card ID
    hero_id: str = ""  # explicit hero ID (if empty, uses current actor)

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        if self.should_skip(context):
            return StepResult(is_finished=True)

        card_id = context.get(self.card_key)
        if not card_id:
            return StepResult(is_finished=True)

        actor_id = self.hero_id or state.current_actor_id
        hero = state.get_hero(HeroID(str(actor_id)))
        if not hero:
            return StepResult(is_finished=True)

        card = next((c for c in hero.deck if c.id == str(card_id)), None)
        if not card:
            logger.debug(f"   [CONVERT] Card {card_id} not found in {actor_id} deck")
            return StepResult(is_finished=True)

        stat = card.item
        if not stat:
            logger.debug(f"   [CONVERT] Card {card_id} has no item stat")
            return StepResult(is_finished=True)

        hero.items[stat] = hero.items.get(stat, 0) + 1
        card.state = CardState.ITEM
        logger.debug(f"   [CONVERT] {card.name} → permanent item (+1 {stat.name})")

        return StepResult(
            is_finished=True,
            events=[
                GameEvent(
                    event_type=GameEventType.ITEM_GAINED,
                    actor_id=str(actor_id),
                    metadata={
                        "stat_type": stat.value,
                        "amount": 1,
                        "source_card_id": card.id,
                    },
                )
            ],
        )


class ResolveUpgradesStep(GameStep):
    """
    Simultaneous Upgrade loop.
    Waits for players to finish their pending upgrades.
    """

    type: StepType = StepType.RESOLVE_UPGRADES

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        # Process input if provided
        if self.pending_input:
            selection = self.pending_input.get("selection")
            if isinstance(selection, dict):
                hero_id = selection.get("hero_id")
                card_id = selection.get("card_id")
                # UPGRADE_PHASE is a 'simultaneous' request and hero_id/card_id
                # come straight from the client. Only apply an upgrade the hero
                # actually earned (has a pending slot) and was actually offered
                # (a legal option) — otherwise a client could grant a free extra
                # upgrade or skip tiers to a higher-tier deck card.
                if hero_id and card_id and self._is_legal_upgrade(state, hero_id, card_id):
                    apply_hero_upgrade(state, hero_id, card_id)
            self.pending_input = None

        if not state.pending_upgrades:
            logger.debug("   [PHASE] All upgrades complete.")
            return StepResult(is_finished=True, new_steps=[RoundResetStep()])

        broadcast_data = {}
        for h_id, count in state.pending_upgrades.items():
            options = self._get_upgrade_options(state, h_id)
            broadcast_data[str(h_id)] = {"remaining": count, "options": options}

        return StepResult(
            requires_input=True,
            input_request=create_input_request(
                request_type=InputRequestType.UPGRADE_PHASE,
                player_id="simultaneous",
                prompt="Mandatory Upgrade Phase",
                players=broadcast_data,
            ),
        )

    def _is_legal_upgrade(self, state: GameState, hero_id: str, card_id: str) -> bool:
        """A submitted upgrade is legal only if the hero has a pending upgrade
        slot AND the chosen card is one of the currently offered option cards."""
        if hero_id not in state.pending_upgrades:
            logger.debug("   [UPGRADE] Rejected upgrade for %s: no pending slot.", hero_id)
            return False
        legal_ids = {
            cid for opt in self._get_upgrade_options(state, hero_id) for cid in opt["pair"]
        }
        if card_id not in legal_ids:
            logger.debug(
                "   [UPGRADE] Rejected upgrade %s for %s: not an offered option.",
                card_id,
                hero_id,
            )
            return False
        return True

    def _get_upgrade_options(self, state: GameState, hero_id: str):
        """
        Returns upgrade options for a hero.

        Note: Ultimate cards (Tier IV) are handled separately - they unlock
        automatically at level 8, so they should never appear as upgrade options.
        """
        hero = state.get_hero(HeroID(hero_id))
        if not hero:
            return []
        non_basic_colors = [CardColor.RED, CardColor.BLUE, CardColor.GREEN]
        hand_non_basics = [c for c in hero.hand if c.color in non_basic_colors]
        if not hand_non_basics:
            return []

        tier_map = {CardTier.I: 1, CardTier.II: 2, CardTier.III: 3}
        min_tier_val = min(tier_map.get(c.tier, 99) for c in hand_non_basics)

        # If all cards are Tier III, there are no upgrade options.
        # Ultimate cards auto-activate at level 8 (handled in _level_up).
        if min_tier_val == 3:
            return []

        eligible_colors = [c.color for c in hand_non_basics if tier_map.get(c.tier) == min_tier_val]
        next_tier_map = {1: CardTier.II, 2: CardTier.III}
        target_tier = next_tier_map.get(min_tier_val)
        if not target_tier:
            return []

        options = []
        for color in eligible_colors:
            pair = [
                c
                for c in hero.deck
                if c.color == color and c.tier == target_tier and c.state == CardState.DECK
            ]
            if len(pair) == 2:
                options.append(
                    {
                        "color": color,
                        "tier": target_tier,
                        "pair": [c.id for c in pair],
                        "card_details": [c.model_dump() for c in pair],
                    }
                )
        return options


class RoundResetStep(GameStep):
    """Resets round state and transitions to Planning."""

    type: StepType = StepType.ROUND_RESET

    def resolve(self, state: GameState, context: dict[str, Any]) -> StepResult:
        from goa2.engine.phases import record_position_snapshot

        state.round += 1
        state.turn = 1
        record_position_snapshot(state)
        state.phase = GamePhase.PLANNING
        state.heroes_defeated_this_round.clear()
        logger.debug(f"   [ROUND START] Round {state.round}, Turn {state.turn}")
        return StepResult(is_finished=True)


def apply_hero_upgrade(state: GameState, hero_id: str, chosen_card_id: str):
    """
    Executes the upgrade transition for a hero.
    1. Removes old tier card of same color.
    2. Adds chosen card to hand.
    3. Tucks pair card as item.
    4. Decrements pending count.
    """
    hero = state.get_hero(HeroID(hero_id))
    if not hero:
        return

    chosen_card = next((c for c in hero.deck if c.id == chosen_card_id), None)
    if not chosen_card:
        logger.debug(f"   [!] Upgrade Error: Chosen card {chosen_card_id} not found in deck.")
        return

    prev_card = None
    if chosen_card.tier != CardTier.IV:
        for c in hero.hand:
            if c.color == chosen_card.color:
                prev_card = c
                break

    pair_card = None
    if chosen_card.tier != CardTier.IV:
        pair_card = next(
            (
                c
                for c in hero.deck
                if c.color == chosen_card.color
                and c.tier == chosen_card.tier
                and c.id != chosen_card.id
            ),
            None,
        )

    if prev_card:
        logger.debug(
            f"   [UPGRADE] Removing {prev_card.id} (Tier {prev_card.tier.name}) from hand."
        )
        hero.hand.remove(prev_card)
        prev_card.state = CardState.RETIRED

    logger.debug(f"   [UPGRADE] Adding {chosen_card.id} (Tier {chosen_card.tier.name}) to hand.")
    chosen_card.state = CardState.HAND
    chosen_card.is_facedown = False
    hero.hand.append(chosen_card)

    if pair_card:
        stat = pair_card.item
        if stat:
            hero.items[stat] = hero.items.get(stat, 0) + 1
            logger.debug(f"   [UPGRADE] Tucking {pair_card.id} as Item (+1 {stat.name}).")
        pair_card.state = CardState.ITEM

    if hero_id in state.pending_upgrades:
        state.pending_upgrades[HeroID(hero_id)] -= 1
        if state.pending_upgrades[HeroID(hero_id)] <= 0:
            del state.pending_upgrades[HeroID(hero_id)]


def _one_man_army_bonus(state: GameState, zone) -> dict[TeamColor, int]:
    """Check for heroes with active one_man_army ultimate in the zone."""
    bonus = {TeamColor.RED: 0, TeamColor.BLUE: 0}
    for team in state.teams.values():
        for hero in team.heroes:
            if hero.level < 8 or not hero.ultimate_card:
                continue
            if hero.ultimate_card.effect_id != "one_man_army":
                continue
            # Owner-level get_position returns None for an unbound multi-piece
            # hero (its pieces hold the positions), which would silently drop
            # the bonus. Safe today because no hero is both multi-piece and
            # one_man_army; if that ever changes, iterate get_positions() and
            # count per-piece zone membership instead.
            hero_loc = state.get_position(str(hero.id))
            if hero_loc and hero_loc in zone.hexes and hero.team is not None:
                bonus[hero.team] += 1
                logger.debug(f"   [BATTLE] {hero.name} counts as a heavy minion (One Man Army)")
    return bonus
