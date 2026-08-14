from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from goa2.domain.models import (
    CardContainerType,
    TargetType,
)
from goa2.domain.models.effect import (
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_hex import (
    ObstacleFilter,
    RangeFilter,
)
from goa2.engine.filters_units import (
    AdjacencyFilter,
    ImmunityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckUnitOnBoardStep,
    CheckUnitTypeStep,
    CombineBooleanContextStep,
    CreateEffectStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    ForEachStep,
    GainCoinsStep,
    GameStep,
    MoveSequenceStep,
    MoveUnitStep,
    MultiSelectStep,
    PerformPrimaryActionStep,
    RemoveUnitStep,
    RetrieveCardStep,
    SelectStep,
    SetContextFlagStep,
    SpendAdditionalLifeCounterStep,
    SwapUnitsStep,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _has_ultimate(hero: Hero) -> bool:
    """Check if Unbound Fury is in passive play.

    Scoped to the card, not just "has an ultimate at level 8": every hero has
    an ultimate, and NebKher's Mind Grip runs an Ursafar card with the
    PERFORMER as ``hero``, so an unscoped check enrages whoever copies the card.
    """
    ult = hero.ultimate_card
    return ult is not None and ult.id == "unbound_fury" and hero.level >= 8


def is_enraged(state: GameState, hero: Hero) -> bool:
    """Check if this hero is currently enraged.

    A hero is enraged if:
    - They own a live ENRAGED effect, i.e. they PERFORMED a card that said
      "This round: You are enraged"
    - OR Unbound Fury (their ultimate) is in passive play

    Ownership is ``ActiveEffect.source_id`` — the performer, which is not
    always the card's owner. NebKher's Mind Grip performs a card sitting in
    Ursafar's turn slot: the rage is NebKher's, while the effect stays bound
    to the Ursafar card (``source_card_id``). Reading ``card.is_active`` off
    ``played_cards`` instead would hand the rage to the wrong hero, since
    EffectManager marks the performed card active whoever performed it.

    Dormant effects don't count: an effect bound to a card that has not
    resolved yet ("played, but not yet resolved") has not turned its
    "this round" clause on.
    """
    if _has_ultimate(hero):
        return True
    return any(
        effect.effect_type == EffectType.ENRAGED
        and effect.source_id == str(hero.id)
        and effect.is_active
        for effect in state.active_effects
    )


def _enraged_effect_step() -> CreateEffectStep:
    """Create the standard 'This round: You are enraged.' step."""
    return CreateEffectStep(
        effect_type=EffectType.ENRAGED,
        scope=EffectScope(
            shape=Shape.POINT,
            affects=AffectsFilter.SELF,
        ),
        duration=DurationType.THIS_ROUND,
        is_active=True,
    )


# =============================================================================
# GROUP A — Simple conditionals
# =============================================================================


# TIER II — BLUE: Cold Ire (MOVEMENT)
# "If enraged, gain +1 Movement. This round: You are enraged."
@register_effect("cold_ire")
class ColdIreEffect(CardEffect):
    """If enraged, gain +1 Movement. This round: You are enraged."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        bonus = 1 if is_enraged(state, hero) else 0
        return [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value + bonus),
            _enraged_effect_step(),
        ]


# TIER III — BLUE: Eyes of Flame (MOVEMENT)
# "If enraged, gain +2 Movement. This round: You are enraged."
@register_effect("eyes_of_flame")
class EyesOfFlameEffect(CardEffect):
    """If enraged, gain +2 Movement. This round: You are enraged."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        bonus = 2 if is_enraged(state, hero) else 0
        return [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value + bonus),
            _enraged_effect_step(),
        ]


# TIER II — RED: Rip (ATTACK)
# "Target a unit adjacent to you.
#  After the attack: If enraged, gain 1 coin.
#  This round: You are enraged."
@register_effect("rip")
class RipEffect(CardEffect):
    """Attack adjacent. If enraged, gain 1 coin. This round: You are enraged."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    SetContextFlagStep(key="self_hero", value=hero.id),
                    GainCoinsStep(hero_key="self_hero", amount=1),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# TIER I — GREEN: Sniff Out (SKILL, RANGED)
# "If enraged, an enemy hero in range discards a card, if able."
@register_effect("sniff_out")
class SniffOutEffect(CardEffect):
    """If enraged, an enemy hero in range discards a card, if able."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        if not is_enraged(state, hero):
            return []
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range to discard a card",
                output_key="victim_id",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range),
                    ImmunityFilter(),
                ],
            ),
            ForceDiscardStep(victim_key="victim_id"),
        ]


# TIER II — GREEN: Eyes on the Prey (SKILL, RANGED)
# "If enraged, an enemy hero in range discards a card, if able."
@register_effect("eyes_on_the_prey")
class EyesOnThePreyEffect(SniffOutEffect):
    """If enraged, an enemy hero in range discards a card, if able."""

    pass


# TIER III — GREEN: Apex Predator (SKILL, RANGED)
# "If enraged, an enemy hero in range discards a card, or is defeated."
@register_effect("apex_predator")
class ApexPredatorEffect(CardEffect):
    """If enraged, an enemy hero in range discards a card, or is defeated."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        if not is_enraged(state, hero):
            return []
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range",
                output_key="victim_id",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range),
                    ImmunityFilter(),
                ],
            ),
            ForceDiscardOrDefeatStep(victim_key="victim_id"),
        ]


# =============================================================================
# GROUP B — Attack + conditional minion removal
# =============================================================================


# TIER I — RED: Prey Drive (ATTACK)
# "Target a unit adjacent to you. After the attack: If enraged, and the
#  target was not removed, remove up to 1 enemy minion in radius.
#  This round: You are enraged."
@register_effect("prey_drive")
class PreyDriveEffect(CardEffect):
    """Attack adjacent. If enraged + target not removed, remove up to 1 enemy minion in radius."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    CheckUnitOnBoardStep(
                        unit_key="victim_id",
                        output_key="target_not_removed",
                    ),
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select an enemy minion in radius to remove (optional).",
                        output_key="remove_target",
                        is_mandatory=False,
                        active_if_key="target_not_removed",
                        filters=[
                            UnitTypeFilter(unit_type="MINION"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=stats.radius),
                        ],
                    ),
                    RemoveUnitStep(
                        unit_key="remove_target",
                        active_if_key="remove_target",
                    ),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# TIER II — RED: Prey Abundance (ATTACK)
# "Target a unit adjacent to you. After the attack: If enraged, and the
#  target was not removed, remove up to 1 enemy minion in radius.
#  This round: You are enraged."
@register_effect("prey_abundance")
class PreyAbundanceEffect(PreyDriveEffect):
    """Same pattern as Prey Drive with different stats."""

    pass


# TIER III — RED: Feeding Frenzy (ATTACK)
# "Target a unit adjacent to you. After the attack: If enraged, and the
#  target was not removed, remove up to 2 enemy minions in radius.
#  This round: You are enraged."
@register_effect("feeding_frenzy")
class FeedingFrenzyEffect(CardEffect):
    """Attack adjacent. If enraged + target not removed, remove up to 2 enemy minions in radius."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    CheckUnitOnBoardStep(
                        unit_key="victim_id",
                        output_key="target_not_removed",
                    ),
                    MultiSelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select enemy minions in radius to remove.",
                        output_key="remove_targets",
                        max_selections=2,
                        min_selections=0,
                        is_mandatory=False,
                        active_if_key="target_not_removed",
                        filters=[
                            UnitTypeFilter(unit_type="MINION"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=stats.radius),
                        ],
                    ),
                    ForEachStep(
                        list_key="remove_targets",
                        item_key="current_remove_target",
                        steps_template=[
                            RemoveUnitStep(unit_key="current_remove_target"),
                        ],
                    ),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# =============================================================================
# GROUP C — Movement + swap + additional movement
# =============================================================================


# TIER I — BLUE: Prowling Brute (MOVEMENT)
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you. This round: You are enraged."
@register_effect("prowling_brute")
class ProwlingBruteEffect(CardEffect):
    """Move. If enraged, may swap with adjacent unit or token."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.UNIT_OR_TOKEN,
                        prompt="Select a unit or token adjacent to you to swap with (optional).",
                        output_key="swap_target",
                        is_mandatory=False,
                        filters=[
                            RangeFilter(max_range=1),
                        ],
                    ),
                    SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# TIER II — BLUE: Rampaging Beast (MOVEMENT)
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you; if you do, move up to 1 additional space.
#  This round: You are enraged."
@register_effect("rampaging_beast")
class RampagingBeastEffect(CardEffect):
    """Move. If enraged, may swap with adjacent unit/token; if you do, move 1 more."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.UNIT_OR_TOKEN,
                        prompt="Select a unit or token adjacent to you to swap with (optional).",
                        output_key="swap_target",
                        is_mandatory=False,
                        filters=[
                            RangeFilter(max_range=1),
                        ],
                    ),
                    SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
                    MoveSequenceStep(
                        unit_id=hero.id,
                        range_val=1,
                        is_mandatory=False,
                        active_if_key="swap_target",
                    ),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# TIER III — BLUE: Unstoppable Force (MOVEMENT)
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you; if you do, move up to 2 additional spaces.
#  This round: You are enraged."
@register_effect("unstoppable_force")
class UnstoppableForceEffect(CardEffect):
    """Move. If enraged, may swap with adjacent unit/token; if you do, move 2 more."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.UNIT_OR_TOKEN,
                        prompt="Select a unit or token adjacent to you to swap with (optional).",
                        output_key="swap_target",
                        is_mandatory=False,
                        filters=[
                            RangeFilter(max_range=1),
                        ],
                    ),
                    SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
                    MoveSequenceStep(
                        unit_id=hero.id,
                        range_val=2,
                        is_mandatory=False,
                        active_if_key="swap_target",
                    ),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# =============================================================================
# GROUP D — Pre-attack movement
# =============================================================================


# UNTIERED — GOLD: Claws That Catch (ATTACK)
# "Before the attack: If enraged, you may move 1 space to a space adjacent
#  to an enemy hero. Target a unit adjacent to you.
#  This round: You are enraged."
@register_effect("claws_that_catch")
class ClawsThatCatchEffect(CardEffect):
    """If enraged, may move 1 to hex adjacent to enemy hero. Attack adjacent. Set enraged."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = []
        if is_enraged(state, hero):
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.HEX,
                        prompt="Move 1 space to a space adjacent to an enemy hero (optional).",
                        output_key="dash_hex",
                        is_mandatory=False,
                        filters=[
                            RangeFilter(max_range=1),
                            ObstacleFilter(),
                            AdjacencyFilter(target_tags=["ENEMY", "HERO"]),
                        ],
                    ),
                    MoveUnitStep(
                        unit_id=hero.id,
                        destination_key="dash_hex",
                        is_mandatory=False,
                    ),
                ]
            )
        steps.extend(
            [
                AttackSequenceStep(damage=stats.primary_value, range_val=1),
                _enraged_effect_step(),
            ]
        )
        return steps


# =============================================================================
# GROUP E — Complex effects / new steps
# =============================================================================


# TIER III — RED: Tear (ATTACK)
# "Target a unit adjacent to you.
#  After the attack: If enraged, gain 2 coins;
#  if you defeated a hero, that hero spends 1 additional Life counter.
#  This round: You are enraged."
@register_effect("tear")
class TearEffect(CardEffect):
    """Attack adjacent. If enraged: gain 2 coins; if defeated hero, spend 1 extra life counter."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
        ]
        if is_enraged(state, hero):
            steps.extend(
                [
                    SetContextFlagStep(key="self_hero", value=hero.id),
                    GainCoinsStep(hero_key="self_hero", amount=2),
                    # Check if target was defeated (block_succeeded=False → 0)
                    CheckContextConditionStep(
                        input_key="block_succeeded",
                        operator="==",
                        threshold=0,
                        output_key="target_defeated",
                    ),
                    # Check if target was a hero
                    CheckUnitTypeStep(
                        unit_key="victim_id",
                        expected_type="HERO",
                        output_key="target_is_hero",
                    ),
                    # Combine: defeated AND hero
                    CombineBooleanContextStep(
                        key_a="target_defeated",
                        key_b="target_is_hero",
                        output_key="hero_defeated",
                        operation="AND",
                    ),
                    SpendAdditionalLifeCounterStep(
                        victim_key="victim_id",
                        active_if_key="hero_defeated",
                    ),
                ]
            )
        steps.append(_enraged_effect_step())
        return steps


# UNTIERED — SILVER: Angry Roar (SKILL)
# "If enraged, perform the primary action on one of your active cards
#  with an active effect.
#  This round: You are enraged."
@register_effect("angry_roar")
class AngryRoarEffect(CardEffect):
    """If enraged, perform primary action of an active card with active effect."""

    def _get_active_effect_card_ids(self, state: GameState, hero: Hero, card: Card) -> list[str]:
        """Get IDs of played cards that are active AND whose effect produces CreateEffectStep."""
        from goa2.engine.effects import CardEffectRegistry
        from goa2.engine.stats import compute_card_stats

        valid_ids = []
        for c in hero.played_cards:
            if c is None or c.id == card.id:
                continue
            if not c.is_active:
                continue
            if not c.current_effect_id:
                continue
            effect = CardEffectRegistry.get(c.current_effect_id)
            if not effect:
                continue
            stats = compute_card_stats(state, hero.id, c)
            steps = effect.build_steps(state, hero, c, stats)
            if any(isinstance(s, CreateEffectStep) for s in steps):
                valid_ids.append(c.id)
        return valid_ids

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = []
        if is_enraged(state, hero):
            valid_card_ids = self._get_active_effect_card_ids(state, hero, card)
            if valid_card_ids:
                steps.extend(
                    [
                        SelectStep(
                            target_type=TargetType.CARD,
                            card_container=CardContainerType.PLAYED,
                            card_is_active=True,
                            prompt="Select an active card with an active effect to perform its primary action.",
                            output_key="roar_card",
                            is_mandatory=True,
                            allowed_card_ids=valid_card_ids,
                        ),
                        PerformPrimaryActionStep(
                            card_key="roar_card",
                            hero_id=hero.id,
                        ),
                    ]
                )
        steps.append(_enraged_effect_step())
        return steps


# TIER II — GREEN: Instinctive Reaction (SKILL)
# "If enraged, choose one —
#  • Perform the primary action on one of your discarded cards.
#  • You may retrieve a discarded card."
@register_effect("instinctive_reaction")
class InstinctiveReactionEffect(CardEffect):
    """If enraged, choose one: perform primary action of discarded card OR retrieve discarded card."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        if not is_enraged(state, hero):
            return []
        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="ir_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Perform the primary action of a discarded card",
                    2: "Retrieve a discarded card",
                },
                is_mandatory=True,
            ),
            # Branch 1: Perform primary action of a discarded card
            CheckContextConditionStep(
                input_key="ir_choice",
                operator="==",
                threshold=1,
                output_key="chose_perform",
            ),
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="Select a discarded card to perform its primary action.",
                output_key="perform_card",
                is_mandatory=True,
                active_if_key="chose_perform",
            ),
            PerformPrimaryActionStep(
                card_key="perform_card",
                hero_id=hero.id,
                active_if_key="chose_perform",
            ),
            # Branch 2: Retrieve a discarded card
            CheckContextConditionStep(
                input_key="ir_choice",
                operator="==",
                threshold=2,
                output_key="chose_retrieve",
            ),
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="Select a discarded card to retrieve.",
                output_key="retrieve_card",
                is_mandatory=False,
                active_if_key="chose_retrieve",
            ),
            RetrieveCardStep(
                card_key="retrieve_card",
                active_if_key="retrieve_card",
            ),
        ]


# TIER III — GREEN: Evolutionary Response (SKILL)
# "If enraged, choose one, or both —
#  • Perform the primary action on one of your discarded cards.
#  • You may retrieve a discarded card."
@register_effect("evolutionary_response")
class EvolutionaryResponseEffect(CardEffect):
    """If enraged, choose one or both: perform primary action of discarded card / retrieve discarded card."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        if not is_enraged(state, hero):
            return []

        # "Choose one, or both" leaves the order to the player: performing a
        # discarded card's primary action can change what is in the discard,
        # so retrieve-then-perform is a materially different line from
        # perform-then-retrieve. Both orders are offered explicitly.
        def _retrieve_steps(slot: str, gate: str) -> list[GameStep]:
            card_key = f"er_retrieve_card_{slot}"
            return [
                SelectStep(
                    target_type=TargetType.CARD,
                    card_container=CardContainerType.DISCARD,
                    prompt="Select a discarded card to retrieve.",
                    output_key=card_key,
                    is_mandatory=False,
                    active_if_key=gate,
                ),
                RetrieveCardStep(card_key=card_key, active_if_key=card_key),
            ]

        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one, or both",
                output_key="er_choice",
                number_options=[1, 2, 3, 4],
                number_labels={
                    1: "Perform the primary action of a discarded card",
                    2: "Retrieve a discarded card",
                    3: "Both — perform first, then retrieve",
                    4: "Both — retrieve first, then perform",
                },
                is_mandatory=True,
            ),
            # Perform on 1, 3 and 4 (i.e. everything except retrieve-only).
            CheckContextConditionStep(
                input_key="er_choice",
                operator="!=",
                threshold=2,
                output_key="do_perform",
            ),
            # Retrieve before the perform only on 4.
            CheckContextConditionStep(
                input_key="er_choice",
                operator="==",
                threshold=4,
                output_key="er_retrieve_first",
            ),
            # Retrieve after the perform on 2 and 3 (2 <= choice <= 3).
            CheckContextConditionStep(
                input_key="er_choice",
                operator=">=",
                threshold=2,
                output_key="_er_ge_2",
            ),
            CheckContextConditionStep(
                input_key="er_choice",
                operator="<=",
                threshold=3,
                output_key="_er_le_3",
            ),
            CombineBooleanContextStep(
                key_a="_er_ge_2",
                key_b="_er_le_3",
                output_key="er_retrieve_last",
                operation="AND",
            ),
            *_retrieve_steps("first", "er_retrieve_first"),
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="Select a discarded card to perform its primary action.",
                output_key="perform_card",
                is_mandatory=True,
                active_if_key="do_perform",
            ),
            PerformPrimaryActionStep(
                card_key="perform_card",
                hero_id=hero.id,
                active_if_key="do_perform",
            ),
            *_retrieve_steps("last", "er_retrieve_last"),
        ]


# =============================================================================
# GROUP F — Ultimate
# =============================================================================


# UNTIERED — PURPLE: Unbound Fury (PASSIVE)
# "You are always enraged, and all your resolved cards count as active."
@register_effect("unbound_fury")
class UnboundFuryEffect(CardEffect):
    """Passive: always enraged, all resolved cards count as active."""

    def on_ultimate_unlocked(self, state: GameState, hero: Hero) -> None:
        """Called when hero reaches level 8. Sets enraged_active_override on all cards."""
        all_cards = hero.deck + hero.hand + hero.played_cards + hero.discard_pile
        if hero.ultimate_card:
            all_cards.append(hero.ultimate_card)
        for card in all_cards:
            if card is not None:
                card.enraged_active_override = True
        logger.debug(
            "[UNBOUND FURY] All %s cards for %s now have enraged_active_override=True",
            len([c for c in all_cards if c is not None]),
            hero.id,
        )

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        # Passive — no active steps
        return []


# =============================================================================
# URSAFAR EFFECTS — HIGH-LEVEL DESIGN
# =============================================================================
#
# CORE MECHANIC: ENRAGED
# ----------------------
# Almost every Ursafar card says "This round: You are enraged."
# Some cards have "If enraged, ..." conditionals that gate bonus effects.
#
# Implementation:
#   - EffectType.ENRAGED in domain/models/effect.py
#   - Cards that SET enraged (red, blue, gold, silver) create an ActiveEffect
#     via CreateEffectStep(effect_type=EffectType.ENRAGED, duration=THIS_ROUND, ...)
#   - Cards that only CHECK enraged (all greens) do NOT create this effect
#   - This also sets card.is_active = True (via EffectManager linkage), which
#     is what Angry Roar's "active cards" chooser reads — but it is NOT what
#     rage is: the mark lands on the performed card whoever performed it.
#   - Helper function is_enraged() checks whether the hero OWNS a live ENRAGED
#     effect (ActiveEffect.source_id == hero), or has Unbound Fury in play
#   - is_enraged() is called at build_steps time to conditionally include bonus steps
#
# WHO OWNS THE RAGE (locked 2026-08-08)
# -------------------------------------
# The performer, not the card's owner. NebKher's Mind Grip performs a card
# sitting in Ursafar's turn slot: the ENRAGED effect's source_id is NebKher
# (so HE becomes enraged for the round) while source_card_id stays the Ursafar
# card it came from. Ursafar gains nothing from being copied. Deriving rage
# from card.is_active on played_cards would invert that, and scoping the
# ultimate check to "has an ultimate at level 8" would enrage every copier —
# both were live bugs, see tests/engine/test_ursafar_enraged_scope.py.
#
# Dormant effects don't count: an effect bound to a card that has not resolved
# yet is "played, but not yet resolved" and its "this round" clause is not on.
#
# ULTIMATE MECHANIC: enraged_active_override on Card
# ----------------------------------------------------
# "All your resolved cards count as active" — the ultimate sets
# card.enraged_active_override = True on ALL of the hero's cards.
# The Card.is_active property returns (is_active_base or enraged_active_override),
# so card.is_active always reads True — for our effects, other heroes' effects,
# filters, views, anything. EffectManager is unchanged; it writes to is_active
# (which hits the setter on is_active_base), but the getter always wins.
# Angry Roar can just check card.is_active directly — no special helper needed.
#
#
# =============================================================================
# CARD EFFECTS BY TIER
# =============================================================================
#
# UNTIERED — GOLD: Claws That Catch (ATTACK)
# --------------------------------------------
# "Before the attack: If enraged, you may move 1 space to a space adjacent
#  to an enemy hero. Target a unit adjacent to you.
#  This round: You are enraged."
#
# Steps:
#   if is_enraged:
#     MoveSequenceStep(range=1, filters=[AdjacencyFilter(to enemy hero)])  (optional)
#   AttackSequenceStep(damage, range=1)
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Reference: Brogan's Mad Dash (movement before attack)
#
#
# UNTIERED — SILVER: Angry Roar (SKILL)
# ---------------------------------------
# "If enraged, perform the primary action on one of your active cards
#  with an active effect.
#  This round: You are enraged."
#
# Steps:
#   if is_enraged:
#     SelectStep(target_type=CARD, filters=[active cards with active effects])
#     PerformPrimaryActionStep(card_key=...) — NEW STEP needed
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Note: Needs a new step to perform the primary action of a selected card.
#       Primary actions are ATTACK (with value), MOVEMENT (with value), or SKILL.
#       This step reads the selected card's primary_action and primary_action_value,
#       then pushes the appropriate sub-steps (AttackSequenceStep or MoveSequenceStep).
#       Just check card.is_active — the enraged_active_override handles the ultimate.
#
#
# TIER I — RED: Prey Drive (ATTACK)
# -----------------------------------
# "Target a unit adjacent to you.
#  After the attack: If enraged, and the target was not removed,
#  remove up to 1 enemy minion in radius.
#  This round: You are enraged."
#
# Steps:
#   AttackSequenceStep(damage, range=1)
#   if is_enraged:
#     CheckContextConditionStep(target not removed)
#     SelectStep(UNIT, filters=[MINION, ENEMY, RangeFilter(radius)], max=1, optional)
#     RemoveUnitStep(unit_key=...)
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Reference: Sabina's Shootout (attack + conditional minion removal)
#
#
# TIER I — BLUE: Prowling Brute (MOVEMENT)
# ------------------------------------------
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you.
#  This round: You are enraged."
#
# Steps:
#   MoveSequenceStep(range=primary_value)
#   if is_enraged:
#     SelectStep(UNIT, filters=[AdjacencyFilter], optional)
#     SwapUnitsStep(hero, selected_unit)
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Reference: Sabina's Back to Back (swap with adjacent unit)
#
#
# TIER I — GREEN: Sniff Out (SKILL, RANGED)  [IMPLEMENTED]
# -------------------------------------------
# "If enraged, an enemy hero in range discards a card, if able."
#
# Reference: Sabina's Close Support (force discard if able)
#
#
# =============================================================================
# TIER II
# =============================================================================
#
# TIER II — RED: Rip (ATTACK)  [IMPLEMENTED]
# -----------------------------
# "Target a unit adjacent to you.
#  After the attack: If enraged, gain 1 coin.
#  This round: You are enraged."
#
# Reference: Sabina's War Drummer (conditional coin gain)
#
#
# TIER II — RED: Prey Abundance (ATTACK)
# ----------------------------------------
# "Target a unit adjacent to you.
#  After the attack: If enraged, and the target was not removed,
#  remove up to 1 enemy minion in radius.
#  This round: You are enraged."
#
# Steps: Same pattern as Prey Drive (Tier I red) with different stats.
#
#
# TIER II — BLUE: Cold Ire (MOVEMENT)  [IMPLEMENTED]
# --------------------------------------
# "If enraged, gain +1 Movement.
#  This round: You are enraged."
#
#
# TIER II — BLUE: Rampaging Beast (MOVEMENT)
# --------------------------------------------
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you; if you do, move up to 1 additional space.
#  This round: You are enraged."
#
# Steps:
#   MoveSequenceStep(range=primary_value)
#   if is_enraged:
#     SelectStep(UNIT, filters=[AdjacencyFilter], optional)
#     SwapUnitsStep(hero, selected_unit)
#     MoveSequenceStep(range=1, optional)  — only if swap happened
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Reference: Sabina's Back to Back + Prowling Brute pattern extended
#
#
# TIER II — GREEN: Instinctive Reaction (SKILL)
# ------------------------------------------------
# "If enraged, choose one —
#  - Perform the primary action on one of your discarded cards.
#  - You may retrieve a discarded card."
#
# Steps:
#   if is_enraged:
#     SelectStep(CHOOSE_ACTION, options=["perform_primary", "retrieve"])
#     Branch A: SelectStep(CARD, discard) → PerformPrimaryActionStep (NEW STEP)
#     Branch B: SelectStep(CARD, discard) → RetrieveCardStep
#   (NO enraged set — green cards only check, never set)
#
# Note: Needs CHOOSE_ACTION + branching pattern.
# Reference: Sabina's Steady Advance (card retrieval from discard)
#
#
# TIER II — GREEN: Eyes on the Prey (SKILL, RANGED)  [IMPLEMENTED]
# ---------------------------------------------------
# "If enraged, an enemy hero in range discards a card, if able."
#
# Steps: Same pattern as Sniff Out (Tier I green) with different stats.
#   (NO enraged set — green cards only check, never set)
#
#
# =============================================================================
# TIER III
# =============================================================================
#
# TIER III — RED: Tear (ATTACK)
# -------------------------------
# "Target a unit adjacent to you.
#  After the attack: If enraged, gain 2 coins;
#  if you defeated a hero, that hero spends 1 additional Life counter.
#  This round: You are enraged."
#
# Steps:
#   AttackSequenceStep(damage, range=1)
#   if is_enraged:
#     GainCoinsStep(hero_key, amount=2)
#     CheckHeroDefeatedThisRoundStep(...)
#     SpendLifeCounterStep(victim_key=...) — NEW STEP or reuse existing defeat logic
#   CreateEffectStep(ENRAGED, THIS_ROUND)
#
# Note: "Spends 1 additional Life counter" is a unique mechanic. Need to check
#       if there's existing support or if a new step is needed.
# Reference: Sabina's War Drummer (coin gain), Brogan's defeat checks
#
#
# TIER III — RED: Feeding Frenzy (ATTACK)
# -----------------------------------------
# "Target a unit adjacent to you.
#  After the attack: If enraged, and the target was not removed,
#  remove up to 2 enemy minions in radius.
#  This round: You are enraged."
#
# Steps: Same pattern as Prey Drive/Prey Abundance but max=2 minions.
#
# Reference: Sabina's Bullet Hell (multi-select minion removal)
#
#
# TIER III — BLUE: Eyes of Flame (MOVEMENT)  [IMPLEMENTED]
# -------------------------------------------
# "If enraged, gain +2 Movement.
#  This round: You are enraged."
#
#
# TIER III — BLUE: Unstoppable Force (MOVEMENT)
# ------------------------------------------------
# "If enraged, after movement, you may swap with a unit or a token
#  adjacent to you; if you do, move up to 2 additional spaces.
#  This round: You are enraged."
#
# Steps: Same pattern as Rampaging Beast but 2 additional spaces.
#
#
# TIER III — GREEN: Evolutionary Response (SKILL)
# -------------------------------------------------
# "If enraged, choose one, or both —
#  - Perform the primary action on one of your discarded cards.
#  - You may retrieve a discarded card."
#
# Steps:
#   if is_enraged:
#     SelectStep(CHOOSE_ACTION, options=["perform_primary", "retrieve", "both"])
#     Branch A: SelectStep(CARD, discard) → PerformPrimaryActionStep
#     Branch B: SelectStep(CARD, discard) → RetrieveCardStep
#     Branch C: Both A then B
#   (NO enraged set — green cards only check, never set)
#
# Note: "Choose one, or both" is the Tier III upgrade of Instinctive Reaction's
#       "choose one". Same mechanic, expanded choice.
#
#
# TIER III — GREEN: Apex Predator (SKILL, RANGED)  [IMPLEMENTED]
# -------------------------------------------------
# "If enraged, an enemy hero in range discards a card, or is defeated."
#
# Reference: Sabina's Covering Fire (discard or defeat)
#
#
# =============================================================================
# ULTIMATE (PURPLE/TIER IV): Unbound Fury (PASSIVE)
# =============================================================================
#
# "You are always enraged, and all your resolved cards count as active."
#
# Implementation:
#   - is_enraged() checks _has_ultimate(hero) first → returns True immediately
#   - On ultimate activation, set card.enraged_active_override = True on ALL
#     of the hero's cards (deck, hand, played, discard). The Card.is_active
#     property returns (is_active_base or enraged_active_override), so
#     card.is_active always reads True for every card. EffectManager unchanged.
#   - This affects Angry Roar, other heroes' effect checks, views — everything.
#
#
# =============================================================================
# EFFECT TIERS / IMPLEMENTATION ORDER
# =============================================================================
#
# Shared foundation (implement first):  [DONE]
#   1. EffectType.ENRAGED in domain/models/effect.py
#   2. is_enraged() helper in this file
#   3. enraged_active_override field on Card model
#   4. _enraged_effect_step() helper in this file
#
# Group A — Simple conditionals (easiest, no new steps):  [DONE]
#   - Cold Ire / Eyes of Flame (movement bonus)
#   - Rip (attack + coin gain) — Tear uses same pattern + extra defeat logic
#   - Sniff Out / Eyes on the Prey (force discard if able)
#   - Apex Predator (force discard or defeat)
#
# Group B — Attack + conditional minion removal:
#   - Prey Drive / Prey Abundance / Feeding Frenzy
#
# Group C — Movement + swap + additional movement:
#   - Prowling Brute / Rampaging Beast / Unstoppable Force
#
# Group D — Pre-attack movement:
#   - Claws That Catch
#
# Group E — Needs new steps or patterns:
#   - Instinctive Reaction / Evolutionary Response (PerformPrimaryActionStep)
#   - Angry Roar (perform primary action of active card — just checks card.is_active)
#   - Tear's "spend 1 additional Life counter" (SpendLifeCounterStep?)
#
# Group F — Ultimate:
#   - Unbound Fury (always enraged + enraged_active_override on all cards)
#
# =============================================================================
