from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import (
    CardContainerType,
    StatType,
    TargetType,
)
from goa2.domain.types import UnitID
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_cards import CardsInContainerFilter
from goa2.engine.filters_composite import OrFilter
from goa2.engine.filters_hex import (
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
)
from goa2.engine.filters_units import (
    AdjacencyFilter,
    AdjacencyToContextFilter,
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.stats import get_computed_stat
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckUnitTypeStep,
    CountCardsStep,
    CountStep,
    DefeatUnitStep,
    ForceDiscardStep,
    GainCoinsStep,
    GameStep,
    MayRepeatOnceStep,
    MoveSequenceStep,
    MoveUnitStep,
    PlaceUnitStep,
    RetrieveCardStep,
    SelectStep,
    SetContextFlagStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# Fabled Lance (Ultimate) helpers — used by all attack effects
# =============================================================================


def _fabled_lance_active(hero: Hero) -> bool:
    """Check if Rowenna's ultimate passive is active (level >= 8)."""
    return hero.level >= 8


def _attack_range(state: GameState, hero: Hero) -> int:
    """Return attack range: 2 + bonuses if ult active, else 1 (melee)."""
    if _fabled_lance_active(hero):
        return get_computed_stat(state, UnitID(hero.id), StatType.RANGE, 2)
    return 1


def _attack_is_ranged(hero: Hero) -> bool:
    """Return whether attacks are ranged (True when ult active)."""
    return _fabled_lance_active(hero)


# =============================================================================
# TIER I - RED: Token of Gratitude (ATTACK)
# =============================================================================


@register_effect("token_of_gratitude")
class TokenOfGratitudeEffect(CardEffect):
    """
    Target a unit adjacent to you. After the attack: A friendly hero in
    radius gains 1 coin.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=_attack_range(state, hero),
                is_ranged=_attack_is_ranged(hero),
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly hero in radius to gain 1 coin",
                output_key="coin_target",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            GainCoinsStep(hero_key="coin_target", amount=1),
        ]


# =============================================================================
# TIER II - RED: Fair Share (ATTACK)
# =============================================================================


@register_effect("fair_share")
class FairShareEffect(TokenOfGratitudeEffect):
    """
    Target a unit adjacent to you. After the attack: A friendly hero in
    radius gains 1 coin.
    """

    pass


# =============================================================================
# TIER III - RED: Paragon of Grace (ATTACK)
# =============================================================================


@register_effect("paragon_of_grace")
class ParagonOfGraceEffect(CardEffect):
    """
    Target a unit adjacent to you. After the attack: A friendly hero in
    radius gains 1 coin. May repeat once on a different target.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        coin_steps = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly hero in radius to gain 1 coin",
                output_key="coin_target",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            GainCoinsStep(hero_key="coin_target", amount=1),
        ]

        range_val = _attack_range(state, hero)
        is_ranged = _attack_is_ranged(hero)

        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=range_val,
                is_ranged=is_ranged,
                target_output_key="victim_id_1",
            ),
            *coin_steps,
            MayRepeatOnceStep(
                prompt="May repeat once on a different target?",
                steps_template=[
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=range_val,
                        is_ranged=is_ranged,
                        target_output_key="victim_id_2",
                        target_filters=[
                            ExcludeIdentityFilter(exclude_keys=["victim_id_1"]),
                        ],
                    ),
                    *coin_steps,
                ],
            ),
        ]


# =============================================================================
# TIER II - RED: Feat of Bravery (ATTACK)
# =============================================================================


@register_effect("feat_of_bravery")
class FeatOfBraveryEffect(CardEffect):
    """
    Target a unit adjacent to you. After the attack: A friendly hero in
    radius may retrieve a discarded card.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=_attack_range(state, hero),
                is_ranged=_attack_is_ranged(hero),
            ),
            # Select friendly hero in radius who has discarded cards
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly hero in radius to retrieve a discarded card",
                output_key="retrieve_ally",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, min_cards=1),
                ],
            ),
            # Ally picks which card to retrieve
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                context_hero_id_key="retrieve_ally",
                override_player_id_key="retrieve_ally",
                prompt="Select a discarded card to retrieve",
                output_key="ally_retrieved_card",
                is_mandatory=True,
                active_if_key="retrieve_ally",
            ),
            RetrieveCardStep(
                card_key="ally_retrieved_card",
                hero_key="retrieve_ally",
                active_if_key="retrieve_ally",
            ),
        ]


# =============================================================================
# TIER III - RED: Paragon of Valor (ATTACK)
# =============================================================================


@register_effect("paragon_of_valor")
class ParagonOfValorEffect(CardEffect):
    """
    Target a unit adjacent to you. After the attack: A friendly hero in
    radius may retrieve a discarded card; if they do, you may repeat
    once on a different target.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        range_val = _attack_range(state, hero)
        is_ranged = _attack_is_ranged(hero)

        ally_retrieve_steps = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="A friendly hero in radius may retrieve a discarded card",
                output_key="retrieve_ally",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, min_cards=1),
                ],
            ),
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                context_hero_id_key="retrieve_ally",
                override_player_id_key="retrieve_ally",
                prompt="Select a discarded card to retrieve",
                output_key="ally_retrieved_card",
                is_mandatory=True,
                active_if_key="retrieve_ally",
            ),
            RetrieveCardStep(
                card_key="ally_retrieved_card",
                hero_key="retrieve_ally",
                active_if_key="retrieve_ally",
            ),
        ]

        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=range_val,
                is_ranged=is_ranged,
                target_output_key="victim_id_1",
            ),
            *ally_retrieve_steps,
            # "if they do" — only offer repeat if ally was selected
            MayRepeatOnceStep(
                prompt="Repeat on a different target?",
                active_if_key="retrieve_ally",
                steps_template=[
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=range_val,
                        is_ranged=is_ranged,
                        target_output_key="victim_id_2",
                        target_filters=[
                            ExcludeIdentityFilter(exclude_keys=["victim_id_1"]),
                        ],
                    ),
                    *ally_retrieve_steps,
                ],
            ),
        ]


# =============================================================================
# TIER I - BLUE: Stand Guard (SKILL)
# =============================================================================


@register_effect("stand_guard")
class StandGuardEffect(CardEffect):
    """
    Swap with a friendly unit in range which is adjacent to an enemy
    hero, or who has a card in the discard.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly unit to swap with",
                output_key="swap_target",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.range),
                    OrFilter(
                        filters=[
                            AdjacencyFilter(target_tags=["ENEMY", "HERO"]),
                            CardsInContainerFilter(
                                container=CardContainerType.DISCARD, min_cards=1
                            ),
                        ]
                    ),
                ],
            ),
            SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
        ]


# =============================================================================
# TIER II - BLUE: Protect the Weak (SKILL)
# =============================================================================


@register_effect("protect_the_weak")
class ProtectTheWeakEffect(StandGuardEffect):
    """
    Swap with a friendly unit in range which is adjacent to an enemy
    hero, or who has a card in the discard.
    """

    pass


# =============================================================================
# TIER III - BLUE: Defend the Innocent (SKILL)
# =============================================================================


@register_effect("defend_the_innocent")
class DefendTheInnocentEffect(CardEffect):
    """
    Swap with a friendly unit in range which is adjacent to an enemy
    hero, or who has a card in the discard.
    You may retrieve a discarded card.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # Swap (same as Stand Guard)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly unit to swap with",
                output_key="swap_target",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.range),
                    OrFilter(
                        filters=[
                            AdjacencyFilter(target_tags=["ENEMY", "HERO"]),
                            CardsInContainerFilter(
                                container=CardContainerType.DISCARD, min_cards=1
                            ),
                        ]
                    ),
                ],
            ),
            SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target"),
            # Retrieve a discarded card (optional)
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="You may retrieve a discarded card",
                output_key="retrieved_card",
                is_mandatory=False,
            ),
            RetrieveCardStep(
                card_key="retrieved_card",
                active_if_key="retrieved_card",
            ),
        ]


# =============================================================================
# TIER II - BLUE: Accept Surrender (SKILL)
# =============================================================================


@register_effect("accept_surrender")
class AcceptSurrenderEffect(CardEffect):
    """
    Defeat an enemy hero adjacent to you with no cards in hand.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero with no cards in hand to defeat",
                output_key="defeat_target",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                    CardsInContainerFilter(container=CardContainerType.HAND, max_cards=0),
                ],
            ),
            DefeatUnitStep(victim_key="defeat_target", killer_id=hero.id),
        ]


# =============================================================================
# TIER III - BLUE: Glorious Triumph (SKILL)
# =============================================================================


@register_effect("glorious_triumph")
class GloriousTriumphEffect(CardEffect):
    """
    Defeat an enemy hero adjacent to you with no cards in hand;
    your friendly heroes gain triple assist coins.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero with no cards in hand to defeat",
                output_key="defeat_target",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                    CardsInContainerFilter(container=CardContainerType.HAND, max_cards=0),
                ],
            ),
            DefeatUnitStep(
                victim_key="defeat_target",
                killer_id=hero.id,
                assist_multiplier=3,
            ),
        ]


# =============================================================================
# TIER II - GREEN: Opening Shots (SKILL)
# =============================================================================


@register_effect("opening_shots")
class OpeningShotsEffect(CardEffect):
    """
    If both you and an enemy hero in radius have no cards in the
    discard, that hero discards a card, if able.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # Check actor's discard is empty
            CountCardsStep(
                card_container=CardContainerType.DISCARD,
                output_key="actor_discard_count",
            ),
            CheckContextConditionStep(
                input_key="actor_discard_count",
                operator="==",
                threshold=0,
                output_key="actor_discard_empty",
            ),
            # Select enemy hero in radius who also has empty discard
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero with no discarded cards",
                output_key="discard_victim",
                is_mandatory=True,
                active_if_key="actor_discard_empty",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.radius),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, max_cards=0),
                ],
            ),
            ForceDiscardStep(
                victim_key="discard_victim",
                active_if_key="discard_victim",
            ),
        ]


# =============================================================================
# TIER III - GREEN: Opening Volley (SKILL)
# =============================================================================


@register_effect("opening_volley")
class OpeningVolleyEffect(CardEffect):
    """
    If both you and an enemy hero in radius have no cards in the
    discard, that hero discards a card, if able. May repeat once.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        opening_shot_steps = [
            CountCardsStep(
                card_container=CardContainerType.DISCARD,
                output_key="actor_discard_count",
            ),
            CheckContextConditionStep(
                input_key="actor_discard_count",
                operator="==",
                threshold=0,
                output_key="actor_discard_empty",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero with no discarded cards",
                output_key="discard_victim",
                is_mandatory=True,
                active_if_key="actor_discard_empty",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.radius),
                    CardsInContainerFilter(container=CardContainerType.DISCARD, max_cards=0),
                ],
            ),
            ForceDiscardStep(
                victim_key="discard_victim",
                active_if_key="discard_victim",
            ),
        ]

        return [
            *opening_shot_steps,
            MayRepeatOnceStep(
                prompt="May repeat once?",
                steps_template=opening_shot_steps,
            ),
        ]


# =============================================================================
# TIER I - GREEN: Close Quarters (MOVEMENT)
# =============================================================================


@register_effect("close_quarters")
class CloseQuartersEffect(CardEffect):
    """
    After movement, if you are adjacent to an enemy hero, you may
    Choose one —
    - Place a friendly minion in radius into a space adjacent to that
      enemy hero.
    - Place an enemy minion in radius into a space adjacent to you.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
            # Check if adjacent to an enemy hero
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
                output_key="adj_enemy_hero_count",
            ),
            CheckContextConditionStep(
                input_key="adj_enemy_hero_count",
                operator=">=",
                threshold=1,
                output_key="has_adj_enemy_hero",
            ),
            # Choose one (optional — "you may")
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="melee_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Place a friendly minion adjacent to enemy hero",
                    2: "Place an enemy minion adjacent to you",
                },
                is_mandatory=False,
                active_if_key="has_adj_enemy_hero",
            ),
            # === Branch 1: Place friendly minion adjacent to enemy hero ===
            CheckContextConditionStep(
                input_key="melee_choice",
                operator="==",
                threshold=1,
                output_key="chose_place_friendly",
            ),
            # First pick which adjacent enemy hero
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select the adjacent enemy hero",
                output_key="target_enemy_hero",
                is_mandatory=True,
                active_if_key="chose_place_friendly",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
            ),
            # Pick friendly minion in radius
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly minion in radius to place",
                output_key="place_minion_1",
                is_mandatory=True,
                active_if_key="chose_place_friendly",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            # Pick hex adjacent to that enemy hero
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space adjacent to the enemy hero",
                output_key="place_dest_1",
                is_mandatory=True,
                active_if_key="chose_place_friendly",
                filters=[
                    RangeFilter(max_range=1, origin_key="target_enemy_hero"),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            PlaceUnitStep(
                unit_key="place_minion_1",
                destination_key="place_dest_1",
                active_if_key="chose_place_friendly",
            ),
            # === Branch 2: Place enemy minion adjacent to you ===
            CheckContextConditionStep(
                input_key="melee_choice",
                operator="==",
                threshold=2,
                output_key="chose_place_enemy",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy minion in radius to place",
                output_key="place_minion_2",
                is_mandatory=True,
                active_if_key="chose_place_enemy",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space adjacent to you",
                output_key="place_dest_2",
                is_mandatory=True,
                active_if_key="chose_place_enemy",
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            PlaceUnitStep(
                unit_key="place_minion_2",
                destination_key="place_dest_2",
                active_if_key="chose_place_enemy",
            ),
        ]


# =============================================================================
# TIER II - GREEN: Melee (MOVEMENT)
# =============================================================================


@register_effect("melee")
class MeleeEffect(CloseQuartersEffect):
    """
    After movement, if you are adjacent to an enemy hero, you may
    Choose one —
    - Place a friendly minion in radius into a space adjacent to that
      enemy hero.
    - Place an enemy minion in radius into a space adjacent to you.
    """

    pass


# =============================================================================
# TIER III - GREEN: Grand Melee (MOVEMENT)
# =============================================================================


@register_effect("grand_melee")
class GrandMeleeEffect(CardEffect):
    """
    After movement, if you are adjacent to an enemy hero, Choose up to
    two times —
    - Place a friendly minion in radius into a space adjacent to that
      enemy hero.
    - Place an enemy minion in radius into a space adjacent to you.
    """

    def _build_choose_and_place(self, hero: Hero, stats: CardStats, suffix: str) -> list[GameStep]:
        """Build one round of the choose-and-place sequence."""
        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key=f"melee_choice_{suffix}",
                number_options=[1, 2],
                number_labels={
                    1: "Place a friendly minion adjacent to enemy hero",
                    2: "Place an enemy minion adjacent to you",
                },
                is_mandatory=False,
                active_if_key="has_adj_enemy_hero",
            ),
            # Branch 1
            CheckContextConditionStep(
                input_key=f"melee_choice_{suffix}",
                operator="==",
                threshold=1,
                output_key=f"chose_friendly_{suffix}",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select the adjacent enemy hero",
                output_key=f"target_hero_{suffix}",
                is_mandatory=True,
                active_if_key=f"chose_friendly_{suffix}",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly minion in radius",
                output_key=f"place_min_{suffix}",
                is_mandatory=True,
                active_if_key=f"chose_friendly_{suffix}",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space adjacent to the enemy hero",
                output_key=f"place_dest_{suffix}",
                is_mandatory=True,
                active_if_key=f"chose_friendly_{suffix}",
                filters=[
                    RangeFilter(max_range=1, origin_key=f"target_hero_{suffix}"),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            PlaceUnitStep(
                unit_key=f"place_min_{suffix}",
                destination_key=f"place_dest_{suffix}",
                active_if_key=f"chose_friendly_{suffix}",
            ),
            # Branch 2
            CheckContextConditionStep(
                input_key=f"melee_choice_{suffix}",
                operator="==",
                threshold=2,
                output_key=f"chose_enemy_{suffix}",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy minion in radius",
                output_key=f"place_emin_{suffix}",
                is_mandatory=True,
                active_if_key=f"chose_enemy_{suffix}",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space adjacent to you",
                output_key=f"place_edest_{suffix}",
                is_mandatory=True,
                active_if_key=f"chose_enemy_{suffix}",
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            PlaceUnitStep(
                unit_key=f"place_emin_{suffix}",
                destination_key=f"place_edest_{suffix}",
                active_if_key=f"chose_enemy_{suffix}",
            ),
        ]

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    UnitTypeFilter(unit_type="HERO"),
                ],
                output_key="adj_enemy_hero_count",
            ),
            CheckContextConditionStep(
                input_key="adj_enemy_hero_count",
                operator=">=",
                threshold=1,
                output_key="has_adj_enemy_hero",
            ),
            # "Choose up to two times" — both optional, no MayRepeatOnceStep
            *self._build_choose_and_place(hero, stats, "a"),
            *self._build_choose_and_place(hero, stats, "b"),
        ]


# =============================================================================
# UNTIERED - GOLD: Code of Chivalry (ATTACK)
# =============================================================================


@register_effect("code_of_chivalry")
class CodeOfChivalryEffect(CardEffect):
    """
    Target a unit adjacent to you. Before the attack: If you target a
    hero, both you and the target may retrieve a discarded card.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        range_val = _attack_range(state, hero)
        is_ranged = _attack_is_ranged(hero)

        return [
            # Select target first (before attack, need to check type)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Target a unit adjacent to you",
                output_key="victim_id",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=range_val),
                    TeamFilter(relation="ENEMY"),
                ],
            ),
            # Check if target is a hero
            CheckUnitTypeStep(
                unit_key="victim_id",
                expected_type="HERO",
                output_key="target_is_hero",
            ),
            # Before the attack: if hero, both may retrieve
            # Target retrieves first
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                context_hero_id_key="victim_id",
                override_player_id_key="victim_id",
                prompt="You may retrieve a discarded card",
                output_key="target_retrieved_card",
                is_mandatory=False,
                active_if_key="target_is_hero",
            ),
            RetrieveCardStep(
                card_key="target_retrieved_card",
                hero_key="victim_id",
                active_if_key="target_retrieved_card",
            ),
            # Actor retrieves
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                prompt="You may retrieve a discarded card",
                output_key="actor_retrieved_card",
                is_mandatory=False,
                active_if_key="target_is_hero",
            ),
            RetrieveCardStep(
                card_key="actor_retrieved_card",
                active_if_key="actor_retrieved_card",
            ),
            # Now attack the pre-selected target
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=range_val,
                is_ranged=is_ranged,
                target_id_key="victim_id",
            ),
        ]


# =============================================================================
# UNTIERED - SILVER: Throw the Gauntlet (SKILL)
# =============================================================================


@register_effect("throw_the_gauntlet")
class ThrowTheGauntletEffect(CardEffect):
    """
    Place yourself into a space in range adjacent to an enemy hero in
    range; that hero may move 1 space; if they do, gain 2 coins.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # Select the enemy hero
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range",
                output_key="target_hero",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range),
                ],
            ),
            # Place self into hex in range, adjacent to that hero
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space in range adjacent to the enemy hero",
                output_key="gauntlet_dest",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=stats.range),
                    ObstacleFilter(is_obstacle=False),
                    AdjacencyToContextFilter(target_key="target_hero"),
                ],
            ),
            PlaceUnitStep(unit_id=hero.id, destination_key="gauntlet_dest"),
            # Enemy hero may move 1 space (their choice)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space (Rowenna gains 2 coins if you do)",
                output_key="enemy_move_dest",
                is_mandatory=False,
                override_player_id_key="target_hero",
                filters=[
                    RangeFilter(max_range=1, origin_key="target_hero"),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_key="target_hero"),
                ],
            ),
            # Actually move the enemy hero if they chose a hex
            MoveUnitStep(
                unit_key="target_hero",
                destination_key="enemy_move_dest",
                range_val=1,
                is_movement_action=False,
                active_if_key="enemy_move_dest",
            ),
            # If they moved, Rowenna gains 2 coins
            SetContextFlagStep(key="rowenna_id", value=hero.id),
            GainCoinsStep(
                hero_key="rowenna_id",
                amount=2,
                active_if_key="enemy_move_dest",
            ),
        ]


# =============================================================================
# TIER IV - PURPLE: Fabled Lance (PASSIVE/ULTIMATE)
# =============================================================================
# Card Text: 'All your attack actions gain the "Ranged" subtype, target
#   a unit in range, and count as having a printed Range value of 2.'
#
# HLD:
#   This is a persistent passive that modifies ALL attack actions.
#   It needs to:
#   - Make every attack action "ranged" (is_ranged=True)
#   - Override target selection to use range instead of adjacency
#   - Set printed range to 2 for all attacks
#
#   This is the most novel mechanic. Approach options:
#   a) PassiveConfig with BEFORE_ATTACK trigger that injects modifiers
#   b) A stat modifier that adds range=2 and ranged flag
#   c) Build-time check in other attack effects (too coupled)
#
#   Best approach: Use PassiveConfig(BEFORE_ATTACK) to inject a
#   modifier that sets is_ranged=True and range=2 on the attack.
#   The AttackSequenceStep would need to respect these overrides.
#
#   TODO: Determine if AttackSequenceStep can dynamically become
#   ranged via context flags or if we need engine-level support.
#   May need a context flag like "force_ranged" and "force_range_value"
#   that AttackSequenceStep checks.
# =============================================================================


@register_effect("fabled_lance")
class FabledLanceEffect(CardEffect):
    """
    All your attack actions gain the "Ranged" subtype, target a unit
    in range, and count as having a printed Range value of 2.

    Implementation: This passive has no trigger steps. Instead, each
    Rowenna attack effect checks _fabled_lance_active() at build time
    and adjusts range_val/is_ranged on AttackSequenceStep accordingly.
    Range items stack on top of the base value of 2.
    """

    pass
