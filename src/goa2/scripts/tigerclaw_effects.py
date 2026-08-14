from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goa2.domain.models import ActionType, TargetType
from goa2.domain.models.effect import (
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.enums import CardContainerType, PassiveTrigger
from goa2.domain.models.marker import MarkerType
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters_geometry import SpaceBehindEmptyFilter
from goa2.engine.filters_hex import (
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
    TerrainFilter,
)
from goa2.engine.filters_units import (
    AdjacencyToContextFilter,
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckLanePushStep,
    ComputeHexStep,
    CountStep,
    CreateEffectStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    GameStep,
    MoveSequenceStep,
    MoveUnitStep,
    PlaceMarkerStep,
    PlaceUnitStep,
    RetrieveCardStep,
    SelectStep,
    SetContextFlagStep,
    StealCoinsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# TIER I - BLUE: Dodge
# =============================================================================


@register_effect("dodge")
class DodgeEffect(CardEffect):
    """
    Card text: "Block a ranged attack."
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        if context.get("attack_is_ranged"):
            return [SetContextFlagStep(key="auto_block", value=True)]
        else:
            return [SetContextFlagStep(key="defense_invalid", value=True)]


# =============================================================================
# TIER I - RED: Hit and Run
# =============================================================================


@register_effect("hit_and_run")
class HitAndRunEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. After the attack:
    You may move 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            # Post-attack: you may move 1 space
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="post_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# TIER II - BLUE: Sidestep
# =============================================================================


@register_effect("sidestep")
class SidestepEffect(CardEffect):
    """
    Card text: "Block a ranged attack. You may move 1 space."
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        if context.get("attack_is_ranged"):
            return [SetContextFlagStep(key="auto_block", value=True)]
        else:
            return [SetContextFlagStep(key="defense_invalid", value=True)]

    def build_on_block_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="sidestep_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=defender.id),
                ],
            ),
            MoveUnitStep(
                unit_id=defender.id,
                destination_key="sidestep_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="sidestep_hex",
            ),
        ]


# =============================================================================
# TIER II - BLUE: Parry
# =============================================================================


@register_effect("parry")
class ParryEffect(CardEffect):
    """
    Card text: "Block a non-ranged attack. The attacker discards a card, if able."
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        if not context.get("attack_is_ranged"):
            return [SetContextFlagStep(key="auto_block", value=True)]
        else:
            return [SetContextFlagStep(key="defense_invalid", value=True)]

    def build_on_block_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        return [
            ForceDiscardStep(
                victim_key="attacker_id",
                immunity_source_id=str(defender.id),
            ),
        ]


# =============================================================================
# TIER III - BLUE: Riposte
# =============================================================================


@register_effect("riposte")
class RiposteEffect(CardEffect):
    """
    Card text: "Block a non-ranged attack. The attacker discards a card, or is defeated."
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        if not context.get("attack_is_ranged"):
            return [SetContextFlagStep(key="auto_block", value=True)]
        else:
            return [SetContextFlagStep(key="defense_invalid", value=True)]

    def build_on_block_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        return [
            ForceDiscardOrDefeatStep(
                victim_key="attacker_id",
                immunity_source_id=str(defender.id),
            ),
        ]


# =============================================================================
# TIER III - RED: Leaping Strike
# =============================================================================


@register_effect("leaping_strike")
class LeapingStrikeEffect(CardEffect):
    """
    Card text: "Before the attack: You may move 1 space.
    Target a unit adjacent to you. After the attack: You may move 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # Pre-attack: you may move 1 space
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Before the attack: You may move 1 space",
                output_key="pre_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="pre_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="pre_move_hex",
            ),
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            # Post-attack: you may move 1 space
            SelectStep(
                target_type=TargetType.HEX,
                prompt="After the attack: You may move 1 space",
                output_key="post_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# TIER II - RED: Backstab
# =============================================================================


@register_effect("backstab")
class BackstabEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you; if a friendly unit is adjacent
    to the target, +2 Attack. (A 'friendly unit' is another hero or a minion
    on your team.)"
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select adjacent enemy target
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent unit to attack",
                output_key="victim_id",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                ],
            ),
            # 2. Count friendly units adjacent to the target (excluding self)
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    AdjacencyToContextFilter(target_key="victim_id"),
                    TeamFilter(relation="FRIENDLY"),
                    ExcludeIdentityFilter(exclude_self=True),
                ],
                output_key="friendly_adjacent_count",
                skip_immunity_filter=True,
            ),
            # 3. Check if any friendly is adjacent
            CheckContextConditionStep(
                input_key="friendly_adjacent_count",
                operator=">=",
                threshold=1,
                output_key="has_flanking",
            ),
            # 4. Set +2 bonus if flanking
            SetContextFlagStep(key="atk_bonus", value=2, active_if_key="has_flanking"),
            # 5. Attack with pre-selected target and conditional bonus
            AttackSequenceStep(
                damage=stats.primary_value,
                target_id_key="victim_id",
                range_val=1,
                damage_bonus_key="atk_bonus",
            ),
        ]


# =============================================================================
# TIER III - RED: Backstab with a Ballista
# =============================================================================


@register_effect("backstab_with_a_ballista")
class BackstabWithABallistaEffect(CardEffect):
    """
    Card text: "Target a unit in range; if a friendly unit is adjacent to the
    target +2 Attack, and the target cannot perform a primary action to defend."

    The flanking bonus reuses the same pattern as Backstab.
    When flanking applies, block_primary_defense prevents the defender from
    using primary DEFENSE cards and suppresses defense effect text.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select target in range
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select unit to attack in range",
                output_key="victim_id",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=stats.range),
                    TeamFilter(relation="ENEMY"),
                ],
            ),
            # 2. Count friendly units adjacent to the target (excluding self)
            CountStep(
                target_type=TargetType.UNIT,
                filters=[
                    AdjacencyToContextFilter(target_key="victim_id"),
                    TeamFilter(relation="FRIENDLY"),
                    ExcludeIdentityFilter(exclude_self=True),
                ],
                output_key="friendly_adjacent_count",
                skip_immunity_filter=True,
            ),
            # 3. Check if any friendly is adjacent
            CheckContextConditionStep(
                input_key="friendly_adjacent_count",
                operator=">=",
                threshold=1,
                output_key="has_flanking",
            ),
            # 4. Set +2 bonus if flanking
            SetContextFlagStep(key="atk_bonus", value=2, active_if_key="has_flanking"),
            # 5. Set "no primary defense" flag if flanking
            SetContextFlagStep(
                key="block_primary_defense",
                value=True,
                active_if_key="has_flanking",
            ),
            # 6. Attack with pre-selected target and conditional bonus
            AttackSequenceStep(
                damage=stats.primary_value,
                target_id_key="victim_id",
                range_val=stats.range,
                is_ranged=True,
                damage_bonus_key="atk_bonus",
            ),
        ]


# =============================================================================
# TIER II - RED: Combat Reflexes
# =============================================================================


@register_effect("combat_reflexes")
class CombatReflexesEffect(CardEffect):
    """
    Card text: "Before the attack: You may move 1 space. Target a unit
    adjacent to you. After the attack: If you did not move before the
    attack, you may move 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Optional pre-move (stores destination in context if taken)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Before the attack: You may move 1 space",
                output_key="pre_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="pre_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="pre_move_hex",
            ),
            # 2. Attack adjacent target
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            # 3. Post-move only if pre-move was NOT taken
            SelectStep(
                target_type=TargetType.HEX,
                prompt="After the attack: You may move 1 space",
                output_key="post_move_hex",
                is_mandatory=False,
                skip_if_key="pre_move_hex",
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# TIER III - BLUE: Evade
# =============================================================================


@register_effect("evade")
class EvadeEffect(CardEffect):
    """
    Card text: "Block a ranged attack. You may move 1 space.
    You may retrieve your resolved or discarded basic skill card."
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        if context.get("attack_is_ranged"):
            return [SetContextFlagStep(key="auto_block", value=True)]
        else:
            return [SetContextFlagStep(key="defense_invalid", value=True)]

    def build_on_block_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        return [
            # 1. Optional move 1 space
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="evade_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=defender.id),
                ],
            ),
            MoveUnitStep(
                unit_id=defender.id,
                destination_key="evade_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="evade_move_hex",
            ),
            # 2. Retrieve a basic skill card from discard (optional)
            SelectStep(
                target_type=TargetType.CARD,
                card_containers=[CardContainerType.PLAYED, CardContainerType.DISCARD],
                prompt="Select a basic skill card to retrieve (optional)",
                output_key="retrieved_card",
                is_mandatory=False,
                card_is_basic=True,
                card_action_types=[ActionType.SKILL],
            ),
            RetrieveCardStep(
                card_key="retrieved_card",
                active_if_key="retrieved_card",
            ),
        ]


# =============================================================================
# TIER I - GREEN: Light-Fingered
# =============================================================================


@register_effect("light_fingered")
class LightFingeredEffect(CardEffect):
    """
    Card text: "You may move 1 space. Take 1 coin from an enemy hero
    adjacent to you; if you do, you may move 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Optional move 1 space
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="pre_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="pre_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="pre_move_hex",
            ),
            # 2. Select adjacent enemy hero to steal from
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent enemy hero to take a coin from",
                output_key="steal_victim",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            # 3. Steal 1 coin
            StealCoinsStep(
                victim_key="steal_victim",
                amount=1,
                output_key="stole_coins",
                active_if_key="steal_victim",
            ),
            # 4. Post-move only if coins were stolen
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="post_move_hex",
                is_mandatory=False,
                active_if_key="stole_coins",
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# TIER II - GREEN: Pick Pocket
# =============================================================================


@register_effect("pick_pocket")
class PickPocketEffect(CardEffect):
    """
    Card text: "Move up to 2 spaces. Take 1 coin from an enemy hero
    adjacent to you; if you do, you may move 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Move up to 2 spaces
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Move up to 2 spaces",
                output_key="pre_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=2),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=2, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="pre_move_hex",
                range_val=2,
                is_movement_action=False,
                active_if_key="pre_move_hex",
            ),
            # 2. Select adjacent enemy hero to steal from
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent enemy hero to take a coin from",
                output_key="steal_victim",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            # 3. Steal 1 coin
            StealCoinsStep(
                victim_key="steal_victim",
                amount=1,
                output_key="stole_coins",
                active_if_key="steal_victim",
            ),
            # 4. Post-move only if coins were stolen
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="post_move_hex",
                is_mandatory=False,
                active_if_key="stole_coins",
                filters=[
                    RangeFilter(max_range=1),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=1, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=1,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# TIER III - GREEN: Master Thief
# =============================================================================


@register_effect("master_thief")
class MasterThiefEffect(CardEffect):
    """
    Card text: "Move up to 2 spaces. Take 1 or 2 coins from an enemy hero
    adjacent to you; if you do, you may move up to 2 spaces."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Move up to 2 spaces
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Move up to 2 spaces",
                output_key="pre_move_hex",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=2),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=2, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="pre_move_hex",
                range_val=2,
                is_movement_action=False,
                active_if_key="pre_move_hex",
            ),
            # 2. Select adjacent enemy hero to steal from
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent enemy hero to take coins from",
                output_key="steal_victim",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            # 3. Choose amount: 1 or 2
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose how many coins to take",
                output_key="steal_amount",
                number_options=[1, 2],
                number_labels={1: "1 Coin", 2: "2 Coins"},
                is_mandatory=True,
                active_if_key="steal_victim",
            ),
            # 4. Steal coins
            StealCoinsStep(
                victim_key="steal_victim",
                amount_key="steal_amount",
                output_key="stole_coins",
                active_if_key="steal_victim",
            ),
            # 5. Post-move up to 2 only if coins were stolen
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move up to 2 spaces",
                output_key="post_move_hex",
                is_mandatory=False,
                active_if_key="stole_coins",
                filters=[
                    RangeFilter(max_range=2),
                    ObstacleFilter(is_obstacle=False),
                    MovementPathFilter(range_val=2, unit_id=hero.id),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="post_move_hex",
                range_val=2,
                is_movement_action=False,
                active_if_key="post_move_hex",
            ),
        ]


# =============================================================================
# UNTIERED - SILVER: Blend Into Shadows
# =============================================================================


@register_effect("blend_into_shadows")
class BlendIntoShadowsEffect(CardEffect):
    """
    Card text: "If you are adjacent to a terrain hex, you may be placed on
    an empty space within 2 spaces. If you do, you are immune to attacks
    next turn."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Count terrain hexes adjacent to hero
            CountStep(
                target_type=TargetType.HEX,
                filters=[
                    RangeFilter(max_range=1),
                    TerrainFilter(is_terrain=True),
                ],
                output_key="terrain_count",
            ),
            # 2. Check if adjacent to terrain
            CheckContextConditionStep(
                input_key="terrain_count",
                operator=">=",
                threshold=1,
                output_key="can_blend",
            ),
            # 3. Select destination hex (only if adjacent to terrain)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select an empty space within 2 spaces to blend into",
                output_key="blend_hex",
                is_mandatory=True,
                active_if_key="can_blend",
                filters=[
                    RangeFilter(max_range=stats.radius or 2),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            # 4. Place hero at destination
            PlaceUnitStep(
                unit_id=hero.id,
                destination_key="blend_hex",
                active_if_key="blend_hex",
            ),
            # 5. Grant attack immunity next turn
            CreateEffectStep(
                effect_type=EffectType.ATTACK_IMMUNITY,
                scope=EffectScope(
                    shape=Shape.POINT,
                    origin_id=hero.id,
                    affects=AffectsFilter.SELF,
                ),
                duration=DurationType.NEXT_TURN,
                active_if_key="blend_hex",
            ),
        ]


# =============================================================================
# UNTIERED - GOLD: Blink Strike
# =============================================================================


@register_effect("blink_strike")
class BlinkStrikeEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you in a straight line;
    move to the space directly behind it, then attack it."

    Player selects the enemy unit to blink through (must have empty hex behind).
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select adjacent enemy with empty space behind
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select enemy to blink through",
                output_key="blink_victim",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    SpaceBehindEmptyFilter(origin_id=hero.id),
                ],
            ),
            # 2. Compute the hex behind the enemy
            ComputeHexStep(
                target_key="blink_victim",
                scale=1,
                output_key="blink_dest",
            ),
            # 3. Place hero behind enemy (teleport through)
            PlaceUnitStep(
                unit_id=hero.id,
                destination_key="blink_dest",
            ),
            # 4. Attack the enemy we passed through
            AttackSequenceStep(
                damage=stats.primary_value,
                target_id_key="blink_victim",
                range_val=1,
            ),
        ]


# =============================================================================
# TIER II - GREEN: Poisoned Dagger
# =============================================================================


@register_effect("poisoned_dagger")
class PoisonedDaggerEffect(CardEffect):
    """
    Card text: "Give a hero in range a Poison marker. The hero with a poison
    marker has -1 Initiative, -1 Attack and -1 Defense."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a hero in range to poison",
                output_key="poison_target",
                is_mandatory=True,
                # Poison may target any hero (allies included) — unlike attack/
                # defeat effects, friendly heroes are legal targets per the
                # rules. Self is excluded by SelectStep's default self-filter.
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    RangeFilter(max_range=stats.range),
                ],
            ),
            PlaceMarkerStep(
                marker_type=MarkerType.POISON,
                target_key="poison_target",
                value=-1,
            ),
        ]


# =============================================================================
# TIER III - GREEN: Poisoned Dart
# =============================================================================


@register_effect("poisoned_dart")
class PoisonedDartEffect(CardEffect):
    """
    Card text: "Give a hero in range a Poison marker. The hero with a poison
    marker has -2 Initiative, -2 Attack and -2 Defense."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a hero in range to poison",
                output_key="poison_target",
                is_mandatory=True,
                # Poison may target any hero (allies included) — unlike attack/
                # defeat effects, friendly heroes are legal targets per the
                # rules. Self is excluded by SelectStep's default self-filter.
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    RangeFilter(max_range=stats.range),
                ],
            ),
            PlaceMarkerStep(
                marker_type=MarkerType.POISON,
                target_key="poison_target",
                value=-2,
            ),
        ]


# =============================================================================
# ULTIMATE (Purple/Tier IV) - Passive: Cloak and Daggers
# =============================================================================


@register_effect("cloak_and_daggers")
class CloakAndDaggersEffect(CardEffect):
    """
    Card text: "After you perform a basic action, you may repeat it once;
    if you repeat an attack action, you cannot target the same unit."

    Passive ability that triggers AFTER_BASIC_ACTION (any action on basic cards).
    Reads the action type, value, and range from context to rebuild the repeat.
    """

    def get_passive_config(self) -> PassiveConfig | None:
        return PassiveConfig(
            trigger=PassiveTrigger.AFTER_BASIC_ACTION,
            uses_per_turn=1,
            is_optional=True,
            prompt="Cloak and Daggers: Repeat the basic action?",
        )

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> list[GameStep]:
        from goa2.engine.effects import CardEffectRegistry

        if trigger != PassiveTrigger.AFTER_BASIC_ACTION:
            return []

        # Check repeat prevention effects (e.g. Enfeeblement)
        result = state.validator.can_repeat_action(state, str(hero.id), context)
        if not result.allowed:
            return []

        action_type = context.get("basic_action_type")
        action_value = context.get("basic_action_value")

        if not action_type or action_value is None:
            return []

        if action_type == ActionType.ATTACK.value:
            # If a primary effect was used, rebuild the full sequence
            effect_id = context.get("basic_action_effect_id")
            if effect_id:
                effect = CardEffectRegistry.get(effect_id)
                basic_card = hero.current_turn_card
                if effect and basic_card:
                    steps = effect.get_steps(state, hero, basic_card)
                    # Inject "cannot target same unit" filter
                    for step in steps:
                        if isinstance(step, AttackSequenceStep) and (
                            not step.target_id_key or step.target_output_key
                        ):
                            step.target_filters.append(
                                ExcludeIdentityFilter(exclude_keys=["last_combat_target"])
                            )
                        elif isinstance(step, SelectStep) and step.target_type == TargetType.UNIT:
                            step.filters.append(
                                ExcludeIdentityFilter(exclude_keys=["last_combat_target"])
                            )
                    return [CheckLanePushStep(), *steps]

            # Fallback: secondary/generic attack
            action_range = context.get("basic_action_range", 1)
            return [
                CheckLanePushStep(),
                AttackSequenceStep(
                    damage=action_value,
                    range_val=action_range,
                    target_filters=[ExcludeIdentityFilter(exclude_keys=["last_combat_target"])],
                ),
            ]
        elif action_type == ActionType.MOVEMENT.value:
            return [
                CheckLanePushStep(),
                MoveSequenceStep(
                    unit_id=hero.id,
                    range_val=action_value,
                ),
            ]
        elif action_type == ActionType.SKILL.value:
            effect_id = context.get("basic_action_effect_id")
            if effect_id:
                effect = CardEffectRegistry.get(effect_id)
                basic_card = hero.current_turn_card
                if effect and basic_card:
                    return [CheckLanePushStep(), *effect.get_steps(state, hero, basic_card)]
            return []

        return []
