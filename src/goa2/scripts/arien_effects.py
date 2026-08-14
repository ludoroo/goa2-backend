from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goa2.domain.models import (
    ActionType,
    AffectsFilter,
    CardColor,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
    TargetType,
)
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_geometry import LineBehindTargetFilter
from goa2.engine.filters_hex import (
    AdjacentSpawnPointFilter,
    HasEmptyNeighborFilter,
    ObstacleFilter,
    RangeFilter,
    SpawnPointFilter,
)
from goa2.engine.filters_units import (
    AdjacencyToContextFilter,
    CanBeMovedByActorFilter,
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckAdjacencyStep,
    CreateEffectStep,
    ForceDiscardOrDefeatStep,
    GameStep,
    MayRepeatOnceStep,
    MoveSequenceStep,
    MoveUnitStep,
    PlaceUnitStep,
    PushUnitStep,
    SelectStep,
    SetContextFlagStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero, TargetType
    from goa2.domain.models.enums import PassiveTrigger
    from goa2.domain.state import GameState
    from goa2.engine.effects import PassiveConfig
    from goa2.engine.stats import CardStats


@register_effect("spell_break")
class SpellBreakEffect(CardEffect):
    """
    Card text: "This turn: Enemy heroes in radius cannot perform skill actions,
    except on gold cards."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            CreateEffectStep(
                effect_type=EffectType.TARGET_PREVENTION,
                scope=EffectScope(
                    shape=Shape.RADIUS,
                    range=stats.radius or 0,
                    origin_id=hero.id,
                    affects=AffectsFilter.ENEMY_HEROES,
                ),
                duration=DurationType.THIS_TURN,
                restrictions=[ActionType.SKILL],
                except_card_colors=[CardColor.GOLD],
            ),
        ]


@register_effect("noble_blade")
class NobleBladeEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. Before the attack: You may move another unit
    that is adjacent to the target 1 space."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select Attack Target (Mandatory)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select target for Noble Blade attack",
                output_key="victim_id",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                ],
                is_mandatory=True,
            ),
            # 2. Select Unit to Nudge (Optional)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent unit to move 1 space (Optional)",
                output_key="nudge_unit_id",
                is_mandatory=False,
                filters=[
                    AdjacencyToContextFilter(target_key="victim_id"),
                    ExcludeIdentityFilter(exclude_self=True, exclude_keys=["victim_id"]),
                    HasEmptyNeighborFilter(),  # Must have somewhere to go
                    CanBeMovedByActorFilter(),  # Cannot move if protected
                ],
            ),
            # 3. Select Destination (Active If Nudge Selected)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination for move",
                output_key="nudge_dest",
                active_if_key="nudge_unit_id",
                filters=[
                    RangeFilter(max_range=1, origin_key="nudge_unit_id"),
                    ObstacleFilter(is_obstacle=False),
                ],
                is_mandatory=True,
            ),
            # 4. Execute Move (Active If Nudge Selected)
            MoveUnitStep(
                unit_key="nudge_unit_id",
                destination_key="nudge_dest",
                range_val=1,
                active_if_key="nudge_unit_id",
            ),
            # 5. Resolve Attack Sequence (Using pre-selected target)
            # Note: range_val=1 is hardcoded as Noble Blade is adjacent-only (not buffable)
            AttackSequenceStep(damage=stats.primary_value, target_id_key="victim_id", range_val=1),
        ]


@register_effect("arcane_whirlpool")
class SwapEnemyMinionEffect(CardEffect):
    """
    Card text: "Swap with an enemy minion in range."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range),
                ],
                prompt="Select an enemy minion to swap with.",
                output_key="swap_target_id",
            ),
            SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target_id"),
        ]


@register_effect("ebb_and_flow")
class EbbAndFlowEffect(CardEffect):
    """
    Card text: "Swap with an enemy minion in range; if it was adjacent to you, may repeat once."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select First Target
            SelectStep(
                target_type=TargetType.UNIT,
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range),
                ],
                prompt="Select an enemy minion to swap with.",
                output_key="swap_target_1",
                is_mandatory=True,
            ),
            # 2. Check Adjacency (Before Swap)
            CheckAdjacencyStep(
                unit_a_id=hero.id,
                unit_b_key="swap_target_1",
                output_key="can_repeat",
            ),
            # 2. Swap 1
            SwapUnitsStep(unit_a_id=hero.id, unit_b_key="swap_target_1"),
            # 3. May Repeat
            MayRepeatOnceStep(
                active_if_key="can_repeat",
                steps_template=[
                    SelectStep(
                        target_type=TargetType.UNIT,
                        filters=[
                            UnitTypeFilter(unit_type="MINION"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=stats.range),  # Range from NEW position
                            ExcludeIdentityFilter(exclude_self=True),
                        ],
                        prompt="Select an enemy minion to swap with (Repeat)",
                        output_key="swap_target_2",
                        is_mandatory=False,
                    ),
                    SwapUnitsStep(
                        unit_a_id=hero.id,
                        unit_b_key="swap_target_2",
                        active_if_key="swap_target_2",  # Only if selected
                    ),
                ],
            ),
        ]


@register_effect("dangerous_current")
class DangerousCurrentEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. Before the attack: Up to 1 enemy hero
    in any of the 2 spaces in a straight line directly behind the target
    discards a card, or is defeated."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select Attack Target (Mandatory)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select target for Dangerous Current attack",
                output_key="victim_id",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                ],
                is_mandatory=True,
            ),
            # 2. Select "Backstab" Victim (Optional - Up to 1)
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select enemy hero behind target to discard/defeat (optional)",
                output_key="backstab_victim_id",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    LineBehindTargetFilter(target_key="victim_id", length=2),
                ],
            ),
            # 3. Resolve Discard/Defeat Logic
            ForceDiscardOrDefeatStep(
                victim_key="backstab_victim_id",
            ),
            # 4. Resolve Attack Sequence
            AttackSequenceStep(damage=stats.primary_value, target_id_key="victim_id", range_val=1),
        ]


@register_effect("raging_stream")
class RagingStreamEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. Before the attack: Up to 1 enemy hero
    in any of the 3 spaces in a straight line directly behind the target
    discards a card, or is defeated."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select target for Raging Stream attack",
                output_key="victim_id",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                ],
                is_mandatory=True,
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select enemy hero behind target to discard/defeat (optional)",
                output_key="backstab_victim_id",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    LineBehindTargetFilter(target_key="victim_id", length=3),
                ],
            ),
            ForceDiscardOrDefeatStep(
                victim_key="backstab_victim_id",
            ),
            AttackSequenceStep(damage=stats.primary_value, target_id_key="victim_id", range_val=1),
        ]


@register_effect("violent_torrent")
class ViolentTorrentEffect(CardEffect):
    """
    Card text: "Target a unit adjacent to you. Before the attack: Up to 1 enemy hero
    in any of the 5 spaces in a straight line directly behind the target
    discards a card, or is defeated. May repeat once on a different unit."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        attack_steps = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select target for Violent Torrent attack",
                output_key="victim_id_1",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                ],
                is_mandatory=True,
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select enemy hero behind target to discard/defeat (optional)",
                output_key="backstab_victim_id_1",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    LineBehindTargetFilter(target_key="victim_id_1", length=5),
                ],
            ),
            ForceDiscardOrDefeatStep(
                victim_key="backstab_victim_id_1",
            ),
            AttackSequenceStep(
                damage=stats.primary_value, target_id_key="victim_id_1", range_val=1
            ),
        ]

        repeat_steps_template = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select second target for Violent Torrent (optional)",
                output_key="victim_id_2",
                filters=[
                    RangeFilter(max_range=1),
                    TeamFilter(relation="ENEMY"),
                    ExcludeIdentityFilter(exclude_keys=["victim_id_1"]),
                ],
                is_mandatory=False,
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select enemy hero behind second target to discard/defeat (optional)",
                output_key="backstab_victim_id_2",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    LineBehindTargetFilter(target_key="victim_id_2", length=5),
                ],
            ),
            ForceDiscardOrDefeatStep(
                victim_key="backstab_victim_id_2",
            ),
            # Guarded: an unset victim_id_2 would make AttackSequenceStep spawn
            # its own target picker, which carries neither the "different unit"
            # exclusion nor anything else from the select above.
            AttackSequenceStep(
                damage=stats.primary_value,
                target_id_key="victim_id_2",
                range_val=1,
                active_if_key="victim_id_2",
            ),
        ]

        return [
            *attack_steps,
            MayRepeatOnceStep(
                active_if_key="victim_id_1",
                steps_template=repeat_steps_template,
            ),
        ]


@register_effect("liquid_leap")
@register_effect("magical_current")
class TeleportStrictEffect(CardEffect):
    """
    Card text: "Place yourself into a space in range without a spawn point
    and not adjacent to an empty spawn point."
    Used by: Liquid Leap, Magical Current
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination for Teleport",
                output_key="target_hex",
                filters=[
                    RangeFilter(max_range=stats.range),
                    ObstacleFilter(is_obstacle=False),
                    SpawnPointFilter(has_spawn_point=False),
                    AdjacentSpawnPointFilter(is_empty=True, must_not_have=True),
                ],
                is_mandatory=True,
            ),
            PlaceUnitStep(unit_id=hero.id, destination_key="target_hex"),
        ]


@register_effect("stranger_tide")
class TeleportNoSpawnEffect(CardEffect):
    """
    Card text: "Place yourself into a space in range without a spawn point."
    Used by: Stranger Tide
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination for Teleport",
                output_key="target_hex",
                filters=[
                    RangeFilter(max_range=stats.range),
                    ObstacleFilter(is_obstacle=False),
                    SpawnPointFilter(has_spawn_point=False),
                ],
                is_mandatory=True,
            ),
            PlaceUnitStep(unit_id=hero.id, destination_key="target_hex"),
        ]


@register_effect("rogue_wave")
class RogueWaveEffect(CardEffect):
    """
    Card text: "Target a unit in range. After the attack: You may push an enemy unit
    adjacent to you up to 2 spaces."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Attack Sequence (selects target, reaction, damage)
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=card.is_ranged,
            ),
            # 2. Optional: Select enemy adjacent to push
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an adjacent enemy to push (optional)",
                output_key="push_target_id",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),  # Adjacent to Arien
                    TeamFilter(relation="ENEMY"),
                ],
            ),
            # 3. Choose push distance (0, 1, or 2) - only if target selected
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose push distance",
                output_key="push_distance",
                number_options=[0, 1, 2],
                number_labels={0: "Don't Push", 1: "Push 1", 2: "Push 2"},
                active_if_key="push_target_id",
            ),
            # 4. Execute push - reads target and distance from context
            PushUnitStep(
                target_key="push_target_id",
                distance_key="push_distance",
                active_if_key="push_target_id",
                is_mandatory=False,
            ),
        ]


@register_effect("tidal_blast")
class TidalBlastEffect(CardEffect):
    """
    Card text: "Target a unit in range. After the attack: You may push an enemy unit
    adjacent to you up to 3 spaces."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Attack Sequence
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=card.is_ranged,
            ),
            # 2. Select adjacent enemy to push
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an adjacent enemy to push (optional)",
                output_key="push_target_id",
                is_mandatory=False,
                filters=[
                    RangeFilter(max_range=1),  # Adjacent
                    TeamFilter(relation="ENEMY"),
                ],
            ),
            # 3. Choose push distance (0-3)
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose push distance",
                output_key="push_distance",
                number_options=[0, 1, 2, 3],
                number_labels={0: "Don't Push", 1: "Push 1", 2: "Push 2", 3: "Push 3"},
                active_if_key="push_target_id",
            ),
            # 4. Execute push
            PushUnitStep(
                target_key="push_target_id",
                distance_key="push_distance",
                active_if_key="push_target_id",
                is_mandatory=False,
            ),
        ]


@register_effect("slippery_ground")
class SlipperyGroundEffect(CardEffect):
    """
    Card text: "This turn: Enemy heroes adjacent to you cannot fast travel,
    or move more than 1 space with a movement action."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            CreateEffectStep(
                effect_type=EffectType.MOVEMENT_ZONE,
                scope=EffectScope(
                    shape=Shape.ADJACENT,
                    origin_id=hero.id,
                    affects=AffectsFilter.ENEMY_HEROES,
                ),
                duration=DurationType.THIS_TURN,
                max_value=1,
                limit_actions_only=True,
                restrictions=[ActionType.FAST_TRAVEL],
            ),
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
        ]


@register_effect("deluge")
class DelugeEffect(CardEffect):
    """
    Card text: "This turn: Enemy heroes in radius cannot fast travel,
    or move more than 1 space with a movement action."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            CreateEffectStep(
                effect_type=EffectType.MOVEMENT_ZONE,
                scope=EffectScope(
                    shape=Shape.RADIUS,
                    range=stats.radius or 0,
                    origin_id=hero.id,
                    affects=AffectsFilter.ENEMY_HEROES,
                ),
                duration=DurationType.THIS_TURN,
                max_value=1,
                limit_actions_only=True,
                restrictions=[ActionType.FAST_TRAVEL],
            ),
            MoveSequenceStep(unit_id=hero.id, range_val=stats.primary_value),
        ]


@register_effect("aspiring_duelist")
class AspiringDuelistEffect(CardEffect):
    """
    Card text: "Ignore all minion defense modifiers."

    This is a primary DEFENSE card. The effect triggers when used to defend.
    Sets a context flag that ResolveCombatStep checks to skip minion modifier calculation.
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        return [SetContextFlagStep(key="ignore_minion_defense", value=True)]


@register_effect("expert_duelist")
class ExpertDuelistEffect(CardEffect):
    """
    Card text: "Ignore all minion defense modifiers. This turn: You are immune to
    attack actions of all enemy heroes, except this attacker."

    This is a primary DEFENSE card (Tier II upgrade of Aspiring Duelist).
    Two effects:
    1. Sets ignore_minion_defense flag (same as Aspiring Duelist)
    2. Creates ATTACK_IMMUNITY effect on self, with current attacker exempted
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        return [
            # Effect 1: Ignore minion defense modifiers
            SetContextFlagStep(key="ignore_minion_defense", value=True),
            # Effect 2: Immune to attacks from other enemy heroes this turn
            CreateEffectStep(
                effect_type=EffectType.ATTACK_IMMUNITY,
                scope=EffectScope(
                    shape=Shape.POINT,
                    origin_id=defender.id,
                    affects=AffectsFilter.SELF,
                ),
                duration=DurationType.THIS_TURN,
                except_attacker_key="attacker_id",  # Read current attacker from context
                is_active=True,  # Immediately active (defense effect)
            ),
        ]


@register_effect("master_duelist")
class MasterDuelistEffect(CardEffect):
    """
    Card text: "Ignore all minion defense modifiers. This round: You are immune to
    attack actions of all enemy heroes, except this attacker."

    This is a primary DEFENSE card (Tier III upgrade of Expert Duelist).
    Same as Expert Duelist but immunity lasts THIS_ROUND instead of THIS_TURN.
    """

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep] | None:
        return [
            # Effect 1: Ignore minion defense modifiers
            SetContextFlagStep(key="ignore_minion_defense", value=True),
            # Effect 2: Immune to attacks from other enemy heroes this ROUND
            CreateEffectStep(
                effect_type=EffectType.ATTACK_IMMUNITY,
                scope=EffectScope(
                    shape=Shape.POINT,
                    origin_id=defender.id,
                    affects=AffectsFilter.SELF,
                ),
                duration=DurationType.THIS_ROUND,  # Lasts entire round
                except_attacker_key="attacker_id",  # Read current attacker from context
                is_active=True,  # Immediately active (defense effect)
            ),
        ]


# =============================================================================
# ULTIMATE (Purple/Tier IV) - Passive Ability
# =============================================================================


@register_effect("living_tsunami")
class LivingTsunamiEffect(CardEffect):
    """
    Ultimate (Purple) - Arien

    Card text: "Once per turn, before performing an Attack action,
    you may move 1 space."

    This is a passive ability that triggers BEFORE_ATTACK.
    As an ultimate, it's always active once the hero reaches Level 8.
    """

    def get_passive_config(self) -> PassiveConfig | None:
        from goa2.domain.models.enums import PassiveTrigger
        from goa2.engine.effects import PassiveConfig

        return PassiveConfig(
            trigger=PassiveTrigger.BEFORE_ATTACK,
            uses_per_turn=1,
            is_optional=True,
            prompt="Living Tsunami: Move 1 space before attacking?",
        )

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> list[GameStep]:
        from goa2.domain.models.enums import PassiveTrigger

        # Only respond to BEFORE_ATTACK trigger
        if trigger != PassiveTrigger.BEFORE_ATTACK:
            return []

        return [
            MoveSequenceStep(
                unit_id=hero.id,
                range_val=1,
                is_mandatory=False,  # "you may move" - can choose to stay
            )
        ]
