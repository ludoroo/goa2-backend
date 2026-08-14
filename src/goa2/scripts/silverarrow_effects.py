from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import (
    ActionType,
    CardContainerType,
    DurationType,
    EffectType,
    PassiveTrigger,
    TargetType,
)
from goa2.domain.models.effect import AffectsFilter, EffectScope, Shape
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters_composite import (
    CountMatchFilter,
    OrFilter,
)
from goa2.engine.filters_geometry import (
    InStraightLineFilter,
    StraightLinePathFilter,
)
from goa2.engine.filters_hex import (
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
)
from goa2.engine.filters_units import (
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckUnitTypeStep,
    ComputeDistanceStep,
    CreateEffectStep,
    FastTravelSequenceStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    GainCoinsStep,
    GameStep,
    MoveUnitStep,
    RecordHexStep,
    RetrieveCardStep,
    SelectStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# FAMILY 1 — RED: Isolated-Target Snipe (Clear Shot / Opportunity Shot / Snap Shot)
# =============================================================================


class _IsolatedSnipeEffect(CardEffect):
    """
    "Choose one —
    • Target a unit in range, which is not adjacent to any other unit.
    • Target a unit adjacent to you."

    Uses OrFilter to offer the disjunction: the long-range branch requires
    no other unit adjacent to the target (CountMatchFilter with max_count=0),
    while the melee fallback requires adjacency to the shooter.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=True,
                target_filters=[
                    OrFilter(
                        filters=[
                            # Long-range branch: target has zero other units
                            # adjacent to it. min_range=1 excludes the candidate
                            # from its own count (distance 0 to itself).
                            CountMatchFilter(
                                sub_filters=[
                                    RangeFilter(
                                        min_range=1,
                                        max_range=1,
                                        origin_hex_key=CountMatchFilter.ORIGIN_HEX_KEY,
                                    ),
                                ],
                                min_count=0,
                                max_count=0,
                            ),
                            # Melee fallback: target is adjacent to shooter.
                            RangeFilter(max_range=1),
                        ]
                    ),
                ],
            ),
        ]


@register_effect("clear_shot")
class ClearShotEffect(_IsolatedSnipeEffect):
    pass


@register_effect("opportunity_shot")
class OpportunityShotEffect(_IsolatedSnipeEffect):
    pass


@register_effect("snap_shot")
class SnapShotEffect(_IsolatedSnipeEffect):
    pass


# =============================================================================
# FAMILY 2 — RED: Maximum-Range Snipe (Long Shot, Rain of Arrows)
# =============================================================================


@register_effect("long_shot")
class LongShotEffect(CardEffect):
    """Card text: Target a unit at maximum range."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=r,
                is_ranged=True,
                target_filters=[RangeFilter(min_range=r, max_range=r)],
            ),
        ]


@register_effect("rain_of_arrows")
class RainOfArrowsEffect(CardEffect):
    """
    Card text:
    "Target a unit at maximum range.
    If you target a hero, repeat once on a different hero;
    if you do, may repeat once on a minion."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        dmg = stats.primary_value
        return [
            # 1. First attack — any unit at max range.
            AttackSequenceStep(
                damage=dmg,
                range_val=r,
                is_ranged=True,
                target_output_key="rain_victim_1",
                target_filters=[RangeFilter(min_range=r, max_range=r)],
            ),
            # 2. Was the first target a hero? Convert bool -> True/None gate.
            CheckUnitTypeStep(
                unit_key="rain_victim_1",
                expected_type="HERO",
                output_key="_rain_first_is_hero",
            ),
            CheckContextConditionStep(
                input_key="_rain_first_is_hero",
                operator="==",
                threshold=1,
                output_key="rain_first_was_hero",
            ),
            # 3. Mandatory second hero shot (different hero, max range).
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a different enemy hero at maximum range",
                output_key="rain_victim_2",
                is_mandatory=True,
                active_if_key="rain_first_was_hero",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(min_range=r, max_range=r),
                    ExcludeIdentityFilter(exclude_keys=["rain_victim_1"]),
                ],
            ),
            AttackSequenceStep(
                damage=dmg,
                range_val=r,
                is_ranged=True,
                target_id_key="rain_victim_2",
                active_if_key="rain_victim_2",
            ),
            # 4. Optional third minion shot (only if second shot resolved).
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="You may target a different enemy minion at maximum range",
                output_key="rain_victim_3",
                is_mandatory=False,
                active_if_key="rain_victim_2",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(min_range=r, max_range=r),
                    ExcludeIdentityFilter(exclude_keys=["rain_victim_1", "rain_victim_2"]),
                ],
            ),
            AttackSequenceStep(
                damage=dmg,
                range_val=r,
                is_ranged=True,
                target_id_key="rain_victim_3",
                active_if_key="rain_victim_3",
            ),
        ]


# =============================================================================
# FAMILY 3 — BLUE: Root Zones (Grappling Branches / Entangling Vines / Grasping Roots)
# =============================================================================


class _RootZoneEffect(CardEffect):
    """
    "This turn: Enemy heroes in radius cannot fast travel,
    or move more than 1 space with a movement action."

    Identical to Arien's Deluge (minus the trailing MoveSequenceStep).
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
        ]


@register_effect("grappling_branches")
class GrapplingBranchesEffect(_RootZoneEffect):
    pass


@register_effect("entangling_vines")
class EntanglingVinesEffect(_RootZoneEffect):
    pass


@register_effect("grasping_roots")
class GraspingRootsEffect(_RootZoneEffect):
    pass


# =============================================================================
# FAMILY 4 — BLUE: End-of-Turn Sentinel (Treetop Sentinel / Warning Shot)
# =============================================================================


def _sentinel_finishing_steps(hero_id: str, radius: int, defeat_on_fail: bool) -> list[GameStep]:
    discard_step: GameStep = (
        ForceDiscardOrDefeatStep(victim_key="sentinel_victim")
        if defeat_on_fail
        else ForceDiscardStep(victim_key="sentinel_victim")
    )
    return [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="End of turn — select an enemy hero in radius",
            output_key="sentinel_victim",
            is_mandatory=False,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=radius, origin_id=hero_id),
            ],
        ),
        discard_step,
    ]


@register_effect("treetop_sentinel")
class TreetopSentinelEffect(CardEffect):
    """
    "End of turn: An enemy hero in radius discards a card, or is defeated."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            CreateEffectStep(
                effect_type=EffectType.DELAYED_TRIGGER,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_TURN,
                is_active=True,
                finishing_steps=_sentinel_finishing_steps(
                    str(hero.id), radius, defeat_on_fail=True
                ),
            ),
        ]


@register_effect("warning_shot")
class WarningShotEffect(CardEffect):
    """
    "End of turn: An enemy hero in radius discards a card, if able."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            CreateEffectStep(
                effect_type=EffectType.DELAYED_TRIGGER,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_TURN,
                is_active=True,
                finishing_steps=_sentinel_finishing_steps(
                    str(hero.id), radius, defeat_on_fail=False
                ),
            ),
        ]


# =============================================================================
# FAMILY 5 — GREEN: Drag-and-Dance (Lead Astray / Divert Attention / Disorient)
# =============================================================================


def _drag_and_dance_steps(hero_id: str, max_drag_distance: int) -> list[GameStep]:
    """
    Move an enemy unit adjacent to you up to N spaces; if you do,
    move up to that number of spaces in a straight line.

    Implementation:
    1. Select adjacent enemy to drag.
    2. Record its starting hex.
    3. Select destination hex for the drag (up to max_drag_distance from target).
    4. Move the target.
    5. Compute how far it actually moved.
    6. Self-move in a straight line up to that distance (effect-side nudge).
    """
    return [
        # 1. Select adjacent enemy — "Move an enemy unit adjacent to you" is
        #    imperative, so neither the unit nor its destination is skippable
        #    (matches Disorient, which shares the wording).
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an adjacent enemy unit to move",
            output_key="drag_target",
            filters=[
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=1),
            ],
        ),
        # 2. Record starting hex before drag
        RecordHexStep(
            unit_key="drag_target",
            output_key="drag_start_hex",
            active_if_key="drag_target",
        ),
        # 3. Select destination for dragged unit
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Select destination for the enemy unit",
            output_key="drag_dest",
            active_if_key="drag_target",
            filters=[
                RangeFilter(
                    max_range=max_drag_distance,
                    origin_key="drag_target",
                ),
                ObstacleFilter(is_obstacle=False),
                MovementPathFilter(range_val=max_drag_distance, unit_key="drag_target"),
            ],
        ),
        # 4. Move the target (forced movement, not a movement action)
        MoveUnitStep(
            unit_key="drag_target",
            destination_key="drag_dest",
            range_val=max_drag_distance,
            is_movement_action=False,
            active_if_key="drag_dest",
        ),
        # 5. Compute distance moved
        ComputeDistanceStep(
            unit_key="drag_target",
            hex_key="drag_start_hex",
            output_key="drag_distance_moved",
            active_if_key="drag_dest",
        ),
        # 6. Self-move in a straight line, up to drag_distance_moved
        SelectStep(
            target_type=TargetType.HEX,
            prompt="You may move in a straight line",
            output_key="dance_dest",
            is_mandatory=False,
            active_if_key="drag_distance_moved",
            filters=[
                RangeFilter(
                    max_range=0,
                    max_range_key="drag_distance_moved",
                    origin_id=hero_id,
                ),
                InStraightLineFilter(origin_id=hero_id),
                StraightLinePathFilter(origin_id=hero_id),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_id=hero_id,
            destination_key="dance_dest",
            range_val=max_drag_distance,
            is_movement_action=False,
            force_straight_line=True,
            active_if_key="dance_dest",
        ),
    ]


@register_effect("lead_astray")
class LeadAstrayEffect(CardEffect):
    """
    "Move an enemy unit adjacent to you up to 3 spaces;
    if you do, move up to that number of spaces in a straight line."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _drag_and_dance_steps(str(hero.id), max_drag_distance=3)


@register_effect("divert_attention")
class DivertAttentionEffect(CardEffect):
    """
    "Move an enemy unit adjacent to you up to 2 spaces;
    if you do, move up to that number of spaces in a straight line."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _drag_and_dance_steps(str(hero.id), max_drag_distance=2)


@register_effect("disorient")
class DisorientEffect(CardEffect):
    """
    "Move an enemy unit adjacent to you 1 space;
    if you do, you may move 1 space."

    Simplified variant — fixed distance 1, no straight-line constraint on self-move.
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select adjacent enemy
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an adjacent enemy unit to move 1 space",
                output_key="disorient_target",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            # 2. Select destination for target (1 space from target)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination for the enemy unit",
                output_key="disorient_dest",
                is_mandatory=True,
                active_if_key="disorient_target",
                filters=[
                    RangeFilter(max_range=1, origin_key="disorient_target"),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            # 3. Move the target
            MoveUnitStep(
                unit_key="disorient_target",
                destination_key="disorient_dest",
                range_val=1,
                is_movement_action=False,
                active_if_key="disorient_dest",
            ),
            # 4. Self-move 1 space (no straight-line constraint)
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key="disorient_self_dest",
                is_mandatory=False,
                active_if_key="disorient_dest",
                filters=[
                    RangeFilter(max_range=1, origin_id=str(hero.id)),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_id=str(hero.id),
                destination_key="disorient_self_dest",
                range_val=1,
                is_movement_action=False,
                active_if_key="disorient_self_dest",
            ),
        ]


# =============================================================================
# FAMILY 6 — GREEN: Gift Retrieve (Nature's Blessing / Fae Healing)
# =============================================================================


class _GiftRetrieveEffect(CardEffect):
    """
    "A hero in radius may retrieve a discarded card;
    if they do, that hero gains N coins."

    "A hero" is unqualified: enemy heroes are legal recipients too. Only
    Silverarrow herself is excluded, per the standing "a hero excludes the
    actor unless stated" convention (SelectStep adds that filter itself).
    """

    coins: int = 0

    def __init_subclass__(cls, coins: int = 0, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.coins = coins

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # 1. Select any hero in radius — friendly or enemy
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a hero in radius to gift a card retrieval",
                output_key="gift_hero",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    RangeFilter(max_range=stats.radius or 0),
                ],
            ),
            # 2. That hero selects a card from their discard
            SelectStep(
                target_type=TargetType.CARD,
                card_container=CardContainerType.DISCARD,
                context_hero_id_key="gift_hero",
                override_player_id_key="gift_hero",
                prompt="Select a discarded card to retrieve",
                output_key="gift_card",
                is_mandatory=False,
                active_if_key="gift_hero",
            ),
            # 3. Retrieve the card
            RetrieveCardStep(
                card_key="gift_card",
                hero_key="gift_hero",
                active_if_key="gift_card",
            ),
            # 4. Gain coins
            GainCoinsStep(
                hero_key="gift_hero",
                amount=self.coins,
                active_if_key="gift_card",
            ),
        ]


@register_effect("natures_blessing")
class NaturesBlessingEffect(_GiftRetrieveEffect, coins=2):
    """A hero in radius may retrieve a discarded card; if they do, that hero gains 2 coins."""

    pass


@register_effect("fae_healing")
class FaeHealingEffect(_GiftRetrieveEffect, coins=1):
    """A hero in radius may retrieve a discarded card; if they do, that hero gains 1 coin."""

    pass


# =============================================================================
# FAMILY 7 — GOLD: Shoot and Scoot (untiered)
# =============================================================================


@register_effect("shoot_and_scoot")
class ShootAndScootEffect(CardEffect):
    """
    "Target a unit at maximum range.
    After the attack: If able, you may fast travel to an adjacent zone."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=r,
                is_ranged=True,
                target_filters=[RangeFilter(min_range=r, max_range=r)],
            ),
            # "to an adjacent zone" — the current zone is not adjacent to itself.
            FastTravelSequenceStep(unit_id=str(hero.id), require_zone_change=True),
        ]


# =============================================================================
# FAMILY 8 — SILVER: Trailblazer (untiered)
# =============================================================================


@register_effect("trailblazer")
class TrailblazerEffect(CardEffect):
    """
    "You may fast travel, if able.
    This round: You and friendly heroes in radius may ignore obstacles while
    performing movement actions."
    """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            FastTravelSequenceStep(unit_id=str(hero.id)),
            CreateEffectStep(
                effect_type=EffectType.MOVEMENT_AURA_ZONE,
                scope=EffectScope(
                    shape=Shape.RADIUS,
                    range=stats.radius or 0,
                    origin_id=str(hero.id),
                    affects=AffectsFilter.SELF_AND_FRIENDLY_HEROES,
                ),
                duration=DurationType.THIS_ROUND,
                grants_pass_through_obstacles=True,
            ),
        ]


# =============================================================================
# FAMILY 9 — ULTIMATE: Wild Hunt
# =============================================================================


@register_effect("wild_hunt")
class WildHuntEffect(CardEffect):
    """
    "Each time before you perform an action, you may move 2 spaces in a
    straight line."

    This is an effect-side nudge, not a MOVEMENT action, so it intentionally
    does not use MoveSequenceStep or inherit Trailblazer's movement-action aura.
    """

    def get_passive_config(self) -> PassiveConfig:
        return PassiveConfig(
            trigger=PassiveTrigger.BEFORE_ACTION,
            uses_per_turn=0,
            is_optional=True,
            prompt="Wild Hunt: Move 2 spaces in a straight line?",
        )

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict,
    ) -> list[GameStep]:
        if trigger != PassiveTrigger.BEFORE_ACTION:
            return []

        hero_id = str(hero.id)
        return [
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Wild Hunt: Select a straight-line destination",
                output_key="wild_hunt_dest",
                is_mandatory=False,
                filters=[
                    # "move 2 spaces", not "up to 2": exactly 2, and never the
                    # hex Silverarrow already stands on.
                    RangeFilter(min_range=2, max_range=2, origin_id=hero_id),
                    InStraightLineFilter(origin_id=hero_id),
                    StraightLinePathFilter(origin_id=hero_id),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_id=hero_id,
                destination_key="wild_hunt_dest",
                range_val=2,
                is_movement_action=False,
                force_straight_line=True,
                active_if_key="wild_hunt_dest",
            ),
        ]
