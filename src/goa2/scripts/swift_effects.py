"""Card effect implementations for the hero Swift."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import CardState, DurationType, EffectType, TargetType
from goa2.domain.models.effect import EffectScope, Shape
from goa2.domain.models.enums import PassiveTrigger
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters_geometry import (
    InStraightLineFilter,
    NotInStraightLineFilter,
    StraightLinePathFilter,
)
from goa2.engine.filters_hex import (
    AdjacentToTerrainFilter,
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
)
from goa2.engine.filters_units import (
    AdjacencyToContextFilter,
    ImmunityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CreateEffectStep,
    DefeatUnitStep,
    FastTravelSequenceStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    ForEachStep,
    GameStep,
    MayRepeatOnceStep,
    MoveUnitStep,
    MultiSelectStep,
    PerformPrimaryActionStep,
    PlaceUnitStep,
    PushUnitStep,
    RecordHexStep,
    SelectStep,
    SetContextFlagStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# Maximum-range, straight-line snipe (Snipe)
# =============================================================================


@register_effect("snipe")
class SnipeEffect(CardEffect):
    """Card text: "Target a unit at maximum range, and in a straight line." """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=r,
                is_ranged=True,
                target_filters=[
                    RangeFilter(min_range=r, max_range=r),
                    InStraightLineFilter(),
                ],
            ),
        ]


class _StraightLineNonAdjacentShot(CardEffect):
    """Card text: "Target a unit in range, in a straight line, and not adjacent
    to you." Differs from Snipe by allowing any range >= 2 (not only maximum)."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=r,
                is_ranged=True,
                target_filters=[
                    RangeFilter(min_range=2, max_range=r),
                    InStraightLineFilter(),
                ],
            ),
        ]


@register_effect("prepared_shot")
class PreparedShotEffect(_StraightLineNonAdjacentShot):
    pass


@register_effect("killshot")
class KillshotEffect(_StraightLineNonAdjacentShot):
    pass


# =============================================================================
# Pre-attack adjacent-discard shots (Shotgun / Super-Shotgun)
# =============================================================================


class _PreAttackDiscardShot(CardEffect):
    """Card text: "Target a unit in range. Before the attack: An enemy hero
    adjacent to the target discards a card[, if able / , or is defeated]."

    Subclasses set ``defeat_on_fail`` to switch between Shotgun (discard if
    able) and Super-Shotgun (discard, or is defeated)."""

    defeat_on_fail: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.range or 0
        bystander_step: GameStep = (
            ForceDiscardOrDefeatStep(victim_key="shotgun_bystander")
            if self.defeat_on_fail
            else ForceDiscardStep(victim_key="shotgun_bystander")
        )
        return [
            # 1. Pick the attack target (any enemy unit in range).
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select attack target",
                output_key="shotgun_victim",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=r),
                    ImmunityFilter(),
                ],
            ),
            # 2. Before the attack: an enemy hero adjacent to the target.
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero adjacent to the target",
                output_key="shotgun_bystander",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    AdjacencyToContextFilter(target_key="shotgun_victim"),
                    ImmunityFilter(),
                ],
            ),
            bystander_step,
            # 3. The attack itself.
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=r,
                is_ranged=True,
                target_id_key="shotgun_victim",
            ),
        ]


@register_effect("shotgun")
class ShotgunEffect(_PreAttackDiscardShot):
    defeat_on_fail: bool = False


@register_effect("super_shotgun")
class SuperShotgunEffect(_PreAttackDiscardShot):
    defeat_on_fail: bool = True


# =============================================================================
# Jump family — place self in a straight line, then push adjacent enemies
# =============================================================================


def _place_in_straight_line_steps(hero: Hero, radius: int) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Place yourself into a space in a straight line in radius",
            output_key="jump_dest",
            is_mandatory=True,
            filters=[
                RangeFilter(max_range=radius),
                InStraightLineFilter(),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        PlaceUnitStep(unit_id=str(hero.id), destination_key="jump_dest"),
    ]


class _JumpSinglePush(CardEffect):
    """Card text: "Place yourself into a space in a straight line in radius.
    Push an enemy unit adjacent to you up to N spaces." """

    max_push: int = 1

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            *_place_in_straight_line_steps(hero, radius),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Push an enemy unit adjacent to you",
                output_key="jump_push_target",
                is_mandatory=False,
                filters=[TeamFilter(relation="ENEMY"), RangeFilter(max_range=1)],
            ),
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose push distance",
                output_key="jump_push_distance",
                number_options=list(range(self.max_push + 1)),
                active_if_key="jump_push_target",
            ),
            PushUnitStep(
                target_key="jump_push_target",
                distance_key="jump_push_distance",
                active_if_key="jump_push_target",
            ),
        ]


@register_effect("steam_jump")
class SteamJumpEffect(_JumpSinglePush):
    max_push: int = 1


@register_effect("assault_jump")
class AssaultJumpEffect(_JumpSinglePush):
    max_push: int = 2


# =============================================================================
# Suppress family — enemy hero in radius, not adjacent to terrain, discards/defeated
# =============================================================================


class _SuppressEffect(CardEffect):
    """Card text: "An enemy hero in radius who is not adjacent to terrain
    discards a card[, if able / , or is defeated]." """

    defeat_on_fail: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        outcome: GameStep = (
            ForceDiscardOrDefeatStep(victim_key="suppress_victim")
            if self.defeat_on_fail
            else ForceDiscardStep(victim_key="suppress_victim")
        )
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in radius not adjacent to terrain",
                output_key="suppress_victim",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=radius),
                    AdjacentToTerrainFilter(is_adjacent=False),
                ],
            ),
            outcome,
        ]


@register_effect("suppress")
class SuppressEffect(_SuppressEffect):
    defeat_on_fail: bool = False


@register_effect("pin_down")
class PinDownEffect(_SuppressEffect):
    defeat_on_fail: bool = False


@register_effect("killing_ground")
class KillingGroundEffect(_SuppressEffect):
    defeat_on_fail: bool = True


# =============================================================================
# Bounce — straight-line move ignoring obstacles, repeats if already resolved
# =============================================================================


def _bounce_move_steps(hero: Hero) -> list[GameStep]:
    return [
        RecordHexStep(unit_id=str(hero.id), output_key="bounce_origin"),
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Move 2 spaces in a straight line, ignoring obstacles",
            output_key="bounce_dest",
            is_mandatory=True,
            filters=[
                RangeFilter(min_range=2, max_range=2),
                InStraightLineFilter(origin_id=str(hero.id)),
                StraightLinePathFilter(origin_id=str(hero.id), pass_through_obstacles=True),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_id=str(hero.id),
            destination_key="bounce_dest",
            range_val=2,
            pass_through_obstacles=True,
            force_straight_line=True,
            active_if_key="bounce_dest",
        ),
    ]


@register_effect("bounce")
class BounceEffect(CardEffect):
    """Card text: "Move 2 spaces in a straight line, ignoring obstacles; if this
    card is already resolved as you perform this action, may repeat once."

    Bounce's primary action is a SKILL (not a Movement action), so it uses
    SelectStep + MoveUnitStep rather than MoveSequenceStep. The repeat only
    applies when the action is performed while the Bounce card is already
    resolved — i.e. re-performed via Reload or Bullet Time, never on the
    initial play.

    Detecting "already resolved" needs both checks: when copied via Reload the
    card object lives in played_cards (state == RESOLVED), but when re-performed
    directly via Bullet Time it is still current_turn_card (state == UNRESOLVED
    until end of turn), so we also honour the re-performance signal set by
    PerformPrimaryActionStep."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps = _bounce_move_steps(hero)
        is_reperformance = state.execution_context.get("reperforming_card_id") == card.id
        if card.state == CardState.RESOLVED or is_reperformance:
            steps = [*steps, MayRepeatOnceStep(steps_template=_bounce_move_steps(hero))]
        return steps


# =============================================================================
# Mark for Death / Hunting Season — move enemy minion(s) + next-turn coin bounty
# =============================================================================


def _minion_defeat_bounty_step(hero: Hero, radius: int, count: int) -> GameStep:
    """'Next turn: the first <count> times an enemy minion in radius is defeated,
    gain 1 coin.' Radius moves with Swift (origin_id), measured at defeat time."""
    return CreateEffectStep(
        effect_type=EffectType.MINION_DEFEAT_BOUNTY,
        scope=EffectScope(shape=Shape.RADIUS, range=radius, origin_id=str(hero.id)),
        duration=DurationType.NEXT_TURN,
        is_active=True,
        max_value=count,
    )


def _move_minion_to_radius_steps(
    hero: Hero, radius: int, minion_key: str, dest_key: str
) -> list[GameStep]:
    """Move a context-selected enemy minion up to 3 spaces to a space in radius."""
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Move the minion to a space in radius (up to 3 spaces)",
            output_key=dest_key,
            is_mandatory=False,
            active_if_key=minion_key,
            filters=[
                RangeFilter(max_range=radius, origin_id=str(hero.id)),
                MovementPathFilter(range_val=3, unit_key=minion_key),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_key=minion_key,
            destination_key=dest_key,
            range_val=3,
            is_movement_action=False,
            active_if_key=dest_key,
        ),
    ]


@register_effect("mark_for_death")
class MarkForDeathEffect(CardEffect):
    """Card text: "Move an enemy minion in radius up to 3 spaces to a space in
    radius. Next turn: The first time an enemy minion in radius is defeated,
    gain 1 coin." """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy minion in radius to move",
                output_key="mfd_minion",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=radius),
                ],
            ),
            *_move_minion_to_radius_steps(hero, radius, "mfd_minion", "mfd_dest"),
            _minion_defeat_bounty_step(hero, radius, count=1),
        ]


@register_effect("hunting_season")
class HuntingSeasonEffect(CardEffect):
    """Card text: "Move up to two enemy minions in radius, up to 3 spaces each,
    to spaces in radius. Next turn: The first two times an enemy minion in
    radius is defeated, gain 1 coin." """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            MultiSelectStep(
                target_type=TargetType.UNIT,
                prompt="Select up to two enemy minions in radius to move",
                output_key="hs_minions",
                max_selections=2,
                min_selections=0,
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=radius),
                ],
            ),
            ForEachStep(
                list_key="hs_minions",
                item_key="hs_minion",
                steps_template=_move_minion_to_radius_steps(hero, radius, "hs_minion", "hs_dest"),
            ),
            _minion_defeat_bounty_step(hero, radius, count=2),
        ]


# =============================================================================
# End-of-turn placement (Delayed Jump / Mobile Scout)
# =============================================================================


def _end_of_turn_jump_steps(hero: Hero, radius: int, fast_travel: bool) -> list[GameStep]:
    steps: list[GameStep] = [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Place yourself into a space in radius not in a straight line",
            output_key="delayed_jump_dest",
            is_mandatory=True,
            filters=[
                RangeFilter(max_range=radius, origin_id=str(hero.id)),
                NotInStraightLineFilter(origin_id=str(hero.id)),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        PlaceUnitStep(unit_id=str(hero.id), destination_key="delayed_jump_dest"),
    ]
    if fast_travel:
        # "You may then fast travel, if able." FastTravelSequenceStep validates
        # ability and is skippable, matching the optional "you may".
        steps.append(FastTravelSequenceStep(unit_id=str(hero.id)))
    return steps


class _EndOfTurnJumpEffect(CardEffect):
    """Card text: "End of turn: Place yourself into a space in radius not in a
    straight line from you. [You may then fast travel, if able.]"

    Modelled as a THIS_TURN DELAYED_TRIGGER whose finishing steps fire when the
    turn ends (same pattern as Silverarrow's Treetop Sentinel)."""

    fast_travel: bool = False

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
                finishing_steps=_end_of_turn_jump_steps(hero, radius, self.fast_travel),
            ),
        ]


@register_effect("delayed_jump")
class DelayedJumpEffect(_EndOfTurnJumpEffect):
    fast_travel: bool = False


@register_effect("mobile_scout")
class MobileScoutEffect(_EndOfTurnJumpEffect):
    fast_travel: bool = True


# =============================================================================
# Reload! — choose: perform rightmost resolved card, or defeat adjacent minion
# =============================================================================


@register_effect("reload")
class ReloadEffect(CardEffect):
    """Card text: "Choose one —
    • Perform the primary action of your rightmost resolved card.
    • Defeat a minion adjacent to you." """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        # Rightmost resolved card = the most recently resolved card, never Reload!
        # itself. RAW, Bullet Time's re-perform would copy Reload! (already
        # resolved); the designer rules it copies the earlier card, per the older
        # "card in the previous turn slot" wording.
        resolved = [c for c in hero.played_cards if c is not None]
        rightmost_id = resolved[-1].id if resolved else None

        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="reload_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Perform primary action of rightmost resolved card",
                    2: "Defeat a minion adjacent to you",
                },
            ),
            # Branch 1 — perform the rightmost resolved card's primary action.
            CheckContextConditionStep(
                input_key="reload_choice",
                operator="==",
                threshold=1,
                output_key="reload_perform",
            ),
            SetContextFlagStep(key="reload_card", value=rightmost_id),
            PerformPrimaryActionStep(
                card_key="reload_card",
                hero_id=str(hero.id),
                active_if_key="reload_perform",
            ),
            # Branch 2 — defeat an adjacent enemy minion.
            CheckContextConditionStep(
                input_key="reload_choice",
                operator="==",
                threshold=2,
                output_key="reload_defeat",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Defeat a minion adjacent to you",
                output_key="reload_minion",
                is_mandatory=True,
                active_if_key="reload_defeat",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            DefeatUnitStep(
                victim_key="reload_minion",
                killer_id=str(hero.id),
                active_if_key="reload_defeat",
            ),
        ]


@register_effect("drop_trooper")
class DropTrooperEffect(CardEffect):
    """Card text: "Place yourself into a space in a straight line in radius.
    Push up to two enemy units adjacent to you up to 2 spaces." """

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        return [
            *_place_in_straight_line_steps(hero, radius),
            MultiSelectStep(
                target_type=TargetType.UNIT,
                prompt="Push up to two adjacent enemy units",
                output_key="dt_targets",
                max_selections=2,
                min_selections=0,
                is_mandatory=False,
                filters=[TeamFilter(relation="ENEMY"), RangeFilter(max_range=1)],
            ),
            ForEachStep(
                list_key="dt_targets",
                item_key="dt_target",
                steps_template=[
                    SelectStep(
                        target_type=TargetType.NUMBER,
                        prompt="Choose push distance",
                        output_key="dt_distance",
                        number_options=[0, 1, 2],
                    ),
                    PushUnitStep(target_key="dt_target", distance_key="dt_distance"),
                ],
            ),
        ]


# =============================================================================
# Bullet Time (ultimate passive)
# =============================================================================


@register_effect("bullet_time")
class BulletTimeEffect(CardEffect):
    """Ultimate passive: "After you resolve a basic card, you may perform the
    primary action on that card; you cannot target the same enemy hero twice in
    the same turn this way."

    Basic card = Gold/Silver (``card.is_basic``). AFTER_BASIC_ACTION fires after
    every basic action, primary or secondary, and publishes
    ``basic_action_card_id``; a card resolved by its secondary still gets its
    primary performed.

    "Cannot target the same enemy hero twice this way": the just-resolved basic
    action's combat target is published by ResolveCombatStep under the standard
    ``last_combat_target`` key (and ``execution_context`` is cleared each turn,
    so it only reflects this turn's basic action). Passing it as
    ``exclude_target_key`` injects an ExcludeIdentityFilter into the re-performed
    action's selections, so the Bullet Time repeat cannot re-hit that target.
    (If the basic action hit a minion rather than a hero, that minion is also
    excluded — a harmless over-restriction.)
    """

    def get_passive_config(self) -> PassiveConfig:
        return PassiveConfig(
            trigger=PassiveTrigger.AFTER_BASIC_ACTION,
            uses_per_turn=0,  # unlimited within a turn
            is_optional=True,
            prompt="Bullet Time: perform the basic card's primary action again?",
        )

    def should_offer_passive(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict,
    ) -> bool:
        if trigger != PassiveTrigger.AFTER_BASIC_ACTION:
            return False
        return bool(context.get("basic_action_card_id"))

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict,
    ) -> list[GameStep]:
        if not context.get("basic_action_card_id"):
            return []
        return [
            PerformPrimaryActionStep(
                card_key="basic_action_card_id",
                hero_id=str(hero.id),
                exclude_target_key="last_combat_target",
            )
        ]
