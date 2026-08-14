"""Hanu card effects.

Hanu is a trickster / positioning / support hero. His kit centers on swaps
(self<->enemy hero, two friendly units), coordinated movement, alternate-target
attacks anchored on a friendly hero, and initiative disruption (Hurry Up!).

Locked interpretations live in project memory (project-hanu-design-decisions):
- Self is never a valid "a friendly unit" / "a hero" option (auto-excluded).
- Hurry Up! targets any hero (not self) and sets a card's BASE initiative to 11;
  other initiative modifiers still stack.
- Journey line: swapped enemy is immune to everyone this turn, then an
  end-of-turn forced swap-back ignores range and immunity.
- Fight and Flight flee (3, straight) is forced if able, only if target survived.
- The Ultimate Trick (Purple): at level 8, Hurry Up! also lets Hanu's player
  choose the target's next action (input remap in engine/handler.py; only the
  decision-maker changes, legality stays with the controlled hero).

Reuse map:
- Monkey Trick/Twist/Business  -> swap two friendly units (Misa pattern)
- Hear/See Nothing             -> swap self with enemy hero
- Helping Hand / Outnumber fam -> attack; bullet A adjacent, bullet B anchored on
                                  a friendly hero (AdjacencyFilter FRIENDLY,HERO)
- This Way / That Way          -> co-directional move of self + friendly hero
- Journey line                 -> swap + immunity + finishing-step swap-back
- Fight and Flight             -> attack + conditional straight-line flee
- Hurry Up!                    -> SetCardInitiativeStep (new)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from goa2.domain.models import TargetType
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_cards import HasUnresolvedCardFilter
from goa2.engine.filters_composite import AndFilter, OrFilter
from goa2.engine.filters_geometry import (
    CoMoverValidHexFilter,
    InStraightLineFilter,
    StraightLinePathFilter,
)
from goa2.engine.filters_hex import MovementPathFilter, ObstacleFilter, RangeFilter
from goa2.engine.filters_units import (
    AdjacencyFilter,
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CoDirectionalDragStep,
    GameStep,
    MoveUnitStep,
    ScheduleActionControlStep,
    ScheduleJourneyReturnStep,
    SelectStep,
    SetCardInitiativeStep,
    SetContextFlagStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


def _optional_self_move_1(hero: Hero, dest_key: str = "hanu_move_dest") -> list[GameStep]:
    """ "You may move 1 space." — an effect-side self nudge (not a movement action),
    respecting obstacles and a real 1-hex movement path."""
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="You may move 1 space",
            output_key=dest_key,
            is_mandatory=False,
            filters=[
                MovementPathFilter(range_val=1, unit_id=hero.id),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_id=hero.id,
            destination_key=dest_key,
            range_val=1,
            is_movement_action=False,
            active_if_key=dest_key,
        ),
    ]


# =============================================================================
# GREEN — Monkey Trick / Twist / Business
# "Swap two friendly units in radius." (Business also: "You may move 1 space.")
# Self is never a valid "friendly unit" (SelectStep auto-excludes the actor).
# =============================================================================


class _MonkeySwapEffect(CardEffect):
    """Swap two friendly units in radius; optionally move 1 space after."""

    move_after: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        r = stats.radius
        steps: list[GameStep] = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select the first friendly unit to swap",
                output_key="swap_a",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=r),
                ],
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select the second friendly unit to swap",
                output_key="swap_b",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=r),
                    ExcludeIdentityFilter(exclude_keys=["swap_a"]),
                ],
            ),
            SwapUnitsStep(unit_a_key="swap_a", unit_b_key="swap_b"),
        ]
        if self.move_after:
            steps.extend(_optional_self_move_1(hero))
        return steps


@register_effect("monkey_trick")
class MonkeyTrickEffect(_MonkeySwapEffect):
    """Radius 1 swap."""


@register_effect("monkey_twist")
class MonkeyTwistEffect(_MonkeySwapEffect):
    """Radius 2 swap."""


@register_effect("monkey_business")
class MonkeyBusinessEffect(_MonkeySwapEffect):
    """Radius 2 swap, then optional 1-space move."""

    move_after: bool = True


# =============================================================================
# GREEN — Hear Nothing / See Nothing
# "Swap with an enemy hero in radius." (See Nothing: "You may move 1 space.")
# =============================================================================


def _swap_with_enemy_hero_step(hero: Hero, radius: int | None, output_key: str) -> SelectStep:
    return SelectStep(
        target_type=TargetType.UNIT,
        prompt="Select an enemy hero to swap with",
        output_key=output_key,
        is_mandatory=True,
        filters=[
            UnitTypeFilter(unit_type="HERO"),
            TeamFilter(relation="ENEMY"),
            RangeFilter(max_range=radius),
        ],
    )


class _EnemyHeroSwapEffect(CardEffect):
    """Swap self with an enemy hero in radius; optionally move 1 space after."""

    move_after: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            _swap_with_enemy_hero_step(hero, stats.radius, "swap_enemy"),
            SwapUnitsStep(unit_a_id=str(hero.id), unit_b_key="swap_enemy"),
        ]
        if self.move_after:
            steps.extend(_optional_self_move_1(hero))
        return steps


@register_effect("hear_nothing")
class HearNothingEffect(_EnemyHeroSwapEffect):
    """Radius 3 swap with an enemy hero."""


@register_effect("see_nothing")
class SeeNothingEffect(_EnemyHeroSwapEffect):
    """Radius 3 swap with an enemy hero, then optional 1-space move."""

    move_after: bool = True


# =============================================================================
# RED — Helping Hand / Even the Odds / Trusted Sidekick (bullet B = hero) and
# Outnumber / Pile On (bullet B = minion).
#   Bullet A: "Target a unit adjacent to you."
#   Bullet B: "Target a hero/minion in range, adjacent to your friendly hero"
#             (Trusted Sidekick: "and not adjacent to you").
# "Choose one" = one attack whose target satisfies either bullet.
# "Choose one, or both" (Trusted Sidekick / Pile On) = one optional attack per
# bullet.
# =============================================================================


def _bullet_b_filters(bullet_b_type: Literal["HERO", "MINION"], not_adjacent: bool) -> list:
    """Filters for "a hero/minion adjacent to your friendly hero" (AdjacencyFilter
    excludes Hanu himself as the anchor). Range is bounded by the caller's
    AttackSequenceStep range_val."""
    filters: list = [
        UnitTypeFilter(unit_type=bullet_b_type),
        AdjacencyFilter(target_tags=["FRIENDLY", "HERO"]),
    ]
    if not_adjacent:
        filters.append(RangeFilter(min_range=2))
    return filters


class _AltTargetAttackEffect(CardEffect):
    """ "Choose one" alternate-target attack: bullet A (adjacent) or bullet B
    (in-range unit adjacent to a friendly hero). One attack."""

    bullet_b_type: Literal["HERO", "MINION"] = "HERO"
    bullet_b_not_adjacent: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range or 1,
                is_ranged=card.is_ranged,
                target_filters=[
                    OrFilter(
                        filters=[
                            RangeFilter(max_range=1),  # bullet A: adjacent to you
                            AndFilter(
                                filters=_bullet_b_filters(
                                    self.bullet_b_type, self.bullet_b_not_adjacent
                                )
                            ),
                        ]
                    )
                ],
            )
        ]


class _AltTargetBothAttackEffect(CardEffect):
    """ "Choose one, or both, in any order" — one optional attack per bullet.

    The player first picks which bullet to resolve first, then each attack is
    offered optionally so they can do just one, both, or (via the order choice)
    resolve them in either order — bullet A's or bullet B's board effects can
    land first. (Mirrors Bain's Hunter-Seeker.)"""

    bullet_b_type: Literal["HERO", "MINION"] = "HERO"
    bullet_b_not_adjacent: bool = True

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        rng = stats.range or 1
        dmg = stats.primary_value
        b_filters = _bullet_b_filters(self.bullet_b_type, self.bullet_b_not_adjacent)

        def bullet_a(active_if_key: str) -> AttackSequenceStep:
            # Bullet A: a unit adjacent to you.
            return AttackSequenceStep(
                damage=dmg,
                range_val=1,
                is_ranged=card.is_ranged,
                is_mandatory=False,
                active_if_key=active_if_key,
            )

        def bullet_b(active_if_key: str) -> AttackSequenceStep:
            # Bullet B: a hero/minion in range adjacent to a friendly hero.
            return AttackSequenceStep(
                damage=dmg,
                range_val=rng,
                is_ranged=card.is_ranged,
                is_mandatory=False,
                target_filters=b_filters,
                active_if_key=active_if_key,
            )

        return [
            # Choose which bullet to resolve first ("in any order").
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose which attack to resolve first",
                output_key="ats_order",
                number_options=[1, 2],
                number_labels={
                    1: "Attack an adjacent unit first",
                    2: "Attack a unit by your friendly hero first",
                },
                is_mandatory=True,
            ),
            CheckContextConditionStep(
                input_key="ats_order",
                operator="==",
                threshold=1,
                output_key="ats_a_first",
            ),
            CheckContextConditionStep(
                input_key="ats_order",
                operator="==",
                threshold=2,
                output_key="ats_b_first",
            ),
            # PATH A first: bullet A then bullet B.
            bullet_a("ats_a_first"),
            bullet_b("ats_a_first"),
            # PATH B first: bullet B then bullet A.
            bullet_b("ats_b_first"),
            bullet_a("ats_b_first"),
        ]


@register_effect("helping_hand")
class HelpingHandEffect(_AltTargetAttackEffect):
    """Attack 3, range 3. Bullet B targets a hero."""


@register_effect("even_the_odds")
class EvenTheOddsEffect(_AltTargetAttackEffect):
    """Attack 4, range 4. Bullet B targets a hero."""


@register_effect("outnumber")
class OutnumberEffect(_AltTargetAttackEffect):
    """Attack 5, range 4. Bullet B targets a minion."""

    bullet_b_type: Literal["HERO", "MINION"] = "MINION"


@register_effect("trusted_sidekick")
class TrustedSidekickEffect(_AltTargetBothAttackEffect):
    """Attack 4, range 4. Bullet B: hero, not adjacent to you. Choose one or both."""

    bullet_b_type: Literal["HERO", "MINION"] = "HERO"


@register_effect("pile_on")
class PileOnEffect(_AltTargetBothAttackEffect):
    """Attack 5, range 4. Bullet B: minion. Choose one or both."""

    bullet_b_type: Literal["HERO", "MINION"] = "MINION"
    bullet_b_not_adjacent: bool = False


# =============================================================================
# BLUE — This Way! / That Way!
# "A friendly hero in radius chooses a distance of 1/2(/3); move both of you
#  that number of spaces in the same direction of your choice. Both must be
#  moved the full distance, or neither one moves."
# The friendly hero picks the distance; Hanu picks the direction (a hex).
# CoMoverValidHexFilter enforces the all-or-nothing co-move.
# =============================================================================


class _CoDirectionalMoveEffect(CardEffect):
    """Move Hanu and a chosen friendly hero the same full distance/direction."""

    max_distance: int = 2

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            # Publish Hanu's id so filters/drag can resolve the anchor from context.
            SetContextFlagStep(key="tw_anchor", value=str(hero.id)),
            # Hanu chooses WHICH friendly hero comes along.
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly hero to move with you",
                output_key="tw_partner",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius),
                ],
            ),
            # That friendly hero chooses the distance.
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose a distance to move",
                output_key="tw_distance",
                is_mandatory=True,
                override_player_id_key="tw_partner",
                context_hero_id_key="tw_partner",
                number_options=list(range(1, self.max_distance + 1)),
            ),
            # Hanu chooses the direction: a hex at exactly the chosen distance in a
            # straight line where BOTH can complete the full move.
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Choose a direction (both move the full distance)",
                output_key="tw_dest",
                is_mandatory=True,
                filters=[
                    RangeFilter(
                        min_range_key="tw_distance",
                        max_range_key="tw_distance",
                        origin_key="tw_anchor",
                    ),
                    InStraightLineFilter(origin_key="tw_anchor"),
                    CoMoverValidHexFilter(anchor_key="tw_anchor", partner_key="tw_partner"),
                ],
            ),
            CoDirectionalDragStep(
                anchor_key="tw_anchor",
                partner_key="tw_partner",
                anchor_dest_key="tw_dest",
                anchor_is_token=False,
            ),
        ]


@register_effect("this_way")
class ThisWayEffect(_CoDirectionalMoveEffect):
    """Radius 3; distance 1-2."""

    max_distance: int = 2


@register_effect("that_way")
class ThatWayEffect(_CoDirectionalMoveEffect):
    """Radius 3; distance 1-3."""

    max_distance: int = 3


# =============================================================================
# BLUE — Unexpected Journey / There and Back Again / Safe Travels
# "Swap with an enemy hero in radius. This turn: That hero is immune.
#  End of turn: Swap with that hero, regardless of radius and immunity."
# (Safe Travels: end-of-turn "You may move 1 space".)
# =============================================================================


class _JourneyEffect(CardEffect):
    """Swap with an enemy hero, grant them immunity this turn, and schedule an
    end-of-turn forced swap-back (regardless of range/immunity)."""

    move_after: bool = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            _swap_with_enemy_hero_step(hero, stats.radius, "journey_enemy"),
            SwapUnitsStep(unit_a_id=str(hero.id), unit_b_key="journey_enemy"),
            ScheduleJourneyReturnStep(enemy_key="journey_enemy", move_after=self.move_after),
        ]


@register_effect("unexpected_journey")
class UnexpectedJourneyEffect(_JourneyEffect):
    """Radius 2."""


@register_effect("there_and_back_again")
class ThereAndBackAgainEffect(_JourneyEffect):
    """Radius 3."""


@register_effect("safe_travels")
class SafeTravelsEffect(_JourneyEffect):
    """Radius 3; end-of-turn optional 1-space move after the swap-back."""

    move_after: bool = True


# =============================================================================
# GOLD — Fight and Flight
# "Target a unit adjacent to you. If the target is not defeated, After the
#  attack: If able, move 3 spaces in a straight line."
# The flee is forced when able (full distance, straight line).
# =============================================================================


@register_effect("fight_and_flight")
class FightAndFlightEffect(CardEffect):
    """Attack an adjacent unit; if it survives, flee 3 in a straight line."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                is_ranged=card.is_ranged,
                target_output_key="ff_victim",
            ),
            # "not defeated" == the attack was blocked (target survived).
            CheckContextConditionStep(
                input_key="block_succeeded",
                operator="==",
                threshold=1,
                output_key="ff_survived",
            ),
            # Forced full 3-space straight-line flee, if able. Mandatory so the
            # move is not declinable when a legal line exists; when survived but
            # no legal line exists there simply are no candidates (harmless abort
            # at the final step — the attack has already resolved).
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Flee: move 3 spaces in a straight line",
                output_key="ff_dest",
                is_mandatory=True,
                active_if_key="ff_survived",
                filters=[
                    RangeFilter(min_range=3, max_range=3),
                    InStraightLineFilter(origin_id=hero.id),
                    StraightLinePathFilter(origin_id=hero.id),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_id=hero.id,
                destination_key="ff_dest",
                range_val=3,
                is_movement_action=False,
                force_straight_line=True,
                active_if_key="ff_dest",
            ),
        ]


# =============================================================================
# SILVER — Hurry Up!
# "Set the printed Initiative value of an unresolved card of a hero in range to
#  11, until it is resolved, or otherwise changes state."
# Targets any hero (not self, auto-excluded) in range 4. Base override so items
# and other Initiative modifiers still stack.
# =============================================================================


@register_effect("hurry_up")
class HurryUpEffect(CardEffect):
    """Set a target hero's unresolved card initiative to 11 until it resolves."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a hero whose unresolved card gets Initiative 11",
                output_key="hurry_target",
                is_mandatory=True,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    RangeFilter(max_range=stats.range),
                    HasUnresolvedCardFilter(),
                ],
            ),
            SetCardInitiativeStep(hero_key="hurry_target", value=11),
            # The Ultimate Trick (level 8): Hanu's player controls the
            # target's next action (this card, this round).
            ScheduleActionControlStep(hero_key="hurry_target"),
        ]


# =============================================================================
# PURPLE — The Ultimate Trick
# "You choose the next action, and how it is performed, for a hero you target
#  with the Hurry Up!." Passive: the behavior is implemented by
# ScheduleActionControlStep (appended by Hurry Up!, gated on level 8) plus the
# player_id remap in engine/handler.py. This effect itself contributes no steps.
# =============================================================================


@register_effect("the_ultimate_trick")
class TheUltimateTrickEffect(CardEffect):
    """Passive marker — control logic lives in Hurry Up! + the handler remap."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return []
