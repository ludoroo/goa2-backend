"""Ignatia card effects — the coin-branch chaos hero.

Almost every card branches on the Tie Breaker coin face (see
``GameState.coin_face``): the blue face runs the :tiebreaker_blue: text, the
orange face runs the :tiebreaker_orange: text. Interpretations are locked in
the project memory (project-ignatia-design-decisions).

Three shared mechanics live on ``_IgnatiaBranchEffect`` so every card inherits
them without engine changes:

* **Coin read** — ``build_steps`` picks the branch from ``state.coin_face`` at
  resolve time (the moment the action is performed).
* **Equilibrium** (Silver) — while a THIS_ROUND ``EffectType.EQUILIBRIUM`` effect
  is active she may pick either branch, so the coin read is replaced by a
  blue/orange prompt with both branches gated.
* **Chaos Incarnate** (ultimate) — after performing, ``_maybe_ultimate`` appends
  a ``MayRepeatOnceStep`` whose template flips the coin and performs the action
  again with different targets. Off Equilibrium the re-perform is the opposite
  face (the flip is deterministic, so it is known at build time); on Equilibrium
  it is another free choice. Prior targets are excluded via
  ``ExcludeIdentityFilter`` reading the first performance's (distinct) keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from goa2.domain.models import TargetType, TokenType
from goa2.domain.models.effect import (
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters_composite import CountMatchFilter, OrFilter
from goa2.engine.filters_geometry import (
    HasStraightLineDestinationFilter,
    InStraightLineFilter,
    NotInStraightLineFilter,
    StraightLinePathFilter,
)
from goa2.engine.filters_hex import ObstacleFilter, RangeFilter
from goa2.engine.filters_units import (
    ExcludeIdentityFilter,
    TeamFilter,
    TokenTypeFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CountStep,
    CreateEffectStep,
    DefeatUnitStep,
    FlipTieBreakerCoinStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    GameStep,
    MayRepeatOnceStep,
    MoveUnitStep,
    PlaceTokenStep,
    PlaceTokenTrailStep,
    RecordHexStep,
    RemoveUnitStep,
    SelectStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


_CHAOS_INCARNATE = "chaos_incarnate"


def _ultimate_active(hero: Hero) -> bool:
    """Chaos Incarnate is unlocked (mirrors the level-8 ultimate convention)."""
    return (
        getattr(hero, "level", 0) >= 8
        and hero.ultimate_card is not None
        and hero.ultimate_card.id == _CHAOS_INCARNATE
    )


def _equilibrium_active(state: GameState, hero: Hero) -> bool:
    """A THIS_ROUND Equilibrium effect this hero created is still in play."""
    hid = str(hero.id)
    return any(
        e.effect_type == EffectType.EQUILIBRIUM and e.source_id == hid for e in state.active_effects
    )


def _excl(exclude: list[str]) -> list[ExcludeIdentityFilter]:
    if not exclude:
        return []
    return [ExcludeIdentityFilter(exclude_self=False, exclude_keys=list(exclude))]


def _repeat_gate(prefix: str, target_filters: list) -> list[GameStep]:
    """Steps that set ``{prefix}_can_repeat`` iff a legal repeat target exists.

    ``MayRepeatOnceStep`` prompts before knowing whether its mandatory select
    can be satisfied, so an unsatisfiable YES aborts the whole action. Carrying
    no ``active_if_key``, these two get the branch flag under Equilibrium, so
    the branch she did not pick leaves ``_can_repeat`` unset.
    """
    count_key = f"{prefix}_targets"
    return [
        CountStep(
            target_type=TargetType.UNIT,
            output_key=count_key,
            filters=target_filters,
            skip_immunity_filter=False,  # match SelectStep, which applies it
        ),
        CheckContextConditionStep(
            input_key=count_key,
            operator=">=",
            threshold=1,
            output_key=f"{prefix}_can_repeat",
        ),
    ]


class _IgnatiaBranchEffect(CardEffect):
    """Base for coin-branch cards. Subclasses implement ``_blue_steps`` /
    ``_orange_steps`` (parameterised by a ``slot`` key namespace and a list of
    context keys to ``exclude``) and declare ``_first_target_keys``."""

    # -- subclass hooks ----------------------------------------------------
    def _blue_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        stats: CardStats,
        slot: str,
        exclude: list[str],
    ) -> list[GameStep]:
        raise NotImplementedError

    def _orange_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        stats: CardStats,
        slot: str,
        exclude: list[str],
    ) -> list[GameStep]:
        raise NotImplementedError

    def _first_target_keys(self, slot: str) -> list[str]:
        """Every context key a branch may store a target under, for this slot
        (union across branches; unused ones are None and harmless to exclude)."""
        raise NotImplementedError

    # -- branch assembly ---------------------------------------------------
    def _branch_for_face(
        self, state, hero, card, stats, face: str, slot: str, exclude: list[str]
    ) -> list[GameStep]:
        if face == "BLUE":
            return self._blue_steps(state, hero, card, stats, slot, exclude)
        return self._orange_steps(state, hero, card, stats, slot, exclude)

    def _equilibrium_branch(
        self, state, hero, card, stats, slot: str, exclude: list[str]
    ) -> list[GameStep]:
        blue = self._blue_steps(state, hero, card, stats, slot, exclude)
        orange = self._orange_steps(state, hero, card, stats, slot, exclude)
        # Gate each branch on the chosen side. Steps that ALREADY carry an
        # active_if_key (e.g. Chaos Gate's optional "move 1 space", gated on its
        # own dest key) are left alone: their key is only ever set inside this
        # same (gated) branch, so they can't fire for the other side. Overwriting
        # would break their own optionality.
        for s in blue:
            if s.active_if_key is None:
                s.active_if_key = f"ign_{slot}_is_blue"
        for s in orange:
            if s.active_if_key is None:
                s.active_if_key = f"ign_{slot}_is_orange"
        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Equilibrium: apply blue or orange text?",
                number_options=[1, 2],
                number_labels={1: "Blue", 2: "Orange"},
                output_key=f"ign_{slot}_choice",
                is_mandatory=True,
            ),
            CheckContextConditionStep(
                input_key=f"ign_{slot}_choice",
                operator="==",
                threshold=1,
                output_key=f"ign_{slot}_is_blue",
            ),
            CheckContextConditionStep(
                input_key=f"ign_{slot}_choice",
                operator="==",
                threshold=2,
                output_key=f"ign_{slot}_is_orange",
            ),
            *blue,
            *orange,
        ]

    def _first_branch(self, state, hero, card, stats) -> list[GameStep]:
        if _equilibrium_active(state, hero):
            return self._equilibrium_branch(state, hero, card, stats, "a", [])
        return self._branch_for_face(state, hero, card, stats, state.coin_face, "a", [])

    def _maybe_ultimate(self, state, hero, card, stats) -> list[GameStep]:
        if not _ultimate_active(hero):
            return []
        exclude = self._first_target_keys("a")
        if _equilibrium_active(state, hero):
            repeat = self._equilibrium_branch(state, hero, card, stats, "b", exclude)
        else:
            opposite = "ORANGE" if state.coin_face == "BLUE" else "BLUE"
            repeat = self._branch_for_face(state, hero, card, stats, opposite, "b", exclude)
        return [
            MayRepeatOnceStep(
                prompt=(
                    "Chaos Incarnate: flip the coin and perform this action again "
                    "with different targets?"
                ),
                steps_template=[FlipTieBreakerCoinStep(), *repeat],
            )
        ]

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return self._first_branch(state, hero, card, stats) + self._maybe_ultimate(
            state, hero, card, stats
        )


# =============================================================================
# F1 — Fire attacks (playing_with_fire / erratic_fireblast / loosely_aimed_firebolts)
#   blue  : "Target a unit in range not in a straight line."
#   orange: "Target a unit in range in a straight line."
# =============================================================================


class _FireAttackEffect(_IgnatiaBranchEffect):
    """Ranged attack whose branch difference is the target's alignment with
    Ignatia: blue targets off the straight line, orange targets on it."""

    def _attack(self, stats, slot: str, line_filter, exclude: list[str]) -> AttackSequenceStep:
        return AttackSequenceStep(
            damage=stats.primary_value,
            range_val=stats.range,
            is_ranged=True,
            target_output_key=f"ign_{slot}_v1",
            target_filters=[line_filter, *_excl(exclude)],
        )

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        return [self._attack(stats, slot, NotInStraightLineFilter(), exclude)]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        return [self._attack(stats, slot, InStraightLineFilter(), exclude)]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


@register_effect("playing_with_fire")
class PlayingWithFireEffect(_FireAttackEffect):
    pass


@register_effect("erratic_fireblast")
class ErraticFireblastEffect(_FireAttackEffect):
    pass


@register_effect("loosely_aimed_firebolts")
class LooselyAimedFireboltsEffect(_FireAttackEffect):
    """Tier III fire attack. Orange additionally "may repeat once on a different
    hero" (the repeat fires even when the first target was not a hero)."""

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        first = self._attack(stats, slot, InStraightLineFilter(), exclude)

        def target_filters() -> list:
            # Fresh instances: the gate and the select agree without aliasing.
            return [
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=stats.range),
                InStraightLineFilter(),
                ExcludeIdentityFilter(exclude_keys=[f"ign_{slot}_v1", *exclude]),
            ]

        return [
            first,
            *_repeat_gate(f"ign_{slot}_lf", target_filters()),
            MayRepeatOnceStep(
                active_if_key=f"ign_{slot}_lf_can_repeat",
                prompt="Repeat once on a different enemy hero in a straight line?",
                steps_template=[
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Target a different enemy hero in range and in a straight line",
                        output_key=f"ign_{slot}_v2",
                        is_mandatory=True,
                        filters=target_filters(),
                    ),
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=stats.range,
                        is_ranged=True,
                        target_id_key=f"ign_{slot}_v2",
                    ),
                ],
            ),
        ]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1", f"ign_{slot}_v2"]


# =============================================================================
# F2 — Range-extreme attacks (crack_of_doom / imminent_eruption)
#   blue  : "Target a unit adjacent to you." (range 1, hardcoded)
#   orange: "Target a unit at maximum range." (exactly the card's range)
# =============================================================================


class _RangeExtremeAttackEffect(_IgnatiaBranchEffect):
    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                is_ranged=True,
                target_output_key=f"ign_{slot}_v1",
                target_filters=_excl(exclude),
            )
        ]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=True,
                target_output_key=f"ign_{slot}_v1",
                target_filters=[
                    RangeFilter(min_range=stats.range, max_range=stats.range),
                    *_excl(exclude),
                ],
            )
        ]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


@register_effect("crack_of_doom")
class CrackOfDoomEffect(_RangeExtremeAttackEffect):
    pass


@register_effect("imminent_eruption")
class ImminentEruptionEffect(_RangeExtremeAttackEffect):
    """Blue additionally "may repeat once on a minion" (adjacent).

    "A minion", not "a different minion": a first target that survives stays
    eligible. (The ultimate's "different targets" exclusion still applies.)
    """

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        first = AttackSequenceStep(
            damage=stats.primary_value,
            range_val=1,
            is_ranged=True,
            target_output_key=f"ign_{slot}_v1",
            target_filters=_excl(exclude),
        )

        def target_filters() -> list:
            return [
                UnitTypeFilter(unit_type="MINION"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=1),
                *_excl(exclude),
            ]

        return [
            first,
            *_repeat_gate(f"ign_{slot}_ie", target_filters()),
            MayRepeatOnceStep(
                active_if_key=f"ign_{slot}_ie_can_repeat",
                prompt="Repeat once on an adjacent minion?",
                steps_template=[
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Target an adjacent minion",
                        output_key=f"ign_{slot}_v2",
                        is_mandatory=True,
                        filters=target_filters(),
                    ),
                    AttackSequenceStep(
                        damage=stats.primary_value,
                        range_val=1,
                        is_ranged=True,
                        target_id_key=f"ign_{slot}_v2",
                    ),
                ],
            ),
        ]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1", f"ign_{slot}_v2"]


# =============================================================================
# F3 — Chaos Bolt (Gold basic)
#   blue  : "Target a minion adjacent to you."
#   orange: "Target a hero in range."
# =============================================================================


@register_effect("chaos_bolt")
class ChaosBoltEffect(_IgnatiaBranchEffect):
    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                is_ranged=True,
                target_output_key=f"ign_{slot}_v1",
                target_filters=[UnitTypeFilter(unit_type="MINION"), *_excl(exclude)],
            )
        ]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=True,
                target_output_key=f"ign_{slot}_v1",
                target_filters=[UnitTypeFilter(unit_type="HERO"), *_excl(exclude)],
            )
        ]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


# =============================================================================
# F4 — Discard/Defeat AoE
#   abrupt_combustion (r3) / spontaneous_immolation (r4):
#     blue  : an enemy hero in radius adjacent to a token or a minion discards
#     orange: remove an enemy minion in radius adjacent to an enemy hero
#   violent_conflagration (r4):
#     blue  : ...discards a card, or is defeated
#     orange: defeat an enemy minion in radius adjacent to an enemy hero
# =============================================================================


def _adjacent_to_token_or_minion() -> CountMatchFilter:
    """Presence check: the candidate hex has a token OR a minion adjacent to it
    (any token; not a bare hero). Measured from the candidate via ORIGIN_HEX_KEY."""
    return CountMatchFilter(
        include_tokens=True,
        min_count=1,
        sub_filters=[
            RangeFilter(min_range=1, max_range=1, origin_hex_key=CountMatchFilter.ORIGIN_HEX_KEY),
            OrFilter(filters=[UnitTypeFilter(unit_type="MINION"), TokenTypeFilter()]),
        ],
    )


def _adjacent_to_enemy_hero() -> CountMatchFilter:
    """Presence check: the candidate (an enemy minion) is adjacent to an enemy
    hero (enemy relative to Ignatia)."""
    return CountMatchFilter(
        min_count=1,
        sub_filters=[
            UnitTypeFilter(unit_type="HERO"),
            TeamFilter(relation="ENEMY"),
            RangeFilter(min_range=1, max_range=1, origin_hex_key=CountMatchFilter.ORIGIN_HEX_KEY),
        ],
    )


class _CombustionEffect(_IgnatiaBranchEffect):
    """Subclasses set ``defeat_on_discard`` (blue "or is defeated") and
    ``defeat_minion`` (orange "defeat" vs "remove")."""

    defeat_on_discard: bool = False
    defeat_minion: bool = False

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        victim = f"ign_{slot}_v1"
        select = SelectStep(
            target_type=TargetType.UNIT,
            prompt="An enemy hero in radius adjacent to a token or minion",
            output_key=victim,
            is_mandatory=True,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=stats.radius),
                _adjacent_to_token_or_minion(),
                *_excl(exclude),
            ],
        )
        if self.defeat_on_discard:
            resolve = ForceDiscardOrDefeatStep(victim_key=victim)
        else:
            resolve = ForceDiscardStep(victim_key=victim)
        return [select, resolve]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        victim = f"ign_{slot}_v1"
        select = SelectStep(
            target_type=TargetType.UNIT,
            prompt="An enemy minion in radius adjacent to an enemy hero",
            output_key=victim,
            is_mandatory=True,
            filters=[
                UnitTypeFilter(unit_type="MINION"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=stats.radius),
                _adjacent_to_enemy_hero(),
                *_excl(exclude),
            ],
        )
        if self.defeat_minion:
            resolve = DefeatUnitStep(victim_key=victim, killer_id=str(hero.id))
        else:
            resolve = RemoveUnitStep(unit_key=victim)
        return [select, resolve]

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


@register_effect("abrupt_combustion")
class AbruptCombustionEffect(_CombustionEffect):
    pass


@register_effect("spontaneous_immolation")
class SpontaneousImmolationEffect(_CombustionEffect):
    pass


@register_effect("violent_conflagration")
class ViolentConflagrationEffect(_CombustionEffect):
    defeat_on_discard = True
    defeat_minion = True


# =============================================================================
# F5 — Move a hero in a straight line (searing_heat / scorching_blaze)
#   blue  : move a friendly hero in radius N spaces in a straight line
#   orange: move an enemy hero in radius N spaces in a straight line
# =============================================================================


class _MoveHeroLineEffect(_IgnatiaBranchEffect):
    """Subclasses set the straight-line distance bounds (searing = exactly 2,
    scorching = 2 or 3)."""

    min_dist: int = 2
    max_dist: int = 2

    def _move_hero(
        self, hero_relation: Literal["FRIENDLY", "ENEMY"], stats, slot: str, exclude: list[str]
    ) -> list[GameStep]:
        hkey = f"ign_{slot}_v1"
        dkey = f"ign_{slot}_dest"
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt=f"Select a {hero_relation.lower()} hero in radius",
                output_key=hkey,
                is_mandatory=True,
                # Moving a friendly hero is not an offensive action.
                skip_immunity_filter=(hero_relation == "FRIENDLY"),
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation=hero_relation),
                    RangeFilter(max_range=stats.radius),
                    HasStraightLineDestinationFilter(
                        distance=self.min_dist, max_distance=self.max_dist
                    ),
                    *_excl(exclude),
                ],
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Move it in a straight line",
                output_key=dkey,
                is_mandatory=True,
                filters=[
                    RangeFilter(min_range=self.min_dist, max_range=self.max_dist, origin_key=hkey),
                    StraightLinePathFilter(origin_key=hkey),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_key=hkey,
                destination_key=dkey,
                range_val=self.max_dist,
                is_movement_action=False,
                force_straight_line=True,
            ),
        ]

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        return self._move_hero("FRIENDLY", stats, slot, exclude)

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        return self._move_hero("ENEMY", stats, slot, exclude)

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


@register_effect("searing_heat")
class SearingHeatEffect(_MoveHeroLineEffect):
    min_dist = 2
    max_dist = 2


@register_effect("scorching_blaze")
class ScorchingBlazeEffect(_MoveHeroLineEffect):
    min_dist = 2
    max_dist = 3


# =============================================================================
# F6 — Swaps (unstable_portal / chaos_gate)
#   blue  : swap with a friendly unit in radius
#   orange: swap with an enemy unit in radius
#   chaos_gate: blue then "may move that unit 1"; orange then "may move 1" (self)
# =============================================================================


class _SwapEffect(_IgnatiaBranchEffect):
    def _swap(
        self, relation: Literal["FRIENDLY", "ENEMY"], hero, stats, slot: str, exclude: list[str]
    ) -> list[GameStep]:
        v = f"ign_{slot}_v1"
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt=f"Swap with a {relation.lower()} unit in radius",
                output_key=v,
                is_mandatory=True,
                skip_immunity_filter=(relation == "FRIENDLY"),
                filters=[
                    TeamFilter(relation=relation),
                    RangeFilter(max_range=stats.radius),
                    *_excl(exclude),
                ],
            ),
            SwapUnitsStep(unit_a_id=str(hero.id), unit_b_key=v),
        ]

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        return self._swap("FRIENDLY", hero, stats, slot, exclude)

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        return self._swap("ENEMY", hero, stats, slot, exclude)

    def _first_target_keys(self, slot):
        return [f"ign_{slot}_v1"]


@register_effect("unstable_portal")
class UnstablePortalEffect(_SwapEffect):
    pass


@register_effect("chaos_gate")
class ChaosGateEffect(_SwapEffect):
    """Adds an optional 1-space move: blue moves the swapped unit, orange moves
    Ignatia herself. The move's own dest-key gate (bdest/odest) is distinct per
    branch, so it composes with Equilibrium gating without clobbering."""

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        v = f"ign_{slot}_v1"
        dest = f"ign_{slot}_bdest"
        return [
            *self._swap("FRIENDLY", hero, stats, slot, exclude),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move that unit 1 space",
                output_key=dest,
                is_mandatory=False,
                filters=[
                    RangeFilter(min_range=1, max_range=1, origin_key=v),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_key=v,
                destination_key=dest,
                range_val=1,
                is_movement_action=False,
                active_if_key=dest,
            ),
        ]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        dest = f"ign_{slot}_odest"
        return [
            *self._swap("ENEMY", hero, stats, slot, exclude),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="You may move 1 space",
                output_key=dest,
                is_mandatory=False,
                # No origin -> measured from Ignatia (current actor), post-swap.
                filters=[
                    RangeFilter(min_range=1, max_range=1),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_id=str(hero.id),
                destination_key=dest,
                range_val=1,
                is_movement_action=False,
                active_if_key=dest,
            ),
        ]


# =============================================================================
# F7 — Path / Magma (path_of_ashes N2 / path_of_cinders N3 / path_of_flames N4)
#   blue  : "Move up to N in a straight line. Place a Magma token in each empty
#            space you moved through, or out of." (origin included, dest excluded)
#   orange: "Place up to N Magma tokens in radius."
# =============================================================================


class _PathEffect(_IgnatiaBranchEffect):
    """Subclasses set ``move_dist`` = N (also the orange placement cap)."""

    move_dist: int = 2

    def _blue_steps(self, state, hero, card, stats, slot, exclude):
        origin = f"ign_{slot}_origin"
        dest = f"ign_{slot}_dest"
        crossed_obstacles = f"ign_{slot}_crossed_obstacles"
        move_succeeded = f"ign_{slot}_move_succeeded"
        return [
            RecordHexStep(unit_id=str(hero.id), output_key=origin),
            SelectStep(
                target_type=TargetType.HEX,
                prompt=f"Move up to {self.move_dist} spaces in a straight line",
                output_key=dest,
                is_mandatory=False,
                filters=[
                    RangeFilter(min_range=1, max_range=self.move_dist),
                    StraightLinePathFilter(),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            MoveUnitStep(
                unit_id=str(hero.id),
                destination_key=dest,
                range_val=self.move_dist,
                is_movement_action=False,
                force_straight_line=True,
                crossed_obstacles_output_key=crossed_obstacles,
                success_output_key=move_succeeded,
                active_if_key=dest,
            ),
            PlaceTokenTrailStep(
                token_type=TokenType.MAGMA,
                origin_hex_key=origin,
                dest_key=dest,
                excluded_hexes_key=crossed_obstacles,
                active_if_key=move_succeeded,
            ),
        ]

    def _orange_steps(self, state, hero, card, stats, slot, exclude):
        steps: list[GameStep] = []
        for i in range(self.move_dist):
            hk = f"ign_{slot}_m{i}"
            steps.append(
                SelectStep(
                    target_type=TargetType.HEX,
                    prompt="You may place a Magma token in radius",
                    output_key=hk,
                    is_mandatory=False,
                    filters=[
                        RangeFilter(max_range=stats.radius),
                        ObstacleFilter(is_obstacle=False),
                    ],
                )
            )
            steps.append(PlaceTokenStep(token_type=TokenType.MAGMA, hex_key=hk, active_if_key=hk))
        return steps

    def _first_target_keys(self, slot):
        return []  # tokens/hexes only — nothing to exclude on the ultimate re-perform


@register_effect("path_of_ashes")
class PathOfAshesEffect(_PathEffect):
    move_dist = 2


@register_effect("path_of_cinders")
class PathOfCindersEffect(_PathEffect):
    move_dist = 3


@register_effect("path_of_flames")
class PathOfFlamesEffect(_PathEffect):
    move_dist = 4


# =============================================================================
# F8 — Equilibrium (Silver basic): raise the THIS_ROUND free-choice flag.
# Not a branch card itself; it is what _equilibrium_active() detects.
# =============================================================================


@register_effect("equilibrium")
class EquilibriumEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            CreateEffectStep(
                effect_type=EffectType.EQUILIBRIUM,
                scope=EffectScope(shape=Shape.POINT, affects=AffectsFilter.SELF),
                duration=DurationType.THIS_ROUND,
            )
        ]
