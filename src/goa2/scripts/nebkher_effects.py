"""NebKher card effects — critical-infrastructure cards.

NebKher is an illusion/topology hero. This module implements the five effects
that ride on new engine primitives (see
docs/superpowers/plans/2026-07-07-nebkher-tdd-paths.md):

- crack_in_reality / shift_reality  -> topology split (TopologyService)
- mind_grip                         -> PerformCardActionStep (enemy prev slot,
                                       Illusion substitution, marker skip)
- diabolical_laughter               -> LaughStep + up-to-3 menu +
                                       SwapResolvedCardsStep
- what_the_hell_are_you (ultimate)  -> AFTER_LAUGH passive

The delegable families (Imbue Doubt line, Illusion placement line,
Phantasmal line) are implemented separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import (
    CardContainerType,
    CardState,
    TokenType,
)
from goa2.domain.models.effect import DurationType, EffectScope, EffectType, Shape
from goa2.domain.models.enums import PassiveTrigger, TargetType
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters import (
    AdjacentToTokenFilter,
    AndFilter,
    CountMatchFilter,
    ExcludeIdentityFilter,
    HasPreviousSlotCardFilter,
    ImmunityFilter,
    MovementPathFilter,
    ObstacleFilter,
    OrFilter,
    RangeFilter,
    TeamFilter,
    TokenTypeFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    ChooseCardColorStep,
    CollectUnitsStep,
    CreateEffectStep,
    DefeatUnitStep,
    ForceDiscardByColorStep,
    ForceDiscardOrDefeatStep,
    ForEachStep,
    GameStep,
    LaughStep,
    MoveUnitStep,
    PerformCardActionStep,
    PlaceTokenStep,
    SelectStep,
    SwapResolvedCardsStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


# =============================================================================
# CRACK IN REALITY / SHIFT REALITY
# "Split the board into two sides with a straight line of spaces drawn
#  through your space. This turn: …cannot interact… as if they did not exist."
# =============================================================================

_AXES: list[tuple[int, str]] = [(1, "q"), (2, "r"), (3, "s")]


@register_effect("crack_in_reality")
class CrackInRealityEffect(CardEffect):
    """Tier-2 split: the two sides cannot interact with each other; the line
    itself bridges both. The line is fixed at cast time."""

    topology_effect_type: EffectType = EffectType.TOPOLOGY_SPLIT

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        hero_hex = state.get_position(str(hero.id))
        if hero_hex is None:
            return []

        steps: list[GameStep] = [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose the line of spaces drawn through your space",
                output_key="reality_axis",
                number_options=[idx for idx, _ in _AXES],
                number_labels={
                    idx: f"Line along the {axis}-axis (through your space)" for idx, axis in _AXES
                },
                is_mandatory=True,
            )
        ]
        for idx, axis in _AXES:
            steps.append(
                CheckContextConditionStep(
                    input_key="reality_axis",
                    operator="==",
                    threshold=idx,
                    output_key=f"reality_axis_{axis}",
                )
            )
            steps.append(
                CreateEffectStep(
                    effect_type=self.topology_effect_type,
                    scope=EffectScope(shape=Shape.GLOBAL, origin_id=str(hero.id)),
                    duration=DurationType.THIS_TURN,
                    is_active=True,
                    split_axis=axis,
                    split_value=getattr(hero_hex, axis),
                    # Fallback anchor only — the live source position takes
                    # precedence for Shift Reality's isolation (S6).
                    isolated_hex=hero_hex,
                    active_if_key=f"reality_axis_{axis}",
                )
            )
        return steps


@register_effect("shift_reality")
class ShiftRealityEffect(CrackInRealityEffect):
    """Tier-3 split: additionally, units on either side cannot interact with
    NebKher himself; only units on the line can. The restriction is one-way —
    NebKher is not affected by his own ability, so for his movement, targeting,
    placement and radius the split does not exist. The protection follows him
    wherever he moves, not the cast-time hex."""

    topology_effect_type: EffectType = EffectType.TOPOLOGY_ISOLATION


# =============================================================================
# MIND GRIP (gold basic skill, ranged 5)
# "Choose one —
#  • Perform an action on the card in the previous turn slot of an enemy hero
#    in range; if you would place any tokens this way, place Illusion tokens
#    instead; skip giving markers.
#  • Defeat a minion adjacent to you."
# =============================================================================


@register_effect("mind_grip")
class MindGripEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Mind Grip — choose one",
                output_key="mg_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Perform an action on an enemy hero's previous card",
                    2: "Defeat a minion adjacent to you",
                },
                is_mandatory=True,
            ),
            # Bullet 1: perform an action from the enemy's previous turn slot.
            CheckContextConditionStep(
                input_key="mg_choice", operator="==", threshold=1, output_key="mg_perform"
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range with a card in their previous turn slot",
                output_key="mg_target_hero",
                is_mandatory=True,
                active_if_key="mg_perform",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range or 0),
                    HasPreviousSlotCardFilter(),
                ],
            ),
            PerformCardActionStep(
                card_owner_key="mg_target_hero",
                previous_slot=True,
                hero_id=str(hero.id),
                token_type_override=TokenType.ILLUSION,
                skip_markers=True,
                active_if_key="mg_target_hero",
            ),
            # Bullet 2: defeat an adjacent minion.
            CheckContextConditionStep(
                input_key="mg_choice", operator="==", threshold=2, output_key="mg_defeat"
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a minion adjacent to you to defeat",
                output_key="mg_minion",
                is_mandatory=True,
                active_if_key="mg_defeat",
                filters=[
                    UnitTypeFilter(unit_type="MINION"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                ],
            ),
            DefeatUnitStep(
                victim_key="mg_minion",
                killer_id=str(hero.id),
                active_if_key="mg_minion",
            ),
        ]


# =============================================================================
# DIABOLICAL LAUGHTER (silver basic skill, radius 4)
# "Laugh diabolically; if you do, choose up to three times —
#  • Swap with an Illusion token in radius.
#  • Place an Illusion token in an adjacent space.
#  • Swap two resolved cards of an enemy hero in radius, without canceling
#    active effects."
# =============================================================================


def _laughter_menu_steps(
    hero_id: str, radius: int, prefix: str, active_if_key: str | None = None
) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.NUMBER,
            prompt="Diabolical Laughter — choose an option",
            output_key=f"{prefix}_pick",
            number_options=[1, 2, 3],
            number_labels={
                1: "Swap with an Illusion token in radius",
                2: "Place an Illusion token in an adjacent space",
                3: "Swap two resolved cards of an enemy hero in radius",
            },
            is_mandatory=False,
            active_if_key=active_if_key,
        ),
        # Bullet 1: swap self with an Illusion token in radius.
        CheckContextConditionStep(
            input_key=f"{prefix}_pick", operator="==", threshold=1, output_key=f"{prefix}_b1"
        ),
        SelectStep(
            target_type=TargetType.UNIT_OR_TOKEN,
            prompt="Select an Illusion token in radius",
            output_key=f"{prefix}_swap_token",
            is_mandatory=True,
            active_if_key=f"{prefix}_b1",
            filters=[
                UnitTypeFilter(unit_type="TOKEN"),
                TokenTypeFilter(token_type=TokenType.ILLUSION),
                RangeFilter(max_range=radius),
            ],
        ),
        SwapUnitsStep(
            unit_a_id=hero_id,
            unit_b_key=f"{prefix}_swap_token",
            active_if_key=f"{prefix}_swap_token",
        ),
        # Bullet 2: place an Illusion token in an adjacent empty space.
        CheckContextConditionStep(
            input_key=f"{prefix}_pick", operator="==", threshold=2, output_key=f"{prefix}_b2"
        ),
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Select an adjacent space for the Illusion token",
            output_key=f"{prefix}_place_hex",
            is_mandatory=True,
            active_if_key=f"{prefix}_b2",
            filters=[
                RangeFilter(max_range=1),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        PlaceTokenStep(
            token_type=TokenType.ILLUSION,
            hex_key=f"{prefix}_place_hex",
            active_if_key=f"{prefix}_place_hex",
        ),
        # Bullet 3: swap two resolved cards of an enemy hero in radius.
        CheckContextConditionStep(
            input_key=f"{prefix}_pick", operator="==", threshold=3, output_key=f"{prefix}_b3"
        ),
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an enemy hero in radius with two resolved cards",
            output_key=f"{prefix}_victim",
            is_mandatory=True,
            active_if_key=f"{prefix}_b3",
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=radius),
                _two_resolved_cards_filter(),
            ],
        ),
        SelectStep(
            target_type=TargetType.CARD,
            prompt="Select the first resolved card to swap",
            output_key=f"{prefix}_card_a",
            card_container=CardContainerType.PLAYED,
            card_states=[CardState.RESOLVED],
            context_hero_id_key=f"{prefix}_victim",
            is_mandatory=True,
            active_if_key=f"{prefix}_victim",
        ),
        SelectStep(
            target_type=TargetType.CARD,
            prompt="Select the second resolved card to swap",
            output_key=f"{prefix}_card_b",
            card_container=CardContainerType.PLAYED,
            card_states=[CardState.RESOLVED],
            exclude_card_id_keys=[f"{prefix}_card_a"],
            context_hero_id_key=f"{prefix}_victim",
            is_mandatory=True,
            active_if_key=f"{prefix}_card_a",
        ),
        SwapResolvedCardsStep(
            hero_key=f"{prefix}_victim",
            card_a_key=f"{prefix}_card_a",
            card_b_key=f"{prefix}_card_b",
            active_if_key=f"{prefix}_card_b",
        ),
    ]


def _two_resolved_cards_filter():
    from goa2.engine.filters import CardsInContainerFilter

    return CardsInContainerFilter(container=CardContainerType.PLAYED, min_cards=2)


@register_effect("diabolical_laughter")
class DiabolicalLaughterEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 0
        hero_id_str = str(hero.id)

        steps: list[GameStep] = [LaughStep(output_key="dl_laughed")]

        # We generate 3 iterations of optional choice steps
        # Chain them so choice_i is only active if choice_{i-1} was made.
        previous_key = "dl_laughed"
        for i in range(3):
            prefix = f"dl_choice_{i}"
            steps.extend(
                _laughter_menu_steps(hero_id_str, radius, prefix, active_if_key=previous_key)
            )
            previous_key = f"{prefix}_pick"

        return steps


# =============================================================================
# ULTIMATE — WHAT THE HELL ARE YOU? (passive, radius 5)
# "Each time after you laugh diabolically as part of performing an action,
#  all enemy heroes in radius discard a card, or are defeated."
# =============================================================================


@register_effect("what_the_hell_are_you")
class WhatTheHellAreYouEffect(CardEffect):
    def get_passive_config(self) -> PassiveConfig:
        return PassiveConfig(
            trigger=PassiveTrigger.AFTER_LAUGH,
            uses_per_turn=0,  # every laugh
            is_optional=False,  # "all enemy heroes … discard, or are defeated"
        )

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict,
    ) -> list[GameStep]:
        if trigger != PassiveTrigger.AFTER_LAUGH:
            return []
        from goa2.domain.types import UnitID
        from goa2.engine.stats import compute_card_stats

        stats = compute_card_stats(state, UnitID(str(hero.id)), card)
        radius = stats.radius or 0
        return [
            CollectUnitsStep(
                target_type=TargetType.UNIT,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=radius),
                ],
                output_key="wthau_victims",
            ),
            ForEachStep(
                list_key="wthau_victims",
                item_key="wthau_victim",
                steps_template=[ForceDiscardOrDefeatStep(victim_key="wthau_victim")],
            ),
        ]


# =============================================================================
# IMBUE DOUBT FAMILY (blue cards)
# "Name a color. Next turn, after playing cards:
#  An enemy hero in radius discards a card of that color, if able."
# =============================================================================


@register_effect("imbue_doubt")
class ImbueDoubtEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 3
        return [
            ChooseCardColorStep(output_key="chosen_color", prompt="Imbue Doubt: Name a card color"),
            CreateEffectStep(
                effect_type=EffectType.AFTER_CARDS_PLAYED_TRIGGER,
                scope=EffectScope(shape=Shape.GLOBAL, origin_id=str(hero.id)),
                duration=DurationType.NEXT_TURN,
                is_active=True,
                named_color_key="chosen_color",
                finishing_steps=[
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select an enemy hero in radius to discard",
                        output_key="doubt_victim",
                        filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=radius),
                            ImmunityFilter(),
                        ],
                    ),
                    ForceDiscardByColorStep(
                        victim_key="doubt_victim",
                        color_key="chosen_color",
                        active_if_key="doubt_victim",
                    ),
                ],
            ),
        ]


@register_effect("time_to_reconsider")
class TimeToReconsiderEffect(ImbueDoubtEffect):
    pass


@register_effect("an_illusion_of_choice")
class AnIllusionOfChoiceEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        radius = stats.radius or 4
        return [
            ChooseCardColorStep(
                output_key="chosen_color", prompt="An Illusion of Choice: Name a card color"
            ),
            CreateEffectStep(
                effect_type=EffectType.AFTER_CARDS_PLAYED_TRIGGER,
                scope=EffectScope(shape=Shape.GLOBAL, origin_id=str(hero.id)),
                duration=DurationType.NEXT_TURN,
                is_active=True,
                named_color_key="chosen_color",
                finishing_steps=[
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select first enemy hero in radius to discard",
                        output_key="choice_victim_1",
                        is_mandatory=False,
                        filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=radius),
                        ],
                    ),
                    ForceDiscardByColorStep(
                        victim_key="choice_victim_1",
                        color_key="chosen_color",
                        active_if_key="choice_victim_1",
                    ),
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select second enemy hero in radius to discard",
                        output_key="choice_victim_2",
                        is_mandatory=False,
                        active_if_key="choice_victim_1",
                        filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=radius),
                            ExcludeIdentityFilter(exclude_keys=["choice_victim_1"]),
                        ],
                    ),
                    ForceDiscardByColorStep(
                        victim_key="choice_victim_2",
                        color_key="chosen_color",
                        active_if_key="choice_victim_2",
                    ),
                ],
            ),
        ]


# =============================================================================
# FLEETING IMAGE FAMILY (green placement & swap cards)
# "Place up to N Illusion tokens in radius.
#  You may swap with an Illusion token in play."
# =============================================================================


@register_effect("fleeting_image")
class FleetingImageEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        if card.id == "fleeting_image":
            n = 1
        elif card.id == "multiple_projections":
            n = 2
        else:
            n = 3

        radius = stats.radius or 2
        steps: list[GameStep] = []

        for i in range(n):
            hex_key = f"fi_place_hex_{i}"
            steps.append(
                SelectStep(
                    target_type=TargetType.HEX,
                    prompt=f"Place Illusion token {i+1}/{n} in radius",
                    output_key=hex_key,
                    is_mandatory=False,
                    filters=[
                        RangeFilter(max_range=radius),
                        ObstacleFilter(is_obstacle=False),
                    ],
                    active_if_key=f"fi_place_hex_{i-1}" if i > 0 else None,
                )
            )
            steps.append(
                PlaceTokenStep(
                    token_type=TokenType.ILLUSION,
                    hex_key=hex_key,
                    active_if_key=hex_key,
                )
            )

        steps.append(
            SelectStep(
                target_type=TargetType.UNIT_OR_TOKEN,
                prompt="Select an Illusion token in play to swap with",
                output_key="fi_swap_token",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="TOKEN"),
                    TokenTypeFilter(token_type=TokenType.ILLUSION),
                ],
            )
        )
        steps.append(
            SwapUnitsStep(
                unit_a_id=str(hero.id),
                unit_b_key="fi_swap_token",
                active_if_key="fi_swap_token",
            )
        )
        return steps


@register_effect("multiple_projections")
class MultipleProjectionsEffect(FleetingImageEffect):
    pass


@register_effect("master_of_illusions")
class MasterOfIllusionsEffect(FleetingImageEffect):
    pass


# =============================================================================
# ILLUSIONARY FORCE FAMILY (green equivalence cards)
# "Place up to N Illusion tokens in radius.
#  This round: While you are performing actions, all Illusion tokens count
#  as both tokens and friendly melee minions."
# =============================================================================


@register_effect("illusionary_force")
class IllusionaryForceEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        n = 2 if card.id == "illusionary_force" else 3

        radius = stats.radius or 4
        steps: list[GameStep] = []

        for i in range(n):
            hex_key = f"if_place_hex_{i}"
            steps.append(
                SelectStep(
                    target_type=TargetType.HEX,
                    prompt=f"Place Illusion token {i+1}/{n} in radius",
                    output_key=hex_key,
                    is_mandatory=False,
                    filters=[
                        RangeFilter(max_range=radius),
                        ObstacleFilter(is_obstacle=False),
                    ],
                    active_if_key=f"if_place_hex_{i-1}" if i > 0 else None,
                )
            )
            steps.append(
                PlaceTokenStep(
                    token_type=TokenType.ILLUSION,
                    hex_key=hex_key,
                    active_if_key=hex_key,
                )
            )

        steps.append(
            CreateEffectStep(
                effect_type=EffectType.ILLUSION_MINION_EQUIVALENCE,
                scope=EffectScope(shape=Shape.GLOBAL, origin_id=str(hero.id)),
                duration=DurationType.THIS_ROUND,
                is_active=True,
            )
        )
        return steps


@register_effect("illusionary_army")
class IllusionaryArmyEffect(IllusionaryForceEffect):
    pass


# =============================================================================
# PHANTASMAL FAMILY (red cards)
# =============================================================================


@register_effect("phantasmal_sentry")
class PhantasmalSentryEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an adjacent unit or qualifying hero in range",
                output_key="ps_target",
                is_mandatory=True,
                filters=[
                    TeamFilter(relation="ENEMY"),
                    OrFilter(
                        filters=[
                            RangeFilter(max_range=1),
                            AndFilter(
                                filters=[
                                    UnitTypeFilter(unit_type="HERO"),
                                    RangeFilter(max_range=stats.range or 0),
                                    AdjacentToTokenFilter(
                                        token_type=TokenType.ILLUSION,
                                        max_range=stats.range or 0,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
            ),
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range or 0,
                is_ranged=True,
                target_id_key="ps_target",
            ),
        ]


@register_effect("phantasmal_warrior")
class PhantasmalWarriorEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        n_spaces = 1 if card.id == "phantasmal_warrior" else 2

        return [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Phantasmal Warrior — choose one",
                output_key="pw_choice",
                number_options=[1, 2],
                number_labels={
                    1: f"Move Illusion token up to {n_spaces} spaces to target hero",
                    2: "Target a unit adjacent to you",
                },
                is_mandatory=True,
            ),
            CheckContextConditionStep(
                input_key="pw_choice", operator="==", threshold=1, output_key="pw_bullet1"
            ),
            SelectStep(
                target_type=TargetType.UNIT_OR_TOKEN,
                prompt="Select an Illusion token in range to move",
                output_key="pw_token",
                is_mandatory=True,
                active_if_key="pw_bullet1",
                filters=[
                    UnitTypeFilter(unit_type="TOKEN"),
                    TokenTypeFilter(token_type=TokenType.ILLUSION),
                    RangeFilter(max_range=stats.range or 0),
                ],
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select destination hex adjacent to a target hero",
                output_key="pw_dest",
                is_mandatory=True,
                active_if_key="pw_token",
                filters=[
                    ObstacleFilter(is_obstacle=False, exclude_id_key="pw_token"),
                    MovementPathFilter(range_val=n_spaces, unit_key="pw_token"),
                    CountMatchFilter(
                        min_count=1,
                        sub_filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            TeamFilter(relation="ENEMY"),
                            ImmunityFilter(),
                            RangeFilter(max_range=1, origin_hex_key="_cmf_origin_hex"),
                            RangeFilter(max_range=stats.range or 0, origin_id=hero.id),
                        ],
                    ),
                ],
            ),
            MoveUnitStep(
                unit_key="pw_token",
                destination_key="pw_dest",
                range_val=n_spaces,
                is_movement_action=False,
                active_if_key="pw_dest",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy hero in range to target",
                output_key="pw_hero",
                is_mandatory=True,
                active_if_key="pw_dest",
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="ENEMY"),
                    ImmunityFilter(),
                    RangeFilter(max_range=stats.range or 0),
                    RangeFilter(max_range=1, origin_hex_key="pw_dest"),
                ],
            ),
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range or 0,
                is_ranged=True,
                target_id_key="pw_hero",
                active_if_key="pw_hero",
            ),
            CheckContextConditionStep(
                input_key="pw_choice", operator="==", threshold=2, output_key="pw_bullet2"
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select adjacent unit to target",
                output_key="ps_target_2",
                is_mandatory=True,
                active_if_key="pw_bullet2",
                filters=[
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=1),
                    ImmunityFilter(),
                ],
            ),
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                is_ranged=True,
                target_id_key="ps_target_2",
                active_if_key="ps_target_2",
            ),
        ]


@register_effect("phantasmal_champion")
class PhantasmalChampionEffect(PhantasmalWarriorEffect):
    pass


# =============================================================================
# TWIST FATE FAMILY (red swap cards)
# =============================================================================


@register_effect("twist_fate")
class TwistFateEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        is_twist_fate = card.id == "twist_fate"

        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range or 0,
                is_ranged=True,
                target_output_key="tf_victim",
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select an enemy unit in range to swap",
                output_key="tf_swap_enemy",
                is_mandatory=False,
                active_if_key="tf_victim",
                filters=[
                    TeamFilter(relation="ENEMY"),
                    RangeFilter(max_range=stats.range or 0),
                ],
            ),
            SelectStep(
                target_type=TargetType.UNIT_OR_TOKEN,
                prompt="Select an Illusion token to swap with",
                output_key="tf_swap_token",
                is_mandatory=False,
                active_if_key="tf_swap_enemy",
                filters=[
                    UnitTypeFilter(unit_type="TOKEN"),
                    TokenTypeFilter(token_type=TokenType.ILLUSION),
                    RangeFilter(max_range=1 if is_twist_fate else (stats.range or 0)),
                ],
            ),
            SwapUnitsStep(
                unit_a_key="tf_swap_enemy",
                unit_b_key="tf_swap_token",
                active_if_key="tf_swap_token",
            ),
        ]


@register_effect("devious_scheme")
class DeviousSchemeEffect(TwistFateEffect):
    pass
