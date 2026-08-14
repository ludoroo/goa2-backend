"""Widget card effects.

Remaining useful precedents:
- Min smoke-bomb effects: single-token placement, selection, and swaps.
- Mortimer zombie effects: optional token movement and repeated choice gates.
- Bain/Brogan/Arien effects: straight-line targeting and discard-or-defeat flows.
- Ursafar effects: selecting a card and using PerformPrimaryActionStep.
- Misa/Silverarrow/Tigerclaw effects: computed or related follow-up movement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goa2.domain.models.enums import (
    ActionType,
    CardContainerType,
    PassiveTrigger,
    TargetType,
    TokenType,
)
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters_composite import OrFilter
from goa2.engine.filters_geometry import CoMoverValidHexFilter, InStraightLineFilter
from goa2.engine.filters_hex import MovementPathFilter, ObstacleFilter, RangeFilter
from goa2.engine.filters_units import (
    AdjacencyToContextFilter,
    CanBeMovedByActorFilter,
    TeamFilter,
    TokenTypeFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CoDirectionalDragStep,
    DefeatUnitStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    GameStep,
    MoveTokenStep,
    MoveUnitStep,
    PerformPrimaryActionStep,
    PlaceTokenStep,
    RemoveTokenStep,
    RemoveUnitStep,
    SelectStep,
    SetContextFlagStep,
    SwapUnitsStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


def _pyro_selection_step(output_key: str, range_val: int, *, is_mandatory: bool) -> SelectStep:
    return SelectStep(
        target_type=TargetType.UNIT_OR_TOKEN,
        prompt="Select Pyro",
        output_key=output_key,
        is_mandatory=is_mandatory,
        auto_select_if_one=True,
        filters=[
            UnitTypeFilter(unit_type="TOKEN"),
            TokenTypeFilter(token_type=TokenType.PYRO),
            RangeFilter(max_range=range_val),
        ],
    )


def _move_pyro_steps(
    *,
    range_val: int,
    move_distance: int,
    pyro_key: str = "pyro_id",
    destination_key: str = "pyro_dest",
    is_mandatory: bool = False,
    active_if_key: str | None = None,
) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.UNIT_OR_TOKEN,
            prompt="Select Pyro to move",
            output_key=pyro_key,
            is_mandatory=is_mandatory,
            active_if_key=active_if_key,
            auto_select_if_one=True,
            filters=[
                UnitTypeFilter(unit_type="TOKEN"),
                TokenTypeFilter(token_type=TokenType.PYRO),
                RangeFilter(max_range=range_val),
            ],
        ),
        SelectStep(
            target_type=TargetType.HEX,
            prompt=f"Select Pyro destination (up to {move_distance})",
            output_key=destination_key,
            is_mandatory=False,
            active_if_key=pyro_key,
            filters=[
                ObstacleFilter(is_obstacle=False),
                MovementPathFilter(range_val=move_distance, unit_key=pyro_key),
            ],
        ),
        MoveTokenStep(
            token_key=pyro_key,
            destination_key=destination_key,
            range_val=move_distance,
            active_if_key=destination_key,
        ),
    ]


def _move_selected_pyro_steps(
    *,
    pyro_key: str,
    destination_key: str,
    move_distance: int,
    active_if_key: str | None,
) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt=f"Select Pyro destination (up to {move_distance})",
            output_key=destination_key,
            is_mandatory=False,
            active_if_key=active_if_key,
            filters=[
                ObstacleFilter(is_obstacle=False),
                MovementPathFilter(range_val=move_distance, unit_key=pyro_key),
            ],
        ),
        MoveTokenStep(
            token_key=pyro_key,
            destination_key=destination_key,
            range_val=move_distance,
            active_if_key=destination_key,
        ),
    ]


def _move_widget_steps(
    *,
    hero_id: str,
    destination_key: str,
    active_if_key: str | None,
) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Select Widget destination (up to 2)",
            output_key=destination_key,
            active_if_key=active_if_key,
            is_mandatory=False,
            filters=[
                ObstacleFilter(is_obstacle=False),
                MovementPathFilter(range_val=2, unit_id=hero_id),
            ],
        ),
        MoveUnitStep(
            unit_id=hero_id,
            destination_key=destination_key,
            range_val=2,
            active_if_key=destination_key,
        ),
    ]


def _swap_with_pyro_steps(
    selection_distance: int,
    *,
    is_mandatory: bool = True,
    include_friendly_heroes: bool = True,
) -> list[GameStep]:
    target_filters = [
        UnitTypeFilter(unit_type="HERO"),
        RangeFilter(max_range=selection_distance),
    ]
    if include_friendly_heroes:
        target_filters.append(
            OrFilter(
                filters=[
                    TeamFilter(relation="SELF"),
                    TeamFilter(relation="FRIENDLY"),
                ]
            )
        )
    else:
        target_filters.append(TeamFilter(relation="SELF"))

    return [
        _pyro_selection_step("pyro_swap_id", selection_distance, is_mandatory=is_mandatory),
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select yourself or a friendly hero to swap with Pyro",
            output_key="pyro_swap_target",
            is_mandatory=is_mandatory,
            active_if_key="pyro_swap_id",
            skip_self_filter=True,
            filters=target_filters,
        ),
        SwapUnitsStep(
            unit_a_key="pyro_swap_id",
            unit_b_key="pyro_swap_target",
            active_if_key="pyro_swap_target",
            is_mandatory=is_mandatory,
        ),
    ]


def _diversionary_steps(damage: int, pyro_move: int) -> list[GameStep]:
    return [
        AttackSequenceStep(damage=damage, range_val=1),
        *_move_pyro_steps(range_val=99, move_distance=pyro_move, is_mandatory=False),
    ]


def _airborne_steps(damage: int, *, radius: int, after_swap: bool) -> list[GameStep]:
    steps: list[GameStep] = [
        *_swap_with_pyro_steps(radius, is_mandatory=False, include_friendly_heroes=False),
        AttackSequenceStep(damage=damage, range_val=1),
    ]
    if after_swap:
        steps.extend(
            _swap_with_pyro_steps(radius, is_mandatory=False, include_friendly_heroes=False)
        )
    return steps


def _pyro_adjacent_enemy_minion_removal_steps(
    *, hero_id: str, range_val: int, defeat: bool
) -> list[GameStep]:
    remove_step: GameStep
    if defeat:
        remove_step = DefeatUnitStep(victim_key="pyro_adjacent_minion", killer_id=hero_id)
    else:
        remove_step = RemoveUnitStep(unit_key="pyro_adjacent_minion")

    return [
        _pyro_selection_step("pyro_skill_id", 99, is_mandatory=True),
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an enemy minion adjacent to Pyro",
            output_key="pyro_adjacent_minion",
            active_if_key="pyro_skill_id",
            is_mandatory=True,
            filters=[
                UnitTypeFilter(unit_type="MINION"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=range_val),
                AdjacencyToContextFilter(target_key="pyro_skill_id"),
            ],
        ),
        remove_step,
        RemoveTokenStep(token_key="pyro_skill_id"),
    ]


def _drag_steps(*, ignore_obstacles: bool) -> list[GameStep]:
    return [
        _pyro_selection_step("drag_pyro_id", 99, is_mandatory=True),
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an enemy unit adjacent to Pyro",
            output_key="drag_enemy_id",
            active_if_key="drag_pyro_id",
            is_mandatory=True,
            filters=[
                TeamFilter(relation="ENEMY"),
                AdjacencyToContextFilter(target_key="drag_pyro_id"),
                CanBeMovedByActorFilter(),
            ],
        ),
        SelectStep(
            target_type=TargetType.HEX,
            prompt="Select Pyro destination (2 or 3 spaces in a straight line)",
            output_key="drag_pyro_dest",
            active_if_key="drag_enemy_id",
            is_mandatory=True,
            filters=[
                RangeFilter(min_range=2, max_range=3, origin_key="drag_pyro_id"),
                InStraightLineFilter(origin_key="drag_pyro_id"),
                CoMoverValidHexFilter(
                    anchor_key="drag_pyro_id",
                    partner_key="drag_enemy_id",
                    ignore_path_obstacles=ignore_obstacles,
                ),
            ],
        ),
        CoDirectionalDragStep(
            anchor_key="drag_pyro_id",
            partner_key="drag_enemy_id",
            anchor_dest_key="drag_pyro_dest",
            ignore_path_obstacles=ignore_obstacles,
        ),
    ]


def _breath_steps(*, range_val: int, defeat_if_unable: bool) -> list[GameStep]:
    discard_step: GameStep
    if defeat_if_unable:
        discard_step = ForceDiscardOrDefeatStep(victim_key="breath_target")
    else:
        discard_step = ForceDiscardStep(victim_key="breath_target")

    return [
        _pyro_selection_step("pyro_breath_id", 99, is_mandatory=True),
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an enemy hero in range and in a straight line from Pyro",
            output_key="breath_target",
            active_if_key="pyro_breath_id",
            is_mandatory=True,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=range_val, origin_key="pyro_breath_id"),
                InStraightLineFilter(origin_key="pyro_breath_id"),
            ],
        ),
        discard_step,
    ]


@register_effect("dragon_bond")
class DragonBondEffect(CardEffect):
    """Choose to place Pyro, or move Widget and Pyro if Pyro is already in play."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            SetContextFlagStep(key="widget_id", value=str(hero.id)),
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose one",
                output_key="dragon_bond_choice",
                number_options=[1, 2],
                number_labels={
                    1: "Place Pyro into a space in radius",
                    2: "Move yourself and Pyro up to 2 spaces",
                },
                is_mandatory=True,
            ),
            CheckContextConditionStep(
                input_key="dragon_bond_choice",
                operator="==",
                threshold=1,
                output_key="dragon_bond_place",
            ),
            SelectStep(
                target_type=TargetType.HEX,
                prompt="Select a space in radius for Pyro",
                output_key="pyro_place_hex",
                active_if_key="dragon_bond_place",
                is_mandatory=True,
                filters=[
                    RangeFilter(max_range=stats.radius),
                    ObstacleFilter(is_obstacle=False),
                ],
            ),
            PlaceTokenStep(
                token_type=TokenType.PYRO,
                hex_key="pyro_place_hex",
                active_if_key="pyro_place_hex",
            ),
            CheckContextConditionStep(
                input_key="dragon_bond_choice",
                operator="==",
                threshold=2,
                output_key="dragon_bond_move",
            ),
            SelectStep(
                target_type=TargetType.UNIT_OR_TOKEN,
                prompt="Select Pyro in play",
                output_key="dragon_bond_pyro",
                active_if_key="dragon_bond_move",
                is_mandatory=True,
                auto_select_if_one=True,
                filters=[
                    UnitTypeFilter(unit_type="TOKEN"),
                    TokenTypeFilter(token_type=TokenType.PYRO),
                    RangeFilter(max_range=99),
                ],
            ),
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Choose who moves first",
                output_key="dragon_bond_order",
                active_if_key="dragon_bond_pyro",
                number_options=[1, 2],
                number_labels={
                    1: "Move Widget first",
                    2: "Move Pyro first",
                },
                is_mandatory=True,
            ),
            CheckContextConditionStep(
                input_key="dragon_bond_order",
                operator="==",
                threshold=1,
                output_key="dragon_bond_widget_first",
            ),
            *_move_widget_steps(
                hero_id=str(hero.id),
                destination_key="dragon_bond_widget_dest_first",
                active_if_key="dragon_bond_widget_first",
            ),
            *_move_selected_pyro_steps(
                pyro_key="dragon_bond_pyro",
                destination_key="dragon_bond_pyro_dest_second",
                move_distance=2,
                active_if_key="dragon_bond_widget_first",
            ),
            CheckContextConditionStep(
                input_key="dragon_bond_order",
                operator="==",
                threshold=2,
                output_key="dragon_bond_pyro_first",
            ),
            *_move_selected_pyro_steps(
                pyro_key="dragon_bond_pyro",
                destination_key="dragon_bond_pyro_dest_first",
                move_distance=2,
                active_if_key="dragon_bond_pyro_first",
            ),
            *_move_widget_steps(
                hero_id=str(hero.id),
                destination_key="dragon_bond_widget_dest_second",
                active_if_key="dragon_bond_pyro_first",
            ),
        ]


@register_effect("take_off")
class TakeOffEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_with_pyro_steps(stats.range)


@register_effect("all_aboard")
class AllAboardEffect(TakeOffEffect):
    pass


@register_effect("safe_landing")
class SafeLandingEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            *_move_pyro_steps(range_val=stats.range, move_distance=1, is_mandatory=False),
            *_swap_with_pyro_steps(stats.range),
        ]


@register_effect("diversionary_strike")
class DiversionaryStrikeEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _diversionary_steps(stats.primary_value, 2)


@register_effect("diversionary_attack")
class DiversionaryAttackEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _diversionary_steps(stats.primary_value, 3)


@register_effect("diversionary_assault")
class DiversionaryAssaultEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _diversionary_steps(stats.primary_value, 4)


def _performable_faceup_skill_card_ids(state: GameState, hero: Hero) -> list[str]:
    """Faceup resolved skill cards whose primary action the hero may perform now.

    Performing one is a Skill action on *that* card, so a prevention effect such
    as Arien's Spell Break (gold cards excepted) rules it out even when the card
    granting the re-performance is gold.
    """
    from goa2.engine.rules import can_perform_card_primary

    return [
        c.id
        for c in hero.played_cards
        if c is not None
        and not c.is_facedown
        and c.primary_action == ActionType.SKILL
        and can_perform_card_primary(state, str(hero.id), c)
    ]


@register_effect("fight_as_one")
class FightAsOneEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        faceup_skill_card_ids = _performable_faceup_skill_card_ids(state, hero)
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                target_output_key="fight_as_one_initial_target",
            ),
            SelectStep(
                target_type=TargetType.CARD,
                prompt="Select a resolved skill card to perform",
                output_key="fight_as_one_skill_card",
                card_container=CardContainerType.PLAYED,
                card_action_types=[ActionType.SKILL],
                allowed_card_ids=faceup_skill_card_ids,
                is_mandatory=False,
            ),
            PerformPrimaryActionStep(
                card_key="fight_as_one_skill_card",
                hero_id=str(hero.id),
                exclude_target_key="fight_as_one_initial_target",
                active_if_key="fight_as_one_skill_card",
            ),
        ]


@register_effect("airborne_attack")
class AirborneAttackEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        assert stats.radius is not None, "airborne_attack requires a radius"
        return _airborne_steps(stats.primary_value, radius=stats.radius, after_swap=False)


@register_effect("airborne_assault")
class AirborneAssaultEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        assert stats.radius is not None, "airborne_assault requires a radius"
        return _airborne_steps(stats.primary_value, radius=stats.radius, after_swap=True)


@register_effect("nibble")
class NibbleEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _pyro_adjacent_enemy_minion_removal_steps(
            hero_id=str(hero.id), range_val=stats.range, defeat=False
        )


@register_effect("gnaw")
class GnawEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _pyro_adjacent_enemy_minion_removal_steps(
            hero_id=str(hero.id), range_val=stats.range, defeat=True
        )


@register_effect("fiery_breath")
class FieryBreathEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _breath_steps(range_val=stats.range, defeat_if_unable=False)


@register_effect("flaming_breath")
class FlamingBreathEffect(FieryBreathEffect):
    pass


@register_effect("scorching_breath")
class ScorchingBreathEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _breath_steps(range_val=stats.range, defeat_if_unable=True)


@register_effect("drag_off")
class DragOffEffect(CardEffect):
    """Move Pyro and an adjacent enemy unit 2-3 spaces in the same direction."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _drag_steps(ignore_obstacles=False)


@register_effect("carry_away")
class CarryAwayEffect(CardEffect):
    """Drag Off, ignoring obstacles on either unit's path."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _drag_steps(ignore_obstacles=True)


@register_effect("dragon_knight")
class DragonKnightEffect(CardEffect):
    """Each time after you perform a movement action, you may perform the
    primary action on one of your faceup skill cards."""

    def get_passive_config(self) -> PassiveConfig | None:
        return PassiveConfig(
            trigger=PassiveTrigger.AFTER_MOVEMENT,
            uses_per_turn=0,  # unlimited
            is_optional=True,
            prompt="Dragon Knight: Perform the primary action of a faceup skill card?",
        )

    def should_offer_passive(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> bool:
        # Only offer when the player actually has a faceup skill card they can perform.
        return bool(_performable_faceup_skill_card_ids(state, hero))

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> list[GameStep]:
        if trigger != PassiveTrigger.AFTER_MOVEMENT:
            return []

        faceup_skill_card_ids = _performable_faceup_skill_card_ids(state, hero)
        if not faceup_skill_card_ids:
            return []

        return [
            SelectStep(
                target_type=TargetType.CARD,
                prompt="Select a faceup skill card to perform",
                output_key="dragon_knight_skill_card",
                card_container=CardContainerType.PLAYED,
                card_action_types=[ActionType.SKILL],
                allowed_card_ids=faceup_skill_card_ids,
                is_mandatory=True,
            ),
            PerformPrimaryActionStep(
                card_key="dragon_knight_skill_card",
                hero_id=str(hero.id),
                active_if_key="dragon_knight_skill_card",
            ),
        ]
