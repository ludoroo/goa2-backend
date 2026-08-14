"""Razzle card effects for the multi-piece hero infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goa2.domain.models import CardContainerType, CardState, TargetType
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.filters import (
    CountMatchFilter,
    ExcludeIdentityFilter,
    FilterCondition,
    HeroPieceFilter,
    ImmunityFilter,
    MovementPathFilter,
    ObstacleFilter,
    RangeFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AddContextValueStep,
    AttackSequenceStep,
    CheckContextConditionStep,
    CountStep,
    GameStep,
    MoveSequenceStep,
    MoveUnitStep,
    RazzleMirroredPushStep,
    RemoveHeroPieceStep,
    RetrieveCardStep,
    SelectStep,
    SetActingPieceStep,
    SpawnHeroPieceStep,
    SwapUnitsStep,
)
from goa2.engine.topology import topology_distance

if TYPE_CHECKING:
    from goa2.domain.models import Card, Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


def _acting_piece(state: GameState, hero: Hero) -> str:
    return state.resolve_board_actor(str(hero.id))


def _has_twin_strike(hero: Hero) -> bool:
    ultimate = hero.ultimate_card
    return bool(
        hero.level >= 8
        and ultimate is not None
        and ultimate.state == CardState.PASSIVE
        and ultimate.current_effect_id == "twin_strike"
    )


def _another_piece_filters(
    extra_filters: list[FilterCondition] | None = None,
) -> list[FilterCondition]:
    return [
        HeroPieceFilter(exclude_acting=True),
        *(extra_filters or []),
    ]


def _move_another_piece_steps(
    *,
    distance: int,
    prefix: str,
) -> list[GameStep]:
    piece_key = f"{prefix}_piece"
    dest_key = f"{prefix}_dest"
    return [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select another one of you to move",
            output_key=piece_key,
            is_mandatory=False,
            filters=_another_piece_filters(),
        ),
        SelectStep(
            target_type=TargetType.HEX,
            prompt=f"Select where to move that piece (up to {distance})",
            output_key=dest_key,
            is_mandatory=False,
            active_if_key=piece_key,
            filters=[
                RangeFilter(max_range=distance, origin_key=piece_key),
                MovementPathFilter(range_val=distance, unit_key=piece_key),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_key=piece_key,
            destination_key=dest_key,
            range_val=distance,
            is_movement_action=False,
            active_if_key=dest_key,
        ),
    ]


def _swap_friendly_hero_steps(
    state: GameState,
    hero: Hero,
    stats: CardStats,
    *,
    move_distance: int,
    prefix: str,
) -> list[GameStep]:
    actor_piece = _acting_piece(state, hero)
    swap_key = f"{prefix}_swap_target"
    return [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select a friendly hero in range to swap with",
            output_key=swap_key,
            is_mandatory=True,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="FRIENDLY"),
                RangeFilter(max_range=stats.range),
            ],
        ),
        SwapUnitsStep(unit_a_id=actor_piece, unit_b_key=swap_key),
        *_move_another_piece_steps(distance=move_distance, prefix=f"{prefix}_move"),
    ]


def _movement_then_move_another_steps(
    state: GameState,
    hero: Hero,
    stats: CardStats,
    *,
    move_distance: int,
    prefix: str,
) -> list[GameStep]:
    return [
        MoveSequenceStep(unit_id=str(hero.id), range_val=stats.primary_value),
        *_move_another_piece_steps(
            distance=move_distance,
            prefix=f"{prefix}_move",
        ),
    ]


def _minion_target_filters(
    *,
    range_val: int,
    origin_key: str | None = None,
    exclude_keys: list[str] | None = None,
) -> list[FilterCondition]:
    return [
        UnitTypeFilter(unit_type="MINION"),
        RangeFilter(max_range=range_val, origin_key=origin_key),
        ExcludeIdentityFilter(exclude_self=False, exclude_keys=exclude_keys or []),
    ]


def _piece_with_minion_target_filter(
    *,
    range_val: int,
    exclude_keys: list[str],
) -> CountMatchFilter:
    return CountMatchFilter(
        min_count=1,
        sub_filters=[
            UnitTypeFilter(unit_type="MINION"),
            RangeFilter(
                max_range=range_val,
                origin_hex_key=CountMatchFilter.ORIGIN_HEX_KEY,
            ),
            ExcludeIdentityFilter(exclude_self=False, exclude_keys=exclude_keys),
            ImmunityFilter(),
        ],
    )


def _swap_minion_steps(
    state: GameState,
    hero: Hero,
    stats: CardStats,
    *,
    prefix: str,
    repeat_once: bool = False,
) -> list[GameStep]:
    actor_piece = _acting_piece(state, hero)
    minion_key = f"{prefix}_minion"
    steps: list[GameStep] = [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select a minion in range to swap with",
            output_key=minion_key,
            is_mandatory=True,
            filters=_minion_target_filters(range_val=stats.range),
        ),
        SwapUnitsStep(unit_a_id=actor_piece, unit_b_key=minion_key),
    ]

    if not repeat_once:
        return steps

    repeater_key = f"{prefix}_repeater"
    repeat_minion_key = f"{prefix}_repeat_minion"
    steps.extend(
        [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select another one of you to repeat the swap",
                output_key=repeater_key,
                is_mandatory=False,
                filters=_another_piece_filters(
                    [
                        _piece_with_minion_target_filter(
                            range_val=stats.range,
                            exclude_keys=[minion_key],
                        )
                    ]
                ),
            ),
            SetActingPieceStep(
                hero_id=str(hero.id),
                piece_key=repeater_key,
                active_if_key=repeater_key,
            ),
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a different minion in range",
                output_key=repeat_minion_key,
                is_mandatory=True,
                active_if_key=repeater_key,
                filters=_minion_target_filters(
                    range_val=stats.range,
                    origin_key=repeater_key,
                    exclude_keys=[minion_key],
                ),
            ),
            SwapUnitsStep(
                unit_a_key=repeater_key,
                unit_b_key=repeat_minion_key,
                active_if_key=repeat_minion_key,
            ),
        ]
    )
    return steps


def _retrieve_card_steps(*, prefix: str, active_if_key: str | None = None) -> list[GameStep]:
    card_key = f"{prefix}_card"
    return [
        SelectStep(
            target_type=TargetType.CARD,
            prompt="Select a discarded card to retrieve",
            output_key=card_key,
            card_container=CardContainerType.DISCARD,
            is_mandatory=False,
            active_if_key=active_if_key,
        ),
        RetrieveCardStep(card_key=card_key, active_if_key=card_key),
    ]


def _other_pieces_in_radius_steps(
    *,
    radius: int,
    count_key: str,
) -> list[GameStep]:
    return [
        CountStep(
            target_type=TargetType.UNIT,
            output_key=count_key,
            filters=[
                HeroPieceFilter(exclude_acting=True),
                RangeFilter(max_range=radius),
            ],
        )
    ]


def _twin_strike_repeat_steps(
    hero: Hero,
    stats: CardStats,
) -> list[GameStep]:
    repeater_key = "twin_strike_repeater"
    repeat_target_key = "twin_strike_victim"
    first_target_key = "victim_id"
    return [
        SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select another one of you to repeat Stunt Doubles",
            output_key=repeater_key,
            is_mandatory=False,
            filters=_another_piece_filters(
                [
                    CountMatchFilter(
                        min_count=1,
                        sub_filters=[
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(
                                max_range=1,
                                origin_hex_key=CountMatchFilter.ORIGIN_HEX_KEY,
                            ),
                            ExcludeIdentityFilter(
                                exclude_self=False,
                                exclude_keys=[first_target_key],
                            ),
                            ImmunityFilter(),
                        ],
                    )
                ]
            ),
        ),
        SetActingPieceStep(
            hero_id=str(hero.id),
            piece_key=repeater_key,
            active_if_key=repeater_key,
        ),
        AttackSequenceStep(
            damage=stats.primary_value,
            range_val=1,
            target_output_key=repeat_target_key,
            target_filters=[
                ExcludeIdentityFilter(exclude_self=False, exclude_keys=[first_target_key])
            ],
            active_if_key=repeater_key,
        ),
        SpawnHeroPieceStep(
            hero_id=str(hero.id),
            max_count=3,
            radius=stats.radius or 1,
            active_if_key=repeater_key,
        ),
    ]


@register_effect("twin_strike")
class TwinStrikeEffect(CardEffect):
    """Registered for completeness; Stunt Doubles owns the approved repeat block."""


@register_effect("alleyoop")
class AlleyoopEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_friendly_hero_steps(
            state,
            hero,
            stats,
            move_distance=1,
            prefix="alleyoop",
        )


@register_effect("group_performance")
class GroupPerformanceEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_friendly_hero_steps(
            state,
            hero,
            stats,
            move_distance=2,
            prefix="group_performance",
        )


@register_effect("team_spirit")
class TeamSpiritEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_friendly_hero_steps(
            state,
            hero,
            stats,
            move_distance=3,
            prefix="team_spirit",
        )


@register_effect("magic_trick")
class MagicTrickEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [RazzleMirroredPushStep(hero_id=str(hero.id), max_distance=2)]


@register_effect("aaaand_its_gone")
class AaaandItsGoneEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [RazzleMirroredPushStep(hero_id=str(hero.id), max_distance=3)]


@register_effect("tightrope")
class TightropeEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _movement_then_move_another_steps(
            state,
            hero,
            stats,
            move_distance=1,
            prefix="tightrope",
        )


@register_effect("high_wire")
class HighWireEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _movement_then_move_another_steps(
            state,
            hero,
            stats,
            move_distance=2,
            prefix="high_wire",
        )


@register_effect("wire_dancers")
class WireDancersEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _movement_then_move_another_steps(
            state,
            hero,
            stats,
            move_distance=3,
            prefix="wire_dancers",
        )


@register_effect("theatrics")
class TheatricsEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_minion_steps(state, hero, stats, prefix="theatrics")


@register_effect("spectacle")
class SpectacleEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return _swap_minion_steps(
            state,
            hero,
            stats,
            prefix="spectacle",
            repeat_once=True,
        )


@register_effect("stunt_doubles")
class StuntDoublesEffect(CardEffect):
    """Target adjacent unit. After the attack, spawn up to 3 more of you."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            SpawnHeroPieceStep(
                hero_id=str(hero.id),
                max_count=3,
                radius=stats.radius or 1,
            ),
        ]
        if _has_twin_strike(hero):
            steps.extend(_twin_strike_repeat_steps(hero, stats))
        return steps


@register_effect("phantom_strike")
class PhantomStrikeEffect(CardEffect):
    """Target adjacent unit. After attack, you may remove one of you if possible."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            RemoveHeroPieceStep(hero_id=str(hero.id), mode="choose_one", min_remaining=1),
        ]


@register_effect("hit_and_gone")
class HitAndGoneEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            RemoveHeroPieceStep(hero_id=str(hero.id), mode="choose_any", min_remaining=1),
        ]


@register_effect("into_thin_air")
class IntoThinAirEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            RemoveHeroPieceStep(hero_id=str(hero.id), mode="choose_any", min_remaining=0),
        ]


@register_effect("rummage")
class RummageEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        count_key = "rummage_piece_count"
        can_retrieve_key = "rummage_can_retrieve"
        return [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            *_other_pieces_in_radius_steps(radius=stats.radius or 3, count_key=count_key),
            CheckContextConditionStep(
                input_key=count_key,
                threshold=1,
                output_key=can_retrieve_key,
            ),
            *_retrieve_card_steps(prefix="rummage", active_if_key=can_retrieve_key),
        ]


@register_effect("ransack")
class RansackEffect(CardEffect):
    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        count_key = "ransack_piece_count"
        steps: list[GameStep] = [
            AttackSequenceStep(damage=stats.primary_value, range_val=1),
            *_other_pieces_in_radius_steps(radius=stats.radius or 3, count_key=count_key),
        ]
        for index in range(1, min(hero.piece_supply, 4)):
            can_retrieve_key = f"ransack_can_retrieve_{index}"
            steps.extend(
                [
                    CheckContextConditionStep(
                        input_key=count_key,
                        threshold=index,
                        output_key=can_retrieve_key,
                    ),
                    *_retrieve_card_steps(
                        prefix=f"ransack_{index}",
                        active_if_key=can_retrieve_key,
                    ),
                ]
            )
        return steps


@register_effect("crowd_control")
class CrowdControlEffect(CardEffect):
    """Skill removes other pieces; defense gains +2 per other piece in radius."""

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        return [RemoveHeroPieceStep(hero_id=str(hero.id), mode="all_others")]

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict,
    ) -> list[GameStep] | None:
        defender_piece = str(context.get("defender_id", defender.id))
        origin = state.get_position(defender_piece)
        if origin is None:
            return []

        radius = stats.radius or card.radius_value or 3
        bonus = 0
        for pid in state.get_piece_ids(str(defender.id)):
            if pid == defender_piece:
                continue
            loc = state.get_position(pid)
            if loc is not None and topology_distance(origin, loc, state) <= radius:
                bonus += 2

        if bonus <= 0:
            return []
        return [AddContextValueStep(key="defense_bonus", amount=bonus)]
