"""Snorri — rune hero card effects.

Core mechanic: ``inscribe_the_runes`` places 4 rune markers (one per turn
slot) via ``PlaceRunesStep`` (see ``engine/steps/markers.py``). A rune is
*active* while it sits below the slot matching ``state.turn``. Every other
Snorri card keys off :func:`active_runes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from goa2.domain.models import (
    ActionType,
    AffectsFilter,
    Card,
    CardColor,
    CardContainerType,
    CardState,
    DurationType,
    EffectScope,
    EffectType,
    RuneType,
    Shape,
    TargetType,
)
from goa2.domain.models.enums import PassiveTrigger
from goa2.engine.effects import CardEffect, PassiveConfig, register_effect
from goa2.engine.filters_hex import MovementPathFilter, ObstacleFilter, RangeFilter
from goa2.engine.filters_units import (
    ContextIdsFilter,
    ExcludeIdentityFilter,
    TeamFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckUnitTypeStep,
    ChooseRuneStep,
    CountStep,
    CreateEffectStep,
    ForceDiscardByColorStep,
    ForceDiscardOrDefeatStep,
    GainCoinsStep,
    GameStep,
    MayRepeatOnceStep,
    MoveSequenceStep,
    MoveUnitStep,
    PlaceRunesStep,
    RetrieveCardStep,
    SelectStep,
    SetContextFlagStep,
    SnapshotAdjacentHeroesStep,
    SwapCardStep,
    SwapItemCardStep,
)

if TYPE_CHECKING:
    from goa2.domain.models import Hero
    from goa2.domain.state import GameState
    from goa2.engine.stats import CardStats


def _find_card_owner(state: GameState, card: Card) -> Hero | None:
    """Locate the hero who owns ``card`` — scans every card container,
    including ``ultimate_card``, on every hero (both teams).

    Deliberately NOT ``state.current_actor_id``: a copied Snorri card
    (e.g. NebKher's Mind Grip) is performed by another hero but must still
    see Snorri's runes (TDD §19).
    """
    for team in state.teams.values():
        for hero in team.heroes:
            if hero.current_turn_card is not None and hero.current_turn_card.id == card.id:
                return hero
            if hero.extra_turn_card is not None and hero.extra_turn_card.id == card.id:
                return hero
            for container in (hero.hand, hero.deck, hero.discard_pile, hero.played_cards):
                if any(c is not None and c.id == card.id for c in container):
                    return hero
            if hero.ultimate_card is not None and hero.ultimate_card.id == card.id:
                return hero
    return None


def active_runes(state: GameState, card: Card, context: dict[str, Any]) -> set[RuneType]:
    """The set of currently-active runes for ``card``.

    Base activity: the rune below the turn slot matching ``state.turn``,
    read from the CARD OWNER's ``rune_slots`` (not the current actor — see
    :func:`_find_card_owner`). Unioned with any rune named by
    ``context["snorri_ult_rune_action"]`` / ``context["snorri_ult_rune_defense"]``
    (Rune Mastery ultimate, wired up in a later task) — read defensively
    since most callers never set them.
    """
    active: set[RuneType] = set()

    owner = _find_card_owner(state, card)
    if owner is not None:
        turn_rune = owner.rune_slots.get(state.turn)
        if turn_rune is not None:
            active.add(RuneType(turn_rune))

    for key in ("snorri_ult_rune_action", "snorri_ult_rune_defense"):
        raw = context.get(key)
        if raw:
            active.add(RuneType(raw))

    return active


@register_effect("inscribe_the_runes")
class InscribeTheRunesEffect(CardEffect):
    def build_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        stats: CardStats,
    ) -> list[GameStep]:
        return [PlaceRunesStep(hero_id=hero.id)]


class OathEffect(CardEffect):
    """Shared rune-gated defense implementation for Snorri's Oath cards."""

    rune_blocks: ClassVar[dict[RuneType, str]] = {}
    choose_one: ClassVar[bool] = False
    _choice_key: ClassVar[str] = "snorri_oath_rune"
    _matches_key: ClassVar[str] = "snorri_oath_rune_matches"

    @staticmethod
    def _blocks(block_type: str, context: dict[str, Any]) -> bool:
        return {
            "basic": bool(context.get("attack_is_basic")),
            "melee": not bool(context.get("attack_is_ranged")),
            "ranged": bool(context.get("attack_is_ranged")),
            "non_basic": not bool(context.get("attack_is_basic")),
        }[block_type]

    def build_defense_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        active = [rune for rune in RuneType if rune in active_runes(state, card, context)]
        eligible = [rune for rune in active if rune in self.rune_blocks]
        matching = [rune for rune in eligible if self._blocks(self.rune_blocks[rune], context)]

        # Reset both combat flags explicitly: an attack context must never
        # inherit a prior defense result.
        steps: list[GameStep] = [
            SetContextFlagStep(key="auto_block", value=False),
            SetContextFlagStep(key="defense_invalid", value=True),
        ]

        if not eligible:
            return steps

        if not self.choose_one or len(eligible) == 1:
            if matching:
                steps.extend(
                    [
                        SetContextFlagStep(key="auto_block", value=True),
                        SetContextFlagStep(key="defense_invalid", value=False),
                    ]
                )
            return steps

        steps.extend(
            [
                ChooseRuneStep(
                    output_key=self._choice_key,
                    options=eligible,
                    prompt="Choose one active rune",
                    matching_options=matching,
                    matches_output_key=self._matches_key,
                ),
                SetContextFlagStep(key="auto_block", value=True, active_if_key=self._matches_key),
                SetContextFlagStep(
                    key="defense_invalid", value=False, active_if_key=self._matches_key
                ),
            ]
        )
        return steps

    def build_on_block_steps(
        self,
        state: GameState,
        defender: Hero,
        card: Card,
        stats: CardStats,
        context: dict[str, Any],
    ) -> list[GameStep]:
        return [
            CreateEffectStep(
                effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                scope=EffectScope(
                    shape=Shape.POINT, origin_id=defender.id, affects=AffectsFilter.SELF
                ),
                duration=DurationType.THIS_TURN,
                is_active=True,
            )
        ]


@register_effect("oath_of_endurance")
class OathOfEnduranceEffect(OathEffect):
    rune_blocks: ClassVar[dict[RuneType, str]] = {
        RuneType.HORN: "basic",
        RuneType.AXE: "melee",
    }


@register_effect("oath_of_fortitude")
class OathOfFortitudeEffect(OathOfEnduranceEffect):
    rune_blocks: ClassVar[dict[RuneType, str]] = {
        **OathOfEnduranceEffect.rune_blocks,
        RuneType.BIRD: "ranged",
    }


@register_effect("oath_of_perseverance")
class OathOfPerseveranceEffect(OathOfFortitudeEffect):
    rune_blocks: ClassVar[dict[RuneType, str]] = {
        **OathOfFortitudeEffect.rune_blocks,
        RuneType.ANVIL: "non_basic",
    }
    choose_one: ClassVar[bool] = True


def _optional_move_one_space(hero_id: str, *, output_key: str) -> list[GameStep]:
    """Return Snorri's optional one-space, effect-side move."""
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt="You may move 1 space",
            output_key=output_key,
            is_mandatory=False,
            filters=[
                RangeFilter(max_range=1),
                MovementPathFilter(range_val=1, unit_id=hero_id),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_id=hero_id,
            destination_key=output_key,
            range_val=1,
            active_if_key=output_key,
        ),
    ]


class RunicMeleeEffect(CardEffect):
    """Shared implementation for Runic Dagger, Hammer, and Battleaxe."""

    has_pre_move: ClassVar[bool] = False
    has_repeat: ClassVar[bool] = False

    def _sequence(
        self,
        hero: Hero,
        stats: CardStats,
        active: set[RuneType],
        *,
        repeat_leg: bool,
    ) -> list[GameStep]:
        steps: list[GameStep] = []
        if self.has_pre_move and RuneType.HORN in active:
            steps.extend(
                _optional_move_one_space(
                    hero.id,
                    output_key=(
                        "runic_melee_repeat_move_hex" if repeat_leg else "runic_melee_move_hex"
                    ),
                )
            )

        target_filters = (
            [UnitTypeFilter(unit_type="MINION"), TeamFilter(relation="ENEMY")] if repeat_leg else []
        )
        steps.append(
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=1,
                target_filters=target_filters,
            )
        )

        if RuneType.ANVIL in active:
            retrieve_key = "runic_melee_repeat_retrieve" if repeat_leg else "runic_melee_retrieve"
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.CARD,
                        card_container=CardContainerType.DISCARD,
                        prompt="You may retrieve a discarded card",
                        output_key=retrieve_key,
                        is_mandatory=False,
                    ),
                    RetrieveCardStep(card_key=retrieve_key, active_if_key=retrieve_key),
                ]
            )
        return steps

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        steps = self._sequence(hero, stats, active, repeat_leg=False)

        if self.has_repeat and RuneType.AXE in active:
            repeat_target_filters = [
                UnitTypeFilter(unit_type="MINION"),
                TeamFilter(relation="ENEMY"),
            ]
            # With Horn, the repeat is a full sequence beginning with an
            # optional move. That move still resolves if the later mandatory
            # attack cannot be completed, so any enemy minion makes the repeat
            # available. Without Horn, a minion must already be adjacent.
            if RuneType.HORN not in active:
                repeat_target_filters.append(RangeFilter(max_range=1))
            steps.extend(
                [
                    CountStep(
                        target_type=TargetType.UNIT,
                        output_key="runic_battleaxe_repeat_targets",
                        filters=repeat_target_filters,
                    ),
                    CheckContextConditionStep(
                        input_key="runic_battleaxe_repeat_targets",
                        operator=">=",
                        threshold=1,
                        output_key="runic_battleaxe_can_repeat",
                    ),
                    MayRepeatOnceStep(
                        active_if_key="runic_battleaxe_can_repeat",
                        prompt="Repeat the attack on an enemy minion?",
                        steps_template=self._sequence(hero, stats, active, repeat_leg=True),
                    ),
                ]
            )
        return steps


@register_effect("rune_mastery")
class RuneMasteryEffect(CardEffect):
    """Adds one inactive placed rune to each Snorri action or defense."""

    def get_passive_configs(self) -> list[PassiveConfig]:
        return [
            PassiveConfig(
                trigger=PassiveTrigger.BEFORE_ACTION, uses_per_turn=-1, is_optional=False
            ),
            PassiveConfig(
                trigger=PassiveTrigger.BEFORE_DEFENSE, uses_per_turn=-1, is_optional=False
            ),
        ]

    def should_offer_passive(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> bool:
        # Primary-defense cards also pass through the generic BEFORE_ACTION
        # hook. Defense reactions have their own BEFORE_DEFENSE hook so Rune
        # Mastery must not prompt twice for the same defense.
        return bool(hero.rune_slots) and not (
            trigger == PassiveTrigger.BEFORE_ACTION
            and context.get("current_action_type") == ActionType.DEFENSE
        )

    def get_passive_steps(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        trigger: PassiveTrigger,
        context: dict[str, Any],
    ) -> list[GameStep]:
        active_turn_rune = hero.rune_slots.get(state.turn)
        options = [
            rune
            for rune in RuneType
            if rune in hero.rune_slots.values() and rune != active_turn_rune
        ]
        if not options:
            return []
        output_key = (
            "snorri_ult_rune_defense"
            if trigger == PassiveTrigger.BEFORE_DEFENSE
            else "snorri_ult_rune_action"
        )
        return [
            ChooseRuneStep(
                output_key=output_key,
                options=options,
                prompt="Rune Mastery: choose a second active rune",
            )
        ]


@register_effect("runic_dagger")
class RunicDaggerEffect(RunicMeleeEffect):
    pass


@register_effect("runic_hammer")
class RunicHammerEffect(RunicMeleeEffect):
    has_pre_move: ClassVar[bool] = True


@register_effect("runic_battleaxe")
class RunicBattleaxeEffect(RunicHammerEffect):
    has_repeat: ClassVar[bool] = True


def _optional_move(hero_id: str, *, distance: int, output_key: str) -> list[GameStep]:
    return [
        SelectStep(
            target_type=TargetType.HEX,
            prompt=f"You may move up to {distance} spaces",
            output_key=output_key,
            is_mandatory=False,
            filters=[
                RangeFilter(max_range=distance),
                MovementPathFilter(range_val=distance, unit_id=hero_id),
                ObstacleFilter(is_obstacle=False),
            ],
        ),
        MoveUnitStep(
            unit_id=hero_id,
            destination_key=output_key,
            range_val=distance,
            active_if_key=output_key,
        ),
    ]


class RunicRangedEffect(CardEffect):
    """Shared implementation for Runecaster and Runeblaster."""

    bird_allows_full_range: ClassVar[bool] = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        range_value = stats.range or 0
        target_filters = [
            TeamFilter(relation="ENEMY"),
            RangeFilter(
                max_range=range_value,
                min_range=(
                    None if self.bird_allows_full_range and RuneType.BIRD in active else range_value
                ),
            ),
        ]
        steps: list[GameStep] = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt=(
                    "Select target at maximum range"
                    if not (self.bird_allows_full_range and RuneType.BIRD in active)
                    else "Select target in range"
                ),
                output_key="target_id",
                filters=target_filters,
            ),
            SnapshotAdjacentHeroesStep(output_key="rc_adjacent"),
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=range_value,
                is_ranged=True,
                target_id_key="target_id",
            ),
        ]

        if RuneType.HORN in active:
            steps.extend(_optional_move(hero.id, distance=2, output_key="runic_ranged_move_hex"))

        if RuneType.AXE in active:
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.UNIT,
                        prompt="Select an enemy hero adjacent to the target",
                        output_key="rc_discard_victim",
                        is_mandatory=False,
                        filters=[ContextIdsFilter(ids_key="rc_adjacent")],
                    ),
                    ForceDiscardOrDefeatStep(
                        victim_key="rc_discard_victim",
                        active_if_key="rc_discard_victim",
                    ),
                ]
            )
        return steps


@register_effect("runecaster")
class RunecasterEffect(RunicRangedEffect):
    pass


@register_effect("runeblaster")
class RuneblasterEffect(RunicRangedEffect):
    bird_allows_full_range: ClassVar[bool] = True


class RuneDiscardEffect(CardEffect):
    """Shared rune-to-colour discard logic for Runetrap and Runebomb."""

    rune_colors: ClassVar[dict[RuneType, CardColor]] = {}
    choose_one: ClassVar[bool] = False
    _rune_order: ClassVar[tuple[RuneType, ...]] = (
        RuneType.HORN,
        RuneType.AXE,
        RuneType.ANVIL,
        RuneType.BIRD,
    )

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        eligible = [
            rune for rune in self._rune_order if rune in active and rune in self.rune_colors
        ]
        if not eligible:
            return []

        target_step = SelectStep(
            target_type=TargetType.UNIT,
            prompt="Select an enemy hero in radius",
            output_key="rt_victim",
            is_mandatory=False,
            filters=[
                UnitTypeFilter(unit_type="HERO"),
                TeamFilter(relation="ENEMY"),
                RangeFilter(max_range=stats.radius or 0),
            ],
        )

        if self.choose_one:
            color_key = "rb_discard_color"
            return [
                ChooseRuneStep(
                    output_key="rb_rune",
                    options=eligible,
                    prompt="Choose one active rune",
                    value_map={rune.value: self.rune_colors[rune].value for rune in eligible},
                    value_output_key=color_key,
                ),
                target_step,
                ForceDiscardByColorStep(
                    victim_key="rt_victim",
                    color_key=color_key,
                    active_if_key="rt_victim",
                ),
            ]

        return [
            target_step,
            *[
                ForceDiscardByColorStep(
                    victim_key="rt_victim", color=self.rune_colors[rune], active_if_key="rt_victim"
                )
                for rune in eligible
            ],
        ]


@register_effect("runetrap")
class RunetrapEffect(RuneDiscardEffect):
    rune_colors: ClassVar[dict[RuneType, CardColor]] = {
        RuneType.HORN: CardColor.GREEN,
        RuneType.AXE: CardColor.SILVER,
        RuneType.ANVIL: CardColor.BLUE,
    }


@register_effect("runebomb")
class RunebombEffect(RuneDiscardEffect):
    rune_colors: ClassVar[dict[RuneType, CardColor]] = {
        **RunetrapEffect.rune_colors,
        RuneType.BIRD: CardColor.GOLD,
    }
    choose_one: ClassVar[bool] = True


class PassageEffect(CardEffect):
    """Shared movement and rune riders for Snorri's Passage cards."""

    has_anvil_immunity: ClassVar[bool] = False
    has_horn_bonus: ClassVar[bool] = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        movement = (stats.primary_value or 0) + (
            2 if self.has_horn_bonus and RuneType.HORN in active else 0
        )
        steps: list[GameStep] = [
            MoveSequenceStep(
                range_val=movement,
                pass_through_obstacles=RuneType.BIRD in active,
            )
        ]
        if self.has_anvil_immunity and RuneType.ANVIL in active:
            steps.append(
                CreateEffectStep(
                    effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
                    scope=EffectScope(
                        shape=Shape.POINT, origin_id=hero.id, affects=AffectsFilter.SELF
                    ),
                    duration=DurationType.THIS_TURN,
                    is_active=True,
                )
            )
        return steps


@register_effect("safe_passage")
class SafePassageEffect(PassageEffect):
    pass


@register_effect("hidden_passage")
class HiddenPassageEffect(PassageEffect):
    has_anvil_immunity: ClassVar[bool] = True


@register_effect("deep_passage")
class DeepPassageEffect(HiddenPassageEffect):
    has_horn_bonus: ClassVar[bool] = True


class AncestralEffect(CardEffect):
    """Shared friendly-hero rune benefits for Ancestral Boon and Grace."""

    has_bird_swap: ClassVar[bool] = False

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        applicable = {RuneType.AXE, RuneType.ANVIL}
        if self.has_bird_swap:
            applicable.add(RuneType.BIRD)
        if not active & applicable:
            return []

        steps: list[GameStep] = [
            SelectStep(
                target_type=TargetType.UNIT,
                prompt="Select a friendly hero in radius",
                output_key="ab_hero",
                is_mandatory=False,
                filters=[
                    UnitTypeFilter(unit_type="HERO"),
                    TeamFilter(relation="FRIENDLY"),
                    RangeFilter(max_range=stats.radius or 0),
                ],
            )
        ]
        if RuneType.AXE in active:
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.CARD,
                        prompt="Choose a resolved card to swap",
                        output_key="ab_resolved_card",
                        context_hero_id_key="ab_hero",
                        override_player_id_key="ab_hero",
                        card_container=CardContainerType.PLAYED,
                        card_states=[CardState.RESOLVED],
                        is_mandatory=False,
                        active_if_key="ab_hero",
                    ),
                    SelectStep(
                        target_type=TargetType.CARD,
                        prompt="Choose a card from hand to swap",
                        output_key="ab_hand_card",
                        context_hero_id_key="ab_hero",
                        override_player_id_key="ab_hero",
                        card_container=CardContainerType.HAND,
                        is_mandatory=False,
                        active_if_key="ab_resolved_card",
                    ),
                    SwapCardStep(
                        source_card_key="ab_resolved_card",
                        target_card_key="ab_hand_card",
                        context_hero_id_key="ab_hero",
                        active_if_key="ab_hand_card",
                    ),
                ]
            )
        if RuneType.ANVIL in active:
            steps.append(
                RetrieveCardStep(
                    hero_key="ab_hero",
                    retrieve_all_discarded=True,
                    active_if_key="ab_hero",
                )
            )
        if self.has_bird_swap and RuneType.BIRD in active:
            steps.extend(
                [
                    SelectStep(
                        target_type=TargetType.CARD,
                        prompt="Choose an item card to replace",
                        output_key="ab_item_card",
                        context_hero_id_key="ab_hero",
                        override_player_id_key="ab_hero",
                        card_container=CardContainerType.DECK,
                        card_states=[CardState.ITEM],
                        selected_card_color_key="ab_item_color",
                        selected_card_tier_key="ab_item_tier",
                        is_mandatory=False,
                        active_if_key="ab_hero",
                    ),
                    SelectStep(
                        target_type=TargetType.CARD,
                        prompt="Choose a same-tier, same-color card",
                        output_key="ab_item_target",
                        context_hero_id_key="ab_hero",
                        override_player_id_key="ab_hero",
                        card_container=CardContainerType.DECK,
                        card_color_key="ab_item_color",
                        card_tier_key="ab_item_tier",
                        card_has_item=True,
                        exclude_card_states=[CardState.ITEM],
                        is_mandatory=False,
                        active_if_key="ab_item_card",
                    ),
                    SwapItemCardStep(
                        hero_key="ab_hero",
                        item_card_key="ab_item_card",
                        target_card_key="ab_item_target",
                        active_if_key="ab_item_target",
                    ),
                ]
            )
        return steps


@register_effect("ancestral_boon")
class AncestralBoonEffect(AncestralEffect):
    pass


@register_effect("ancestral_grace")
class AncestralGraceEffect(AncestralEffect):
    has_bird_swap: ClassVar[bool] = True


@register_effect("rune_sigils")
class RuneSigilsEffect(CardEffect):
    """Gold ranged attack with rune-gated targeting, damage, coins, and repeat."""

    def _attack_steps(
        self,
        *,
        target_key: str,
        hero_type_key: str,
        damage: int,
        range_value: int,
        actor_key: str,
        anvil_active: bool,
        target_output_key: str | None = None,
        target_filters: list[Any] | None = None,
    ) -> list[GameStep]:
        steps: list[GameStep] = [
            AttackSequenceStep(
                damage=damage,
                range_val=range_value,
                is_ranged=True,
                target_id_key=target_key,
                target_output_key=target_output_key,
                target_filters=target_filters or [],
            ),
            CheckUnitTypeStep(
                unit_key=target_key,
                expected_type="HERO",
                output_key=hero_type_key,
            ),
        ]
        if anvil_active:
            steps.append(GainCoinsStep(hero_key=actor_key, amount=3, active_if_key=hero_type_key))
        return steps

    def build_steps(
        self, state: GameState, hero: Hero, card: Card, stats: CardStats
    ) -> list[GameStep]:
        active = active_runes(state, card, state.execution_context)
        damage = (stats.primary_value or 0) + (3 if RuneType.AXE in active else 0)
        range_value = stats.range or 0
        anvil_active = RuneType.ANVIL in active
        steps: list[GameStep] = [SetContextFlagStep(key="rs_actor", value=hero.id)]

        if RuneType.BIRD in active:
            steps.append(
                SelectStep(
                    target_type=TargetType.UNIT,
                    prompt="You may target an enemy minion in range instead",
                    output_key="rs_target",
                    is_mandatory=False,
                    filters=[
                        UnitTypeFilter(unit_type="MINION"),
                        TeamFilter(relation="ENEMY"),
                        RangeFilter(max_range=range_value),
                    ],
                )
            )

        steps.extend(
            self._attack_steps(
                target_key="rs_target",
                hero_type_key="rs_target_is_hero",
                damage=damage,
                range_value=range_value,
                actor_key="rs_actor",
                anvil_active=anvil_active,
                target_output_key="rs_target",
                target_filters=[RangeFilter(max_range=1)],
            )
        )
        if RuneType.HORN in active:
            steps.extend(
                [
                    CountStep(
                        target_type=TargetType.UNIT,
                        output_key="rs_repeat_targets",
                        skip_immunity_filter=False,
                        filters=[
                            UnitTypeFilter(unit_type="HERO"),
                            TeamFilter(relation="ENEMY"),
                            RangeFilter(max_range=range_value),
                            ExcludeIdentityFilter(exclude_self=False, exclude_keys=["rs_target"]),
                        ],
                    ),
                    CheckContextConditionStep(
                        input_key="rs_repeat_targets",
                        operator=">=",
                        threshold=1,
                        output_key="rs_can_repeat",
                    ),
                    MayRepeatOnceStep(
                        active_if_key="rs_can_repeat",
                        prompt="Repeat the attack on a different enemy hero?",
                        steps_template=[
                            SelectStep(
                                target_type=TargetType.UNIT,
                                prompt="Select a different enemy hero in range",
                                output_key="rs_repeat_target",
                                filters=[
                                    UnitTypeFilter(unit_type="HERO"),
                                    TeamFilter(relation="ENEMY"),
                                    RangeFilter(max_range=range_value),
                                    ExcludeIdentityFilter(
                                        exclude_self=False, exclude_keys=["rs_target"]
                                    ),
                                ],
                            ),
                            *self._attack_steps(
                                target_key="rs_repeat_target",
                                hero_type_key="rs_repeat_target_is_hero",
                                damage=damage,
                                range_value=range_value,
                                actor_key="rs_actor",
                                anvil_active=anvil_active,
                            ),
                        ],
                    ),
                ]
            )
        return steps
