"""Tests for Silverarrow PR2 card effects (Families 2, 3, 4, 7).

Covered:
- Family 2 RED Max-Range Snipe: long_shot, rain_of_arrows
- Family 3 BLUE Root Zones: grappling_branches, entangling_vines, grasping_roots
- Family 4 BLUE End-of-Turn Sentinel: treetop_sentinel, warning_shot
- Family 7 GOLD Max-Range + Escape: shoot_and_scoot
"""

import pytest

import goa2.data.heroes.silverarrow
import goa2.scripts.silverarrow_effects  # noqa: F401 — registers effects
from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActionType,
    DurationType,
    EffectType,
    Team,
    TeamColor,
)
from goa2.domain.models.effect import AffectsFilter, Shape
from goa2.domain.state import GameState
from goa2.engine.effects import CardEffectRegistry
from goa2.engine.filters import (
    ExcludeIdentityFilter,
    RangeFilter,
    UnitTypeFilter,
)
from goa2.engine.steps import (
    AttackSequenceStep,
    CheckContextConditionStep,
    CheckUnitTypeStep,
    CreateEffectStep,
    FastTravelSequenceStep,
    ForceDiscardOrDefeatStep,
    ForceDiscardStep,
    SelectStep,
)


def _card_by_id(card_id: str):
    hero = HeroRegistry.get("Silverarrow")
    card = next((c for c in hero.deck if c.id == card_id), None)
    assert card is not None, f"Silverarrow has no card {card_id}"
    return card


@pytest.fixture
def silver_state():
    board = Board()
    hexes = set()
    for q in range(-5, 6):
        for r in range(-5, 6):
            s = -q - r
            if abs(s) <= 5:
                hexes.add(Hex(q=q, r=r, s=s))
    z1 = Zone(id="z1", hexes=hexes, neighbors=[])
    board.zones = {"z1": z1}
    board.populate_tiles_from_zones()

    silver = HeroRegistry.get("Silverarrow")
    silver.team = TeamColor.RED
    silver.hand = []

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[silver], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
    )
    state.place_entity("hero_silverarrow", Hex(q=0, r=0, s=0))
    state.current_actor_id = "hero_silverarrow"
    return state


# =============================================================================
# Family 2 — Max-Range Snipe
# =============================================================================


class TestLongShot:
    def test_registered(self):
        assert CardEffectRegistry.get("long_shot") is not None

    def test_steps_attack_at_max_range(self, silver_state):
        effect = CardEffectRegistry.get("long_shot")
        hero = silver_state.get_hero("hero_silverarrow")
        card = _card_by_id("long_shot")

        steps = effect.get_steps(silver_state, hero, card)
        assert len(steps) == 1
        atk = steps[0]
        assert isinstance(atk, AttackSequenceStep)
        assert atk.is_ranged is True
        assert atk.range_val == card.range_value  # 3
        # target_filters should contain a RangeFilter pinning min==max==range
        rfs = [f for f in atk.target_filters if isinstance(f, RangeFilter)]
        assert len(rfs) == 1
        assert rfs[0].min_range == card.range_value
        assert rfs[0].max_range == card.range_value


class TestRainOfArrows:
    def test_registered(self):
        assert CardEffectRegistry.get("rain_of_arrows") is not None

    def test_step_structure(self, silver_state):
        effect = CardEffectRegistry.get("rain_of_arrows")
        hero = silver_state.get_hero("hero_silverarrow")
        card = _card_by_id("rain_of_arrows")

        steps = effect.get_steps(silver_state, hero, card)
        classes = [type(s) for s in steps]
        assert classes == [
            AttackSequenceStep,  # first shot, stores rain_victim_1
            CheckUnitTypeStep,  # rain_victim_1 HERO? -> bool
            CheckContextConditionStep,  # bool -> rain_first_was_hero True/None
            SelectStep,  # pick second hero (mandatory if gate active)
            AttackSequenceStep,  # second shot
            SelectStep,  # optional minion pick
            AttackSequenceStep,  # third shot
        ]

        first_attack: AttackSequenceStep = steps[0]
        assert first_attack.target_id_key is None
        assert first_attack.target_output_key == "rain_victim_1"
        assert first_attack.is_ranged is True

        check_type: CheckUnitTypeStep = steps[1]
        assert check_type.unit_key == "rain_victim_1"
        assert check_type.expected_type == "HERO"

        gate: CheckContextConditionStep = steps[2]
        assert gate.output_key == "rain_first_was_hero"

        second_select: SelectStep = steps[3]
        assert second_select.active_if_key == "rain_first_was_hero"
        assert second_select.is_mandatory is True
        assert any(
            isinstance(f, UnitTypeFilter) and f.unit_type == "HERO" for f in second_select.filters
        )
        assert any(
            isinstance(f, ExcludeIdentityFilter) and "rain_victim_1" in f.exclude_keys
            for f in second_select.filters
        )

        second_attack: AttackSequenceStep = steps[4]
        assert second_attack.target_id_key == "rain_victim_2"
        assert second_attack.active_if_key == "rain_victim_2"

        third_select: SelectStep = steps[5]
        assert third_select.active_if_key == "rain_victim_2"
        assert third_select.is_mandatory is False
        assert any(
            isinstance(f, UnitTypeFilter) and f.unit_type == "MINION" for f in third_select.filters
        )
        assert any(
            isinstance(f, ExcludeIdentityFilter)
            and set(f.exclude_keys) == {"rain_victim_1", "rain_victim_2"}
            for f in third_select.filters
        )

        third_attack: AttackSequenceStep = steps[6]
        assert third_attack.target_id_key == "rain_victim_3"
        assert third_attack.active_if_key == "rain_victim_3"


# =============================================================================
# Family 3 — Root Zones
# =============================================================================


@pytest.mark.parametrize(
    "effect_id,expected_radius",
    [
        ("grasping_roots", 2),
        ("entangling_vines", 3),
        ("grappling_branches", 4),
    ],
)
class TestRootZones:
    def test_registered(self, effect_id, expected_radius):
        assert CardEffectRegistry.get(effect_id) is not None

    def test_creates_movement_zone(self, silver_state, effect_id, expected_radius):
        effect = CardEffectRegistry.get(effect_id)
        hero = silver_state.get_hero("hero_silverarrow")
        card = _card_by_id(effect_id)

        steps = effect.get_steps(silver_state, hero, card)
        assert len(steps) == 1
        create = steps[0]
        assert isinstance(create, CreateEffectStep)
        assert create.effect_type == EffectType.MOVEMENT_ZONE
        assert create.scope.shape == Shape.RADIUS
        assert create.scope.range == expected_radius
        assert create.scope.affects == AffectsFilter.ENEMY_HEROES
        assert create.duration == DurationType.THIS_TURN
        assert create.max_value == 1
        assert create.limit_actions_only is True
        assert ActionType.FAST_TRAVEL in create.restrictions


# =============================================================================
# Family 4 — End-of-Turn Sentinel
# =============================================================================


class TestSentinel:
    @pytest.mark.parametrize(
        "effect_id,defeat_on_fail",
        [
            ("warning_shot", False),
            ("treetop_sentinel", True),
        ],
    )
    def test_schedules_delayed_trigger(self, silver_state, effect_id, defeat_on_fail):
        effect = CardEffectRegistry.get(effect_id)
        hero = silver_state.get_hero("hero_silverarrow")
        card = _card_by_id(effect_id)

        steps = effect.get_steps(silver_state, hero, card)
        assert len(steps) == 1
        create = steps[0]
        assert isinstance(create, CreateEffectStep)
        assert create.effect_type == EffectType.DELAYED_TRIGGER
        assert create.duration == DurationType.THIS_TURN
        assert create.is_active is True

        # finishing_steps: [SelectStep (hero in radius), ForceDiscard(OrDefeat)]
        assert len(create.finishing_steps) == 2
        select_step, discard_step = create.finishing_steps
        assert isinstance(select_step, SelectStep)
        # "An enemy hero in radius discards…, if able" — victim choice is optional.
        assert select_step.is_mandatory is False
        assert select_step.output_key == "sentinel_victim"
        assert any(
            isinstance(f, UnitTypeFilter) and f.unit_type == "HERO" for f in select_step.filters
        )
        # Radius anchored at Silverarrow's id, resolved at resolve time
        rfs = [f for f in select_step.filters if isinstance(f, RangeFilter)]
        assert len(rfs) == 1
        assert rfs[0].origin_id == "hero_silverarrow"
        assert rfs[0].max_range == card.radius_value

        if defeat_on_fail:
            assert isinstance(discard_step, ForceDiscardOrDefeatStep)
        else:
            assert isinstance(discard_step, ForceDiscardStep)
        assert discard_step.victim_key == "sentinel_victim"


# =============================================================================
# Family 7 — Shoot and Scoot
# =============================================================================


class TestShootAndScoot:
    def test_registered(self):
        assert CardEffectRegistry.get("shoot_and_scoot") is not None

    def test_attack_then_fast_travel(self, silver_state):
        effect = CardEffectRegistry.get("shoot_and_scoot")
        hero = silver_state.get_hero("hero_silverarrow")
        card = _card_by_id("shoot_and_scoot")

        steps = effect.get_steps(silver_state, hero, card)
        assert len(steps) == 2

        atk = steps[0]
        assert isinstance(atk, AttackSequenceStep)
        assert atk.is_ranged is True
        assert atk.range_val == card.range_value  # 2
        rfs = [f for f in atk.target_filters if isinstance(f, RangeFilter)]
        assert rfs and rfs[0].min_range == rfs[0].max_range == card.range_value

        scoot = steps[1]
        assert isinstance(scoot, FastTravelSequenceStep)
        assert scoot.unit_id == "hero_silverarrow"
        # Audit §5.1: "adjacent zone" excludes Silverarrow's current zone.
        assert scoot.require_zone_change is True
