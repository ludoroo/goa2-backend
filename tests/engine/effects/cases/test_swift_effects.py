"""Effect flow tests for the hero Swift."""

import pytest

from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _hex_disk(radius: int) -> list[tuple[int, int, int]]:
    return [
        (q, r, -q - r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if abs(q + r) <= radius
    ]


def _option_set(run) -> set:
    """Set of selectable values from the current request (raw metadata or option id)."""
    assert run.latest_request is not None
    options = set()
    for option in run.latest_request.options:
        if hasattr(option, "metadata") and option.metadata and "raw" in option.metadata:
            options.add(option.metadata.get("raw"))
        elif hasattr(option, "id"):
            options.add(option.id)
        else:
            options.add(option)
    return options


# =============================================================================
# RED — Snipe: "Target a unit at maximum range, and in a straight line."
# =============================================================================


@pytest.mark.effect_flow
def test_snipe_can_target_unit_at_max_range_in_straight_line() -> None:
    # Snipe is a ranged attack (range 4). A minion exactly 4 hexes away in a
    # straight line is a legal target.
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero(
            "hero_swift",
            at=(0, 0, 0),
            current_card=hero_card("Swift", "snipe"),
        )
        .blue_minion("blue_minion", at=(4, 0, -4))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert "blue_minion" in _option_set(run)
    run.choose("blue_minion").finish()

    combat_events = [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]
    assert combat_events


@pytest.mark.effect_flow
def test_snipe_cannot_target_unit_closer_than_max_range() -> None:
    # A minion at range 2 (not maximum range) must not be a legal Snipe target.
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero(
            "hero_swift",
            at=(0, 0, 0),
            current_card=hero_card("Swift", "snipe"),
        )
        .blue_minion("blue_minion", at=(2, 0, -2))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK")
    # No legal target at maximum range -> the attack finds nothing to select.
    run.finish()
    combat_events = [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]
    assert not combat_events


# =============================================================================
# Prepared Shot / Killshot:
# "Target a unit in range, in a straight line, and not adjacent to you."
# =============================================================================


@pytest.mark.parametrize("card_id", ["prepared_shot", "killshot"])
@pytest.mark.effect_flow
def test_straight_line_shot_allows_in_range_non_adjacent(card_id: str) -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", card_id))
        .blue_minion("blue_minion", at=(3, 0, -3))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert "blue_minion" in _option_set(run)
    run.choose("blue_minion").finish()
    assert [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]


@pytest.mark.parametrize("card_id", ["prepared_shot", "killshot"])
@pytest.mark.effect_flow
def test_straight_line_shot_rejects_adjacent_target(card_id: str) -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", card_id))
        .blue_minion("blue_minion", at=(1, 0, -1))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK")
    run.finish()
    assert not [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]


# =============================================================================
# Shotgun: "Target a unit in range. Before the attack: An enemy hero adjacent
# to the target discards a card, if able."
# Super-Shotgun: "...discards a card, or is defeated."
# =============================================================================


def test_shotgun_pre_attack_discards_adjacent_enemy_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "shotgun"))
        .blue_minion("blue_minion", at=(2, 0, -2))
        .blue_hero("hero_arien", at=(3, 0, -3))
        .with_actor("hero_swift")
        .build()
    )
    arien = state.get_hero("hero_arien")
    arien.hand = [hero_card("Swift", "snipe")]

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    # Pick the attack target (the minion).
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_UNIT)
    # Pre-attack: the adjacent enemy hero is the only valid discard target.
    assert "hero_arien" in _option_set(run)
    run.choose("hero_arien").expect_input(InputRequestType.SELECT_CARD)
    # The victim chooses which card to discard.
    run.choose("snipe").finish()

    assert len(arien.hand) == 0  # discarded its only card
    assert [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]


def test_super_shotgun_defeats_adjacent_hero_with_no_cards() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "super_shotgun"))
        .blue_minion("blue_minion", at=(2, 0, -2))
        .blue_hero("hero_arien", at=(3, 0, -3))
        .with_actor("hero_swift")
        .build()
    )
    arien = state.get_hero("hero_arien")
    arien.hand = []  # no cards -> "or is defeated"

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("hero_arien").finish()

    defeats = [
        e
        for e in run.events
        if e.event_type == GameEventType.UNIT_DEFEATED and e.target_id == "hero_arien"
    ]
    assert defeats


# =============================================================================
# Jump family — "Place yourself into a space in a straight line in radius.
# Push [an enemy unit / up to two enemy units] adjacent to you up to N spaces."
# =============================================================================


def test_steam_jump_places_self_in_line_then_pushes_adjacent_enemy() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "steam_jump"))
        .blue_minion("blue_minion", at=(3, 0, -3))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 2, "r": 0, "s": -2}).expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).finish()

    assert state.entity_locations.get("hero_swift") == Hex(q=2, r=0, s=-2)
    # Pushed 1 space directly away from Swift's new position.
    assert state.entity_locations.get("blue_minion") == Hex(q=4, r=0, s=-4)


def test_assault_jump_pushes_adjacent_enemy_up_to_two() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "assault_jump"))
        .blue_minion("blue_minion", at=(3, 0, -3))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 2, "r": 0, "s": -2}).expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).finish()

    assert state.entity_locations.get("blue_minion") == Hex(q=5, r=0, s=-5)


def test_drop_trooper_pushes_up_to_two_adjacent_enemies() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "drop_trooper"))
        .blue_minion("minion_a", at=(2, 0, -2))
        .blue_minion("minion_b", at=(1, -1, 0))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    # Place adjacent to both minions, in a straight line from origin.
    run.choose({"q": 1, "r": 0, "s": -1}).expect_input(InputRequestType.SELECT_UNIT)
    # Multi-select both adjacent enemies (auto-finishes at max=2).
    run.choose("minion_a")
    run.expect_input(InputRequestType.SELECT_UNIT)
    run.choose("minion_b")
    # Each pushed in turn; choose distance 1 for both.
    run.expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1)
    run.expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).finish()

    assert state.entity_locations.get("minion_a") == Hex(q=3, r=0, s=-3)
    assert state.entity_locations.get("minion_b") == Hex(q=1, r=-2, s=1)


# =============================================================================
# Suppress / Pin Down: "An enemy hero in radius who is not adjacent to terrain
# discards a card, if able."
# Killing Ground: "...discards a card, or is defeated."
# =============================================================================


@pytest.mark.parametrize("card_id", ["suppress", "pin_down", "killing_ground"])
def test_suppress_family_forces_discard_on_clear_enemy_hero(card_id: str) -> None:
    # Radius-2 disk so the enemy hero can sit at a fully interior hex (1,0,-1):
    # the board edge counts as terrain, so an edge hero would be skipped.
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(2))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", card_id))
        .blue_hero("hero_arien", at=(1, 0, -1))
        .with_actor("hero_swift")
        .build()
    )
    arien = state.get_hero("hero_arien")
    arien.hand = [hero_card("Swift", "snipe")]

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert "hero_arien" in _option_set(run)
    run.choose("hero_arien").expect_input(InputRequestType.SELECT_CARD)
    run.choose("snipe").finish()

    assert len(arien.hand) == 0


@pytest.mark.parametrize("card_id", ["suppress", "pin_down", "killing_ground"])
def test_suppress_family_skips_enemy_hero_adjacent_to_terrain(card_id: str) -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(2))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", card_id))
        .blue_hero("hero_arien", at=(1, 0, -1))
        .with_actor("hero_swift")
        .build()
    )
    # Make an on-map terrain tile adjacent to the interior enemy hero -> not a legal target.
    state.board.tiles[Hex(q=2, r=0, s=-2)].is_terrain = True
    arien = state.get_hero("hero_arien")
    arien.hand = [hero_card("Swift", "snipe")]

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL")
    run.finish()

    assert len(arien.hand) == 1  # untouched


def test_killing_ground_defeats_clear_enemy_hero_with_no_cards() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(2))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "killing_ground"))
        .blue_hero("hero_arien", at=(1, 0, -1))
        .with_actor("hero_swift")
        .build()
    )
    state.get_hero("hero_arien").hand = []

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("hero_arien").finish()

    assert any(
        e.event_type == GameEventType.UNIT_DEFEATED and e.target_id == "hero_arien"
        for e in run.events
    )


# =============================================================================
# Bounce: "Move 2 spaces in a straight line, ignoring obstacles; if this card
# is already resolved as you perform this action, may repeat once."
# =============================================================================


def test_bounce_moves_in_straight_line_through_obstacle() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "bounce"))
        .with_actor("hero_swift")
        .build()
    )
    # Terrain in the path at (1,0,-1); Bounce ignores it.
    state.board.tiles[Hex(q=1, r=0, s=-1)].is_terrain = True

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 2, "r": 0, "s": -2}).finish()

    assert state.entity_locations.get("hero_swift") == Hex(q=2, r=0, s=-2)


def test_bounce_repeats_only_when_card_already_resolved() -> None:
    from goa2.domain.models import CardState
    from goa2.engine.effects import CardEffectRegistry
    from goa2.engine.stats import compute_card_stats
    from goa2.engine.steps import MayRepeatOnceStep

    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0))
        .with_actor("hero_swift")
        .build()
    )
    hero = state.get_hero("hero_swift")
    card = hero_card("Swift", "bounce")
    effect = CardEffectRegistry.get("bounce")
    stats = compute_card_stats(state, hero.id, card)

    card.state = CardState.UNRESOLVED
    fresh = effect.build_steps(state, hero, card, stats)
    assert not any(isinstance(s, MayRepeatOnceStep) for s in fresh)

    card.state = CardState.RESOLVED
    resolved = effect.build_steps(state, hero, card, stats)
    assert any(isinstance(s, MayRepeatOnceStep) for s in resolved)


# =============================================================================
# Reload!: "Choose one — • Perform the primary action of your rightmost
# resolved card. • Defeat a minion adjacent to you."
# =============================================================================


def test_reload_performs_rightmost_resolved_card() -> None:
    from goa2.domain.models import CardState

    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "reload"))
        .blue_minion("blue_minion", at=(4, 0, -4))
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")
    snipe = hero_card("Swift", "snipe")
    snipe.state = CardState.RESOLVED
    swift.played_cards = [snipe]  # rightmost resolved card

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # snipe's target
    run.choose("blue_minion").finish()

    assert [e for e in run.events if e.event_type == GameEventType.COMBAT_RESOLVED]


def test_reload_defeats_adjacent_minion() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "reload"))
        .blue_minion("blue_minion", at=(1, 0, -1))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_UNIT)  # defeat adjacent minion
    run.choose("blue_minion").finish()

    assert state.entity_locations.get("blue_minion") is None
    assert any(
        e.event_type == GameEventType.UNIT_DEFEATED and e.target_id == "blue_minion"
        for e in run.events
    )


# =============================================================================
# Delayed Jump / Mobile Scout: "End of turn: Place yourself into a space in
# radius not in a straight line from you. [Mobile Scout: You may then fast
# travel, if able.]"
# =============================================================================


def _delayed_jump_create_step(card_id: str):
    from goa2.engine.effects import CardEffectRegistry
    from goa2.engine.stats import compute_card_stats

    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0))
        .with_actor("hero_swift")
        .build()
    )
    hero = state.get_hero("hero_swift")
    card = hero_card("Swift", card_id)
    effect = CardEffectRegistry.get(card_id)
    stats = compute_card_stats(state, hero.id, card)
    steps = effect.build_steps(state, hero, card, stats)
    return steps


@pytest.mark.parametrize("card_id", ["delayed_jump", "mobile_scout"])
def test_delayed_jump_schedules_end_of_turn_placement(card_id: str) -> None:
    from goa2.domain.models import DurationType
    from goa2.domain.models.effect import EffectType
    from goa2.engine.filters_geometry import NotInStraightLineFilter
    from goa2.engine.steps import CreateEffectStep, PlaceUnitStep, SelectStep

    steps = _delayed_jump_create_step(card_id)
    assert len(steps) == 1
    create = steps[0]
    assert isinstance(create, CreateEffectStep)
    assert create.effect_type == EffectType.DELAYED_TRIGGER
    assert create.duration == DurationType.THIS_TURN

    fin = create.finishing_steps
    assert isinstance(fin[0], SelectStep)
    assert any(isinstance(f, NotInStraightLineFilter) for f in fin[0].filters)
    assert any(isinstance(s, PlaceUnitStep) for s in fin)


def test_mobile_scout_adds_fast_travel_to_end_of_turn() -> None:
    from goa2.engine.steps import FastTravelSequenceStep

    steps = _delayed_jump_create_step("mobile_scout")
    fin = steps[0].finishing_steps
    assert any(isinstance(s, FastTravelSequenceStep) for s in fin)

    delayed = _delayed_jump_create_step("delayed_jump")
    assert not any(isinstance(s, FastTravelSequenceStep) for s in delayed[0].finishing_steps)


def test_delayed_jump_finishing_steps_place_self_not_in_straight_line() -> None:
    # Execute the scheduled end-of-turn finishing steps directly (the turn-end
    # orchestration that fires them is the THIS_TURN DELAYED_TRIGGER machinery
    # shared with Treetop Sentinel). This proves the finishing steps themselves
    # place the hero correctly off the straight line.
    from goa2.engine.handler import process_stack, push_steps

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_swift", at=(0, 0, 0))
        .with_actor("hero_swift")
        .build()
    )
    fin = _delayed_jump_create_step("delayed_jump")[0].finishing_steps
    push_steps(state, [s.model_copy(deep=True) for s in fin])

    result = process_stack(state)
    assert result.input_request is not None
    assert result.input_request.request_type.value == "SELECT_HEX"
    # A straight-line hex like (2,0,-2) must NOT be offered; (1,1,-2) is.
    raws = {
        opt.metadata.get("raw")
        for opt in result.input_request.options
        if opt.metadata and "raw" in opt.metadata
    }
    assert Hex(q=2, r=0, s=-2) not in raws
    assert Hex(q=1, r=1, s=-2) in raws

    state.execution_stack[-1].pending_input = {"selection": {"q": 1, "r": 1, "s": -2}}
    process_stack(state)
    assert state.entity_locations.get("hero_swift") == Hex(q=1, r=1, s=-2)


# =============================================================================
# Mark for Death / Hunting Season: move enemy minion(s) + schedule next-turn
# minion-defeat coin bounty.
# =============================================================================


def test_mark_for_death_moves_minion_and_schedules_bounty() -> None:
    from goa2.domain.models import DurationType
    from goa2.domain.models.effect import EffectType

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "mark_for_death"))
        .blue_minion("blue_minion", at=(2, 0, -2))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_minion").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 1, "r": 1, "s": -2}).finish()

    assert state.entity_locations.get("blue_minion") == Hex(q=1, r=1, s=-2)
    bounties = [e for e in state.active_effects if e.effect_type == EffectType.MINION_DEFEAT_BOUNTY]
    assert len(bounties) == 1
    assert bounties[0].max_value == 1
    assert bounties[0].duration == DurationType.NEXT_TURN
    assert bounties[0].source_id == "hero_swift"


def test_hunting_season_schedules_bounty_for_two() -> None:
    from goa2.domain.models.effect import EffectType

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "hunting_season"))
        .with_actor("hero_swift")
        .build()
    )

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    # No minions in radius -> multi-select finds nothing, proceeds to schedule.
    run.choose("SKILL").finish()

    bounties = [e for e in state.active_effects if e.effect_type == EffectType.MINION_DEFEAT_BOUNTY]
    assert len(bounties) == 1
    assert bounties[0].max_value == 2


# =============================================================================
# Bullet Time (ultimate passive): "After you resolve a basic card, you may
# perform the primary action on that card; you cannot target the same enemy
# hero twice in the same turn this way."
# =============================================================================


def _bullet_time_setup():
    from goa2.data.heroes.registry import HeroRegistry
    from goa2.engine.effects import CardEffectRegistry

    state = (
        EffectScenarioBuilder()
        .line_board(length=7)
        .red_hero("hero_swift", at=(0, 0, 0))
        .with_actor("hero_swift")
        .build()
    )
    hero = state.get_hero("hero_swift")
    ult = HeroRegistry.get("Swift").ultimate_card
    effect = CardEffectRegistry.get("bullet_time")
    return state, hero, ult, effect


@pytest.mark.effect_contract
def test_bullet_time_registers_after_basic_action_passive() -> None:
    from goa2.domain.models.enums import PassiveTrigger

    _, _, _, effect = _bullet_time_setup()
    cfg = effect.get_passive_config()
    assert cfg is not None
    assert cfg.trigger == PassiveTrigger.AFTER_BASIC_ACTION
    assert cfg.is_optional is True


@pytest.mark.effect_contract
def test_bullet_time_offers_only_when_basic_card_resolved() -> None:
    from goa2.domain.models.enums import PassiveTrigger

    state, hero, ult, effect = _bullet_time_setup()
    assert (
        effect.should_offer_passive(
            state, hero, ult, PassiveTrigger.AFTER_BASIC_ACTION, {"basic_action_card_id": "bounce"}
        )
        is True
    )
    assert (
        effect.should_offer_passive(state, hero, ult, PassiveTrigger.AFTER_BASIC_ACTION, {})
        is False
    )


@pytest.mark.effect_contract
def test_bullet_time_performs_resolved_card_excluding_repeat_targets() -> None:
    from goa2.domain.models.enums import PassiveTrigger
    from goa2.engine.steps import PerformPrimaryActionStep

    state, hero, ult, effect = _bullet_time_setup()
    steps = effect.get_passive_steps(
        state, hero, ult, PassiveTrigger.AFTER_BASIC_ACTION, {"basic_action_card_id": "bounce"}
    )
    perform = [s for s in steps if isinstance(s, PerformPrimaryActionStep)]
    assert len(perform) == 1
    assert perform[0].card_key == "basic_action_card_id"
    # Excludes the just-resolved basic action's combat target (published by
    # ResolveCombatStep under last_combat_target, cleared each turn).
    assert perform[0].exclude_target_key == "last_combat_target"


def test_bullet_time_reperform_excludes_previous_target() -> None:
    # Drive the exact step Bullet Time emits: re-performing a ranged attack card
    # must NOT offer the hero recorded in last_combat_target.
    from goa2.domain.models import CardState
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.steps import PerformPrimaryActionStep

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_swift", at=(0, 0, 0))
        .blue_hero("hero_a", at=(4, 0, -4))  # straight line, max range 4
        .blue_hero("hero_b", at=(0, 4, -4))  # straight line, max range 4 (other axis)
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")
    snipe = hero_card("Swift", "snipe")
    snipe.state = CardState.RESOLVED
    swift.played_cards = [snipe]

    # Simulate that the original basic action already hit hero_a this turn.
    state.execution_context["last_combat_target"] = "hero_a"
    state.execution_context["bt_card"] = "snipe"

    push_steps(
        state,
        [
            PerformPrimaryActionStep(
                card_key="bt_card",
                hero_id="hero_swift",
                exclude_target_key="last_combat_target",
            )
        ],
    )
    result = process_stack(state)

    assert result.input_request is not None
    assert result.input_request.request_type.value == "SELECT_UNIT"
    options = set(result.input_request.to_dict()["valid_options"])
    assert "hero_a" not in options  # previously targeted -> excluded
    assert "hero_b" in options


@pytest.mark.effect_flow
def test_bullet_time_via_reload_copy_cannot_retarget_same_hero() -> None:
    # End-to-end: a red card (Snipe) resolved last turn. This turn Swift plays
    # the gold basic card Reload! and chooses "perform the primary action of your
    # rightmost resolved card", copying Snipe to hit hero_a. Because Reload is a
    # basic card, the Bullet Time ultimate then offers to perform Reload's
    # primary action again. The copied Snipe must NOT be allowed to re-target the
    # hero already hit this turn (hero_a); only hero_b remains selectable.
    from goa2.data.heroes.registry import HeroRegistry
    from goa2.domain.models import CardState

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "reload"))
        .blue_hero("hero_a", at=(4, 0, -4))  # straight line (q-axis), max range 4
        .blue_hero("hero_b", at=(0, 4, -4))  # straight line (r-axis), max range 4
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")

    # Swift's red Snipe was played and resolved last turn (rightmost resolved).
    snipe = hero_card("Swift", "snipe")
    snipe.state = CardState.RESOLVED
    swift.played_cards = [snipe]

    # Unlock the Bullet Time ultimate (active at level >= 8).
    swift.level = 8
    swift.ultimate_card = HeroRegistry.get("Swift").ultimate_card.model_copy(deep=True)

    # hero_a fully blocks the hit so it SURVIVES on the board -- this proves the
    # re-target is prevented by the exclusion filter, not merely because the
    # target was defeated and removed. hero_b keeps no defense.
    blocker = hero_card("Arien", "violent_torrent")  # secondary DEFENSE 7 > attack
    state.get_hero("hero_a").hand = [blocker]
    state.get_hero("hero_b").hand = []

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # copied Snipe's target

    # Both enemy heroes are valid Snipe targets before anyone is hit.
    assert {"hero_a", "hero_b"} <= _option_set(run)
    run.choose("hero_a").expect_input(InputRequestType.SELECT_CARD_OR_PASS)
    run.choose(blocker.id)  # hero_a fully blocks and stays on the board

    # Sanity: hero_a survived the copied Snipe.
    assert state.entity_locations.get("hero_a") is not None

    # Reload is a basic card -> Bullet Time offers to repeat its primary action.
    run.expect_input(InputRequestType.CONFIRM_PASSIVE)
    run.choose("YES").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # copied Snipe again

    options = _option_set(run)
    assert "hero_a" not in options  # already hit this turn -> excluded
    assert "hero_b" in options


@pytest.mark.effect_flow
def test_bullet_time_via_reload_copy_allows_bounce_repeat() -> None:
    # End-to-end: Bounce (red) resolved last turn. This turn Swift plays the gold
    # basic card Reload! and copies Bounce. Bounce's text grants "may repeat once"
    # only when the card is already resolved as the action is performed -- which is
    # exactly the case when it is copied. Reload is basic, so the Bullet Time
    # ultimate then re-performs Reload, copying Bounce a second time; that copy
    # must ALSO offer the Bounce repeat.
    from goa2.data.heroes.registry import HeroRegistry
    from goa2.domain.models import CardState

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(6))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "reload"))
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")

    # Swift's red Bounce was played and resolved last turn (rightmost resolved).
    bounce = hero_card("Swift", "bounce")
    bounce.state = CardState.RESOLVED
    swift.played_cards = [bounce]

    # Unlock the Bullet Time ultimate (active at level >= 8).
    swift.level = 8
    swift.ultimate_card = HeroRegistry.get("Swift").ultimate_card.model_copy(deep=True)

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_HEX)  # copied Bounce's move

    # Resolved Bounce grants the repeat -> after moving, the repeat is offered.
    run.choose({"q": 2, "r": 0, "s": -2}).expect_input(InputRequestType.SELECT_OPTION)
    assert "YES" in {o.id for o in run.latest_request.options}
    run.choose("NO")  # decline this repeat, keep position predictable
    assert state.entity_locations.get("hero_swift") == Hex(q=2, r=0, s=-2)

    # Reload is a basic card -> Bullet Time offers to repeat its primary action.
    run.expect_input(InputRequestType.CONFIRM_PASSIVE)
    run.choose("YES").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_HEX)  # Bounce copied again

    # The Bullet Time copy must ALSO allow the Bounce repeat.
    run.choose({"q": 4, "r": 0, "s": -4}).expect_input(InputRequestType.SELECT_OPTION)
    assert "YES" in {o.id for o in run.latest_request.options}
    run.choose("YES").expect_input(InputRequestType.SELECT_HEX)  # the repeat move
    run.choose({"q": 6, "r": 0, "s": -6}).finish()

    assert state.entity_locations.get("hero_swift") == Hex(q=6, r=0, s=-6)


@pytest.mark.effect_flow
def test_bullet_time_reperform_of_bounce_allows_repeat() -> None:
    # Bounce is itself a basic (Silver) card, so it triggers Bullet Time directly:
    #   play Bounce -> move -> ultimate triggers -> move -> may repeat -> move again
    # The initial play does NOT grant the repeat (the card isn't resolved yet), but
    # by the time Bullet Time re-performs Bounce the action has resolved once, so the
    # re-performance MUST offer "may repeat once".
    from goa2.data.heroes.registry import HeroRegistry

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(6))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "bounce"))
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")
    swift.level = 8
    swift.ultimate_card = HeroRegistry.get("Swift").ultimate_card.model_copy(deep=True)

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)  # initial Bounce move
    # Initial play: no repeat -> the next prompt is the Bullet Time offer itself.
    run.choose({"q": 2, "r": 0, "s": -2}).expect_input(InputRequestType.CONFIRM_PASSIVE)

    # Ultimate re-performs Bounce.
    run.choose("YES").expect_input(InputRequestType.SELECT_HEX)  # Bullet Time move
    run.choose({"q": 4, "r": 0, "s": -4}).expect_input(InputRequestType.SELECT_OPTION)
    assert "YES" in {o.id for o in run.latest_request.options}  # repeat is offered
    run.choose("YES").expect_input(InputRequestType.SELECT_HEX)  # the repeat move
    run.choose({"q": 6, "r": 0, "s": -6}).finish()

    assert state.entity_locations.get("hero_swift") == Hex(q=6, r=0, s=-6)


@pytest.mark.effect_flow
def test_bullet_time_after_basic_secondary_performs_the_primary_action() -> None:
    # Reload! used for its secondary movement still resolves a basic card, so
    # Bullet Time offers Reload!'s primary action, which copies last turn's Killshot.
    from goa2.data.heroes.registry import HeroRegistry
    from goa2.domain.models import CardState

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_swift", at=(0, 0, 0), current_card=hero_card("Swift", "reload"))
        .blue_hero("hero_a", at=(4, 0, -4))
        .with_actor("hero_swift")
        .build()
    )
    swift = state.get_hero("hero_swift")
    killshot = hero_card("Swift", "killshot")
    killshot.state = CardState.RESOLVED
    swift.played_cards = [killshot]
    swift.level = 8
    swift.ultimate_card = HeroRegistry.get("Swift").ultimate_card.model_copy(deep=True)
    state.get_hero("hero_a").hand = []

    run = run_card(state, "hero_swift")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("MOVEMENT").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 1, "r": 0, "s": -1}).expect_input(InputRequestType.CONFIRM_PASSIVE)

    run.choose("YES").expect_input(InputRequestType.SELECT_NUMBER)  # Reload!'s choice
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # Killshot's target
    assert "hero_a" in _option_set(run)
    run.choose("hero_a").expect_input(InputRequestType.SELECT_CARD_OR_PASS)
    run.choose("PASS").finish()

    assert [
        e
        for e in run.events
        if e.event_type == GameEventType.COMBAT_RESOLVED and e.target_id == "hero_a"
    ]
