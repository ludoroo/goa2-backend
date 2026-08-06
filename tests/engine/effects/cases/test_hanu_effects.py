"""Effect flow tests for the hero Hanu."""

import pytest

from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType
from goa2.domain.models import Token, TokenType
from goa2.domain.models.effect import (
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.engine.effect_manager import EffectManager

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _combat_values(run) -> list:
    return [
        e.metadata.get("attack_value")
        for e in run.events
        if e.event_type == GameEventType.COMBAT_RESOLVED
    ]


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


def _pos(state, uid) -> tuple:
    h = state.entity_locations.get(uid)
    return (h.q, h.r, h.s) if h is not None else None


def _place_passable_mine(state, at: tuple[int, int, int], owner_id: str) -> None:
    mine = Token(
        id="mine_1",
        name="Mine",
        token_type=TokenType.MINE_DUD,
        owner_id=owner_id,
        is_passable=True,
    )
    state.register_entity(mine, "token")
    state.token_pool.setdefault(TokenType.MINE_DUD, []).append(mine)
    state.place_entity("mine_1", Hex(q=at[0], r=at[1], s=at[2]))


# =============================================================================
# GREEN — Monkey Trick / Twist / Business: "Swap two friendly units in radius."
# (Trick r1, Twist r2, Business r2 + optional move 1). Self is NOT selectable.
# =============================================================================


@pytest.mark.effect_flow
def test_monkey_trick_swaps_two_friendly_minions() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "monkey_trick"))
        .red_minion("red_a", at=(1, 0, -1))
        .red_minion("red_b", at=(0, 1, -1))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_a").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_b").finish()

    assert _pos(state, "red_a") == (0, 1, -1)
    assert _pos(state, "red_b") == (1, 0, -1)


@pytest.mark.effect_flow
def test_monkey_trick_excludes_self_enemy_and_out_of_radius() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "monkey_trick"))
        .red_minion("red_ok", at=(1, 0, -1))  # friendly, in radius 1
        .red_minion("red_far", at=(3, 0, -3))  # friendly, out of radius 1
        .blue_minion("blue_enemy", at=(0, 1, -1))  # enemy, in radius
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    options = _option_set(run)
    assert "red_ok" in options
    assert "hero_hanu" not in options  # self excluded
    assert "blue_enemy" not in options  # enemy excluded
    assert "red_far" not in options  # out of radius


@pytest.mark.effect_flow
def test_monkey_trick_aborts_without_two_friendly_units() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "monkey_trick"))
        .red_minion("red_only", at=(1, 0, -1))  # only one friendly other than self
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    # First pick is the only friendly; second mandatory pick has no candidate -> abort.
    run.choose("red_only").finish()
    # No swap happened (nothing to swap with).
    assert _pos(state, "red_only") == (1, 0, -1)


@pytest.mark.effect_flow
def test_monkey_business_optional_move_after_swap() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "monkey_business"))
        .red_minion("red_a", at=(1, 0, -1))
        .red_minion("red_b", at=(2, 0, -2))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_a").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_b").expect_input(InputRequestType.SELECT_HEX)
    # Swap resolved; now Hanu may move 1 space.
    assert _pos(state, "red_a") == (2, 0, -2)
    assert _pos(state, "red_b") == (1, 0, -1)
    run.choose({"q": 0, "r": 1, "s": -1}).finish()
    assert _pos(state, "hero_hanu") == (0, 1, -1)


@pytest.mark.effect_flow
def test_monkey_business_move_is_skippable() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "monkey_business"))
        .red_minion("red_a", at=(1, 0, -1))
        .red_minion("red_b", at=(2, 0, -2))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_a").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_b").expect_input(InputRequestType.SELECT_HEX)
    run.skip().finish()
    assert _pos(state, "hero_hanu") == (0, 0, 0)


# =============================================================================
# GREEN — Hear Nothing / See Nothing: "Swap with an enemy hero in radius."
# (both r3; See Nothing also: "You may move 1 space.")
# =============================================================================


@pytest.mark.effect_flow
def test_hear_nothing_swaps_with_enemy_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hear_nothing"))
        .blue_hero("blue_hero", at=(3, 0, -3))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert "blue_hero" in _option_set(run)
    run.choose("blue_hero").finish()
    assert _pos(state, "hero_hanu") == (3, 0, -3)
    assert _pos(state, "blue_hero") == (0, 0, 0)


@pytest.mark.effect_flow
def test_hear_nothing_rejects_minion_friendly_and_out_of_radius() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(6))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hear_nothing"))
        .blue_hero("blue_ok", at=(2, 0, -2))  # eligible
        .blue_minion("blue_minion", at=(1, 0, -1))  # not a hero
        .red_hero("red_ally", at=(0, 1, -1))  # friendly
        .blue_hero("blue_far", at=(5, 0, -5))  # out of radius 3
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    options = _option_set(run)
    assert "blue_ok" in options
    assert "blue_minion" not in options
    assert "red_ally" not in options
    assert "blue_far" not in options


@pytest.mark.effect_flow
def test_hear_nothing_skill_unavailable_with_no_enemy_hero_in_radius() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hear_nothing"))
        .blue_minion("blue_minion", at=(1, 0, -1))  # only an enemy minion
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.expect_option_absent("SKILL")
    assert _pos(state, "hero_hanu") == (0, 0, 0)


@pytest.mark.effect_flow
def test_see_nothing_swaps_then_optional_move() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "see_nothing"))
        .blue_hero("blue_hero", at=(3, 0, -3))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_hero").expect_input(InputRequestType.SELECT_HEX)
    assert _pos(state, "hero_hanu") == (3, 0, -3)
    # Move 1 from Hanu's new spot.
    run.choose({"q": 2, "r": 0, "s": -2}).finish()
    assert _pos(state, "hero_hanu") == (2, 0, -2)


# =============================================================================
# RED — Helping Hand / Even the Odds / Trusted Sidekick (attack; bullet B on
# hero) and Outnumber / Pile On (bullet B on minion).
# Bullet A: "a unit adjacent to you". Bullet B: "a hero/minion in range,
# adjacent to your friendly hero" (Trusted Sidekick/Pile On: "or both"; Trusted
# Sidekick also "and not adjacent to you").
# =============================================================================


@pytest.mark.effect_flow
def test_helping_hand_bullet_a_attacks_adjacent_unit() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "helping_hand"))
        .blue_minion("blue_adj", at=(1, 0, -1))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_adj").finish()
    assert 3 in _combat_values(run)


@pytest.mark.effect_flow
def test_helping_hand_bullet_b_attacks_hero_adjacent_to_friendly_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "helping_hand"))
        .red_hero("red_ally", at=(2, 0, -2))
        .blue_hero("blue_target", at=(3, 0, -3))  # adj to red_ally, range 3, not adj Hanu
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert "blue_target" in _option_set(run)
    run.choose("blue_target").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").finish()
    assert 3 in _combat_values(run)


@pytest.mark.effect_flow
def test_helping_hand_bullet_b_excludes_hero_adjacent_only_to_hanu() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "helping_hand"))
        .blue_minion("blue_adj", at=(1, 0, -1))  # bullet A target
        .blue_hero("blue_lone", at=(2, 0, -2))  # range 2, no friendly hero adjacent
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    options = _option_set(run)
    assert "blue_adj" in options
    assert "blue_lone" not in options  # not adjacent to any *other* friendly hero


@pytest.mark.effect_flow
def test_even_the_odds_bullet_b_at_range_four() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "even_the_odds"))
        .red_hero("red_ally", at=(3, 0, -3))
        .blue_hero("blue_target", at=(4, 0, -4))  # range 4, adj to red_ally
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert "blue_target" in _option_set(run)
    run.choose("blue_target").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").finish()
    assert 4 in _combat_values(run)


@pytest.mark.effect_flow
def test_outnumber_bullet_b_requires_minion_not_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "outnumber"))
        .red_hero("red_ally", at=(2, 0, -2))
        .blue_hero("blue_hero", at=(3, 0, -3))  # hero adj to ally -> NOT valid (needs minion)
        .blue_minion("blue_minion", at=(2, 1, -3))  # minion adj to ally -> valid
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    options = _option_set(run)
    assert "blue_minion" in options
    assert "blue_hero" not in options


@pytest.mark.effect_flow
def test_trusted_sidekick_attacks_both_bullets() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "trusted_sidekick"))
        .blue_minion("blue_adj", at=(1, 0, -1))  # bullet A
        .red_hero("red_ally", at=(2, 0, -2))
        .blue_hero("blue_far", at=(3, 0, -3))  # bullet B: adj ally, not adj Hanu
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)  # choose order
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # bullet A first
    run.choose("blue_adj").expect_input(InputRequestType.SELECT_UNIT)  # bullet B
    run.choose("blue_far").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").finish()
    values = _combat_values(run)
    assert len(values) == 2
    assert all(v == 4 for v in values)


@pytest.mark.effect_flow
def test_trusted_sidekick_can_attack_bullet_b_before_bullet_a() -> None:
    """ "Choose one, or both, in any order": the player may resolve bullet B
    (friendly-hero-anchored attack) before bullet A (adjacent attack)."""
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "trusted_sidekick"))
        .blue_minion("blue_adj", at=(1, 0, -1))  # bullet A
        .red_hero("red_ally", at=(2, 0, -2))
        .blue_hero("blue_far", at=(3, 0, -3))  # bullet B: adj ally, not adj Hanu
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)  # choose order
    assert _option_set(run) == {1, 2}
    run.choose(2).expect_input(InputRequestType.SELECT_UNIT)  # bullet B first
    assert "blue_far" in _option_set(run)
    run.choose("blue_far").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").expect_input(InputRequestType.SELECT_UNIT)  # then bullet A
    run.choose("blue_adj").finish()
    values = _combat_values(run)
    assert len(values) == 2
    assert all(v == 4 for v in values)


@pytest.mark.effect_flow
def test_trusted_sidekick_bullet_b_excludes_target_adjacent_to_you() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "trusted_sidekick"))
        .red_hero("red_ally", at=(2, 0, -2))
        .blue_hero("blue_near", at=(1, 0, -1))  # adj to ally AND adj to Hanu -> excluded from B
        .blue_hero("blue_far", at=(3, 0, -3))  # adj to ally, not adj Hanu -> valid B
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)  # choose order
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # bullet A first
    run.skip().expect_input(InputRequestType.SELECT_UNIT)  # bullet B
    options = _option_set(run)
    assert "blue_far" in options
    assert "blue_near" not in options


@pytest.mark.effect_flow
def test_pile_on_completes_with_only_one_bullet_available() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "pile_on"))
        .blue_minion("blue_adj", at=(1, 0, -1))  # only bullet A available (no friendly ally)
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)  # choose order
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # bullet A first
    run.choose("blue_adj").finish()  # bullet B auto-skips (no target)
    values = _combat_values(run)
    assert values == [5]


# =============================================================================
# RED — This Way! / That Way!: "A friendly hero in radius chooses a distance of
# 1/2(/3); move both of you that number of spaces in the same direction of your
# choice." (This Way d1-2, That Way d1-3.) Both move full distance or neither.
# =============================================================================


@pytest.mark.effect_flow
def test_this_way_moves_both_in_same_direction() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_hero("red_ally", at=(0, 1, -1))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 2, "r": 0, "s": -2}).finish()

    assert _pos(state, "hero_hanu") == (2, 0, -2)
    assert _pos(state, "red_ally") == (2, 1, -3)  # same +q offset


@pytest.mark.effect_flow
@pytest.mark.parametrize(
    "mine_hex",
    [(1, 0, -1), (1, 1, -2)],
    ids=["hanu-path", "ally-path"],
)
@pytest.mark.parametrize(
    "mine_owner,should_trigger",
    [("blue_mine_owner", True), ("red_mine_owner", False)],
    ids=["enemy-mine", "same-team-mine"],
)
def test_this_way_crosses_passable_mine_regardless_of_owner(
    mine_hex: tuple[int, int, int], mine_owner: str, should_trigger: bool
) -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_hero("red_ally", at=(0, 1, -1))
        .red_hero("red_mine_owner", at=(-4, 0, 4))
        .blue_hero("blue_mine_owner", at=(-5, 0, 5))
        .with_actor("hero_hanu")
        .build()
    )
    _place_passable_mine(state, mine_hex, mine_owner)

    destination = Hex(q=2, r=0, s=-2)
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_HEX)

    assert destination in _option_set(run)
    run.choose(destination).finish()

    assert _pos(state, "hero_hanu") == (2, 0, -2)
    assert _pos(state, "red_ally") == (2, 1, -3)
    triggered = [
        event.target_id for event in run.events if event.event_type == GameEventType.MINE_TRIGGERED
    ]
    assert triggered == (["mine_1"] if should_trigger else [])
    assert (_pos(state, "mine_1") is None) is should_trigger


@pytest.mark.effect_flow
def test_this_way_rejects_direction_when_ally_would_land_on_mine() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_hero("red_ally", at=(0, 1, -1))
        .red_hero("red_mine_owner", at=(-4, 0, 4))
        .with_actor("hero_hanu")
        .build()
    )
    _place_passable_mine(state, (2, 1, -3), "red_mine_owner")

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_HEX)

    assert Hex(q=2, r=0, s=-2) not in _option_set(run)


@pytest.mark.effect_flow
def test_this_way_distance_options_are_one_or_two() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_hero("red_ally", at=(0, 1, -1))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    assert _option_set(run) == {"1", "2"}


@pytest.mark.effect_flow
def test_that_way_distance_options_are_one_two_or_three() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "that_way"))
        .red_hero("red_ally", at=(0, 1, -1))
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    assert _option_set(run) == {"1", "2", "3"}


@pytest.mark.effect_flow
def test_this_way_all_or_nothing_excludes_direction_that_drops_partner_off_board() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_hero("red_ally", at=(3, 0, -3))  # on the +q edge
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(2).expect_input(InputRequestType.SELECT_HEX)
    from goa2.domain.hex import Hex

    options = _option_set(run)
    # +q would send the partner off the board -> not a valid co-move.
    assert Hex(q=2, r=0, s=-2) not in options
    # -q keeps both on-board -> valid.
    assert Hex(q=-2, r=0, s=2) in options


@pytest.mark.effect_flow
def test_this_way_aborts_without_a_friendly_hero_in_radius() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "this_way"))
        .red_minion("red_minion", at=(0, 1, -1))  # friendly, but a minion
        .blue_hero("blue_hero", at=(1, 0, -1))  # a hero, but enemy
        .with_actor("hero_hanu")
        .build()
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()  # no friendly hero -> mandatory partner select aborts
    assert _pos(state, "hero_hanu") == (0, 0, 0)
    assert _pos(state, "red_minion") == (0, 1, -1)


# =============================================================================
# RED — Unexpected Journey / There and Back Again / Safe Travels
# "Swap with an enemy hero in radius. This turn: That hero is immune.
#  End of turn: Swap with that hero, regardless of radius and immunity."
# (Safe Travels: end-of-turn "You may move 1 space".)
# =============================================================================


def _build_effect_steps(state, hero_id, card_id):
    from goa2.engine.effects import CardEffectRegistry
    from goa2.engine.stats import compute_card_stats

    hero = state.get_hero(hero_id)
    card = hero_card("Hanu", card_id)
    effect = CardEffectRegistry.get(card_id)
    stats = compute_card_stats(state, hero.id, card)
    # get_steps_with_stats, not build_steps: the public API is what binds the
    # card onto the effect-creating steps.
    return effect.get_steps_with_stats(state, hero, card, stats)


def _journey_state(card_id, radius_disk=4):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(radius_disk))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", card_id))
        .red_hero("red_ally", at=(0, 2, -2))
        .blue_hero("blue_enemy", at=(2, 0, -2))
        .with_actor("hero_hanu")
        .build()
    )


def _immunity_effects(state):
    from goa2.domain.models.effect import EffectType

    return [e for e in state.active_effects if e.effect_type == EffectType.IMMUNITY_ENEMY_ACTIONS]


def _delayed_effects(state):
    from goa2.domain.models.effect import EffectType

    return [
        e
        for e in state.active_effects
        if e.effect_type == EffectType.DELAYED_TRIGGER and e.finishing_steps
    ]


@pytest.mark.effect_flow
def test_unexpected_journey_swaps_and_grants_immunity_to_everyone() -> None:
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.rules import is_immune

    state = _journey_state("unexpected_journey")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))

    res = process_stack(state)
    assert res.input_request.request_type.value == "SELECT_UNIT"
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    res = process_stack(state)
    assert res.input_request is None

    # Swap happened.
    assert _pos(state, "hero_hanu") == (2, 0, -2)
    assert _pos(state, "blue_enemy") == (0, 0, 0)

    # Displaced enemy is immune (heavy-style) to Hanu's team this turn.
    imm = _immunity_effects(state)
    assert len(imm) == 1
    # Hanu owns it; the displaced hero is its subject.
    assert imm[0].source_id == "hero_hanu"
    assert imm[0].protected_unit_id == "blue_enemy"
    state.current_actor_id = "red_ally"
    assert is_immune(state.get_unit("blue_enemy"), state) is True

    # A swap-back is scheduled for end of turn.
    assert len(_delayed_effects(state)) == 1


@pytest.mark.effect_flow
def test_journey_immunity_blocks_even_the_displaced_heros_own_allies() -> None:
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.rules import is_immune

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "unexpected_journey"))
        .blue_hero("blue_enemy", at=(2, 0, -2))
        .blue_hero("blue_ally", at=(1, 1, -2))  # same team as blue_enemy
        .with_actor("hero_hanu")
        .build()
    )
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)

    unit = state.get_unit("blue_enemy")
    # Immune to everyone this turn (heavy-style): an enemy actor...
    state.current_actor_id = "hero_hanu"
    assert is_immune(unit, state) is True
    # ...AND the displaced hero's OWN ally cannot target it either.
    state.current_actor_id = "blue_ally"
    assert is_immune(unit, state) is True
    # But it is never immune to its own actions.
    state.current_actor_id = "blue_enemy"
    assert is_immune(unit, state) is False


@pytest.mark.effect_flow
def test_unexpected_journey_end_of_turn_swaps_back_regardless_of_range() -> None:
    from goa2.domain.hex import Hex
    from goa2.engine.handler import process_stack, push_steps

    state = _journey_state("unexpected_journey")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)

    # Move the displaced enemy far away — swap-back ignores range.
    state.move_unit("blue_enemy", Hex(q=4, r=0, s=-4))

    fin = _delayed_effects(state)[0].finishing_steps
    push_steps(state, [s.model_copy(deep=True) for s in fin])
    process_stack(state)

    # Positions swapped back regardless of the distance between them.
    assert _pos(state, "hero_hanu") == (4, 0, -4)
    assert _pos(state, "blue_enemy") == (2, 0, -2)


@pytest.mark.effect_flow
def test_journey_immunity_expires_at_end_of_turn() -> None:
    from goa2.engine.effect_manager import EffectManager
    from goa2.engine.handler import process_stack, push_steps

    state = _journey_state("unexpected_journey")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)
    assert len(_immunity_effects(state)) == 1

    EffectManager.expire_active_turn_effects(state)
    assert _immunity_effects(state) == []


@pytest.mark.effect_flow
def test_unexpected_journey_aborts_with_no_enemy_hero() -> None:
    from goa2.engine.handler import process_stack, push_steps

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "unexpected_journey"))
        .blue_minion("blue_minion", at=(1, 0, -1))
        .with_actor("hero_hanu")
        .build()
    )
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)

    assert _pos(state, "hero_hanu") == (0, 0, 0)
    assert _immunity_effects(state) == []
    assert _delayed_effects(state) == []


@pytest.mark.effect_flow
def test_there_and_back_again_swaps_at_radius_three() -> None:
    from goa2.engine.handler import process_stack, push_steps

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "there_and_back_again"))
        .blue_hero("blue_enemy", at=(3, 0, -3))  # radius 3
        .with_actor("hero_hanu")
        .build()
    )
    push_steps(state, _build_effect_steps(state, "hero_hanu", "there_and_back_again"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)
    assert _pos(state, "hero_hanu") == (3, 0, -3)
    assert _pos(state, "blue_enemy") == (0, 0, 0)


@pytest.mark.effect_flow
def test_safe_travels_end_of_turn_move_after_swap_back() -> None:
    from goa2.engine.handler import process_stack, push_steps

    state = _journey_state("safe_travels")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "safe_travels"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)

    fin = _delayed_effects(state)[0].finishing_steps
    push_steps(state, [s.model_copy(deep=True) for s in fin])
    res = process_stack(state)
    # After the swap-back, Hanu (now at (0,0,0)) may move 1 space.
    assert _pos(state, "hero_hanu") == (0, 0, 0)
    assert res.input_request.request_type.value == "SELECT_HEX"
    state.execution_stack[-1].pending_input = {"selection": {"q": 0, "r": 1, "s": -1}}
    process_stack(state)
    assert _pos(state, "hero_hanu") == (0, 1, -1)


# =============================================================================
# GOLD — Fight and Flight: "Target a unit adjacent to you. If the target is not
# defeated, After the attack: If able, move 3 spaces in a straight line."
# =============================================================================


@pytest.mark.effect_flow
def test_fight_and_flight_flees_when_target_survives() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "fight_and_flight"))
        .blue_hero("blue_hero", at=(1, 0, -1))
        .with_actor("hero_hanu")
        .build()
    )
    # Give the enemy a blockable card (secondary DEFENSE 4 > attack 2) so it survives.
    blue = state.get_hero("blue_hero")
    defense_card = hero_card("Hanu", "there_and_back_again")
    defense_card.id = "blue_def"
    blue.hand = [defense_card]

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_hero").expect_input("SELECT_CARD_OR_PASS")
    run.choose("blue_def").expect_input(InputRequestType.SELECT_HEX)
    # +q is blocked by blue_hero at (1,0,-1); flee -q for a clear straight line.
    run.choose({"q": -3, "r": 0, "s": 3}).finish()
    assert _pos(state, "hero_hanu") == (-3, 0, 3)


@pytest.mark.effect_flow
def test_fight_and_flight_no_flee_when_target_defeated() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "fight_and_flight"))
        .blue_hero("blue_hero", at=(1, 0, -1))
        .with_actor("hero_hanu")
        .build()
    )
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_hero").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").finish()  # defeated -> no flee prompt
    assert _pos(state, "hero_hanu") == (0, 0, 0)


@pytest.mark.effect_flow
def test_fight_and_flight_no_flee_when_no_straight_line_available() -> None:
    # Board radius 2: no hex is a full 3 spaces away, so the flee is not "able".
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(2))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "fight_and_flight"))
        .blue_hero("blue_hero", at=(1, 0, -1))
        .with_actor("hero_hanu")
        .build()
    )
    blue = state.get_hero("blue_hero")
    defense_card = hero_card("Hanu", "there_and_back_again")
    defense_card.id = "blue_def"
    blue.hand = [defense_card]

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_hero").expect_input("SELECT_CARD_OR_PASS")
    run.choose("blue_def").finish()  # survives, but cannot flee -> completes
    assert _pos(state, "hero_hanu") == (0, 0, 0)


@pytest.mark.effect_flow
def test_fight_and_flight_aborts_with_no_adjacent_unit() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "fight_and_flight"))
        .blue_hero("blue_far", at=(3, 0, -3))  # not adjacent
        .with_actor("hero_hanu")
        .build()
    )
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").finish()  # no adjacent target -> abort
    assert _pos(state, "hero_hanu") == (0, 0, 0)
    assert _combat_values(run) == []


# =============================================================================
# SILVER — Hurry Up!: "Set the printed Initiative value of an unresolved card of
# a hero in range to 11, until it is resolved, or otherwise changes state."
# Targets any hero (not self) in range 4. Base override; modifiers still stack.
# =============================================================================


def _hurry_state(*, target_init=2):
    target_card = hero_card("Hanu", "monkey_trick")  # printed initiative 2
    target_card.initiative = target_init
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .blue_hero("blue_enemy", at=(2, 0, -2), current_card=target_card)
        .with_actor("hero_hanu")
        .build()
    )


@pytest.mark.effect_flow
def test_hurry_up_sets_target_card_initiative_to_eleven() -> None:
    state = _hurry_state(target_init=2)
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_enemy").finish()
    assert state.get_hero("blue_enemy").current_turn_card.initiative == 11


@pytest.mark.effect_flow
def test_hurry_up_targets_any_hero_but_not_self() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(6))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .red_hero("red_ally", at=(2, 0, -2), current_card=hero_card("Hanu", "monkey_trick"))
        .blue_hero("blue_enemy", at=(0, 2, -2), current_card=hero_card("Hanu", "monkey_trick"))
        .blue_hero("blue_far", at=(5, 0, -5), current_card=hero_card("Hanu", "monkey_trick"))
        .with_actor("hero_hanu")
        .build()
    )
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    options = _option_set(run)
    assert "red_ally" in options  # friendly allowed
    assert "blue_enemy" in options  # enemy allowed
    assert "hero_hanu" not in options  # self excluded
    assert "blue_far" not in options  # out of range 4


@pytest.mark.effect_flow
def test_hurry_up_excludes_immune_enemy_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .red_hero("red_ally", at=(0, 2, -2), current_card=hero_card("Hanu", "monkey_trick"))
        .blue_hero(
            "blue_immune",
            at=(2, 0, -2),
            current_card=hero_card("Hanu", "monkey_trick"),
        )
        .with_actor("hero_hanu")
        .build()
    )
    EffectManager.create_effect(
        state=state,
        source_id="blue_immune",
        effect_type=EffectType.IMMUNITY_ENEMY_ACTIONS,
        scope=EffectScope(
            shape=Shape.POINT,
            origin_id="blue_immune",
            affects=AffectsFilter.SELF,
        ),
        duration=DurationType.THIS_TURN,
        is_active=True,
    )

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT)

    options = _option_set(run)
    assert "red_ally" in options
    assert "blue_immune" not in options


@pytest.mark.effect_flow
def test_hurry_up_initiative_modifiers_still_stack() -> None:
    from goa2.domain.models import StatType
    from goa2.engine.stats import get_computed_stat

    state = _hurry_state(target_init=2)
    # Target carries an Initiative item (+1) that must still apply on top of 11.
    state.get_hero("blue_enemy").items[StatType.INITIATIVE] = 1

    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_enemy").finish()

    card = state.get_hero("blue_enemy").current_turn_card
    base = card.get_base_stat_value(StatType.INITIATIVE)
    assert base == 11
    assert get_computed_stat(state, "blue_enemy", StatType.INITIATIVE, base) == 12


@pytest.mark.effect_flow
def test_hurry_up_restores_initiative_at_end_of_turn() -> None:
    from goa2.domain.models.effect import EffectType
    from goa2.engine.handler import process_stack, push_steps

    state = _hurry_state(target_init=3)
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("blue_enemy").finish()
    assert state.get_hero("blue_enemy").current_turn_card.initiative == 11

    # Execute the scheduled end-of-turn restore.
    delayed = [
        e
        for e in state.active_effects
        if e.effect_type == EffectType.DELAYED_TRIGGER and e.finishing_steps
    ]
    assert delayed
    push_steps(state, [s.model_copy(deep=True) for s in delayed[0].finishing_steps])
    process_stack(state)
    assert state.get_hero("blue_enemy").current_turn_card.initiative == 3


@pytest.mark.effect_flow
def test_hurry_up_skill_unavailable_with_no_hero_in_range() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(6))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .blue_hero("blue_far", at=(5, 0, -5), current_card=hero_card("Hanu", "monkey_trick"))
        .with_actor("hero_hanu")
        .build()
    )
    run = run_card(state, "hero_hanu")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.expect_option_absent("SKILL")
    assert state.get_hero("blue_far").current_turn_card.initiative != 11


# =============================================================================
# PURPLE — The Ultimate Trick: "You choose the next action, and how it is
# performed, for a hero you target with the Hurry Up!."
# Level 8 passive. Hurry Up! records a CONTROL_NEXT_ACTION effect; the handler
# reroutes the controlled hero's inputs to Hanu while they resolve that card.
# Only the decision-maker changes — actor/legality stay the controlled hero.
# =============================================================================


def _ultimate_state(*, enemy_card=None):
    """Level-8 Hanu with ultimate, about to play Hurry Up! on blue_enemy."""
    target_card = enemy_card if enemy_card is not None else hero_card("Hanu", "monkey_trick")
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .blue_hero("blue_enemy", at=(2, 0, -2), current_card=target_card)
        .with_unresolved_heroes(["blue_enemy"])
        .with_actor("hero_hanu")
        .build()
    )
    hanu = state.get_hero("hero_hanu")
    hanu.level = 8
    hanu.ultimate_card = hero_card("Hanu", "the_ultimate_trick")
    return state


def _play_hurry_up(run):
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    return run.choose("blue_enemy")


@pytest.mark.effect_flow
def test_ultimate_trick_records_control_effect_at_level_8() -> None:
    from goa2.domain.models.effect import DurationType, EffectType

    state = _ultimate_state()
    run = run_card(state, "hero_hanu")
    _play_hurry_up(run).finish()

    effects = [e for e in state.active_effects if e.effect_type == EffectType.CONTROL_NEXT_ACTION]
    assert len(effects) == 1
    effect = effects[0]
    assert effect.source_id == "hero_hanu"
    assert effect.scope.origin_id == "blue_enemy"
    assert effect.controlled_card_id == "monkey_trick"
    assert effect.duration == DurationType.THIS_ROUND
    assert effect.is_active
    assert any(
        e.event_type == GameEventType.EFFECT_CREATED
        and e.metadata.get("effect") == "action_control"
        for e in run.events
    )


@pytest.mark.effect_flow
def test_ultimate_trick_inactive_below_level_8() -> None:
    from goa2.domain.models.effect import EffectType

    state = _ultimate_state()
    state.get_hero("hero_hanu").level = 7
    run = run_card(state, "hero_hanu")
    _play_hurry_up(run).finish()

    assert not [e for e in state.active_effects if e.effect_type == EffectType.CONTROL_NEXT_ACTION]
    # Hurry Up!'s own initiative effect still applies without the ultimate.
    assert state.get_hero("blue_enemy").current_turn_card.initiative == 11


@pytest.mark.effect_flow
def test_ultimate_trick_requires_ultimate_card() -> None:
    from goa2.domain.models.effect import EffectType

    state = _ultimate_state()
    state.get_hero("hero_hanu").ultimate_card = None
    run = run_card(state, "hero_hanu")
    _play_hurry_up(run).finish()

    assert not [e for e in state.active_effects if e.effect_type == EffectType.CONTROL_NEXT_ACTION]


@pytest.mark.effect_contract
def test_ultimate_trick_is_registered_and_inert() -> None:
    from goa2.engine.effects import CardEffectRegistry

    effect = CardEffectRegistry.get("the_ultimate_trick")
    assert effect is not None
    # No PassiveConfig: behavior lives in Hurry Up!'s ScheduleActionControlStep
    # plus the handler player_id remap.
    assert effect.get_passive_config() is None


def _basic_attack_card():
    from goa2.domain.models import ActionType, Card, CardColor, CardState, CardTier

    card = Card(
        id="test_basic_attack",
        name="Basic Attack",
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=2,
        primary_action=ActionType.ATTACK,
        primary_action_value=3,
        secondary_actions={},
        is_ranged=True,
        range_value=2,
        effect_id="",
        effect_text="",
        is_facedown=False,
    )
    card.state = CardState.UNRESOLVED
    return card


@pytest.mark.effect_flow
def test_controlled_heroes_inputs_reroute_to_hanu() -> None:
    state = _ultimate_state()
    run = run_card(state, "hero_hanu", finalize_turn=True)
    _play_hurry_up(run)
    # FinalizeHeroTurnStep -> FindNextActorStep picks blue_enemy (initiative 11).
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    assert run.latest_request.player_id == "hero_hanu"
    assert run.latest_request.context["controlled_hero_id"] == "blue_enemy"
    # Client contract: to_dict() surfaces the controlled hero.
    assert run.latest_request.to_dict()["controlled_hero_id"] == "blue_enemy"
    # The remap must not freeze rollback: Hanu confirms/rolls back this action.
    assert state.execution_context.get("rollback_frozen") is None


@pytest.mark.effect_flow
def test_controlled_hero_legality_is_relative_to_that_hero() -> None:
    # Pedro (Hanu) controls Wuk but cannot make Wuk attack Wuk's allies:
    # options are computed relative to the controlled hero, not the controller.
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .blue_hero("blue_enemy", at=(2, 0, -2), current_card=_basic_attack_card())
        .blue_minion("blue_ally_minion", at=(3, 0, -3))  # adjacent to blue_enemy
        .red_minion("red_minion", at=(2, 1, -3))  # adjacent to blue_enemy
        .with_unresolved_heroes(["blue_enemy"])
        .with_actor("hero_hanu")
        .build()
    )
    hanu = state.get_hero("hero_hanu")
    hanu.level = 8
    hanu.ultimate_card = hero_card("Hanu", "the_ultimate_trick")

    run = run_card(state, "hero_hanu", finalize_turn=True)
    _play_hurry_up(run)
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    assert run.latest_request.player_id == "hero_hanu"
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    assert run.latest_request.player_id == "hero_hanu"
    options = _option_set(run)
    assert "red_minion" in options  # blue_enemy's enemies are attackable
    assert "blue_ally_minion" not in options  # blue_enemy's own ally is not


@pytest.mark.effect_flow
def test_defender_inputs_during_controlled_turn_stay_with_defender() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_hanu", at=(0, 0, 0), current_card=hero_card("Hanu", "hurry_up"))
        .red_hero("red_ally", at=(2, 1, -3))  # adjacent to blue_enemy
        .blue_hero("blue_enemy", at=(2, 0, -2), current_card=_basic_attack_card())
        .with_unresolved_heroes(["blue_enemy"])
        .with_actor("hero_hanu")
        .build()
    )
    hanu = state.get_hero("hero_hanu")
    hanu.level = 8
    hanu.ultimate_card = hero_card("Hanu", "the_ultimate_trick")

    run = run_card(state, "hero_hanu", finalize_turn=True)
    _play_hurry_up(run)
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("red_ally").expect_input("SELECT_CARD_OR_PASS")
    assert run.latest_request.player_id == "red_ally"
    assert "controlled_hero_id" not in run.latest_request.context


@pytest.mark.effect_flow
def test_control_fizzles_if_targeted_card_changes() -> None:
    from goa2.domain.types import HeroID

    state = _ultimate_state()
    run = run_card(state, "hero_hanu")
    _play_hurry_up(run).finish()

    # The targeted card leaves UNRESOLVED some other way (replaced here);
    # control must not latch onto the replacement.
    state.get_hero("blue_enemy").current_turn_card = hero_card("Hanu", "this_way")
    state.current_actor_id = HeroID("blue_enemy")
    run2 = run_card(state, "blue_enemy")
    run2.expect_input(InputRequestType.CHOOSE_ACTION)
    assert run2.latest_request.player_id == "blue_enemy"
    assert "controlled_hero_id" not in run2.latest_request.context


@pytest.mark.effect_flow
def test_control_expires_at_end_of_round() -> None:
    from goa2.domain.models.effect import DurationType, EffectType
    from goa2.engine.effect_manager import EffectManager

    state = _ultimate_state()
    run = run_card(state, "hero_hanu")
    _play_hurry_up(run).finish()

    EffectManager.expire_effects(state, DurationType.THIS_ROUND)
    assert not [e for e in state.active_effects if e.effect_type == EffectType.CONTROL_NEXT_ACTION]


@pytest.mark.effect_contract
def test_journey_effects_belong_to_the_journey_card() -> None:
    """Both halves of the Journey are the card's active effect.

    Card binding is what lets Hanu's defeat end the immunity: the immunity is
    registered against the *displaced* hero (that is how is_immune_to_actor
    identifies who is protected), so nothing else ties it back to Hanu.
    """
    from goa2.engine.handler import process_stack, push_steps

    state = _journey_state("unexpected_journey")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)

    assert _immunity_effects(state)[0].source_card_id == "unexpected_journey"
    assert _delayed_effects(state)[0].source_card_id == "unexpected_journey"


@pytest.mark.effect_contract
def test_hanu_defeat_ends_the_journey_immunity() -> None:
    """A defeated hero's card stops protecting the hero it displaced."""
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.rules import is_immune
    from goa2.engine.steps import DefeatUnitStep

    state = _journey_state("unexpected_journey")
    push_steps(state, _build_effect_steps(state, "hero_hanu", "unexpected_journey"))
    process_stack(state)
    state.execution_stack[-1].pending_input = {"selection": "blue_enemy"}
    process_stack(state)
    assert _immunity_effects(state)

    push_steps(state, [DefeatUnitStep(victim_id="hero_hanu", killer_id="blue_enemy")])
    process_stack(state)

    assert _immunity_effects(state) == []
    state.current_actor_id = "red_ally"
    assert is_immune(state.get_unit("blue_enemy"), state) is False
