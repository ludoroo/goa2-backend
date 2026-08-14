"""Effect flow tests for Ignatia — the coin-branch chaos hero.

Every branch card reads the Tie Breaker coin face: BLUE face -> the
:tiebreaker_blue: text, ORANGE face -> the :tiebreaker_orange: text. The coin
is the same bit as ``state.tie_breaker_team`` (BLUE team -> blue face, RED team
-> orange face); see GameState.coin_face.
"""

import pytest

from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.events import GameEventType
from goa2.domain.input import InputRequestType
from goa2.domain.models import TeamColor

from ..builders import EffectScenarioBuilder, hero_card, hex_at
from ..runner import run_card


def _enable_ultimate(state, hero_id: str = "hero_ignatia") -> None:
    """Unlock Chaos Incarnate: level 8 + the ultimate card in hand-of-record."""
    ig = state.get_hero(hero_id)
    ig.level = 8
    ig.ultimate_card = HeroRegistry.get("Ignatia").ultimate_card


def _enable_equilibrium(state, hero_id: str = "hero_ignatia") -> None:
    """Activate a THIS_ROUND Equilibrium effect on Ignatia (as her Silver would)."""
    from goa2.domain.models.effect import (
        ActiveEffect,
        DurationType,
        EffectScope,
        EffectType,
        Shape,
    )

    ig = state.get_hero(hero_id)
    state.add_effect(
        ActiveEffect(
            id="equilibrium_test",
            source_id=str(ig.id),
            effect_type=EffectType.EQUILIBRIUM,
            scope=EffectScope(shape=Shape.POINT, origin_id=str(ig.id)),
            duration=DurationType.THIS_ROUND,
            created_at_turn=state.turn,
            created_at_round=state.round,
        )
    )


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


def _set_coin(state, face: str) -> None:
    state.tie_breaker_team = TeamColor.BLUE if face == "BLUE" else TeamColor.RED


# Straight-line axis from origin: (2,0,-2) is on a straight line; (2,-1,-1) is not.
ON_AXIS = (2, 0, -2)
OFF_AXIS = (2, -1, -1)


# =============================================================================
# F1 — Fire attacks: playing_with_fire / erratic_fireblast / loosely_aimed_firebolts
#   blue  : target a unit in range NOT in a straight line
#   orange: target a unit in range in a straight line
# =============================================================================


def _fire_state(card_id: str):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .blue_minion("on_axis", at=ON_AXIS)
        .blue_minion("off_axis", at=OFF_AXIS)
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_playing_with_fire_blue_targets_only_off_straight_line() -> None:
    state = _fire_state("playing_with_fire")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "off_axis" in opts
    assert "on_axis" not in opts


@pytest.mark.effect_flow
def test_playing_with_fire_orange_targets_only_straight_line() -> None:
    state = _fire_state("playing_with_fire")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "on_axis" in opts
    assert "off_axis" not in opts


@pytest.mark.effect_flow
def test_playing_with_fire_blue_resolves_attack_on_off_axis_target() -> None:
    state = _fire_state("playing_with_fire")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    # The minion has no defense card, so combat auto-resolves on selection.
    run.choose("off_axis").finish()

    assert any(e.event_type == GameEventType.COMBAT_RESOLVED for e in run.events)


@pytest.mark.effect_flow
def test_loosely_aimed_firebolts_orange_skips_the_repeat_with_no_second_hero() -> None:
    """An unsatisfiable YES would abort the action, ultimate included."""
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero(
            "hero_ignatia",
            at=(0, 0, 0),
            current_card=hero_card("Ignatia", "loosely_aimed_firebolts"),
        )
        .blue_hero("h_on", at=ON_AXIS)  # the only enemy hero anywhere
        .with_actor("hero_ignatia")
        .build()
    )
    _set_coin(state, "ORANGE")
    _enable_ultimate(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("h_on").expect_input("SELECT_CARD_OR_PASS")
    # Straight past the repeat, to the ultimate's own prompt.
    run.choose("PASS").expect_input(InputRequestType.SELECT_OPTION)
    assert "Chaos Incarnate" in run.latest_request.prompt


@pytest.mark.effect_flow
def test_loosely_aimed_firebolts_repeat_gate_does_not_leak_across_equilibrium() -> None:
    """The repeat's ``active_if_key`` exempts it from Equilibrium's branch
    gating, so its gate must stay unset on the blue branch."""
    state = _fire_state("loosely_aimed_firebolts")
    _set_coin(state, "ORANGE")
    _enable_equilibrium(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # 1 = Blue
    # Blue has no repeat clause; the orange gate must not fire.
    run.choose("off_axis").finish()


def _fire_hero_state(card_id: str):
    """Fire scenario where targets are heroes (so the first target survives the
    attack and the ultimate re-perform's exclusion is observable)."""
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .blue_hero("h_on", at=ON_AXIS)
        .blue_hero("h_off", at=OFF_AXIS)
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_playing_with_fire_ultimate_flips_coin_and_reperforms_opposite_branch() -> None:
    state = _fire_hero_state("playing_with_fire")
    _set_coin(state, "BLUE")
    _enable_ultimate(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    # Blue: first attack targets off the straight line.
    run.choose("h_off").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").expect_input(InputRequestType.SELECT_OPTION)
    # Chaos Incarnate: flip the coin and perform again.
    run.choose("YES").expect_input(InputRequestType.SELECT_UNIT)

    # The coin flipped BLUE -> orange, so the re-perform runs the orange branch.
    assert state.coin_face == "ORANGE"
    opts = _option_set(run)
    assert "h_on" in opts  # orange -> in a straight line
    assert "h_off" not in opts  # opposite branch (and first target) excluded


@pytest.mark.effect_flow
def test_playing_with_fire_ultimate_can_be_declined() -> None:
    state = _fire_hero_state("playing_with_fire")
    _set_coin(state, "BLUE")
    _enable_ultimate(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("h_off").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").expect_input(InputRequestType.SELECT_OPTION)
    # Decline: no flip, no re-perform.
    run.choose("NO").finish()

    assert state.coin_face == "BLUE"  # coin unchanged


@pytest.mark.effect_flow
def test_equilibrium_lets_her_pick_blue_against_an_orange_coin() -> None:
    state = _fire_state("playing_with_fire")
    _set_coin(state, "ORANGE")  # coin shows orange (in-line)...
    _enable_equilibrium(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    # ...but Equilibrium prompts her to choose the side instead of reading the coin.
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # 1 = Blue

    opts = _option_set(run)
    assert "off_axis" in opts  # blue -> not in a straight line
    assert "on_axis" not in opts


@pytest.mark.effect_flow
def test_ultimate_with_equilibrium_makes_the_reperform_a_free_choice() -> None:
    state = _fire_hero_state("playing_with_fire")
    _set_coin(state, "BLUE")
    _enable_ultimate(state)
    _enable_equilibrium(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)  # first: choose side
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # blue
    run.choose("h_off").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").expect_input(InputRequestType.SELECT_OPTION)  # ultimate prompt
    # YES flips the coin, but the re-perform is another free choice (SELECT_NUMBER),
    # not a forced flipped-face attack.
    run.choose("YES").expect_input(InputRequestType.SELECT_NUMBER)

    assert state.coin_face == "ORANGE"  # the flip still happened (matters for future ties)


@pytest.mark.effect_flow
def test_erratic_fireblast_blue_excludes_straight_line_target() -> None:
    # Same branch logic as playing_with_fire, Tier II stats (range 3).
    state = _fire_state("erratic_fireblast")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "off_axis" in opts
    assert "on_axis" not in opts


# ---- loosely_aimed_firebolts (Tier III, range 3) --------------------------
#   orange adds: "May repeat once on a different hero." (repeat fires even if
#   the first target was not a hero; the repeat target must be a hero, in a
#   straight line, and different from the first.)


def _loosely_state():
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero(
            "hero_ignatia",
            at=(0, 0, 0),
            current_card=hero_card("Ignatia", "loosely_aimed_firebolts"),
        )
        .blue_minion("m_on", at=(2, 0, -2))  # on-axis minion (first target)
        .blue_minion("m_on2", at=(0, -2, 2))  # on-axis minion (not a valid repeat)
        .blue_hero("h_on", at=(-2, 0, 2))  # on-axis hero (valid repeat)
        .blue_hero("h_off", at=(2, -1, -1))  # off-axis hero (not in line)
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_loosely_orange_repeat_targets_only_a_different_hero_in_line() -> None:
    state = _loosely_state()
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    # First target is a (non-hero) minion in a straight line — repeat still fires.
    run.choose("m_on").expect_input(InputRequestType.SELECT_OPTION)
    run.choose("YES").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "h_on" in opts  # a different hero, in a straight line
    assert "h_off" not in opts  # off the straight line
    assert "m_on" not in opts  # the first target is excluded
    assert "m_on2" not in opts  # repeat must be a hero, not a minion


@pytest.mark.effect_flow
def test_loosely_blue_attacks_off_axis_without_repeat() -> None:
    state = _loosely_state()
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "h_off" in opts  # off the straight line -> valid under blue
    assert "m_on" not in opts  # on the straight line -> excluded under blue
    # Blue has no repeat: after the defender passes, the action ends (no repeat prompt).
    run.choose("h_off").expect_input("SELECT_CARD_OR_PASS")
    run.choose("PASS").finish()


# =============================================================================
# F2 — Range-extreme attacks (crack_of_doom / imminent_eruption), range 5
#   blue  : target a unit adjacent to you (range 1)
#   orange: target a unit at maximum range (exactly the card's range)
# =============================================================================


def _range_state(card_id: str):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .blue_minion("adj", at=(1, 0, -1))  # range 1
        .blue_minion("mid", at=(0, 3, -3))  # range 3
        .blue_minion("far", at=(-5, 0, 5))  # range 5 (clear axis)
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_crack_of_doom_blue_targets_only_adjacent() -> None:
    state = _range_state("crack_of_doom")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "adj" in opts
    assert "mid" not in opts
    assert "far" not in opts


@pytest.mark.effect_flow
def test_crack_of_doom_orange_targets_only_maximum_range() -> None:
    state = _range_state("crack_of_doom")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "far" in opts  # exactly at max range
    assert "adj" not in opts
    assert "mid" not in opts  # closer than max range -> excluded


@pytest.mark.effect_flow
def test_imminent_eruption_orange_targets_only_maximum_range() -> None:
    state = _range_state("imminent_eruption")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "far" in opts
    assert "adj" not in opts
    assert "mid" not in opts


def _imminent_blue_state():
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero(
            "hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "imminent_eruption")
        )
        .blue_minion("m1", at=(1, 0, -1))  # adjacent minion
        .blue_minion("m2", at=(0, 1, -1))  # adjacent minion
        .blue_hero("h1", at=(-1, 0, 1))  # adjacent hero
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_imminent_eruption_blue_repeats_on_another_adjacent_minion() -> None:
    state = _imminent_blue_state()
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    # First adjacent target is a minion -> auto-resolves.
    run.choose("m1").expect_input(InputRequestType.SELECT_OPTION)
    run.choose("YES").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "m2" in opts  # another adjacent minion
    assert "m1" not in opts  # the first target died to the attack
    assert "h1" not in opts  # repeat must be a minion, not a hero


@pytest.mark.effect_flow
def test_imminent_eruption_blue_skips_the_repeat_with_no_minion_left() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero(
            "hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "imminent_eruption")
        )
        .blue_minion("m1", at=(1, 0, -1))  # the only adjacent minion
        .with_actor("hero_ignatia")
        .build()
    )
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("m1").finish()  # m1 dies; nothing left to repeat on

    assert "m1" not in state.entity_locations


def _protect_minion(state, minion_id: str, protector_id: str, token_at) -> None:
    """Cover ``minion_id`` with a totem-style MINION_PROTECTION, so an attack
    defeats it without removing it from the board (see the Brogan/Tali ruling)."""
    from goa2.domain.models import Token, TokenType
    from goa2.domain.models.effect import (
        ActiveEffect,
        AffectsFilter,
        DurationType,
        EffectScope,
        EffectType,
        Shape,
    )

    token = Token(id="totem_1", name="Totem", token_type=TokenType.TOTEM, owner_id=protector_id)
    state.misc_entities[token.id] = token
    state.place_entity(token.id, hex_at(token_at))
    state.add_effect(
        ActiveEffect(
            id="totem_protection",
            source_id=protector_id,
            token_id=token.id,
            effect_type=EffectType.MINION_PROTECTION,
            scope=EffectScope(
                shape=Shape.ADJACENT, origin_id=token.id, affects=AffectsFilter.FRIENDLY_UNITS
            ),
            duration=DurationType.PASSIVE,
            sacrifice_origin_token=True,
            created_at_turn=state.turn,
            created_at_round=state.round,
        )
    )


@pytest.mark.effect_flow
def test_imminent_eruption_blue_may_repeat_on_the_same_minion_if_it_survives() -> None:
    """ "May repeat once on a minion" — not "a different minion"."""
    state = _imminent_blue_state()
    _set_coin(state, "BLUE")
    _protect_minion(state, "m1", protector_id="h1", token_at=(2, 0, -2))

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("m1").expect_input(InputRequestType.SELECT_OPTION)
    run.choose("YES").expect_input(InputRequestType.SELECT_UNIT)

    assert "m1" in state.entity_locations  # the totem saved it
    opts = _option_set(run)
    assert "m1" in opts  # same minion may be hit again
    assert "m2" in opts


# =============================================================================
# F3 — Chaos Bolt (Gold basic), range 3
#   blue  : target a minion adjacent to you
#   orange: target a hero in range
# =============================================================================


def _chaos_bolt_state():
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "chaos_bolt"))
        .blue_minion("am", at=(1, 0, -1))  # adjacent minion
        .blue_hero("ah", at=(0, 1, -1))  # adjacent hero
        .blue_minion("fm", at=(3, 0, -3))  # range-3 minion
        .blue_hero("rh", at=(-3, 0, 3))  # range-3 hero
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_chaos_bolt_blue_targets_only_adjacent_minion() -> None:
    state = _chaos_bolt_state()
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "am" in opts
    assert "ah" not in opts  # blue wants a minion, not a hero
    assert "fm" not in opts  # not adjacent
    assert "rh" not in opts


@pytest.mark.effect_flow
def test_chaos_bolt_orange_targets_only_hero_in_range() -> None:
    state = _chaos_bolt_state()
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("ATTACK").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "ah" in opts  # adjacent hero, in range
    assert "rh" in opts  # range-3 hero
    assert "am" not in opts  # orange wants a hero, not a minion
    assert "fm" not in opts


# =============================================================================
# F4 — Discard/Defeat AoE
#   abrupt_combustion (r3) / spontaneous_immolation (r4):
#     blue  : an enemy hero in radius adjacent to a token or a minion discards
#     orange: remove an enemy minion in radius adjacent to an enemy hero
#   violent_conflagration (r4):
#     blue  : ...discards a card, OR is defeated
#     orange: DEFEAT an enemy minion in radius adjacent to an enemy hero
# =============================================================================


def _place_token(state, token_id, token_type, at):
    from goa2.domain.hex import Hex
    from goa2.domain.models import Token

    tok = Token(id=token_id, name=token_id, token_type=token_type)
    state.register_entity(tok)
    state.place_entity(token_id, Hex(q=at[0], r=at[1], s=at[2]))


@pytest.mark.effect_flow
def test_abrupt_combustion_blue_targets_hero_adjacent_to_any_token_or_minion() -> None:
    from goa2.domain.models import TokenType

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero(
            "hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "abrupt_combustion")
        )
        .blue_hero("h_tok", at=(2, 0, -2))  # adjacent to a token
        .blue_hero("h_min", at=(0, 2, -2))  # adjacent to a minion
        .blue_hero("h_alone", at=(-2, 0, 2))  # adjacent to nothing
        .blue_minion("supp", at=(0, 3, -3))  # anchor for h_min
        .with_actor("hero_ignatia")
        .build()
    )
    # A non-Magma token proves the check is "any token", not Magma-specific.
    _place_token(state, "rock1", TokenType.ROCK, at=(3, 0, -3))
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)

    opts = _option_set(run)
    assert "h_tok" in opts
    assert "h_min" in opts
    assert "h_alone" not in opts


def _orange_minion_state(card_id: str):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .blue_minion("supp", at=(1, 0, -1))  # adjacent to enemy hero eh
        .blue_hero("eh", at=(2, 0, -2))
        .blue_minion("lone_m", at=(0, 2, -2))  # not adjacent to any enemy hero
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_abrupt_combustion_orange_removes_supported_minion_no_coins() -> None:
    state = _orange_minion_state("abrupt_combustion")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "supp" in opts
    assert "lone_m" not in opts  # not adjacent to an enemy hero

    run.choose("supp").finish()
    assert state.entity_locations.get("supp") is None  # removed
    assert state.get_hero("hero_ignatia").gold == 0  # remove -> no coins


@pytest.mark.effect_flow
def test_violent_conflagration_orange_defeats_supported_minion_for_coins() -> None:
    state = _orange_minion_state("violent_conflagration")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("supp").finish()

    assert state.entity_locations.get("supp") is None  # defeated
    assert state.get_hero("hero_ignatia").gold > 0  # defeat -> coins


@pytest.mark.effect_flow
def test_violent_conflagration_blue_defeats_cardless_eligible_hero() -> None:
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero(
            "hero_ignatia",
            at=(0, 0, 0),
            current_card=hero_card("Ignatia", "violent_conflagration"),
        )
        .blue_hero("victim", at=(2, 0, -2))
        .blue_minion("anchor", at=(3, 0, -3))  # makes victim eligible
        .with_actor("hero_ignatia")
        .build()
    )
    state.get_hero("victim").hand = []  # no cards -> "or is defeated"
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("victim").finish()

    assert any(e.event_type == GameEventType.UNIT_DEFEATED for e in run.events)


# =============================================================================
# F5 — Move a hero in a straight line (searing_heat r3 / scorching_blaze r3)
#   blue  : move a friendly hero in radius N spaces in a straight line
#   orange: move an enemy hero in radius N spaces in a straight line
#   searing: N = 2; scorching: N = 2 or 3
# =============================================================================


def _hex_coords(run) -> set:
    return {(o.q, o.r, o.s) for o in _option_set(run) if hasattr(o, "q")}


def _move_hero_state(card_id: str):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .red_hero("ally", at=(2, 0, -2))  # friendly hero
        .blue_hero("enemy", at=(0, 2, -2))  # enemy hero
        .with_actor("hero_ignatia")
        .build()
    )


@pytest.mark.effect_flow
def test_searing_heat_blue_moves_a_friendly_hero_two_in_a_straight_line() -> None:
    state = _move_hero_state("searing_heat")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "ally" in opts  # a friendly hero
    assert "enemy" not in opts  # not the enemy (blue = friendly)
    assert "hero_ignatia" not in opts  # never yourself

    run.choose("ally").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 4, "r": 0, "s": -4}).finish()  # 2 straight-line from (2,0,-2)

    moved = state.entity_locations.get("ally")
    assert (moved.q, moved.r, moved.s) == (4, 0, -4)


@pytest.mark.effect_flow
def test_searing_heat_orange_targets_an_enemy_hero() -> None:
    state = _move_hero_state("searing_heat")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "enemy" in opts
    assert "ally" not in opts


@pytest.mark.effect_flow
def test_searing_heat_distance_is_exactly_two() -> None:
    state = _move_hero_state("searing_heat")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("ally").expect_input(InputRequestType.SELECT_HEX)
    coords = _hex_coords(run)
    assert (4, 0, -4) in coords  # distance 2 offered
    assert (5, 0, -5) not in coords  # distance 3 NOT offered


@pytest.mark.effect_flow
def test_scorching_blaze_allows_distance_two_or_three() -> None:
    state = _move_hero_state("scorching_blaze")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("ally").expect_input(InputRequestType.SELECT_HEX)
    coords = _hex_coords(run)
    assert (4, 0, -4) in coords  # distance 2
    assert (5, 0, -5) in coords  # distance 3


def _blocked_ally_state(card_id: str, *, leave_lane_open: bool):
    """Ignatia at (0,0,0) with a friendly ally at (1,0,-1) whose straight lines
    are all blocked at the first step — except, when ``leave_lane_open``, the
    (0,-1,1) ray: a passable Mine sits 2 spaces out at (1,-2,1), so the ally can
    be moved 3 spaces to (1,-3,2) but cannot land at 2."""
    from goa2.domain.models import Token, TokenType

    builder = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(5))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .red_hero("ally", at=(1, 0, -1))
        # Four of the ally's six neighbours; Ignatia herself blocks the fifth.
        .blue_minion("b1", at=(2, -1, -1))
        .blue_minion("b2", at=(2, 0, -2))
        .blue_minion("b3", at=(1, 1, -2))
        .blue_minion("b4", at=(0, 1, -1))
    )
    if not leave_lane_open:
        builder = builder.blue_minion("b5", at=(1, -1, 0))  # seals the last ray
    state = builder.with_actor("hero_ignatia").build()

    if leave_lane_open:
        mine = Token(id="mine_1", name="Mine", token_type=TokenType.MINE_DUD, is_passable=True)
        state.misc_entities[mine.id] = mine
        state.place_entity(mine.id, hex_at((1, -2, 1)))
    return state


@pytest.mark.effect_flow
def test_searing_heat_does_not_offer_a_hero_who_cannot_be_moved() -> None:
    """Picking a boxed-in hero would dead-end at the destination select."""
    state = _blocked_ally_state("searing_heat", leave_lane_open=False)
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()  # no legal target at all -> nothing to ask

    assert _pos(state, "ally") == (1, 0, -1)


@pytest.mark.effect_flow
def test_searing_heat_does_not_offer_a_hero_who_can_only_move_three() -> None:
    state = _blocked_ally_state("searing_heat", leave_lane_open=True)
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()  # Searing Heat moves exactly 2 — no landing

    assert _pos(state, "ally") == (1, 0, -1)


@pytest.mark.effect_flow
def test_scorching_blaze_offers_a_hero_who_can_only_move_three() -> None:
    """Scorching Blaze moves 2 OR 3, so the same ally is a legal target."""
    state = _blocked_ally_state("scorching_blaze", leave_lane_open=True)
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    assert "ally" in _option_set(run)

    run.choose("ally").expect_input(InputRequestType.SELECT_HEX)
    assert _hex_coords(run) == {(1, -3, 2)}  # over the passable Mine, not onto it

    run.choose({"q": 1, "r": -3, "s": 2}).finish()
    assert _pos(state, "ally") == (1, -3, 2)


# =============================================================================
# F6 — Swaps (unstable_portal r4 / chaos_gate r4)
#   blue  : swap with a friendly unit in radius (chaos_gate: then move that unit 1)
#   orange: swap with an enemy unit in radius (chaos_gate: then move yourself 1)
# =============================================================================


def _swap_state(card_id: str):
    return (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .red_minion("fm", at=(2, 0, -2))  # friendly unit
        .blue_minion("em", at=(0, 2, -2))  # enemy unit
        .with_actor("hero_ignatia")
        .build()
    )


def _pos(state, uid):
    h = state.entity_locations.get(uid)
    return (h.q, h.r, h.s) if h is not None else None


@pytest.mark.effect_flow
def test_unstable_portal_blue_swaps_with_friendly_unit() -> None:
    state = _swap_state("unstable_portal")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "fm" in opts
    assert "em" not in opts  # blue = friendly

    run.choose("fm").finish()
    assert _pos(state, "hero_ignatia") == (2, 0, -2)
    assert _pos(state, "fm") == (0, 0, 0)


@pytest.mark.effect_flow
def test_unstable_portal_orange_swaps_with_enemy_unit() -> None:
    state = _swap_state("unstable_portal")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    opts = _option_set(run)
    assert "em" in opts
    assert "fm" not in opts

    run.choose("em").finish()
    assert _pos(state, "hero_ignatia") == (0, 2, -2)
    assert _pos(state, "em") == (0, 0, 0)


@pytest.mark.effect_flow
def test_chaos_gate_blue_swaps_then_moves_that_unit() -> None:
    state = _swap_state("chaos_gate")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("fm").expect_input(InputRequestType.SELECT_HEX)  # move that unit (optional)
    run.choose({"q": 1, "r": 0, "s": -1}).finish()

    assert _pos(state, "hero_ignatia") == (2, 0, -2)
    assert _pos(state, "fm") == (1, 0, -1)  # swapped to origin, then moved 1


@pytest.mark.effect_flow
def test_chaos_gate_orange_swaps_then_moves_self() -> None:
    state = _swap_state("chaos_gate")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_UNIT)
    run.choose("em").expect_input(InputRequestType.SELECT_HEX)  # move yourself (optional)
    run.choose({"q": 0, "r": 1, "s": -1}).finish()

    assert _pos(state, "em") == (0, 0, 0)
    assert _pos(state, "hero_ignatia") == (0, 1, -1)  # swapped to em's spot, then moved 1


@pytest.mark.effect_flow
def test_chaos_gate_blue_optional_move_under_equilibrium_still_works() -> None:
    # Regression: Equilibrium gating must not clobber the optional move's own gate.
    state = _swap_state("chaos_gate")
    _set_coin(state, "ORANGE")
    _enable_equilibrium(state)

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_NUMBER)
    run.choose(1).expect_input(InputRequestType.SELECT_UNIT)  # 1 = Blue
    run.choose("fm").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 1, "r": 0, "s": -1}).finish()

    assert _pos(state, "fm") == (1, 0, -1)


# =============================================================================
# F7 — Path / Magma (path_of_ashes r3 N2 / path_of_cinders r4 N3 / path_of_flames r4 N4)
#   blue  : move up to N in a straight line; place a Magma in each empty space
#           moved through, or out of (origin included, destination excluded)
#   orange: place up to N Magma tokens in radius
# =============================================================================


def _add_magma_pool(state, count: int = 4) -> None:
    from goa2.domain.models import Token, TokenType

    state.token_pool[TokenType.MAGMA] = []
    for i in range(count):
        tok = Token(id=f"magma_pool_{i}", name="Magma", token_type=TokenType.MAGMA)
        state.register_entity(tok)
        state.token_pool[TokenType.MAGMA].append(tok)


def _magma_at(state, coord) -> bool:
    from goa2.domain.hex import Hex
    from goa2.domain.models import Token, TokenType

    h = Hex(q=coord[0], r=coord[1], s=coord[2])
    for eid, loc in state.entity_locations.items():
        if loc == h:
            ent = state.get_entity(eid)
            if isinstance(ent, Token) and ent.token_type == TokenType.MAGMA:
                return True
    return False


def _path_state(card_id: str):
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", card_id))
        .with_actor("hero_ignatia")
        .build()
    )
    _add_magma_pool(state)
    return state


@pytest.mark.effect_flow
def test_path_of_ashes_blue_lays_a_magma_trail_origin_included_dest_excluded() -> None:
    state = _path_state("path_of_ashes")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 2, "r": 0, "s": -2}).finish()  # move 2 in a straight line

    assert _pos(state, "hero_ignatia") == (2, 0, -2)
    assert _magma_at(state, (0, 0, 0))  # moved out of
    assert _magma_at(state, (1, 0, -1))  # moved through
    assert not _magma_at(state, (2, 0, -2))  # destination excluded


@pytest.mark.effect_flow
def test_path_of_ashes_does_not_replace_crossed_mine_with_magma() -> None:
    from goa2.domain.hex import Hex
    from goa2.domain.models import Token, TokenType
    from goa2.domain.types import BoardEntityID, HeroID

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(4))
        .red_hero(
            "hero_ignatia",
            at=(0, 0, 0),
            current_card=hero_card("Ignatia", "path_of_ashes"),
        )
        .blue_hero("mine_owner", at=(0, 3, -3))
        .with_actor("hero_ignatia")
        .build()
    )
    _add_magma_pool(state)
    _set_coin(state, "BLUE")
    mine = Token(
        id=BoardEntityID("mine_1"),
        name="Mine",
        token_type=TokenType.MINE_DUD,
        owner_id=HeroID("mine_owner"),
        is_passable=True,
    )
    state.token_pool[TokenType.MINE_DUD] = [mine]
    state.register_entity(mine, "token")
    state.place_entity(mine.id, Hex(q=1, r=0, s=-1))

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=2, r=0, s=-2))
    run.finish()

    assert _pos(state, "hero_ignatia") == (2, 0, -2)
    assert state.get_position("mine_1") is None
    assert _magma_at(state, (0, 0, 0))
    assert not _magma_at(state, (1, 0, -1))


@pytest.mark.effect_flow
def test_path_of_ashes_places_no_trail_when_move_fails() -> None:
    from goa2.domain.hex import Hex
    from goa2.domain.models import Token, TokenType
    from goa2.domain.types import BoardEntityID
    from goa2.engine.handler import process_stack, push_steps
    from goa2.engine.steps import MoveUnitStep, PlaceTokenTrailStep

    state = _path_state("path_of_ashes")
    blocker = Token(
        id=BoardEntityID("late_blocker"),
        name="Late blocker",
        token_type=TokenType.SMOKE_BOMB,
    )
    state.register_entity(blocker, "token")
    state.place_entity(blocker.id, Hex(q=2, r=0, s=-2))
    state.execution_context["origin"] = Hex(q=0, r=0, s=0)
    state.execution_context["destination"] = Hex(q=2, r=0, s=-2)
    push_steps(
        state,
        [
            MoveUnitStep(
                unit_id="hero_ignatia",
                destination_key="destination",
                range_val=2,
                force_straight_line=True,
                success_output_key="move_succeeded",
            ),
            PlaceTokenTrailStep(
                token_type=TokenType.MAGMA,
                origin_hex_key="origin",
                dest_key="destination",
                active_if_key="move_succeeded",
            ),
        ],
    )
    process_stack(state)

    assert _pos(state, "hero_ignatia") == (0, 0, 0)
    assert not _magma_at(state, (0, 0, 0))
    assert not _magma_at(state, (1, 0, -1))


@pytest.mark.effect_flow
def test_path_of_ashes_blue_moving_zero_places_nothing() -> None:
    state = _path_state("path_of_ashes")
    _set_coin(state, "BLUE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.skip().finish()  # move up to 2 -> choose 0

    assert _pos(state, "hero_ignatia") == (0, 0, 0)
    assert not _magma_at(state, (0, 0, 0))
    assert not _magma_at(state, (1, 0, -1))


@pytest.mark.effect_flow
def test_path_blue_supply_short_removes_board_magma_before_trail() -> None:
    # Two pool Magma tokens sit off-path; free supply is 2 < 3 trail hexes.
    # The shortfall removal is prompted BEFORE any trail placement, offering
    # only the pre-existing board tokens (never this trail's own tokens).
    state = _path_state("path_of_cinders")
    _set_coin(state, "BLUE")
    from goa2.domain.hex import Hex

    state.place_entity("magma_pool_0", Hex(q=0, r=2, s=-2))
    state.place_entity("magma_pool_1", Hex(q=0, r=-2, s=2))

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 3, "r": 0, "s": -3}).expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN)
    assert _option_set(run) == {"magma_pool_0", "magma_pool_1"}
    run.choose("magma_pool_0").finish()

    assert _pos(state, "hero_ignatia") == (3, 0, -3)
    for coord in ((0, 0, 0), (1, 0, -1), (2, 0, -2)):
        assert _magma_at(state, coord)
    assert not _magma_at(state, (3, 0, -3))  # destination excluded
    assert not _magma_at(state, (0, 2, -2))  # removed for the shortfall
    assert _magma_at(state, (0, -2, 2))  # the other board token stays


@pytest.mark.effect_flow
def test_path_of_ashes_orange_places_up_to_two_magma_in_radius() -> None:
    state = _path_state("path_of_ashes")
    _set_coin(state, "ORANGE")

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 1, "r": 0, "s": -1}).expect_input(InputRequestType.SELECT_HEX)
    run.choose({"q": 0, "r": 1, "s": -1}).finish()

    assert _magma_at(state, (1, 0, -1))
    assert _magma_at(state, (0, 1, -1))


# =============================================================================
# F8 — Equilibrium (Silver): "This round: each time you perform or repeat a
# primary action, you may apply either blue or orange text, regardless of the
# coin." Playing it raises a THIS_ROUND flag that every branch card reads.
# =============================================================================


@pytest.mark.effect_flow
def test_equilibrium_card_creates_this_round_flag() -> None:
    from goa2.domain.models.effect import DurationType, EffectType

    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(2))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "equilibrium"))
        .with_actor("hero_ignatia")
        .build()
    )

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()

    flags = [e for e in state.active_effects if e.effect_type == EffectType.EQUILIBRIUM]
    assert len(flags) == 1
    assert flags[0].source_id == "hero_ignatia"
    assert flags[0].duration == DurationType.THIS_ROUND


@pytest.mark.effect_flow
def test_equilibrium_played_then_branch_card_offers_choice_same_round() -> None:
    # End-to-end: play the real Equilibrium card, then a branch card this round
    # is driven off the flag it raised (no test-only activation).
    state = (
        EffectScenarioBuilder()
        .with_hexes(_hex_disk(3))
        .red_hero("hero_ignatia", at=(0, 0, 0), current_card=hero_card("Ignatia", "equilibrium"))
        .blue_minion("am", at=(1, 0, -1))
        .with_actor("hero_ignatia")
        .build()
    )

    run = run_card(state, "hero_ignatia")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()  # Equilibrium raises the flag

    # Now Ignatia plays Chaos Bolt this round on an orange coin.
    state.get_hero("hero_ignatia").current_turn_card = hero_card("Ignatia", "chaos_bolt")
    _set_coin(state, "ORANGE")

    run2 = run_card(state, "hero_ignatia")
    run2.expect_input(InputRequestType.CHOOSE_ACTION)
    # Equilibrium is active -> she is prompted to choose the side, not read the coin.
    run2.choose("ATTACK").expect_input(InputRequestType.SELECT_NUMBER)
