import pytest

import goa2.scripts.dodger_effects  # noqa: F401
from goa2.domain.input import InputRequestType
from goa2.domain.models import CardColor, CardState, CardTier
from goa2.domain.models.card import Card
from goa2.domain.models.effect import EffectType
from goa2.domain.models.enums import ActionType, StatType
from goa2.domain.types import BoardEntityID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import AttackSequenceStep, PerformPrimaryActionStep

from ..assertions import assert_valid_options
from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _option_ids(run) -> set[str]:
    """Return the set of unit IDs exposed as valid options in the current request."""
    assert run.latest_request is not None
    ids: set[str] = set()
    for option in run.latest_request.options:
        if hasattr(option, "id"):
            ids.add(str(option.id))
        else:
            ids.add(str(option))
    return ids


def _dummy_discard_card(card_id: str = "burned_card") -> Card:
    """A plain resolved card that can be seeded into an enemy hero's discard pile
    to satisfy `CardsInContainerFilter(container=DISCARD, min_cards=1)`."""
    return Card(
        id=card_id,
        name=card_id.replace("_", " ").title(),
        tier=CardTier.UNTIERED,
        color=CardColor.SILVER,
        initiative=5,
        primary_action=ActionType.SKILL,
        primary_action_value=None,
        secondary_actions={},
        is_ranged=False,
        effect_id="",
        effect_text="",
        is_facedown=False,
        state=CardState.RESOLVED,
    )


def _setup_dodger(hexes, *, with_ultimate: bool = True):
    """Level-8 Dodger about to play Darkest Ritual.

    Level 8 satisfies the card's "If you have your Ultimate" clause. When
    ``with_ultimate`` is set, the real Tide of Darkness ultimate is assigned so
    this is a faithful level-8 Dodger (and its passive override is in effect for
    spawn-point counting).
    """
    state = (
        EffectScenarioBuilder()
        .with_hexes(hexes)
        .red_hero(
            "hero_dodger",
            at=(0, 0, 0),
            current_card=hero_card("Dodger", "darkest_ritual"),
        )
        .with_actor("hero_dodger")
        .build()
    )
    dodger = state.get_hero("hero_dodger")
    dodger.level = 8
    if with_ultimate:
        dodger.ultimate_card = hero_card("Dodger", "tide_of_darkness")
    return state, dodger


@pytest.mark.effect_flow
def test_darkest_ritual_grants_ultimate_item_even_without_coins() -> None:
    """Darkest Ritual's two clauses are independent.

    Card text: "If there are 2 or more empty spawn points in radius ..., gain 2
    coins. If you have your Ultimate, gain an Attack item."

    With fewer than 2 empty spawn points in radius (here a 2-hex board → only 1
    free hex), the coin clause is skipped, but a level-8 Dodger must STILL gain
    the Attack item. This currently fails because the GainItemStep reads
    context["self"], which is only set inside the coin branch.
    """
    # 2-hex board: hero on one hex, exactly one free hex → < 2 empty spawn points
    state, dodger = _setup_dodger([(0, 0, 0), (1, 0, -1)])
    gold_before = dodger.gold

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()

    # Coin clause correctly skipped (< 2 empty spawn points)...
    assert dodger.gold == gold_before
    # ...but the independent Ultimate item clause must still fire.
    assert dodger.items.get(StatType.ATTACK, 0) == 1


@pytest.mark.effect_flow
def test_darkest_ritual_grants_item_when_coins_also_gained() -> None:
    """Control: with 2+ empty spawn points, both clauses fire.

    The item only comes through here because the coin branch sets
    context["self"] first — exactly the coupling the bug exposes.
    """
    # Roomy board: many free hexes in radius → 2+ empty spawn points (override)
    state, dodger = _setup_dodger([(q, 0, -q) for q in range(5)])
    gold_before = dodger.gold

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION)
    run.choose("SKILL").finish()

    assert dodger.gold == gold_before + 2
    assert dodger.items.get(StatType.ATTACK, 0) == 1


# =============================================================================
# Tide of Darkness while defending
# =============================================================================


def _defending_dodger(*, with_ultimate: bool):
    """Level-8 Dodger about to block a 5-damage attack with Shield of Decay.

    The board carries no spawn points and no battle zone, so the card's "+2 if
    there are 2 or more empty spawn points in radius" clause can only hold via
    Tide of Darkness, which counts every free space as a spawn point.
    """
    state = (
        EffectScenarioBuilder()
        .with_hexes(
            [(q, r, -q - r) for q in range(-1, 3) for r in range(-1, 3) if -1 <= -q - r <= 2]
        )
        .red_hero("hero_attacker", at=(0, 0, 0))
        .blue_hero("hero_dodger", at=(1, 0, -1))
        .with_actor("hero_attacker")
        .build()
    )
    dodger = state.get_hero("hero_dodger")
    dodger.level = 8
    if with_ultimate:
        dodger.ultimate_card = hero_card("Dodger", "tide_of_darkness")
    shield = hero_card("Dodger", "shield_of_decay")
    shield.is_facedown = False
    dodger.hand = [shield]
    return state


def _block_five_damage(state):
    """Drive a plain 5-damage melee attack onto Dodger, blocked by Shield of Decay."""
    push_steps(state, [AttackSequenceStep(damage=5, range_val=1)])
    result = process_stack(state)
    assert result.input_request.request_type.value == "SELECT_UNIT"
    state.execution_stack[-1].pending_input = {"selection": "hero_dodger"}
    result = process_stack(state)
    assert result.input_request.request_type.value == "SELECT_CARD_OR_PASS"
    state.execution_stack[-1].pending_input = {"selection": "shield_of_decay"}
    process_stack(state)
    return BoardEntityID("hero_dodger") in state.entity_locations


@pytest.mark.effect_flow
def test_tide_of_darkness_counts_while_dodger_defends() -> None:
    """Blocking with a primary DEFENSE card is performing an action.

    Shield of Decay has Defense 3; the +2 clause is what survives a 5-damage
    attack. Tide of Darkness must satisfy that clause on Dodger's behalf even
    though the current actor during a defense is the attacker.
    """
    assert _block_five_damage(_defending_dodger(with_ultimate=True))


@pytest.mark.effect_flow
def test_defense_bonus_needs_the_clause_without_tide_of_darkness() -> None:
    """Control: no Tide, no spawn points, so the +2 never fires and Defense 3 falls.

    Without this, the test above could pass because the bonus applies
    unconditionally rather than because Tide satisfied the clause.
    """
    assert not _block_five_damage(_defending_dodger(with_ultimate=False))


@pytest.mark.effect_contract
def test_enfeeblement_creates_one_instance_per_payload() -> None:
    """The card's two clauses are two payloads of a single active effect."""
    state = (
        EffectScenarioBuilder()
        .small_arena()
        .red_hero(
            "hero_dodger",
            at=(0, 0, 0),
            current_card=hero_card("Dodger", "enfeeblement"),
        )
        .with_actor("hero_dodger")
        .build()
    )

    run_card(state, "hero_dodger").expect_input(InputRequestType.CHOOSE_ACTION).choose(
        "SKILL"
    ).finish()

    bound = [e for e in state.active_effects if e.source_card_id == "enfeeblement"]
    assert {e.effect_type for e in bound} == {
        EffectType.AREA_STAT_MODIFIER,
        EffectType.REPEAT_PREVENTION,
    }


@pytest.mark.effect_contract
def test_repeating_enfeeblement_does_not_duplicate_its_effect() -> None:
    """Only one instance of an active effect per card can be active."""
    state = (
        EffectScenarioBuilder()
        .small_arena()
        .red_hero(
            "hero_dodger",
            at=(0, 0, 0),
            current_card=hero_card("Dodger", "enfeeblement"),
        )
        .with_actor("hero_dodger")
        .build()
    )

    run_card(state, "hero_dodger").expect_input(InputRequestType.CHOOSE_ACTION).choose(
        "SKILL"
    ).finish()
    before = [e.id for e in state.active_effects]

    state.execution_context["selected_card"] = "enfeeblement"
    push_steps(state, [PerformPrimaryActionStep(hero_id="hero_dodger")])
    process_stack(state)

    assert [e.id for e in state.active_effects] == before


# =============================================================================
# Finger family: flattened single SELECT_UNIT
# =============================================================================
#
# Card text (Littlefinger / Finger of Death):
#   "Choose one —
#      • Target a unit adjacent to you.
#      • Target a hero in range who has one or more cards in the discard."
#
# The two bullets are *targeting constraints*, not a mandatory mode dial: the
# player is picking a target, and both bullets describe legal shapes for that
# single target. Flattening: after CHOOSE_ACTION the player must see ONE
# SELECT_UNIT whose valid options are (adjacent enemy units) or (in-range enemy
# heroes with 1+ discard). No SELECT_NUMBER mode prompt.


def _finger_family_state(effect_card_id: str, range_value: int):
    """Board with one target of each qualifying shape and two near-miss decoys.

    Layout (works for range_value in {2, 3} without collisions):
      (0,0,0)   Dodger (RED, actor) with the given attack card
      (1,0,-1)  adjacent enemy minion (BLUE) — qualifies via "adjacent unit"
      (range,0,-range)       enemy hero with 1 card in discard — qualifies
                             via "hero in range with discard"
      (0,2,-2)               enemy hero with EMPTY discard pile — near miss:
                             non-adjacent (distance 2) and in range for both
                             range 2 and range 3, but discard is empty
      (range+1,0,-(range+1)) enemy hero with 1 card in discard — near miss:
                             has discard but out of range

    The no-discard decoy sits off the r=0 axis so its hex never coincides with
    the adjacent minion (at range=2) or the qualifying ranged hero (at
    range=3), while still being both a "hero" and "in range, non-adjacent".
    """
    axis_hexes = [(q, 0, -q) for q in range(range_value + 2)]
    off_axis_hexes = [(0, 1, -1), (0, 2, -2)]
    hexes = axis_hexes + off_axis_hexes

    hero_with_discard = "enemy_hero_with_discard"
    hero_no_discard = "enemy_hero_no_discard"
    hero_out_of_range = "enemy_hero_out_of_range"

    state = (
        EffectScenarioBuilder()
        .with_hexes(hexes)
        .red_hero(
            "hero_dodger",
            at=(0, 0, 0),
            current_card=hero_card("Dodger", effect_card_id),
        )
        .blue_minion("adjacent_minion", at=(1, 0, -1))
        .blue_hero(hero_no_discard, at=(0, 2, -2))
        .blue_hero(hero_with_discard, at=(range_value, 0, -range_value))
        .blue_hero(hero_out_of_range, at=(range_value + 1, 0, -(range_value + 1)))
        .with_actor("hero_dodger")
        .build()
    )
    # Seed discard piles so the "hero in range with 1+ discard" clause is
    # discriminating: only enemy_hero_with_discard and enemy_hero_out_of_range
    # have a card there.
    state.get_hero("enemy_hero_with_discard").discard_pile = [_dummy_discard_card("in_range_burn")]
    state.get_hero("enemy_hero_out_of_range").discard_pile = [
        _dummy_discard_card("out_of_range_burn")
    ]
    state.get_hero("enemy_hero_no_discard").discard_pile = []
    return state


@pytest.mark.effect_contract
def test_littlefinger_of_death_offers_single_combined_select_unit() -> None:
    """Littlefinger of Death (range 2) must flatten mode selection into one
    SELECT_UNIT that unions both targeting bullets.

    Currently RED because the effect emits a SELECT_NUMBER mode prompt before
    any target selection; the flattened behavior removes that prompt.
    """
    state = _finger_family_state("littlefinger_of_death", range_value=2)

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

    # The very next prompt must be the combined target picker, not a mode dial.
    run.expect_input(InputRequestType.SELECT_UNIT)

    assert_valid_options(
        run.latest_request,
        contains=[
            "adjacent_minion",  # adjacent unit — 1st bullet
            "enemy_hero_with_discard",  # hero in range 2 with discard — 2nd bullet
        ],
        excludes=[
            "enemy_hero_no_discard",  # in range but discard empty
            "enemy_hero_out_of_range",  # has discard but beyond range 2
            "hero_dodger",  # never target self
        ],
    )


@pytest.mark.effect_contract
def test_finger_of_death_offers_single_combined_select_unit() -> None:
    """Finger of Death (range 3) — same flattened contract as Littlefinger.

    The reach differs (3 vs. 2) but both cards share the same targeting
    grammar, so the union of legal targets is derived from the card's range.
    """
    state = _finger_family_state("finger_of_death", range_value=3)

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

    run.expect_input(InputRequestType.SELECT_UNIT)

    assert_valid_options(
        run.latest_request,
        contains=[
            "adjacent_minion",
            "enemy_hero_with_discard",
        ],
        excludes=[
            "enemy_hero_no_discard",
            "enemy_hero_out_of_range",
            "hero_dodger",
        ],
    )


@pytest.mark.effect_contract
def test_finger_family_does_not_emit_select_number_mode_prompt() -> None:
    """Regression guard: the flattened design MUST NOT emit SELECT_NUMBER.

    The prior (obsolete) design prompted the player to pick a mode (1: adjacent,
    2: ranged) before the target picker. This explicit anti-assertion catches
    any regression that reintroduces that prompt for either Finger variant.
    """
    for card_id, rng in (("littlefinger_of_death", 2), ("finger_of_death", 3)):
        state = _finger_family_state(card_id, range_value=rng)
        run = run_card(state, "hero_dodger")
        run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

        # Whatever the next input is, it must not be a SELECT_NUMBER mode dial.
        # We drive one more step and check the request type.
        result = process_stack(state)
        run.latest_request = result.input_request
        run.events.extend(result.events)
        assert run.latest_request is not None, f"Expected an input request for {card_id!r}"
        assert run.latest_request.request_type != InputRequestType.SELECT_NUMBER, (
            f"{card_id!r} must not emit a SELECT_NUMBER mode prompt after CHOOSE_ACTION; "
            f"got {run.latest_request.request_type!r}"
        )


# =============================================================================
# Dread Razor: direct target selection, range gated by empty spawn adjacency
# =============================================================================
#
# Card text: "Choose one —
#              • Target a unit adjacent to you.
#              • If you are adjacent to an empty spawn point in the battle zone,
#                target a unit in range."
#
# Flattened: NO mode prompt. Direct SELECT_UNIT after CHOOSE_ACTION. The card's
# range (2) is available only when Dodger is adjacent to an empty spawn point
# in the battle zone; otherwise the picker is restricted to range 1.


def _dread_razor_state(*, with_spawn_adjacent: bool):
    """Two-row arena that keeps a real melee target available in both branches.

      (0,0,0)   Dodger (RED, actor) with Dread Razor
      (1,0,-1)  spawn point (RED, HERO type) — empty when
                with_spawn_adjacent=True, absent otherwise
      (2,0,-2)  BLUE minion at range 2 — the discriminating target
      (3,0,-3)  BLUE minion at range 3 — always out of reach
      (0,1,-1)  BLUE minion adjacent to Dodger — always a legal melee target

    The adjacent minion at (0,1,-1) keeps the melee bullet non-empty even in
    the "no spawn adjacent" branch, so the SELECT_UNIT prompt is actually
    issued and the picker's option set is meaningful to inspect.

    When with_spawn_adjacent is True the (1,0,-1) hex hosts an empty spawn
    point in the active battle zone, unlocking the "unit in range" bullet.
    Otherwise (2,0,-2) must NOT appear in the picker's options.
    """
    from goa2.domain.models import TeamColor
    from goa2.domain.models.spawn import SpawnType

    builder = (
        EffectScenarioBuilder()
        .with_hexes([(q, 0, -q) for q in range(4)] + [(0, 1, -1)])
        .red_hero(
            "hero_dodger",
            at=(0, 0, 0),
            current_card=hero_card("Dodger", "dread_razor"),
        )
        .blue_minion("adjacent_minion", at=(0, 1, -1))
        .blue_minion("range_two_minion", at=(2, 0, -2))
        .blue_minion("range_three_minion", at=(3, 0, -3))
        .with_actor("hero_dodger")
    )
    if with_spawn_adjacent:
        builder = builder.spawn_point((1, 0, -1), team=TeamColor.RED, spawn_type=SpawnType.HERO)
    state = builder.build()
    if with_spawn_adjacent:
        # Put the spawn point's zone into the active battle zone list so
        # `_is_adjacent_to_empty_spawn_in_battle_zone` sees it.
        state.active_zone_id = "z1"
    return state


@pytest.mark.effect_contract
def test_dread_razor_without_spawn_adjacency_only_offers_adjacent_target() -> None:
    """No empty spawn adjacent → range 1 only.

    Currently RED because the effect emits a SELECT_NUMBER mode prompt when
    both bullets are available, and even in the melee-only branch the target
    picker is reached via that indirection rather than as the first prompt.
    Here `with_spawn_adjacent=False` collapses to melee-only, but the flattened
    contract still forbids any mode dial: we must land on SELECT_UNIT
    immediately after CHOOSE_ACTION.
    """
    state = _dread_razor_state(with_spawn_adjacent=False)

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

    run.expect_input(InputRequestType.SELECT_UNIT)

    options = _option_ids(run)
    # Range 2 minion must be excluded — the ranged bullet is unavailable.
    assert "range_two_minion" not in options, (
        "Dread Razor without empty-spawn adjacency must only offer adjacent "
        f"targets (range 1); got options={options!r}"
    )
    # And of course the range-3 minion is never reachable.
    assert "range_three_minion" not in options


@pytest.mark.effect_contract
def test_dread_razor_with_spawn_adjacency_offers_full_range() -> None:
    """Empty spawn adjacent in battle zone → range 2 target picker.

    The range-2 minion becomes selectable; the out-of-range range-3 minion
    stays excluded. No SELECT_NUMBER mode dial in between.
    """
    state = _dread_razor_state(with_spawn_adjacent=True)

    run = run_card(state, "hero_dodger")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

    run.expect_input(InputRequestType.SELECT_UNIT)

    assert_valid_options(
        run.latest_request,
        contains=[
            "adjacent_minion",  # melee bullet
            "range_two_minion",  # unlocked by the spawn-adjacency clause
        ],
        excludes=[
            "range_three_minion",  # still out of the card's range 2
            "hero_dodger",  # never target self
        ],
    )


@pytest.mark.effect_contract
def test_dread_razor_does_not_emit_select_number_mode_prompt() -> None:
    """Regression guard: neither branch of Dread Razor may emit SELECT_NUMBER.

    Historically the effect prompted a 1/2 mode dial when both bullets were
    live; the flattened contract collapses that into a single SELECT_UNIT that
    already encodes the correct range.
    """
    for adjacent in (False, True):
        state = _dread_razor_state(with_spawn_adjacent=adjacent)
        run = run_card(state, "hero_dodger")
        run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")

        result = process_stack(state)
        run.latest_request = result.input_request
        run.events.extend(result.events)
        assert (
            run.latest_request is not None
        ), f"Expected input after CHOOSE_ACTION (spawn_adjacent={adjacent})"
        assert run.latest_request.request_type != InputRequestType.SELECT_NUMBER, (
            "Dread Razor must not emit a SELECT_NUMBER mode prompt "
            f"(spawn_adjacent={adjacent}); got {run.latest_request.request_type!r}"
        )
