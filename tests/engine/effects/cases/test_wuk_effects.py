"""Wuk effect flow tests."""

from __future__ import annotations

import pytest

import goa2.data.heroes.wuk
import goa2.scripts.wuk_effects  # noqa: F401  (register effects)
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType
from goa2.domain.models import MinionType, Token, TokenType
from goa2.domain.models.effect import EffectType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import EndPhaseStep

from ..builders import EffectScenarioBuilder, hero_card
from ..runner import run_card


def _add_tree(state: GameState, token_id: str, at: Hex) -> None:
    tree = Token(id=BoardEntityID(token_id), name="Tree", token_type=TokenType.TREE)
    state.register_entity(tree, "token")
    state.place_entity(token_id, at)


def _add_tree_pool(state: GameState, count: int = 3) -> None:
    """Register `count` unplaced Tree tokens into the supply pool."""
    state.token_pool[TokenType.TREE] = []
    for i in range(count):
        tree = Token(id=BoardEntityID(f"tree_pool_{i}"), name="Tree", token_type=TokenType.TREE)
        state.register_entity(tree, "token")
        state.token_pool[TokenType.TREE].append(tree)


def _placed_trees(state: GameState) -> list[str]:
    return [
        str(t.id)
        for t in state.token_pool.get(TokenType.TREE, [])
        if BoardEntityID(str(t.id)) in state.entity_locations
    ]


def _wuk_at_origin(card_id: str) -> EffectScenarioBuilder:
    return (
        EffectScenarioBuilder()
        .line_board(5)
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", card_id))
        .with_actor("hero_wuk")
    )


def _drive_canopy_swap(
    state: GameState,
    *,
    tree_id: str,
    participant_id: str,
    expect_participant_options_superset: set[str] | None = None,
    forbidden_participant_options: set[str] | None = None,
) -> None:
    """Order-neutral driver for one Into-the-Canopy swap.

    Consumes exactly one SELECT_UNIT_OR_TOKEN (tree) prompt and exactly one
    SELECT_UNIT (participant) prompt in whichever order they occur.

    Explicitly rejects SELECT_NUMBER and any unexpected prompt at any point
    before both selections are consumed.

    After both selections are consumed, the driver peeks at the next pending
    request WITHOUT consuming it: if it is a SELECT_OPTION (e.g. the Treetop
    Ride repeat prompt), the driver returns immediately, leaving the request
    on top of the stack so the caller's own expect_input(SELECT_OPTION) can
    handle it. Any other pending prompt at that point is treated as
    unexpected and raises.

    Since a peek does not set pending_input on the waiting step, the step
    remains on the execution stack and re-issues the same request on the
    caller's next process_stack call (its pending_request_id is preserved).

    Optionally inspects the participant prompt's options when it appears
    (asserting a required subset present, and forbidden ids absent).
    """
    tree_chosen = False
    participant_chosen = False
    saw_participant_prompt = False

    for _ in range(16):
        result = process_stack(state)
        req = result.input_request
        if req is None:
            break

        request_type = req.request_type
        both_done = tree_chosen and participant_chosen

        if both_done:
            # Peek only: don't consume. Hand SELECT_OPTION back to the caller
            # (Treetop Ride repeat gate). Any other prompt here is unexpected.
            if request_type == InputRequestType.SELECT_OPTION:
                return
            raise AssertionError(
                "Canopy swap completed both selections but next prompt is "
                f"not SELECT_OPTION; got type={request_type!r}, "
                f"prompt={req.to_dict()!r}"
            )

        if request_type == InputRequestType.SELECT_NUMBER:
            raise AssertionError(
                f"Canopy swap must not use SELECT_NUMBER; got prompt={req.to_dict()!r}"
            )
        if request_type == InputRequestType.SELECT_UNIT_OR_TOKEN:
            assert (
                not tree_chosen
            ), f"Tree prompt appeared twice in one canopy swap; latest={req.to_dict()!r}"
            state.execution_stack[-1].pending_input = {"selection": tree_id}
            tree_chosen = True
            continue
        if request_type == InputRequestType.SELECT_UNIT:
            assert (
                not participant_chosen
            ), f"Participant prompt appeared twice in one canopy swap; latest={req.to_dict()!r}"
            saw_participant_prompt = True
            option_ids = {str(option.id) for option in req.options}
            if expect_participant_options_superset is not None:
                missing = expect_participant_options_superset - option_ids
                assert not missing, (
                    f"Participant prompt missing required options {sorted(missing)!r}; "
                    f"got options={sorted(option_ids)!r}"
                )
            if forbidden_participant_options is not None:
                leaked = forbidden_participant_options & option_ids
                assert not leaked, (
                    f"Participant prompt exposes forbidden options {sorted(leaked)!r}; "
                    f"got options={sorted(option_ids)!r}"
                )
            state.execution_stack[-1].pending_input = {"selection": participant_id}
            participant_chosen = True
            continue
        raise AssertionError(
            f"Unexpected prompt during canopy swap: type={request_type!r}, "
            f"prompt={req.to_dict()!r}"
        )

    assert tree_chosen, "Canopy swap never asked for a Tree token"
    assert participant_chosen, "Canopy swap never asked for a participant"
    assert saw_participant_prompt, "Canopy swap never surfaced the participant prompt"


@pytest.mark.effect_flow
def test_toss_away_throws_adjacent_enemy_into_range() -> None:
    state = _wuk_at_origin("toss_away").blue_minion("blue_minion", at=(1, 0, -1)).build()

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("blue_minion")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    run.finish()

    assert state.entity_locations["blue_minion"] == Hex(q=3, r=0, s=-3)


@pytest.mark.effect_flow
def test_toss_away_throws_tree_token() -> None:
    state = _wuk_at_origin("toss_away").build()
    _add_tree(state, "tree_1", Hex(q=1, r=0, s=-1))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=2, r=0, s=-2))
    run.finish()

    assert state.entity_locations["tree_1"] == Hex(q=2, r=0, s=-2)


@pytest.mark.effect_flow
def test_monstrous_throw_repeats_on_second_target() -> None:
    state = (
        EffectScenarioBuilder()
        .small_arena()
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "monstrous_throw"))
        .with_actor("hero_wuk")
        .blue_minion("m1", at=(1, 0, -1))
        .blue_minion("m2", at=(0, 1, -1))
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    # First throw
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("m1")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=4, r=0, s=-4))
    # Repeat -> yes, second throw
    run.expect_input(InputRequestType.SELECT_OPTION).choose("YES")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("m2")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    run.finish()

    assert state.entity_locations["m1"] == Hex(q=4, r=0, s=-4)
    assert state.entity_locations["m2"] == Hex(q=3, r=0, s=-3)


@pytest.mark.effect_flow
def test_into_the_canopy_swap_self_with_tree() -> None:
    # Flattened, order-neutral: the effect issues one Tree selection and one
    # combined participant selection in EITHER order; no SELECT_NUMBER
    # mode-picker appears. The final swap must land regardless of order.
    state = _wuk_at_origin("into_the_canopy").build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    _drive_canopy_swap(state, tree_id="tree_1", participant_id="hero_wuk")

    assert state.entity_locations["hero_wuk"] == Hex(q=2, r=0, s=-2)
    assert state.entity_locations["tree_1"] == Hex(q=0, r=0, s=0)


@pytest.mark.effect_flow
def test_into_the_canopy_swap_friendly_with_tree() -> None:
    # Order-neutral: the participant prompt (whenever it appears) accepts a
    # friendly unit directly; no SELECT_NUMBER precedes or follows.
    state = _wuk_at_origin("into_the_canopy").red_minion("ally", at=(1, 0, -1)).build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    _drive_canopy_swap(state, tree_id="tree_1", participant_id="ally")

    assert state.entity_locations["ally"] == Hex(q=2, r=0, s=-2)
    assert state.entity_locations["tree_1"] == Hex(q=1, r=0, s=-1)


@pytest.mark.effect_flow
def test_into_the_canopy_offers_self_and_friendly_together() -> None:
    # Whenever the combined participant prompt appears (either before or after
    # the tree pick), it must offer Wuk AND a friendly unit as options in the
    # SAME prompt. Enemy units in radius must not appear as participants.
    state = (
        _wuk_at_origin("into_the_canopy")
        .red_minion("ally", at=(1, 0, -1))
        .blue_minion("enemy", at=(0, 1, -1))
        .build()
    )
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    _drive_canopy_swap(
        state,
        tree_id="tree_1",
        participant_id="ally",
        expect_participant_options_superset={"hero_wuk", "ally"},
        forbidden_participant_options={"enemy"},
    )

    # Swap outcome still lands regardless of prompt order.
    assert state.entity_locations["ally"] == Hex(q=2, r=0, s=-2)
    assert state.entity_locations["tree_1"] == Hex(q=1, r=0, s=-1)


@pytest.mark.effect_flow
def test_treetop_ride_swaps_twice() -> None:
    # Treetop Ride keeps the per-iteration NUMBER mode-picker (mode 1 = swap
    # self with a Tree token) so the outer MayRepeatNTimesStep SELECT_OPTION
    # confirmation interleaves cleanly between iterations. The Into-the-Canopy
    # flattening deliberately does not extend to Treetop Ride's repeat flow.
    state = _wuk_at_origin("treetop_ride").build()
    _add_tree(state, "tree_1", Hex(q=1, r=0, s=-1))
    _add_tree(state, "tree_2", Hex(q=2, r=0, s=-2))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_OPTION).choose("YES")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(1)
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_OPTION).choose("YES")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(1)
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_2")
    run.finish()

    assert state.entity_locations["hero_wuk"] == Hex(q=2, r=0, s=-2)
    assert state.entity_locations["tree_2"] == Hex(q=1, r=0, s=-1)
    assert state.entity_locations["tree_1"] == Hex(q=0, r=0, s=0)


def _exclusion_effects(state: GameState):
    return [e for e in state.active_effects if e.effect_type == EffectType.MINION_BATTLE_EXCLUSION]


@pytest.mark.effect_flow
def test_claim_dominance_creates_exclusion_cap_1() -> None:
    state = _wuk_at_origin("claim_dominance").build()
    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.finish()

    effects = _exclusion_effects(state)
    assert len(effects) == 1
    assert effects[0].max_value == 1
    assert effects[0].source_id == "hero_wuk"
    assert effects[0].is_active is True


@pytest.mark.effect_flow
def test_assert_dominance_creates_exclusion_cap_2() -> None:
    state = _wuk_at_origin("assert_dominance").build()
    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.finish()

    effects = _exclusion_effects(state)
    assert len(effects) == 1
    assert effects[0].max_value == 2


@pytest.mark.effect_flow
def test_claim_dominance_excludes_minion_in_real_end_phase_battle() -> None:
    # Wuk (RED) in the active battle zone, adjacent to one BLUE minion.
    # Zone counts: RED 1 minion, BLUE 1 minion -> without dominance the battle
    # is a tie (nobody removed). Claim Dominance makes the adjacent BLUE minion
    # not count -> BLUE 0 -> BLUE loses its minion. This drives the REAL
    # EndPhaseStep (THIS_ROUND expiry + lazy minion battle), so it catches the
    # effect expiring before the battle.
    state = (
        EffectScenarioBuilder()
        .with_hexes([(0, 0, 0), (1, 0, -1), (0, 1, -1), (2, 0, -2)])
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "claim_dominance"))
        .with_actor("hero_wuk")
        .red_minion("r1", at=(2, 0, -2))
        .blue_minion("b1", at=(1, 0, -1))  # adjacent to Wuk, in the zone
        .build()
    )
    state.active_zone_id = "z1"

    # Play the card (creates the exclusion effect).
    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.finish()

    # Resolve the real End Phase (battle included).
    push_steps(state, [EndPhaseStep()])
    result = process_stack(state)
    assert result.input_request is None  # auto-resolves (no choice needed)

    # BLUE minion excluded -> BLUE loses it; RED minion survives.
    assert state.entity_locations.get("b1") is None
    assert state.entity_locations.get("r1") is not None
    # And the exclusion effect is cleaned up by end-of-round (no leak into next round).
    assert not _exclusion_effects(state)


@pytest.mark.effect_flow
def test_gifts_of_nature_removes_tree_and_retrieves() -> None:
    state = _wuk_at_origin("gifts_of_nature").build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    wuk = state.get_hero("hero_wuk")
    discarded = hero_card("Wuk", "tree_slam")
    wuk.discard_pile = [discarded]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_CARD).choose("tree_slam")
    run.finish()

    assert state.entity_locations.get("tree_1") is None  # tree removed (cost)
    assert discarded in wuk.hand
    assert discarded not in wuk.discard_pile


@pytest.mark.effect_flow
def test_gifts_of_nature_requires_tree_in_radius() -> None:
    # No tree in radius means the skill is unavailable; nothing is retrieved.
    state = _wuk_at_origin("gifts_of_nature").build()
    wuk = state.get_hero("hero_wuk")
    discarded = hero_card("Wuk", "tree_slam")
    wuk.discard_pile = [discarded]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).expect_option_absent("SKILL")

    assert discarded in wuk.discard_pile  # not retrieved (no tree to remove)


@pytest.mark.effect_flow
def test_tree_of_plenty_friendly_hero_retrieves() -> None:
    # After the common Tree removal there is no numeric mode-picker. A single
    # optional beneficiary selector directly picks the friendly hero, who then
    # chooses which discarded card to retrieve.
    state = _wuk_at_origin("tree_of_plenty").red_hero("ally", at=(1, 0, -1)).build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    ally = state.get_hero("ally")
    ally_card = hero_card("Wuk", "trample")
    ally.discard_pile = [ally_card]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("ally")
    run.expect_input(InputRequestType.SELECT_CARD).choose("trample")
    run.finish()

    assert ally_card in ally.hand
    assert ally_card not in ally.discard_pile


@pytest.mark.effect_flow
def test_tree_of_plenty_self_retrieves_via_unified_beneficiary_selector() -> None:
    # Wuk himself is a valid beneficiary in the unified selector when he has a
    # discard; no numeric mode prompt precedes the beneficiary pick.
    state = _wuk_at_origin("tree_of_plenty").build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    wuk = state.get_hero("hero_wuk")
    self_card = hero_card("Wuk", "tree_slam")
    wuk.discard_pile = [self_card]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("hero_wuk")
    run.expect_input(InputRequestType.SELECT_CARD).choose("tree_slam")
    run.finish()

    assert self_card in wuk.hand
    assert self_card not in wuk.discard_pile


@pytest.mark.effect_flow
def test_tree_of_plenty_offers_self_and_friendly_hero_together() -> None:
    # Single beneficiary selector offers Wuk AND a friendly hero (both with
    # discards) as options in the same prompt. Friendly heroes WITHOUT a
    # discard, and enemy heroes, are excluded.
    state = (
        _wuk_at_origin("tree_of_plenty")
        .red_hero("ally_with_discard", at=(1, 0, -1))
        .red_hero("ally_empty", at=(0, 1, -1))
        .blue_hero("enemy", at=(-1, 1, 0))
        .build()
    )
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    wuk = state.get_hero("hero_wuk")
    wuk.discard_pile = [hero_card("Wuk", "tree_slam")]
    ally_with_discard = state.get_hero("ally_with_discard")
    ally_with_discard.discard_pile = [hero_card("Wuk", "trample")]
    # ally_empty has NO discard pile entries.

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_UNIT)

    option_ids = {str(option.id) for option in run.latest_request.options}
    assert {"hero_wuk", "ally_with_discard"} <= option_ids
    assert "ally_empty" not in option_ids
    assert "enemy" not in option_ids


@pytest.mark.effect_flow
def test_tree_of_plenty_beneficiary_selector_is_optional() -> None:
    # The unified beneficiary selector is optional: skipping it after the tree
    # removal ends the effect without retrieving any card, even though Wuk
    # himself and a friendly hero are eligible beneficiaries.
    state = _wuk_at_origin("tree_of_plenty").red_hero("ally", at=(1, 0, -1)).build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    wuk = state.get_hero("hero_wuk")
    self_card = hero_card("Wuk", "tree_slam")
    wuk.discard_pile = [self_card]
    ally = state.get_hero("ally")
    ally_card = hero_card("Wuk", "trample")
    ally.discard_pile = [ally_card]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_UNIT).skip()
    run.finish()

    # Nothing retrieved, tree cost still paid.
    assert state.entity_locations.get("tree_1") is None
    assert self_card in wuk.discard_pile
    assert ally_card in ally.discard_pile


@pytest.mark.effect_flow
def test_abundance_retrieves_both_self_and_friendly() -> None:
    state = _wuk_at_origin("abundance").red_hero("ally", at=(1, 0, -1)).build()
    _add_tree(state, "tree_1", Hex(q=2, r=0, s=-2))
    wuk = state.get_hero("hero_wuk")
    self_card = hero_card("Wuk", "tree_slam")
    wuk.discard_pile = [self_card]
    ally = state.get_hero("ally")
    ally_card = hero_card("Wuk", "trample")
    ally.discard_pile = [ally_card]

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_CARD).choose("tree_slam")  # self
    run.expect_input(InputRequestType.SELECT_UNIT).choose("ally")
    run.expect_input(InputRequestType.SELECT_CARD).choose("trample")  # friendly
    run.finish()

    assert self_card in wuk.hand
    assert ally_card in ally.hand


@pytest.mark.effect_flow
def test_natures_protector_targets_unit_adjacent_to_tree() -> None:
    # Enemy minion at range 2, with a tree adjacent to it (not adjacent to Wuk-as-melee).
    state = _wuk_at_origin("natures_protector").blue_minion("victim", at=(2, 0, -2)).build()
    _add_tree(state, "tree_1", Hex(q=3, r=0, s=-3))  # adjacent to victim

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("victim")
    run.finish()

    # value-2 minion takes 5 -> defeated/removed
    assert state.entity_locations.get("victim") is None


@pytest.mark.effect_flow
def test_natures_protector_requires_tree_for_nonadjacent_unit() -> None:
    state = (
        _wuk_at_origin("natures_protector")
        .blue_minion("victim", at=(2, 0, -2))
        .blue_hero("adjacent_hero", at=(1, 0, -1))
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT)
    option_ids = {option.id for option in run.latest_request.options}
    assert "adjacent_hero" in option_ids
    assert "victim" not in option_ids


@pytest.mark.effect_flow
def test_natures_protector_targets_adjacent_hero() -> None:
    state = _wuk_at_origin("natures_protector").blue_hero("enemy", at=(1, 0, -1)).build()

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("enemy")
    run.expect_input(InputRequestType.SELECT_CARD_OR_PASS).choose("PASS")
    run.finish()

    combat = [e for e in run.events if e.event_type.value == "COMBAT_RESOLVED"]
    assert combat


@pytest.mark.effect_flow
def test_natures_protector_offers_both_target_categories_together() -> None:
    state = (
        _wuk_at_origin("natures_protector")
        .blue_hero("adjacent_hero", at=(1, 0, -1))
        .blue_minion("tree_target", at=(2, 0, -2))
        .build()
    )
    _add_tree(state, "tree_1", Hex(q=3, r=0, s=-3))

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT)

    option_ids = {option.id for option in run.latest_request.options}
    assert {"adjacent_hero", "tree_target"} <= option_ids


@pytest.mark.effect_flow
def test_natures_champion_attacks_both_targets() -> None:
    state = (
        _wuk_at_origin("natures_champion")
        .blue_hero("enemy", at=(1, 0, -1))
        .blue_minion("victim", at=(2, 0, -2))
        .build()
    )
    _add_tree(state, "tree_1", Hex(q=3, r=0, s=-3))  # adjacent to victim

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(1)  # hero first
    # mode 1: adjacent hero (optional)
    run.expect_input(InputRequestType.SELECT_UNIT).choose("enemy")
    run.expect_input(InputRequestType.SELECT_CARD_OR_PASS).choose("PASS")
    # mode 2: unit in range adjacent to tree (different target)
    run.expect_input(InputRequestType.SELECT_UNIT).choose("victim")
    run.finish()

    assert state.entity_locations.get("victim") is None  # 6 dmg defeats value-2 minion
    combat = [e for e in run.events if e.event_type.value == "COMBAT_RESOLVED"]
    assert len(combat) >= 2


@pytest.mark.effect_flow
def test_natures_champion_can_attack_tree_unit_before_hero() -> None:
    """ "In any order": the tree-anchored attack (mode 2) may resolve before the
    adjacent-hero attack (mode 1)."""
    state = (
        _wuk_at_origin("natures_champion")
        .blue_hero("enemy", at=(1, 0, -1))
        .blue_minion("victim", at=(2, 0, -2))
        .build()
    )
    _add_tree(state, "tree_1", Hex(q=3, r=0, s=-3))  # adjacent to victim

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(2)  # tree unit first
    # mode 2: unit in range adjacent to tree
    run.expect_input(InputRequestType.SELECT_UNIT).choose("victim")
    # mode 1: adjacent enemy hero (different target)
    run.expect_input(InputRequestType.SELECT_UNIT).choose("enemy")
    run.expect_input(InputRequestType.SELECT_CARD_OR_PASS).choose("PASS")
    run.finish()

    assert state.entity_locations.get("victim") is None  # 6 dmg defeats value-2 minion
    combat = [e for e in run.events if e.event_type.value == "COMBAT_RESOLVED"]
    assert len(combat) >= 2


@pytest.mark.effect_flow
def test_mystic_saplings_places_three_trees() -> None:
    state = _wuk_at_origin("mystic_saplings").build()
    _add_tree_pool(state, 3)

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=1, r=0, s=-1))
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=2, r=0, s=-2))
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    run.finish()

    assert len(_placed_trees(state)) == 3


@pytest.mark.effect_flow
def test_mystic_saplings_can_stop_early() -> None:
    state = _wuk_at_origin("mystic_saplings").build()
    _add_tree_pool(state, 3)

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=1, r=0, s=-1))
    run.expect_input(InputRequestType.SELECT_HEX).skip()  # stop after one
    run.finish()

    assert len(_placed_trees(state)) == 1


@pytest.mark.effect_flow
def test_tree_slam_mode1_attacks_adjacent_minion() -> None:
    state = _wuk_at_origin("tree_slam").blue_minion("victim", at=(1, 0, -1)).build()

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(1)
    run.expect_input(InputRequestType.SELECT_UNIT).choose("victim")
    run.finish()

    assert state.entity_locations.get("victim") is None  # 4 dmg defeats value-2 minion


@pytest.mark.effect_flow
def test_tree_slam_mode2_removes_tree_then_attacks_in_range() -> None:
    state = _wuk_at_origin("tree_slam").blue_minion("victim", at=(2, 0, -2)).build()
    _add_tree(state, "tree_1", Hex(q=1, r=0, s=-1))  # adjacent to Wuk

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(2)
    run.expect_input(InputRequestType.SELECT_UNIT_OR_TOKEN).choose("tree_1")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("victim")
    run.finish()

    assert state.entity_locations.get("tree_1") is None  # tree removed (cost)
    assert state.entity_locations.get("victim") is None  # attacked in range


@pytest.mark.effect_flow
def test_march_of_nature_places_tree_after_resolving_card() -> None:
    state = _wuk_at_origin("claim_dominance").build()
    wuk = state.get_hero("hero_wuk")
    wuk.level = 8
    wuk.ultimate_card = hero_card("Wuk", "march_of_nature")
    _add_tree_pool(state, 3)

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")  # claim_dominance
    run.expect_input(InputRequestType.CONFIRM_PASSIVE).choose("YES")  # March
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=2, r=0, s=-2))
    run.finish()

    assert len(_placed_trees(state)) == 1


@pytest.mark.effect_flow
def test_march_of_nature_can_decline() -> None:
    state = _wuk_at_origin("claim_dominance").build()
    wuk = state.get_hero("hero_wuk")
    wuk.level = 8
    wuk.ultimate_card = hero_card("Wuk", "march_of_nature")
    _add_tree_pool(state, 3)

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("SKILL")
    run.expect_input(InputRequestType.CONFIRM_PASSIVE).choose("NO")
    run.finish()

    assert len(_placed_trees(state)) == 0


@pytest.mark.effect_flow
def test_trample_crosses_hero_and_minion() -> None:
    state = (
        EffectScenarioBuilder()
        .line_board(6)
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "trample"))
        .with_actor("hero_wuk")
        .blue_hero("eh", at=(1, 0, -1))  # no cards -> defeated by discard-or-defeat
        .blue_minion("em", at=(2, 0, -2))
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("MOVEMENT")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(2)
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    # crossed heroes are auto-affected (no player choice); only the optional
    # minion defeat prompts.
    run.expect_input(InputRequestType.SELECT_UNIT).choose("em")
    run.finish()

    assert state.entity_locations.get("hero_wuk") == Hex(q=3, r=0, s=-3)
    assert state.entity_locations.get("eh") is None  # defeated (no cards)
    assert state.entity_locations.get("em") is None  # defeated minion


@pytest.mark.effect_flow
def test_trample_affects_all_crossed_heroes_without_choice() -> None:
    # Two enemy heroes crossed (no cards); both must be defeated with NO
    # selection prompt — the player cannot choose which heroes are affected.
    state = (
        EffectScenarioBuilder()
        .line_board(6)
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "trample"))
        .with_actor("hero_wuk")
        .blue_hero("eh1", at=(1, 0, -1))
        .blue_hero("eh2", at=(2, 0, -2))
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("MOVEMENT")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(2)
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    # no hero prompt, no minion candidates -> resolves with no further input
    run.finish()

    assert state.entity_locations.get("eh1") is None
    assert state.entity_locations.get("eh2") is None


@pytest.mark.effect_flow
def test_angry_stampede_defeats_support_then_heavy() -> None:
    # Heavy is immune while supported by the adjacent normal minion; defeating
    # the support first must unlock the heavy for the second minion select.
    state = (
        EffectScenarioBuilder()
        .line_board(6)
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "angry_stampede"))
        .with_actor("hero_wuk")
        .blue_minion("supp", at=(1, 0, -1))
        .blue_minion("hvy", at=(2, 0, -2), minion_type=MinionType.HEAVY)
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("MOVEMENT")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(2)
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=3, r=0, s=-3))
    # no heroes crossed -> hero multi-select auto-finishes
    # minion 1: only the support is selectable (heavy is immune)
    run.expect_input(InputRequestType.SELECT_UNIT).choose("supp")
    # minion 2: heavy now unsupported -> selectable
    run.expect_input(InputRequestType.SELECT_UNIT).choose("hvy")
    run.finish()

    assert state.entity_locations.get("supp") is None
    assert state.entity_locations.get("hvy") is None


@pytest.mark.effect_flow
def test_trample_normal_mode_may_detour_to_aligned_destination() -> None:
    state = (
        EffectScenarioBuilder()
        .small_arena()
        .red_hero("hero_wuk", at=(0, 0, 0), current_card=hero_card("Wuk", "trample"))
        .with_actor("hero_wuk")
        .blue_hero("eh", at=(1, 0, -1))
        .build()
    )

    run = run_card(state, "hero_wuk")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("MOVEMENT")
    run.expect_input(InputRequestType.SELECT_NUMBER).choose(1)
    run.expect_input(InputRequestType.SELECT_HEX).choose(Hex(q=2, r=0, s=-2))
    run.finish()

    assert state.entity_locations.get("hero_wuk") == Hex(q=2, r=0, s=-2)
    assert state.entity_locations.get("eh") == Hex(q=1, r=0, s=-1)
