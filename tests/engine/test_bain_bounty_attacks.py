"""
Tests for Bain's bounty attack cards (Dead or Alive, Hand Crossbow,
Hunter-Seeker), the HasMarkerFilter, and the bounty defeat penalty.
"""

import pytest

import goa2.scripts.bain_effects  # noqa: F401 — registers effects
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardTier,
    GamePhase,
    Hero,
    Minion,
    MinionType,
    Team,
    TeamColor,
)
from goa2.domain.models.marker import MarkerType
from goa2.domain.state import GameState
from goa2.engine.filters import HasMarkerFilter, TeamFilter, UnitTypeFilter
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.stats import CardStats
from goa2.engine.steps import (
    AttackSequenceStep,
    DefeatUnitStep,
    PlaceMarkerStep,
    SelectStep,
)
from tests.engine.effects.runner import run_card

# =============================================================================
# Card Factories
# =============================================================================


def _make_attack_card(card_id, name, effect_id, primary_value=4, **overrides):
    defaults = dict(
        id=card_id,
        name=name,
        tier=CardTier.UNTIERED,
        color=CardColor.GOLD,
        initiative=11,
        primary_action=ActionType.ATTACK,
        primary_action_value=primary_value,
        secondary_actions={},
        is_ranged=False,
        effect_id=effect_id,
        effect_text="",
        is_facedown=False,
    )
    defaults.update(overrides)
    return Card(**defaults)


def _make_ranged_attack_card(card_id, name, effect_id, primary_value=4, range_value=3):
    return Card(
        id=card_id,
        name=name,
        tier=CardTier.II,
        color=CardColor.RED,
        initiative=8,
        primary_action=ActionType.ATTACK,
        primary_action_value=primary_value,
        secondary_actions={},
        is_ranged=True,
        range_value=range_value,
        effect_id=effect_id,
        effect_text="",
        is_facedown=False,
    )


def _make_filler_card(card_id="filler", color=CardColor.GOLD):
    return Card(
        id=card_id,
        name="Filler",
        tier=CardTier.UNTIERED,
        color=color,
        initiative=1,
        primary_action=ActionType.ATTACK,
        secondary_actions={},
        is_ranged=False,
        range_value=0,
        primary_action_value=1,
        effect_id="filler",
        effect_text="",
        is_facedown=False,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def game_state():
    """Board with enough hexes for ranged tests."""
    board = Board()
    hexes = set()
    for q in range(-2, 6):
        for r in range(-2, 3):
            s = -q - r
            if abs(s) <= 5:
                hexes.add(Hex(q=q, r=r, s=s))
    z1 = Zone(id="z1", hexes=hexes, neighbors=[])
    board.zones = {"z1": z1}
    board.populate_tiles_from_zones()

    hero = Hero(id="hero_bain", name="Bain", team=TeamColor.RED, deck=[], level=1)
    enemy = Hero(id="enemy_hero", name="Enemy", team=TeamColor.BLUE, deck=[], level=1)
    enemy.hand = [_make_filler_card("e_card")]
    enemy2 = Hero(id="enemy_hero_2", name="Enemy2", team=TeamColor.BLUE, deck=[], level=1)
    enemy2.hand = [_make_filler_card("e2_card")]
    minion = Minion(id="minion_1", name="Grunt", team=TeamColor.BLUE, type=MinionType.MELEE)

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[hero], minions=[]),
            TeamColor.BLUE: Team(
                color=TeamColor.BLUE,
                heroes=[enemy, enemy2],
                minions=[minion],
            ),
        },
    )
    state.place_entity("hero_bain", Hex(q=0, r=0, s=0))
    state.place_entity("enemy_hero", Hex(q=1, r=0, s=-1))
    state.place_entity("enemy_hero_2", Hex(q=3, r=0, s=-3))
    state.place_entity("minion_1", Hex(q=0, r=1, s=-1))
    state.current_actor_id = "hero_bain"
    return state


# =============================================================================
# HasMarkerFilter Tests
# =============================================================================


def test_has_marker_filter_matches_bounty_holder(game_state):
    """HasMarkerFilter passes for the hero holding the specified marker."""
    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero", value=0, source_id="hero_bain")

    f = HasMarkerFilter(marker_type=MarkerType.BOUNTY)
    assert f.apply("enemy_hero", game_state, {}) is True
    assert f.apply("enemy_hero_2", game_state, {}) is False


def test_has_marker_filter_no_marker_in_play(game_state):
    """HasMarkerFilter rejects all when marker is not placed."""
    f = HasMarkerFilter(marker_type=MarkerType.BOUNTY)
    assert f.apply("enemy_hero", game_state, {}) is False


def test_has_marker_filter_rejects_non_string(game_state):
    """HasMarkerFilter rejects non-string candidates (e.g., Hex)."""
    f = HasMarkerFilter(marker_type=MarkerType.BOUNTY)
    assert f.apply(Hex(q=0, r=0, s=0), game_state, {}) is False


# =============================================================================
# Bounty Defeat Penalty Tests
# =============================================================================


def test_bounty_defeat_adds_extra_life_counter_penalty(game_state):
    """Defeating a hero with Bounty marker costs +1 additional life counter."""
    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero", value=0, source_id="hero_bain")

    initial_counters = game_state.teams[TeamColor.BLUE].life_counters

    push_steps(game_state, [DefeatUnitStep(victim_id="enemy_hero")])
    result = process_stack(game_state).input_request
    while result is not None:
        result = process_stack(game_state).input_request

    # Level 1 hero: base penalty 1 + bounty 1 = 2
    assert game_state.teams[TeamColor.BLUE].life_counters == initial_counters - 2


def test_defeat_without_bounty_normal_penalty(game_state):
    """Defeating a hero without Bounty marker uses normal penalty."""
    initial_counters = game_state.teams[TeamColor.BLUE].life_counters

    push_steps(game_state, [DefeatUnitStep(victim_id="enemy_hero")])
    result = process_stack(game_state).input_request
    while result is not None:
        result = process_stack(game_state).input_request

    # Level 1 hero: base penalty 1
    assert game_state.teams[TeamColor.BLUE].life_counters == initial_counters - 1


def test_markers_survive_their_placers_defeat(game_state):
    """No marker leaves play because the hero who placed it was defeated.

    Bounty is the case the audit called out (§4.7), but the rule is general —
    Venom and Poison behave the same way.
    """
    bounty = game_state.get_marker(MarkerType.BOUNTY)
    bounty.place(target_id="enemy_hero", value=0, source_id="hero_bain")
    venom = game_state.get_marker(MarkerType.VENOM)
    venom.place(target_id="enemy_hero_2", value=-1, source_id="hero_bain")

    push_steps(game_state, [DefeatUnitStep(victim_id="hero_bain")])
    result = process_stack(game_state).input_request
    while result is not None:
        result = process_stack(game_state).input_request

    assert bounty.target_id == "enemy_hero"
    assert venom.target_id == "enemy_hero_2"


def test_bounty_survives_bains_own_defeat(game_state):
    """Audit §4.7: Bounty stays on its target after Bain is defeated."""
    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero", value=0, source_id="hero_bain")

    push_steps(game_state, [DefeatUnitStep(victim_id="hero_bain")])
    result = process_stack(game_state).input_request
    while result is not None:
        result = process_stack(game_state).input_request

    assert marker.target_id == "enemy_hero"

    # ...and it still imposes the extra life loss.
    initial_counters = game_state.teams[TeamColor.BLUE].life_counters
    push_steps(game_state, [DefeatUnitStep(victim_id="enemy_hero")])
    result = process_stack(game_state).input_request
    while result is not None:
        result = process_stack(game_state).input_request

    assert game_state.teams[TeamColor.BLUE].life_counters == initial_counters - 2


# =============================================================================
# Dead or Alive Tests
# =============================================================================


def test_dead_or_alive_builds_attack_then_bounty_select(game_state):
    """Dead or Alive produces attack + optional bounty select + place marker."""
    from goa2.scripts.bain_effects import DeadOrAliveEffect

    effect = DeadOrAliveEffect()
    hero = game_state.get_hero("hero_bain")
    card = _make_attack_card("doa", "Dead or Alive", "dead_or_alive")
    stats = CardStats(primary_value=4, range=1)

    steps = effect.build_steps(game_state, hero, card, stats)

    assert len(steps) == 3
    assert isinstance(steps[0], AttackSequenceStep)
    assert steps[0].range_val == 1
    assert isinstance(steps[1], SelectStep)
    assert steps[1].is_mandatory is False
    # Filters should include ENEMY + HERO
    filter_types = [type(f) for f in steps[1].filters]
    assert TeamFilter in filter_types
    assert UnitTypeFilter in filter_types
    assert isinstance(steps[2], PlaceMarkerStep)
    assert steps[2].marker_type == MarkerType.BOUNTY


# =============================================================================
# Hand Crossbow Tests
# =============================================================================


def _give_hand_crossbow(state):
    """Attach Hand Crossbow as the resolving card for hero_bain."""
    state.phase = GamePhase.RESOLUTION
    card = _make_ranged_attack_card("hc", "Hand Crossbow", "hand_crossbow")
    hero = state.get_hero("hero_bain")
    hero.current_turn_card = card
    return card


# Reply value for each post-selection input type we may encounter while
# draining an attack's follow-up prompts. Non-skippable dialogs (e.g. the
# defender's reaction window) explicitly want PASS, not SKIP.
_DECLINE_REPLY_BY_TYPE = {
    InputRequestType.SELECT_CARD_OR_PASS: "PASS",
    InputRequestType.CONFIRM_PASSIVE: "NO",
    InputRequestType.CHOOSE_ACTION: "PASS",
}


def _drain_to_completion(run):
    """Drive an attack's post-selection prompts to completion.

    Reply is chosen per input type: PASS for defender/reaction windows,
    NO for passive confirmations, SKIP only for genuinely skippable
    selections (optional SELECT_UNIT etc.). Events emitted along the way
    are accumulated on `run.events` so downstream assertions can inspect
    COMBAT_RESOLVED and friends.
    """
    while True:
        result = process_stack(run.state)
        run.events.extend(result.events)
        req = result.input_request
        if req is None:
            return
        reply = _DECLINE_REPLY_BY_TYPE.get(req.request_type, "SKIP")
        run.state.execution_stack[-1].pending_input = {"selection": reply}


def test_hand_crossbow_flattens_to_single_select_unit(game_state):
    """
    Hand Crossbow should present ONE combined SELECT_UNIT after CHOOSE_ACTION,
    with no intermediate SELECT_NUMBER mode picker.

    Card text: "Choose one —
      * Target a hero in range with a Bounty marker.
      * Target a unit adjacent to you."

    The two alternatives are equivalent target choices from the player's
    perspective and should be flattened into one target selection whose valid
    candidates are the union of the two branches.
    """
    # Bounty-mark the nonadjacent enemy hero at (3,0,-3) — range 3, not adjacent.
    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero_2", value=0, source_id="hero_bain")

    _give_hand_crossbow(game_state)

    run = run_card(game_state, "hero_bain")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    # No SELECT_NUMBER in between: the next prompt is directly a SELECT_UNIT.
    run.expect_input(InputRequestType.SELECT_UNIT)


def test_hand_crossbow_offers_adjacent_and_bounty_targets_together(game_state):
    """
    The single SELECT_UNIT must offer BOTH categories together:
      - adjacent ordinary enemies (adjacent minion, adjacent enemy hero),
      - nonadjacent enemy heroes with the Bounty marker (in range).
    """
    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero_2", value=0, source_id="hero_bain")

    _give_hand_crossbow(game_state)

    run = run_card(game_state, "hero_bain")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT)
    option_ids = {option.id for option in run.latest_request.options}

    # Adjacent category: adjacent minion + adjacent enemy hero both qualify.
    assert (
        "minion_1" in option_ids
    ), f"Adjacent minion should qualify via adjacent-unit branch; got {sorted(option_ids)!r}"
    assert (
        "enemy_hero" in option_ids
    ), f"Adjacent enemy hero should qualify via adjacent-unit branch; got {sorted(option_ids)!r}"
    # Bounty category: nonadjacent bounty-marked enemy hero qualifies.
    assert (
        "enemy_hero_2" in option_ids
    ), f"Nonadjacent bounty-marked hero should qualify via bounty branch; got {sorted(option_ids)!r}"


def test_hand_crossbow_excludes_nonqualifying_ranged_targets(game_state):
    """
    A nonadjacent enemy hero WITHOUT the Bounty marker is in range but
    matches neither branch (not adjacent, not bounty-marked) and must be
    excluded from the combined SELECT_UNIT options.
    """
    # Add a third enemy hero at (2,0,-2): in range 3 but not adjacent, no bounty.
    enemy3 = Hero(id="enemy_hero_3", name="Enemy3", team=TeamColor.BLUE, deck=[], level=1)
    game_state.teams[TeamColor.BLUE].heroes.append(enemy3)
    game_state.place_entity("enemy_hero_3", Hex(q=2, r=0, s=-2))

    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero_2", value=0, source_id="hero_bain")

    _give_hand_crossbow(game_state)

    run = run_card(game_state, "hero_bain")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT)
    option_ids = {option.id for option in run.latest_request.options}

    # The nonadjacent, non-bounty enemy is in range but does not qualify.
    assert (
        "enemy_hero_3" not in option_ids
    ), f"Nonadjacent enemy without Bounty must be excluded; got {sorted(option_ids)!r}"
    # Sanity: legitimate targets are still offered.
    assert "enemy_hero" in option_ids  # adjacent
    assert "enemy_hero_2" in option_ids  # bounty


def test_hand_crossbow_resolves_adjacent_category_target(game_state):
    """
    Picking an adjacent-category target (adjacent minion) resolves an
    attack whose victim is specifically that minion — asserted via the
    COMBAT_RESOLVED event's target_id.
    """
    from goa2.domain.events import GameEventType

    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero_2", value=0, source_id="hero_bain")

    _give_hand_crossbow(game_state)

    run = run_card(game_state, "hero_bain")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("minion_1")

    _drain_to_completion(run)

    combat_targets = [
        event.target_id for event in run.events if event.event_type == GameEventType.COMBAT_RESOLVED
    ]
    assert combat_targets == ["minion_1"], (
        "Adjacent-category attack must resolve with minion_1 as the "
        f"combat target; got COMBAT_RESOLVED target_ids={combat_targets!r}"
    )


def test_hand_crossbow_resolves_bounty_category_target(game_state):
    """
    Picking a bounty-category target (nonadjacent bounty-marked enemy hero)
    resolves an attack whose victim is specifically that hero — asserted
    via the COMBAT_RESOLVED event's target_id, which is the stable
    per-victim signal emitted by AttackSequenceStep.
    """
    from goa2.domain.events import GameEventType

    marker = game_state.get_marker(MarkerType.BOUNTY)
    marker.place(target_id="enemy_hero_2", value=0, source_id="hero_bain")

    _give_hand_crossbow(game_state)

    run = run_card(game_state, "hero_bain")
    run.expect_input(InputRequestType.CHOOSE_ACTION).choose("ATTACK")
    run.expect_input(InputRequestType.SELECT_UNIT).choose("enemy_hero_2")

    _drain_to_completion(run)

    combat_targets = [
        event.target_id for event in run.events if event.event_type == GameEventType.COMBAT_RESOLVED
    ]
    assert combat_targets == ["enemy_hero_2"], (
        "Bounty-category attack must resolve with enemy_hero_2 as the "
        f"combat target; got COMBAT_RESOLVED target_ids={combat_targets!r}"
    )


# =============================================================================
# Hunter-Seeker Tests
# =============================================================================


def test_hunter_seeker_builds_dual_path_steps(game_state):
    """Hunter-Seeker produces choice + two paths with optional second attack."""
    from goa2.scripts.bain_effects import HunterSeekerEffect

    effect = HunterSeekerEffect()
    hero = game_state.get_hero("hero_bain")
    card = _make_ranged_attack_card("hs", "Hunter-Seeker", "hunter_seeker", primary_value=5)
    stats = CardStats(primary_value=5, range=3)

    steps = effect.build_steps(game_state, hero, card, stats)

    # SelectStep(NUMBER) + 2 CheckContext + PATH A (Attack + Select + Attack) + PATH B (Attack + Select + Attack)
    assert len(steps) == 9


def test_hunter_seeker_bounty_path_uses_has_marker_filter(game_state):
    """Hunter-Seeker bounty attacks use HasMarkerFilter."""
    from goa2.scripts.bain_effects import HunterSeekerEffect

    effect = HunterSeekerEffect()
    hero = game_state.get_hero("hero_bain")
    card = _make_ranged_attack_card("hs", "Hunter-Seeker", "hunter_seeker", primary_value=5)
    stats = CardStats(primary_value=5, range=3)

    steps = effect.build_steps(game_state, hero, card, stats)

    # Path A first attack (index 3) = bounty attack
    bounty_attack_a = steps[3]
    assert isinstance(bounty_attack_a, AttackSequenceStep)
    filter_types_a = [type(f) for f in bounty_attack_a.target_filters]
    assert HasMarkerFilter in filter_types_a

    # Path B second select (index 7) filters for bounty + excludes first target
    bounty_select_b = steps[7]
    assert isinstance(bounty_select_b, SelectStep)
    from goa2.engine.filters import ExcludeIdentityFilter

    filter_types_b = [type(f) for f in bounty_select_b.filters]
    assert HasMarkerFilter in filter_types_b
    assert ExcludeIdentityFilter in filter_types_b
