"""No-target ATTACK/SKILL options are pruned from the CHOOSE_ACTION menu.

An action that cannot do anything (its first mandatory target selection has no
valid candidate from the current position) is removed from the menu, so the
ISMCTS search / heuristic prior never expand a guaranteed no-op branch. Actions
whose legality is satisfied later (move-then-target, deferred end-of-turn
effects, auras/self, optional selects) must remain available.
"""

import pytest

from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.board import Board, Tile, Zone
from goa2.domain.hex import Hex
from goa2.domain.models import (
    Card,
    CardState,
    Hero,
    Minion,
    MinionType,
    Team,
    TeamColor,
    Token,
    TokenType,
)
from goa2.domain.models.marker import Marker, MarkerType
from goa2.domain.models.spawn import SpawnPoint, SpawnType
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import ResolveCardStep


@pytest.fixture(autouse=True, scope="module")
def _register_effects():
    """Card effects must be registered for scripted primaries (mad_dash,
    warning_shot, angry_roar). In production the server does this at startup;
    make the test self-sufficient rather than relying on collection order."""
    from goa2.server.app import register_all_effects

    register_all_effects()


def _card(hero_name: str, card_id: str) -> Card:
    hero = HeroRegistry.get(hero_name)
    assert hero is not None, hero_name
    cards = [*hero.deck] + ([hero.ultimate_card] if hero.ultimate_card else [])
    for c in cards:
        if c.id == card_id:
            playable = c.model_copy(deep=True)
            playable.state = CardState.UNRESOLVED
            playable.is_facedown = False
            return playable
    raise AssertionError(f"card {card_id} not found for {hero_name}")


@pytest.fixture
def arena():
    """A radius-4 hex board with one Red hero as the current actor."""
    board = Board()
    for q in range(-4, 5):
        for r in range(-4, 5):
            s = -q - r
            if abs(s) <= 4:
                h = Hex(q=q, r=r, s=s)
                board.tiles[h] = Tile(hex=h)
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        entity_locations={},
    )
    state.active_zone_id = "z1"
    board.zones["z1"] = Zone(id="z1", label="Z1", hexes=list(board.tiles.keys()))
    return state


def _menu_ids(state: GameState, hero: Hero) -> set[str]:
    state.teams[TeamColor.RED].heroes.append(hero)
    state.current_actor_id = hero.id
    push_steps(state, [ResolveCardStep(hero_id=hero.id)])
    req = process_stack(state).input_request
    assert req is not None and req["type"] == "CHOOSE_ACTION"
    return {o["id"] for o in req["options"]}


def _enemy(state: GameState, at: Hex, mid: str = "m1") -> None:
    m = Minion(id=mid, name=mid, type=MinionType.MELEE, team=TeamColor.BLUE)
    state.teams[TeamColor.BLUE].minions.append(m)
    state.place_entity(mid, at)


# --- Basic ATTACK (Brogan onslaught: mandatory adjacent-enemy select up front) ---


def test_basic_attack_pruned_when_no_enemy_in_range(arena):
    hero = Hero(id="hero_brogan", name="Brogan", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Brogan", "onslaught")
    arena.place_entity("hero_brogan", Hex(q=0, r=0, s=0))
    assert "ATTACK" not in _menu_ids(arena, hero)


def test_basic_attack_present_when_enemy_in_range(arena):
    hero = Hero(id="hero_brogan", name="Brogan", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Brogan", "onslaught")
    arena.place_entity("hero_brogan", Hex(q=0, r=0, s=0))
    _enemy(arena, Hex(q=1, r=0, s=-1))  # adjacent
    assert "ATTACK" in _menu_ids(arena, hero)


# --- Move-then-target (Brogan mad_dash): must NOT be pruned when a dash
#     destination adjacent to an enemy exists, even though no enemy is
#     adjacent right now. ---


def test_move_then_target_present_when_dash_destination_exists(arena):
    hero = Hero(id="hero_brogan", name="Brogan", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Brogan", "mad_dash")
    arena.place_entity("hero_brogan", Hex(q=0, r=0, s=0))
    # Enemy 3 away in a straight line: hero can dash 2 to (2,0,-2), adjacent to it.
    _enemy(arena, Hex(q=3, r=0, s=-3))
    assert "ATTACK" in _menu_ids(arena, hero)


def test_move_then_target_pruned_when_no_dash_destination(arena):
    hero = Hero(id="hero_brogan", name="Brogan", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Brogan", "mad_dash")
    arena.place_entity("hero_brogan", Hex(q=0, r=0, s=0))
    # No enemies at all -> no dash destination adjacent to an enemy.
    assert "ATTACK" not in _menu_ids(arena, hero)


# --- Pre-target passive movement (Arien living_tsunami): the menu is built
#     before the optional move, so current-position targeting cannot prune it. ---


def test_attack_present_when_before_attack_passive_can_move_into_range(arena):
    template = HeroRegistry.get("Arien")
    assert template is not None and template.ultimate_card is not None
    hero = Hero(
        id="hero_arien",
        name="Arien",
        team=TeamColor.RED,
        deck=[],
        level=8,
        ultimate_card=template.ultimate_card.model_copy(deep=True),
    )
    hero.current_turn_card = _card("Arien", "noble_blade")
    arena.place_entity("hero_arien", Hex(q=0, r=0, s=0))
    _enemy(arena, Hex(q=2, r=0, s=-2))

    assert "ATTACK" in _menu_ids(arena, hero)


def test_multi_piece_attack_not_pruned_before_piece_is_chosen(arena):
    from goa2.engine.hero_pieces import create_hero_pieces, piece_id

    hero = Hero(
        id="hero_razzle",
        name="Razzle",
        team=TeamColor.RED,
        deck=[],
        piece_supply=4,
    )
    hero.current_turn_card = _card("Razzle", "stunt_doubles")
    arena.teams[TeamColor.RED].heroes.append(hero)
    arena.current_actor_id = hero.id
    create_hero_pieces(arena, hero)
    arena.place_entity(piece_id("hero_razzle", 1), Hex(q=0, r=0, s=0))
    arena.place_entity(piece_id("hero_razzle", 2), Hex(q=1, r=0, s=-1))

    push_steps(arena, [ResolveCardStep(hero_id=hero.id)])
    req = process_stack(arena).input_request

    assert req is not None
    assert "ATTACK" in {option.id for option in req.options}


@pytest.mark.parametrize(
    ("hero_name", "hero_id", "card_id", "action_id"),
    [
        ("Brynn", "hero_brynn", "familiar_ground", "ATTACK"),
        ("Brynn", "hero_brynn", "bear_trap", "SKILL"),
        ("NebKher", "hero_nebkher", "phantasmal_sentry", "ATTACK"),
        ("Wuk", "hero_wuk", "natures_protector", "ATTACK"),
    ],
)
def test_flattened_alternative_target_action_pruned_without_any_target(
    arena, hero_name, hero_id, card_id, action_id
):
    hero = Hero(id=hero_id, name=hero_name, team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card(hero_name, card_id)
    arena.place_entity(hero_id, Hex(q=0, r=0, s=0))

    assert action_id not in _menu_ids(arena, hero)


# --- Deferred skill (Silverarrow warning_shot): end-of-turn target, gate lives
#     in finishing_steps, not top-level. Must remain available. ---


def test_deferred_skill_present_with_no_enemy(arena):
    hero = Hero(id="hero_silverarrow", name="Silverarrow", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Silverarrow", "warning_shot")
    arena.place_entity("hero_silverarrow", Hex(q=0, r=0, s=0))
    assert "SKILL" in _menu_ids(arena, hero)


# --- Self-aura / conditional skill (Ursafar angry_roar): useful regardless of
#     target (applies the enraged self-effect). Must remain available. ---


def test_angry_roar_present_with_no_target(arena):
    hero = Hero(id="hero_ursafar", name="Ursafar", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Ursafar", "angry_roar")
    arena.place_entity("hero_ursafar", Hex(q=0, r=0, s=0))
    assert "SKILL" in _menu_ids(arena, hero)


# --- Flattened combined-target-gate cards ---------------------------------
#
# After the flattening pass, several ATTACK/SKILL effects still expose two
# alternative target categories behind a leading NUMBER mode-choice select
# (Bain Hand Crossbow, Dodger Littlefinger/Finger of Death, Tali Spirit Wolf,
# Wuk Into the Canopy). The behavioural contract is a *combined* target gate:
# the option belongs in the CHOOSE_ACTION menu iff at least one alternative —
# adjacent OR the non-adjacent alternate category — has a legal target from
# the current position; otherwise it must be pruned.
#
# The current pruning probe short-circuits on the leading NUMBER select and
# leaves the option available regardless of downstream target validity. The
# negative parametrized tests below fail on that gap. The positive tests
# encode the other half of the contract in two shapes:
#   1. adjacent enemy present -> action stays;
#   2. NO adjacent enemy, but the non-adjacent alternate category has a
#      valid target (bounty-marked ranged hero, ranged hero with a discarded
#      card, ordinary ranged target) -> action must still stay.
# Each case isolates target validation so failures reflect the pruning
# contract, not setup drift.


def _place_tree(state: GameState, at: Hex, tid: str = "tree_1") -> None:
    tree = Token(id=BoardEntityID(tid), name="Tree", token_type=TokenType.TREE)
    state.register_entity(tree, "token")
    state.place_entity(tid, at)


def _place_enemy_hero(state: GameState, at: Hex, hid: str = "hero_victim") -> Hero:
    victim = Hero(id=hid, name="Victim", team=TeamColor.BLUE, deck=[])
    state.teams[TeamColor.BLUE].heroes.append(victim)
    state.place_entity(hid, at)
    return victim


def _give_hero_a_discarded_card(hero: Hero) -> None:
    """Attach a real, validated Card to ``hero.discard_pile``. Copies a card
    from an unrelated hero's registry deck so CardsInContainerFilter (which
    counts entries in ``hero.discard_pile``) sees at least one card."""
    donor = HeroRegistry.get("Brogan")
    assert donor is not None and donor.deck, "Brogan deck required for donor card"
    card = donor.deck[0].model_copy(deep=True)
    card.state = CardState.DISCARD
    hero.discard_pile.append(card)


@pytest.mark.parametrize(
    ("hero_name", "hero_id", "card_id", "action_id"),
    [
        # Bain Hand Crossbow: bounty-hero-in-range OR adjacent unit.
        ("Bain", "hero_bain", "hand_crossbow", "ATTACK"),
        # Dodger Finger family: adjacent unit OR hero-in-range-with-discard.
        ("Dodger", "hero_dodger", "littlefinger_of_death", "ATTACK"),
        ("Dodger", "hero_dodger", "finger_of_death", "ATTACK"),
        # Tali Spirit Wolf: unit-in-range OR adjacent enemy hero.
        ("Tali", "hero_tali", "spirit_wolf", "ATTACK"),
    ],
)
def test_combined_target_attack_pruned_when_every_alternative_has_no_target(
    arena, hero_name, hero_id, card_id, action_id
):
    """Empty board -> neither the adjacent nor the alternate branch has a
    candidate -> action is a guaranteed no-op and must be pruned."""
    hero = Hero(id=hero_id, name=hero_name, team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card(hero_name, card_id)
    arena.place_entity(hero_id, Hex(q=0, r=0, s=0))

    assert action_id not in _menu_ids(arena, hero)


@pytest.mark.parametrize(
    ("hero_name", "hero_id", "card_id", "action_id"),
    [
        ("Bain", "hero_bain", "hand_crossbow", "ATTACK"),
        ("Dodger", "hero_dodger", "littlefinger_of_death", "ATTACK"),
        ("Dodger", "hero_dodger", "finger_of_death", "ATTACK"),
        ("Tali", "hero_tali", "spirit_wolf", "ATTACK"),
    ],
)
def test_combined_target_attack_present_when_adjacent_branch_has_target(
    arena, hero_name, hero_id, card_id, action_id
):
    """A single adjacent enemy satisfies the melee branch -> action stays."""
    hero = Hero(id=hero_id, name=hero_name, team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card(hero_name, card_id)
    arena.place_entity(hero_id, Hex(q=0, r=0, s=0))
    _enemy(arena, Hex(q=1, r=0, s=-1))

    assert action_id in _menu_ids(arena, hero)


# The next three tests remove the adjacent branch entirely (no enemy at
# range 1) and instead satisfy ONLY the alternate category for each card, so
# a correct implementation must inspect BOTH sides of the combined gate.


def test_hand_crossbow_present_when_only_bounty_hero_in_range(arena):
    """Hand Crossbow: bounty-marked enemy hero at range with nothing adjacent
    -> the melee branch is a no-op but the bounty branch is legal."""
    hero = Hero(id="hero_bain", name="Bain", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Bain", "hand_crossbow")
    arena.place_entity("hero_bain", Hex(q=0, r=0, s=0))
    victim = _place_enemy_hero(arena, Hex(q=3, r=0, s=-3))
    arena.markers[MarkerType.BOUNTY] = Marker(
        type=MarkerType.BOUNTY,
        target_id=str(victim.id),
        value=1,
        source_id="hero_bain",
    )

    assert "ATTACK" in _menu_ids(arena, hero)


@pytest.mark.parametrize(
    ("card_id", "victim_hex"),
    [
        # Finger of Death targets a hero in range with discard; range 3.
        ("finger_of_death", Hex(q=3, r=0, s=-3)),
        # Littlefinger of Death is the same effect at range 2.
        ("littlefinger_of_death", Hex(q=2, r=0, s=-2)),
    ],
)
def test_finger_family_present_when_only_discard_hero_in_range(arena, card_id, victim_hex):
    """Finger / Littlefinger of Death: an enemy hero with a discarded card
    sitting at range (not adjacent) satisfies ONLY the ranged branch."""
    hero = Hero(id="hero_dodger", name="Dodger", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Dodger", card_id)
    arena.place_entity("hero_dodger", Hex(q=0, r=0, s=0))
    victim = _place_enemy_hero(arena, victim_hex)
    _give_hero_a_discarded_card(victim)

    assert "ATTACK" in _menu_ids(arena, hero)


def test_spirit_wolf_present_when_only_ranged_target_available(arena):
    """Spirit Wolf ranged branch takes any enemy in range; place a minion
    outside adjacency so only that branch is legal."""
    hero = Hero(id="hero_tali", name="Tali", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Tali", "spirit_wolf")
    arena.place_entity("hero_tali", Hex(q=0, r=0, s=0))
    _enemy(arena, Hex(q=2, r=0, s=-2))  # range 2, not adjacent

    assert "ATTACK" in _menu_ids(arena, hero)


# --- Wuk Into the Canopy (SKILL) ------------------------------------------
#
# "Choose one — Swap with a Tree token in radius. / Swap a friendly unit in
# radius with a Tree token in radius." Both alternatives take a mandatory
# Tree token select, so absence of any Tree in radius makes every branch a
# guaranteed no-op. Current pruning masks this behind the leading NUMBER
# mode-choice select.


def test_into_the_canopy_pruned_when_no_tree_in_radius(arena):
    hero = Hero(id="hero_wuk", name="Wuk", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Wuk", "into_the_canopy")
    arena.place_entity("hero_wuk", Hex(q=0, r=0, s=0))

    assert "SKILL" not in _menu_ids(arena, hero)


def test_into_the_canopy_present_when_tree_in_radius(arena):
    hero = Hero(id="hero_wuk", name="Wuk", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Wuk", "into_the_canopy")
    arena.place_entity("hero_wuk", Hex(q=0, r=0, s=0))
    _place_tree(arena, Hex(q=1, r=0, s=-1))

    assert "SKILL" in _menu_ids(arena, hero)


# --- Wuk Tree of Plenty (SKILL) -------------------------------------------
#
# Structure differs: the first mandatory step is the Tree-removal COST
# (SelectStep UNIT_OR_TOKEN, TREE filter) — NOT a NUMBER mode choice — so the
# existing probe correctly prunes when no Tree is in radius. The downstream
# retrieval branches are optional (self-retrieve / friendly-hero-retrieve),
# so once the cost is satisfiable the action stays available regardless of
# whether either retrieval branch would actually retrieve a card.


def test_tree_of_plenty_pruned_when_no_tree_in_radius(arena):
    hero = Hero(id="hero_wuk", name="Wuk", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Wuk", "tree_of_plenty")
    arena.place_entity("hero_wuk", Hex(q=0, r=0, s=0))

    assert "SKILL" not in _menu_ids(arena, hero)


def test_tree_of_plenty_present_when_tree_in_radius_even_if_retrieves_would_noop(arena):
    """Retrieval branches are optional -> action stays as long as the
    mandatory Tree-removal cost has a candidate. Wuk has no discard and no
    friendly hero is on the board, but SKILL must remain in the menu."""
    hero = Hero(id="hero_wuk", name="Wuk", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Wuk", "tree_of_plenty")
    arena.place_entity("hero_wuk", Hex(q=0, r=0, s=0))
    _place_tree(arena, Hex(q=1, r=0, s=-1))

    assert "SKILL" in _menu_ids(arena, hero)


# --- Dodger Dread Razor (untiered gold, ATTACK) ----------------------------
#
# "Choose one — Target a unit adjacent to you. If you are adjacent to an
# empty spawn point in the battle zone, target a unit in range."
#
# The effect builder collapses to a bare AttackSequenceStep when the hero is
# NOT adjacent to an empty spawn -> the existing probe already prunes that
# case. When the hero IS adjacent to an empty spawn point, the effect emits
# a combined melee-or-ranged target gate; with no enemy adjacent AND no enemy
# in range, every alternative is a guaranteed no-op and the option must be
# pruned. Current pruning masks this second case.


def _place_empty_spawn_adjacent(state: GameState, hero_hex: Hex) -> Hex:
    """Attach an unoccupied minion spawn point to a tile adjacent to hero.

    The tile must live in the current battle zone (``zone_id`` set) and have
    ``spawn_point`` populated but no occupant. The arena fixture registers
    zones by hex membership but leaves ``Tile.zone_id`` unset, so we also
    stamp the zone here rather than mutate the shared fixture.
    """
    spawn_hex = Hex(q=hero_hex.q + 1, r=hero_hex.r, s=hero_hex.s - 1)
    tile = state.board.tiles[spawn_hex]
    tile.zone_id = "z1"
    tile.spawn_point = SpawnPoint(
        location=spawn_hex,
        team=TeamColor.BLUE,
        type=SpawnType.MINION,
        minion_type=MinionType.MELEE,
    )
    return spawn_hex


def test_dread_razor_pruned_when_no_enemy_and_no_spawn_adjacency(arena):
    """Melee-only fallback path: without a spawn adjacency the effect emits a
    single AttackSequenceStep, which is target-gated and correctly pruned."""
    hero = Hero(id="hero_dodger", name="Dodger", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Dodger", "dread_razor")
    arena.place_entity("hero_dodger", Hex(q=0, r=0, s=0))

    assert "ATTACK" not in _menu_ids(arena, hero)


def test_dread_razor_pruned_when_spawn_adjacency_but_no_enemy_anywhere(arena):
    """Combined melee-or-ranged path: spawn adjacency enables the ranged
    alternative, but with no enemy adjacent AND no enemy in range every
    alternative is a no-op."""
    hero = Hero(id="hero_dodger", name="Dodger", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Dodger", "dread_razor")
    hero_hex = Hex(q=0, r=0, s=0)
    arena.place_entity("hero_dodger", hero_hex)
    _place_empty_spawn_adjacent(arena, hero_hex)

    assert "ATTACK" not in _menu_ids(arena, hero)


def test_dread_razor_present_when_spawn_adjacency_and_enemy_in_range(arena):
    hero = Hero(id="hero_dodger", name="Dodger", team=TeamColor.RED, deck=[])
    hero.current_turn_card = _card("Dodger", "dread_razor")
    hero_hex = Hex(q=0, r=0, s=0)
    arena.place_entity("hero_dodger", hero_hex)
    _place_empty_spawn_adjacent(arena, hero_hex)
    # Enemy two hexes away in a different direction -> only the ranged
    # alternative applies; the combined gate must not veto that.
    _enemy(arena, Hex(q=-2, r=0, s=2))

    assert "ATTACK" in _menu_ids(arena, hero)
