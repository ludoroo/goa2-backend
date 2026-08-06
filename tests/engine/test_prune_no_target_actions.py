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
    ActionType,
    Card,
    CardState,
    Hero,
    Minion,
    MinionType,
    Team,
    TeamColor,
)
from goa2.domain.state import GameState
from goa2.engine.effects import CardEffect, CardEffectRegistry
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import AttackSequenceStep, ResolveCardStep
from goa2.engine.steps.cards import action_passes_initial_target_gate


class _KeyedAttackEffect(CardEffect):
    def build_steps(self, state, hero, card, stats):
        return [
            AttackSequenceStep(
                damage=1,
                range_val=1,
                target_id_key="selected_target",
            )
        ]


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


@pytest.mark.parametrize(
    ("target_preselected", "expected"),
    [(False, False), (True, True)],
)
def test_keyed_attack_requires_target_key_to_exist_in_context(
    arena, monkeypatch, target_preselected, expected
):
    effect_id = "test_keyed_attack"
    monkeypatch.setitem(CardEffectRegistry._effects, effect_id, _KeyedAttackEffect())

    hero = Hero(id="hero_test", name="Test", team=TeamColor.RED, deck=[])
    card = _card("Brogan", "onslaught")
    card.effect_id = effect_id
    hero.current_turn_card = card
    arena.teams[TeamColor.RED].heroes.append(hero)
    arena.current_actor_id = hero.id
    arena.place_entity(hero.id, Hex(q=0, r=0, s=0))
    if target_preselected:
        arena.execution_context["selected_target"] = "existing_target"

    assert (
        action_passes_initial_target_gate(
            arena,
            hero,
            card,
            ActionType.ATTACK,
            is_primary=True,
        )
        is expected
    )


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
