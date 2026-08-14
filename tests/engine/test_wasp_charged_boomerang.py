"""
Tests for Wasp's Charged Boomerang effect.

Card Text: "Target a unit in range and not in a straight line.
(Units adjacent to you are in a straight line from you.)"
"""

import pytest

import goa2.scripts.wasp_effects  # noqa: F401 - Register wasp effects
from goa2.domain.board import Board, Zone
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardTier,
    Hero,
    Team,
    TeamColor,
)
from goa2.domain.state import GameState
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import ResolveCardStep


@pytest.fixture
def wasp_boomerang_state():
    """
    Board setup for testing straight line targeting:

    Hex grid (q, r, s):
                    (-1,-1,2)  (0,-1,1)  (1,-1,0)
                 (-1,0,1) [WASP] (1,0,-1) (2,0,-2)
                    (-1,1,0)   (0,1,-1)  (1,1,-2)

    - (0,0,0): Wasp (attacker)
    - (1,0,-1): Adjacent enemy (in straight line - INVALID target)
    - (2,0,-2): Enemy at range 2, same q-axis (in straight line - INVALID target)
    - (1,1,-2): Enemy at range 2, NOT in straight line (VALID target)
    - (0,-1,1): Enemy at range 1, same s-axis (adjacent, in straight line - INVALID)
    """
    board = Board()
    hexes = {
        Hex(q=0, r=0, s=0),  # Wasp
        Hex(q=1, r=0, s=-1),  # Adjacent, straight line (q-axis)
        Hex(q=2, r=0, s=-2),  # Range 2, straight line (q-axis)
        Hex(q=1, r=1, s=-2),  # Range 2, NOT straight line (diagonal)
        Hex(q=0, r=-1, s=1),  # Adjacent, straight line (s-axis)
        Hex(q=-1, r=0, s=1),  # Adjacent, straight line (r-axis)
        Hex(q=2, r=-1, s=-1),  # Range 2, NOT straight line (diagonal)
        Hex(q=3, r=0, s=-3),  # Range 3, straight line (q-axis)
        Hex(q=-1, r=2, s=-1),  # Range 2, NOT straight line (diagonal)
    }
    z1 = Zone(id="z1", hexes=hexes, neighbors=[])
    board.zones = {"z1": z1}
    board.populate_tiles_from_zones()

    wasp = Hero(id="wasp", name="Wasp", team=TeamColor.RED, deck=[], level=1)
    card = Card(
        id="charged_boomerang",
        name="Charged Boomerang",
        tier=CardTier.II,
        color=CardColor.RED,
        initiative=9,
        primary_action=ActionType.ATTACK,
        primary_action_value=3,
        is_ranged=True,
        range_value=3,
        effect_id="charged_boomerang",
        effect_text="Target a unit in range and not in a straight line.",
        is_facedown=False,
    )
    wasp.current_turn_card = card

    # Enemies at various positions
    enemy_adjacent = Hero(
        id="enemy_adjacent", name="Adjacent", team=TeamColor.BLUE, deck=[], level=1
    )
    enemy_straight = Hero(
        id="enemy_straight", name="Straight", team=TeamColor.BLUE, deck=[], level=1
    )
    enemy_diagonal = Hero(
        id="enemy_diagonal", name="Diagonal", team=TeamColor.BLUE, deck=[], level=1
    )
    enemy_diagonal2 = Hero(
        id="enemy_diagonal2", name="Diagonal2", team=TeamColor.BLUE, deck=[], level=1
    )

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[wasp], minions=[]),
            TeamColor.BLUE: Team(
                color=TeamColor.BLUE,
                heroes=[
                    enemy_adjacent,
                    enemy_straight,
                    enemy_diagonal,
                    enemy_diagonal2,
                ],
                minions=[],
            ),
        },
    )

    state.place_entity("wasp", Hex(q=0, r=0, s=0))
    state.place_entity("enemy_adjacent", Hex(q=1, r=0, s=-1))  # Adjacent, straight line
    state.place_entity("enemy_straight", Hex(q=2, r=0, s=-2))  # Range 2, straight line
    state.place_entity("enemy_diagonal", Hex(q=1, r=1, s=-2))  # Range 2, NOT straight line
    state.place_entity("enemy_diagonal2", Hex(q=2, r=-1, s=-1))  # Range 2, NOT straight line

    state.current_actor_id = "wasp"
    return state


def test_not_in_straight_line_filter_excludes_straight_line():
    """Test the filter correctly identifies straight line hexes."""
    # Same q coordinate = straight line
    origin = Hex(q=0, r=0, s=0)

    # Same q-axis (q=0)
    assert origin.is_straight_line(Hex(q=0, r=1, s=-1)) is True
    assert origin.is_straight_line(Hex(q=0, r=-2, s=2)) is True

    # Same r-axis (r=0)
    assert origin.is_straight_line(Hex(q=1, r=0, s=-1)) is True
    assert origin.is_straight_line(Hex(q=-2, r=0, s=2)) is True

    # Same s-axis (s=0)
    assert origin.is_straight_line(Hex(q=1, r=-1, s=0)) is True
    assert origin.is_straight_line(Hex(q=-2, r=2, s=0)) is True

    # NOT straight line (no matching coordinate)
    assert origin.is_straight_line(Hex(q=1, r=1, s=-2)) is False
    assert origin.is_straight_line(Hex(q=2, r=-1, s=-1)) is False
    assert origin.is_straight_line(Hex(q=-1, r=2, s=-1)) is False


def test_charged_boomerang_valid_targets(wasp_boomerang_state):
    """
    Test that only targets NOT in a straight line are offered.
    """
    step = ResolveCardStep(hero_id="wasp")
    push_steps(wasp_boomerang_state, [step])

    # 1. Action Choice (Attack)
    req = process_stack(wasp_boomerang_state).input_request
    assert req["type"] == "CHOOSE_ACTION"
    wasp_boomerang_state.execution_stack[-1].pending_input = {"selection": "ATTACK"}

    # 2. Select Attack Target - should only include diagonal enemies
    req = process_stack(wasp_boomerang_state).input_request
    assert req["type"] == "SELECT_UNIT"

    valid_options = req["valid_options"]

    # Should include: enemy_diagonal, enemy_diagonal2 (not in straight line)
    assert "enemy_diagonal" in valid_options, "Diagonal target should be valid"
    assert "enemy_diagonal2" in valid_options, "Second diagonal target should be valid"

    # Should NOT include: enemy_adjacent (adjacent = in straight line)
    assert (
        "enemy_adjacent" not in valid_options
    ), "Adjacent targets should be excluded (in straight line)"

    # Should NOT include: enemy_straight (same q-axis = in straight line)
    assert "enemy_straight" not in valid_options, "Straight line targets should be excluded"


def test_charged_boomerang_attack_resolves(wasp_boomerang_state):
    """
    Test that the attack resolves correctly on a valid diagonal target.
    """
    step = ResolveCardStep(hero_id="wasp")
    push_steps(wasp_boomerang_state, [step])

    # Action Choice
    req = process_stack(wasp_boomerang_state).input_request
    wasp_boomerang_state.execution_stack[-1].pending_input = {"selection": "ATTACK"}

    # Select diagonal target
    req = process_stack(wasp_boomerang_state).input_request
    assert "enemy_diagonal" in req["valid_options"]
    wasp_boomerang_state.execution_stack[-1].pending_input = {"selection": "enemy_diagonal"}

    # Reaction window - pass
    req = process_stack(wasp_boomerang_state).input_request
    assert req["type"] == "SELECT_CARD_OR_PASS"
    wasp_boomerang_state.execution_stack[-1].pending_input = {"selection": "PASS"}

    # Finish resolution
    _ = process_stack(wasp_boomerang_state).input_request

    # Target should be defeated (no defense, damage 3 vs defense None)
    assert "enemy_diagonal" not in wasp_boomerang_state.entity_locations


def test_charged_boomerang_no_valid_targets():
    """
    Test behavior when all enemies are in a straight line (no valid targets).
    """
    board = Board()
    hexes = {
        Hex(q=0, r=0, s=0),  # Wasp
        Hex(q=1, r=0, s=-1),  # Adjacent, straight line
        Hex(q=2, r=0, s=-2),  # Range 2, straight line
        Hex(q=3, r=0, s=-3),  # Range 3, straight line
    }
    z1 = Zone(id="z1", hexes=hexes, neighbors=[])
    board.zones = {"z1": z1}
    board.populate_tiles_from_zones()

    wasp = Hero(id="wasp", name="Wasp", team=TeamColor.RED, deck=[], level=1)
    card = Card(
        id="charged_boomerang",
        name="Charged Boomerang",
        tier=CardTier.II,
        color=CardColor.RED,
        initiative=9,
        primary_action=ActionType.ATTACK,
        primary_action_value=3,
        is_ranged=True,
        range_value=3,
        effect_id="charged_boomerang",
        effect_text="Target a unit in range and not in a straight line.",
        is_facedown=False,
    )
    wasp.current_turn_card = card

    enemy = Hero(id="enemy", name="Enemy", team=TeamColor.BLUE, deck=[], level=1)

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[wasp], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[enemy], minions=[]),
        },
    )

    state.place_entity("wasp", Hex(q=0, r=0, s=0))
    state.place_entity("enemy", Hex(q=2, r=0, s=-2))  # Straight line only
    state.current_actor_id = "wasp"

    step = ResolveCardStep(hero_id="wasp")
    push_steps(state, [step])

    # ATTACK is unavailable because every enemy is in a straight line.
    req = process_stack(state).input_request
    assert req is not None
    assert "ATTACK" not in {option.id for option in req.options}
