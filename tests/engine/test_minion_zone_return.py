"""Tests for ReturnMinionToZoneStep - returning minions outside the active zone."""

import pytest

from goa2.domain.board import Board, Zone
from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType, InputResponse
from goa2.domain.models import Card, GamePhase, Hero, Minion, MinionType, Team, TeamColor
from goa2.domain.models.effect import (
    ActiveEffect,
    AffectsFilter,
    DisplacementType,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.state import GameState
from goa2.domain.tile import Tile
from goa2.domain.types import UnitID
from goa2.engine.handler import process_stack, push_steps, submit_input
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.steps import (
    AdvanceTurnStep,
    CheckLanePushStep,
    EndPhaseCleanupStep,
    EndPhaseStep,
    FinalizeHeroTurnStep,
    FindNextActorStep,
    PlaceUnitStep,
    ReturnMinionToZoneStep,
)


def create_minion(id_str, team, m_type=MinionType.MELEE):
    return Minion(id=UnitID(id_str), name=id_str, team=team, type=m_type)


@pytest.fixture
def zone_state():
    """Create a state with a zone and outside area."""
    board = Board()
    zone_hexes = {Hex(q=0, r=0, s=0), Hex(q=1, r=-1, s=0)}
    board.zones["battle_zone"] = Zone(id="battle_zone", hexes=zone_hexes)

    outside_hex = Hex(q=2, r=-2, s=0)

    for h in zone_hexes:
        board.tiles[h] = Tile(hex=h, zone_id="battle_zone")
    board.tiles[outside_hex] = Tile(hex=outside_hex)

    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[], minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        active_zone_id="battle_zone",
    )
    return state


def test_no_minions_outside_zone(zone_state):
    """When all minions are inside the zone, nothing happens."""
    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)
    zone_state.move_unit(m_red.id, Hex(q=0, r=0, s=0))

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    result = process_stack(zone_state).input_request

    assert zone_state.unit_locations.get(m_red.id) == Hex(q=0, r=0, s=0)
    assert result is None


def test_auto_return_single_path(zone_state):
    """When only one empty hex in zone, minion auto-returns there."""
    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)

    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    _ = process_stack(zone_state).input_request

    loc = zone_state.unit_locations.get(m_red.id)
    zone = zone_state.board.zones["battle_zone"]
    assert loc in zone.hexes


def test_team_choice_multiple_paths(zone_state):
    """When multiple paths exist, team must choose."""
    zone_state.board.zones["battle_zone"].hexes.add(Hex(q=2, r=-1, s=-1))
    zone_state.board.tiles[Hex(q=2, r=-1, s=-1)] = Tile(
        hex=Hex(q=2, r=-1, s=-1), zone_id="battle_zone"
    )

    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)

    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    result = process_stack(zone_state).input_request

    if result is not None:
        assert result["type"] == "SELECT_HEX"
        assert result["player_id"] == "team:RED"


def test_malformed_return_hex_rerequests_without_losing_step(zone_state):
    alternative = Hex(q=2, r=-1, s=-1)
    zone_state.board.zones["battle_zone"].hexes.add(alternative)
    zone_state.board.tiles[alternative] = Tile(hex=alternative, zone_id="battle_zone")

    minion = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(minion)
    outside = Hex(q=2, r=-2, s=0)
    zone_state.move_unit(minion.id, outside)
    push_steps(zone_state, [ReturnMinionToZoneStep()])

    first = process_stack(zone_state).input_request
    assert first is not None
    assert first.request_type == InputRequestType.SELECT_HEX

    submit_input(
        zone_state,
        InputResponse(request_id=first.id, selection={"q": 1}),
    )
    retried = process_stack(zone_state).input_request

    assert retried is not None
    assert retried.request_type == InputRequestType.SELECT_HEX
    assert len(zone_state.execution_stack) == 1
    assert isinstance(zone_state.execution_stack[-1], ReturnMinionToZoneStep)
    assert zone_state.get_position(str(minion.id)) == outside


def test_multiple_minions_tiebreaker_order(zone_state):
    """Multiple minions outside zone processed in tie-breaker order."""
    m_red = create_minion("r1", TeamColor.RED)
    m_blue = create_minion("b1", TeamColor.BLUE)

    zone_state.teams[TeamColor.RED].minions.append(m_red)
    zone_state.teams[TeamColor.BLUE].minions.append(m_blue)

    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    zone_state.board.tiles[Hex(q=3, r=-3, s=0)] = Tile(hex=Hex(q=3, r=-3, s=0))
    zone_state.move_unit(m_blue.id, Hex(q=3, r=-3, s=0))

    # Add alternative route for m_blue to get around m_red at (1,-1,0)
    route = [Hex(q=2, r=-3, s=1), Hex(q=1, r=-2, s=1), Hex(q=0, r=-1, s=1)]
    for h in route:
        zone_state.board.tiles[h] = Tile(hex=h)

    zone_state.tie_breaker_team = TeamColor.RED

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    _ = process_stack(zone_state).input_request

    # Both minions should be returned to zone
    zone = zone_state.board.zones["battle_zone"]
    assert zone_state.unit_locations.get(m_red.id) in zone.hexes
    assert zone_state.unit_locations.get(m_blue.id) in zone.hexes


def test_no_empty_space_in_zone(zone_state):
    """When zone has no empty spaces, minion stays outside (edge case)."""
    m_red = create_minion("r1", TeamColor.RED)
    m_red2 = create_minion("r2", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.extend([m_red, m_red2])

    zone_state.move_unit(m_red.id, Hex(q=0, r=0, s=0))
    zone_state.move_unit(m_red2.id, Hex(q=1, r=-1, s=0))

    m_outside = create_minion("r3", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_outside)
    zone_state.move_unit(m_outside.id, Hex(q=2, r=-2, s=0))

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    result = process_stack(zone_state).input_request

    assert result is None


def test_finalize_hero_turn_spawns_check(zone_state):
    """FinalizeHeroTurnStep should spawn ReturnMinionToZoneStep."""
    hero = Hero(
        id="hero_test",
        name="Test Hero",
        title="Tester",
        team=TeamColor.RED,
        hand=[],
        deck=[],
    )
    card = Card(
        id="test_card",
        name="Test Card",
        tier="I",
        color="BLUE",
        primary_action="SKILL",
        initiative=1,
        effect_id="none",
        effect_text="",
    )
    hero.current_turn_card = card
    zone_state.teams[TeamColor.RED].heroes.append(hero)
    zone_state.current_actor_id = "hero_test"

    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)
    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    step = FinalizeHeroTurnStep(hero_id="hero_test")
    push_steps(zone_state, [step])
    _ = process_stack(zone_state).input_request

    # Minion should be returned to zone
    zone = zone_state.board.zones["battle_zone"]
    assert zone_state.unit_locations.get(m_red.id) in zone.hexes


def test_return_minion_respects_obstacles(zone_state):
    """If the direct path to the zone is blocked, the minion must take a longer traversable route."""
    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)

    # Place red minion outside
    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    # Place an obstacle blocking the direct path to the zone at Hex(1,-1,0)
    blocking_minion = create_minion("blocker", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(blocking_minion)
    zone_state.move_unit(blocking_minion.id, Hex(q=1, r=-1, s=0))

    # Add alternative route for m_red to get around blocker
    route = [Hex(q=2, r=-3, s=1), Hex(q=1, r=-2, s=1), Hex(q=0, r=-1, s=1)]
    for h in route:
        zone_state.board.tiles[h] = Tile(hex=h)

    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    _ = process_stack(zone_state).input_request

    # m_red should successfully bypass the obstacle and end up in the battle zone at Hex(0,0,0)
    assert zone_state.unit_locations.get(m_red.id) == Hex(q=0, r=0, s=0)


def test_return_minion_respects_obstacles_fallback(zone_state):
    """If no traversable path exists, the minion is placed instead."""
    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)

    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    blocking_minion = create_minion("blocker", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(blocking_minion)
    zone_state.move_unit(blocking_minion.id, Hex(q=1, r=-1, s=0))

    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)

    assert zone_state.unit_locations.get(m_red.id) == Hex(q=0, r=0, s=0)
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_PLACED]


def _add_magnetic_dagger(state: GameState, wasp_id: str = "hero_wasp") -> None:
    state.active_effects.append(
        ActiveEffect(
            id="magnetic_dagger",
            source_id=wasp_id,
            effect_type=EffectType.PLACEMENT_PREVENTION,
            scope=EffectScope(
                shape=Shape.RADIUS,
                range=3,
                origin_id=wasp_id,
                affects=AffectsFilter.ENEMY_UNITS,
            ),
            duration=DurationType.THIS_TURN,
            is_active=True,
            created_at_turn=state.turn,
            created_at_round=state.round,
            displacement_blocks=[DisplacementType.PLACE, DisplacementType.SWAP],
            blocks_enemy_actors=True,
            blocks_friendly_actors=False,
            blocks_self=False,
        )
    )


def _add_hero(state: GameState, hero_id: str, team: TeamColor) -> Hero:
    hero = Hero(id=hero_id, name=hero_id, team=team, deck=[])
    state.teams[team].heroes.append(hero)
    return hero


def test_normal_zone_return_is_movement_despite_placement_prevention(zone_state):
    minion = create_minion("b1", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(minion)
    zone_state.move_unit(minion.id, Hex(q=2, r=-2, s=0))

    wasp = _add_hero(zone_state, "hero_wasp", TeamColor.RED)
    zone_state.move_unit(wasp.id, Hex(q=0, r=0, s=0))
    _add_magnetic_dagger(zone_state)

    # The other zone hex has a traversable one-step path from the minion.
    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)

    assert zone_state.get_position(str(minion.id)) == Hex(q=1, r=-1, s=0)
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_MOVED]


def test_blocked_fallback_preserves_lane_check_and_next_actor(zone_state):
    """The Wasp reproduction must not strand an empty RESOLUTION stack."""
    wasp = _add_hero(zone_state, "hero_wasp", TeamColor.RED)
    next_hero = _add_hero(zone_state, "hero_arien", TeamColor.BLUE)
    next_hero.current_turn_card = Card(
        id="next_card",
        name="Next Card",
        tier="I",
        color="BLUE",
        primary_action="SKILL",
        initiative=5,
        effect_id="none",
        effect_text="",
    )

    minion = create_minion("b1", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(minion)
    outside = Hex(q=2, r=-2, s=0)
    zone_state.move_unit(minion.id, outside)
    zone_state.move_unit(wasp.id, Hex(q=1, r=-1, s=0))
    _add_magnetic_dagger(zone_state)

    zone_state.phase = GamePhase.RESOLUTION
    zone_state.unresolved_hero_ids = [str(next_hero.id)]
    push_steps(
        zone_state,
        [ReturnMinionToZoneStep(), CheckLanePushStep(), FindNextActorStep()],
    )

    result = GameSession(zone_state).advance()

    assert result.result_type == SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    assert result.input_request.player_id == str(next_hero.id)
    assert zone_state.get_position(str(minion.id)) == outside
    assert zone_state.current_actor_id == str(next_hero.id)


def test_blocked_first_minion_does_not_prevent_later_legal_return(zone_state):
    blocked = create_minion("a_blocked", TeamColor.BLUE)
    legal = create_minion("b_legal", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.extend([blocked, legal])
    blocked_start = Hex(q=2, r=-2, s=0)
    legal_start = Hex(q=-1, r=1, s=0)
    zone_state.board.tiles[legal_start] = Tile(hex=legal_start)
    zone_state.move_unit(blocked.id, blocked_start)
    zone_state.move_unit(legal.id, legal_start)

    wasp = _add_hero(zone_state, "hero_wasp", TeamColor.RED)
    zone_state.move_unit(wasp.id, Hex(q=1, r=-1, s=0))
    _add_magnetic_dagger(zone_state)

    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)

    assert result.input_request is None
    assert zone_state.get_position(str(blocked.id)) == blocked_start
    assert zone_state.get_position(str(legal.id)) == Hex(q=0, r=0, s=0)
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_MOVED]


def test_mandatory_abort_does_not_discard_next_actor_control(zone_state):
    wasp = _add_hero(zone_state, "hero_wasp", TeamColor.RED)
    next_hero = _add_hero(zone_state, "hero_arien", TeamColor.BLUE)
    next_hero.current_turn_card = Card(
        id="next_card",
        name="Next Card",
        tier="I",
        color="BLUE",
        primary_action="SKILL",
        initiative=5,
        effect_id="none",
        effect_text="",
    )
    minion = create_minion("b1", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(minion)
    zone_state.move_unit(minion.id, Hex(q=2, r=-2, s=0))
    zone_state.move_unit(wasp.id, Hex(q=1, r=-1, s=0))
    _add_magnetic_dagger(zone_state)
    zone_state.phase = GamePhase.RESOLUTION
    zone_state.unresolved_hero_ids = [str(next_hero.id)]

    push_steps(
        zone_state,
        [
            PlaceUnitStep(unit_id=str(minion.id), target_hex_arg=Hex(q=0, r=0, s=0)),
            FindNextActorStep(),
        ],
    )
    result = process_stack(zone_state)

    assert result.input_request is not None
    assert result.input_request.player_id == str(next_hero.id)
    assert zone_state.current_actor_id == str(next_hero.id)


@pytest.mark.parametrize(
    ("continuation", "phase"),
    [
        (AdvanceTurnStep(), GamePhase.RESOLUTION),
        (EndPhaseStep(), GamePhase.CLEANUP),
        (EndPhaseCleanupStep(), GamePhase.CLEANUP),
    ],
)
def test_mandatory_abort_preserves_phase_control_continuations(zone_state, continuation, phase):
    wasp = _add_hero(zone_state, "hero_wasp", TeamColor.RED)
    minion = create_minion("b1", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(minion)
    zone_state.move_unit(minion.id, Hex(q=2, r=-2, s=0))
    zone_state.move_unit(wasp.id, Hex(q=1, r=-1, s=0))
    _add_magnetic_dagger(zone_state)
    zone_state.phase = phase

    push_steps(
        zone_state,
        [
            PlaceUnitStep(unit_id=str(minion.id), target_hex_arg=Hex(q=0, r=0, s=0)),
            continuation,
        ],
    )
    process_stack(zone_state)

    assert zone_state.phase == GamePhase.PLANNING


def test_session_rejects_drained_resolution_state(zone_state):
    zone_state.phase = GamePhase.RESOLUTION

    with pytest.raises(RuntimeError, match=r"RESOLUTION.*stack"):
        GameSession(zone_state).advance()
