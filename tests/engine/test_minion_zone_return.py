"""Tests for ReturnMinionToZoneStep - returning minions outside the active zone."""

import pytest

from goa2.domain.board import Board, Zone
from goa2.domain.events import GameEventType
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequestType, InputResponse, selection_value
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
from goa2.engine.steps import FinalizeHeroTurnStep, ReturnMinionToZoneStep


def create_minion(id_str, team, m_type=MinionType.MELEE):
    return Minion(id=UnitID(id_str), name=id_str, team=team, type=m_type)


def magnetic_dagger_effect(source_id: str) -> ActiveEffect:
    return ActiveEffect(
        id="magnetic_dagger",
        source_id=source_id,
        effect_type=EffectType.PLACEMENT_PREVENTION,
        scope=EffectScope(
            shape=Shape.RADIUS,
            range=3,
            origin_id=source_id,
            affects=AffectsFilter.ENEMY_UNITS,
        ),
        duration=DurationType.THIS_TURN,
        is_active=True,
        created_at_turn=1,
        created_at_round=1,
        displacement_blocks=[DisplacementType.PLACE, DisplacementType.SWAP],
        blocks_enemy_actors=True,
        blocks_friendly_actors=False,
        blocks_self=False,
    )


def add_blocked_then_legal_minions(state: GameState):
    """A blocked RED placement precedes a legal BLUE move in tie-break order."""
    bridge = Hex(q=1, r=-1, s=0)
    state.board.tiles[bridge].is_terrain = True
    blocked = create_minion("a_blocked", TeamColor.RED)
    legal = create_minion("b_legal", TeamColor.BLUE)
    state.teams[TeamColor.RED].minions.append(blocked)
    state.teams[TeamColor.BLUE].minions.append(legal)
    state.move_unit(blocked.id, Hex(q=2, r=-2, s=0))
    legal_start = Hex(q=0, r=1, s=-1)
    state.board.tiles[legal_start] = Tile(hex=legal_start)
    state.move_unit(legal.id, legal_start)
    caster = Hero(id="hero_wasp", name="Wasp", team=TeamColor.BLUE, deck=[])
    state.teams[TeamColor.BLUE].heroes.append(caster)
    caster_hex = Hex(q=1, r=0, s=-1)
    state.board.tiles[caster_hex] = Tile(hex=caster_hex)
    state.move_unit(caster.id, caster_hex)
    state.active_effects.append(magnetic_dagger_effect(str(caster.id)))
    return blocked, legal, caster


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
    result = process_stack(zone_state)

    # The empty detour takes three steps, not the two-hex geometric distance.
    assert zone_state.unit_locations.get(m_red.id) == Hex(q=0, r=0, s=0)
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_MOVED]
    assert result.events[0].metadata["range"] == 3


def test_return_minion_respects_obstacles_fallback(zone_state):
    """If the direct path to the zone is completely blocked (no traversable path), the minion is Placed instead (ignores path obstacles)."""
    m_red = create_minion("r1", TeamColor.RED)
    zone_state.teams[TeamColor.RED].minions.append(m_red)

    # Place red minion outside
    zone_state.move_unit(m_red.id, Hex(q=2, r=-2, s=0))

    # Place an obstacle blocking the only possible route to the zone at Hex(1,-1,0)
    blocking_minion = create_minion("blocker", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(blocking_minion)
    zone_state.move_unit(blocking_minion.id, Hex(q=1, r=-1, s=0))

    # No alternative path is added, so it is completely blocked.
    # When ReturnMinionToZoneStep runs, it should fail to find a traversable path (since q=1, r=-1, s=0 is blocked),
    # but then fall back to placement (ignoring obstacles for path propagation) and return m_red to the empty space (q=0, r=0, s=0).
    step = ReturnMinionToZoneStep()
    push_steps(zone_state, [step])
    result = process_stack(zone_state)

    # m_red should be Placed at Hex(0,0,0) (via the fallback placement rule)
    assert zone_state.unit_locations.get(m_red.id) == Hex(q=0, r=0, s=0)
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_PLACED]


@pytest.mark.parametrize("choose_destination", [False, True], ids=["automatic", "team-choice"])
def test_zone_return_moves_despite_placement_prevention(zone_state, choose_destination):
    # Isolate the return step under the placement restriction used by Magnetic
    # Dagger. This is not a test of Wasp's attack or effect creation sequence.
    minion = create_minion("b1", TeamColor.BLUE)
    zone_state.teams[TeamColor.BLUE].minions.append(minion)
    outside = Hex(q=2, r=-2, s=0)
    zone_state.move_unit(minion.id, outside)
    wasp = Hero(id="hero_wasp", name="Wasp", team=TeamColor.RED, deck=[])
    zone_state.teams[TeamColor.RED].heroes.append(wasp)
    zone_state.move_unit(wasp.id, Hex(q=0, r=0, s=0))
    zone_state.active_effects.append(magnetic_dagger_effect(str(wasp.id)))
    target = Hex(q=1, r=-1, s=0)
    if choose_destination:
        target = Hex(q=2, r=-1, s=-1)
        zone_state.board.zones["battle_zone"].hexes.add(target)
        zone_state.board.tiles[target] = Tile(hex=target, zone_id="battle_zone")

    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)
    if choose_destination:
        assert result.input_request is not None
        assert result.input_request.request_type == InputRequestType.SELECT_HEX
        assert result.input_request.player_id == "team:BLUE"
        assert zone_state.get_position(str(minion.id)) == outside
        submit_input(
            zone_state,
            InputResponse(request_id=result.input_request.id, selection=target.model_dump()),
        )
        result = process_stack(zone_state)

    assert result.input_request is None
    assert zone_state.get_position(str(minion.id)) == target
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_MOVED]
    assert result.events[0].metadata["range"] == 1


@pytest.mark.parametrize("split_active", [False, True], ids=["open-route", "split-route"])
def test_return_distinguishes_walking_from_placement_across_reality_split(split_active):
    # Both endpoints lie on the positive side of q=0, but the only empty path
    # crosses onto the negative side. A split forbids walking that path; it
    # does not forbid placement between the same-side endpoints.
    route = [
        Hex(q=1, r=0, s=-1),
        Hex(q=0, r=0, s=0),
        Hex(q=-1, r=0, s=1),
        Hex(q=-1, r=-1, s=2),
        Hex(q=-1, r=-2, s=3),
        Hex(q=0, r=-3, s=3),
        Hex(q=1, r=-4, s=3),
    ]
    start, target = route[0], route[-1]
    caster_hex = Hex(q=0, r=1, s=-1)
    minion = create_minion("r1", TeamColor.RED)
    caster = Hero(id="hero_nebkher", name="NebKher", team=TeamColor.BLUE, deck=[])
    state = GameState(
        board=Board(
            zones={"battle_zone": Zone(id="battle_zone", hexes={target})},
            tiles={hex_: Tile(hex=hex_) for hex_ in [*route, caster_hex]},
        ),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, minions=[minion]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[caster]),
        },
        active_zone_id="battle_zone",
    )
    state.move_unit(minion.id, start)
    state.move_unit(caster.id, caster_hex)
    if split_active:
        state.active_effects.append(
            ActiveEffect(
                id="reality_split",
                source_id=caster.id,
                effect_type=EffectType.TOPOLOGY_SPLIT,
                split_axis="q",
                split_value=0,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_ROUND,
                created_at_turn=state.turn,
                created_at_round=state.round,
                is_active=True,
            )
        )

    push_steps(state, [ReturnMinionToZoneStep()])
    result = process_stack(state)

    assert result.input_request is None
    assert state.get_position(str(minion.id)) == target
    expected_event = GameEventType.UNIT_PLACED if split_active else GameEventType.UNIT_MOVED
    assert [event.event_type for event in result.events] == [expected_event]


@pytest.mark.parametrize("choose_destination", [False, True], ids=["automatic", "team-choice"])
def test_placement_fallback_skips_nearer_hex_across_reality_split(choose_destination):
    start = Hex(q=-1, r=0, s=1)
    across_split = Hex(q=1, r=0, s=-1)
    legal = Hex(q=-1, r=-3, s=4)
    route = [
        start,
        Hex(q=0, r=0, s=0),
        across_split,
        Hex(q=-1, r=-1, s=2),
        Hex(q=-1, r=-2, s=3),
        legal,
    ]
    destinations = {across_split, legal}
    if choose_destination:
        alternative = Hex(q=-2, r=-2, s=4)
        route.append(alternative)
        destinations.add(alternative)
    board = Board(
        zones={"battle_zone": Zone(id="battle_zone", hexes=destinations)},
        tiles={hex_: Tile(hex=hex_) for hex_ in route},
    )
    # Terrain on both exits forces the rule's placement fallback.
    board.tiles[route[1]].is_terrain = True
    board.tiles[route[3]].is_terrain = True
    minion = create_minion("r1", TeamColor.RED)
    caster = Hero(id="hero_nebkher", name="NebKher", team=TeamColor.BLUE, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, minions=[minion]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[caster]),
        },
        active_zone_id="battle_zone",
    )
    state.move_unit(minion.id, start)
    caster_hex = Hex(q=0, r=1, s=-1)
    state.board.tiles[caster_hex] = Tile(hex=caster_hex)
    state.move_unit(caster.id, caster_hex)
    state.active_effects.append(
        ActiveEffect(
            id="reality_split",
            source_id=str(caster.id),
            effect_type=EffectType.TOPOLOGY_SPLIT,
            split_axis="q",
            split_value=0,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )

    push_steps(state, [ReturnMinionToZoneStep()])
    result = process_stack(state)
    if choose_destination:
        assert result.input_request is not None
        assert result.input_request.player_id == "team:RED"
        assert {Hex(**selection_value(option)) for option in result.input_request.options} == {
            legal,
            alternative,
        }
        submit_input(
            state,
            InputResponse(request_id=result.input_request.id, selection=legal.model_dump()),
        )
        result = process_stack(state)

    assert result.input_request is None
    assert state.get_position(str(minion.id)) == legal
    assert [event.event_type for event in result.events] == [GameEventType.UNIT_PLACED]


@pytest.mark.parametrize("resume_from_json", [False, True], ids=["automatic", "saved-choice"])
def test_blocked_minion_does_not_prevent_later_legal_return(zone_state, resume_from_json):
    blocked, legal, _ = add_blocked_then_legal_minions(zone_state)
    destination = Hex(q=0, r=0, s=0)
    if resume_from_json:
        destination = Hex(q=-1, r=1, s=0)
        zone_state.board.zones["battle_zone"].hexes.add(destination)
        zone_state.board.tiles[destination] = Tile(hex=destination, zone_id="battle_zone")

    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)
    if resume_from_json:
        assert result.input_request is not None
        # Save while choosing for the second minion. The already-blocked first
        # minion must not be reconsidered after loading or an invalid answer.
        zone_state = GameState.model_validate_json(zone_state.model_dump_json())
        submit_input(
            zone_state,
            InputResponse(
                request_id=result.input_request.id, selection=Hex(q=2, r=-2, s=0).model_dump()
            ),
        )
        result = process_stack(zone_state)
        assert result.input_request is not None
        assert str(legal.id) in result.input_request.prompt
        submit_input(
            zone_state,
            InputResponse(request_id=result.input_request.id, selection=destination.model_dump()),
        )
        result = process_stack(zone_state)

    assert result.input_request is None
    assert zone_state.get_position(str(blocked.id)) == Hex(q=2, r=-2, s=0)
    assert zone_state.get_position(str(legal.id)) == destination


def _turn_card(card_id: str, initiative: int = 1) -> Card:
    return Card(
        id=card_id,
        name=card_id,
        tier="I",
        color="BLUE",
        primary_action="SKILL",
        initiative=initiative,
        effect_id="none",
        effect_text="",
        is_facedown=False,
    )


@pytest.mark.parametrize("has_next_actor", [True, False], ids=["next-actor", "end-turn"])
def test_blocked_post_finalize_return_cannot_stall_phase_progress(zone_state, has_next_actor):
    # Generic step regression using Magnetic Dagger's real placement restriction;
    # this intentionally does not drive Wasp's card flow.
    blocked, legal, actor = add_blocked_then_legal_minions(zone_state)
    zone_state.remove_entity(legal.id)
    zone_state.teams[TeamColor.BLUE].minions.remove(legal)
    actor.current_turn_card = _turn_card("current", initiative=2)
    actor.hand = [_turn_card("future")]
    zone_state.phase = GamePhase.RESOLUTION
    zone_state.current_actor_id = actor.id
    zone_state.resolution_owner_id = actor.id

    next_hero = None
    if has_next_actor:
        next_hero = Hero(
            id="hero_next",
            name="Next",
            team=TeamColor.RED,
            deck=[],
            hand=[_turn_card("next_future")],
            current_turn_card=_turn_card("next_current", initiative=3),
        )
        zone_state.teams[TeamColor.RED].heroes.append(next_hero)
        next_hex = Hex(q=-1, r=1, s=0)
        zone_state.board.tiles[next_hex] = Tile(hex=next_hex)
        zone_state.move_unit(next_hero.id, next_hex)
        zone_state.unresolved_hero_ids = [next_hero.id]

    push_steps(zone_state, [FinalizeHeroTurnStep(hero_id=str(actor.id))])
    result = process_stack(zone_state)

    assert zone_state.get_position(str(blocked.id)) == Hex(q=2, r=-2, s=0)
    if has_next_actor:
        assert result.input_request is not None
        assert result.input_request.player_id == str(next_hero.id)
        assert zone_state.current_actor_id == next_hero.id
        assert any(effect.id == "magnetic_dagger" for effect in zone_state.active_effects)
    else:
        assert result.input_request is None
        assert zone_state.turn == 2
        assert zone_state.phase == GamePhase.PLANNING


@pytest.mark.parametrize("actions_only", [False, True], ids=["all-movement", "actions-only"])
def test_movement_restriction_uses_placement_fallback_when_needed(zone_state, actions_only):
    # The bounding rule also permits placement when the minion cannot move;
    # automatic returns are not movement actions. Isolate those two contracts.
    target = Hex(q=0, r=0, s=0)
    zone_state.board.zones["battle_zone"].hexes = {target}
    zone_state.board.tiles[Hex(q=1, r=-1, s=0)].zone_id = None
    minion = create_minion("r1", TeamColor.RED)
    caster = Hero(id="limiter", name="Limiter", team=TeamColor.BLUE, deck=[])
    zone_state.teams[TeamColor.RED].minions.append(minion)
    zone_state.teams[TeamColor.BLUE].heroes.append(caster)
    caster_hex = Hex(q=3, r=-2, s=-1)
    zone_state.board.tiles[caster_hex] = Tile(hex=caster_hex)
    zone_state.move_unit(caster.id, caster_hex)
    zone_state.move_unit(minion.id, Hex(q=2, r=-2, s=0))
    zone_state.active_effects.append(
        ActiveEffect(
            id="movement_cap",
            source_id=caster.id,
            effect_type=EffectType.MOVEMENT_ZONE,
            scope=EffectScope(
                shape=Shape.ADJACENT,
                origin_id=caster.id,
                affects=AffectsFilter.ENEMY_UNITS,
            ),
            max_value=1,
            limit_actions_only=actions_only,
            duration=DurationType.THIS_TURN,
            created_at_turn=zone_state.turn,
            created_at_round=zone_state.round,
            is_active=True,
        )
    )

    push_steps(zone_state, [ReturnMinionToZoneStep()])
    result = process_stack(zone_state)

    assert result.input_request is None
    assert zone_state.get_position(str(minion.id)) == target
    expected = GameEventType.UNIT_MOVED if actions_only else GameEventType.UNIT_PLACED
    assert [event.event_type for event in result.events] == [expected]
