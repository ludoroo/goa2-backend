from goa2.domain.board import Board
from goa2.domain.hex import Hex
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    CardTier,
    Hero,
    Team,
    TeamColor,
    Token,
    TokenType,
)
from goa2.domain.state import GameState
from goa2.domain.tile import Tile
from goa2.domain.types import BoardEntityID, HeroID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.steps import MoveSequenceStep, MoveUnitStep, TriggerMineStep


def _make_state_with_mine():
    board = Board()
    for h in [Hex(q=0, r=0, s=0), Hex(q=1, r=-1, s=0), Hex(q=2, r=-2, s=0)]:
        board.tiles[h] = Tile(hex=h)

    hero = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.BLUE, deck=[])
    mine_owner = Hero(id=HeroID("hero_min"), name="Min", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[hero], minions=[]),
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[mine_owner], minions=[]),
        },
        current_actor_id=HeroID("hero_a"),
    )
    state.place_entity(BoardEntityID("hero_a"), Hex(q=0, r=0, s=0))

    mine = Token(
        id=BoardEntityID("mine_1"),
        name="Mine",
        token_type=TokenType.MINE_BLAST,
        owner_id=HeroID("hero_min"),
        is_passable=True,
        is_facedown=True,
    )
    state.token_pool[TokenType.MINE_BLAST] = [mine]
    state.misc_entities[BoardEntityID("mine_1")] = mine
    state.place_entity(BoardEntityID("mine_1"), Hex(q=1, r=-1, s=0))
    return state


def test_mine_triggered_and_removed_after_movement():
    """Moving through a mine triggers it and removes the token."""
    state = _make_state_with_mine()
    push_steps(state, [MoveSequenceStep(range_val=2)])

    req = process_stack(state).input_request
    assert req["type"] == "SELECT_HEX"

    state.execution_stack[-1].pending_input = {"selection": {"q": 2, "r": -2, "s": 0}}
    _ = process_stack(state).input_request

    assert state.entity_locations[BoardEntityID("hero_a")] == Hex(q=2, r=-2, s=0)
    assert BoardEntityID("mine_1") not in state.entity_locations


def test_trigger_mine_step_directly():
    """TriggerMineStep removes mines from context and emits events."""
    state = _make_state_with_mine()
    state.execution_context["triggered_mine_ids"] = ["mine_1"]

    push_steps(state, [TriggerMineStep()])
    _ = process_stack(state).input_request

    assert BoardEntityID("mine_1") not in state.entity_locations


def test_trigger_mine_step_no_mines():
    """TriggerMineStep with empty mine list does nothing."""
    state = _make_state_with_mine()
    state.execution_context["triggered_mine_ids"] = []

    push_steps(state, [TriggerMineStep()])
    result = process_stack(state).input_request

    assert result is None


def test_no_mine_triggered_when_no_passable_tokens():
    """Movement without passable tokens does not trigger anything."""
    board = Board()
    for h in [Hex(q=0, r=0, s=0), Hex(q=1, r=-1, s=0), Hex(q=2, r=-2, s=0)]:
        board.tiles[h] = Tile(hex=h)

    hero = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.BLUE, deck=[])
    state = GameState(
        board=board,
        teams={TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[hero], minions=[])},
        current_actor_id=HeroID("hero_a"),
    )
    state.place_entity(BoardEntityID("hero_a"), Hex(q=0, r=0, s=0))

    push_steps(state, [MoveSequenceStep(range_val=2)])

    req = process_stack(state).input_request
    assert req["type"] == "SELECT_HEX"

    state.execution_stack[-1].pending_input = {"selection": {"q": 2, "r": -2, "s": 0}}
    _ = process_stack(state).input_request

    assert state.entity_locations[BoardEntityID("hero_a")] == Hex(q=2, r=-2, s=0)


def test_forced_movement_triggers_mine():
    """MoveUnitStep without MinePathChoiceStep still triggers mines (forced movement)."""
    state = _make_state_with_mine()
    state.execution_context["target_hex"] = {"q": 2, "r": -2, "s": 0}

    push_steps(state, [MoveUnitStep(unit_id="hero_a", destination_key="target_hex", range_val=2)])
    _ = process_stack(state).input_request

    assert state.entity_locations[BoardEntityID("hero_a")] == Hex(q=2, r=-2, s=0)
    assert BoardEntityID("mine_1") not in state.entity_locations


def test_movement_cannot_end_on_mine() -> None:
    state = _make_state_with_mine()
    state.execution_context["target_hex"] = {"q": 1, "r": -1, "s": 0}

    push_steps(state, [MoveUnitStep(unit_id="hero_a", destination_key="target_hex", range_val=1)])
    process_stack(state)

    assert state.entity_locations[BoardEntityID("hero_a")] == Hex(q=0, r=0, s=0)
    assert state.entity_locations[BoardEntityID("mine_1")] == Hex(q=1, r=-1, s=0)


def test_blast_mine_forces_discard():
    """Walking through a blast mine forces the moved hero to discard a card."""
    state = _make_state_with_mine()
    hero = state.get_hero(HeroID("hero_a"))
    card = Card(
        id="card_1",
        name="Test Card",
        tier=CardTier.I,
        color=CardColor.RED,
        primary_action=ActionType.ATTACK,
        primary_action_value=2,
        secondary_actions={},
        effect_id="e",
        effect_text="t",
        initiative=5,
        state=CardState.HAND,
        is_facedown=False,
    )
    hero.hand.append(card)

    push_steps(state, [MoveSequenceStep(range_val=2)])
    req = process_stack(state).input_request
    assert req["type"] == "SELECT_HEX"

    state.execution_stack[-1].pending_input = {"selection": {"q": 2, "r": -2, "s": 0}}
    req = process_stack(state).input_request

    # Blast mine triggered — hero must discard
    assert req is not None
    assert req["type"] == "SELECT_CARD"
    assert req["player_id"] == "hero_a"

    state.execution_stack[-1].pending_input = {"selection": "card_1"}
    _ = process_stack(state).input_request

    assert len(hero.hand) == 0
    assert any(c.id == "card_1" for c in hero.discard_pile)


def test_second_move_in_same_turn_triggers_mine():
    """A second movement in the same turn must still trigger mines.

    Regression: multi-piece heroes (e.g. Razzle's "another you") move several
    pieces in one turn sharing one execution_context. The first move used to
    leave ``triggered_mine_ids`` in the context (even as an empty list), which
    made the second move's mine detection short-circuit and walk over a mine
    without triggering it.
    """
    board = Board()
    # Corridor A: hero_a moves here, no mine. Corridor B: hero_b crosses a mine.
    for h in [
        Hex(q=0, r=0, s=0),
        Hex(q=1, r=-1, s=0),
        Hex(q=2, r=-2, s=0),
        Hex(q=0, r=2, s=-2),
        Hex(q=1, r=1, s=-2),
        Hex(q=2, r=0, s=-2),
    ]:
        board.tiles[h] = Tile(hex=h)

    a = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.BLUE, deck=[])
    b = Hero(id=HeroID("hero_b"), name="B", team=TeamColor.BLUE, deck=[])
    mine_owner = Hero(id=HeroID("hero_min"), name="Min", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[a, b], minions=[]),
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[mine_owner], minions=[]),
        },
        current_actor_id=HeroID("hero_a"),
    )
    state.place_entity(BoardEntityID("hero_a"), Hex(q=0, r=0, s=0))
    state.place_entity(BoardEntityID("hero_b"), Hex(q=0, r=2, s=-2))

    mine = Token(
        id=BoardEntityID("mine_1"),
        name="Mine",
        token_type=TokenType.MINE_DUD,
        owner_id=HeroID("hero_min"),
        is_passable=True,
        is_facedown=True,
    )
    state.token_pool[TokenType.MINE_DUD] = [mine]
    state.misc_entities[BoardEntityID("mine_1")] = mine
    state.place_entity(BoardEntityID("mine_1"), Hex(q=1, r=1, s=-2))

    # First move: hero_a, no mine on its path (populates execution_context).
    state.execution_context["target_a"] = {"q": 2, "r": -2, "s": 0}
    push_steps(
        state,
        [MoveUnitStep(unit_id="hero_a", destination_key="target_a", range_val=2)],
    )
    process_stack(state)

    # Second move (same turn/context): hero_b walks across the mine.
    state.execution_context["target_b"] = {"q": 2, "r": 0, "s": -2}
    push_steps(
        state,
        [MoveUnitStep(unit_id="hero_b", destination_key="target_b", range_val=2)],
    )
    process_stack(state)

    assert state.entity_locations[BoardEntityID("hero_b")] == Hex(q=2, r=0, s=-2)
    assert BoardEntityID("mine_1") not in state.entity_locations


def _blast_card(cid: str) -> Card:
    return Card(
        id=cid,
        name="Test Card",
        tier=CardTier.I,
        color=CardColor.RED,
        primary_action=ActionType.ATTACK,
        primary_action_value=2,
        secondary_actions={},
        effect_id="e",
        effect_text="t",
        initiative=5,
        state=CardState.HAND,
        is_facedown=False,
    )


def _make_two_corridor_state_with_blast_mines() -> tuple[GameState, Hero, Hero]:
    """Two disjoint corridors, one enemy MINE_BLAST on each; each hero has one card."""
    board = Board()
    for h in [
        Hex(q=0, r=0, s=0),
        Hex(q=1, r=-1, s=0),
        Hex(q=2, r=-2, s=0),
        Hex(q=0, r=2, s=-2),
        Hex(q=1, r=1, s=-2),
        Hex(q=2, r=0, s=-2),
    ]:
        board.tiles[h] = Tile(hex=h)

    a = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.BLUE, deck=[])
    b = Hero(id=HeroID("hero_b"), name="B", team=TeamColor.BLUE, deck=[])
    a.hand.append(_blast_card("a_card"))
    b.hand.append(_blast_card("b_card"))
    mine_owner = Hero(id=HeroID("hero_min"), name="Min", team=TeamColor.RED, deck=[])
    state = GameState(
        board=board,
        teams={
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[a, b], minions=[]),
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[mine_owner], minions=[]),
        },
        current_actor_id=HeroID("hero_a"),
    )
    state.place_entity(BoardEntityID("hero_a"), Hex(q=0, r=0, s=0))
    state.place_entity(BoardEntityID("hero_b"), Hex(q=0, r=2, s=-2))

    m1 = Token(
        id=BoardEntityID("mine_1"),
        name="Mine",
        token_type=TokenType.MINE_BLAST,
        owner_id=HeroID("hero_min"),
        is_passable=True,
        is_facedown=True,
    )
    m2 = Token(
        id=BoardEntityID("mine_2"),
        name="Mine",
        token_type=TokenType.MINE_BLAST,
        owner_id=HeroID("hero_min"),
        is_passable=True,
        is_facedown=True,
    )
    state.token_pool[TokenType.MINE_BLAST] = [m1, m2]
    state.misc_entities[BoardEntityID("mine_1")] = m1
    state.misc_entities[BoardEntityID("mine_2")] = m2
    state.place_entity(BoardEntityID("mine_1"), Hex(q=1, r=-1, s=0))
    state.place_entity(BoardEntityID("mine_2"), Hex(q=1, r=1, s=-2))

    return state, a, b


def test_two_moves_each_trigger_own_blast_mine():
    """Two separate moves in one turn each trigger their own blast mine.

    Guards against regressions where per-move mine state leaks across moves:
    each mine must be removed and each move's blast must force *its own*
    victim (not the previous mover) to discard.
    """
    state, a, b = _make_two_corridor_state_with_blast_mines()

    # Move 1: hero_a across mine_1 -> blast forces hero_a to discard.
    state.execution_context["target_a"] = {"q": 2, "r": -2, "s": 0}
    push_steps(
        state,
        [MoveUnitStep(unit_id="hero_a", destination_key="target_a", range_val=2)],
    )
    req = process_stack(state).input_request
    assert req is not None and req["type"] == "SELECT_CARD"
    assert req["player_id"] == "hero_a"
    state.execution_stack[-1].pending_input = {"selection": "a_card"}
    process_stack(state)

    assert BoardEntityID("mine_1") not in state.entity_locations
    assert a.hand == []

    # Move 2 (same turn/context): hero_b across mine_2 -> blast forces hero_b.
    state.execution_context["target_b"] = {"q": 2, "r": 0, "s": -2}
    push_steps(
        state,
        [MoveUnitStep(unit_id="hero_b", destination_key="target_b", range_val=2)],
    )
    req = process_stack(state).input_request
    assert req is not None and req["type"] == "SELECT_CARD"
    # The second blast must target the second mover, not hero_a.
    assert req["player_id"] == "hero_b"
    state.execution_stack[-1].pending_input = {"selection": "b_card"}
    process_stack(state)

    assert BoardEntityID("mine_2") not in state.entity_locations
    assert b.hand == []


def test_batched_moves_route_each_blast_to_its_own_victim():
    """Batched MoveUnitStep resolves must not clobber each other's mine victim.

    Both moves commit before either TriggerMineStep runs, so a shared-context
    routing scheme would misroute the first blast to hero B.
    """
    state, a, b = _make_two_corridor_state_with_blast_mines()

    move_a = MoveUnitStep(unit_id="hero_a", destination_key="target_a", range_val=2)
    move_b = MoveUnitStep(unit_id="hero_b", destination_key="target_b", range_val=2)

    state.execution_context["target_a"] = {"q": 2, "r": -2, "s": 0}
    state.execution_context["target_b"] = {"q": 2, "r": 0, "s": -2}

    # Resolve both moves on the shared context before any trigger fires.
    result_a = move_a.resolve(state, state.execution_context)
    result_b = move_b.resolve(state, state.execution_context)

    assert state.entity_locations[BoardEntityID("hero_a")] == Hex(q=2, r=-2, s=0)
    assert state.entity_locations[BoardEntityID("hero_b")] == Hex(q=2, r=0, s=-2)
    assert any(isinstance(s, TriggerMineStep) for s in result_a.new_steps)
    assert any(isinstance(s, TriggerMineStep) for s in result_b.new_steps)

    # LIFO push: A_chain runs first, then B_chain.
    push_steps(state, [*result_a.new_steps, *result_b.new_steps])

    req = process_stack(state).input_request
    assert req is not None and req["type"] == "SELECT_CARD"
    assert req["player_id"] == "hero_a"
    assert req["valid_options"] == ["a_card"]
    state.execution_stack[-1].pending_input = {"selection": "a_card"}

    req = process_stack(state).input_request
    assert req is not None and req["type"] == "SELECT_CARD"
    assert req["player_id"] == "hero_b"
    assert req["valid_options"] == ["b_card"]
    state.execution_stack[-1].pending_input = {"selection": "b_card"}
    process_stack(state)

    assert BoardEntityID("mine_1") not in state.entity_locations
    assert BoardEntityID("mine_2") not in state.entity_locations
    assert a.hand == [] and any(c.id == "a_card" for c in a.discard_pile)
    assert b.hand == [] and any(c.id == "b_card" for c in b.discard_pile)


def test_dud_mine_no_discard():
    """Walking through a dud mine does NOT force a discard."""
    board = Board()
    for h in [Hex(q=0, r=0, s=0), Hex(q=1, r=-1, s=0), Hex(q=2, r=-2, s=0)]:
        board.tiles[h] = Tile(hex=h)

    hero = Hero(id=HeroID("hero_a"), name="A", team=TeamColor.BLUE, deck=[])
    mine_owner = Hero(id=HeroID("hero_min"), name="Min", team=TeamColor.RED, deck=[])
    card = Card(
        id="card_1",
        name="Test Card",
        tier=CardTier.I,
        color=CardColor.RED,
        primary_action=ActionType.ATTACK,
        primary_action_value=2,
        secondary_actions={},
        effect_id="e",
        effect_text="t",
        initiative=5,
        state=CardState.HAND,
        is_facedown=False,
    )
    hero.hand.append(card)

    state = GameState(
        board=board,
        teams={
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[hero], minions=[]),
            TeamColor.RED: Team(color=TeamColor.RED, heroes=[mine_owner], minions=[]),
        },
        current_actor_id=HeroID("hero_a"),
    )
    state.place_entity(BoardEntityID("hero_a"), Hex(q=0, r=0, s=0))

    mine = Token(
        id=BoardEntityID("mine_1"),
        name="Mine",
        token_type=TokenType.MINE_DUD,
        owner_id=HeroID("hero_min"),
        is_passable=True,
        is_facedown=True,
    )
    state.token_pool[TokenType.MINE_DUD] = [mine]
    state.misc_entities[BoardEntityID("mine_1")] = mine
    state.place_entity(BoardEntityID("mine_1"), Hex(q=1, r=-1, s=0))

    push_steps(state, [MoveSequenceStep(range_val=2)])
    req = process_stack(state).input_request
    assert req["type"] == "SELECT_HEX"

    state.execution_stack[-1].pending_input = {"selection": {"q": 2, "r": -2, "s": 0}}
    req = process_stack(state).input_request

    # Dud mine — no discard, movement completes
    assert req is None
    assert len(hero.hand) == 1
    assert BoardEntityID("mine_1") not in state.entity_locations
