"""Behavioral contracts for topology- and public-state-aware heuristics."""

from __future__ import annotations

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.hex import Hex
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models.card import Card
from goa2.domain.models.effect import (
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.enums import (
    ActionType,
    CardColor,
    CardTier,
    StatType,
    TeamColor,
)
from goa2.domain.state import GameState
from goa2.engine.hero_pieces import piece_id
from goa2.engine.map_logic import zones_between
from goa2.engine.setup import GameSetup
from goa2.engine.stats import calculate_minion_defense_modifier
from goa2.engine.topology import get_topology_service


@pytest.fixture
def state() -> GameState:
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp", "Xargatha"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=1,
    )


def _clear_board(state: GameState) -> None:
    for entity_id in list(state.entity_locations):
        state.remove_entity(entity_id)


def _hex_option(value: Hex) -> InputOption:
    return InputOption.from_value(value.model_dump())


def _movement_request(options: list[Hex]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.MOVEMENT_HEX,
        player_id="hero_wasp",
        options=[_hex_option(value) for value in options],
    )


def _unit_request(*unit_ids: str) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.SELECT_ENEMY,
        player_id="hero_wasp",
        options=[InputOption(id=unit_id, text=unit_id) for unit_id in unit_ids],
    )


def _action_request(*options: tuple[ActionType, int]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_wasp",
        options=[
            InputOption(
                id=action.name,
                text=action.name,
                metadata={"type": action, "value": value},
            )
            for action, value in options
        ],
    )


def _attack(card_id: str = "tactical_attack") -> Card:
    return Card(
        id=card_id,
        name=card_id,
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.ATTACK,
        primary_action_value=2,
        secondary_actions={},
        effect_id="",
        effect_text="",
        is_ranged=True,
        range_value=1,
        is_facedown=False,
    )


def _movement_or_fast_travel_card(movement: int) -> Card:
    return Card(
        id="travel_card",
        name="Travel Card",
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.MOVEMENT,
        primary_action_value=movement,
        secondary_actions={ActionType.FAST_TRAVEL: 0},
        effect_id="scripted_movement",
        effect_text="",
        is_facedown=False,
    )


def _secondary_attack_card(attack: int) -> Card:
    return Card(
        id="secondary_attack",
        name="Secondary Attack",
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.SKILL,
        primary_action_value=None,
        secondary_actions={ActionType.ATTACK: attack},
        effect_id="scripted_skill",
        effect_text="",
        is_ranged=True,
        range_value=1,
        is_facedown=False,
    )


@pytest.mark.parametrize(
    ("movement", "expected"),
    [(6, "MOVEMENT"), (1, "FAST_TRAVEL")],
    ids=["movement-reaches-a-better-zone", "fast-travel-finds-a-better-position"],
)
def test_choose_action_compares_best_legal_movement_and_fast_travel_destinations(
    state: GameState, movement: int, expected: str
) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=-4, r=-4, s=8))
    state.place_entity("hero_arien", Hex(q=-1, r=-3, s=4))
    wasp.current_turn_card = _movement_or_fast_travel_card(movement)
    request = _action_request(
        (ActionType.MOVEMENT, movement),
        (ActionType.FAST_TRAVEL, 0),
    )

    assert HeuristicAgent(0).choose_input(state, request) == expected


@pytest.mark.parametrize(
    ("movement", "positional_action"),
    [(6, ActionType.MOVEMENT), (1, ActionType.FAST_TRAVEL)],
)
def test_generic_skill_outranks_large_dynamic_positional_gain(
    state: GameState, movement: int, positional_action: ActionType
) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=-2, r=4, s=-2))
    wasp.current_turn_card = _movement_or_fast_travel_card(movement)
    request = _action_request(
        (positional_action, movement if positional_action == ActionType.MOVEMENT else 0),
        (ActionType.SKILL, 0),
    )
    agent = HeuristicAgent(0)
    positional_score, skill_score = (
        agent.score_option(state, request, option) for option in request.options
    )
    chosen = agent.choose_input(state, request)

    assert skill_score > positional_score
    assert chosen == "SKILL"


def test_secondary_attack_score_uses_attack_value(state: GameState) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    state.place_entity("hero_arien", Hex(q=1, r=-1, s=0))
    agent = HeuristicAgent(0)

    wasp.current_turn_card = _secondary_attack_card(2)
    low_request = _action_request((ActionType.ATTACK, 2))
    low_score = agent.score_option(state, low_request, low_request.options[0])
    wasp.current_turn_card = _secondary_attack_card(5)
    high_request = _action_request((ActionType.ATTACK, 5))
    high_score = agent.score_option(state, high_request, high_request.options[0])

    assert high_score > low_score


def test_secondary_attack_score_uses_best_legal_target_quality(state: GameState) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    state.place_entity("hero_arien", Hex(q=1, r=-1, s=0))
    wasp.current_turn_card = _secondary_attack_card(5)
    request = _action_request((ActionType.ATTACK, 5))
    agent = HeuristicAgent(0)
    hero_score = agent.score_option(state, request, request.options[0])

    state.remove_entity("hero_arien")
    minion = state.teams[TeamColor.BLUE].minions[0]
    state.place_entity(minion.id, Hex(q=1, r=-1, s=0))
    minion_score = agent.score_option(state, request, request.options[0])

    assert hero_score > minion_score


def test_secondary_attack_without_a_legal_target_is_not_attractive(state: GameState) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    wasp.current_turn_card = _secondary_attack_card(5)
    request = _action_request((ActionType.ATTACK, 5), (ActionType.HOLD, 0))
    agent = HeuristicAgent(0)

    assert agent.score_option(state, request, request.options[0]) <= agent.score_option(
        state, request, request.options[1]
    )


def test_basic_primary_attack_without_a_legal_target_is_not_attractive(
    state: GameState,
) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    wasp.current_turn_card = _attack()
    request = _action_request((ActionType.ATTACK, 2), (ActionType.HOLD, 0))
    agent = HeuristicAgent(0)

    assert agent.score_option(state, request, request.options[0]) <= agent.score_option(
        state, request, request.options[1]
    )


@pytest.mark.parametrize("action", [ActionType.ATTACK, ActionType.SKILL])
def test_scripted_primary_action_keeps_safe_fallback_score(
    state: GameState, action: ActionType
) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.current_actor_id = wasp.id
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    wasp.current_turn_card = _attack().model_copy(
        update={"primary_action": action, "effect_id": "scripted_primary"}
    )
    agent = HeuristicAgent(0)
    scores = []
    for value in (1, 9):
        request = _action_request((action, value))
        scores.append(agent.score_option(state, request, request.options[0]))
    state.place_entity("hero_arien", Hex(q=1, r=-1, s=0))
    request = _action_request((action, 9))
    scores.append(agent.score_option(state, request, request.options[0]))

    assert scores[0] == scores[1] == scores[2]


def test_enemy_approach_score_is_clamped_beyond_ten_hexes(state: GameState) -> None:
    _clear_board(state)
    farther = Hex(q=-5, r=3, s=2)
    nearer = Hex(q=-4, r=2, s=2)
    enemy = Hex(q=8, r=-3, s=-5)
    state.place_entity("hero_arien", enemy)
    topology = get_topology_service()

    assert any(farther in zone.hexes and nearer in zone.hexes for zone in state.board.zones.values())
    assert topology.distance(farther, enemy, state) > topology.distance(nearer, enemy, state) >= 10
    request = _movement_request([farther, nearer])
    agent = HeuristicAgent(0)

    assert agent.score_option(state, request, request.options[0]) == agent.score_option(
        state, request, request.options[1]
    )


def test_topology_aware_approach_cannot_override_a_full_zone_step(state: GameState) -> None:
    _clear_board(state)
    local_hex = Hex(q=0, r=0, s=0)
    enemy_hex = Hex(q=0, r=-2, s=2)
    state.place_entity("hero_wasp", Hex(q=-2, r=2, s=0))
    state.place_entity("hero_arien", enemy_hex)
    state.active_effects.append(
        ActiveEffect(
            id="split",
            source_id="hero_xargatha",
            effect_type=EffectType.TOPOLOGY_SPLIT,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_ROUND,
            split_axis="q",
            split_value=0,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    agent = HeuristicAgent(0)
    lane_id = next(iter(state.battle_zones))
    current_zone = next(z.id for z in state.board.zones.values() if local_hex in z.hexes)
    farther_zone = max(
        state.board.zones,
        key=lambda zone_id: zones_between(state, TeamColor.RED, lane_id, zone_id),
    )
    farther_hex = next(
        h for h in state.board.zones[farther_zone].hexes if not state.board.get_tile(h).occupant_id
    )
    assert zones_between(state, TeamColor.RED, lane_id, farther_zone) > zones_between(
        state, TeamColor.RED, lane_id, current_zone
    )
    assert agent.score_option(
        state, _movement_request([farther_hex]), _hex_option(farther_hex)
    ) > agent.score_option(state, _movement_request([local_hex]), _hex_option(local_hex))


def test_target_score_uses_public_timing_and_minion_support(state: GameState) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    arien = state.get_hero("hero_arien")
    brogan = state.get_hero("hero_brogan")
    assert wasp and arien and brogan
    blue_minions = state.teams[TeamColor.BLUE].minions
    target_minion, support = blue_minions[:2]
    state.place_entity(wasp.id, Hex(q=0, r=0, s=0))
    state.place_entity(arien.id, Hex(q=1, r=-1, s=0))
    state.place_entity(brogan.id, Hex(q=-1, r=1, s=0))
    state.place_entity(target_minion.id, Hex(q=0, r=1, s=-1))
    state.unresolved_hero_ids = [arien.id]
    arien.current_turn_card = _attack("hidden_fast")
    brogan.current_turn_card = _attack("hidden_slow")
    arien.current_turn_card.is_facedown = True
    brogan.current_turn_card.is_facedown = True

    agent = HeuristicAgent(0)
    request = _unit_request(arien.id, brogan.id, target_minion.id)
    timing_scores = [agent.score_option(state, request, option) for option in request.options]
    assert timing_scores[0] > timing_scores[1]

    state.place_entity(support.id, Hex(q=-2, r=1, s=1))
    state.unresolved_hero_ids = [arien.id, brogan.id]
    assert calculate_minion_defense_modifier(state, arien.id) < calculate_minion_defense_modifier(
        state, brogan.id
    )
    before = [agent.score_option(state, request, option) for option in request.options]
    assert before[0] > before[1] > before[2]

    arien.current_turn_card, brogan.current_turn_card = (
        brogan.current_turn_card,
        arien.current_turn_card,
    )
    arien.hand, brogan.hand = brogan.hand, arien.hand
    after = [agent.score_option(state, request, option) for option in request.options]
    assert after == before


def test_target_score_resolves_razzle_piece_to_enemy_hero_owner() -> None:
    register_all_effects()
    state = GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp", "Xargatha"],
        ["Razzle", "Brogan"],
        game_type="QUICK",
        seed=1,
    )
    _clear_board(state)
    target_id = piece_id("hero_razzle", 1)
    minion_id = state.teams[TeamColor.BLUE].minions[0].id
    state.place_entity(target_id, Hex(q=0, r=0, s=0))
    state.place_entity(minion_id, Hex(q=1, r=-1, s=0))
    state.unresolved_hero_ids = ["hero_razzle"]
    request = _unit_request(target_id, minion_id)
    piece_score, minion_score = (
        HeuristicAgent(0).score_option(state, request, option) for option in request.options
    )

    assert piece_score > minion_score


def test_score_card_uses_computed_stats_and_rejects_disconnected_target(state: GameState) -> None:
    _clear_board(state)
    wasp = state.get_hero("hero_wasp")
    assert wasp
    state.place_entity(wasp.id, Hex(q=-1, r=0, s=1))
    state.place_entity("hero_arien", Hex(q=1, r=0, s=-1))
    card = _attack()
    agent = HeuristicAgent(0)

    printed = agent.score_card(state, wasp, card)
    wasp.items[StatType.RANGE] = 1
    in_range = agent.score_card(state, wasp, card)
    wasp.items[StatType.ATTACK] = 2
    buffed = agent.score_card(state, wasp, card)
    assert in_range > printed
    assert buffed > in_range

    state.active_effects.append(
        ActiveEffect(
            id="split",
            source_id="hero_xargatha",
            effect_type=EffectType.TOPOLOGY_SPLIT,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_ROUND,
            split_axis="q",
            split_value=0,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    assert get_topology_service().distance(
        state.get_position(wasp.id), state.get_position("hero_arien"), state
    ) == float("inf")
    assert agent.score_card(state, wasp, card) < buffed


@pytest.mark.parametrize(
    ("reverse_created_turn", "hand_order", "expected_id"),
    [
        (None, ("low", "high"), "high"),
        (-1, ("high", "low"), "low"),
        (-2, ("low", "high"), "high"),
    ],
    ids=["normal", "live-reversal", "stale-reversal"],
)
def test_choose_card_ties_use_computed_initiative_and_live_reversal(
    state: GameState,
    monkeypatch: pytest.MonkeyPatch,
    reverse_created_turn: int | None,
    hand_order: tuple[str, str],
    expected_id: str,
) -> None:
    hero = state.get_hero("hero_wasp")
    enemy = state.get_hero("hero_arien")
    assert hero and enemy
    cards = {name: _attack(name).model_copy(update={"initiative": 5}) for name in hand_order}
    hero.hand = [cards[name] for name in hand_order]
    enemy.hand = [_attack("hidden_enemy").model_copy(update={"is_facedown": True})]
    if reverse_created_turn is not None:
        state.active_effects.append(
            ActiveEffect(
                id="reverse",
                source_id=enemy.id,
                effect_type=EffectType.REVERSED_INITIATIVE,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.NEXT_TURN,
                created_at_turn=state.turn + reverse_created_turn,
                created_at_round=state.round,
                is_active=True,
            )
        )

    calls: list[tuple[str, str]] = []

    def computed_initiative(
        _state: GameState,
        unit_id: str,
        stat_type: StatType,
        base_value: int = 0,
        performing_card: Card | None = None,
    ) -> int:
        assert unit_id == hero.id
        assert stat_type == StatType.INITIATIVE
        assert base_value == 5
        assert performing_card is not None
        calls.append((unit_id, performing_card.id))
        return {"low": 2, "high": 9}[performing_card.id]

    monkeypatch.setattr("goa2.engine.stats.get_computed_stat", computed_initiative)
    agent = HeuristicAgent(0)
    monkeypatch.setattr(agent, "score_card", lambda *_args: 0.0)

    assert agent.choose_card(state, hero).id == expected_id
    assert {card_id for _, card_id in calls} == {"low", "high"}
