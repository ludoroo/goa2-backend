from __future__ import annotations

import pytest

import automata.agents.heuristic_agent as heuristic_module
from automata.agents.heuristic_agent import HeuristicAgent
from automata.harness.game_runner import DEFAULT_MAP
from goa2.domain.models import (
    ActionType,
    CardColor,
    CardState,
    CardTier,
    StatType,
    TeamColor,
)
from goa2.domain.models.card import Card
from goa2.domain.models.effect import (
    ActiveEffect,
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID
from goa2.engine.phases import commit_card
from goa2.engine.setup import GameSetup


def _new_game() -> tuple[GameState, Hero]:
    state = GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=["Xargatha"],
        blue_heroes=["Arien"],
        game_type="QUICK",
        seed=1,
    )
    hero = state.get_hero(HeroID("hero_xargatha"))
    assert hero is not None
    return state, hero


def _card(hero: Hero, card_id: str) -> Card:
    return next(card for card in hero.hand if card.id == card_id)


def _place_enemy_adjacent(state: GameState, hero: Hero) -> None:
    enemy = state.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    hero_position = state.get_position(str(hero.id))
    assert hero_position is not None
    adjacent = next(
        candidate
        for candidate in hero_position.neighbors()
        if state.board.is_on_map(candidate)
        and not state.board.get_tile(candidate).is_obstacle
        and not state.board.get_tile(candidate).is_occupied
    )
    state.move_unit(UnitID(str(enemy.id)), adjacent)


def test_initial_xargatha_targetless_attacks_are_interpreted_as_secondary_movement() -> None:
    state, xargatha = _new_game()
    agent = HeuristicAgent(seed=3)

    charm_score = agent.score_card(state, xargatha, _card(xargatha, "charm"))
    attack_scores = [
        agent.score_card(state, xargatha, _card(xargatha, card_id))
        for card_id in ("threatening_slash", "cleave")
    ]

    assert attack_scores == pytest.approx([charm_score, charm_score])
    assert all(score < charm_score for score in attack_scores)
    assert agent.choose_planning(state, xargatha).card.id == "charm"


@pytest.mark.parametrize("card_id", ["threatening_slash", "cleave"])
def test_targetless_attack_later_scores_as_movement_not_the_old_attack_estimate(
    card_id: str,
) -> None:
    state, xargatha = _new_game()
    state.round = 2
    agent = HeuristicAgent()

    movement_score = agent.score_card(state, xargatha, _card(xargatha, "charm"))
    score = agent.score_card(state, xargatha, _card(xargatha, card_id))

    assert score == pytest.approx(movement_score)
    assert score < movement_score  # Prefer the primary action when action values tie.
    assert score not in (11, 12)  # Previous targetless ATTACK estimates.


@pytest.mark.parametrize(
    ("card_id", "expected_score"),
    [("threatening_slash", 16), ("cleave", 19)],
)
def test_reachable_attacks_retain_their_attack_score(card_id: str, expected_score: float) -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, card_id))

    assert score == expected_score


def test_card_scoring_uses_best_of_multiple_executable_secondary_actions() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    card = Card(
        id="defensive_options",
        name="Defensive Options",
        tier=CardTier.UNTIERED,
        color=CardColor.GOLD,
        initiative=1,
        primary_action=ActionType.DEFENSE,
        secondary_actions={
            ActionType.MOVEMENT: 5,
            ActionType.SKILL: 0,
            ActionType.ATTACK: 2,
        },
        effect_id="defensive_options",
        effect_text="",
    )

    score = HeuristicAgent().score_card(state, xargatha, card)

    # The proactive primary DEFENSE is unavailable. Of the executable
    # secondaries, ATTACK (15 + 2) outranks movement and skill.
    assert score == pytest.approx(13)
    assert score < 13  # Secondary loses only the primary-action tie-break.


def test_attack_score_has_small_bounded_risk_only_when_an_earlier_card_can_escape() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    agent = HeuristicAgent()

    exposed_score = agent.score_card(state, xargatha, _card(xargatha, "threatening_slash"))
    too_fast_to_dodge_score = agent.score_card(state, xargatha, _card(xargatha, "cleave"))

    assert 2 <= 20 - exposed_score <= 4
    assert too_fast_to_dodge_score == pytest.approx(19)


def test_reversed_initiative_reverses_which_attack_is_exposed_to_dodge_risk() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    state.active_effects.append(
        ActiveEffect(
            id="reverse_time",
            source_id="hero_arien",
            effect_type=EffectType.REVERSED_INITIATIVE,
            scope=EffectScope(shape=Shape.GLOBAL, affects=AffectsFilter.ALL_HEROES),
            duration=DurationType.NEXT_TURN,
            created_at_turn=state.turn - 1,
            created_at_round=state.round,
            is_active=True,
        )
    )

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "cleave"))

    assert 2 <= 19 - score <= 4


def test_publicly_played_earlier_movement_cards_are_not_counted_as_remaining() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    spent_ids = {"noble_blade", "aspiring_duelist", "dangerous_current"}
    spent = [card for card in arien.hand if str(card.id) in spent_ids]
    for card in spent:
        arien.hand.remove(card)
        card.state = CardState.RESOLVED
        card.is_facedown = False
    arien.played_cards = spent

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert score == pytest.approx(20)


def test_movement_restriction_removes_dodge_risk() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    state.active_effects.append(
        ActiveEffect(
            id="movement_lock",
            source_id=str(xargatha.id),
            effect_type=EffectType.PETRIFY,
            scope=EffectScope(shape=Shape.GLOBAL, affects=AffectsFilter.ENEMY_HEROES),
            restrictions=[ActionType.MOVEMENT],
            duration=DurationType.THIS_ROUND,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert score == pytest.approx(20)


def test_ranged_attack_uses_its_actual_range_for_escape_risk() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    ranged_attack = Card(
        id="ranged_test_attack",
        name="Ranged Test Attack",
        tier=CardTier.UNTIERED,
        color=CardColor.GOLD,
        initiative=7,
        primary_action=ActionType.ATTACK,
        primary_action_value=5,
        secondary_actions={},
        effect_id="ranged_test_attack",
        effect_text="",
        is_ranged=True,
        range_value=3,
        state=CardState.HAND,
        is_facedown=False,
    )

    agent = HeuristicAgent()
    score = agent.score_card(state, xargatha, ranged_attack)
    no_escape_score = agent.score_card(
        state,
        xargatha,
        ranged_attack.model_copy(update={"range_value": 20}),
    )

    assert 2 <= 20 - score <= 4
    assert no_escape_score == pytest.approx(20)


def test_legal_minion_target_makes_attack_dodge_risk_zero() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    hero_position = state.get_position(str(xargatha.id))
    assert hero_position is not None
    minion = state.teams[TeamColor.BLUE].minions[1]
    minion_destination = next(
        candidate
        for candidate in hero_position.neighbors()
        if state.board.is_on_map(candidate)
        and not state.board.get_tile(candidate).is_obstacle
        and not state.board.get_tile(candidate).is_occupied
    )
    state.move_unit(UnitID(str(minion.id)), minion_destination)

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert score == pytest.approx(20)


@pytest.mark.parametrize(
    ("risky_card_ids", "expected_penalty"),
    [
        ((), 0),
        (("noble_blade",), 2),
        (("noble_blade", "aspiring_duelist"), 3),
        (("noble_blade", "aspiring_duelist", "dangerous_current"), 4),
    ],
)
def test_dodge_risk_count_maps_to_bounded_penalty(
    monkeypatch: pytest.MonkeyPatch,
    risky_card_ids: tuple[str, ...],
    expected_penalty: int,
) -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    candidates = [card for card in arien.hand if str(card.id) in risky_card_ids]
    agent = HeuristicAgent()
    monkeypatch.setattr(agent, "_public_target_loadouts", lambda *args, **kwargs: [candidates])

    score = agent.score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == expected_penalty


@pytest.mark.parametrize("tie_breaker", [TeamColor.RED, TeamColor.BLUE])
def test_equal_initiative_uses_both_tie_break_teams(tie_breaker: TeamColor) -> None:
    state, xargatha = _new_game()
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    state.tie_breaker_team = tie_breaker

    assert HeuristicAgent._initiative_resolves_before(state, arien, 8, xargatha, 8) is (
        tie_breaker == TeamColor.BLUE
    )
    assert HeuristicAgent._initiative_resolves_before(state, xargatha, 8, arien, 8) is (
        tie_breaker == TeamColor.RED
    )


@pytest.mark.parametrize(
    ("reversed_initiative", "first", "second", "expected"),
    [
        (False, 9, 8, True),
        (False, 8, 9, False),
        (True, 9, 8, False),
        (True, 8, 9, True),
    ],
)
def test_initiative_direction_handles_both_orders_when_reversed(
    reversed_initiative: bool,
    first: int,
    second: int,
    expected: bool,
) -> None:
    state, xargatha = _new_game()
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    if reversed_initiative:
        state.active_effects.append(
            ActiveEffect(
                id="reverse_time",
                source_id="hero_arien",
                effect_type=EffectType.REVERSED_INITIATIVE,
                scope=EffectScope(shape=Shape.GLOBAL, affects=AffectsFilter.ALL_HEROES),
                duration=DurationType.NEXT_TURN,
                created_at_turn=state.turn - 1,
                created_at_round=state.round,
                is_active=True,
            )
        )

    assert (
        HeuristicAgent._initiative_resolves_before(state, arien, first, xargatha, second)
        is expected
    )


def test_nonstandard_public_inference_uses_only_minimum_uncertainty_penalty() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.name = "Dodger"  # Explicitly excluded from ordinary loadout inference.

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == 2


def test_inconsistent_public_inference_uses_only_minimum_uncertainty_penalty() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 2}  # Impossible after one ordinary upgrade.

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == 2


def test_score_card_computes_targets_and_public_knowledge_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=["Xargatha"],
        blue_heroes=["Arien", "Wasp"],
        game_type="QUICK",
        seed=1,
    )
    xargatha = state.get_hero(HeroID("hero_xargatha"))
    assert xargatha is not None
    origin = state.get_position(str(xargatha.id))
    assert origin is not None
    open_neighbors = [
        candidate
        for candidate in origin.neighbors()
        if state.board.is_on_map(candidate)
        and not state.board.get_tile(candidate).is_obstacle
        and not state.board.get_tile(candidate).is_occupied
    ]
    for target_id, destination in zip(("hero_arien", "hero_wasp"), open_neighbors[:2], strict=True):
        state.move_unit(UnitID(target_id), destination)

    agent = HeuristicAgent()
    target_calls = 0
    knowledge_calls = 0
    original_targets = agent._legal_attack_targets
    original_knowledge = heuristic_module.build_public_card_knowledge

    def counted_targets(*args, **kwargs):
        nonlocal target_calls
        target_calls += 1
        return original_targets(*args, **kwargs)

    def counted_knowledge(*args, **kwargs):
        nonlocal knowledge_calls
        knowledge_calls += 1
        return original_knowledge(*args, **kwargs)

    monkeypatch.setattr(agent, "_legal_attack_targets", counted_targets)
    monkeypatch.setattr(heuristic_module, "build_public_card_knowledge", counted_knowledge)

    agent.score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert target_calls == 1
    assert knowledge_calls == 1


def test_safest_of_multiple_hero_targets_sets_attack_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=["Xargatha"],
        blue_heroes=["Arien", "Wasp"],
        game_type="QUICK",
        seed=1,
    )
    xargatha = state.get_hero(HeroID("hero_xargatha"))
    assert xargatha is not None
    origin = state.get_position(str(xargatha.id))
    assert origin is not None
    open_neighbors = [
        candidate
        for candidate in origin.neighbors()
        if state.board.is_on_map(candidate)
        and not state.board.get_tile(candidate).is_obstacle
        and not state.board.get_tile(candidate).is_occupied
    ]
    for target_id, destination in zip(("hero_arien", "hero_wasp"), open_neighbors[:2], strict=True):
        state.move_unit(UnitID(target_id), destination)

    agent = HeuristicAgent()

    def target_penalty(_state, _attacker, _card, _source, target_id, _range, **_kwargs):
        return 4.0 if target_id == "hero_arien" else 2.0

    monkeypatch.setattr(agent, "_target_dodge_risk_penalty", target_penalty)

    score = agent.score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == 2


def test_public_reveal_narrows_upgraded_hypothesis_and_replaces_starting_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 1}
    state.record_public_revealed_card(arien.id, "magical_current")
    agent = HeuristicAgent()
    seen_ids: set[str] = set()

    def only_upgrades_can_dodge(_state, _attacker, _initiative, candidate, *args):
        seen_ids.add(str(candidate.id))
        return str(candidate.id) in {"magical_current", "raging_stream"}

    monkeypatch.setattr(agent, "_card_can_dodge_before_attack", only_upgrades_can_dodge)

    score = agent.score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == 2
    assert "magical_current" in seen_ids
    assert "raging_stream" not in seen_ids
    assert "liquid_leap" not in seen_ids


def test_faceup_current_and_extra_turn_cards_are_publicly_unavailable() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    risky = {
        str(card.id): card
        for card in arien.hand
        if str(card.id) in {"noble_blade", "aspiring_duelist", "dangerous_current"}
    }
    for card in risky.values():
        arien.hand.remove(card)
        card.state = CardState.UNRESOLVED
        card.is_facedown = False
    arien.current_turn_card = risky["noble_blade"]
    arien.extra_turn_card = risky["aspiring_duelist"]
    arien.played_cards = [risky["dangerous_current"]]

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert score == 20


def test_multi_piece_hero_target_uses_its_owner_public_loadout() -> None:
    state = GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=["Xargatha"],
        blue_heroes=["Razzle"],
        game_type="QUICK",
        seed=1,
    )
    xargatha = state.get_hero(HeroID("hero_xargatha"))
    razzle = state.get_hero(HeroID("hero_razzle"))
    assert xargatha is not None and razzle is not None
    piece_id = state.get_piece_ids(str(razzle.id))[0]
    origin = state.get_position(str(xargatha.id))
    assert origin is not None
    destination = next(
        candidate
        for candidate in origin.neighbors()
        if state.board.is_on_map(candidate)
        and not state.board.get_tile(candidate).is_obstacle
        and not state.board.get_tile(candidate).is_occupied
    )
    state.move_unit(UnitID(piece_id), destination)

    score = HeuristicAgent().score_card(state, xargatha, _card(xargatha, "threatening_slash"))

    assert 20 - score == 4


def test_attack_dodge_risk_is_invariant_to_hidden_commitment_from_level_two_baseline() -> None:
    state, xargatha = _new_game()
    _place_enemy_adjacent(state, xargatha)
    arien = state.get_hero(HeroID("hero_arien"))
    assert arien is not None
    arien.level = 2
    arien.items = {StatType.DEFENSE: 1}
    first, second = arien.hand[:2]
    agent = HeuristicAgent()
    attack = _card(xargatha, "threatening_slash")

    baseline_score = agent.score_card(state, xargatha, attack)
    commit_card(state, arien.id, first)
    first_score = agent.score_card(state, xargatha, attack)
    arien.unplay_card(first)
    arien.current_turn_card = None
    del state.pending_inputs[arien.id]
    commit_card(state, arien.id, second)
    second_score = agent.score_card(state, xargatha, attack)

    assert baseline_score == first_score == second_score
