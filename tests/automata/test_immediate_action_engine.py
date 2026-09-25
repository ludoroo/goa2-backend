from __future__ import annotations

from collections.abc import Callable

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import ActionBoundaryKind, DecisionDescriptor
from automata.evaluation.learned_matrix import LearnedMatrixCell, build_learned_ismcts
from automata.models.contracts.inference import LearnedModelOutput
from automata.runtime.clone import clone_state
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafEvaluation, LeafMode, SearchContext
from automata.search.ismcts.engine import (
    _ActionBoundary,
    _rollout,
    _Simulator,
    legal_keys,
    search,
)
from automata.search.node import action_key
from automata.search.root import RootTarget
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequest, InputRequestType, selection_value
from goa2.domain.models import CardState, GamePhase, StepType, TargetType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.effects import CardEffectRegistry
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    AttackSequenceStep,
    ConfirmResolutionStep,
    ResolveCardStep,
    ResolveTieBreakerStep,
    RespawnHeroStep,
    SelectStep,
)


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _game(red: str, blue: str):
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        [red],
        [blue],
        game_type="QUICK",
        seed=41,
    )
    for entity_id in list(state.entity_locations):
        state.remove_entity(entity_id)
    state.phase = GamePhase.RESOLUTION
    return state


def _open_line(state, length: int = 5) -> list[Hex]:
    for start, tile in state.board.tiles.items():
        if tile.is_terrain:
            continue
        for direction in range(6):
            line = [start]
            for _ in range(length - 1):
                line.append(line[-1].neighbor(direction))
            if all(
                point in state.board.tiles and not state.board.tiles[point].is_terrain
                for point in line
            ):
                return line
    raise AssertionError("map has no open straight test line")


def _card(hero, card_id: str):
    cards = [
        hero.current_turn_card,
        hero.ultimate_card,
        *hero.hand,
        *hero.deck,
        *hero.played_cards,
        *hero.discard_pile,
        *hero.spells,
    ]
    card = next((candidate for candidate in cards if candidate and candidate.id == card_id), None)
    assert card is not None
    return card


def _prepare_actor(state, hero_id: str, card_id: str):
    hero = state.get_hero(HeroID(hero_id))
    assert hero is not None
    card = _card(hero, card_id)
    card.state = CardState.UNRESOLVED
    hero.current_turn_card = card
    state.current_actor_id = HeroID(hero_id)
    state.resolution_owner_id = HeroID(hero_id)
    return hero, card


def _first_request(state) -> InputRequest:
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    return result.input_request


class _Continuation:
    def __init__(self, choose: Callable[[object, InputRequest], object]) -> None:
        self._choose = choose

    def choose_input(self, state, request: InputRequest):
        return self._choose(state, request)


class _BoundaryLeaf:
    def evaluate(self, context: SearchContext, state) -> LeafEvaluation:
        assert context.is_action_boundary
        return LeafEvaluation(value=0.0)


def _run_to_boundary(
    state,
    request: InputRequest,
    root_selection: object,
    continuation: _Continuation,
    *,
    environment_policy: object | None = None,
):
    world = clone_state(state)
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({request.player_id}),
        decision_owner_hero_id=request.player_id,
        request=request,
    )
    sim = _Simulator(
        world,
        TeamColor.RED,
        environment_policy or HeuristicAgent(7),  # type: ignore[arg-type]
        owned_hero_ids=target.owned_hero_ids,
        cfg=SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
    )
    decision = sim.advance_to_root(target)
    actor_id = str(world.resolution_owner_id or world.current_actor_id)
    boundary = _ActionBoundary(request.player_id, actor_id, world.round)
    next_decision = sim.apply_ours(
        decision,
        action_key(root_selection),
        action_boundary=boundary,
    )
    reward = _rollout(
        sim,
        next_decision,
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
        continuation,  # type: ignore[arg-type]
        _BoundaryLeaf(),
        SearchContext(request.player_id, TeamColor.RED, request.player_id, decision),
        action_boundary=boundary,
    )
    assert reward == 0.5
    return sim.state


def _assert_confirm_boundary(state, actor_id: str, round_number: int) -> None:
    assert state.round == round_number
    assert str(state.current_actor_id) == actor_id
    assert state.execution_stack[-1].type is StepType.CONFIRM_RESOLUTION


@pytest.mark.parametrize("cell", list(LearnedMatrixCell))
def test_immediate_action_supports_every_learned_matrix_cell(cell: LearnedMatrixCell) -> None:
    state = _game("Brogan", "Arien")
    line = _open_line(state)
    hero, card = _prepare_actor(state, "hero_brogan", "brutal_jab")
    enemy = state.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    state.place_entity(hero.id, line[0])
    state.place_entity(enemy.id, line[2])
    effect = CardEffectRegistry.get("brutal_jab")
    assert effect is not None
    push_steps(
        state,
        [*effect.get_steps(state, hero, card), ConfirmResolutionStep(hero_id=str(hero.id))],
    )
    request = _first_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    assert len(legal) > 1
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({str(hero.id)}),
        decision_owner_hero_id=str(hero.id),
        request=request,
    )

    class Runtime:
        def evaluate(self, observation):
            candidate_ids = tuple(candidate.candidate_id for candidate in observation.candidates)
            size = len(candidate_ids)
            return LearnedModelOutput(
                candidate_ids=candidate_ids,
                policy_logits=(0.0,) * size,
                probabilities=(1.0 / size,) * size,
                value=0.2,
            )

    strategy = build_learned_ismcts(
        cell,
        runtime=None if cell is LearnedMatrixCell.HH else Runtime(),  # type: ignore[arg-type]
        default_policy=HeuristicAgent(9),
        config=SearchConfig(iterations=2, leaf_mode=LeafMode.IMMEDIATE_ACTION, seed=5),
    )
    result = strategy.select(state, TeamColor.RED, target, legal)

    assert result.selected_candidate in legal
    assert result.search_result is not None
    assert result.search_result.root.visits >= 2


def test_immediate_action_respawn_places_hero_and_beats_pass() -> None:
    state = _game("Wasp", "Arien")
    hero, _card_value = _prepare_actor(state, "hero_wasp", "shock")
    state.remove_entity(hero.id)
    push_steps(
        state,
        [RespawnHeroStep(hero_id=str(hero.id)), ResolveCardStep(hero_id=str(hero.id))],
    )
    request = _first_request(state)
    assert request.request_type is InputRequestType.CHOOSE_RESPAWN
    decision = DecisionDescriptor("INPUT", request=request)
    legal = tuple(legal_keys(decision))
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({str(hero.id)}),
        decision_owner_hero_id=str(hero.id),
        request=request,
    )
    observed: list[tuple[bool, StepType | None]] = []

    class FirstOptionContinuation:
        def choose_input(self, _state, pending: InputRequest):
            return selection_value(pending.options[0])

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(12),
        SearchConfig(iterations=8, leaf_mode=LeafMode.IMMEDIATE_ACTION, seed=17),
        root_target=target,
        continuation_policy=FirstOptionContinuation(),  # type: ignore[arg-type]
        cutoff_observer=lambda leaf, _team, _value: observed.append(
            (
                leaf.has_board_presence(str(hero.id)),
                leaf.execution_stack[-1].type if leaf.execution_stack else None,
            )
        ),
    )

    diagnostics = {item.action: item for item in result.root_action_diagnostics}
    assert result.best_key == "RESPAWN"
    assert diagnostics["RESPAWN"].mean_value > diagnostics["PASS"].mean_value
    assert any(placed and top is StepType.RESOLVE_CARD for placed, top in observed)


def test_brutal_jab_move_is_resolved_before_push_and_confirm_boundary() -> None:
    state = _game("Brogan", "Arien")
    line = _open_line(state)
    hero, card = _prepare_actor(state, "hero_brogan", "brutal_jab")
    enemy = state.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    state.place_entity(hero.id, line[0])
    state.place_entity(enemy.id, line[2])
    effect = CardEffectRegistry.get("brutal_jab")
    assert effect is not None
    push_steps(
        state,
        [*effect.get_steps(state, hero, card), ConfirmResolutionStep(hero_id=str(hero.id))],
    )
    request = _first_request(state)
    assert request.request_type is InputRequestType.SELECT_HEX
    start_round = state.round

    resolved = _run_to_boundary(
        state,
        request,
        line[1].model_dump(),
        _Continuation(
            lambda _state, pending: (
                str(enemy.id)
                if pending.request_type is InputRequestType.SELECT_UNIT_OR_TOKEN
                else 1
            )
        ),
    )

    assert resolved.get_position(hero.id) == line[1]
    assert resolved.get_position(enemy.id) == line[3]
    _assert_confirm_boundary(resolved, str(hero.id), start_round)


def test_sirens_call_pull_resolves_before_confirm_boundary() -> None:
    state = _game("Xargatha", "Arien")
    line = _open_line(state)
    hero, card = _prepare_actor(state, "hero_xargatha", "sirens_call")
    enemy = state.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    state.place_entity(hero.id, line[0])
    state.place_entity(enemy.id, line[3])
    effect = CardEffectRegistry.get("sirens_call")
    assert effect is not None
    push_steps(
        state,
        [*effect.get_steps(state, hero, card), ConfirmResolutionStep(hero_id=str(hero.id))],
    )
    request = _first_request(state)
    assert request.request_type is InputRequestType.SELECT_UNIT
    start_round = state.round

    resolved = _run_to_boundary(
        state,
        request,
        str(enemy.id),
        _Continuation(lambda _state, _request: line[1].model_dump()),
    )

    assert resolved.get_position(enemy.id) == line[1]
    assert resolved.get_position(hero.id).distance(resolved.get_position(enemy.id)) == 1
    _assert_confirm_boundary(resolved, str(hero.id), start_round)


def test_living_tsunami_move_enables_attack_before_confirm_boundary() -> None:
    state = _game("Arien", "Brogan")
    line = _open_line(state)
    hero, _card_value = _prepare_actor(state, "hero_arien", "noble_blade")
    hero.level = 8
    minion = state.teams[TeamColor.BLUE].minions[0]
    state.place_entity(hero.id, line[0])
    state.place_entity(minion.id, line[2])
    push_steps(
        state,
        [ResolveCardStep(hero_id=str(hero.id)), ConfirmResolutionStep(hero_id=str(hero.id))],
    )
    action = _first_request(state)
    assert action.request_type is InputRequestType.CHOOSE_ACTION
    start_round = state.round

    def continue_attack(_state, request: InputRequest):
        if request.request_type is InputRequestType.CONFIRM_PASSIVE:
            return "YES"
        if request.request_type is InputRequestType.SELECT_HEX:
            return line[1].model_dump()
        return str(minion.id)

    resolved = _run_to_boundary(
        state,
        action,
        "ATTACK",
        _Continuation(continue_attack),
    )

    assert resolved.get_position(hero.id) == line[1]
    assert resolved.get_position(minion.id) is None
    _assert_confirm_boundary(resolved, str(hero.id), start_round)


def test_skipped_perform_action_boundary_does_not_hide_mind_grip_minion_defeat() -> None:
    state = _game("NebKher", "Arien")
    line = _open_line(state)
    hero, card = _prepare_actor(state, "hero_nebkher", "mind_grip")
    minion = state.teams[TeamColor.BLUE].minions[0]
    state.place_entity(hero.id, line[0])
    state.place_entity(minion.id, line[1])
    effect = CardEffectRegistry.get("mind_grip")
    assert effect is not None
    push_steps(
        state,
        [*effect.get_steps(state, hero, card), ConfirmResolutionStep(hero_id=str(hero.id))],
    )
    request = _first_request(state)
    assert request.request_type is InputRequestType.SELECT_NUMBER
    start_round = state.round

    resolved = _run_to_boundary(
        state,
        request,
        2,
        _Continuation(lambda _state, _request: str(minion.id)),
    )

    assert resolved.get_position(minion.id) is None
    _assert_confirm_boundary(resolved, str(hero.id), start_round)


def test_tie_breaker_step_interrupts_immediate_action() -> None:
    state = _game("Wasp", "Arien")
    hero, _card_value = _prepare_actor(state, "hero_wasp", "shock")
    state.execution_stack = [ResolveTieBreakerStep(tied_hero_ids=[hero.id, HeroID("hero_arien")])]
    boundary = _ActionBoundary(str(hero.id), str(hero.id), state.round)
    sim = _Simulator(
        state,
        TeamColor.RED,
        HeuristicAgent(7),
        owned_hero_ids=frozenset({str(hero.id)}),
        cfg=SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
    )

    decision = sim.advance(action_boundary=boundary)

    assert decision.kind == "BOUNDARY"
    assert decision.action_boundary_kind is ActionBoundaryKind.INTERRUPTED
    assert state.execution_stack[-1].type is StepType.RESOLVE_TIE_BREAKER


def test_team_owned_follow_up_interrupts_immediate_action() -> None:
    state = _game("Wasp", "Arien")
    hero, _card_value = _prepare_actor(state, "hero_wasp", "shock")
    state.execution_context["team_owner"] = "team:RED"
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Team choice",
                number_options=[1, 2],
                override_player_id_key="team_owner",
            ),
            ConfirmResolutionStep(hero_id=str(hero.id)),
        ],
    )
    boundary = _ActionBoundary(str(hero.id), str(hero.id), state.round)
    sim = _Simulator(
        state,
        TeamColor.RED,
        HeuristicAgent(7),
        owned_hero_ids=frozenset({str(hero.id)}),
        cfg=SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
    )

    decision = sim.advance(action_boundary=boundary)

    assert decision.kind == "BOUNDARY"
    assert decision.action_boundary_kind is ActionBoundaryKind.INTERRUPTED
    assert state.execution_stack[-1].type is StepType.SELECT


def test_foreign_defense_reaction_resolves_before_confirm_boundary() -> None:
    state = _game("Arien", "Brogan")
    line = _open_line(state)
    attacker, _card_value = _prepare_actor(state, "hero_arien", "noble_blade")
    defender = state.get_hero(HeroID("hero_brogan"))
    assert defender is not None
    state.place_entity(attacker.id, line[0])
    state.place_entity(defender.id, line[1])
    push_steps(
        state,
        [
            AttackSequenceStep(damage=4, range_val=1),
            ConfirmResolutionStep(hero_id=str(attacker.id)),
        ],
    )
    request = _first_request(state)
    assert request.request_type is InputRequestType.SELECT_UNIT
    start_round = state.round
    defender_hand = {card.id for card in defender.hand}

    def choose_foreign(_state, pending: InputRequest):
        return next(
            selection_value(option)
            for option in pending.options
            if selection_value(option) != "PASS"
        )

    resolved = _run_to_boundary(
        state,
        request,
        str(defender.id),
        _Continuation(lambda _state, _request: pytest.fail("foreign input was crossed")),
        environment_policy=_Continuation(choose_foreign),
    )

    resolved_defender = resolved.get_hero(HeroID(str(defender.id)))
    assert resolved_defender is not None
    assert {card.id for card in resolved_defender.hand} < defender_hand
    _assert_confirm_boundary(resolved, str(attacker.id), start_round)
