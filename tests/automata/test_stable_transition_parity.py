"""Search and actual play encode the same completed transition.

Raw stacks isolate progression, not a particular card effect. Determinization is
replaced with a faithful clone so both paths play the same sampled world. Actual
play still uses the normal driver and the behavior-neutral boundary observer.
The observations below are compared, never labeled with a censored game's result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from automata.agents.contracts import PlanningDecision
from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP, continue_game
from automata.models.contracts import canonical_json_bytes
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.clone import clone_state
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableValueBoundary,
    StableValueBoundaryKind,
    detect_stable_value_boundary,
)
from automata.search.config import SearchConfig
from automata.search.contracts import LeafEvaluation, LeafMode
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts import legal_keys, search
from automata.search.root import RootTarget
from goa2.domain.input import InputRequest, InputRequestType, selection_value
from goa2.domain.models import CardState, GamePhase, TargetType, TeamColor
from goa2.domain.models.effect import ActiveEffect, DurationType, EffectScope, EffectType, Shape
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    FinalizeHeroTurnStep,
    LogMessageStep,
    ResolveTieBreakerStep,
    SelectStep,
)


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _game(red: list[str] | None = None) -> GameState:
    return GameSetup.create_game(
        DEFAULT_MAP, red or ["Wasp"], ["Arien"], game_type="QUICK", seed=83
    )


class _FirstAgent(HeuristicAgent):
    def __init__(self, *, planning_card: str | None = None, actor_choice: str | None = None):
        super().__init__(0)
        self.planning_card = planning_card
        self.actor_choice = actor_choice

    def choose_planning(self, state, hero):
        if planning_open_for_second_card(state, HeroID(hero.id)):
            return PlanningDecision.finish()
        if not hero.hand:
            return PlanningDecision.pass_()
        chosen = next((card for card in hero.hand if card.id == self.planning_card), hero.hand[0])
        return PlanningDecision.commit(chosen)

    def choose_input(self, state, request, **kwargs):
        if request.request_type is InputRequestType.CHOOSE_ACTOR and any(
            selection_value(option) == self.actor_choice for option in request.options
        ):
            return self.actor_choice
        if request.options:
            return selection_value(request.options[0])
        # Real simultaneous upgrades use the same explicit heuristic choice in
        # actual play and search. They are not invented scalar candidates.
        return super().choose_input(state, request, **kwargs)


@dataclass(frozen=True)
class _Capture:
    boundary: StableValueBoundary
    encoded: bytes


def _capture(state, boundary, viewer_id: str, team: TeamColor) -> _Capture:
    observation = encode_stable_value(
        state, boundary, viewer_hero_id=viewer_id, perspective_team=team
    )
    assert observation.state.candidate_ids == ()
    encoded = canonical_json_bytes(observation)
    assert b'"candidates"' not in encoded
    assert b'"input_request_type"' not in encoded
    assert b'"semantic_role"' not in encoded
    assert b"CONFIRM" not in encoded
    return _Capture(boundary, encoded)


class _ActualValues:
    def __init__(self, viewer_id: str, team: TeamColor):
        self.viewer_id = viewer_id
        self.team = team
        self.records: list[_Capture] = []

    def record_boundary(self, state, boundary, *, viewer_hero_ids):
        if self.viewer_id in viewer_hero_ids:
            self.records.append(_capture(state, boundary, self.viewer_id, self.team))

    def record_outcome(self, **_kwargs):
        pass


class _SearchValues(HeuristicLeafEvaluator):
    def __init__(self, viewer_id: str, team: TeamColor):
        self.viewer_id = viewer_id
        self.team = team
        self.records: list[_Capture] = []

    def evaluate(self, _context, _state):
        pytest.fail("candidate-bearing leaf evaluation used at a stable transition")

    def evaluate_stable_value(self, context, state) -> LeafEvaluation:
        assert context.root_viewer_id == self.viewer_id
        assert context.perspective_team is self.team
        assert not hasattr(context, "decision")
        assert context.boundary == detect_stable_value_boundary(state)
        captured = _capture(state, context.boundary, self.viewer_id, self.team)

        # Neither reaching a foreign actor nor resolving their turn grants the
        # fixed root viewer access to that hero's remaining private cards.
        changed = clone_state(state)
        hidden = next(
            hero
            for roster in changed.teams.values()
            for hero in roster.heroes
            if hero.id != self.viewer_id and hero.hand
        )
        hidden.hand[0].id = "unseen_private_card_replacement"
        assert _capture(changed, context.boundary, self.viewer_id, self.team) == captured
        self.records.append(captured)
        return LeafEvaluation(value=0.0)


def _actual_first(
    state: GameState,
    viewer_id: str,
    team: TeamColor,
    *,
    planning_card: str | None = None,
    actor_choice: str | None = None,
) -> _Capture:
    observer = _ActualValues(viewer_id, team)
    agents = {
        str(hero.id): _FirstAgent(planning_card=planning_card, actor_choice=actor_choice)
        for roster in state.teams.values()
        for hero in roster.heroes
    }
    continue_game(clone_state(state), agents, max_steps=20, boundary_observer=observer)
    assert observer.records
    return observer.records[0]


def _search_values(monkeypatch, state, target, legal, viewer_id, team) -> list[_Capture]:
    import automata.search.ismcts.engine as engine

    monkeypatch.setattr(
        engine, "determinize", lambda original, _viewer, _rng: clone_state(original)
    )
    before = state.model_dump_json()
    evaluator = _SearchValues(viewer_id, team)
    # Visit every root candidate, then revisit an existing child. In particular,
    # planning roots must not fall back to a shorter/deeper policy-leaf path.
    iterations = len(legal) + 1
    result = search(
        state,
        team,
        legal,
        _FirstAgent(),
        SearchConfig(
            iterations=iterations,
            leaf_mode=LeafMode.STABLE_TRANSITION,
            widening_c=float(len(legal) + 1),
            widening_alpha=0.0,
            seed=11,
        ),
        root_target=target,
        leaf_evaluator=evaluator,
    )
    assert state.model_dump_json() == before
    assert result.effective_leaf_mode is LeafMode.STABLE_TRANSITION
    assert result.root.visits == iterations
    assert len(evaluator.records) == iterations
    assert all(item.visits > 0 for item in result.root_action_diagnostics)
    return evaluator.records


def _give_turn_card(state: GameState, hero_id: str) -> None:
    hero = state.get_hero(HeroID(hero_id))
    assert hero is not None
    card = hero.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    hero.current_turn_card = card


def _pending_request(state: GameState) -> InputRequest:
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    return result.input_request


def _input_target(request: InputRequest, viewer_id: str) -> RootTarget:
    return RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({viewer_id}),
        decision_owner_hero_id=viewer_id,
        request=request,
    )


@pytest.mark.parametrize("first_commit", [True, False])
def test_planning_search_matches_live_actor_ready_encoding(monkeypatch, first_commit: bool) -> None:
    state = _game()
    if first_commit:
        viewer_id, team = "hero_wasp", TeamColor.RED
    else:
        wasp = state.get_hero(HeroID("hero_wasp"))
        assert wasp is not None
        GameSession(state).commit_card(wasp.id, wasp.hand[0])
        viewer_id, team = "hero_arien", TeamColor.BLUE
    hero = state.get_hero(HeroID(viewer_id))
    assert hero is not None
    legal = tuple(card.id for card in hero.hand)
    assert len(legal) > 1
    target = RootTarget.card(hero_id=viewer_id, owned_hero_ids=frozenset({viewer_id}))
    expected = {_actual_first(state, viewer_id, team, planning_card=key) for key in legal}

    actual = _search_values(monkeypatch, state, target, legal, viewer_id, team)

    assert all(record.boundary.kind is StableValueBoundaryKind.ACTOR_READY for record in actual)
    assert set(actual) == expected


@pytest.mark.parametrize(
    ("next_actor", "turn", "reaction", "respawn", "delayed_context"),
    [
        (True, 1, False, False, False),
        (True, 1, True, True, False),
        (False, 1, False, False, True),
        (False, 4, True, False, False),
    ],
)
def test_resolution_search_matches_live_boundary_encoding(
    monkeypatch, next_actor: bool, turn: int, reaction: bool, respawn: bool, delayed_context: bool
) -> None:
    state = _game()
    state.phase = GamePhase.RESOLUTION
    state.execution_stack.clear()
    state.pending_inputs.clear()
    state.turn = turn
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_arien")] if next_actor else []
    _give_turn_card(state, "hero_wasp")
    if next_actor:
        _give_turn_card(state, "hero_arien")
    if respawn:
        state.remove_entity("hero_arien")
    if delayed_context:
        state.active_effects.append(
            ActiveEffect(
                id="parity_finished_actor_context",
                source_id="hero_arien",
                effect_type=EffectType.DELAYED_TRIGGER,
                scope=EffectScope(shape=Shape.GLOBAL),
                duration=DurationType.THIS_TURN,
                created_at_turn=state.turn,
                created_at_round=state.round,
                finishing_steps=[LogMessageStep(message="finished")],
            )
        )
    viewer_id, team = ("hero_arien", TeamColor.BLUE) if reaction else ("hero_wasp", TeamColor.RED)
    state.execution_context["root_viewer"] = viewer_id
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Parity probe",
                number_options=[1, 2],
                override_player_id_key="root_viewer",
            ),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    request = _pending_request(state)
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    expected = _actual_first(state, viewer_id, team)

    actual = _search_values(
        monkeypatch, state, _input_target(request, viewer_id), legal, viewer_id, team
    )

    assert all(record == expected for record in actual)
    if next_actor:
        assert expected.boundary.kind is StableValueBoundaryKind.ACTOR_READY
        assert expected.boundary.actor_id == "hero_arien"
    else:
        assert expected.boundary.kind is StableValueBoundaryKind.PLANNING_READY
        assert expected.boundary.actor_id is None
        assert expected.boundary.round == state.round + (turn == 4)
        assert expected.boundary.turn == (1 if turn == 4 else 2)


def test_actorless_team_tie_search_matches_live_selected_actor_encoding(monkeypatch) -> None:
    state = _game(["Wasp", "Xargatha"])
    state.phase = GamePhase.RESOLUTION
    state.execution_stack.clear()
    state.pending_inputs.clear()
    state.current_actor_id = None
    state.resolution_owner_id = None
    state.unresolved_hero_ids = [HeroID("hero_wasp"), HeroID("hero_xargatha")]
    for hero_id in state.unresolved_hero_ids:
        _give_turn_card(state, str(hero_id))
    push_steps(state, [ResolveTieBreakerStep(tied_hero_ids=list(state.unresolved_hero_ids))])
    request = _pending_request(state)
    assert request.player_id == "team:RED"
    legal = tuple(legal_keys(DecisionDescriptor("INPUT", request=request)))
    expected = {
        _actual_first(state, "hero_wasp", TeamColor.RED, actor_choice=str(key)) for key in legal
    }

    actual = _search_values(
        monkeypatch,
        state,
        _input_target(request, "hero_wasp"),
        legal,
        "hero_wasp",
        TeamColor.RED,
    )

    assert set(actual) == expected
    assert {record.boundary.actor_id for record in actual} == set(legal)
