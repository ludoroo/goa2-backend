from __future__ import annotations

from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode, SearchContext
from automata.search.ismcts import legal_keys, search
from automata.search.ismcts.engine import _rollout, terminal_reward
from automata.search.root import RootTarget
from goa2.domain.input import InputRequest
from goa2.domain.models import GamePhase, TargetType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps import SelectStep, TriggerGameOverStep


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _state():
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    return state


@pytest.mark.parametrize(
    ("winner", "perspective", "expected"),
    [
        ("RED", TeamColor.RED, 1.0),
        ("red", TeamColor.BLUE, 0.0),
        ("BLUE", TeamColor.BLUE, 1.0),
        ("blue", TeamColor.RED, 0.0),
        ("hero_wasp", TeamColor.RED, 1.0),
        ("hero_wasp", TeamColor.BLUE, 0.0),
        ("hero_arien", TeamColor.BLUE, 1.0),
        ("hero_arien", TeamColor.RED, 0.0),
    ],
)
def test_terminal_reward_resolves_team_and_individual_winners(
    winner: str, perspective: TeamColor, expected: float
) -> None:
    state = _state()

    assert terminal_reward(winner, perspective, state=state) == expected


def test_terminal_reward_preserves_a_genuine_draw() -> None:
    assert terminal_reward(None, TeamColor.RED, state=_state()) == 0.5


@pytest.mark.parametrize("winner", ["GREEN", "hero_missing", "not-a-winner"])
def test_terminal_reward_rejects_unknown_non_null_winner(winner: str) -> None:
    with pytest.raises(ValueError, match="unknown terminal winner"):
        terminal_reward(winner, TeamColor.RED, state=_state())


def test_terminal_reward_rejects_a_team_absent_from_state() -> None:
    state = _state()
    del state.teams[TeamColor.BLUE]

    with pytest.raises(ValueError, match="not present"):
        terminal_reward("BLUE", TeamColor.RED, state=state)


def test_terminal_reward_rejects_a_perspective_team_absent_from_state() -> None:
    state = _state()
    del state.teams[TeamColor.RED]

    with pytest.raises(ValueError, match="perspective team"):
        terminal_reward("BLUE", TeamColor.RED, state=state)


def test_terminal_reward_rejects_a_piece_id_as_an_individual_winner() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Razzle"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )
    piece_ids = state.get_piece_ids("hero_razzle")
    assert piece_ids

    with pytest.raises(ValueError, match="unknown terminal winner"):
        terminal_reward(str(piece_ids[0]), TeamColor.RED, state=state)


def test_terminal_reward_rejects_an_individual_winner_without_a_team() -> None:
    state = _state()
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    hero.team = None

    with pytest.raises(ValueError, match="has no team"):
        terminal_reward(hero.id, TeamColor.RED, state=state)


class _FailingLeafEvaluator:
    def evaluate(self, _context: SearchContext, _state: Any) -> Any:
        pytest.fail("terminal outcome reached the leaf evaluator")


class _TerminalSim:
    def __init__(self, state, team: TeamColor) -> None:
        self.state = state
        self.our_team = team


@pytest.mark.parametrize(
    ("winner", "perspective", "expected"),
    [
        ("hero_wasp", TeamColor.RED, 1.0),
        ("hero_wasp", TeamColor.BLUE, 0.0),
        ("hero_arien", TeamColor.RED, 0.0),
        ("hero_arien", TeamColor.BLUE, 1.0),
    ],
)
def test_rollout_terminal_path_bypasses_leaf_evaluation(
    winner: str, perspective: TeamColor, expected: float
) -> None:
    state = _state()
    context = SearchContext("hero_wasp", perspective, "hero_wasp", DecisionDescriptor("CARD"))

    reward = _rollout(
        _TerminalSim(state, perspective),  # type: ignore[arg-type]
        DecisionDescriptor("OVER", winner=winner),
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE),
        HeuristicAgent(0),
        _FailingLeafEvaluator(),  # type: ignore[arg-type]
        context,
    )

    assert reward == expected


def _terminal_input_root(winner: str) -> tuple[Any, InputRequest, tuple[Any, ...]]:
    state = _state()
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.execution_context["root_owner"] = "hero_wasp"
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Root choice",
                number_options=[1, 2],
                override_player_id_key="root_owner",
            ),
            TriggerGameOverStep(
                individual_winner_id=HeroID(winner),
                condition="TEST",
            ),
        ],
    )
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    decision = DecisionDescriptor("INPUT", request=result.input_request)
    return state, result.input_request, tuple(legal_keys(decision))


def _target(request: InputRequest) -> RootTarget:
    return RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
        request=request,
    )


@pytest.mark.parametrize(("winner", "expected"), [("hero_wasp", 1.0), ("hero_arien", 0.0)])
def test_search_terminal_rollout_and_tree_backup_use_individual_winner_team(
    winner: str, expected: float
) -> None:
    state, request, legal = _terminal_input_root(winner)

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(0),
        SearchConfig(iterations=3, leaf_mode=LeafMode.BOUNDED_CONTINUATION, seed=5),
        root_target=_target(request),
        leaf_evaluator=_FailingLeafEvaluator(),  # type: ignore[arg-type]
    )

    assert result.root.visits == 3
    assert result.root.q == expected
    assert all(item.mean_value == expected for item in result.root_action_diagnostics)


def test_search_fails_closed_on_an_unknown_individual_winner() -> None:
    state, request, legal = _terminal_input_root("hero_missing")

    with pytest.raises(ValueError, match="unknown terminal winner"):
        search(
            state,
            TeamColor.RED,
            legal,
            HeuristicAgent(0),
            SearchConfig(iterations=1, leaf_mode=LeafMode.BOUNDED_CONTINUATION, seed=5),
            root_target=_target(request),
            leaf_evaluator=_FailingLeafEvaluator(),  # type: ignore[arg-type]
        )
