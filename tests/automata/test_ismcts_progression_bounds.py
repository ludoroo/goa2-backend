from __future__ import annotations

from dataclasses import asdict

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.search.ismcts import (
    RootTarget,
    SearchProgressionError,
    search,
    validate_search_root,
)
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.engine.setup import GameSetup


def _state(*, two_red: bool = False):
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp", "Xargatha"] if two_red else ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )


def _wasp_root(state, *, own_xargatha: bool = False):
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    owned = {wasp.id}
    if own_xargatha:
        owned.add("hero_xargatha")
    return (
        wasp,
        RootTarget.card(hero_id=wasp.id, owned_hero_ids=frozenset(owned)),
        tuple(card.id for card in wasp.hand),
    )


def test_unchanged_action_complete_fails_closed_with_diagnostics(monkeypatch) -> None:
    from automata.search.ismcts import engine

    class StalledSession(GameSession):
        def advance(self, response=None):
            return SessionResult(
                result_type=SessionResultType.ACTION_COMPLETE,
                current_phase=self.state.phase,
            )

    state = _state()
    _, target, legal = _wasp_root(state)
    state.phase = GamePhase.RESOLUTION
    monkeypatch.setattr(engine, "GameSession", StalledSession)

    with pytest.raises(SearchProgressionError, match="unchanged ACTION_COMPLETE") as raised:
        validate_search_root(
            state,
            TeamColor.RED,
            target,
            legal,
            HeuristicAgent(2),
            cfg=SearchConfig(max_advance_transitions=8),
        )

    error = raised.value
    assert error.phase == GamePhase.RESOLUTION.value
    assert error.round == state.round
    assert error.actor is None
    assert error.pending_request is None
    assert error.stack_depth == 0
    assert error.top_step is None
    assert error.transition_counts["session_advances"] == 1


def test_changing_action_complete_still_hits_advance_transition_bound(monkeypatch) -> None:
    from automata.search.ismcts import engine

    class ChangingButUnboundedSession(GameSession):
        def advance(self, response=None):
            self.state.turn += 1
            return SessionResult(
                result_type=SessionResultType.ACTION_COMPLETE,
                current_phase=self.state.phase,
            )

    state = _state()
    _, target, legal = _wasp_root(state)
    state.phase = GamePhase.RESOLUTION
    monkeypatch.setattr(engine, "GameSession", ChangingButUnboundedSession)

    with pytest.raises(SearchProgressionError, match="transition limit exceeded") as raised:
        validate_search_root(
            state,
            TeamColor.RED,
            target,
            legal,
            HeuristicAgent(2),
            cfg=SearchConfig(max_advance_transitions=3),
        )

    assert raised.value.transition_counts["current_advance_transitions"] == 4
    assert raised.value.transition_counts["session_advances"] == 4


def test_repeating_environment_input_decision_fails_closed(monkeypatch) -> None:
    from automata.search.ismcts import engine

    request = InputRequest(
        id="repeating-blue-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_arien",
        options=[InputOption.from_value("a"), InputOption.from_value("b")],
    )

    class RepeatingInputSession(GameSession):
        def advance(self, response=None):
            return SessionResult(
                result_type=SessionResultType.INPUT_NEEDED,
                input_request=request,
                current_phase=self.state.phase,
            )

    state = _state()
    _, target, legal = _wasp_root(state)
    state.phase = GamePhase.RESOLUTION
    monkeypatch.setattr(engine, "GameSession", RepeatingInputSession)

    with pytest.raises(SearchProgressionError, match="repeated environment input") as raised:
        validate_search_root(
            state,
            TeamColor.RED,
            target,
            legal,
            HeuristicAgent(2),
            cfg=SearchConfig(max_advance_transitions=8),
        )

    assert raised.value.pending_request == request.id
    assert raised.value.transition_counts["environment_inputs"] == 2
    assert raised.value.transition_counts["session_advances"] == 2


@pytest.mark.parametrize(
    ("leaf_mode", "iterations"),
    [(LeafMode.IMMEDIATE, 2), (LeafMode.BOUNDED_CONTINUATION, 1)],
)
def test_repeated_forced_tree_decision_fails_closed(
    monkeypatch, leaf_mode: LeafMode, iterations: int
) -> None:
    from automata.search.ismcts import engine

    class ForcedCycleSession(GameSession):
        def pass_turn(self, hero_id):
            if hero_id == HeroID("hero_xargatha"):
                return SessionResult(
                    result_type=SessionResultType.ACTION_COMPLETE,
                    current_phase=self.state.phase,
                )
            return super().pass_turn(hero_id)

    state = _state(two_red=True)
    wasp, target, legal = _wasp_root(state, own_xargatha=True)
    xargatha = state.get_hero(HeroID("hero_xargatha"))
    assert xargatha is not None
    real_legal_keys = engine.legal_keys

    def no_legal_xargatha(decision):
        if decision.hero is not None and decision.hero.id == xargatha.id:
            return []
        return real_legal_keys(decision)

    monkeypatch.setattr(engine, "GameSession", ForcedCycleSession)
    monkeypatch.setattr(engine, "legal_keys", no_legal_xargatha)

    with pytest.raises(SearchProgressionError, match="repeated forced decision") as raised:
        search(
            state,
            TeamColor.RED,
            legal,
            HeuristicAgent(2),
            SearchConfig(
                iterations=iterations,
                cutoff_limit=1,
                leaf_mode=leaf_mode,
                root_widening_c=0.1,
                max_forced_decisions=8,
            ),
            root_target=target,
        )

    assert raised.value.phase == GamePhase.PLANNING.value
    assert raised.value.actor == xargatha.id
    assert raised.value.transition_counts["forced_decisions"] == 1
    assert wasp.hand


def test_default_progression_bounds_allow_normal_complex_continuation() -> None:
    state = _state(two_red=True)
    _, target, legal = _wasp_root(state, own_xargatha=True)

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(iterations=2, cutoff_limit=1, seed=11),
        root_target=target,
    )

    assert result.best_key in legal
    assert result.root.visits == 2


def test_progression_bounds_do_not_change_exact_eight_iteration_budget() -> None:
    state = _state()
    _, target, legal = _wasp_root(state)

    result = search(
        state,
        TeamColor.RED,
        legal,
        HeuristicAgent(2),
        SearchConfig(iterations=8, leaf_mode=LeafMode.IMMEDIATE, seed=11),
        root_target=target,
    )

    assert result.root.visits == 8
    assert sum(item.visits for item in result.root_action_diagnostics) == 8


@pytest.mark.parametrize("field", ["max_advance_transitions", "max_forced_decisions"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_progression_bounds_must_be_positive_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        SearchConfig(**{field: value})


def test_progression_bounds_participate_in_config_identity() -> None:
    baseline = SearchConfig()
    changed = SearchConfig(max_advance_transitions=baseline.max_advance_transitions + 1)

    assert baseline != changed
    assert "max_advance_transitions" in asdict(baseline)
    assert "max_forced_decisions" in asdict(baseline)
