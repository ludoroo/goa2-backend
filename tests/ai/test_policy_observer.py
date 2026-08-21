"""Behavioral contract for injectable search priors and root observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import automata.search.agent as agent_module
from automata.agents.random_agent import RandomAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import ISMCTSAgent, SearchConfig, policy_candidate_features
from automata.search.ismcts import Decision, SearchResult
from automata.search.node import Node
from automata.search.prior import HeuristicPrior, PolicyResult
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.engine.setup import GameSetup

if TYPE_CHECKING:
    from automata.search.node import Key


RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _state():
    register_all_effects()
    return GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)


def _config(*, seed: int = 7, use_prior: bool = True) -> SearchConfig:
    return SearchConfig(
        iterations=1,
        cutoff_rounds=0,
        widening_c=1.0,
        widening_alpha=0.5,
        seed=seed,
        use_prior=use_prior,
    )


def _new_agent(config: SearchConfig, **kwargs: Any) -> ISMCTSAgent:
    """Call the planned constructor without hiding runtime signature failures."""
    constructor: Any = ISMCTSAgent
    return constructor(config, **kwargs)


class _RecordingPrior:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __call__(
        self, state: object, decision: Decision, legal: list[object]
    ) -> PolicyResult:
        del state
        self.calls.append((decision.kind, tuple(legal)))
        return PolicyResult(order=list(reversed(legal)))


def test_explicit_prior_drives_search_with_nonheuristic_rollout_policy() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    legal = [card.id for card in hero.hand]
    assert len(legal) > 1
    prior = _RecordingPrior()

    chosen = _new_agent(
        _config(), default_policy=RandomAgent(3), prior=prior
    ).choose_card(state, hero)

    assert prior.calls
    assert prior.calls[0] == ("CARD", tuple(legal))
    assert chosen is not None
    assert chosen.id == legal[-1]


def test_explicit_prior_is_rejected_when_priors_are_disabled() -> None:
    with pytest.raises(ValueError, match="prior"):
        _new_agent(
            _config(use_prior=False), prior=_RecordingPrior()
        )


def test_card_root_observer_reports_complete_read_only_search_statistics() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    legal: list[Key] = [card.id for card in hero.hand]
    observations: list[tuple[GameState, Any]] = []
    agent = _new_agent(
        _config(), root_observer=lambda root, observation: observations.append((root, observation))
    )

    chosen = agent.choose_card(state, hero)

    assert chosen is not None
    assert len(observations) == 1
    observed_state, observation = observations[0]
    assert observed_state is state
    assert observation.decision_owner_hero_id == hero.id
    assert observation.decision_kind == "CARD"
    assert observation.request is None
    assert set(observation.legal_keys) == set(legal)
    assert observation.chosen_key == chosen.id
    expected_first = HeuristicPrior()(state, Decision("CARD", hero=hero), legal).order[0]
    assert chosen.id == expected_first
    assert set(observation.child_stats) == set(legal)

    expanded = [stats for stats in observation.child_stats.values() if stats.visits]
    unexpanded = [stats for stats in observation.child_stats.values() if not stats.visits]
    assert len(expanded) == 1
    assert unexpanded
    for stats in observation.child_stats.values():
        assert stats.q == pytest.approx(
            stats.total_value / stats.visits if stats.visits else 0.0
        )

    with pytest.raises((AttributeError, TypeError)):
        observation.decision_kind = "INPUT"
    with pytest.raises(TypeError):
        observation.child_stats[chosen.id] = expanded[0]


def test_card_observer_can_reconstruct_policy_features_for_every_legal_key() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    rows: list[dict[Key, dict[str, float]]] = []

    def extract(root: GameState, observation: Any) -> None:
        observed_hero = root.get_hero(observation.decision_owner_hero_id)
        assert observed_hero is not None
        decision = Decision("CARD", hero=observed_hero)
        rows.append(policy_candidate_features(root, decision, observation.legal_keys))

    _new_agent(_config(), root_observer=extract).choose_card(state, hero)

    assert len(rows) == 1
    assert set(rows[0]) == {card.id for card in hero.hand}
    assert all(rows[0][key] for key in rows[0])


def test_input_observer_receives_exact_predecision_root_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    request = InputRequest(
        id="observed-input",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("hold")],
    )
    state.input_stack.append(request)
    observed: list[tuple[GameState, Any]] = []

    def canned_search(*args: Any, **kwargs: Any) -> SearchResult:
        del args, kwargs
        return SearchResult(root=Node(), best_key="hold")

    monkeypatch.setattr(agent_module, "search", canned_search)
    result = _new_agent(
        _config(), root_observer=lambda root, row: observed.append((root, row))
    ).choose_input(
        state,
        request,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
    )

    assert result == "hold"
    assert len(observed) == 1
    assert observed[0][0] is state
    assert observed[0][1].request is request


def test_root_observer_exceptions_propagate() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]

    def reject_row(state: object, observation: object) -> None:
        del state, observation
        raise RuntimeError("dataset write failed")

    agent = _new_agent(_config(), root_observer=reject_row)
    with pytest.raises(RuntimeError, match="dataset write failed"):
        agent.choose_card(state, hero)


def test_absent_root_observer_does_not_change_deterministic_choice() -> None:
    def choose(observer: object | None, *, use_prior: bool = True) -> str | None:
        state = _state()
        hero = state.teams[TeamColor.RED].heroes[0]
        kwargs: dict[str, Any] = (
            {} if observer is None else {"root_observer": observer}
        )
        card = ISMCTSAgent(
            _config(seed=11, use_prior=use_prior),
            default_policy=RandomAgent(5),
            **kwargs,
        ).choose_card(state, hero)
        return card.id if card is not None else None

    observed: list[Any] = []
    baseline = choose(None)
    assert baseline == choose(None)
    # A non-heuristic default policy receives no implicit prior, even when the
    # config enables priors.
    assert baseline == choose(None, use_prior=False)
    assert baseline == choose(lambda state, row: observed.append((state, row)))
    assert len(observed) == 1
