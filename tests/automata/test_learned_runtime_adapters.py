from __future__ import annotations

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.models.contracts.inference import LearnedModelOutput
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import (
    LeafMode,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackLeafEvaluator, FallbackSearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.learned import LearnedLeafEvaluator, LearnedSearchPolicy
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.setup import GameSetup
from goa2.engine.steps.selection import AskConfirmationStep

MAP = "src/goa2/data/maps/forgotten_island.json"


class RecordingRuntime:
    def __init__(self, logits: tuple[float, ...], value: float = 0.25) -> None:
        self.logits = logits
        self.value = value
        self.observations = []

    def evaluate(self, observation):
        self.observations.append(observation)
        ids = tuple(candidate.candidate_id for candidate in observation.candidates)
        return LearnedModelOutput(
            candidate_ids=ids,
            policy_logits=self.logits,
            probabilities=tuple(1.0 / len(ids) for _ in ids),
            value=self.value,
        )


class DynamicRecordingRuntime:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.observations = []

    def evaluate(self, observation):
        self.observations.append(observation)
        ids = tuple(candidate.candidate_id for candidate in observation.candidates)
        return LearnedModelOutput(
            candidate_ids=ids,
            policy_logits=tuple(0.0 for _ in ids),
            probabilities=tuple(1.0 / len(ids) for _ in ids),
            value=self.value,
        )


def _state():
    return GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)


def _live_resolution_input():
    register_all_effects()
    state = GameSetup.create_game(MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=3)
    session = GameSession(state)
    wasp = state.get_hero(HeroID("hero_wasp"))
    arien = state.get_hero(HeroID("hero_arien"))
    assert wasp is not None and arien is not None
    session.commit_card(HeroID(wasp.id), wasp.hand[0])
    result = session.commit_card(HeroID(arien.id), arien.hand[0])
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    assert not state.input_stack
    return state, result.input_request


def test_learned_policy_returns_exact_caller_aligned_raw_logits_from_fixed_viewer() -> None:
    state = _state()
    request = InputRequest(
        id="public-units",
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_razzle",
        options=[
            InputOption.from_value("hero_razzle_piece_1"),
            InputOption.from_value("hero_arien"),
        ],
    )
    legal = ("hero_razzle_piece_1", "hero_arien")
    runtime = RecordingRuntime(tuple(float(index) for index in range(len(legal))))
    context = SearchContext(
        "hero_arien",
        TeamColor.BLUE,
        "hero_razzle",
        DecisionDescriptor("INPUT", request=request),
    )

    assert not state.input_stack

    scores = LearnedSearchPolicy(runtime).score(context, state, legal)

    assert scores == PolicyScores(legal, runtime.logits, ScoreSemantics.LOGITS)
    assert runtime.observations[0].state.viewer.private_hero_id == "hero_arien"
    assert runtime.observations[0].state.viewer.perspective_team == "BLUE"
    owner_token = next(
        token
        for token in runtime.observations[0].state.tokens
        if token.kind == "HERO" and token.features["hero_id"] == "hero_razzle"
    )
    assert owner_token.features["is_decision_owner"] is True


def test_learned_leaf_uses_same_runtime_and_rejects_out_of_contract_value() -> None:
    state = _state()
    owner = state.get_hero(HeroID("hero_razzle"))
    assert owner is not None
    runtime = RecordingRuntime(tuple(0.0 for _ in owner.hand), 0.75)
    context = SearchContext(
        "hero_razzle",
        TeamColor.RED,
        "hero_razzle",
        DecisionDescriptor("CARD", hero=owner),
    )

    assert LearnedLeafEvaluator(runtime).evaluate(context, state).value == 0.75

    runtime.value = 1.5
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        LearnedLeafEvaluator(runtime).evaluate(context, state)


def test_changed_card_owner_keeps_root_viewer_and_candidate_alignment() -> None:
    state = _state()
    owner = state.get_hero(HeroID("hero_razzle"))
    assert owner is not None
    legal = tuple(card.id for card in owner.hand)
    runtime = RecordingRuntime(tuple(float(index) for index in range(len(legal))))

    scores = LearnedSearchPolicy(runtime).score(
        SearchContext(
            "hero_arien",
            TeamColor.BLUE,
            "hero_razzle",
            DecisionDescriptor("CARD", hero=owner),
        ),
        state,
        legal,
    )

    assert scores.actions == legal
    assert runtime.observations[0].state.viewer.private_hero_id == "hero_arien"
    targets = {candidate.target_ref for candidate in runtime.observations[0].candidates}
    assert len(targets) == 1
    assert targets == {"hero:hero_razzle"}


def test_inference_failure_falls_back_per_component_but_contract_errors_propagate() -> None:
    state = _state()
    owner = state.get_hero(HeroID("hero_razzle"))
    assert owner is not None
    legal = tuple(card.id for card in owner.hand)
    context = SearchContext(
        "hero_razzle",
        TeamColor.RED,
        "hero_razzle",
        DecisionDescriptor("CARD", hero=owner),
    )

    class FailedRuntime:
        def evaluate(self, observation):
            raise RuntimeError("device inference failed")

    fallback_policy = FallbackSearchPolicy(
        LearnedSearchPolicy(FailedRuntime()), HeuristicPrior(HeuristicAgent(seed=1))
    )
    fallback_leaf = FallbackLeafEvaluator(
        LearnedLeafEvaluator(FailedRuntime()), HeuristicLeafEvaluator()
    )

    assert fallback_policy.score(context, state, legal).actions == legal
    assert -1.0 <= fallback_leaf.evaluate(context, state).value <= 1.0

    class ContractBrokenRuntime(RecordingRuntime):
        def evaluate(self, observation):
            output = super().evaluate(observation)
            return LearnedModelOutput(
                candidate_ids=tuple(reversed(output.candidate_ids)),
                policy_logits=output.policy_logits,
                probabilities=output.probabilities,
                value=output.value,
            )

    broken = ContractBrokenRuntime(tuple(0.0 for _ in legal))
    with pytest.raises(ValueError, match="exact legal action order"):
        FallbackSearchPolicy(
            LearnedSearchPolicy(broken), HeuristicPrior(HeuristicAgent(seed=1))
        ).score(context, state, legal)


def test_heuristic_prior_scores_the_explicit_live_input_without_input_stack() -> None:
    state = _state()
    request = InputRequest(
        id="live-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_razzle",
        options=[InputOption.from_value("LOW"), InputOption.from_value("HIGH")],
    )

    class ExplicitScores(HeuristicAgent):
        def score_option(self, state, request, option):
            return {"LOW": -2.0, "HIGH": 3.0}[option.id]

    context = SearchContext(
        "hero_razzle",
        TeamColor.RED,
        "hero_razzle",
        DecisionDescriptor("INPUT", request=request),
    )
    scores = HeuristicPrior(ExplicitScores(seed=1)).score(
        context, state, ["LOW", "HIGH"]
    )

    assert not state.input_stack
    assert scores.scores == (-2.0, 3.0)


@pytest.mark.parametrize(
    ("policy_source", "value_source"),
    [("learned", "heuristic"), ("heuristic", "learned"), ("learned", "learned")],
    ids=["L-H", "H-L", "L-L"],
)
def test_live_session_input_runs_ismcts_with_independent_learned_components(
    policy_source: str, value_source: str
) -> None:
    state, request = _live_resolution_input()
    heuristic = HeuristicAgent(seed=1)
    policy_runtime = RecordingRuntime((0.0, 1.0))
    leaf_runtime = RecordingRuntime((0.0,), value=0.4)
    prior = (
        LearnedSearchPolicy(policy_runtime)
        if policy_source == "learned"
        else HeuristicPrior(heuristic)
    )
    leaf = (
        LearnedLeafEvaluator(leaf_runtime)
        if value_source == "learned"
        else HeuristicLeafEvaluator()
    )
    agent = ISMCTSAgent(
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE, seed=2),
        environment_policy=heuristic,
        prior=prior,
        leaf_evaluator=leaf,
    )

    selection = agent.choose_input(
        state,
        request,
        owned_hero_ids=frozenset({"hero_arien"}),
        decision_owner_hero_id="hero_arien",
    )

    assert selection in {"SKILL", "HOLD"}
    assert bool(policy_runtime.observations) is (policy_source == "learned")
    assert bool(leaf_runtime.observations) is (value_source == "learned")
    for runtime in (policy_runtime, leaf_runtime):
        assert all(
            observation.state.viewer.private_hero_id == "hero_arien"
            for observation in runtime.observations
        )


def test_team_scoped_live_session_input_uses_eligible_root_viewer_for_learned_search() -> None:
    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.execution_stack = [
        AskConfirmationStep(prompt="leaf", player_id="team:RED"),
        AskConfirmationStep(prompt="root", player_id="team:RED"),
    ]
    result = GameSession(state).advance()
    assert result.result_type is SessionResultType.INPUT_NEEDED
    assert result.input_request is not None
    assert result.input_request.player_id == "team:RED"
    assert not state.input_stack

    policy_runtime = RecordingRuntime((0.0, 1.0))
    leaf_runtime = RecordingRuntime((0.0, 0.0), value=0.2)
    heuristic = HeuristicAgent(seed=1)
    agent = ISMCTSAgent(
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE, seed=2),
        environment_policy=heuristic,
        prior=LearnedSearchPolicy(policy_runtime),
        leaf_evaluator=LearnedLeafEvaluator(leaf_runtime),
    )

    selection = agent.choose_input(
        state,
        result.input_request,
        owned_hero_ids=frozenset({"hero_razzle"}),
        decision_owner_hero_id="hero_razzle",
    )

    assert selection in {"YES", "NO"}
    assert policy_runtime.observations and leaf_runtime.observations
    for runtime in (policy_runtime, leaf_runtime):
        for observation in runtime.observations:
            assert observation.state.viewer.private_hero_id == "hero_razzle"
            owner = next(
                token
                for token in observation.state.tokens
                if token.kind == "HERO" and token.features["hero_id"] == "hero_razzle"
            )
            assert owner.features["is_decision_owner"] is True


def test_learned_leaf_advances_past_an_owned_hero_forced_zero_hand_pass() -> None:
    register_all_effects()
    state = GameSetup.create_game(
        MAP,
        ["Razzle", "Wasp"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=3,
    )
    razzle = state.get_hero(HeroID("hero_razzle"))
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert razzle is not None and wasp is not None
    wasp.hand.clear()
    runtime = DynamicRecordingRuntime(value=0.1)
    heuristic = HeuristicAgent(seed=1)
    agent = ISMCTSAgent(
        SearchConfig(iterations=1, leaf_mode=LeafMode.IMMEDIATE, seed=2),
        environment_policy=heuristic,
        prior=HeuristicPrior(heuristic),
        leaf_evaluator=LearnedLeafEvaluator(runtime),
    )

    planning = agent.choose_planning(
        state,
        razzle,
        owned_hero_ids=frozenset({"hero_razzle", "hero_wasp"}),
    )

    assert planning.card in razzle.hand
    assert runtime.observations
    assert all(observation.candidates for observation in runtime.observations)
