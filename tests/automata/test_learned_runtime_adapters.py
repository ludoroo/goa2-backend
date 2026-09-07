from __future__ import annotations

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.models.contracts.inference import LearnedModelOutput
from automata.search.contracts import (
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackLeafEvaluator, FallbackSearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.learned import LearnedLeafEvaluator, LearnedSearchPolicy
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

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


def _state():
    return GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)


def test_learned_policy_returns_exact_caller_aligned_raw_logits_from_fixed_viewer() -> None:
    state = _state()
    state.input_stack.append(
        InputRequest(
            id="public-units",
            request_type=InputRequestType.SELECT_UNIT,
            player_id="hero_razzle",
            options=[
                InputOption.from_value("hero_razzle_piece_1"),
                InputOption.from_value("hero_arien"),
            ],
        )
    )
    legal = ("hero_razzle_piece_1", "hero_arien")
    runtime = RecordingRuntime(tuple(float(index) for index in range(len(legal))))
    context = SearchContext("hero_arien", TeamColor.BLUE, "hero_razzle")

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
    context = SearchContext("hero_razzle", TeamColor.RED, "hero_razzle")

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
        SearchContext("hero_arien", TeamColor.BLUE, "hero_razzle"), state, legal
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
    context = SearchContext("hero_razzle", TeamColor.RED, "hero_razzle")

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
