"""Actual engine outcomes reach diagnostic and learning sinks without ambiguity.

Raw-stack fixtures isolate outcome plumbing, not a character effect. This tests
retained joint publication; it is not the future Gen1 policy/value row format.
All recorded decisions are played in the real harness, not search leaves.
"""

from __future__ import annotations

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import DecisionDescriptor
from automata.evaluation.protocol import EvaluationGameResult, summarize
from automata.harness.game_runner import DEFAULT_MAP, continue_game
from automata.harness.trajectory import InMemoryRecorder
from automata.observation import encode_decision, legal_keys_for_decision
from automata.runtime.effects import register_all_effects
from automata.training.dataset import JointDatasetRecorder, load_joint_dataset
from goa2.domain.models import GamePhase, TargetType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import SelectStep, TriggerGameOverStep


class _PlayedDecisions:
    def __init__(self, recorder):
        self.recorder = recorder
        self.outcomes = []
        self.decision_count = 0

    def record_decision(self, state, decision):
        assert decision.request is not None
        hero = state.get_hero(decision.hero_id)
        assert hero is not None and hero.team is not None
        descriptor = DecisionDescriptor("INPUT", request=decision.request)
        observation = encode_decision(
            state,
            descriptor,
            legal_keys_for_decision(descriptor),
            decision_owner_hero_id=str(hero.id),
            perspective_team=hero.team.value,
        )
        selected = next(
            candidate
            for candidate in observation.candidates
            if candidate.selection == decision.selection
        )
        self.recorder.record_decision(
            observation=observation,
            policy_source="HEURISTIC",
            policy_target=tuple(
                float(candidate == selected) for candidate in observation.candidates
            ),
            selected_candidate_id=selected.candidate_id,
            selected_selection=selected.selection,
        )
        self.decision_count += 1

    def record_outcome(self, *, winner_side, rounds, reason):
        self.outcomes.append((winner_side, rounds, reason))
        self.recorder.record_outcome(winner_side=winner_side, rounds=rounds, reason=reason)


class _BoundaryOutcomes:
    def __init__(self):
        self.outcomes = []

    def record_boundary(self, *_args, **_kwargs):
        pytest.fail("this terminal fixture has no intervening stable boundary")

    def record_outcome(self, *, winner_side, rounds, reason):
        self.outcomes.append((winner_side, rounds, reason))


@pytest.mark.parametrize("winning_team", [TeamColor.RED, TeamColor.BLUE])
@pytest.mark.parametrize("ending", ["terminal", "max_steps", "invalid_winner", "missing_winner"])
def test_actual_outcome_separates_raw_evidence_from_normalized_labels(
    tmp_path, monkeypatch, winning_team, ending
):
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=101)
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    raw_winner = (
        "hero_missing" if ending == "invalid_winner" else state.teams[winning_team].heroes[0].id
    )
    push_steps(
        state,
        [
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Played RED choice",
                number_options=[1, 2],
                override_player_id="hero_wasp",
            ),
            SelectStep(
                target_type=TargetType.NUMBER,
                prompt="Played BLUE choice",
                number_options=[1, 2],
                override_player_id="hero_arien",
            ),
            TriggerGameOverStep(individual_winner_id=HeroID(raw_winner), condition="TEST"),
        ],
    )
    if ending == "missing_winner":
        # Inject a malformed terminal state after real engine decisions. There
        # is no engine draw rule, so absent markers must not publish zero labels.
        resolve = TriggerGameOverStep.resolve

        def resolve_without_winner(step, state, context):
            result = resolve(step, state, context)
            state.individual_winner_id = None
            state.winner = None
            return result

        monkeypatch.setattr(TriggerGameOverStep, "resolve", resolve_without_winner)

    output = tmp_path / "data" / "completed.jsonl.zst"
    recorder = JointDatasetRecorder(
        output,
        game_id="outcome-contract",
        world_seed=101,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
        generation_id="outcome-test",
        source_revision="test-source",
        dirty_tree_hash="test-tree",
        source_model_digest=None,
        search_config_id="heuristic-choice",
        generator_config_id="outcome-test",
    )
    decisions = _PlayedDecisions(recorder)
    boundaries = _BoundaryOutcomes()
    diagnostic = InMemoryRecorder()
    agents = {"hero_wasp": HeuristicAgent(0), "hero_arien": HeuristicAgent(1)}

    try:
        if ending in {"invalid_winner", "missing_winner"}:
            with pytest.raises(ValueError, match="winner"):
                continue_game(
                    state,
                    agents,
                    recorder=diagnostic,
                    decision_observer=decisions,
                    boundary_observer=boundaries,
                )
            assert decisions.decision_count == 2
            assert decisions.outcomes == boundaries.outcomes == []
            assert diagnostic.outcome is not None
            assert diagnostic.outcome["winner"] == (
                raw_winner if ending == "invalid_winner" else None
            )
            assert diagnostic.outcome["reason"] == "game_over"
        else:
            result = continue_game(
                state,
                agents,
                max_steps=2 if ending == "max_steps" else 20,
                recorder=diagnostic,
                decision_observer=decisions,
                boundary_observer=boundaries,
            )
            expected_side = winning_team.value if ending == "terminal" else None
            expected_reason = "game_over" if ending == "terminal" else "max_steps"
            assert result.reason == expected_reason
            assert result.winner_side == expected_side
            assert result.winner == (raw_winner if ending == "terminal" else None)
            assert diagnostic.outcome is not None
            assert diagnostic.outcome["winner"] == result.winner
            assert (
                decisions.outcomes
                == boundaries.outcomes
                == [(expected_side, state.round, expected_reason)]
            )
            summary = summarize(
                [
                    EvaluationGameResult(
                        case_id="actual-outcome",
                        world_seed=101,
                        a_side="RED",
                        winner_side=result.winner_side,
                        rounds=result.rounds,
                        steps=result.steps,
                        reason=result.reason,
                    )
                ]
            )
            assert summary.draws == 0
            assert summary.a_wins == int(ending == "terminal" and winning_team is TeamColor.RED)
            assert summary.b_wins == int(ending == "terminal" and winning_team is TeamColor.BLUE)
            if ending == "max_steps":
                assert summary.max_step_terminations == 1
                assert not summary.screening_passes()
                assert not summary.promotion_passes()
    finally:
        recorder.close()

    if ending == "terminal":
        rows = load_joint_dataset(output).rows
        assert decisions.decision_count == len(rows) == 2
        assert {row.perspective_team for row in rows} == {"RED", "BLUE"}
        for row in rows:
            assert row.terminal_winner == winning_team.value
            assert row.value_target == (1 if row.perspective_team == winning_team.value else -1)
    else:
        assert decisions.decision_count > 0
        # No completed dataset or provisional data survive censorship/failure.
        assert list(output.parent.iterdir()) == []
