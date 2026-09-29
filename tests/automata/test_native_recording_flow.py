"""Native samples come from played roots and real completed transitions.

Raw-stack fixtures isolate recording, not a hero effect. Only ResolveCardStep's
body is replaced with a deterministic terminal sequence; real search, selection,
turn finalization, actor/planning boundaries, and terminal steps still run.
"""

from __future__ import annotations

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import DEFAULT_MAP, continue_game
from automata.harness.trajectory import InMemoryRecorder
from automata.models.contracts import canonical_json_bytes
from automata.observation import encode_decision
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.strategy import ISMCTSStrategy
from automata.training.native_dataset import (
    NativeGameIdentity,
    iter_native_game_records,
    native_game_id,
)
from automata.training.native_recorder import NativeDatasetRecorder
from automata.training.search_targets import search_policy_target_from_result
from goa2.domain.models import CardState, GamePhase, TargetType
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.setup import GameSetup
from goa2.engine.steps import (
    FinalizeHeroTurnStep,
    ResolveCardStep,
    SelectStep,
    StepResult,
    TriggerGameOverStep,
)


def _identity():
    fields = dict(
        world_seed=61,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
        generation_id="native-flow",
        source_revision="test-source",
        dirty_tree_hash="test-tree",
        source_model_digest=None,
        search_config_id="stable-transition-test",
        generator_config_id="native-flow-v1",
    )
    return NativeGameIdentity(game_id=native_game_id(**fields), **fields)


def _number(hero_id):
    return SelectStep(
        target_type=TargetType.NUMBER,
        prompt="Played choice",
        number_options=[1, 2],
        override_player_id=hero_id,
    )


def _game(*, next_actor):
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=61)
    state.phase = GamePhase.RESOLUTION
    state.pending_inputs.clear()
    state.execution_stack.clear()
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    state.unresolved_hero_ids = [HeroID("hero_arien")] if next_actor else []
    for hero_id in ("hero_wasp", "hero_arien") if next_actor else ("hero_wasp",):
        hero = state.get_hero(HeroID(hero_id))
        card = hero.hand.pop()
        card.state = CardState.UNRESOLVED
        card.is_facedown = False
        hero.current_turn_card = card
    push_steps(
        state,
        [
            _number("hero_arien"),
            _number("hero_arien"),
            _number("hero_wasp"),
            FinalizeHeroTurnStep(hero_id="hero_wasp"),
        ],
    )
    return state


def _terminal_card(monkeypatch, *, winner, missing_winner=False):
    def resolve(step, _state, _context):
        return StepResult(
            is_finished=True,
            new_steps=[
                _number(str(step.hero_id)),
                TriggerGameOverStep(individual_winner_id=HeroID(winner), condition="TEST"),
            ],
        )

    monkeypatch.setattr(ResolveCardStep, "resolve", resolve)
    if missing_winner:
        original = TriggerGameOverStep.resolve

        def without_winner(step, state, context):
            result = original(step, state, context)
            state.winner = None
            state.individual_winner_id = None
            return result

        monkeypatch.setattr(TriggerGameOverStep, "resolve", without_winner)


class _ObservedLeaf(HeuristicLeafEvaluator):
    def __init__(self):
        self.state_ids = []

    def evaluate_stable_value(self, context, state):
        self.state_ids.append(id(state))
        return super().evaluate_stable_value(context, state)


class _PlayedRootSink:
    """Test-only wiring of a real root strategy to the new policy-target sink."""

    strategy_id = "native-recording-test"

    def __init__(self, delegate, recorder):
        self.delegate = delegate
        self.recorder = recorder
        self.selected = []

    def select(self, state, team, root, legal):
        result = self.delegate.select(state, team, root, legal)
        self.selected.append(result.selected_candidate)
        if self.recorder is not None:
            owner = state.get_hero(HeroID(root.decision_owner_hero_id))
            descriptor = DecisionDescriptor(
                root.kind,
                hero=owner if root.kind == "CARD" else None,
                request=root.request,
                can_finish_planning=root.kind == "CARD" and None in legal,
            )
            observation = encode_decision(
                state,
                descriptor,
                legal,
                decision_owner_hero_id=root.decision_owner_hero_id,
                perspective_team=team.value,
            )
            self.recorder.record_policy(
                observation=observation,
                target=search_policy_target_from_result(result, observation.candidates),
                perspective_team=team.value,
            )
        return result


def _agents(recorder=None):
    config = SearchConfig(iterations=2, leaf_mode=LeafMode.STABLE_TRANSITION, seed=19)
    leaf = _ObservedLeaf()
    policy = HeuristicAgent(19)
    delegate = ISMCTSStrategy(
        environment_policy=policy,
        config=config,
        prior=HeuristicPrior(policy),
        leaf_evaluator=leaf,
    )
    sink = _PlayedRootSink(delegate, recorder)
    agent = ISMCTSAgent(config, strategy=sink)
    return {"hero_wasp": agent, "hero_arien": agent}, sink, leaf


class _ObservedRecorder(NativeDatasetRecorder):
    def __init__(self, path, *, game):
        super().__init__(path, game=game)
        self.destination = path
        self.boundaries = []
        self.expected_observations = []
        self.outcomes = []

    def record_boundary(self, state, boundary, *, viewer_hero_ids):
        assert not self.destination.exists()
        self.boundaries.append((id(state), boundary, viewer_hero_ids))
        for viewer in viewer_hero_ids:
            hero = state.get_hero(HeroID(viewer))
            observation = encode_stable_value(
                state, boundary, viewer_hero_id=viewer, perspective_team=hero.team
            )
            self.expected_observations.append(canonical_json_bytes(observation))
        return super().record_boundary(state, boundary, viewer_hero_ids=viewer_hero_ids)

    def record_outcome(self, **outcome):
        self.outcomes.append(outcome)
        return super().record_outcome(**outcome)


def _trace(raw):
    return [(row["player_id"], row["chosen_key"]) for row in raw.decisions]


@pytest.mark.parametrize("next_actor", [False, True], ids=["planning-ready", "actor-ready"])
@pytest.mark.parametrize("winner_side", ["RED", "BLUE"])
def test_native_records_only_played_roots_and_actual_boundaries(
    tmp_path, monkeypatch, next_actor, winner_side
):
    winner = "hero_wasp" if winner_side == "RED" else "hero_arien"
    _terminal_card(monkeypatch, winner=winner)
    plain = _game(next_actor=next_actor)
    plain_raw = InMemoryRecorder()
    plain_agents, _, _ = _agents()
    expected = continue_game(plain, plain_agents, max_steps=30, recorder=plain_raw)
    assert expected.reason == "game_over"

    state = _game(next_actor=next_actor)
    destination = tmp_path / "played.jsonl.zst"
    raw = InMemoryRecorder()
    with _ObservedRecorder(destination, game=_identity()) as recorder:
        agents, sink, leaf = _agents(recorder)
        actual = continue_game(
            state, agents, max_steps=30, recorder=raw, boundary_observer=recorder
        )

    assert actual == expected
    assert _trace(raw) == _trace(plain_raw)
    assert state.entity_locations == plain.entity_locations
    assert state.phase == plain.phase == GamePhase.GAME_OVER
    assert len(recorder.outcomes) == 1
    assert recorder.outcomes[0]["winner_side"] == winner_side
    assert raw.outcome["winner"] == winner
    assert recorder.boundaries
    first_state_id, first_boundary, first_viewers = recorder.boundaries[0]
    assert first_state_id == id(state)
    assert first_boundary.kind.value == ("ACTOR_READY" if next_actor else "PLANNING_READY")
    assert first_viewers == ("hero_arien", "hero_wasp")
    assert leaf.state_ids and id(state) not in leaf.state_ids
    assert all(state_id == id(state) for state_id, _, _ in recorder.boundaries)

    records = list(iter_native_game_records(destination))
    policies = [row for row in records if row.sample_kind == "POLICY"]
    values = [row for row in records if row.sample_kind == "VALUE"]
    assert len(policies) == len(raw.decisions) == len(sink.selected)
    assert [row.sample_index for row in records] == list(range(len(records)))
    assert [row.policy_index for row in policies] == list(range(len(policies)))
    assert [row.target.selected_candidate_id for row in policies] == [
        next(
            candidate.candidate_id
            for candidate in row.observation.candidates
            if candidate.selection == chosen
        )
        for row, chosen in zip(policies, sink.selected, strict=True)
    ]
    assert [
        canonical_json_bytes(row.observation) for row in values
    ] == recorder.expected_observations
    assert len(values) == sum(len(viewers) for _, _, viewers in recorder.boundaries)
    assert {row.perspective_team for row in values} == {"RED", "BLUE"}
    for row in policies:
        assert "value_target" not in row.model_dump()
        assert "terminal_winner" not in row.model_dump()
    for row in values:
        assert row.terminal_winner == winner_side
        assert row.value_target == (1 if row.perspective_team == winner_side else -1)
        assert "target" not in row.model_dump()
        assert "candidates" not in row.observation.model_dump()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("ending", ["max_steps", "exception", "invalid_winner", "missing_winner"])
def test_native_play_discards_both_heads_after_censoring_or_failure(tmp_path, monkeypatch, ending):
    _terminal_card(
        monkeypatch,
        winner="hero_missing" if ending == "invalid_winner" else "hero_wasp",
        missing_winner=ending == "missing_winner",
    )
    state = _game(next_actor=True)
    destination = tmp_path / "discarded.jsonl.zst"
    recorder = _ObservedRecorder(destination, game=_identity())
    agents, _, _ = _agents(recorder)

    def progress(_round, steps):
        if ending == "exception" and steps >= 4:
            raise RuntimeError("interrupted after actual boundary")

    def play():
        with recorder:
            return continue_game(
                state,
                agents,
                max_steps=4 if ending == "max_steps" else 30,
                boundary_observer=recorder,
                progress_callback=progress,
            )

    if ending == "max_steps":
        assert play().reason == "max_steps"
    elif ending == "exception":
        with pytest.raises(RuntimeError, match="interrupted after actual boundary"):
            play()
    else:
        with pytest.raises(ValueError, match="winner"):
            play()

    assert recorder.boundaries  # Both heads were provisional before the failure.
    assert recorder.expected_observations
    assert list(tmp_path.iterdir()) == []
