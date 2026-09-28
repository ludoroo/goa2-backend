from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.decision import ActionBoundaryKind, DecisionDescriptor
from automata.models.contracts.inference import LearnedModelOutput
from automata.runtime.clone import clone_state
from automata.runtime.effects import register_all_effects
from automata.search.config import SearchConfig
from automata.search.contracts import (
    ComponentInferenceError,
    LeafEvaluation,
    LeafMode,
    PolicyScores,
    ScoreSemantics,
    SearchContext,
)
from automata.search.fallback import FallbackLeafEvaluator
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.ismcts import RootTarget, search
from automata.search.ismcts.engine import _ActionBoundary
from automata.search.learned import LearnedLeafEvaluator
from automata.search.node import Key, action_key
from automata.search.public_consequence import PublicConsequenceSnapshot
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import ActionType, GamePhase, StepType, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup
from goa2.engine.steps import ConfirmResolutionStep, LogMessageStep


def _state():
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp", "Xargatha"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )


def test_respawn_opportunity_shapes_only_an_executable_immediate_edge() -> None:
    state = _state()
    request = InputRequest(
        id="respawn",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_wasp",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)
    evaluator = HeuristicLeafEvaluator()

    prepared = evaluator.prepare_immediate_edge(context, state, decision, "RESPAWN")
    continuation = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="respawn-hex",
            request_type=InputRequestType.CHOOSE_RESPAWN_HEX,
            player_id="hero_wasp",
            options=[InputOption.from_value({"q": 0, "r": 0, "s": 0})],
        ),
    )
    shaped = evaluator.evaluate_immediate_edge(
        context.for_decision(continuation, owner_id=context.current_owner_id), state, prepared
    )
    ordinary = evaluator.evaluate(
        context.for_decision(continuation, owner_id=context.current_owner_id), state
    )

    assert evaluator.recipe_id == "public-consequence-v4"
    assert ordinary.value == 0.0
    assert shaped.value == pytest.approx(math.tanh(1.0))

    fizzled = evaluator.evaluate_immediate_edge(context, state, prepared)
    assert fizzled == evaluator.evaluate(context, state)


def test_teammate_respawn_completion_checks_request_owner_presence() -> None:
    state = _state()
    teammate_hex = state.get_position("hero_xargatha")
    assert teammate_hex is not None
    state.remove_entity("hero_xargatha")
    request = InputRequest(
        id="teammate-respawn",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_xargatha",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_xargatha", decision)
    evaluator = HeuristicLeafEvaluator()
    prepared = evaluator.prepare_immediate_edge(context, state, decision, "RESPAWN")
    respawned = clone_state(state)
    respawned.place_entity("hero_xargatha", teammate_hex)
    boundary = context.for_action_boundary()

    assert (
        evaluator.evaluate_immediate_edge(boundary, respawned, prepared).value
        > evaluator.evaluate(boundary, respawned).value
    )


def test_public_board_consequence_is_directional_and_hidden_card_invariant() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    start = state.get_position(wasp.id)
    assert start is not None
    lane = state.board.lane_of_zone(state.board.get_zone_for_hex(start) or "")
    assert lane is not None
    lane_zones = state.board.lanes[lane]
    current_zone_index = lane_zones.index(state.board.get_zone_for_hex(start) or "")
    forward_zone = state.board.zones[lane_zones[current_zone_index + 2]]
    destination = sorted(forward_zone.hexes, key=lambda item: (item.q, item.r, item.s))[0]

    request = InputRequest(
        id="hex",
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_wasp",
        options=[InputOption.from_value(destination)],
        can_skip=True,
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)
    evaluator = HeuristicLeafEvaluator()
    prepared = evaluator.prepare_immediate_edge(
        context, state, decision, action_key(destination.model_dump())
    )

    moved = clone_state(state)
    moved.move_unit(HeroID(wasp.id), destination)
    useful = evaluator.evaluate_immediate_edge(context, moved, prepared)
    skipped = evaluator.evaluate_immediate_edge(
        context,
        state,
        evaluator.prepare_immediate_edge(context, state, decision, "SKIP"),
    )
    harmful = clone_state(state)
    harmful.remove_entity("hero_wasp")
    harmed = evaluator.evaluate_immediate_edge(context, harmful, prepared)
    assert useful.value > skipped.value
    assert harmed == skipped

    hidden_variant = clone_state(state)
    enemy = hidden_variant.teams[TeamColor.BLUE].heroes[0]
    enemy.hand.reverse()
    hidden_prepared = evaluator.prepare_immediate_edge(
        context, hidden_variant, decision, action_key(destination.model_dump())
    )
    assert hidden_prepared == prepared


def test_objective_consequence_averages_multi_piece_heroes_across_lanes() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/across_the_river.json",
        ["Razzle"],
        ["Arien"],
        game_type="QUICK",
        seed=31,
    )
    piece_one = state.get_piece_ids("hero_razzle")[0]
    for entity_id in list(state.entity_locations):
        if str(entity_id) != piece_one:
            state.remove_entity(entity_id)
    one_piece_objective = PublicConsequenceSnapshot.capture(
        state, "hero_razzle", TeamColor.RED
    ).before.objectives

    first = state.get_positions("hero_razzle")[0]
    first_zone = state.board.get_zone_for_hex(first)
    assert first_zone is not None
    first_lane = state.lane_of_zone(first_zone)
    assert first_lane is not None
    second_lane = "lane_2" if first_lane != "lane_2" else "lane_1"
    second_zone = state.board.lanes[second_lane][-2]
    second = sorted(
        state.board.zones[second_zone].hexes, key=lambda item: (item.q, item.r, item.s)
    )[0]

    second_only = clone_state(state)
    second_only.remove_entity(piece_one)
    second_only.place_entity("hero_razzle_piece_2", second)
    second_piece_objective = PublicConsequenceSnapshot.capture(
        second_only, "hero_razzle", TeamColor.RED
    ).before.objectives

    state.place_entity("hero_razzle_piece_2", second)
    combined = PublicConsequenceSnapshot.capture(
        state, "hero_razzle", TeamColor.RED
    ).before.objectives

    assert combined == pytest.approx((one_piece_objective + second_piece_objective) / 2.0)


def test_terminal_authority_is_unchanged_by_public_consequence_shaping() -> None:
    state = _state()
    request = InputRequest(
        id="edge",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("x")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)
    evaluator = HeuristicLeafEvaluator()
    prepared = evaluator.prepare_immediate_edge(context, state, decision, "x")

    won = clone_state(state)
    won.winner = TeamColor.RED
    lost = clone_state(state)
    lost.winner = TeamColor.BLUE

    assert evaluator.evaluate_immediate_edge(context, won, prepared).value == 1.0
    assert evaluator.evaluate_immediate_edge(context, lost, prepared).value == -1.0


def test_material_value_includes_public_on_board_minions_and_is_zero_sum() -> None:
    state = _state()
    red_context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", DecisionDescriptor("CARD"))
    blue_context = SearchContext(
        "hero_arien", TeamColor.BLUE, "hero_arien", DecisionDescriptor("CARD")
    )
    evaluator = HeuristicLeafEvaluator()
    enemy_minion = min(
        (
            minion
            for minion in state.teams[TeamColor.BLUE].minions
            if state.has_board_presence(str(minion.id))
        ),
        key=lambda minion: minion.value,
    )
    before = evaluator.evaluate(red_context, state)

    state.remove_entity(str(enemy_minion.id))

    red_value = evaluator.evaluate(red_context, state).value
    blue_value = evaluator.evaluate(blue_context, state).value
    assert red_value > before.value
    assert red_value == pytest.approx(-blue_value)


def _choose_action_edge(action_type: ActionType = ActionType.SKILL):
    state = _state()
    request = InputRequest(
        id="action",
        request_type=InputRequestType.CHOOSE_ACTION,
        player_id="hero_wasp",
        options=[
            InputOption(
                id=action_type.value,
                text=action_type.value.title(),
                metadata={"type": action_type},
            ),
            InputOption(id="HOLD", text="Hold", metadata={"type": ActionType.HOLD}),
        ],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)
    continuation = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="target",
            request_type=InputRequestType.SELECT_HEX,
            player_id="hero_wasp",
            options=[InputOption.from_value({"q": 0, "r": 0, "s": 0})],
        ),
    )
    evaluator = HeuristicLeafEvaluator()
    prepared = evaluator.prepare_immediate_edge(context, state, decision, action_type.value)
    held = evaluator.evaluate_immediate_edge(
        context,
        state,
        evaluator.prepare_immediate_edge(context, state, decision, "HOLD"),
    )
    boundary = context.for_decision(
        continuation, owner_id=context.current_owner_id
    ).for_action_boundary()
    return state, context, continuation, evaluator, prepared, held, boundary


@pytest.mark.parametrize(
    "action_type",
    [ActionType.ATTACK, ActionType.SKILL, ActionType.MOVEMENT, ActionType.HOLD],
)
def test_equal_realized_no_change_actions_tie_regardless_of_label(
    action_type: ActionType,
) -> None:
    state, _, _, evaluator, prepared, held, boundary = _choose_action_edge(action_type)

    action = evaluator.evaluate_immediate_edge(boundary, state, prepared)
    interrupted = evaluator.evaluate_immediate_edge(
        boundary.for_action_boundary(ActionBoundaryKind.INTERRUPTED), state, prepared
    )

    assert action == held
    assert interrupted == held


def test_smallest_minion_kill_outvalues_no_change_action() -> None:
    state, context, _, evaluator, prepared, held, boundary = _choose_action_edge()
    state.execution_context["current_action_type"] = ActionType.SKILL
    unchanged = evaluator.evaluate_immediate_edge(boundary, state, prepared)
    killed = clone_state(state)
    enemy_minion = min(
        (
            minion
            for minion in killed.teams[TeamColor.BLUE].minions
            if killed.has_board_presence(str(minion.id))
        ),
        key=lambda minion: minion.value,
    )
    assert 0.25 * enemy_minion.value == 0.5
    killed.remove_entity(str(enemy_minion.id))
    decision = context.decision
    assert decision is not None
    no_proxy = evaluator.prepare_immediate_edge(context, state, decision, "HOLD")

    realized = evaluator.evaluate_immediate_edge(boundary, killed, prepared)
    no_change_logit = math.atanh(unchanged.value) - math.atanh(held.value)
    realized_logit = math.atanh(realized.value) - math.atanh(held.value)

    assert no_change_logit == pytest.approx(0.0)
    assert realized == evaluator.evaluate_immediate_edge(boundary, killed, no_proxy)
    assert realized_logit > 0.0
    assert realized_logit > no_change_logit


@pytest.mark.parametrize("kind", list(ActionBoundaryKind))
def test_choose_action_boundary_adds_normalized_public_material_delta(
    kind: ActionBoundaryKind,
) -> None:
    state, _, _, evaluator, prepared, _, boundary = _choose_action_edge()
    state.execution_context["current_action_type"] = ActionType.SKILL
    state.teams[TeamColor.RED].life_counters += 1
    typed_boundary = boundary.for_action_boundary(kind)

    action = evaluator.evaluate_immediate_edge(typed_boundary, state, prepared)

    ordinary = evaluator.evaluate(typed_boundary, state)
    assert action.value > ordinary.value


def test_choose_action_boundary_keeps_visible_harm_below_hold() -> None:
    state, _, _, evaluator, prepared, held, boundary = _choose_action_edge()
    state.teams[TeamColor.RED].life_counters -= 1

    action = evaluator.evaluate_immediate_edge(boundary, state, prepared)

    assert action.value < held.value


def test_choose_action_legacy_immediate_no_change_has_no_label_proxy() -> None:
    state, context, continuation, evaluator, prepared, held, _ = _choose_action_edge()
    state.execution_stack = [ConfirmResolutionStep(hero_id="hero_wasp")]

    action = evaluator.evaluate_immediate_edge(
        context.for_decision(continuation, owner_id=context.current_owner_id), state, prepared
    )

    assert action == held


def test_unknown_action_at_boundary_without_same_action_evidence_is_neutral() -> None:
    state, _, _, evaluator, prepared, _, boundary = _choose_action_edge()

    action = evaluator.evaluate_immediate_edge(boundary, state, prepared)

    assert action == evaluator.evaluate(boundary, state)


class _FlatPolicy:
    def score(
        self, context: SearchContext, state: object, legal_actions: Sequence[Key]
    ) -> PolicyScores:
        actions = tuple(legal_actions)
        return PolicyScores(actions, (0.0,) * len(actions), ScoreSemantics.LOGITS)


class _RespawnSimulator:
    request: InputRequest
    respawn_hex: object

    def __init__(self, state, our_team, environment_policy, *, owned_hero_ids, cfg=None):
        self.state = state
        self.our_team = our_team

    def advance(self):
        return DecisionDescriptor("INPUT", request=self.request)

    def advance_to_root(self, target):
        return self.advance()

    def apply_ours(self, decision, key):
        if decision.request.request_type is InputRequestType.CHOOSE_RESPAWN_HEX:
            self.state.move_unit(HeroID("hero_wasp"), self.respawn_hex)
            request_type = InputRequestType.SELECT_OPTION
        else:
            request_type = (
                InputRequestType.CHOOSE_RESPAWN_HEX
                if key == "RESPAWN"
                else InputRequestType.SELECT_OPTION
            )
        request = InputRequest(
            id=f"after-{key}",
            request_type=request_type,
            player_id="hero_wasp",
            options=[InputOption.from_value(self.respawn_hex)],
        )
        return DecisionDescriptor("INPUT", request=request)


def test_flat_prior_real_search_prefers_executable_respawn_without_rewriting_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from automata.search.ismcts import engine

    state = _state()
    request = InputRequest(
        id="respawn-root",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_wasp",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )
    respawn_hex = state.get_position("hero_wasp")
    assert respawn_hex is not None
    state.remove_entity("hero_wasp")
    _RespawnSimulator.request = request
    _RespawnSimulator.respawn_hex = respawn_hex
    monkeypatch.setattr(engine, "_Simulator", _RespawnSimulator)
    monkeypatch.setattr(engine, "determinize", lambda state, viewer_id, rng: clone_state(state))
    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
        request=request,
    )

    result = search(
        state,
        TeamColor.RED,
        ("RESPAWN", "PASS"),
        HeuristicAgent(0),
        SearchConfig(iterations=8, leaf_mode=LeafMode.IMMEDIATE, seed=4),
        _FlatPolicy(),
        root_target=target,
    )

    diagnostics = {item.action: item for item in result.root_action_diagnostics}
    assert result.best_key == "RESPAWN"
    assert diagnostics["RESPAWN"].mean_value > diagnostics["PASS"].mean_value
    assert diagnostics["RESPAWN"].visits >= diagnostics["PASS"].visits
    assert tuple(
        item.prior_probability for item in result.root_action_diagnostics
    ) == pytest.approx((0.5, 0.5))
    respawn_mass = diagnostics["RESPAWN"].visits ** 2 / sum(
        item.visits**2 for item in diagnostics.values()
    )
    assert respawn_mass >= 0.5

    repeated = search(
        state,
        TeamColor.RED,
        ("RESPAWN", "PASS"),
        HeuristicAgent(0),
        SearchConfig(iterations=8, leaf_mode=LeafMode.IMMEDIATE, seed=4),
        _FlatPolicy(),
        root_target=target,
    )
    assert repeated.best_key == result.best_key
    assert repeated.root_action_diagnostics == result.root_action_diagnostics


def test_immediate_edge_is_excluded_from_terminals_and_bounded_continuation() -> None:
    from automata.search.ismcts.engine import _rollout

    state = _state()
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", DecisionDescriptor("CARD"))

    class Evaluator:
        def evaluate(self, context, state):
            return LeafEvaluation(value=-0.25)

        def prepare_immediate_edge(self, context, state, decision, action):
            return "edge"

        def evaluate_immediate_edge(self, context, state, prepared):
            return LeafEvaluation(value=1.0)

    class Sim:
        our_team = TeamColor.RED

        def __init__(self):
            self.state = state

    evaluator = Evaluator()
    terminal = _rollout(
        Sim(),  # type: ignore[arg-type]
        DecisionDescriptor("OVER", winner="RED"),
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE),
        HeuristicAgent(0),
        evaluator,
        context,
        immediate_edge="edge",
    )
    terminal_loss = _rollout(
        Sim(),  # type: ignore[arg-type]
        DecisionDescriptor("OVER", winner="BLUE"),
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE),
        HeuristicAgent(0),
        evaluator,
        context,
        immediate_edge="edge",
    )
    bounded = _rollout(
        Sim(),  # type: ignore[arg-type]
        DecisionDescriptor(
            "INPUT",
            request=InputRequest(
                id="leaf",
                request_type=InputRequestType.SELECT_OPTION,
                player_id="hero_wasp",
                options=[InputOption.from_value("x")],
            ),
        ),
        SearchConfig(
            cutoff_limit=0,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
        ),
        HeuristicAgent(0),
        evaluator,
        context,
        immediate_edge="edge",
    )
    assert terminal == 1.0
    assert terminal_loss == 0.0
    assert bounded == pytest.approx(0.375)


def test_immediate_action_continues_owned_inputs_until_confirm_boundary() -> None:
    from automata.search.ismcts.engine import _rollout

    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.current_actor_id = "hero_wasp"
    state.execution_stack = [LogMessageStep(message="continue")]
    first = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="target",
            request_type=InputRequestType.SELECT_OPTION,
            player_id="hero_wasp",
            options=[InputOption.from_value("target")],
        ),
    )
    confirm = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="confirm",
            request_type=InputRequestType.CHOOSE_ACTION,
            player_id="hero_wasp",
            options=[InputOption.from_value("CONFIRM")],
        ),
    )

    class Sim:
        our_team = TeamColor.RED

        def __init__(self) -> None:
            self.state = state
            self.advances = 0

        def apply_ours(self, current, key, *, action_boundary=None):
            assert current is first
            assert key == "target"
            return self.advance(None, action_boundary=action_boundary)

        def advance(self, response, *, action_boundary=None):
            assert action_boundary == _ActionBoundary("hero_wasp", "hero_wasp", state.round)
            self.advances += 1
            self.state.execution_stack = [ConfirmResolutionStep(hero_id="hero_wasp")]
            return confirm

        def progression_error(self, *args, **kwargs):
            return AssertionError((args, kwargs))

    class Continuation:
        def choose_input(self, state, request):
            return "target"

    class Evaluator:
        immediate_edge_enabled = True

        def evaluate(self, context, state):
            raise AssertionError("prepared edge must be evaluated")

        def prepare_immediate_edge(self, context, state, decision, action):
            return "edge"

        def evaluate_immediate_edge(self, context, state, prepared):
            assert prepared == "edge"
            assert context.decision == confirm
            return LeafEvaluation(value=1.0)

    sim = Sim()
    reward = _rollout(
        sim,  # type: ignore[arg-type]
        first,
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
        Continuation(),  # type: ignore[arg-type]
        Evaluator(),
        SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", first),
        immediate_edge="edge",
        action_boundary=_ActionBoundary("hero_wasp", "hero_wasp", state.round),
    )

    assert reward == 1.0
    assert sim.advances == 1


@pytest.mark.parametrize(
    ("player_id", "step_type"),
    [("team:RED", StepType.SELECT), ("hero_wasp", StepType.CONFIRM_RESOLUTION)],
)
def test_immediate_action_stops_at_foreign_or_action_boundary(
    player_id: str, step_type: StepType
) -> None:
    from automata.search.ismcts.engine import _rollout

    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.current_actor_id = "hero_wasp"
    state.execution_stack = [
        (
            ConfirmResolutionStep(hero_id="hero_wasp")
            if step_type is StepType.CONFIRM_RESOLUTION
            else LogMessageStep(message="continue")
        )
    ]
    decision = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="stop",
            request_type=InputRequestType.SELECT_OPTION,
            player_id=player_id,
            options=[InputOption.from_value("x")],
        ),
    )

    class Sim:
        our_team = TeamColor.RED

        def __init__(self) -> None:
            self.state = state

        def advance(self, response):
            raise AssertionError("boundary input must not be continued")

    class Continuation:
        def choose_input(self, state, request):
            raise AssertionError("boundary input must not be selected")

    reward = _rollout(
        Sim(),  # type: ignore[arg-type]
        decision,
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
        Continuation(),  # type: ignore[arg-type]
        HeuristicLeafEvaluator(),
        SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision),
        action_boundary=_ActionBoundary("hero_wasp", "hero_wasp", state.round),
    )
    assert 0.0 <= reward <= 1.0


def test_immediate_action_boundary_retains_an_encodable_learned_value_context() -> None:
    from automata.search.ismcts.engine import _rollout

    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.current_actor_id = "hero_wasp"
    state.execution_stack = [ConfirmResolutionStep(hero_id="hero_wasp")]
    root = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="root-action",
            request_type=InputRequestType.SELECT_OPTION,
            player_id="hero_wasp",
            options=[InputOption.from_value("A"), InputOption.from_value("B")],
        ),
    )

    class Runtime:
        def evaluate(self, observation):
            assert observation.decision_kind == "INPUT"
            assert tuple(candidate.selection for candidate in observation.candidates) == (
                "CONFIRM",
            )
            candidate_ids = tuple(candidate.candidate_id for candidate in observation.candidates)
            return LearnedModelOutput(
                candidate_ids=candidate_ids,
                policy_logits=(0.0,),
                probabilities=(1.0,),
                value=0.25,
            )

    class Sim:
        our_team = TeamColor.RED

        def __init__(self) -> None:
            self.state = state

    reward = _rollout(
        Sim(),  # type: ignore[arg-type]
        DecisionDescriptor("BOUNDARY", action_boundary_kind=ActionBoundaryKind.COMPLETE),
        SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
        HeuristicAgent(0),
        LearnedLeafEvaluator(Runtime()),  # type: ignore[arg-type]
        SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", root),
        action_boundary=_ActionBoundary("hero_wasp", "hero_wasp", state.round),
    )

    assert reward == pytest.approx(0.625)


def test_immediate_action_repeated_decision_fails_closed() -> None:
    from automata.search.ismcts.engine import _rollout

    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.current_actor_id = "hero_wasp"
    state.execution_stack = [LogMessageStep(message="continue")]
    decision = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="repeat",
            request_type=InputRequestType.SELECT_OPTION,
            player_id="hero_wasp",
            options=[InputOption.from_value("x")],
        ),
    )

    class Sim:
        our_team = TeamColor.RED

        def __init__(self) -> None:
            self.state = state

        def apply_ours(self, current, key, *, action_boundary=None):
            assert current is decision
            assert key == "x"
            return self.advance(None, action_boundary=action_boundary)

        def advance(self, response, *, action_boundary=None):
            assert action_boundary == _ActionBoundary("hero_wasp", "hero_wasp", state.round)
            return decision

        def progression_error(self, reason, decision, **kwargs):
            return RuntimeError(reason)

    class Continuation:
        def choose_input(self, state, request):
            return "x"

    with pytest.raises(RuntimeError, match="repeated same-action decision"):
        _rollout(
            Sim(),  # type: ignore[arg-type]
            decision,
            SearchConfig(leaf_mode=LeafMode.IMMEDIATE_ACTION),
            Continuation(),  # type: ignore[arg-type]
            HeuristicLeafEvaluator(),
            SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision),
            action_boundary=_ActionBoundary("hero_wasp", "hero_wasp", state.round),
        )


def test_fallback_preparation_recovers_and_produces_a_usable_fallback_token() -> None:
    state = _state()
    request = InputRequest(
        id="respawn",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_wasp",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    continuation = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="hex",
            request_type=InputRequestType.CHOOSE_RESPAWN_HEX,
            player_id="hero_wasp",
            options=[InputOption.from_value({"q": 0, "r": 0, "s": 0})],
        ),
    )
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)

    class BrokenPrimary:
        immediate_edge_enabled = True
        contextual_root_coverage_enabled = True

        def evaluate(self, context, state):
            raise AssertionError("failed preparation must pin this edge to fallback")

        def prepare_immediate_edge(self, context, state, decision, action):
            raise ComponentInferenceError("preparation failed")

        def evaluate_immediate_edge(self, context, state, prepared):
            raise AssertionError("failed preparation must pin this edge to fallback")

    fallback = HeuristicLeafEvaluator()
    evaluator = FallbackLeafEvaluator(BrokenPrimary(), fallback)
    prepared = evaluator.prepare_immediate_edge(context, state, decision, "RESPAWN")
    result = evaluator.evaluate_immediate_edge(
        context.for_decision(continuation, owner_id=context.current_owner_id), state, prepared
    )

    assert result.value > fallback.evaluate(context, state).value


def test_learned_leaf_success_is_unshaped_but_recoverable_failure_uses_edge_fallback() -> None:
    state = _state()
    request = InputRequest(
        id="respawn",
        request_type=InputRequestType.CHOOSE_RESPAWN,
        player_id="hero_wasp",
        options=[InputOption.from_value("RESPAWN"), InputOption.from_value("PASS")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    continuation = DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id="hex",
            request_type=InputRequestType.CHOOSE_RESPAWN_HEX,
            player_id="hero_wasp",
            options=[InputOption.from_value({"q": 0, "r": 0, "s": 0})],
        ),
    )
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)

    class Learned:
        def __init__(self, fail: bool):
            self.fail = fail

        def evaluate(self, context, state):
            if self.fail:
                raise ComponentInferenceError("failed")
            return LeafEvaluation(value=0.25)

    successful = FallbackLeafEvaluator(Learned(False), HeuristicLeafEvaluator())
    prepared = successful.prepare_immediate_edge(context, state, decision, "RESPAWN")
    assert successful.evaluate_immediate_edge(
        context.for_decision(continuation, owner_id=context.current_owner_id), state, prepared
    ) == LeafEvaluation(value=0.25)

    failed = FallbackLeafEvaluator(Learned(True), HeuristicLeafEvaluator())
    prepared = failed.prepare_immediate_edge(context, state, decision, "RESPAWN")
    assert (
        failed.evaluate_immediate_edge(
            context.for_decision(continuation, owner_id=context.current_owner_id), state, prepared
        ).value
        > HeuristicLeafEvaluator().evaluate(context, state).value
    )
