from __future__ import annotations

import math

import pytest

from automata.decision import DecisionDescriptor, DecisionSemanticRole
from automata.search import (
    LEGACY_SCHEDULE_ID,
    REQUEST_AWARE_SCHEDULE_V1_ID,
    REQUEST_AWARE_SCHEDULE_V2_ID,
)
from automata.search.config import SearchConfig, parse_learned_lh_search_config
from automata.search.contracts import LeafMode
from automata.search.scheduling import resolve_root_search_plan
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.domain.types import HeroID


def _state() -> GameState:
    return GameState.model_construct(execution_context={})


def _input_decision(
    request_type: InputRequestType,
    count: int,
    *,
    can_skip: bool = False,
) -> tuple[DecisionDescriptor, tuple[object, ...]]:
    options = [InputOption.from_value(f"choice-{index}") for index in range(count)]
    request = InputRequest(
        id="scheduled-root",
        request_type=request_type,
        player_id="hero_wasp",
        options=options,
        can_skip=can_skip,
    )
    return DecisionDescriptor("INPUT", request=request), tuple(
        [f"choice-{index}" for index in range(count)] + (["SKIP"] if can_skip else [])
    )


def _hex_decision(
    count: int,
    request_type: InputRequestType = InputRequestType.SELECT_HEX,
) -> tuple[DecisionDescriptor, tuple[object, ...]]:
    request = InputRequest(
        id="spatial-root",
        request_type=request_type,
        player_id="hero_wasp",
        options=[
            InputOption.from_value({"q": index, "r": 0, "s": -index}) for index in range(count)
        ],
    )
    return DecisionDescriptor("INPUT", request=request), tuple(
        ("hex", index, 0, -index) for index in range(count)
    )


def test_request_schedule_version_is_strict_and_part_of_search_identity() -> None:
    assert SearchConfig().request_schedule_version is None
    for invalid in (True, 1.0, 0, 3):
        with pytest.raises(ValueError, match="request_schedule_version"):
            SearchConfig(request_schedule_version=invalid)  # type: ignore[arg-type]

    for version in (1, 2):
        config, identity = parse_learned_lh_search_config({"request_schedule_version": version})
        assert config.request_schedule_version == version
        assert identity["request_schedule_version"] == version

    stable, stable_identity = parse_learned_lh_search_config(
        {"leaf_mode": "STABLE_TURN", "request_schedule_version": 2}
    )
    assert stable.leaf_mode is LeafMode.STABLE_TURN
    assert stable_identity["leaf_mode"] == "STABLE_TURN"


def test_request_schedule_v2_uses_stable_turn_only_for_actor_bound_resolution_inputs() -> None:
    decision, legal = _input_decision(InputRequestType.SELECT_OPTION, 3)
    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_wasp")
    config = SearchConfig(
        iterations=3,
        leaf_mode=LeafMode.BOUNDED_CONTINUATION,
        request_schedule_version=2,
    )

    plan = resolve_root_search_plan(state, decision, legal, config)
    actorless = resolve_root_search_plan(_state(), decision, legal, config)
    tie_breaker, tie_legal = _input_decision(InputRequestType.TIE_BREAKER, 2)
    tie = resolve_root_search_plan(state, tie_breaker, tie_legal, config)
    card = resolve_root_search_plan(state, DecisionDescriptor("CARD"), ("card-a", "card-b"), config)
    stable_card = resolve_root_search_plan(
        state,
        DecisionDescriptor("CARD"),
        ("card-a", "card-b"),
        SearchConfig(leaf_mode=LeafMode.STABLE_TURN, request_schedule_version=2),
    )

    assert plan.schedule_id == REQUEST_AWARE_SCHEDULE_V2_ID
    assert plan.effective_leaf_mode is LeafMode.STABLE_TURN
    assert actorless.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert tie.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert card.effective_leaf_mode is LeafMode.BOUNDED_CONTINUATION
    assert stable_card.effective_leaf_mode is LeafMode.IMMEDIATE


def test_request_schedule_v2_preserves_all_v1_budgets_and_coverage() -> None:
    state = _state()
    state.phase = GamePhase.RESOLUTION
    state.resolution_owner_id = HeroID("hero_wasp")
    roots = [
        _hex_decision(25),
        _input_decision(InputRequestType.DEFENSE_CARD, 2),
        _input_decision(InputRequestType.CHOOSE_ACTION, 3),
        _input_decision(InputRequestType.SELECT_OPTION, 4),
        (DecisionDescriptor("CARD"), ("card-a", "card-b")),
    ]

    for decision, legal in roots:
        v1 = resolve_root_search_plan(
            state,
            decision,
            legal,
            SearchConfig(iterations=3, request_schedule_version=1),
        )
        v2 = resolve_root_search_plan(
            state,
            decision,
            legal,
            SearchConfig(iterations=3, request_schedule_version=2),
        )
        assert (
            v2.requested_iterations,
            v2.effective_iterations,
            v2.root_coverage_target,
            v2.request_type,
            v2.semantic_role,
        ) == (
            v1.requested_iterations,
            v1.effective_iterations,
            v1.root_coverage_target,
            v1.request_type,
            v1.semantic_role,
        )


def test_explicit_stable_turn_falls_back_safely_without_an_enclosing_actor() -> None:
    decision, legal = _input_decision(InputRequestType.SELECT_OPTION, 2)
    actorless = resolve_root_search_plan(
        _state(), decision, legal, SearchConfig(leaf_mode=LeafMode.STABLE_TURN)
    )
    planning = resolve_root_search_plan(
        _state(),
        DecisionDescriptor("CARD"),
        ("card-a", "card-b"),
        SearchConfig(leaf_mode=LeafMode.STABLE_TURN),
    )

    assert actorless.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION
    assert planning.effective_leaf_mode is LeafMode.IMMEDIATE


def test_request_schedule_none_preserves_legacy_plan_exactly() -> None:
    decision, legal = _input_decision(InputRequestType.SELECT_OPTION, 5)
    config = SearchConfig(iterations=3, leaf_mode=LeafMode.BOUNDED_CONTINUATION)

    plan = resolve_root_search_plan(_state(), decision, legal, config)

    assert plan.schedule_id == LEGACY_SCHEDULE_ID
    assert plan.request_type == InputRequestType.SELECT_OPTION.value
    assert plan.semantic_role is DecisionSemanticRole.OPTION_SELECTION
    assert plan.requested_iterations == 3
    assert plan.effective_iterations == 3
    assert plan.root_coverage_target is None
    assert plan.effective_leaf_mode is LeafMode.BOUNDED_CONTINUATION


@pytest.mark.parametrize(
    ("candidate_count", "coverage", "iterations"),
    [(2, 2, 16), (15, 4, 16), (25, 5, 16), (51, 8, 16), (100, 10, 20), (130, 12, 24)],
)
def test_request_schedule_v1_spatial_formula(
    candidate_count: int, coverage: int, iterations: int
) -> None:
    decision, legal = _hex_decision(candidate_count)

    plan = resolve_root_search_plan(
        _state(),
        decision,
        legal,
        SearchConfig(
            iterations=2,
            leaf_mode=LeafMode.BOUNDED_CONTINUATION,
            request_schedule_version=1,
            adaptive_hex_root_schedule_version=1,
        ),
    )

    expected = min(candidate_count, 12, max(4, math.ceil(math.sqrt(candidate_count))))
    assert expected == coverage
    assert plan.schedule_id == REQUEST_AWARE_SCHEDULE_V1_ID
    assert plan.effective_iterations == iterations
    assert plan.root_coverage_target == coverage
    assert plan.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION


@pytest.mark.parametrize(("base_iterations", "effective_iterations"), [(1, 2), (3, 3), (20, 4)])
def test_request_schedule_v1_caps_binary_reactions_at_four(
    base_iterations: int, effective_iterations: int
) -> None:
    defense, legal = _input_decision(InputRequestType.DEFENSE_CARD, 2)

    plan = resolve_root_search_plan(
        _state(),
        defense,
        legal,
        SearchConfig(iterations=base_iterations, request_schedule_version=1),
    )

    assert plan.effective_iterations == effective_iterations
    assert plan.root_coverage_target == 2


def test_request_schedule_v1_includes_respawn_destination_in_hex_scheduling() -> None:
    respawn, legal = _hex_decision(15, InputRequestType.CHOOSE_RESPAWN_HEX)

    plan = resolve_root_search_plan(
        _state(),
        respawn,
        legal,
        SearchConfig(iterations=2, request_schedule_version=1),
    )

    assert plan.semantic_role is DecisionSemanticRole.RESPAWN_DESTINATION
    assert plan.effective_iterations == 16
    assert plan.root_coverage_target == 4
    assert plan.effective_leaf_mode is LeafMode.IMMEDIATE_ACTION


def test_request_schedule_v1_freezes_reaction_action_other_card_and_singleton_plans() -> None:
    defense, defense_legal = _input_decision(InputRequestType.DEFENSE_CARD, 2)
    action, action_legal = _input_decision(InputRequestType.CHOOSE_ACTION, 3)
    other, other_legal = _input_decision(InputRequestType.SELECT_OPTION, 3)
    singleton, singleton_legal = _input_decision(InputRequestType.CONFIRM_PASSIVE, 1)
    config = SearchConfig(
        iterations=1,
        leaf_mode=LeafMode.BOUNDED_CONTINUATION,
        request_schedule_version=1,
    )

    defense_plan = resolve_root_search_plan(_state(), defense, defense_legal, config)
    action_plan = resolve_root_search_plan(_state(), action, action_legal, config)
    other_plan = resolve_root_search_plan(_state(), other, other_legal, config)
    singleton_plan = resolve_root_search_plan(_state(), singleton, singleton_legal, config)
    card_plan = resolve_root_search_plan(
        _state(), DecisionDescriptor("CARD"), ("card-a", "card-b"), config
    )

    assert (
        defense_plan.effective_iterations,
        defense_plan.root_coverage_target,
        defense_plan.effective_leaf_mode,
    ) == (2, 2, LeafMode.IMMEDIATE_ACTION)
    assert (
        action_plan.effective_iterations,
        action_plan.root_coverage_target,
        action_plan.effective_leaf_mode,
    ) == (8, None, LeafMode.IMMEDIATE_ACTION)
    assert (
        other_plan.effective_iterations,
        other_plan.root_coverage_target,
        other_plan.effective_leaf_mode,
    ) == (1, None, LeafMode.IMMEDIATE_ACTION)
    assert singleton_plan.effective_iterations == 0
    assert singleton_plan.root_coverage_target is None
    assert (
        card_plan.effective_iterations,
        card_plan.root_coverage_target,
        card_plan.effective_leaf_mode,
    ) == (1, None, LeafMode.BOUNDED_CONTINUATION)
