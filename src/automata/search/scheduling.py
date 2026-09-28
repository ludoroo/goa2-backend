"""Immutable root-search plans derived from a strictly validated decision.

The resolver never changes legality or mutates :class:`SearchConfig`. Callers
must invoke it only after root validation has surfaced the exact decision and
canonical legal candidate set.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from automata.decision import (
    DecisionDescriptor,
    DecisionSemanticRole,
    classify_decision,
)
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState

from .config import SearchConfig
from .contracts import LeafMode

LEGACY_SCHEDULE_ID = "legacy"
REQUEST_AWARE_SCHEDULE_V1_ID = "request-aware-v1"
REQUEST_AWARE_SCHEDULE_V2_ID = "request-aware-v2"

_STABLE_TURN_ROLES = frozenset(
    {
        DecisionSemanticRole.ACTION_CHOICE,
        DecisionSemanticRole.MOVEMENT_DESTINATION,
        DecisionSemanticRole.ATTACK_TARGET,
        DecisionSemanticRole.DEFENSE_REACTION,
        DecisionSemanticRole.PASSIVE_REACTION,
        DecisionSemanticRole.RESPAWN_CHOICE,
        DecisionSemanticRole.RESPAWN_DESTINATION,
        DecisionSemanticRole.CARD_SELECTION,
        DecisionSemanticRole.UNIT_SELECTION,
        DecisionSemanticRole.SPATIAL_SELECTION,
        DecisionSemanticRole.NUMBER_SELECTION,
        DecisionSemanticRole.OPTION_SELECTION,
    }
)


@dataclass(frozen=True, slots=True)
class RootSearchPlan:
    """Complete bounded schedule and typed provenance for one validated root."""

    schedule_id: str
    effective_leaf_mode: LeafMode
    request_type: str | None
    semantic_role: DecisionSemanticRole
    requested_iterations: int
    effective_iterations: int
    root_coverage_target: int | None


RootSearchPlanObserver = Callable[[RootSearchPlan], None]
_ROOT_PLAN_OBSERVER: ContextVar[RootSearchPlanObserver | None] = ContextVar(
    "root_search_plan_observer", default=None
)


@contextmanager
def observe_root_search_plans(observer: RootSearchPlanObserver) -> Iterator[None]:
    """Observe plans resolved by searches in this bounded execution context."""
    token = _ROOT_PLAN_OBSERVER.set(observer)
    try:
        yield
    finally:
        _ROOT_PLAN_OBSERVER.reset(token)


def _publish_root_search_plan(plan: RootSearchPlan) -> None:
    """Publish validated plan metadata without reconstructing caller state."""
    observer = _ROOT_PLAN_OBSERVER.get()
    if observer is not None:
        observer(plan)


def _is_hex_root(legal_candidates: Sequence[object]) -> bool:
    has_hex = False
    for key in legal_candidates:
        if key == "SKIP":
            continue
        if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "hex"):
            return False
        has_hex = True
    return has_hex


def _coverage_target(candidate_count: int) -> int:
    return min(candidate_count, 12, max(4, math.ceil(math.sqrt(candidate_count))))


def _stable_turn_eligible(
    state: GameState,
    decision: DecisionDescriptor,
    semantic_role: DecisionSemanticRole,
) -> bool:
    return bool(
        decision.kind == "INPUT"
        and state.phase is GamePhase.RESOLUTION
        and state.resolution_owner_id is not None
        and semantic_role in _STABLE_TURN_ROLES
    )


def _legacy_plan(
    config: SearchConfig,
    *,
    candidate_count: int,
    is_hex_root: bool,
    request_type: str | None,
    semantic_role: DecisionSemanticRole,
) -> RootSearchPlan:
    effective_iterations = config.iterations
    coverage_target: int | None = None
    if config.adaptive_hex_root_schedule_version == 1 and candidate_count > 8 and is_hex_root:
        coverage_target = _coverage_target(candidate_count)
        effective_iterations = max(effective_iterations, 2 * coverage_target)
    return RootSearchPlan(
        schedule_id=LEGACY_SCHEDULE_ID,
        effective_leaf_mode=config.leaf_mode,
        request_type=request_type,
        semantic_role=semantic_role,
        requested_iterations=config.iterations,
        effective_iterations=effective_iterations,
        root_coverage_target=coverage_target,
    )


def resolve_root_search_plan(
    state: GameState,
    decision: DecisionDescriptor,
    legal_candidates: Sequence[object],
    config: SearchConfig,
) -> RootSearchPlan:
    """Resolve the frozen schedule for an already validated root.

    Candidate order and membership are read-only. A singleton always takes the
    validated zero-simulation fast path. With no request schedule configured,
    this reproduces the legacy adaptive-HEX scheduling rules exactly.
    """
    candidate_count = len(legal_candidates)
    if candidate_count == 0:
        raise ValueError("root search plan requires at least one legal candidate")

    semantics = classify_decision(state, decision)
    is_hex_root = _is_hex_root(legal_candidates)
    if config.request_schedule_version is None:
        plan = _legacy_plan(
            config,
            candidate_count=candidate_count,
            is_hex_root=is_hex_root,
            request_type=semantics.input_request_type,
            semantic_role=semantics.semantic_role,
        )
        if config.leaf_mode is LeafMode.STABLE_TURN and not _stable_turn_eligible(
            state, decision, semantics.semantic_role
        ):
            plan = RootSearchPlan(
                schedule_id=plan.schedule_id,
                effective_leaf_mode=(
                    LeafMode.IMMEDIATE_ACTION if decision.kind == "INPUT" else LeafMode.IMMEDIATE
                ),
                request_type=plan.request_type,
                semantic_role=plan.semantic_role,
                requested_iterations=plan.requested_iterations,
                effective_iterations=plan.effective_iterations,
                root_coverage_target=plan.root_coverage_target,
            )
    else:
        effective_iterations = config.iterations
        coverage_target: int | None = None
        effective_leaf_mode = (
            LeafMode.IMMEDIATE
            if config.leaf_mode is LeafMode.STABLE_TURN and decision.kind != "INPUT"
            else config.leaf_mode
        )

        if decision.kind == "INPUT":
            stable_turn_eligible = bool(
                config.request_schedule_version == 2
                and _stable_turn_eligible(state, decision, semantics.semantic_role)
            )
            effective_leaf_mode = (
                LeafMode.STABLE_TURN if stable_turn_eligible else LeafMode.IMMEDIATE_ACTION
            )
            if candidate_count == 2 and semantics.semantic_role in {
                DecisionSemanticRole.DEFENSE_REACTION,
                DecisionSemanticRole.PASSIVE_REACTION,
            }:
                effective_iterations = max(2, min(config.iterations, 4))
                coverage_target = 2
            elif (
                semantics.semantic_role
                in {
                    DecisionSemanticRole.MOVEMENT_DESTINATION,
                    DecisionSemanticRole.RESPAWN_DESTINATION,
                    DecisionSemanticRole.SPATIAL_SELECTION,
                }
                and is_hex_root
            ):
                coverage_target = _coverage_target(candidate_count)
                effective_iterations = max(
                    config.iterations,
                    16,
                    2 * coverage_target,
                )
            elif semantics.semantic_role is DecisionSemanticRole.ACTION_CHOICE:
                effective_iterations = max(config.iterations, 8)

        schedule_id = (
            REQUEST_AWARE_SCHEDULE_V2_ID
            if config.request_schedule_version == 2
            else REQUEST_AWARE_SCHEDULE_V1_ID
        )
        plan = RootSearchPlan(
            schedule_id=schedule_id,
            effective_leaf_mode=effective_leaf_mode,
            request_type=semantics.input_request_type,
            semantic_role=semantics.semantic_role,
            requested_iterations=config.iterations,
            effective_iterations=effective_iterations,
            root_coverage_target=coverage_target,
        )

    if candidate_count == 1:
        return RootSearchPlan(
            schedule_id=plan.schedule_id,
            effective_leaf_mode=plan.effective_leaf_mode,
            request_type=plan.request_type,
            semantic_role=plan.semantic_role,
            requested_iterations=plan.requested_iterations,
            effective_iterations=0,
            root_coverage_target=None,
        )
    return plan


__all__ = [
    "LEGACY_SCHEDULE_ID",
    "REQUEST_AWARE_SCHEDULE_V1_ID",
    "REQUEST_AWARE_SCHEDULE_V2_ID",
    "RootSearchPlan",
    "RootSearchPlanObserver",
    "observe_root_search_plans",
    "resolve_root_search_plan",
]
