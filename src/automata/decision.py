"""Neutral descriptor for an engine decision consumed by search and encoders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import ActionType
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState


class ActionBoundaryKind(StrEnum):
    """Whether an IMMEDIATE_ACTION cutoff completed or interrupted the action."""

    COMPLETE = "COMPLETE"
    INTERRUPTED = "INTERRUPTED"


class DecisionSemanticRole(StrEnum):
    """Stable model-facing role derived from typed engine decision facts."""

    PLANNING = "PLANNING"
    ACTION_CHOICE = "ACTION_CHOICE"
    MOVEMENT_DESTINATION = "MOVEMENT_DESTINATION"
    # Approximate: a generic SELECT_UNIT while the typed current action is ATTACK.
    # It identifies attack-related unit selection, not proven downstream consequences.
    ATTACK_TARGET = "ATTACK_TARGET"
    DEFENSE_REACTION = "DEFENSE_REACTION"
    PASSIVE_REACTION = "PASSIVE_REACTION"
    RESPAWN_CHOICE = "RESPAWN_CHOICE"
    RESPAWN_DESTINATION = "RESPAWN_DESTINATION"
    ACTOR_CHOICE = "ACTOR_CHOICE"
    UPGRADE_CHOICE = "UPGRADE_CHOICE"
    CARD_SELECTION = "CARD_SELECTION"
    UNIT_SELECTION = "UNIT_SELECTION"
    SPATIAL_SELECTION = "SPATIAL_SELECTION"
    NUMBER_SELECTION = "NUMBER_SELECTION"
    OPTION_SELECTION = "OPTION_SELECTION"


@dataclass(frozen=True, slots=True)
class DecisionSemantics:
    input_request_type: str | None
    can_skip: bool
    semantic_role: DecisionSemanticRole


_INPUT_SEMANTIC_ROLES: dict[InputRequestType, DecisionSemanticRole] = {
    InputRequestType.NONE: DecisionSemanticRole.OPTION_SELECTION,
    InputRequestType.ACTION_CHOICE: DecisionSemanticRole.ACTION_CHOICE,
    InputRequestType.MOVEMENT_HEX: DecisionSemanticRole.MOVEMENT_DESTINATION,
    InputRequestType.DEFENSE_CARD: DecisionSemanticRole.DEFENSE_REACTION,
    InputRequestType.TIE_BREAKER: DecisionSemanticRole.ACTOR_CHOICE,
    InputRequestType.SELECT_ALLY: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.FAST_TRAVEL_DESTINATION: DecisionSemanticRole.MOVEMENT_DESTINATION,
    InputRequestType.SELECT_ENEMY: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.UPGRADE_CHOICE: DecisionSemanticRole.UPGRADE_CHOICE,
    InputRequestType.SELECT_UNIT: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.SELECT_UNIT_OR_TOKEN: DecisionSemanticRole.UNIT_SELECTION,
    InputRequestType.SELECT_HEX: DecisionSemanticRole.SPATIAL_SELECTION,
    InputRequestType.SELECT_CARD: DecisionSemanticRole.CARD_SELECTION,
    InputRequestType.SELECT_NUMBER: DecisionSemanticRole.NUMBER_SELECTION,
    InputRequestType.CHOOSE_ACTION: DecisionSemanticRole.ACTION_CHOICE,
    InputRequestType.SELECT_CARD_OR_PASS: DecisionSemanticRole.DEFENSE_REACTION,
    InputRequestType.SELECT_OPTION: DecisionSemanticRole.OPTION_SELECTION,
    InputRequestType.CHOOSE_ACTOR: DecisionSemanticRole.ACTOR_CHOICE,
    InputRequestType.CHOOSE_RESPAWN: DecisionSemanticRole.RESPAWN_CHOICE,
    InputRequestType.CHOOSE_RESPAWN_HEX: DecisionSemanticRole.RESPAWN_DESTINATION,
    InputRequestType.UPGRADE_PHASE: DecisionSemanticRole.UPGRADE_CHOICE,
    InputRequestType.CONFIRM_PASSIVE: DecisionSemanticRole.PASSIVE_REACTION,
}

if set(_INPUT_SEMANTIC_ROLES) != set(InputRequestType):
    raise RuntimeError("InputRequestType semantic-role mapping is incomplete")


@dataclass
class DecisionDescriptor:
    kind: str
    hero: Hero | None = None
    request: InputRequest | None = None
    winner: str | None = None
    can_finish_planning: bool = False
    action_boundary_kind: ActionBoundaryKind | None = None

    @property
    def is_terminal(self) -> bool:
        return self.kind == "OVER"


def classify_decision(state: GameState, decision: DecisionDescriptor) -> DecisionSemantics:
    """Classify a branchable decision without reading private/free-form request data."""
    if decision.kind == "CARD":
        return DecisionSemantics(None, False, DecisionSemanticRole.PLANNING)
    if decision.kind != "INPUT":
        raise ValueError(f"unsupported decision kind: {decision.kind!r}")
    if decision.request is None:
        raise ValueError("INPUT decision requires a request")

    request_type = decision.request.request_type
    role = _INPUT_SEMANTIC_ROLES[request_type]
    if (
        request_type is InputRequestType.SELECT_UNIT
        and state.execution_context.get("current_action_type") == ActionType.ATTACK
    ):
        role = DecisionSemanticRole.ATTACK_TARGET
    return DecisionSemantics(request_type.value, decision.request.can_skip, role)


__all__ = [
    "ActionBoundaryKind",
    "DecisionDescriptor",
    "DecisionSemanticRole",
    "DecisionSemantics",
    "classify_decision",
]
