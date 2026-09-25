"""Shared recognition of public decisions with one contextual no-op."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from automata.decision import DecisionDescriptor
from goa2.domain.input import InputRequest, InputRequestType, selection_value

from .node import Key, action_key


class ContextualNoopKind(StrEnum):
    RESPAWN_PASS = "RESPAWN_PASS"
    ACTION_HOLD = "ACTION_HOLD"
    NARROW_HEX_SKIP = "NARROW_HEX_SKIP"
    BROAD_HEX_SKIP = "BROAD_HEX_SKIP"


@dataclass(frozen=True, slots=True)
class ContextualNoopShape:
    kind: ContextualNoopKind
    noop_key: Key


def request_action_keys(request: InputRequest) -> tuple[Key, ...]:
    """Return the canonical unique keys represented by a surfaced request."""
    keys = dict.fromkeys(action_key(selection_value(option)) for option in request.options)
    if request.can_skip:
        keys["SKIP"] = None
    return tuple(keys)


def _type_name(value: object) -> str | None:
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else None


def _is_hold_option(request: InputRequest, key: Key, index: int) -> bool:
    option = request.options[index]
    return _type_name(option.metadata.get("type")) == "HOLD" or key == "HOLD"


def contextual_noop_shape(
    decision: DecisionDescriptor,
    legal: Sequence[Key],
) -> ContextualNoopShape | None:
    """Recognize only approved, unambiguous contextual no-op shapes."""
    request = decision.request
    if request is None:
        return None

    if (
        request.request_type is InputRequestType.CHOOSE_RESPAWN
        and len(legal) == 2
        and set(legal) == {"RESPAWN", "PASS"}
    ):
        return ContextualNoopShape(ContextualNoopKind.RESPAWN_PASS, "PASS")

    if request.request_type is InputRequestType.CHOOSE_ACTION:
        option_keys = [action_key(selection_value(option)) for option in request.options]
        hold_keys = [
            key for index, key in enumerate(option_keys) if _is_hold_option(request, key, index)
        ]
        if (
            len(hold_keys) == 1
            and hold_keys[0] in legal
            and any(key != hold_keys[0] for key in legal)
        ):
            return ContextualNoopShape(ContextualNoopKind.ACTION_HOLD, hold_keys[0])

    is_hex_or_skip = all(
        key == "SKIP" or (isinstance(key, tuple) and len(key) == 4 and key[0] == "hex")
        for key in legal
    )
    if (
        len(legal) >= 2
        and request.can_skip
        and "SKIP" in legal
        and is_hex_or_skip
        and any(isinstance(key, tuple) and key[0] == "hex" for key in legal)
    ):
        kind = (
            ContextualNoopKind.BROAD_HEX_SKIP
            if len(legal) > 8
            else ContextualNoopKind.NARROW_HEX_SKIP
        )
        return ContextualNoopShape(kind, "SKIP")

    return None


__all__ = [
    "ContextualNoopKind",
    "ContextualNoopShape",
    "contextual_noop_shape",
    "request_action_keys",
]
