"""Stable candidate identities and decision-observation contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from .observation import LearnedObservation, _Contract


class FinishCandidateID(_Contract[Literal[1]]):
    kind: Literal["FINISH"] = "FINISH"


class SkipCandidateID(_Contract[Literal[1]]):
    kind: Literal["SKIP"] = "SKIP"


class CardCandidateID(_Contract[Literal[1]]):
    kind: Literal["CARD"] = "CARD"
    card_id: str = Field(min_length=1)


class UnitCandidateID(_Contract[Literal[1]]):
    kind: Literal["UNIT"] = "UNIT"
    unit_id: str = Field(min_length=1)


class HexCandidateID(_Contract[Literal[1]]):
    kind: Literal["HEX"] = "HEX"
    q: int
    r: int
    s: int

    @model_validator(mode="after")
    def _valid_cube_coordinate(self) -> HexCandidateID:
        if self.q + self.r + self.s != 0:
            raise ValueError("hex candidate coordinates must sum to zero")
        return self


class NumberCandidateID(_Contract[Literal[1]]):
    kind: Literal["NUMBER"] = "NUMBER"
    value: int | float

    @model_validator(mode="after")
    def _finite_value(self) -> NumberCandidateID:
        if not math.isfinite(float(self.value)):
            raise ValueError("number candidate value must be finite")
        return self


class OptionCandidateID(_Contract[Literal[1]]):
    kind: Literal["OPTION"] = "OPTION"
    option_id: str = Field(min_length=1)


class ActionCandidateID(_Contract[Literal[1]]):
    kind: Literal["ACTION"] = "ACTION"
    action_id: str = Field(min_length=1)


class EntityCandidateID(_Contract[Literal[1]]):
    """Identity for a non-unit graph entity without exposing engine identity."""

    kind: Literal["ENTITY"] = "ENTITY"
    entity_ref: str = Field(min_length=1)


CandidateID = Annotated[
    FinishCandidateID
    | SkipCandidateID
    | CardCandidateID
    | UnitCandidateID
    | HexCandidateID
    | NumberCandidateID
    | OptionCandidateID
    | ActionCandidateID
    | EntityCandidateID,
    Field(discriminator="kind"),
]


def _require_unique(candidates: tuple[CandidateID, ...]) -> None:
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate IDs contain a duplicate candidate")


class EncodedCandidate(_Contract[Literal[1]]):
    """One legal choice aligned to its exact JSON-safe engine selection."""

    candidate_id: CandidateID
    selection: JsonValue
    target_ref: str | None = Field(default=None, min_length=1)
    features: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identity_matches_selection(self) -> EncodedCandidate:
        candidate_id = self.candidate_id
        expected: JsonValue | None
        if isinstance(candidate_id, FinishCandidateID):
            expected = None
        elif isinstance(candidate_id, SkipCandidateID):
            expected = "SKIP"
        elif isinstance(candidate_id, CardCandidateID):
            expected = candidate_id.card_id
        elif isinstance(candidate_id, UnitCandidateID):
            expected = candidate_id.unit_id
        elif isinstance(candidate_id, HexCandidateID):
            expected = {"q": candidate_id.q, "r": candidate_id.r, "s": candidate_id.s}
        elif isinstance(candidate_id, NumberCandidateID):
            expected = candidate_id.value
            if isinstance(self.selection, bool):
                raise ValueError("number candidate selection must be numeric, not boolean")
        elif isinstance(candidate_id, OptionCandidateID):
            expected = self.selection
            if (
                self.selection != candidate_id.option_id
                and str(self.selection) != candidate_id.option_id
            ):
                raise ValueError("option candidate identity must align with its engine selection")
        elif isinstance(candidate_id, ActionCandidateID):
            expected = self.selection
            if (
                self.selection != candidate_id.action_id
                and str(self.selection) != candidate_id.action_id
            ):
                raise ValueError("action candidate identity must align with its engine selection")
        elif isinstance(candidate_id, EntityCandidateID):
            expected = self.selection
            if self.target_ref != candidate_id.entity_ref:
                raise ValueError("entity candidate identity must align with its target reference")
        else:
            raise ValueError("unsupported candidate identity variant")
        if self.selection != expected:
            raise ValueError("candidate identity must align with its engine selection")
        return self


class DecisionObservation(_Contract[Literal[3]]):
    """A v2 state graph paired with an ordered legal decision in schema v3."""

    state: LearnedObservation
    decision_kind: str = Field(min_length=1)
    candidates: tuple[EncodedCandidate, ...]

    @model_validator(mode="after")
    def _valid_decision_observation(self) -> DecisionObservation:
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        _require_unique(candidate_ids)
        tokens = {token.local_ref: token for token in self.state.tokens}
        for candidate in self.candidates:
            if candidate.target_ref is not None and candidate.target_ref not in tokens:
                raise ValueError("candidate target reference must identify an observation token")
            candidate_id = candidate.candidate_id
            if (
                isinstance(
                    candidate_id,
                    (CardCandidateID, UnitCandidateID, HexCandidateID, EntityCandidateID),
                )
                and candidate.target_ref is None
            ):
                raise ValueError("graph-bound candidate requires a target reference")
            target = tokens.get(candidate.target_ref) if candidate.target_ref is not None else None
            if isinstance(candidate_id, CardCandidateID):
                visible_card = (
                    target is not None
                    and target.kind == "CARD"
                    and target.features.get("card_id") == candidate_id.card_id
                )
                legal_card_without_visible_token = (
                    target is not None
                    and target.kind == "HERO"
                    and target.features.get("is_decision_owner") is True
                )
                if not (visible_card or legal_card_without_visible_token):
                    raise ValueError(
                        "card candidate reference must identify its visible CARD token "
                        "or hidden decision owner"
                    )
            elif isinstance(candidate_id, UnitCandidateID):
                identity = None
                if target and target.kind == "UNIT":
                    identity = target.features.get("entity_id", target.features.get("unit_id"))
                if target and target.kind == "HERO":
                    identity = target.features.get("hero_id")
                if identity != candidate_id.unit_id:
                    raise ValueError(
                        "unit candidate reference must identify its UNIT or HERO token"
                    )
            elif isinstance(candidate_id, HexCandidateID):
                coordinates = (candidate_id.q, candidate_id.r, candidate_id.s)
                target_coordinates = (
                    (
                        target.features.get("q"),
                        target.features.get("r"),
                        target.features.get("s"),
                    )
                    if target and target.kind == "TILE"
                    else None
                )
                if target_coordinates != coordinates:
                    raise ValueError("hex candidate reference must identify its TILE token")
            elif isinstance(candidate_id, EntityCandidateID):
                if target is None or target.kind not in {"TOKEN", "ENTITY"}:
                    raise ValueError(
                        "entity candidate reference must identify an entity graph token"
                    )
            elif isinstance(
                candidate_id,
                (
                    FinishCandidateID,
                    SkipCandidateID,
                    NumberCandidateID,
                    OptionCandidateID,
                    ActionCandidateID,
                ),
            ):
                if target is not None:
                    raise ValueError("non-graph candidate cannot carry a target reference")
            else:
                raise ValueError("unsupported candidate identity variant")
        return self


__all__ = [
    "ActionCandidateID",
    "CandidateID",
    "CardCandidateID",
    "DecisionObservation",
    "EncodedCandidate",
    "EntityCandidateID",
    "FinishCandidateID",
    "HexCandidateID",
    "NumberCandidateID",
    "OptionCandidateID",
    "SkipCandidateID",
    "UnitCandidateID",
]
