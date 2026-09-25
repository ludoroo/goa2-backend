"""Framework-independent observation contracts."""

from __future__ import annotations

import math
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

VersionT = TypeVar("VersionT", bound=int)


class _Contract(BaseModel, Generic[VersionT]):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: VersionT

    @model_validator(mode="after")
    def _all_numbers_are_finite(self) -> _Contract[VersionT]:
        def check(value: object) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("serialized contract numbers must be finite")
            if isinstance(value, BaseModel):
                for field_value in value.__dict__.values():
                    check(field_value)
            elif isinstance(value, dict):
                for field_value in value.values():
                    check(field_value)
            elif isinstance(value, (tuple, list)):
                for field_value in value:
                    check(field_value)

        for item in self.__dict__.values():
            check(item)
        return self


class Viewer(_Contract[Literal[2]]):
    """Orthogonal visibility entitlement and value-orientation context."""

    private_hero_id: str | None = None
    perspective_team: str | None = None


class PublicSnapshot(_Contract[Literal[2]]):
    viewer: Viewer
    map_id: str
    game_type: str
    public_state: dict[str, JsonValue]


class ObservationToken(_Contract[Literal[1]]):
    """One immutable, observation-local graph token."""

    local_ref: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    features: dict[str, JsonValue]


class ObservationRelationship(_Contract[Literal[1]]):
    """One immutable directed edge between observation-local token refs."""

    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    features: dict[str, JsonValue]


class LearnedObservation(_Contract[Literal[2]]):
    """Graph observation using the current viewer contract."""

    viewer: Viewer
    decision_kind: Literal["SNAPSHOT"] = "SNAPSHOT"
    # Retained solely for rejection of legacy sparse observations. Candidate
    # identity belongs to candidates.py; graph observations require this empty.
    candidate_ids: tuple[JsonValue, ...] = ()
    features: dict[str, float] = Field(default_factory=dict)
    tokens: tuple[ObservationToken, ...] = ()
    relationships: tuple[ObservationRelationship, ...] = ()

    @model_validator(mode="after")
    def _valid_graph_observation(self) -> LearnedObservation:
        if self.candidate_ids or self.features:
            raise ValueError("graph observations have no candidates or sparse features")
        refs = {token.local_ref for token in self.tokens}
        if len(refs) != len(self.tokens):
            raise ValueError("graph observation token refs must be unique")
        if any(
            edge.source_ref not in refs or edge.target_ref not in refs
            for edge in self.relationships
        ):
            raise ValueError("graph relationship refs must identify observation tokens")
        return self


__all__ = [
    "LearnedObservation",
    "ObservationRelationship",
    "ObservationToken",
    "PublicSnapshot",
    "Viewer",
]
