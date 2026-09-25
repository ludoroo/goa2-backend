"""Torch-free, artifact-pinnable feature declarations and vectorization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from functools import cached_property
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from automata.models.contracts import CandidateID, DecisionObservation, EncodedCandidate

_RESERVED = ("PAD", "UNK", "MISSING")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NumericFeature(_Frozen):
    source: str = Field(min_length=1)
    dtype: Literal["BOOLEAN", "INTEGER", "FLOAT"]
    default: bool | int | float
    normalization: Literal["NONE", "STANDARD", "MIN_MAX", "SIGNED_LOG"] = "NONE"
    policy: Literal["DIRECT", "COUNT", "SUM", "MEAN", "MAX"] = "DIRECT"


class CategoricalFeature(_Frozen):
    source: str = Field(min_length=1)
    vocabulary: tuple[str, ...]
    policy: Literal["DIRECT"] = "DIRECT"

    @model_validator(mode="after")
    def _valid_vocabulary(self) -> CategoricalFeature:
        if self.vocabulary[:3] != _RESERVED or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("categorical vocabulary must have unique PAD/UNK/MISSING prefixes")
        return self


class ReferenceFeature(_Frozen):
    source: str = Field(min_length=1)
    required: bool
    policy: Literal["DIRECT"] = "DIRECT"


class IgnoredFeature(_Frozen):
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    policy: Literal["IGNORE"] = "IGNORE"


class RecordFeatureSchema(_Frozen):
    kind: str = Field(min_length=1)
    numeric: tuple[NumericFeature, ...] = ()
    categorical: tuple[CategoricalFeature, ...] = ()
    references: tuple[ReferenceFeature, ...] = ()
    ignored: tuple[IgnoredFeature, ...] = ()

    @model_validator(mode="after")
    def _unique_sources(self) -> RecordFeatureSchema:
        sources = [item.source for item in self.numeric]
        sources.extend(item.source for item in self.categorical)
        sources.extend(item.source for item in self.references)
        sources.extend(item.source for item in self.ignored)
        if len(sources) != len(set(sources)):
            raise ValueError(f"duplicate feature declaration in {self.kind}")
        return self


class VectorizedToken(_Frozen):
    local_ref: str
    kind: str
    numeric: tuple[float, ...]
    numeric_valid: tuple[bool, ...]
    categorical: tuple[int, ...]
    references: tuple[int, ...]
    reference_valid: tuple[bool, ...]


class VectorizedRelationship(_Frozen):
    source_ref: str
    target_ref: str
    source_index: int
    target_index: int
    kind: str
    numeric: tuple[float, ...]
    numeric_valid: tuple[bool, ...]
    categorical: tuple[int, ...]


class VectorizedCandidate(_Frozen):
    kind: str
    numeric: tuple[float, ...]
    numeric_valid: tuple[bool, ...]
    categorical: tuple[int, ...]
    references: tuple[int, ...]
    reference_valid: tuple[bool, ...]


class VectorizedDecision(_Frozen):
    tokens: tuple[VectorizedToken, ...]
    relationships: tuple[VectorizedRelationship, ...]
    candidates: tuple[VectorizedCandidate, ...]
    candidate_ids: tuple[CandidateID, ...]


def _n(
    source: str,
    dtype: Literal["BOOLEAN", "INTEGER", "FLOAT"] = "INTEGER",
    *,
    default: bool | int | float = 0,
    normalization: Literal["NONE", "STANDARD", "MIN_MAX", "SIGNED_LOG"] = "NONE",
    policy: Literal["DIRECT", "COUNT", "SUM", "MEAN", "MAX"] = "DIRECT",
) -> NumericFeature:
    return NumericFeature(
        source=source,
        dtype=dtype,
        default=default,
        normalization=normalization,
        policy=policy,
    )


def _b(source: str) -> NumericFeature:
    return _n(source, "BOOLEAN", default=False)


def _c(source: str, *values: str) -> CategoricalFeature:
    return CategoricalFeature(source=source, vocabulary=(*_RESERVED, *values))


def _r(source: str, required: bool = True) -> ReferenceFeature:
    return ReferenceFeature(source=source, required=required)


def _i(source: str, reason: str) -> IgnoredFeature:
    return IgnoredFeature(source=source, reason=reason)


def _record(
    kind: str,
    *,
    numeric: tuple[NumericFeature, ...] = (),
    categorical: tuple[CategoricalFeature, ...] = (),
    references: tuple[ReferenceFeature, ...] = (),
    ignored: tuple[IgnoredFeature, ...] = (),
) -> RecordFeatureSchema:
    return RecordFeatureSchema(
        kind=kind,
        numeric=numeric,
        categorical=categorical,
        references=references,
        ignored=ignored,
    )


def _token_schemas() -> tuple[RecordFeatureSchema, ...]:
    raw_id = "Observation-local/public ID is retained only for reference alignment."
    return (
        _record(
            "GLOBAL",
            numeric=tuple(
                _n(name)
                for name in (
                    "round",
                    "turn",
                    "team_count",
                    "hero_count",
                    "lane_count",
                    "playable_tile_count",
                )
            ),
            categorical=(
                _c(
                    "map_id",
                    "forgotten_island",
                    "narrow_passages",
                    "across_the_river",
                    "vexing_cliffs",
                ),
                _c("game_type", "QUICK", "STANDARD"),
                _c("phase", "SETUP", "PLANNING", "RESOLUTION", "END"),
                _c("tie_breaker_team", "RED", "BLUE"),
            ),
        ),
        _record(
            "TEAM",
            numeric=(
                *(
                    _n(name)
                    for name in (
                        "life_counters",
                        "hero_count",
                        "alive_hero_count",
                        "physical_piece_count",
                        "minion_count",
                        "total_level",
                        "total_gold",
                    )
                ),
                _n("mean_level", "FLOAT"),
                _n("mean_gold", "FLOAT"),
            ),
            categorical=(_c("relation", "OWN", "ENEMY", "PUBLIC"),),
            ignored=(_i("team_id", raw_id),),
        ),
        _record(
            "HERO",
            numeric=(
                _n("level"),
                _n("gold"),
                _n("items", policy="COUNT"),
                _n("wish_cast_count"),
                _n("rune_slots", policy="COUNT"),
                _b("is_current_actor"),
                _b("is_decision_owner"),
                _b("is_acting_piece"),
            ),
            categorical=(
                _c("name", "Razzle", "Wasp", "Arien", "Brogan"),
                _c("title"),
                _c("relation", "SELF", "ALLY", "ENEMY", "PUBLIC"),
            ),
            references=(_r("team_ref"),),
            ignored=(
                _i("hero_id", raw_id),
                _i("team_id", raw_id),
                _i(
                    "adapter_features",
                    "Hero adapter tensors require their own versioned declaration.",
                ),
            ),
        ),
        _record(
            "TILE",
            numeric=(*(_n(name) for name in ("q", "r", "s")), _b("is_terrain"), _b("has_occupant")),
            categorical=(_c("spawn_team", "RED", "BLUE"), _c("spawn_type", "HERO", "MINION")),
            ignored=(_i("zone_id", raw_id),),
        ),
        _record(
            "ZONE",
            numeric=(_n("neighbor_count"), _n("spawn_point_count"), _b("is_battle_zone")),
            ignored=(_i("zone_id", raw_id),),
        ),
        _record(
            "LANE",
            numeric=(
                _n("zone_count"),
                _n("battle_zone_index"),
                _n("signed_battle_zone_advantage", "FLOAT"),
                _n("wave_counter"),
            ),
            references=(_r("battle_zone_ref", False),),
            ignored=(
                _i("lane_id", raw_id),
                _i("ordered_zone_refs", "Lane membership is represented by ZONE_IN_LANE edges."),
            ),
        ),
        _record(
            "UNIT",
            numeric=(
                _b("is_current_actor"),
                _b("is_decision_owner"),
                _b("is_acting_piece"),
                _b("is_positioned"),
                _n("value"),
                _b("is_heavy"),
            ),
            categorical=(
                _c("unit_type", "HERO", "HERO_PIECE", "MINION"),
                _c("relation", "SELF", "ALLY", "ENEMY", "PUBLIC"),
                _c("minion_type", "MELEE", "RANGED"),
            ),
            references=(
                _r("owner_ref"),
                _r("tile_ref", False),
                _r("zone_ref", False),
                _r("lane_ref", False),
            ),
            ignored=(_i("entity_id", raw_id), _i("team_id", raw_id)),
        ),
        _record(
            "CARD",
            numeric=(
                _n("count"),
                _n("primary_action_value"),
                _n("secondary_actions", policy="COUNT"),
                _n("initiative"),
                _b("is_facedown"),
                _b("is_ranged"),
                _n("range_value"),
                _n("radius_value"),
                _b("is_active"),
                _n("spell_rank"),
            ),
            categorical=tuple(
                _c(name)
                for name in (
                    "area",
                    "visibility",
                    "name",
                    "image_id",
                    "tier",
                    "color",
                    "primary_action",
                    "effect_id",
                    "state",
                    "item",
                )
            ),
            references=(_r("owner_ref"),),
            ignored=(
                _i("card_id", raw_id),
                _i("effect_text", "Free-form display text is not a stable model identity."),
            ),
        ),
        _record(
            "EFFECT",
            numeric=(
                _b("is_active"),
                _n("stat_value", "FLOAT"),
                _n("split_value", "FLOAT"),
            ),
            categorical=(
                _c("effect_type"),
                _c("duration", "THIS_TURN", "NEXT_TURN", "THIS_ROUND", "PASSIVE"),
                _c("stat_type"),
                _c("split_axis"),
                _c("named_color"),
            ),
            ignored=(
                _i("effect_id", raw_id),
                _i(
                    "scope",
                    "Structured effect scope needs separately declared shape, range, and target semantics.",
                ),
            ),
        ),
        _record(
            "MARKER",
            numeric=(_n("value", "FLOAT"),),
            categorical=(_c("marker_type"),),
            references=(_r("target_ref", False), _r("source_ref", False)),
        ),
        _record(
            "TOKEN",
            numeric=(_b("is_facedown"), _b("is_passable")),
            categorical=(_c("name"), _c("token_type")),
            references=(_r("owner_ref", False), _r("tile_ref")),
            ignored=(_i("entity_id", raw_id),),
        ),
        _record(
            "ENTITY",
            numeric=(_b("is_obstacle"),),
            categorical=(_c("name"), _c("entity_kind")),
            references=(_r("owner_ref", False), _r("tile_ref", False)),
            ignored=(_i("entity_id", raw_id),),
        ),
    )


def _relationship_schemas() -> tuple[RecordFeatureSchema, ...]:
    empty = tuple(
        _record(kind)
        for kind in (
            "HEX_ADJACENT",
            "IN_LANE",
            "IN_ZONE",
            "OWNS",
            "POSITIONED_AT",
            "TILE_IN_ZONE",
            "ZONE_ADJACENT",
            "ZONE_IN_LANE",
        )
    )
    unit = _record(
        "UNIT_TO_UNIT",
        numeric=(
            *(_n(name) for name in ("delta_q", "delta_r", "delta_s", "hex_distance")),
            _n("path_distance", "FLOAT"),
            *(
                _b(name)
                for name in (
                    "path_exists",
                    "is_adjacent",
                    "topology_is_adjacent",
                    "is_straight_line",
                    "has_line_of_sight",
                    "same_zone",
                    "same_lane",
                )
            ),
            _n("lane_progress_delta", "FLOAT"),
            *(
                _b(name)
                for name in (
                    "reachable",
                    "threatens",
                    "supports",
                    "path_distance_valid",
                    "has_line_of_sight_valid",
                    "reachable_valid",
                    "threatens_valid",
                    "supports_valid",
                )
            ),
        ),
        categorical=(_c("relation", "SELF", "ALLY", "ENEMY", "PUBLIC"),),
    )
    return (*empty, unit)


def _candidate_schemas() -> tuple[RecordFeatureSchema, ...]:
    raw = "Engine selection identity remains Python-side and is never embedded categorically."
    common_ignored = (
        _i("selection", raw),
        _i("features", "No candidate extension fields are declared in schema v1."),
    )
    return (
        _record("FINISH", references=(_r("target_ref", False),), ignored=common_ignored),
        _record("SKIP", references=(_r("target_ref", False),), ignored=common_ignored),
        _record(
            "CARD", references=(_r("target_ref"),), ignored=(*common_ignored, _i("card_id", raw))
        ),
        _record(
            "UNIT", references=(_r("target_ref"),), ignored=(*common_ignored, _i("unit_id", raw))
        ),
        _record(
            "HEX",
            numeric=(_n("q"), _n("r"), _n("s")),
            references=(_r("target_ref"),),
            ignored=common_ignored,
        ),
        _record(
            "NUMBER",
            numeric=(_n("value", "FLOAT"),),
            references=(_r("target_ref", False),),
            ignored=common_ignored,
        ),
        _record(
            "OPTION",
            references=(_r("target_ref", False),),
            ignored=(*common_ignored, _i("option_id", raw)),
        ),
        _record(
            "ACTION",
            references=(_r("target_ref", False),),
            ignored=(*common_ignored, _i("action_id", raw)),
        ),
        _record(
            "ENTITY",
            references=(_r("target_ref"),),
            ignored=(
                *common_ignored,
                _i("entity_ref", "Duplicated by the typed target_ref reference."),
            ),
        ),
    )


def _schema_digest(
    *,
    schema_version: int,
    schema_id: str,
    observation_schema_version: int,
    tokens: tuple[RecordFeatureSchema, ...],
    relationships: tuple[RecordFeatureSchema, ...],
    candidates: tuple[RecordFeatureSchema, ...],
) -> str:
    payload = {
        "schema_version": schema_version,
        "schema_id": schema_id,
        "observation_schema_version": observation_schema_version,
        "tokens": [item.model_dump(mode="json") for item in tokens],
        "relationships": [item.model_dump(mode="json") for item in relationships],
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


class TensorFeatureSchema(_Frozen):
    """Immutable declarations that convert observations to primitive arrays."""

    schema_version: Literal[1] = 1
    schema_id: str = "goa2-tensor-features-v1"
    observation_schema_version: Literal[3] = 3
    tokens: tuple[RecordFeatureSchema, ...]
    relationships: tuple[RecordFeatureSchema, ...]
    candidates: tuple[RecordFeatureSchema, ...]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    _CURRENT: ClassVar[TensorFeatureSchema | None] = None

    @cached_property
    def _token_by_kind(self) -> dict[str, RecordFeatureSchema]:
        return {item.kind: item for item in self.tokens}

    @cached_property
    def _relationship_by_kind(self) -> dict[str, RecordFeatureSchema]:
        return {item.kind: item for item in self.relationships}

    @cached_property
    def _candidate_by_kind(self) -> dict[str, RecordFeatureSchema]:
        return {item.kind: item for item in self.candidates}

    @model_validator(mode="after")
    def _valid_digest_and_kinds(self) -> TensorFeatureSchema:
        for collection in (self.tokens, self.relationships, self.candidates):
            kinds = [item.kind for item in collection]
            if len(kinds) != len(set(kinds)):
                raise ValueError("record schema kinds must be unique")
        expected = _schema_digest(
            schema_version=self.schema_version,
            schema_id=self.schema_id,
            observation_schema_version=self.observation_schema_version,
            tokens=self.tokens,
            relationships=self.relationships,
            candidates=self.candidates,
        )
        if self.digest != expected:
            raise ValueError("tensor feature schema digest does not match its declarations")
        return self

    @classmethod
    def current(cls) -> TensorFeatureSchema:
        if cls._CURRENT is None:
            tokens = _token_schemas()
            relationships = _relationship_schemas()
            candidates = _candidate_schemas()
            digest = _schema_digest(
                schema_version=1,
                schema_id="goa2-tensor-features-v1",
                observation_schema_version=3,
                tokens=tokens,
                relationships=relationships,
                candidates=candidates,
            )
            cls._CURRENT = cls(
                tokens=tokens,
                relationships=relationships,
                candidates=candidates,
                digest=digest,
            )
        assert cls._CURRENT is not None
        return cls._CURRENT

    def vectorize(self, observation: DecisionObservation, *, training: bool) -> VectorizedDecision:
        if observation.schema_version != self.observation_schema_version:
            raise ValueError("observation schema version is incompatible with tensor schema")
        if not observation.candidates:
            raise ValueError("candidate collection cannot be empty")
        index_by_ref = {
            token.local_ref: index for index, token in enumerate(observation.state.tokens)
        }

        tokens: list[VectorizedToken] = []
        for token in observation.state.tokens:
            schema = self._require_kind(self._token_by_kind, token.kind, "token")
            numeric, valid, categorical = self._values(schema, token.features, training=training)
            references, reference_valid = self._references(schema, token.features, index_by_ref)
            tokens.append(
                VectorizedToken(
                    local_ref=token.local_ref,
                    kind=token.kind,
                    numeric=numeric,
                    numeric_valid=valid,
                    categorical=categorical,
                    references=references,
                    reference_valid=reference_valid,
                )
            )

        relationships: list[VectorizedRelationship] = []
        for edge in observation.state.relationships:
            schema = self._require_kind(self._relationship_by_kind, edge.kind, "relationship")
            numeric, valid, categorical = self._values(schema, edge.features, training=training)
            # Observation contracts validate these refs; repeat the check for model safety.
            if edge.source_ref not in index_by_ref or edge.target_ref not in index_by_ref:
                raise ValueError("relationship reference does not identify a token")
            relationships.append(
                VectorizedRelationship(
                    source_ref=edge.source_ref,
                    target_ref=edge.target_ref,
                    source_index=index_by_ref[edge.source_ref],
                    target_index=index_by_ref[edge.target_ref],
                    kind=edge.kind,
                    numeric=numeric,
                    numeric_valid=valid,
                    categorical=categorical,
                )
            )

        candidates: list[VectorizedCandidate] = []
        for candidate in observation.candidates:
            if training and candidate.features:
                raise ValueError("undeclared candidate extension fields")
            kind = candidate.candidate_id.kind
            schema = self._require_kind(self._candidate_by_kind, kind, "candidate")
            source = self._candidate_source(candidate)
            numeric, valid, categorical = self._values(schema, source, training=training)
            references, reference_valid = self._references(schema, source, index_by_ref)
            candidates.append(
                VectorizedCandidate(
                    kind=kind,
                    numeric=numeric,
                    numeric_valid=valid,
                    categorical=categorical,
                    references=references,
                    reference_valid=reference_valid,
                )
            )
        return VectorizedDecision(
            tokens=tuple(tokens),
            relationships=tuple(relationships),
            candidates=tuple(candidates),
            candidate_ids=tuple(item.candidate_id for item in observation.candidates),
        )

    @staticmethod
    def _require_kind(
        by_kind: dict[str, RecordFeatureSchema], kind: str, record_type: str
    ) -> RecordFeatureSchema:
        try:
            return by_kind[kind]
        except KeyError as exc:
            raise ValueError(f"unknown {record_type} kind: {kind!r}") from exc

    @staticmethod
    def _candidate_source(candidate: EncodedCandidate) -> dict[str, object]:
        # Candidate IDs are inspected structurally, never converted to strings.
        identity = candidate.candidate_id.model_dump(
            mode="json", exclude={"schema_version", "kind"}
        )
        return {
            **identity,
            "selection": candidate.selection,
            "target_ref": candidate.target_ref,
            "features": candidate.features,
        }

    @staticmethod
    def _values(
        schema: RecordFeatureSchema,
        values: Mapping[str, object],
        *,
        training: bool,
    ) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...]]:
        declared = {item.source for item in schema.numeric}
        declared.update(item.source for item in schema.categorical)
        declared.update(item.source for item in schema.references)
        declared.update(item.source for item in schema.ignored)
        unknown = set(values) - declared
        if training and unknown:
            raise ValueError(f"undeclared {schema.kind} fields: {sorted(unknown)!r}")
        numeric_values: list[float] = []
        numeric_valid: list[bool] = []
        for feature in schema.numeric:
            raw = values.get(feature.source)
            value, valid = TensorFeatureSchema._numeric(feature, raw)
            numeric_values.append(value)
            numeric_valid.append(valid)
        categorical_values = tuple(
            TensorFeatureSchema._categorical(feature, values.get(feature.source))
            for feature in schema.categorical
        )
        return tuple(numeric_values), tuple(numeric_valid), categorical_values

    @staticmethod
    def _numeric(feature: NumericFeature, raw: object) -> tuple[float, bool]:
        if raw is None:
            return float(feature.default), False
        if feature.policy == "COUNT":
            if not isinstance(raw, (dict, list)):
                raise ValueError(f"{feature.source} must be a collection for COUNT")
            return float(len(raw)), True
        if feature.policy != "DIRECT":
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{feature.source} must be a non-empty numeric list")
            numbers: list[float] = []
            for item in raw:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(item)
                ):
                    raise ValueError(f"{feature.source} contains malformed numeric values")
                numbers.append(float(item))
            if feature.policy == "SUM":
                return sum(numbers), True
            if feature.policy == "MEAN":
                return sum(numbers) / len(numbers), True
            return max(numbers), True
        if feature.dtype == "BOOLEAN":
            if not isinstance(raw, bool):
                raise ValueError(f"{feature.source} must be boolean")
            return float(raw), True
        if feature.dtype == "INTEGER":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{feature.source} must be an integer")
            return float(raw), True
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{feature.source} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{feature.source} must be finite")
        return value, True

    @staticmethod
    def _categorical(feature: CategoricalFeature, raw: object) -> int:
        if raw is None:
            return 2
        if not isinstance(raw, str):
            raise ValueError(f"{feature.source} must be a string categorical value")
        try:
            return feature.vocabulary.index(raw)
        except ValueError:
            return 1

    @staticmethod
    def _references(
        schema: RecordFeatureSchema,
        values: Mapping[str, object],
        index_by_ref: dict[str, int],
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        indexes: list[int] = []
        valid: list[bool] = []
        for feature in schema.references:
            raw = values.get(feature.source)
            if raw is None:
                if feature.required:
                    raise ValueError(f"required reference {feature.source!r} is missing")
                indexes.append(-1)
                valid.append(False)
            elif not isinstance(raw, str) or raw not in index_by_ref:
                raise ValueError(f"reference {feature.source!r} does not identify a token")
            else:
                indexes.append(index_by_ref[raw])
                valid.append(True)
        return tuple(indexes), tuple(valid)


__all__ = [
    "CategoricalFeature",
    "IgnoredFeature",
    "NumericFeature",
    "RecordFeatureSchema",
    "ReferenceFeature",
    "TensorFeatureSchema",
    "VectorizedCandidate",
    "VectorizedDecision",
    "VectorizedRelationship",
    "VectorizedToken",
]
