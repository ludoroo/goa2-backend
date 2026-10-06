"""Strict contracts for finite native Gen1 paired gameplay evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from automata.models.contracts import (
    GEN1_RUNTIME_COMPATIBILITY_VERSION,
    canonical_json_bytes,
)
from automata.models.shared_encoder.gen1_model import GEN1_ARCHITECTURE_ID
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.runtime.effects import register_all_effects
from automata.training.native_gen1 import current_gen1_artifact_scope
from automata.training.native_run_contracts import NativeRunSearchConfig
from automata.training.native_splits import NativeSplitConfig, native_seed_purpose

NativeGameplayComparisonKind = Literal[
    "CANDIDATE_VS_HEURISTIC_FULL_SEARCH",
    "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY",
    "CANDIDATE_VS_HEURISTIC_POLICY_ONLY",
    "CANDIDATE_VS_GEN1_PARENT_FULL_SEARCH",
]

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_MANDATORY_COMPARISONS: tuple[NativeGameplayComparisonKind, ...] = (
    "CANDIDATE_VS_HEURISTIC_FULL_SEARCH",
    "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY",
    "CANDIDATE_VS_HEURISTIC_POLICY_ONLY",
)
_PARENT_COMPARISON: NativeGameplayComparisonKind = "CANDIDATE_VS_GEN1_PARENT_FULL_SEARCH"
_PAIRING_RECIPE: Literal["fixed-board-rosters-contestant-side-swap-v1"] = (
    "fixed-board-rosters-contestant-side-swap-v1"
)
_MAPS_ROOT = Path(__file__).resolve().parents[2] / "goa2" / "data" / "maps"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _CanonicalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def canonical_bytes(self) -> bytes:
        validated = type(self).model_validate(self.model_dump(mode="python"), strict=True)
        return canonical_json_bytes(validated)


class _DigestedCanonicalModel(_CanonicalModel):
    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _strict_model(value: _ModelT, expected: type[_ModelT], *, label: str) -> _ModelT:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}")
    return expected.model_validate(value.model_dump(mode="python"), strict=True)


def _nonempty_trimmed(value: str, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _digest_mapping(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_absolute_path(value: Path, *, label: str) -> Path:
    if not value.is_absolute():
        raise ValueError(f"{label} must be absolute")
    normalized = Path(os.path.abspath(value))
    if ".." in value.parts or value != normalized:
        raise ValueError(f"{label} must be lexically normalized")
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")


def _strict_schema_version(value: Any) -> Any:
    if type(value) is not int:
        raise ValueError("schema_version must be the integer 1")
    return value


class NativeEvaluationFixture(_DigestedCanonicalModel):
    fixture_id: str
    map_id: str
    game_type: str
    red_composition: tuple[str, ...] = Field(min_length=1)
    blue_composition: tuple[str, ...] = Field(min_length=1)
    world_seeds: tuple[StrictInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_fixture(self) -> NativeEvaluationFixture:
        for label in ("fixture_id", "map_id", "game_type"):
            _nonempty_trimmed(getattr(self, label), label=label)
        heroes = (*self.red_composition, *self.blue_composition)
        if any(type(hero) is not str or not hero or hero != hero.strip() for hero in heroes):
            raise ValueError("evaluation heroes must be nonempty trimmed strings")
        if len(heroes) != len(set(heroes)):
            raise ValueError("evaluation fixture heroes must be unique across both teams")
        if any(seed < 0 for seed in self.world_seeds):
            raise ValueError("evaluation world seeds must be nonnegative")
        if len(self.world_seeds) != len(set(self.world_seeds)):
            raise ValueError("evaluation fixture world seeds must be unique")
        return self


class NativePairedEvaluationConfig(_DigestedCanonicalModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    source_revision: str
    dirty_tree_hash: str
    candidate_model_digest: str = Field(pattern=_DIGEST_PATTERN)
    parent_model_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    split_config: NativeSplitConfig
    comparisons: tuple[NativeGameplayComparisonKind, ...] = Field(min_length=1)
    fixtures: tuple[NativeEvaluationFixture, ...] = Field(min_length=1)
    search: NativeRunSearchConfig
    random_stream_namespace: str
    max_steps: StrictInt = Field(gt=0)
    max_rounds: StrictInt = Field(gt=0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_is_strict(cls, value: Any) -> Any:
        return _strict_schema_version(value)

    @model_validator(mode="after")
    def _valid_config(self) -> NativePairedEvaluationConfig:
        for label in (
            "evaluation_id",
            "source_revision",
            "dirty_tree_hash",
            "random_stream_namespace",
        ):
            _nonempty_trimmed(getattr(self, label), label=label)
        split = NativeSplitConfig.model_validate(
            self.split_config.model_dump(mode="python"), strict=True
        )
        search = NativeRunSearchConfig.model_validate(
            self.search.model_dump(mode="python"), strict=True
        )
        fixtures = tuple(
            NativeEvaluationFixture.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.fixtures
        )
        if not search.use_prior:
            raise ValueError("native paired evaluation search use_prior must be true")
        if len(self.comparisons) != len(set(self.comparisons)):
            raise ValueError("native paired evaluation comparisons must not contain duplicates")
        missing = set(_MANDATORY_COMPARISONS) - set(self.comparisons)
        if missing:
            raise ValueError(
                "native paired evaluation mandatory comparisons are missing: "
                f"{sorted(missing)!r}"
            )
        has_parent_comparison = _PARENT_COMPARISON in self.comparisons
        if has_parent_comparison != (self.parent_model_digest is not None):
            raise ValueError("parent digest and parent comparison must be present together")
        if self.parent_model_digest == self.candidate_model_digest:
            raise ValueError("candidate and parent model digests must differ")
        fixture_ids = tuple(item.fixture_id for item in fixtures)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("evaluation fixture IDs must be unique")
        scenario_ids = tuple(
            (
                item.map_id,
                item.game_type,
                item.red_composition,
                item.blue_composition,
                item.world_seeds,
            )
            for item in fixtures
        )
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("evaluation fixture scenario identities must be unique")
        for fixture in fixtures:
            for seed in fixture.world_seeds:
                if native_seed_purpose(split, seed) != "evaluation":
                    raise ValueError(f"world seed {seed} must have declared evaluation purpose")
        return self


class NativePairedEvaluationAuthorities(_DigestedCanonicalModel):
    output_root: Path
    candidate_artifact_path: Path
    parent_artifact_path: Path | None

    @field_validator("output_root")
    @classmethod
    def _valid_output_root(cls, value: Path) -> Path:
        return _normalized_absolute_path(value, label="evaluation output root")

    @field_validator("candidate_artifact_path")
    @classmethod
    def _valid_candidate_path(cls, value: Path) -> Path:
        return _normalized_absolute_path(value, label="candidate artifact path")

    @field_validator("parent_artifact_path")
    @classmethod
    def _valid_parent_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return _normalized_absolute_path(value, label="parent artifact path")

    @model_validator(mode="after")
    def _disjoint_authorities(self) -> NativePairedEvaluationAuthorities:
        paths = [self.output_root, self.candidate_artifact_path]
        if self.parent_artifact_path is not None:
            paths.append(self.parent_artifact_path)
        if any(
            _paths_overlap(left, right)
            for index, left in enumerate(paths)
            for right in paths[index + 1 :]
        ):
            raise ValueError("evaluation output and artifact authorities must be disjoint")
        return self


class NativeEvaluationScopeIdentity(_CanonicalModel):
    supported_heroes: tuple[str, ...]
    supported_maps: tuple[str, ...]
    supported_game_types: tuple[str, ...]
    hero_adapter_versions: dict[str, StrictInt]
    map_schema_version: StrictInt = Field(gt=0)
    digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _valid_identity(self) -> NativeEvaluationScopeIdentity:
        for label, values in (
            ("heroes", self.supported_heroes),
            ("maps", self.supported_maps),
            ("game types", self.supported_game_types),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"evaluation scope {label} must be unique and sorted")
            if any(type(value) is not str or not value for value in values):
                raise ValueError(f"evaluation scope {label} must be nonempty strings")
        if set(self.hero_adapter_versions) != {"generic", *self.supported_heroes}:
            raise ValueError("evaluation scope adapter versions must cover generic and all heroes")
        if any(version <= 0 for version in self.hero_adapter_versions.values()):
            raise ValueError("evaluation scope adapter versions must be positive")
        if self.digest != _scope_digest_fields(
            supported_heroes=self.supported_heroes,
            supported_maps=self.supported_maps,
            supported_game_types=self.supported_game_types,
            hero_adapter_versions=self.hero_adapter_versions,
            map_schema_version=self.map_schema_version,
        ):
            raise ValueError("evaluation scope digest does not match its fields")
        return self


class NativeEvaluationMapIdentity(_DigestedCanonicalModel):
    map_id: str
    sha256: str = Field(pattern=_DIGEST_PATTERN)
    length: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def _valid_map(self) -> NativeEvaluationMapIdentity:
        _nonempty_trimmed(self.map_id, label="map_id")
        return self


class NativePlannedEvaluationCase(_DigestedCanonicalModel):
    ordinal: StrictInt = Field(ge=0)
    case_id: str = Field(pattern=_DIGEST_PATTERN)
    pair_id: str = Field(pattern=_DIGEST_PATTERN)
    comparison: NativeGameplayComparisonKind
    fixture_id: str
    world_seed: StrictInt = Field(ge=0)
    candidate_side: Literal["RED", "BLUE"]

    @model_validator(mode="after")
    def _valid_case(self) -> NativePlannedEvaluationCase:
        _nonempty_trimmed(self.fixture_id, label="fixture_id")
        return self


class NativePairedEvaluationManifest(_DigestedCanonicalModel):
    schema_version: Literal[1] = 1
    config: NativePairedEvaluationConfig
    authorities: NativePairedEvaluationAuthorities
    current_scope: NativeEvaluationScopeIdentity
    maps: tuple[NativeEvaluationMapIdentity, ...]
    planned_cases: tuple[NativePlannedEvaluationCase, ...] = Field(min_length=1)
    pairing_recipe: Literal["fixed-board-rosters-contestant-side-swap-v1"] = _PAIRING_RECIPE

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_is_strict(cls, value: Any) -> Any:
        return _strict_schema_version(value)

    @model_validator(mode="after")
    def _valid_derived_manifest(self) -> NativePairedEvaluationManifest:
        config = NativePairedEvaluationConfig.model_validate(
            self.config.model_dump(mode="python"), strict=True
        )
        authorities = NativePairedEvaluationAuthorities.model_validate(
            self.authorities.model_dump(mode="python"), strict=True
        )
        current_scope = NativeEvaluationScopeIdentity.model_validate(
            self.current_scope.model_dump(mode="python"), strict=True
        )
        maps = tuple(
            NativeEvaluationMapIdentity.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.maps
        )
        planned = tuple(
            NativePlannedEvaluationCase.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.planned_cases
        )
        expected_scope, expected_maps, expected_plan = _derive_manifest_parts(config)
        if current_scope != expected_scope:
            raise ValueError("evaluation manifest current scope is not canonical")
        if maps != expected_maps:
            raise ValueError("evaluation manifest map identities are not canonical")
        if planned != expected_plan:
            raise ValueError("evaluation manifest planned cases are not canonical")
        parent_enabled = config.parent_model_digest is not None
        if parent_enabled != (authorities.parent_artifact_path is not None):
            raise ValueError(
                "parent digest, comparison, and artifact path must be present together"
            )
        return self


class NativeEvaluatedArtifactIdentity(_DigestedCanonicalModel):
    role: Literal["CANDIDATE", "PARENT"]
    model_digest: str = Field(pattern=_DIGEST_PATTERN)
    manifest_sha256: str = Field(pattern=_DIGEST_PATTERN)
    model_id: str
    runtime_compatibility_version: StrictInt = Field(gt=0)
    decision_tensor_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    stable_value_tensor_schema_digest: str = Field(pattern=_DIGEST_PATTERN)
    value_semantics: str
    current_scope_digest: str = Field(pattern=_DIGEST_PATTERN)
    device: Literal["cpu"]
    floating_dtype: Literal["float32"]

    @model_validator(mode="after")
    def _valid_artifact_identity(self) -> NativeEvaluatedArtifactIdentity:
        for label in ("model_id", "value_semantics"):
            _nonempty_trimmed(getattr(self, label), label=label)
        return self


class NativeEvaluationObservation(_DigestedCanonicalModel):
    case_id: str = Field(pattern=_DIGEST_PATTERN)
    pair_id: str = Field(pattern=_DIGEST_PATTERN)
    comparison: NativeGameplayComparisonKind
    fixture_id: str
    world_seed: StrictInt = Field(ge=0)
    candidate_side: Literal["RED", "BLUE"]
    status: Literal["COMPLETED", "CENSORED"]
    raw_winner: str | None
    winner_side: Literal["RED", "BLUE"] | None
    reason: Literal["game_over", "max_steps", "max_rounds"]
    rounds: StrictInt = Field(ge=0)
    turns: StrictInt = Field(ge=0)
    steps: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _valid_outcome(self) -> NativeEvaluationObservation:
        _nonempty_trimmed(self.fixture_id, label="fixture_id")
        if self.status == "COMPLETED":
            if self.reason != "game_over":
                raise ValueError("completed evaluation observations require game_over")
            if self.raw_winner is None or self.winner_side is None:
                raise ValueError(
                    "completed evaluation observations require raw and normalized winners; "
                    "the engine has no draw rule"
                )
            _nonempty_trimmed(self.raw_winner, label="raw_winner")
        elif (
            self.reason not in {"max_steps", "max_rounds"}
            or self.raw_winner is not None
            or self.winner_side is not None
        ):
            raise ValueError("censored observations require a cap reason and no winner")
        return self


class NativeCompletedPairScore(_DigestedCanonicalModel):
    pair_id: str = Field(pattern=_DIGEST_PATTERN)
    comparison: NativeGameplayComparisonKind
    fixture_id: str
    world_seed: StrictInt = Field(ge=0)
    candidate_red_won: bool
    candidate_blue_won: bool
    candidate_pair_score: float

    @field_validator("candidate_pair_score", mode="before")
    @classmethod
    def _strict_pair_score(cls, value: Any) -> Any:
        if type(value) is not float or value not in {0.0, 0.5, 1.0}:
            raise ValueError("candidate pair score must be exactly 0.0, 0.5, or 1.0")
        return value

    @model_validator(mode="after")
    def _valid_score(self) -> NativeCompletedPairScore:
        _nonempty_trimmed(self.fixture_id, label="fixture_id")
        expected = (float(self.candidate_red_won) + float(self.candidate_blue_won)) / 2.0
        if self.candidate_pair_score != expected:
            raise ValueError("candidate pair score does not match its decisive games")
        return self


class NativeComparisonAggregate(_DigestedCanonicalModel):
    comparison: NativeGameplayComparisonKind
    planned_pair_count: StrictInt = Field(ge=0)
    attempted_case_count: StrictInt = Field(ge=0)
    completed_case_count: StrictInt = Field(ge=0)
    censored_case_count: StrictInt = Field(ge=0)
    completed_pair_count: StrictInt = Field(ge=0)
    censored_pair_count: StrictInt = Field(ge=0)
    excluded_completed_singletons: StrictInt = Field(ge=0)
    candidate_game_wins_in_completed_pairs: StrictInt = Field(ge=0)
    comparator_game_wins_in_completed_pairs: StrictInt = Field(ge=0)
    descriptive_mean_completed_pair_score: float | None
    censor_reasons: dict[str, StrictInt]

    @field_validator("descriptive_mean_completed_pair_score", mode="before")
    @classmethod
    def _strict_finite_mean(cls, value: Any) -> Any:
        if value is not None and (
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError("descriptive pair score mean must be a finite strict float in [0, 1]")
        return value

    @model_validator(mode="after")
    def _valid_counts(self) -> NativeComparisonAggregate:
        if set(self.censor_reasons) - {"max_steps", "max_rounds"}:
            raise ValueError("aggregate contains an unknown censor reason")
        if any(count <= 0 for count in self.censor_reasons.values()):
            raise ValueError("aggregate censor reason counts must be positive")
        if self.attempted_case_count != self.completed_case_count + self.censored_case_count:
            raise ValueError("aggregate attempted count must equal completed plus censored")
        if self.attempted_case_count != 2 * self.planned_pair_count:
            raise ValueError("aggregate must cover every planned pair leg")
        if self.completed_pair_count + self.censored_pair_count != self.planned_pair_count:
            raise ValueError("aggregate pair counts must cover the full fixed schedule")
        if self.excluded_completed_singletons > self.censored_pair_count:
            raise ValueError("excluded singleton count cannot exceed censored pairs")
        if sum(self.censor_reasons.values()) != self.censored_case_count:
            raise ValueError("aggregate censor reasons must cover every censored case")
        if (
            self.candidate_game_wins_in_completed_pairs
            + self.comparator_game_wins_in_completed_pairs
            != 2 * self.completed_pair_count
        ):
            raise ValueError("completed decisive pair win counts must cover both games")
        if (self.descriptive_mean_completed_pair_score is None) != (self.completed_pair_count == 0):
            raise ValueError("descriptive mean exists exactly when completed pairs exist")
        return self


class NativeEvaluationFailure(_DigestedCanonicalModel):
    case_id: str | None
    category: Literal[
        "INFERENCE_FAILURE",
        "COMPONENT_UNAVAILABLE",
        "OUTCOME_NORMALIZATION_FAILURE",
        "GAMEPLAY_FAILURE",
        "PUBLICATION_FAILURE",
    ]
    error_type: str
    message: str = Field(max_length=4096)

    @model_validator(mode="after")
    def _valid_failure(self) -> NativeEvaluationFailure:
        if self.case_id is not None and (
            len(self.case_id) != 64
            or any(character not in "0123456789abcdef" for character in self.case_id)
        ):
            raise ValueError("failure case_id must be a SHA-256 digest")
        _nonempty_trimmed(self.error_type, label="failure error_type")
        _nonempty_trimmed(self.message, label="failure message")
        return self


class NativePairedEvaluationResult(_DigestedCanonicalModel):
    schema_version: Literal[1] = 1
    status: Literal["FAILED", "SUCCEEDED"]
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    config_digest: str = Field(pattern=_DIGEST_PATTERN)
    artifact_identities: tuple[NativeEvaluatedArtifactIdentity, ...] = Field(min_length=1)
    planned_case_count: StrictInt = Field(ge=0)
    attempted_case_ids: tuple[str, ...]
    observations: tuple[NativeEvaluationObservation, ...]
    completed_pairs: tuple[NativeCompletedPairScore, ...]
    aggregates: tuple[NativeComparisonAggregate, ...]
    failure: NativeEvaluationFailure | None
    evaluation_scope: Literal["DECLARED_EVALUATION_SEEDS"] = "DECLARED_EVALUATION_SEEDS"
    artifact_training_exposure: Literal["UNKNOWN"] = "UNKNOWN"
    strength_claim: Literal["DESCRIPTIVE_COMPLETED_PAIRS_ONLY"] = "DESCRIPTIVE_COMPLETED_PAIRS_ONLY"

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_is_strict(cls, value: Any) -> Any:
        return _strict_schema_version(value)

    @model_validator(mode="after")
    def _valid_status_payload(self) -> NativePairedEvaluationResult:
        tuple(
            NativeEvaluatedArtifactIdentity.model_validate(
                item.model_dump(mode="python"), strict=True
            )
            for item in self.artifact_identities
        )
        tuple(
            NativeEvaluationObservation.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.observations
        )
        tuple(
            NativeCompletedPairScore.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.completed_pairs
        )
        tuple(
            NativeComparisonAggregate.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.aggregates
        )
        if self.failure is not None:
            NativeEvaluationFailure.model_validate(
                self.failure.model_dump(mode="python"), strict=True
            )
        if (self.failure is not None) != (self.status == "FAILED"):
            raise ValueError("FAILED evaluation status exists exactly with one failure")
        if self.status == "FAILED" and (self.completed_pairs or self.aggregates):
            raise ValueError("FAILED evaluation results cannot expose strength aggregates")
        return self


class NativePairedEvaluationCompletion(_DigestedCanonicalModel):
    schema_version: Literal[1] = 1
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    result_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_is_strict(cls, value: Any) -> Any:
        return _strict_schema_version(value)


def _scope_digest_fields(
    *,
    supported_heroes: tuple[str, ...],
    supported_maps: tuple[str, ...],
    supported_game_types: tuple[str, ...],
    hero_adapter_versions: dict[str, int],
    map_schema_version: int,
) -> str:
    return _digest_mapping(
        {
            "supported_heroes": supported_heroes,
            "supported_maps": supported_maps,
            "supported_game_types": supported_game_types,
            "hero_adapter_versions": hero_adapter_versions,
            "map_schema_version": map_schema_version,
        }
    )


def _current_scope_identity() -> NativeEvaluationScopeIdentity:
    # Scope identity must not depend on whether gameplay happened to bootstrap
    # optional hero modules earlier in this process.
    register_all_effects()
    scope = current_gen1_artifact_scope()
    heroes = tuple(sorted(scope.supported_heroes))
    maps = tuple(sorted(scope.supported_maps))
    game_types = tuple(sorted(scope.supported_game_types))
    adapters = dict(sorted(scope.hero_adapter_versions.items()))
    digest = _scope_digest_fields(
        supported_heroes=heroes,
        supported_maps=maps,
        supported_game_types=game_types,
        hero_adapter_versions=adapters,
        map_schema_version=scope.map_schema_version,
    )
    return NativeEvaluationScopeIdentity(
        supported_heroes=heroes,
        supported_maps=maps,
        supported_game_types=game_types,
        hero_adapter_versions=adapters,
        map_schema_version=scope.map_schema_version,
        digest=digest,
    )


def _map_identities(
    config: NativePairedEvaluationConfig,
) -> tuple[NativeEvaluationMapIdentity, ...]:
    identities: list[NativeEvaluationMapIdentity] = []
    for map_id in sorted({fixture.map_id for fixture in config.fixtures}):
        path = _MAPS_ROOT / f"{map_id}.json"
        if not path.is_file():
            raise ValueError(f"evaluation map source is absent: {map_id!r}")
        payload = path.read_bytes()
        identities.append(
            NativeEvaluationMapIdentity(
                map_id=map_id,
                sha256=hashlib.sha256(payload).hexdigest(),
                length=len(payload),
            )
        )
    return tuple(identities)


def _pair_id(
    config: NativePairedEvaluationConfig,
    fixture: NativeEvaluationFixture,
    comparison: NativeGameplayComparisonKind,
    world_seed: int,
) -> str:
    return _digest_mapping(
        {
            "recipe": "native-gen1-paired-evaluation-pair-v1",
            "evaluation_id": config.evaluation_id,
            "comparison": comparison,
            "fixture_id": fixture.fixture_id,
            "map_id": fixture.map_id,
            "game_type": fixture.game_type,
            "red_composition": fixture.red_composition,
            "blue_composition": fixture.blue_composition,
            "world_seed": world_seed,
            "pairing_recipe": _PAIRING_RECIPE,
        }
    )


def _case_id(pair_id: str, *, candidate_side: Literal["RED", "BLUE"], leg: int) -> str:
    return _digest_mapping(
        {
            "recipe": "native-gen1-paired-evaluation-case-v1",
            "pair_id": pair_id,
            "candidate_side": candidate_side,
            "leg": leg,
        }
    )


def _planned_cases(
    config: NativePairedEvaluationConfig,
) -> tuple[NativePlannedEvaluationCase, ...]:
    cases: list[NativePlannedEvaluationCase] = []
    for comparison in config.comparisons:
        for fixture in config.fixtures:
            for seed in fixture.world_seeds:
                pair_id = _pair_id(config, fixture, comparison, seed)
                sides: tuple[Literal["RED", "BLUE"], ...] = ("RED", "BLUE")
                for leg, side in enumerate(sides):
                    cases.append(
                        NativePlannedEvaluationCase(
                            ordinal=len(cases),
                            case_id=_case_id(pair_id, candidate_side=side, leg=leg),
                            pair_id=pair_id,
                            comparison=comparison,
                            fixture_id=fixture.fixture_id,
                            world_seed=seed,
                            candidate_side=side,
                        )
                    )
    return tuple(cases)


def _derive_manifest_parts(
    config: NativePairedEvaluationConfig,
) -> tuple[
    NativeEvaluationScopeIdentity,
    tuple[NativeEvaluationMapIdentity, ...],
    tuple[NativePlannedEvaluationCase, ...],
]:
    scope = _current_scope_identity()
    supported_heroes = set(scope.supported_heroes)
    for fixture in config.fixtures:
        if fixture.map_id not in scope.supported_maps:
            raise ValueError(f"unknown current native evaluation map {fixture.map_id!r}")
        if fixture.game_type not in scope.supported_game_types:
            raise ValueError(f"unknown current native evaluation game type {fixture.game_type!r}")
        unknown = set((*fixture.red_composition, *fixture.blue_composition)) - supported_heroes
        if unknown:
            raise ValueError(f"unknown current native evaluation heroes: {sorted(unknown)!r}")
    return scope, _map_identities(config), _planned_cases(config)


def create_native_paired_evaluation_manifest(
    config: NativePairedEvaluationConfig,
    authorities: NativePairedEvaluationAuthorities,
) -> NativePairedEvaluationManifest:
    """Strictly reconstruct inputs and derive the complete fixed paired schedule."""
    validated_config = _strict_model(config, NativePairedEvaluationConfig, label="config")
    validated_authorities = _strict_model(
        authorities, NativePairedEvaluationAuthorities, label="authorities"
    )
    for label, path in (
        ("evaluation output root", validated_authorities.output_root),
        ("candidate artifact", validated_authorities.candidate_artifact_path),
    ):
        _require_no_symlink_components(path, label=label)
    if validated_authorities.parent_artifact_path is not None:
        _require_no_symlink_components(
            validated_authorities.parent_artifact_path, label="parent artifact"
        )
    parent_enabled = validated_config.parent_model_digest is not None
    if parent_enabled != (validated_authorities.parent_artifact_path is not None):
        raise ValueError("parent digest, comparison, and artifact path must be present together")
    scope, maps, planned = _derive_manifest_parts(validated_config)
    return NativePairedEvaluationManifest(
        config=validated_config,
        authorities=validated_authorities,
        current_scope=scope,
        maps=maps,
        planned_cases=planned,
    )


def _validate_observation_for_case(
    observation: NativeEvaluationObservation,
    case: NativePlannedEvaluationCase,
    manifest: NativePairedEvaluationManifest,
) -> None:
    identity = (
        observation.case_id,
        observation.pair_id,
        observation.comparison,
        observation.fixture_id,
        observation.world_seed,
        observation.candidate_side,
    )
    expected = (
        case.case_id,
        case.pair_id,
        case.comparison,
        case.fixture_id,
        case.world_seed,
        case.candidate_side,
    )
    if identity != expected:
        raise ValueError("evaluation observation identity does not match its planned case")
    if observation.steps > manifest.config.max_steps:
        raise ValueError("evaluation observation exceeds the declared max_steps budget")
    # The harness starts a new game at round 1 and defines max_rounds relative
    # to that starting counter. A max_rounds censor therefore truthfully reports
    # 1 + max_rounds when the configured limit is reached.
    if observation.rounds > manifest.config.max_rounds + 1:
        raise ValueError("evaluation observation exceeds the declared max_rounds budget")


def aggregate_native_evaluation(
    manifest: NativePairedEvaluationManifest,
    observations: Sequence[NativeEvaluationObservation],
) -> tuple[
    tuple[NativeCompletedPairScore, ...],
    tuple[NativeComparisonAggregate, ...],
]:
    """Purely aggregate one fully recorded schedule, scoring only decisive pairs."""
    validated_manifest = _strict_model(manifest, NativePairedEvaluationManifest, label="manifest")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be an explicit sequence")
    values = tuple(
        _strict_model(item, NativeEvaluationObservation, label="observation")
        for item in observations
    )
    planned = validated_manifest.planned_cases
    if len(values) != len(planned):
        raise ValueError("aggregation requires one observation for every planned case")
    for observation, case in zip(values, planned, strict=True):
        _validate_observation_for_case(observation, case, validated_manifest)

    completed_pairs: list[NativeCompletedPairScore] = []
    aggregates: list[NativeComparisonAggregate] = []
    for comparison in validated_manifest.config.comparisons:
        comparison_cases = tuple(case for case in planned if case.comparison == comparison)
        comparison_observations = tuple(
            observation for observation in values if observation.comparison == comparison
        )
        by_pair: dict[str, list[NativeEvaluationObservation]] = {}
        for observation in comparison_observations:
            by_pair.setdefault(observation.pair_id, []).append(observation)
        censored_pair_count = 0
        excluded_singletons = 0
        comparison_scores: list[NativeCompletedPairScore] = []
        for pair_id in dict.fromkeys(case.pair_id for case in comparison_cases):
            mates = by_pair[pair_id]
            if len(mates) != 2 or {mate.candidate_side for mate in mates} != {"RED", "BLUE"}:
                raise ValueError("evaluation pair does not contain its exact RED and BLUE mates")
            completed = tuple(mate for mate in mates if mate.status == "COMPLETED")
            if len(completed) != 2:
                censored_pair_count += 1
                if len(completed) == 1:
                    excluded_singletons += 1
                continue
            red = next(mate for mate in completed if mate.candidate_side == "RED")
            blue = next(mate for mate in completed if mate.candidate_side == "BLUE")
            red_won = red.winner_side == "RED"
            blue_won = blue.winner_side == "BLUE"
            score = (float(red_won) + float(blue_won)) / 2.0
            comparison_scores.append(
                NativeCompletedPairScore(
                    pair_id=pair_id,
                    comparison=comparison,
                    fixture_id=red.fixture_id,
                    world_seed=red.world_seed,
                    candidate_red_won=red_won,
                    candidate_blue_won=blue_won,
                    candidate_pair_score=score,
                )
            )
        completed_pairs.extend(comparison_scores)
        completed_cases = sum(item.status == "COMPLETED" for item in comparison_observations)
        censored_cases = len(comparison_observations) - completed_cases
        candidate_wins = sum(
            int(item.candidate_red_won) + int(item.candidate_blue_won) for item in comparison_scores
        )
        censor_reasons = Counter(
            item.reason for item in comparison_observations if item.status == "CENSORED"
        )
        aggregates.append(
            NativeComparisonAggregate(
                comparison=comparison,
                planned_pair_count=len(comparison_cases) // 2,
                attempted_case_count=len(comparison_observations),
                completed_case_count=completed_cases,
                censored_case_count=censored_cases,
                completed_pair_count=len(comparison_scores),
                censored_pair_count=censored_pair_count,
                excluded_completed_singletons=excluded_singletons,
                candidate_game_wins_in_completed_pairs=candidate_wins,
                comparator_game_wins_in_completed_pairs=(
                    2 * len(comparison_scores) - candidate_wins
                ),
                descriptive_mean_completed_pair_score=(
                    float(
                        sum(item.candidate_pair_score for item in comparison_scores)
                        / len(comparison_scores)
                    )
                    if comparison_scores
                    else None
                ),
                censor_reasons=dict(sorted(censor_reasons.items())),
            )
        )
    return tuple(completed_pairs), tuple(aggregates)


def _validate_artifact_identities(
    manifest: NativePairedEvaluationManifest,
    identities: tuple[NativeEvaluatedArtifactIdentity, ...],
) -> None:
    expected_roles = (
        ("CANDIDATE", "PARENT")
        if manifest.config.parent_model_digest is not None
        else ("CANDIDATE",)
    )
    if tuple(item.role for item in identities) != expected_roles:
        raise ValueError("evaluation artifact identities do not match configured roles")
    decision_digest = TensorFeatureSchema.current().digest
    stable_value_digest = StableValueTensorSchema.current().digest
    for identity in identities:
        expected_digest = (
            manifest.config.candidate_model_digest
            if identity.role == "CANDIDATE"
            else manifest.config.parent_model_digest
        )
        if (
            identity.model_digest != expected_digest
            or identity.model_id != GEN1_ARCHITECTURE_ID
            or identity.runtime_compatibility_version != GEN1_RUNTIME_COMPATIBILITY_VERSION
            or identity.decision_tensor_schema_digest != decision_digest
            or identity.stable_value_tensor_schema_digest != stable_value_digest
            or identity.value_semantics != "stable-boundary-outcome-v1"
            or identity.current_scope_digest != manifest.current_scope.digest
        ):
            raise ValueError("evaluation artifact identity is incompatible or forged")


def validate_native_evaluation_result(
    manifest: NativePairedEvaluationManifest,
    result: NativePairedEvaluationResult,
) -> NativePairedEvaluationResult:
    """Strictly validate runner/loader evidence against its immutable manifest."""
    validated_manifest = _strict_model(manifest, NativePairedEvaluationManifest, label="manifest")
    validated_result = _strict_model(result, NativePairedEvaluationResult, label="result")
    if (
        validated_result.manifest_digest != validated_manifest.digest
        or validated_result.config_digest != validated_manifest.config.digest
        or validated_result.planned_case_count != len(validated_manifest.planned_cases)
    ):
        raise ValueError("evaluation result identity or planned budget differs from manifest")
    _validate_artifact_identities(validated_manifest, validated_result.artifact_identities)

    planned_ids = tuple(case.case_id for case in validated_manifest.planned_cases)
    attempted = validated_result.attempted_case_ids
    if attempted != planned_ids[: len(attempted)]:
        raise ValueError("evaluation attempted case IDs must be an exact planned prefix")
    if len(attempted) > len(planned_ids):
        raise ValueError("evaluation attempted case count exceeds the plan")
    for index, observation in enumerate(validated_result.observations):
        if index >= len(attempted):
            raise ValueError("evaluation observation exists without an attempted case")
        _validate_observation_for_case(
            observation,
            validated_manifest.planned_cases[index],
            validated_manifest,
        )

    if validated_result.status == "SUCCEEDED":
        if attempted != planned_ids or len(validated_result.observations) != len(planned_ids):
            raise ValueError("SUCCEEDED evaluation must record the exact complete schedule")
        expected_pairs, expected_aggregates = aggregate_native_evaluation(
            validated_manifest, validated_result.observations
        )
        if validated_result.completed_pairs != expected_pairs:
            raise ValueError("evaluation completed pair scores are not derivable")
        if validated_result.aggregates != expected_aggregates:
            raise ValueError("evaluation aggregates are not derivable")
    else:
        failure = validated_result.failure
        assert failure is not None
        observed_count = len(validated_result.observations)
        if failure.case_id is None:
            if failure.category != "PUBLICATION_FAILURE" or observed_count != len(attempted):
                raise ValueError("case-free evaluation failure must describe publication")
        elif (
            not attempted
            or failure.case_id != attempted[-1]
            or observed_count != len(attempted) - 1
        ):
            raise ValueError("evaluation failure must name the attempted unrecorded case")
    return validated_result


def _require_canonical_file(path: Path, model: type[_ModelT]) -> _ModelT:
    _require_no_symlink_components(path, label=f"native evaluation {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"native evaluation file is missing: {path}")
    payload = path.read_bytes()
    try:
        value = model.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid native evaluation {path.name}: {exc}") from exc
    if not isinstance(value, _CanonicalModel):  # pragma: no cover - internal type guard
        raise TypeError("native evaluation file model is not canonical")
    if payload != value.canonical_bytes():
        raise ValueError(f"native evaluation {path.name} is not canonical JSON")
    return value


def load_native_paired_evaluation_manifest(
    path: str | Path,
) -> NativePairedEvaluationManifest:
    value = _require_canonical_file(Path(path), NativePairedEvaluationManifest)
    assert isinstance(value, NativePairedEvaluationManifest)
    for label, authority in (
        ("evaluation output root", value.authorities.output_root),
        ("candidate artifact", value.authorities.candidate_artifact_path),
    ):
        _require_no_symlink_components(authority, label=label)
    if value.authorities.parent_artifact_path is not None:
        _require_no_symlink_components(
            value.authorities.parent_artifact_path, label="parent artifact"
        )
    return value


def load_native_paired_evaluation_result(path: str | Path) -> NativePairedEvaluationResult:
    value = _require_canonical_file(Path(path), NativePairedEvaluationResult)
    assert isinstance(value, NativePairedEvaluationResult)
    return value


def load_completed_native_paired_evaluation(
    root: str | Path,
) -> tuple[NativePairedEvaluationManifest, NativePairedEvaluationResult]:
    """Load only canonical, digest-bound success evidence for the exact current maps."""
    path = Path(root)
    _require_no_symlink_components(path, label="native completed evaluation root")
    if not path.is_dir():
        raise FileNotFoundError(f"native completed evaluation root is missing: {path}")
    manifest = load_native_paired_evaluation_manifest(path / "manifest.json")
    if manifest.authorities.output_root != path.absolute():
        raise ValueError("completed evaluation root differs from its output authority")
    result = load_native_paired_evaluation_result(path / "result.json")
    result = validate_native_evaluation_result(manifest, result)
    if result.status != "SUCCEEDED":
        raise ValueError("native paired evaluation is not succeeded")
    completion = _require_canonical_file(path / "complete.json", NativePairedEvaluationCompletion)
    assert isinstance(completion, NativePairedEvaluationCompletion)
    if completion.manifest_digest != manifest.digest or completion.result_digest != result.digest:
        raise ValueError("native evaluation completion marker digest binding is stale")
    return manifest, result


__all__ = [
    "NativeComparisonAggregate",
    "NativeCompletedPairScore",
    "NativeEvaluatedArtifactIdentity",
    "NativeEvaluationFailure",
    "NativeEvaluationFixture",
    "NativeEvaluationMapIdentity",
    "NativeEvaluationObservation",
    "NativeEvaluationScopeIdentity",
    "NativeGameplayComparisonKind",
    "NativePairedEvaluationAuthorities",
    "NativePairedEvaluationCompletion",
    "NativePairedEvaluationConfig",
    "NativePairedEvaluationManifest",
    "NativePairedEvaluationResult",
    "NativePlannedEvaluationCase",
    "aggregate_native_evaluation",
    "create_native_paired_evaluation_manifest",
    "load_completed_native_paired_evaluation",
    "load_native_paired_evaluation_manifest",
    "load_native_paired_evaluation_result",
    "validate_native_evaluation_result",
]
