"""Strict contracts for one finite, non-resumable native Gen1 run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from automata.models.contracts import canonical_json_bytes
from automata.search.config import SearchConfig
from automata.search.contracts import CutoffUnit, LeafMode
from automata.training.native_dataset import native_game_id
from automata.training.native_generation import (
    NativeGenerationConfig,
    NativeGenerationGame,
    NativeTeacherKind,
)
from automata.training.native_replay import NativeReplayConfig
from automata.training.native_splits import (
    NativeSeedSplitLedger,
    NativeSplitConfig,
    NativeSplitName,
    create_native_split_ledger,
    extend_native_split_ledger,
    native_seed_purpose,
)
from automata.training.native_trainer import (
    NativeTrainerConfig,
    NativeTrainerInitialization,
    NativeTrainingStepProvenance,
)
from automata.training.native_validation import NativeValidationMetrics

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_FLOAT_FIELDS = (
    "uct_c",
    "widening_c",
    "widening_alpha",
    "puct_c",
)
_OPTIONAL_FLOAT_FIELDS = (
    "decision_timeout_seconds",
    "root_widening_c",
    "root_widening_alpha",
    "root_puct_c",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class _CanonicalModel(_FrozenModel):
    def canonical_bytes(self) -> bytes:
        validated = type(self).model_validate(self.model_dump(mode="python"), strict=True)
        return canonical_json_bytes(validated)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _nonempty_trimmed(value: str, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_value(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve(strict=False)
    resolved_right = right.resolve(strict=False)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


class NativeRunSearchConfig(_FrozenModel):
    """Serializable, explicit mirror of every ``SearchConfig`` field."""

    iterations: StrictInt = Field(gt=0)
    decision_timeout_seconds: float | None
    max_advance_steps: StrictInt = Field(gt=0)
    uct_c: float = Field(ge=0.0)
    cutoff_limit: StrictInt = Field(ge=0)
    cutoff_unit: CutoffUnit
    max_advance_transitions: StrictInt = Field(gt=0)
    max_forced_decisions: StrictInt = Field(gt=0)
    leaf_mode: LeafMode
    widening_c: float = Field(gt=0.0)
    widening_alpha: float = Field(ge=0.0, le=1.0)
    root_widening_c: float | None
    root_widening_alpha: float | None
    adaptive_hex_root_schedule_version: StrictInt | None
    request_schedule_version: StrictInt | None
    seed: StrictInt
    use_prior: bool
    puct_c: float = Field(ge=0.0)
    root_puct_c: float | None

    @field_validator(*_FLOAT_FIELDS, mode="before")
    @classmethod
    def _strict_finite_float(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("native run search floating values must be strict finite floats")
        return value

    @field_validator(*_OPTIONAL_FLOAT_FIELDS, mode="before")
    @classmethod
    def _strict_finite_optional_float(cls, value: Any) -> Any:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError(
                "native run optional search floating values must be strict finite floats or None"
            )
        return value

    @model_validator(mode="after")
    def _valid_offline_search(self) -> NativeRunSearchConfig:
        if self.seed != 0:
            raise ValueError("native run search seed must be zero")
        if self.decision_timeout_seconds is not None:
            raise ValueError("native run search timeout must be None")
        if self.request_schedule_version is not None:
            raise ValueError("native run search request schedule must be None")
        if self.leaf_mode is not LeafMode.STABLE_TRANSITION:
            raise ValueError("native run search leaf mode must be STABLE_TRANSITION")
        # SearchConfig owns the remaining cross-field constraints.
        self._build_search_config()
        return self

    def _build_search_config(self) -> SearchConfig:
        return SearchConfig(**self.model_dump(mode="python"))

    def to_search_config(self) -> SearchConfig:
        validated = NativeRunSearchConfig.model_validate(
            self.model_dump(mode="python"), strict=True
        )
        return validated._build_search_config()


def _generation_config(config: NativeRunConfig) -> NativeGenerationConfig:
    return NativeGenerationConfig(
        generation_id=config.generation_id,
        source_revision=config.source_revision,
        dirty_tree_hash=config.dirty_tree_hash,
        teacher_kind=config.teacher_kind,
        source_model_digest=config.source_model_digest,
        search_config=config.search.to_search_config(),
        split_config=config.split_config,
        random_stream_namespace=config.random_stream_namespace,
        visit_temperature=config.visit_temperature,
        max_steps=config.max_steps,
        max_rounds=config.max_rounds,
    )


class NativeRunConfig(_CanonicalModel):
    schema_version: Literal[1] = 1
    run_id: str
    generation_id: str
    source_revision: str
    dirty_tree_hash: str
    teacher_kind: NativeTeacherKind
    source_model_digest: str | None = Field(pattern=_DIGEST_PATTERN)
    search: NativeRunSearchConfig
    split_config: NativeSplitConfig
    random_stream_namespace: str
    visit_temperature: float = Field(ge=0.0)
    max_steps: StrictInt = Field(gt=0)
    max_rounds: StrictInt | None = Field(gt=0)
    games: tuple[NativeGenerationGame, ...] = Field(min_length=1)
    replay_config: NativeReplayConfig
    trainer_config: NativeTrainerConfig
    initialization: NativeTrainerInitialization
    optimizer_steps: StrictInt = Field(gt=0)
    games_per_step: StrictInt = Field(gt=0)
    sampling_seeds: tuple[StrictInt, ...]
    index_chunk_size: StrictInt = Field(gt=0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("visit_temperature", mode="before")
    @classmethod
    def _strict_temperature(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("visit_temperature must be a strict finite float")
        return value

    @model_validator(mode="after")
    def _valid_run_config(self) -> NativeRunConfig:
        for label in (
            "run_id",
            "generation_id",
            "source_revision",
            "dirty_tree_hash",
            "random_stream_namespace",
        ):
            _nonempty_trimmed(getattr(self, label), label=label)
        # Reconstruct all nested public inputs, including unsafe model copies.
        NativeRunSearchConfig.model_validate(self.search.model_dump(mode="python"), strict=True)
        NativeSplitConfig.model_validate(self.split_config.model_dump(mode="python"), strict=True)
        NativeReplayConfig.model_validate(self.replay_config.model_dump(mode="python"), strict=True)
        NativeTrainerConfig.model_validate(
            self.trainer_config.model_dump(mode="python"), strict=True
        )
        NativeTrainerInitialization.model_validate(
            self.initialization.model_dump(mode="python"), strict=True
        )
        games = tuple(
            NativeGenerationGame.model_validate(game.model_dump(mode="python"), strict=True)
            for game in self.games
        )
        generation = _generation_config(self)
        if self.replay_config.split_config != self.split_config:
            raise ValueError("replay config split_config must equal the run split_config")
        parent_mode = self.teacher_kind == "GEN1_PARENT"
        if parent_mode:
            if (
                self.initialization.mode != "GEN1_PARENT"
                or self.initialization.parent_model_digest != self.source_model_digest
            ):
                raise ValueError("GEN1_PARENT teacher and initialization must use one digest")
        elif (
            self.initialization.mode != "FRESH_BOOTSTRAP"
            or self.source_model_digest is not None
            or self.initialization.parent_model_digest is not None
        ):
            raise ValueError(
                "HEURISTIC_BOOTSTRAP requires fresh initialization and no parent digest"
            )
        if len(self.sampling_seeds) != self.optimizer_steps:
            raise ValueError("sampling_seeds length must equal optimizer_steps")

        ledger = extend_native_split_ledger(
            create_native_split_ledger(self.split_config),
            (game.world_seed for game in games),
        )
        split_by_seed = {
            assignment.world_seed: assignment.split for assignment in ledger.assignments
        }
        for game in games:
            declared = native_seed_purpose(self.split_config, game.world_seed)
            if declared != game.seed_purpose:
                raise ValueError("planned game seed_purpose does not match split configuration")
        expected_ids = tuple(_expected_game_id(game, generation) for game in games)
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("planned native game identities must be unique")
        train_count = sum(split_by_seed[game.world_seed] == "train" for game in games)
        validation_count = len(games) - train_count
        if train_count == 0 or validation_count == 0:
            raise ValueError("native run requires nonempty TRAIN and validation cohorts")
        if self.games_per_step > train_count:
            raise ValueError("games_per_step cannot exceed the planned TRAIN game count")
        if self.replay_config.max_games is None:
            raise ValueError("native run replay max_games must be explicitly bounded")
        if self.replay_config.max_games < train_count:
            raise ValueError("native run replay max_games must retain the full TRAIN cohort")
        return self


def _expected_game_id(game: NativeGenerationGame, generation: NativeGenerationConfig) -> str:
    return native_game_id(
        world_seed=game.world_seed,
        map_id=game.map_id,
        game_type=game.game_type,
        red_composition=game.red_composition,
        blue_composition=game.blue_composition,
        generation_id=generation.generation_id,
        source_revision=generation.source_revision,
        dirty_tree_hash=generation.dirty_tree_hash,
        source_model_digest=generation.source_model_digest,
        search_config_id=generation.search_config_id,
        generator_config_id=generation.generator_config_id,
    )


class NativeRunAuthorities(_FrozenModel):
    output_root: Path
    parent_artifact_path: Path | None

    @field_validator("output_root")
    @classmethod
    def _absolute_output_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("native run output_root must be absolute")
        if ".." in value.parts or value != Path(os.path.abspath(value)):
            raise ValueError("native run output_root must be lexically normalized")
        return value

    @field_validator("parent_artifact_path")
    @classmethod
    def _absolute_parent(cls, value: Path | None) -> Path | None:
        if value is not None:
            if not value.is_absolute():
                raise ValueError("native run parent artifact path must be absolute")
            if ".." in value.parts or value != Path(os.path.abspath(value)):
                raise ValueError("native run parent artifact path must be lexically normalized")
        return value

    @model_validator(mode="after")
    def _disjoint_authorities(self) -> NativeRunAuthorities:
        if self.parent_artifact_path is not None and _paths_overlap(
            self.output_root, self.parent_artifact_path
        ):
            raise ValueError("native run output and parent artifact paths must be disjoint")
        return self


class NativePlannedGame(_FrozenModel):
    ordinal: StrictInt = Field(ge=0)
    game: NativeGenerationGame
    split: NativeSplitName
    expected_game_id: str = Field(pattern=_DIGEST_PATTERN)
    source_logical_name: str = Field(pattern=r"^games/[0-9]{8}\.jsonl\.zst$")
    completion_relative_path: str = Field(pattern=r"^receipts/games/[0-9]{8}\.json$")

    @model_validator(mode="after")
    def _strict_nested_game(self) -> NativePlannedGame:
        NativeGenerationGame.model_validate(self.game.model_dump(mode="python"), strict=True)
        return self


class NativeRunManifest(_CanonicalModel):
    schema_version: Literal[1] = 1
    config: NativeRunConfig
    authorities: NativeRunAuthorities
    planned_split_ledger: NativeSeedSplitLedger
    planned_games: tuple[NativePlannedGame, ...] = Field(min_length=1)
    layout_version: Literal["native-one-run-layout-v1"] = "native-one-run-layout-v1"

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_derived_manifest(self) -> NativeRunManifest:
        config = NativeRunConfig.model_validate(self.config.model_dump(mode="python"), strict=True)
        authorities = NativeRunAuthorities.model_validate(
            self.authorities.model_dump(mode="python"), strict=True
        )
        expected_ledger, expected_games = _derive_plan(config)
        actual_ledger = NativeSeedSplitLedger.model_validate(
            self.planned_split_ledger.model_dump(mode="python"), strict=True
        )
        actual_games = tuple(
            NativePlannedGame.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.planned_games
        )
        if actual_ledger != expected_ledger or actual_games != expected_games:
            raise ValueError("native run manifest planned identities are not canonical")
        parent_mode = config.teacher_kind == "GEN1_PARENT"
        if parent_mode != (authorities.parent_artifact_path is not None):
            raise ValueError("parent artifact path must exist exactly for GEN1_PARENT mode")
        return self


def _derive_plan(
    config: NativeRunConfig,
) -> tuple[NativeSeedSplitLedger, tuple[NativePlannedGame, ...]]:
    generation = _generation_config(config)
    ledger = extend_native_split_ledger(
        create_native_split_ledger(config.split_config),
        (game.world_seed for game in config.games),
    )
    planned = tuple(
        NativePlannedGame(
            ordinal=ordinal,
            game=game,
            split=ledger.split_for_seed(game.world_seed),
            expected_game_id=_expected_game_id(game, generation),
            source_logical_name=f"games/{ordinal:08d}.jsonl.zst",
            completion_relative_path=f"receipts/games/{ordinal:08d}.json",
        )
        for ordinal, game in enumerate(config.games)
    )
    return ledger, planned


class NativeTrainingStepRecord(_FrozenModel):
    optimizer_step: StrictInt = Field(gt=0)
    policy_cross_entropy: float
    policy_entropy: float
    value_bce: float
    regularization: float
    total_loss: float
    gradient_norm_before_clip: float
    provenance: NativeTrainingStepProvenance

    @model_validator(mode="after")
    def _strict_provenance(self) -> NativeTrainingStepRecord:
        NativeTrainingStepProvenance.model_validate(
            self.provenance.model_dump(mode="python"), strict=True
        )
        return self

    @field_validator(
        "policy_cross_entropy",
        "policy_entropy",
        "value_bce",
        "regularization",
        "total_loss",
        "gradient_norm_before_clip",
        mode="before",
    )
    @classmethod
    def _finite_step_scalar(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("native training step scalars must be strict finite floats")
        return value


class NativeRunTrainingStep(_FrozenModel):
    sampling_seed: StrictInt
    selected_game_ids: tuple[str, ...] = Field(min_length=1)
    result: NativeTrainingStepRecord

    @model_validator(mode="after")
    def _valid_selection(self) -> NativeRunTrainingStep:
        result = NativeTrainingStepRecord.model_validate(
            self.result.model_dump(mode="python"), strict=True
        )
        if (
            len(set(self.selected_game_ids)) != len(self.selected_game_ids)
            or any(not _is_digest(game_id) for game_id in self.selected_game_ids)
            or result.provenance.selected_game_ids != self.selected_game_ids
        ):
            raise ValueError("native run step selection and provenance must agree uniquely")
        return self


class NativeRunProducts(_FrozenModel):
    validation_scope: Literal["CURRENT_RUN_UPDATES"] = "CURRENT_RUN_UPDATES"
    parent_training_exposure: Literal["NONE_FRESH_INITIALIZATION", "UNKNOWN"]
    generated_game_ids: tuple[str, ...] = Field(min_length=1)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    catalog_digest: str = Field(pattern=_DIGEST_PATTERN)
    validation_ledger_digest: str = Field(pattern=_DIGEST_PATTERN)
    initial_validation: NativeValidationMetrics
    training_steps: tuple[NativeRunTrainingStep, ...] = Field(min_length=1)
    final_validation: NativeValidationMetrics
    artifact_model_digest: str = Field(pattern=_DIGEST_PATTERN)
    artifact_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _valid_products(self) -> NativeRunProducts:
        if len(set(self.generated_game_ids)) != len(self.generated_game_ids) or any(
            not _is_digest(game_id) for game_id in self.generated_game_ids
        ):
            raise ValueError("generated game IDs must be unique SHA-256 digests")
        initial = NativeValidationMetrics.model_validate(
            self.initial_validation.model_dump(mode="python"), strict=True
        )
        final = NativeValidationMetrics.model_validate(
            self.final_validation.model_dump(mode="python"), strict=True
        )
        identity_fields = (
            "validation_scope",
            "validation_ledger_digest",
            "game_ids",
            "dataset_digests",
            "source_digests",
            "completion_receipt_digests",
            "game_count",
            "policy_contributing_game_count",
            "policy_row_count",
            "value_contributing_game_count",
            "value_row_count",
        )
        if any(getattr(initial, name) != getattr(final, name) for name in identity_fields):
            raise ValueError("initial and final validation physical identities must match")
        if initial.validation_ledger_digest != self.validation_ledger_digest:
            raise ValueError("validation metrics do not match the product ledger digest")
        steps = tuple(
            NativeRunTrainingStep.model_validate(step.model_dump(mode="python"), strict=True)
            for step in self.training_steps
        )
        for expected, step in enumerate(steps, start=1):
            if step.result.optimizer_step != expected:
                raise ValueError("native run training steps must be contiguous")
        return self


class NativeRunProgress(_FrozenModel):
    phase: Literal[
        "CLAIMED",
        "GENERATING",
        "INDEXING",
        "REPLAY",
        "INITIAL_VALIDATION",
        "TRAINING",
        "FINAL_VALIDATION",
        "EXPORTING",
        "VERIFYING",
        "COMPLETE",
    ]
    attempted_game_ids: tuple[str, ...] = ()
    uncertified_game_ids: tuple[str, ...] = ()
    completed_game_ids: tuple[str, ...] = ()
    optimizer_steps_completed: StrictInt = Field(ge=0)
    training_steps: tuple[NativeRunTrainingStep, ...] = ()

    @model_validator(mode="after")
    def _truthful_progress(self) -> NativeRunProgress:
        for label, values in (
            ("attempted", self.attempted_game_ids),
            ("uncertified", self.uncertified_game_ids),
            ("completed", self.completed_game_ids),
        ):
            if len(set(values)) != len(values) or any(
                not _is_digest(game_id) for game_id in values
            ):
                raise ValueError(f"{label} game IDs must be unique SHA-256 digests")
        attempted = set(self.attempted_game_ids)
        if not set(self.completed_game_ids) <= attempted:
            raise ValueError("completed games must have been attempted")
        if not set(self.uncertified_game_ids) <= attempted:
            raise ValueError("uncertified games must have been attempted")
        if set(self.completed_game_ids) & set(self.uncertified_game_ids):
            raise ValueError("a game cannot be both completed and uncertified")
        steps = tuple(
            NativeRunTrainingStep.model_validate(step.model_dump(mode="python"), strict=True)
            for step in self.training_steps
        )
        if self.optimizer_steps_completed != len(steps):
            raise ValueError("optimizer progress must retain every completed step diagnostic")
        for expected, step in enumerate(steps, start=1):
            if step.result.optimizer_step != expected:
                raise ValueError("progress training steps must be contiguous")
        return self


class NativeRunFailure(_FrozenModel):
    phase: str
    error_type: str
    message: str

    @model_validator(mode="after")
    def _valid_failure(self) -> NativeRunFailure:
        _nonempty_trimmed(self.phase, label="failure phase")
        _nonempty_trimmed(self.error_type, label="failure error_type")
        _nonempty_trimmed(self.message, label="failure message")
        return self


class NativeRunResult(_CanonicalModel):
    schema_version: Literal[1] = 1
    status: Literal["RUNNING", "FAILED", "SUCCEEDED"]
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    config_digest: str = Field(pattern=_DIGEST_PATTERN)
    progress: NativeRunProgress
    products: NativeRunProducts | None
    failure: NativeRunFailure | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_status_payload(self) -> NativeRunResult:
        NativeRunProgress.model_validate(self.progress.model_dump(mode="python"), strict=True)
        if self.products is not None:
            NativeRunProducts.model_validate(self.products.model_dump(mode="python"), strict=True)
        if self.failure is not None:
            NativeRunFailure.model_validate(self.failure.model_dump(mode="python"), strict=True)
        if (self.products is not None) != (self.status == "SUCCEEDED"):
            raise ValueError("native run products exist exactly for SUCCEEDED")
        if (self.failure is not None) != (self.status == "FAILED"):
            raise ValueError("native run failure exists exactly for FAILED")
        if self.status == "SUCCEEDED" and self.progress.phase != "COMPLETE":
            raise ValueError("SUCCEEDED native run progress must be COMPLETE")
        return self


class NativeRunCompletion(_CanonicalModel):
    schema_version: Literal[1] = 1
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    result_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value


def create_native_run_manifest(
    config: NativeRunConfig,
    authorities: NativeRunAuthorities,
) -> NativeRunManifest:
    """Strictly reconstruct configuration and derive its entire finite plan."""
    if not isinstance(config, NativeRunConfig):
        raise TypeError("config must be a NativeRunConfig")
    if not isinstance(authorities, NativeRunAuthorities):
        raise TypeError("authorities must be NativeRunAuthorities")
    validated_config = NativeRunConfig.model_validate(config.model_dump(mode="python"), strict=True)
    validated_authorities = NativeRunAuthorities.model_validate(
        authorities.model_dump(mode="python"), strict=True
    )
    ledger, games = _derive_plan(validated_config)
    return NativeRunManifest(
        config=validated_config,
        authorities=validated_authorities,
        planned_split_ledger=ledger,
        planned_games=games,
    )


def _require_canonical_file(path: Path, model: type[_CanonicalModel]) -> _CanonicalModel:
    _require_no_symlink_components(path, label=f"native run {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"native run file is missing: {path}")
    payload = path.read_bytes()
    try:
        value = model.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise ValueError(f"invalid native run {path.name}: {exc}") from exc
    if payload != value.canonical_bytes():
        raise ValueError(f"native run {path.name} is not canonical JSON")
    return value


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")


def load_native_run_manifest(path: str | Path) -> NativeRunManifest:
    value = _require_canonical_file(Path(path), NativeRunManifest)
    assert isinstance(value, NativeRunManifest)
    return value


def load_native_run_result(path: str | Path) -> NativeRunResult:
    value = _require_canonical_file(Path(path), NativeRunResult)
    assert isinstance(value, NativeRunResult)
    return value


def _validate_completed_relationships(manifest: NativeRunManifest, result: NativeRunResult) -> None:
    config = manifest.config
    if result.status != "SUCCEEDED" or result.products is None:
        raise ValueError("native run is not succeeded")
    products = result.products
    planned_ids = tuple(item.expected_game_id for item in manifest.planned_games)
    if (
        result.manifest_digest != manifest.digest
        or result.config_digest != config.digest
        or result.progress.phase != "COMPLETE"
        or result.progress.attempted_game_ids != planned_ids
        or result.progress.completed_game_ids != planned_ids
        or result.progress.uncertified_game_ids
        or products.generated_game_ids != planned_ids
        or result.progress.optimizer_steps_completed != config.optimizer_steps
        or result.progress.training_steps != products.training_steps
        or len(products.training_steps) != config.optimizer_steps
    ):
        raise ValueError("completed native run result does not match its manifest budgets")
    expected_exposure = (
        "NONE_FRESH_INITIALIZATION"
        if config.initialization.mode == "FRESH_BOOTSTRAP"
        else "UNKNOWN"
    )
    if products.parent_training_exposure != expected_exposure:
        raise ValueError("completed native run parent exposure label is invalid")
    validation_ids = tuple(
        item.expected_game_id for item in manifest.planned_games if item.split == "validation"
    )
    if (
        products.initial_validation.game_ids != validation_ids
        or products.initial_validation.dataset_digests != (products.dataset_digest,)
        or products.initial_validation.source_digests != (products.source_receipt_digest,)
        or products.initial_validation.completion_receipt_digests
        != (products.completion_receipt_digest,)
    ):
        raise ValueError("completed native run validation authorities differ from products")
    train_ids = {item.expected_game_id for item in manifest.planned_games if item.split == "train"}
    for index, step in enumerate(products.training_steps):
        if (
            step.sampling_seed != config.sampling_seeds[index]
            or len(step.selected_game_ids) != config.games_per_step
            or not set(step.selected_game_ids) <= train_ids
            or step.result.provenance.replay_catalog_digest != products.catalog_digest
            or step.result.provenance.dataset_digests != (products.dataset_digest,)
            or step.result.provenance.source_digests != (products.source_receipt_digest,)
            or step.result.provenance.completion_receipt_digests
            != (products.completion_receipt_digest,)
        ):
            raise ValueError("completed native run training budget differs from its manifest")


def _validate_completed_physical_products(
    root: Path,
    manifest: NativeRunManifest,
    result: NativeRunResult,
) -> None:
    """Reopen every durable authority; success files alone are not sufficient."""
    assert result.products is not None
    config = manifest.config
    products = result.products
    from automata.training.native_gen1 import load_current_gen1_parent_artifact
    from automata.training.native_indexed_dataset import load_native_source_receipt
    from automata.training.native_receipts import load_native_dataset_completion_receipt
    from automata.training.native_replay import load_native_replay_catalog
    from automata.training.native_trainer import (
        NativeReplayDatasetBinding,
        open_bound_native_dataset,
    )
    from automata.training.native_validation import (
        NativeValidationLedger,
        create_native_validation_ledger,
    )

    completion_path = root / "receipts" / "completion-set.json"
    source_path = root / "receipts" / "source-inventory.json"
    completion = load_native_dataset_completion_receipt(completion_path)
    source = load_native_source_receipt(source_path)
    binding = NativeReplayDatasetBinding(
        dataset_digest=products.dataset_digest,
        source_digest=products.source_receipt_digest,
        completion_receipt_digest=products.completion_receipt_digest,
        source_root=root,
        source_receipt_path=source_path,
        completion_receipt_path=completion_path,
        index_cache_dir=root / "index",
        chunk_size=manifest.config.index_chunk_size,
    )
    dataset = open_bound_native_dataset(binding, rebuild_index=False)
    catalog = load_native_replay_catalog(root / "replay" / "catalog.json")
    ledger_path = root / "receipts" / "validation-ledger.json"
    ledger_payload = ledger_path.read_bytes()
    ledger = NativeValidationLedger.model_validate_json(ledger_payload, strict=True)
    if ledger_payload != ledger.canonical_bytes():
        raise ValueError("completed native run validation ledger is not canonical JSON")
    derived_ledger = create_native_validation_ledger(
        manifest.planned_split_ledger,
        datasets=(dataset,),
        completion_receipts=(completion,),
    )
    artifact = load_current_gen1_parent_artifact(
        root / "artifact", expected_model_digest=products.artifact_model_digest
    )
    artifact_manifest_bytes = (root / "artifact" / "manifest.json").read_bytes()
    provenance_path = root / "artifact" / "provenance.json"
    try:
        provenance_bytes = provenance_path.read_bytes()
        provenance = json.loads(provenance_bytes)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("completed native run artifact provenance is invalid") from exc
    if not isinstance(provenance, dict) or provenance_bytes != _canonical_json_value(provenance):
        raise ValueError("completed native run artifact provenance is not canonical JSON")
    optimizer = provenance.get("optimizer")
    expected_successful_steps = [
        step.result.model_dump(mode="json") for step in products.training_steps
    ]
    if (
        _canonical_json_value(provenance.get("trainer_config"))
        != _canonical_json_value(config.trainer_config.model_dump(mode="json"))
        or _canonical_json_value(provenance.get("initialization"))
        != _canonical_json_value(config.initialization.model_dump(mode="json"))
        or not isinstance(optimizer, dict)
        or type(optimizer.get("step_count")) is not int
        or optimizer.get("step_count") != config.optimizer_steps
        or _canonical_json_value(provenance.get("successful_steps"))
        != _canonical_json_value(expected_successful_steps)
    ):
        raise ValueError("completed native run artifact provenance differs from run products")
    if (
        completion.digest != products.completion_receipt_digest
        or source.digest != products.source_receipt_digest
        or dataset.digest != products.dataset_digest
        or catalog.digest != products.catalog_digest
        or catalog.split_ledger != manifest.planned_split_ledger
        or ledger != derived_ledger
        or ledger.digest != products.validation_ledger_digest
        or artifact.manifest.model_digest != products.artifact_model_digest
        or hashlib.sha256(artifact_manifest_bytes).hexdigest() != products.artifact_manifest_digest
    ):
        raise ValueError("completed native run physical products differ from terminal result")


def load_completed_native_run(
    root: str | Path,
) -> tuple[NativeRunManifest, NativeRunResult]:
    """Load only a digest-bound canonical success marker and matching terminal state."""
    path = Path(root)
    _require_no_symlink_components(path, label="native completed run root")
    if not path.is_dir():
        raise FileNotFoundError(f"native completed run root is missing: {path}")
    manifest = load_native_run_manifest(path / "manifest.json")
    if manifest.authorities.output_root != path.absolute():
        raise ValueError("completed run root does not match manifest output authority")
    result = load_native_run_result(path / "result.json")
    completion_value = _require_canonical_file(path / "complete.json", NativeRunCompletion)
    assert isinstance(completion_value, NativeRunCompletion)
    _validate_completed_relationships(manifest, result)
    if (
        completion_value.manifest_digest != manifest.digest
        or completion_value.result_digest != result.digest
    ):
        raise ValueError("native run completion marker digest binding is stale")
    _validate_completed_physical_products(path.absolute(), manifest, result)
    return manifest, result


__all__ = [
    "NativePlannedGame",
    "NativeRunAuthorities",
    "NativeRunCompletion",
    "NativeRunConfig",
    "NativeRunFailure",
    "NativeRunManifest",
    "NativeRunProducts",
    "NativeRunProgress",
    "NativeRunResult",
    "NativeRunSearchConfig",
    "NativeRunTrainingStep",
    "NativeTrainingStepRecord",
    "create_native_run_manifest",
    "load_completed_native_run",
    "load_native_run_manifest",
    "load_native_run_result",
]
