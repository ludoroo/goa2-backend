"""Receipt-bound replay training for one native Gen1 logical batch.

This module deliberately provides one optimizer update rather than an epoch or
checkpoint loop.  Dropout is constrained to zero in this first trainer contract,
so a training step consumes no process-global RNG state.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)
from torch import Tensor

from automata.models.contracts import canonical_json_bytes
from automata.models.shared_encoder.artifacts import (
    Gen1ModelArtifactManifest,
    export_gen1_model_artifact,
)
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.training.native_indexed_dataset import (
    IndexedNativeDataset,
    create_native_source_receipt_from_completions,
    load_native_source_receipt,
    open_native_indexed_dataset,
)
from automata.training.native_losses import native_policy_loss, native_stable_value_loss
from automata.training.native_receipts import (
    load_native_dataset_completion_receipt,
    validate_native_dataset_completion,
)
from automata.training.native_replay import (
    NativeReplayCatalog,
    NativeReplayGameRef,
    NativeReplaySample,
)

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_FLOAT_FIELDS = (
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "policy_weight",
    "value_weight",
    "entropy_weight",
    "l2_weight",
    "max_gradient_norm",
    "dropout",
)


class NativeTrainerConfig(BaseModel):
    """Complete immutable recipe for one-step native Gen1 optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    seed: StrictInt = Field(ge=0)
    learning_rate: float = Field(gt=0)
    adam_beta1: float = Field(gt=0, lt=1)
    adam_beta2: float = Field(gt=0, lt=1)
    adam_epsilon: float = Field(gt=0)
    policy_weight: float = Field(ge=0)
    value_weight: float = Field(ge=0)
    entropy_weight: float = Field(ge=0)
    l2_weight: float = Field(ge=0)
    max_gradient_norm: float = Field(gt=0)
    token_width: StrictInt = Field(gt=0)
    state_width: StrictInt = Field(gt=0)
    candidate_width: StrictInt = Field(gt=0)
    message_passing_layers: StrictInt = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator(*_FLOAT_FIELDS, mode="before")
    @classmethod
    def _strict_finite_float(cls, value: Any) -> Any:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("trainer floating configuration values must be strict finite floats")
        return value

    @model_validator(mode="after")
    def _valid_objective(self) -> NativeTrainerConfig:
        if self.policy_weight == 0.0 and self.value_weight == 0.0:
            raise ValueError("at least one of policy_weight or value_weight must be positive")
        if self.dropout != 0.0:
            raise ValueError("native trainer currently requires dropout=0 for RNG isolation")
        return self

    def _revalidated(self) -> NativeTrainerConfig:
        return NativeTrainerConfig.model_validate(self.model_dump(mode="python"), strict=True)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._revalidated())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class NativeTrainerInitialization(BaseModel):
    """Weight initialization identity; optimizer state is never resumed."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    mode: Literal["FRESH_BOOTSTRAP", "GEN1_PARENT"]
    parent_model_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _valid_parent(self) -> NativeTrainerInitialization:
        if (self.parent_model_digest is not None) != (self.mode == "GEN1_PARENT"):
            raise ValueError("parent_model_digest is required exactly for GEN1_PARENT")
        return self


class NativeReplayDatasetBinding(BaseModel):
    """Explicit physical authority for one replay dataset digest.

    Authority paths must be absolute physical paths: symlinks in any ancestor
    component are rejected so aliases such as ``/tmp`` for ``/private/tmp``
    are not interchangeable controlled authorities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_root: Path
    source_receipt_path: Path
    completion_receipt_path: Path
    index_cache_dir: Path
    chunk_size: StrictInt = Field(gt=0)

    @field_validator(
        "source_root",
        "source_receipt_path",
        "completion_receipt_path",
        "index_cache_dir",
    )
    @classmethod
    def _absolute_authority_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("native replay binding authority paths must be absolute")
        return value


@dataclass(frozen=True, slots=True)
class BoundNativeReplayGame:
    reference: NativeReplayGameRef
    dataset: IndexedNativeDataset


@dataclass(frozen=True, slots=True)
class BoundNativeReplaySample:
    catalog_digest: str
    sample: NativeReplaySample
    games: tuple[BoundNativeReplayGame, ...]
    catalog: NativeReplayCatalog
    bindings: tuple[NativeReplayDatasetBinding, ...]


class NativeTrainingStepProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    replay_catalog_digest: str = Field(pattern=_DIGEST_PATTERN)
    selected_game_ids: tuple[str, ...]
    dataset_digests: tuple[str, ...]
    source_digests: tuple[str, ...]
    completion_receipt_digests: tuple[str, ...]
    policy_contributing_game_count: StrictInt = Field(ge=0)
    value_contributing_game_count: StrictInt = Field(ge=0)


@dataclass(frozen=True, slots=True)
class NativeTrainingStepResult:
    """One successful update; regularization is zero when L2 is disabled."""

    optimizer_step: int
    policy_cross_entropy: float
    policy_entropy: float
    value_bce: float
    regularization: float
    total_loss: float
    gradient_norm_before_clip: float
    provenance: NativeTrainingStepProvenance


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _strict_model(value: _ModelT, expected: type[_ModelT], *, label: str) -> _ModelT:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}")
    return expected.model_validate(value.model_dump(mode="python"), strict=True)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve(strict=False)
    resolved_right = right.resolve(strict=False)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


def _require_no_symlink_components(path: Path) -> None:
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("native dataset binding authority paths must not contain symlinks")


def open_bound_native_dataset(
    binding: NativeReplayDatasetBinding,
    *,
    rebuild_index: bool = False,
) -> IndexedNativeDataset:
    """Strictly bind completion, inventory, source bytes, and index identity.

    ``rebuild_index=False`` is read-only/fail-closed. ``True`` may rebuild only
    the binding's disposable owned index cache. Every authority must use its
    physical path without symlinked ancestors (for example ``/private/tmp``,
    not its ``/tmp`` alias). This helper makes no TRAIN or validation membership
    claim.
    """
    validated = _strict_model(binding, NativeReplayDatasetBinding, label="binding")
    if type(rebuild_index) is not bool:
        raise TypeError("rebuild_index must be a strict boolean")
    all_authorities = (
        validated.source_root,
        validated.source_receipt_path,
        validated.completion_receipt_path,
        validated.index_cache_dir,
    )
    for authority in all_authorities:
        _require_no_symlink_components(authority)
    physical_authorities = all_authorities[1:]
    if any(
        _paths_overlap(left, right)
        for index, left in enumerate(physical_authorities)
        for right in physical_authorities[index + 1 :]
    ):
        raise ValueError("native dataset binding physical authorities must not overlap")

    completion = load_native_dataset_completion_receipt(validated.completion_receipt_path)
    if completion.digest != validated.completion_receipt_digest:
        raise ValueError("native completion receipt digest does not match replay binding")
    completion_semantic_digest = validate_native_dataset_completion(
        validated.source_root, completion
    )
    source_receipt = load_native_source_receipt(validated.source_receipt_path)
    derived_receipt = create_native_source_receipt_from_completions(
        validated.source_root, completion
    )
    if source_receipt != derived_receipt or source_receipt.digest != validated.source_digest:
        raise ValueError("native source receipt does not match completion provenance or binding")

    dataset = open_native_indexed_dataset(
        validated.source_root,
        validated.source_receipt_path,
        validated.index_cache_dir,
        chunk_size=validated.chunk_size,
        rebuild=rebuild_index,
    )
    manifest = dataset.manifest
    if (
        dataset.digest != validated.dataset_digest
        or completion_semantic_digest != validated.dataset_digest
        or dataset.source_digest != validated.source_digest
        or manifest.source_receipt != source_receipt
    ):
        raise ValueError("native index identity does not match replay binding")

    if tuple(item.game.game_id for item in completion.games) != dataset.game_ids:
        raise ValueError("completion and native index game order or identity differs")
    for completed, indexed in zip(completion.games, manifest.games, strict=True):
        if (
            completed.game != indexed.identity
            or completed.logical_name != indexed.source_logical_name
            or completed.row_count != indexed.row_count
            or completed.policy_row_count != indexed.policy_row_count
            or completed.value_row_count != indexed.value_row_count
            or completed.boundary_count != indexed.boundary_count
        ):
            raise ValueError("completion and native index game metadata differs")
    return dataset


def _validate_dataset_binding(
    binding: NativeReplayDatasetBinding,
    references: tuple[NativeReplayGameRef, ...],
    *,
    rebuild_index: bool,
) -> IndexedNativeDataset:
    dataset = open_bound_native_dataset(binding, rebuild_index=rebuild_index)
    completion = load_native_dataset_completion_receipt(binding.completion_receipt_path)
    completed_by_id = {item.game.game_id: item for item in completion.games}
    indexed_by_id = {item.game_id: item for item in dataset.manifest.games}
    for reference in references:
        selected_completion = completed_by_id.get(reference.game.game_id)
        selected_index = indexed_by_id.get(reference.game.game_id)
        if selected_completion is None or selected_index is None:
            raise ValueError("sampled replay game is absent from its bound physical dataset")
        if (
            reference.game != selected_completion.game
            or reference.source_logical_name != selected_completion.logical_name
            or reference.policy_row_count != selected_completion.policy_row_count
            or reference.value_row_count != selected_completion.value_row_count
            or reference.boundary_count != selected_completion.boundary_count
            or reference.dataset_digest != dataset.digest
            or reference.source_digest != dataset.source_digest
            or reference.completion_receipt_digest != completion.digest
        ):
            raise ValueError("sampled replay reference does not match physical game metadata")
        if dataset.validate_game(reference.game.game_id) != selected_index:
            raise ValueError("selected native index game failed exact source validation")
    return dataset


def _bind_native_replay_sample(
    catalog: NativeReplayCatalog,
    sample: NativeReplaySample,
    *,
    bindings: Sequence[NativeReplayDatasetBinding],
    rebuild_index: bool,
) -> BoundNativeReplaySample:
    validated_catalog = _strict_model(catalog, NativeReplayCatalog, label="catalog")
    validated_sample = _strict_model(sample, NativeReplaySample, label="sample")
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise TypeError("bindings must be an explicit sequence")
    validated_bindings = tuple(
        _strict_model(item, NativeReplayDatasetBinding, label="binding") for item in bindings
    )

    catalog_by_id = {item.game.game_id: item for item in validated_catalog.games}
    split_by_seed = {
        assignment.world_seed: assignment.split
        for assignment in validated_catalog.split_ledger.assignments
    }
    for reference in validated_sample.games:
        if catalog_by_id.get(reference.game.game_id) != reference:
            raise ValueError("sampled replay reference is not exact retained catalog membership")
        if split_by_seed.get(reference.game.world_seed) != "train":
            raise ValueError("sampled replay reference is not assigned to the train split")

    selected_digests = _ordered_unique(
        tuple(reference.dataset_digest for reference in validated_sample.games)
    )
    bound_digests = tuple(item.dataset_digest for item in validated_bindings)
    if len(set(bound_digests)) != len(bound_digests):
        raise ValueError("native replay bindings must be unambiguous by dataset digest")
    if set(bound_digests) != set(selected_digests):
        raise ValueError("native replay bindings must contain exactly the selected datasets")
    binding_by_digest = {item.dataset_digest: item for item in validated_bindings}

    datasets: dict[str, IndexedNativeDataset] = {}
    for digest in selected_digests:
        references = tuple(
            reference for reference in validated_sample.games if reference.dataset_digest == digest
        )
        dataset = _validate_dataset_binding(
            binding_by_digest[digest], references, rebuild_index=rebuild_index
        )
        compatibility = validated_catalog.compatibility
        if compatibility is None or (
            compatibility.decision_tensor_schema_id,
            compatibility.decision_tensor_schema_version,
            compatibility.decision_tensor_schema_digest,
            compatibility.stable_value_tensor_schema_id,
            compatibility.stable_value_tensor_schema_version,
            compatibility.stable_value_tensor_schema_digest,
        ) != (
            dataset.manifest.decision_tensor_schema_id,
            dataset.manifest.decision_tensor_schema_version,
            dataset.manifest.decision_tensor_schema_digest,
            dataset.manifest.stable_value_tensor_schema_id,
            dataset.manifest.stable_value_tensor_schema_version,
            dataset.manifest.stable_value_tensor_schema_digest,
        ):
            raise ValueError("replay catalog and physical dataset tensor compatibility differ")
        datasets[digest] = dataset

    return BoundNativeReplaySample(
        catalog_digest=validated_catalog.digest,
        sample=validated_sample,
        games=tuple(
            BoundNativeReplayGame(
                reference=reference,
                dataset=datasets[reference.dataset_digest],
            )
            for reference in validated_sample.games
        ),
        catalog=validated_catalog,
        bindings=validated_bindings,
    )


def bind_native_replay_sample(
    catalog: NativeReplayCatalog,
    sample: NativeReplaySample,
    *,
    bindings: Sequence[NativeReplayDatasetBinding],
) -> BoundNativeReplaySample:
    """Bind sampled TRAIN references to non-symlinked physical authorities.

    Disposable physical indexes may be rebuilt when needed.
    """
    return _bind_native_replay_sample(
        catalog,
        sample,
        bindings=bindings,
        rebuild_index=True,
    )


def _expected_model_config(
    config: NativeTrainerConfig,
    decision_schema: TensorFeatureSchema,
    stable_value_schema: StableValueTensorSchema,
) -> Gen1ModelConfig:
    return Gen1ModelConfig(
        decision_schema_digest=decision_schema.digest,
        stable_value_schema_digest=stable_value_schema.digest,
        token_width=config.token_width,
        state_width=config.state_width,
        candidate_width=config.candidate_width,
        message_passing_layers=config.message_passing_layers,
        dropout=config.dropout,
    )


def _assert_cpu_float32_model(model: Gen1PolicyValueModel) -> None:
    state = model.state_dict()
    if not state:
        raise ValueError("native trainer model must have state")
    for name, tensor in state.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"native trainer model tensor must be on CPU: {name}")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"native trainer model floating tensor must be float32: {name}")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("native trainer parameters must be CPU float32")


def _assert_finite_parameters(model: Gen1PolicyValueModel) -> None:
    if any(not torch.isfinite(parameter.detach()).all().item() for parameter in model.parameters()):
        raise ValueError("native trainer model parameters must be finite")


def _assert_cpu_batch(value: object) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError("native training batches and outputs must remain on CPU")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError("native training floating tensors must use float32")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_cpu_batch(getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_cpu_batch(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu_batch(item)


def _clear_gradients(model: Gen1PolicyValueModel) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _strict_nonempty_string(value: str, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _step_payload(step: NativeTrainingStepResult) -> dict[str, Any]:
    return {
        "optimizer_step": step.optimizer_step,
        "policy_cross_entropy": step.policy_cross_entropy,
        "policy_entropy": step.policy_entropy,
        "value_bce": step.value_bce,
        "regularization": step.regularization,
        "total_loss": step.total_loss,
        "gradient_norm_before_clip": step.gradient_norm_before_clip,
        "provenance": step.provenance.model_dump(mode="json"),
    }


def _validate_step_result(step: NativeTrainingStepResult, *, expected_step: int) -> None:
    if not isinstance(step, NativeTrainingStepResult):
        raise TypeError("successful-step lineage contains a foreign result")
    if type(step.optimizer_step) is not int or step.optimizer_step != expected_step:
        raise ValueError("successful-step lineage optimizer ordering is invalid")
    scalars = (
        step.policy_cross_entropy,
        step.policy_entropy,
        step.value_bce,
        step.regularization,
        step.total_loss,
        step.gradient_norm_before_clip,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in scalars):
        raise ValueError("successful-step lineage contains a non-finite scalar")
    _strict_model(
        step.provenance,
        NativeTrainingStepProvenance,
        label="successful-step provenance",
    )


@dataclass(slots=True)
class NativeTrainer:
    config: NativeTrainerConfig
    initialization: NativeTrainerInitialization
    decision_schema: TensorFeatureSchema
    stable_value_schema: StableValueTensorSchema
    model: Gen1PolicyValueModel
    optimizer: torch.optim.Adam
    optimizer_steps: int = 0
    _poisoned: bool = field(default=False, init=False, repr=False)
    _successful_steps: list[NativeTrainingStepResult] = field(
        default_factory=list, init=False, repr=False
    )
    _last_successful_step: NativeTrainingStepResult | None = field(
        default=None, init=False, repr=False
    )
    _initial_config_digest: str = field(init=False, repr=False)
    _initialization_snapshot: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = _strict_model(self.config, NativeTrainerConfig, label="trainer config")
        initialization = _strict_model(
            self.initialization,
            NativeTrainerInitialization,
            label="trainer initialization",
        )
        self._initial_config_digest = config.digest
        self._initialization_snapshot = canonical_json_bytes(initialization)

    def _revalidate_self(self) -> tuple[NativeTrainerConfig, NativeTrainerInitialization]:
        config = _strict_model(self.config, NativeTrainerConfig, label="trainer config")
        initialization = _strict_model(
            self.initialization,
            NativeTrainerInitialization,
            label="trainer initialization",
        )
        if config.digest != self._initial_config_digest:
            raise ValueError("native trainer config changed from its pinned initial digest")
        if canonical_json_bytes(initialization) != self._initialization_snapshot:
            raise ValueError("native trainer initialization changed from its pinned snapshot")
        if self._poisoned:
            raise RuntimeError("native trainer is poisoned after an optimizer failure")
        if self.optimizer_steps != len(self._successful_steps) or self.optimizer_steps < 0:
            raise ValueError("native trainer successful-step lineage is inconsistent")
        for expected_step, step in enumerate(self._successful_steps, start=1):
            _validate_step_result(step, expected_step=expected_step)
        if self._successful_steps:
            if self._last_successful_step is not self._successful_steps[-1]:
                raise ValueError("native trainer latest successful-step binding changed")
        elif self._last_successful_step is not None:
            raise ValueError("native trainer has a result without successful-step lineage")
        decision_schema = _strict_model(
            self.decision_schema, TensorFeatureSchema, label="decision schema"
        )
        stable_value_schema = _strict_model(
            self.stable_value_schema,
            StableValueTensorSchema,
            label="stable-value schema",
        )
        if (
            decision_schema != TensorFeatureSchema.current()
            or stable_value_schema != StableValueTensorSchema.current()
        ):
            raise ValueError("native trainer tensor schemas changed")
        if self.model.config != _expected_model_config(
            config, decision_schema, stable_value_schema
        ):
            raise ValueError("native trainer model configuration changed")
        _assert_cpu_float32_model(self.model)
        parameter_ids = [id(parameter) for parameter in self.model.parameters()]
        optimized_ids = [
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        ]
        if optimized_ids != parameter_ids:
            raise ValueError("native trainer optimizer parameters changed")
        if len(self.optimizer.param_groups) != 1:
            raise ValueError("native trainer requires exactly one Adam parameter group")
        group = self.optimizer.param_groups[0]
        expected = {
            "lr": config.learning_rate,
            "betas": (config.adam_beta1, config.adam_beta2),
            "eps": config.adam_epsilon,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        }
        if any(group.get(name) != value for name, value in expected.items()):
            raise ValueError("native trainer Adam contract changed")
        return config, initialization

    def _revalidate_selection(self, selection: BoundNativeReplaySample) -> BoundNativeReplaySample:
        if not isinstance(selection, BoundNativeReplaySample):
            raise TypeError("selection must be a BoundNativeReplaySample")
        catalog = _strict_model(selection.catalog, NativeReplayCatalog, label="bound catalog")
        sample = _strict_model(selection.sample, NativeReplaySample, label="bound sample")
        bindings = tuple(
            _strict_model(binding, NativeReplayDatasetBinding, label="bound binding")
            for binding in selection.bindings
        )
        if selection.catalog_digest != catalog.digest:
            raise ValueError("bound replay catalog digest changed")
        if len(selection.games) != len(sample.games):
            raise ValueError("bound replay game order changed")
        binding_by_digest = {binding.dataset_digest: binding for binding in bindings}
        for supplied, reference in zip(selection.games, sample.games, strict=True):
            if not isinstance(supplied, BoundNativeReplayGame) or supplied.reference != reference:
                raise ValueError("bound replay game reference changed")
            binding = binding_by_digest.get(reference.dataset_digest)
            if binding is None:
                raise ValueError("bound replay game lost its physical binding")
            if (
                supplied.dataset.source_root != binding.source_root
                or supplied.dataset.cache_dir != binding.index_cache_dir
                or supplied.dataset.digest != binding.dataset_digest
                or supplied.dataset.source_digest != binding.source_digest
            ):
                raise ValueError("bound replay dataset authority changed")
        rebound = _bind_native_replay_sample(
            catalog,
            sample,
            bindings=bindings,
            rebuild_index=False,
        )
        if tuple(game.reference for game in rebound.games) != tuple(
            game.reference for game in selection.games
        ):
            raise ValueError("rebound replay selection differs from retained authority")
        return rebound

    def train_logical_batch(
        self,
        selection: BoundNativeReplaySample,
    ) -> NativeTrainingStepResult:
        """Stream one logical sample through both heads and perform exactly one Adam step."""
        config, _ = self._revalidate_self()
        rebound = self._revalidate_selection(selection)
        sample = rebound.sample
        policy_enabled = config.policy_weight > 0.0
        value_enabled = config.value_weight > 0.0
        if not (
            (policy_enabled and sample.policy_contributing_game_count > 0)
            or (value_enabled and sample.value_contributing_game_count > 0)
        ):
            raise ValueError("native logical batch has no enabled contributing head")
        _assert_finite_parameters(self.model)

        provenance = NativeTrainingStepProvenance(
            replay_catalog_digest=rebound.catalog_digest,
            selected_game_ids=sample.game_ids,
            dataset_digests=_ordered_unique(
                tuple(game.reference.dataset_digest for game in rebound.games)
            ),
            source_digests=_ordered_unique(
                tuple(game.reference.source_digest for game in rebound.games)
            ),
            completion_receipt_digests=_ordered_unique(
                tuple(game.reference.completion_receipt_digest for game in rebound.games)
            ),
            policy_contributing_game_count=sample.policy_contributing_game_count,
            value_contributing_game_count=sample.value_contributing_game_count,
        )
        self.optimizer.zero_grad(set_to_none=True)
        policy_cross_entropy = 0.0
        policy_entropy = 0.0
        value_bce = 0.0
        objective_without_regularization = 0.0
        regularization_value = 0.0
        optimizer_started = False
        try:
            for bound in rebound.games:
                reference = bound.reference
                if policy_enabled and reference.policy_row_count > 0:
                    for policy_batch in bound.dataset.iter_policy_training_chunks(
                        reference.game.game_id
                    ):
                        _assert_cpu_batch(policy_batch)
                        policy_output = self.model.forward_policy(policy_batch.batch)
                        _assert_cpu_batch(policy_output)
                        policy_loss = native_policy_loss(
                            policy_output,
                            legal_mask=policy_batch.batch.candidates.mask,
                            policy_targets=policy_batch.policy_targets,
                            row_weights=policy_batch.row_weights,
                            row_mask=policy_batch.row_mask,
                            head_normalizer=float(sample.policy_contributing_game_count),
                            entropy_weight=config.entropy_weight,
                        )
                        weighted = config.policy_weight * policy_loss.total
                        if not torch.isfinite(weighted).item():
                            raise ValueError("native policy objective is non-finite")
                        weighted.backward()
                        cross_entropy = float(policy_loss.cross_entropy.detach())
                        entropy = float(policy_loss.entropy.detach())
                        policy_cross_entropy += cross_entropy
                        policy_entropy += entropy
                        objective_without_regularization += config.policy_weight * (
                            cross_entropy - config.entropy_weight * entropy
                        )
                if value_enabled and reference.value_row_count > 0:
                    for value_batch in bound.dataset.iter_value_training_chunks(
                        reference.game.game_id
                    ):
                        _assert_cpu_batch(value_batch)
                        value_output = self.model.forward_stable_value(value_batch.batch)
                        _assert_cpu_batch(value_output)
                        value_loss = native_stable_value_loss(
                            value_output,
                            value_targets=value_batch.value_targets,
                            row_weights=value_batch.row_weights,
                            row_mask=value_batch.row_mask,
                            head_normalizer=float(sample.value_contributing_game_count),
                        )
                        weighted = config.value_weight * value_loss.total
                        if not torch.isfinite(weighted).item():
                            raise ValueError("native stable-value objective is non-finite")
                        weighted.backward()
                        bce = float(value_loss.bce.detach())
                        value_bce += bce
                        objective_without_regularization += config.value_weight * bce

            if config.l2_weight > 0.0:
                regularization = sum(
                    (parameter.square().sum() for parameter in self.model.parameters()),
                    torch.zeros((), dtype=torch.float32),
                )
                if not torch.isfinite(regularization).item():
                    raise ValueError("native L2 regularizer is non-finite")
                regularization_value = float(regularization.detach())
                (config.l2_weight * regularization).backward()
            gradients = tuple(
                parameter.grad
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )
            if not gradients or any(
                gradient.device.type != "cpu"
                or gradient.dtype != torch.float32
                or not torch.isfinite(gradient).all().item()
                for gradient in gradients
            ):
                raise ValueError("native trainer gradients must be finite CPU float32 tensors")
            total_loss = objective_without_regularization + (
                config.l2_weight * regularization_value
            )
            if not math.isfinite(total_loss):
                raise ValueError("native total loss is non-finite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.max_gradient_norm,
                error_if_nonfinite=True,
            )
            gradient_norm_value = float(gradient_norm.detach())
            if not math.isfinite(gradient_norm_value):
                raise ValueError("native pre-clip gradient norm is non-finite")

            next_step = self.optimizer_steps + 1
            result = NativeTrainingStepResult(
                optimizer_step=next_step,
                policy_cross_entropy=policy_cross_entropy,
                policy_entropy=policy_entropy,
                value_bce=value_bce,
                regularization=regularization_value,
                total_loss=total_loss,
                gradient_norm_before_clip=gradient_norm_value,
                provenance=provenance,
            )
            optimizer_started = True
            self.optimizer.step()
            if any(
                not torch.isfinite(parameter.detach()).all().item()
                for parameter in self.model.parameters()
            ):
                raise RuntimeError("Adam produced non-finite native trainer parameters")
        except BaseException:
            _clear_gradients(self.model)
            if optimizer_started:
                self._poisoned = True
            raise

        self.optimizer_steps = result.optimizer_step
        self._successful_steps.append(result)
        self._last_successful_step = result
        return result

    def export_artifact(
        self,
        destination: str | Path,
        *,
        step: NativeTrainingStepResult,
        source_revision: str,
        dirty_tree_hash: str,
    ) -> Gen1ModelArtifactManifest:
        """Export exact current weights with complete ordered successful-step lineage."""
        config, initialization = self._revalidate_self()
        _strict_nonempty_string(source_revision, label="source_revision")
        _strict_nonempty_string(dirty_tree_hash, label="dirty_tree_hash")
        if self.optimizer_steps < 1 or self._last_successful_step is None:
            raise ValueError("native trainer requires a completed optimizer step before export")
        if step is not self._last_successful_step or step.optimizer_step != self.optimizer_steps:
            raise ValueError(
                "native trainer export requires its exact latest successful step object"
            )
        _assert_finite_parameters(self.model)
        from automata.training.native_gen1 import current_gen1_artifact_scope

        provenance = {
            "schema_version": 1,
            "trainer_config": config.model_dump(mode="json"),
            "trainer_config_digest": config.digest,
            "initialization": initialization.model_dump(mode="json"),
            "optimizer": {
                "kind": "torch.optim.Adam",
                "learning_rate": config.learning_rate,
                "beta1": config.adam_beta1,
                "beta2": config.adam_beta2,
                "epsilon": config.adam_epsilon,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
                "step_count": self.optimizer_steps,
            },
            "successful_steps": [_step_payload(item) for item in self._successful_steps],
            "source_revision": source_revision,
            "dirty_tree_hash": dirty_tree_hash,
        }
        return export_gen1_model_artifact(
            destination,
            model=self.model,
            decision_schema=self.decision_schema,
            stable_value_schema=self.stable_value_schema,
            scope=current_gen1_artifact_scope(),
            provenance=provenance,
        )


def create_native_trainer(
    config: NativeTrainerConfig,
    *,
    initialization: NativeTrainerInitialization,
    parent_artifact_path: str | Path | None = None,
) -> NativeTrainer:
    """Create isolated fresh/parent weights and a new empty-state explicit Adam."""
    validated_config = _strict_model(config, NativeTrainerConfig, label="trainer config")
    validated_initialization = _strict_model(
        initialization,
        NativeTrainerInitialization,
        label="trainer initialization",
    )
    decision_schema = TensorFeatureSchema.current()
    stable_value_schema = StableValueTensorSchema.current()
    expected_config = _expected_model_config(validated_config, decision_schema, stable_value_schema)

    if validated_initialization.mode == "FRESH_BOOTSTRAP":
        if parent_artifact_path is not None:
            raise ValueError("FRESH_BOOTSTRAP must not receive a parent artifact path")
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.default_generator.manual_seed(validated_config.seed)
            model = Gen1PolicyValueModel(
                decision_schema=decision_schema,
                stable_value_schema=stable_value_schema,
                config=expected_config,
            )
    else:
        if parent_artifact_path is None:
            raise ValueError("GEN1_PARENT requires a parent artifact path")
        assert validated_initialization.parent_model_digest is not None
        from automata.training.native_gen1 import load_current_gen1_parent_artifact

        # Safe loading reconstructs the module before applying exact parent
        # weights; isolate that constructor's otherwise irrelevant RNG draws.
        with torch.random.fork_rng(devices=[], enabled=True):
            loaded = load_current_gen1_parent_artifact(
                parent_artifact_path,
                expected_model_digest=validated_initialization.parent_model_digest,
            )
        if loaded.config != expected_config:
            raise ValueError("Gen1 parent architecture does not match trainer configuration")
        if any(
            tensor.dtype != "float32"
            for tensor in loaded.manifest.tensors.values()
            if tensor.dtype.startswith("float") or tensor.dtype.startswith("bfloat")
        ):
            raise ValueError("Gen1 parent serialized floating weights must be exact float32")
        model = loaded.model
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    model.train()
    _assert_cpu_float32_model(model)
    _assert_finite_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=validated_config.learning_rate,
        betas=(validated_config.adam_beta1, validated_config.adam_beta2),
        eps=validated_config.adam_epsilon,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state:
        raise RuntimeError("new native trainer Adam unexpectedly has optimizer state")
    return NativeTrainer(
        config=validated_config,
        initialization=validated_initialization,
        decision_schema=decision_schema,
        stable_value_schema=stable_value_schema,
        model=model,
        optimizer=optimizer,
    )


__all__ = [
    "BoundNativeReplayGame",
    "BoundNativeReplaySample",
    "NativeReplayDatasetBinding",
    "NativeTrainer",
    "NativeTrainerConfig",
    "NativeTrainerInitialization",
    "NativeTrainingStepProvenance",
    "NativeTrainingStepResult",
    "bind_native_replay_sample",
    "create_native_trainer",
    "open_bound_native_dataset",
]
