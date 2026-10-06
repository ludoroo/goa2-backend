"""Completion-bound, current-run held-out metrics for native Gen1 data.

Validation here is split-isolated from updates in the current run only.  A model
artifact does not carry enough ancestral replay history to claim global or
parent-training held-out status.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import PurePosixPath
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
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.training.native_dataset import NativeGameIdentity
from automata.training.native_indexed_dataset import (
    IndexedNativeDataset,
    NativeIndexedDatasetManifest,
    create_native_source_receipt_from_completions,
)
from automata.training.native_losses import native_policy_loss, native_stable_value_loss
from automata.training.native_receipts import (
    NativeDatasetCompletionReceipt,
    load_native_dataset_completion_receipt,
    validate_native_dataset_completion,
)
from automata.training.native_splits import NativeSeedSplitLedger
from automata.training.native_trainer import (
    NativeReplayDatasetBinding,
    open_bound_native_dataset,
)

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_VALIDATION_SCOPE: Literal["CURRENT_RUN_UPDATES"] = "CURRENT_RUN_UPDATES"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def canonical_bytes(self) -> bytes:
        validated = type(self).model_validate(self.model_dump(mode="python"), strict=True)
        return canonical_json_bytes(validated)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _safe_source_logical_name(value: str) -> str:
    if type(value) is not str:
        raise ValueError("validation source logical name must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or not (value.endswith(".jsonl") or value.endswith(".jsonl.zst"))
    ):
        raise ValueError(
            "validation source logical name must be a normalized relative native JSONL path"
        )
    return value


class NativeValidationDatasetRef(_FrozenModel):
    """Portable authority for one supplied dataset, including TRAIN-only data."""

    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    validation_game_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _valid_dataset_reference(self) -> NativeValidationDatasetRef:
        if len(set(self.validation_game_ids)) != len(self.validation_game_ids):
            raise ValueError("validation dataset game IDs must be unique")
        if any(
            len(game_id) != 64 or any(character not in "0123456789abcdef" for character in game_id)
            for game_id in self.validation_game_ids
        ):
            raise ValueError("validation dataset game IDs must be SHA-256 digests")
        return self


class NativeValidationGameRef(_FrozenModel):
    """One completion-bound validation game; this contains no rows or tensors."""

    game: NativeGameIdentity
    dataset_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    completion_receipt_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_logical_name: str
    policy_row_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    boundary_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _valid_reference(self) -> NativeValidationGameRef:
        _safe_source_logical_name(self.source_logical_name)
        # Reconstruct nested instances because model_copy(update=...) can bypass
        # validation on both this object and its game identity.
        game = NativeGameIdentity.model_validate(self.game.model_dump(mode="python"), strict=True)
        if game != self.game:
            raise ValueError("validation game identity changed during strict reconstruction")
        if self.policy_row_count + self.value_row_count <= 0:
            raise ValueError("validation game must contribute at least one row")
        if (self.value_row_count == 0) != (self.boundary_count == 0):
            raise ValueError("validation boundary count must agree with value rows")
        if self.boundary_count > self.value_row_count:
            raise ValueError("validation boundary count cannot exceed value rows")
        return self


class NativeValidationLedger(_FrozenModel):
    """Immutable exact held-out game inventory in physical dataset order."""

    schema_version: Literal[1] = 1
    split_ledger: NativeSeedSplitLedger
    datasets: tuple[NativeValidationDatasetRef, ...] = Field(min_length=1)
    games: tuple[NativeValidationGameRef, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def _valid_ledger(self) -> NativeValidationLedger:
        split_ledger = NativeSeedSplitLedger.model_validate(
            self.split_ledger.model_dump(mode="python"), strict=True
        )
        datasets = tuple(
            NativeValidationDatasetRef.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.datasets
        )
        refs = tuple(
            NativeValidationGameRef.model_validate(item.model_dump(mode="python"), strict=True)
            for item in self.games
        )
        dataset_digests = tuple(item.dataset_digest for item in datasets)
        if len(set(dataset_digests)) != len(dataset_digests):
            raise ValueError("validation ledger dataset authorities must be unambiguous")
        dataset_by_digest = {item.dataset_digest: item for item in datasets}
        game_ids = tuple(item.game.game_id for item in refs)
        dataset_game_ids = tuple((item.dataset_digest, item.game.game_id) for item in refs)
        if len(set(game_ids)) != len(game_ids) or len(set(dataset_game_ids)) != len(
            dataset_game_ids
        ):
            raise ValueError("validation ledger game IDs must be unique")
        expected_game_ids: list[str] = []
        for dataset in datasets:
            dataset_games = tuple(
                item for item in refs if item.dataset_digest == dataset.dataset_digest
            )
            if tuple(item.game.game_id for item in dataset_games) != dataset.validation_game_ids:
                raise ValueError(
                    "validation ledger must name every and only validation game in each "
                    "dataset inventory order"
                )
            if any(
                item.source_digest != dataset.source_digest
                or item.completion_receipt_digest != dataset.completion_receipt_digest
                for item in dataset_games
            ):
                raise ValueError("validation game refs must match their dataset authorities")
            expected_game_ids.extend(dataset.validation_game_ids)
        if tuple(expected_game_ids) != game_ids:
            raise ValueError("validation ledger games must follow physical dataset order")
        if any(item.dataset_digest not in dataset_by_digest for item in refs):
            raise ValueError("validation game ref names an unknown dataset authority")

        split_by_seed = {
            assignment.world_seed: assignment.split for assignment in split_ledger.assignments
        }
        for item in refs:
            split = split_by_seed.get(item.game.world_seed)
            if split is None:
                raise ValueError(
                    f"validation game world seed {item.game.world_seed} is not enrolled"
                )
            if split != "validation":
                raise ValueError("validation ledger cannot contain a train-split game")
        return self


class NativeValidationMetrics(_FrozenModel):
    """Game-normalized diagnostics held out from current-run updates only."""

    schema_version: Literal[1] = 1
    validation_scope: Literal["CURRENT_RUN_UPDATES"] = _VALIDATION_SCOPE
    validation_ledger_digest: str = Field(pattern=_DIGEST_PATTERN)
    game_ids: tuple[str, ...]
    dataset_digests: tuple[str, ...]
    source_digests: tuple[str, ...]
    completion_receipt_digests: tuple[str, ...]
    game_count: StrictInt = Field(ge=0)
    policy_contributing_game_count: StrictInt = Field(ge=0)
    policy_row_count: StrictInt = Field(ge=0)
    policy_cross_entropy: float | None
    policy_entropy: float | None
    value_contributing_game_count: StrictInt = Field(ge=0)
    value_row_count: StrictInt = Field(ge=0)
    value_bce: float | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator(
        "policy_cross_entropy",
        "policy_entropy",
        "value_bce",
        mode="before",
    )
    @classmethod
    def _strict_finite_optional_metric(cls, value: Any) -> Any:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("validation metric values must be strict finite floats or None")
        return value

    @model_validator(mode="after")
    def _valid_metrics(self) -> NativeValidationMetrics:
        authorities = (
            self.dataset_digests,
            self.source_digests,
            self.completion_receipt_digests,
        )
        if len(self.game_ids) != self.game_count:
            raise ValueError("validation metric game IDs must align with game_count")
        if self.game_count == 0:
            raise ValueError("validation metrics require at least one game")
        if len(set(self.game_ids)) != self.game_count:
            raise ValueError("validation metric game IDs must be unique")
        authority_count = len(self.dataset_digests)
        if authority_count == 0 or any(
            len(values) != authority_count or len(set(values)) != len(values)
            for values in authorities
        ):
            raise ValueError(
                "validation metric authority arrays must be nonempty aligned unique inventories"
            )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for values in authorities
            for value in values
        ):
            raise ValueError("validation metric authority identities must be SHA-256 digests")
        if self.policy_contributing_game_count > self.game_count:
            raise ValueError("policy contributing game count cannot exceed game_count")
        if self.value_contributing_game_count > self.game_count:
            raise ValueError("value contributing game count cannot exceed game_count")
        policy_absent = self.policy_contributing_game_count == 0
        if policy_absent != (self.policy_row_count == 0) or policy_absent != (
            self.policy_cross_entropy is None and self.policy_entropy is None
        ):
            raise ValueError("absent policy metrics must use zero counts and None values")
        if not policy_absent and (
            self.policy_row_count == 0
            or self.policy_cross_entropy is None
            or self.policy_entropy is None
        ):
            raise ValueError("contributing policy games require rows and finite metrics")
        value_absent = self.value_contributing_game_count == 0
        if value_absent != (self.value_row_count == 0) or value_absent != (self.value_bce is None):
            raise ValueError("absent value metrics must use zero counts and None value")
        if not value_absent and (self.value_row_count == 0 or self.value_bce is None):
            raise ValueError("contributing value games require rows and a finite metric")
        return self


def _strict_model(value: _ModelT, expected: type[_ModelT], *, label: str) -> _ModelT:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}")
    return expected.model_validate(value.model_dump(mode="python"), strict=True)


def _strict_sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an explicit sequence")
    return value


def _strict_manifest(dataset: IndexedNativeDataset) -> NativeIndexedDatasetManifest:
    if not isinstance(dataset, IndexedNativeDataset):
        raise TypeError("datasets must contain only IndexedNativeDataset instances")
    return NativeIndexedDatasetManifest.model_validate(
        dataset.manifest.model_dump(mode="python"), strict=True
    )


def _validate_dataset_pair(
    dataset: IndexedNativeDataset,
    completion: NativeDatasetCompletionReceipt,
) -> tuple[NativeIndexedDatasetManifest, NativeDatasetCompletionReceipt]:
    manifest = _strict_manifest(dataset)
    validated_completion = _strict_model(
        completion,
        NativeDatasetCompletionReceipt,
        label="completion receipt",
    )
    source_receipt = create_native_source_receipt_from_completions(
        dataset.source_root, validated_completion
    )
    semantic_digest = validate_native_dataset_completion(dataset.source_root, validated_completion)
    if (
        source_receipt != manifest.source_receipt
        or source_receipt.digest != manifest.source_digest
        or dataset.source_digest != manifest.source_digest
        or semantic_digest != manifest.semantic_dataset_digest
        or dataset.digest != manifest.semantic_dataset_digest
    ):
        raise ValueError(
            "validation completion, source inventory, and indexed dataset identities differ"
        )
    if tuple(item.game.game_id for item in validated_completion.games) != tuple(
        item.game_id for item in manifest.games
    ):
        raise ValueError("validation completion and index game order or identity differs")
    for completed, indexed in zip(validated_completion.games, manifest.games, strict=True):
        if (
            completed.game != indexed.identity
            or completed.logical_name != indexed.source_logical_name
            or completed.row_count != indexed.row_count
            or completed.policy_row_count != indexed.policy_row_count
            or completed.value_row_count != indexed.value_row_count
            or completed.boundary_count != indexed.boundary_count
        ):
            raise ValueError("validation completion and indexed game metadata differs")
        if dataset.validate_game(indexed.game_id) != indexed:
            raise ValueError("validation indexed game failed exact source validation")
    return manifest, validated_completion


def _ref_from_physical(
    *,
    dataset_digest: str,
    source_digest: str,
    completion_digest: str,
    completed: Any,
    indexed: Any,
) -> NativeValidationGameRef:
    return NativeValidationGameRef(
        game=indexed.identity,
        dataset_digest=dataset_digest,
        source_digest=source_digest,
        completion_receipt_digest=completion_digest,
        source_logical_name=completed.logical_name,
        policy_row_count=indexed.policy_row_count,
        value_row_count=indexed.value_row_count,
        boundary_count=indexed.boundary_count,
    )


def create_native_validation_ledger(
    split_ledger: NativeSeedSplitLedger,
    *,
    datasets: Sequence[IndexedNativeDataset],
    completion_receipts: Sequence[NativeDatasetCompletionReceipt],
) -> NativeValidationLedger:
    """Create refs for every and only validation-assigned game in physical order."""
    validated_split = _strict_model(split_ledger, NativeSeedSplitLedger, label="split ledger")
    dataset_values = tuple(_strict_sequence(datasets, label="datasets"))
    completion_values = tuple(_strict_sequence(completion_receipts, label="completion_receipts"))
    if not dataset_values or len(dataset_values) != len(completion_values):
        raise ValueError(
            "validation requires one completion receipt for each nonempty dataset sequence"
        )
    split_by_seed = {
        assignment.world_seed: assignment.split for assignment in validated_split.assignments
    }
    dataset_refs: list[NativeValidationDatasetRef] = []
    refs: list[NativeValidationGameRef] = []
    dataset_digests: set[str] = set()
    all_game_ids: set[str] = set()
    for dataset_value, completion_value in zip(dataset_values, completion_values, strict=True):
        if not isinstance(dataset_value, IndexedNativeDataset):
            raise TypeError("datasets must contain only IndexedNativeDataset instances")
        if not isinstance(completion_value, NativeDatasetCompletionReceipt):
            raise TypeError(
                "completion_receipts must contain only NativeDatasetCompletionReceipt instances"
            )
        manifest, completion = _validate_dataset_pair(dataset_value, completion_value)
        if manifest.semantic_dataset_digest in dataset_digests:
            raise ValueError("validation datasets must be unambiguous by dataset digest")
        dataset_digests.add(manifest.semantic_dataset_digest)
        validation_game_ids: list[str] = []
        for completed, indexed in zip(completion.games, manifest.games, strict=True):
            if indexed.game_id in all_game_ids:
                raise ValueError("validation source game IDs must be unique across datasets")
            all_game_ids.add(indexed.game_id)
            split = split_by_seed.get(indexed.world_seed)
            if split is None:
                raise ValueError(
                    f"world seed {indexed.world_seed} is not enrolled in the native split ledger"
                )
            if split == "validation":
                validation_game_ids.append(indexed.game_id)
                refs.append(
                    _ref_from_physical(
                        dataset_digest=manifest.semantic_dataset_digest,
                        source_digest=manifest.source_digest,
                        completion_digest=completion.digest,
                        completed=completed,
                        indexed=indexed,
                    )
                )
        dataset_refs.append(
            NativeValidationDatasetRef(
                dataset_digest=manifest.semantic_dataset_digest,
                source_digest=manifest.source_digest,
                completion_receipt_digest=completion.digest,
                validation_game_ids=tuple(validation_game_ids),
            )
        )
    if not refs:
        raise ValueError("native validation requires at least one validation-assigned game")
    return NativeValidationLedger(
        split_ledger=validated_split,
        datasets=tuple(dataset_refs),
        games=tuple(refs),
    )


def _assert_cpu_value(value: object, *, label: str, finite: bool = False) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{label} tensors must remain on CPU")
        if value.is_floating_point():
            if value.dtype != torch.float32:
                raise ValueError(f"{label} floating tensors must use float32")
            if finite and not torch.isfinite(value).all().item():
                raise ValueError(f"{label} floating tensors must be finite")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_cpu_value(getattr(value, item.name), label=label, finite=finite)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_cpu_value(item, label=label, finite=finite)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu_value(item, label=label, finite=finite)


def _validate_model(model: Gen1PolicyValueModel) -> None:
    if type(model) is not Gen1PolicyValueModel:
        raise TypeError("model must be an exact Gen1PolicyValueModel")
    if type(model.config) is not Gen1ModelConfig:
        raise TypeError("Gen1 validation model config must be an exact Gen1ModelConfig")
    reconstructed = Gen1ModelConfig(**asdict(model.config))
    if reconstructed != model.config:
        raise ValueError("Gen1 validation model config failed strict reconstruction")
    if reconstructed.dropout != 0.0:
        raise ValueError("native validation requires model dropout == 0.0")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            location = name or "<root>"
            raise ValueError(
                "native validation requires every live torch.nn.Dropout.p == 0; "
                f"{location} has p={module.p!r}"
            )
    decision = TensorFeatureSchema.current()
    value = StableValueTensorSchema.current()
    if (
        reconstructed.decision_schema_digest != decision.digest
        or reconstructed.stable_value_schema_digest != value.digest
    ):
        raise ValueError("native validation model does not match current tensor schemas")
    with torch.random.fork_rng(devices=[], enabled=True), torch.device("meta"):
        expected_model = Gen1PolicyValueModel(
            decision_schema=decision,
            stable_value_schema=value,
            config=reconstructed,
        ).to(dtype=torch.float32)
    actual_signature = tuple(
        (name, tuple(tensor.shape), tensor.dtype) for name, tensor in model.state_dict().items()
    )
    expected_signature = tuple(
        (name, tuple(tensor.shape), tensor.dtype)
        for name, tensor in expected_model.state_dict().items()
    )
    if actual_signature != expected_signature:
        raise ValueError("native validation model structure does not match its configuration")
    state = model.state_dict()
    if not state:
        raise ValueError("native validation model must have state")
    for name, tensor in state.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"native validation model tensor must be on CPU: {name}")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise ValueError(f"native validation model floating tensor must be float32: {name}")
            if not torch.isfinite(tensor.detach()).all().item():
                raise ValueError(f"native validation model tensor must be finite: {name}")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("native validation parameters must be CPU float32")
        if not torch.isfinite(parameter.detach()).all().item():
            raise ValueError("native validation model parameters must be finite")


class _ModelSnapshot:
    def __init__(self, model: Gen1PolicyValueModel) -> None:
        self.modules = tuple(model.modules())
        self.flags = tuple(module.training for module in self.modules)
        self.parameters = tuple(model.named_parameters())
        self.parameter_values = tuple(
            parameter.detach().clone() for _, parameter in self.parameters
        )
        self.requires_grad = tuple(parameter.requires_grad for _, parameter in self.parameters)
        self.gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for _, parameter in self.parameters
        )
        self.buffers = tuple(model.named_buffers())
        self.buffer_values = tuple(buffer.detach().clone() for _, buffer in self.buffers)
        self.rng = torch.random.get_rng_state().clone()

    def restore_and_report_mutation(self, model: Gen1PolicyValueModel) -> bool:
        changed = False
        current_modules = tuple(model.modules())
        current_parameters = tuple(model.named_parameters())
        current_buffers = tuple(model.named_buffers())
        if (
            tuple(id(item) for item in current_modules) != tuple(id(item) for item in self.modules)
            or tuple(name for name, _ in current_parameters)
            != tuple(name for name, _ in self.parameters)
            or tuple(name for name, _ in current_buffers) != tuple(name for name, _ in self.buffers)
        ):
            raise RuntimeError("native validation model structure changed during inference")

        for module, expected in zip(self.modules, self.flags, strict=True):
            if module.training != expected:
                changed = True
                module.training = expected
        for (_, parameter), expected_value, expected_requires, expected_grad in zip(
            self.parameters,
            self.parameter_values,
            self.requires_grad,
            self.gradients,
            strict=True,
        ):
            if not torch.equal(parameter.detach(), expected_value):
                changed = True
                with torch.inference_mode():
                    parameter.copy_(expected_value)
            if parameter.requires_grad != expected_requires:
                changed = True
                parameter.requires_grad_(expected_requires)
            actual_grad = parameter.grad
            if expected_grad is None:
                if actual_grad is not None:
                    changed = True
                    parameter.grad = None
            elif actual_grad is None or not torch.equal(actual_grad, expected_grad):
                changed = True
                parameter.grad = expected_grad.clone()
        for (_, buffer), expected_value in zip(self.buffers, self.buffer_values, strict=True):
            if not torch.equal(buffer.detach(), expected_value):
                changed = True
                with torch.inference_mode():
                    buffer.copy_(expected_value)
        current_rng = torch.random.get_rng_state()
        if not torch.equal(current_rng, self.rng):
            changed = True
            torch.random.set_rng_state(self.rng)
        return changed


def _open_and_match_ledger(
    ledger: NativeValidationLedger,
    bindings: Sequence[NativeReplayDatasetBinding],
) -> dict[str, IndexedNativeDataset]:
    validated_bindings = tuple(
        _strict_model(item, NativeReplayDatasetBinding, label="binding") for item in bindings
    )
    bound_digests = tuple(item.dataset_digest for item in validated_bindings)
    if len(set(bound_digests)) != len(bound_digests):
        raise ValueError("native validation bindings must be unambiguous by dataset digest")
    ledger_digests = tuple(item.dataset_digest for item in ledger.datasets)
    if set(bound_digests) != set(ledger_digests):
        raise ValueError("native validation bindings must contain exactly the ledger datasets")
    binding_by_digest = {item.dataset_digest: item for item in validated_bindings}
    split_by_seed = {
        assignment.world_seed: assignment.split for assignment in ledger.split_ledger.assignments
    }
    datasets: dict[str, IndexedNativeDataset] = {}
    expected_datasets: list[NativeValidationDatasetRef] = []
    expected_refs: list[NativeValidationGameRef] = []
    all_game_ids: set[str] = set()
    for ledger_dataset in ledger.datasets:
        digest = ledger_dataset.dataset_digest
        binding = binding_by_digest[digest]
        dataset = open_bound_native_dataset(binding, rebuild_index=False)
        completion = load_native_dataset_completion_receipt(binding.completion_receipt_path)
        datasets[digest] = dataset
        validation_game_ids: list[str] = []
        for completed, indexed in zip(completion.games, dataset.manifest.games, strict=True):
            if indexed.game_id in all_game_ids:
                raise ValueError("validation source game IDs must be unique across datasets")
            all_game_ids.add(indexed.game_id)
            split = split_by_seed.get(indexed.world_seed)
            if split is None:
                raise ValueError(
                    f"world seed {indexed.world_seed} is not enrolled in the validation ledger"
                )
            if split == "validation":
                validation_game_ids.append(indexed.game_id)
                expected_refs.append(
                    _ref_from_physical(
                        dataset_digest=dataset.digest,
                        source_digest=dataset.source_digest,
                        completion_digest=completion.digest,
                        completed=completed,
                        indexed=indexed,
                    )
                )
        expected_datasets.append(
            NativeValidationDatasetRef(
                dataset_digest=dataset.digest,
                source_digest=dataset.source_digest,
                completion_receipt_digest=completion.digest,
                validation_game_ids=tuple(validation_game_ids),
            )
        )
    if tuple(expected_datasets) != ledger.datasets:
        raise ValueError("native validation dataset authority inventory changed")
    if tuple(expected_refs) != ledger.games:
        raise ValueError(
            "native validation ledger does not name every and only validation game in physical order"
        )
    return datasets


def _evaluate(
    model: Gen1PolicyValueModel,
    ledger: NativeValidationLedger,
    *,
    bindings: Sequence[NativeReplayDatasetBinding],
) -> NativeValidationMetrics:
    _validate_model(model)
    datasets = _open_and_match_ledger(ledger, bindings)
    policy_games = sum(item.policy_row_count > 0 for item in ledger.games)
    value_games = sum(item.value_row_count > 0 for item in ledger.games)
    policy_cross_entropy = 0.0
    policy_entropy = 0.0
    value_bce = 0.0

    with torch.inference_mode():
        for reference in ledger.games:
            dataset = datasets[reference.dataset_digest]
            if reference.policy_row_count > 0:
                for policy_batch in dataset.iter_policy_training_chunks(reference.game.game_id):
                    _assert_cpu_value(policy_batch, label="native validation batch")
                    policy_output = model.forward_policy(policy_batch.batch)
                    _assert_cpu_value(policy_output, label="native validation output", finite=True)
                    policy_loss = native_policy_loss(
                        policy_output,
                        legal_mask=policy_batch.batch.candidates.mask,
                        policy_targets=policy_batch.policy_targets,
                        row_weights=policy_batch.row_weights,
                        row_mask=policy_batch.row_mask,
                        head_normalizer=float(policy_games),
                        entropy_weight=0.0,
                    )
                    cross_entropy = float(policy_loss.cross_entropy)
                    entropy = float(policy_loss.entropy)
                    if not math.isfinite(cross_entropy) or not math.isfinite(entropy):
                        raise ValueError("native validation policy metrics must be finite")
                    policy_cross_entropy += cross_entropy
                    policy_entropy += entropy
            if reference.value_row_count > 0:
                for value_batch in dataset.iter_value_training_chunks(reference.game.game_id):
                    _assert_cpu_value(value_batch, label="native validation batch")
                    value_output = model.forward_stable_value(value_batch.batch)
                    _assert_cpu_value(value_output, label="native validation output", finite=True)
                    value_loss = native_stable_value_loss(
                        value_output,
                        value_targets=value_batch.value_targets,
                        row_weights=value_batch.row_weights,
                        row_mask=value_batch.row_mask,
                        head_normalizer=float(value_games),
                    )
                    bce = float(value_loss.bce)
                    if not math.isfinite(bce):
                        raise ValueError("native validation value metric must be finite")
                    value_bce += bce
    if not all(math.isfinite(value) for value in (policy_cross_entropy, policy_entropy, value_bce)):
        raise ValueError("native validation accumulated metrics must be finite")
    return NativeValidationMetrics(
        validation_ledger_digest=ledger.digest,
        game_ids=tuple(item.game.game_id for item in ledger.games),
        dataset_digests=tuple(item.dataset_digest for item in ledger.datasets),
        source_digests=tuple(item.source_digest for item in ledger.datasets),
        completion_receipt_digests=tuple(
            item.completion_receipt_digest for item in ledger.datasets
        ),
        game_count=len(ledger.games),
        policy_contributing_game_count=policy_games,
        policy_row_count=sum(item.policy_row_count for item in ledger.games),
        policy_cross_entropy=(float(policy_cross_entropy) if policy_games > 0 else None),
        policy_entropy=float(policy_entropy) if policy_games > 0 else None,
        value_contributing_game_count=value_games,
        value_row_count=sum(item.value_row_count for item in ledger.games),
        value_bce=float(value_bce) if value_games > 0 else None,
    )


def evaluate_native_validation(
    model: Gen1PolicyValueModel,
    ledger: NativeValidationLedger,
    *,
    bindings: Sequence[NativeReplayDatasetBinding],
) -> NativeValidationMetrics:
    """Read-only, chunk-streamed held-out metrics over exact ledger authorities."""
    if type(model) is not Gen1PolicyValueModel:
        raise TypeError("model must be an exact Gen1PolicyValueModel")
    validated_ledger = _strict_model(ledger, NativeValidationLedger, label="validation ledger")
    binding_values = tuple(_strict_sequence(bindings, label="bindings"))
    if any(not isinstance(item, NativeReplayDatasetBinding) for item in binding_values):
        raise TypeError("bindings must contain only NativeReplayDatasetBinding instances")

    snapshot = _ModelSnapshot(model)
    result: NativeValidationMetrics | None = None
    caught: tuple[type[BaseException], BaseException, Any] | None = None
    try:
        result = _evaluate(
            model,
            validated_ledger,
            bindings=tuple(binding_values),  # type: ignore[arg-type]
        )
    except BaseException:
        caught = sys.exc_info()  # type: ignore[assignment]
    try:
        mutated = snapshot.restore_and_report_mutation(model)
    except BaseException as restore_error:
        if caught is not None:
            _, original, traceback = caught
            original.add_note(
                "native validation snapshot restoration also failed: "
                f"{type(restore_error).__name__}: {restore_error}"
            )
            raise original.with_traceback(traceback) from restore_error
        raise
    if caught is not None:
        _, original, traceback = caught
        raise original.with_traceback(traceback)
    if mutated:
        raise RuntimeError("native validation mutated model state or CPU Torch RNG")
    assert result is not None
    return result


__all__ = [
    "NativeValidationDatasetRef",
    "NativeValidationGameRef",
    "NativeValidationLedger",
    "NativeValidationMetrics",
    "create_native_validation_ledger",
    "evaluate_native_validation",
]
