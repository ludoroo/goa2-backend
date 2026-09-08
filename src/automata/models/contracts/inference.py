"""Learned inference input/output and runtime contracts."""

from __future__ import annotations

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .candidates import CandidateID, DecisionObservation, _require_unique
from .observation import _Contract


def _require_finite(values: tuple[float, ...], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite numbers")


def _require_probability_distribution(values: tuple[float, ...], name: str) -> None:
    _require_finite(values, name)
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} probabilities must be in the range [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name} probabilities must sum to one")


class PolicyValueOutput(_Contract[Literal[1]]):
    candidate_ids: tuple[CandidateID, ...]
    policy_logits: tuple[float, ...]
    value: float

    @model_validator(mode="after")
    def _valid_output(self) -> PolicyValueOutput:
        _require_unique(self.candidate_ids)
        if not self.candidate_ids:
            raise ValueError("policy output must contain at least one candidate")
        if len(self.candidate_ids) != len(self.policy_logits):
            raise ValueError("candidate IDs and policy logits must have aligned lengths")
        _require_finite(self.policy_logits, "policy logits")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError("value must be finite and in the range [-1, 1]")
        return self


class SearchOutcome(_Contract[Literal[1]]):
    candidate_ids: tuple[CandidateID, ...]
    prior_probabilities: tuple[float, ...]
    sample_counts: tuple[int, ...]
    mean_values: tuple[float, ...]
    value_variances: tuple[float, ...]
    improved_probabilities: tuple[float, ...]
    selected_candidate_id: CandidateID

    @model_validator(mode="after")
    def _valid_outcome(self) -> SearchOutcome:
        _require_unique(self.candidate_ids)
        aligned = (
            self.prior_probabilities,
            self.sample_counts,
            self.mean_values,
            self.value_variances,
            self.improved_probabilities,
        )
        if any(len(values) != len(self.candidate_ids) for values in aligned):
            raise ValueError("candidate statistics must have aligned lengths")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must belong to candidate IDs")
        if any(count < 0 for count in self.sample_counts):
            raise ValueError("sample counts cannot be negative")
        _require_probability_distribution(self.prior_probabilities, "prior")
        _require_probability_distribution(self.improved_probabilities, "improved")
        _require_finite(self.mean_values, "mean values")
        if any(not math.isfinite(value) or value < 0.0 for value in self.value_variances):
            raise ValueError("value variances must be finite and non-negative")
        if any(not -1.0 <= value <= 1.0 for value in self.mean_values):
            raise ValueError("mean values must be in the range [-1, 1]")
        return self


class LearnedModelOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_ids: tuple[CandidateID, ...]
    policy_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    value: float

    @field_validator("policy_logits", "probabilities")
    @classmethod
    def finite_scores(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("learned model scores must be finite")
        return values

    @model_validator(mode="after")
    def aligned_output(self) -> LearnedModelOutput:
        size = len(self.candidate_ids)
        if len(self.policy_logits) != size or len(self.probabilities) != size:
            raise ValueError("learned model scores must align with candidate IDs")
        if any(not 0.0 <= probability <= 1.0 for probability in self.probabilities):
            raise ValueError("probabilities must be in [0, 1]")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError("value must be finite and in [-1, 1]")
        return self


class LearnedModelRuntime(Protocol):
    def evaluate(self, observation: DecisionObservation) -> LearnedModelOutput: ...


__all__ = [
    "LearnedModelOutput",
    "LearnedModelRuntime",
    "PolicyValueOutput",
    "SearchOutcome",
]
