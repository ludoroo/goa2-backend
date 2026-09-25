"""Neutral contracts for canonical offline policy-ranking evidence."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from automata.decision import DecisionSemanticRole

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PolicyRankingBucket:
    """One ranking bucket; ``count`` is informative pairwise decisions only."""

    count: int
    multi_candidate_count: int
    game_count: int
    pairwise_accuracy: float | None

    def __post_init__(self) -> None:
        if self.pairwise_accuracy is not None and type(self.pairwise_accuracy) is not float:
            raise ValueError("policy ranking pairwise accuracy must be a float or null")
        for name in ("count", "multi_candidate_count", "game_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"policy ranking {name} must be a non-negative integer")
        if self.count > self.multi_candidate_count:
            raise ValueError("informative pairwise count cannot exceed multi-candidate count")
        if self.game_count > self.count:
            raise ValueError("policy ranking game count cannot exceed informative count")
        if self.count == 0:
            if self.game_count != 0 or self.pairwise_accuracy is not None:
                raise ValueError("empty policy ranking evidence cannot have games or accuracy")
        elif (
            self.game_count == 0
            or self.pairwise_accuracy is None
            or not math.isfinite(self.pairwise_accuracy)
            or not (0.0 <= self.pairwise_accuracy <= 1.0)
        ):
            raise ValueError("populated policy ranking evidence must have games and accuracy")

    def canonical_data(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "game_count": self.game_count,
            "multi_candidate_count": self.multi_candidate_count,
            "pairwise_accuracy": self.pairwise_accuracy,
        }


@dataclass(frozen=True, slots=True)
class PolicyRankingSnapshot:
    """Pinned artifact ranking evidence on one exact validation membership."""

    artifact_digest: str
    dataset_digest: str
    split_digest: str
    overall: PolicyRankingBucket
    by_semantic_role: Mapping[DecisionSemanticRole, PolicyRankingBucket]
    schema_version: int = 1
    split_name: str = "validation"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or type(self.split_name) is not str:
            raise ValueError("policy ranking snapshot schema/split types are invalid")
        if self.schema_version != 1 or self.split_name != "validation":
            raise ValueError("policy ranking snapshot schema/split is unsupported")
        for name in ("artifact_digest", "dataset_digest", "split_digest"):
            if _DIGEST_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if any(not isinstance(role, DecisionSemanticRole) for role in self.by_semantic_role):
            raise ValueError("policy ranking role keys must be DecisionSemanticRole values")
        if any(
            not isinstance(bucket, PolicyRankingBucket) for bucket in self.by_semantic_role.values()
        ):
            raise ValueError("policy ranking role values must be PolicyRankingBucket values")
        frozen = {
            role: self.by_semantic_role[role]
            for role in sorted(self.by_semantic_role, key=lambda item: item.value)
        }
        object.__setattr__(self, "by_semantic_role", MappingProxyType(frozen))

    def canonical_data(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
            "by_semantic_role": {
                role.value: bucket.canonical_data()
                for role, bucket in self.by_semantic_role.items()
            },
            "dataset_digest": self.dataset_digest,
            "overall": self.overall.canonical_data(),
            "schema_version": self.schema_version,
            "split_digest": self.split_digest,
            "split_name": self.split_name,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> PolicyRankingSnapshot:
        try:
            value = json.loads(payload)
            expected = {
                "artifact_digest",
                "by_semantic_role",
                "dataset_digest",
                "overall",
                "schema_version",
                "split_digest",
                "split_name",
            }
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError("snapshot fields are incomplete")
            raw_roles = value["by_semantic_role"]
            if not isinstance(raw_roles, dict):
                raise ValueError("snapshot semantic-role evidence is invalid")
            snapshot = cls(
                artifact_digest=value["artifact_digest"],
                dataset_digest=value["dataset_digest"],
                split_digest=value["split_digest"],
                overall=PolicyRankingBucket(**value["overall"]),
                by_semantic_role={
                    DecisionSemanticRole(role): PolicyRankingBucket(**bucket)
                    for role, bucket in raw_roles.items()
                },
                schema_version=value["schema_version"],
                split_name=value["split_name"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("policy ranking snapshot is invalid") from exc
        if payload != snapshot.canonical_bytes():
            raise ValueError("policy ranking snapshot is not canonical JSON")
        return snapshot


def _ranking_bucket_from_metrics(value: Any, *, label: str) -> PolicyRankingBucket:
    if not isinstance(value, Mapping):
        raise ValueError(f"policy metrics {label} bucket is missing")
    required = (
        "pairwise_count",
        "multi_candidate_count",
        "pairwise_game_count",
        "pairwise_accuracy",
    )
    if any(name not in value for name in required):
        raise ValueError(f"policy metrics {label} bucket lacks ranking evidence")
    accuracy = value["pairwise_accuracy"]
    if accuracy is not None and (
        isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
    ):
        raise ValueError(f"policy metrics {label} pairwise accuracy is invalid")
    return PolicyRankingBucket(
        count=value["pairwise_count"],
        multi_candidate_count=value["multi_candidate_count"],
        game_count=value["pairwise_game_count"],
        pairwise_accuracy=None if accuracy is None else float(accuracy),
    )


def snapshot_from_policy_metrics(
    policy_metrics: Mapping[str, Any],
    *,
    artifact_digest: str,
    dataset_digest: str,
    split_digest: str,
) -> PolicyRankingSnapshot:
    """Build strict gate evidence from one validation policy-metrics object."""

    by_role = policy_metrics.get("by_semantic_role")
    if not isinstance(by_role, Mapping):
        raise ValueError("policy metrics are missing semantic-role buckets")
    expected_roles = {role.value for role in DecisionSemanticRole}
    if set(by_role) != expected_roles:
        raise ValueError("policy metrics semantic-role buckets are incomplete")
    return PolicyRankingSnapshot(
        artifact_digest=artifact_digest,
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        overall=_ranking_bucket_from_metrics(policy_metrics.get("overall"), label="overall"),
        by_semantic_role={
            role: _ranking_bucket_from_metrics(by_role[role.value], label=role.value)
            for role in DecisionSemanticRole
        },
    )


__all__ = [
    "PolicyRankingBucket",
    "PolicyRankingSnapshot",
    "snapshot_from_policy_metrics",
]
