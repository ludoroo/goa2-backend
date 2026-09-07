"""Feature schema and vectorization for the shared-encoder model family."""

from .feature_schema import (
    CategoricalFeature,
    IgnoredFeature,
    NumericFeature,
    RecordFeatureSchema,
    ReferenceFeature,
    TensorFeatureSchema,
    VectorizedCandidate,
    VectorizedDecision,
    VectorizedRelationship,
    VectorizedToken,
)

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
