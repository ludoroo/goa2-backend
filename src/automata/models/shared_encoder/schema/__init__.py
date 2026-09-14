"""Feature schema and vectorization for the shared-encoder model family."""

from .feature_schema import (
    TENSOR_SCHEMA_ID,
    TENSOR_SCHEMA_VERSION,
    CategoricalFeature,
    HashedStringFeature,
    IgnoredFeature,
    NumericFeature,
    RecordFeatureSchema,
    ReferenceFeature,
    TensorFeatureSchema,
    TensorSchemaID,
    TensorSchemaVersion,
    VectorizedCandidate,
    VectorizedDecision,
    VectorizedRelationship,
    VectorizedToken,
    expanded_numeric_width,
)

__all__ = [
    "TENSOR_SCHEMA_ID",
    "TENSOR_SCHEMA_VERSION",
    "CategoricalFeature",
    "HashedStringFeature",
    "IgnoredFeature",
    "NumericFeature",
    "RecordFeatureSchema",
    "ReferenceFeature",
    "TensorFeatureSchema",
    "TensorSchemaID",
    "TensorSchemaVersion",
    "VectorizedCandidate",
    "VectorizedDecision",
    "VectorizedRelationship",
    "VectorizedToken",
    "expanded_numeric_width",
]
