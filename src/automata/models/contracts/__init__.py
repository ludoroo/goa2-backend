"""Generic, framework-independent learned-model contracts."""

from .artifacts import (
    ArtifactError,
    ArtifactScope,
    RuntimeRequirements,
)
from .candidates import (
    ActionCandidateID,
    CandidateID,
    CardCandidateID,
    DecisionObservation,
    EncodedCandidate,
    EntityCandidateID,
    FinishCandidateID,
    HexCandidateID,
    NumberCandidateID,
    OptionCandidateID,
    SkipCandidateID,
    UnitCandidateID,
)
from .compatibility import (
    CURRENT_MAP_SCHEMA_VERSION,
    CURRENT_RUNTIME_COMPATIBILITY_VERSION,
    INITIAL_RELATIONSHIP_NAMES,
)
from .inference import LearnedModelOutput, LearnedModelRuntime, PolicyValueOutput, SearchOutcome
from .observation import (
    LearnedObservation,
    ObservationRelationship,
    ObservationToken,
    PublicSnapshot,
    Viewer,
)
from .serialization import canonical_json_bytes, from_canonical_json

__all__ = [
    "CURRENT_MAP_SCHEMA_VERSION",
    "CURRENT_RUNTIME_COMPATIBILITY_VERSION",
    "INITIAL_RELATIONSHIP_NAMES",
    "ActionCandidateID",
    "ArtifactError",
    "ArtifactScope",
    "CandidateID",
    "CardCandidateID",
    "DecisionObservation",
    "EncodedCandidate",
    "EntityCandidateID",
    "FinishCandidateID",
    "HexCandidateID",
    "LearnedModelOutput",
    "LearnedModelRuntime",
    "LearnedObservation",
    "NumberCandidateID",
    "ObservationRelationship",
    "ObservationToken",
    "OptionCandidateID",
    "PolicyValueOutput",
    "PublicSnapshot",
    "RuntimeRequirements",
    "SearchOutcome",
    "SkipCandidateID",
    "UnitCandidateID",
    "Viewer",
    "canonical_json_bytes",
    "from_canonical_json",
]
