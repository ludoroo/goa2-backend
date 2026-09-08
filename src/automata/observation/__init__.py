"""Information-safe observation projection."""

from .decision_encoder import encode_decision, encode_search_context, legal_keys_for_decision
from .graph import encode_snapshot
from .hero_adapters import HeroObservationAdapter, HeroObservationAdapterRegistry
from .projector import project_snapshot

__all__ = [
    "HeroObservationAdapter",
    "HeroObservationAdapterRegistry",
    "encode_decision",
    "encode_search_context",
    "encode_snapshot",
    "legal_keys_for_decision",
    "project_snapshot",
]
