"""Encode candidate-free observations at stable value boundaries."""

from __future__ import annotations

from automata.models.contracts import StableValueObservation, Viewer
from automata.runtime.value_boundary import (
    StableValueBoundary,
    detect_stable_value_boundary,
)
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from .graph.encoder import encode_snapshot
from .projector import project_snapshot


def encode_stable_value(
    state: GameState,
    boundary: StableValueBoundary,
    *,
    viewer_hero_id: str,
    perspective_team: TeamColor,
) -> StableValueObservation:
    """Encode one actual stable boundary without inventing a policy decision."""
    actual = detect_stable_value_boundary(state)
    if actual is None:
        raise ValueError("state is not at a stable value boundary")
    if boundary != actual:
        raise ValueError("stable value boundary does not match the current state")

    viewer = Viewer(
        schema_version=2,
        private_hero_id=viewer_hero_id,
        perspective_team=perspective_team.value,
    )
    snapshot = project_snapshot(state, viewer, current_owner_id=boundary.actor_id)
    # Finishing effects may leave an advisory actor on the live state after
    # planning opens. A value boundary, not that historical actor, determines
    # graph context. Override the owner's None explicitly too: the generic
    # projector otherwise falls back to the state's resolution owner.
    snapshot = snapshot.model_copy(
        update={
            "public_state": {
                **snapshot.public_state,
                "current_actor_id": boundary.actor_id,
                "automata_context": {
                    "resolution_owner_id": boundary.actor_id,
                    "acting_piece_id": state.acting_piece_id if boundary.actor_id else None,
                },
            }
        }
    )
    graph = encode_snapshot(snapshot)
    return StableValueObservation(
        schema_version=1,
        state=graph,
        boundary_kind=boundary.kind.value,
    )


__all__ = ["encode_stable_value"]
