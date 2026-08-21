"""Root-search observations and their synchronous observer contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from goa2.domain.input import InputRequest
from goa2.domain.state import GameState

from .node import Key


@dataclass(frozen=True, slots=True)
class RootChildStats:
    """Search statistics for one legal root action."""

    visits: int
    total_value: float
    q: float


@dataclass(frozen=True, slots=True)
class RootSearchObservation:
    """Read-only snapshot emitted after one successful root search."""

    decision_owner_hero_id: str
    decision_kind: Literal["CARD", "INPUT"]
    request: InputRequest | None
    legal_keys: tuple[Key, ...]
    chosen_key: Key | None
    child_stats: Mapping[Key, RootChildStats]


class RootSearchObserver(Protocol):
    """Receive the root state and completed search snapshot synchronously.

    ``state`` is borrowed, read-only, and is the exact pre-decision root state.
    Observers must extract what they need during this call and must not retain
    or mutate it.
    """

    def __call__(self, state: GameState, observation: RootSearchObservation) -> None: ...
