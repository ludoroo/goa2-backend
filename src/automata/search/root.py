"""Shared contracts and validation for classic search root decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from goa2.domain.input import InputRequest
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from .node import Key

RootKind = Literal["CARD", "INPUT"]


class RootMismatchError(ValueError):
    """The current state did not surface the root requested by a caller."""


class RootDecision(Protocol):
    kind: str
    hero: Any | None
    request: InputRequest | None


@dataclass(frozen=True)
class RootTarget:
    kind: RootKind
    owned_hero_ids: frozenset[str]
    decision_owner_hero_id: str
    hero_id: str | None = None
    request_id: str | None = None
    player_id: str | None = None

    def __post_init__(self) -> None:
        if not self.owned_hero_ids:
            raise ValueError("RootTarget requires a non-empty owned_hero_ids set")
        if not self.decision_owner_hero_id:
            raise ValueError("RootTarget requires decision_owner_hero_id")
        if self.decision_owner_hero_id not in self.owned_hero_ids:
            raise ValueError("RootTarget decision owner must be in owned_hero_ids")
        if self.kind == "CARD":
            if self.hero_id is None or self.hero_id not in self.owned_hero_ids:
                raise ValueError("CARD RootTarget requires an owned hero_id")
            if self.request_id is not None or self.player_id is not None:
                raise ValueError("CARD RootTarget must not carry request_id/player_id")
        elif self.kind == "INPUT":
            if self.request_id is None or self.player_id is None:
                raise ValueError("INPUT RootTarget requires request_id and player_id")
            if self.hero_id is not None:
                raise ValueError("INPUT RootTarget must not carry hero_id")
        else:
            raise ValueError(f"Unknown RootTarget kind: {self.kind!r}")

    @classmethod
    def card(cls, *, hero_id: str, owned_hero_ids: frozenset[str]) -> RootTarget:
        return cls("CARD", owned_hero_ids, hero_id, hero_id=hero_id)

    @classmethod
    def input(
        cls,
        *,
        request_id: str,
        player_id: str,
        owned_hero_ids: frozenset[str],
        decision_owner_hero_id: str | None = None,
    ) -> RootTarget:
        if decision_owner_hero_id is None:
            raise ValueError("INPUT RootTarget requires a decision owner")
        return cls(
            "INPUT",
            owned_hero_ids,
            decision_owner_hero_id,
            request_id=request_id,
            player_id=player_id,
        )

    def matches(self, decision: RootDecision) -> bool:
        if decision.kind != self.kind:
            return False
        if self.kind == "CARD":
            return decision.hero is not None and getattr(decision.hero, "id", None) == self.hero_id
        return (
            decision.request is not None
            and decision.request.id == self.request_id
            and decision.request.player_id == self.player_id
        )


def _team_of_player(state: GameState, player_id: str) -> TeamColor | None:
    if player_id.startswith("team:"):
        name = player_id.split(":", 1)[1]
        return next(
            (color for color in state.teams if color.value == name or color.name == name), None
        )
    hero = state.get_hero(HeroID(player_id))
    return hero.team if hero is not None else None


def validate_root_decision(
    state: GameState,
    perspective_team: TeamColor,
    target: RootTarget,
    surfaced: RootDecision,
) -> None:
    if target.kind == "CARD":
        assert target.hero_id is not None
        if state.get_hero(HeroID(target.hero_id)) is None:
            raise RootMismatchError(
                f"CARD RootTarget references hero_id {target.hero_id!r} that does not exist"
            )
    else:
        assert target.player_id is not None
        if _team_of_player(state, target.player_id) is None:
            raise RootMismatchError(
                f"INPUT RootTarget references player_id {target.player_id!r} that does not exist"
            )

    owner = state.get_hero(HeroID(target.decision_owner_hero_id))
    if owner is None:
        raise RootMismatchError(f"decision owner {target.decision_owner_hero_id!r} does not exist")
    if owner.team != perspective_team:
        raise RootMismatchError("decision owner team disagrees with value perspective")
    if target.kind == "CARD" and target.hero_id != target.decision_owner_hero_id:
        raise RootMismatchError("CARD target owner must be the targeted hero")
    if target.kind == "INPUT" and owner.team != _team_of_player(state, target.player_id or ""):
        raise RootMismatchError("decision owner is ineligible for addressed player/team")
    if not target.matches(surfaced):
        surfaced_id = (
            getattr(surfaced.hero, "id", None)
            if surfaced.hero is not None
            else surfaced.request.id if surfaced.request is not None else None
        )
        surfaced_player = surfaced.request.player_id if surfaced.request is not None else None
        raise RootMismatchError(
            f"simulator surfaced {surfaced.kind}(id={surfaced_id!r}, "
            f"player_id={surfaced_player!r}) but root target was {target.kind}("
            f"hero_id={target.hero_id!r}, request_id={target.request_id!r}, "
            f"player_id={target.player_id!r})"
        )


def validate_root_legal(
    legal_candidates: Sequence[Key], canonical_legal: Sequence[Key]
) -> tuple[Key, ...]:
    if not legal_candidates:
        raise ValueError("root validation requires non-empty legal candidates")
    caller_counts = Counter(legal_candidates)
    canonical_counts = Counter(canonical_legal)
    if caller_counts != canonical_counts:
        missing = canonical_counts - caller_counts
        extra = caller_counts - canonical_counts
        raise RootMismatchError(
            "caller legal candidates disagree with canonical legal keys: "
            f"missing={sorted(missing.elements(), key=repr)!r}, "
            f"extra={sorted(extra.elements(), key=repr)!r} "
            f"(caller={list(legal_candidates)!r}, canonical={list(canonical_legal)!r})"
        )
    return tuple(legal_candidates)


@dataclass(frozen=True)
class ValidatedRoot:
    target: RootTarget
    decision: RootDecision
    legal_candidates: tuple[Key, ...]

    @property
    def is_singleton(self) -> bool:
        return len(self.legal_candidates) == 1

    @property
    def singleton_candidate(self) -> Key:
        if not self.is_singleton:
            raise ValueError("root does not have exactly one legal candidate")
        return self.legal_candidates[0]


def validate_root(
    state: GameState,
    perspective_team: TeamColor,
    target: RootTarget,
    surfaced: RootDecision,
    *,
    legal_candidates: Sequence[Key],
    canonical_legal: Sequence[Key],
) -> ValidatedRoot:
    validate_root_decision(state, perspective_team, target, surfaced)
    legal = validate_root_legal(legal_candidates, canonical_legal)
    return ValidatedRoot(target, surfaced, legal)
