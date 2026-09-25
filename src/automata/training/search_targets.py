"""Training targets and traces derived from completed root searches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from automata.decision import DecisionDescriptor
from automata.models.contracts import CandidateID, EncodedCandidate
from automata.search.node import Key
from goa2.domain.input import InputRequest
from goa2.domain.state import GameState
from goa2.domain.types import HeroID


class SearchActionTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    candidate: EncodedCandidate
    prior_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_count: StrictInt = Field(ge=0)
    mean_value: float = Field(ge=-1.0, le=1.0)
    value_variance: float = Field(ge=0.0)
    improved_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    selected: bool = False

    @model_validator(mode="after")
    def _finite_statistics(self) -> SearchActionTarget:
        values = (self.mean_value, self.value_variance)
        probabilities = (self.prior_probability, self.improved_probability)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action values must be finite")
        if not all(value is None or math.isfinite(value) for value in probabilities):
            raise ValueError("action probabilities must be finite")
        return self


class SearchPolicyTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    actions: tuple[SearchActionTarget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_alignment(self) -> SearchPolicyTarget:
        candidate_ids = tuple(action.candidate.candidate_id for action in self.actions)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("action statistics contain a duplicate candidate")
        if sum(action.selected for action in self.actions) > 1:
            raise ValueError("at most one action may be selected")
        self._validate_optional_distribution("prior_probability")
        self._validate_optional_distribution("improved_probability")
        return self

    def _validate_optional_distribution(
        self, field: Literal["prior_probability", "improved_probability"]
    ) -> None:
        values = tuple(getattr(action, field) for action in self.actions)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError(f"{field} must be present for every action or none")
        total = sum(value for value in values if value is not None)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{field} must sum to one")

    @property
    def selected_candidate_id(self) -> CandidateID | None:
        selected = next((action for action in self.actions if action.selected), None)
        return selected.candidate.candidate_id if selected is not None else None

    @property
    def has_selection(self) -> bool:
        return any(action.selected for action in self.actions)


@dataclass(frozen=True, slots=True)
class RootActionStatistics:
    visits: int
    total_value: float
    q: float


@dataclass(frozen=True, slots=True)
class RootSearchTrace:
    decision_owner_hero_id: str
    decision_kind: Literal["CARD", "INPUT"]
    request: InputRequest | None
    legal_keys: tuple[Key, ...]
    chosen_key: Key | None
    action_statistics: Mapping[Key, RootActionStatistics]
    policy_target: SearchPolicyTarget | None = None


def root_search_trace_from_result(
    *,
    state: GameState,
    owner_hero_id: str,
    decision_kind: Literal["CARD", "INPUT"],
    request: InputRequest | None,
    legal: Sequence[Key],
    result: object,
    include_policy_target: bool = False,
) -> RootSearchTrace:
    """Adapt one model-neutral ISMCTS result into a training trace."""
    from automata.search.ismcts import SearchResult

    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    statistics: dict[Key, RootActionStatistics] = {}
    total_samples = 0
    for key in legal:
        child = result.root.children.get(key)
        stats = (
            RootActionStatistics(visits=0, total_value=0.0, q=0.0)
            if child is None
            else RootActionStatistics(child.visits, child.total_value, child.q)
        )
        statistics[key] = stats
        total_samples += stats.visits

    policy_target = None
    if include_policy_target:
        from automata.observation import encode_decision

        hero = state.get_hero(HeroID(owner_hero_id))
        if hero is None or hero.team is None:
            raise ValueError(f"unknown decision owner {owner_hero_id!r}")
        decision = DecisionDescriptor(
            decision_kind,
            hero=hero if decision_kind == "CARD" else None,
            request=request,
            can_finish_planning=decision_kind == "CARD" and None in legal,
        )
        encoded = encode_decision(
            state,
            decision,
            legal,
            decision_owner_hero_id=owner_hero_id,
            perspective_team=hero.team.value,
        )
        has_selection = result.best_key in legal
        actions = tuple(
            SearchActionTarget(
                candidate=candidate,
                sample_count=statistics[key].visits,
                mean_value=statistics[key].q,
                value_variance=(
                    result.root.children[key].value_variance if key in result.root.children else 0.0
                ),
                improved_probability=(
                    statistics[key].visits / total_samples if total_samples else None
                ),
                selected=has_selection and key == result.best_key,
            )
            for key, candidate in zip(legal, encoded.candidates, strict=True)
        )
        policy_target = SearchPolicyTarget(actions=actions)

    return RootSearchTrace(
        decision_owner_hero_id=owner_hero_id,
        decision_kind=decision_kind,
        request=request,
        legal_keys=tuple(legal),
        chosen_key=result.best_key,
        action_statistics=MappingProxyType(statistics),
        policy_target=policy_target,
    )


__all__ = [
    "RootActionStatistics",
    "RootSearchTrace",
    "SearchActionTarget",
    "SearchPolicyTarget",
    "root_search_trace_from_result",
]
