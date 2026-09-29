"""Training targets and traces derived from completed root searches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from automata.decision import DecisionDescriptor
from automata.models.contracts import CandidateID, EncodedCandidate
from automata.search.node import Key, action_key
from goa2.domain.input import InputRequest
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

if TYPE_CHECKING:
    from automata.search.ismcts.strategy import StrategyResult


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


def search_policy_target_from_result(
    result: StrategyResult[Key], candidates: Sequence[EncodedCandidate]
) -> SearchPolicyTarget:
    """Build an aligned policy target from one completed strategy result."""
    if tuple(action_key(candidate.selection) for candidate in candidates) != result.candidates:
        raise ValueError("encoded candidates must align with the exact ordered search choices")
    statistics = result.search_result
    if statistics is None:
        raise ValueError("self-play search strategy must return improved action statistics")
    if statistics.best_key not in result.candidates:
        raise ValueError("search statistics best action is outside the legal root")
    if any(key not in result.candidates for key in statistics.root.children):
        raise ValueError("search statistics contain actions outside the legal root")

    visits = tuple(
        statistics.root.children[key].visits if key in statistics.root.children else 0
        for key in result.candidates
    )
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in visits):
        raise ValueError("self-play root visits must be non-negative integers")
    total_visits = sum(visits)
    if (
        statistics.effective_iterations is not None
        and total_visits != statistics.effective_iterations
    ):
        # Live serving may recover a completed prefix after a cooperative
        # deadline. That is not complete offline teacher evidence, even if
        # the actual game could continue to an otherwise normal outcome.
        raise ValueError("self-play search did not complete its effective visit budget")
    improved_probabilities: tuple[float, ...]
    if not total_visits:
        if len(visits) != 1:
            raise ValueError("self-play search statistics contain no root visits")
        # Search intentionally leaves validated forced roots unvisited. Keep
        # the zero sample/value sentinel while recording the only policy mass.
        improved_probabilities = (1.0,)
    else:
        improved_probabilities = tuple(count / total_visits for count in visits)

    diagnostics = {}
    for diagnostic in statistics.root_action_diagnostics:
        if diagnostic.action not in result.candidates:
            raise ValueError("search diagnostics contain actions outside the legal root")
        if diagnostic.action in diagnostics:
            raise ValueError("search diagnostics contain a duplicate root action")
        child = statistics.root.children.get(diagnostic.action)
        expected_visits = child.visits if child is not None else 0
        expected_mean = child.q if child is not None else 0.0
        expected_variance = child.value_variance if child is not None else 0.0
        if (
            diagnostic.visits != expected_visits
            or not math.isclose(diagnostic.mean_value, expected_mean)
            or not math.isclose(diagnostic.value_variance, expected_variance)
        ):
            raise ValueError("search diagnostics disagree with root statistics")
        prior = diagnostic.prior_probability
        if prior is not None and (
            isinstance(prior, bool)
            or not isinstance(prior, (int, float))
            or not math.isfinite(prior)
            or not 0.0 <= prior <= 1.0
        ):
            raise ValueError("search diagnostics contain an invalid root prior")
        diagnostics[diagnostic.action] = diagnostic

    has_complete_priors = len(diagnostics) == len(result.candidates) and all(
        diagnostics[key].prior_probability is not None for key in result.candidates
    )
    if has_complete_priors:
        priors: tuple[float | None, ...] = tuple(
            cast(float, diagnostics[key].prior_probability) for key in result.candidates
        )
        if not math.isclose(sum(cast(tuple[float, ...], priors)), 1.0):
            raise ValueError("search diagnostics root priors must sum to one")
    else:
        # The action-target contract is all-or-none. Partial diagnostics are
        # not evidence for a full prior distribution and must not be filled
        # from visit counts.
        priors = (None,) * len(result.candidates)

    return SearchPolicyTarget(
        actions=tuple(
            SearchActionTarget(
                schema_version=1,
                candidate=candidate,
                prior_probability=priors[index],
                sample_count=visits[index],
                mean_value=(
                    statistics.root.children[key].q if key in statistics.root.children else 0.0
                ),
                value_variance=(
                    statistics.root.children[key].value_variance
                    if key in statistics.root.children
                    else 0.0
                ),
                improved_probability=improved_probabilities[index],
                selected=index == result.selected_index,
            )
            for index, (key, candidate) in enumerate(
                zip(result.candidates, candidates, strict=True)
            )
        )
    )


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
    "search_policy_target_from_result",
]
