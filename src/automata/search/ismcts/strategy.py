"""Learned-model-neutral strategy boundary for classic action selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from automata.agents.contracts import Agent
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from ..config import SearchConfig
from ..contracts import LeafEvaluator, SearchPolicy
from ..node import Key
from ..root import RootTarget
from .engine import CutoffObserver, SearchResult, search

CandidateT = TypeVar("CandidateT")
SearchRunner = Callable[..., SearchResult]


@dataclass(frozen=True, slots=True)
class StrategyResult(Generic[CandidateT]):
    candidates: tuple[CandidateT, ...]
    selected_index: int
    search_result: SearchResult | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("StrategyResult requires at least one candidate")
        if not 0 <= self.selected_index < len(self.candidates):
            raise ValueError("selected_index must identify a candidate")

    @property
    def selected_candidate(self) -> CandidateT:
        return self.candidates[self.selected_index]


class SearchStrategy(Protocol):
    strategy_id: str

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]: ...


class ISMCTSStrategy:
    strategy_id = "ismcts"

    def __init__(
        self,
        *,
        environment_policy: Agent,
        config: SearchConfig,
        prior: SearchPolicy | None = None,
        leaf_evaluator: LeafEvaluator | None = None,
        continuation_policy: Agent | None = None,
        cutoff_observer: CutoffObserver | None = None,
        search_runner: SearchRunner | None = None,
    ) -> None:
        self._environment_policy = environment_policy
        self._config = config
        self._prior = prior
        self._leaf_evaluator = leaf_evaluator
        self._continuation_policy = continuation_policy or environment_policy
        self._cutoff_observer = cutoff_observer
        self._search_runner = search_runner or search

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]:
        candidates = tuple(legal_candidates)
        result = self._search_runner(
            state,
            perspective_team,
            candidates,
            self._environment_policy,
            self._config,
            self._prior,
            root_target=root_target,
            cutoff_observer=self._cutoff_observer,
            continuation_policy=self._continuation_policy,
            leaf_evaluator=self._leaf_evaluator,
        )
        try:
            selected_index = candidates.index(result.best_key)
        except ValueError as exc:
            raise ValueError("ISMCTS selected a candidate outside the legal root") from exc
        return StrategyResult(candidates, selected_index, search_result=result)


__all__ = ["ISMCTSStrategy", "SearchStrategy", "StrategyResult"]
