"""Learned-model-neutral strategy boundary for classic action selection."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, TypeVar

from automata.agents.contracts import Agent
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

from ..config import SearchConfig
from ..continuation import AgentContinuationPolicy, as_continuation_policy
from ..contracts import ContinuationPolicy, LeafEvaluator, SearchPolicy
from ..node import Key
from ..root import RootTarget
from .engine import CutoffObserver, SearchResult, search

CandidateT = TypeVar("CandidateT")
SearchRunner = Callable[..., SearchResult]
TemperatureProvider = Callable[[GameState], float]


def _validated_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("temperature must be finite and non-negative")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    return temperature


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


class VisitSamplingStrategy:
    """Sample self-play actions from a delegate's root visit distribution.

    This wrapper is deliberately opt-in: callers that use ``ISMCTSStrategy``
    directly retain robust-child/argmax selection. A zero temperature also
    preserves the delegate's selected action after validating its statistics.
    """

    strategy_id = "ismcts-visit-sampling"

    def __init__(
        self,
        delegate: SearchStrategy,
        *,
        temperature: float | None = None,
        temperature_provider: TemperatureProvider | None = None,
        seed: int,
    ) -> None:
        if (temperature is None) == (temperature_provider is None):
            raise ValueError("provide exactly one of temperature or temperature_provider")
        if temperature_provider is not None and not callable(temperature_provider):
            raise ValueError("temperature_provider must be callable")
        self._delegate = delegate
        self._temperature = _validated_temperature(temperature) if temperature is not None else None
        self._temperature_provider = temperature_provider
        self._rng = random.Random(seed)

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]:
        legal = tuple(legal_candidates)
        result = self._delegate.select(state, perspective_team, root_target, legal)
        if result.candidates != legal:
            raise ValueError("search strategy changed or reordered the legal candidates")

        statistics = result.search_result
        if statistics is None:
            raise ValueError("visit sampling requires search statistics")
        if statistics.best_key != result.selected_candidate:
            raise ValueError("search statistics best action disagrees with the delegated action")
        if any(key not in legal for key in statistics.root.children):
            raise ValueError("search statistics contain actions outside the legal root")

        visits = tuple(
            statistics.root.children[key].visits if key in statistics.root.children else 0
            for key in legal
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in visits
        ):
            raise ValueError("root visits must be non-negative integers")
        temperature = (
            self._temperature
            if self._temperature_provider is None
            else _validated_temperature(self._temperature_provider(state))
        )
        assert temperature is not None
        positive = tuple(index for index, count in enumerate(visits) if count > 0)
        if not positive:
            if len(legal) != 1:
                raise ValueError("visit sampling requires positive root visits")
            return StrategyResult(legal, 0, search_result=statistics)
        if temperature == 0:
            return result

        # Subtracting the largest log-weight is a common scaling factor and
        # keeps visits ** (1 / temperature) finite for very small temperatures.
        log_weights = tuple(math.log(visits[index]) / temperature for index in positive)
        maximum = max(log_weights)
        weights = tuple(math.exp(weight - maximum) for weight in log_weights)
        selected_index = self._rng.choices(positive, weights=weights, k=1)[0]
        return StrategyResult(legal, selected_index, search_result=statistics)


class ISMCTSStrategy:
    strategy_id = "ismcts"

    def __init__(
        self,
        *,
        environment_policy: Agent,
        config: SearchConfig,
        prior: SearchPolicy | None = None,
        leaf_evaluator: LeafEvaluator | None = None,
        continuation_policy: ContinuationPolicy | Agent | None = None,
        cutoff_observer: CutoffObserver | None = None,
        search_runner: SearchRunner | None = None,
    ) -> None:
        self._environment_policy = environment_policy
        self._config = config
        self._prior = prior
        self._leaf_evaluator = leaf_evaluator
        self._continuation_policy = (
            AgentContinuationPolicy(environment_policy)
            if continuation_policy is None
            else as_continuation_policy(continuation_policy)
        )
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


__all__ = [
    "ISMCTSStrategy",
    "SearchStrategy",
    "StrategyResult",
    "VisitSamplingStrategy",
]
