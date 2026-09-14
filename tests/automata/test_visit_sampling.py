from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from automata.search import VisitSamplingStrategy
from automata.search.ismcts import SearchResult
from automata.search.ismcts.strategy import StrategyResult
from automata.search.node import Node


class _FixedStrategy:
    strategy_id = "fixed"

    def __init__(
        self,
        candidates: tuple[str, ...],
        visits: tuple[int, ...],
        *,
        selected_index: int = 0,
        best_key: str | None = None,
        include_statistics: bool = True,
    ) -> None:
        self._candidates = candidates
        self._visits = visits
        self._selected_index = selected_index
        self._best_key = candidates[selected_index] if best_key is None else best_key
        self._include_statistics = include_statistics

    def select(self, *args: Any, **kwargs: Any) -> StrategyResult[str]:
        del args, kwargs
        statistics = None
        if self._include_statistics:
            statistics = SearchResult(
                Node(
                    children={
                        candidate: Node(visits=visits)
                        for candidate, visits in zip(self._candidates, self._visits, strict=True)
                    }
                ),
                self._best_key,
            )
        return StrategyResult(
            self._candidates,
            self._selected_index,
            search_result=statistics,
        )


def _select(strategy: VisitSamplingStrategy) -> StrategyResult[str]:
    return strategy.select(None, None, None, ("best", "other"))  # type: ignore[arg-type]


def test_zero_temperature_preserves_delegated_argmax_selection() -> None:
    delegate = _FixedStrategy(("best", "other"), (9, 1))

    result = _select(VisitSamplingStrategy(delegate, temperature=0, seed=2))

    assert result.selected_index == 0
    assert result.selected_candidate == result.search_result.best_key == "best"


def test_positive_temperature_samples_reproducibly_and_can_choose_non_best() -> None:
    first = VisitSamplingStrategy(_FixedStrategy(("best", "other"), (9, 1)), temperature=1, seed=2)
    second = VisitSamplingStrategy(_FixedStrategy(("best", "other"), (9, 1)), temperature=1, seed=2)

    first_sequence = [_select(first).selected_index for _ in range(5)]
    second_sequence = [_select(second).selected_index for _ in range(5)]

    assert first_sequence == second_sequence
    assert first_sequence[0] == 1
    assert (
        _select(
            VisitSamplingStrategy(_FixedStrategy(("best", "other"), (9, 1)), temperature=10, seed=5)
        ).selected_index
        == 1
    )
    assert (
        _select(
            VisitSamplingStrategy(_FixedStrategy(("best", "other"), (9, 1)), temperature=1, seed=5)
        ).selected_index
        == 0
    )


def test_positive_temperature_never_samples_zero_visit_actions() -> None:
    strategy = VisitSamplingStrategy(
        _FixedStrategy(("best", "other"), (0, 3), selected_index=1),
        temperature=100,
        seed=7,
    )

    assert {_select(strategy).selected_index for _ in range(10)} == {1}


def test_singleton_zero_visit_root_remains_forced() -> None:
    strategy = VisitSamplingStrategy(_FixedStrategy(("forced",), (0,)), temperature=1, seed=7)

    result = strategy.select(None, None, None, ("forced",))  # type: ignore[arg-type]

    assert result.selected_index == 0
    assert result.selected_candidate == "forced"


@pytest.mark.parametrize(
    ("delegate", "message"),
    [
        (_FixedStrategy(("best", "other"), (0, 0)), "positive root visits"),
        (_FixedStrategy(("other", "best"), (1, 1)), "changed or reordered"),
        (
            _FixedStrategy(("best", "other"), (1, -1)),
            "non-negative integers",
        ),
        (
            _FixedStrategy(
                ("best", "other"),
                (1, 1),
                selected_index=1,
                best_key="best",
            ),
            "best action",
        ),
        (
            _FixedStrategy(("best", "other"), (1, 1), include_statistics=False),
            "search statistics",
        ),
    ],
)
def test_sampling_fails_closed_on_malformed_or_misaligned_search_results(
    delegate: _FixedStrategy, message: str
) -> None:
    strategy = VisitSamplingStrategy(delegate, temperature=1, seed=7)

    with pytest.raises(ValueError, match=message):
        _select(strategy)


def test_sampling_configuration_requires_finite_non_negative_temperature() -> None:
    delegate = _FixedStrategy(("best", "other"), (1, 1))

    for temperature in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            VisitSamplingStrategy(delegate, temperature=temperature, seed=1)


def test_state_dependent_temperature_provider_must_be_callable() -> None:
    with pytest.raises(ValueError, match="temperature_provider"):
        VisitSamplingStrategy(
            _FixedStrategy(("best", "other"), (9, 1)),
            temperature_provider=1.0,  # type: ignore[arg-type]
            seed=2,
        )


def test_state_dependent_temperature_is_resolved_and_validated_on_every_call() -> None:
    rounds: list[int] = []

    def temperature(state: Any) -> float:
        rounds.append(state.round)
        return 1.0 if state.round < 9 else 0.0

    strategy = VisitSamplingStrategy(
        _FixedStrategy(("best", "other"), (9, 1)),
        temperature_provider=temperature,
        seed=2,
    )

    assert strategy.select(SimpleNamespace(round=1), None, None, ("best", "other")).selected_index == 1  # type: ignore[arg-type]
    assert strategy.select(SimpleNamespace(round=9), None, None, ("best", "other")).selected_index == 0  # type: ignore[arg-type]
    assert rounds == [1, 9]

    invalid = VisitSamplingStrategy(
        _FixedStrategy(("best", "other"), (9, 1)),
        temperature_provider=lambda _state: float("nan"),
        seed=2,
    )
    with pytest.raises(ValueError, match="temperature"):
        invalid.select(SimpleNamespace(round=1), None, None, ("best", "other"))  # type: ignore[arg-type]


def test_constant_provider_is_exactly_compatible_with_static_temperature() -> None:
    static = VisitSamplingStrategy(
        _FixedStrategy(("best", "other"), (9, 1)), temperature=1.0, seed=2
    )
    provided = VisitSamplingStrategy(
        _FixedStrategy(("best", "other"), (9, 1)),
        temperature_provider=lambda _state: 1.0,
        seed=2,
    )

    static_sequence = [_select(static).selected_index for _ in range(8)]
    provided_sequence = [
        provided.select(SimpleNamespace(round=round_number), None, None, ("best", "other")).selected_index  # type: ignore[arg-type]
        for round_number in range(1, 9)
    ]

    assert provided_sequence == static_sequence
