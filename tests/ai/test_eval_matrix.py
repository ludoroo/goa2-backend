"""Tests for the (legacy) evaluation matrix CLI helpers.

These are cheap unit-level tests that exercise ``run_matrix`` **without**
running any real games or ISMCTS searches — every matchup's ``evaluate`` call
is monkeypatched to return a canned :class:`MatchupResult`. Coverage kept:

- All configured matchups are dispatched, with the expected labels.
- ``games`` vs ``search_games`` sample-size split is honored.
- ``_result_dict`` produces the JSON-serializable shape a baseline consumer
  depends on.

Full-strength ISMCTS matrix runs are integration territory; unit tests must
never invoke them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from automata.evaluation import cli as cli_module
from automata.evaluation.cli import _result_dict, run_matrix
from automata.evaluation.matchup import MatchupResult


def _fake_result(label_a: str, label_b: str, games: int) -> MatchupResult:
    """Canned matchup outcome the fake ``evaluate`` returns."""
    return MatchupResult(
        label_a=label_a,
        label_b=label_b,
        games=games,
        a_wins=games,
        b_wins=0,
        draws=0,
        avg_rounds=12.0,
    )


@pytest.fixture()
def fake_evaluate(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace :func:`evaluate` inside the CLI module with a recorder.

    Returns the list of calls so tests can inspect labels + sample sizes
    without paying the cost of a real game.
    """
    calls: list[dict[str, Any]] = []

    def _fake(
        a_factory: Callable[[int], Any],
        b_factory: Callable[[int], Any],
        *,
        red_heroes: list[str],
        blue_heroes: list[str],
        games: int,
        base_seed: int,
        label_a: str,
        label_b: str,
        **_: Any,
    ) -> MatchupResult:
        calls.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                "games": games,
                "base_seed": base_seed,
                "red_heroes": list(red_heroes),
                "blue_heroes": list(blue_heroes),
            }
        )
        return _fake_result(label_a, label_b, games)

    monkeypatch.setattr(cli_module, "evaluate", _fake)
    return calls


def test_run_matrix_dispatches_every_configured_matchup(
    fake_evaluate: list[dict[str, Any]],
) -> None:
    results = run_matrix(games=8, base_seed=0, search_games=2, search_iters=2)

    # Each configured matchup produces exactly one result.
    assert len(results) == 4
    labels = [(r.label_a, r.label_b) for r in results]
    assert ("random", "random") in labels
    assert ("heuristic", "random") in labels
    assert ("ismcts", "heuristic") in labels
    assert ("ismcts", "ismcts_noprior") in labels

    # Fast matchups use `games`; ISMCTS matchups use the smaller `search_games`.
    fast_games = {c["games"] for c in fake_evaluate if "ismcts" not in c["label_a"]}
    search_games = {c["games"] for c in fake_evaluate if c["label_a"] == "ismcts"}
    assert fast_games == {8}
    assert search_games == {2}


def test_result_dict_is_json_serializable(fake_evaluate: list[dict[str, Any]]) -> None:
    results = run_matrix(games=4, base_seed=0, search_games=1, search_iters=2)
    for r in results:
        d = _result_dict(r)
        restored = json.loads(json.dumps(d))
        assert set(restored) >= {"a", "b", "games", "a_winrate", "wilson_ci"}
        assert 0.0 <= restored["a_winrate"] <= 1.0
        lo, hi = restored["wilson_ci"]
        assert 0.0 <= lo <= hi <= 1.0
