from __future__ import annotations

from types import SimpleNamespace

import pytest

from automata.evaluation import matchup
from automata.harness.game_runner import RunResult
from automata.runtime.outcomes import WinnerSide


def _factory(seed: int) -> object:
    return object()


def _result(
    *,
    winner: str | None,
    winner_side: WinnerSide | None,
    reason: str = "game_over",
    rounds: int = 3,
) -> RunResult:
    return RunResult(
        winner=winner,
        winner_side=winner_side,
        reason=reason,
        rounds=rounds,
        turns=2,
        steps=20,
    )


def test_evaluate_uses_normalized_side_and_preserves_alternating_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter(
        [
            _result(winner="RED", winner_side="RED"),
            _result(winner="hero_arien", winner_side="BLUE"),
            _result(winner=None, winner_side=None),
            _result(winner="hero_wasp", winner_side="RED"),
        ]
    )
    monkeypatch.setattr(matchup, "run_game", lambda *_args, **_kwargs: next(outcomes))

    result = matchup.evaluate(
        _factory,
        _factory,
        red_heroes=["Wasp"],
        blue_heroes=["Arien"],
        games=4,
    )

    assert (result.a_wins, result.b_wins, result.draws) == (2, 1, 1)
    assert result.avg_rounds == 3.0


def test_evaluate_does_not_infer_a_side_from_raw_hero_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        matchup,
        "run_game",
        lambda *_args, **_kwargs: _result(winner="hero_wasp", winner_side="BLUE"),
    )

    result = matchup.evaluate(
        _factory,
        _factory,
        red_heroes=["Wasp"],
        blue_heroes=["Arien"],
        games=1,
    )

    assert (result.a_wins, result.b_wins, result.draws) == (0, 1, 0)


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_result(winner=None, winner_side=None, reason="max_steps"), "non-game_over"),
        (
            SimpleNamespace(
                winner="mystery",
                winner_side="GREEN",
                reason="game_over",
                rounds=3,
            ),
            "winner_side",
        ),
    ],
)
def test_evaluate_rejects_censored_or_unknown_normalized_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    outcome: RunResult | SimpleNamespace,
    message: str,
) -> None:
    monkeypatch.setattr(matchup, "run_game", lambda *_args, **_kwargs: outcome)

    with pytest.raises(ValueError, match=message):
        matchup.evaluate(
            _factory,
            _factory,
            red_heroes=["Wasp"],
            blue_heroes=["Arien"],
            games=1,
        )
