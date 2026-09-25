"""Offline teachers must not inherit serving's partial-deadline recovery."""

import pytest

from automata.search.config import SearchConfig, parse_learned_lh_search_config


@pytest.mark.parametrize("seconds", [0.01, 30.0])
def test_offline_search_rejects_wall_clock_early_returns(seconds: float) -> None:
    with pytest.raises(ValueError, match=r"offline.*decision_timeout_seconds.*None"):
        parse_learned_lh_search_config({"decision_timeout_seconds": seconds})


def test_offline_search_pins_disabled_internal_deadlines_in_its_identity() -> None:
    for raw in ({}, {"decision_timeout_seconds": None}):
        config, identity = parse_learned_lh_search_config(raw)
        assert config.decision_timeout_seconds is None
        assert identity["decision_timeout_seconds"] is None


def test_serving_search_keeps_cooperative_deadline_support() -> None:
    config = SearchConfig(decision_timeout_seconds=1.0)
    assert config.decision_timeout_seconds == 1.0
