import pytest

from automata.agents.capabilities import BoundedComputeCapability, bounded_compute_capability
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.agents.random_agent import RandomAgent
from automata.search.config import PROD_DEFAULT_DECISION_TIMEOUT_SECONDS, SearchConfig


def test_ismcts_advertises_generic_bounded_compute() -> None:
    capability = bounded_compute_capability(ISMCTSAgent(SearchConfig(iterations=1)))
    assert capability is not None
    assert capability.decision_timeout_seconds == PROD_DEFAULT_DECISION_TIMEOUT_SECONDS
    assert isinstance(capability.create_fallback(seed=9), HeuristicAgent)


def test_bounded_compute_capability_is_structural_and_validated() -> None:
    capability = BoundedComputeCapability(0.25, RandomAgent)
    agent = type("FutureAgent", (), {"bounded_compute": capability})()
    assert bounded_compute_capability(agent) is capability
    with pytest.raises(ValueError):
        BoundedComputeCapability(0, RandomAgent)
