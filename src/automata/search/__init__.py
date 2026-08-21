"""Search: Information-Set MCTS agent for the GoA2 engine.

Cut B — opponents are folded into a fixed default policy and their hidden
commits into `determinize`, so the tree is single-perspective (all MAX nodes).
Rollouts truncate at a round-count cutoff and defer to `evaluate_state`.
"""

from .agent import ISMCTSAgent
from .config import SearchConfig
from .learned_policy import LearnedPolicy
from .observation import RootChildStats, RootSearchObservation, RootSearchObserver
from .policy_features import POLICY_FEATURE_SCHEMA_ID, policy_candidate_features
from .prior import Policy

__all__ = [
    "POLICY_FEATURE_SCHEMA_ID",
    "ISMCTSAgent",
    "LearnedPolicy",
    "Policy",
    "RootChildStats",
    "RootSearchObservation",
    "RootSearchObserver",
    "SearchConfig",
    "policy_candidate_features",
]
