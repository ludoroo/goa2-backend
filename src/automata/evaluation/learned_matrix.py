"""Canonical Learned policy/value ablation matrix for ISMCTS."""

from __future__ import annotations

from enum import StrEnum

from automata.agents.heuristic_agent import HeuristicAgent
from automata.models.contracts import LearnedModelRuntime
from automata.search.config import SearchConfig
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.strategy import ISMCTSStrategy
from automata.search.learned import LearnedLeafEvaluator, LearnedSearchPolicy


class LearnedMatrixCell(StrEnum):
    """Policy/leaf sources: H is Heuristic and L is Learned."""

    HH = "H/H"
    HL = "H/L"
    LH = "L/H"
    LL = "L/L"


def build_learned_ismcts(
    cell: LearnedMatrixCell,
    *,
    runtime: LearnedModelRuntime | None,
    default_policy: HeuristicAgent,
    config: SearchConfig,
) -> ISMCTSStrategy:
    """Build ISMCTS without conflating its policy and leaf-value seams."""
    uses_learned_policy = cell in {LearnedMatrixCell.LH, LearnedMatrixCell.LL}
    uses_learned_leaf = cell in {LearnedMatrixCell.HL, LearnedMatrixCell.LL}
    if (uses_learned_policy or uses_learned_leaf) and runtime is None:
        raise ValueError(f"{cell.value} requires a Learned runtime")
    if runtime is None:
        assert not uses_learned_policy and not uses_learned_leaf

    prior = (
        LearnedSearchPolicy(runtime)
        if runtime and uses_learned_policy
        else HeuristicPrior(default_policy)
    )
    leaf = (
        LearnedLeafEvaluator(runtime) if runtime and uses_learned_leaf else HeuristicLeafEvaluator()
    )
    return ISMCTSStrategy(
        environment_policy=default_policy,
        config=config,
        prior=prior,
        leaf_evaluator=leaf,
    )


__all__ = ["LearnedMatrixCell", "build_learned_ismcts"]
