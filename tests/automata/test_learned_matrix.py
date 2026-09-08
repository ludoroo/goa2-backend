from automata.agents.heuristic_agent import HeuristicAgent
from automata.evaluation.learned_matrix import LearnedMatrixCell, build_learned_ismcts
from automata.search.config import SearchConfig
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.learned import LearnedLeafEvaluator, LearnedSearchPolicy


class Runtime:
    def evaluate(self, observation):  # pragma: no cover - construction seam only
        raise AssertionError


def test_hh_hl_lh_ll_map_only_to_ismcts_policy_and_leaf_seams() -> None:
    runtime = Runtime()
    expected = {
        LearnedMatrixCell.HH: (False, False),
        LearnedMatrixCell.HL: (False, True),
        LearnedMatrixCell.LH: (True, False),
        LearnedMatrixCell.LL: (True, True),
    }

    for cell, (learned_policy, learned_leaf) in expected.items():
        strategy = build_learned_ismcts(
            cell,
            runtime=runtime,
            default_policy=HeuristicAgent(seed=3),
            config=SearchConfig(iterations=1, seed=4),
        )
        assert isinstance(strategy._prior, LearnedSearchPolicy) is learned_policy
        assert isinstance(strategy._leaf_evaluator, LearnedLeafEvaluator) is learned_leaf
        assert isinstance(strategy._leaf_evaluator, HeuristicLeafEvaluator) is not learned_leaf


def test_learned_cells_require_a_runtime_but_hh_does_not() -> None:
    build_learned_ismcts(
        LearnedMatrixCell.HH,
        runtime=None,
        default_policy=HeuristicAgent(seed=1),
        config=SearchConfig(iterations=1),
    )

    for cell in (LearnedMatrixCell.HL, LearnedMatrixCell.LH, LearnedMatrixCell.LL):
        try:
            build_learned_ismcts(
                cell,
                runtime=None,
                default_policy=HeuristicAgent(seed=1),
                config=SearchConfig(iterations=1),
            )
        except ValueError as exc:
            assert "runtime" in str(exc)
        else:
            raise AssertionError(f"{cell} accepted no Learned runtime")
