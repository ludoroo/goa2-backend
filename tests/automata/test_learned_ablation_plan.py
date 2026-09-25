from automata.evaluation.ablation import AblationPlan, BudgetKind
from automata.evaluation.learned_matrix import LearnedMatrixCell
from automata.search.contracts import LeafMode


def test_ablation_plan_crosses_matrix_horizon_and_budget_without_changing_seeds() -> None:
    cases = AblationPlan(seeds=(101, 102), iteration_budget=32, wall_clock_seconds=0.25).cases()

    assert len(cases) == 2 * 4 * 2 * 2
    assert {case.cell for case in cases} == set(LearnedMatrixCell)
    assert {case.leaf_mode for case in cases} == {
        LeafMode.IMMEDIATE,
        LeafMode.BOUNDED_CONTINUATION,
    }
    assert {case.budget_kind for case in cases} == set(BudgetKind)
    assert {case.seed for case in cases} == {101, 102}
    assert len({case.case_id for case in cases}) == len(cases)


def test_matrix_exposes_policy_and_value_marginal_pairs() -> None:
    pairs = AblationPlan(seeds=(1,), iteration_budget=1, wall_clock_seconds=0.1).marginals()

    assert pairs["policy"] == (
        (LearnedMatrixCell.HH, LearnedMatrixCell.LH),
        (LearnedMatrixCell.HL, LearnedMatrixCell.LL),
    )
    assert pairs["value"] == (
        (LearnedMatrixCell.HH, LearnedMatrixCell.HL),
        (LearnedMatrixCell.LH, LearnedMatrixCell.LL),
    )
