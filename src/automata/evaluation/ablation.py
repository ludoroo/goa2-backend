"""Predeclared, paired Learned ablation and search-boundary schedule."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from automata.search.contracts import LeafMode

from .learned_matrix import LearnedMatrixCell


class BudgetKind(StrEnum):
    ITERATIONS = "equal_iterations"
    WALL_CLOCK = "equal_wall_clock"


@dataclass(frozen=True, slots=True)
class AblationCase:
    case_id: str
    seed: int
    cell: LearnedMatrixCell
    leaf_mode: LeafMode
    budget_kind: BudgetKind
    iterations: int | None
    wall_clock_seconds: float | None


@dataclass(frozen=True, slots=True)
class AblationPlan:
    seeds: tuple[int, ...]
    iteration_budget: int
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("ablation seeds must be non-empty and unique")
        if self.iteration_budget <= 0 or self.wall_clock_seconds <= 0:
            raise ValueError("ablation budgets must be positive")

    def cases(self) -> tuple[AblationCase, ...]:
        cases = []
        for seed in self.seeds:
            for cell in LearnedMatrixCell:
                for leaf_mode in (LeafMode.IMMEDIATE, LeafMode.BOUNDED_CONTINUATION):
                    for budget in BudgetKind:
                        identity = f"{seed}:{cell.value}:{leaf_mode.value}:{budget.value}"
                        cases.append(
                            AblationCase(
                                case_id=hashlib.sha256(identity.encode()).hexdigest(),
                                seed=seed,
                                cell=cell,
                                leaf_mode=leaf_mode,
                                budget_kind=budget,
                                iterations=(
                                    self.iteration_budget
                                    if budget is BudgetKind.ITERATIONS
                                    else None
                                ),
                                wall_clock_seconds=(
                                    self.wall_clock_seconds
                                    if budget is BudgetKind.WALL_CLOCK
                                    else None
                                ),
                            )
                        )
        return tuple(cases)

    def marginals(
        self,
    ) -> dict[str, tuple[tuple[LearnedMatrixCell, LearnedMatrixCell], ...]]:
        return {
            "policy": (
                (LearnedMatrixCell.HH, LearnedMatrixCell.LH),
                (LearnedMatrixCell.HL, LearnedMatrixCell.LL),
            ),
            "value": (
                (LearnedMatrixCell.HH, LearnedMatrixCell.HL),
                (LearnedMatrixCell.LH, LearnedMatrixCell.LL),
            ),
        }


__all__ = ["AblationCase", "AblationPlan", "BudgetKind"]
