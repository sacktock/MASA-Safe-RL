from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints import Constraint, BaseConstraintEnv, CostFn

class ProbabilisticSafety(Constraint):
    """Undiscounted probabilistic constraint: track fraction of unsafe steps and compare to alpha."""

    def __init__(self, cost_fn: CostFn, alpha: float):
        self.cost_fn = cost_fn
        self.alpha = alpha

    def reset(self):
        self.unsafe = 0
        self.total = 0

    def update(self, labels):
        cost = self.cost_fn(labels)
        self.unsafe = float(cost >= 0.5)
        self.total += 1

    def prob_unsafe(self) -> float:
        return (self.unsafe / self.total)

    def satisfied(self) -> bool:
        return self.prob_unsafe() <= self.alpha

    def episode_metric(self) -> Dict[str, float]:
        return {"p_unsafe": self.prob_unsafe(), "satisfied": float(self.satisfied())}
    
    @property
    def constraint_type(self) -> str:
        return "prob"


class ProbabilisticSafetyEnv(BaseConstraintEnv):
    """Gymnasium wrapper for probabilistic constraint."""

    def __init__(self, env: gym.Env, cost_fn: CostFn, alpha: float):
        super().__init__(env, ProbabilisticSafety(cost_fn=cost_fn, alpha=alpha))

