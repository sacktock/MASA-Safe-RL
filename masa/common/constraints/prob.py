from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints import Constraint, BaseConstraintEnv, CostFn
from masa.examples.dummy import cost_fn as dummy_cost_fn

class ProbabilisticSafety(Constraint):
    """Undiscounted probabilistic constraint: track fraction of unsafe steps and compare to alpha."""

    def __init__(self, cost_fn: CostFn, alpha: float):
        self.cost_fn = cost_fn
        self.alpha = alpha

    def reset(self):
        self.total = 0
        self.total_unsafe = 0.0
        self.step_cost = 0.0

    def update(self, labels):
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe += float(self.step_cost >= 0.5)
        self.total += 1

    def prob_unsafe(self) -> float:
        return (self.total_unsafe / self.total)

    def satisfied(self) -> bool:
        return self.prob_unsafe() <= self.alpha

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_unsafe": float(self.total_unsafe), "p_unsafe": self.prob_unsafe(), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": self.step_cost, "cum_unsafe": float(self.total_unsafe), "p_unsafe": self.prob_unsafe(), "satisfied": float(self.satisfied())}
    
    @property
    def constraint_type(self) -> str:
        return "prob"


class ProbabilisticSafetyEnv(BaseConstraintEnv):
    """Gymnasium wrapper for probabilistic constraint."""

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, alpha: float = 0.01, **kw):
        super().__init__(env, ProbabilisticSafety(cost_fn=cost_fn, alpha=alpha), **kw)

