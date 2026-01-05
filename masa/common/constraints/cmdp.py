from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints.base import Constraint, BaseConstraintEnv, CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn

class CumulativeCost(Constraint):
    """CMDP-style cumulative cost constraint with threshold."""

    def __init__(self, cost_fn: CostFn, budget: float):
        self.cost_fn = cost_fn
        self.budget = budget

    def reset(self):
        self.total = 0.0
        self.step_cost = 0.0

    def update(self, labels: Iterable[str]):
        self.step_cost = self.cost_fn(labels)
        self.total += self.step_cost

    def satisfied(self) -> bool:
        return self.total <= self.budget

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_cost": self.total, "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5), "cum_cost": self.total}

    @property
    def constraint_type(self) -> str:
        return "cmdp"

class CumulativeCostEnv(BaseConstraintEnv):
    """Gymnasium wrapper for CMDP-style constraint."""

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, budget: float = 20.0, **kw):
        super().__init__(env, CumulativeCost(cost_fn=cost_fn, budget=budget), **kw)



    
