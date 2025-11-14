from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints import Constraint, BaseConstraintEnv, CostFn
from masa.examples.dummy import cost_fn as dummy_cost_fn

class CumulativeCost(Constraint):
    """CMDP-style cumulative cost constraint with threshold."""

    def __init__(self, cost_fn: CostFn, budget: float):
        self.cost_fn = cost_fn
        self.budget = budget

    def reset(self):
        self.total = 0.0

    def update(self, labels):
        cost = self.cost_fn(labels)
        self.total += cost

    def satisfied(self) -> bool:
        return self.total <= self.budget

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_cost": self.total, "satisfied": float(self.satisfied())}

    @property
    def constraint_type(self) -> str:
        return "cmdp"

class CumulativeCostEnv(BaseConstraintEnv):
    """Gymnasium wrapper for CMDP-style constraint."""

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, budget: float = 20.0):
        super().__init__(env, CumulativeCost(cost_fn=cost_fn, budget=budget))



    
