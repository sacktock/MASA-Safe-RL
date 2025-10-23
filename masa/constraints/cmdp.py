from __future__ import annotations
from typing import Any, Dict
from .base import ConstraintEnv, CostFn

class CumulativeCostEnv(ConstraintEnv):
    """CMDP-style cumulative cost with threshold."""

    def __init__(self, env: gym.Env, cost_fn: CostFn, budget: float):
        super().__init__(env, cost_fn=cost_fn)
        self.budget = budget

    def _reset(self):
        self.total = 0.0

    def _update(self, info):
        cost = self.cost_fn(info)
        self.total += cost

    def satisfied(self) -> bool:
        return self.total <= self.budget

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_cost": self.total, "budget": self.budget, "satisfied": float(self.satisfied())}


    
