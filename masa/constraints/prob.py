from __future__ import annotations
from typing import Any, Dict
from .base import ConstraintEnv, CostFn

class StepWiseProbabilisticEnv(ConstraintEnv):
    """Undiscounted probabilistic constraint: track fraction of unsafe steps and compare to alpha."""

    def __init__(self, env: gym.Env, cost_fn: CostFn, alpha: float):
        super().__init__(env, cost_fn=cost_fn)
        self.alpha = alpha

    def _reset(self):
        self.unsafe = 0
        self.total = 0

    def _update(self, info):
        cost = self.cost_fn(info)
        self.unsafe = float(cost >= 0.5)
        self.total += 1

    def prob_unsafe(self) -> float:
        return (self.unsafe / self.total)

    def satisfied(self) -> bool:
        return self.prob_unsafe() <= self.alpha

    def episode_metric(self) -> Dict[str, float]:
        return {"p_unsafe": self.prob_unsafe(), "alpha": self.alpha, "satisfied": float(self.satisfied())}

