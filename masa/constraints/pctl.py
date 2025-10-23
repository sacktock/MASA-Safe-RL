from __future__ import annotations
from typing import Any, Dict
from .base import ConstraintEnv, CostFn

class PCTLEnv(ConstraintEnv):
    """Probabilistic CTL constraint (on the initial states): monitors the undiscounetd probability of being safe"""

    def __init__(self, env: gym.Env, cost_fn: CostFn, alpha: float):
        super().__init__(env, cost_fn=cost_fn)

    def _reset(self):
        self.safe = True

    def _udpate(self, info):
        cost = self.cost_fn(info)
        unsafe = float(cost >= 0.5)
        self.safe = self.safe and (not unsafe)

    def satisfied(self) -> bool:
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        return {"satisfied": float(self.satisfied())}

