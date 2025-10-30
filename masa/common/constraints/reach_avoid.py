from __future__ import annotations
from typing import Any, Dict
from .base import ConstraintEnv, CostFn

class ReachAvoidEnv(ConstraintEnv):
    """Reach target set while avoiding unsafe labels (cost==1)."""

    def __init__(self, env: gym.Env, cost_fn: CostFn, target_label: str):
        super().__init__(env, cost_fn=cost_fn)
        self.target_label = target_label

    def _reset(self):
        self.reached = False
        self.violated = False   

    def _update(self, info):
        cost = self.cost_fn(info)
        violated = float(cost) >= 0.5
        self.reached = self.target_label in info["labels"]

    def satisfied(self) -> bool:
        return self.reached and not self.violated

    def episode_metric(self) -> Dict[str, float]:
        return {"reached": self.reached, "violated": self.violated, "satisfied": float(self.satisfied())}


    
