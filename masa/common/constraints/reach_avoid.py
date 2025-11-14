from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints import Constraint, BaseConstraintEnv

class ReachAvoid(Constraint):
    """Reach target label set while avoiding unsafe label set."""

    def __init__(avoid_label: str, reach_label: str):
        self.avoid_label = avoid_label
        self.reach_label = reach_label

    def reset(self):
        self.reached = False
        self.violated = False  

    def update(self, labels):
        self.violated = self.avoid_label in lables
        self.reached = self.reach_label in labels

    def satisfied(self) -> bool:
        return self.reached and not self.violated

    def episode_metric(self) -> Dict[str, float]:
        return {"reached": self.reached, "violated": self.violated, "satisfied": float(self.satisfied())}

    @property
    def constraint_type(self) -> str:
        return "reach_avoid"


class ReachAvoidEnv(BaseConstraintEnv):
    """Gymansium wrapper for Reach-avoid."""

    def __init__(self, env: gym.Env, avoid_label: str = "unsafe", reach_label: str = "target"):
        super().__init__(env, ReachAvoid(avoid_label=avoid_label, reach_label=reach_label))


    
