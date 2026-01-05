from __future__ import annotations
from typing import Any, Dict, Iterable
from masa.common.constraints.base import Constraint, BaseConstraintEnv

class ReachAvoid(Constraint):
    """Reach target label set while avoiding unsafe label set."""

    def __init__(avoid_label: str, reach_label: str):
        self.avoid_label = avoid_label
        self.reach_label = reach_label

    def reset(self):
        self.reached = False
        self.violated = False
        self.satisfied = False

    def update(self, labels: Iterable[str]):
        self.reach = self.reach_label in labels
        self.avoid = self.avoid_label not in labels

        self.reached = self.reached or self.reach
        self.violated = self.violated or bool(not self.avoid)

        self.satisfied = self.satisfied or (self.reached and bool(not self.violated))

    def episode_metric(self) -> Dict[str, float]:
        return {"reached": self.reached, "violated": self.violated, "satisfied": float(self.satisfied)}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": float(not self.avoid), "violation": bool(not self.avoid), "reached": self.reach}

    @property
    def constraint_type(self) -> str:
        return "reach_avoid"

class ReachAvoidEnv(BaseConstraintEnv):
    """Gymansium wrapper for Reach-avoid."""

    def __init__(self, env: gym.Env, avoid_label: str = "unsafe", reach_label: str = "target", **kw):
        super().__init__(env, ReachAvoid(avoid_label=avoid_label, reach_label=reach_label), **kw)


    
