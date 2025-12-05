from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints import Constraint, BaseConstraintEnv, CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn

class PCTL(Constraint):
    """Probabilistic CTL constraint (on the initial states): monitors the undiscounetd probability of being safe."""

    def __init__(self, cost_fn: CostFn, alpha: float):
        self.cost_fn = cost_fn
        self.alpha = alpha

    def reset(self):
        self.safe = True
        self.step_cost = 0.0
        self.total_unsafe = 0.0
    
    def update(self, labels: Iterable[str]):
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe = float(self.step_cost >= 0.5)
        self.safe = self.safe and (not self.total_unsafe)

    def satisfied(self) -> bool:
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_unsafe": float(self.total_unsafe), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5)}

    @property
    def constraint_type(self) -> str:
        return "pctl"

class PCTLEnv(BaseConstraintEnv):
    """ Gymnasium wrapper for Probabilistic CTL constraint."""

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, alpha: float = 0.01, **kw):
        super().__init__(env, PCTL(cost_fn=cost_fn, alpha=alpha), **kw)

