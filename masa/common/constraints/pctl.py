"""
Overview
--------

Probabilistic CTL (PCTL) style constraint monitor (simplified).

The current implementation mirrors the safety-style structure used elsewhere:
it accumulates unsafe occurrences based on a per-step cost function applied to
labels. While named "PCTL", this monitor presently behaves like a boolean
"safety so far" tracker under the local convention ``cost >= 0.5``.

Conceptually, a PCTL-style safety constraint might aim to bound the probability
of reaching unsafe states, e.g.:

.. math::

   \\Pr(\\Diamond\\,\\mathsf{unsafe}) \\le \\alpha,

but note that this file's current implementation does not compute an explicit
probability estimate; it tracks whether any unsafe event occurred.

API Reference
-------------
"""

from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints.base import Constraint, BaseConstraintEnv, CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn

class PCTL(Constraint):
    """Simplified PCTL-named monitor tracking whether any unsafe step occurred.

    Args:
        cost_fn: Mapping from label sets to scalar cost.
        alpha: Threshold parameter stored for downstream use (not currently used
            in the logic in this file).

    Attributes:
        cost_fn: Cost function ``labels -> float``.
        alpha: User-specified parameter (reserved for probabilistic variants).
        safe: True until an unsafe cost is observed.
        step_cost: Most recent cost value.
        total_unsafe: Count of unsafe steps (as floats).

    """

    def __init__(self, cost_fn: CostFn, alpha: float):
        self.cost_fn = cost_fn
        self.alpha = alpha

    def reset(self):
        """Reset episode counters."""
        self.safe = True
        self.step_cost = 0.0
        self.total_unsafe = 0.0
    
    def update(self, labels: Iterable[str]):
        """Update safety flags from the current label set.

        Args:
            labels: Iterable of atomic propositions.
        """
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe += float(self.step_cost >= 0.5)
        self.safe = self.safe and (not self.total_unsafe)

    def satisfied(self) -> bool:
        """Whether the episode remains safe so far."""
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        """End-of-episode metrics.

        Returns:
            Dict containing:

            - ``"cum_unsafe"``: count of unsafe steps,
            - ``"satisfied"``: 1.0 if safe else 0.0.
        """
        return {"cum_unsafe": float(self.total_unsafe), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        """Per-step metrics.

        Returns:
            Dict containing:

            - ``"cost"``: current step cost,
            - ``"violation"``: 1.0 if ``cost >= 0.5`` else 0.0.
        """
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5)}

    @property
    def constraint_type(self) -> str:
        """Stable identifier string: ``"pctl"``."""
        return "pctl"

class PCTLEnv(BaseConstraintEnv):
    """Gymnasium wrapper for the :class:`PCTL` monitor.

    Args:
        env: Base environment (must be a :class:`~masa.common.labelled_env.LabelledEnv`).
        cost_fn: Cost function mapping labels to a scalar.
        alpha: Threshold parameter stored on the monitor (see :class:`PCTL`).
        **kw: Extra keyword arguments forwarded to :class:`BaseConstraintEnv`.

    """

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, alpha: float = 0.01, **kw):
        super().__init__(env, PCTL(cost_fn=cost_fn, alpha=alpha), **kw)

