"""
Overview
--------

Cumulative-cost constraints in the CMDP style.

This module provides a simple *budgeted cumulative cost* constraint, commonly
used to model constrained MDPs (CMDPs). At each step a cost is computed from the
current label set:

.. math::

   c_t \\triangleq c(L(s_t)),

and accumulated over the episode:

.. math::

   C_T \\triangleq \\sum_{t=0}^{T-1} c_t.

The episode is considered *satisfied* when:

.. math::

   C_T \\le B,

where :math:`B` is the user-specified budget.

The wrapper :class:`CumulativeCostEnv` updates the monitor each step by reading
``info["labels"]`` from the wrapped :class:`~masa.common.labelled_env.LabelledEnv`.

API Reference
-------------
"""

from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints.base import Constraint, BaseConstraintEnv, CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn

class CumulativeCost(Constraint):
    """CMDP-style cumulative cost constraint with a fixed budget.

    The monitor keeps:

    - ``step_cost``: the instantaneous cost :math:`c_t`,
    - ``total``: the accumulated cost :math:`C_T`.

    Args:
        cost_fn: Mapping from a label set to a scalar cost.
        budget: Episode budget :math:`B`. The episode is satisfied if
            ``total <= budget``.

    Attributes:
        cost_fn: The cost function ``labels -> float``.
        budget: Maximum allowed cumulative cost.
        total: Running cumulative cost for the current episode.
        step_cost: Cost at the most recent update.

    """

    def __init__(self, cost_fn: CostFn, budget: float):
        self.cost_fn = cost_fn
        self.budget = budget

    def reset(self):
        """Reset episode counters."""
        self.total = 0.0
        self.step_cost = 0.0

    def update(self, labels: Iterable[str]):
        """Update costs from the current label set.

        Args:
            labels: Iterable of atomic proposition strings for the current step.
        """
        self.step_cost = self.cost_fn(labels)
        self.total += self.step_cost

    def satisfied(self) -> bool:
        """Check whether the episode remains within budget.

        Returns:
            ``True`` iff ``total <= budget``.
        """
        return self.total <= self.budget

    def episode_metric(self) -> Dict[str, float]:
        """End-of-episode metrics.

        Returns:
            A dict containing:

            - ``"cum_cost"``: cumulative cost over the episode,
            - ``"satisfied"``: ``1.0`` if within budget else ``0.0``.
        """
        return {"cum_cost": self.total, "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        """Per-step metrics.

        Returns:
            A dict containing:

            - ``"cost"``: instantaneous cost,
            - ``"violation"``: 1.0 if the instantaneous cost is considered unsafe
              under the local convention ``cost >= 0.5``,
            - ``"cum_cost"``: running total.
        """
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5), "cum_cost": self.total}

    @property
    def constraint_type(self) -> str:
        """Stable identifier string: ``"cmdp"``."""
        return "cmdp"

class CumulativeCostEnv(BaseConstraintEnv):
    """Gymnasium wrapper that attaches :class:`CumulativeCost` to an environment.

    Args:
        env: Base environment (must be a :class:`~masa.common.labelled_env.LabelledEnv`).
        cost_fn: Cost function mapping label sets to float cost.
        budget: Cumulative cost budget :math:`B`.
        **kw: Extra keyword arguments forwarded to :class:`BaseConstraintEnv`.

    """

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, budget: float = 20.0, **kw):
        super().__init__(env, CumulativeCost(cost_fn=cost_fn, budget=budget), **kw)



    
