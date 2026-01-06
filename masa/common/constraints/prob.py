"""
Undiscounted probabilistic safety constraint.

This monitor tracks the empirical fraction of unsafe steps in an episode and
requires it to be at most ``alpha``.

Let:

- :math:`u_t \\in \\{0,1\\}` indicate whether step :math:`t` is unsafe,
  computed from labels by thresholding a cost function:

  .. math::

     u_t = \\mathbf{1}[\\mathrm{cost}(L(s_t)) \\ge 0.5].

- Then the empirical unsafe fraction after :math:`T` steps is:

  .. math::

     \\hat{p}_{\\text{unsafe}} = \\frac{1}{T} \\sum_{t=0}^{T-1} u_t.

The episode is satisfied iff:

.. math::

   \\hat{p}_{\\text{unsafe}} \\le \\alpha.

"""

from __future__ import annotations
from typing import Any, Dict
from masa.common.constraints.base import Constraint, BaseConstraintEnv, CostFn
from masa.common.dummy import cost_fn as dummy_cost_fn

class ProbabilisticSafety(Constraint):
    """Undiscounted probabilistic constraint based on unsafe-step frequency.

    Args:
        cost_fn: Mapping from a label set to a scalar cost.
        alpha: Allowed maximum fraction of unsafe steps in an episode.

    Attributes:
        total: Number of steps observed so far.
        total_unsafe: Number of steps considered unsafe so far.
        step_cost: Most recent cost.

    """

    def __init__(self, cost_fn: CostFn, alpha: float):
        self.cost_fn = cost_fn
        self.alpha = alpha

    def reset(self):
        """Reset episode counters."""
        self.total = 0
        self.total_unsafe = 0.0
        self.step_cost = 0.0

    def update(self, labels: Iterable[str]):
        """Update counters from the current label set.

        Args:
            labels: Iterable of atomic propositions for the current step.
        """
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe += float(self.step_cost >= 0.5)
        self.total += 1

    def prob_unsafe(self) -> float:
        """Return the empirical fraction of unsafe steps.

        Returns:
            ``total_unsafe / total``.
        """
        if not self.total:
            return 0.0
        else:
            return self.total_unsafe / self.total

    def satisfied(self) -> bool:
        """Check whether the unsafe fraction is within the threshold."""
        return self.prob_unsafe() <= self.alpha

    def episode_metric(self) -> Dict[str, float]:
        """End-of-episode metrics.

        Returns:
            Dict containing:

            - ``"cum_unsafe"``: count of unsafe steps,
            - ``"p_unsafe"``: proportion of unsafe states in the current trace,
            - ``"satisfied"``: 1.0 if p_unsafe <= self.alpha else 0.0.
        """
        return {"cum_unsafe": float(self.total_unsafe), "p_unsafe": self.prob_unsafe(), "satisfied": float(self.satisfied())}

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
        """Stable identifier string: ``"prob"``."""
        return "prob"


class ProbabilisticSafetyEnv(BaseConstraintEnv):
    """Gymnasium wrapper for :class:`ProbabilisticSafety`.

    Args:
        env: Base environment (must be a :class:`~masa.common.labelled_env.LabelledEnv`).
        cost_fn: Cost function mapping labels to a scalar.
        alpha: Allowed maximum unsafe-step fraction.
        **kw: Extra keyword arguments forwarded to :class:`BaseConstraintEnv`.

    """

    def __init__(self, env: gym.Env, cost_fn: CostFn = dummy_cost_fn, alpha: float = 0.01, **kw):
        super().__init__(env, ProbabilisticSafety(cost_fn=cost_fn, alpha=alpha), **kw)

