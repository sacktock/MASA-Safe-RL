"""Independent Q-learning with episode-level Lagrangian dual updates."""
from __future__ import annotations

from collections.abc import Mapping
from operator import index

import numpy as np

from masa.algorithms.multi_agent.iql_lambda import IQLLambda


class IQLLagrangian(IQLLambda):
    r"""Adapt one multiplier per selected CMG budget.

    Every multiplier is fixed during an episode.  After a non-overlapping batch
    of complete episodes, the dual update is

    .. math::

       \lambda_b \leftarrow \Pi_{[0,\lambda_{\max}]}
       \left(\lambda_b + \eta(\overline C_b - B_b)\right),

    where :math:`\overline C_b` is the mean cumulative episode cost for budget
    :math:`b`, and :math:`B_b` is its configured cap.

    Args:
        cost_lambda: Initial multiplier for every budget, or a mapping selecting
            the budgets that receive independent dual variables.
        lambda_lr: Positive dual step size :math:`\eta`.
        dual_update_every: Number of complete training episodes per dual update.
            A partial batch is retained across calls to :meth:`train`.
        lambda_max: Optional nonnegative projection ceiling.
        **kwargs: Forwarded to
            :class:`~masa.algorithms.multi_agent.iql_lambda.IQLLambda`.

    Notes:
        CMG budgets are undiscounted episode sums.  This implementation requires
        ``gamma=1.0`` in ``ql_kwargs`` so the penalised task objective and the
        dual accounting use the same finite-episode convention.  It is a simple
        independent-learning baseline, not a convergence or per-trajectory
        safety guarantee.
    """

    def __init__(
        self,
        *args,
        cost_lambda: float | Mapping[str, float] = 0.0,
        lambda_lr: float = 0.01,
        dual_update_every: int = 1,
        lambda_max: float | None = None,
        **kwargs,
    ):
        ql_kwargs = dict(kwargs.pop("ql_kwargs", {}) or {})
        ql_kwargs.setdefault("gamma", 1.0)
        super().__init__(
            *args,
            cost_lambda=cost_lambda,
            ql_kwargs=ql_kwargs,
            **kwargs,
        )

        if any(learner.gamma != 1.0 for learner in self.learners.values()):
            raise ValueError(
                "IQLLagrangian requires ql_kwargs={'gamma': 1.0} for "
                "undiscounted finite-episode CMG budgets."
            )

        self.lambda_lr = float(lambda_lr)
        self.dual_update_every = index(dual_update_every)
        self.lambda_max = None if lambda_max is None else float(lambda_max)

        if not np.isfinite(self.lambda_lr) or self.lambda_lr <= 0.0:
            raise ValueError("lambda_lr must be finite and positive.")
        if self.dual_update_every < 1:
            raise ValueError("dual_update_every must be a positive integer.")
        if self.lambda_max is not None and (
            not np.isfinite(self.lambda_max) or self.lambda_max < 0.0
        ):
            raise ValueError("lambda_max must be finite and nonnegative, or None.")
        if self.lambda_max is not None and any(
            value > self.lambda_max for value in self.lambdas.values()
        ):
            raise ValueError("Initial multipliers must not exceed lambda_max.")
        if any(
            not np.isfinite(self.budgets[name].amount)
            or self.budgets[name].amount < 0.0
            for name in self.lambdas
        ):
            raise ValueError("Selected CMG budget caps must be finite and nonnegative.")

        self._dual_cost_sums = {name: 0.0 for name in self.lambdas}
        self._dual_episode_count = 0

    def _after_episode(self, metrics: Mapping[str, float]) -> None:
        for name in self.lambdas:
            metric_name = f"{name}_cum_cost"
            if metric_name not in metrics:
                raise KeyError(f"CMG episode metrics omitted {metric_name!r}.")
            self._dual_cost_sums[name] += float(metrics[metric_name])

        self._dual_episode_count += 1
        if self._dual_episode_count < self.dual_update_every:
            return

        for name, current in tuple(self.lambdas.items()):
            mean_cost = self._dual_cost_sums[name] / self._dual_episode_count
            updated = max(
                0.0,
                current
                + self.lambda_lr * (mean_cost - float(self.budgets[name].amount)),
            )
            if self.lambda_max is not None:
                updated = min(updated, self.lambda_max)
            self.lambdas[name] = float(updated)
            self._dual_cost_sums[name] = 0.0

        self._dual_episode_count = 0


__all__ = ["IQLLagrangian"]
