"""Independent Q-learning with fixed CMG budget penalties."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from masa.algorithms.tabular.multi_agent.iql import IQL


class IQLLambda(IQL):
    r"""Independent Q-learning with fixed nonnegative multipliers.

    For a selected CMG budget :math:`b` covering agents :math:`G_b`, MASA's CMG
    monitor reports the aggregate step cost

    .. math::

       c_{b,t} = \sum_{j \in G_b} c_{j,t}.

    A member agent learns from

    .. math::

       \widetilde r_{i,t} = r_{i,t}
       - \sum_{b : i \in G_b} \lambda_b c_{b,t}.

    Args:
        cost_lambda: A single multiplier applied to every CMG budget, or a
            mapping from selected budget names to multipliers.  A mapping is
            recommended when budgets overlap, because only listed budgets enter
            the learning signal.  Unselected budgets remain monitored.
        **kwargs: Forwarded to :class:`~masa.algorithms.tabular.multi_agent.iql.IQL`.
    """

    def __init__(
        self,
        *args,
        cost_lambda: float | Mapping[str, float] = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(cost_lambda, Mapping):
            weights = dict(cost_lambda)
        else:
            weights = {name: cost_lambda for name in self.budgets}

        unknown = set(weights) - set(self.budgets)
        if unknown:
            raise ValueError(f"Unknown CMG budgets: {sorted(unknown)}.")
        if not weights:
            raise ValueError("Select at least one budget or use IQL instead.")

        self.lambdas = {name: float(value) for name, value in weights.items()}
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self.lambdas.values()
        ):
            raise ValueError("All multipliers must be finite and nonnegative.")

    def _penalty(self, agent: str, cmg_step_metrics: Mapping[str, float]) -> float:
        penalty = 0.0
        for name, multiplier in self.lambdas.items():
            if agent not in self.budgets[name].agents:
                continue
            # CMG cost/budget accounting
            metric_name = f"{name}_cost"
            if metric_name not in cmg_step_metrics:
                raise KeyError(f"CMG step metrics omitted {metric_name!r}.")
            penalty += multiplier * float(cmg_step_metrics[metric_name])
        return float(penalty)


__all__ = ["IQLLambda"]
