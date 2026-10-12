"""Independent multi-agent learning coordinators."""

from masa.algorithms.tabular.multi_agent.iql import IQL
from masa.algorithms.tabular.multi_agent.iql_lambda import IQLLambda
from masa.algorithms.tabular.multi_agent.iql_lagrangian import IQLLagrangian

__all__ = ["IQL", "IQLLambda", "IQLLagrangian"]
