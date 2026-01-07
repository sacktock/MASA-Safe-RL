from __future__ import annotations
from typing import Optional
import importlib
import warnings
from masa.plugins.helpers import load_plugins
from masa.common.registry import ENV_REGISTRY, CONSTRAINT_REGISTRY
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor
from masa.common.labelled_env import LabelledEnv

def load_callable(path: str):
    """Load a callable from 'module_path:object_name' string."""
    if ":" not in path:
        raise ValueError("Expected 'module:callable' for --label-fn/--cost-fn")
    mod, name = path.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), name)
    except AttributeError:
        warnings.warn(f"Could not load object from path: {path}")
        return None

def make_env(
    env_id: str, 
    constraint: str, 
    max_episode_steps: int, 
    *,
    label_fn: Optional[LabelFn] = None, 
    **constraint_kwargs
):
    """
    Construct a fully wrapped MASA environment using the canonical wrapper order.

    This helper creates a Gymnasium environment and applies MASA wrappers in the
    **recommended and enforced order**:

    .. math::

        \\texttt{TimeLimit}
        \\;\\rightarrow\\;
        \\texttt{LabelledEnv}
        \\;\\rightarrow\\;
        \\texttt{BaseConstraintEnv}
        \\;\\rightarrow\\;
        \\texttt{ConstraintMonitor}
        \\;\\rightarrow\\;
        \\texttt{RewardMonitor}

    The resulting environment exposes atomic predicate labels, constraint metrics,
    and reward summaries via the standard Gymnasium ``info`` dictionary.

    Args:
        env_id (str | gymnasium.Env):
            Either a Gymnasium environment ID (passed to :func:`gymnasium.make`)
            or an already-constructed environment instance.
        label_fn (:class:`masa.common.label_fn.LabelFn`):
            Function mapping observations to an iterable of atomic predicate names.
            Labels are computed on every call to :meth:`reset` and :meth:`step` and
            stored under ``info["labels"]``.
        cost_fn (:class:`masa.common.constraints.base.CostFn`, optional):
            Cost function mapping a set of atomic predicates to a scalar cost.
            Required if ``constraint_cls`` expects a cost function.
        constraint_cls (type, optional):
            Constraint wrapper class to apply (e.g.,
            :class:`masa.common.constraints.cumulative.CumulativeConstraintEnv`).
            If ``None``, no constraint wrapper is applied.
        max_episode_steps (int, optional):
            Maximum episode length. If provided, the environment is wrapped in
            :class:`gymnasium.wrappers.TimeLimit` *before* any MASA-specific wrappers.
        **constraint_kwargs:
            Additional keyword arguments forwarded to ``constraint_cls``.

    Returns:
        gymnasium.Env:
            A fully wrapped Gymnasium environment compatible with MASA algorithms,
            monitors, and logging utilities.

    Notes:
        - Wrapper order matters and is enforced by this function.
        - Constraints are reset automatically on environment reset.
        - All semantic information (labels, costs, violations, metrics) is communicated
          exclusively via the ``info`` dictionary; observations and rewards are left
          unchanged.

    See Also:
        - :class:`masa.common.labelled_env.LabelledEnv`
        - :class:`masa.common.constraints.base.BaseConstraintEnv`
        - :class:`masa.common.wrappers.ConstraintMonitor`
        - :class:`masa.common.wrappers.RewardMonitor`
    """
    
    env_ctor = ENV_REGISTRY.get(env_id)
    constraint_ctor = CONSTRAINT_REGISTRY.get(constraint)
    env = env_ctor()
    # must wrap time limit first
    env = TimeLimit(env, max_episode_steps)
    if label_fn is not None:
        env = LabelledEnv(env, label_fn)
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    return env

load_plugins()