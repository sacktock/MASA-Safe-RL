from __future__ import annotations
from typing import Optional
import importlib
import warnings
from masa.plugins.helpers import load_plugins
from masa.common.registry import ENV_REGISTRY, CONSTRAINT_REGISTRY
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor
from masa.common.labelled_env import LabelledEnv
from masa.common.label_fn import LabelFn

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
) -> gym.Env:
    """
    Construct a fully wrapped MASA environment using the canonical wrapper order.

    This helper creates a Gymnasium environment and applies MASA wrappers in the
    **recommended and enforced order**:

    ``TimeLimit`` :math:`\rightarrow` ``LabelledEnv`` :math:`\rightarrow` ``BaseconstraintEnv`` :math:`\rightarrow` ``ConstraintMonitor`` :math:`\rightarrow` ``RewardMonitor``

    The resulting environment exposes labels, constraint metrics, and reward
    summaries exclusively via the Gymnasium ``info`` dictionary. Observations
    and rewards themselves are left unchanged.

    Args:
        env_id:
            Environment identifier registered in ``ENV_REGISTRY``.
        constraint:
            Constraint identifier registered in ``CONSTRAINT_REGISTRY``.
        max_episode_steps:
            Maximum number of steps per episode. Applied via ``TimeLimit`` as
            the outermost wrapper.
        label_fn:
            Optional function mapping observations to atomic predicate labels.
            If provided, labels are computed on every ``reset`` and ``step`` and
            stored under ``info["labels"]``.
        **constraint_kwargs:
            Additional keyword arguments forwarded to the constraint wrapper
            constructor.

    Returns:
        A fully wrapped Gymnasium environment compatible with MASA algorithms,
        monitors, and logging utilities.

    Notes:
        - Wrapper order is fixed and enforced.
        - Constraints are reset automatically on environment reset.
        - All semantic metadata (labels, costs, violations, metrics) is communicated
          via the ``info`` dictionary.

    See Also:
        masa.common.labelled_env.LabelledEnv
        masa.common.constraints.base.BaseConstraintEnv
        masa.common.wrappers.ConstraintMonitor
        masa.common.wrappers.RewardMonitor
    """

    load_plugins()
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

