from __future__ import annotations
from typing import Optional
import importlib
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