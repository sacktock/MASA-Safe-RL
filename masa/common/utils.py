from __future__ import annotations
from typing import Any, Callable, Optional
import importlib
import warnings
import gymnasium as gym
from gymnasium.wrappers import RecordVideo as GymnasiumRecordVideo
from pettingzoo import ParallelEnv
from masa.plugins.helpers import load_plugins
from masa.common.registry import (
    CONSTRAINT_REGISTRY,
    ENV_REGISTRY,
    MARL_CONSTRAINT_REGISTRY,
    MARL_ENV_REGISTRY,
)
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor
from masa.common.labelled_env import LabelledEnv
from masa.common.labelled_pz_env import LabelledParallelEnv
from masa.common.label_fn import LabelFn
from masa.common.pettingzoo_record_video import RecordVideoParallel

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


def _resolve_video_kwargs(
    video_kwargs: Optional[dict[str, Any]],
    record_video_episode_trigger: Optional[Callable[[int], bool]],
) -> dict[str, Any]:
    resolved = dict(video_kwargs or {})
    if record_video_episode_trigger is not None:
        if "episode_trigger" in resolved:
            raise ValueError(
                "Pass either record_video_episode_trigger or "
                "video_kwargs['episode_trigger'], not both."
            )
        resolved["episode_trigger"] = record_video_episode_trigger
    return resolved


def make_env(
    env_id: str, 
    constraint: str, 
    max_episode_steps: int, 
    *,
    label_fn: Optional[LabelFn] = None, 
    env_kwargs: Optional[dict[str, Any]] = None,
    record_video: bool = False,
    record_video_episode_trigger: Optional[Callable[[int], bool]] = None,
    video_folder: str = "videos",
    video_kwargs: Optional[dict[str, Any]] = None,
    **constraint_kwargs
) -> gym.Env:
    r"""
    Construct a fully wrapped MASA environment using the canonical wrapper order.

    This helper creates a Gymnasium environment and applies MASA wrappers in the
    **recommended and enforced order**:

    :class:`~gymnasium.wrappers.TimeLimit` :math:`\rightarrow` 
    :class:`~masa.common.labelled_env.LabelledEnv` :math:`\rightarrow` 
    :class:`~masa.common.constraints.base.BaseConstraintEnv`  :math:`\rightarrow` 
    :class:`~masa.common.wrappers.ConstraintMonitor` :math:`\rightarrow` 
    :class:`~masa.common.wrappers.RewardMonitor`

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
        env_kwargs:
            Optional keyword arguments forwarded to the base environment
            constructor.
        record_video:
            Whether to wrap the resulting environment with Gymnasium's
            :class:`~gymnasium.wrappers.RecordVideo`. Defaults to ``False``.
        record_video_episode_trigger:
            Optional predicate called with the episode id to decide whether to
            record that episode. This is forwarded as ``episode_trigger`` to
            :class:`~gymnasium.wrappers.RecordVideo`.
        video_folder:
            Output directory for recorded videos when ``record_video=True``.
        video_kwargs:
            Optional keyword arguments forwarded to
            :class:`~gymnasium.wrappers.RecordVideo`.
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
        - :class:`masa.common.labelled_env.LabelledEnv` 
        - :class:`masa.common.constraints.base.BaseConstraintEnv` 
        - :class:`masa.common.wrappers.ConstraintMonitor` 
        - :class:`masa.common.wrappers.RewardMonitor`
    """

    load_plugins()
    env_ctor = ENV_REGISTRY.get(env_id)
    constraint_ctor = CONSTRAINT_REGISTRY.get(constraint)
    env = env_ctor(**dict(env_kwargs or {}))
    # must wrap time limit first
    env = TimeLimit(env, max_episode_steps)
    if label_fn is not None:
        env = LabelledEnv(env, label_fn)
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    if record_video:
        env = GymnasiumRecordVideo(
            env,
            video_folder=video_folder,
            **_resolve_video_kwargs(video_kwargs, record_video_episode_trigger),
        )
    return env


def make_marl_env(
    env_id: str,
    constraint: str,
    *,
    label_fn: Optional[dict[str, LabelFn] | LabelFn] = None,
    env_kwargs: Optional[dict[str, Any]] = None,
    record_video: bool = False,
    record_video_episode_trigger: Optional[Callable[[int], bool]] = None,
    video_folder: str = "videos",
    video_kwargs: Optional[dict[str, Any]] = None,
    **constraint_kwargs,
) -> ParallelEnv:
    r"""
    Construct a fully wrapped MASA multi-agent environment.

    This helper creates a PettingZoo parallel environment and applies the
    standard MARL wrapper order:

    :class:`~masa.common.labelled_pz_env.LabelledParallelEnv` :math:`\rightarrow`
    :class:`~masa.common.constraints.multi_agent.cmg.ConstrainedMarkovGameEnv`

    Args:
        env_id:
            Multi-agent environment identifier registered in ``MARL_ENV_REGISTRY``.
        constraint:
            Multi-agent constraint identifier registered in
            ``MARL_CONSTRAINT_REGISTRY``.
        label_fn:
            Optional labelling function, or per-agent mapping of labelling
            functions. If omitted, the base environment's ``label_fn`` attribute
            is used.
        env_kwargs:
            Optional keyword arguments forwarded to the base environment
            constructor.
        record_video:
            Whether to wrap the resulting PettingZoo parallel environment with
            :class:`~masa.common.pettingzoo_record_video.RecordVideoParallel`.
            Defaults to ``False``.
        record_video_episode_trigger:
            Optional predicate called with the episode id to decide whether to
            record that episode. This is forwarded as ``episode_trigger`` to
            :class:`~masa.common.pettingzoo_record_video.RecordVideoParallel`.
        video_folder:
            Output directory for recorded videos when ``record_video=True``.
        video_kwargs:
            Optional keyword arguments forwarded to
            :class:`~masa.common.pettingzoo_record_video.RecordVideoParallel`.
        **constraint_kwargs:
            Additional keyword arguments forwarded to the constraint wrapper
            constructor. If ``cost_fn`` is omitted and the base environment
            exposes one, it is forwarded automatically.

    Returns:
        A wrapped PettingZoo parallel environment compatible with MASA MARL
        constraints.
    """

    load_plugins()
    env_ctor = MARL_ENV_REGISTRY.get(env_id)
    constraint_ctor = MARL_CONSTRAINT_REGISTRY.get(constraint)

    raw_env = env_ctor(**dict(env_kwargs or {}))
    resolved_label_fn = label_fn if label_fn is not None else getattr(raw_env, "label_fn", None)
    if resolved_label_fn is None:
        raise ValueError(
            f"MARL env '{env_id}' does not expose a default label_fn. "
            "Pass label_fn=... to make_marl_env."
        )

    if "cost_fn" not in constraint_kwargs:
        cost_fn = getattr(raw_env, "cost_fn", None)
        if cost_fn is not None:
            constraint_kwargs["cost_fn"] = cost_fn

    env = LabelledParallelEnv(raw_env, resolved_label_fn)
    env = constraint_ctor(env, **constraint_kwargs)
    if record_video:
        env = RecordVideoParallel(
            env,
            video_folder=video_folder,
            **_resolve_video_kwargs(video_kwargs, record_video_episode_trigger),
        )
    return env
