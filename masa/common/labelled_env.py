"""
Gymnasium wrapper for labelled environments.

This module provides :class:`LabelledEnv`, a lightweight Gymnasium wrapper
that augments the ``info`` dictionary returned by :meth:`reset` and
:meth:`step` with a set of atomic proposition labels derived from the
current observation.

This wrapper is the canonical bridge between raw Gymnasium environments
and MASA components that reason over labels, such as constraints, DFAs,
and cost functions.
"""

from __future__ import annotations
from typing import Any, Dict
import gymnasium as gym
from masa.common.label_fn import LabelFn

class LabelledEnv(gym.Wrapper):
    """
    Gymnasium wrapper that attaches a labelling function to an environment.

    At every call to :meth:`reset` and :meth:`step`, the wrapped environment's
    observation is passed through a user-provided :class:`LabelFn`. The
    resulting set of atomic propositions is stored under the ``"labels"``
    key in the ``info`` dictionary.

    This induces a **labelled MDP** without modifying the observation or
    reward spaces, enabling downstream components to reason symbolically
    over environment behaviour.

    Attributes:
        label_fn (:class:`LabelFn`):
            Function mapping observations to an iterable of atomic
            proposition names.
    """

    def __init__(self, env: gym.Env, label_fn: LabelFn):
        """
        Initialize the labelled environment wrapper.

        Args:
            env (gym.Env):
                The base Gymnasium environment to wrap.
            label_fn (:class:`LabelFn`):
                Labelling function mapping observations to atomic
                propositions.

        Notes:
            The wrapper does **not** alter the observation, reward,
            termination, or truncation signals. All labelling information
            is communicated exclusively via the ``info`` dictionary.
        """
        super().__init__(env)
        self.label_fn = label_fn

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        Reset the environment and compute initial labels.

        The observation returned by the underlying environment is passed
        through :attr:`label_fn`, and the resulting set of labels is stored
        under ``info["labels"]``.

        Args:
            seed (int | None, optional):
                Random seed passed to the underlying environment reset.
            options (Dict[str, Any] | None, optional):
                Optional reset options forwarded to the environment.

        Returns:
            Tuple[Any, Dict[str, Any]]:
                A tuple ``(obs, info)`` where:
                  - ``obs`` is the initial observation.
                  - ``info`` contains all original entries from the wrapped
                    environment, plus a ``"labels"`` entry of type
                    ``set[str]``.

        Notes:
            The ``labels`` entry is always a **set**, even if the labelling
            function returns a different iterable type.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        info["labels"] = set(self.label_fn(obs))
        return obs, info

    def step(self, action):
        """
        Step the environment and compute labels for the next observation.

        After stepping the wrapped environment, the resulting observation
        is labelled using :attr:`label_fn`. The labels are attached to the
        ``info`` dictionary under the ``"labels"`` key.

        Args:
            action (Any):
                Action to apply to the environment.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
                A tuple ``(obs, reward, terminated, truncated, info)`` where:
                  - ``obs`` is the next observation,
                  - ``reward`` is the scalar reward,
                  - ``terminated`` indicates episode termination,
                  - ``truncated`` indicates episode truncation,
                  - ``info`` includes a ``"labels"`` entry of type
                    ``set[str]``.

        Notes:
            The labelling function is applied **after** the environment
            transition, meaning labels correspond to the *post-transition*
            state.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info["labels"] = set(self.label_fn(obs))
        return obs, reward, terminated, truncated, info