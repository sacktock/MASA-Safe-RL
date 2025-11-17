from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import Base_Algorithm
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.q_learning import QL
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

class QL_Lambda(QL):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        alpha: float = 0.1,
        cost_lambda: float = 1.0,
        gamma: float = 0.9,
        exploration: str = 'boltzmann',
        boltzmann_temp: float = 0.05,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: str = 'linear',
        epsilon_decay_frames: int = 10000, 
    ):

        super().__init__(
            env, 
            tensorboard_logdir=tensorboard_logdir,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            env_fn=env_fn,
            eval_env=eval_env,
            alpha=alpha,
            gamma=gamma,
            exploration=exploration,
            boltzmann_temp=boltzmann_temp,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_decay_frames=epsilon_decay_frames,
        )

        self.cost_lambda = cost_lambda

    def optimize(self, step, logger: Optional[TrainLogger] = None):
        super().optimize(step, logger)
        if logger:
            logger.add("train/stats", {"cost_lambda": self.cost_lambda})

    def rollout(self, step, logger: Optional[TrainLogger] = None):

        self.key, subkey = jr.split(self.key)
        action = self.act(subkey, self._last_obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info["constraint"]["step"].get("cost", 0.0)
        penalty = -self.cost_lambda * cost
        self.buffer = (self._last_obs, action, reward + penalty, next_obs, terminated)

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
        else:
            self._last_obs = next_obs

        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)

        if logger:
            logger.add("train/rollout", info)
