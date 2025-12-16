from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.q_learning import QL
from masa.common.ltl import DFACostFn, DFA
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

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, cost, violation, next_state, terminal) in self.buffer:

            penalty = -self.cost_lambda * cost

            current = self.Q[next_state]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * ((reward + penalty) + (1 - violation) * (1 - terminal) * self.gamma * np.max(current))

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"cost_lambda": self.cost_lambda})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})

