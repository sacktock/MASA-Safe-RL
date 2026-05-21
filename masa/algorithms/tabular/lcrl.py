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

class LCRL(QL):

    def __init__(
        self,
        *args,
        r_min: float = 0.0,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.r_min = r_min

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, _, violation, next_state, terminal) in self.buffer:

            current = self.Q[next_state]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * (reward * (1 - violation) + float(violation) * (self.r_min / (1.0 - self.gamma)) \
            + (1 - violation) * (1 - terminal) * self.gamma * np.max(current))

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})