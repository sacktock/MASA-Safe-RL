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

class SEM(QL):

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
        ql_alpha: float = 0.1,
        ql_gamma: float = 0.9,
        dm_alpha: float = 0.1,
        dm_gamma: float = 0.99,
        cm_alpha: float = 0.05,
        cm_gamma: float = 0.999,
        r_min: float = 0.0,
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
            alpha=ql_alpha,
            gamma=ql_gamma,
            exploration=exploration,
            boltzmann_temp=boltzmann_temp,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_decay_frames=epsilon_decay_frames,
        )

        self.dm_alpha = dm_alpha
        self.dm_gamma = dm_gamma
        self.cm_alpha = cm_alpha
        self.cm_gamma = cm_gamma
        self.r_min = r_min
        
    def _setup_q_table(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.D = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.C = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q tables with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, action, reward, _, violation, next_state, terminal) in self.buffer:

            penalty = 1.0 if violation else 0.0

            current = self.Q[next_state]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * ((reward - self.r_min) + (1 - terminal) * self.gamma * np.max(current))

            current = self.D[next_state]
            self.D[state, action] = (1 - self.dm_alpha) * self.D[state, action] \
            + self.dm_alpha * (penalty + (1 - terminal) * (1 - violation) * self.dm_gamma * np.max(current))

            current = self.C[next_state]
            self.C[state, action] = (1 - self.cm_alpha) * self.C[state, action] \
            + self.cm_alpha * (-penalty + (1 - terminal) * (1 - violation) * self.cm_gamma * np.max(current))

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"dm_alpha": self.dm_alpha})
            logger.add("train/stats", {"cm_alpha": self.cm_alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})

    def act(self, key, obs, deterministic=False):
        if deterministic:
            action = self.select_action(
                jnp.asarray(self.Q[obs], dtype=jnp.float32), 
                jnp.asarray(self.D[obs], dtype=jnp.float32),
                jnp.asarray(self.C[obs], dtype=jnp.float32),
            )
        else:
            action = self.sample_action(
                key, 
                jnp.asarray(self.Q[obs], dtype=jnp.float32), 
                jnp.asarray(self.D[obs], dtype=jnp.float32),
                jnp.asarray(self.C[obs], dtype=jnp.float32),
                self.boltzmann_temp, 
                self._epsilon, 
                exploration=self.exploration
            )
        return self.prepare_act(action)

    @staticmethod
    @jit
    def select_action(q_values, d_values, c_values):
        c_values = jnp.clip(c_values, -1.0, 0.0)
        d_values = jnp.clip(-d_values, -1.0, 0.0)
        X = jnp.exp(jnp.minimum(c_values, d_values))
        q_X = q_values * X
        return np.argmax(q_X)

    @staticmethod
    @partial(jit, static_argnames=["exploration"])
    def sample_action(key, q_values, d_values, c_values, tmp, eps, exploration="boltzmann"):
        c_values = jnp.clip(c_values, -1.0, 0.0)
        d_values = jnp.clip(-d_values, -1.0, 0.0)
        X = jnp.exp(jnp.minimum(c_values, d_values))
        q_X = q_values * X
        #q_X = q_X / (jnp.sum(q_X) + 1e-6)
        #probs = q_X / (jnp.sum(q_X) + 1e-6)
        #return jr.choice(key, q_values.shape[0], p=probs)
        if exploration == "boltzmann":
            probs = q_X / (jnp.sum(q_X) + 1e-6)
            log_probs = jnp.log(probs)
            scaled_log_probs = log_probs - log_probs
            exp = jnp.exp(scaled_log_probs / tmp)
            probs = exp / (jnp.sum(exp) + 1e-6)
        if exploration == "epsilon_greedy":
            probs = jnp.zeros(q_X.shape[0])
            probs = probs.at[jnp.argmax(q_X)].set(1.0 - eps)
            probs += eps * (q_X / (jnp.sum(q_X) + 1e-6))
        return jr.choice(key, q_values.shape[0], p=probs)