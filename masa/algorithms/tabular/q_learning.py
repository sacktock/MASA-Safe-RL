
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import Base_Algorithm
from masa.common.metrics import TrainLogger
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

class QL(Base_Algorithm):

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
            supported_action_spaces=(spaces.Discrete,),
            supported_observation_spaces=(spaces.Discrete,),
            env_fn=env_fn,
            eval_env=eval_env,
        )

        assert exploration in ["boltzmann", "epsilon-greedy"], f"Unsupported exploration type {exploration}"
        assert epsilon_decay in ["linear"], f"Unsupported epsilon decay type {epsilon_decay}"

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.boltzmann_temp = boltzmann_temp
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_frames = epsilon_decay_frames

        self._setup_q_table()
        self._setup_buffer()
        self._setup_decay_schedule()
        
    def _setup_q_table(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def _setup_buffer(self):
        """Setup a simple buffer that stores one tuple to update"""
        self.buffer = tuple()

    def _setup_decay_schedule(self):
        self.epsilon = self.initial_epsilon
        # TODO

    def optimize(self, step, logger: Optional[TrainLogger] = None):
        """Update the Q tabkle with one tuple of experience"""
        if not self.buffer:
            return

        state, action, reward, next_state, terminal = self.buffer
        current = self.Q[next_state]
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
        + self.alpha * (reward + (1 - terminal) * self.gamma * np.max(current))

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon-greedy":
                logger.add("train/stats", {"epsilon": self.epsilon})

    def rollout(self, step, logger: Optional[TrainLogger] = None):

        self.key, subkey = jr.split(self.key)
        action = self.act(subkey, self._last_obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer = (self._last_obs, action, reward, next_obs, terminated)

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
        else:
            self._last_obs = next_obs

        if logger:
            logger.add("train/rollout", info)

    def act(self, key, obs, deterministic=False):
        if deterministic:
            action = self.select_action(self.Q[obs])
        else:
            action = self.sample_action(
                key, 
                jnp.asarray(self.Q[obs], dtype=jnp.float32), 
                self.boltzmann_temp, 
                self.epsilon, 
                exploration=self.exploration
            )
        return self.prepare_act(action)
            
    def prepare_act(self, act):
        return int(np.asarray(act).item())

    @staticmethod
    def select_action(q_values):
        return np.argmax(q_values)

    @staticmethod
    @partial(jit, static_argnames=["exploration"])
    def sample_action(key, q_values, tmp, eps, exploration="boltzmann"):
        if exploration == "boltzmann":
            scaled_q = q_values - jnp.max(q_values)
            exp = jnp.exp(scaled_q / tmp)
            probs = exp / (np.sum(exp) + 1e-6)
        if exploration == "epsilon_greedy":
            probs = jnp.zeros(q_values.shape[0])
            probs[jnp.argmax(q_values)] = 1.0 - eps
            probs += (epsilon / q_values.shape[0])
        return jr.choice(key, q_values.shape[0], p=probs)

    @property
    def train_ratio(self):
        return 1