from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, List, Callable
from gymnasium import spaces
import gymnasium as gym
import jax.random as jr
from masa.common.metrics import RolloutLogger, StatsLogger, TrainLogger
from abc import ABC, abstractmethod
import tensorflow as tf
import math
from tqdm import tqdm

def _allowed_names(allowed: tuple[type[spaces.Space], ...]) -> str:
    return ", ".join(t.__name__ for t in allowed)

class Base_Algorithm(ABC):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        supported_observation_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
    ):
        self.env = env
        self.tensorboard_logdir = tensorboard_logdir
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.supported_action_spaces = supported_action_spaces
        self.supported_observation_spaces = supported_observation_spaces

        act_sp = getattr(env, "action_space", None)
        obs_sp = getattr(env, "observation_space", None)

        if act_sp is None or obs_sp is None:
            raise ValueError(
                f"{self.__class__.__name__}: env is missing action_space/observation_space."
            )

        if self.supported_action_spaces is not None and not isinstance(act_sp, self.supported_action_spaces):
            raise TypeError(
                f"{self.__class__.__name__} does not support action space { type(act_sp).__name__ }.\n"
                f"Allowed types: { _allowed_names(self.supported_action_spaces) }."
            )

        if self.supported_observation_spaces is not None and not isinstance(obs_sp, self.supported_observation_spaces):
            raise TypeError(
                f"{self.__class__.__name__} does not support observation space { type(obs_sp).__name__ }.\n"
                f"Allowed types: { _allowed_names(self.supported_observation_spaces) }."
            )

        self._env_fn = env_fn
        self._eval_env = eval_env

        self.key = jr.PRNGKey(0 if seed is None else seed)

    def train(self, 
        num_frames: int,
        num_eval_episodes: Optional[int] = None,
        eval_freq: int = 0,
        log_freq: int = 1,
        prefill: Optional[int] = None,
        save_freq: int = 0,
        stats_window_size: int = 100,
    ):

        if self.tensorboard_logdir is not None:
            summary_writer = tf.summary.create_file_writer(self.tensorboard_logdir)
        else:
            summary_writer = None

        logger = TrainLogger(
            {"train/rollout": RolloutLogger, "eval/rollout": RolloutLogger, "train/stats": StatsLogger},
            tensorboard=bool(summary_writer is not None),
            summary_writer=summary_writer, 
            stats_window_size=stats_window_size,
            prefix='',
        )

        total_steps = 0

        next_eval = eval_freq
        next_save = save_freq
        next_log = log_freq

        self._last_obs, _ = self.env.reset(seed=self.seed)

        for step in tqdm(range(math.ceil((num_frames)/self.train_ratio))):
            self.rollout(step, logger=logger)
            self.optimize(step, logger=logger)

            total_steps += self.train_ratio

            if eval_freq and (total_steps >= next_eval):
                next_eval += eval_freq
                self.eval(num_eval_episodes, seed=step, logger=logger)

            if save_freq and (total_steps >= next_save):
                next_save += save_freq
                self.save(step)

            if log_freq and (total_steps >= next_log):
                next_log += log_freq
                logger.log(step*self.train_ratio)

    def _get_eval_env(self) -> gym.Env:
        if self._eval_env is not None:
            return self._eval_env
        if self._env_fn is not None:
            self._eval_env = self._env_fn()
            return self._eval_env
        raise RuntimeError(
                "Cannot construct eval env; please pass env_fn or eval_env."
            )

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Optimizes the policy and other auxilliary models"""
        raise NotImplementedError

    def rollout(self, step: int, logger: Optional[TrainLogger] = None):
        """Collects rollouts/experience for the policy to learn from"""
        raise NotImplementedError

    def eval(self, num_episodes: int, seed: Optional[int] = None, logger: Optional[TrainLogger] = None) -> List[float]:
        eval_env = self._get_eval_env()

        base = 0 if self.seed is None else int(self.seed)
        eval_seed = base + 10_000 if seed is None else int(seed) + 10_000

        eval_key = jr.PRNGKey(eval_seed)

        returns = []

        for ep in range(num_episodes):
            obs, info = eval_env.reset(seed=eval_seed + ep)
            done = False
            ret = 0.0
            while not done:
                eval_key, subkey = jr.split(eval_key)
                action = self.act(subkey, obs, deterministic=False)
                obs, rew, terminated, truncated, info = eval_env.step(action)
                ret += float(rew)
                done = terminated or truncated

            returns.append(ret)

            if logger is not None:
                logger.add("eval/rollout", info)

        return returns

    def act(self, key, obs, deterministic=False):
        """Implements the agent's policy: action selection (deterministic) or sampling"""
        raise NotImplementedError

    def prepare_act(self, act):
        """Prepares the action to be used in the environment"""
        raise NotImplementedError

    def save(self, step):
        """Save the relevant algorithm/model parameters"""
        raise NotImplementedError
    
    def load(self):
        """Load the relevant algorithm/model parameters"""
        raise NotImplementedError

    @property
    def train_ratio(self):
        """How often to do an update step during training"""
        raise NotImplementedError

