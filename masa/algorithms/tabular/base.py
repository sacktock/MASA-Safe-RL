from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import Base_Algorithm
import gymnasium as gym
from gymnasium import spaces


class Tabular_Algorithm(Base_Algorithm):

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