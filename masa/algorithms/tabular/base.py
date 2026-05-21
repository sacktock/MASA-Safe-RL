from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import BaseAlgorithm
import gymnasium as gym
from gymnasium import spaces


class TabularAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            supported_action_spaces=(spaces.Discrete,),
            supported_observation_spaces=(spaces.Discrete,),
            **kwargs,
        )