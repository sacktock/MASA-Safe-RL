from __future__ import annotations
from typing import Any, Dict
import gymnasium as gym
from .base import LabelFn


class LabelledEnv(gym.Wrapper):
    """Gymnasium wrapper that attaches the labelling function"""

    def __init__(self, env: gym.Env, label_fn: LabelFn):
        super().__init__(env)
        self.label_fn = label_fn

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        assert isinstance(info, dict)
        info["labels"] = set(self.label_fn(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        labels = set(self.label_fn(obs))
        assert isinstance(info, dict)
        info.update({"labels": labels})
        return obs, reward, terminated, truncated, info