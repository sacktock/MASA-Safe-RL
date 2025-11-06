from __future__ import annotations
import gymnasium as gym
from masa.common.constraints import BaseConstraintEnv

class TimeLimit(gym.Wrapper):

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ConstraintMonitor(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env, BaseConstraintEnv):  # type: ignore[arg-type]
            raise TypeError(
                "ConstraintMonitor requires env to implement BaseConstraintEnv "
                "(wrap your env with CumulativeCostEnv/StepWiseProbabilisticEnv/...)."
            )
        self._constraint_env: BaseConstraintEnv = env  # type: ignore[assignment]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        info.setdefault("constraint", {})["type"] = self._constraint_env.constraint_type
        info["constraint"]["step"] = self._step_metrics()
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info.setdefault("constraint", {})["type"] = self._constraint_env.constraint_type
        info["constraint"]["step"] = self._step_metrics()
        if terminated or truncated:
            info["constraint"]["episode"] = self._episode_metrics()
        return observation, reward, terminated, truncated, info

    def _step_metrics(self) -> Dict[str, float]:
        try:
            return dict(self._constraint_env.constraint_step_metrics())
        except Exception:
            return {}

    def _episode_metrics(self) -> Dict[str, float]:
        try:
            return dict(self._constraint_env.constraint_episode_metrics())
        except Exception:
            return {}

class RewardMonitor(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_reward = 0.0
        self.total_steps = 0
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.total_steps += 1
        info = dict(info or {})
        info.setdefault("metrics", {})
        info["metrics"]["step"] = {"reward": reward}
        if terminated or truncated:
            info["metrics"]["episode"] = self._episode_metrics()
        return observation, reward, terminated, truncated, info
        
    def _episode_metrics(self):
        return {"ep_reward": self.total_reward, "ep_length": self.total_steps}

        

        
