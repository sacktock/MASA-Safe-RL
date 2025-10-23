from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Protocol

CostFn = Callable[Iterable[str], float]

class ConstraintEnv(gym.Wrapper):
    """Gymnasium wrapper that attaches the constraint and optional cost function"""

    def __init__(self, env: gym.Env, cost_fn: CostFn | None = None):
        super().__init__(env)
        self.cost_fn = cost_fn
        assert self.label_fn is not None, "It looks like there is no labelling function, please attach a labelling function before wrapping the environment with a constraint"
        self._reset()

    def _reset(self):
        pass

    def _update(self, info):
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset()
        self._update(info)
        assert isinstance(info, dict)
        info.update(self.episode_metric())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update(info)
        assert isinstance(info, dict)
        info.update(self.episode_metric())
        return obs, info

    def satisfied(self) -> bool:
        raise NotImplementedError

    def episode_metric(self) -> Dict[str, float]:
        raise NotImplementedError