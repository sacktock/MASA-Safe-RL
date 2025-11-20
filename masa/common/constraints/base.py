from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Protocol, Callable
from masa.common.labelled_env import LabelledEnv
from typing import Dict, Protocol, Any
import gymnasium as gym

CostFn = Callable[Iterable[str], float]

class Constraint(Protocol):
    """
    Protocol for any Gymnasium wrapper that carries a constraint and can
    expose per-step and per-episode metrics in a uniform way.
    """

    @property
    def constraint_type(self) -> str:
        """A stable identifier"""
        raise NotImplementedError

    
    def constraint_step_metrics(self) -> Dict[str, float]:
        """
        Metrics that make sense at *any* step (cheap, non-destructive).
        e.g., running cum_cost, p_unsafe_estimate, reached/violated flags, dfa_accepting.
        """
        raise NotImplementedError

    def constraint_episode_metrics(self) -> Dict[str, float]:
        """
        Metrics at episode end (terminal/truncated). Should summarize what matters for logging.
        """
        raise NotImplementedError

class BaseConstraintEnv(gym.Wrapper, Constraint):
    """
    Common base for specific constraint wrappers, providing default
    step/episode metric behavior and a place to hold the underlying object.
    """

    def __init__(self, env: gym.Env, constraint: Constraint, **kw):
        if not isinstance(env, LabelledEnv):
            raise TypeError(
                f"{self.__class__.__name__} must wrap a LabelledEnv, "
                f"but got {type(env).__name__}. "
                "Please wrap your environment with LabelledEnv before applying a constraint wrapper."
            )

        super().__init__(env)
        self._constraint = constraint

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._constraint.reset()

        labels = info.get("labels", set())
        if not isinstance(labels, (set, frozenset)):
            raise ValueError(
                f"Expected 'labels' in info to be a set of atomic propositions, got {type(labels).__name__}"
            )

        self._constraint.update(labels)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        labels = info.get("labels", set())
        if not isinstance(labels, (set, frozenset)):
            raise ValueError(
                f"Expected 'labels' in info to be a set of atomic propositions, got {type(labels).__name__}"
            )
            
        self._constraint.update(labels)
        return obs, reward, terminated, truncated, info

    @property
    def cost_fn(self):
        if self._constraint is not None:
            return getattr(self.env._constraint, "cost_fn", None)
        else:
            return None

    @property
    def constraint_type(self) -> str:
        return self._constraint.constraint_type

    def constraint_step_metrics(self) -> Dict[str, float]:
        return self._constraint.step_metric()

    def constraint_episode_metrics(self) -> Dict[str, float]:
        return self._constraint.episode_metric()