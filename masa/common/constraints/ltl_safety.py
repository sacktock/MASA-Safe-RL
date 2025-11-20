from __future__ import annotations
from typing import Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from masa.common.constraints import Constraint, BaseConstraintEnv
from masa.common.ltl import DFA, dfa_to_costfn
from masa.examples.dummy import dfa as dummy_dfa

class LTLSafety(Constraint):

    def __init__(self, dfa: DFA):
        self.cost_fn = dfa_to_costfn(dfa)

    def reset(self):
        self.safe = True
        self.step_cost = 0.0
        self.total_unsafe = 0.0
        self.cost_fn.reset()

    def update(self, labels: Iterable[str]):
        self.step_cost = self.cost_fn(labels)
        self.total_unsafe = float(self.step_cost >= 0.5)
        self.safe = self.safe and (not self.total_unsafe)

    def get_automaton_state(self):
        return self.cost_fn.automaton_state

    def get_dfa(self):
        return self.cost_fn.dfa

    def satisfied(self) -> bool:
        return self.safe

    def episode_metric(self) -> Dict[str, float]:
        return {"cum_unsafe": float(self.total_unsafe), "satisfied": float(self.satisfied())}

    def step_metric(self) -> Dict[str, float]:
        return {"cost": self.step_cost, "violation": float(self.step_cost >= 0.5)}

    @property
    def constraint_type(self) -> str:
        return "ltl_safety"


class LTLSafetyEnv(BaseConstraintEnv):

    def __init__(self, env: gym.Env, dfa: DFA = dummy_dfa, **kw):
        super().__init__(env, LTLSafety(dfa=dfa), **kw)
        self._num_automaton_states = int(dfa.num_automaton_states)
        if self._num_automaton_states <= 0:
            raise ValueError("dfa.num_automaton_states must be positive")
        self._orig_obs_space = env.observation_space
        self.observation_space = self._make_augmented_obs_space(self._orig_obs_space)
        self._box_dtype = np.float32

    def _make_augmented_obs_space(self, orig: spaces.Space) -> spaces.Space:
        if isinstance(orig, spaces.Discrete):
            num_states = int(orig.n)
            aug = spaces.Discrete(num_states * self._num_automaton_states)
            return aug
        if isinstance(orig, spaces.Box):
            if orig.shape is None or len(orig.shape) != 1:
                raise TypeError(
                    f"LTLSafetyEnv only supports 1-D Box for augmentation; got shape {orig.shape}"
                )
            n = int(orig.shape[0])
            low = np.concatenate([orig.low.astype(self._box_dtype, copy=False),
                                  np.zeros(self._num_automaton_states, dtype=self._box_dtype)])
            high = np.concatenate([orig.high.astype(self._box_dtype, copy=False),
                                   np.ones(self._num_automaton_states, dtype=self._box_dtype)])
            aug = spaces.Box(low=low, high=high, dtype=self._box_dtype)
            return aug
        if isinstance(orig, spaces.Dict):
            automaton_space = spaces.Box(low=0.0, high=1.0, shape=(self._num_automaton_states,), dtype=self._box_dtype)
            new_spaces = dict(orig.spaces)
            new_spaces["automaton"] = automaton_space
            aug = spaces.Dict(new_spaces)
            return aug

        raise TypeError(
            f"LTLSafetyEnv does not support observation space type {type(orig).__name__}. "
            "Supported: Discrete, 1-D Box, Dict."
        )

    def _one_hot(self, q: int) -> np.ndarray:
        enc = np.zeros(self._num_automaton_states, dtype=self._box_dtype)
        if 0 <= q < self._num_automaton_states:
            enc[q] = 1
        return enc

    def _augment_obs(self, obs: Any) -> Any:
        q = self._constraint.get_automaton_state()
        if isinstance(self.observation_space, spaces.Discrete):
            if not (isinstance(obs, (int, np.integer))):
                raise TypeError(f"Expected Discrete obs as int, got {type(obs).__name__}")
            return self._orig_obs_space.n * int(q) + int(obs)
        if isinstance(self.observation_space, spaces.Box):
            if not isinstance(obs, np.ndarray):
                obs = np.asarray(obs, dtype=self._box_dtype)
            if obs.ndim != 1:
                raise TypeError(f"Expected 1-D Box observation, got shape {getattr(obs, 'shape', None)}")
            enc = self._one_hot(q, dtype=self._box_dtype)
            return np.concatenate([obs.astype(self._box_dtype, copy=False), enc], axis=0)
        if isinstance(self.observation_space, spaces.Dict):
            out = dict(obs) if isinstance(obs, dict) else {}
            out["automaton"] = self._one_hot(q, dtype=np.float32)
            return out

        raise RuntimeError(f"Unexpected observation space type {self.observation_space}")

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._constraint.reset()
        labels = info.get("labels", set())
        self._constraint.update(labels)
        info['automaton_state'] = self._constraint.get_automaton_state()
        return self._augment_obs(obs), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        labels = info.get("labels", set())
        self._constraint.update(labels)
        info['automaton_state'] = self._constraint.get_automaton_state()
        return self._augment_obs(obs), reward, terminated, truncated, info
    

        