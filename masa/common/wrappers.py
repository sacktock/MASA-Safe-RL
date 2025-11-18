from __future__ import annotations
import gymnasium as gym
from masa.common.constraints import BaseConstraintEnv
from masa.common.ltl import DFACostFn, DFA, ShapedCostFn
import numpy as np
from collections import deque
from tqdm import tqdm

class ConstraintPersistentWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def _constraint(self):
        return getattr(self.env, "_constraint", None)

    @property
    def cost_fn(self):
        if self._constraint is not None:
            return getattr(self.env._constraint, "cost_fn", None)
        else:
            return None
        
class TimeLimit(ConstraintPersistentWrapper):

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

class ConstraintMonitor(ConstraintPersistentWrapper):

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

class RewardMonitor(ConstraintPersistentWrapper):

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

class RewardShapingWrapper(ConstraintPersistentWrapper):

    def __init__(self, env: gym.Env, gamma: float = 0.99, impl: str = "none"):
        super().__init__(env)
        self._last_potential = 0.0
        self._gamma = gamma
        self._impl = impl

        self._setup_potential_fn()
        self._setup_cost_fn()

    def _setup_cost_fn(self):
        if hasattr(self._constraint, "cost_fn") and isinstance(self._constraint.cost_fn, DFACostFn):
            self.shaped_cost_fn = ShapedCostFn(self._constraint.cost_fn.dfa, self.potential_fn, gamma=self._gamma)
        else:
            self.shaped_cost_fn = lambda q: 0.0

    def _setup_potential_fn(self):

        if self._impl != "none":
            assert hasattr(self._constraint, "cost_fn"), \
            ("RewardShapingWrapper requires env to implement a BaseConstraintEnv that exposes a cost_fn")
            assert isinstance(getattr(self._constraint, "cost_fn", None), DFACostFn), \
            ("RewardShapingWrapper requires env to implement a LTLSafetyEnv with cost_fn class: DFACostFn")

            dfa: DFA = self.env._constraint.cost_fn.dfa
        else:
            self.potential_fn = lambda q: 0.0

        if self._impl == "vi":

            VI_STEPS = 100
            GAMMA = 0.9
            self.V = {q: 0.0 for q in dfa.states}
            assert GAMMA <= self._gamma

            print("Reward shaping DFA ...")
            for i in tqdm(range(VI_STEPS)):
                diff = 0.0
                for u in dfa.states:
                    V_u = self.V[u]
                    self.V[u] = 1.0/(1.0 - GAMMA) if u in dfa.accepting else \
                        np.max([GAMMA * self.V[v] for v in dfa.edges[u].keys()])
                    diff = max(diff, np.abs(V_u - self.V[u]))
                if diff < 1e-6:
                    break

            self.potential_fn = lambda q: self.V[q]

        if self._impl == "cycle":
            
            print("Reward shaping DFA ...")
            edges_rev = {v: set() for v in dfa.states}
            for u in dfa.states:
                if u in dfa.edges:
                    reachable_states = set(dfa.edges[u].keys())
                    for v in reachable_states:
                        edges_rev[v].add(u)

            dist_to_accepting = {u: np.inf for u in dfa.states}
            queue = deque()

            for a in dfa.accepting:
                dist_to_accepting[a] = 0.0
                queue.append(a)

            max_dist = -1
            furthest_state = None

            while queue:
                current = queue.popleft()
                current_dist = dist_to_accepting[current]

                if current_dist > max_dist:
                    max_dist = current_dist
                    furthest_state = current

                for w in edges_rev.get(current, []):
                    if dist_to_accepting[w] == np.inf:
                        dist_to_accepting[w] = current_dist + 1
                        queue.append(w)

            if furthest_state is None:
                furthest_state = dfa.initial

            self.dist_to_furthest = {u: np.inf for u in dfa.states}
            max_finite_dist = 0.0

            if furthest_state is not None:
                u_target = furthest_state
                self.dist_to_furthest[u_target] = 0.0
                queue = deque([u_target])

                while queue:
                    current = queue.popleft()
                    current_dist = self.dist_to_furthest[current]
        
                    if current_dist > max_finite_dist:
                        max_finite_dist = current_dist
                    
                    for w in edges_rev.get(current, []):
                        if self.dist_to_furthest[w] == np.inf:
                            self.dist_to_furthest[w] = current_dist + 1
                            queue.append(w)

            replacement_value = max_finite_dist + 1.0

            for u in dfa.states:
                if self.dist_to_furthest[u] == np.inf:
                    self.dist_to_furthest[u] = replacement_value

            self.potential_fn = lambda q: self.dist_to_furthest[q]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        automaton_state = info.get("automaton_state", 0)
        self._last_potential = self.potential_fn(automaton_state)
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        cost = info["constraint"]["step"].get("cost", 0.0)
        automaton_state = info.get("automaton_state", 0)
        potential = self.potential_fn(automaton_state)
        info["constraint"]["step"]["cost"] = cost + self._gamma * potential - self._last_potential
        self._last_potential = potential
        return observation, reward, terminated, truncated, info

    @property
    def cost_fn(self):
        return self.shaped_cost_fn
        

        

        
