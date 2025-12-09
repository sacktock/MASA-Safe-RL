from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
from masa.common.constraints import BaseConstraintEnv
from masa.common.ltl import DFACostFn, DFA, ShapedCostFn
from masa.common.running_mean_std import RunningMeanStd
import numpy as np
from collections import deque
from tqdm import tqdm

def is_wrapped(env: gym.Env, wrapper_class: gym.Wrapper) -> bool:
    """
    Check if env is wrapped anywhere in the chain by wrapper_class.
    Works for both Gymnasium-style wrappers (.env) and VecEnv-style (.venv).
    """
    current = env
    visited = set()

    while True:
        if id(current) in visited:
            return False
        visited.add(id(current))

        if isinstance(current, wrapper_class):
            return True

        if hasattr(current, "venv"):
            current = current.venv
            continue

        if isinstance(current, gym.Wrapper):
            current = current.env
            continue

        return False

def get_wrapped(env: gym.Env, wrapper_class: gym.Wrapper) -> gym.Env:

    current = env
    visited = set()

    while True:
        if id(current) in visited:
            return None
        visited.add(id(current))

        if isinstance(current, wrapper_class):
            return current

        if hasattr(current, "venv"):
            current = current.venv
            continue

        if isinstance(current, gym.Wrapper):
            current = current.env
            continue

        return None

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

    @property
    def label_fn(self):
        return getattr(self.env, "label_fn", None)

class ConstraintPersistentObsWrapper(ConstraintPersistentWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _get_obs(obs: Any) -> Any:
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._get_obs(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._get_obs(obs), rew, term, trunc, info
        
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
        info["constraint"]["episode"] = self._episode_metrics()
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

    def step(self, action: Any):
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

class NormWrapper(ConstraintPersistentWrapper):

    def __init__(
        self, 
        env: gym.Env, 
        norm_obs: bool = True,
        norm_rew: bool = True,
        training: bool = True,
        clip_obs: float = 10.0,
        clip_rew: float = 10.0,
        gamma: float = 0.99,
        eps: float = 1e-8
    ):
        assert not isinstance(
            env, VecEnvWrapperBase
        ), "NormWrapper does not expect a vectorized environment (DummyVecWrapper / VecWrapper). Please use VecNormWrapper instead"

        assert norm_obs and isinstance(
            env.observation_space, spaces.Box
        ), "NormWrapper only supports Box observation spaces when norm_obs=True."

        super().__init__(env)

        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.training = training
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.eps = eps

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.rew_rms = RunningMeanStd(shape=())

        self.returns = np.zeros(1, dtype=np.float32)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps),
            -self.clip_obs,
            self.clip_obs
        )

    def _normalize_rew(self, rew: float) -> float:
        return np.clip(
            rew / np.sqrt(self.rew_rms.var + self.eps),
            -self.clip_rew,
            self.clip_rew,
        )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs)

        if self.norm_rew and self.training:
            self.returns[:] = 0.0

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs)

        if self.norm_rew:
            self.returns = self.returns * self.gamma + rew
            if self.training:
                self.rew_rms.update(self.returns)

            rew = self._normalize_rew(rew)

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, rew, term, trunc, info

class OneHotObsWrapper(ConstraintPersistentObsWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._orig_obs_space = self.env.observation_space

        if isinstance(self._orig_obs_space, spaces.Discrete):
            self._mode = "discrete"
            n = self._orig_obs_space.n
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n,),
                dtype=np.float32,
            )
            

        elif isinstance(self._orig_obs_space, spaces.Dict):
            self._mode = "dict"

            new_spaces: Dict[str, spaces.Space] = {}
            for key, subspace in self._orig_obs_space.spaces.items():
                if isinstance(subspace, spaces.Discrete):
                    n = subspace.n
                    new_spaces[key] = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(n,),
                        dtype=np.float32,
                    )
                else:
                    # Preserve non-Discrete subspace as-is
                    new_spaces[key] = subspace

            self.observation_space = spaces.Dict(new_spaces)

        else:
            self._mode = "pass"
            self.observation_space = self._orig_obs_space

    @staticmethod
    def _one_hot_scalar(idx: int, n: int) -> np.ndarray:
        """One-hot encode a single integer index into a 1D vector of length n."""
        one_hot = np.zeros(n, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _get_obs(self, obs: Union[int, Dict[str, Any], np.ndarray]) -> np.ndarray:
        if self._mode == "discrete":
            # Original obs_space is Discrete; obs is an int-like
            idx = int(obs)
            n = self._orig_obs_space.n
            return self._one_hot_scalar(idx, n)

        elif self._mode == "dict":
            assert isinstance(obs, dict), (
                f"Expected dict observation for Dict space, got {type(obs)}"
            )

            new_obs: Dict[str, Any] = {}
            for key, subspace in self._orig_obs_space.spaces.items():
                value = obs[key]

                if isinstance(subspace, spaces.Discrete):
                    idx = int(value)
                    new_obs[key] = self._one_hot_scalar(idx, subspace.n)
                else:
                    # Leave non-Discrete parts unchanged
                    new_obs[key] = value

            return new_obs
        else: 
            # pass
            return obs

class FlattenDictObsWrapper(ConstraintPersistentObsWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._orig_obs_space = self.env.observation_space

        if not isinstance(self._orig_obs_space, spaces.Dict):
            raise TypeError(
                f"FlattenDictObsWrapper requires Dict observation space, got {type(self._orig_obs_space)}"
            )

        # To be able to reconstruct if needed, keep slices for each key
        self._key_slices: dict[str, slice] = {}

        low_parts = []
        high_parts = []
        offset = 0

        # Sort keys alphabetically for deterministic ordering
        for key in sorted(self._orig_obs_space.spaces.keys()):
            subspace = self._orig_obs_space.spaces[key]

            if isinstance(subspace, spaces.Box):
                # Flatten Box
                low = np.asarray(subspace.low, dtype=np.float32).reshape(-1)
                high = np.asarray(subspace.high, dtype=np.float32).reshape(-1)
                length = low.shape[0]

                low_parts.append(low)
                high_parts.append(high)

            elif isinstance(subspace, spaces.Discrete):
                # One-hot will be in [0, 1]
                length = subspace.n
                low_parts.append(np.zeros(length, dtype=np.float32))
                high_parts.append(np.ones(length, dtype=np.float32))

            else:
                raise TypeError(
                    f"Unsupported subspace type for key '{key}': {type(subspace)}"
                )

            self._key_slices[key] = slice(offset, offset + length)
            offset += length

        low = np.concatenate(low_parts).astype(np.float32)
        high = np.concatenate(high_parts).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _get_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        assert isinstance(obs, dict), (
                f"Expected dict observation for Dict space, got {type(obs)}"
            )

        parts = []
        for key in sorted(self._orig_obs_space.spaces.keys()):
            subspace = self._orig_obs_space.spaces[key]
            value = obs[key]

            if not isinstance(subspace, spaces.Box):
                raise TypeError(
                    f"FlattenDictObsWrapper only supports Box subspaces, "
                    f"got {type(subspace)} for key '{key}'"
                )

            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            parts.append(arr)

        return np.concatenate(parts, axis=0).astype(np.float32)

class VecEnvWrapperBase(ConstraintPersistentWrapper):

    n_envs: int

    def __init__(self, env: gym.Env):
        # For DummyVecWrapper: env is the single env
        # For VecWrapper: env is envs[0]
        # For VecNormWrapper: env is a VecEnvWrapperBase
        super().__init__(env)

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        raise NotImplementedError
        
class DummyVecWrapper(VecEnvWrapperBase):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_envs = 1
        self.envs: List[gym.Env] = [env]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return [obs], [info]

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        dones = list(dones)
        assert len(dones) == 1
        if dones[0]:
            return self.reset(seed=seed, options=options)
        else:
            [None], [{}]

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return [obs], [rew], [term], [trunc], [info]

class VecWrapper(VecEnvWrapperBase):

    def __init__(self, envs: List[gym.Env]):
        assert len(envs) > 0, "VecWrapper requires at least one environment"
        super().__init__(envs[0]) # maintain API compatibility
        self.envs: List[gym.Env] = envs
        self.n_envs = len(envs)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            obs, info = env.reset(seed=s, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        dones = list(dones)
        assert len(dones) == self.n_envs

        reset_obs = [None] * self.n_envs
        reset_infos = [{} for _ in range(self.n_envs)]

        for i, done in enumerate(dones):
            if done:
                s = None if seed is None else seed + i
                obs, info = self.envs[i].reset(seed=s, options=options)
                reset_obs[i] = obs
                reset_infos[i] = info

        return reset_obs, reset_infos

    def step(self, action):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(action)
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        return obs_list, rew_list, term_list, trunc_list, info_list


class VecNormWrapper(VecEnvWrapperBase):

    def __init__(
        self, 
        env: Union[gym.Env, List[gym.Env]], 
        norm_obs: bool = True,
        norm_rew: bool = True,
        training: bool = True,
        clip_obs: float = 10.0,
        clip_rew: float = 10.0,
        gamma: float = 0.99,
        eps: float = 1e-8
    ):
        assert isinstance(
            env, VecEnvWrapperBase
        ), "VecNormWrapper expects a vectorized environment (DummyVecWrapper / VecWrapper)."

        assert norm_obs and isinstance(
            env.observation_space, spaces.Box
        ), "VecNormWrapper only supports Box observation spaces when norm_obs=True."

        super().__init__(env)

        self.n_envs = env.n_envs
        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.training = training
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew
        self.gamma = gamma
        self.eps = eps

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.rew_rms = RunningMeanStd(shape=())

        self.returns = np.zeros(self.n_envs, dtype=np.float32)

    def _normalize_obs(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        obs_arr = np.asarray(obs_list, dtype=np.float32)
        norm = (obs_arr - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
        norm = np.clip(norm, -self.clip_obs, self.clip_obs)
        return norm.tolist()

    def _normalize_rew(self, rew_list: List[float]) -> List[float]:
        rew_arr = np.asarray(rew_list, dtype=np.float32)
        norm = rew_arr / np.sqrt(self.rew_rms.var + self.eps)
        norm = np.clip(norm, -self.clip_rew, self.clip_rew)
        return norm.tolist()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs_list, info_list = self.env.reset(seed=seed, options=options)

        if self.norm_obs and self.training:
            self.obs_rms.update(np.asarray(obs_list, dtype=np.float32))

        self.returns[:] = 0.0

        if self.norm_obs:
            obs_list = self._normalize_obs(obs_list)

        return obs_list, info_list

    def reset_done(
        self, 
        dones: Union[List[bool], np.ndarray],
        *, 
        seed: int | None = None, 
        options: Dict[str, Any] | None = None
    ):
        reset_obs, reset_infos = self.env.reset_done(
            dones, seed=seed, options=options
        )

        obs_arr = np.asarray(
            [o for o in reset_obs if o is not None],
            dtype=np.float32,
        ) if any(o is not None for o in reset_obs) else None

        if self.norm_obs and self.training and obs_arr is not None:
            self.obs_rms.update(obs_arr)

        for i, done in enumerate(dones):
            if done:
                self.returns[i] = 0.0

        if self.norm_obs:
            norm_reset_obs: List[Any] = list(reset_obs)
            # Only normalize indices that were reset
            for i, done in enumerate(dones):
                if done and reset_obs[i] is not None:
                    o = np.asarray(reset_obs[i], dtype=np.float32)
                    norm = (o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
                    norm = np.clip(norm, -self.clip_obs, self.clip_obs)
                    norm_reset_obs[i] = norm
            reset_obs = norm_reset_obs

        return reset_obs, reset_infos

    def step(self, actions):
        obs_list, rew_list, term_list, trunc_list, infos = self.env.step(actions)

        obs_arr = np.asarray(obs_list, dtype=np.float32)
        rew_arr = np.asarray(rew_list, dtype=np.float32)

        if self.norm_obs and self.training:
            self.obs_rms.update(obs_arr)

        if self.norm_rew:
            self.returns = self.returns * self.gamma + rew_arr
            if self.training:
                self.rew_rms.update(self.returns)

            rew_list = self._normalize_rew(rew_list)

        if self.norm_obs:
            obs_list = self._normalize_obs(obs_list)

        return obs_list, rew_list, term_list, trunc_list, infos

        