import warnings
import gymnasium as gym
from gymnasium import spaces
from masa.common.wrappers import ConstraintPersistentWrapper, is_wrapped
from masa.common.constraints.ltl_safety import LTLSafetyEnv
from typing import Union, Any, Tuple, Dict, List, Optional, Callable
from masa.common.constraints.base import CostFn
from masa.common.label_fn import LabelFn
from masa.prob_shield.helpers import build_successor_states_matrix
from masa.prob_shield.interval_bound_vi import interval_bound_value_iteration
import numpy as np
import copy

def projection_bar(intersections: np.ndarray, i: int) -> np.ndarray:
    assert intersections.shape[0] > 0, "No intersections provided."

    target_vertex = np.zeros_like(intersections[0])
    target_vertex[i] = 1.0

    diffs = intersections - target_vertex[None, :]
    distances = np.linalg.norm(diffs, axis=1) 

    if np.any(distances == 0.0):
        raise RuntimeError("Unexpected null distance in projection_bar")

    weights = 1.0 / distances

    total_weight = weights.sum()
    if total_weight == 0.0:
        raise RuntimeError("Unexpected null total_weight in projection_bar")

    weighted_sum = (weights[:, None] * intersections).sum(axis=0)

    result_vertex = weighted_sum / total_weight
    return result_vertex.astype(np.float64)


def projection_bar_min(intersections: np.ndarray, i: int, j: int) -> np.ndarray:
    assert intersections.shape[0] > 0, "No intersections provided."

    target_vertex_j = np.zeros_like(intersections[0])
    target_vertex_j[j] = 1.0

    target_vertex_i = np.zeros_like(intersections[0])
    target_vertex_i[i] = 1.0

    diffs_i = intersections - target_vertex_i[None, :]
    diffs_j = intersections - target_vertex_j[None, :]

    distances_i = np.linalg.norm(diffs_i, axis=1)
    distances_j = np.linalg.norm(diffs_j, axis=1)

    distances = np.minimum(distances_i, distances_j)  

    if np.any(distances == 0.0):
        raise RuntimeError("Unexpected null distance in projection_bar_min")

    weights = 1.0 / distances
    total_weight = weights.sum()
    if total_weight == 0.0:
        raise RuntimeError("Unexpected null total_weight in projection_bar_min")

    weighted_sum = (weights[:, None] * intersections).sum(axis=0)
    result_vertex = weighted_sum / total_weight

    return result_vertex.astype(np.float64)

class ProbShieldWrapperBase(ConstraintPersistentWrapper):

    def __init__(
        self,
        env: gym.Env,
        label_fn: Optional[LabelFn] = None,
        cost_fn: Optional[CostFn] = None,
        safety_abstraction: Optional[Callable[[Any], int]] = None,
        theta: float = 1e-10,
        max_vi_steps: int = 1000,
        init_safety_bound: float = 0.5,
    ):
        super().__init__(env)

        self.safety_abstraction = safety_abstraction
        self.theta = theta
        self.max_vi_steps = max_vi_steps
        self.init_safety_bound = init_safety_bound
        self._box_dtype = np.float32

        # Sanity checks
        if is_wrapped(self.env, ProbShieldWrapperDisc):
            raise RuntimeError(
                "Environment is already wrapped in ProbShieldWrapperDisc. "
                "Double-wrapping can cause undefined behaviour."
            )
        if not isinstance(self.env, ConstraintPersistentWrapper):
            raise TypeError(
                "ProbShieldWrapperDisc expects `env` to be an instance of "
                f"ConstraintPersistentWrapper, got {type(env).__name__}."
            )
        if not isinstance(self.env.observation_space, spaces.Discrete) and safety_abstraction is None:
            raise TypeError(
                "ProbShieldWrapperDisc only supports environments with a "
                f"Discrete observation space or a discrete safety abstraction, got: {type(self.env.observation_space).__name__}"
            )

        self.successor_states_matrix, self.probabilities, self.max_successors, \
        label_fn, cost_fn, safe_set = build_successor_states_matrix(
            self.env, label_fn=label_fn, cost_fn=cost_fn,
        )

        v_inf, v_sup, _, _, = interval_bound_value_iteration(
            self.successor_states_matrix, self.probabilities, label_fn, cost_fn, safe_set, \
            theta=self.theta, max_steps=self.max_vi_steps
        )

        start_state = None
        if is_wrapped(self.env, LTLSafetyEnv):
            dfa: DFA = self.env._constraint.get_dfa()
            if hasattr(self.env.unwrapped, "_start_state"):
                aut_states = list(dfa.states)
                n_aut = len(aut_states)
                aut_index = {q: i for i, q in enumerate(aut_states)}
                start_state = aut_index[dfa.initial] * n_states + int(self.env.unwrapped._start_state)
        else:
            if hasattr(self.env.unwrapped, "_start_state"):
                start_state = int(self.env.unwrapped._start_state)

        if start_state is not None:
            assert v_sup[start_state] <= self.init_safety_bound, f"Value iteration could not verify that the initial safety bound {self.init_safety_bound} is achievable from the initial state"
            print("Initial state lower bound:", v_sup[start_state])

        self.safety_lb = v_sup

        self._orig_obs_space = self.env.observation_space
        self._orig_act_space = self.env.action_space

        assert isinstance(self._orig_act_space, spaces.Discrete)

        self.observation_space = self._make_augmented_obs_space(self._orig_obs_space)
        self.action_space = self._make_augmented_act_space(self._orig_act_space)

    def _abstraction(self, obs: Any) -> int:
        if isinstance(self._orig_obs_space, spaces.Discrete):
            return int(obs)
        else:
            assert self.safety_abstraction is not None, f"If the env observation space type is {type(orig).__name__}, then you must supply a discrete safety abstraction"
            abstr_obs = self.safety_abstraction(obs)
            try:
                abstr_obs = int(abstr_obs)
            except:
                raise TypeError(
                    f"Could not cast abstracted state as type `int`, given type {type(abstr_obs)} instead."
                )
            return abstr_obs

    def _make_augmented_obs_space(self, orig: spaces.Space) -> spaces.Space:
        if isinstance(orig, (spaces.Discrete, spaces.Box)):
            aug = spaces.Dict({
                "orig_obs": orig,
                "safety_bound": spaces.Box(low=0, high=1, shape=(1,), dtype=self._box_dtype)
            })
        elif isinstance(orig, spaces.Dict):
            aug = dict(orig.spaces)
            aug["safety_bound"] = spaces.Box(low=0, high=1, shape=(1,), dtype=self._box_dtype)
        else:
            raise TypeError(
                f"ProbShieldWrapperDisc does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, Box, Dict."
            )
        return aug

    def _make_augmented_act_space(self, orig: spaces.Discrete) -> Any:
        raise NotImplementedError
    
    def _augment_obs(self, obs: Any) -> Dict[str, Any]:
        # No one-hotting here, we let the algorithm decide how they want to handle discrete observations
        if isinstance(self._orig_obs_space, (spaces.Discrete, spaces.Box)):
            return dict({
                "orig_obs": obs,
                "safety_bound": float(self._current_safety_bound)
            })
        elif isinstance(self._orig_obs_space, (spaces.Dict)):
            obs = dict(obs)
            obs["safety_bound"] = float(self._current_safety_bound)
            return obs
        else:
            raise TypeError(
                f"ProbShieldWrapperDisc does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, Box, Dict."
            )

    def _one_hot(self, s: int) -> np.ndarray:
        raise NotImplementedError

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        raise NotImplementedError 

    def _project_act(self, i: int, j: int, betas: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:

        if not np.all(np.isfinite(betas)):
            raise RuntimeError(f"Non-finite betas: {betas}")
        
        # Calculate non-zero successors
        probs_curr = self.probabilities[:, self._current_obs, :].copy()
        mask = np.any(probs_curr > 0.0, axis=1)
        successors_s = np.nonzero(mask)[0]
        k = len(successors_s)
        probs_curr = probs_curr[successors_s]

        betas = betas[:k]
        obs_successor_states = self.successor_states_matrix[successors_s, self._current_obs]
        obs_safety_lb = self.safety_lb[obs_successor_states]
        safety_bounds = obs_safety_lb + (1.0 - obs_safety_lb) * betas
        proj_safety_bounds = safety_bounds.copy()

        # Project the infeasible or unnecessarily safe alphas
        all_out_and_nonzero_safety_bounds = True
        all_in_and_nonzero_safety_bounds = True 

        expected_safety = np.sum(probs_curr * safety_bounds[:, None], axis=0)
        diffs = self._current_safety_bound - expected_safety

        all_out_and_nonzero_safety_bounds = bool(np.all(diffs <= -eps))
        all_in_and_nonzero_safety_bounds = bool(np.all(diffs >= eps))

        if all_out_and_nonzero_safety_bounds:
            numerators = self._current_safety_bound  - np.sum(probs_curr * safety_bounds[:, None], axis=0)
            diff_sb = obs_safety_lb - safety_bounds 
            denominators = np.sum(probs_curr * diff_sb[:, None], axis=0)

            nonzero_mask = denominators != 0.0

            if np.any(nonzero_mask):
                 # candidates for all valid actions
                candidates = numerators[nonzero_mask] / denominators[nonzero_mask]
                # take minimum over valid actions
                max_safe_scaling = np.min(candidates)
            else:
                max_safe_scaling = 1.0

            max_safe_scaling = np.clip(max_safe_scaling, 0.0, 1.0)
            proj_safety_bounds = (1 - max_safe_scaling) * proj_safety_bounds + max_safe_scaling * obs_safety_lb

        if all_in_and_nonzero_safety_bounds:
            numerators = self._current_safety_bound  - np.sum(probs_curr * safety_bounds[:, None], axis=0)
            denominators = 1.0 - np.sum(probs_curr * safety_bounds[:, None], axis=0)

            nonzero_mask = denominators != 0.0

            if np.any(nonzero_mask):
                 # candidates for all valid actions
                candidates = numerators[nonzero_mask] / denominators[nonzero_mask]
                # take minimum over valid actions
                max_unsafe_scaling = np.min(candidates)
            else:
                max_unsafe_scaling = 1.0

            max_unsafe_scaling = np.clip(max_unsafe_scaling, 0.0, 1.0)
            proj_safety_bounds = (1 - max_unsafe_scaling) * proj_safety_bounds + max_unsafe_scaling * np.ones_like(proj_safety_bounds)

        # Compute the intersections of the hyperplane and the probability polytope
        intersections_list = []

        expected_proj_safety = np.sum(probs_curr * proj_safety_bounds[:, None], axis=0)
        proj_diffs = self._current_safety_bound - expected_proj_safety
        n_actions = proj_diffs.shape[0]

        # For the diagonals
        diag_mask = np.abs(proj_diffs) < eps
        diag_indices = np.nonzero(diag_mask)[0]

        if diag_indices.size > 0:
            # Take corresponding rows of identity: shape (n_diag, n_actions)
            diag_intersections = np.eye(n_actions, dtype=np.float64)[diag_indices]
            intersections_list.append(diag_intersections)

        # For the off diagonals
        # Upper triangular indices
        i_idx, j_idx = np.triu_indices(n_actions, k=1) 

        diff_i = proj_diffs[i_idx]
        diff_j = proj_diffs[j_idx] 

        # Sign change condition with eps tolerance
        sign_change_mask = (
            ((diff_i >  eps) & (diff_j < -eps)) |
            ((diff_i < -eps) & (diff_j >  eps))
        )

        if np.any(sign_change_mask):
            i_valid = i_idx[sign_change_mask]
            j_valid = j_idx[sign_change_mask]
            diff_i_valid = diff_i[sign_change_mask]
            diff_j_valid = diff_j[sign_change_mask]

            # Compute barycentric coordinates for all valid pairs at once
            alpha_i = diff_j_valid / (diff_j_valid - diff_i_valid)
            alpha_j = diff_i_valid / (diff_i_valid - diff_j_valid)

            # Build intersection matrix: each row is a point in the simplex
            n_valid = i_valid.shape[0]
            edge_intersections = np.zeros((n_valid, n_actions), dtype=np.float64)

            rows = np.arange(n_valid)
            edge_intersections[rows, i_valid] = alpha_i
            edge_intersections[rows, j_valid] = alpha_j

            intersections_list.append(edge_intersections)

        # Stack everything
        if intersections_list:
            intersections = np.vstack(intersections_list)
        else:
            intersections = np.empty((0, n_actions), dtype=np.float64)

        if intersections.shape[0] == 0:
            warning.warn("Empty intersections in _project_act ... falling back to the closesly vertex.")
            idx = int(np.nanargmin(np.abs(self._current_safety_bound - expected_proj_safety)))
            safe_vertex = np.zeros(n_actions, dtype=np.float64)
            safe_vertex[idx] = 1.0
            return safe_vertex, proj_safety_bounds

        # Compute the vertex corresponding to the action
        safe_vertex = np.zeros(n_actions, dtype=np.float64)

        diff_act_i = self._current_safety_bound - expected_proj_safety[i]
        diff_act_j = self._current_safety_bound - expected_proj_safety[j]

        if i == j:
            if np.abs(diff_act_i) < eps:
                safe_vertex[i] = 1.0
            else:
                safe_vertex = projection_bar(intersections, i)
        else:
            if np.abs(diff_act_i) < eps:
                safe_vertex[i] = 1.0
            else:
                if np.abs(diff_act_j) < eps:
                    safe_vertex[j] = 1.0
                elif (diff_act_i < -eps and diff_act_j < -eps) or (diff_act_i > eps and diff_act_j > eps):
                    safe_vertex = projection_bar_min(intersections, i, j)
                else:
                    safe_vertex = np.zeros(n_actions, dtype=np.float64)
                    safe_vertex[i] = diff_act_j / (diff_act_j - diff_act_i)
                    safe_vertex[j] = diff_act_i / (diff_act_i - diff_act_j)

        return safe_vertex, proj_safety_bounds

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_safety_bound = self.init_safety_bound
        abstr_obs = self._abstraction(obs)
        self._current_obs = abstr_obs
        return self._augment_obs(obs), info

    def step(self, action):
        i, j, betas = self._parse_act(action)
        safe_vertex, proj_safety_bounds = self._project_act(i, j, betas)

        # Renormalize safe_vertex:
        #   safe_vertex can often be affected by floating point errors
        safe_vertex = np.abs(safe_vertex)
        safe_vertex = safe_vertex / np.sum(safe_vertex)

        orig_action = self.np_random.choice(len(safe_vertex), p=safe_vertex)
        orig_obs, reward, terminated, truncated, info = self.env.step(orig_action)

        abstr_obs = self._abstraction(orig_obs)

        succ_list = self.successor_states_matrix[:, self._current_obs]
        matches = np.where(succ_list == abstr_obs)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"Something went wrong! Next abstract state {abstr_obs} is not in successor list for current abstract state {self._current_obs}."
            )
        next_obs_idx = int(matches[0])
        
        self._current_safety_bound = proj_safety_bounds[next_obs_idx]
        self._current_obs = abstr_obs

        return self._augment_obs(orig_obs), reward, terminated, truncated, info

class ProbShieldWrapperDisc(ProbShieldWrapperBase):

    def __init__(
        self, 
        env: gym.Env,
        label_fn: Optional[LabelFn] = None,
        cost_fn: Optional[CostFn] = None,
        safety_abstraction: Optional[Callable[[Any], int]] = None,
        theta: float = 1e-10,
        max_vi_steps: int = 1000,
        init_safety_bound: float = 0.5,
        granularity: int = 20,
    ):

        self.granularity = granularity

        super().__init__(
            env, 
            label_fn=label_fn, 
            cost_fn=cost_fn, 
            safety_abstraction=safety_abstraction, 
            theta=theta, 
            max_vi_steps=max_vi_steps, 
            init_safety_bound=init_safety_bound
        )

    def _make_augmented_act_space(self, orig: spaces.Discrete) -> spaces.MultiDiscrete:
        n_actions = orig.n
        aug = spaces.MultiDiscrete([n_actions]*2+[self.granularity+1]*self.max_successors)
        return aug

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        i, j = action[0], action[1]
        betas = action[2:] / self.granularity
        return i, j, betas

class ProbShieldWrapperCont(ProbShieldWrapperBase):

    def __init__(
        self, 
        env: gym.Env,
        label_fn: Optional[LabelFn] = None,
        cost_fn: Optional[CostFn] = None,
        safety_abstraction: Optional[Callable[[Any], int]] = None,
        theta: float = 1e-10,
        max_vi_steps: int = 1000,
        init_safety_bound: float = 0.5,
    ):

        super().__init__(
            env, 
            label_fn=label_fn, 
            cost_fn=cost_fn, 
            safety_abstraction=safety_abstraction, 
            theta=theta, 
            max_vi_steps=max_vi_steps, 
            init_safety_bound=init_safety_bound
        )

    def _make_augmented_act_space(self, orig: spaces.Discrete) -> spaces.Dict:
        n_actions = orig.n
        aug = spaces.Dict({
            "multi_discrete": spaces.MultiDiscrete([n_actions]*2),
            "box": spaces.Box(low=0, high=1, shape=(self.max_successors,), dtype=self._box_dtype)
        })
        return aug

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        i, j = int(action[0].item()), int(action[1].item())
        betas = action[2:].astype(np.float64)
        return i, j, betas

    

    




    