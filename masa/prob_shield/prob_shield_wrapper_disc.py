
import gymnasium as gym
from gymnasium import spaces
from masa.common.wrappers import ConstraintPersistentWrapper
from masa.envs.tabular.base import TabularEnv
from masa.envs.discrete.base import DiscreteEnv
from typing import Union, Any, Tuple, Dict, List
from masa.common.constraints import CostFn
from masa.common.label_fn import LabelFn
from jax import jit
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import copy

@jit
def vi_one_step(
    v_inf: jnp.ndarray, 
    v_sup: jnp.ndarray, 
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray
):
    """
    v_inf, v_sup: (n_states,)
    successor_states_matrix: (max_successors, n_states) int
    probabilities: (max_successors, n_states, n_actions)
    """

    next_v_inf = jnp.take(v_inf, successor_states_matrix, axis=0)
    next_v_sup = jnp.take(v_sup, successor_states_matrix, axis=0)

    exp_inf = jnp.sum(next_v_inf[..., None] * probabilities, axis=0)
    exp_sup = jnp.sum(next_v_sup[..., None] * probabilities, axis=0)

    v_inf_new = jnp.min(exp_inf, axis=-1)
    v_sup_new = jnp.min(exp_sup, axis=-1)

    delta = jnp.max(jnp.abs(v_sup_new - v_inf_new))

    return v_inf_new, v_sup_new, delta

def jax_value_iteration(
    v_inf_init: jnp.ndarray,
    v_sup_init: jnp.ndarray,
    successor_states_matrix: jnp.ndarray,
    probabilities: jnp.ndarray,
    theta: float,
    max_steps: int,
):

    v_inf = v_inf_init
    v_sup = v_sup_init
    delta = jnp.inf
    steps = 0

    def cond(carry):
        v_inf, v_sup, delta, steps = carry
        return jnp.logical_and(steps < max_steps, delta > theta)

    def body(carry):
        v_inf, v_sup, _, steps = carry
        v_inf_new, v_sup_new, delta_new = vi_one_step(
            v_inf, v_sup, successor_states_matrix, probabilities
        )
        return v_inf_new, v_sup_new, delta_new, steps + 1

    v_inf, v_sup, delta, steps = lax.while_loop(
        cond,
        body,
        (v_inf, v_sup, delta, steps)
    )
    return v_inf, v_sup, delta, steps

def fast_value_iteration(
    v_inf: np.ndarray,
    v_sup: np.ndarray,
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    theta: float = 1e-10,
    max_steps: int = 1000,
    device: str = 'auto',
):
    print("Initializing value iteration ...")

    v_inf_j = jnp.array(v_inf)
    v_sup_j = jnp.array(v_sup)
    successor_states_j = jnp.array(successor_states_matrix, dtype=jnp.int32)
    probabilities_j = jnp.array(probabilities, dtype=jnp.float32)
    
    v_inf_j, v_sup_j, delta, steps = jax_value_iteration(
        v_inf_j,
        v_sup_j,
        successor_states_j,
        probabilities_j,
        theta,
        max_steps,
    )

    print("Completed value iteration ...")

    v_inf_final = np.array(v_inf_j)
    v_sup_final = np.array(v_sup_j)

    return v_inf_final, v_sup_final, delta, int(steps)

def build_successor_state_matrix_and_probabilities(
    env: TabularEnv,
    label_fn: LabelFn,
    cost_fn: CostFn,
):

    print("Calculating the maximum number of successor states ...")

    use_transition_matrix: bool = env.has_transition_matrix
    use_successor_states_dict: bool = env.has_successor_states_dict and not use_transition_matrix

    if use_transition_matrix:
        probs = np.array(env.get_transition_matrix())
        n_states = probs.shape[0]
        max_successors = np.max(np.count_nonzero(probs, axis=0))

    if use_successor_states_dict:
        successor_states, probs_dict = env.get_successor_states_dict()
        max_successors = 0
        n_states = 0
        for s in successor_states.keys():
            max_successors = max(max_successors, len(successor_states[s]))
            n_states += 1

    n_actions = env.action_space.n

    print(f"Calculated maximum number of successor states [{max_successors}] ...")

    # Initial bounds
    v_inf = np.zeros(n_states, dtype=np.float32)
    v_sup = np.ones(n_states, dtype=np.float32)

    successor_states_matrix = np.zeros((max_successors, n_states), dtype=np.int64)
    probabilities = np.zeros((max_successors, n_states, n_actions), dtype=np.float32)

    print("Building successor state matrix and probabilities ...")

    for s in range(n_states):
        safe_unsafe_flag = False
        if cost_fn(label_fn(s)):
            v_inf[s] = 1.0
            safe_unsafe_flag = True 
        if s in env.safe_end_component:
            v_sup[s] = 0.0
            safe_unsafe_flag = True 

        if safe_unsafe_flag:  # absorbing
            successor_states_matrix[:, s] = s
            probabilities[:, s, :] = 0.0
            probabilities[0, s, :] = 1.0
        else:
            if use_transition_matrix:
                mask = np.any(probs[:, s, :] > 0.0, axis=1)
                successors_s = np.nonzero(mask)[0]
                k = len(successors_s)

                successor_states_matrix[:k, s] = successors_s

                for a in range(n_actions):
                    probabilities[:k, s, a] = probs[successors_s, s, a]

            if use_successor_states_dict:
                successors_s = np.array(successor_states[s], dtype=np.int64)#
                k = len(successors_s)
                successor_states_matrix[:k, s] = successors_s

                for a in range(n_actions):
                    probabilities[:k, s, a] = np.array(
                        probs_dict[(s, a)],
                        dtype=np.float32
                    )

            # remaining slots already zeroed

    return v_inf, v_sup, successor_states_matrix, probabilities, max_successors

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
    return result_vertex.astype(np.float32)


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

    return result_vertex.astype(np.float32)

class ProbShieldWrapperDisc(ConstraintPersistentWrapper):

    def __init__(
        self, 
        env: gym.Env,
        theta: float = 1e-10,
        max_vi_steps: int = 1000,
        init_safety_bound: float = 0.5,
        granularity: int = 20,
    ):

        # Sanity checks
        if isinstance(env, ProbShieldWrapperDisc):
            raise RuntimeError(
                "Environment is already wrapped in ProbShieldWrapperDisc. "
                "Double-wrapping can cause undefined behaviour."
            )
        for method_name in ("has_transition_matrix", "has_successor_states_dict"):
            if not hasattr(env.unwrapped, method_name):
                raise TypeError(
                    f"Env of type {type(env.unwrapped).__name__} must define a callable "
                    f"'{method_name}()' method."
                )
        has_tm = env.unwrapped.has_transition_matrix
        has_ssd = env.unwrapped.has_successor_states_dict
        if not (has_tm or has_ssd):
            raise ValueError(
                "Environment must expose either a transition matrix or a "
                "successor-states dictionary (or both), but neither was found."
            )
        if not isinstance(env, ConstraintPersistentWrapper):
            raise TypeError(
                "ProbShieldWrapperDisc expects `env` to be an instance of "
                f"ConstraintPersistentWrapper, got {type(env).__name__}."
            )
        if not hasattr(env, "cost_fn"):
            raise AttributeError(
                "Environment must define a `cost_fn` attribute."
            )
        if env.cost_fn is None:
            raise ValueError(
                "Environment attribute `cost_fn` must not be None."
            )
        if not hasattr(env, "label_fn"):
            raise AttributeError(
                "Environment must define a `label_fn` attribute."
            )
        if env.label_fn is None:
            raise ValueError(
                "Environment attribute `label_fn` must not be None."
            )
        if not hasattr(env, "action_space"):
            raise AttributeError("Environment must have an `action_space` attribute.")
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                "ProbShieldWrapperDisc only supports environments with a "
                f"Discrete action space, got: {type(env.action_space).__name__}."
            )
        if not isinstance(env.observation_space, spaces.Discrete): # TODO: or has a safety abstraction
            raise TypeError(
                "ProbShieldWrapperDisc only supports environments with a "
                f"Discrete observation space or discrete safety abstracyion, got: {type(env.observation_space).__name__}"
            )
        if not hasattr(env.unwrapped, "safe_end_component"):
            raise AttributeError(
                "Environment must define a `safe_end_component` attribute "
                "containing terminal safe states."
            )
        sec = env.unwrapped.safe_end_component
        try:
            iter(sec)
        except TypeError:
            raise TypeError(
                "`env.safe_end_component` must be an iterable (e.g. list, set, tuple) "
                f"but got type {type(sec).__name__}."
            )
        if len(sec) == 0:
            raise ValueError(
                "`env.safe_end_component` must be a non-empty iterable. "
                "The shielding procedure requires at least one absorbing safe state."
            )

        super().__init__(env)

        self.theta = theta
        self.max_vi_steps = max_vi_steps
        self.init_safety_bound = init_safety_bound
        self.granularity = granularity
        self._box_dtype = np.float32

        v_inf, v_sup, self.successor_states_matrix, self.probabilities, self.max_successors = \
            build_successor_state_matrix_and_probabilities(
                self.env.unwrapped, self.env.label_fn, self.env.cost_fn,
            )

        v_inf, v_sup, _, _, = fast_value_iteration(
            v_inf, v_sup, self.successor_states_matrix, self.probabilities, self.theta, self.max_vi_steps
        )

        self.safety_lb = v_sup

        self._orig_obs_space = self.env.observation_space
        self._orig_act_space = self.env.action_space

        assert isinstance(self._orig_act_space, spaces.Discrete)

        self.observation_space = self._make_augmented_obs_space(self._orig_obs_space)
        self.action_space = self._make_augmented_act_space(self._orig_act_space)

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
                f"LTLSafetyEnv does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, Box, Dict."
            )
        return aug

    def _make_augmented_act_space(self, orig: spaces.Discrete) -> spaces.MultiDiscrete:
        n_actions = orig.n
        aug = spaces.MultiDiscrete([n_actions]*2+[self.granularity+1]*self.max_successors)
        return aug

    def _augment_obs(self, obs: Any) -> Dict[str, Any]:
        # No one-hotting here, we let the algorithm decide how they want to handle discrete observations
        if isinstance(self._orig_obs_space, (spaces.Discrete, spaces.Box)):
            return dict({
                "orig_obs": obs,
                "safety_bound": self._current_safety_bound
            })
        elif isinstance(self._orig_obs_space, (spaces.Dict)):
            obs = dict(obs)
            obs["safety_bound"] = self._current_safety_bound
            return obs
        else:
            raise TypeError(
                f"LTLSafetyEnv does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, Box, Dict."
            )

    def _one_hot(self, s: int) -> np.ndarray:
        raise NotImplementedError

    def _project_act(self, action: Any, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:

        # betas chosen by the agent
        betas = action[2:] / self.granularity
        obs_successor_states = self.successor_states_matrix[:, self._current_obs]
        obs_safety_lb = self.safety_lb[obs_successor_states]
        safety_bounds = obs_safety_lb + (1.0 - obs_safety_lb) * betas
        proj_safety_bounds = safety_bounds.copy()

        # Project the infeasible or unnecessarily safe alphas
        all_out_and_nonzero_safety_bounds = True
        all_in_and_nonzero_safety_bounds = True 

        probs_curr = self.probabilities[:, self._current_obs, :]  

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
            diag_intersections = np.eye(n_actions, dtype=np.float32)[diag_indices]
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
            edge_intersections = np.zeros((n_valid, n_actions), dtype=np.float32)

            rows = np.arange(n_valid)
            edge_intersections[rows, i_valid] = alpha_i
            edge_intersections[rows, j_valid] = alpha_j

            intersections_list.append(edge_intersections)

        # Stack everything
        if intersections_list:
            intersections = np.vstack(intersections_list)
        else:
            intersections = np.empty((0, n_actions), dtype=np.float32)

        i = action[0]
        j = action[1]

        # Compute the vertex corresponding to the action
        safe_vertex = np.zeros(n_actions, dtype=np.float32)

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
                    safe_vertex = np.zeros(n_actions, dtype=np.float32)
                    safe_vertex[i] = diff_act_j / (diff_act_j - diff_act_i)
                    safe_vertex[j] = diff_act_i / (diff_act_i - diff_act_j)

        return safe_vertex, proj_safety_bounds
                    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_safety_bound = self.init_safety_bound
        self._current_obs = int(obs)
        return self._augment_obs(obs), info

    def step(self, action):
        safe_vertex, proj_safety_bounds = self._project_act(action)

        orig_action = self.np_random.choice(len(safe_vertex), p=safe_vertex)
        orig_obs, reward, terminated, truncated, info = self.env.step(orig_action)

        succ_list = self.successor_states_matrix[:, self._current_obs]
        matches = np.where(succ_list == orig_obs)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"Something went wrong! Next state {orig_obs} is not in successor list for state {self._current_obs}."
            )
        next_obs_idx = int(matches[0])
        
        self._current_safety_bound = proj_safety_bounds[next_obs_idx]
        self._current_obs = int(orig_obs)

        return self._augment_obs(orig_obs), reward, terminated, truncated, info




    