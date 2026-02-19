import warnings
import gymnasium as gym
from gymnasium import spaces
from masa.common.wrappers import ConstraintPersistentWrapper, is_wrapped, get_wrapped
from masa.common.constraints.ltl_safety import LTLSafetyEnv
from typing import Union, Any, Tuple, Dict, List, Optional, Callable
from masa.common.constraints.base import CostFn
from masa.common.label_fn import LabelFn
from masa.prob_shield.helpers import build_successor_states_matrix
from masa.prob_shield.interval_bound_vi import interval_bound_value_iteration
import numpy as np
import copy

def _egalitarian_constant_margin_fill(beta: np.ndarray, w: np.ndarray, q: float, eps: float) -> np.ndarray:
    """
    Find W in [beta,1]^k s.t. w·W = q, maximizing the minimum additive margin (W-beta),
    i.e., raise all non-saturated coordinates equally, freezing any that hit 1.
    Assumes w >= 0, sum(w)=1 (we renormalize outside).
    """
    W = beta.astype(np.float64).copy()
    w = w.astype(np.float64)

    # target increase in expectation over beta
    lower = float(np.dot(w, W))
    delta = float(q - lower)
    if delta <= eps:
        return W

    # only indices that matter for the constraint (w_i > 0) and can still increase
    active = np.where((w > eps) & (W < 1.0 - eps))[0]

    while delta > eps and active.size > 0:
        sum_w = float(w[active].sum())
        if sum_w <= eps:
            break

        # max uniform raise before someone saturates
        min_slack = float(np.min(1.0 - W[active]))

        # raising all active by x increases w·W by x * sum_w
        x_needed = delta / sum_w
        x = min(x_needed, min_slack)

        W[active] += x
        delta -= x * sum_w

        # freeze saturated
        active = active[(1.0 - W[active]) > eps]

    return np.clip(W, beta, 1.0)

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
        if is_wrapped(self.env, ProbShieldWrapperBase):
            raise RuntimeError(
                "Environment is already wrapped in ProbShieldWrapperBase. "
                "Double-wrapping can cause undefined behaviour."
            )
        if not isinstance(self.env, ConstraintPersistentWrapper):
            raise TypeError(
                "ProbShieldWrapperBase expects `env` to be an instance of "
                f"ConstraintPersistentWrapper, got {type(env).__name__}."
            )
        if not isinstance(self.env.observation_space, spaces.Discrete) and safety_abstraction is None:
            raise TypeError(
                "ProbShieldWrapperBase only supports environments with a "
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
            ltl_safety_env = get_wrapped(env, LTLSafetyEnv)
            n_states = ltl_safety_env._orig_obs_space.n
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
            aug = spaces.Dict(aug)
        else:
            raise TypeError(
                f"ProbShieldWrapperBase does not support observation space type {type(orig).__name__}. "
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
                f"ProbShieldWrapperBase does not support observation space type {type(orig).__name__}. "
                "Supported: Discrete, Box, Dict."
            )

    def _one_hot(self, s: int) -> np.ndarray:
        raise NotImplementedError

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        raise NotImplementedError 

    def _safe_action_from_beta(self, probs_curr: np.ndarray, beta: np.ndarray) -> int:
        exp_beta = probs_curr.T @ beta  # (n_actions,)
        return int(np.argmin(exp_beta))

    def _project_act(
        self, i: int, j: int, t: float, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_actions = int(self._orig_act_space.n)

        # --- current transition table slice (valid successors only) ---
        succ_list = self.successor_states_matrix[:, self._current_obs]
        k = int(np.sum(succ_list != -1))

        if k <= 0:
            print("Warning: no valid successors from current state; returning safe default action.")
            # no successors; just execute something reasonable
            pi = np.zeros(n_actions, dtype=np.float64)
            a = int(np.clip(i, 0, n_actions - 1))
            pi[a] = 1.0
            proj_safety_bounds_full = np.ones(self.max_successors, dtype=np.float64)
            return pi, proj_safety_bounds_full

        succ_states = succ_list[:k]
        probs_curr = self.probabilities[:k, self._current_obs, :].astype(np.float64)  # (k, A)

        beta = self.safety_lb[succ_states].astype(np.float64)  # (k,)
        q = float(np.clip(float(self._current_safety_bound), 0.0, 1.0))
        #print("Projecting action with q =", q, "and beta =", beta)
        # clamp inputs
        i = int(np.clip(i, 0, n_actions - 1))
        j = int(np.clip(j, 0, n_actions - 1))
        t = float(np.clip(t, 0.0, 1.0))

        # --- safe action wrt beta ---
        a_safe = self._safe_action_from_beta(probs_curr, beta)
        pi_safe = np.zeros(n_actions, dtype=np.float64)
        pi_safe[a_safe] = 1.0

        # --- beta-cost per pure action: c[a] = E[beta | a] ---
        c = (probs_curr.T @ beta).astype(np.float64)  # (A,)
        c_safe = float(c[a_safe])

        # Ensure feasibility baseline numerically
        if c_safe > q + eps:
            warnings.warn(" safe action is not actually safe under current q; adjusting q to ensure feasibility.")
            q = c_safe

        # ------------------------------------------------------------
        # Step 1: build pi_base
        # ------------------------------------------------------------
        pi_base = np.zeros(n_actions, dtype=np.float64)

        # EDGE MODE (old action selection logic)
        if i == j:
            ci = float(c[i])
            if ci <= q + eps:
                pi_base[i] = 1.0
            else:
                denom = (ci - c_safe)
                if abs(denom) <= 1e-18:
                    pi_base[a_safe] = 1.0
                else:
                    tau = (q - c_safe) / denom
                    tau = float(np.clip(tau, 0.0, 1.0))
                    pi_base[a_safe] = 1.0 - tau
                    pi_base[i] = tau
        else:
            ci = float(c[i])
            cj = float(c[j])

            both_feasible = (ci <= q + eps) and (cj <= q + eps)
            one_feasible = ((ci <= q + eps) != (cj <= q + eps))

            if both_feasible:
                # keep riskier feasible endpoint (your preference)
                if ci >= cj:
                    pi_base[i] = 1.0
                else:
                    pi_base[j] = 1.0

            elif one_feasible:
                # unique boundary point on the edge where cost=q
                denom = (cj - ci)
                if abs(denom) <= 1e-18:
                    # costs equal; pick the feasible endpoint
                    if ci <= q + eps:
                        pi_base[i] = 1.0
                    else:
                        pi_base[j] = 1.0
                else:
                    lam = (q - ci) / denom
                    lam = float(np.clip(lam, 0.0, 1.0))
                    pi_base[i] = 1.0 - lam
                    pi_base[j] = lam

            else:
                # both infeasible: pick less risky endpoint then mix with safe to hit cost=q
                k_idx = i if ci <= cj else j
                ck = float(c[k_idx])

                denom = (ck - c_safe)
                if abs(denom) <= 1e-18:
                    pi_base[a_safe] = 1.0
                else:
                    tau = (q - c_safe) / denom
                    tau = float(np.clip(tau, 0.0, 1.0))
                    pi_base[a_safe] = 1.0 - tau
                    pi_base[k_idx] = tau

        # final pi = mix toward safe using t
        pi = (1.0 - t) * pi_base + t * pi_safe
        pi = np.clip(pi, 0.0, 1.0)
        ssum = float(pi.sum())
        if ssum !=1:
            warnings.warn(f"Sum of pi is {ssum}, which is not 1. Renormalizing.")   
        if ssum <= 0.0 or not np.isfinite(ssum):
            pi = pi_safe.copy()
        else:
            pi /= ssum

        # ------------------------------------------------------------
        # Step 2: compute W (anti-collapse default)
        # ------------------------------------------------------------
        w = probs_curr @ pi
        w_sum = float(w.sum())
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            pi = pi_safe.copy()
            w = probs_curr @ pi
            w_sum = float(w.sum())

        if abs(w_sum - 1.0) > 1e-10:
            w = w / w_sum

        lower = float(np.dot(w, beta))
        if q <= lower + eps:
            W = beta.copy()
        else:
            W = _egalitarian_constant_margin_fill(beta, w, q, eps)

        proj_safety_bounds_full = np.ones(self.max_successors, dtype=np.float64)
        proj_safety_bounds_full[:k] = W
        return pi.astype(np.float64), proj_safety_bounds_full

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_safety_bound = float(self.init_safety_bound)
        self._current_obs = self._abstraction(obs)
        return self._augment_obs(obs), info

    def step(self, action):
        i, j, t = self._parse_act(action)
        pi, proj_safety_bounds_full = self._project_act(i, j, t)

        pi = np.clip(pi, 0.0, 1.0)
        s = float(pi.sum())
        if s <= 0.0:
            pi = np.zeros_like(pi)
            pi[0] = 1.0
        else:
            pi = pi / s

        orig_action = self.np_random.choice(len(pi), p=pi)
        orig_obs, reward, terminated, truncated, info = self.env.step(orig_action)

        abstr_obs = self._abstraction(orig_obs)

        succ_list = self.successor_states_matrix[:, self._current_obs]
        matches = np.where(succ_list == abstr_obs)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"Something went wrong! Next abstract state {abstr_obs} is not in successor list for current abstract state {self._current_obs}."
            )
        next_obs_idx = int(matches[0])
        
        self._current_safety_bound = proj_safety_bounds_full[next_obs_idx]
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
        aug = spaces.MultiDiscrete([n_actions]*2+[self.granularity+1])
        return aug

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        i, j = action[0], action[1]
        mix = action[2] / self.granularity
        return i, j, mix

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
            "box": spaces.Box(low=0, high=1, shape=(1,), dtype=self._box_dtype)
        })
        return aug

    def _parse_act(self, action: np.ndarray) -> Tuple[int, int, np.ndarray]:
        i, j = int(action[0].item()), int(action[1].item())
        mix = float(action[2].item())
        return i, j, mix

    

    




    