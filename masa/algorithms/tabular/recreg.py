from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, Callable, Dict, Tuple, List
from masa.common.metrics import TrainLogger
from masa.algorithms.tabular.q_learning import QL
from masa.common.ltl import DFACostFn, DFA
import masa.common.pctl as pctl
from collections import defaultdict
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from tqdm import tqdm
from masa.prob_shield.helpers import build_successor_states_matrix
from masa.prob_shield.eventual_discounted_vi import ev_value_iteration
from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.wrappers import is_wrapped, get_wrapped

def build_pctl_formula(
    env: gym.Env,
    bound_param: int,
    prob: float,
    label_fn: Optional[LabelFn] = None,
    cost_fn: Optional[CostFn] = None,
) -> pctl.BoundedPCTLFormula:

    if not hasattr(env, "cost_fn") and cost_fn is None and env.cost_fn is None:
        raise AttributeError("Environment must define a `cost_fn` attribute.")
    cost_fn = cost_fn if cost_fn else env.cost_fn

    if not hasattr(env, "label_fn") and label_fn is None and env.label_fn is None:
        raise AttributeError("Environment must define a `label_fn` attribute.")
    label_fn = label_fn if label_fn else env.label_fn

    if not hasattr(env, "observation_space"):
        raise AttributeError("Environment must have an `observation_space` attribute.")
    if isinstance(env.observation_space, spaces.Discrete):
        n_states = env.observation_space.n
    else:
        raise NotImplementedError(f"Unexpected observation_space {type(env.observation_space).__name__}")

    unsafe_dict = defaultdict(set)

    for s in range(n_states):
        if cost_fn(label_fn(s)):
            unsafe_dict[s] = {"unsafe"}
            
    new_label_fn = lambda obs: unsafe_dict[obs]

    formula = pctl.Always(1.0-prob, bound_param, pctl.Neg(pctl.Atom("unsafe")))

    return formula, new_label_fn

def update_compact(
    succ: np.ndarray,
    counts: np.ndarray,
    succ_index: List[Dict[int, int]],
    next_free: np.ndarray,
    s: int,
    a: int,
    s_next: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[int, int]], np.ndarray]:

    m = succ_index[s]
    if s_next in m:
        k = m[s_next]
    else:
        k = int(next_free[s])
        if k >= succ.shape[0] or succ[k, s] != -1:
            succ, counts, succ_index, next_free, k = handle_overflow(
                succ, counts, succ_index, next_free, s=s, s_next=s_next
            )
        else:
            succ[k, s] = int(s_next)
            m[int(s_next)] = k
            next_free[s] = np.int32(k + 1)

    counts[k, s, a] += 1.0
    return succ, counts, succ_index, next_free

def handle_overflow(
    succ: np.ndarray,
    counts: np.ndarray,
    succ_index: List[Dict[int, int]],
    next_free: np.ndarray,
    s: int,
    s_next: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[int, int]], np.ndarray, int]:

    K, S = succ.shape
    K2, S2, A = counts.shape
    assert K == K2 and S == S2
    assert len(succ_index) == S
    assert next_free.shape == (S,)

    new_K = max(1, 2 * K)

    succ_new = -np.ones((new_K, S), dtype=succ.dtype)
    succ_new[:K, :] = succ

    counts_new = np.zeros((new_K, S, A), dtype=counts.dtype)
    counts_new[:K, :, :] = counts

    # Allocate slot for this state
    # Prefer next_free[s] if it points to an empty slot, otherwise find one.
    k_slot = int(next_free[s])
    if k_slot >= new_K or succ_new[k_slot, s] != -1:
        free = np.where(succ_new[:, s] == -1)[0]
        if free.size == 0:
            raise RuntimeError(f"handle_overflow failed: no free slot after resizing K {K}->{new_K}")
        k_slot = int(free[0])

    succ_new[k_slot, s] = int(s_next)
    succ_index[s][int(s_next)] = k_slot

    # Bump next_free pointer if we used the next_free slot
    if k_slot == int(next_free[s]):
        next_free[s] = np.int32(k_slot + 1)

    return succ_new, counts_new, succ_index, next_free, k_slot

def compact_probabilities(counts):
    return counts / counts.sum(axis=0, keepdims=True)

def q_to_boltzmann_policy(Q: np.ndarray, tau: float) -> np.ndarray:
    prefs = Q / tau
    max_prefs = np.max(prefs, axis=1, keepdims=True)
    exp_prefs = np.exp(prefs - max_prefs)
    return exp_prefs / np.sum(exp_prefs, axis=1, keepdims=True) 

def q_to_argmax_policy(Q):
    # No tie breaking (deterministic) argmax policy 
    Q = np.asarray(Q)
    policy = np.zeros(Q.shape, dtype=np.float64)
    policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1.0
    return policy

class RECREG(QL):

    def __init__(
        self,
        *args,
        task_alpha: float = 0.1,
        task_gamma: float = 0.95,
        safe_alpha: float = 0.1,
        safe_gamma: float = 0.95,
        model_free_alpha: float = 0.1,
        mode: str = "model_free",
        model_checking: str = "exact",
        samples: int = 8192,
        horizon: int = 5,
        safety_prob: float = 0.01,
        r_min: float = 0.0,
        cost_coef: float = 10.0,
        **kwargs,
    ):

        super().__init__(
            *args,
            alpha=task_alpha,
            gamma=task_gamma,
            **kwargs,
        )

        assert mode in ["exact", "model_based", "model_free"], f"Unsupported model implementation {mode}"
        assert model_checking in ["exact", "statistical", "none"], f"Unsupported model-checking type {rollout}"
        assert not model_checking == "none" or mode == "model_free", f"Must specify a model checking type when using mode: {mode}"
        assert samples > 0, "Number of samples must be greater than 0"

        self.safe_alpha = safe_alpha
        self.safe_gamma = safe_gamma
        self.model_free_alpha = model_free_alpha
        self.mode = mode
        self.model_checking = model_checking
        self.samples = samples
        self.horizon = horizon
        self.safety_prob = safety_prob
        self.r_min = r_min
        self.cost_coef = cost_coef

        self._setup_models()

    def _setup_models(self):

        if self.mode == "exact":

            successor_states_matrix, probabilities, _, label_fn, cost_fn, _ = build_successor_states_matrix(self.env)

            # Backup policy Q values
            self.B = ev_value_iteration(
                successor_states_matrix, probabilities, label_fn, cost_fn, gamma=self.safe_gamma
            )

            self.backup_policy = self._backup_policy_matrix(deterministic=False)

            formula, pctl_label_fn = build_pctl_formula(
                self.env, self.horizon, self.safety_prob, label_fn=label_fn, cost_fn=cost_fn,
            )

            self.model_checker = pctl.ExactModelChecker(
                formula, 
                pctl_label_fn, 
                {"unsafe"}, # atomic predicates
                successor_states=successor_states_matrix,
                probabilities=probabilities,
            )

            # Satisfaction relation: the probability of being unsafe in H timesteps,
            # computed for each backup the shield may execute (keyed by _deterministic_backup).
            self._S_exact = {}
            for det in sorted({self._deterministic_backup(False), self._deterministic_backup(True)}):
                self._S_exact[det] = 1.0 - self.model_checker.check_state_action(
                    self.key,
                    self._backup_policy_matrix(deterministic=det),
                )[np.newaxis, :]
            self.S = self._S_exact[self._deterministic_backup(False)]  # used by the training mask

            self.unsafe_mask = np.array(
                [bool(cost_fn(label_fn(s))) for s in range(self.n_states)],
                dtype=bool,
            )

        elif self.mode in ["model_based", "model_free"]:

            # Handle LTLSafetyEnv case
            if is_wrapped(self.env, LTLSafetyEnv):
                from masa.common.constraints.ltl_safety import create_product_label_fn, product_cost_fn

                ltl_safety_env: LTLSafetyEnv | None = get_wrapped(self.env, LTLSafetyEnv)
                dfa = self.env._constraint.get_dfa()
                if isinstance(ltl_safety_env._orig_obs_space, spaces.Discrete):
                    n_states = ltl_safety_env._orig_obs_space.n
                else:
                    raise NotImplementedError(f"Unexpected observation space: {ltl_safety_env._orig_obs_space}")

                label_fn = create_product_label_fn(n_states, dfa)
                cost_fn = product_cost_fn
            else:
                if not hasattr(self.env, "label_fn") and self.env.label_fn is None:
                    raise AttributeError("Environment must define a `label_fn` attribute.")
                label_fn = self.env.label_fn
                if not hasattr(self.env, "cost_fn") and self.env.cost_fn is None:
                    raise AttributeError("Environment must define a `cost_fn` attribute.")
                cost_fn = self.env.cost_fn

            self.unsafe_mask = np.array(
                [bool(cost_fn(label_fn(s))) for s in range(self.n_states)],
                dtype=bool,
            )

            if self.mode == "model_based":

                # Backup policy Q values
                self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

                formula, pctl_label_fn = build_pctl_formula(
                    self.env, self.horizon, self.safety_prob, label_fn=label_fn, cost_fn=cost_fn
                )

                max_succesors = 5 # modify this if need be

                # Successor states and counts to estimate the transition kernel
                self.successors = -np.ones((max_succesors, self.n_states), dtype=np.int64)
                self.successors[0, :] = np.arange(self.n_states, dtype=np.int64) 
                self.counts = np.zeros((max_succesors, self.n_states, self.n_actions), dtype=np.float32)
                self.counts[0, :, :] = 1e-3

                self.successor_index = [{int(s): 0} for s in range(self.n_states)]
                self.next_free = np.ones(self.n_states, dtype=np.int32)

                if self.model_checking == "exact":
                    model_checker_cls = pctl.ExactModelChecker
                elif self.model_checking == "statistical":
                    model_checker_cls = pctl.StatisticalModelChecker
                else:
                    raise NotImplementedError(f"Unexpected model checker class: {self.model_checking}")

                self.model_checker = model_checker_cls(
                    formula, 
                    pctl_label_fn, 
                    {"unsafe"}, # atomic predicates
                    successor_states=self.successors,
                    probabilities=compact_probabilities(self.counts)
                )

            if self.mode == "model_free":
                # Backup policy Q values
                self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
                # Satisfaction relation: the probability of being unsafe in H timesteps
                self.S = np.zeros((self.horizon, self.n_states, self.n_actions), dtype=np.float32)

        else:
            raise NotImplementedError(f"Unexpected mode: {self.mode}")

    def _deterministic_backup(self, deterministic: bool) -> bool:
        # act_overrid executes argmax(B) when deterministic or under epsilon-greedy,
        # and a Boltzmann(B, 0.01) sample otherwise.
        return bool(deterministic or self.exploration == "epsilon_greedy")

    def _backup_policy_matrix(self, deterministic: bool) -> np.ndarray:
        if self.exploration not in ("boltzmann", "epsilon_greedy"):
            raise NotImplementedError(f"Unexpected exploration: {self.exploration}")
        if self._deterministic_backup(deterministic):
            return q_to_argmax_policy(self.B)
        return q_to_boltzmann_policy(self.B, 0.01)

    def _backup_row(self, s: int, deterministic: bool) -> np.ndarray:
        q = np.asarray(self.B[s], dtype=np.float64)
        if self._deterministic_backup(deterministic):
            p = np.zeros_like(q)
            p[int(np.argmax(q))] = 1.0
            return p
        z = np.exp((q - q.max()) / 0.01)
        return z / z.sum()

    def _risk_table(self, deterministic: bool) -> np.ndarray:
        # H-step risk q_H(s, a) of taking a and then following the executed backup.
        if self.mode == "exact":
            return self._S_exact[self._deterministic_backup(deterministic)]
        return self.S  # model_based: recomputed in act_override; model_free: learned with argmax backup

    def _backup_tail_risk_state(self, s: int, deterministic: bool = False) -> float:
        # rho(s) = sum_a pi_backup(a|s) q_H(s, a): risk of handing control to the backup at s.
        S = self._risk_table(deterministic)
        return float(np.dot(self._backup_row(s, deterministic), S[0, s]))

    def _immediate_risk(self, s: int, a: int) -> float:
        """p(s, a) = P(next state is unsafe | s, a), for the executed action a."""
        if self.mode == "model_free":
            return float(self.S[-1, s, a])  # base case of the recursion: one-step risk
        succ = np.asarray(self.model_checker.successor_states)[:, s]   # (K,), -1 = empty slot
        probs = np.asarray(self.model_checker.probabilities)[:, s, a]  # (K,)
        valid = succ >= 0
        return float(np.sum(probs[valid] * self.unsafe_mask[succ[valid]]))

    def _update_next_budget(
        self, s: int, a: int, next_obs: int, delta: float, override: bool = False, deterministic: bool = False
    ) -> float:
        """Recovery-aware residual budget (Theorem 1b, with the 1/(1-p) factor so its a true Martingale)."""
        if self.unsafe_mask[next_obs]:
            return 0.0

        if override:
            tail = self._backup_tail_risk_state(s, deterministic)
        else:
            tail = float(self._risk_table(deterministic)[0, s, a])
        rho_next = self._backup_tail_risk_state(next_obs, deterministic)

        slack = delta - tail
        if slack < -1e-9 and not self.unsafe_mask[s]:
            # Invariant b_t >= rho(s_t) broken (inaccurate estimates or backup unsafe from s_0).
            self._budget_deficits += 1

        p = min(max(self._immediate_risk(s, a), 0.0), 1.0)
        scale = 1.0 / (1.0 - p) if p < 1.0 - 1e-6 else 1.0
        return float(np.clip(rho_next + max(slack, 0.0) * scale, 0.0, 1.0))

    def _check_initial_budget(self, obs: int, deterministic: bool) -> None:
        if self._backup_tail_risk_state(obs, deterministic) > self.safety_prob + 1e-9:
            self._infeasible_starts += 1

    def train(self, *args, **kwargs):
        self._overrides = []
        self._step_wise_prob = self.safety_prob
        self._budget_deficits = 0
        self._infeasible_starts = 0
        super().train(*args, **kwargs)

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, safe_action, action, override, reward, cost, violation, next_state, terminal, info) in self.buffer:

            # 0-return update for the risky action if overridden (all modes)
            if override:
                targ = self.r_min / (1 - self.gamma)
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * targ

            # Generate (real + counterfactual) experiences
            if hasattr(self.env, "cost_fn") and isinstance(self.env.cost_fn, DFACostFn) and self.use_counter_factual_experiences:
                cf_exp = self.generate_counter_factuals(
                    state, safe_action, reward, next_state, terminal, info, getattr(self.env, "cost_fn", None)
                )
            else:
                # no DFA: treat the actual transition as the only "cf" experience
                cf_exp = [(state, safe_action, reward, cost, violation, next_state, terminal)]

            for exp in cf_exp:
                cf_state, cf_act, cf_rew, cf_cost, cf_viol, cf_next_state, cf_term = exp

                # --- model-based: update pseudo-counts with cf exp ---
                if self.mode == "model_based":
                    self.successors, self.counts, self.successor_index, self.next_free = update_compact(
                        self.successors, self.counts, self.successor_index, self.next_free, s=cf_state,  a=cf_act, s_next=cf_next_state
                    )

                # --- model-free: update satisfaction relation S with cf exp ---
                if self.mode == "model_free":
                    next_b = self.B[cf_next_state]
                    for t in reversed(range(self.horizon)):
                        if t == self.horizon - 1:
                            next_S = min(1.0, float(cf_viol))
                        else:
                            next_S = min(
                                1.0,
                                float(cf_viol)
                                + (1 - cf_term) * self.S[t + 1, cf_next_state, np.argmax(next_b)],
                            )
                        self.S[t, cf_state, cf_act] = (
                            (1 - self.model_free_alpha) * self.S[t, cf_state, cf_act] + self.model_free_alpha * next_S
                        )

                # --- update backup policy B with cf exp ---
                if self.mode in ["model_free", "model_based"]:
                    cf_pen = -float(cf_cost) * self.cost_coef
                    cf_gamma = self.safe_gamma if cf_viol else 1.0
                    next_b = self.B[cf_next_state]
                    target_b = cf_pen + (1 - cf_term) * cf_gamma * np.max(next_b)
                    self.B[cf_state, cf_act] = (1 - self.safe_alpha) * self.B[cf_state, cf_act]  \
                                                + self.safe_alpha * target_b

                # --- update task policy Q with cf exp ---
                if self.mode in ["model_free", "exact", "model_based"]:
                    next_q = self.Q[cf_next_state]
                    if self.S[0, cf_state, cf_act] <= self.safety_prob:
                        target_q = cf_rew + (1 - cf_term) * self.gamma * np.max(next_q)
                    else:
                        target_q = self.r_min / (1 - self.gamma)
                    self.Q[cf_state, cf_act] = (1 - self.alpha) * self.Q[cf_state, cf_act] \
                                                + self.alpha * target_q

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"safe_alpha": self.safe_alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})
            

    def rollout(self, step: int, logger: Optional[TrainLogger] = None):

        self.key, subkey = jr.split(self.key)
        safe_act, act, override = self.act_override(subkey, self._last_obs, self._step_wise_prob)
        next_obs, reward, terminated, truncated, info = self.env.step(safe_act)

        self._overrides.append(override)

        cost = info["constraint"]["step"].get("cost", 0.0)
        violation = info["constraint"]["step"].get("violation", False)

        self._step_wise_prob = self._update_next_budget(
                self._last_obs, safe_act, next_obs, self._step_wise_prob, override=bool(override)
            )

        self.buffer.append(
            (self._last_obs, safe_act, act, override, reward, cost, violation, next_obs, terminated, info)
        )

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
            self._step_wise_prob = self.safety_prob
            self._check_initial_budget(self._last_obs, deterministic=False)
            info["constraint"]["episode"].update({
                "override_rate": float(np.mean(self._overrides)),
                "budget_deficits": int(self._budget_deficits),
                "infeasible_starts": int(self._infeasible_starts),
            })
            self._overrides.clear()
        else:
            self._last_obs = next_obs

        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)

        if logger:
            logger.add("train/rollout", info)

    def eval(self, num_episodes: int, seed: Optional[int] = None, logger: Optional[TrainLogger] = None) -> List[float]:
        eval_env = self._get_eval_env()
        base = 0 if self.seed is None else int(self.seed)
        eval_seed = base + 10_000 if seed is None else int(seed) + 10_000
        eval_key = jr.PRNGKey(eval_seed)
        returns = []
        overrides = []

        with tqdm(
            total=num_episodes,
            desc="evaluation",
            position=1,
            leave=False,
            dynamic_ncols=True,
            colour="yellow"
        ) as pbar:

            for ep in range(num_episodes):
                obs, info = eval_env.reset(seed=eval_seed + ep)
                done = False
                ret = 0.0
                self._eval_step_wise_prob = self.safety_prob
                saved = (self._budget_deficits, self._infeasible_starts) if hasattr(self, "_budget_deficits") else (0, 0)
                self._budget_deficits, self._infeasible_starts = 0, 0
                self._check_initial_budget(obs, deterministic=True)
                while not done:
                    eval_key, subkey = jr.split(eval_key)
                    action, _, override = self.act_override(subkey, obs, self._eval_step_wise_prob, deterministic=True)
                    next_obs, rew, terminated, truncated, info = eval_env.step(action)

                    self._eval_step_wise_prob = self._update_next_budget(
                            obs, action, next_obs, self._eval_step_wise_prob,
                            override=bool(override), deterministic=True,
                        )
                    
                    obs = next_obs
                    overrides.append(override)

                    ret += float(rew)
                    done = terminated or truncated

                info["constraint"]["episode"].update({
                    "override_rate": float(np.mean(overrides)),
                    "budget_deficits": int(self._budget_deficits),
                    "infeasible_starts": int(self._infeasible_starts),
                })
                self._budget_deficits, self._infeasible_starts = saved
                overrides.clear()

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add("eval/rollout", info)

        return returns

    def act_override(self, key, obs, step_wise_prob, deterministic=False):
        key, key1, key2, key3 = jr.split(key, 4)

        safe_act = act = self._act(key1, obs, self.Q, deterministic, self.boltzmann_temp, self._epsilon)

        if self.mode == "model_based":
            self.model_checker.update_kernel(
                successor_states=self.successors, probabilities=compact_probabilities(self.counts)
            )

            self.backup_policy = self._backup_policy_matrix(deterministic)

            if self.model_checking == "exact":
                self.S = 1.0 - self.model_checker.check_state_action(
                    None, self.backup_policy
                )[np.newaxis, :]
            else:
                raise NotImplementedError(f"Model checker class: {self.model_checking} currently not supported")

            override = self.S[0, obs, act] > step_wise_prob
        
        elif self.mode in ["model_free", "exact"]:
            override = self._risk_table(deterministic)[0, obs, act] > step_wise_prob
        else:
            raise NotImplementedError(f"Unexpected mode: {self.mode}")

        if override:
            # Use boltzmann temperature=0.01 or epsilon=0.001 for backup policy
            if self.exploration == "boltzmann":
                safe_act = self._act(key3, obs, self.B, deterministic, 0.01, 0.001)
            elif self.exploration == "epsilon_greedy":
                safe_act = self._act(key3, obs, self.B, True, 0.01, 0.001)
            else:
                raise NotImplementedError(f"Unexpected exploration: {self.exploration}")
    
        return self.prepare_act(safe_act), self.prepare_act(act), override

    def act(self, key, obs, deterministic=False):
        # No episode context here: use the full budget as a fixed per-step threshold.
        action, _, _ = self.act_override(key, obs, self.safety_prob, deterministic=deterministic)
        return action

    def _act(self, key, obs, Q, deterministic, boltzmann_temp, epsilon):
        if deterministic:
            return self.select_action(Q[obs])
        else:
            return self.sample_action(
                key, jnp.asarray(Q[obs], dtype=jnp.float32), boltzmann_temp, epsilon, exploration=self.exploration
            )


class RECREG_EXACT(RECREG):

    def __init__(
        self,
        *args,
        mode: str = "exact",
        model_checking: str = "exact",
        **kwargs,
    ):

        super().__init__(
            *args,
            mode=mode,
            model_checking=model_checking,
            **kwargs,
        )


class RECREG_MODEL_BASED(RECREG):

    def __init__(
        self,
        *args,
        mode: str = "model_based",
        model_checking: str = "exact",
        **kwargs,
    ):

        super().__init__(
            *args,
            mode=mode,
            model_checking=model_checking,
            **kwargs,
        )