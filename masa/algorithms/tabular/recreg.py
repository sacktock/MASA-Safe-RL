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
from masa.common.constraints import LTLSafetyEnv
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
    n_states, n_actions = Q.shape
    max_vals = np.max(Q, axis=1, keepdims=True)
    greedy_mask = (Q == max_vals).astype(float)
    greedy_policy = greedy_mask / greedy_mask.sum(axis=1, keepdims=True)
    return greedy_policy

class RECREG(QL):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        task_alpha: float = 0.1,
        task_gamma: float = 0.9,
        safe_alpha: float = 0.1,
        safe_gamma: float = 0.9,
        mode: str = "model_based",
        model_checking: str = "statistical",
        samples: int = 512,
        horizon: int = 10,
        step_wise_prob: float = 0.99,
        exploration: str = "boltzmann",
        boltzmann_temp: float = 0.05,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: str = "linear",
        epsilon_decay_frames: int = 10000, 
    ):

        super().__init__(
            env, 
            tensorboard_logdir=tensorboard_logdir,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            env_fn=env_fn,
            eval_env=eval_env,
            alpha=task_alpha,
            gamma=task_gamma,
            exploration=exploration,
            boltzmann_temp=boltzmann_temp,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_decay_frames=epsilon_decay_frames,
        )

        assert mode in ["exact", "model_based", "model_free"], f"Unsupported model implementation {mode}"
        assert model_checking in ["exact", "statistical", "none"], f"Unsupported model-checking type {rollout}"
        assert not model_checking == "none" or mode == "model_free", f"Must specify a model checking type when using mode: {mode}"
        assert samples > 0, "Number of samples must be greater than 0"

        self.safe_alpha = safe_alpha
        self.safe_gamma = safe_gamma
        self.mode = mode
        self.model_checking = model_checking
        self.samples = samples
        self.horizon = horizon
        self.step_wise_prob = step_wise_prob

        self._setup_models()

    def _setup_models(self):

        if self.mode == "exact":

            successor_states_matrix, probabilities, _, label_fn, cost_fn, _ = build_successor_states_matrix(self.env)

            # Backup policy Q values
            self.B = ev_value_iteration(
                successor_states_matrix, probabilities, label_fn, cost_fn, gamma=self.safe_gamma
            )

            if self.exploration == "boltzmann":
                self.backup_policy = q_to_boltzmann_policy(self.B, self.boltzmann_temp)
            elif self.exploration == "epsilon_greedy":
                self.backup_policy = q_to_argmax_policy(self.B)
            else:
                raise NotImplementedError(f"Unexpected exploration: {self.exploration}")

            formula, pctl_label_fn = build_pctl_formula(
                self.env, self.horizon, self.step_wise_prob, label_fn=label_fn, cost_fn=cost_fn,
            )

            self.model_checker = pctl.ExactModelChecker(
                formula, 
                pctl_label_fn, 
                {"unsafe"}, # atomic predicates
                successor_states=successor_states_matrix,
                probabilities=probabilities,
            )

            # Satisfaction relation: the probability of being unsafe in H timesteps
            self.S = 1.0 - self.model_checker.check_state_action(
                self.key,
                self.backup_policy,
            )[np.newaxis, :]

        elif self.mode == "model_based":

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

            # Backup policy Q values
            self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

            formula, pctl_label_fn = build_pctl_formula(
                self.env, self.horizon, self.step_wise_prob, label_fn=label_fn, cost_fn=cost_fn
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

        elif self.mode == "model_free":

            # Backup policy Q values
            self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
            # Satisfaction relation: the probability of being unsafe in H timesteps
            self.S = np.zeros((self.horizon, self.n_states, self.n_actions), dtype=np.float32)

        else:
            raise NotImplementedError(f"Unexpected mode: {self.mode}")

    def train(self, *args, **kwargs):
        self._overrides = []
        super().train(*args, **kwargs)

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the Q table with tuples of experience"""
        if len(self.buffer) == 0:
            return

        for (state, safe_action, action, override, reward, cost, violation, next_state, terminal, info) in self.buffer:

            # 0-return update for the risky action if overridden (all modes)
            if override:
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action]

            # Generate (real + counterfactual) experiences
            if hasattr(self.env, "cost_fn") and isinstance(self.env.cost_fn, DFACostFn):
                cf_exp = self.generate_counter_factuals(
                    state, safe_action, reward, next_state, terminal, info, getattr(self.env._constraint, "cost_fn", None)
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
                            (1 - self.safe_alpha) * self.S[t, cf_state, cf_act] + self.safe_alpha * next_S
                        )

                # --- update backup policy B with cf exp ---
                if self.mode in ["model_free", "model_based"]:
                    cf_pen = -float(cf_cost)
                    cf_gamma = self.safe_gamma if cf_viol else 1.0
                    next_b = self.B[cf_next_state]
                    target_b = cf_pen + (1 - cf_term) * cf_gamma * np.max(next_b)
                    self.B[cf_state, cf_act] = (1 - self.safe_alpha) * self.B[cf_state, cf_act]  \
                                                + self.safe_alpha * target_b

                # --- update task policy Q with cf exp ---
                if self.mode in ["model_free", "exact"]:
                    next_q = self.Q[cf_next_state]
                    if self.S[0, cf_state, cf_act] <= self.step_wise_prob:
                        target_q = cf_rew + (1 - cf_term) * self.gamma * np.max(next_q)
                    else:
                        target_q = 0.0
                    self.Q[cf_state, cf_act] = (1 - self.alpha) * self.Q[cf_state, cf_act] \
                                                + self.alpha * target_q

            # Q-update for model_based on the *actual* safe action (no CF gating)
            if self.mode == "model_based":
                next_q = self.Q[next_state]
                target_q = reward + (1 - terminal) * self.gamma * np.max(next_q)
                self.Q[state, safe_action] = (1 - self.alpha) * self.Q[state, safe_action] \
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
        safe_act, act, override = self.act_override(subkey, self._last_obs)
        next_obs, reward, terminated, truncated, info = self.env.step(safe_act)

        self._overrides.append(override)

        cost = info["constraint"]["step"].get("cost", 0.0)
        violation = info["constraint"]["step"].get("violation", False)

        self.buffer.append(
            (self._last_obs, safe_act, act, override, reward, cost, violation, next_obs, terminated, info)
        )

        if terminated or truncated:
            self._last_obs, _ = self.env.reset()
            info["constraint"]["episode"].update({"override_rate": float(np.mean(self._overrides))})
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
                while not done:
                    eval_key, subkey = jr.split(eval_key)
                    action, _, override = self.act_override(subkey, obs, deterministic=False)
                    overrides.append(override)
                    obs, rew, terminated, truncated, info = eval_env.step(action)
                    ret += float(rew)
                    done = terminated or truncated

                info["constraint"]["episode"].update({"override_rate": float(np.mean(overrides))})
                overrides.clear()

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add("eval/rollout", info)

        return returns

    def act_override(self, key, obs, deterministic=False):
        key, key1, key2, key3 = jr.split(key, 4)

        safe_act = act = self._act(key1, obs, self.Q, deterministic, self.boltzmann_temp, self._epsilon)

        if self.mode in ["exact", "model_free"]:
            override = False if self.S[0, obs, act] <= self.step_wise_prob else True

        elif self.mode == "model_based":
            self.model_checker.update_kernel(
                successor_states=self.successors, probabilities=compact_probabilities(self.counts)
            )

            if self.exploration == "boltzmann":
                self.backup_policy = q_to_boltzmann_policy(self.B, self.boltzmann_temp)
            elif self.exploration == "epsilon_greedy":
                self.backup_policy = q_to_argmax_policy(self.B)
            else:
                raise NotImplementedError(f"Unexpected exploration: {self.exploration}")

            if self.model_checking == "exact":
                result = self.model_checker.check_state_action(
                    None, self.backup_policy
                )[obs, act]

            if self.model_checking == "statistical":
                result = self.model_checker.check_state_action(
                    key2, self.backup_policy, obs, act, self.samples
                )

            override = False if result else True
        else:
            raise NotImplementedError(f"Unexpected mode: {self.mode}")
            
        if override:
            # Use boltzmann temperature or epsilon=0.01 for backup policy
            safe_act = self._act(key3, obs, self.B, deterministic, self.boltzmann_temp, 0.01)
        return self.prepare_act(safe_act), self.prepare_act(act), override

    def act(self, key, obs, deterministic=False):
        action, _, _ = self.act_override(key, obs, deterministic=deterministic)
        return action

    def _act(self, key, obs, Q, deterministic, boltzmann_temp, epsilon):
        if deterministic:
            return self.select_action(Q[obs])
        else:
            return self.sample_action(
                key, jnp.asarray(Q[obs], dtype=jnp.float32), boltzmann_temp, epsilon, exploration=self.exploration
            )


