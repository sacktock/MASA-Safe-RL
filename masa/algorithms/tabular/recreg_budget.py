from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from gymnasium import spaces
from tqdm import tqdm

import masa.common.pctl as pctl
from masa.algorithms.tabular.q_learning import QL
from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.ltl import DFA, DFACostFn
from masa.common.metrics import TrainLogger
from masa.common.wrappers import get_wrapped, is_wrapped
from masa.prob_shield.eventual_discounted_vi import ev_value_iteration
from masa.prob_shield.helpers import build_successor_states_matrix


def build_pctl_formula(
    env: gym.Env,
    bound_param: int,
    prob: float,
    label_fn: Optional[LabelFn] = None,
    cost_fn: Optional[CostFn] = None,
) -> pctl.BoundedPCTLFormula:

    cost_fn = cost_fn if cost_fn is not None else getattr(env, "cost_fn", None)
    if cost_fn is None:
        raise AttributeError("Environment must define a `cost_fn` attribute.")

    label_fn = label_fn if label_fn is not None else getattr(env, "label_fn", None)
    if label_fn is None:
        raise AttributeError("Environment must define a `label_fn` attribute.")

    if not hasattr(env, "observation_space"):
        raise AttributeError("Environment must have an `observation_space` attribute.")
    if isinstance(env.observation_space, spaces.Discrete):
        n_states = env.observation_space.n
    else:
        raise NotImplementedError(
            f"Unexpected observation_space {type(env.observation_space).__name__}"
        )

    unsafe_dict = defaultdict(set)
    for s in range(n_states):
        if cost_fn(label_fn(s)):
            unsafe_dict[s] = {"unsafe"}

    new_label_fn = lambda obs: unsafe_dict[obs]
    formula = pctl.Always(1.0 - prob, bound_param, pctl.Neg(pctl.Atom("unsafe")))
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

    k_slot = int(next_free[s])
    if k_slot >= new_K or succ_new[k_slot, s] != -1:
        free = np.where(succ_new[:, s] == -1)[0]
        if free.size == 0:
            raise RuntimeError(
                f"handle_overflow failed: no free slot after resizing K {K}->{new_K}"
            )
        k_slot = int(free[0])

    succ_new[k_slot, s] = int(s_next)
    succ_index[s][int(s_next)] = k_slot

    if k_slot == int(next_free[s]):
        next_free[s] = np.int32(k_slot + 1)

    return succ_new, counts_new, succ_index, next_free, k_slot


def compact_probabilities(counts: np.ndarray) -> np.ndarray:
    return counts / counts.sum(axis=0, keepdims=True)


def q_to_argmax_policy(Q: np.ndarray) -> np.ndarray:
    # Deterministic argmax policy (ties broken by lowest index, as np.argmax)
    Q = np.asarray(Q)
    policy = np.zeros(Q.shape, dtype=np.float64)
    policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1.0
    return policy


class RECREG_BUDGET(QL):
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
        n_bins: int = 100,
        slack_max: Optional[float] = None,
        slack_max_coef: float = 2.0,
        backup_explore: bool = True,
        backup_temp: float = 0.01,
        backup_epsilon: float = 0.001,
        **kwargs,
    ):
        """
        Args (new w.r.t. RECREG):
            n_bins: number K of slack bins (K >= 2). Bin k has slack σ_k = k·Δ.
            slack_max: top of the slack grid σ_max. Slack above it is floored to σ_max
                (sound). Defaults to slack_max_coef * safety_prob.
            slack_max_coef: see slack_max.
            backup_explore: in MODELBASED/MODELFREE training, sample backup actions from
                the live backup Q table (Boltzmann with backup_temp, or epsilon-greedy with
                backup_epsilon) instead of the snapshot's greedy backup (Assumption 5.2).
                EXACT mode and evaluation always use the snapshot's greedy backup.
        """

        super().__init__(
            *args,
            alpha=task_alpha,
            gamma=task_gamma,
            **kwargs,
        )

        assert mode in ["exact", "model_based", "model_free"], (
            f"Unsupported mode {mode}"
        )
        assert model_checking in ["exact", "statistical", "none"], (
            f"Unsupported model-checking type {model_checking}"
        )
        assert not model_checking == "none" or mode == "model_free", (
            f"Must specify a model checking type when using mode: {mode}"
        )
        assert samples > 0, "Number of samples must be greater than 0"
        assert n_bins >= 2, "Need at least two slack bins"

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
        self.n_bins = int(n_bins)
        self.backup_explore = backup_explore
        self.backup_temp = backup_temp
        self.backup_epsilon = backup_epsilon

        # Target for masked (uncertified) pairs, Eq. (11) with r_min offset
        self._masked_value = self.r_min / (1.0 - self.gamma)

        self._setup_slack_grid(slack_max, slack_max_coef)
        self._setup_budget_q_table()
        self._setup_models()

        # Episode / diagnostics state
        self._budget_deficits = 0
        self._infeasible_starts = 0
        self._ep_overrides: List[bool] = []
        self._ep_bins: List[int] = []
        self._k = 0
        self._needs_refresh = True  # begin-of-episode work pending

        # Initial risk snapshot (so act/eval work before training)
        self._snapshot_ready = False
        self._refresh_risk_snapshot()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_slack_grid(self, slack_max: Optional[float], slack_max_coef: float):
        sm = slack_max if slack_max is not None else slack_max_coef * self.safety_prob
        self.slack_max = float(max(sm, 0.0))
        self._grid = np.linspace(0.0, self.slack_max, self.n_bins, dtype=np.float64)
        self._delta = (
            self.slack_max / (self.n_bins - 1) if self.slack_max > 0.0 else np.inf
        )
        self._bin_ids = np.arange(self.n_bins)

    def _setup_budget_q_table(self):
        # Replaces the (n_states, n_actions) table allocated by QL.__init__
        self.Q = np.zeros(
            (self.n_states, self.n_bins, self.n_actions), dtype=np.float32
        )

    def _setup_models(self):

        if self.mode == "exact":
            successor_states_matrix, probabilities, _, label_fn, cost_fn, _ = (
                build_successor_states_matrix(self.env)
            )

            # Backup policy Q values (eventually discounted VI, Prop. 4.2)
            self.B = ev_value_iteration(
                successor_states_matrix,
                probabilities,
                label_fn,
                cost_fn,
                gamma=self.safe_gamma,
            )

            formula, pctl_label_fn = build_pctl_formula(
                self.env,
                self.horizon,
                self.safety_prob,
                label_fn=label_fn,
                cost_fn=cost_fn,
            )

            self.model_checker = pctl.ExactModelChecker(
                formula,
                pctl_label_fn,
                {"unsafe"},
                successor_states=successor_states_matrix,
                probabilities=probabilities,
            )

            self._exact_successors = np.asarray(successor_states_matrix)
            self._exact_probabilities = np.asarray(probabilities)

            self.unsafe_mask = np.array(
                [bool(cost_fn(label_fn(s))) for s in range(self.n_states)],
                dtype=bool,
            )

        elif self.mode in ["model_based", "model_free"]:
            if is_wrapped(self.env, LTLSafetyEnv):
                from masa.common.constraints.ltl_safety import (
                    create_product_label_fn,
                    product_cost_fn,
                )

                ltl_safety_env: LTLSafetyEnv | None = get_wrapped(
                    self.env, LTLSafetyEnv
                )
                dfa = self.env._constraint.get_dfa()
                if isinstance(ltl_safety_env._orig_obs_space, spaces.Discrete):
                    n_states = ltl_safety_env._orig_obs_space.n
                else:
                    raise NotImplementedError(
                        f"Unexpected observation space: {ltl_safety_env._orig_obs_space}"
                    )

                label_fn = create_product_label_fn(n_states, dfa)
                cost_fn = product_cost_fn
            else:
                label_fn = getattr(self.env, "label_fn", None)
                if label_fn is None:
                    raise AttributeError(
                        "Environment must define a `label_fn` attribute."
                    )
                cost_fn = getattr(self.env, "cost_fn", None)
                if cost_fn is None:
                    raise AttributeError(
                        "Environment must define a `cost_fn` attribute."
                    )

            self.unsafe_mask = np.array(
                [bool(cost_fn(label_fn(s))) for s in range(self.n_states)],
                dtype=bool,
            )

            # Backup policy Q values (negative eventually discounted cost)
            self.B = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

            if self.mode == "model_based":
                formula, pctl_label_fn = build_pctl_formula(
                    self.env,
                    self.horizon,
                    self.safety_prob,
                    label_fn=label_fn,
                    cost_fn=cost_fn,
                )

                max_succesors = 5  # grows on overflow

                self.successors = -np.ones(
                    (max_succesors, self.n_states), dtype=np.int64
                )
                self.successors[0, :] = np.arange(self.n_states, dtype=np.int64)
                self.counts = np.zeros(
                    (max_succesors, self.n_states, self.n_actions), dtype=np.float32
                )
                self.counts[0, :, :] = 1e-3  # self-loop prior for unvisited pairs

                self.successor_index = [{int(s): 0} for s in range(self.n_states)]
                self.next_free = np.ones(self.n_states, dtype=np.int32)

                if self.model_checking == "exact":
                    model_checker_cls = pctl.ExactModelChecker
                elif self.model_checking == "statistical":
                    model_checker_cls = pctl.StatisticalModelChecker
                else:
                    raise NotImplementedError(
                        f"Unexpected model checker class: {self.model_checking}"
                    )

                self.model_checker = model_checker_cls(
                    formula,
                    pctl_label_fn,
                    {"unsafe"},
                    successor_states=self.successors,
                    probabilities=compact_probabilities(self.counts),
                )

            if self.mode == "model_free":
                # Risk critics: S[t] estimates the (H - t)-step risk of (s, a) then backup;
                # S[0] is the H-step risk, S[H-1] the immediate (one-step) risk.
                self.S = np.zeros(
                    (self.horizon, self.n_states, self.n_actions), dtype=np.float32
                )

        else:
            raise NotImplementedError(f"Unexpected mode: {self.mode}")

    # ------------------------------------------------------------------
    # Risk snapshot (frozen per episode)
    # ------------------------------------------------------------------

    def _immediate_risk_table(self, succ: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """p(s, a) = P(next state unsafe | s, a) from a compact (K, S) / (K, S, A) kernel."""
        succ = np.asarray(succ)
        probs = np.asarray(probs, dtype=np.float64)
        valid = succ >= 0
        unsafe_succ = np.zeros(succ.shape, dtype=np.float64)
        unsafe_succ[valid] = self.unsafe_mask[succ[valid]]
        return np.einsum("ks,ksa->sa", unsafe_succ, probs)

    def _refresh_risk_snapshot(self) -> None:
        """Recompute U(s, a), ρ(s), p(s, a) and the greedy backup from the current
        data (MODELBASED) or critics (MODELFREE). EXACT computes them once."""

        if self.mode == "exact":
            if self._snapshot_ready:
                return
            backup_policy = q_to_argmax_policy(self.B)
            U = 1.0 - np.asarray(
                self.model_checker.check_state_action(self.key, backup_policy)
            )
            p = self._immediate_risk_table(
                self._exact_successors, self._exact_probabilities
            )

        elif self.mode == "model_based":
            probs = compact_probabilities(self.counts)
            self.model_checker.update_kernel(
                successor_states=self.successors,
                probabilities=probs,
            )
            if self.model_checking != "exact":
                raise NotImplementedError(
                    f"Model checker class: {self.model_checking} currently not supported"
                )
            backup_policy = q_to_argmax_policy(self.B)
            U = 1.0 - np.asarray(
                self.model_checker.check_state_action(None, backup_policy)
            )
            p = self._immediate_risk_table(self.successors, probs)

        else:  # model_free
            U = np.asarray(self.S[0], dtype=np.float64)
            p = np.asarray(self.S[-1], dtype=np.float64)

        backup_act = np.argmax(np.asarray(self.B), axis=1).astype(np.int64)

        self._U = np.clip(np.asarray(U, dtype=np.float64), 0.0, 1.0)
        self._p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        self._backup_act = backup_act
        # ρ(s) = U(s, π_safe(s)), read from the same array so U(s, π_safe(s)) == ρ(s) exactly
        self._rho = self._U[np.arange(self.n_states), backup_act]
        self._snapshot_ready = True

    # ------------------------------------------------------------------
    # Vectorized slack helpers (all use the frozen snapshot)
    # ------------------------------------------------------------------

    def _floor_bins(self, sigma: np.ndarray) -> np.ndarray:
        """Floor slack values onto the grid; values above σ_max go to the top bin."""
        idx = np.floor(np.asarray(sigma, dtype=np.float64) / self._delta)
        return np.clip(idx, 0, self.n_bins - 1).astype(np.int64)

    def _survival_scale(self, p: np.ndarray) -> np.ndarray:
        """1 / (1 - p), with 1 when p ≈ 1 (conservative: no survival rescaling)."""
        p = np.asarray(p, dtype=np.float64)
        ok = p < 1.0 - 1e-6
        return np.where(ok, 1.0 / np.where(ok, 1.0 - p, 1.0), 1.0)

    def _certified_mask(self, s: np.ndarray) -> np.ndarray:
        """(n,) product states -> (n, K, A) bool: U(s, a) <= ρ(s) + σ_k."""
        U = self._U[s]  # (n, A)
        thr = self._rho[s][:, None] + self._grid[None, :]  # (n, K)
        return U[:, None, :] <= thr[:, :, None]

    def _next_bins(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(n,) states, (n,) executed actions -> next slack bins (n, K) for every
        current bin, and the slack numerators σ_k + ρ(s) - U(s, a) (n, K)."""
        num = self._grid[None, :] + (self._rho[s] - self._U[s, a])[:, None]
        sigma_next = np.maximum(num, 0.0) * self._survival_scale(self._p[s, a])[:, None]
        return self._floor_bins(sigma_next), num

    def _masked_max(self, s: np.ndarray, k: np.ndarray) -> np.ndarray:
        """(n,) states, (n, K) bins -> (n, K) values max_{a ∈ A(s, k)} Q(s, k, a),
        with max ∅ := r_min / (1 - γ)."""
        q = self.Q[s[:, None], k]  # (n, K, A)
        thr = self._rho[s][:, None] + self._grid[k]  # (n, K)
        cert = self._U[s][:, None, :] <= thr[:, :, None]  # (n, K, A)
        vals = np.where(cert, q, -np.inf).max(axis=-1)
        return np.where(cert.any(axis=-1), vals, self._masked_value)

    def _initial_bin(self, obs: int, count: bool = True) -> int:
        """σ_0 = ε - ρ(s_0), floored. Counts an infeasible start if ρ(s_0) > ε."""
        sigma0 = self.safety_prob - float(self._rho[obs])
        if sigma0 < -1e-9 and count:
            self._infeasible_starts += 1
        return int(self._floor_bins(np.array([max(sigma0, 0.0)]))[0])

    def _next_bin_executed(self, s: int, a: int, s_next: int, k: int) -> int:
        """Residual-budget update for the executed transition (floored supermartingale).

        Under the frozen snapshot, certified task actions and the greedy backup always give
        a non-negative numerator, so deficits can only arise from exploratory backup
        actions during training in the learned modes.
        """
        if self.unsafe_mask[s_next]:
            return 0
        bins, num = self._next_bins(
            np.array([s], dtype=np.int64), np.array([a], dtype=np.int64)
        )
        if num[0, k] < -1e-9 and not self.unsafe_mask[s]:
            self._budget_deficits += 1
        return int(bins[0, k])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, *args, **kwargs):
        self._budget_deficits = 0
        self._infeasible_starts = 0
        self._ep_overrides = []
        self._ep_bins = []
        self._needs_refresh = True
        super().train(*args, **kwargs)

    def _begin_episode(self, obs: int) -> None:
        # Recompute the snapshot with everything learned during the previous episode
        # (including the last transition, which optimize() has processed by now).
        self._refresh_risk_snapshot()
        self._k = self._initial_bin(obs)
        self._needs_refresh = False

    def rollout(self, step: int, logger: Optional[TrainLogger] = None):

        if self._needs_refresh:
            self._begin_episode(self._last_obs)

        self.key, subkey = jr.split(self.key)
        obs, k = self._last_obs, self._k
        exec_act, prop_act, override = self.act_override(subkey, obs, k)
        next_obs, reward, terminated, truncated, info = self.env.step(exec_act)

        cost = info["constraint"]["step"].get("cost", 0.0)
        violation = info["constraint"]["step"].get("violation", False)

        next_k = self._next_bin_executed(obs, exec_act, next_obs, k)

        self._ep_overrides.append(bool(override))
        self._ep_bins.append(k)

        self.buffer.append(
            (
                obs,
                k,
                exec_act,
                prop_act,
                override,
                reward,
                cost,
                violation,
                next_obs,
                terminated,
                info,
            )
        )

        if terminated or truncated:
            info["constraint"]["episode"].update(
                {
                    "override_rate": float(np.mean(self._ep_overrides)),
                    "budget_deficits": int(self._budget_deficits),
                    "infeasible_starts": int(self._infeasible_starts),
                    "expected_bin_occupancy": float(np.mean(self._ep_bins)),
                }
            )
            self._ep_overrides.clear()
            self._ep_bins.clear()
            self._last_obs, _ = self.env.reset()
            self._needs_refresh = True  # snapshot + initial bin at the next rollout
        else:
            self._last_obs = next_obs
            self._k = next_k

        self._step += 1
        self._epsilon = self._epsilon_decay_schedule(self._step)

        if logger:
            logger.add("train/rollout", info)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        """Update the risk models, backup and task Q tables with (real + counterfactual)
        experience. The stored bin k is one of the K bins swept by the slack
        counterfactuals, so it is not needed to build targets."""
        if len(self.buffer) == 0:
            return

        use_cf = (
            hasattr(self.env, "cost_fn")
            and isinstance(self.env.cost_fn, DFACostFn)
            and self.use_counter_factual_experiences
        )

        for (
            state,
            _k,
            exec_act,
            _prop_act,
            _override,
            reward,
            cost,
            violation,
            next_state,
            terminal,
            info,
        ) in self.buffer:
            if use_cf:
                cf_exp = self.generate_counter_factuals(
                    state,
                    exec_act,
                    reward,
                    next_state,
                    terminal,
                    info,
                    self.env.cost_fn,
                )
            else:
                cf_exp = [
                    (state, exec_act, reward, cost, violation, next_state, terminal)
                ]

            cf_s, _, cf_r, cf_c, cf_v, cf_next, cf_term = zip(*cf_exp)
            s = np.asarray(cf_s, dtype=np.int64)
            s_next = np.asarray(cf_next, dtype=np.int64)
            r = np.asarray(cf_r, dtype=np.float64)
            c = np.asarray(cf_c, dtype=np.float64)
            v = np.asarray(cf_v, dtype=bool)
            term = np.asarray(cf_term, dtype=np.float64)
            a = int(exec_act)

            if self.mode == "model_based":
                for s_i, sn_i in zip(s.tolist(), s_next.tolist()):
                    (
                        self.successors,
                        self.counts,
                        self.successor_index,
                        self.next_free,
                    ) = update_compact(
                        self.successors,
                        self.counts,
                        self.successor_index,
                        self.next_free,
                        s=s_i,
                        a=a,
                        s_next=sn_i,
                    )

            if self.mode == "model_free":
                self._update_risk_critics(s, a, v, s_next, term)

            if self.mode in ["model_free", "model_based"]:
                self._update_backup(s, a, c, v, s_next, term)

            self._update_task(s, a, r, s_next, term)

        self.buffer.clear()

        if logger:
            logger.add("train/stats", {"alpha": self.alpha})
            logger.add("train/stats", {"safe_alpha": self.safe_alpha})
            if self.exploration == "boltzmann":
                logger.add("train/stats", {"temp": self.boltzmann_temp})
            if self.exploration == "epsilon_greedy":
                logger.add("train/stats", {"epsilon": self._epsilon})

    def _update_risk_critics(
        self,
        s: np.ndarray,
        a: int,
        v: np.ndarray,
        s_next: np.ndarray,
        term: np.ndarray,
    ) -> None:
        """Eq. (15), vectorized over counterfactual states, greedy (live) backup."""
        b_next = np.argmax(self.B[s_next], axis=1)
        viol = v.astype(np.float64)
        cont = 1.0 - term
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_S = np.minimum(1.0, viol)
            else:
                next_S = np.minimum(1.0, viol + cont * self.S[t + 1, s_next, b_next])
            self.S[t, s, a] = (1.0 - self.model_free_alpha) * self.S[
                t, s, a
            ] + self.model_free_alpha * next_S

    def _update_backup(
        self,
        s: np.ndarray,
        a: int,
        c: np.ndarray,
        v: np.ndarray,
        s_next: np.ndarray,
        term: np.ndarray,
    ) -> None:
        """Eventually discounted backup update (Prop. 4.2), vectorized."""
        pen = -c * self.cost_coef
        g = np.where(v, self.safe_gamma, 1.0)
        target = pen + (1.0 - term) * g * np.max(self.B[s_next], axis=1)
        self.B[s, a] = (1.0 - self.safe_alpha) * self.B[s, a] + self.safe_alpha * target

    def _update_task(
        self,
        s: np.ndarray,
        a: int,
        r: np.ndarray,
        s_next: np.ndarray,
        term: np.ndarray,
    ) -> None:
        """Task update for every (counterfactual state, slack bin) at once, then the
        zero sweep over the remaining uncertified (k, a') pairs.

        s holds distinct product states (one per DFA state), so fancy-index writes are safe.
        """
        n = s.shape[0]
        alpha = self.alpha

        cert = self._certified_mask(s)  # (n, K, A)
        cert_a = cert[:, :, a]  # (n, K)

        k_next, _ = self._next_bins(s, np.full(n, a, dtype=np.int64))  # (n, K)
        v_next = self._masked_max(s_next, k_next)  # (n, K)

        target = r[:, None] + (1.0 - term)[:, None] * self.gamma * v_next
        target = np.where(cert_a, target, self._masked_value)

        q_sa = self.Q[s, :, a]  # (n, K)
        self.Q[s, :, a] = (1.0 - alpha) * q_sa + alpha * target

        # Zero sweep: every other uncertified (k, a') at each counterfactual state
        sweep = ~cert
        sweep[:, :, a] = False  # already updated above
        q_s = self.Q[s]  # (n, K, A)
        self.Q[s] = np.where(
            sweep, (1.0 - alpha) * q_s + alpha * self._masked_value, q_s
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act_override(self, key, obs: int, k: int, deterministic: bool = False):
        """Propose from Q[obs, k], certify with U <= ρ + σ_k, otherwise back up.

        Deterministic proposals are greedy over the certified set (Prop. 4.5); stochastic
        proposals range over all actions, so every action is proposed infinitely often.
        """
        key, key1, key2 = jr.split(key, 3)

        cert_row = self._U[obs] <= self._rho[obs] + self._grid[k]  # (A,)

        if deterministic:
            prop = int(np.argmax(np.where(cert_row, self.Q[obs, k], -np.inf)))
        else:
            prop = self.prepare_act(
                self.sample_action(
                    key1,
                    jnp.asarray(self.Q[obs, k], dtype=jnp.float32),
                    self.boltzmann_temp,
                    self._epsilon,
                    exploration=self.exploration,
                )
            )

        override = not bool(cert_row[prop])
        exec_act = prop

        if override:
            explore_backup = (
                not deterministic
                and self.backup_explore
                and self.mode in ["model_based", "model_free"]
            )
            if explore_backup:
                exec_act = self.prepare_act(
                    self.sample_action(
                        key2,
                        jnp.asarray(self.B[obs], dtype=jnp.float32),
                        self.backup_temp,
                        self.backup_epsilon,
                        exploration=self.exploration,
                    )
                )
            else:
                exec_act = int(self._backup_act[obs])

        return int(exec_act), int(prop), override

    def act(self, key, obs, deterministic=False):
        # No episode context: start from the initial slack σ_0 = ε - ρ(obs).
        k = self._initial_bin(obs, count=False)
        action, _, _ = self.act_override(key, obs, k, deterministic=deterministic)
        return action

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(
        self,
        num_episodes: int,
        seed: Optional[int] = None,
        logger: Optional[TrainLogger] = None,
    ) -> List[float]:
        """Evaluate the shield with the current (frozen) risk snapshot."""
        eval_env = self._get_eval_env()
        base = 0 if self.seed is None else int(self.seed)
        eval_seed = base + 10_000 if seed is None else int(seed) + 10_000
        eval_key = jr.PRNGKey(eval_seed)
        returns = []

        with tqdm(
            total=num_episodes,
            desc="evaluation",
            position=1,
            leave=False,
            dynamic_ncols=True,
            colour="yellow",
        ) as pbar:
            for ep in range(num_episodes):
                obs, info = eval_env.reset(seed=eval_seed + ep)
                done = False
                ret = 0.0
                overrides: List[bool] = []
                bins: List[int] = []

                saved = (self._budget_deficits, self._infeasible_starts)
                self._budget_deficits, self._infeasible_starts = 0, 0

                k = self._initial_bin(obs)
                while not done:
                    eval_key, subkey = jr.split(eval_key)
                    action, _, override = self.act_override(
                        subkey, obs, k, deterministic=True
                    )
                    next_obs, rew, terminated, truncated, info = eval_env.step(action)

                    overrides.append(bool(override))
                    bins.append(k)

                    k = self._next_bin_executed(obs, action, next_obs, k)
                    obs = next_obs

                    ret += float(rew)
                    done = terminated or truncated

                info["constraint"]["episode"].update(
                    {
                        "override_rate": float(np.mean(overrides)),
                        "budget_deficits": int(self._budget_deficits),
                        "infeasible_starts": int(self._infeasible_starts),
                        "expected_bin_occupancy": float(np.mean(bins)),
                    }
                )
                self._budget_deficits, self._infeasible_starts = saved

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add("eval/rollout", info)

        return returns


class RECREG_BUDGET_EXACT(RECREG_BUDGET):
    def __init__(
        self, *args, mode: str = "exact", model_checking: str = "exact", **kwargs
    ):
        super().__init__(*args, mode=mode, model_checking=model_checking, **kwargs)


class RECREG_BUDGET_MODEL_BASED(RECREG_BUDGET):
    def __init__(
        self, *args, mode: str = "model_based", model_checking: str = "exact", **kwargs
    ):
        super().__init__(*args, mode=mode, model_checking=model_checking, **kwargs)


class RECREG_BUDGET_MODEL_FREE(RECREG_BUDGET):
    def __init__(
        self, *args, mode: str = "model_free", model_checking: str = "none", **kwargs
    ):
        super().__init__(*args, mode=mode, model_checking=model_checking, **kwargs)
