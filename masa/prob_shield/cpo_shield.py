from __future__ import annotations

from contextlib import nullcontext
from typing import Union, Any, Tuple, Dict, List, Optional, Callable

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from gymnasium import spaces
from tqdm.auto import tqdm

import tensorflow_probability.substrates.jax as tfp

from masa.algorithms.on_policy.cpo import CPO
from masa.common.buffers import VTraceCostRolloutBuffer
from masa.common.metrics import TrainLogger
from masa.common.wrappers import ConstraintPersistentWrapper, is_wrapped, get_wrapped
from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperBase
from masa.common.constraints.base import CostFn
from masa.common.label_fn import LabelFn
from masa.prob_shield.helpers import build_successor_states_matrix
from masa.prob_shield.interval_bound_vi import interval_bound_value_iteration


tfd = tfp.distributions

class CPOShield(CPO):
    """
    CPO with probabilistic action shielding.

    The actor remains a policy over the original discrete action space.
    The shield modifies the rollout behaviour distribution only.
    """

    def __init__(
        self,
        env,
        *args,
        clip_rho: float = 1.0,
        clip_c: float = 1.0,
        clip_traj: bool = False,
        beta_clip_eps: float = 1e-6,
        label_fn: Optional[LabelFn] = None,
        cost_fn: Optional[CostFn] = None,
        safety_abstraction: Optional[Callable[[Any], int]] = None,
        theta: float = 1e-10,
        max_vi_steps: int = 1000,
        init_safety_bound: float = 0.5,
        **kwargs,
    ):

        self.clip_rho = clip_rho
        self.clip_c = clip_c
        self.clip_traj = clip_traj
        self.beta_clip_eps = beta_clip_eps

        self.safety_abstraction = safety_abstraction
        self.theta = theta
        self.max_vi_steps = max_vi_steps
        self.init_safety_bound = init_safety_bound

        self._orig_obs_space = env.observation_space
        self._orig_act_space = env.action_space

        # Sanity checks
        if is_wrapped(env, ProbShieldWrapperBase):
            raise RuntimeError(
                "Environment is already wrapped in ProbShieldWrapperBase. "
                "Please do not warp the env in ProbShieldWrapperBase before passing to CPOShield."
            )
        if not isinstance(env, ConstraintPersistentWrapper):
            raise TypeError(
                "ProbShieldWrapperBase expects `env` to be an instance of "
                f"ConstraintPersistentWrapper, got {type(env).__name__}."
            )
        if not isinstance(env.observation_space, spaces.Discrete) and safety_abstraction is None:
            raise TypeError(
                "ProbShieldWrapperBase only supports environments with a "
                f"Discrete observation space or a discrete safety abstraction, got: {type(env.observation_space).__name__}"
            )

        assert isinstance(self._orig_act_space, spaces.Discrete)

        super().__init__(env, *args, **kwargs)

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

    def _setup_buffer(self):
        self.rollout_buffer = VTraceCostRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_envs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
            clip_rho=self.clip_rho,
            clip_c=self.clip_c,
            clip_traj=self.clip_traj,
        )

    def _abstraction(self, obs: Any) -> Union[int, np.ndarray]:
        """
        Convert one-hot observations back to discrete state ids, then optionally
        apply the safety abstraction per environment.

        Expected batched input:
            obs.shape == (n_envs, n_states)

        Also supports a single one-hot vector:
            obs.shape == (n_states,)
        """
        obs_arr = np.asarray(obs)

        single_obs = obs_arr.ndim == 1
        if single_obs:
            obs_arr = obs_arr[None, :]  # (1, n_states)

        if obs_arr.ndim != 2:
            raise ValueError(
                f"Expected obs with shape (n_envs, n_states) or (n_states,), "
                f"got shape {obs_arr.shape}."
            )

        # Un-one-hot encode per env.
        decoded_obs = np.argmax(obs_arr, axis=1).astype(np.int64)

        # Optional safety abstraction.
        if self.safety_abstraction is not None:
            abstr_obs = []
            for s in decoded_obs:
                try:
                    abstr_obs.append(int(self.safety_abstraction(int(s))))
                except Exception as exc:
                    raise TypeError(
                        f"Could not cast abstracted state as int. "
                        f"Got value from safety_abstraction({int(s)!r}) with error: {exc}"
                    ) from exc

            abstr_obs = np.asarray(abstr_obs, dtype=np.int64)
        else:
            # If no safety abstraction is supplied, the decoded discrete state is
            # already the abstract safety state.
            abstr_obs = decoded_obs

        if single_obs:
            return int(abstr_obs[0])

        return abstr_obs

    def action_projection(
        self,
        actor_probs: np.ndarray,
        raw_beta_proposals: np.ndarray,
        current_obs: int,
        current_safety_bound: float,
        eps: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Project a CPO actor distribution into a shielded behaviour distribution.

        actor_probs:
            Shape: [n_actions]. The CPO actor distribution pi(.|s).

        raw_beta_proposals:
            Shape: [max_successors] or [k]. Raw next-state unsafety-bound proposals
            in the same coordinate as self.safety_lb, not residual beta coordinates.

        Returns:
            shield_probs:
                Shape [n_actions]. Behaviour distribution used for sampling.

            proj_safety_bounds:
                Shape [k]. Projected raw next-state beta bounds.

            metrics:
                Useful diagnostics.
        """
        actor_probs = np.asarray(actor_probs, dtype=np.float64)
        actor_probs = np.clip(actor_probs, 0.0, 1.0)
        actor_probs = actor_probs / np.sum(actor_probs)

        probs_curr = self.probabilities[:, current_obs, :].copy()
        mask = np.any(probs_curr > 0.0, axis=1)
        successors_s = np.nonzero(mask)[0]
        k = len(successors_s)

        probs_curr = probs_curr[successors_s]  # [k, n_actions]

        successor_states = self.successor_states_matrix[successors_s, current_obs]
        beta_star = self.safety_lb[successor_states].astype(np.float64)

        # Cost critic gives raw beta proposals. Enforce beta >= beta*.
        beta = np.asarray(raw_beta_proposals, dtype=np.float64)[:k]
        beta = np.clip(beta, 0.0, 1.0)
        beta = np.maximum(beta, beta_star)

        # Expected next unsafety bound under each base action.
        exp_by_action = np.sum(probs_curr * beta[:, None], axis=0)
        exp_actor = float(np.dot(actor_probs, exp_by_action))

        # If CPO is unnecessarily conservative, inflate uniformly toward 1
        # until the actor distribution uses as much risk budget as possible.
        inflated = False
        if exp_actor < current_safety_bound - eps:
            eta = (
                (current_safety_bound - exp_actor)
                / (1.0 - exp_actor + eps)
            )
            eta = float(np.clip(eta, 0.0, 1.0))
            beta = (1.0 - eta) * beta + eta * np.ones_like(beta)
            inflated = eta > 0.0

            exp_by_action = np.sum(probs_curr * beta[:, None], axis=0)
            exp_actor = float(np.dot(actor_probs, exp_by_action))

        shield_probs = actor_probs.copy()
        interpolated = False
        interp_alpha = 0.0

        # If CPO distribution violates the current risk bound, interpolate toward
        # the safest action distribution.
        if exp_actor > current_safety_bound + eps:
            # Safest action according to the projected beta proposals.
            safest_action = int(np.argmin(exp_by_action))
            exp_safe = float(exp_by_action[safest_action])

            # If even the safest action violates, pull beta back toward beta*.
            # This mirrors the "project infeasible alphas" branch in _project_act.
            if exp_safe > current_safety_bound + eps:
                exp_star_by_action = np.sum(probs_curr * beta_star[:, None], axis=0)
                exp_star_safe = float(exp_star_by_action[safest_action])

                eta = (
                    (exp_safe - current_safety_bound)
                    / (exp_safe - exp_star_safe + eps)
                )
                eta = float(np.clip(eta, 0.0, 1.0))
                beta = (1.0 - eta) * beta + eta * beta_star

                exp_by_action = np.sum(probs_curr * beta[:, None], axis=0)
                exp_actor = float(np.dot(actor_probs, exp_by_action))
                exp_safe = float(exp_by_action[safest_action])

            safe_probs = np.zeros_like(actor_probs)
            safe_probs[safest_action] = 1.0

            interp_alpha = (
                (exp_actor - current_safety_bound)
                / (exp_actor - exp_safe + eps)
            )
            interp_alpha = float(np.clip(interp_alpha, 0.0, 1.0))

            shield_probs = (1.0 - interp_alpha) * actor_probs + interp_alpha * safe_probs
            shield_probs = np.clip(shield_probs, 0.0, 1.0)
            shield_probs = shield_probs / np.sum(shield_probs)
            interpolated = interp_alpha > 0.0

        exp_shield = float(np.dot(shield_probs, np.sum(probs_curr * beta[:, None], axis=0)))

        metrics = {
            "shield/exp_actor_beta": exp_actor,
            "shield/exp_shield_beta": exp_shield,
            "shield/current_beta": float(current_safety_bound),
            "shield/inflated_beta": float(inflated),
            "shield/interpolated_policy": float(interpolated),
            "shield/interp_alpha": float(interp_alpha),
        }

        return shield_probs, beta, metrics

    def _beta_proposals_from_cost_critic(self, current_obs: int) -> np.ndarray:
        probs_curr = self.probabilities[:, current_obs, :].copy()
        mask = np.any(probs_curr > 0.0, axis=1)
        successors_s = np.nonzero(mask)[0]
        k = len(successors_s)

        assert k <= self.max_successors, (
            f"k={k} exceeds max_successors={self.max_successors}"
        )

        successor_states = self.successor_states_matrix[successors_s, current_obs]
        n = self.safety_lb.shape[0]

        successor_obs = np.zeros((self.max_successors, n), dtype=np.float32)
        successor_obs[:k] = np.eye(n, dtype=np.float32)[successor_states]
        successor_obs = np.eye(n, dtype=np.float32)[successor_states]

        betas = self._predict_betas(self.policy.featurizer_state, self.policy.cost_critic_state, successor_obs)

        valid_betas = betas[:k]

        beta_star = self.safety_lb[successor_states]
        valid_betas = np.clip(valid_betas, 0.0, 1.0)
        valid_betas = np.maximum(valid_betas, beta_star)

        out = np.ones(self.max_successors, dtype=np.float64)
        out[:k] = valid_betas
        return out

    def train(self, *args, **kwargs):
        self._current_safety_bound = [self.init_safety_bound]*self.n_envs
        super().train(*args, **kwargs)

    def rollout(
        self, 
        step: int,
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1,
    ):
        steps = 0
        self.rollout_buffer.reset()
        self._last_obs = np.array(self._last_obs)
        self._current_obs = np.array(self._abstraction(self._last_obs)).reshape((self.n_envs,))
        self._last_episode_start = np.array(self._last_episode_start)
        self._last_cost_episode_start = np.array(self._last_cost_episode_start)

        ep_costs = []

        pbar_context = (
            tqdm(
                total=self.n_steps,
                desc="rollout",
                position=tqdm_position,
                leave=False,
                dynamic_ncols=True,
                colour="green",
            )
            if self.use_tqdm_rollout else nullcontext()
        )

        with pbar_context as pbar:
            while steps < self.n_steps:
                self.policy.reset_noise()

                obs = self.prepare_obs(self._last_obs, n_envs=self.n_envs)

                self._current_obs = np.array(self._abstraction(self._last_obs)).reshape((self.n_envs,))

                actor_probs, values, cost_values = self._predict_all(
                    self.policy.featurizer_state, 
                    self.policy.actor_state, 
                    self.policy.critic_state, 
                    self.policy.cost_critic_state,
                    obs,
                )

                actor_probs = np.array(actor_probs)
                values = np.array(values)
                cost_values = np.array(cost_values)

                actions = np.zeros(self.n_envs, dtype=np.int64)
                actor_log_probs = np.zeros(self.n_envs, dtype=np.float32)
                shield_log_probs = np.zeros(self.n_envs, dtype=np.float32)
                proj_bounds_per_env = []

                for env_idx in range(self.n_envs):

                    beta_props = self._beta_proposals_from_cost_critic(self._current_obs[env_idx])

                    shield_probs, proj_bounds, shield_metrics = self.action_projection(
                        actor_probs[env_idx],
                        beta_props,
                        self._current_obs[env_idx],
                        self._current_safety_bound[env_idx],
                    )

                    action = tfd.Categorical(probs=shield_probs).sample(seed=self.policy.noise_key)

                    actions[env_idx] = action
                    actor_log_probs[env_idx] = np.log(actor_probs[env_idx, action] + 1e-8)
                    shield_log_probs[env_idx] = np.log(shield_probs[action] + 1e-8)

                    proj_bounds_per_env.append((proj_bounds, shield_metrics))

                new_obs_list = []
                reward_list = []
                terminated_list = []
                truncated_list = []
                info_list = []

                for env_idx, action in enumerate(actions):
                    proj_bounds, shield_metrics = proj_bounds_per_env[env_idx]

                    new_obs, reward, terminated, truncated, info = self.env.envs[env_idx].step(int(action))

                    abstr_obs = int(self._abstraction(new_obs))

                    succ_list = self.successor_states_matrix[:, self._current_obs[env_idx]]
                    matches = np.where(succ_list == abstr_obs)[0]
                    if matches.size == 0:
                        raise RuntimeError(
                            f"Next abstract state {abstr_obs} is not in successor list "
                            f"for current abstract state {self._current_obs[env_idx]}."
                        )

                    next_obs_idx = int(matches[0])

                    self._current_safety_bound[env_idx] = float(proj_bounds[next_obs_idx])
                    self._current_obs[env_idx] = abstr_obs

                    new_obs_list.append(new_obs)
                    reward_list.append(reward)
                    terminated_list.append(terminated)
                    truncated_list.append(truncated)
                    info_list.append(info)

                new_obs = np.array(new_obs_list)
                rewards = np.array(reward_list)
                terminated = np.array(terminated_list)
                truncated = np.array(truncated_list)
                infos = info_list

                costs = np.array([info["constraint"]["step"].get("cost", 0.0) for info in infos])
                cost_dones = np.array([info["constraint"]["step"].get("cost_done", False) for info in infos])

                for info in infos:
                    if "episode" in info["constraint"]:
                        if self.constraint_type == "CMDP":
                            ep_costs.append(info["constraint"]["episode"].get("cum_cost", 0.0))
                        elif self.constraint_type == "PCTL":
                            ep_costs.append(1.0 - info["constraint"]["episode"].get("satisfied", 0.0))
                        elif self.constraint_type == "LTL_SAFETY":
                            ep_costs.append(1.0 - info["constraint"]["episode"].get("satisfied", 0.0))
                        elif self.constraint_type == "REACH_AVOID":
                            ep_costs.append(info["constraint"]["episode"].get("violated", 0.0))
                        elif self.constraint_type == "PROB":
                            ep_costs.append(info["constraint"]["episode"].get("cum_unsafe", 0.0))
                        else:
                            ep_costs.append(info["constraint"]["episode"].get("cum_cost", 0.0))
            
                steps += 1
                
                if self.use_tqdm_rollout:
                    pbar.update(1)

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = np.array(actions)
                    actions = actions.reshape(-1, 1)

                dones = np.array([False]*self.n_envs)

                for idx, info in enumerate(infos):
                    if truncated[idx]:
                        truncated_obs = new_obs[idx].reshape(1, -1)
                        feats = self.featurizer.apply(self.policy.featurizer_state.params, truncated_obs)
                        terminal_value = np.array(
                            self.critic.apply(
                                self.policy.critic_state.params,
                                feats,
                            ).flatten()
                        ).item()
                        rewards[idx] += self.gamma * terminal_value

                        if not cost_dones[idx]:
                            terminal_cost_value = np.array(
                                self.cost_critic.apply(
                                    self.policy.cost_critic_state.params,
                                    feats,
                                ).flatten()
                            ).item()
                            costs[idx] += self.cost_gamma * terminal_cost_value

                    if terminated[idx] or truncated[idx]:
                        dones[idx] = True

                self.rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    costs,
                    self._last_episode_start,
                    self._last_cost_episode_start,
                    values,
                    cost_values,
                    actor_log_probs,
                    shield_log_probs,
                )

                if np.any(cost_dones):
                    for i, cost_done in enumerate(cost_dones):
                        if self.constraint_type == "REACH_AVOID":
                            dones[i] = cost_done # terminate the episode if we reach the avoid state
                        #if self.constraint_type == "LTL_SAFETY":
                        #    self.env.envs[i]._constraint.cost_fn.reset() # reset the DFA const function and continue the episode

                if np.any(dones):
                    reset_obs, _ = self.env.reset_done(dones)
                    for i, done in enumerate(dones):
                        if done and reset_obs[i] is not None:
                            new_obs[i] = reset_obs[i]
                            self._current_safety_bound[i] = self.init_safety_bound

                self._last_obs = new_obs
                self._last_episode_start = dones
                self._last_cost_episode_start = np.logical_or(dones, cost_dones)

                if logger:
                    for info in infos:
                        logger.add("train/rollout", info)
                        logger.add("train/stats", shield_metrics)

        assert isinstance(self._last_obs, np.ndarray) 
        final_obs = self.prepare_obs(self._last_obs, n_envs=self.n_envs)
        feats = self.featurizer.apply(self.policy.featurizer_state.params, final_obs)
        last_value = np.array(
            self.critic.apply(
                self.policy.critic_state.params,
                feats,
            ).flatten()
        )
        last_cost_value = np.array(
            self.policy.cost_critic.apply(
                self.policy.cost_critic_state.params,
                feats,
            ).flatten()
        )
        last_cost_value = np.where(
            self._last_cost_episode_start,
            0.0,
            last_cost_value,
        )

        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value, last_cost_value=last_cost_value, done=self._last_episode_start, cost_done=self._last_cost_episode_start
        )

        if len(ep_costs) > 0:
            self.mean_ep_cost = float(np.mean(ep_costs))
        else:
            self.mean_ep_cost = 0.0

    @staticmethod
    @jit
    def _predict_all(
        featurizer_state: TrainState, actor_state: TrainState, critic_state: TrainState, cost_critic_state: TrainState, observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_state.params, features)
        actor_probs = dist.probs_parameter()
        values = critic_state.apply_fn(critic_state.params, features).flatten()
        cost_values = cost_critic_state.apply_fn(cost_critic_state.params, features).flatten()
        return actor_probs, values, cost_values

    @staticmethod
    @jit
    def _predict_betas(
        featurizer_state: TrainState, cost_critic_state: TrainState, observations: jnp.ndarray
    ) -> jnp.ndarray:
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        betas = cost_critic_state.apply_fn(cost_critic_state.params, features).flatten()
        return betas

    @staticmethod
    @jit
    def _predict_actor_probs(featurizer_state: TrainState, actor_state: TrainState, observations: jnp.ndarray) -> jnp.ndarray:
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_state.params, features)
        actor_probs = dist.probs_parameter()
        return actor_probs

    def act(self, key: jax.Array, obs: np.ndarray, safety_bound: float, deterministic: bool = False) -> Union[int, np.ndarray]:
        obs = self.prepare_obs(obs, n_envs=1)
        actor_probs = self._predict_actor_probs(self.policy.featurizer_state, self.policy.actor_state, obs)
        actor_probs = np.array(actor_probs).flatten()
        current_obs = int(self._abstraction(obs))
        beta_props = self._beta_proposals_from_cost_critic(current_obs)
        shield_probs, proj_bounds, shield_metrics = self.action_projection(
            actor_probs,
            beta_props,
            current_obs,
            safety_bound,
        )
        if deterministic:
            action = tfd.Categorical(probs=shield_probs).sample(seed=key)
        else:
            action = tfd.Categorical(probs=shield_probs).mode()
        action = self.prepare_act(action, n_envs=1)
        return action, proj_bounds

    def eval(self, num_episodes: int, seed: Optional[int] = None, logger: Optional[TrainLogger] = None) -> List[float]:
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
            colour="yellow"
        ) as pbar:

            for ep in range(num_episodes):
                obs, info = eval_env.reset(seed=eval_seed + ep)
                current_obs = int(self._abstraction(obs))
                current_eval_safety_bound = self.init_safety_bound
                done = False
                ret = 0.0
                while not done:
                    eval_key, subkey = jr.split(eval_key)
                    action, proj_bounds = self.act(subkey, obs, current_eval_safety_bound, deterministic=False)
                    
                    obs, rew, terminated, truncated, info = eval_env.step(action)

                    abstr_obs = int(self._abstraction(obs))
                    succ_list = self.successor_states_matrix[:, current_obs]
                    matches = np.where(succ_list == abstr_obs)[0]
                    if matches.size == 0:
                        raise RuntimeError(
                            f"Next abstract state {abstr_obs} is not in successor list "
                            f"for current abstract state {current_obs}."
                        )

                    next_obs_idx = int(matches[0])
                    current_eval_safety_bound = float(proj_bounds[next_obs_idx])
                    current_obs = abstr_obs

                    ret += float(rew)
                    done = terminated or truncated

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add("eval/rollout", info)

        return returns