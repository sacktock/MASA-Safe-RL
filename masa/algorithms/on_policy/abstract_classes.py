from __future__ import annotations
from abc import ABC
import warnings
from contextlib import nullcontext
from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from gymnasium import spaces
from masa.common.on_policy_algorithm import OnPolicyAlgorithm
from masa.common.buffers import CostRolloutBuffer
from masa.common.metrics import TrainLogger

class OnPolicyCostAlgorithm(OnPolicyAlgorithm, ABC):
    """Abstract cost-aware on-policy algorithm mixin."""

    # Smoothing for the running (EWMA) estimate of ``mean_ep_cost``.
    cost_ewma_beta: float = 0.05

    def _wrap_env(self, env: gym.Env):
        env = super()._wrap_env(env)

        if not hasattr(env, "_constraint") or env._constraint is None:
            raise AttributeError(
                f"{self.__class__.__name__} requires env to expose `_constraint`."
            )

        self.constraint = env._constraint
        self.constraint_type = str(getattr(self.constraint, "constraint_type", ""))

        if self.constraint_type == "CMDP":
            budget = getattr(self.constraint, "budget", None)
            if budget is not None and self.cost_limit != budget:
                warnings.warn(
                    f"cost_limit={self.cost_limit} does not match CMDP "
                    f"cost_budget={budget}; using constraint.cost_budget.",
                )
                self.cost_limit = budget

        elif self.constraint_type == "PCTL":
            alpha = getattr(self.constraint, "alpha", None)
            if alpha is not None and self.cost_limit != alpha:
                warnings.warn(
                    f"cost_limit={self.cost_limit} does not match PCTL "
                    f"alpha={alpha}; using constraint.alpha.",
                )
                self.cost_limit = alpha

        elif self.constraint_type in ["LTL_SAFETY", "REACH_AVOID", "PROB"]:
            pass

        else:
            warnings.warn(
                f"Constraint type {constraint_type!r} not recognised; using "
                f"provided cost_limit={self.cost_limit}. Please double-check "
                "constraint semantics are compatible with cost-style CMDP algorithms.",
            )

        return env

    def _setup_buffer(self):
        self.rollout_buffer = CostRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_envs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
        )

    def _setup_model(self):
        super()._setup_model()
        self.cost_critic = self.policy.cost_critic  # type: ignore[assignment]

    def train(self, *args, **kwargs):
        self._last_cost_episode_start = [True]*self.n_envs
        self._cost_accumulator = np.zeros(self.n_envs, dtype=np.float32)
        if not hasattr(self, "_mean_ep_cost"):
            self._mean_ep_cost = None
            self.mean_ep_cost = 0.0
        super().train(*args, **kwargs)

    def _bootstrap_inflight_cost(self, last_cost_value=np.array(0.0)) -> float:
        """Cold-start estimate of the mean episode cost from in-flight trajectories."""
        # Accumulated raw step cost of the current episode + discounted cost-critic tail.
        est = self._cost_accumulator + last_cost_value
        if self.constraint_type in ("PCTL", "LTL_SAFETY", "REACH_AVOID"):
            # Episode cost is a [0, 1] indicator for these constraints.
            est = np.clip(est, 0.0, 1.0)
        return float(np.mean(est))

    def _update_mean_ep_cost(self, ep_costs: list, last_cost_value=np.array(0.0)) -> None:
        """Update the running per-episode EWMA estimate of the mean episode cost."""
        b = self.cost_ewma_beta
        prev = self._mean_ep_cost

        if len(ep_costs) > 0:
            # Per-episode cost expoentially-weighted moving average update
            for c in ep_costs:
                prev = float(c) if prev is None else (1.0 - b) * prev + b * float(c)
        elif prev is None:
            # Fall-back to a (biased, undercounting) in-flight bootstrap to avoid a spurious 0.0.
            prev = self._bootstrap_inflight_cost(last_cost_value)
        # If nothing terminated just use the current estimate to avoid double counting episodes

        self._mean_ep_cost = prev
        self.mean_ep_cost = prev

    def rollout(
        self, 
        step: int,
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1,
    ):
        steps = 0
        self.rollout_buffer.reset()
        self._last_obs = np.array(self._last_obs)
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
                actions, log_probs, values, cost_values = self.policy.predict_all(self.policy.noise_key, obs)

                actions = np.array(actions)
                log_probs = np.array(log_probs)
                values = np.array(values)
                cost_values = np.array(cost_values)

                new_obs, rewards, terminated, truncated, infos = self.env.step(self.prepare_act(actions, n_envs=self.n_envs))

                costs = np.array([info["constraint"]["step"].get("cost", 0.0) for info in infos])
                cost_dones = np.array([info["constraint"]["step"].get("cost_done", False) for info in infos])

                self._cost_accumulator += costs

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

                new_obs = np.array(new_obs)
                rewards = np.array(rewards)
            
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
                    log_probs,
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

                self._last_obs = new_obs
                self._last_episode_start = dones
                self._last_cost_episode_start = np.logical_or(dones, cost_dones)
                self._cost_accumulator[dones] = 0.0

                if logger:
                    for info in infos:
                        logger.add("train/rollout", info)

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

        self._update_mean_ep_cost(ep_costs, last_cost_value=last_cost_value)

        return ep_costs

class OnPolicyNaiveLagrangeAlgorithm(OnPolicyCostAlgorithm, ABC):
    """Abstract cost-aware on-policy algorithm with naive Lagrange dual updates."""

    def _update_lagrange_multiplier(self, mean_ep_cost: float):
        self.lagrangian_multiplier += self.lambda_lr * (mean_ep_cost - self.cost_limit)
        self.lagrangian_multiplier = max(0.0, self.lagrangian_multiplier)
        if self.lagrangian_upper_bound is not None:
            self.lagrangian_multiplier = min(self.lagrangian_multiplier, self.lagrangian_upper_bound)


    def rollout(
        self,
        step: int,
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1,
    ):
        super().rollout(
            step=step,
            logger=logger,
            tqdm_position=tqdm_position,
        )

        self._update_lagrange_multiplier(self.mean_ep_cost)
            