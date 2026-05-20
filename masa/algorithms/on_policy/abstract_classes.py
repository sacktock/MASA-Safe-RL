from __future__ import annotations
from abc import ABC
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

                for info in infos:
                    if "episode" in info["constraint"]:
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
                    values,
                    cost_values,
                    log_probs,
                )

                if np.any(dones):
                    reset_obs, _ = self.env.reset_done(dones)
                    for i, done in enumerate(dones):
                        if done and reset_obs[i] is not None:
                            new_obs[i] = reset_obs[i]

                self._last_obs = new_obs
                self._last_episode_start = dones

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

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value, last_cost_value=last_cost_value, done=self._last_episode_start)

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
        ep_costs = super().rollout(
            step=step,
            logger=logger,
            tqdm_position=tqdm_position,
        )

        if len(ep_costs) > 0:
            mean_ep_cost = float(np.mean(ep_costs))
            self._update_lagrange_multiplier(mean_ep_cost)
            