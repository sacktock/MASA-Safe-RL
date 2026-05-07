from __future__ import annotations

import numpy as np
import jax.random as jr
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional
from jax import jit
import jax
from functools import partial
from flax.training.train_state import TrainState

from masa.algorithms.ppo import PPO
from masa.common.base_class import BaseJaxPolicy
from masa.prob_shield.parameterized_policy import ParameterizedPPOPolicy
from masa.common.metrics import Stats, Dist

from tqdm.auto import tqdm

class ParameterizedPPO(PPO):

    def __init__(self, *args, policy_class: type[BaseJaxPolicy] = ParameterizedPPOPolicy, **kwargs):
        super().__init__(*args, policy_class=policy_class, **kwargs)

        self.margin_stats = {f"margin_{t}": Stats(prefix=f"margin_{t}") for t in [0, 50, 100, 150, 200]}
        self.margin_dists = {f"margin_{t}": Dist(prefix=f"margin_{t}") for t in [0, 50, 100, 150, 200]}

    def _validate_and_extract_action_specs(self):

        if not isinstance(self.action_space, spaces.Dict):
            raise TypeError(
                "ParameterizedPPO requires env.action_space to be gymnasium.spaces.Dict with keys "
                "['multi_discrete', 'box']. Got "
                f"{type(self.action_space).__name__}. "
                "Consider using the original PPO class instead."
            )

        if "multi_discrete" not in self.action_space.spaces or "box" not in self.action_space.spaces:
            raise KeyError(
                "ParameterizedPPO requires env.action_space = Dict({'multi_discrete': ..., 'box': ...}). "
                f"Got keys: {list(self.action_space.spaces.keys())}. "
                "Consider using the original PPO class instead."
            )

        md = self.action_space.spaces["multi_discrete"]
        bx = self.action_space.spaces["box"]

        if not isinstance(md, spaces.MultiDiscrete):
            raise TypeError(
                "ParameterizedPPO requires action_space['multi_discrete'] to be MultiDiscrete([n_actions, n_actions]). "
                f"Got {type(md).__name__}. "
                "Consider using the original PPO class instead."
            )

        if not isinstance(bx, spaces.Box):
            raise TypeError(
                "ParameterizedPPO requires action_space['box'] to be Box(low=0, high=1, shape=(max_successors,)). "
                f"Got {type(bx).__name__}. "
                "Consider using the original PPO class instead."
            )

        if md.nvec.ndim != 1 or md.nvec.shape[0] != 2:
            raise ValueError(
                "ParameterizedPPO expects action_space['multi_discrete'] to have nvec shape (2,), "
                f"but got nvec={md.nvec}."
            )

        if int(md.nvec[0]) != int(md.nvec[1]):
            raise ValueError(
                "ParameterizedPPO expects action_space['multi_discrete'] = MultiDiscrete([n_actions, n_actions]). "
                f"Got nvec={md.nvec}."
            )

        if bx.shape is None or len(bx.shape) != 1:
            raise ValueError(
                "ParameterizedPPO expects action_space['box'] to be 1D with shape (max_successors,). "
                f"Got shape={bx.shape}."
            )

        self.n_actions = int(md.nvec[0])
        self.max_successors = int(bx.shape[0])

        # Sanity check
        if np.any(bx.low != 0) or np.any(bx.high != 1):
            raise ValueError(
                "ParameterizedPPO expects action_space['box'] bounds to be exactly [0,1] for betas. "
                f"Got low(min)={float(np.min(bx.low))}, high(max)={float(np.max(bx.high))}."
            )

    def _setup_model(self):
        if hasattr(self, "policy") and self.policy is not None:
            return

        self._validate_and_extract_action_specs()

        if self.policy_kwargs is None:
            self.policy_kwargs = {}

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.n_actions,
            self.max_successors,
            **self.policy_kwargs,
        )

        self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

        self.featurizer = self.policy.featurizer  # type: ignore[assignment]
        self.actor = self.policy.actor            # type: ignore[assignment]
        self.critic = self.policy.critic          # type: ignore[assignment]

    @staticmethod
    @partial(jit, static_argnames=["normalize_advantage"])
    def _one_update(
        featurizer_state: TrainState,
        actor_state: TrainState,
        critic_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
    ):
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_critic_loss(featurizer_params, actor_params, critic_params):
            features = featurizer_state.apply_fn(featurizer_params, observations)
            dist = actor_state.apply_fn(actor_params, features)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy(actions)
            
            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob)
            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss

            # Critic loss
            critic_values = critic_state.apply_fn(critic_params, features).flatten()
            value_loss = vf_coef * ((returns - critic_values)**2).mean()

            total_loss = total_policy_loss + value_loss
            return total_loss, (total_policy_loss, value_loss)

        (loss, (pg_loss, vf_loss)), grads = jax.value_and_grad(actor_critic_loss, argnums=(0, 1, 2), has_aux=True)(
            featurizer_state.params, actor_state.params, critic_state.params
        )

        featurizer_state = featurizer_state.apply_gradients(grads=grads[0])
        actor_state = actor_state.apply_gradients(grads=grads[1])
        critic_state = critic_state.apply_gradients(grads=grads[2])

        return (featurizer_state, actor_state, critic_state), (pg_loss, vf_loss)

    def optimize(
        self,
        step: int, 
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1
    ):
        
        clip_range = self.clip_range_schedule(step)
        current_lr = self.lr_schedule(step)

        beta_stats = Stats(prefix="betas")

        with tqdm(
            total=self.n_epochs*self.n_steps//(self.batch_size//self.n_envs),
            desc="optimize",
            position=tqdm_position,
            leave=False,
            dynamic_ncols=True,
            colour="cyan",
        ) as pbar:

            for _ in range(self.n_epochs):
                self.key, subkey = jr.split(self.key)
                for rollout_data in self.rollout_buffer.get(subkey, self.batch_size//self.n_envs):

                    observations, actions, rewards, values, returns, advantages, old_log_probs = rollout_data

                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to int
                        actions = actions.flatten().astype(np.int32)

                    (self.policy.featurizer_state, self.policy.actor_state, self.policy.critic_state), (pg_loss, vf_loss) = \
                    self._one_update(
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        critic_state=self.policy.critic_state,
                        observations=observations,
                        actions=actions,
                        advantages=advantages,
                        returns=returns,
                        old_log_prob=old_log_probs,
                        clip_range=clip_range,
                        ent_coef=self.ent_coef,
                        vf_coef=self.vf_coef,
                        normalize_advantage=self.normalize_advantage,
                    )

                    pbar.update(1)

                    beta_stats.update(actions[:, 2:])
                
        if logger:
            logger.add("train/stats", {
                "betas": beta_stats,
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
                "clip_range": float(clip_range),
                "lr": float(current_lr)
            })
            logger.add("train/stats", {k: v for k, v in self.margin_stats.items() if v.n != 0})
            logger.add("train/stats", {k: v for k, v in self.margin_dists.items() if v.n != 0})
            self.margin_stats = {f"margin_{t}": Stats(prefix=f"margin_{t}") for t in [0, 50, 100, 150, 200]}
            # do not reinstatiate self.margin_dists: we keep previous samples
            

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
                actions, log_probs, values = self.policy.predict_all(self.policy.noise_key, obs)

                actions = np.array(actions)
                log_probs = np.array(log_probs)
                values = np.array(values)

                new_obs, rewards, terminated, truncated, infos = self.env.step(self.prepare_act(actions, n_envs=self.n_envs))

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

                    if terminated[idx] or truncated[idx]:
                        dones[idx] = True

                self.rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_start,
                    values,
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
                        logger.add("train/stats", {k: v for k, v in info.items() if k in ["margin_penalty", "proj_penalty"]})
                        for t in [0, 50, 100, 200, 250]:
                            if f"margin_{t}" in info:
                                self.margin_stats[f"margin_{t}"].update(info[f"margin_{t}"])
                                self.margin_dists[f"margin_{t}"].update(info[f"margin_{t}"])

        assert isinstance(self._last_obs, np.ndarray) 
        final_obs = self.prepare_obs(self._last_obs, n_envs=self.n_envs)
        feats = self.featurizer.apply(self.policy.featurizer_state.params, final_obs)
        last_value = np.array(
            self.critic.apply(
                self.policy.critic_state.params,
                feats,
            ).flatten()
        )

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value, done=self._last_episode_start)

    def prepare_act(self, act: Any, n_envs: int = 1) -> np.ndarray:

        if isinstance(self.action_space, spaces.Dict):

            if isinstance(act, dict):
                md = act.get("multi_discrete", None)
                bx = act.get("box", None)
                if md is None or bx is None:
                    raise KeyError(
                        "ParameterizedPPO.prepare_act expected dict with keys "
                        "['multi_discrete', 'box'], got keys "
                        f"{list(act.keys())}."
                    )

                md = np.asarray(md)
                bx = np.asarray(bx, dtype=np.float32)

                md = md.reshape(n_envs, 2)
                bx = bx.reshape(n_envs, self.max_successors)

                i = np.clip(md[:, 0], 0, self.n_actions - 1).astype(np.float32)
                j = np.clip(md[:, 1], 0, self.n_actions - 1).astype(np.float32)
                betas = np.clip(bx, 0.0, 1.0).astype(np.float32)

                flat = np.concatenate([i[:, None], j[:, None], betas], axis=1).astype(np.float32)

                if n_envs == 1:
                    return flat[0]
                return flat

            flat = np.array(act, dtype=np.float32, copy=True).reshape(n_envs, 2 + self.max_successors)
            flat[:, 0] = np.clip(flat[:, 0], 0, self.n_actions - 1)
            flat[:, 1] = np.clip(flat[:, 1], 0, self.n_actions - 1)
            flat[:, 2:] = np.clip(flat[:, 2:], 0.0, 1.0)

            if n_envs == 1:
                return flat[0]
            return flat

        return super().prepare_act(act, n_envs=n_envs)

