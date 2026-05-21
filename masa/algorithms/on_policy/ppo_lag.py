from __future__ import annotations
import jax.random as jr
import jax.numpy as jnp
import optax
from jax import jit
import jax
from functools import partial
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Tuple, Optional, Union, Callable
from masa.common.base_class import BaseJaxPolicy
from masa.algorithms.on_policy import PPO
from masa.algorithms.on_policy.abstract_classes import OnPolicyNaiveLagrangeAlgorithm
from masa.common.policies import PPOLagPolicy
from masa.common.metrics import TrainLogger
from tqdm.auto import tqdm

class PPOLag(OnPolicyNaiveLagrangeAlgorithm, PPO):

    def __init__(
        self,
        *args,
        policy_class: type[BaseJaxPolicy] = PPOLagPolicy,
        normalize_reward_advantages: bool = True,
        normalize_cost_advantages: bool = True,
        # Lagrange parameters
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.01,
        lagrangian_upper_bound: Optional[float] = None,
        **kwargs,
        
    ):
        self.normalize_reward_advantages = normalize_reward_advantages
        self.normalize_cost_advantages = normalize_cost_advantages

        self.cost_limit = cost_limit
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound

        self.lagrangian_multiplier = float(
            max(lagrangian_multiplier_init, 0.0)
        )

        super().__init__(*args, policy_class=policy_class, **kwargs)

    @staticmethod
    @hit
    def _one_update(
        featurizer_state: TrainState,
        actor_state: TrainState,
        critic_state: TrainState,
        cost_critic_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        cost_returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
    ):
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_critic_loss(featurizer_params, actor_params, critic_params, cost_critic_params):
            features = featurizer_state.apply_fn(featurizer_params, observations)
            dist = actor_state.apply_fn(actor_params, features)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

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
            cost_critic_values = cost_critic_state.apply_fn(cost_critic_params, features).flatten()
            value_loss = vf_coef * (
                ((returns - critic_values)**2).mean() +
                ((cost_returns - cost_critic_values)**2).mean()
            )

            total_loss = total_policy_loss + value_loss
            return total_loss, (total_policy_loss, value_loss)

        (loss, (pg_loss, vf_loss)), grads = jax.value_and_grad(actor_critic_loss, argnums=(0, 1, 2, 3), has_aux=True)(
            featurizer_state.params, actor_state.params, critic_state.params, cost_critic_state.params
        )

        featurizer_state = featurizer_state.apply_gradients(grads=grads[0])
        actor_state = actor_state.apply_gradients(grads=grads[1])
        critic_state = critic_state.apply_gradients(grads=grads[2])
        cost_critic_state = cost_critic_state.apply_gradients(grads=grads[3])

        return (featurizer_state, actor_state, critic_state, cost_critic_state), (pg_loss, vf_loss)

    def optimize(
        self,
        step: int, 
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1
    ):
        
        clip_range = self.clip_range_schedule(step)
        current_lr = self.lr_schedule(step)

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

                    observations, actions, rewards, costs, values, cost_values, returns, cost_returns, reward_advantages, cost_advantages, old_log_probs = rollout_data

                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to int
                        actions = actions.flatten().astype(np.int32)

                    if self.normalize_reward_advantages and len(reward_advantages) > 1:
                        reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std() + 1e-8)

                    if self.normalize_cost_advantages:
                        cost_advantages = cost_advantages - cost_advantages.mean()

                    advantages = (reward_advantages - self.lagrangian_multiplier * cost_advantages) / (1.0 + self.lagrangian_multiplier)

                    (self.policy.featurizer_state, self.policy.actor_state, self.policy.critic_state, self.policy.cost_critic_state), (pg_loss, vf_loss) = \
                    self._one_update(
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        critic_state=self.policy.critic_state,
                        cost_critic_state=self.policy.cost_critic_state,
                        observations=observations,
                        actions=actions,
                        advantages=advantages,
                        returns=returns,
                        cost_returns=cost_returns,
                        old_log_prob=old_log_probs,
                        clip_range=clip_range,
                        ent_coef=self.ent_coef,
                        vf_coef=self.vf_coef,
                    )

                    pbar.update(1)
                
        if logger:
            logger.add("train/stats", {
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
                "clip_range": float(clip_range),
                "lr": float(current_lr),
                "lagrangian_multiplier": float(self.lagrangian_multiplier),
                "mean_cost_advantage": float(cost_advantages.mean()),
                "mean_cost_return": float(cost_returns.mean()),
            })