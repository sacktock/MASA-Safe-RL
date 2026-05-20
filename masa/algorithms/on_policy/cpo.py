from __future__ import annotations
import jax.random as jr
import jax.numpy as jnp
import optax
from jax import jit
import jax
from functools import partial
from contextlib import nullcontext
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Callable, Union, Any
from tqdm.auto import tqdm
from masa.common.base_class import BaseJaxPolicy
from masa.common.metrics import TrainLogger
from masa.algorithms.on_policy import TRPO
from masa.algorithms.on_policy.abstract_classes import OnPolicyCostAlgorithm
from masa.common.policies import PPOLagPolicy

class CPO(OnPolicyCostAlgorithm, TRPO):

    def __init__(
        self,
        *args,
        policy_class: type[BaseJaxPolicy] = PPOLagPolicy,
        # CPO parameters
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        **kwargs,
    ):

        self.cost_limit = cost_limit
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda

        super().__init__(*args, policy_class=policy_class, **kwargs)

    @staticmethod
    @jit
    def _value_update(
        featurizer_state: TrainState,
        critic_state: TrainState,
        cost_critic_state: TrainState,
        observations: jnp.ndarray,
        returns: jnp.ndarray,
        cost_returns: jnp.ndarray,
        vf_coef: float,
    ):
        def critic_loss(featurizer_params, critic_params, cost_critic_params):
            features = featurizer_state.apply_fn(featurizer_params, observations)
            # Critic loss
            critic_values = critic_state.apply_fn(critic_params, features).flatten()
            cost_critic_values = cost_critic_state.apply_fn(cost_critic_params, features).flatten()
            value_loss = vf_coef * (
                ((returns - critic_values)**2).mean() +
                ((cost_returns - cost_critic_values)**2).mean()
            )
            return value_loss

        loss, grads = jax.value_and_grad(critic_loss, argnums=(0, 1, 2))(
            featurizer_state.params, critic_state.params, cost_critic_state.params,
        )

        featurizer_state = featurizer_state.apply_gradients(grads=grads[0])
        critic_state = critic_state.apply_gradients(grads=grads[1])
        cost_critic_state = cost_critic_state.apply_gradients(grads=grads[2])

        return (featurizer_state, critic_state, cost_critic_state), loss

    @staticmethod
    @jit
    def _cost_surrogate_loss(
        actor_params: Any,
        featurizer_state: TrainState,
        actor_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        cost_advantages: jnp.ndarray,
        old_log_prob: jnp.ndarray,
    ):
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_params, features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio = jnp.exp(log_prob - old_log_prob)
        return (ratio * cost_advantages).mean() # positive - minimize cost

    def optimize(
        self,
        step: int, 
        logger: Optional[TrainLogger] = None,
        tqdm_position: int = 1
    ):

        self.key, subkey = jr.split(self.key)
        rollout_data = next(self.rollout_buffer.get(subkey, self.n_steps))
        observations, actions, rewards, costs, values, cost_values, returns, cost_returns, reward_advantages, cost_advantages, old_log_probs = rollout_data

        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to int
            actions = actions.flatten().astype(np.int32)

        if self.normalize_advantage and len(reward_advantages) > 1:
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std() + 1e-8)

        old_actor_params = self.policy.actor_state.params

        def actor_reward_loss(actor_params):
            return self._surrogate_loss(
                actor_params,
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
                actions,
                reward_advantages,
                old_log_probs,
                self.ent_coef,
            )

        def actor_cost_loss(actor_params):
            return self._cost_surrogate_loss(
                actor_params,
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
                actions,
                cost_advantages,
                old_log_probs,
            )

        fvp_observations = observations[::self.fvp_sample_freq]

        def fvp(v):
            return self._fisher_vector_product(
                old_actor_params,
                v,
                self.policy.featurizer_state,
                self.policy.actor_state,
                fvp_observations,
                self.cg_damping,
            )

        reward_loss_before, reward_grads = jax.value_and_grad(actor_reward_loss)(old_actor_params)
        cost_loss_before, cost_grads = jax.value_and_grad(actor_cost_loss)(old_actor_params)

        reward_grads = self.flatten_params(reward_grads)
        cost_grads = self.flatten_params(cost_grads)

        x = self.conjugate_gradient(fvp, -reward_grads, self.cg_iters)
        p = self.conjugate_gradient(fvp, cost_grads, self.cg_iters)

        c = self.mean_ep_cost - self.cost_limit

        denom = jnp.dot(cost_grads, p)

        lambda_cpo = jnp.where(
            denom > 0,
            jnp.maximum(0.0, (jnp.dot(cost_grads, x) + c)/ (denom + 1e-8)),
            0.0
        )

        step_direction = x - lambda_cpo * p

        shs = 0.5 * jnp.dot(
            step_direction,
            fvp(step_direction),
        )

        step_size = jnp.sqrt(
            self.target_kl / (shs + 1e-8)
        )

        full_step = step_direction * step_size

        old_flat = self.flatten_params(old_actor_params)

        expected_improve = -jnp.dot(reward_grads, full_step)
        expected_cost_diff = jnp.dot(cost_grads, full_step)

        accepted = False

        for i in range(self.line_search_steps):
            frac = self.line_search_decay ** i
            candidate = old_flat + frac * full_step
            candidate_params = self.unflatten_params(
                candidate,
                old_actor_params,
            )
            candidate_reward_loss = actor_reward_loss(candidate_params)
            candidate_cost_loss = actor_cost_loss(candidate_params)
            kl = self._mean_kl(
                old_actor_params, 
                candidate_params, 
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
            )
            reward_improve = reward_loss_before - candidate_reward_loss 
            candidate_cost_diff = candidate_cost_loss - cost_loss_before
            last_kl = kl
            if reward_improve > 0 and candidate_cost_diff <= max(-c, 0.0) and kl < self.target_kl:
                self.policy.actor_state = self.policy.actor_state.replace(params=candidate_params)
                accepted = True
                break

        with tqdm(
            total=self.n_critic_updates,
            desc="optimize_critic",
            position=tqdm_position,
            leave=False,
            dynamic_ncols=True,
            colour="cyan",
        ) as pbar:

            for _ in range(self.n_critic_updates):

                (self.policy.featurizer_state, self.policy.critic_state, self.policy.cost_critic_state), vf_loss = \
                self._value_update(
                    featurizer_state=self.policy.featurizer_state,
                    critic_state=self.policy.critic_state,
                    cost_critic_state=self.policy.cost_critic_state,
                    observations=observations,
                    returns=returns,
                    cost_returns=cost_returns,
                    vf_coef=self.vf_coef,
                )

                pbar.update(1)

        if logger:
            logger.add(
                "train/stats",{
                "reward_policy_loss": float(reward_loss_before),
                "cost_policy_loss": float(cost_loss_before),
                "value_loss": float(vf_loss),
                "kl": float(kl),
                "accepted": float(accepted),
                "expected_improve": float(expected_improve),
                "expected_cost_diff": float(expected_cost_diff),
                "step_size": float(step_size),
                "reward_grad_norm": float(jnp.linalg.norm(reward_grads)),
                "cost_grad_norm": float(jnp.linalg.norm(cost_grads)),
                "xHx": float(shs * 2.0),
                "final_kl": float(last_kl),
                "candidate_cost_diff": float(candidate_cost_diff),
                "lr": float(self.lr_schedule(step)),
                "mean_cost_advantage": float(cost_advantages.mean()),
                "mean_cost_return": float(cost_returns.mean()),
            })

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
            self.mean_ep_cost = float(np.mean(ep_costs))
        else:
            self.mean_ep_cost = 0.0
