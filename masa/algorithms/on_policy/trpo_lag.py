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
from masa.common.buffers import CostRolloutBuffer
from masa.common.metrics import TrainLogger
from masa.algorithms.on_policy import TRPO
from masa.algorithms.on_policy.abstract_classes import OnPolicyNaiveLagrangeAlgorithm
from masa.common.policies import PPOLagPolicy

class TRPOLag(OnPolicyNaiveLagrangeAlgorithm, TRPO):

    def __init__(
        self,
        *args,
        # Lagrange parameters
        policy_class: type[BaseJaxPolicy] = PPOLagPolicy,
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.01,
        lagrangian_upper_bound: Optional[float] = None,
        **kwargs,
    ):

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

        advantages = (reward_advantages - self.lagrangian_multiplier * cost_advantages) / (1.0 + self.lagrangian_multiplier)

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_actor_params = self.policy.actor_state.params

        def actor_loss(actor_params):
            return self._surrogate_loss(
                actor_params,
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
                actions,
                advantages,
                old_log_probs,
                self.ent_coef,
            )

        loss_before, grads = jax.value_and_grad(actor_loss)(old_actor_params)

        flat_grads = self.flatten_params(grads)

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

        step_direction = self.conjugate_gradient(
            fvp,
            -flat_grads,
            self.cg_iters,
        )

        shs = 0.5 * jnp.dot(
            step_direction,
            fvp(step_direction),
        )

        step_size = jnp.sqrt(
            self.target_kl / (shs + 1e-8)
        )

        full_step = step_direction * step_size

        old_flat = self.flatten_params(old_actor_params)

        expected_improve = -jnp.dot(flat_grads, full_step)

        accepted = False

        for i in range(self.line_search_steps):
            frac = self.line_search_decay ** i
            candidate = old_flat + frac * full_step
            candidate_params = self.unflatten_params(
                candidate,
                old_actor_params,
            )
            loss = actor_loss(candidate_params)
            kl = self._mean_kl(
                old_actor_params, 
                candidate_params, 
                self.policy.featurizer_state,
                self.policy.actor_state,
                observations,
            )
            improve = loss_before - loss
            last_kl = kl
            if improve > 0 and kl < self.target_kl:
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
                "policy_loss": float(loss_before),
                "value_loss": float(vf_loss),
                "kl": float(kl),
                "accepted": float(accepted),
                "expected_improve": float(expected_improve),
                "step_size": float(step_size),
                "grad_norm": float(jnp.linalg.norm(flat_grads)),
                "xHx": float(shs * 2.0),
                "final_kl": float(last_kl),
                "lr": float(self.lr_schedule(step)),
                "lagrangian_multiplier": float(self.lagrangian_multiplier),
                "mean_cost_advantage": float(cost_advantages.mean()),
                "mean_cost_return": float(cost_returns.mean()),
            })