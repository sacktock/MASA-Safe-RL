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
        normalize_reward_advantages: bool = True,
        normalize_cost_advantages: bool = True,
        # CPO parameters
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        **kwargs,
    ):
        self.normalize_reward_advantages = normalize_reward_advantages
        self.normalize_cost_advantages = normalize_cost_advantages

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

    def _determine_case(
        self,
        cost_grads: jnp.ndarray,
        c: float,
        q: jnp.ndarray,
        r: jnp.ndarray,
        s: jnp.ndarray,
    ):

        if jnp.linalg.norm(cost_grads) <= 1e-8 and c < 0:
            A = jnp.zeros(1)
            B = jnp.zeros(1)
            optim_case = 4
        else:
            A = q - (r ** 2) / (s + 1e-8)
            B = 2.0 * self.target_kl - (c ** 2) / (s + 1e-8)

            if c < 0 and B < 0: # entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0: # part of the trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0: # entire trust region is infeasible
                optim_case = 1
            else: # entire trust region is infeasible
                optim_case = 0

        return optim_case, A, B

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

        if self.normalize_reward_advantages and len(reward_advantages) > 1:
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std() + 1e-8)

        if self.normalize_cost_advantages:
            cost_advantages = cost_advantages - cost_advantages.mean()

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

        def fvp(v):
            return self._fisher_vector_product(
                old_actor_params,
                v,
                self.policy.featurizer_state,
                self.policy.actor_state,
                fvp_observations,
                self.cg_damping,
            )

        fvp_observations = observations[::self.fvp_sample_freq]
        old_actor_params = self.policy.actor_state.params

        reward_loss_before, reward_grads = jax.value_and_grad(actor_reward_loss)(old_actor_params)
        reward_grads = self.flatten_params(reward_grads)

        x = self.conjugate_gradient(fvp, -reward_grads, self.cg_iters)
        assert jnp.all(jnp.isfinite(x)), "x is not finite"
        xHx = jnp.dot(x, fvp(x))
        assert xHx.item() >= 0, "xHx is negative"
        alpha = jnp.sqrt(2 * self.target_kl / (xHx + 1e-8))

        cost_loss_before, cost_grads = jax.value_and_grad(actor_cost_loss)(old_actor_params)
        cost_grads = self.flatten_params(cost_grads)

        p = self.conjugate_gradient(fvp, cost_grads, self.cg_iters)
        c = self.mean_ep_cost - self.cost_limit
        q = xHx
        r = jnp.dot(-reward_grads, p) 
        s = jnp.dot(cost_grads, p) 

        optim_case, A, B = self._determine_case(cost_grads, c, q, r, s)

        if optim_case in (3, 4):
            # feasible use usual TRPO step
            step_direction = alpha * x
            lambda_star = 1.0 / (alpha + 1e-8)
            nu_star = 0.0
        elif optim_case in (1, 2):
            # QCQP step
            A_safe = jnp.maximum(A, 0.0)
            B_safe = jnp.maximum(B, 1e-8)

            lambda_a = jnp.sqrt(A_safe / B_safe)
            lambda_b = jnp.sqrt(q / (2.0 * self.target_kl + 1e-8))

            eps_c = c + 1e-8

            if c < 0:
                lambda_a_star = jnp.clip(lambda_a, 0.0, r / eps_c)
                lambda_b_star = jnp.maximum(lambda_b, r / eps_c)
            else:
                lambda_a_star = jnp.maximum(lambda_a, r / eps_c)
                lambda_b_star = jnp.clip(lambda_b, 0.0, r / eps_c)

            f_a = -0.5 * (A_safe / (lambda_a_star + 1e-8) + B_safe * lambda_a_star) - r * c / (s + 1e-8)
            f_b = -0.5 * (q / (lambda_b_star + 1e-8) + 2.0 * self.target_kl * lambda_b_star)

            lambda_star = jnp.where(f_a >= f_b, lambda_a_star, lambda_b_star)
            nu_star = jnp.maximum(0.0, lambda_star * c - r) / (s + 1e-8)
            step_direction = (1.0 / (lambda_star + 1e-8)) * (x - nu_star * p)
        else:
            # Infeasible recovery: reduce cost
            lambda_star = 0.0
            nu_star = jnp.sqrt(2.0 * self.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        old_flat = self.flatten_params(old_actor_params)

        expected_improve = -jnp.dot(reward_grads, step_direction)
        expected_cost_diff = jnp.dot(cost_grads, step_direction)

        accepted = False
        last_kl = jnp.array(0.0)
        candidate_cost_diff = jnp.array(0.0)
        reward_improve = jnp.array(0.0)

        for i in range(self.line_search_steps):
            frac = self.line_search_decay ** i
            candidate = old_flat + frac * step_direction
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

            reward_ok = reward_improve > 0 if optim_case > 1 else True
            cost_ok = candidate_cost_diff <= max(-c, 0.0)
            kl_ok = kl < self.target_kl

            if reward_ok and cost_ok and kl_ok:
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
                "lambda_star": float(lambda_star),
                "nu_star": float(nu_star),
                "reward_grad_norm": float(jnp.linalg.norm(reward_grads)),
                "cost_grad_norm": float(jnp.linalg.norm(cost_grads)),
                "xHx": float(xHx),
                "actual_improve": float(reward_improve),
                "actual_cost_diff": float(candidate_cost_diff),
                "final_kl": float(last_kl),
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
