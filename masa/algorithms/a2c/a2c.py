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
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.base_class import BaseJaxPolicy
from masa.common.on_policy_algorithm import OnPolicyAlgorithm
from masa.common.policies import PPOPolicy
from tqdm.auto import tqdm

class A2C(OnPolicyAlgorithm):

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
        learning_rate: Union[float, optax.Schedule] = 3e-4,
        n_steps: int = 16,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        policy_class: type[BaseJaxPolicy] = PPOPolicy,
        policy_kwargs: Optional[dict[str, Any]] = None,
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
            use_tqdm_rollout=False, # Turn off tqdm progress bar for rollout
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs
        )

        if normalize_advantage:
            assert n_steps * self.n_envs > 1, "n_steps * n_envs must be > 1 when normalize_advantage = True"

        self.normalize_advantage = normalize_advantage

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
            entropy = dist.entropy()

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = jnp.mean(-entropy)

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
        tqdm_position: int = 1 # unused
    ):

        current_lr = self.lr_schedule(step)

        self.key, subkey = jr.split(self.key)
        for rollout_data in self.rollout_buffer.get(subkey, None):
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
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                normalize_advantage=self.normalize_advantage,
            )

        if logger:
            logger.add("train/stats", {
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
                "lr": float(current_lr)
            })

    @property
    def train_ratio(self):
        return self.n_steps * self.n_envs