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
from masa.common.policies import PPOLagPolicy

class TRPOLag(TRPO):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        learning_rate: Union[float, optax.Schedule] = 3e-4,
        n_steps: int = 2048,
        n_critic_updates: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        fvp_sample_freq: int = 4,
        line_search_steps: int = 10,
        line_search_decay: float = 0.8,
        policy_class: type[BaseJaxPolicy] = PPOLagPolicy,
        policy_kwargs: Optional[dict[str, Any]] = None,
        # Lagrange parameters
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.01,
        lagrangian_upper_bound: Optional[float] = None,
    ):

        self.cost_limit = cost_limit
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound

        self.lagrangian_multiplier = float(
            max(lagrangian_multiplier_init, 0.0)
        )

        super().__init__(
            env=env,
            tensorboard_logdir=tensorboard_logdir,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            env_fn=env_fn,
            eval_env=eval_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            n_critic_updates=n_critic_updates,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            cg_iters=cg_iters,
            cg_damping=cg_damping,
            fvp_sample_freq=fvp_sample_freq,
            line_search_steps=line_search_steps,
            line_search_decay=line_search_decay,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
        )

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

    def _update_lagrange_multiplier(self, mean_ep_cost: float):
        self.lagrangian_multiplier += self.lambda_lr * (mean_ep_cost - self.cost_limit)
        self.lagrangian_multiplier = max(0.0, self.lagrangian_multiplier)
        if self.lagrangian_upper_bound is not None:
            self.lagrangian_multiplier = min(self.lagrangian_multiplier, self.lagrangian_upper_bound)

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

        # update lagrange multiplier after rollout
        if len(ep_costs) > 0:
            mean_ep_cost = float(np.mean(ep_costs))
            self._update_lagrange_multiplier(mean_ep_cost)

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