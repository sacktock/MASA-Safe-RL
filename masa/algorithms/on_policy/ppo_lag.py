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
from masa.common.buffers import CostRolloutBuffer
from masa.algorithms.on_policy import PPO
from masa.common.policies import PPOPolicy, Critic
from masa.common.metrics import TrainLogger
from tqdm.auto import tqdm

class PPOLagPolicy(PPOPolicy):

    def build(self, key: jax.Array, lr_schedule: Union[optax.Schedule, float], max_grad_norm: float) -> jax.Array:
        key = super().build(key, lr_schedule, max_grad_norm)
        key, cost_critic_key = jr.split(key, 2)

        optimizer = self.optimizer_class(
            learning_rate=lr_schedule,
            **self.optimizer_kwargs,
        )
        
        obs = jnp.array([self.observation_space.sample()])
        obs = self.featurizer.apply(self.featurizer_state.params, obs)

        self.cost_critic = Critic(
            net_arch=self.critic_net_arch,
            activation_fn=self.activation_fn,
        )

        self.cost_critic_state = TrainState.create(
            apply_fn=self.cost_critic.apply,
            params=self.cost_critic.init(cost_critic_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.cost_critic.apply = jit(self.cost_critic.apply)

        return key

    def predict_all(self, key: jax.Array, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_all(key, self.featurizer_state, self.actor_state, self.critic_state, self.cost_critic_state, observation)

    @staticmethod
    @jit
    def _predict_all(
        key: jax.Array, featurizer_state: TrainState, actor_state: TrainState, critic_state: TrainState, cost_critic_state: TrainState, observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_state.params, features)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = critic_state.apply_fn(critic_state.params, features).flatten()
        cost_values = cost_critic_state.apply_fn(cost_critic_state.params, features).flatten()
        return actions, log_probs, values, cost_values

class PPOLag(PPO):

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
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, optax.Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
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
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
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
    @partial(jit, static_argnames=["normalize_advantage"])
    def _one_update(
        featurizer_state: TrainState,
        actor_state: TrainState,
        critic_state: TrainState,
        cost_critic_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        reward_advantages: jnp.ndarray,
        cost_advantages: jnp.ndarray,
        returns: jnp.ndarray,
        cost_returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        lagrangian_multiplier: float,
        normalize_advantage: bool = True,
    ):
        advantages = (reward_advantages - lagrangian_multiplier * cost_advantages) / (1.0 + lagrangian_multiplier)

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
                ((returns - critic_values)**2).mean()
                + ((cost_returns - cost_critic_values)**2).mean()
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

                    (self.policy.featurizer_state, self.policy.actor_state, self.policy.critic_state, self.policy.cost_critic_state), (pg_loss, vf_loss) = \
                    self._one_update(
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        critic_state=self.policy.critic_state,
                        cost_critic_state=self.policy.cost_critic_state,
                        observations=observations,
                        actions=actions,
                        reward_advantages=reward_advantages,
                        cost_advantages=cost_advantages,
                        returns=returns,
                        cost_returns=cost_returns,
                        old_log_prob=old_log_probs,
                        clip_range=clip_range,
                        ent_coef=self.ent_coef,
                        vf_coef=self.vf_coef,
                        lagrangian_multiplier=self.lagrangian_multiplier,
                        normalize_advantage=self.normalize_advantage,
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