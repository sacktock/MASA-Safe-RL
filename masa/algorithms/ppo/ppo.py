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
from masa.common.base_class import BaseAlgorithm, BaseJaxPolicy
from masa.algorithms.ppo.buffers import RolloutBuffer
from masa.algorithms.ppo.policies import PPOPolicy
from masa.common.wrappers import DummyVecWrapper, VecEnvWrapperBase, NormWrapper, VecNormWrapper
from tqdm import tqdm

class PPO(BaseAlgorithm):

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
            supported_action_spaces=(spaces.Discrete, spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete),
            supported_observation_spaces=(spaces.Discrete, spaces.Box, spaces.Dict),
            env_fn=env_fn,
            eval_env=eval_env,
        )

        if normalize_advantage:
            assert batch_size > 1, "batch_size must be > 1 when normalize_advantage = True"

        if policy_kwargs is None:
            policy_kwargs = {}

        if not isinstance(self.env, VecEnvWrapperBase):
            self.env = DummyVecWrapper(self.env)

        self.n_envs = self.env.n_envs

        if isinstance(learning_rate, float):
            self.lr_schedule = optax.schedules.constant_schedule(learning_rate)
        else:
            assert isinstance(learning_rate, optax.Schedule), f"learning_rate for class PPO must be float or optax.Schedule not {learning_rate}"
            self.lr_schedule = learning_rate

        if isinstance(clip_range, float):
            self.clip_range_schedule = optax.schedules.constant_schedule(clip_range)
        else:
            assert isinstance(clip_range, optax.Schedule), f"clip_range for class PPO must be float or optax.Schedule not {clip_range}"
            self.clip_range_schedule = clip_range

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

        self._setup_buffer()
        self._setup_model()

    def _setup_buffer(self):
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_envs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _setup_model(self):
        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                **self.policy_kwargs
            )

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            self.featurizer = self.policy.featurizer # type: ignore[assignment]
            self.actor = self.policy.actor  # type: ignore[assignment]
            self.critic = self.policy.critic  # type: ignore[assignment]
    
    def _get_eval_env(self) -> gym.Env:
        if self._eval_env is not None:
            eval_env = self._eval_env
        elif getattr(self, "_env_fn", None) is not None:
            eval_env = self._env_fn()
            self._eval_env = eval_env
        else:
            raise RuntimeError(
                "Cannot construct eval env; please pass env_fn or eval_env."
            )

        assert not isinstance(self._eval_env, VecEnvWrapperBase), "Please do not wrap your eval environment in a DummyVecEnvWrapper / VecEnvWrapper, this will interfere with the underlying evalutaion code!"

        train_env = self.env
        if isinstance(train_env, VecNormWrapper):
            train_norm_env: VecNormWrapper = train_env

            if not isinstance(self._eval_env, NormWrapper):
                self._eval_env = NormWrapper(
                    env=self._eval_env,  # already VecEnvWrapperBase
                    norm_obs=train_norm_env.norm_obs,
                    norm_rew=train_norm_env.norm_rew,
                    training=False,  # important: do not update stats during eval
                    clip_obs=train_norm_env.clip_obs,
                    clip_rew=train_norm_env.clip_rew,
                    gamma=train_norm_env.gamma,
                    eps=train_norm_env.eps,
                )
            else:
                self._eval_env.training = False

            self._eval_env: NormWrapper = self._eval_env

            self._eval_env.obs_rms = train_norm_env.obs_rms.copy()
            self._eval_env.rew_rms = train_norm_env.rew_rms.copy()

        return self._eval_env

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

    def train(self, *args, **kwargs):
        self._last_episode_start = [True]*self.n_envs
        super().train(*args, **kwargs)

    def optimize(self, step: int, logger: Optional[TrainLogger] = None):
        
        clip_range = self.clip_range_schedule(step)
        current_lr = self.lr_schedule(step)

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
                
        if logger:
            logger.add("train/stats", {
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
                "clip_range": float(clip_range),
                "lr": float(current_lr)
            })
                
    def rollout(self, step: int, logger: Optional[TrainLogger] = None):
        steps = 0
        self.rollout_buffer.reset()
        self._last_obs = np.array(self._last_obs)
        self._last_episode_start = np.array(self._last_episode_start)

        with tqdm(total=self.n_steps) as pbar:
            while steps < self.n_steps:
                self.policy.reset_noise()

                obs = self.prepare_obs(self._last_obs)
                actions, log_probs, values = self.policy.predict_all(self.policy.noise_key, obs)

                actions = np.array(actions)
                log_probs = np.array(log_probs)
                values = np.array(values)

                new_obs, rewards, terminated, truncated, infos = self.env.step(self.prepare_act(actions))

                new_obs = np.array(new_obs)
                rewards = np.array(rewards)
            
                steps += 1
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

        assert isinstance(self._last_obs, np.ndarray) 
        final_obs = self.prepare_obs(self._last_obs)
        feats = self.featurizer.apply(self.policy.featurizer_state.params, final_obs)
        last_value = np.array(
            self.critic.apply(
                self.policy.critic_state.params,
                feats,
            ).flatten()
        )

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value, done=self._last_episode_start)

    def act(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False):
        obs = self.prepare_obs(obs)
        action = self.policy.forward(key, obs, deterministic=deterministic)
        return self.prepare_act(action)

    def prepare_act(self, act: np.ndarray):
        if isinstance(self.action_space, spaces.Box):
            act = np.clip(act, self.action_space.low, self.action_space.high, dtype=np.float32)
            act = act.reshape(self.n_envs, *self.action_space.shape)
            if self.n_envs == 1:
                act = np.squeeze(act, axis=0)
        if isinstance(self.action_space, spaces.Discrete):
            act = np.array(act, dtype=np.int64)
            act = act.reshape(self.n_envs)
            if self.n_envs == 1:
                act = int(act.item())
        return act
        
    def prepare_obs(self, obs: np.ndarray):
        obs = np.array(obs)
        return obs.reshape(self.n_envs, *self.observation_space.shape)

    @property
    def train_ratio(self):
        return self.n_steps * self.n_envs
