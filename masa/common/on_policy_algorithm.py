from __future__ import annotations
from contextlib import contextmanager
import jax.random as jr
import jax.numpy as jnp
import optax
from jax import jit
import jax
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from masa.common.base_class import BaseAlgorithm, BaseJaxPolicy
from tqdm import tqdm
from typing import Any, Optional, TypeVar, Union, Callable
from masa.common.wrappers import DummyVecWrapper, VecEnvWrapperBase, NormWrapper, VecNormWrapper, OneHotObsWrapper, FlattenDictObsWrapper, is_wrapped
from masa.common.buffers import RolloutBuffer
from masa.common.policies import PPOPolicy
from tqdm.auto import tqdm

@contextmanager
def nullcontext(arg=None):
    yield arg

class OnPolicyAlgorithm(BaseAlgorithm):

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
        use_tqdm_rollout: bool = False,
        learning_rate: Union[float, optax.Schedule] = 3e-4,
        n_steps: int = 16,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        policy_class: type[BaseJaxPolicy] = PPOPolicy,
        policy_kwargs: Optional[dict[str, Any]] = None,
    ):

        env = self._wrap_env(env)

        super().__init__(
            env, 
            tensorboard_logdir=tensorboard_logdir,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            supported_action_spaces=(spaces.Discrete, spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete, spaces.Dict),
            supported_observation_spaces=(spaces.Box,),
            env_fn=env_fn,
            eval_env=eval_env,
        )

        if policy_kwargs is None:
            policy_kwargs = {}

        self.n_envs = self.env.n_envs

        if isinstance(learning_rate, float):
            self.lr_schedule = optax.schedules.constant_schedule(learning_rate)
        else:
            assert callable(learning_rate), f"learning_rate for class PPO must be float or optax.Schedule not {type(learning_rate)}"
            self.lr_schedule = learning_rate

        self.use_tqdm_rollout = use_tqdm_rollout
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
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

    def _wrap_env(self, env: gym.Env):

        if is_wrapped(env, VecEnvWrapperBase):
            assert is_wrapped(env, OneHotObsWrapper) or not isinstance(env.observation_space, spaces.Discrete), \
            "Before wrapping your environment in a DummyVecWrapper / VecWrapper, please wrap your environment in a OneHotObsWrapper if it has type(observation_space)__name__ = Discrete"
            assert not isinstance(env.observation_space, spaces.Dict), \
            "Before wrapping your environment in a DummyVecWrapper / VecWrapper, please wrap your environment in a FlattenDictObsWrapper if it has type(observation_space)__name__ = Dict"

        if isinstance(env.observation_space, (spaces.Dict, spaces.Discrete)):
            if not is_wrapped(env, OneHotObsWrapper):
                env = OneHotObsWrapper(env)
            if isinstance(env.observation_space, spaces.Dict):
                env = FlattenDictObsWrapper(env)

        if not isinstance(env, VecEnvWrapperBase) and not is_wrapped(env, VecEnvWrapperBase):
            env = DummyVecWrapper(env)

        return env

    def _get_eval_env(self) -> gym.Env:
        self._eval_env = super()._get_eval_env()

        if is_wrapped(self._eval_env, VecEnvWrapperBase):
            raise TypeError(
                "Please do not wrap your eval environment in a DummyVecWrapper / VecWrapper"
            )

        if isinstance(self._eval_env.observation_space, (spaces.Dict, spaces.Discrete)):
            if not is_wrapped(self._eval_env, OneHotObsWrapper):
                self._eval_env = OneHotObsWrapper(self._eval_env)
            if isinstance(self._eval_env.observation_space, spaces.Dict):
                self._eval_env = FlattenDictObsWrapper(self._eval_env)

        train_env = self.env

        if is_wrapped(train_env, (VecNormWrapper, NormWrapper)):
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

    def train(self, *args, **kwargs):
        self._last_episode_start = [True]*self.n_envs
        super().train(*args, **kwargs)

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

    def act(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> Union[int, np.ndarray]:
        obs = self.prepare_obs(obs, n_envs=1)
        action = self.policy.forward(key, obs, deterministic=deterministic)
        return self.prepare_act(action, n_envs=1)

    def prepare_act(self, act: np.ndarray, n_envs: int = 1) -> Union[int, np.ndarray]:
        if isinstance(self.action_space, spaces.Box):
            act = np.clip(act, self.action_space.low, self.action_space.high, dtype=np.float32)
            act = act.reshape(n_envs, *self.action_space.shape)
            if n_envs == 1:
                act = np.squeeze(act, axis=0)
        elif isinstance(self.action_space, spaces.Discrete):
            act = np.array(act, dtype=np.int64)
            act = act.reshape(n_envs)
            if n_envs == 1:
                act = int(act.item())
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            act = np.array(act, dtype=np.int64)
            act = act.reshape(n_envs, len(self.action_space.nvec))
            if n_envs == 1:
                act = act[0]
        elif isinstance(self.action_space, spaces.MultiBinary):
            n = self.action_space.n
            act = np.array(act, dtype=np.int64)
            act = act.reshape(n_envs, n)
            if n_envs == 1:
                act = act[0]
        else:
            raise NotImplementedError(f"Unexpected action space {type(self.action_space).__name__}")
        return act
        
    def prepare_obs(self, obs: np.ndarray, n_envs: int = 1) -> np.ndarray:
        obs = np.array(obs)
        return obs.reshape(n_envs, *self.observation_space.shape)


        
