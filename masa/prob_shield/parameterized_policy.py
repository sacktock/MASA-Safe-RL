import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from dataclasses import dataclass
import tensorflow_probability.substrates.jax as tfp
from masa.common.base_class import BaseJaxPolicy
from flax.training.train_state import TrainState
import optax
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import flax.linen as nn
from gymnasium import spaces
from masa.common.layers import Identity, NatureCNN, Flatten
from masa.common.policies import Critic
from typing import Callable, Optional, Sequence, Union, Tuple, Any

tfd = tfp.distributions

@dataclass
class ParamActionDist:
    # tensors from the network
    logits_i: jnp.ndarray          # (B, N)
    logits_j: jnp.ndarray          # (B, N)
    beta_loc: callable             # function(features, i, j) -> (B, K)
    beta_scale: callable           # same -> (B, K)
    features: jnp.ndarray          # (B, F)
    max_successors: int
    eps: float = 1e-6

    def _beta_dist(self, i: jnp.ndarray, j: jnp.ndarray):
        # i,j shape: (B,)
        loc = self.beta_loc(self.features, i, j)
        scale = self.beta_scale(self.features, i, j)

        # squashed Gaussian to [0,1]
        base = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        # Use sigmoid squash: beta = sigmoid(z)
        # log_prob needs change-of-variables term; easiest is to use TFP TransformedDistribution
        bij = tfp.bijectors.Sigmoid()
        dist = tfd.TransformedDistribution(distribution=base, bijector=bij)
        return dist

    def sample(self, seed):
        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)
        # sample i, j
        key_i, key_j, key_b = tfp.random.split_seed(seed, n=3)
        i = di.sample(seed=key_i)  # (B,)
        j = dj.sample(seed=key_j)  # (B,)
        betas = self._beta_dist(i, j).sample(seed=key_b)  # (B, K)
        # pack
        return jnp.concatenate([i[:, None], j[:, None], betas], axis=1)

    def mode(self):
        i = jnp.argmax(self.logits_i, axis=1)
        j = jnp.argmax(self.logits_j, axis=1)
        # use mean of sigmoid-normal approximately via loc (not exact); ok for deterministic eval
        betas = jnp.clip(jax.nn.sigmoid(self.beta_loc(self.features, i, j)), 0.0, 1.0)
        return jnp.concatenate([i[:, None], j[:, None], betas], axis=1)

    def log_prob(self, actions):
        # actions shape (B, 2+K), first two are i,j (possibly floats)
        i = actions[:, 0].astype(jnp.int32)
        j = actions[:, 1].astype(jnp.int32)
        betas = actions[:, 2:]

        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)
        lp = di.log_prob(i) + dj.log_prob(j)
        lp += self._beta_dist(i, j).log_prob(betas)
        return lp

    def entropy(self):
        # exact entropies for categorical; approximate for sigmoid-normal by MC or use -E[logp]
        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)
        # Approx beta entropy via 1 sample (cheap & common)
        # NOTE: entropy coeff in PPO is usually small; stochastic (per-batch) approximation is acceptable
        key = jax.random.PRNGKey(jnp.sum(self.logits_i + self.logits_j))
        a = self.sample(seed=key)
        return di.entropy() + dj.entropy() - self.log_prob(a)

class ParameterizedActor(nn.Module):
    n_actions: int
    max_successors: int
    trunk_arch: Sequence[int]
    head_arch: Sequence[int]
    shared_trunk: bool = True
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    embed_dim: int = 16
    min_scale: float = 1e-3
    init_log_scale: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.array) -> tfd.Distribution:
        x = Flatten()(x)
        if self.shared_trunk:
            for h in self.trunk_arch:
                x = nn.Dense(h)(x)
                x = self.activation_fn(x)

        logits_i = nn.Dense(self.n_actions)(x)
        logits_j = nn.Dense(self.n_actions)(x)

        emb_i_tbl = self.param("emb_i", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))
        emb_j_tbl = self.param("emb_j", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))

        def beta_net(feats, i, j):
            ei = emb_i_tbl[i]
            ej = emb_j_tbl[j]
            z = jnp.concatenate([feats, ei, ej], axis=1)
            y = z
            for h in self.head_arch:
                y = nn.Dense(h)(y)
                y = self.activation_fn(y)

            loc = nn.Dense(self.max_successors)(y)
            log_scale = nn.Dense(self.max_successors)(y) + self.init_log_scale
            scale = jnp.maximum(jnp.exp(log_scale), self.min_scale)
            return loc, scale

        def beta_loc(feats, i, j):
            loc, _ = beta_net(feats, i, j)
            return loc

        def beta_scale(feats, i, j):
            _, scale = beta_net(feats, i, j)
            return scale

        return ParamActionDist(
            logits_i=logits_i,
            logits_j=logits_j,
            beta_loc=beta_loc,
            beta_scale=beta_scale,
            features=x if self.shared_trunk else features,
            max_successors=self.max_successors,
        )

class ParameterizedPPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_actions: int,
        max_successors: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        log_std_init: float = -1.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        shared_trunk: bool = True,
        features_extractor_class: Optional[type[NatureCNN]] = None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_class: type[nn.Module] = ParameterizedActor,
        critic_class: type[nn.Module] = Critic,
        
    ):
        super().__init__(observation_space, action_space)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        if net_arch is not None:
            if isinstance(net_arch, list):
                self.actor_net_arch = self.critic_net_arch = net_arch
            else:
                assert isinstance(net_arch, dict)
                self.actor_net_arch = net_arch["actor"]
                self.critic_net_arch = net_arch["critic"]
        else:
            self.actor_net_arch = self.critic_net_arch = [64, 64]

        if features_extractor_class is not None:
            if features_extractor_kwargs is None:
                features_extractor_kwargs = {}

        self.n_actions = n_actions
        self.max_successors = max_successors
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        self.shared_trunk = shared_trunk
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.actor_class = actor_class
        self.critic_class = critic_class

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Union[optax.Schedule, float], max_grad_norm: float) -> jax.Array:
        key, feat_key, actor_key, critic_key = jax.random.split(key, 4)
        key, self.key = jax.random.split(key, 2)

        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if self.features_extractor_class is not None:
            self.featurizer = self.features_extractor_class(
                **self.features_extractor_kwargs
            )
        else:
            self.featurizer = Identity()

        optimizer = self.optimizer_class(
            learning_rate=lr_schedule,
            **self.optimizer_kwargs,
        )

        self.featurizer_state = TrainState.create(
            apply_fn=self.featurizer.apply,
            params=self.featurizer.init(feat_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.featurizer.apply = jit(self.featurizer.apply)

        obs = self.featurizer.apply(self.featurizer_state.params, obs)

        self.actor = self.actor_class(
            n_actions=self.n_actions,
            max_successors=self.max_successors,
            trunk_arch=self.actor_net_arch,
            head_arch=self.actor_net_arch,
            shared_trunk=self.shared_trunk,
            activation_fn=self.activation_fn,
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.critic = self.critic_class(
            net_arch=self.critic_net_arch, 
            activation_fn=self.activation_fn
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.actor.apply = jit(self.actor.apply)
        self.critic.apply = jit(self.critic.apply)

        return key

    def reset_noise(self):
        self.key, self.noise_key = jax.random.split(self.key)
 
    def forward(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(key, obs, deterministic=deterministic)

    def predict_all(self, key: jax.Array, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_all(key, self.featurizer_state, self.actor_state, self.critic_state, observation)

    def _predict(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self.select_action(self.featurizer_state, self.actor_state, obs)
        else:
            return self.sample_action(key, self.featurizer_state, self.actor_state, obs)

    @staticmethod
    @jit
    def _predict_all(
        key: jax.Array, featurizer_state: TrainState, actor_state: TrainState, critic_state: TrainState, observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        feats = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_state.params, feats)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = critic_state.apply_fn(critic_state.params, feats).flatten()
        return actions, log_probs, values

    @staticmethod
    @jit
    def select_action(
        featurizer_state: TrainState, actor_state: TrainState, obs: jnp.ndarray
    ) -> jnp.ndarray:
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        return actor_state.apply_fn(actor_state.params, feats).mode()
        
    @staticmethod
    @jit
    def sample_action(
        key: jax.Array, featurizer_state: TrainState, actor_state: TrainState, obs: jnp.ndarray
    ) -> jnp.ndarray:
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        dist = actor_state.apply_fn(actor_state.params, feats)
        action = dist.sample(seed=key)
        return action
