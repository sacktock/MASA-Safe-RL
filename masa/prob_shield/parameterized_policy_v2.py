import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from dataclasses import dataclass
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

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

@dataclass
class ParamActionDist:
    # tensors from the network
    logits_i: jnp.ndarray   # (B, N)
    logits_j: jnp.ndarray   # (B, N)

    # Conditional mix lookup tables (pre-sigmoid Gaussian params)
    mix_loc_table: jnp.ndarray    # (B, N, N, E, 1)
    mix_scale_table: jnp.ndarray  # (B, N, N, E, 1)
    eps: float = 1e-6

    def _gather_mix_params(self, i: jnp.ndarray, j: jnp.ndarray):
        """
        i, j, e: (B,) int32
        returns loc, scale: (B, 1)
        """
        B = i.shape[0]

        # Step 1: gather along axis=1 (i)
        loc_i = jnp.take_along_axis(self.mix_loc_table, i.reshape(B, 1, 1, 1), axis=1)
        sc_i  = jnp.take_along_axis(self.mix_scale_table, i.reshape(B, 1, 1, 1), axis=1)

        # Step 2: gather along axis=2 (j)
        loc_ij = jnp.take_along_axis(loc_i, j.reshape(B, 1, 1, 1), axis=2)
        sc_ij  = jnp.take_along_axis(sc_i,  j.reshape(B, 1, 1, 1), axis=2)

        # Squeeze to (B,1)
        loc = loc_ij.reshape(B, 1)
        scale = sc_ij.reshape(B, 1)
        return loc, scale

    def _mix_dist(self, i: jnp.ndarray, j: jnp.ndarray):
        loc, scale = self._gather_mix_params(i, j)  # (B,1)
        base = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        eps = jnp.array(self.eps, dtype=loc.dtype)
        bij = tfb.Chain([
            tfb.Shift(eps),
            tfb.Scale(1.0 - 2.0 * eps),
            tfb.Sigmoid(),
        ])
        return tfd.TransformedDistribution(distribution=base, bijector=bij)

    def sample(self, seed):
        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)
        # sample i, j, e
        key_i, key_j, key_m = tfp.random.split_seed(seed, n=3)
        i = di.sample(seed=key_i)                      # (B,)
        j = dj.sample(seed=key_j)                      # (B,)
        mix = self._mix_dist(i, j).sample(seed=key_m) # (B,1) in (0,1)

        # Flat action expected by env: [i, j, edge_dim, mix]
        return jnp.concatenate([i[:, None], j[:, None], mix], axis=1)

    def mode(self):
        i = jnp.argmax(self.logits_i, axis=1)
        j = jnp.argmax(self.logits_j, axis=1)
        
        # use mean of sigmoid-normal approximately via loc (not exact); ok for deterministic eval
        loc, _ = self._gather_mix_params(i, j)
        mix = jnp.clip(jax.nn.sigmoid(loc), 0.0, 1.0)
        return jnp.concatenate([i[:, None], j[:, None], mix], axis=1)

    def log_prob(self, actions):
        # actions shape (B, 2+K), first two are i,j (possibly floats)
        i = actions[:, 0].astype(jnp.int32)
        j = actions[:, 1].astype(jnp.int32)
        mix = actions[:, 2].reshape((-1, 1)).astype(self.mix_loc_table.dtype)  # (B,1)
        #eps = jnp.array(1e-6, dtype=betas.dtype)
        #betas_safe = jnp.clip(betas, eps, 1.0 - eps)
        #betas_safe = jax.lax.stop_gradient(betas_safe - betas) + betas

        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)
        dm = self._mix_dist(i, j)

        lp = di.log_prob(i) + dj.log_prob(j)
        lp = lp + dm.log_prob(mix)  # (B,)
        return lp

    def entropy(self, actions):
        di = tfd.Categorical(logits=self.logits_i)
        dj = tfd.Categorical(logits=self.logits_j)

        ent = di.entropy() + dj.entropy()

        # deterministic proxy for entropy based on Jacobian-at-mean correction
        i = actions[:, 0].astype(jnp.int32)
        j = actions[:, 1].astype(jnp.int32)
        loc, scale = self._gather_mix_params(i, j)  # (B,1)
        base = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        base_ent = base.entropy()  # (B,)

        # log sigmoid'(x) = -softplus(-x) - softplus(x)
        log_sigmoid_prime_at_mean = -jax.nn.softplus(-loc) - jax.nn.softplus(loc)  # (B, K)
        jacobian_term = jnp.sum(log_sigmoid_prime_at_mean, axis=-1)          # (B,)

        beta_ent_approx = base_ent + jacobian_term
        return ent + beta_ent_approx

class ParameterizedActorV2(nn.Module):
    n_actions: int
    trunk_arch: Sequence[int]
    head_arch: Sequence[int]
    shared_trunk: bool = True
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    embed_dim: int = 64
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    eps: float = 1e-6
    mean_clip: Optional[float] = 10.0
    smooth_mean_clip: bool = True

    @nn.compact
    def __call__(self, x: jnp.array) -> ParamActionDist:
        x = Flatten()(x)
        if self.shared_trunk:
            for h in self.trunk_arch:
                x = nn.Dense(h)(x)
                x = self.activation_fn(x)

        logits_i = nn.Dense(self.n_actions)(x)
        logits_j = nn.Dense(self.n_actions)(x)

        emb_i_tbl = self.param("emb_i", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))
        emb_j_tbl = self.param("emb_j", nn.initializers.normal(0.02), (self.n_actions, self.embed_dim))

        # (B, N, E)
        ei = jnp.broadcast_to(emb_i_tbl[None, :, :], (x.shape[0], self.n_actions, self.embed_dim))
        ej = jnp.broadcast_to(emb_j_tbl[None, :, :], (x.shape[0], self.n_actions, self.embed_dim))

        xb = x[:, None, None, :]
        ei_grid = ei[:, :, None, :]
        ej_grid = ej[:, None, :, :]
        z = jnp.concatenate(
            [
                jnp.broadcast_to(xb, (x.shape[0], self.n_actions, self.n_actions, x.shape[1])),
                jnp.broadcast_to(ei_grid, (x.shape[0], self.n_actions, self.n_actions, self.embed_dim)),
                jnp.broadcast_to(ej_grid, (x.shape[0], self.n_actions, self.n_actions, self.embed_dim))
            ], axis=-1
        )

        y = z.reshape((x.shape[0]*self.n_actions*self.n_actions, -1))
        for h in self.head_arch:
            y = nn.Dense(h)(y)
            y = self.activation_fn(y)

        mix_loc = nn.Dense(1)(y)
        if self.mean_clip is not None:
            if self.smooth_mean_clip:
                mix_loc = self.mean_clip * jnp.tanh(mix_loc / self.mean_clip)
            else:
                mix_loc = jnp.clip(mix_loc, -self.mean_clip, self.mean_clip)

        mix_log_std = nn.Dense(1)(y) + self.log_std_init
        mix_log_std = jnp.clip(mix_log_std, self.log_std_min, self.log_std_max)
        mix_scale = jnp.exp(mix_log_std) + self.eps

        mix_loc = mix_loc.reshape((x.shape[0], self.n_actions, self.n_actions, 1))
        mix_scale = mix_scale.reshape((x.shape[0], self.n_actions, self.n_actions, 1))

        return ParamActionDist(
            logits_i=logits_i,
            logits_j=logits_j,
            mix_loc_table=mix_loc,
            mix_scale_table=mix_scale,
            eps=self.eps,
        )

class ParameterizedPPOPolicyV2(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_actions: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        log_std_init: float = -1.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        shared_trunk: bool = True,
        features_extractor_class: Optional[type[NatureCNN]] = None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_class: type[nn.Module] = ParameterizedActorV2,
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
            trunk_arch=self.actor_net_arch,
            head_arch=self.actor_net_arch,
            shared_trunk=self.shared_trunk,
            activation_fn=self.activation_fn,
            log_std_init=self.log_std_init
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
