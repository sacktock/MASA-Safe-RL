from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional

from masa.algorithms.ppo import PPO
from masa.common.base_class import BaseJaxPolicy
from masa.prob_shield.parameterized_policy_v2 import ParameterizedPPOPolicyV2

class ParameterizedPPOV2(PPO):

    def __init__(self, *args, policy_class: type[BaseJaxPolicy] = ParameterizedPPOPolicyV2, **kwargs):
        super().__init__(*args, policy_class=policy_class, **kwargs)

    def _validate_and_extract_action_specs(self):

        if not isinstance(self.action_space, spaces.Dict):
            raise TypeError(
                "ParameterizedPPOV2 requires env.action_space to be gymnasium.spaces.Dict with keys "
                "['multi_discrete', 'box']. Got "
                f"{type(self.action_space).__name__}. "
                "Consider using the original PPO class instead."
            )

        if "multi_discrete" not in self.action_space.spaces or "box" not in self.action_space.spaces:
            raise KeyError(
                "ParameterizedPPOV2 requires env.action_space = Dict({'multi_discrete': ..., 'box': ...}). "
                f"Got keys: {list(self.action_space.spaces.keys())}. "
                "Consider using the original PPO class instead."
            )

        md = self.action_space.spaces["multi_discrete"]
        bx = self.action_space.spaces["box"]

        if not isinstance(md, spaces.MultiDiscrete):
            raise TypeError(
                "ParameterizedPPOV2 requires action_space['multi_discrete'] to be MultiDiscrete([n_actions, n_actions]). "
                f"Got {type(md).__name__}. "
                "Consider using the original PPO class instead."
            )

        if not isinstance(bx, spaces.Box):
            raise TypeError(
                "ParameterizedPPOV2 requires action_space['box'] to be Box(low=0, high=1, shape=(1,)). "
                f"Got {type(bx).__name__}. "
                "Consider using the original PPO class instead."
            )

        if md.nvec.ndim != 1 or md.nvec.shape[0] != 2:
            raise ValueError(
                "ParameterizedPPOV2 expects action_space['multi_discrete'] to have nvec shape (2,), "
                f"but got nvec={md.nvec}."
            )

        if int(md.nvec[0]) != int(md.nvec[1]):
            raise ValueError(
                "ParameterizedPPOV2 expects action_space['multi_discrete'] = MultiDiscrete([n_actions, n_actions]). "
                f"Got nvec={md.nvec}."
            )

        if bx.shape is None or bx.shape != (1,):
            raise ValueError(
                "ParameterizedPPOV2 expects action_space['box'] to be 1D with shape (1,). "
                f"Got shape={bx.shape}."
            )

        self.n_actions = int(md.nvec[0])

        # Sanity check
        if np.any(bx.low != 0) or np.any(bx.high != 1):
            raise ValueError(
                "ParameterizedPPOV2 expects action_space['box'] bounds to be exactly [0,1] for mixing parameter. "
                f"Got low(min)={float(np.min(bx.low))}, high(max)={float(np.max(bx.high))}."
            )

    def _setup_model(self):
        if hasattr(self, "policy") and self.policy is not None:
            return

        self._validate_and_extract_action_specs()

        if self.policy_kwargs is None:
            self.policy_kwargs = {}

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.n_actions,
            **self.policy_kwargs,
        )

        self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

        self.featurizer = self.policy.featurizer  # type: ignore[assignment]
        self.actor = self.policy.actor            # type: ignore[assignment]
        self.critic = self.policy.critic          # type: ignore[assignment]

    def prepare_act(self, act: Any, n_envs: int = 1) -> np.ndarray:

        if isinstance(self.action_space, spaces.Dict):

            if isinstance(act, dict):
                md = act.get("multi_discrete", None)
                bx = act.get("box", None)
                if md is None or bx is None:
                    raise KeyError(
                        "ParameterizedPPO.prepare_act expected dict with keys "
                        "['multi_discrete', 'box'], got keys "
                        f"{list(act.keys())}."
                    )

                md = np.asarray(md)
                bx = np.asarray(bx, dtype=np.float32)

                md = md.reshape(n_envs, 3)
                bx = bx.reshape(n_envs, 1)

                i = np.clip(md[:, 0], 0, self.n_actions - 1).astype(np.float32)
                j = np.clip(md[:, 1], 0, self.n_actions - 1).astype(np.float32)
                mix = np.clip(bx, 0.0, 1.0).astype(np.float32)

                flat = np.stack([i, j, mix]).astype(np.float32)
                assert len(flat.shape) == 1, f"Wrong shape: {flat.shape}"

                if n_envs == 1:
                    return flat[0]
                return flat

            flat = np.array(act, dtype=np.float32, copy=True).reshape(n_envs, 3)
            flat[:, 0] = np.clip(flat[:, 0], 0, self.n_actions - 1)
            flat[:, 1] = np.clip(flat[:, 1], 0, self.n_actions - 1)
            flat[:, 2] = np.clip(flat[:, 2], 0.0, 1.0)

            if n_envs == 1:
                return flat[0]
            return flat

        return super().prepare_act(act, n_envs=n_envs)

