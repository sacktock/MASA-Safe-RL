from __future__ import annotations
from typing import Any
from gymnasium import spaces
import numpy as np
import math
from masa.common.label_fn import LabelFn
from masa.envs.continuous.base import ContinuousEnv

THETA_THESHOLD_RADIANS = 0.2095 # ~ 3/4 pi radians
X_THRESHOLD = 2.4

def label_fn(obs):
    x, x_dot, theta, theta_dot = obs
    if np.abs(theta) <= THETA_THESHOLD_RADIANS \
            and np.abs(x) <= X_THRESHOLD:
        return {"stable"}
    else:
        return set()

cost_fn = lambda labels: 0.0 if "stable" in labels else 1.0

class ContinuousCartPole(ContinuousEnv):

    def __init__(self):

        self._gravity = 9.8
        self._masscart = 1.0
        self._masspole = 0.1
        self._total_mass = self._masspole + self._masscart
        self._length = 0.5
        self._polemass_length = self._masspole * self._length
        self._force_mag = 10.0
        self._tau = 0.02
        self._kinematics_integrator = "euler"

        self._theta_threshold_radians = THETA_THESHOLD_RADIANS
        self._x_threshold = X_THRESHOLD

        self._x_vel_threshold = 2.0
        self._theta_vel_threshold = 0.5

        high = np.array(
            [
                self._x_threshold*2,
                self._x_vel_threshold*2,
                self._theta_threshold_radians*2,
                self._theta_vel_threshold*2,
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _obs(self):
        return self._state

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = self.np_random.uniform(
            low=np.array([-0.05, -0.05, -0.05, -0.05]), 
            high=np.array([0.05, 0.05, 0.05, 0.05]), 
            size=(4,)
        )

        return self._obs(), {}

    def step(self, action: Any):
        assert self.action_space.contains(action), f"Invalid action {action}!"

        x, x_dot, theta, theta_dot = self._state
        force = self._force_mag * float(action[0])
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self._polemass_length * theta_dot ** 2 * sintheta
        ) / self._total_mass
        thetaacc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._masspole * costheta ** 2 / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * costheta / self._total_mass

        if self._kinematics_integrator == "euler":
            x = x + self._tau * x_dot
            x_dot = x_dot + self._tau * xacc
            theta = theta + self._tau * theta_dot
            theta_dot = theta_dot + self._tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self._tau * xacc
            x = x + self._tau * x_dot
            theta_dot = theta_dot + self._tau * thetaacc
            theta = theta + self._tau * theta_dot

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        stable = np.abs(theta) <= self._theta_threshold_radians \
            and np.abs(x) <= self._x_threshold

        return self._obs(), 1.0, not stable, False, {}