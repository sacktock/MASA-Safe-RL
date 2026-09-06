"""Winning-region shielding for finite, fully observed MDPs and safety DFAs."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.wrappers import ConstraintPersistentWrapper
from masa.deterministic_shield import support, winning_region

# TODO: add pre-emptive shield,
# TODO: add alternate action replacement strategies for post-posted shielding
class DeterministicLTLShield(ConstraintPersistentWrapper):
    """Postposed shield: preserve a safe proposal, otherwise use first safe action.

    Place directly outside LTLSafetyEnv(obs_type="discrete"), before observation
    transforms or vectorization. Base state/action IDs must be zero-based.
    The DFA's accepting states must mean *bad prefixes*, as in MASA.

    The model, DFA, and deterministic state-label function must stay unchanged.
    The model support must include every possible real transition. No action or
    observation remapping may occur between the model and this wrapper.
    Auto-reset wrappers must also be outside this shield.

    Terminals need explicit model dynamics (normally absorbing). Synthesis uses
    infinite-horizon safety, not finite-trace LTL or a time-limit-dependent region.
    """

    def __init__(self, env: LTLSafetyEnv):
        if not isinstance(env, LTLSafetyEnv):
            raise TypeError("Place the shield directly outside LTLSafetyEnv.")
        if env._obs_type != "discrete":
            raise TypeError("LTLSafetyEnv must use obs_type='discrete'.")
        for space in (env._orig_obs_space, env.action_space):
            if not isinstance(space, spaces.Discrete) or space.start != 0:
                raise TypeError("Base states and actions must be zero-based Discrete.")
        super().__init__(env)
        self._state: int | None = None
        self._n_states = int(env._orig_obs_space.n)
        n_actions = int(env.action_space.n)

        dfa = env._constraint.get_dfa()
        # Use the monitor's actual encoding, not the DFA state names as indices.
        q_index = env._automaton_states_idx
        rejecting = np.zeros(len(q_index), dtype=bool)
        for q in dfa.accepting:
            rejecting[q_index[q]] = True
        if dfa.initial in dfa.accepting:
            raise ValueError("The DFA already rejects the empty prefix.")

        labels = [set(env.label_fn(s)) for s in range(self._n_states)]
        next_q = np.empty((len(q_index), self._n_states), dtype=np.intp)
        for q, i in q_index.items():
            for s, label in enumerate(labels):
                next_q[i, s] = q_index[dfa.transition(q, label)]

        successors, self._support = _read_support(
            env.unwrapped, self._n_states, n_actions
        )
        # The live monitor has ALREADY consumed L(s). Next it consumes L(s').
        self._targets = next_q[:, successors] * self._n_states + successors
        self._initial_states = (
            next_q[q_index[dfa.initial]] * self._n_states + np.arange(self._n_states)
        )
        self.winning_region, self.safe_actions = winning_region(
            self._targets, self._support, rejecting
        )
        self._fallback = self.safe_actions.argmax(axis=1)
        self.winning_region.flags.writeable = False
        self.safe_actions.flags.writeable = False

    def _check_state(self, obs: int) -> int:
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Invalid product observation: {obs!r}.")
        state = int(obs)
        if not self.winning_region[state]:
            raise RuntimeError(
                f"Product state {state} is outside the winning region; "
                "safety cannot be guaranteed."
            )
        return state

    def action_masks(self) -> np.ndarray:
        """Return a copy of the current safe-action mask; requires an active episode."""
        if self._state is None:
            raise RuntimeError("Call reset() before requesting actions.")
        return self.safe_actions[self._state].copy()

    def reset(self, *, seed=None, options=None):
        self._state = None
        obs, info = self.env.reset(seed=seed, options=options)
        state = self._check_state(obs)
        if state != self._initial_states[state % self._n_states]:
            raise RuntimeError("Reset labels/DFA disagree with the synthesized model.")
        self._state = state
        return obs, {**info, "action_mask": self.action_masks()}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() before step(), including after episode end.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}.")
        proposed = int(action)
        previous = self._state
        executed = (
            proposed if self.safe_actions[previous, proposed]
            else int(self._fallback[previous])
        )

        # Any failed step/check invalidates the episode until reset().
        self._state = None
        obs, reward, terminated, truncated, info = self.env.step(executed)
        state = self._check_state(obs)
        q, s = divmod(previous, self._n_states)
        possible = self._support[s, executed] & (self._targets[q, s] == state)
        if not possible.any():
            raise RuntimeError("Observed transition is absent from the shield model.")

        done = bool(terminated or truncated)
        if not done:
            self._state = state
        info = {
            **info,
            "action_mask": (
                np.zeros(self.action_space.n, dtype=bool)
                if done else self.action_masks()
            ),
            "shield_intervened": executed != proposed,
            "shield_proposed_action": proposed,
            "shield_executed_action": executed,
        }
        return obs, reward, terminated, truncated, info