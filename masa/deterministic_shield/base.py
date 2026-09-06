"""Shared LTL-product synthesis and episode bookkeeping for both shields."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.wrappers import ConstraintPersistentWrapper

from .support import read_support
from .winning_region import winning_region


class _LTLShieldBase(ConstraintPersistentWrapper):
    """Finite, fully observed MDP; accepting DFA states denote bad prefixes.

    Wrap LTLSafetyEnv(obs_type="discrete") directly. State/action IDs must be
    zero-based. The model, DFA and deterministic state-label function must stay
    unchanged. Real transition support must be contained in the modeled support.
    Do not remap actions/observations between the base model and this wrapper.
    Put observation transforms, auto-reset and vectorization outside the shield.

    Synthesis is infinite-horizon safety, not finite-trace LTL. Terminal states
    need explicit model dynamics (normally absorbing); time limits do not shorten
    the synthesis horizon. Runtime model checks detect mismatches after a step,
    and cannot undo an unsafe transition caused by an incomplete model.

    After truncation, info['action_mask'] still describes the returned observation
    for value bootstrapping. After termination it is empty. In either case,
    step()/action_masks() require reset() before another action can be taken.
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

        successors, self._support = read_support(
            env.unwrapped, self._n_states, n_actions
        )
        # The live monitor already consumed L(s); its next update consumes L(s').
        self._targets = next_q[:, successors] * self._n_states + successors
        self._initial_states = (
            next_q[q_index[dfa.initial]] * self._n_states
            + np.arange(self._n_states)
        )
        self.winning_region, self.safe_actions = winning_region(
            self._targets, self._support, rejecting
        )
        self.winning_region.flags.writeable = False
        self.safe_actions.flags.writeable = False

    def _require_state(self) -> int:
        if self._state is None:
            raise RuntimeError("Call reset() before acting, including after episode end.")
        return self._state

    def _check_state(self, obs) -> int:
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Invalid product observation: {obs!r}.")
        state = int(obs)
        if not self.winning_region[state]:
            raise RuntimeError(
                f"Product state {state} is outside the winning region; "
                "safety cannot be guaranteed."
            )
        return state

    def _validate_proposal(self, action) -> int:
        self._require_state()
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}.")
        return int(action)

    def action_masks(self) -> np.ndarray:
        """A copy of the current safe-action mask; requires an active episode."""
        return self.safe_actions[self._require_state()].copy()

    def reset(self, *, seed=None, options=None):
        self._state = None
        obs, info = self.env.reset(seed=seed, options=options)
        state = self._check_state(obs)
        if state != self._initial_states[state % self._n_states]:
            raise RuntimeError("Reset labels/DFA disagree with the synthesized model.")
        self._state = state
        return obs, {**info, "action_mask": self.action_masks()}

    def _step_safe(self, proposed: int, executed):
        previous = self._require_state()
        # Validate BEFORE stepping. A bad proposal/selector must not advance time.
        if not self.action_space.contains(executed):
            raise ValueError(f"Invalid executed action: {executed!r}.")
        executed = int(executed)
        if not self.safe_actions[previous, executed]:
            raise ValueError(
                f"Action {executed} is not safe in product state {previous}; "
                "choose an action allowed by action_masks()."
            )

        # A failed environment step or model check requires a fresh reset.
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
                if terminated else self.safe_actions[state].copy()
            ),
            "shield_intervened": executed != proposed,
            "shield_proposed_action": proposed,
            "shield_executed_action": executed,
        }
        return obs, reward, terminated, truncated, info

    def step(self, action):
        # Do not inherit gym.Wrapper.step(), which would bypass shielding.
        raise NotImplementedError("Use PreemptiveLTLShield or PostposedLTLShield.")

    def close(self):
        self._state = None
        return self.env.close()
