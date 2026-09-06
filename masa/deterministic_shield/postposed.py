"""Preserve safe proposals and replace unsafe ones using a selected rule."""
from __future__ import annotations

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from .base import _LTLShieldBase
from .replacement_strategies import Replacement


class PostposedLTLShield(_LTLShieldBase):
    """Postposed shielding with a deterministic lowest-index default.

    replacement=None uses a precomputed first-safe action (constant-time lookup).
    Otherwise replacement(state, proposed, mask) chooses an action; state is the
    product observation, and mask is a copy. Use random_safe(), highest_score(),
    or a custom callable. Safe proposals never call replacement.

    Random replacement changes action selection, not the hard safe-action mask.
    random_safe(seed) owns a separate RNG that env.reset(seed=...) does not reset.

    Every replacement is checked against the original mask before env.step().
    A callback error, invalid action or unsafe replacement does not step the
    environment, provided the callback itself does not mutate/step it. Callback
    state (including an RNG) is not rolled back after a failure.
    """

    def __init__(self, env: LTLSafetyEnv, *, replacement: Replacement | None = None):
        if replacement is not None and not callable(replacement):
            raise TypeError("replacement must be callable or None.")
        super().__init__(env)
        self._replacement = replacement
        self._fallback = self.safe_actions.argmax(axis=1)

    def step(self, action):
        proposed = self._validate_proposal(action)
        state = self._require_state()
        if self.safe_actions[state, proposed]:
            executed = proposed
        elif self._replacement is None:
            executed = int(self._fallback[state])
        else:
            executed = self._replacement(state, proposed, self.action_masks())
        return self._step_safe(proposed, executed)
