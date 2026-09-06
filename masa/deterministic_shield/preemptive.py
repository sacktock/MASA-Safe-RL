"""Expose safe actions before the policy chooses; never replace its choice."""
from .base import _LTLShieldBase


class PreemptiveLTLShield(_LTLShieldBase):
    """Restrict action selection through action_masks()/info['action_mask'].

    The policy must select from the current mask, including during exploration.
    An unsafe action raises ValueError BEFORE the environment is stepped. The
    episode remains active so the caller can retry with a safe action. Valid
    actions are executed unchanged; this wrapper never silently substitutes one.

    Exposing a mask does not make an arbitrary learning algorithm mask-aware.
    """

    def step(self, action):
        proposed = self._validate_proposal(action)
        return self._step_safe(proposed, proposed)
