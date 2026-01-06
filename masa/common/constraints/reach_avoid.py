"""
Reach-avoid constraint monitor.

A reach-avoid property requires eventually reaching a target set while never
visiting an unsafe set. Using atomic propositions:

- ``reach_label`` indicates a target condition,
- ``avoid_label`` indicates an unsafe condition.

A typical reach-avoid specification can be described as:

.. math::

   (\\neg\\mathsf{avoid})\\ \\mathcal{U}\\ \\mathsf{reach}

i.e., "avoid is never true until reach becomes true" (informally).

This implementation tracks:

- ``reached``: whether ``reach_label`` has been observed at least once,
- ``violated``: whether ``avoid_label`` has been observed at least once,
- ``satisfied``: whether the property has been satisfied so far.

"""

from __future__ import annotations
from typing import Any, Dict, Iterable
from masa.common.constraints.base import Constraint, BaseConstraintEnv

class ReachAvoid(Constraint):
    """Reach target label set while avoiding unsafe label set.

    At each step, given a label set ``labels``:

    - reaching condition:
    ``reach = (reach_label in labels)``
    - avoiding condition:
    ``avoid_ok = (avoid_label not in labels)``

    State updates:

    - ``reached`` becomes true once reach is observed,
    - ``violated`` becomes true once avoid is violated,
    - ``satisfied`` becomes true once reached is true and violated is false.

    Args:
        avoid_label: Atomic proposition name indicating unsafe/avoid condition.
        reach_label: Atomic proposition name indicating the target condition.

    Attributes:
        avoid_label: Name of unsafe label.
        reach_label: Name of target label.
        reached: Whether target has been reached at least once.
        violated: Whether unsafe has been observed at least once.
        satisfied: Whether reach-avoid has been satisfied so far.

    """

    def __init__(avoid_label: str, reach_label: str):
        self.avoid_label = avoid_label
        self.reach_label = reach_label

    def reset(self):
        """Reset episode flags."""
        self.reached = False
        self.violated = False
        self.satisfied = False

    def update(self, labels: Iterable[str]):
        """Update reach/avoid flags from the current label set.

        Args:
            labels: Iterable of atomic propositions for the current step.
        """
        self.reach = self.reach_label in labels
        self.avoid = self.avoid_label not in labels

        self.reached = self.reached or self.reach
        self.violated = self.violated or bool(not self.avoid)

        self.satisfied = self.satisfied or (self.reached and bool(not self.violated))

    def episode_metric(self) -> Dict[str, float]:
        """End-of-episode metrics.

        Returns:
            Dict containing:

            - ``"reached"``: whether the target was ever reached,
            - ``"violated"``: whether unsafe was ever visited,
            - ``"satisfied"``: 1.0 if satisfied else 0.0.
        """
        return {"reached": self.reached, "violated": self.violated, "satisfied": float(self.satisfied)}

    def step_metric(self) -> Dict[str, float]:
        """Per-step metrics.

        Returns:
            Dict containing:

            - ``"cost"``: 1.0 if avoid is violated at this step else 0.0,
            - ``"violation"``: 1.0 if avoid violated else 0.0,
            - ``"reached"``: 1.0 if reach holds at this step else 0.0.
        """
        return {"cost": float(not self.avoid), "violation": bool(not self.avoid), "reached": self.reach}

    @property
    def constraint_type(self) -> str:
        """Stable identifier string: ``"reach_avoid"``."""
        return "reach_avoid"

class ReachAvoidEnv(BaseConstraintEnv):
    """Gymnasium wrapper for the :class:`ReachAvoid` monitor.

    Args:
        env: Base environment (must be a :class:`~masa.common.labelled_env.LabelledEnv`).
        avoid_label: Atomic proposition name for unsafe/avoid condition.
        reach_label: Atomic proposition name for target condition.
        **kw: Extra keyword arguments forwarded to :class:`BaseConstraintEnv`.

    """

    def __init__(self, env: gym.Env, avoid_label: str = "unsafe", reach_label: str = "target", **kw):
        super().__init__(env, ReachAvoid(avoid_label=avoid_label, reach_label=reach_label), **kw)


    
