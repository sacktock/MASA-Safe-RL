"""Coalition membership for multi-agent constraints and shields."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Coalition:
    """A non-empty set of agents with an optional display name.

    Coalition membership is order-independent. Algorithms should call
    :meth:`ordered` with an environment's ``possible_agents`` to obtain the
    canonical tuple used for joint-action encoding.
    """

    agents: tuple[str, ...]
    name: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.agents, (str, bytes)):
            raise TypeError("Coalition agents must be a sequence of agent names.")
        agents = tuple(self.agents)
        if not agents:
            raise ValueError("A coalition must contain at least one agent.")
        if any(not isinstance(agent, str) or not agent for agent in agents):
            raise TypeError("Coalition agents must be non-empty strings.")
        if len(set(agents)) != len(agents):
            raise ValueError("Coalition agents must be unique.")
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name
        ):
            raise TypeError("Coalition name must be a non-empty string or None.")
        # Store a canonical order so equality and hashing follow set membership;
        # the optional display name is not part of coalition identity.
        object.__setattr__(self, "agents", tuple(sorted(agents)))

    def ordered(self, possible_agents: Sequence[str]) -> tuple[str, ...]:
        """Return members in the environment's canonical agent order."""
        possible = tuple(possible_agents)
        if len(set(possible)) != len(possible):
            raise ValueError("possible_agents must be unique.")
        members = set(self.agents)
        unknown = members - set(possible)
        if unknown:
            raise ValueError(
                f"Coalition contains unknown agents: {sorted(unknown)}."
            )
        return tuple(agent for agent in possible if agent in members)

    def __contains__(self, agent: str) -> bool:
        return agent in self.agents
