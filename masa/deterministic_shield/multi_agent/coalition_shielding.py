"""Winning-region LTL shielding for PettingZoo parallel environments."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.common.ltl import DFA
from masa.common.multi_agent.coalition import Coalition
from masa.envs.multiagent.tabular_env import TabularParallelEnv
from masa.common.multi_agent.labelled_pz_env import LabelledParallelEnv

from .coalition_support import (
    CoalitionSupport,
    build_coalition_support,
    rectangular_action_masks,
)
from masa.deterministic_shield.replacement_strategies import Replacement
from masa.deterministic_shield.winning_region import winning_region

ShieldMode = Literal["preemptive", "postposed"]
ExecutionMode = Literal["centralised", "decentralised"]
ReplacementSpec = Replacement | Mapping[str, Replacement | None] | None
LabelCombiner = Callable[[Mapping[str, frozenset[str]]], Iterable[str]]


def _union_labels(labels_by_agent: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """Default shared alphabet: union labels from every possible agent."""
    return frozenset().union(*labels_by_agent.values())


def _normalise_execution(value: str) -> ExecutionMode:
    aliases = {
        "centralised": "centralised",
        "centralized": "centralised",
        "decentralised": "decentralised",
        "decentralized": "decentralised",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            "execution must be 'centralised' or 'decentralised'."
        ) from exc


class CoalitionLTLShield(ParallelEnv):
    """Shield a coalition against every action of its complement.

    This wrapper must directly wrap ``LabelledParallelEnv(TabularParallelEnv)``.
    ``dfa.accepting`` must contain bad-prefix states. The tabular transition model
    must include every real successor with non-zero probability.

    ``execution='centralised'`` exposes or repairs a coalition *joint* action.
    ``execution='decentralised'`` exposes one local mask per coalition member.
    Those masks form a certified Cartesian rectangle: coalition members may choose
    simultaneously without observing one another's current choice. They rely on
    teammates respecting their masks; agents outside the coalition are treated as
    unrestricted adversaries and are never modified by the shield.

    ``mode='preemptive'`` rejects unsafe proposals before stepping. In
    ``mode='postposed'`` unsafe proposals are replaced. A centralised replacement
    operates on a coalition-action index. Decentralised replacements operate
    independently per agent and must be supplied as separate callbacks in a mapping
    when the coalition contains more than one agent.

    The wrapper leaves observations unchanged. It publishes product/DFA state and
    masks through methods and per-coalition-agent ``infos``. Separately deployed
    coalition members must reconstruct the same global model state and DFA state.
    This implementation does not solve partial-observation shielding.

    Agent membership must remain fixed during an episode, and all agents must end
    an episode together. The transition model, labels, DFA and action spaces must
    remain fixed after construction.
    """

    metadata = {"name": "coalition_ltl_shield"}

    def __init__(
        self,
        env: LabelledParallelEnv,
        *,
        coalition: Coalition | Sequence[str],
        dfa: DFA,
        mode: ShieldMode = "preemptive",
        execution: str = "centralised",
        replacement: ReplacementSpec = None,
        label_combiner: LabelCombiner | None = None,
    ) -> None:
        if not isinstance(env, LabelledParallelEnv):
            raise TypeError("CoalitionLTLShield must wrap a LabelledParallelEnv.")
        if not isinstance(dfa, DFA):
            raise TypeError("dfa must be a masa.common.ltl.DFA.")
        if mode not in ("preemptive", "postposed"):
            raise ValueError("mode must be 'preemptive' or 'postposed'.")
        if isinstance(coalition, (str, bytes)):
            raise TypeError("coalition must be a Coalition or sequence of agent names.")
        if not isinstance(coalition, Coalition):
            coalition = Coalition(tuple(coalition))

        self.env = env
        self.metadata = getattr(env, "metadata", self.metadata)
        self.possible_agents = list(env.possible_agents)
        self.agents = list(getattr(env, "agents", self.possible_agents))
        if not isinstance(env.env, TabularParallelEnv):
            raise TypeError(
                "CoalitionLTLShield requires LabelledParallelEnv to wrap a "
                "TabularParallelEnv directly."
            )
        self.tabular_env = env.env
        if tuple(self.tabular_env.possible_agents) != tuple(self.possible_agents):
            raise ValueError(
                "TabularParallelEnv and LabelledParallelEnv possible_agents "
                "must have the same order."
            )
        self._action_sizes = self.tabular_env.action_sizes
        self._agent_index = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }
        self.coalition = coalition
        self.mode: ShieldMode = mode
        self.execution = _normalise_execution(execution)

        self.coalition_agents = coalition.ordered(self.possible_agents)
        coalition_set = set(self.coalition_agents)
        self.complement_agents = tuple(
            agent for agent in self.possible_agents if agent not in coalition_set
        )

        for agent in self.possible_agents:
            space = env.action_space(agent)
            if not isinstance(space, spaces.Discrete) or space.start != 0:
                raise TypeError(
                    "Coalition shielding requires zero-based Discrete action spaces."
                )

        self._configure_replacements(replacement)
        self._dfa = dfa
        self._dfa_states = tuple(dfa.states)
        self._q_index = {
            state: index for index, state in enumerate(self._dfa_states)
        }
        if len(self._q_index) != len(self._dfa_states):
            raise ValueError("DFA states must be unique.")
        unknown_accepting = set(dfa.accepting) - set(self._dfa_states)
        if unknown_accepting:
            raise ValueError(
                f"DFA accepting states are unknown: {sorted(unknown_accepting)}."
            )
        if dfa.initial not in self._q_index:
            raise ValueError("DFA initial state must be listed in dfa.states.")
        if dfa.initial in dfa.accepting:
            raise ValueError("The DFA already rejects the empty prefix.")

        if label_combiner is not None and not callable(label_combiner):
            raise TypeError("label_combiner must be callable or None.")
        self._label_combiner = label_combiner or _union_labels
        labels = [
            self._labels_for_state(state)
            for state in range(self.tabular_env.n_states)
        ]
        self._state_labels = tuple(labels)
        next_q = np.empty(
            (len(self._dfa_states), self.tabular_env.n_states), dtype=np.intp
        )
        for q_index, q_state in enumerate(self._dfa_states):
            for state, state_labels in enumerate(labels):
                q_next = dfa.transition(q_state, state_labels)
                if q_next not in self._q_index:
                    raise ValueError(
                        f"DFA transition returned unknown state {q_next!r}."
                    )
                next_q[q_index, state] = self._q_index[q_next]
        self._next_q = next_q

        rejecting = np.zeros(len(self._dfa_states), dtype=bool)
        for state in dfa.accepting:
            rejecting[self._q_index[state]] = True

        game = build_coalition_support(self.tabular_env, self.coalition_agents)
        self._game: CoalitionSupport = game
        self.coalition_actions = game.coalition_actions
        self._coalition_action_index = {
            action: index for index, action in enumerate(self.coalition_actions)
        }
        self._targets = (
            next_q[:, game.successors] * self.tabular_env.n_states + game.successors
        )
        self.winning_region, self.safe_joint_actions = winning_region(
            self._targets, game.support, rejecting
        )
        self.winning_region.flags.writeable = False
        self.safe_joint_actions.flags.writeable = False
        self._joint_fallback = self.safe_joint_actions.argmax(axis=1)

        self.independent_joint_actions: np.ndarray | None = None
        self.local_safe_actions: Mapping[str, np.ndarray] = MappingProxyType({})
        self._local_fallback: dict[str, np.ndarray] = {}
        if self.execution == "decentralised":
            coalition_sizes = tuple(
                self._action_sizes[agent] for agent in self.coalition_agents
            )
            local_masks, rectangle = rectangular_action_masks(
                self.safe_joint_actions,
                self.coalition_actions,
                coalition_sizes,
            )
            self.independent_joint_actions = rectangle
            local_by_agent = dict(zip(self.coalition_agents, local_masks))
            self.local_safe_actions = MappingProxyType(local_by_agent)
            self._local_fallback = {
                agent: mask.argmax(axis=1)
                for agent, mask in local_by_agent.items()
            }

        self._base_state: int | None = None
        self._q: int | None = None
        self._product_state: int | None = None

    def _labels_from_observations(
        self, observations: Mapping[str, object]
    ) -> frozenset[str]:
        if not isinstance(observations, Mapping):
            raise TypeError("Tabular observations must be an agent mapping.")
        expected = set(self.possible_agents)
        supplied = set(observations)
        if supplied != expected:
            raise ValueError(
                "State observations must contain exactly possible_agents; "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}."
            )

        label_fn = self.env.label_fn
        labels_by_agent: dict[str, frozenset[str]] = {}
        for agent in self.possible_agents:
            if isinstance(label_fn, Mapping):
                try:
                    fn = label_fn[agent]
                except KeyError as exc:
                    raise ValueError(
                        f"No labelling function was supplied for {agent!r}."
                    ) from exc
            else:
                fn = label_fn
            if not callable(fn):
                raise TypeError(f"Labelling function for {agent!r} is not callable.")
            raw = fn(observations[agent])
            if isinstance(raw, (str, bytes)):
                raise TypeError("Labels must be an iterable of proposition names.")
            labels = frozenset(raw)
            if any(not isinstance(label, str) or not label for label in labels):
                raise TypeError("Labels must be non-empty strings.")
            labels_by_agent[agent] = labels

        raw_combined = self._label_combiner(MappingProxyType(labels_by_agent))
        if isinstance(raw_combined, (str, bytes)):
            raise TypeError(
                "label_combiner must return an iterable of proposition names."
            )
        combined = frozenset(raw_combined)
        if any(not isinstance(label, str) or not label for label in combined):
            raise TypeError("Combined labels must be non-empty strings.")
        return combined

    def _labels_for_state(self, state: int) -> frozenset[str]:
        observations = self.tabular_env.observations_from_state(state)
        return self._labels_from_observations(observations)

    def _check_runtime_labels(
        self, observations: Mapping[str, object], state: int
    ) -> None:
        # Some Parallel environments omit terminal observations. When all are
        # present, verify that the live observation and tabular-state encodings
        # induce the same shared propositions.
        if isinstance(observations, Mapping) and set(observations) == set(
            self.possible_agents
        ):
            actual = self._labels_from_observations(observations)
            expected = self._state_labels[state]
            if actual != expected:
                raise RuntimeError(
                    "Runtime labels disagree with observations_from_state() for "
                    f"tabular state {state}: expected {set(expected)}, "
                    f"got {set(actual)}."
                )

    def _legal_actions(self, state: int, agent: str) -> tuple[int, ...]:
        return self._game.legal_actions[state][self._agent_index[agent]]

    def _configure_replacements(self, replacement: ReplacementSpec) -> None:
        if self.execution == "centralised":
            if isinstance(replacement, Mapping):
                raise TypeError(
                    "A centralised replacement is one callback over joint-action indices."
                )
            if replacement is not None and not callable(replacement):
                raise TypeError("replacement must be callable or None.")
            self._joint_replacement = replacement
            self._replacement_by_agent: dict[str, Replacement | None] = {}
            return

        self._joint_replacement = None
        if isinstance(replacement, Mapping):
            unknown = set(replacement) - set(self.coalition_agents)
            if unknown:
                raise ValueError(
                    "Replacement mapping contains non-coalition agents: "
                    f"{sorted(unknown)}."
                )
            configured: dict[str, Replacement | None] = {}
            for agent in self.coalition_agents:
                callback = replacement.get(agent)
                if callback is not None and not callable(callback):
                    raise TypeError(
                        f"Replacement for {agent!r} must be callable or None."
                    )
                configured[agent] = callback
            callbacks = [callback for callback in configured.values() if callback is not None]
            if len({id(callback) for callback in callbacks}) != len(callbacks):
                raise ValueError(
                    "Decentralised agents must use separate replacement callback "
                    "instances; do not share one stateful selector or RNG."
                )
            self._replacement_by_agent = configured
            return

        if replacement is not None and not callable(replacement):
            raise TypeError("replacement must be callable, a mapping, or None.")
        if replacement is not None and len(self.coalition_agents) > 1:
            raise TypeError(
                "For a multi-agent decentralised coalition, provide a mapping "
                "with one replacement callback per agent."
            )
        self._replacement_by_agent = {
            agent: replacement for agent in self.coalition_agents
        }

    def __getattr__(self, name: str):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def state(self):
        return self.env.state()

    def render(self):
        return self.env.render()

    def close(self):
        self._clear_runtime_state()
        return self.env.close()

    @property
    def product_state(self) -> int:
        """Current flattened ``(DFA state, tabular state)`` index."""
        return self._require_product_state()

    @property
    def tabular_state(self) -> int:
        """Current finite state ID of the wrapped ``TabularParallelEnv``."""
        self._require_product_state()
        assert self._base_state is not None
        return self._base_state

    def labels_for_state(self, state: int) -> frozenset[str]:
        """Shared DFA propositions precomputed for one finite game state."""
        if not isinstance(state, (int, np.integer)) or isinstance(state, bool):
            raise TypeError("Tabular state must be an integer.")
        state = int(state)
        if not 0 <= state < self.tabular_env.n_states:
            raise ValueError(
                f"Tabular state {state} is outside "
                f"[0, {self.tabular_env.n_states})."
            )
        return self._state_labels[state]

    @property
    def automaton_state(self):
        """Current DFA state value, rather than its integer encoding."""
        if self._q is None:
            raise RuntimeError("Call reset() before requesting the automaton state.")
        return self._dfa_states[self._q]

    @property
    def constraint_type(self) -> str:
        return "COALITION_LTL_SHIELD"

    def encode_coalition_action(
        self, action: Mapping[str, int] | Sequence[int]
    ) -> int:
        """Encode a coalition action in canonical environment-agent order."""
        if isinstance(action, Mapping):
            if set(action) != set(self.coalition_agents):
                raise ValueError(
                    "Coalition action mapping must contain exactly the coalition agents."
                )
            key = tuple(int(action[agent]) for agent in self.coalition_agents)
        else:
            if isinstance(action, (str, bytes)):
                raise TypeError("Coalition action must be a mapping or action sequence.")
            key = tuple(int(primitive) for primitive in action)
        try:
            return self._coalition_action_index[key]
        except KeyError as exc:
            raise ValueError(f"Invalid coalition action: {key!r}.") from exc

    def decode_coalition_action(self, index: int) -> dict[str, int]:
        """Decode a coalition-action index to an agent-action mapping."""
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("Coalition action index must be an integer.")
        if not 0 <= int(index) < len(self.coalition_actions):
            raise ValueError(f"Invalid coalition action index: {index!r}.")
        action = self.coalition_actions[int(index)]
        return dict(zip(self.coalition_agents, action))

    def robust_coalition_action_mask(self) -> np.ndarray:
        """Full coalition relation safe against every complement action."""
        return self.safe_joint_actions[self._require_product_state()].copy()

    def coalition_action_mask(self) -> np.ndarray:
        """Current executable coalition relation in coalition-action index order.

        In centralised execution this is the full robust safe relation. In
        decentralised execution it is the selected Cartesian subset represented
        by the local masks.
        """
        product_state = self._require_product_state()
        if self.execution == "centralised":
            return self.safe_joint_actions[product_state].copy()
        assert self.independent_joint_actions is not None
        return self.independent_joint_actions[product_state].copy()

    def safe_coalition_actions(self) -> tuple[dict[str, int], ...]:
        """Decode every currently executable coalition joint action."""
        return tuple(
            self.decode_coalition_action(index)
            for index in np.flatnonzero(self.coalition_action_mask())
        )

    def local_action_mask(self, agent: str) -> np.ndarray:
        """Current independent mask for one decentralised coalition member."""
        if self.execution != "decentralised":
            raise RuntimeError(
                "Local masks are available only for decentralised execution."
            )
        if agent not in self.local_safe_actions:
            raise ValueError(f"{agent!r} is not in the coalition.")
        return self.local_safe_actions[agent][self._require_product_state()].copy()

    def local_action_masks(self) -> dict[str, np.ndarray]:
        """Copies of all current decentralised local masks."""
        return {
            agent: self.local_action_mask(agent)
            for agent in self.coalition_agents
        }

    def reset(self, seed=None, options=None):
        self._clear_runtime_state()
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        if tuple(self.agents) != tuple(self.possible_agents):
            raise RuntimeError(
                "CoalitionLTLShield requires every possible agent to be active at reset."
            )

        state = self.tabular_env.get_state_id()
        self._check_runtime_labels(observations, state)
        q = int(self._next_q[self._q_index[self._dfa.initial], state])
        product_state = self._checked_product_state(q, state)
        infos = self._decorate_infos(
            infos,
            product_state=product_state,
            masks_available=True,
            proposed=None,
            executed=None,
        )
        # Publish an active runtime state only after every reset check succeeds.
        self._base_state = state
        self._q = q
        self._product_state = product_state
        return observations, infos

    def step(self, actions):
        previous_product = self._require_product_state()
        previous_state = self._base_state
        previous_q = self._q
        assert previous_state is not None and previous_q is not None
        proposed = self._validate_action_mapping(actions, previous_state)

        if self.execution == "centralised":
            executed = self._centralised_actions(proposed, previous_product)
        else:
            executed = self._decentralised_actions(proposed, previous_product)

        full_action = tuple(
            executed[agent] for agent in self.tabular_env.possible_agents
        )
        full_action_index = self.tabular_env.encode_joint_action(full_action)
        # Any environment failure or post-step model mismatch invalidates the
        # runtime state until reset. Pre-step validation failures leave it intact.
        self._clear_runtime_state()
        observations, rewards, terminations, truncations, infos = self.env.step(
            executed
        )

        state = self.tabular_env.get_state_id()
        if state not in self._game.full_successors[previous_state][full_action_index]:
            raise RuntimeError(
                "Observed transition is absent from the tabular transition model."
            )
        self._check_runtime_labels(observations, state)
        q = int(self._next_q[previous_q, state])
        product_state = self._checked_product_state(q, state)

        expected_agents = tuple(self.possible_agents)
        done_by_agent = {
            agent: bool(terminations.get(agent, False))
            or bool(truncations.get(agent, False))
            for agent in expected_agents
        }
        if any(done_by_agent.values()) and not all(done_by_agent.values()):
            raise RuntimeError(
                "Per-agent removal/termination is not supported; all agents must "
                "finish the modeled game simultaneously."
            )
        done = all(done_by_agent.values())
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        if not done and tuple(self.agents) != expected_agents:
            raise RuntimeError(
                "Dynamic agent populations are not supported by this shield."
            )

        true_termination = done and any(
            bool(terminations.get(agent, False)) for agent in expected_agents
        )
        infos = self._decorate_infos(
            infos,
            product_state=product_state,
            masks_available=not true_termination,
            proposed=proposed,
            executed=executed,
        )
        # Publish the next active state only after all post-step checks succeed.
        if not done:
            self._base_state = state
            self._q = q
            self._product_state = product_state
        return observations, rewards, terminations, truncations, infos

    def _validate_action_mapping(self, actions, state: int) -> dict[str, int]:
        if not isinstance(actions, Mapping):
            raise TypeError("Parallel actions must be a mapping from agent to action.")
        expected = set(self.possible_agents)
        supplied = set(actions)
        if supplied != expected:
            raise ValueError(
                "Action mapping must contain exactly the currently modeled agents; "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}."
            )

        result: dict[str, int] = {}
        for agent in self.possible_agents:
            action = actions[agent]
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action!r} for {agent!r}.")
            primitive = int(action)
            if primitive not in self._legal_actions(state, agent):
                raise ValueError(
                    f"Action {primitive} is not model-legal for {agent!r} "
                    f"in state {state}."
                )
            result[agent] = primitive
        return result

    def _centralised_actions(
        self, proposed: dict[str, int], product_state: int
    ) -> dict[str, int]:
        coalition_action = tuple(
            proposed[agent] for agent in self.coalition_agents
        )
        proposed_index = self._coalition_action_index[coalition_action]
        mask = self.safe_joint_actions[product_state]
        if mask[proposed_index]:
            executed_index = proposed_index
        elif self.mode == "preemptive":
            raise ValueError(
                f"Coalition action {coalition_action} is unsafe in product state "
                f"{product_state}."
            )
        elif self._joint_replacement is None:
            executed_index = int(self._joint_fallback[product_state])
        else:
            executed_index = self._joint_replacement(
                product_state, proposed_index, mask.copy()
            )
        executed_index = self._validate_joint_replacement(
            executed_index, mask, product_state
        )

        executed = dict(proposed)
        for agent, primitive in self.decode_coalition_action(executed_index).items():
            executed[agent] = primitive
        return executed

    def _decentralised_actions(
        self, proposed: dict[str, int], product_state: int
    ) -> dict[str, int]:
        executed = dict(proposed)
        unsafe = [
            agent
            for agent in self.coalition_agents
            if not self.local_safe_actions[agent][product_state, proposed[agent]]
        ]
        if unsafe and self.mode == "preemptive":
            raise ValueError(
                "Unsafe decentralised actions for "
                + ", ".join(f"{agent}={proposed[agent]}" for agent in unsafe)
                + f" in product state {product_state}."
            )

        for agent in unsafe:
            mask = self.local_safe_actions[agent][product_state]
            callback = self._replacement_by_agent[agent]
            if callback is None:
                replacement = int(self._local_fallback[agent][product_state])
            else:
                replacement = callback(
                    product_state, proposed[agent], mask.copy()
                )
            if (
                not isinstance(replacement, (int, np.integer))
                or isinstance(replacement, bool)
                or not 0 <= int(replacement) < mask.size
                or not mask[int(replacement)]
            ):
                raise ValueError(
                    f"Replacement {replacement!r} is not safe for {agent!r} "
                    f"in product state {product_state}."
                )
            executed[agent] = int(replacement)

        coalition_action = tuple(
            executed[agent] for agent in self.coalition_agents
        )
        coalition_index = self._coalition_action_index[coalition_action]
        assert self.independent_joint_actions is not None
        if not self.independent_joint_actions[product_state, coalition_index]:
            raise AssertionError(
                "Internal error: local shield outputs left the certified rectangle."
            )
        return executed

    @staticmethod
    def _validate_joint_replacement(
        replacement, mask: np.ndarray, product_state: int
    ) -> int:
        if (
            not isinstance(replacement, (int, np.integer))
            or isinstance(replacement, bool)
            or not 0 <= int(replacement) < mask.size
            or not mask[int(replacement)]
        ):
            raise ValueError(
                f"Replacement coalition-action index {replacement!r} is not safe "
                f"in product state {product_state}."
            )
        return int(replacement)

    def _checked_product_state(self, q: int, state: int) -> int:
        product_state = q * self.tabular_env.n_states + state
        if not self.winning_region[product_state]:
            raise RuntimeError(
                f"Product state {product_state} is outside the coalition winning "
                "region; the safety property cannot be guaranteed."
            )
        return product_state

    def _require_product_state(self) -> int:
        if self._product_state is None:
            raise RuntimeError(
                "Call reset() before acting or requesting masks, including after "
                "episode end or a failed environment/model check."
            )
        return self._product_state

    def _clear_runtime_state(self) -> None:
        self._base_state = None
        self._q = None
        self._product_state = None

    def _decorate_infos(
        self,
        infos,
        *,
        product_state: int,
        masks_available: bool,
        proposed: Mapping[str, int] | None,
        executed: Mapping[str, int] | None,
    ):
        if not isinstance(infos, Mapping):
            raise TypeError("Parallel infos must be a mapping.")
        out = dict(infos)
        q_index, state = divmod(product_state, self.tabular_env.n_states)
        for agent in self.coalition_agents:
            raw = out.get(agent, {})
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping):
                raise TypeError(f"Info for {agent!r} must be a mapping or None.")
            info = dict(raw)
            info["shield_mode"] = self.mode
            info["shield_execution"] = self.execution
            info["shield_coalition"] = self.coalition.name or self.coalition_agents
            info["shield_product_state"] = product_state
            info["shield_automaton_state"] = self._dfa_states[q_index]
            info["shield_labels"] = set(self._state_labels[state])

            if self.execution == "centralised":
                info["shield_joint_action_mask"] = (
                    self.safe_joint_actions[product_state].copy()
                    if masks_available
                    else np.zeros(len(self.coalition_actions), dtype=bool)
                )
            else:
                info["shield_action_mask"] = (
                    self.local_safe_actions[agent][product_state].copy()
                    if masks_available
                    else np.zeros(self._action_sizes[agent], dtype=bool)
                )

            if proposed is not None and executed is not None:
                info["shield_intervened"] = proposed[agent] != executed[agent]
                info["shield_proposed_action"] = proposed[agent]
                info["shield_executed_action"] = executed[agent]
                info["shield_joint_intervened"] = any(
                    proposed[member] != executed[member]
                    for member in self.coalition_agents
                )
                info["shield_proposed_coalition_action"] = tuple(
                    proposed[member] for member in self.coalition_agents
                )
                info["shield_executed_coalition_action"] = tuple(
                    executed[member] for member in self.coalition_agents
                )
            out[agent] = info
        return out
