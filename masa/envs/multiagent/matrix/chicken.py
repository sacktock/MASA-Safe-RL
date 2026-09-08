from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np
from gymnasium.spaces import Box, Discrete

from masa.envs.multiagent.tabular_env import TabularParallelEnv
from masa.envs.multiagent.matrix._label_utils import binary_cost, flatten_binary_obs


class Actions(IntEnum):
    Swerve = 0
    Straight = 1


def label_fn(obs):
    obs_vec = flatten_binary_obs(obs)
    if obs_vec.size != 5:
        raise ValueError(
            f"ChickenMatrix label_fn expected 5 channels, got {obs_vec.size}."
        )

    labels = set()
    if obs_vec[0]:
        labels.add("player_0_swerve")
    if obs_vec[1]:
        labels.add("player_0_straight")
    if obs_vec[2]:
        labels.add("player_1_swerve")
    if obs_vec[3]:
        labels.add("player_1_straight")

    if obs_vec[0] and obs_vec[2]:
        labels.add("swerve_swerve")
    elif obs_vec[0] and obs_vec[3]:
        labels.add("swerve_straight")
    elif obs_vec[1] and obs_vec[2]:
        labels.add("straight_swerve")
    elif obs_vec[1] and obs_vec[3]:
        labels.add("straight_straight")

    if obs_vec[4]:
        labels.update({"crash", "unsafe"})

    return labels


def cost_fn(labels):
    return binary_cost(labels)


class ChickenMatrix(TabularParallelEnv):
    """Repeated two-player Chicken with an exact five-state tabular model.

    Stage game (row versus column):

    * Straight/Swerve: ``(T, S)``
    * Swerve/Straight: ``(S, T)``
    * Swerve/Swerve: ``(R, R)``
    * Straight/Straight: ``(P, P)`` and a crash

    Tabular state ``0`` is the reset state. States ``1`` through ``4`` encode
    the previous joint action in canonical order ``(0,0), (0,1), (1,0),
    (1,1)``. The transition model is therefore deterministic and independent
    of the previous state.
    """

    metadata = {
        "name": "chicken_matrix_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_moves: int = 200,
        T: float = 3.0,
        R: float = 2.0,
        S: float = 1.0,
        P: float = 0.0,
        flatten_observations: bool = True,
        render_mode=None,
        seed: int | None = None,
    ):
        super().__init__()
        assert num_agents == 2, "ChickenMatrix currently supports exactly 2 agents."
        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]

        self.T, self.R, self.S, self.P = float(T), float(R), float(S), float(P)
        self.max_moves = int(max_moves)

        self.n_obs_types = 2 * self.n_agents + 1
        self.flatten_observations = bool(flatten_observations)
        self.rng = np.random.RandomState(0 if seed is None else seed)

        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}
        self._last_crash = False
        self._state = 0

        self.render_mode = render_mode
        self._renderer = None
        self._cum_rewards: dict[str, float] = {}
        self.label_fn = label_fn
        self.cost_fn = cost_fn

        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        self._n_states = 5
        self._transition_matrix = self._make_transition_matrix()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        del agent
        if self.flatten_observations:
            return Box(low=0, high=1, shape=(self.n_obs_types,), dtype=np.uint8)
        return Box(
            low=0,
            high=1,
            shape=(1, 1, self.n_obs_types),
            dtype=np.uint8,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        del agent
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {agent: None for agent in self.agents}
        self._last_crash = False
        self._state = 0
        self._cum_rewards = {agent: 0.0 for agent in self.agents}

        observations = self.observations_from_state(self._state)
        infos = {
            agent: {
                "round": self._round,
                "last_actions": None,
                "crash": False,
            }
            for agent in self.agents
        }
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        a0, a1 = self.possible_agents
        act0 = int(actions.get(a0, Actions.Swerve))
        act1 = int(actions.get(a1, Actions.Swerve))

        self._last_crash = (
            act0 == Actions.Straight and act1 == Actions.Straight
        )
        if act0 == Actions.Straight and act1 == Actions.Swerve:
            r0, r1 = self.T, self.S
            outcome = "Straight-Swerve"
        elif act0 == Actions.Swerve and act1 == Actions.Straight:
            r0, r1 = self.S, self.T
            outcome = "Swerve-Straight"
        elif act0 == Actions.Swerve and act1 == Actions.Swerve:
            r0, r1 = self.R, self.R
            outcome = "Swerve-Swerve"
        else:
            r0, r1 = self.P, self.P
            outcome = "Straight-Straight"

        rewards = {a0: float(r0), a1: float(r1)}
        self._last_actions[a0] = act0
        self._last_actions[a1] = act1
        self._state = 1 + self.encode_joint_action((act0, act1))
        self._cum_rewards[a0] += r0
        self._cum_rewards[a1] += r1

        self._round += 1
        env_trunc = self._round >= self.max_moves
        terminations = {a0: False, a1: False}
        truncations = {a0: env_trunc, a1: env_trunc}
        infos = {
            a0: {
                "round": self._round,
                "last_actions": (act0, act1),
                "outcome": outcome,
                "crash": self._last_crash,
            },
            a1: {
                "round": self._round,
                "last_actions": (act1, act0),
                "outcome": outcome,
                "crash": self._last_crash,
            },
        }

        observations = self.observations_from_state(self._state)
        if env_trunc:
            self.agents = []

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None
        if not self._renderer:
            raise ValueError(
                "Renderer missing; create env with render_mode='human' or "
                "'rgb_array'."
            )

        pay = {"T": self.T, "R": self.R, "S": self.S, "P": self.P}
        last = (
            self._last_actions.get("player_0"),
            self._last_actions.get("player_1"),
        )
        return self._renderer.render(
            round_no=self._round,
            last_actions=last,
            last_crash=self._last_crash,
            cum_rewards=self._cum_rewards,
            payoffs=pay,
        )

    def close(self):
        if self._renderer:
            self._renderer.close()

    def _make_transition_matrix(self) -> np.ndarray:
        matrix = np.zeros(
            (self.n_states, self.n_states, self.n_joint_actions),
            dtype=np.float32,
        )
        for state in range(self.n_states):
            for action_index in range(self.n_joint_actions):
                matrix[1 + action_index, state, action_index] = 1.0
        return matrix

    def _observation_from_state(self, state: int) -> np.ndarray:
        state = self._check_state(state)
        values = np.zeros(self.n_obs_types, dtype=np.uint8)
        if state:
            action = self.decode_joint_action(state - 1)
            action_0 = action["player_0"]
            action_1 = action["player_1"]
            values[action_0] = 1
            values[2 + action_1] = 1
            values[4] = int(
                action_0 == Actions.Straight and action_1 == Actions.Straight
            )
        if self.flatten_observations:
            return values
        return values.reshape(1, 1, -1)

    def observations_from_state(self, state: int) -> dict[str, np.ndarray]:
        observation = self._observation_from_state(state)
        return {
            agent: observation.copy() for agent in self.possible_agents
        }

    def _obs(self):
        return self._observation_from_state(self.get_state_id())

    @property
    def state_space(self):
        if self.flatten_observations:
            return Box(
                low=0,
                high=1,
                shape=(self.n_obs_types,),
                dtype=np.uint8,
            )
        return Box(
            low=0,
            high=1,
            shape=(1, 1, self.n_obs_types),
            dtype=np.uint8,
        )

    def state(self):
        return self._obs()

    def num_cells(self) -> int:
        """Chicken is non-spatial, so the global observation has one cell."""
        return 1

    def channel_names(self) -> list[str]:
        return [
            "player_0_swerve",
            "player_0_straight",
            "player_1_swerve",
            "player_1_straight",
            "crash",
        ]

    def action_names(self, action: int) -> str:
        try:
            name = Actions(int(action)).name
        except ValueError:
            return f"action_{action}"
        return "".join(
            f"_{char.lower()}" if char.isupper() and index > 0 else char.lower()
            for index, char in enumerate(name)
        )

    def get_state(self):
        """Return a fully restorable runtime snapshot."""
        agents = tuple(self.agents)
        p0, p1 = self.possible_agents
        last_actions = (
            self._last_actions.get(p0),
            self._last_actions.get(p1),
        )
        cum_rewards = (
            float(self._cum_rewards.get(p0, 0.0)),
            float(self._cum_rewards.get(p1, 0.0)),
        )
        return (
            agents,
            int(self._round),
            last_actions,
            bool(self._last_crash),
            cum_rewards,
            self.rng.get_state(),
        )

    def set_state(self, state):
        (
            agents,
            round_no,
            last_actions,
            last_crash,
            cum_rewards,
            rng_state,
        ) = state

        if len(last_actions) != 2:
            raise ValueError(
                "ChickenMatrix.set_state: last_actions must have length 2."
            )
        if len(cum_rewards) != 2:
            raise ValueError(
                "ChickenMatrix.set_state: cum_rewards must have length 2."
            )

        self.agents = list(agents)
        self._round = int(round_no)
        p0, p1 = self.possible_agents
        self._last_actions = {
            p0: None if last_actions[0] is None else int(last_actions[0]),
            p1: None if last_actions[1] is None else int(last_actions[1]),
        }
        self._last_crash = bool(last_crash)
        self._cum_rewards = {
            p0: float(cum_rewards[0]),
            p1: float(cum_rewards[1]),
        }

        values = tuple(self._last_actions.values())
        if values == (None, None):
            if self._last_crash:
                raise ValueError("Reset Chicken state cannot contain a crash.")
            self._state = 0
        elif any(value is None for value in values):
            raise ValueError("Chicken last actions must both be set or both be None.")
        else:
            action = tuple(int(value) for value in values)
            expected_crash = action == (Actions.Straight, Actions.Straight)
            if self._last_crash != expected_crash:
                raise ValueError("Chicken crash flag disagrees with last actions.")
            self._state = 1 + self.encode_joint_action(action)

        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))
