from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.envs.multiagent.tabular.renderers.markov_stag_hunt import (
    MarkovStagHuntRenderer,
    validate_renderer_options,
)

Position = tuple[int, int]


class Actions(IntEnum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
    Stay = 4


def label_fn(obs: np.ndarray) -> set[str]:
    values = np.asarray(obs, dtype=np.uint8).reshape(-1)
    if values.size == 0:
        raise ValueError("MarkovStagHunt label_fn received an empty observation.")
    if values[-1]:
        return {"mauled", "unsafe"}
    return {"safe"}


def cost_fn(labels: set[str]) -> float:
    return float("unsafe" in labels)


class MarkovStagHunt(ParallelEnv):
    """Fully observable cooperative stag-hunt gridworld."""

    metadata = {
        "name": "markov_stag_hunt_v0",
        "render_modes": ["ansi", "human", "rgb_array"],
        "render_fps": 5,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (5, 5),
        num_agents: int = 2,
        num_plants: int = 2,
        num_stags: int = 1,
        max_moves: int = 500,
        stag_move_prob: float = 0.2,
        min_hunters_for_stag: int = 2,
        plant_reward: float = 2.0,
        stag_reward: float = 10.0,
        maul_penalty: float = -2.0,
        flatten_observations: bool = True,
        render_mode: Literal["ansi", "human", "rgb_array"] | None = None,
        render_window_size: int = 720,
        seed: int | None = None,
    ) -> None:
        rows, cols = (int(grid_size[0]), int(grid_size[1]))
        validate_renderer_options(render_mode, render_window_size)
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_size dimensions must be positive.")
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if min(num_plants, num_stags) < 0:
            raise ValueError("num_plants and num_stags must be non-negative.")
        if num_plants + num_stags >= rows * cols:
            raise ValueError("The grid needs free space for at least one agent.")
        if max_moves <= 0:
            raise ValueError("max_moves must be positive.")
        if not 0.0 <= stag_move_prob <= 1.0:
            raise ValueError("stag_move_prob must be in [0, 1].")
        if min_hunters_for_stag <= 0:
            raise ValueError("min_hunters_for_stag must be positive.")

        self.grid_size = (rows, cols)
        self._rows = rows
        self._cols = cols
        self.n_agents = int(num_agents)
        self.num_plants_target = int(num_plants)
        self.num_stags_target = int(num_stags)
        self.max_moves = int(max_moves)
        self.stag_move_prob = float(stag_move_prob)
        self.min_hunters_for_stag = int(min_hunters_for_stag)
        self.plant_reward = float(plant_reward)
        self.stag_reward = float(stag_reward)
        self.maul_penalty = float(maul_penalty)
        self.flatten_observations = bool(flatten_observations)
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self.label_fn = label_fn
        self.cost_fn = cost_fn

        self.possible_agents = [f"player_{index}" for index in range(self.n_agents)]
        self.agents: list[str] = []
        self._plant_channel = self.n_agents
        self._stag_channel = self.n_agents + 1
        self._maul_channel = self.n_agents + 2
        self.n_obs_types = self.n_agents + 3
        self._rng = np.random.default_rng(seed)
        self._action_deltas = np.asarray(
            [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)],
            dtype=np.int8,
        )

        obs_shape = (
            (rows * cols * self.n_obs_types,)
            if self.flatten_observations
            else (rows, cols, self.n_obs_types)
        )
        self.observation_spaces = {
            agent: spaces.Box(0, 1, shape=obs_shape, dtype=np.uint8)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(len(Actions)) for agent in self.possible_agents
        }
        self.state_space = spaces.Box(0, 1, shape=obs_shape, dtype=np.uint8)

        self.num_moves = 0
        self.results = {"stags_hunted": 0, "plants_harvested": 0, "maulings": 0}
        self.agent_positions: dict[str, Position] = {}
        self.agent_counts = np.zeros(self.grid_size, dtype=np.int16)
        self.plants: set[Position] = set()
        self.stags: set[Position] = set()
        self._last_mauled_agents: set[str] = set()
        self._last_hunters: set[str] = set()
        self._last_foragers: set[str] = set()
        self._renderer = MarkovStagHuntRenderer(self)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.num_moves = 0
        self.results = {"stags_hunted": 0, "plants_harvested": 0, "maulings": 0}
        self.agent_positions = {}
        self.agent_counts = np.zeros(self.grid_size, dtype=np.int16)
        self.plants = set()
        self.stags = set()
        self._last_mauled_agents = set()
        self._last_hunters = set()
        self._last_foragers = set()

        for agent in self.possible_agents:
            position = (
                int(self._rng.integers(self._rows)),
                int(self._rng.integers(self._cols)),
            )
            self.agent_positions[agent] = position
            self.agent_counts[position] += 1
        self._respawn_entities()
        if self.render_mode == "human":
            self.render()
        return self._observations(), self._infos()

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        current_agents = list(self.agents)
        self._last_mauled_agents = set()
        self._last_hunters = set()
        self._last_foragers = set()

        new_positions: dict[str, Position] = {}
        new_counts = np.zeros_like(self.agent_counts)
        for agent in current_agents:
            action = int(actions.get(agent, Actions.Stay))
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent}.")
            row, col = self.agent_positions[agent]
            d_row, d_col = self._action_deltas[action]
            position = (
                int(np.clip(row + d_row, 0, self._rows - 1)),
                int(np.clip(col + d_col, 0, self._cols - 1)),
            )
            new_positions[agent] = position
            new_counts[position] += 1

        rewards = {agent: 0.0 for agent in current_agents}
        harvested_cells: set[Position] = set()
        for agent, position in new_positions.items():
            if position in self.plants:
                rewards[agent] += self.plant_reward
                harvested_cells.add(position)
                self._last_foragers.add(agent)
        self.plants.difference_update(harvested_cells)
        self.results["plants_harvested"] += len(harvested_cells)

        hunted_stags: set[Position] = set()
        for stag in self.stags:
            present = [agent for agent, position in new_positions.items() if position == stag]
            if len(present) >= self.min_hunters_for_stag:
                hunted_stags.add(stag)
                self._last_hunters.update(present)
                for agent in present:
                    rewards[agent] += self.stag_reward
            elif present:
                self._last_mauled_agents.update(present)
                for agent in present:
                    rewards[agent] += self.maul_penalty
                self.results["maulings"] += 1
        self.stags.difference_update(hunted_stags)
        self.results["stags_hunted"] += len(hunted_stags)

        self.agent_positions = new_positions
        self.agent_counts = new_counts
        self.num_moves += 1

        if self._last_mauled_agents:
            observations = self._observations()
            rewards_out = rewards
            terminations = {agent: True for agent in current_agents}
            truncations = {agent: False for agent in current_agents}
            infos = self._infos(maul_reset=True)
            self.agents = []
            if self.render_mode == "human":
                self.render()
            return observations, rewards_out, terminations, truncations, infos

        self._move_stags_towards_agents()
        self._respawn_entities()
        truncated = self.num_moves >= self.max_moves
        observations = self._observations()
        terminations = {agent: False for agent in current_agents}
        truncations = {agent: truncated for agent in current_agents}
        infos = self._infos()
        if truncated:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self._observation(None)

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)

    def channel_names(self) -> list[str]:
        return [*self.possible_agents, "plant", "stag", "self_mauled"]

    def action_names(self, action: int) -> str:
        try:
            return Actions(int(action)).name.lower()
        except ValueError:
            return f"action_{action}"

    def env_logging_info(self, suffix: str = "") -> dict[str, int]:
        return {f"{name}{suffix}": value for name, value in self.results.items()}

    def _observations(self) -> dict[str, np.ndarray]:
        return {agent: self._observation(agent) for agent in self.possible_agents}

    def _observation(self, agent: str | None) -> np.ndarray:
        layers = np.zeros((*self.grid_size, self.n_obs_types), dtype=np.uint8)
        for index, player in enumerate(self.possible_agents):
            position = self.agent_positions.get(player)
            if position is not None:
                layers[position][index] = 1
        for position in self.plants:
            layers[position][self._plant_channel] = 1
        for position in self.stags:
            layers[position][self._stag_channel] = 1
        if agent is None:
            mauled = bool(self._last_mauled_agents)
        else:
            mauled = agent in self._last_mauled_agents
        layers[:, :, self._maul_channel] = int(mauled)
        return layers.reshape(-1) if self.flatten_observations else layers

    def _infos(self, *, maul_reset: bool = False) -> dict[str, dict[str, Any]]:
        return {
            agent: {
                "position": self.agent_positions[agent],
                "mauled": agent in self._last_mauled_agents,
                "maul_reset": maul_reset,
                "hunted_stag": agent in self._last_hunters,
                "harvested_plant": agent in self._last_foragers,
                "results": dict(self.results),
            }
            for agent in self.possible_agents
        }

    def _free_cells(self) -> list[Position]:
        occupied = set(self.agent_positions.values()) | self.plants | self.stags
        return [
            (row, col)
            for row in range(self._rows)
            for col in range(self._cols)
            if (row, col) not in occupied
        ]

    def _respawn_entities(self) -> None:
        free = self._free_cells()
        self._rng.shuffle(free)
        while len(self.plants) < self.num_plants_target and free:
            self.plants.add(free.pop())
        while len(self.stags) < self.num_stags_target and free:
            self.stags.add(free.pop())

    def _move_stags_towards_agents(self) -> None:
        if not self.stags or not self.agents:
            return
        agent_positions = np.asarray(
            [self.agent_positions[agent] for agent in self.agents], dtype=np.int16
        )
        moved: set[Position] = set()
        for row, col in sorted(self.stags):
            if self._rng.random() >= self.stag_move_prob:
                moved.add((row, col))
                continue
            distances = np.abs(agent_positions[:, 0] - row) + np.abs(
                agent_positions[:, 1] - col
            )
            target_row, target_col = agent_positions[int(np.argmin(distances))]
            d_row = int(np.sign(target_row - row))
            d_col = int(np.sign(target_col - col))
            if d_row and d_col:
                if self._rng.random() < 0.5:
                    d_col = 0
                else:
                    d_row = 0
            target = (
                int(np.clip(row + d_row, 0, self._rows - 1)),
                int(np.clip(col + d_col, 0, self._cols - 1)),
            )
            occupied = (
                target in self.agent_positions.values()
                or target in self.plants
                or target in self.stags
                or target in moved
            )
            moved.add((row, col) if occupied else target)
        self.stags = moved


__all__ = ["Actions", "MarkovStagHunt", "cost_fn", "label_fn"]
