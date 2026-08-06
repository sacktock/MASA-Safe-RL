from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.envs.multiagent.tabular.renderers.clean_up import (
    CleanUpRenderer,
    validate_renderer_options,
)

Agent = str
Position = tuple[int, int]

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WHFFFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFHFFHHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFFFFFHFHFHFHFHFHFHHFHFFFHFW
W==============+~FHHHHHHf====W
W   P    P      ===+~SSf     W
W     P     P   P  <~Sf  P   W
W             P   P<~S>      W
W   P    P         <~S>   P  W
W               P  <~S>P     W
W     P           P<~S>      W
W           P      <~S> P    W
W  P             P <~S>      W
W^T^T^T^T^T^T^T^T^T;~S,^T^T^TW
WBBBBBBBBBBBBBBBBBBBssBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
""".strip()

ORIENTATIONS = ("N", "E", "S", "W")
ABS_DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RELATIVE_MOVE_TO_OFFSET = {1: 0, 2: 1, 3: 2, 4: 3}
ACTION_TO_COMPONENTS = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0),
    2: (3, 0, 0, 0),
    3: (4, 0, 0, 0),
    4: (2, 0, 0, 0),
    5: (0, -1, 0, 0),
    6: (0, 1, 0, 0),
    7: (0, 0, 1, 0),
    8: (0, 0, 0, 1),
}

SAND_CHARS = {" ", "P", "=", "+", "f", "<", ">", ";", ",", "^", "T"}
GRASS_CHARS = {"B", "s"}
GRASS_EDGE_CHARS = {"T", "^", ";", ","}
RIVER_CHARS = {"H", "F", "~", "S"}
APPLE_CHARS = {"B", "T"}
DIRT_CHARS = {"H", "F"}

FEATURE_NAMES = (
    "position_x",
    "position_y",
    "orientation_n",
    "orientation_e",
    "orientation_s",
    "orientation_w",
    "active",
    "respawn_timer",
    "dirt_count",
    "clean_dirt_count",
    "ready_to_shoot",
    "others_cleaned",
    "player_cleaned",
    "edible_consumed",
    "dirt_spawned",
    "fired",
    "got_zapped",
    "avatar_respawned",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class Apple:
    position: Position
    live: bool = False


@dataclass
class Dirt:
    position: Position
    active: bool = False


@dataclass
class Avatar:
    agent: Agent
    index: int
    position: Position
    orientation: int
    reward: float = 0.0
    active: bool = True
    respawn_timer: int = 0
    zap_cooldown: int = 0
    clean_cooldown: int = 0


def label_fn(obs: np.ndarray | list[float]) -> set[str]:
    values = np.asarray(obs).reshape(-1)
    if values.size != N_FEATURES:
        raise ValueError(f"CleanUp label_fn expected {N_FEATURES} features, got {values.size}.")

    labels = {"active" if values[FEATURE_INDEX["active"]] else "inactive"}
    if not values[FEATURE_INDEX["active"]]:
        labels.add("respawning")
    if values[FEATURE_INDEX["others_cleaned"]]:
        labels.add("team_cleaning")
    if values[FEATURE_INDEX["player_cleaned"]]:
        labels.add("cleaned")
    if values[FEATURE_INDEX["edible_consumed"]]:
        labels.add("ate_apple")
    if values[FEATURE_INDEX["dirt_spawned"]]:
        labels.add("dirt_spawned")
    if values[FEATURE_INDEX["fired"]]:
        labels.add("fired")
    if values[FEATURE_INDEX["got_zapped"]]:
        labels.update({"got_zapped", "unsafe"})
    if values[FEATURE_INDEX["avatar_respawned"]]:
        labels.add("respawned")

    dirt = values[FEATURE_INDEX["dirt_count"]]
    clean = values[FEATURE_INDEX["clean_dirt_count"]]
    if dirt + clean > 0 and dirt / (dirt + clean) >= 0.4:
        labels.add("dirty_world")
    return labels


def cost_fn(labels: set[str]) -> float:
    return float("unsafe" in labels)


class CleanUp(ParallelEnv):
    """Tabular Clean Up social dilemma with simultaneous actions."""

    metadata = {
        "name": "clean_up_tabular_v0",
        "render_modes": ["ansi", "human", "rgb_array"],
        "render_fps": 8,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        num_agents: int = 7,
        max_episode_steps: int = 5000,
        render_mode: Literal["ansi", "human", "rgb_array"] | None = None,
        render_window_size: int = 900,
        flatten_observations: bool = True,
        max_apple_growth_rate: float = 0.05,
        dirt_spawn_probability: float = 0.5,
        delay_start_of_dirt_spawning: int = 50,
        frames_till_respawn: int = 50,
        stochastic_min_steps: int = 1000,
        stochastic_interval: int = 100,
        stochastic_termination_probability: float = 0.2,
        seed: int | None = None,
    ) -> None:
        validate_renderer_options(render_mode, render_window_size)
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if not 0.0 <= max_apple_growth_rate <= 1.0:
            raise ValueError("max_apple_growth_rate must be in [0, 1].")
        if not 0.0 <= dirt_spawn_probability <= 1.0:
            raise ValueError("dirt_spawn_probability must be in [0, 1].")
        if not 0.0 <= stochastic_termination_probability <= 1.0:
            raise ValueError("stochastic_termination_probability must be in [0, 1].")
        if min(frames_till_respawn, stochastic_interval) <= 0:
            raise ValueError("frames_till_respawn and stochastic_interval must be positive.")

        self._num_agents = int(num_agents)
        self.max_episode_steps = int(max_episode_steps)
        self.max_apple_growth_rate = float(max_apple_growth_rate)
        self.dirt_spawn_probability = float(dirt_spawn_probability)
        self.delay_start_of_dirt_spawning = int(delay_start_of_dirt_spawning)
        self.frames_till_respawn = int(frames_till_respawn)
        self.stochastic_min_steps = int(stochastic_min_steps)
        self.stochastic_interval = int(stochastic_interval)
        self.stochastic_termination_probability = float(stochastic_termination_probability)
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self.flatten_observations = bool(flatten_observations)
        self.label_fn = label_fn
        self.cost_fn = cost_fn

        self.possible_agents = [f"player_{index}" for index in range(self._num_agents)]
        self.agents: list[Agent] = []
        self._rng = np.random.default_rng(seed)
        self._grid = ASCII_MAP.splitlines()
        self._rows = len(self._grid)
        self._cols = len(self._grid[0])
        self._step_count = 0
        self._dirt_spawn_step = 1
        self._last_events: list[dict[str, Any]] = []
        self._last_zap_cells: set[Position] = set()
        self._last_clean_cells: set[Position] = set()
        self._cleaned_agents: set[Agent] = set()
        self._avatars: dict[Agent, Avatar] = {}
        self._parse_map()

        if self._num_agents > len(self._spawn_points):
            raise ValueError("num_agents exceeds the available spawn points.")
        obs_shape = (N_FEATURES,) if self.flatten_observations else (1, 1, N_FEATURES)
        self.observation_spaces = {
            agent: spaces.Box(0.0, np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(len(ACTION_TO_COMPONENTS))
            for agent in self.possible_agents
        }
        self.state_space = spaces.Box(
            0.0,
            np.inf,
            shape=(self._num_agents * N_FEATURES,),
            dtype=np.float32,
        )
        self._renderer = CleanUpRenderer(self)

    def observation_space(self, agent: Agent):
        return self.observation_spaces[agent]

    def action_space(self, agent: Agent):
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[Agent, np.ndarray], dict[Agent, dict[str, Any]]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._dirt_spawn_step = 1
        self._last_events = []
        self._last_zap_cells = set()
        self._last_clean_cells = set()
        self._cleaned_agents = set()
        self._parse_map()

        spawns = list(self._spawn_points)
        self._rng.shuffle(spawns)
        self._avatars = {
            agent: Avatar(
                agent=agent,
                index=index,
                position=spawns[index],
                orientation=int(self._rng.integers(4)),
            )
            for index, agent in enumerate(self.possible_agents)
        }
        if self.render_mode == "human":
            self.render()
        return self._observations(), self._infos()

    def step(self, actions: dict[Agent, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        current_agents = list(self.agents)
        self._last_events = []
        self._last_zap_cells = set()
        self._last_clean_cells = set()
        self._cleaned_agents = set()
        for avatar in self._avatars.values():
            avatar.reward = 0.0

        self._advance_respawns()
        decoded = {
            agent: ACTION_TO_COMPONENTS[int(actions.get(agent, 0))]
            for agent in current_agents
        }
        for agent, (_, turn, _, _) in decoded.items():
            avatar = self._avatars[agent]
            if avatar.active:
                avatar.orientation = (avatar.orientation + turn) % 4

        self._fire_zaps(current_agents, decoded)
        self._fire_cleaners(current_agents, decoded)
        self._move(current_agents, decoded)
        for avatar in self._avatars.values():
            if avatar.active:
                self._collect_apple(avatar)
        self._grow_apples()
        self._spawn_dirt()
        self._step_count += 1

        terminated = self._should_terminate()
        truncated = self._step_count >= self.max_episode_steps and not terminated
        observations = self._observations()
        rewards = {agent: self._avatars[agent].reward for agent in current_agents}
        terminations = {agent: terminated for agent in current_agents}
        truncations = {agent: truncated for agent in current_agents}
        infos = self._infos()
        if terminated or truncated:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        if not self._avatars:
            return np.zeros(self.state_space.shape, dtype=np.float32)
        return np.concatenate(
            [self._observation(agent).reshape(-1) for agent in self.possible_agents]
        )

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)

    def _parse_map(self) -> None:
        if len({len(row) for row in self._grid}) != 1:
            raise ValueError("Clean Up map must be rectangular.")
        self._walls: set[Position] = set()
        self._sand: set[Position] = set()
        self._grass: set[Position] = set()
        self._grass_edges: set[Position] = set()
        self._river: set[Position] = set()
        self._spawn_points: list[Position] = []
        self._apples: dict[Position, Apple] = {}
        self._dirts: dict[Position, Dirt] = {}
        for y, row in enumerate(self._grid):
            for x, char in enumerate(row):
                position = (x, y)
                if char == "W":
                    self._walls.add(position)
                    continue
                if char in SAND_CHARS:
                    self._sand.add(position)
                if char in GRASS_CHARS:
                    self._grass.add(position)
                if char in GRASS_EDGE_CHARS:
                    self._grass_edges.add(position)
                if char in RIVER_CHARS:
                    self._river.add(position)
                if char == "P":
                    self._spawn_points.append(position)
                if char in APPLE_CHARS:
                    self._apples[position] = Apple(position)
                if char in DIRT_CHARS:
                    self._dirts[position] = Dirt(position, active=char == "F")

    def _advance_respawns(self) -> None:
        for avatar in self._avatars.values():
            if avatar.active:
                continue
            avatar.respawn_timer -= 1
            if avatar.respawn_timer <= 0:
                occupied = {other.position for other in self._avatars.values() if other.active}
                candidates = [p for p in self._spawn_points if p not in occupied]
                if not candidates:
                    candidates = self._spawn_points
                avatar.position = candidates[int(self._rng.integers(len(candidates)))]
                avatar.orientation = int(self._rng.integers(4))
                avatar.active = True
                avatar.respawn_timer = 0
                avatar.zap_cooldown = 0
                avatar.clean_cooldown = 0
                self._last_events.append(
                    {"event": "avatar_respawned", "player_index": avatar.index}
                )

    def _move(
        self,
        agents: list[Agent],
        decoded: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        occupied = {
            avatar.position: avatar.agent for avatar in self._avatars.values() if avatar.active
        }
        desired: dict[Agent, Position] = {}
        for agent in agents:
            avatar = self._avatars[agent]
            move = decoded[agent][0]
            if not avatar.active:
                continue
            if not move:
                desired[agent] = avatar.position
                continue
            direction = (avatar.orientation + RELATIVE_MOVE_TO_OFFSET[move]) % 4
            dx, dy = ABS_DELTAS[direction]
            target = (avatar.position[0] + dx, avatar.position[1] + dy)
            desired[agent] = avatar.position if self._blocked(target) else target
        counts = Counter(desired.values())
        for agent, target in desired.items():
            avatar = self._avatars[agent]
            occupant = occupied.get(target)
            if target != avatar.position and counts[target] == 1 and occupant in (None, agent):
                avatar.position = target

    def _fire_zaps(
        self,
        agents: list[Agent],
        decoded: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        for agent in agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            if avatar.zap_cooldown > 0:
                avatar.zap_cooldown -= 1
            elif decoded[agent][2]:
                avatar.zap_cooldown = 10
                self._last_events.append({"event": "fired", "player_index": avatar.index})
                self._zap_from(avatar)

    def _fire_cleaners(
        self,
        agents: list[Agent],
        decoded: dict[Agent, tuple[int, int, int, int]],
    ) -> None:
        for agent in agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            if avatar.clean_cooldown > 0:
                avatar.clean_cooldown -= 1
            elif decoded[agent][3]:
                avatar.clean_cooldown = 2
                self._clean_from(avatar)

    def _zap_from(self, source: Avatar) -> None:
        hit: set[Agent] = set()
        for ray in self._beam_rays(source, length=3, radius=1):
            for position in ray:
                if self._blocked(position):
                    break
                self._last_zap_cells.add(position)
                target = self._avatar_at(position, exclude=source.agent)
                if target is None or target.agent in hit:
                    continue
                hit.add(target.agent)
                target.active = False
                target.respawn_timer = self.frames_till_respawn
                target.zap_cooldown = 0
                target.clean_cooldown = 0
                self._last_events.append(
                    {"event": "zap", "source": source.index, "target": target.index}
                )
                break

    def _clean_from(self, cleaner: Avatar) -> None:
        cleaned = False
        for ray in self._beam_rays(cleaner, length=3, radius=1):
            for position in ray:
                if self._blocked(position):
                    break
                self._last_clean_cells.add(position)
                dirt = self._dirts.get(position)
                if dirt is not None and dirt.active:
                    dirt.active = False
                    cleaned = True
                    self._last_events.append(
                        {"event": "player_cleaned", "player_index": cleaner.index}
                    )
                    break
        if cleaned:
            self._cleaned_agents.add(cleaner.agent)

    def _beam_rays(self, avatar: Avatar, *, length: int, radius: int) -> list[list[Position]]:
        rays = []
        for lateral in range(-radius, radius + 1):
            ray_length = length - abs(lateral)
            rays.append(
                [
                    _relative_to_world(avatar.position, avatar.orientation, lateral, forward)
                    for forward in range(1, ray_length + 1)
                ]
            )
        return rays

    def _collect_apple(self, avatar: Avatar) -> None:
        apple = self._apples.get(avatar.position)
        if apple is None or not apple.live:
            return
        apple.live = False
        avatar.reward += 1.0
        self._last_events.append(
            {"event": "edible_consumed", "player_index": avatar.index}
        )

    def _grow_apples(self) -> None:
        dirt = self._dirt_count()
        clean = self._clean_dirt_count()
        if dirt + clean == 0:
            return
        interpolation = np.clip((dirt / (dirt + clean) - 0.4) / -0.4, 0.0, 1.0)
        probability = self.max_apple_growth_rate * interpolation
        for apple in self._apples.values():
            if not apple.live and self._rng.random() < probability:
                apple.live = True

    def _spawn_dirt(self) -> None:
        if (
            self._dirt_spawn_step > self.delay_start_of_dirt_spawning
            and self._rng.random() < self.dirt_spawn_probability
        ):
            candidates = [dirt for dirt in self._dirts.values() if not dirt.active]
            if candidates:
                dirt = candidates[int(self._rng.integers(len(candidates)))]
                dirt.active = True
                self._last_events.append({"event": "dirt_spawned", "position": dirt.position})
        self._dirt_spawn_step += 1

    def _should_terminate(self) -> bool:
        return (
            self._step_count >= self.stochastic_min_steps
            and self._step_count % self.stochastic_interval == 0
            and self._rng.random() < self.stochastic_termination_probability
        )

    def _dirt_count(self) -> int:
        return sum(dirt.active for dirt in self._dirts.values())

    def _clean_dirt_count(self) -> int:
        return sum(not dirt.active for dirt in self._dirts.values())

    def _blocked(self, position: Position) -> bool:
        x, y = position
        return not (0 <= x < self._cols and 0 <= y < self._rows) or position in self._walls

    def _avatar_at(self, position: Position, exclude: Agent | None = None) -> Avatar | None:
        return next(
            (
                avatar
                for avatar in self._avatars.values()
                if avatar.active and avatar.agent != exclude and avatar.position == position
            ),
            None,
        )

    def _event_count(self, name: str, index: int, key: str = "player_index") -> float:
        return float(
            sum(event.get("event") == name and event.get(key) == index for event in self._last_events)
        )

    def _observation(self, agent: Agent) -> np.ndarray:
        avatar = self._avatars[agent]
        orientation = [float(avatar.orientation == index) for index in range(4)]
        values = np.asarray(
            [
                avatar.position[0],
                avatar.position[1],
                *orientation,
                float(avatar.active),
                avatar.respawn_timer,
                self._dirt_count(),
                self._clean_dirt_count(),
                float(avatar.active) * max(0.0, 1.0 - avatar.zap_cooldown / 10),
                sum(cleaner != agent for cleaner in self._cleaned_agents),
                self._event_count("player_cleaned", avatar.index),
                self._event_count("edible_consumed", avatar.index),
                sum(event.get("event") == "dirt_spawned" for event in self._last_events),
                self._event_count("fired", avatar.index),
                self._event_count("zap", avatar.index, "target"),
                self._event_count("avatar_respawned", avatar.index),
            ],
            dtype=np.float32,
        )
        return values if self.flatten_observations else values.reshape(1, 1, -1)

    def _observations(self) -> dict[Agent, np.ndarray]:
        return {agent: self._observation(agent) for agent in self.possible_agents}

    def _infos(self) -> dict[Agent, dict[str, Any]]:
        return {
            agent: {
                "position": avatar.position,
                "orientation": ORIENTATIONS[avatar.orientation],
                "active": avatar.active,
                "respawn_timer": avatar.respawn_timer,
                "dirt_count": self._dirt_count(),
                "clean_dirt_count": self._clean_dirt_count(),
                "events": list(self._last_events),
            }
            for agent, avatar in self._avatars.items()
        }


def _relative_to_world(
    origin: Position,
    orientation: int,
    rel_right: int,
    rel_forward: int,
) -> Position:
    if orientation == 0:
        dx, dy = rel_right, -rel_forward
    elif orientation == 1:
        dx, dy = rel_forward, rel_right
    elif orientation == 2:
        dx, dy = -rel_right, rel_forward
    else:
        dx, dy = -rel_forward, -rel_right
    return origin[0] + dx, origin[1] + dy


__all__ = [
    "ACTION_TO_COMPONENTS",
    "ASCII_MAP",
    "CleanUp",
    "FEATURE_NAMES",
    "cost_fn",
    "label_fn",
]
