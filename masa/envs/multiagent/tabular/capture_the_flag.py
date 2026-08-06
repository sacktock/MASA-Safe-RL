from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from masa.envs.multiagent.tabular.renderers.capture_the_flag import (
    CaptureTheFlagRenderer,
    validate_renderer_options,
)

Agent = str
Position = tuple[int, int]
Team = Literal["red", "blue"]

ASCII_MAP = """
IIIIIIIIIIIIIIIIIIIIIII
IWWWWWWWWWWWWWWWWWWWWWI
IWPPP,PPPP,F,PPPP,PPPWI
IWPPP,,PP,,,,,PP,,PPPWI
IWPPP,,,,,,,,,,,,,PPPWI
IWP,,WW,,,,,,,,,WW,,PWI
IWHHWWW,WWWWWWW,WWWHHWI
IWHHW,D,,,,,,,,,D,WHHWI
IWHH,,W,,,WWW,,,W,,HHWI
IW,,,,W,,,,,,,,,W,,,,WI
IW,,,,WWW,,,,,WWW,,,,WI
IW,,,,,,,,,I,,,,,,,,,WI
IW,,,,WWW,,,,,WWW,,,,WI
IW,,,,W,,,,,,,,,W,,,,WI
IWHH,,W,,,WWW,,,W,,HHWI
IWHHW,D,,,,,,,,,D,WHHWI
IWHHWWW,WWWWWWW,WWWHHWI
IWQ,,WW,,,,,,,,,WW,,QWI
IWQQQ,,,,,,,,,,,,,QQQWI
IWQQQ,,QQ,,,,,QQ,,QQQWI
IWQQQ,QQQQ,G,QQQQ,QQQWI
IWWWWWWWWWWWWWWWWWWWWWI
IIIIIIIIIIIIIIIIIIIIIII
""".strip()

ORIENTATIONS = ("N", "E", "S", "W")
ABS_DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RELATIVE_MOVE_TO_OFFSET = {1: 0, 2: 1, 3: 2, 4: 3}
ACTION_TO_COMPONENTS = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (3, 0, 0),
    3: (4, 0, 0),
    4: (2, 0, 0),
    5: (0, -1, 0),
    6: (0, 1, 0),
    7: (0, 0, 1),
    8: (0, 0, 2),
}

FEATURE_NAMES = (
    "position_x",
    "position_y",
    "orientation_n",
    "orientation_e",
    "orientation_s",
    "orientation_w",
    "team_red",
    "team_blue",
    "active",
    "health",
    "respawn_timer",
    "ready_to_shoot",
    "carrying_red",
    "carrying_blue",
    "both_flags_home",
    "red_flag_home",
    "blue_flag_home",
    "got_hit",
    "got_zapped",
    "flag_picked_up",
    "flag_returned",
    "flag_captured",
    "flag_dropped",
    "zapped_flag_carrier",
    "avatar_respawned",
    "fired",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class Avatar:
    agent: Agent
    index: int
    team: Team
    position: Position
    orientation: int
    health: int
    active: bool = True
    respawn_timer: int = 0
    cooling_timer: int = 0
    previous_position: Position | None = None
    carrying_flag: Team | None = None
    reward: float = 0.0


@dataclass
class Flag:
    team: Team
    home: Position
    position: Position
    carried_by: Agent | None = None

    @property
    def at_home(self) -> bool:
        return self.carried_by is None and self.position == self.home


@dataclass
class Wall:
    position: Position
    health: int

    @property
    def active(self) -> bool:
        return self.health > 0


def label_fn(obs: np.ndarray | list[float]) -> set[str]:
    values = np.asarray(obs).reshape(-1)
    if values.size != N_FEATURES:
        raise ValueError(
            f"CaptureTheFlag label_fn expected {N_FEATURES} features, got {values.size}."
        )

    labels = {"active" if values[FEATURE_INDEX["active"]] else "inactive"}
    if values[FEATURE_INDEX["team_red"]]:
        labels.add("team_red")
    if values[FEATURE_INDEX["team_blue"]]:
        labels.add("team_blue")
    if values[FEATURE_INDEX["carrying_red"]] or values[FEATURE_INDEX["carrying_blue"]]:
        labels.add("carrying_flag")

    red_home = bool(values[FEATURE_INDEX["red_flag_home"]])
    blue_home = bool(values[FEATURE_INDEX["blue_flag_home"]])
    if red_home and blue_home:
        labels.add("both_flags_home")
    elif red_home:
        labels.add("red_flag_home")
    elif blue_home:
        labels.add("blue_flag_home")
    else:
        labels.add("no_flags_home")

    event_labels = {
        "got_hit": "got_hit",
        "got_zapped": "got_zapped",
        "flag_picked_up": "flag_picked_up",
        "flag_returned": "flag_returned",
        "flag_captured": "flag_captured",
        "flag_dropped": "flag_dropped",
        "zapped_flag_carrier": "zapped_flag_carrier",
        "avatar_respawned": "respawned",
        "fired": "fired",
    }
    for feature, label in event_labels.items():
        if values[FEATURE_INDEX[feature]]:
            labels.add(label)

    if "got_hit" in labels or "got_zapped" in labels:
        labels.add("unsafe")
    if not values[FEATURE_INDEX["active"]]:
        labels.add("respawning")
    elif values[FEATURE_INDEX["health"]] <= 1:
        labels.add("low_health")
    return labels


def cost_fn(labels: set[str]) -> float:
    return float("unsafe" in labels)


class CaptureTheFlag(ParallelEnv):
    """Two-team tabular Capture the Flag with simultaneous actions."""

    metadata = {
        "name": "capture_the_flag_tabular_v0",
        "render_modes": ["ansi", "human", "rgb_array"],
        "render_fps": 8,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        num_agents: int = 8,
        max_episode_steps: int = 1000,
        render_mode: Literal["ansi", "human", "rgb_array"] | None = None,
        render_window_size: int = 900,
        flatten_observations: bool = True,
        primary_zap_cooldown: int = 2,
        secondary_zap_cooldown: int = 4,
        frames_till_respawn: int = 80,
        seed: int | None = None,
    ) -> None:
        validate_renderer_options(render_mode, render_window_size)
        if num_agents <= 0 or num_agents % 2:
            raise ValueError("num_agents must be a positive even number.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if min(primary_zap_cooldown, secondary_zap_cooldown, frames_till_respawn) <= 0:
            raise ValueError("cooldowns and frames_till_respawn must be positive.")

        self._num_agents = int(num_agents)
        self.max_episode_steps = int(max_episode_steps)
        self.primary_zap_cooldown = int(primary_zap_cooldown)
        self.secondary_zap_cooldown = int(secondary_zap_cooldown)
        self.frames_till_respawn = int(frames_till_respawn)
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
        self._last_events: list[dict[str, Any]] = []
        self._last_zap_cells: set[Position] = set()
        self._avatars: dict[Agent, Avatar] = {}
        self._parse_map()

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
        self._renderer = CaptureTheFlagRenderer(self)

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
        self._last_events = []
        self._last_zap_cells = set()
        self._parse_map()

        red_spawns = self._shuffled(self._spawn_points["red"])
        blue_spawns = self._shuffled(self._spawn_points["blue"])
        team_size = self._num_agents // 2
        if len(red_spawns) < team_size or len(blue_spawns) < team_size:
            raise ValueError("num_agents exceeds the available team spawn points.")

        offsets = {"red": 0, "blue": 0}
        self._avatars = {}
        for index, agent in enumerate(self.possible_agents):
            team: Team = "red" if index % 2 == 0 else "blue"
            spawns = red_spawns if team == "red" else blue_spawns
            position = spawns[offsets[team]]
            offsets[team] += 1
            self._avatars[agent] = Avatar(
                agent=agent,
                index=index,
                team=team,
                position=position,
                orientation=int(self._rng.integers(4)),
                health=2,
            )

        if self.render_mode == "human":
            self.render()
        return self._observations(), self._infos()

    def step(self, actions: dict[Agent, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        current_agents = list(self.agents)
        self._last_events = []
        self._last_zap_cells = set()
        for avatar in self._avatars.values():
            avatar.reward = 0.0

        self._advance_respawns()
        decoded = {
            agent: ACTION_TO_COMPONENTS[int(actions.get(agent, 0))]
            for agent in current_agents
        }
        for agent, (_, turn, _) in decoded.items():
            avatar = self._avatars[agent]
            if avatar.active:
                avatar.orientation = (avatar.orientation + turn) % 4

        self._fire(current_agents, decoded)
        self._move(current_agents, decoded)
        self._update_carried_flags()
        self._process_flags(current_agents)
        self._regenerate_health()
        self._step_count += 1

        truncated = self._step_count >= self.max_episode_steps
        observations = self._observations()
        rewards = {agent: self._avatars[agent].reward for agent in current_agents}
        terminations = {agent: False for agent in current_agents}
        truncations = {agent: truncated for agent in current_agents}
        infos = self._infos()
        if truncated:
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
        self._solid_walls: set[Position] = set()
        self._destroyable_walls: dict[Position, Wall] = {}
        self._terrain: dict[Position, str] = {}
        self._indicator_tiles: set[Position] = set()
        self._spawn_points: dict[Team, list[Position]] = {"red": [], "blue": []}
        self._flags: dict[Team, Flag] = {}
        for y, row in enumerate(self._grid):
            for x, char in enumerate(row):
                position = (x, y)
                if char == "W":
                    self._solid_walls.add(position)
                elif char in {"D", "H"}:
                    initially_destroyed = (
                        char == "D" and self._rng.integers(10) == 9
                    ) or (char == "H" and self._rng.integers(4) == 3)
                    self._destroyable_walls[position] = Wall(
                        position, 0 if initially_destroyed else 5
                    )
                elif char in {"P", "Q", ",", "F", "G"}:
                    self._terrain[position] = "clean"
                    if char == "P":
                        self._spawn_points["red"].append(position)
                    elif char == "Q":
                        self._spawn_points["blue"].append(position)
                    elif char == "F":
                        self._flags["red"] = Flag("red", position, position)
                    elif char == "G":
                        self._flags["blue"] = Flag("blue", position, position)
                elif char == "I":
                    self._indicator_tiles.add(position)

    def _shuffled(self, positions: list[Position]) -> list[Position]:
        result = list(positions)
        self._rng.shuffle(result)
        return result

    def _advance_respawns(self) -> None:
        for avatar in self._avatars.values():
            if avatar.active:
                continue
            avatar.respawn_timer -= 1
            if avatar.respawn_timer <= 0:
                occupied = {other.position for other in self._avatars.values() if other.active}
                candidates = [p for p in self._spawn_points[avatar.team] if p not in occupied]
                if not candidates:
                    candidates = self._spawn_points[avatar.team]
                avatar.position = candidates[int(self._rng.integers(len(candidates)))]
                avatar.orientation = int(self._rng.integers(4))
                avatar.active = True
                avatar.health = 2
                avatar.respawn_timer = 0
                avatar.cooling_timer = 0
                avatar.previous_position = None
                self._last_events.append(
                    {"event": "avatar_respawned", "player_index": avatar.index}
                )

    def _move(self, agents: list[Agent], decoded: dict[Agent, tuple[int, int, int]]) -> None:
        occupied = {
            avatar.position: avatar.agent for avatar in self._avatars.values() if avatar.active
        }
        desired: dict[Agent, Position] = {}
        for agent in agents:
            avatar = self._avatars[agent]
            move = decoded[agent][0]
            if not avatar.active or not move or self._on_enemy_colour(avatar):
                if avatar.active:
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

    def _fire(self, agents: list[Agent], decoded: dict[Agent, tuple[int, int, int]]) -> None:
        for agent in agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            fire = decoded[agent][2]
            if avatar.cooling_timer > 0:
                avatar.cooling_timer -= 1
            elif fire == 1:
                avatar.cooling_timer = self.primary_zap_cooldown
                self._paint(avatar.position, avatar.team)
                self._zap_from(avatar, length=3, radius=1)
                self._last_events.append({"event": "fired", "player_index": avatar.index})
            elif fire == 2 and avatar.previous_position == avatar.position:
                avatar.cooling_timer = self.secondary_zap_cooldown
                self._zap_from(avatar, length=6, radius=0)
                self._last_events.append({"event": "fired", "player_index": avatar.index})
            avatar.previous_position = avatar.position

    def _zap_from(self, source: Avatar, *, length: int, radius: int) -> None:
        hit_agents: set[Agent] = set()
        for ray in self._beam_rays(source, length, radius):
            for position in ray:
                if not self._in_bounds(position) or position in self._solid_walls:
                    break
                self._last_zap_cells.add(position)
                wall = self._destroyable_walls.get(position)
                if wall is not None and wall.active:
                    wall.health -= 1
                    if wall.health == 2:
                        self._last_events.append({"event": "wall_damaged", "position": position})
                    if wall.health <= 0:
                        self._last_events.append({"event": "wall_destroyed", "position": position})
                    else:
                        break
                self._paint(position, source.team)
                target = self._avatar_at(position, exclude=source.agent)
                if target is None or target.agent in hit_agents:
                    continue
                hit_agents.add(target.agent)
                if target.team == source.team:
                    continue
                target.health -= 1
                self._last_events.append(
                    {"event": "avatar_hit", "source": source.index, "target": target.index}
                )
                if target.health <= 0:
                    if target.carrying_flag is not None:
                        self._last_events.append(
                            {"event": "zapped_flag_carrier", "player_index": source.index}
                        )
                    self._drop_flag(target)
                    target.active = False
                    target.respawn_timer = self.frames_till_respawn
                    target.cooling_timer = 0
                    self._last_events.append(
                        {"event": "zap", "source": source.index, "target": target.index}
                    )
                    break

    def _beam_rays(self, avatar: Avatar, length: int, radius: int) -> list[list[Position]]:
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

    def _process_flags(self, agents: list[Agent]) -> None:
        for agent in agents:
            avatar = self._avatars[agent]
            if not avatar.active:
                continue
            for flag in self._flags.values():
                if flag.carried_by is not None or flag.position != avatar.position:
                    continue
                if flag.team != avatar.team:
                    flag.carried_by = agent
                    avatar.carrying_flag = flag.team
                    self._last_events.append(
                        {"event": "flag_picked_up", "player_index": avatar.index}
                    )
                elif not flag.at_home:
                    flag.position = flag.home
                    self._last_events.append(
                        {"event": "flag_returned", "player_index": avatar.index}
                    )
                elif avatar.carrying_flag is not None:
                    self._capture_flag(avatar)

    def _capture_flag(self, carrier: Avatar) -> None:
        for avatar in self._avatars.values():
            avatar.reward += 1.0 if avatar.team == carrier.team else -1.0
            avatar.carrying_flag = None
        for flag in self._flags.values():
            flag.position = flag.home
            flag.carried_by = None
        self._last_events.append(
            {"event": "flag_captured", "player_index": carrier.index, "team": carrier.team}
        )

    def _drop_flag(self, avatar: Avatar) -> None:
        if avatar.carrying_flag is None:
            return
        flag = self._flags[avatar.carrying_flag]
        flag.carried_by = None
        flag.position = avatar.position
        avatar.carrying_flag = None
        self._last_events.append({"event": "flag_dropped", "player_index": avatar.index})

    def _update_carried_flags(self) -> None:
        for flag in self._flags.values():
            if flag.carried_by is not None:
                flag.position = self._avatars[flag.carried_by].position

    def _regenerate_health(self) -> None:
        for avatar in self._avatars.values():
            if not avatar.active or self._rng.random() >= 0.05:
                continue
            colour = self._terrain.get(avatar.position)
            maximum = 3 if colour == avatar.team else 1 if colour in {"red", "blue"} else 2
            avatar.health = min(avatar.health + 1, maximum)

    def _paint(self, position: Position, team: Team) -> None:
        if position in self._terrain:
            self._terrain[position] = team

    def _on_enemy_colour(self, avatar: Avatar) -> bool:
        colour = self._terrain.get(avatar.position)
        return colour in {"red", "blue"} and colour != avatar.team

    def _blocked(self, position: Position) -> bool:
        wall = self._destroyable_walls.get(position)
        return (
            not self._in_bounds(position)
            or position in self._solid_walls
            or (wall is not None and wall.active)
        )

    def _in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self._cols and 0 <= y < self._rows

    def _avatar_at(self, position: Position, exclude: Agent | None = None) -> Avatar | None:
        return next(
            (
                avatar
                for avatar in self._avatars.values()
                if avatar.active and avatar.agent != exclude and avatar.position == position
            ),
            None,
        )

    def _event(self, name: str, index: int, key: str = "player_index") -> float:
        return float(
            any(event.get("event") == name and event.get(key) == index for event in self._last_events)
        )

    def _observation(self, agent: Agent) -> np.ndarray:
        avatar = self._avatars[agent]
        red_home = float(self._flags["red"].at_home)
        blue_home = float(self._flags["blue"].at_home)
        orientation = [float(avatar.orientation == index) for index in range(4)]
        values = np.asarray(
            [
                avatar.position[0],
                avatar.position[1],
                *orientation,
                float(avatar.team == "red"),
                float(avatar.team == "blue"),
                float(avatar.active),
                avatar.health,
                avatar.respawn_timer,
                float(avatar.active) * max(0.0, 1.0 - avatar.cooling_timer / self.primary_zap_cooldown),
                float(avatar.carrying_flag == "red"),
                float(avatar.carrying_flag == "blue"),
                red_home * blue_home,
                red_home,
                blue_home,
                self._event("avatar_hit", avatar.index, "target"),
                self._event("zap", avatar.index, "target"),
                self._event("flag_picked_up", avatar.index),
                self._event("flag_returned", avatar.index),
                self._event("flag_captured", avatar.index),
                self._event("flag_dropped", avatar.index),
                self._event("zapped_flag_carrier", avatar.index),
                self._event("avatar_respawned", avatar.index),
                self._event("fired", avatar.index),
            ],
            dtype=np.float32,
        )
        return values if self.flatten_observations else values.reshape(1, 1, -1)

    def _observations(self) -> dict[Agent, np.ndarray]:
        return {agent: self._observation(agent) for agent in self.possible_agents}

    def _infos(self) -> dict[Agent, dict[str, Any]]:
        red_home = self._flags["red"].at_home
        blue_home = self._flags["blue"].at_home
        return {
            agent: {
                "position": avatar.position,
                "orientation": ORIENTATIONS[avatar.orientation],
                "team": avatar.team,
                "active": avatar.active,
                "health": avatar.health,
                "respawn_timer": avatar.respawn_timer,
                "carrying_flag": avatar.carrying_flag,
                "flags_home": {"red": red_home, "blue": blue_home},
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
    "CaptureTheFlag",
    "FEATURE_NAMES",
    "cost_fn",
    "label_fn",
]
