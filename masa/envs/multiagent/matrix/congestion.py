from __future__ import annotations

import functools
from typing import Dict
from enum import IntEnum

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class Actions(IntEnum):
    RoadA = 0
    RoadB = 1

class CongestionMatrix(ParallelEnv):
    """
    6-agent congestion routing with 2 roads (A/B). Each agent picks a road.
    Per-road cost is linear in the number of users on that road:
        cost_A = base_A + slope_A * nA
        cost_B = base_B + slope_B * nB
    Agent reward = - cost_(chosen road)  (more congestion => lower reward)
    Optional jam penalty adds extra negative reward to agents on an overfull road.

    Defaults: 6 agents, base_A=1, base_B=1, slope_A=1, slope_B=1,
              jam_threshold=5, jam_penalty=5.

    Observations (global, binary channels), shape is (C,) if flattened or (1,1,C):
      - For each agent i in {0..N-1}:
            ch 2*i + 0 = 1 if last action was RoadA, else 0
            ch 2*i + 1 = 1 if last action was RoadB, else 0
      - One-hot for last round load of RoadA over (N+1) buckets [0..N]
      - One-hot for last round load of RoadB over (N+1) buckets [0..N]
      - Jam flag (1 if max(nA, nB) >= jam_threshold else 0)

      On the first round (after reset), all action bits are 0 and load one-hots
      reflect (0 users on A, 0 users on B); jam=0.
    """

    metadata = {"name": "congestion_matrix_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 6,
        max_moves: int = 200,
        base_A: float = 1.0,
        base_B: float = 1.0,
        slope_A: float = 1.0,
        slope_B: float = 1.0,
        jam_threshold: int = 5,
        jam_penalty: float = 5.0,
        flatten_observations: bool = True,
        render_mode=None,
        seed: int | None = None,
    ):
        assert num_agents >= 2, "CongestionMatrix needs at least 2 agents."
        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]

        # Cost params
        self.base_A = float(base_A)
        self.base_B = float(base_B)
        self.slope_A = float(slope_A)
        self.slope_B = float(slope_B)

        # Jam params
        self.jam_threshold = int(jam_threshold)
        self.jam_penalty = float(jam_penalty)

        # Horizon
        self.max_moves = int(max_moves)

        # Observation config
        #  - 2 bits per agent for last actions
        #  - (N+1) one-hot for loadA + (N+1) one-hot for loadB
        #  - 1 jam flag
        self.flatten_observations = bool(flatten_observations)
        self.n_obs_types = 2 * self.n_agents + 2 * (self.n_agents + 1) + 1

        # RNG
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # Runtime
        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}
        self._loadA: int = 0
        self._loadB: int = 0
        self._jam: bool = False
        self._cum_rewards: dict[str, float] = {}

        # Rendering
        self.render_mode = render_mode
        self._renderer = None
        # self._renderer: CongestionRenderer | None = None
        # if self.render_mode is not None:
        #     self._renderer = CongestionRenderer(self.render_mode)

        # Spaces
        self.observation_spaces = {a: self.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.action_space(a) for a in self.possible_agents}

    # ─────────────────────────── PettingZoo API ───────────────────────────
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        C = self.n_obs_types
        if self.flatten_observations:
            return Box(low=0, high=1, shape=(C,), dtype=np.uint8)
        return Box(low=0, high=1, shape=(1, 1, C), dtype=np.uint8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)  # 0=A, 1=B

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {a: None for a in self.agents}
        self._loadA = 0
        self._loadB = 0
        self._jam = False
        self._cum_rewards = {a: 0.0 for a in self.agents}

        obs = {a: self._obs() for a in self.agents}
        infos = {
            a: {"round": self._round, "last_actions": None, "loads": (self._loadA, self._loadB), "jam": self._jam}
            for a in self.agents
        }
        return obs, infos

    def step(self, actions: dict[str, int]):
        # Check that the episode is not already over
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Require one action per active agent
        expected = set(self.agents)
        received = set(actions.keys())
        if expected != received:
            missing = expected - received
            extra = received - expected
            msg_parts = []
            if missing:
                msg_parts.append(f"missing actions for agents: {sorted(missing)}")
            if extra:
                msg_parts.append(f"got actions for non-existent agents: {sorted(extra)}")
            raise ValueError("Invalid action dict: " + "; ".join(msg_parts))

        chosen: dict[str, int] = {}
        for a in self.agents:
            chosen[a] = int(actions[a])

        nA = sum(1 for a in self.possible_agents if chosen[a] == Actions.RoadA)
        nB = self.n_agents - nA

        # Costs & rewards
        costA = self.base_A + self.slope_A * float(nA)
        costB = self.base_B + self.slope_B * float(nB)

        jamA = nA >= self.jam_threshold
        jamB = nB >= self.jam_threshold
        jam_now = jamA or jamB

        rewards: Dict[str, float] = {}
        for a in self.possible_agents:
            if chosen[a] == Actions.RoadA:
                r = -costA
                if jamA:
                    r -= self.jam_penalty
            else:
                r = -costB
                if jamB:
                    r -= self.jam_penalty
            rewards[a] = float(r)

        # Update runtime state
        for a in self.possible_agents:
            self._last_actions[a] = chosen[a]
            self._cum_rewards[a] += rewards[a]
        self._loadA, self._loadB = nA, nB
        self._jam = bool(jam_now)

        # Time & termination
        self._round += 1
        env_trunc = self._round >= self.max_moves

        terminations = {a: False for a in self.possible_agents}
        truncations = {a: env_trunc for a in self.possible_agents}
        infos = {
            a: {
                "round": self._round,
                "last_actions": tuple(chosen[x] for x in self.possible_agents),
                "loads": (nA, nB),
                "jam": jam_now,
                "costs": (costA, costB),
            }
            for a in self.possible_agents
        }

        obs = {a: self._obs() for a in self.agents}
        if env_trunc:
            self.agents = []

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return
        if not self._renderer:
            raise ValueError("Renderer missing; create env with render_mode='human' or 'rgb_array'.")

        params = {
            "base_A": self.base_A,
            "base_B": self.base_B,
            "slope_A": self.slope_A,
            "slope_B": self.slope_B,
            "nA": self._loadA,
            "nB": self._loadB,
            "jam": self._jam,
        }
        last_list = [self._last_actions.get(a) for a in self.possible_agents]

        frame = self._renderer.render(
            round_no=self._round,
            last_actions=last_list,
            cum_rewards=self._cum_rewards,
            params=params,
        )
        return frame

    def close(self):
        if self._renderer:
            self._renderer.close()

    # ───────────────────────────── Internals ─────────────────────────────
    def _obs(self):
        """
        Layout (C = 2N + (N+1) + (N+1) + 1):
          [ per-agent action bits | loadA one-hot | loadB one-hot | jam ]
        """
        N = self.n_agents
        C = self.n_obs_types
        v = np.zeros((C,), dtype=np.uint8)

        # per-agent bits
        for i, pid in enumerate(self.possible_agents):
            a = self._last_actions.get(pid, None)
            base = 2 * i
            if a is None:
                continue
            if a == Actions.RoadA:
                v[base + 0] = 1
            elif a == Actions.RoadB:
                v[base + 1] = 1

        # load one-hots
        offset = 2 * N
        v[offset + int(self._loadA)] = 1
        offset += N + 1
        v[offset + int(self._loadB)] = 1
        offset += N + 1

        # jam flag
        v[offset] = 1 if self._jam else 0

        if self.flatten_observations:
            return v
        return v.reshape(1, 1, -1)

    # ────────────────────────────────────────────────────────────────────
    #                          State helpers
    # ────────────────────────────────────────────────────────────────────
    @property
    def state_space(self):
        C = self.n_obs_types
        if self.flatten_observations:
            return Box(low=0, high=1, shape=(C,), dtype=np.uint8)
        return Box(low=0, high=1, shape=(1, 1, C), dtype=np.uint8)

    def state(self):
        return self._obs()

    def num_cells(self) -> int:
        """Non-spatial global state → single cell."""
        return 1

    def channel_names(self) -> list[str]:
        """
        Human-friendly names for each observation channel index.

        Layout (see _obs):
          0..2*N-1              : per-agent action bits (A/B) in blocks of 2
          2*N .. 2*N+N          : loadA one-hot over [0..N]
          2*N+N+1 .. 2*N+2*N+1  : loadB one-hot over [0..N]
          last                  : jam flag
        """
        names: list[str] = ["" for _ in range(self.n_obs_types)]

        N = self.n_agents

        # Per-agent action bits
        for i in range(N):
            a_idx = 2 * i
            b_idx = 2 * i + 1
            if a_idx < self.n_obs_types:
                names[a_idx] = f"player_{i}_roadA"
            if b_idx < self.n_obs_types:
                names[b_idx] = f"player_{i}_roadB"

        # loadA one-hot bins
        offset = 2 * N
        for k in range(N + 1):
            idx = offset + k
            if idx >= self.n_obs_types:
                break
            names[idx] = f"loadA_{k}"

        # loadB one-hot bins
        offset += N + 1
        for k in range(N + 1):
            idx = offset + k
            if idx >= self.n_obs_types:
                break
            names[idx] = f"loadB_{k}"

        # jam flag
        offset += N + 1
        if offset < self.n_obs_types:
            names[offset] = "jam"

        # Fallback for any unnamed channels
        for i, name in enumerate(names):
            if not name:
                names[i] = f"channel_{i}"

        return names

    def action_names(self, action: int) -> str:
        """
        Human-friendly name for an action integer ID.
        """
        try:
            name = Actions(int(action)).name
        except ValueError:
            return f"action_{action}"
        return "".join(
            f"_{ch.lower()}" if ch.isupper() and i > 0 else ch.lower()
            for i, ch in enumerate(name)
        )

    # ────────────────────────────────────────────────────────────────────
    #               Lightweight state serialization
    # ────────────────────────────────────────────────────────────────────
    def get_state(self):
        """
        Returns a lightweight, fully-restorable snapshot.

        Tuple layout:
          (
            agents: tuple[str, ...],
            round_no: int,
            last_actions: tuple[int|None, ...]   # len = n_agents, in possible_agents order
            loadA: int,
            loadB: int,
            jam: bool,
            cum_rewards: tuple[float, ...]       # len = n_agents, in possible_agents order
            rng_state: object                    # np.random.RandomState.get_state()
          )
        """
        agents = tuple(self.agents) if getattr(self, "agents", None) is not None else tuple()
        last_actions = tuple(self._last_actions.get(a, None) for a in self.possible_agents)
        cum_rewards = tuple(float(self._cum_rewards.get(a, 0.0)) for a in self.possible_agents)
        rng_state = self.rng.get_state()  # fully reproducible
        return (
            agents,
            int(self._round),
            last_actions,
            int(self._loadA),
            int(self._loadB),
            bool(self._jam),
            cum_rewards,
            rng_state,
        )

    def set_state(self, state):
        """
        Restore a snapshot produced by get_state().
        """
        (
            agents,
            round_no,
            last_actions,
            loadA,
            loadB,
            jam,
            cum_rewards,
            rng_state,
        ) = state

        self.agents = list(agents)
        self._round = int(round_no)

        # restore last actions / cum rewards in possible_agents order
        if len(last_actions) != self.n_agents:
            raise ValueError(
                f"CongestionMatrix.set_state: last_actions len {len(last_actions)} != n_agents {self.n_agents}"
            )
        if len(cum_rewards) != self.n_agents:
            raise ValueError(
                f"CongestionMatrix.set_state: cum_rewards len {len(cum_rewards)} != n_agents {self.n_agents}"
            )

        self._last_actions = {
            a: (None if last_actions[i] is None else int(last_actions[i])) for i, a in enumerate(self.possible_agents)
        }
        self._cum_rewards = {a: float(cum_rewards[i]) for i, a in enumerate(self.possible_agents)}

        self._loadA = int(loadA)
        self._loadB = int(loadB)
        self._jam = bool(jam)

        # restore RNG
        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))