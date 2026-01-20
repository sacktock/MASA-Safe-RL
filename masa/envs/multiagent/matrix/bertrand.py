from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

class Actions(IntEnum):
    High = 0    # Collusive/high price
    Low  = 1    # Undercut/low price

class BertrandMatrix(ParallelEnv):
    """
    Repeated 2-player Bertrand (Parallel PettingZoo)

    Stage game (Row vs Column) with payoffs:
        - High   vs Low : (S, T)
        - Low    vs High: (T, S)
        - High   vs High: (R, R)
        - Low    vs Low : (P, P)  # price war

    Defaults: T=8, R=5, S=0, P=0 (all floats).

    Observations (global, binary channels), shape is either (C,) if flattened or (1,1,C):
      For each agent i in {0,1}:
        - ch 2*i + 0 = 1 if agent_i's last action was High, else 0
        - ch 2*i + 1 = 1 if agent_i's last action was Low, else 0
      Plus:
        - ch 2*num_agents = 1 if last outcome was a price war (both Low), else 0

      On the first round (after reset), all channels are 0.
    """

    metadata = {"name": "bertrand_matrix_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_moves: int = 200,
        T: float = 8.0,
        R: float = 5.0,
        S: float = 0.0,
        P: float = 0.0,
        flatten_observations: bool = True,
        render_mode=None,
        seed: int | None = None,
    ):
        assert num_agents == 2, "BertrandMatrix currently supports exactly 2 agents."
        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]

        # Payoffs
        self.T, self.R, self.S, self.P = float(T), float(R), float(S), float(P)

        # Repetition horizon
        self.max_moves = int(max_moves)

        # Observation config (1x1xC binary channels, optionally flattened)
        # C = 2*num_agents + 1 (two action-bits per agent + 1 war bit)
        self.n_obs_types = 2 * self.n_agents + 1
        self.flatten_observations = bool(flatten_observations)

        # RNG (not strictly needed; kept for parity)
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # Runtime state
        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}  # Actions.{High,Low} or None
        self._last_war: bool = False

        # rendering-relevant runtime state
        self.render_mode = render_mode
        self._renderer = None
        # self._renderer: BertrandRenderer | None = None
        # if self.render_mode is not None:
        #     self._renderer = BertrandRenderer(self.render_mode)
        self._cum_rewards: dict[str, float] = {}

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
        # 0 = High, 1 = Low
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {a: None for a in self.agents}
        self._last_war = False
        self._cum_rewards = {a: 0.0 for a in self.agents}

        obs = {a: self._obs() for a in self.agents}
        infos = {a: {"round": self._round, "last_actions": None, "war": False} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Only two agents supported
        a0, a1 = self.possible_agents
        act0 = int(actions.get(a0, Actions.High))
        act1 = int(actions.get(a1, Actions.High))

        # Stage payoffs (Bertrand):
        # (High,High)->(R,R); (High,Low)->(S,T); (Low,High)->(T,S); (Low,Low)->(P,P, war)
        if act0 == Actions.High and act1 == Actions.High:
            r0, r1 = self.R, self.R
            self._last_war = False
            outcome = "High-High"
        elif act0 == Actions.High and act1 == Actions.Low:
            r0, r1 = self.S, self.T
            self._last_war = False
            outcome = "High-Low"
        elif act0 == Actions.Low and act1 == Actions.High:
            r0, r1 = self.T, self.S
            self._last_war = False
            outcome = "Low-High"
        else:  # Low-Low
            r0, r1 = self.P, self.P
            self._last_war = True
            outcome = "Low-Low"

        rewards = {a0: float(r0), a1: float(r1)}

        # Update last actions for observation
        self._last_actions[a0] = act0
        self._last_actions[a1] = act1
        self._cum_rewards[a0] += r0
        self._cum_rewards[a1] += r1

        # Time & termination
        self._round += 1
        env_trunc = self._round >= self.max_moves

        terminations = {a0: False, a1: False}
        truncations = {a0: env_trunc, a1: env_trunc}
        infos = {
            a0: {"round": self._round, "last_actions": (act0, act1), "outcome": outcome, "war": self._last_war},
            a1: {"round": self._round, "last_actions": (act1, act0), "outcome": outcome, "war": self._last_war},
        }

        obs = {a: self._obs() for a in self.agents}
        if env_trunc:
            for a in self.agents:
                truncations[a] = True
            self.agents = []

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return
        if not self._renderer:
            raise ValueError("Renderer missing; create env with render_mode='human' or 'rgb_array'.")

        pay = {"T": self.T, "R": self.R, "S": self.S, "P": self.P}
        last = (self._last_actions.get("player_0"), self._last_actions.get("player_1"))

        frame = self._renderer.render(
            round_no=self._round,
            last_actions=last,
            last_war=self._last_war,
            cum_rewards=self._cum_rewards,
            payoffs=pay,
        )
        return frame

    def close(self):
        if self._renderer:
            self._renderer.close()

    # ───────────────────────────── Internals ─────────────────────────────
    def _obs(self):
        """
        Binary channels, either (C,) or (1,1,C):
          ch0 = p0 High, ch1 = p0 Low,
          ch2 = p1 High, ch3 = p1 Low,
          ch4 = war flag (both Low)
        """
        C = self.n_obs_types
        v = np.zeros((C,), dtype=np.uint8)

        def set_bits(pid: str, base: int):
            a = self._last_actions.get(pid, None)
            if a is None:
                return
            if a == Actions.High:
                v[base + 0] = 1
            elif a == Actions.Low:
                v[base + 1] = 1

        set_bits("player_0", 0)
        set_bits("player_1", 2)
        v[2 * self.n_agents] = 1 if self._last_war else 0

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
        """Number of spatial cells in the layered observation grid.

        BertrandMatrix is non-spatial: the entire game state is a single global cell.
        This returns 1 regardless of whether observations are flattened or shaped (1, 1, C).
        """
        return 1

    def channel_names(self) -> list[str]:
        """
        Human-friendly names for each observation channel index.

        Index i of the returned list corresponds to channel i in the
        observation vector (or the last axis of the (1, 1, C) tensor).
        """
        names: list[str] = ["" for _ in range(self.n_obs_types)]

        # Per-agent action bits: (High, Low)
        for i in range(self.n_agents):
            hi_idx = 2 * i
            lo_idx = 2 * i + 1
            if hi_idx < self.n_obs_types:
                names[hi_idx] = f"player_{i}_high"
            if lo_idx < self.n_obs_types:
                names[lo_idx] = f"player_{i}_low"

        # Price war flag
        war_idx = 2 * self.n_agents
        if war_idx < self.n_obs_types:
            names[war_idx] = "price_war"

        # Fallback for any unnamed channels (future-proofing)
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
        Fully restorable snapshot (no deep clone).

        Tuple layout:
          (
            agents: tuple[str, ...],
            round_no: int,
            last_actions: tuple[int|None, int|None],  # (player_0, player_1)
            last_war: bool,
            cum_rewards: tuple[float, float],         # (player_0, player_1)
            rng_state: object                         # np.random.RandomState.get_state()
          )
        """
        agents = tuple(self.agents) if getattr(self, "agents", None) is not None else tuple()

        p0, p1 = self.possible_agents
        last_actions = (self._last_actions.get(p0, None), self._last_actions.get(p1, None))
        cum_rewards = (float(self._cum_rewards.get(p0, 0.0)), float(self._cum_rewards.get(p1, 0.0)))

        rng_state = self.rng.get_state()
        return (
            agents,
            int(self._round),
            last_actions,
            bool(self._last_war),
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
            last_war,
            cum_rewards,
            rng_state,
        ) = state

        self.agents = list(agents)
        self._round = int(round_no)

        p0, p1 = self.possible_agents

        if len(last_actions) != 2:
            raise ValueError(f"BertrandMatrix.set_state: last_actions must be len 2, got {len(last_actions)}")
        if len(cum_rewards) != 2:
            raise ValueError(f"BertrandMatrix.set_state: cum_rewards must be len 2, got {len(cum_rewards)}")

        self._last_actions = {
            p0: (None if last_actions[0] is None else int(last_actions[0])),
            p1: (None if last_actions[1] is None else int(last_actions[1])),
        }
        self._last_war = bool(last_war)

        self._cum_rewards = {
            p0: float(cum_rewards[0]),
            p1: float(cum_rewards[1]),
        }

        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))