from __future__ import annotations

import functools

import numpy as np
from enum import IntEnum
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

class InspectorActions(IntEnum):
    NotInspect = 0
    Inspect    = 1

class InspecteeActions(IntEnum):
    Comply  = 0
    Violate = 1

class InspectionMatrix(ParallelEnv):
    """
    Repeated 2-player Inspection game (Parallel PettingZoo)

    Players:
      - Row (player_0): Inspector — actions {NotInspect=0, Inspect=1}
      - Col (player_1): Inspectee — actions {Comply=0, Violate=1}

    Stage payoffs (Inspector, Inspectee):
      (NotInspect, Comply)  -> (0, 0)
      (NotInspect, Violate) -> (-h, b)        # undetected violation
      (Inspect, Comply)     -> (-c, 0)
      (Inspect, Violate)    -> (v - c, -f)

    Defaults: b=5, f=10, c=2, h=4, v=3  (all floats)

    Observations (global, binary channels), shape is either (C,) if flattened or (1,1,C):
      For each agent i in {0,1}:
        - ch 2*i + 0 = 1 if agent_i's last action was action 0, else 0
        - ch 2*i + 1 = 1 if agent_i's last action was action 1, else 0
      Plus:
        - ch 2*num_agents = 1 if last outcome was an undetected violation ((NotInspect, Violate)), else 0

      On the first round (after reset), all channels are 0.
    """

    metadata = {"name": "inspection_matrix_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_moves: int = 200,
        b: float = 5.0,
        f: float = 10.0,
        c: float = 2.0,
        h: float = 4.0,
        v: float = 3.0,
        flatten_observations: bool = True,
        render_mode=None,
        seed: int | None = None,
    ):
        assert num_agents == 2, "InspectionMatrix currently supports exactly 2 agents."
        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]

        # Payoff parameters
        self.b, self.f, self.c, self.h, self.v = float(b), float(f), float(c), float(h), float(v)

        # Repetition horizon
        self.max_moves = int(max_moves)

        # Observation config (1x1xC binary channels, optionally flattened)
        # C = 2*num_agents + 1 (two action-bits per agent + 1 undetected-violation bit)
        self.n_obs_types = 2 * self.n_agents + 1
        self.flatten_observations = bool(flatten_observations)

        # RNG (parity with Chicken)
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # Runtime state
        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}  # {player: 0/1 or None}
        self._last_undetected: bool = False

        # rendering-relevant runtime state
        self.render_mode = render_mode
        self._renderer = None
        # self._renderer: InspectionRenderer | None = None
        # if self.render_mode is not None:
        #     self._renderer = InspectionRenderer(self.render_mode)
        # self._cum_rewards: dict[str, float] = {}

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
        # 0/1 for both players
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {a: None for a in self.agents}
        self._last_undetected = False
        self._cum_rewards = {a: 0.0 for a in self.agents}

        obs = {a: self._obs() for a in self.agents}
        infos = {a: {"round": self._round, "last_actions": None, "undetected": False} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Only two agents supported
        p0, p1 = self.possible_agents
        # Defaults mirror Chicken's style (choose action 0 if missing)
        act0 = int(actions.get(p0, InspectorActions.NotInspect))
        act1 = int(actions.get(p1, InspecteeActions.Comply))

        # Stage payoffs & undetected flag
        if act0 == InspectorActions.NotInspect and act1 == InspecteeActions.Comply:  # (N, C)
            r0, r1 = 0.0, 0.0
            undetected = False
            outcome = "NotInspect-Comply"
        elif act0 == InspectorActions.NotInspect and act1 == InspecteeActions.Violate:  # (N, V)
            r0, r1 = -self.h, self.b
            undetected = True
            outcome = "NotInspect-Violate"
        elif act0 == InspectorActions.Inspect and act1 == InspecteeActions.Comply:  # (I, C)
            r0, r1 = -self.c, 0.0
            undetected = False
            outcome = "Inspect-Comply"
        else:  # (I, V)
            r0, r1 = self.v - self.c, -self.f
            undetected = False
            outcome = "Inspect-Violate"

        rewards = {p0: float(r0), p1: float(r1)}

        # Update last actions / rewards
        self._last_actions[p0] = int(act0)
        self._last_actions[p1] = int(act1)
        self._last_undetected = bool(undetected)
        self._cum_rewards[p0] += r0
        self._cum_rewards[p1] += r1

        # Time & termination
        self._round += 1
        env_trunc = self._round >= self.max_moves

        terminations = {p0: False, p1: False}
        truncations = {p0: env_trunc, p1: env_trunc}
        infos = {
            p0: {
                "round": self._round,
                "last_actions": (act0, act1),
                "outcome": outcome,
                "undetected": self._last_undetected,
            },
            p1: {
                "round": self._round,
                "last_actions": (act1, act0),
                "outcome": outcome,
                "undetected": self._last_undetected,
            },
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

        last = (self._last_actions.get("player_0"), self._last_actions.get("player_1"))
        params = {"b": self.b, "f": self.f, "c": self.c, "h": self.h, "v": self.v}

        frame = self._renderer.render(
            round_no=self._round,
            last_actions=last,
            last_undetected=self._last_undetected,
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
        Binary channels, either (C,) or (1,1,C):
          ch0 = p0 action==0 (NotInspect), ch1 = p0 action==1 (Inspect),
          ch2 = p1 action==0 (Comply),     ch3 = p1 action==1 (Violate),
          ch4 = undetected flag ((NotInspect, Violate))
        """
        C = self.n_obs_types
        v = np.zeros((C,), dtype=np.uint8)

        def set_bits(pid: str, base: int):
            a = self._last_actions.get(pid, None)
            if a is None:
                return
            if a == 0:
                v[base + 0] = 1
            elif a == 1:
                v[base + 1] = 1

        set_bits("player_0", 0)
        set_bits("player_1", 2)
        v[2 * self.n_agents] = 1 if self._last_undetected else 0

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

        InspectionMatrix is non-spatial: the entire game state is a single global cell.
        This returns 1 regardless of whether observations are flattened or shaped (1, 1, C).
        """
        return 1

    def channel_names(self) -> list[str]:
        """
        Human-friendly names for each observation channel index.

        By construction in _obs:
          ch0 = p0 action==0 (NotInspect)
          ch1 = p0 action==1 (Inspect)
          ch2 = p1 action==0 (Comply)
          ch3 = p1 action==1 (Violate)
          ch4 = undetected flag ((NotInspect, Violate))
        """
        names: list[str] = ["" for _ in range(self.n_obs_types)]

        # Player 0 = Inspector
        if self.n_obs_types > 0:
            names[0] = "inspector_notinspect"
        if self.n_obs_types > 1:
            names[1] = "inspector_inspect"

        # Player 1 = Inspectee
        if self.n_obs_types > 2:
            names[2] = "inspectee_comply"
        if self.n_obs_types > 3:
            names[3] = "inspectee_violate"

        # Undetected violation flag
        undetected_idx = 2 * self.n_agents
        if undetected_idx < self.n_obs_types:
            names[undetected_idx] = "undetected_violation"

        # Fallback for any unnamed channels (future-proofing)
        for i, name in enumerate(names):
            if not name:
                names[i] = f"channel_{i}"

        return names

    def action_names(self, action: int) -> str:
        """
        Human-friendly name for an action integer ID.
        """
        action = int(action)
        if action == int(InspectorActions.NotInspect):
            return "not_inspect_or_comply"
        if action == int(InspectorActions.Inspect):
            return "inspect_or_violate"
        return f"action_{action}"

    # ────────────────────────────────────────────────────────────────────
    #               Lightweight state serialization (no deep clone)
    # ────────────────────────────────────────────────────────────────────
    def get_state(self):
        """
        Tuple layout:
          (
            agents: tuple[str, ...],
            round_no: int,
            last_actions: tuple[int|None, int|None],  # (p0, p1)
            last_undetected: bool,
            cum_rewards: tuple[float, float],         # (p0, p1)
            rng_state: object
          )
        """
        agents = tuple(self.agents) if getattr(self, "agents", None) is not None else tuple()
        p0, p1 = self.possible_agents
        last = (self._last_actions.get(p0, None), self._last_actions.get(p1, None))
        cum = (float(self._cum_rewards.get(p0, 0.0)), float(self._cum_rewards.get(p1, 0.0)))
        rng_state = self.rng.get_state()
        return (agents, int(self._round), last, bool(self._last_undetected), cum, rng_state)

    def set_state(self, state):
        (agents, round_no, last, last_undetected, cum, rng_state) = state

        self.agents = list(agents)
        self._round = int(round_no)

        p0, p1 = self.possible_agents
        if len(last) != 2:
            raise ValueError(f"InspectionMatrix.set_state: last_actions must be len 2, got {len(last)}")
        if len(cum) != 2:
            raise ValueError(f"InspectionMatrix.set_state: cum_rewards must be len 2, got {len(cum)}")

        self._last_actions = {
            p0: (None if last[0] is None else int(last[0])),
            p1: (None if last[1] is None else int(last[1])),
        }
        self._last_undetected = bool(last_undetected)
        self._cum_rewards = {p0: float(cum[0]), p1: float(cum[1])}

        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))