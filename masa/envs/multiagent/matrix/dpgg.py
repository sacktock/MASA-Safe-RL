from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

class Actions(IntEnum):
    Contribute = 0
    Withhold   = 1

class DPGGMatrix(ParallelEnv):
    """
    Dynamic Public Goods Game (Parallel PettingZoo), 2 players.

    (Markov + binary obs):
      - Pot is discretized to a fixed step size `pot_step`.
      - Internal pot state is an integer index: self._pot_idx
      - Observations include the exact pot index encoded as binary bits (uint8 0/1)

    Observation channels (binary):
      For each agent i in {0,1}:
        ch 2*i + 0 = 1 if agent_i last action was Contribute
        ch 2*i + 1 = 1 if agent_i last action was Withhold
      Plus:
        pot_bits bits encoding pot_idx (LSB-first), where
          pot_idx in [0, pot_levels-1]
          pot_value = pot_idx * pot_step
          pot_levels = floor(pot_cap / pot_step) + 1
    """

    metadata = {"name": "dpgg_matrix_v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_moves: int = 200,
        # DPGG parameters
        contribution_cost: float = 10.0,
        alpha: float = 1.5,
        payout_rate: float = 0.3,
        decay: float = 0.0,
        initial_pot: float = 0.0,
        pot_cap: float = 500.0,
        # NEW: discretization granularity (keep it > 0)
        pot_step: float = 1.0,
        flatten_observations: bool = True,
        render_mode=None,
        seed: int | None = None,
    ):
        assert num_agents == 2, "DPGGMatrix currently supports exactly 2 agents."
        self.n_agents = int(num_agents)
        self.possible_agents = [f"player_{i}" for i in range(self.n_agents)]

        # Parameters
        self.c = float(contribution_cost)
        self.alpha = float(alpha)
        self.payout_rate = float(np.clip(payout_rate, 0.0, 1.0))
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.initial_pot = float(max(0.0, initial_pot))
        self.pot_cap = float(max(0.0, pot_cap))

        # Pot discretization
        self.pot_step = float(pot_step)
        if not np.isfinite(self.pot_step) or self.pot_step <= 0.0:
            raise ValueError(f"pot_step must be > 0, got {pot_step}")

        # Number of discrete pot levels: 0, pot_step, 2*pot_step, ..., <= pot_cap
        self.pot_levels = int(np.floor(self.pot_cap / self.pot_step)) + 1
        self.pot_levels = max(2, self.pot_levels)

        # Bits needed to represent pot_idx
        self.pot_bits = int(np.ceil(np.log2(self.pot_levels)))
        self.pot_bits = max(1, self.pot_bits)

        # Repetition horizon
        self.max_moves = int(max_moves)

        # Observation config (binary channels, optionally flattened)
        # C = 2*num_agents + pot_bits
        self.n_obs_types = 2 * self.n_agents + self.pot_bits
        self.flatten_observations = bool(flatten_observations)

        # RNG (parity with your other envs)
        self.rng = np.random.RandomState(0 if seed is None else seed)

        # Runtime state
        self.agents: list[str] = []
        self._round = 0
        self._last_actions: dict[str, int | None] = {}
        self._pot_idx: int = 0  # discrete pot state
        self._cum_rewards: dict[str, float] = {}

        # rendering
        self.render_mode = render_mode
        self._renderer = None
        # self._renderer: DPGGRenderer | None = None
        # if self.render_mode is not None:
        #     self._renderer = DPGGRenderer(self.render_mode)

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
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self._round = 0
        self._last_actions = {a: None for a in self.agents}
        self._cum_rewards = {a: 0.0 for a in self.agents}

        # Quantize initial pot onto discrete grid
        self._pot_idx = self._pot_value_to_idx(self.initial_pot)

        obs = {a: self._obs() for a in self.agents}
        infos = {
            a: {
                "round": self._round,
                "last_actions": None,
                "pot": self._pot_value(),
                "pot_idx": self._pot_idx,
                "payout": 0.0,
                "contributors": 0,
            }
            for a in self.agents
        }
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        a0, a1 = self.possible_agents
        act0 = int(actions.get(a0, Actions.Contribute))
        act1 = int(actions.get(a1, Actions.Contribute))

        # Current discrete pot value
        pot_before = self._pot_value()
        payout_total = self.payout_rate * pot_before
        share = payout_total / self.n_agents

        cost0 = self.c if act0 == Actions.Contribute else 0.0
        cost1 = self.c if act1 == Actions.Contribute else 0.0

        r0 = share - cost0
        r1 = share - cost1
        rewards = {a0: float(r0), a1: float(r1)}

        # Update last actions & cumulative rewards
        self._last_actions[a0] = act0
        self._last_actions[a1] = act1
        self._cum_rewards[a0] += r0
        self._cum_rewards[a1] += r1

        # Pot update (float math), then quantize back to discrete grid
        pot_after = pot_before - payout_total
        pot_after = max(0.0, pot_after) * (1.0 - self.decay)

        contributors = (1 if act0 == Actions.Contribute else 0) + (1 if act1 == Actions.Contribute else 0)
        pot_after += self.alpha * self.c * contributors
        pot_after = float(np.clip(pot_after, 0.0, self.pot_cap))

        self._pot_idx = self._pot_value_to_idx(pot_after)

        # Time & termination
        self._round += 1
        env_trunc = self._round >= self.max_moves

        terminations = {a0: False, a1: False}
        truncations = {a0: env_trunc, a1: env_trunc}

        infos = {
            a0: {
                "round": self._round,
                "last_actions": (act0, act1),
                "pot_before": pot_before,
                "pot": self._pot_value(),
                "pot_idx": self._pot_idx,
                "payout": payout_total,
                "contributors": contributors,
            },
            a1: {
                "round": self._round,
                "last_actions": (act1, act0),
                "pot_before": pot_before,
                "pot": self._pot_value(),
                "pot_idx": self._pot_idx,
                "payout": payout_total,
                "contributors": contributors,
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
        params = {
            "c": self.c,
            "alpha": self.alpha,
            "payout_rate": self.payout_rate,
            "decay": self.decay,
            "pot": self._pot_value(),
        }

        frame = self._renderer.render(
            round_no=self._round,
            last_actions=last,
            cum_rewards=self._cum_rewards,
            params=params,
        )
        return frame

    def close(self):
        if self._renderer:
            self._renderer.close()

    # ───────────────────────────── Internals ─────────────────────────────
    def _pot_value(self) -> float:
        """Current pot value (float) implied by discrete pot index."""
        return float(self._pot_idx) * self.pot_step

    def _pot_value_to_idx(self, pot_value: float) -> int:
        """
        Quantize a pot value onto the discrete grid.

        We use deterministic "round half up" to avoid Python's bankers-rounding:
          idx = floor(x + 0.5)
        """
        x = float(np.clip(pot_value, 0.0, self.pot_cap)) / self.pot_step
        idx = int(np.floor(x + 0.5))
        if idx < 0:
            idx = 0
        if idx > self.pot_levels - 1:
            idx = self.pot_levels - 1
        return idx

    def _obs(self):
        """
        Binary channels, either (C,) or (1,1,C):
          ch0 = p0 Contribute, ch1 = p0 Withhold,
          ch2 = p1 Contribute, ch3 = p1 Withhold,
          ch(4..4+pot_bits-1) = pot_idx bits (LSB-first)
        """
        C = self.n_obs_types
        v = np.zeros((C,), dtype=np.uint8)

        def set_bits(pid: str, base: int):
            a = self._last_actions.get(pid, None)
            if a is None:
                return
            if a == Actions.Contribute:
                v[base + 0] = 1
            elif a == Actions.Withhold:
                v[base + 1] = 1

        set_bits("player_0", 0)
        set_bits("player_1", 2)

        # pot bits
        start = 2 * self.n_agents
        idx = int(self._pot_idx)
        for b in range(self.pot_bits):
            v[start + b] = (idx >> b) & 1

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
        return 1

    def channel_names(self) -> list[str]:
        names: list[str] = ["" for _ in range(self.n_obs_types)]

        # Per-agent action bits
        for i in range(self.n_agents):
            c_idx = 2 * i
            w_idx = 2 * i + 1
            names[c_idx] = f"player_{i}_contribute"
            names[w_idx] = f"player_{i}_withhold"

        # Pot bits
        start = 2 * self.n_agents
        for b in range(self.pot_bits):
            names[start + b] = f"pot_bit_{b}"  # LSB-first

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
        Tuple layout:
          (
            agents: tuple[str, ...],
            round_no: int,
            last_actions: tuple[int|None, int|None],  # (p0, p1)
            pot_idx: int,
            cum_rewards: tuple[float, float],         # (p0, p1)
            rng_state: object
          )
        """
        agents = tuple(self.agents) if getattr(self, "agents", None) is not None else tuple()
        a0, a1 = self.possible_agents
        last = (self._last_actions.get(a0, None), self._last_actions.get(a1, None))
        cum = (float(self._cum_rewards.get(a0, 0.0)), float(self._cum_rewards.get(a1, 0.0)))
        rng_state = self.rng.get_state()
        return (agents, int(self._round), last, int(self._pot_idx), cum, rng_state)

    def set_state(self, state):
        (agents, round_no, last, pot_idx, cum, rng_state) = state

        self.agents = list(agents)
        self._round = int(round_no)

        a0, a1 = self.possible_agents
        if len(last) != 2:
            raise ValueError(f"DPGGMatrix.set_state: last_actions must be len 2, got {len(last)}")
        if len(cum) != 2:
            raise ValueError(f"DPGGMatrix.set_state: cum_rewards must be len 2, got {len(cum)}")

        self._last_actions = {
            a0: (None if last[0] is None else int(last[0])),
            a1: (None if last[1] is None else int(last[1])),
        }

        self._pot_idx = int(np.clip(int(pot_idx), 0, self.pot_levels - 1))
        self._cum_rewards = {a0: float(cum[0]), a1: float(cum[1])}

        if rng_state is not None:
            self.rng.set_state(rng_state)

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        self.rng.set_state(rng_state)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState(int(seed))