from __future__ import annotations
from typing import Any, Dict
from pettingzoo import ParallelEnv
from masa.common.label_fn import LabelFn


class LabelledParallelEnv(ParallelEnv):
    """PettingZoo parallel API wrapper that attaches the per-agent labelling function."""
    metadata = {"name": "labelled_parallel_env"}


    def __init__(self, env: ParallelEnv, label_fn: Dict[str, LabelFn] | LabelFn):
        self.env = env
        self.metadata = getattr(env, "metadata", self.metadata)
        self.agents = list(getattr(env, "agents", env.possible_agents))
        self.label_fn = label_fn
        self.cost_fn = getattr(env, "cost_fn", None)

    def __getattr__(self, name: str):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        for a in obs:
            lf = self.label_fn[a] if isinstance(self.label_fn, dict) else self.label_fn
            agent_info = info.setdefault(a, {})
            if agent_info is None:
                agent_info = {}
                info[a] = agent_info
            agent_info["labels"] = set(lf(obs[a]))
        return obs, info

    def step(self, actions):
        obs, rewards, term, trunc, infos = self.env.step(actions)
        self.agents = list(getattr(self.env, "agents", self.possible_agents))
        for a in obs:
            lf = self.label_fn[a] if isinstance(self.label_fn, dict) else self.label_fn
            labels = set(lf(obs[a]))
            agent_info = infos.setdefault(a, {})
            if agent_info is None:
                agent_info = {}
                infos[a] = agent_info
            agent_info["labels"] = labels

        return obs, rewards, term, trunc, infos

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def state(self):
        return self.env.state()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
