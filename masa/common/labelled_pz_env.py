from __future__ import annotations
from typing import Any, Dict
from pettingzoo.utils import BaseParallelEnv
from masa.common.label_fn import LabelFn


class LabelledParallelEnv(BaseParallelEnv):
    """PettingZoo parallel API wrapper that attaches the per-agent labelling function."""
    metadata = {"name": "labelled_parallel_env"}


    def __init__(self, env: BaseParallelEnv, label_fn: Dict[str, LabelFn] | LabelFn):
        self.env = env
        self.agents = env.possible_agents
        self.label_fn = label_fn

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for a in obs:
            lf = self.label_fn[a] if isinstance(self.label_fn, dict) else self.label_fn
            info.setdefault(a, {})["labels"] = set(lf(obs[a]))
        return obs, info

    def step(self, actions):
        obs, rewards, term, trunc, infos = self.env.step(actions)
        # compute labels and cost per agent, update shared constraint using a simple merge
        for a in obs:
            lf = self.label_fn[a] if isinstance(self.label_fn, dict) else self.label_fn
            labels = set(lf(obs[a]))
            infos.setdefault(a, {})["labels"] = labels

        return obs, rewards, term, trunc, infos

    @property
    def possible_agents(self):
        return self.env.possible_agents