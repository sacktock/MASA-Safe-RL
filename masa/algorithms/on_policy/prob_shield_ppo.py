from __future__ import annotations
import gymnasium as gym
from masa.prob_shield.parameterized_ppo import ParameterizedPPO
from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperCont
from masa.prob_shield.parameterized_policy import ParameterizedPPOPolicy
from masa.common.base_class import BaseJaxPolicy
from masa.common.policies import PPOLagPolicy

class ProbShieldPPO(ParameterizedPPO):

    def __init__(
        self,
        env: gym.Env,
        *args,
        env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None, 
        policy_class: type[BaseJaxPolicy] = ParameterizedPPOPolicy,
        init_safety_bound: float = 0.01,
        theta: float = 1e-15,
        max_vi_steps: int = 10_000,
        **kwargs,
    ):

        self.init_safety_bound = init_safety_bound
        self.theta = theta
        self.max_vi_steps = max_vi_steps

        env = ProbShieldWrapperCont(
            env,
            init_safety_bound=self.init_safety_bound,
            theta=self.theta,
            max_vi_steps=self.max_vi_steps,
        )

        if eval_env is None and env_fn is not None:
            eval_env = env_fn()

        eval_env = ProbShieldWrapperCont(
            eval_env,
            init_safety_bound=self.init_safety_bound,
            theta=self.theta,
            max_vi_steps=self.max_vi_steps,
        )

        super().__init__(env, *args, env_fn=None, eval_env=eval_env, policy_class=policy_class, **kwargs)

