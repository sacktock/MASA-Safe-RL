from masa.common.utils import ENV_REGISTRY, CONSTRAINT_REGISTRY, ALGO_REGISTRY

from gymnasium.envs.classic_control import CartPoleEnv
from masa.envs.colour_grid_world import ColourGridWorld

from masa.common.constraints import CumulativeCostEnv
from masa.common.constraints import ProbabilisticSafetyEnv
from masa.common.constraints import LTLSafetyEnv
from masa.common.constraints import ReachAvoidEnv
from masa.common.constraints import PCTLEnv

from masa.algorithms.tabular import QL


# Register supported envs
ENV_REGISTRY.register("cartpole", lambda **kw: CartPoleEnv(**kw))
ENV_REGISTRY.register("colour_grid_world", lambda **kw: ColourGridWorld(**kw))

# Register supported constraints
CONSTRAINT_REGISTRY.register("cmdp", lambda env, **kw: CumulativeCostEnv(env, kw['cost_fn'], kw['cost_budget']))
CONSTRAINT_REGISTRY.register("prob", lambda env, **kw: ProbabilisticSafetyEnv(env, kw['cost_fn'], kw['alpha']))
CONSTRAINT_REGISTRY.register("reach_avoid", lambda env, **kw: ReachAvoidEnv(env, kw['avoid_label'], kw['reach_label']))
CONSTRAINT_REGISTRY.register("ltl_dfa", lambda env, **kw: LTLSafetyEnv(env, kw['dfa']))
CONSTRAINT_REGISTRY.register("pctl", lambda env, **kw: PCTLEnv(env, kw['cost_fn'], kw['alpha']))

# Register supported algorithms
ALGO_REGISTRY.register("q_learning", QL)
