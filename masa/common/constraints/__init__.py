from masa.common import registry

registry.CONSTRAINT_REGISTRY.register(
    "CMDP",
    "masa.common.constraints.cmdp:CumulativeCostEnv",
)

registry.CONSTRAINT_REGISTRY.register(
    "LTL_SAFETY",
    "masa.common.constraints.ltl_safety:LTLSafetyEnv",
)

registry.CONSTRAINT_REGISTRY.register(
    "PCTL",
    "masa.common.constraints.pctl:PCTLEnv",
)

registry.CONSTRAINT_REGISTRY.register(
    "PROB",
    "masa.common.constraints.prob:ProbabilisticSafetyEnv",
)

registry.CONSTRAINT_REGISTRY.register(
    "REACH_AVOID",
    "masa.common.constraints.reach_avoid:ReachAvoidEnv",
)

registry.MARL_CONSTRAINT_REGISTRY.register(
    "CMG",
    "masa.common.constraints.multi_agent.cmg:ConstrainedMarkovGameEnv"
)

CONSTRAINTS = registry.CONSTRAINT_REGISTRY.keys()
MARL_CONSTRAINTS = registry.MARL_CONSTRAINT_REGISTRY.keys()

__all__ = [
    *CONSTRAINTS,
    *MARL_CONSTRAINTS,
]