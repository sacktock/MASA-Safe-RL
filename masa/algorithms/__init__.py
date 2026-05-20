from masa.common import registry

registry.ALGO_REGISTRY.register(
    "QL",
    "masa.algorithms.tabular:QL",
)

registry.ALGO_REGISTRY.register(
    "QL_LAMBDA",
    "masa.algorithms.tabular:QL_LAMBDA",
)

registry.ALGO_REGISTRY.register(
    "RECOVERY_RL",
    "masa.algorithms.tabular:RECOVERY_RL",
)

registry.ALGO_REGISTRY.register(
    "RECREG",
    "masa.algorithms.tabular:RECREG",
)

registry.ALGO_REGISTRY.register(
    "SEM",
    "masa.algorithms.tabular:SEM",
)

registry.ALGO_REGISTRY.register(
    "LCRL",
    "masa.algorithms.tabular:LCRL",
)

registry.ALGO_REGISTRY.register(
    "PPO",
    "masa.algorithms.on_policy:PPO",
)

registry.ALGO_REGISTRY.register(
    "PPOLag",
    "masa.algorithms.on_policy:PPOLag",
)

registry.ALGO_REGISTRY.register(
    "A2C",
    "masa.algorithms.on_policy:A2C",
)

ALGORITHMS = registry.ALGO_REGISTRY.keys()

__all__ = ALGORITHMS