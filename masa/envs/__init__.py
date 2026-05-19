from masa.common import registry

registry.ENV_REGISTRY.register(
    "ContinuousCartpole", "masa.envs.continuous.cartpole:ContinuousCartpole"
)

registry.ENV_REGISTRY.register(
    "DiscreteCartpole", "masa.envs.discrete.cartpole:DiscreteCartpole"
)

registry.ENV_REGISTRY.register(
    "ConveyorBelt", "masa.envs.discrete.conveyor_belt:ConveyorBelt"
)

registry.ENV_REGISTRY.register(
    "IslandNavigation", "masa.envs.discrete.island_navigation:IslandNavigation"
)

registry.ENV_REGISTRY.register(
    "MiniPacmanWithCoins", "masa.envs.discrete.mini_pacman_with_coins:MiniPacmanWithCoins"
)

registry.ENV_REGISTRY.register(
    "PacmanWithCoins", "masa.envs.discrete.pacman_with_coins:PacmanWithCoins"
)

registry.ENV_REGISTRY.register(
    "Sokoban", "masa.envs.discrete.sokoban:Sokoban"
)

registry.MARL_ENV_REGISTRY.register(
    "BertrandMatrix",
    "masa.envs.multiagent.matrix.bertrand:BertrandMatrix",
)

registry.MARL_ENV_REGISTRY.register(
    "ChickenMatrix",
    "masa.envs.multiagent.matrix.chicken:ChickenMatrix",
)

registry.MARL_ENV_REGISTRY.register(
    "CongestionMatrix",
    "masa.envs.multiagent.matrix.congestion:CongestionMatrix",
)

registry.MARL_ENV_REGISTRY.register(
    "DPGGMatrix",
    "masa.envs.multiagent.matrix.dpgg:DPGGMatrix",
)

registry.MARL_ENV_REGISTRY.register(
    "InspectionMatrix",
    "masa.envs.multiagent.matrix.inspection:InspectionMatrix",
)

registry.ENV_REGISTRY.register(
    "BridgeCrossing",
    "masa.envs.tabular.bridge_crossing:BridgeCrossing",
)

registry.ENV_REGISTRY.register(
    "BridgeCrossingV2",
    "masa.envs.tabular.bridge_crossing_v2:BridgeCrossingV2",
)

registry.ENV_REGISTRY.register(
    "ColourBombGridWorld",
    "masa.envs.tabular.colour_bomb_grid_world:ColourBombGridWorld",
)

registry.ENV_REGISTRY.register(
    "ColourBombGridWorldV2",
    "masa.envs.tabular.colour_bomb_grid_world_v2:ColourBombGridWorldV2",
)

registry.ENV_REGISTRY.register(
    "ColourBombGridWorldV3",
    "masa.envs.tabular.colour_bomb_grid_world_v3:ColourBombGridWorldV3",
)

registry.ENV_REGISTRY.register(
    "ColourGridWorld",
    "masa.envs.tabular.colour_grid_world:ColourGridWorld",
)

registry.ENV_REGISTRY.register(
    "MediaStreaming",
    "masa.envs.tabular.media_streaming:MediaStreaming",
)

registry.ENV_REGISTRY.register(
    "MiniPacman",
    "masa.envs.tabular.mini_pacman:MiniPacman",
)

registry.ENV_REGISTRY.register(
    "Pacman",
    "masa.envs.tabular.pacman:Pacman",
)

ENVIRONMENTS = registry.ENV_REGISTRY.keys()
MARL_ENVIRONMENTS = registry.MARL_ENV_REGISTRY.keys()

__all__ = [
    *ENVIRONMENTS,
    *MARL_ENVIRONMENTS,
]