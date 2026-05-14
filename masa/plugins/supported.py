from masa.common.registry import (
    ALGO_REGISTRY,
    CONSTRAINT_REGISTRY,
    ENV_REGISTRY,
    MARL_CONSTRAINT_REGISTRY,
    MARL_ENV_REGISTRY,
)

# Register supported envs
ENV_REGISTRY.register("cont_cartpole", "masa.envs.continuous.cartpole:ContinuousCartPole")
ENV_REGISTRY.register("disc_cartpole", "masa.envs.discrete.cartpole:DiscreteCartPole")
ENV_REGISTRY.register("colour_grid_world", "masa.envs.tabular.colour_grid_world:ColourGridWorld")
ENV_REGISTRY.register("colour_bomb_grid_world", "masa.envs.tabular.colour_bomb_grid_world:ColourBombGridWorld")
ENV_REGISTRY.register("colour_bomb_grid_world_v2", "masa.envs.tabular.colour_bomb_grid_world_v2:ColourBombGridWorldV2")
ENV_REGISTRY.register("colour_bomb_grid_world_v3", "masa.envs.tabular.colour_bomb_grid_world_v3:ColourBombGridWorldV3")
ENV_REGISTRY.register("mini_pacman", "masa.envs.tabular.mini_pacman:MiniPacman")
ENV_REGISTRY.register("pacman", "masa.envs.tabular.pacman:Pacman")
ENV_REGISTRY.register("mini_pacman_with_coins", "masa.envs.discrete.mini_pacman_with_coins:MiniPacmanWithCoins")
ENV_REGISTRY.register("pacman_with_coins", "masa.envs.discrete.pacman_with_coins:PacmanWithCoins")
ENV_REGISTRY.register("island_navigation", "masa.envs.discrete.island_navigation:IslandNavigation")
ENV_REGISTRY.register("conveyor_belt", "masa.envs.discrete.conveyor_belt:ConveyorBelt")
ENV_REGISTRY.register("sokoban", "masa.envs.discrete.sokoban:Sokoban")
ENV_REGISTRY.register("bridge_crossing", "masa.envs.tabular.bridge_crossing:BridgeCrossing")
ENV_REGISTRY.register("bridge_crossing_v2", "masa.envs.tabular.bridge_crossing_v2:BridgeCrossingV2")
ENV_REGISTRY.register("media_streaming", "masa.envs.tabular.media_streaming:MediaStreaming")

# Register supported multi-agent envs
MARL_ENV_REGISTRY.register("bertrand_matrix", "masa.envs.multiagent.matrix.bertrand:BertrandMatrix")
MARL_ENV_REGISTRY.register("chicken_matrix", "masa.envs.multiagent.matrix.chicken:ChickenMatrix")
MARL_ENV_REGISTRY.register("congestion_matrix", "masa.envs.multiagent.matrix.congestion:CongestionMatrix")
MARL_ENV_REGISTRY.register("dpgg_matrix", "masa.envs.multiagent.matrix.dpgg:DPGGMatrix")
MARL_ENV_REGISTRY.register("inspection_matrix", "masa.envs.multiagent.matrix.inspection:InspectionMatrix")

# Register supported constraints
CONSTRAINT_REGISTRY.register("cmdp", "masa.common.constraints.cmdp:CumulativeCostEnv")
CONSTRAINT_REGISTRY.register("prob", "masa.common.constraints.prob:ProbabilisticSafetyEnv")
CONSTRAINT_REGISTRY.register("reach_avoid", "masa.common.constraints.reach_avoid:ReachAvoidEnv")
CONSTRAINT_REGISTRY.register("ltl_safety", "masa.common.constraints.ltl_safety:LTLSafetyEnv")
CONSTRAINT_REGISTRY.register("pctl", "masa.common.constraints.pctl:PCTLEnv")

# Register supported multi-agent constraints
MARL_CONSTRAINT_REGISTRY.register("cmg", "masa.common.constraints.multi_agent.cmg:ConstrainedMarkovGameEnv")

# Register supported algorithms
ALGO_REGISTRY.register("q_learning", "masa.algorithms.tabular:QL")
ALGO_REGISTRY.register("q_learning_lambda", "masa.algorithms.tabular:QL_Lambda")
ALGO_REGISTRY.register("sem", "masa.algorithms.tabular:SEM")
ALGO_REGISTRY.register("lcrl", "masa.algorithms.tabular:LCRL")
ALGO_REGISTRY.register("recreg", "masa.algorithms.tabular:RECREG")
ALGO_REGISTRY.register("recovery_rl", "masa.algorithms.tabular:RECOVERY_RL")
ALGO_REGISTRY.register("ppo", "masa.algorithms.ppo:PPO")
ALGO_REGISTRY.register("a2c", "masa.algorithms.a2c:A2C")
