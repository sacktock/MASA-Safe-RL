from masa.common.utils import ENV_REGISTRY, CONSTRAINT_REGISTRY, ALGO_REGISTRY

# Register supported envs
ENV_REGISTRY.register("cartpole", "gymnasium.envs.classic_control:CartPoleEnv")
ENV_REGISTRY.register("colour_grid_world", "masa.envs.tabular.colour_grid_world:ColourGridWorld")
ENV_REGISTRY.register("colour_bomb_grid_world", "masa.envs.tabular.colour_bomb_grid_world:ColourBombGridWorld")
ENV_REGISTRY.register("colour_bomb_grid_world_v2", "masa.envs.tabular.colour_bomb_grid_world_v2:ColourBombGridWorldV2")
ENV_REGISTRY.register("colour_bomb_grid_world_v3", "masa.envs.tabular.colour_bomb_grid_world_v3:ColourBombGridWorldV3")
ENV_REGISTRY.register("mini_pacman", "masa.envs.tabular.mini_pacman:MiniPacman")
ENV_REGISTRY.register("pacman", "masa.envs.tabular.pacman:Pacman")
ENV_REGISTRY.register("bridge_crossing", "masa.envs.tabular.bridge_crossing:BridgeCrossing")
ENV_REGISTRY.register("bridge_crossing_v2", " masa.envs.tabular.bridge_crossing_v2:BridgeCrossingV2")
ENV_REGISTRY.register("media_streaming", "masa.envs.tabular.media_streaming:MediaStreaming")

# Register supported constraints
CONSTRAINT_REGISTRY.register("cmdp", "masa.common.constraints:CumulativeCostEnv")
CONSTRAINT_REGISTRY.register("prob", "masa.common.constraints:ProbabilisticSafetyEnv")
CONSTRAINT_REGISTRY.register("reach_avoid", "masa.common.constraints:ReachAvoidEnv")
CONSTRAINT_REGISTRY.register("ltl_dfa", "masa.common.constraints:LTLSafetyEnv")
CONSTRAINT_REGISTRY.register("pctl", "masa.common.constraints:PCTLEnv")

# Register supported algorithms
ALGO_REGISTRY.register("q_learning", "masa.algorithms.tabular:QL")
ALGO_REGISTRY.register("q_learning_lambda", "masa.algorithms.tabular:QL_Lambda")
ALGO_REGISTRY.register("sem", "masa.algorithms.tabular:SEM")
ALGO_REGISTRY.register("lcrl", "masa.algorithms.tabular:LCRL")
ALGO_REGISTRY.register("recreg", "masa.algorithms.tabular:RECREG")