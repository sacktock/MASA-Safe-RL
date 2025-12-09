from masa.common.wrappers import RewardShapingWrapper
from masa.algorithms.tabular import RECREG, QL, QL_Lambda

def main():
    # Import the masa make_env function
    from masa.common.utils import make_env
    '''
    def make_env(
        env_id: str, 
        constraint: str, 
        max_episode_steps: int, 
        *,
        label_fn: Optional[LabelFn] = None, 
        **constraint_kwargs
    ):
    '''

    # Import the labelling and cost functions for the ColourBombGridWorld
    from masa.envs.tabular.colour_bomb_grid_world import label_fn, cost_fn

    # Import example DFA 
    from masa.examples.colour_bomb_grid_world.property_2 import make_dfa

    # We're going to use the LTL safety constraint, which just has one key word args: (dfa: DFA = dummy_dfa) 
    constraint_kwargs = constraint_kwargs = dict(
        dfa=make_dfa()
    )

    # Intialize the environment (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> LTLSafetyEnv -> ConstraintMonitor -> RewardMonitor
    env = make_env("colour_bomb_grid_world", "ltl_dfa", 400, label_fn=label_fn, **constraint_kwargs)

    # Now we're going to wrap our environment in RewardShapingWrapper
    # The wrapper takes one arg: env
    #   and two key word args: (gamma: float = 0.99, impl: str = "none")
    env = RewardShapingWrapper(
        env, 
        gamma=0.99,
        impl="vi" # which implementation to use ("none", "vi", "cycle")
    )

    # The RewardShapingWrapper implements two different types of potential-based reward shaping
    #   1. "vi": performs value iteration on the DFA computing the relative "cost" of being in each DFA state.
    #   2. "cycle": scores automaton states based on their distance to the DFA state that is furthest from the (unsafe) accepting state.

    # Were going to use (model-free) RECREG to solve the problem
    # RECREG is an off-policy tabular algorithm based on Q learning that takes one arg: env
    #   and key word args:
    #   tensorboard_logdir: Optional[str] = None,
    #   seed: Optional[int] = None,
    #   monitor: bool = True,
    #   device: str = "auto",
    #   verbose: int = 0,
    #   env_fn: Optional[Callable[[], gym.Env]] = None,
    #   eval_env: Optional[gym.Env] = None, 
    #   task_alpha: float = 0.1,
    #   task_gamma: float = 0.9,
    #   safe_alpha: float = 0.1,
    #   safe_gamma: float = 0.9,
    #   model_impl: str = 'model-based',
    #   model_checking: str = 'sample',
    #   samples: int = 512,
    #   horizon: int = 10,
    #   step_wise_prob: float = 0.99,
    #   model_prior: str = 'identity',
    #   exploration: str = 'boltzmann',
    #   boltzmann_temp: float = 0.05,
    #   initial_epsilon: float = 1.0,
    #   final_epsilon: float = 0.1,
    #   epsilon_decay: str = 'linear',
    #   epsilon_decay_frames: int = 10000, 

    # First lets initialize the eval_env
    eval_env = make_env("colour_bomb_grid_world", "ltl_dfa", 400, label_fn=label_fn, **constraint_kwargs)

    # Now let's initialize RECREG
    algo = RECREG(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="cpu", # keep everything on the cpu 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
        model_impl = "model-free", # using model-free RECREG as it is much quicker
        horizon = 5, # recoverability horizon for RECREG
        step_wise_prob = 1e-4, # step-wise epsilon_t values
        # Using the RECREG specific defaults after this
    )

    # Now we begin training
    algo.train(
        num_frames=100_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=2_000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=2_000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size = 100, # sliding window size for metrics logging
    )

if __name__ == "__main__":
    main()