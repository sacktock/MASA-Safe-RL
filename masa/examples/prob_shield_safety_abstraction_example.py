from masa.prob_shield.prob_shield_wrapper_disc import ProbShieldWrapperDisc
from masa.algorithms.ppo import PPO

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

    # Import labelling and cost functions for the BridgeCrossing
    from masa.envs.discrete.pacman_with_coins import safety_abstraction, abstr_label_fn, label_fn, cost_fn

    # We're going to use the PCTL constraint, which has key word args: (cost_fn CostFn: = DummyCostFn, alpha: float = 0.01) 
    constraint_kwargs = constraint_kwargs = dict(
        cost_fn=cost_fn,
        alpha=0.01,
    )

    # Intialize the environment (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> PCTLEnv -> ConstraintMonitor -> RewardMonitor
    env = make_env("pacman_with_coins", "pctl", 1000, label_fn=label_fn, **constraint_kwargs)

    # Now we're going to wrap our environment in ProbShieldWrapperDisc
    # The wrapper takes one arg: env
    #   and key word args: 
    #   theta: float = 1e-10,
    #   max_vi_steps: int = 1000,
    #   init_safety_bound: float = 0.5,
    #   granularity: int = 20,
    env = ProbShieldWrapperDisc(
        env, 
        label_fn=abstr_label_fn, # labelling function for the abstract discrete states
        cost_fn=cost_fn, # the usual cost function for the environment: 1.0 if ghost else 0.0
        safety_abstraction=safety_abstraction, # discrete safety absraction for the environment: maps observations to concerete discrete states
        theta = 1e-15, # early stopping condition for value iteration
        max_vi_steps= 10_000, # number of value iteration steps
        init_safety_bound = 0.01, # Safety constraint from the intial state
        granularity = 20, # Granulairty with which is discretize the successor state betas
    )

    # PPO is a on-policy algorithm that takes one arg: env
    #   and key word args:
    #   tensorboard_logdir: Optional[str] = None,
    #   seed: Optional[int] = None,
    #   monitor: bool = True,
    #   device: str = "auto",
    #   verbose: int = 0,
    #   env_fn: Optional[Callable[[], gym.Env]] = None,
    #   eval_env: Optional[gym.Env] = None, 
    #   learning_rate: Union[float, optax.Schedule] = 3e-4,
    #   n_steps: int = 2048,
    #   batch_size: int = 64,
    #   n_epochs: int = 10,
    #   gamma: float = 0.99,
    #   gae_lambda: float = 0.95,
    #   clip_range: Union[float, optax.Schedule] = 0.2,
    #   normalize_advantage: bool = True,
    #   ent_coef: float = 0.0,
    #   vf_coef: float = 1.0,
    #   max_grad_norm: float = 0.5,
    #   policy_class: type[BaseJaxPolicy] = PPOPolicy,
    #   policy_kwargs: Optional[dict[str, Any]] = None,

    # First lets initialize the eval_env
    eval_env = ProbShieldWrapperDisc(
        make_env("pacman_with_coins", "pctl", 1000, label_fn=label_fn, **constraint_kwargs), 
        label_fn=abstr_label_fn,
        cost_fn=cost_fn,
        safety_abstraction=safety_abstraction,
        theta = 1e-15,
        max_vi_steps= 10_000,
        init_safety_bound = 0.01,
        granularity = 20,
    )

    # Now let's initialize PPO
    # PPO will automatically one-hot encode any discrete observations and flatten any dict observations
    algo = PPO(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="auto", 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
        # Using the PPO specific defaults after this
    )

    # Now we begin training
    algo.train(
        num_frames=500_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=10_000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=10_000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size = 100, # sliding window size for metrics logging
    )

if __name__ == "__main__":
    main()