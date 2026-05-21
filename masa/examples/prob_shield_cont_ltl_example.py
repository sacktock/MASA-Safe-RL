from masa.prob_shield.prob_shield_wrapper_v1 import ProbShieldWrapperCont
from masa.prob_shield.parameterized_ppo import ParameterizedPPO

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
        constraint_kwargs: Optional[dict[str, Any]] = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        record_video: bool = False,
        record_video_episode_trigger: Optional[Callable[[int], bool]] = None,
        video_folder: str = "videos",
        video_kwargs: Optional[dict[str, Any]] = None,
        **kw
    ) -> gym.Env:
    '''

    # Import the labelling function for the ColourBombGridWorldV2 environment
    from masa.envs.tabular.colour_bomb_grid_world_v2 import label_fn
    # Import make_dfa for property 3 
    from masa.examples.colour_bomb_grid_world.property_3 import make_dfa

    # We're going to use the LTLSafety constraint, which has key word args: (dfa: DFA, obs_type: str) 
    constraint_kwargs = dict(
        dfa=make_dfa(), # creates an instance of the dfa
    )

    # First lets initialize the eval_env (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> LTLSafetyEnv -> ConstraintMonitor -> RewardMonitor
    env = make_env("ColourBombGridWorldV2", "LTL_SAFETY", 250, label_fn=label_fn, constraint_kwargs=constraint_kwargs)

    # Now we're going to wrap our environment in ProbShieldWrapperCont
    # The wrapper takes one arg: env
    #   and key word args: 
    #   theta: float = 1e-10,
    #   max_vi_steps: int = 1000,
    #   init_safety_bound: float = 0.5,
    env = ProbShieldWrapperCont(
        env, 
        init_safety_bound=0.01, # safety constraint from the intial state
        theta=1e-15, # early stopping condition for value iteration
        max_vi_steps=10_000, # number of value iteration steps
    )

    # ParameterizedPPO is a on-policy algorithm that takes one arg: env
    #   and key word args:
    #   tensorboard_logdir: Optional[str] = None,
    #   wandb_project: Optional[str] = None,
    #   wandb_name: Optional[str] = None,
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
    #   policy_class: type[BaseJaxPolicy] = ParameterizedPPOPolicy,
    #   policy_kwargs: Optional[dict[str, Any]] = None,

    # First lets initialize the eval_env
    # We can reuse constraint kwargs here as dfa_to_costfn internally creates a deepcopy of the dfa
    eval_env = ProbShieldWrapperCont(
        make_env("ColourBombGridWorldV2", "LTL_SAFETY", 250, label_fn=label_fn, constraint_kwargs=constraint_kwargs),
        init_safety_bound=0.01,
        theta=1e-15,
        max_vi_steps=10_000,
    )

    policy_kwargs = dict(
        log_std_init=-1.5,
        conditional_beta_network=True
    )

    algo = ParameterizedPPO(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="auto", 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
        policy_kwargs=policy_kwargs
        # Using the ParameterizedPPO specific defaults after this
    )

    # Now we begin training
    algo.train(
        num_frames=10_000_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=10_000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=10_000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size=100, # sliding window size for metrics logging
    )

if __name__ == "__main__":
    main()