from masa.prob_shield.cpo_shield import CPOShield

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

    # Import the labelling and cost functions for Mini-Pacman
    from masa.envs.tabular.colour_bomb_grid_world import label_fn, cost_fn

    # We're going to use the PCTL constraint, which has key word args: (cost_fn CostFn: = DummyCostFn, alpha: float = 0.01) 
    constraint_kwargs = dict(
        cost_fn=cost_fn,
        alpha=0.05,
    )

    # Intialize the environment (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> PCTLEnv -> ConstraintMonitor -> RewardMonitor
    env = make_env("ColourBombGridWorld", "PCTL", 100, label_fn=label_fn, constraint_kwargs=constraint_kwargs)
    eval_env = make_env("ColourBombGridWorld", "PCTL", 100, label_fn=label_fn, constraint_kwargs=constraint_kwargs)

    # Now let's initialize CPOShield
    # CPOShield will automatically one-hot encode any discrete observations and flatten any dict observations
    algo = CPOShield(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="auto", 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
        cost_limit=0.05
        # Using the CPOShield specific defaults after this
    )

    # Now we begin training
    algo.train(
        num_frames=200_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=2_000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=2_000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size=100, # sliding window size for metrics logging
    )

if __name__ == "__main__":
    main()