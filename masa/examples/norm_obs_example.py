from masa.common.wrappers import NormWrapper, VecNormWrapper, DummyVecWrapper
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

    # Import the labelling and cost functions for the ContinuousCartpole
    from masa.envs.continuous.cartpole import label_fn, cost_fn

    # We're going to use the PCTL constraint, which has key word args: (cost_fn CostFn: = DummyCostFn, alpha: float = 0.01) 
    constraint_kwargs = constraint_kwargs = dict(
        cost_fn=cost_fn,
        alpha=0.01,
    )

    # Intialize the environment (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> PCTLEnv -> ConstraintMonitor -> RewardMonitor
    env = make_env("cont_cartpole", "pctl", 500, label_fn=label_fn, **constraint_kwargs)
    # PPO will automatically wrap any env in DummyVecWrapper it is has not already been wrapped
    # Since we want VecNormWrapper to be the top level we wrap it before passing to PPO
    env = DummyVecWrapper(env)

    # Now we're going to wrap our environment in NormWrapper
    # The wrapper takes one arg: env
    #   and key word args: 
    #   norm_obs: bool = True,
    #   norm_rew: bool = True,
    #   training: bool = True,
    #   clip_obs: float = 10.0,
    #   clip_rew: float = 10.0,
    #   gamma: float = 0.99,
    #   eps: float = 1e-8
    env = VecNormWrapper(
        env, 
        norm_obs=True, # normalize observations with running mean and std
        norm_rew=False, # normalize the reward with running mean and std of the returns
        training=True, # if training=True then the running mean and stds are updated
        clip_obs= 10.0, # observations are clipped in the range [-10, 10] after normalization for stability
        clip_rew=10.0, # rewards are clipped in the range [-10, 10] after normalization for stability
        gamma=0.99, # discount factor
        eps=1e-8, # small epsilon for divide by zero issues
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
    eval_env = make_env("cont_cartpole", "pctl", 500, label_fn=label_fn, **constraint_kwargs)
    eval_env = NormWrapper(
        eval_env,
        norm_obs=True,
        norm_rew=False,
        training=False, # we don't want to update the running means and std during eval episodes
        clip_obs= 10.0,
        clip_rew=10.0,
        gamma=0.99,
        eps=1e-8,
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
        num_frames=100_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=5000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=5000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size = 100, # sliding window size for metrics logging
    )

if __name__ == "__main__":
    main()