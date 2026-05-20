from __future__ import annotations
from masa.algorithms.on_policy import PPO
from pathlib import Path


class RecordNextEpisodeEveryNSteps:
    """Episode trigger that records after crossing total-step intervals."""

    def __init__(self, interval: int):
        if interval < 1:
            raise ValueError("interval must be at least 1")

        self.interval = interval
        self.total_steps = 0
        self.next_threshold = interval
        self.pending_recordings = 0

    def observe_step(self) -> None:
        self.total_steps += 1

        while self.total_steps >= self.next_threshold:
            self.pending_recordings += 1
            self.next_threshold += self.interval

    def __call__(self, episode_id: int) -> bool:
        del episode_id

        if self.pending_recordings < 1:
            return False

        self.pending_recordings -= 1
        return True

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

    # Import the labelling and cost functions for the PacmanWithCoins
    from masa.envs.discrete.pacman_with_coins import label_fn, cost_fn

    # We're going to use the PCTL constraint, which has key word args: (cost_fn CostFn: = DummyCostFn, alpha: float = 0.01) 
    constraint_kwargs = dict(
        cost_fn=cost_fn,
        alpha=0.01,
    )

    # Intialize the environment (env_id, constraint, max_epsiode_steps) with video logging
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> PCTLEnv -> ConstraintMonitor -> RewardMonitor -> RecordVideo
    env = make_env(
        "PacmanWithCoins", 
        "PCTL", 
        1000, 
        label_fn=label_fn, 
        constraint_kwargs=constraint_kwargs,
        env_kwargs = {"render_mode": "rgb_array"},
        record_video=True,
        record_video_episode_trigger=RecordNextEpisodeEveryNSteps(10_000),
        video_folder="videos/PPO-PacmanWithCoins-Seed-0",
        video_kwargs={
            "name_prefix": "training",
            "video_length": 1000,
        },
    )

    # PPO is a on-policy algorithm that takes one arg: env
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
    #   policy_class: type[BaseJaxPolicy] = PPOPolicy,
    #   policy_kwargs: Optional[dict[str, Any]] = None,

    # First lets initialize the eval_env with video logging 
    eval_env = make_env(
        "PacmanWithCoins", 
        "PCTL", 
        1000, 
        label_fn=label_fn, 
        constraint_kwargs=constraint_kwargs,
        env_kwargs = {"render_mode": "rgb_array"},
        record_video=True,
        record_video_episode_trigger=lambda x: x % 10 == 0, # log one evale epsiode every 10 epsiodes
        video_folder="videos/PPO-PacmanWithCoins-Seed-0",
        video_kwargs={
            "name_prefix": "eval",
        },
    )

    algo = PPO(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="auto", 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
    )

    # Now we begin training
    algo.train(
        num_frames=500_000, # total number of frames (environment interactions)
        num_eval_episodes=10, # total number of evaluation episodes to run
        eval_freq=10_000, # how frequently to run evaluation (default=0 => never run evaluation)
        log_freq=10_000, # how frequenntly to log metrics to stdout or tensorboard
        # prefill: Optional[int] = None (not implemented yet)
        # save_freq: int = 0, (not implemented yet)
        stats_window_size=100, # sliding window size for metrics logging
    )

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()