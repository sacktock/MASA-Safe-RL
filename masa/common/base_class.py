from typing import Any, Optional, TypeVar, Union
from gymnasium import spaces, Env
import jax
from masa.metrics import RolloutLogger, MetricsLogger

class Base_Algorithm:

    def __init__(self,
        env: Env,
        configs: Optional[dict[str, Any]] = None,
        tensorboard_logdir: Optional[str] = None,
        monitor: bool = True,
        seed: Optional[int] = None,
        device: str = "auto",
        verbose: int = 0,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        self.env = env
        self.configs = configs
        self.tensorboard_logdir = tensorboard_logdir
        self.monitor = monitor
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.supported_action_spaces = supported_action_spaces

        self.key = jax.random.PRNGKey(0)

    def train(self, 
        num_frames: int,
        num_eval: Optional[int] = None,
        eval_freq: int = 0,
        log_freq: int = 0,
        prefill: Optional[int] = None,
        save: bool = False,
        mets_logger_kwargs: dict = {},
        rollout_logger_kwargs: dict = {},
    ):

        if self.tensorboard_logdir is not None:
            summary_writer = tf.summary.create_file_writer(self.tensorboard_logdir)
        else:
            summary_writer = None

        if self.monitor:
            train_logger = MetricsLogger(
                tensorboard=(self.tensorboard_logdir is not None),
                summary_writer=summary_writer,
                prefix="train/metrics",
                **mets_logger_kwargs
            )
        
        rollout_logger = RolloutLogger(
            tensorboard=(self.tensorboard_logdir is not None),
            summary_writer=summary_writer,
            prefix="train/rollout",
            **rollout_logger_kwargs
        )

        eval_logger = RolloutLogger(
            tensorboard=(self.tensorboard_logdir is not None),
            summary_writer=summary_writer,
            prefix="eval/rollout",
            **rollout_logger_kwargs
        )

        total_steps = 0

        next_eval = eval_freq
        next_log = log_freq

        for step in range(math.ceil((num_frames)/self.train_ratio)):
            rollout_metrics = self.rollout()
            train_metrics = self.optimize()

            rollout_logger.add(rollout_metrics)
            train_logger.add(train_metrics)

            total_steps += self.train_ratio

            if eval_freq and (total_steps >= next_eval):
                next_eval += eval_freq
                eval_metrics = self.eval()
                eval_logger.add(eval_metrics)
                if save:
                    self.save()

            if log_freq and (total_steps >= log_freq):
                next_log += log_freq
                if self.monitor:
                    train_logger.log()
                rollout_logger.log()
                eval_logger.log()


    def optimize(self):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    def eval(self, num_eval: int):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError

    @property
    def train_ratio(self):
        raise NotImplementedError
