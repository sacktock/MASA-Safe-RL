from masa.prob_shield.prob_shield_wrapper_disc import ProbShieldWrapperDisc
from masa.algorithms.ppo import PPO
from masa.common.wrappers import FlattenDictObsWrapper, VecWrapper, OneHotObsWrapper
from typing import Dict, Any
import flax.linen as nn
import optax

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

    # Import the labelling function for the ColourBombGridWorldV2 environment
    from masa.envs.tabular.colour_bomb_grid_world_v2 import label_fn
    # Import make_dfa for property 3 
    from masa.examples.colour_bomb_grid_world.property_3 import make_dfa

    # We're going to use the LTLSafety constraint, which has key word args: (dfa: DFA, obs_type: str) 
    constraint_kwargs = constraint_kwargs = dict(
        dfa=make_dfa(), # creates an instance of the dfa
        obs_type="discrete", # the observations should be in dict format
    )

    # First lets initialize the eval_env (env_id, constraint, max_epsiode_steps)
    # make_env wraps the environment in TimeLimit -> LabelledEnv -> LTLSafetyEnv -> ConstraintMonitor -> RewardMonitor
    eval_env = make_env("colour_bomb_grid_world_v2", "ltl_safety", 250, label_fn=label_fn, **constraint_kwargs)
    # Because we're using obs_stype=dict here we need to use a safety abstraction 
    #   as ProbShieldWrapperDisc expected either a discrete state space or obs -> discrete state safety abstraction
    #orig_space_n = eval_env.observation_space["orig"].n
    #def safety_abstraction(obs: Dict[str, Any]) -> int:
    #    state = obs["orig"]
    #    aut_state = obs["automaton"]
    #    return orig_space_n * aut_state + state

    # Now we're going to wrap our environment in ProbShieldWrapperDisc
    # The wrapper takes one arg: env
    #   and key word args: 
    #   theta: float = 1e-10,
    #   max_vi_steps: int = 1000,
    #   init_safety_bound: float = 0.5,
    eval_env = ProbShieldWrapperDisc(
        eval_env,
        #safety_abstraction=safety_abstraction, # discrete safety abstraction for the environment: maps observations to concerete discrete states
        init_safety_bound = 0.01, # Safety constraint from the intial state
        theta = 1e-15, # early stopping condition for value iteration
        max_vi_steps= 10_000, # number of value iteration steps
        granularity = 20,
    )

    # We're going to use VecWrapper for the training env so we need to define a environment creatation function
    def create_env():
        # Intialize the environment 
        env = make_env("colour_bomb_grid_world_v2", "ltl_safety", 250, label_fn=label_fn, **constraint_kwargs)

        # Wrap in ProbShieldWrapperDisc
        env = ProbShieldWrapperDisc(
            env, 
            #safety_abstraction=safety_abstraction, 
            init_safety_bound = 0.01, 
            theta = 1e-15, 
            max_vi_steps= 10_000, 
            granularity = 20,
        )

        # Now we're going to wrap our environment in OneHotObsWrappe and FlattenDictObsWrapper so the observations are compatible with ParameterizedPPO
        env = OneHotObsWrapper(env)
        env = FlattenDictObsWrapper(env)

        return env

    # Now we're going to use VecWrapper to synchronously run 8 environments
    n_envs = 8
    envs = [create_env() for _ in range(n_envs)]
    env = VecWrapper(envs)

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

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=nn.relu
    )

    algo = PPO(
        env,
        tensorboard_logdir=None, # ignoring tensorboard logging
        seed=0,
        monitor=True, # monitors training progress
        device="auto", 
        verbose=0, # verbosity level for monitoring
        eval_env=eval_env, # separate environment instance for eval
        learning_rate=optax.schedules.linear_schedule(2.4e-4, 0.0, 300_000),
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        clip_range=optax.schedules.linear_schedule(0.1, 0.0, 300_000),
        ent_coef=0.02,
        vf_coef=0.5,
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