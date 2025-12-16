from __future__ import annotations
from typing import Optional
import argparse
import gymnasium as gym
import ruamel.yaml as yaml
from masa.plugins.helpers import load_plugins
from masa.common.configs import Config, Flags, Path
from masa.common.utils import make_env, load_callable
from masa.common.registry import ALGO_REGISTRY

def parse_config(args, unknown) -> Config:
    configs = yaml.YAML(typ='safe').load(
      (Path(__file__).parent / args.configs).read())
    config = Config(configs["defaults"])
    for name in args.algo_configs:
        config = config.update(configs[name])
    config = Flags(config).parse(unknown)
    if args.env_id is not None:
        config = config.update({"env.id": args.env_id})
    if args.max_episode_steps is not None:
        config = config.update({"env.max_episode_steps": args.max_episode_steps})
    if args.label_fn is not None:
        config = config.update({"env.label_fn": args.label_fn})
    if args.cost_fn is not None:
        config = config.update({"env.cost_fn": args.cost_fn})
    if args.dfa is not None:
        config = config.update({"constraint.dfa": args.dfa})
    if args.total_timesteps is not None:
        config = config.update({"run.total_timesteps": args.total_timesteps})
    config = config.update({"run.seed": args.seed})
    return config

def print_config(config, algo):
    print();print("(Environment)");print(config.env)
    print();print("(Constraint)");print(config.constraint)
    print();print("(Run)");print(config.run)
    print();print(f"({algo.upper()})");print(config[algo])
    print();print();print("Beginning training ...")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-level CLI args for MASA-Safe-RL")
    parser.add_argument("--env-id", type=str, default=None, help="Override env id (otherwise from YAML).")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                   help="Override max episode steps (otherwise from YAML).")
    parser.add_argument("--label-fn", type=str, default=None, help="module:callable returning Iterable[str]") # "common.dummy:label_fn"
    parser.add_argument("--cost-fn", type=str, default=None, help="module:callable returning 0/1 cost") # "common.dummy:cost_fn"
    parser.add_argument("--dfa", type=str, default=None, help="module:callable returning a DFA object") # "common.dummy:make_dfa"
    parser.add_argument("--constraint", type=str, default=None)
    parser.add_argument("--configs", type=str, default="configs.yaml",
                   help="Path to YAML with env/algo/run configs.")
    parser.add_argument("--algo", type=str, default="q_learning",
                   choices=ALGO_REGISTRY.keys(),
                   help="Which algorithm to run (used to pick section from YAML).")
    parser.add_argument("--algo-configs", type=str, nargs='+', default=[],
                   help="algorithm configs to use from configs file.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                   help="Override run total_timesteps (otherwise from YAML).")
    parser.add_argument("--seed", type=int, default=0)
    return parser

def main():

    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    config = parse_config(args, unknown)
    print_config(config, args.algo)

    algo_kwargs = {**config[args.algo]}

    label_fn = load_callable(config.env.label_fn)
    cost_fn = load_callable(config.env.cost_fn)
    make_dfa = load_callable(config.constraint.dfa)

    constraint_kwargs = dict(
        cost_fn=cost_fn,
        budget=config.constraint.cost_budget,
        alpha=config.constraint.alpha,
        avoid_label=config.constraint.avoid_label,
        reach_label=config.constraint.reach_label,
        dfa=make_dfa()
    )

    env_fn = lambda: make_env(
        config.env.id, 
        config.constraint.type, 
        config.env.max_episode_steps,
        label_fn=label_fn,
        **constraint_kwargs,
    )

    train_env = env_fn()

    algo_cls = ALGO_REGISTRY.get(args.algo)
    base_kwargs = dict(
        tensorboard_logdir=config.run.logdir,
        seed=config.run.seed,
        device=config.run.device,
        verbose=config.run.verbose,
        env_fn=env_fn,
    )

    merged = {**base_kwargs, **algo_kwargs}
    algo = algo_cls(train_env, **merged)

    run_kwargs = dict(
        num_eval_episodes=config.run.eval_episodes,
        eval_freq=config.run.eval_every,
        log_freq=config.run.log_every,
        prefill=config.run.prefill,
        save_freq=config.run.save_every,
        stats_window_size=config.run.stats_window_size,
    )

    algo.train(config.run.total_timesteps, **run_kwargs)

if __name__ == "__main__":
    load_plugins() # required
    main()