from __future__ import annotations
from typing import Optional, Any, List
import sys
import importlib.resources as resources
import argparse

import gymnasium as gym
import ruamel.yaml as yaml
from masa.plugins.helpers import load_plugins
from masa.common.configs import Config, Flags, Path
from masa.common.utils import make_env, load_callable
from masa.common.registry import ALGO_REGISTRY, ENV_REGISTRY, CONSTRAINT_REGISTRY

import runpy
import typer
from rich.console import Console

app = typer.Typer(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
console = Console()

def load_yaml(package: str, filename: str) -> dict[str, Any]:
    path = resources.files(package).joinpath(filename)
    return yaml.YAML(typ="safe").load(path.read_text())

def select_cfg(
    package: str,
    filename: str,
    selected: str,
) -> dict[str, Any]:
    configs = load_yaml(package, filename)
    config = Config(configs["defaults"])
    for name in selected:
        config = config.update(configs[name])
    return config

def load_custom_cfg(path: Optional[str]) -> dict[str, Any]:
    if path is None:
        return {}
    return yaml.YAML(typ="safe").load(Path(path).read()) or {}


def deep_update(config: Config, overrides: dict[str, Any]) -> Config:
    for section, values in overrides.items():
        if values is None:
            continue
        for key, value in values.items():
            config = config.update({f"{section}.{key}": value})
    return config
    

def parse_config(env_id, env_cfgs, algo, algo_cfgs) -> Config:
    configs = load_yaml("masa.configs", "defaults.yaml")
    config = Config(configs["defaults"])

    env_configs = load_yaml("masa.configs.envs", f"{env_id}.yaml")
    env_config = Config(env_configs["defaults"])
    config = config.update(env_config)
    for name in env_cfgs:
        config = config.update(env_configs[name])

    algo_config = select_cfg(
        "masa.configs.algorithms",
        f"{algo}.yaml",
        algo_cfgs,
    )

    config = config.update(env_config)
    base = {algo: {**algo_config}, **config}
    config = Config(base)

    return config


def print_config(config: Config, algo: str) -> None:
    console.print()
    console.print("[bold](Environment)[/]")
    console.print(config.env)

    console.print()
    console.print("[bold](Constraint)[/]")
    console.print(config.constraint)

    console.print()
    console.print("[bold](Run)[/]")
    console.print(config.run)

    console.print()
    console.print(f"[bold]({algo.upper()})[/]")
    console.print(config[algo])

    console.print()
    console.print("[bold green]Beginning training ...[/]")


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def run(
    ctx: typer.Context,
    env_id: str = typer.Option(
        "media_streaming", 
        "--env-id", 
        help="Supported environment id in ENV_REGISTRY"
    ),
    algo: str = typer.Option(
        "q_learning",
        "--algo",
        help="Algorithm name registered in ALGO_REGISTRY.",
    ),
    env_cfgs: List[str] = typer.Option(
        [],
        "--env-cfgs",
        help="Environment config variant(s)",
    ),
    algo_cfgs: List[str] = typer.Option(
        [],
        "--algo-cfgs",
        help="Algorithm config variant(s)",
    ),
    constraint: Optional[str] = typer.Option(
        None,
        "--constraint",
        help="Constraint config name.",
    ),
    custom_cfgs: Optional[str] = typer.Option(
        None,
        "--custom-cfgs",
        help="Path to a custom configs YAML file.",
    ),
    total_timesteps: Optional[int] = typer.Option(
        None,
        "--total-timesteps",
        help="Override run.total_timesteps.",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed.")
) -> None:
    r"""
    Train a MASA agent.

    Examples:

        masa run --env-id media_streaming --algo q_learning --constraint pctl

        masa run --env-id bridge_crossing --algo ppo --constraint cmdp \
            --env.max_episode_steps 300 \
            --ppo.learning_rate 0.0003 \
            --constraint.cost_budget 10
    """
    load_plugins()

    if algo not in ALGO_REGISTRY:
        raise typer.BadParameter(
            f"Unknown algorithm '{algo}'. Available: {list(ALGO_REGISTRY.keys())}"
        )
    
    if env_id not in ENV_REGISTRY:
        raise typer.BadParameter(
            f"Unknown env '{env_id}'. Available: {list(ENV_REGISTRY.keys())}"
        )

    if constraint is not None and constraint not in CONSTRAINT_REGISTRY:
        raise typer.BadParameter(
            f"Unknown constraint '{constraint}'. Available: {list(CONSTRAINT_REGISTRY.keys())}"
        )

    config = parse_config(env_id, env_cfgs, algo, algo_cfgs)

    custom_config = load_custom_cfg(custom_cfgs)
    config = deep_update(config, custom_config)

    config = Flags(config).parse(list(ctx.args))

    if total_timesteps is not None:
        config = config.update({"run.total_timesteps": total_timesteps})
    config = config.update({"run.seed": seed})
    
    print_config(config, algo)

    label_fn = load_callable(getattr(config.env, "label_fn", "masa.common.dummy:label_fn"))
    cost_fn = load_callable(getattr(config.env, "cost_fn", "masa.common.dummy:cost_fn"))
    make_dfa = load_callable(getattr(config.constraint, "dfa", "masa.common.dummy:make_dfa"))

    constraint_kwargs = {
        "cost_fn": cost_fn,
        "budget": getattr(config.constraint, "cost_budget", None),
        "alpha": getattr(config.constraint, "alpha", None),
        "avoid_label": getattr(config.constraint, "avoid_label", None),
        "reach_label": getattr(config.constraint, "reach_label", None),
        "dfa": make_dfa()
    }

    constraint_kwargs = {
        key: value for key, value in constraint_kwargs.items() if value is not None
    }

    env_fn = lambda: make_env(
        config.env.id,
        config.constraint.type,
        config.env.max_episode_steps,
        label_fn=label_fn,
        **constraint_kwargs,
    )

    train_env = env_fn()
    
    algo_cls = ALGO_REGISTRY.get(algo)
    algo_kwargs = dict(config[algo])

    base_kwargs = {
        "tensorboard_logdir": config.run.logdir,
        "seed": config.run.seed,
        "device": config.run.device,
        "verbose": config.run.verbose,
        "env_fn": env_fn,
    }

    model = algo_cls(train_env, **base_kwargs, **algo_kwargs)

    run_kwargs = {
        "num_eval_episodes": config.run.eval_episodes,
        "eval_freq": config.run.eval_every,
        "log_freq": config.run.log_every,
        "prefill": config.run.prefill,
        "save_freq": config.run.save_every,
        "stats_window_size": config.run.stats_window_size,
    }

    model.train(config.run.total_timesteps, **run_kwargs)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def example(
    ctx: typer.Context,
    example_script: str = typer.Argument(help="Example script in masa.examples."),
)-> None:
    r"""
    Run an example module from masa.examples.

    Example:

        masa examples prob_shield_example
    """
    module = f"masa.examples.{example_script}"
    sys.argv = [module, *ctx.args]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    app()