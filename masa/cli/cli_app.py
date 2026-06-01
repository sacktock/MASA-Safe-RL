from __future__ import annotations
from typing import Optional, Any, List
import sys
import importlib.resources as resources
import argparse
import warnings

import gymnasium as gym
import ruamel.yaml as yaml
import masa
from masa.algorithms import ALGORITHMS
from masa.envs import ENVIRONMENTS
from masa.common.constraints import CONSTRAINTS
from masa.common import registry
from masa.common.configs import Config, Flags, Path
from masa.common.utils import make_env, load_callable, format_algo_id, format_constraint_id, format_env_id

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

def resolve_id(
    value: str,
    valid_ids: set[str],
    formatter,
    kind: str,
) -> str:
    if value in valid_ids:
        return value
    formatted = formatter(value)
    if formatted in valid_ids:
        typer.echo(
            f"Warning: unknown {kind} '{value}', using '{formatted}' instead.",
            err=True,
        )
        return formatted
    raise typer.BadParameter(
        f"Unknown {kind} '{value}'. Available: {sorted(valid_ids)}"
    )

def load_yaml(package: str, filename: str) -> dict[str, Any]:
    path = resources.files(package).joinpath(filename)
    return yaml.YAML(typ="safe").load(path.read_text())

def select_cfg(
    package: str,
    filename: str,
    selected: List[str],
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
    configs = load_yaml("masa.configs", "Defaults.yaml")
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

    base = {algo: {**algo_config}, **config}
    config = Config(base)

    return config

def parse_benchmark(env_id, env_variant, algo) -> Config:
    configs = load_yaml("masa.configs", "Defaults.yaml")
    config = Config(configs["defaults"])

    env_configs = load_yaml("masa.configs.envs", f"{env_id}.yaml")
    env_config = Config(env_configs["defaults"])
    config = config.update(env_config)
    
    if env_variant is not None:
        if env_variant not in env_configs:
            raise ValueError(
                f"Benchmark variant '{env_variant}' not found in {env_id}.yaml"
            )
        config = config.update(env_configs[env_variant])

    algo_variant =  f"{env_id}_{env_variant}" if env_variant is not None else env_id

    try:
        algo_config = select_cfg(
            "masa.configs.algorithms",
            f"{algo}.yaml",
            [algo_variant],
        )
    except KeyError:
        warnings.warn(f"No benchmark configuration for {algo}:{algo_variant}. Using defaults.")
        algo_configs = load_yaml("masa.configs.algorithms", f"{algo}.yaml")
        algo_config = Config(algo_configs["defaults"])

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

def run_with_config(config, algo):
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

    if config.run.record_video and config.run.record_every == 0:
        warnings.warn(
            "Video recording is enabled, but record_every=0. "
            "Only evaluation episodes will be recorded; training episodes will not be recorded."
        )

    env_kwargs = {"render_mode": "rgb_array"} if config.run.record_video else {} 

    train_env = make_env(
        config.env.id,
        config.constraint.type,
        config.env.max_episode_steps,
        label_fn=label_fn,
        constraint_kwargs=constraint_kwargs,
        env_kwargs=env_kwargs,
        record_video=config.run.record_video,
        record_video_episode_trigger=None,
        video_folder=f"{config.run.logdir}/videos",
        video_kwargs={
            "step_trigger": lambda x: (x % config.run.record_every == 0) and (config.run.record_every != 0),
            "video_length": config.env.max_episode_steps,
            "name_prefix": "training",
        },
    )

    eval_env_fn = lambda: make_env(
        config.env.id,
        config.constraint.type,
        config.env.max_episode_steps,
        label_fn=label_fn,
        constraint_kwargs=constraint_kwargs,
        env_kwargs=env_kwargs,
        record_video=config.run.record_video,
        record_video_episode_trigger=lambda x: (x % config.run.eval_episodes == 0) and (config.run.eval_episodes != 0),
        video_folder=f"{config.run.logdir}/videos",
        video_kwargs={
            "name_prefix": "eval",
        },
    )
    
    algo_cls = registry.get_algorithm(algo)
    algo_kwargs = dict(config[algo])

    base_kwargs = {
        "tensorboard_logdir": config.run.logdir,
        "seed": config.run.seed,
        "device": config.run.device,
        "verbose": config.run.verbose,
        "env_fn": eval_env_fn,
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
def run(
    ctx: typer.Context,
    env_id: str = typer.Option(
        ..., 
        "--env-id", 
        help="Supported environment id in ENV_REGISTRY"
    ),
    algo: str = typer.Option(
        ...,
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

    env_id = resolve_id(
        env_id,
        set(registry.ENV_REGISTRY.keys()),
        format_env_id,
        "env",
    )

    algo = resolve_id(
        algo,
        set(registry.ALGO_REGISTRY.keys()),
        format_algo_id,
        "algorithm",
    )

    if constraint is not None:
        constraint = resolve_id(
            constraint,
            set(registry.CONSTRAINT_REGISTRY.keys()),
            format_constraint_id,
            "constraint",
        )

    config = parse_config(env_id, env_cfgs, algo, algo_cfgs)

    custom_config = load_custom_cfg(custom_cfgs)
    config = deep_update(config, custom_config)

    config = Flags(config).parse(list(ctx.args))

    if constraint is not None:
        config = config.update({"constraint.type": constraint})

    if total_timesteps is not None:
        config = config.update({"run.total_timesteps": total_timesteps})
    config = config.update({"run.seed": seed})

    run_with_config(config, algo)

@app.command()
def benchmark(
    env: str = typer.Argument(
        ...,
        help="Benchmark environment."
    ),
    variant: Optional[str] = typer.Argument(
        None,
        help="Benchmark variant."
    ),
    algo: str = typer.Option(
        ...,
        "--algo",
        help="Algorithm."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
):

    env_id = resolve_id(
        env,
        set(registry.ENV_REGISTRY.keys()),
        format_env_id,
        "env",
    )

    benchmark_id = f"{env_id}_{variant}" if variant is not None else env_id

    algo = resolve_id(
        algo,
        set(registry.ALGO_REGISTRY.keys()),
        format_algo_id,
        "algorithm",
    )

    config = parse_benchmark(env_id, variant, algo)
    config = config.update({"run.seed": seed})
    config = config.update({"run.logdir": f"benchmarks/masa/{benchmark_id}/{algo}_{seed}"})

    run_with_config(config, algo)

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