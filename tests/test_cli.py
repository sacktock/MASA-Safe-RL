from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from masa.cli import cli_app
from masa.common import registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "masa" / "examples"
EXAMPLE_NAMES = tuple(sorted(path.stem for path in EXAMPLES_DIR.glob("*_example.py")))
runner = CliRunner()


def assert_cli_success(result) -> None:
    assert result.exit_code == 0, result.output or repr(result.exception)


def stub_config_loading(monkeypatch, calls):
    config = cli_app.parse_config("MediaStreaming", [], "QL", [])

    def fake_parse_config(env_id, env_cfgs, algo, algo_cfgs):
        calls.append((env_id, env_cfgs, algo, algo_cfgs))
        return config

    monkeypatch.setattr(cli_app, "parse_config", fake_parse_config)


def test_cli_help_exposes_run_and_example_commands():
    result = runner.invoke(cli_app.app, ["--help"])

    assert_cli_success(result)
    assert "run" in result.output
    assert "example" in result.output


def test_run_applies_config_layers_and_cli_overrides(monkeypatch, tmp_path):
    captured = []
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        """
run:
  total_timesteps: 99
  log_every: 41
QL:
  alpha: 0.2
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_app,
        "run_with_config",
        lambda config, algo: captured.append((config, algo)),
    )

    result = runner.invoke(
        cli_app.app,
        [
            "run",
            "--env-id",
            "MediaStreaming",
            "--algo",
            "QL",
            "--algo-cfgs",
            "MediaStreaming",
            "--constraint",
            "PCTL",
            "--custom-cfgs",
            str(custom_config),
            "--total-timesteps",
            "7",
            "--seed",
            "13",
            "--QL.gamma",
            "0.8",
        ],
    )

    assert_cli_success(result)
    assert len(captured) == 1
    config, algo = captured[0]
    assert algo == "QL"
    assert config.env.id == "MediaStreaming"
    assert config.constraint.type == "PCTL"
    assert config.run.total_timesteps == 7
    assert config.run.log_every == 41
    assert config.run.seed == 13
    assert config.QL.alpha == pytest.approx(0.2)
    assert config.QL.gamma == pytest.approx(0.8)


@pytest.mark.parametrize("env_id", sorted(registry.ENV_REGISTRY.keys()))
def test_run_command_accepts_every_registered_environment(monkeypatch, env_id):
    captured = []
    parse_calls = []
    stub_config_loading(monkeypatch, parse_calls)
    monkeypatch.setattr(
        cli_app,
        "run_with_config",
        lambda config, algo: captured.append((config, algo)),
    )

    result = runner.invoke(
        cli_app.app,
        ["run", "--env-id", env_id, "--algo", "QL"],
    )

    assert_cli_success(result)
    _, algo = captured[0]
    assert parse_calls == [(env_id, [], "QL", [])]
    assert algo == "QL"


@pytest.mark.parametrize("algo", sorted(registry.ALGO_REGISTRY.keys()))
def test_run_command_accepts_every_registered_algorithm(monkeypatch, algo):
    captured = []
    parse_calls = []
    stub_config_loading(monkeypatch, parse_calls)
    monkeypatch.setattr(
        cli_app,
        "run_with_config",
        lambda config, selected_algo: captured.append((config, selected_algo)),
    )

    result = runner.invoke(
        cli_app.app,
        ["run", "--env-id", "MediaStreaming", "--algo", algo],
    )

    assert_cli_success(result)
    _, selected_algo = captured[0]
    assert parse_calls == [("MediaStreaming", [], algo, [])]
    assert selected_algo == algo


@pytest.mark.parametrize("constraint", sorted(registry.CONSTRAINT_REGISTRY.keys()))
def test_run_command_accepts_every_registered_constraint(monkeypatch, constraint):
    captured = []
    parse_calls = []
    stub_config_loading(monkeypatch, parse_calls)
    monkeypatch.setattr(
        cli_app,
        "run_with_config",
        lambda config, algo: captured.append((config, algo)),
    )

    result = runner.invoke(
        cli_app.app,
        [
            "run",
            "--env-id",
            "MediaStreaming",
            "--algo",
            "QL",
            "--constraint",
            constraint,
        ],
    )

    assert_cli_success(result)
    config, algo = captured[0]
    assert config.constraint.type == constraint
    assert parse_calls == [("MediaStreaming", [], "QL", [])]
    assert algo == "QL"


def test_run_rejects_unknown_registered_ids(monkeypatch):
    called = False

    def fail_if_called(config, algo):
        nonlocal called
        called = True

    monkeypatch.setattr(cli_app, "run_with_config", fail_if_called)

    result = runner.invoke(
        cli_app.app,
        ["run", "--env-id", "not_an_env", "--algo", "QL"],
    )

    assert result.exit_code == 2
    assert "Unknown env 'not_an_env'" in result.output
    assert "Available:" in result.output
    assert called is False


def test_run_with_config_builds_training_and_evaluation_environments(monkeypatch):
    make_env_calls = []
    algorithm_instances = []
    train_calls = []
    label_fn = object()
    cost_fn = object()
    dfa = object()

    config = cli_app.parse_config("MediaStreaming", [], "QL", [])
    config = config.update(
        {
            "run.total_timesteps": 7,
            "run.eval_every": 3,
            "run.eval_episodes": 2,
            "run.log_every": 4,
            "run.record_video": True,
            "run.record_every": 5,
            "run.logdir": "test-runs",
        }
    )

    def fake_load_callable(path):
        if path.endswith(":label_fn"):
            return label_fn
        if path.endswith(":cost_fn"):
            return cost_fn
        if path.endswith(":make_dfa"):
            return lambda: dfa
        raise AssertionError(f"Unexpected callable path: {path}")

    def fake_make_env(*args, **kwargs):
        env = f"env-{len(make_env_calls)}"
        make_env_calls.append((args, kwargs, env))
        return env

    class FakeAlgorithm:
        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs
            self.eval_env = kwargs["env_fn"]()
            algorithm_instances.append(self)

        def train(self, *args, **kwargs):
            train_calls.append((args, kwargs))

    monkeypatch.setattr(cli_app, "load_callable", fake_load_callable)
    monkeypatch.setattr(cli_app, "make_env", fake_make_env)
    monkeypatch.setattr(cli_app.registry, "get_algorithm", lambda algo: FakeAlgorithm)

    cli_app.run_with_config(config, "QL")

    assert len(make_env_calls) == 2
    train_args, train_kwargs, train_env = make_env_calls[0]
    eval_args, eval_kwargs, eval_env = make_env_calls[1]
    assert train_args == ("MediaStreaming", "PCTL", 40)
    assert eval_args == train_args
    assert train_kwargs["label_fn"] is label_fn
    assert train_kwargs["constraint_kwargs"]["cost_fn"] is cost_fn
    assert train_kwargs["constraint_kwargs"]["dfa"] is dfa
    assert train_kwargs["record_video"] is True
    assert train_kwargs["video_folder"] == "test-runs/videos"
    assert train_kwargs["video_kwargs"]["step_trigger"](5) is True
    assert train_kwargs["video_kwargs"]["step_trigger"](6) is False
    assert eval_kwargs["record_video_episode_trigger"](2) is True
    assert eval_kwargs["record_video_episode_trigger"](3) is False

    assert len(algorithm_instances) == 1
    algorithm = algorithm_instances[0]
    assert algorithm.env == train_env
    assert algorithm.eval_env == eval_env
    assert algorithm.kwargs["seed"] == config.run.seed
    assert algorithm.kwargs["alpha"] == config.QL.alpha
    assert algorithm.kwargs["gamma"] == config.QL.gamma
    assert train_calls == [
        (
            (7,),
            {
                "num_eval_episodes": 2,
                "eval_freq": 3,
                "log_freq": 4,
                "prefill": config.run.prefill,
                "save_freq": config.run.save_every,
                "stats_window_size": config.run.stats_window_size,
            },
        )
    ]


def test_example_inventory_is_not_empty():
    assert EXAMPLE_NAMES


@pytest.mark.parametrize("example_name", EXAMPLE_NAMES)
def test_example_command_dispatches_every_example(monkeypatch, example_name):
    calls = []

    def fake_run_module(module, *, run_name):
        calls.append((module, run_name, list(sys.argv)))

    monkeypatch.setattr(cli_app.runpy, "run_module", fake_run_module)
    monkeypatch.setattr(cli_app.sys, "argv", ["pytest"])

    result = runner.invoke(
        cli_app.app,
        ["example", example_name, "--example-option", "value"],
    )

    assert_cli_success(result)
    module = f"masa.examples.{example_name}"
    assert calls == [(module, "__main__", [module, "--example-option", "value"])]


def test_norm_obs_example_initializes_model_through_cli(monkeypatch):
    from masa.algorithms.on_policy import PPO

    train_calls = []

    def fake_train(self, *args, **kwargs):
        train_calls.append((args, kwargs))
        self.env.close()
        if self._eval_env is not None:
            self._eval_env.close()

    monkeypatch.setattr(PPO, "train", fake_train)

    result = runner.invoke(cli_app.app, ["example", "norm_obs_example"])

    assert_cli_success(result)
    assert train_calls == [
        (
            (),
            {
                "num_frames": 100_000,
                "num_eval_episodes": 10,
                "eval_freq": 5_000,
                "log_freq": 5_000,
                "stats_window_size": 100,
            },
        )
    ]
