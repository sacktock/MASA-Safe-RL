from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_01 = REPO_ROOT / "notebooks" / "tutorials" / "01_first_masa_experiment.ipynb"
TUTORIAL_02 = REPO_ROOT / "notebooks" / "tutorials" / "02_labels_costs_and_infos.ipynb"
TUTORIAL_03 = REPO_ROOT / "notebooks" / "tutorials" / "03_wrapper_stack.ipynb"
TUTORIAL_04 = REPO_ROOT / "notebooks" / "tutorials" / "04_constraints_tour.ipynb"
TUTORIAL_05 = REPO_ROOT / "notebooks" / "tutorials" / "05_ltl_safety_colour_bomb.ipynb"
TUTORIAL_06 = REPO_ROOT / "notebooks" / "tutorials" / "06_tabular_safe_rl_baselines.ipynb"
TUTORIAL_07 = REPO_ROOT / "notebooks" / "tutorials" / "07_continuous_safe_rl_baselines.ipynb"
TUTORIAL_08 = REPO_ROOT / "notebooks" / "tutorials" / "08_create_a_new_environment.ipynb"
TUTORIAL_09 = REPO_ROOT / "notebooks" / "tutorials" / "09_probabilistic_shielding_minipacman.ipynb"
TUTORIAL_10 = REPO_ROOT / "notebooks" / "tutorials" / "10_safety_abstractions_pacman_coins.ipynb"


def _notebook_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])


def test_first_masa_experiment_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_01, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "make_env",
        "bridge_crossing",
        "pctl",
        "QL",
        "num_frames=20",
        'info["labels"]',
        'info["constraint"]',
    ):
        assert token in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_labels_costs_and_infos_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_02, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "colour_grid_world",
        "cmdp",
        "label_fn",
        "cost_fn",
        'info["labels"]',
        'info["constraint"]',
        "terminated",
        "truncated",
        "budget=0.0",
    ):
        assert token in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_wrapper_stack_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_03, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "TimeLimit",
        "LabelledEnv",
        "CumulativeCostEnv",
        "ConstraintMonitor",
        "RewardMonitor",
        "make_env",
        "is_wrapped",
        "get_wrapped",
        "colour_grid_world",
    ):
        assert token in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_constraints_tour_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_04, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "cmdp",
        "prob",
        "pctl",
        "reach_avoid",
        "ltl_safety",
        "DFA",
        "Atom",
        "colour_grid_world",
        "avoid_label",
        "reach_label",
        "notebooks.tutorials.helpers.constraints_tour",
        "SVG",
        "render_cmdp_prob_svg",
        "render_grid_trace_svg",
        "render_constraint_semantics_svg",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_ltl_safety_colour_bomb_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_05, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "colour_bomb_grid_world",
        "colour_bomb_grid_world_v2",
        "ltl_safety",
        "DFA",
        "Atom",
        "property_2",
        "property_3",
        'obs_type="dict"',
        "automaton_state",
        "cum_unsafe",
        "satisfied",
        "render_colour_bomb_trace_svg",
        "render_ltl_rollout_timeline_svg",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_tabular_safe_rl_baselines_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_06, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "colour_grid_world",
        "cmdp",
        "QL",
        "QL_Lambda",
        "LCRL",
        "SEM",
        "RECREG",
        "q_learning",
        "q_learning_lambda",
        "lcrl",
        "sem",
        "recreg",
        "num_frames=20",
        "cost_lambda",
        "override_rate",
        "satisfied",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_continuous_safe_rl_baselines_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_07, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "continuous",
        "cont_cartpole",
        "cmdp",
        "PPO",
        "CPO",
        "PPO Lagrangian",
        "PPO_STUB_CONFIG",
        "make_continuous_safe_env",
        "BASELINE_STATUS",
        "stub",
        "constraint",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_create_a_new_environment_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_08, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "TinyDeliveryEnv",
        "gymnasium",
        "spaces.Discrete",
        "ENV_REGISTRY",
        "make_env",
        "cmdp",
        "label_fn",
        "cost_fn",
        "tutorial_tiny_delivery",
        "safe_actions",
        "unsafe_actions",
        "test_tiny_delivery_wrapped_env",
        'info["labels"]',
        'info["constraint"]',
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_probabilistic_shielding_minipacman_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_09, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "mini_pacman",
        "pctl",
        "ProbShieldWrapperDisc",
        "init_safety_bound",
        "safety_lb",
        "successor_states_matrix",
        "probabilities",
        "max_successors",
        "orig_obs",
        "safety_bound",
        "_project_act",
        "project_candidate_action",
        "safe_actions",
        "bounds",
        "alpha=0.01",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()


def test_safety_abstractions_pacman_coins_notebook_is_valid_and_executable():
    notebook = nbformat.read(TUTORIAL_10, as_version=4)

    assert notebook["nbformat"] == 4

    source = _notebook_source(notebook)
    for token in (
        "mini_pacman_with_coins",
        "pacman_with_coins",
        "ProbShieldWrapperDisc",
        "safety_abstraction",
        "abstr_label_fn",
        "original_state_upper_bound",
        "safety_abstract_states",
        "compression_factor",
        "coin_bits",
        "orig_obs",
        "safety_bound",
        "successor_states_matrix",
        "probabilities",
        "project_candidate_action",
        "alpha=0.01",
    ):
        assert token in source
    assert "<svg" not in source

    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
