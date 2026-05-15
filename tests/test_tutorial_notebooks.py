from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_01 = REPO_ROOT / "notebooks" / "tutorials" / "01_first_masa_experiment.ipynb"
TUTORIAL_02 = REPO_ROOT / "notebooks" / "tutorials" / "02_labels_costs_and_infos.ipynb"
TUTORIAL_03 = REPO_ROOT / "notebooks" / "tutorials" / "03_wrapper_stack.ipynb"
TUTORIAL_04 = REPO_ROOT / "notebooks" / "tutorials" / "04_constraints_tour.ipynb"


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
