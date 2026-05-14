from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_01 = REPO_ROOT / "notebooks" / "tutorials" / "01_first_masa_experiment.ipynb"
TUTORIAL_02 = REPO_ROOT / "notebooks" / "tutorials" / "02_labels_costs_and_infos.ipynb"


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
