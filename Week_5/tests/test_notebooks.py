from pathlib import Path

import nbformat
import pytest


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))


def test_monitoring_notebooks_are_present():
    assert {
        "model_monitoring_demo.ipynb",
        "model_monitoring_observability.ipynb",
        "prometheus_integration_guide.ipynb",
    }.issubset({path.name for path in NOTEBOOKS})


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_is_valid_json(notebook_path):
    notebook = nbformat.reads(
        notebook_path.read_text(encoding="utf-8"),
        as_version=4,
    )

    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert cell["id"]
        if cell["cell_type"] == "code":
            assert isinstance(cell["outputs"], list)
    nbformat.validate(notebook)
