import json
from pathlib import Path


def test_kaggle_notebook_code_cells_compile_and_clone_evaluator():
    for path in (
        Path("notebooks/kaggle_anomalyclip.ipynb"),
        Path("notebooks/kaggle_aaclip.ipynb"),
    ):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(source, str(path), "exec")
        assert "https://github.com/Parsagh05/adversarial-perturbation-evaluator.git" in source
        assert "discover_kaggle_inputs" in source
        assert "inventory_attack_setups" in source
