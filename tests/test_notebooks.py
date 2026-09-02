import json
from pathlib import Path


def test_kaggle_notebook_code_cells_compile_and_clone_evaluator():
    for path in (
        Path("notebooks/kaggle_anomalyclip.ipynb"),
        Path("notebooks/kaggle_aaclip.ipynb"),
        Path("notebooks/kaggle_adaclip.ipynb"),
        Path("notebooks/kaggle_faprompt.ipynb"),
        Path("notebooks/kaggle_crane.ipynb"),
        Path("notebooks/kaggle_aprilgan.ipynb"),
        Path("notebooks/kaggle_fbclip.ipynb"),
        Path("notebooks/kaggle_tipsomaly.ipynb"),
        Path("notebooks/kaggle_vcpclip.ipynb"),
        Path("notebooks/kaggle_filo.ipynb"),
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


def test_every_adapter_ships_a_config_notebook_and_script():
    """Adding an adapter without its entry points is easy to miss."""
    from fpeval.adapters import adapter_names

    # Adapters are registered under aliases too; the canonical name is the one
    # the config's "model" field uses.
    canonical = {
        "anomalyclip", "aaclip", "adaclip", "faprompt", "crane",
        "aprilgan", "fbclip", "tipsomaly", "vcpclip", "filo",
    }
    assert canonical <= set(adapter_names())
    for model in sorted(canonical):
        for path in (
            Path("configs") / f"{model}.example.json",
            Path("notebooks") / f"kaggle_{model}.ipynb",
            Path("scripts") / f"kaggle_run_{model}.py",
        ):
            assert path.is_file(), f"missing {path}"
        config = json.loads(
            (Path("configs") / f"{model}.example.json").read_text(encoding="utf-8")
        )
        assert config["model"] == model
        script = (Path("scripts") / f"kaggle_run_{model}.py").read_text(encoding="utf-8")
        assert f"{model}.json" in script
        assert f"configs/{model}.example.json" in script
