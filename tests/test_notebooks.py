import json
from pathlib import Path


# Adapters are registered under aliases too; the canonical name is the one the
# config's "model" field uses. Each lives under the folder for its regime.
ZERO_SHOT = {
    "anomalyclip", "aaclip", "adaclip", "faprompt", "crane",
    "aprilgan", "fbclip", "tipsomaly", "vcpclip", "filo",
    "bayespfl", "afclip", "cops", "mrad", "winclip",
}
FEW_SHOT = {"winclip_fewshot", "afclip_fewshot", "aprilgan_fewshot"}


def regime(model: str) -> str:
    return "few_shot" if model in FEW_SHOT else "zero_shot"


def notebook_path(model: str) -> Path:
    return Path("notebooks") / regime(model) / f"kaggle_{model}.ipynb"


def script_path(model: str) -> Path:
    return Path("scripts") / regime(model) / f"kaggle_run_{model}.py"


def test_kaggle_notebook_code_cells_compile_and_clone_evaluator():
    for model in sorted(ZERO_SHOT | FEW_SHOT):
        path = notebook_path(model)
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

    canonical = ZERO_SHOT | FEW_SHOT
    assert canonical <= set(adapter_names())
    for model in sorted(canonical):
        for path in (
            Path("configs") / f"{model}.example.json",
            notebook_path(model),
            script_path(model),
        ):
            assert path.is_file(), f"missing {path}"
        config = json.loads(
            (Path("configs") / f"{model}.example.json").read_text(encoding="utf-8")
        )
        assert config["model"] == model
        script = script_path(model).read_text(encoding="utf-8")
        assert f"{model}.json" in script
        assert f"configs/{model}.example.json" in script


def test_notebooks_and_scripts_are_split_by_regime():
    """Nothing may sit loose in notebooks/ or scripts/ once the split exists."""
    for folder, pattern in (
        (Path("notebooks"), "kaggle_*.ipynb"),
        (Path("scripts"), "kaggle_run_*.py"),
    ):
        stray = sorted(str(path) for path in folder.glob(pattern))
        assert not stray, f"move these into {folder}/zero_shot or {folder}/few_shot: {stray}"
        for regime_name in ("zero_shot", "few_shot"):
            assert (folder / regime_name).is_dir(), f"missing {folder / regime_name}"
