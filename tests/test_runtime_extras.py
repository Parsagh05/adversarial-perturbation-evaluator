import tomllib
import json
from pathlib import Path


def _extras():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]


def test_smoke_regression_dependencies_are_declared():
    extras = _extras()
    assert "tensorboard>=2.10" in extras["adaclip"]
    assert "tabulate>=0.9" in extras["bayespfl"]
    assert "humanhash3>=0.0.6" in extras["craneplus"]
    assert "transformers>=4.30,<5" in extras["filo"]


def test_aaclip_example_uses_automatic_checkpoint_resolution():
    config = json.loads(
        Path("configs/aaclip.example.json").read_text(encoding="utf-8")
    )
    for target, kwargs in config["model_kwargs_by_target"].items():
        assert kwargs["target_dataset"] == target
        assert kwargs["download_root"]
        assert "image_checkpoint" not in kwargs
        assert "text_checkpoint" not in kwargs
