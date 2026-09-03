import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.afclip import WEIGHT_FILES, ZERO_SHOT_SOURCE, resolve_weights


def test_adapter_is_registered_under_both_spellings():
    assert "afclip" in adapter_names()
    assert "af-clip" in adapter_names()


def test_zero_shot_weights_never_train_on_the_evaluated_dataset():
    # main.py names the files after the TRAINING dataset, and test.sh pairs
    # them across datasets.
    assert ZERO_SHOT_SOURCE == {"mvtec": "visa", "visa": "mvtec"}
    for target, source in ZERO_SHOT_SOURCE.items():
        assert target != source


def test_weight_file_names_match_the_official_loader():
    assert WEIGHT_FILES == ("{source}_prompt.pt", "{source}_adaptor.pt")


def test_resolve_weights_finds_the_in_repo_pair(tmp_path):
    folder = tmp_path / "weight"
    folder.mkdir()
    for name in ("visa_prompt.pt", "visa_adaptor.pt"):
        (folder / name).write_bytes(b"weights")
    prompt, adaptor = resolve_weights(tmp_path, "visa")
    assert prompt.name == "visa_prompt.pt"
    assert adaptor.name == "visa_adaptor.pt"


def test_resolve_weights_accepts_an_explicit_directory(tmp_path):
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    for name in ("mvtec_prompt.pt", "mvtec_adaptor.pt"):
        (folder / name).write_bytes(b"weights")
    prompt, _ = resolve_weights(tmp_path, "mvtec", str(folder))
    assert prompt.parent == folder.resolve()


def test_resolve_weights_lists_what_is_available(tmp_path):
    folder = tmp_path / "weight"
    folder.mkdir()
    (folder / "visa_prompt.pt").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="visa_prompt.pt"):
        resolve_weights(tmp_path, "visa")


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("afclip", **common, image_size=336)
    with pytest.raises(ValueError, match=r"ViT-L/14@336px"):
        create_adapter("afclip", **common, clip_model="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("afclip", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("afclip", repository=str(tmp_path), target_dataset="mvtec")
