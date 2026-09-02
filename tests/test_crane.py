import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.crane import ZERO_SHOT_CHECKPOINT, resolve_checkpoint


def test_adapter_is_registered():
    assert "crane" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    # test.sh evaluates MVTec with the VisA-trained weights and vice versa.
    assert ZERO_SHOT_CHECKPOINT == {
        "mvtec": "trained_on_visa_crane",
        "visa": "trained_on_mvtec_crane",
    }
    assert "mvtec" not in ZERO_SHOT_CHECKPOINT["mvtec"]
    assert "visa" not in ZERO_SHOT_CHECKPOINT["visa"]


def test_named_checkpoints_resolve_inside_the_repository(tmp_path):
    # Crane ships its released weights in-repo, one file per epoch.
    for name in ZERO_SHOT_CHECKPOINT.values():
        folder = tmp_path / "checkpoints" / name
        folder.mkdir(parents=True)
        (folder / "epoch_5.pth").write_bytes(b"weights")
        (folder / "epoch_1.pth").write_bytes(b"weights")
    assert resolve_checkpoint(tmp_path, "trained_on_visa_crane", 5).name == "epoch_5.pth"
    assert resolve_checkpoint(tmp_path, "trained_on_visa_crane", 1).name == "epoch_1.pth"


def test_resolve_checkpoint_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "custom.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(tmp_path, str(path)) == path.resolve()


def test_resolve_checkpoint_lists_what_is_available(tmp_path):
    (tmp_path / "checkpoints" / "trained_on_visa_crane").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="trained_on_visa_crane"):
        resolve_checkpoint(tmp_path, "trained_on_visa_cranep", 5)


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("crane", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L/14@336px"):
        create_adapter("crane", **common, backbone="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("crane", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("crane", repository=str(tmp_path), target_dataset="mvtec")
