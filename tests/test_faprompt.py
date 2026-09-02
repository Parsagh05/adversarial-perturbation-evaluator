import hashlib

import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.faprompt import (
    CHECKPOINTS,
    ZERO_SHOT_CHECKPOINT,
    resolve_checkpoint,
)


def test_adapter_is_registered_under_both_spellings():
    assert "faprompt" in adapter_names()
    assert "fa-prompt" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    assert ZERO_SHOT_CHECKPOINT == {
        "mvtec": "train_on_visa",
        "visa": "train_on_mvtecad",
    }
    for target, name in ZERO_SHOT_CHECKPOINT.items():
        assert target not in name


def test_released_checkpoints_carry_pinned_ids_and_digests():
    assert set(CHECKPOINTS) == {"train_on_mvtecad", "train_on_visa"}
    for filename, file_id, digest in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_resolve_checkpoint_accepts_an_existing_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(path) == path.resolve()


def test_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        resolve_checkpoint(tmp_path / "missing.pth")


def test_resolve_checkpoint_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, file_id, _ = CHECKPOINTS["train_on_visa"]
    payload = b"cached-faprompt-weights"
    cached = tmp_path / filename
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(CHECKPOINTS, "train_on_visa", (filename, file_id, digest))
    assert resolve_checkpoint("train_on_visa", download_root=tmp_path) == cached


def test_resolve_checkpoint_rejects_a_corrupt_download(tmp_path, monkeypatch):
    filename, file_id, _ = CHECKPOINTS["train_on_visa"]
    monkeypatch.setitem(CHECKPOINTS, "train_on_visa", (filename, file_id, "0" * 64))
    gdown = pytest.importorskip("gdown")

    def write_wrong_bytes(*, id, output, quiet=True):
        del id, quiet
        with open(output, "wb") as handle:
            handle.write(b"corrupt")

    monkeypatch.setattr(gdown, "download", write_wrong_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint("train_on_visa", download_root=tmp_path)
    assert not (tmp_path / filename).exists()


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("faprompt", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L/14@336px"):
        create_adapter("faprompt", **common, backbone="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("faprompt", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("faprompt", repository=str(tmp_path), target_dataset="mvtec")
