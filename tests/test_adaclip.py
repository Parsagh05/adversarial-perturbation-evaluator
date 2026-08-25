import hashlib

import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.adaclip import (
    CHECKPOINTS,
    ZERO_SHOT_CHECKPOINT,
    resolve_checkpoint,
)


def test_adapter_is_registered_under_both_spellings():
    assert "adaclip" in adapter_names()
    assert "ada-clip" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    # test.sh evaluates MVTec with the VisA-trained weights and vice versa.
    assert ZERO_SHOT_CHECKPOINT == {"mvtec": "visa_clinicdb", "visa": "mvtec_colondb"}
    for target, name in ZERO_SHOT_CHECKPOINT.items():
        assert target not in name


def test_released_checkpoints_carry_pinned_digests():
    assert set(CHECKPOINTS) == {"mvtec_colondb", "visa_clinicdb", "all"}
    for filename, digest in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_resolve_checkpoint_accepts_an_existing_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(path) == path.resolve()


def test_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        resolve_checkpoint(tmp_path / "missing.pth")


def test_resolve_checkpoint_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, _ = CHECKPOINTS["all"]
    payload = b"cached-adaclip-weights"
    cached = tmp_path / filename
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(CHECKPOINTS, "all", (filename, digest))

    def fail(*args, **kwargs):
        raise AssertionError("a verified cache must not be downloaded again")

    monkeypatch.setattr("urllib.request.urlretrieve", fail)
    assert resolve_checkpoint("all", download_root=tmp_path) == cached


def test_resolve_checkpoint_rejects_a_corrupt_download(tmp_path, monkeypatch):
    filename, _ = CHECKPOINTS["all"]
    monkeypatch.setitem(CHECKPOINTS, "all", (filename, "0" * 64))

    def write_wrong_bytes(url, destination):
        del url
        with open(destination, "wb") as handle:
            handle.write(b"corrupt")

    monkeypatch.setattr("urllib.request.urlretrieve", write_wrong_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint("all", download_root=tmp_path)
    assert not (tmp_path / filename).exists()


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec", "device": "cuda"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("adaclip", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L-14-336"):
        create_adapter("adaclip", **common, backbone="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter(
            "adaclip", repository=str(tmp_path), target_dataset="btad", device="cuda"
        )


def test_adapter_requires_cuda(tmp_path):
    with pytest.raises(ValueError, match="requires a CUDA device"):
        create_adapter(
            "adaclip",
            repository=str(tmp_path),
            target_dataset="mvtec",
            device="cpu",
        )


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter(
            "adaclip",
            repository=str(tmp_path),
            target_dataset="mvtec",
            device="cuda",
        )
