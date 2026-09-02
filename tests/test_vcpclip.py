import hashlib

import numpy as np
import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.vcpclip import (
    CHECKPOINTS,
    CLIP_BACKBONE_SHA256,
    CLIP_BACKBONE_URL,
    ZERO_SHOT_CHECKPOINT,
    VCPCLIPAdapter,
    resolve_checkpoint,
    resolve_clip_backbone,
)


def test_adapter_is_registered_under_both_spellings():
    assert "vcpclip" in adapter_names()
    assert "vcp-clip" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    # test.sh evaluates MVTec with train_visa.pth and VisA with train_mvtec.pth.
    assert ZERO_SHOT_CHECKPOINT == {"mvtec": "train_visa", "visa": "train_mvtec"}
    for target, name in ZERO_SHOT_CHECKPOINT.items():
        assert target not in name


def test_released_checkpoints_carry_pinned_ids_and_digests():
    assert set(CHECKPOINTS) == {"train_visa", "train_mvtec"}
    for filename, file_id, digest in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_clip_backbone_url_carries_its_own_digest():
    # OpenAI serves the file under a path named by its sha256.
    assert len(CLIP_BACKBONE_SHA256) == 64
    assert CLIP_BACKBONE_SHA256 in CLIP_BACKBONE_URL
    assert CLIP_BACKBONE_URL.endswith("ViT-L-14-336px.pt")


def test_resolve_checkpoint_accepts_an_existing_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(path) == path.resolve()


def test_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        resolve_checkpoint(tmp_path / "missing.pth")


def test_resolve_checkpoint_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, file_id, _ = CHECKPOINTS["train_visa"]
    payload = b"cached-vcp-weights"
    cached = tmp_path / filename
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(CHECKPOINTS, "train_visa", (filename, file_id, digest))
    assert resolve_checkpoint("train_visa", download_root=tmp_path) == cached


def test_resolve_checkpoint_rejects_a_corrupt_download(tmp_path, monkeypatch):
    filename, file_id, _ = CHECKPOINTS["train_visa"]
    monkeypatch.setitem(CHECKPOINTS, "train_visa", (filename, file_id, "0" * 64))
    gdown = pytest.importorskip("gdown")

    def write_wrong_bytes(*, id, output, quiet=True):
        del id, quiet
        with open(output, "wb") as handle:
            handle.write(b"corrupt")

    monkeypatch.setattr(gdown, "download", write_wrong_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint("train_visa", download_root=tmp_path)
    assert not (tmp_path / filename).exists()


def test_clip_backbone_reuses_a_verified_cache_without_downloading(
    tmp_path, monkeypatch
):
    import fpeval.adapters.vcpclip as module

    payload = b"cached-backbone"
    (tmp_path / "ViT-L-14-336px.pt").write_bytes(payload)
    monkeypatch.setattr(
        module, "CLIP_BACKBONE_SHA256", hashlib.sha256(payload).hexdigest()
    )

    def fail(*args, **kwargs):
        raise AssertionError("a verified cache must not be re-downloaded")

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fail)
    assert resolve_clip_backbone(tmp_path) == tmp_path / "ViT-L-14-336px.pt"


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("vcpclip", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L-14-336"):
        create_adapter("vcpclip", **common, backbone="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("vcpclip", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("vcpclip", repository=str(tmp_path), target_dataset="mvtec")


def _adapter_without_weights():
    return VCPCLIPAdapter.__new__(VCPCLIPAdapter)


def test_postprocess_normalizes_each_category_independently():
    adapter = _adapter_without_weights()
    categories = ["bottle", "bottle", "cable"]
    scores = np.array([1.0, 3.0, 7.0], dtype=np.float32)
    maps = np.stack(
        [
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 6.0, dtype=np.float32),
            np.full((2, 2), 9.0, dtype=np.float32),
        ]
    )
    normalized = adapter.postprocess_image_scores(scores, None, None, categories)
    assert normalized.tolist() == [0.0, 1.0, 0.0]
    normalized_maps = adapter.postprocess_anomaly_maps(maps, categories)
    assert normalized_maps[0].max() == 0.0
    assert normalized_maps[1].min() == 1.0
    # A single-image category has zero spread and must not divide by zero.
    assert normalized_maps[2].tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_reference_postprocess_freezes_the_clean_range():
    adapter = _adapter_without_weights()
    categories = ["bottle", "bottle"]
    reference = np.array([1.0, 3.0], dtype=np.float32)
    # An adversarial score above the clean maximum must exceed 1, not be
    # re-normalized back into [0, 1].
    adversarial = np.array([1.0, 5.0], dtype=np.float32)
    frozen = adapter.postprocess_image_scores_with_reference(
        adversarial,
        None,
        None,
        categories,
        reference_scores=reference,
        reference_map_mins=None,
        reference_map_maxs=None,
        reference_categories=categories,
    )
    assert frozen.tolist() == [0.0, 2.0]


def test_reference_postprocess_leaves_unmatched_categories_untouched():
    adapter = _adapter_without_weights()
    scores = np.array([4.0], dtype=np.float32)
    frozen = adapter.postprocess_image_scores_with_reference(
        scores,
        None,
        None,
        ["screw"],
        reference_scores=np.array([1.0, 3.0], dtype=np.float32),
        reference_map_mins=None,
        reference_map_maxs=None,
        reference_categories=["bottle", "bottle"],
    )
    assert frozen.tolist() == [4.0]
