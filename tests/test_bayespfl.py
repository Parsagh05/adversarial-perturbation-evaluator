import hashlib

import numpy as np
import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.bayespfl import (
    CHECKPOINTS,
    CLIP_BACKBONE_SHA256,
    DEFAULT_TOP_PIXELS,
    FINE_GRAINED_CATEGORIES,
    FINE_GRAINED_TOP_PIXELS,
    ZERO_SHOT_CHECKPOINT,
    BayesPFLAdapter,
    resolve_checkpoint,
    top_pixels_for,
)


def test_adapter_is_registered_under_both_spellings():
    assert "bayespfl" in adapter_names()
    assert "bayes-pfl" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    assert ZERO_SHOT_CHECKPOINT == {"mvtec": "train_visa", "visa": "train_mvtec"}
    for target, name in ZERO_SHOT_CHECKPOINT.items():
        assert target not in name


def test_released_checkpoints_carry_pinned_ids_and_digests():
    assert set(CHECKPOINTS) == {"train_visa", "train_mvtec"}
    for filename, file_id, digest in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert len(CLIP_BACKBONE_SHA256) == 64


def test_top_pixels_follow_the_official_category_split():
    # calcuate_metric_pixel uses can_k = -20 for these, -2000 for the rest.
    assert FINE_GRAINED_CATEGORIES == {
        "capsules", "macaroni1", "macaroni2", "pipe_fryum",
        "screw", "cashew", "chewinggum",
    }
    assert top_pixels_for("screw") == FINE_GRAINED_TOP_PIXELS
    assert top_pixels_for("bottle") == DEFAULT_TOP_PIXELS
    # The evaluator hands over directory names, which keep the underscore.
    assert top_pixels_for("pipe_fryum") == FINE_GRAINED_TOP_PIXELS
    assert top_pixels_for("pipe fryum") == DEFAULT_TOP_PIXELS


def test_resolve_checkpoint_accepts_an_existing_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(path) == path.resolve()


def test_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        resolve_checkpoint(tmp_path / "missing.pth")


def test_resolve_checkpoint_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, file_id, _ = CHECKPOINTS["train_visa"]
    payload = b"cached-bayespfl-weights"
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


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("bayespfl", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L-14-336"):
        create_adapter("bayespfl", **common, backbone="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("bayespfl", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("bayespfl", repository=str(tmp_path), target_dataset="mvtec")


def _adapter_without_weights(alpha=0.5):
    adapter = BayesPFLAdapter.__new__(BayesPFLAdapter)
    adapter.alpha = alpha
    return adapter


def _maps(peaks):
    """One 64x64 map per peak value, so the top-k mean tracks the peak."""
    return np.stack([np.full((64, 64), peak, dtype=np.float32) for peak in peaks])


def test_score_fusion_averages_the_two_normalized_parts():
    adapter = _adapter_without_weights()
    categories = ["bottle", "bottle", "bottle"]
    scores = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    maps = _maps([1.0, 3.0, 5.0])
    fused = adapter.postprocess_image_scores(scores, None, None, categories, maps=maps)
    # Both parts are min-max normalized to [0, 1] and averaged at 0.5/0.5.
    assert fused[0] == pytest.approx(0.0, abs=1e-6)
    assert fused[2] == pytest.approx(1.0, abs=1e-6)
    assert fused[1] == pytest.approx(0.5, abs=1e-6)


def test_score_fusion_normalizes_each_category_independently():
    adapter = _adapter_without_weights()
    categories = ["bottle", "bottle", "cable", "cable"]
    scores = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    maps = _maps([1.0, 2.0, 10.0, 20.0])
    fused = adapter.postprocess_image_scores(scores, None, None, categories, maps=maps)
    assert fused[0] == pytest.approx(0.0, abs=1e-6)
    assert fused[1] == pytest.approx(1.0, abs=1e-6)
    assert fused[2] == pytest.approx(0.0, abs=1e-6)
    assert fused[3] == pytest.approx(1.0, abs=1e-6)


def test_reference_fusion_freezes_the_clean_range():
    adapter = _adapter_without_weights()
    categories = ["bottle", "bottle"]
    reference_scores = np.array([0.0, 1.0], dtype=np.float32)
    reference_maps = _maps([1.0, 2.0])
    # Both parts sit above the clean maximum, so the fused score exceeds 1.
    fused = adapter.postprocess_image_scores_with_reference(
        np.array([0.0, 2.0], dtype=np.float32),
        None,
        None,
        categories,
        reference_scores=reference_scores,
        reference_map_mins=None,
        reference_map_maxs=None,
        reference_categories=categories,
        maps=_maps([1.0, 3.0]),
        reference_maps=reference_maps,
    )
    assert fused[0] == pytest.approx(0.0, abs=1e-6)
    assert fused[1] == pytest.approx(2.0, abs=1e-6)


def test_score_fusion_requires_the_cohort_maps():
    adapter = _adapter_without_weights()
    with pytest.raises(ValueError, match="needs the cohort maps"):
        adapter.postprocess_image_scores(
            np.array([1.0], dtype=np.float32), None, None, ["bottle"], maps=None
        )
