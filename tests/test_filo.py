import ast
import hashlib
import os
from pathlib import Path

import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.filo import (
    ANOMALY_DETAIL,
    ANOMALY_STATUS_GENERAL,
    CHECKPOINTS,
    MVTEC_ANOMALY_DETAIL,
    POSITIONS,
    VISA_ANOMALY_DETAIL,
    ZERO_SHOT_CHECKPOINT,
    _phrase_matches,
    resolve_checkpoint,
)


def test_adapter_is_registered():
    assert "filo" in adapter_names()


def test_zero_shot_checkpoints_never_train_on_the_evaluated_dataset():
    # test.sh pairs both the FiLo head and its Grounding DINO across datasets.
    assert ZERO_SHOT_CHECKPOINT == {"mvtec": "train_on_visa", "visa": "train_on_mvtec"}
    for target, suffix in ZERO_SHOT_CHECKPOINT.items():
        assert target not in suffix
        assert f"filo_{suffix}" in CHECKPOINTS
        assert f"grounding_{suffix}" in CHECKPOINTS


def test_released_checkpoints_carry_pinned_digests():
    assert len(CHECKPOINTS) == 4
    for filename, digest in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_position_grid_tiles_the_official_image():
    assert list(POSITIONS) == [
        "top left", "top", "top right",
        "left", "center", "right",
        "bottom left", "bottom", "bottom right",
    ]
    # PromptLearner_abnormal slices its prompts with [position_idx::9].
    assert len(POSITIONS) == 9
    assert POSITIONS["top left"][0] == (0, 0)
    assert POSITIONS["bottom right"][1] == (517, 517)


def test_descriptions_cover_every_category_of_each_dataset():
    assert len(ANOMALY_DETAIL["mvtec"]) == 15
    assert len(ANOMALY_DETAIL["visa"]) == 12
    # The datasets hand over spaced names, so no key may carry an underscore.
    for table in ANOMALY_DETAIL.values():
        for name, descriptions in table.items():
            assert "_" not in name
            assert descriptions and all(descriptions)


def test_phrase_matching_reproduces_check_elements_in_array():
    assert _phrase_matches(["cracks", "corrosion"], "cracks(0.42)")
    assert _phrase_matches(ANOMALY_STATUS_GENERAL, "defect(0.31)")
    assert not _phrase_matches(["cracks"], "wood(0.55)")


def test_resolve_checkpoint_accepts_an_existing_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint(path) == path.resolve()


def test_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        resolve_checkpoint(tmp_path / "missing.pth")


def test_resolve_checkpoint_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, _ = CHECKPOINTS["filo_train_on_visa"]
    payload = b"cached-filo-weights"
    cached = tmp_path / filename
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(CHECKPOINTS, "filo_train_on_visa", (filename, digest))
    assert resolve_checkpoint("filo_train_on_visa", download_root=tmp_path) == cached


def test_resolve_checkpoint_rejects_a_corrupt_download(tmp_path, monkeypatch):
    import fpeval.adapters.filo as module

    filename, _ = CHECKPOINTS["filo_train_on_visa"]
    monkeypatch.setitem(CHECKPOINTS, "filo_train_on_visa", (filename, "0" * 64))
    hub = pytest.importorskip("huggingface_hub")

    def write_wrong_bytes(*, repo_id, filename, local_dir):
        del repo_id
        path = Path(local_dir) / filename
        path.write_bytes(b"corrupt")
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", write_wrong_bytes)
    monkeypatch.setattr(module, "hf_hub_download", write_wrong_bytes, raising=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint("filo_train_on_visa", download_root=tmp_path)


def test_adapter_rejects_non_official_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("filo", **common, image_size=336)
    with pytest.raises(ValueError, match="ViT-L-14-336"):
        create_adapter("filo", **common, clip_model="ViT-B-16")
    with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
        create_adapter("filo", repository=str(tmp_path), target_dataset="btad")


def test_adapter_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository is incomplete"):
        create_adapter("filo", repository=str(tmp_path), target_dataset="mvtec")


def _official_tables(source: str) -> dict[str, dict[str, str]]:
    """Read the dict literals out of the official test.py without importing it."""
    wanted = {
        "mvtec_anomaly_detail_gpt",
        "visa_anomaly_detail_gpt",
        "anomaly_status_general",
    }
    found = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found.setdefault(target.id, ast.literal_eval(node.value))
    return found


def test_description_tables_match_the_official_test_py():
    """The GPT descriptions are paper content; drift would change the captions."""
    repository = os.environ.get("FPEVAL_FILO_REPOSITORY")
    if not repository:
        pytest.skip("set FPEVAL_FILO_REPOSITORY to check against the official repo")
    source = (Path(repository) / "test.py").read_text(encoding="utf-8")
    official = _official_tables(source)
    assert official["mvtec_anomaly_detail_gpt"] == MVTEC_ANOMALY_DETAIL
    assert official["visa_anomaly_detail_gpt"] == VISA_ANOMALY_DETAIL
    assert official["anomaly_status_general"] == ANOMALY_STATUS_GENERAL
