import hashlib

import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters import aprilgan, fbclip, tipsomaly


def test_all_three_are_registered():
    for name in ("aprilgan", "april-gan", "fbclip", "fb-clip", "tipsomaly"):
        assert name in adapter_names()


def test_zero_shot_pairings_never_train_on_the_evaluated_dataset():
    assert aprilgan.ZERO_SHOT_CHECKPOINT == {
        "mvtec": "visa_pretrained",
        "visa": "mvtec_pretrained",
    }
    assert fbclip.ZERO_SHOT_CHECKPOINT == {
        "mvtec": "train_on_visa",
        "visa": "train_on_mvtec",
    }
    assert tipsomaly.ZERO_SHOT_CHECKPOINT == {
        "mvtec": "trained_on_visa_default",
        "visa": "trained_on_mvtec_default",
    }
    for module in (aprilgan, fbclip, tipsomaly):
        for target, name in module.ZERO_SHOT_CHECKPOINT.items():
            assert target not in name


def test_fbclip_checkpoints_carry_pinned_ids_and_digests():
    assert set(fbclip.CHECKPOINTS) == {"train_on_mvtec", "train_on_visa"}
    for filename, file_id, digest in fbclip.CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_aprilgan_resolves_in_repo_weights(tmp_path):
    folder = tmp_path / "exps" / "pretrained"
    folder.mkdir(parents=True)
    (folder / "visa_pretrained.pth").write_bytes(b"weights")
    assert aprilgan.resolve_checkpoint(tmp_path, "visa_pretrained").name == "visa_pretrained.pth"
    with pytest.raises(FileNotFoundError, match="visa_pretrained"):
        aprilgan.resolve_checkpoint(tmp_path, "mvtec_pretrained")


def test_tipsomaly_resolves_in_repo_prompts(tmp_path):
    folder = tmp_path / "workspaces" / "trained_on_visa_default" / "vegan-arkansas" / "checkpoints"
    folder.mkdir(parents=True)
    (folder / "learnable_params_2.pth").write_bytes(b"prompts")
    resolved = tipsomaly.resolve_checkpoint(tmp_path, "trained_on_visa_default", 2)
    assert resolved.name == "learnable_params_2.pth"
    with pytest.raises(FileNotFoundError, match="trained_on_visa_default"):
        tipsomaly.resolve_checkpoint(tmp_path, "trained_on_mvtec_default", 2)


def test_fbclip_reuses_a_verified_cache(tmp_path, monkeypatch):
    filename, file_id, _ = fbclip.CHECKPOINTS["train_on_visa"]
    payload = b"cached-fbclip-weights"
    (tmp_path / filename).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(fbclip.CHECKPOINTS, "train_on_visa", (filename, file_id, digest))
    assert fbclip.resolve_checkpoint("train_on_visa", download_root=tmp_path) == tmp_path / filename


def test_fbclip_rejects_a_corrupt_download(tmp_path, monkeypatch):
    filename, file_id, _ = fbclip.CHECKPOINTS["train_on_visa"]
    monkeypatch.setitem(fbclip.CHECKPOINTS, "train_on_visa", (filename, file_id, "0" * 64))
    gdown = pytest.importorskip("gdown")

    def write_wrong_bytes(*, id, output, quiet=True):
        del id, quiet
        with open(output, "wb") as handle:
            handle.write(b"corrupt")

    monkeypatch.setattr(gdown, "download", write_wrong_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        fbclip.resolve_checkpoint("train_on_visa", download_root=tmp_path)


def test_adapters_reject_non_official_settings(tmp_path):
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("aprilgan", repository=str(tmp_path), target_dataset="mvtec",
                       image_size=224)
    with pytest.raises(ValueError, match="ViT-L-14-336"):
        create_adapter("aprilgan", repository=str(tmp_path), target_dataset="mvtec",
                       backbone="ViT-B-16")
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("fbclip", repository=str(tmp_path), target_dataset="mvtec",
                       image_size=336)
    with pytest.raises(ValueError, match="cls_token_index must be 0 or 1"):
        create_adapter("tipsomaly", repository=str(tmp_path), target_dataset="mvtec",
                       models_dir=str(tmp_path), cls_token_index=2)
    for name in ("aprilgan", "fbclip"):
        with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
            create_adapter(name, repository=str(tmp_path), target_dataset="btad")


def test_adapters_report_an_incomplete_repository(tmp_path):
    for name, extra in (
        ("aprilgan", {}),
        ("fbclip", {}),
        ("tipsomaly", {"models_dir": str(tmp_path)}),
    ):
        with pytest.raises(FileNotFoundError, match="repository is incomplete"):
            create_adapter(name, repository=str(tmp_path), target_dataset="mvtec", **extra)
