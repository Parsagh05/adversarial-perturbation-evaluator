import hashlib

import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters import cops, mrad, winclip


def test_all_three_are_registered():
    for name in ("cops", "mrad", "winclip", "win-clip"):
        assert name in adapter_names()


def test_zero_shot_pairings_never_train_on_the_evaluated_dataset():
    # CoPS ships one epoch per training set; test.sh lower-cases the name.
    assert cops.ZERO_SHOT_CHECKPOINT == {"mvtec": ("visa", 10), "visa": ("mvtec", 5)}
    # MRAD retrieves from the memory bank built on the other dataset.
    assert mrad.ZERO_SHOT_MEMORY == {"mvtec": "visa", "visa": "mvtec"}
    for target, (source, _) in cops.ZERO_SHOT_CHECKPOINT.items():
        assert target != source
    for target, source in mrad.ZERO_SHOT_MEMORY.items():
        assert target != source


def test_cops_weights_follow_the_test_dataset_not_the_trained_one():
    # test.py branches on whether "visa" appears in args.dataset, the TEST set.
    assert cops.SCORE_WEIGHTS["visa"] == (0.35, 1.0)
    assert cops.SCORE_WEIGHTS["mvtec"] == (0.26, 0.9)
    assert cops.NEIGHBOUR_KERNEL == {"visa": 3, "mvtec": 5}


def test_cops_resolves_in_repo_weights(tmp_path):
    folder = tmp_path / "results" / "models" / "visa"
    folder.mkdir(parents=True)
    (folder / "epoch_10.pth").write_bytes(b"weights")
    assert cops.resolve_checkpoint(tmp_path, "visa", 10).name == "epoch_10.pth"
    with pytest.raises(FileNotFoundError, match="Cloning the repository"):
        cops.resolve_checkpoint(tmp_path, "mvtec", 5)


def test_mrad_pins_every_released_file():
    assert set(mrad.CHECKPOINTS) == {"test_on_mvtec", "test_on_visa"}
    assert set(mrad.MEMORY_BANKS) == {
        "cache_model_mvtec", "cache_model_visa",
        "cache_patch_model_mvtec", "cache_patch_model_visa",
    }
    for table in (mrad.CHECKPOINTS, mrad.MEMORY_BANKS):
        for filename, file_id, digest in table.values():
            assert filename.endswith((".pth", ".pt"))
            assert len(file_id) > 20
            assert len(digest) == 64 and int(digest, 16) >= 0


def test_mrad_memory_banks_come_in_image_and_patch_pairs(tmp_path, monkeypatch):
    payload = b"cached-bank"
    digest = hashlib.sha256(payload).hexdigest()
    for key in ("cache_model_visa", "cache_patch_model_visa"):
        filename = mrad.MEMORY_BANKS[key][0]
        (tmp_path / filename).write_bytes(payload)
        monkeypatch.setitem(
            mrad.MEMORY_BANKS, key, (filename, mrad.MEMORY_BANKS[key][1], digest)
        )
    image_bank, patch_bank = mrad.resolve_memory_banks("visa", download_root=tmp_path)
    assert image_bank.name == "cache_model_visa.pt"
    assert patch_bank.name == "cache_patch_model_visa.pt"


def test_mrad_rejects_an_unknown_memory_bank():
    with pytest.raises(KeyError, match="no released memory bank"):
        mrad.resolve_memory_banks("btad")


def test_mrad_resolve_checkpoint_accepts_a_path(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"weights")
    assert mrad.resolve_checkpoint(path) == path.resolve()


def test_mrad_resolve_checkpoint_rejects_an_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pass a path or one of"):
        mrad.resolve_checkpoint(tmp_path / "missing.pth")


def test_mrad_rejects_a_corrupt_download(tmp_path, monkeypatch):
    filename, file_id, _ = mrad.CHECKPOINTS["test_on_mvtec"]
    monkeypatch.setitem(mrad.CHECKPOINTS, "test_on_mvtec", (filename, file_id, "0" * 64))
    gdown = pytest.importorskip("gdown")

    def write_wrong_bytes(*, id, output, quiet=True):
        del id, quiet
        with open(output, "wb") as handle:
            handle.write(b"corrupt")

    monkeypatch.setattr(gdown, "download", write_wrong_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        mrad.resolve_checkpoint("test_on_mvtec", download_root=tmp_path)
    assert not (tmp_path / filename).exists()


def test_winclip_needs_no_checkpoint():
    # WinCLIP is training-free, so the module exposes no weight table at all.
    assert not hasattr(winclip, "CHECKPOINTS")
    assert not hasattr(winclip, "ZERO_SHOT_CHECKPOINT")


def test_adapters_reject_non_official_settings(tmp_path):
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("cops", repository=str(tmp_path), target_dataset="mvtec",
                       image_size=336)
    with pytest.raises(ValueError, match=r"ViT-L/14@336px"):
        create_adapter("cops", repository=str(tmp_path), target_dataset="mvtec",
                       clip_model="ViT-B-16")
    with pytest.raises(ValueError, match="image_size=518"):
        create_adapter("mrad", repository=str(tmp_path), target_dataset="mvtec",
                       image_size=336)
    with pytest.raises(ValueError, match="model_type must be one of"):
        create_adapter("mrad", repository=str(tmp_path), target_dataset="mvtec",
                       model_type="mrad-xl")
    with pytest.raises(ValueError, match="ViT-B-16-plus-240"):
        create_adapter("winclip", repository=str(tmp_path), target_dataset="mvtec",
                       backbone="ViT-L-14-336")
    for name in ("cops", "mrad", "winclip"):
        with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
            create_adapter(name, repository=str(tmp_path), target_dataset="btad")


def test_adapters_report_an_incomplete_repository(tmp_path):
    for name in ("cops", "mrad", "winclip"):
        with pytest.raises(FileNotFoundError, match="repository is incomplete"):
            create_adapter(name, repository=str(tmp_path), target_dataset="mvtec")
