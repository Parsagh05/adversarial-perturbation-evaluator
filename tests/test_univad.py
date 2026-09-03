import hashlib
import os
from pathlib import Path

import pytest

from fpeval.adapters import adapter_names
from fpeval.adapters import univad


def test_univad_is_registered():
    for name in ("univad", "uni-vad"):
        assert name in adapter_names()


def test_univad_pins_the_official_defaults():
    # test.sh runs every dataset at --image_size 448.
    assert univad.MODEL_IMAGE_SIZE == 448
    # UniVAD.__init__ hardcodes these; they are not command-line options.
    assert univad.CLIP_MODEL == "ViT-L-14-336"
    assert univad.CLIP_PRETRAINED == "openai"
    assert univad.OUT_LAYERS == (6, 12, 18, 24)
    assert univad.DINOV2_MODEL == "dinov2_vitg14"


def test_univad_pins_both_segmentation_checkpoints():
    # The two files the README's wget lines fetch into pretrained_ckpts/.
    assert set(univad.CHECKPOINTS) == {
        "groundingdino_swint_ogc.pth",
        "sam_hq_vit_h.pth",
    }
    for url, digest in univad.CHECKPOINTS.values():
        assert url.startswith("https://")
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_univad_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="recurse-submodules"):
        univad._import_official_repository(tmp_path)


def test_univad_rejects_an_unknown_target(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        univad.UniVADAdapter(repository=str(tmp_path), target_dataset="mvtec_loco")


def test_univad_rejects_a_negative_round(tmp_path):
    with pytest.raises(ValueError, match="offset"):
        univad.UniVADAdapter(
            repository=str(tmp_path), target_dataset="mvtec", round_index=-1
        )


def test_working_directory_restores_the_previous_one(tmp_path):
    before = Path.cwd()
    with univad._working_directory(tmp_path):
        assert Path.cwd().resolve() == tmp_path.resolve()
    assert Path.cwd() == before


def test_working_directory_restores_after_a_failure(tmp_path):
    before = Path.cwd()
    with pytest.raises(RuntimeError):
        with univad._working_directory(tmp_path):
            raise RuntimeError("boom")
    assert Path.cwd() == before


def test_fetch_pretrained_accepts_a_local_source(tmp_path):
    source = tmp_path / "supplied"
    source.mkdir()
    repository = tmp_path / "UniVAD"
    payloads = {}
    for filename, (_, _) in univad.CHECKPOINTS.items():
        payload = filename.encode()
        (source / filename).write_bytes(payload)
        payloads[filename] = hashlib.sha256(payload).hexdigest()
    # Point the pinned digests at the stand-in payloads for this test only.
    patched = {
        filename: (url, payloads[filename])
        for filename, (url, _) in univad.CHECKPOINTS.items()
    }
    original = univad.CHECKPOINTS.copy()
    univad.CHECKPOINTS.clear()
    univad.CHECKPOINTS.update(patched)
    try:
        resolved = univad.fetch_pretrained(repository, source=source)
        assert set(resolved) == set(patched)
        for filename, path in resolved.items():
            assert path.parent.name == "pretrained_ckpts"
            assert path.read_bytes() == filename.encode()
        # A supplied file that does not match its pin must be refused.
        (source / "sam_hq_vit_h.pth").write_bytes(b"tampered")
        (repository / "pretrained_ckpts" / "sam_hq_vit_h.pth").unlink()
        with pytest.raises(ValueError, match="checksum mismatch"):
            univad.fetch_pretrained(repository, source=source)
    finally:
        univad.CHECKPOINTS.clear()
        univad.CHECKPOINTS.update(original)


def test_fetch_pretrained_reports_a_missing_supplied_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        univad.fetch_pretrained(tmp_path / "UniVAD", source=tmp_path / "empty")
