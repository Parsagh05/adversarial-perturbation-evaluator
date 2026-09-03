import hashlib

import pytest

from fpeval.adapters import adapter_names
from fpeval.adapters import fade, inctrl


def test_both_are_registered():
    for name in ("fade", "inctrl", "in-ctrl"):
        assert name in adapter_names()


# ------------------------------------------------------------------- FADE ---
def test_fade_prompt_sets_match_run_fade():
    # run_fade.py classifies with winclip prompts only and segments with those
    # plus the five ChatGPT sets, the other four files staying commented out.
    assert fade.CLASSIFICATION_PROMPTS == ("winclip_prompt.json",)
    assert fade.SEGMENTATION_PROMPTS == (
        "winclip_prompt.json",
        "chatgpt3.5_prompt1.json",
        "chatgpt3.5_prompt2.json",
        "chatgpt3.5_prompt3.json",
        "chatgpt3.5_prompt4.json",
        "chatgpt3.5_prompt5.json",
    )
    # Only the classification prompts take a category name.
    assert fade.SEGMENTATION_CLASSNAME == "object"


def test_fade_scales_and_normalization():
    assert fade.SEGMENTATION_IMG_SIZES == (240, 448, 896)
    # datasets/base.py calls these IMAGENET_* but assigns OpenCLIP's values.
    assert fade.CLIP_MEAN == (0.48145466, 0.4578275, 0.40821073)
    assert fade.CLIP_STD == (0.26862954, 0.26130258, 0.27577711)


def test_fade_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        fade._import_official_repository(tmp_path)


def test_fade_rejects_the_ablation_modes(tmp_path):
    # The paper's few-shot setting is cm_both_sm_both; nothing else is wired up.
    with pytest.raises(ValueError, match="cm_both_sm_both"):
        fade.FADEAdapter(
            repository=str(tmp_path),
            target_dataset="mvtec",
            classification_mode="language",
        )


def test_fade_rejects_an_unknown_target(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        fade.FADEAdapter(repository=str(tmp_path), target_dataset="mvtec_loco")


# ----------------------------------------------------------------- InCTRL ---
def test_inctrl_shot_values_match_the_released_checkpoints():
    # Each model archive holds checkpoints/{2,4,8}/checkpoint.pyth.
    assert inctrl.SHOT_VALUES == (2, 4, 8)


def test_inctrl_never_scores_the_dataset_it_trained_on():
    # It is a generalist detector: train on one dataset, evaluate on the other.
    assert inctrl.AUXILIARY_OF == {"mvtec": "visa", "visa": "mvtec"}
    for target, auxiliary in inctrl.AUXILIARY_OF.items():
        assert target != auxiliary


def test_inctrl_pins_every_released_archive():
    assert set(inctrl.MODEL_ARCHIVES) == {"mvtec", "visa"}
    assert set(inctrl.SAMPLE_ARCHIVES) == {"mvtec", "visa"}
    assert set(inctrl.MODEL_DIGESTS) == set(inctrl.MODEL_ARCHIVES)
    assert set(inctrl.SAMPLE_DIGESTS) == set(inctrl.SAMPLE_ARCHIVES)
    for table in (inctrl.MODEL_ARCHIVES, inctrl.SAMPLE_ARCHIVES):
        for filename, file_id in table.values():
            assert filename.endswith(".zip")
            assert len(file_id) > 20
    for table in (inctrl.MODEL_DIGESTS, inctrl.SAMPLE_DIGESTS):
        for digest in table.values():
            assert len(digest) == 64 and int(digest, 16) >= 0


def test_inctrl_refuses_to_train_and_score_on_one_dataset(tmp_path):
    with pytest.raises(ValueError, match="generalist"):
        inctrl.InCTRLAdapter(
            repository=str(tmp_path), target_dataset="mvtec", train_dataset="mvtec"
        )


def test_inctrl_rejects_an_unreleased_shot(tmp_path):
    with pytest.raises(ValueError, match="shot in"):
        inctrl.InCTRLAdapter(
            repository=str(tmp_path), target_dataset="mvtec", shot=1
        )


def test_inctrl_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        inctrl._import_official_repository(tmp_path)


def test_inctrl_accepts_a_supplied_checkpoint(tmp_path):
    weights = tmp_path / "checkpoint.pyth"
    weights.write_bytes(b"weights")
    assert inctrl.resolve_checkpoint("visa", 2, checkpoint=str(weights)) == weights
    with pytest.raises(FileNotFoundError, match="not found"):
        inctrl.resolve_checkpoint("visa", 2, checkpoint=str(tmp_path / "absent.pyth"))


def test_inctrl_verifies_an_archive_digest(tmp_path, monkeypatch):
    import sys
    import types

    payload = b"archive"
    archive = tmp_path / "trained_on_visa.zip"
    archive.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    # A matching digest short-circuits the download entirely.
    assert inctrl._fetch_archive("trained_on_visa.zip", "id", digest, tmp_path) == archive

    # A mismatch has to re-download, and the re-download has to be checked too.
    stub = types.ModuleType("gdown")
    stub.download = lambda id, output, quiet: open(output, "wb").write(b"wrong")
    monkeypatch.setitem(sys.modules, "gdown", stub)
    with pytest.raises(ValueError, match="checksum mismatch"):
        inctrl._fetch_archive("trained_on_visa.zip", "id", "0" * 64, tmp_path)
    # The bad download must not be left behind as if it were valid.
    assert archive.read_bytes() == payload


def test_inctrl_extracts_the_requested_shot(tmp_path):
    import zipfile

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for shot in (2, 4, 8):
            bundle.writestr(f"checkpoints/{shot}/checkpoint.pyth", f"weights-{shot}")
    destination = tmp_path / "visa_4shot_checkpoint.pyth"
    extracted = inctrl._extract_member(
        archive, "checkpoints/4/checkpoint.pyth", destination
    )
    assert extracted.read_bytes() == b"weights-4"
    with pytest.raises(KeyError, match="no member"):
        inctrl._extract_member(archive, "checkpoints/16/checkpoint.pyth", tmp_path / "x")


def test_inctrl_finds_the_shot_folder_of_sample_prompts(tmp_path):
    folder = tmp_path / "fs_visa" / "visa" / "2"
    folder.mkdir(parents=True)
    (folder / "candle.pt").write_bytes(b"prompt")
    resolved = inctrl.resolve_few_shot_dir(
        "visa", 2, few_shot_dir=str(folder)
    )
    assert resolved == folder


def test_inctrl_records_a_direct_forward_call():
    """The official code calls ``self.diff_head.forward(x)``, not the module.

    register_forward_hook only fires through ``Module.__call__``, so a hook would
    silently never run and the recovered map would be missing entirely.
    """
    import torch

    class Head(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    head = Head()
    inputs, outputs = [], []
    inctrl.record_forward(head, inputs, outputs)
    value = torch.arange(6.0).reshape(2, 3)

    # Called the way InCTRL calls it.
    head.forward(value)
    assert len(inputs) == 1 and torch.equal(inputs[0], value)
    assert torch.equal(outputs[0], value.sum(dim=1, keepdim=True))

    # And still recorded through the module, for good measure.
    head(value)
    assert len(inputs) == 2 and len(outputs) == 2


def test_inctrl_recovers_the_patch_map_from_the_recorded_call():
    """patch_ref_map = holistic - (holistic.max() - fg), with fg = 2*final - hl."""
    import torch

    patch_map = torch.tensor([[0.1, 0.7, 0.3, 0.2]])
    text_score, image_reference = 0.25, 0.4
    holistic = patch_map + text_score + image_reference
    head_score = torch.tensor([[0.9]])
    foreground = patch_map.max(dim=1).values
    final_score = (head_score.reshape(-1) + foreground) / 2

    recovered_foreground = 2.0 * final_score - head_score.reshape(-1)
    offset = holistic.max(dim=1).values - recovered_foreground
    recovered = holistic - offset.unsqueeze(1)
    assert torch.allclose(recovered, patch_map, atol=1e-6)
