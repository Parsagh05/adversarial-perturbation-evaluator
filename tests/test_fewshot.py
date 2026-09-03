import numpy as np
import pytest

from fpeval.adapters import adapter_names, create_adapter
from fpeval.adapters.aprilgan import AprilGANFewShotAdapter
from fpeval.adapters.winclip import (
    EXPERIMENT_SEEDS,
    SEED_FILE,
    SHOT_VALUES,
    read_seed_selection,
)
from fpeval.data import discover_normal_reference


def test_few_shot_adapters_are_registered():
    for name in (
        "winclip_fewshot", "winclip-fewshot",
        "afclip_fewshot", "af-clip-fewshot",
        "aprilgan_fewshot", "april-gan-fewshot",
    ):
        assert name in adapter_names()


def test_winclip_shot_values_match_the_official_assert():
    # load_mvtec asserts k_shot in [0, 1, 5, 10] and experiment_indx in [0, 1, 2].
    assert SHOT_VALUES == (1, 5, 10)
    assert EXPERIMENT_SEEDS == (111, 333, 999)


def test_winclip_reads_the_committed_selection(tmp_path):
    folder = tmp_path / "datasets" / "seeds_mvtec" / "bottle"
    folder.mkdir(parents=True)
    (folder / "selected_samples_per_run.txt").write_text(
        "0-1: 053\n0-5: 013 048 053 142 161\n1-1: 081\n", encoding="utf-8"
    )
    assert read_seed_selection(tmp_path, "bottle", 1, 0) == ["053"]
    assert read_seed_selection(tmp_path, "bottle", 5, 0) == [
        "013", "048", "053", "142", "161"
    ]
    assert read_seed_selection(tmp_path, "bottle", 1, 1) == ["081"]
    with pytest.raises(ValueError, match="no entry for 0-10"):
        read_seed_selection(tmp_path, "bottle", 10, 0)


def test_winclip_reports_a_missing_seed_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="ships with the repository"):
        read_seed_selection(tmp_path, "bottle", 1, 0)


def test_seed_file_template_matches_the_official_layout():
    assert SEED_FILE == "datasets/seeds_mvtec/{category}/selected_samples_per_run.txt"


def test_few_shot_adapters_reject_bad_shot_counts(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match=r"k_shot in \(1, 5, 10\)"):
        create_adapter("winclip_fewshot", **common, k_shot=4)
    with pytest.raises(ValueError, match="experiment_indx must be 0, 1 or 2"):
        create_adapter("winclip_fewshot", **common, k_shot=1, experiment_indx=3)
    with pytest.raises(ValueError, match="needs k_shot >= 1"):
        create_adapter("afclip_fewshot", **common, k_shot=0)
    with pytest.raises(ValueError, match="needs k_shot >= 1"):
        create_adapter("aprilgan_fewshot", **common, k_shot=0)


def test_few_shot_adapters_still_validate_the_repository(tmp_path):
    for name in ("winclip_fewshot", "afclip_fewshot", "aprilgan_fewshot"):
        with pytest.raises(FileNotFoundError, match="repository is incomplete"):
            create_adapter(name, repository=str(tmp_path), target_dataset="mvtec")


def _mvtec_tree(root):
    for category, count in (("bottle", 4), ("cable", 3)):
        folder = root / category / "train" / "good"
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"{index:03d}.png").write_bytes(b"x")
        (root / category / "test").mkdir()


def test_normal_reference_discovery_groups_and_sorts(tmp_path):
    _mvtec_tree(tmp_path)
    grouped = discover_normal_reference("mvtec", mvtec_root=tmp_path, visa_root=None)
    assert sorted(grouped) == ["bottle", "cable"]
    assert [s.image_path.name for s in grouped["bottle"]] == [
        "000.png", "001.png", "002.png", "003.png"
    ]
    # References come from the train split, never from the evaluated cohort.
    assert {s.split for samples in grouped.values() for s in samples} == {"train"}
    assert {s.label for samples in grouped.values() for s in samples} == {0}


def test_normal_reference_rejects_a_missing_root(tmp_path):
    with pytest.raises(ValueError, match="No MVTec training images"):
        discover_normal_reference("mvtec", mvtec_root=tmp_path, visa_root=None)


def _aprilgan_without_weights():
    adapter = AprilGANFewShotAdapter.__new__(AprilGANFewShotAdapter)
    return adapter


def test_aprilgan_few_shot_score_blends_text_and_map_peak():
    adapter = _aprilgan_without_weights()
    categories = ["bottle", "bottle", "bottle"]
    scores = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    peaks = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    fused = adapter.postprocess_image_scores(scores, None, peaks, categories)
    # 0.5 * (text + min-max normalized peak)
    assert fused[0] == pytest.approx(0.0, abs=1e-6)
    assert fused[1] == pytest.approx(0.5, abs=1e-6)
    assert fused[2] == pytest.approx(1.0, abs=1e-6)


def test_aprilgan_few_shot_score_freezes_the_clean_peak_range():
    adapter = _aprilgan_without_weights()
    categories = ["bottle", "bottle"]
    fused = adapter.postprocess_image_scores_with_reference(
        np.array([0.0, 1.0], dtype=np.float32),
        None,
        np.array([1.0, 3.0], dtype=np.float32),
        categories,
        reference_scores=np.array([0.0, 1.0], dtype=np.float32),
        reference_map_mins=None,
        reference_map_maxs=np.array([1.0, 2.0], dtype=np.float32),
        reference_categories=categories,
    )
    # The adversarial peak of 3 sits above the clean maximum of 2, so the
    # normalized part exceeds 1 instead of being squashed back into range.
    assert fused[0] == pytest.approx(0.0, abs=1e-6)
    assert fused[1] == pytest.approx(0.5 * (1.0 + 2.0), abs=1e-6)


def test_subspacead_and_inpformer_are_registered():
    for name in ("subspacead", "subspace-ad", "inpformer", "inp-former"):
        assert name in adapter_names()


def test_subspacead_follows_the_benchmark_script():
    from fpeval.adapters.subspacead import LAYERS, NO_AUG_CATEGORIES, SHOT_VALUES

    assert LAYERS == (-12, -13, -14, -15, -16, -17, -18)
    assert SHOT_VALUES == (1, 2, 4)
    # main.py hardcodes transistor as the no-augmentation category.
    assert NO_AUG_CATEGORIES == {"transistor"}


def test_inpformer_releases_one_model_per_dataset_and_shot():
    from fpeval.adapters.inpformer import CHECKPOINTS, SHOT_VALUES

    assert SHOT_VALUES == (1, 2, 4)
    assert set(CHECKPOINTS) == {
        (dataset, shot) for dataset in ("mvtec", "visa") for shot in SHOT_VALUES
    }
    for filename, file_id in CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20


def test_inpformer_rejects_an_unreleased_shot_count(tmp_path):
    from fpeval.adapters.inpformer import resolve_checkpoint

    with pytest.raises(KeyError, match="no released few-shot model"):
        resolve_checkpoint("mvtec", 8, download_root=tmp_path)
    with pytest.raises(KeyError, match="no released few-shot model"):
        resolve_checkpoint("btad", 4, download_root=tmp_path)


def test_inpformer_accepts_an_explicit_checkpoint(tmp_path):
    from fpeval.adapters.inpformer import resolve_checkpoint

    path = tmp_path / "model.pth"
    path.write_bytes(b"weights")
    assert resolve_checkpoint("mvtec", 4, checkpoint=str(path)) == path.resolve()


def test_new_few_shot_adapters_reject_bad_settings(tmp_path):
    common = {"repository": str(tmp_path), "target_dataset": "mvtec"}
    with pytest.raises(ValueError, match=r"k_shot in \(1, 2, 4\)"):
        create_adapter("subspacead", **common, k_shot=3)
    with pytest.raises(ValueError, match="image_res=672"):
        create_adapter("subspacead", **common, image_res=518)
    with pytest.raises(ValueError, match=r"few-shot models for \(1, 2, 4\)"):
        create_adapter("inpformer", **common, shot=8)
    with pytest.raises(ValueError, match="input_size=448 and crop_size=392"):
        create_adapter("inpformer", **common, input_size=518, crop_size=518)
    for name in ("subspacead", "inpformer"):
        with pytest.raises(ValueError, match="must be 'mvtec' or 'visa'"):
            create_adapter(name, repository=str(tmp_path), target_dataset="btad")


def test_new_few_shot_adapters_report_an_incomplete_repository(tmp_path):
    for name in ("subspacead", "inpformer"):
        with pytest.raises(FileNotFoundError, match="repository is incomplete"):
            create_adapter(name, repository=str(tmp_path), target_dataset="mvtec")
