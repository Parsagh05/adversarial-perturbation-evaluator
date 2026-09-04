import pytest

from fpeval.adapters import adapter_names
from fpeval.adapters import aaclip, promptad
from fpeval.kaggle import download_kaggle_dataset, find_kaggle_files


def test_promptad_is_registered():
    for name in ("promptad", "prompt-ad"):
        assert name in adapter_names()


def test_promptad_checkpoint_naming_matches_get_dir_from_args():
    # utils/training_utils.py builds
    # f"{TASK}-Seed_{seed}-{class_name}-check_point.pt".
    assert promptad.CHECKPOINT_TEMPLATE == "{task}-Seed_{seed}-{category}-check_point.pt"
    rendered = promptad.CHECKPOINT_TEMPLATE.format(
        task="CLS", seed=111, category="bottle"
    )
    assert rendered == "CLS-Seed_111-bottle-check_point.pt"
    assert promptad.TASKS == ("CLS", "SEG")
    assert promptad.CHECKPOINT_SEED == 111
    assert promptad.SHOT_VALUES == (1, 2, 4)


def test_promptad_saves_only_the_three_inference_buffers():
    """save_check_point keeps exactly these keys, which is why the official
    test scripts can load with strict=False."""
    assert promptad.CHECKPOINT_KEYS == (
        "feature_gallery1", "feature_gallery2", "text_features",
    )


def test_promptad_release_covers_every_class_shot_and_task():
    # 15 MVTec + 12 VisA classes, three shots, two tasks.
    assert (15 + 12) * len(promptad.SHOT_VALUES) * len(promptad.TASKS) == 162


def test_promptad_constants():
    assert promptad.MODEL_IMAGE_SIZE == 240
    assert promptad.RESOLUTION == 400
    assert promptad.KAGGLE_DATASET.count("/") == 1


def test_promptad_rejects_unsupported_settings(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        promptad.PromptADAdapter(repository=str(tmp_path), target_dataset="btad")
    with pytest.raises(ValueError, match="k_shot"):
        promptad.PromptADAdapter(
            repository=str(tmp_path), target_dataset="mvtec", k_shot=8
        )


def test_promptad_rejects_an_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task"):
        promptad.resolve_checkpoint(
            "mvtec", 1, "bottle", "BOTH", checkpoint_root=str(tmp_path)
        )
    with pytest.raises(ValueError, match="k_shot"):
        promptad.resolve_checkpoint(
            "mvtec", 3, "bottle", "CLS", checkpoint_root=str(tmp_path)
        )


def test_promptad_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        promptad._import_official_repository(tmp_path)


def _build_release(root, *, nested):
    """The six released configurations, as the Kaggle package lays them out.

    Every wrapper directory holds the same file names, so only the dataset and
    shot count separate them - which is exactly what the resolver has to get
    right. ``nested`` toggles between PromptAD's own
    ``result/<dataset>/k_<n>/checkpoint`` tree and a flattened ``result/``.
    """
    for dataset, categories in (("mvtec", ["bottle", "cable"]), ("visa", ["candle"])):
        for k_shot in promptad.SHOT_VALUES:
            top = root / "PromptAD" / f"promptad_retrained_{dataset}_{k_shot}shot"
            (top / "logs").mkdir(parents=True, exist_ok=True)
            (top / "checkpoint_index.json").write_text("{}", encoding="utf-8")
            base = (
                top / "result" / dataset / f"k_{k_shot}" / "checkpoint"
                if nested
                else top / "result"
            )
            base.mkdir(parents=True, exist_ok=True)
            for category in categories:
                for task in promptad.TASKS:
                    name = promptad.CHECKPOINT_TEMPLATE.format(
                        task=task, seed=promptad.CHECKPOINT_SEED, category=category
                    )
                    (base / name).write_bytes(b"x")


@pytest.mark.parametrize("nested", [True, False])
def test_promptad_resolves_every_configuration_of_the_release(tmp_path, nested):
    """All six configurations must resolve, under either packaging."""
    _build_release(tmp_path, nested=nested)
    for dataset, category in (("mvtec", "bottle"), ("visa", "candle")):
        for k_shot in promptad.SHOT_VALUES:
            for task in promptad.TASKS:
                found = promptad.resolve_checkpoint(
                    dataset, k_shot, category, task, checkpoint_root=str(tmp_path)
                )
                assert found.name == promptad.CHECKPOINT_TEMPLATE.format(
                    task=task, seed=promptad.CHECKPOINT_SEED, category=category
                )
                # The wrapper directory names the configuration it belongs to,
                # so a mixed-up shot or dataset would show up here.
                assert f"{dataset}_{k_shot}shot" in str(found)


def test_promptad_reports_a_class_it_has_no_checkpoint_for(tmp_path):
    _build_release(tmp_path, nested=True)
    with pytest.raises(FileNotFoundError, match="No SEG-Seed_111-nosuch"):
        promptad.resolve_checkpoint(
            "mvtec", 1, "nosuch", "SEG", checkpoint_root=str(tmp_path)
        )


def test_promptad_shot_marker_does_not_confuse_neighbouring_counts():
    from pathlib import Path

    assert promptad._mentions_shot(Path("a/k_1/b"), 1)
    assert promptad._mentions_shot(Path("a/promptad_retrained_mvtec_1shot/b"), 1)
    assert not promptad._mentions_shot(Path("a/k_1/b"), 2)
    # A longer number must not match a shorter one.
    assert not promptad._mentions_shot(Path("a/run_11shot/b"), 1)
    assert not promptad._mentions_shot(Path("a/k_11/b"), 1)


# ----------------------------------------------------------------- AA-CLIP ---
def test_aaclip_checkpoints_are_optional_and_cross_dataset():
    import inspect

    parameters = inspect.signature(aaclip.AACLIPAdapter.__init__).parameters
    # Both are now resolved automatically when not supplied.
    assert parameters["image_checkpoint"].default is None
    assert parameters["text_checkpoint"].default is None
    assert "download_root" in parameters
    assert aaclip.ZERO_SHOT_TRAINING == {
        "mvtec": "TrainOnVisA", "visa": "TrainOnMVTec"
    }
    for target, training in aaclip.ZERO_SHOT_TRAINING.items():
        assert target not in training.lower()
    assert aaclip.KAGGLE_DATASET.count("/") == 1


def test_aaclip_resolves_adapters_from_the_released_tree(tmp_path, monkeypatch):
    """The released layout is AA-CLIP_Checkpoints/TrainOn{MVTec,VisA}/*.pth."""
    for training in ("TrainOnMVTec", "TrainOnVisA"):
        folder = tmp_path / "AA-CLIP_Checkpoints" / training
        folder.mkdir(parents=True)
        (folder / "image_adapter.pth").write_bytes(b"x")
        (folder / "text_adapter.pth").write_bytes(b"x")
    monkeypatch.setattr(aaclip, "download_kaggle_dataset", lambda *a, **k: tmp_path)

    image, text = aaclip.resolve_checkpoints("mvtec")
    assert image.parent.name == "TrainOnVisA"
    assert text == image.parent / "text_adapter.pth"
    image, text = aaclip.resolve_checkpoints("visa")
    assert image.parent.name == "TrainOnMVTec"
    assert text == image.parent / "text_adapter.pth"
    with pytest.raises(ValueError, match="mvtec"):
        aaclip.resolve_checkpoints("btad")


def test_aaclip_resolves_adapters_from_a_local_tree(tmp_path, monkeypatch):
    run = tmp_path / "AA-CLIP" / "TrainOnVisA"
    run.mkdir(parents=True)
    image = run / "image_adapter_5.pth"
    image.write_bytes(b"x")
    (run / "text_adapter.pth").write_bytes(b"x")
    other = tmp_path / "AA-CLIP" / "TrainOnMVTec"
    other.mkdir(parents=True)
    (other / "image_adapter_5.pth").write_bytes(b"x")

    monkeypatch.setattr(aaclip, "download_kaggle_dataset", lambda *a, **k: tmp_path)
    found_image, found_text = aaclip.resolve_checkpoints("mvtec")
    assert found_image == image.resolve()
    assert found_text == (run / "text_adapter.pth")
    # VisA takes the MVTec-trained run instead.
    found_image, _ = aaclip.resolve_checkpoints("visa")
    assert found_image == (other / "image_adapter_5.pth").resolve()
    with pytest.raises(ValueError, match="mvtec"):
        aaclip.resolve_checkpoints("btad")


def test_aaclip_reports_an_ambiguous_or_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(aaclip, "download_kaggle_dataset", lambda *a, **k: tmp_path)
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        aaclip.resolve_checkpoints("mvtec")
    run = tmp_path / "TrainOnVisA"
    run.mkdir()
    (run / "image_adapter_1.pth").write_bytes(b"x")
    (run / "image_adapter_2.pth").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        aaclip.resolve_checkpoints("mvtec")


# ------------------------------------------------------------ shared helper ---
def test_kaggle_helper_reuses_a_cached_copy_instead_of_downloading(tmp_path):
    """A populated cache short-circuits the download, so no network is needed.

    kagglehub is not installed here, so reaching the download path at all would
    raise ImportError - which is what makes this assertion meaningful.
    """
    cache = tmp_path / "my-dataset"
    cache.mkdir()
    (cache / "weights.pt").write_bytes(b"x")
    found = download_kaggle_dataset("owner/my-dataset", download_root=str(tmp_path))
    assert found == cache

    # An empty directory is not a usable cache, so it falls through.
    (cache / "weights.pt").unlink()
    with pytest.raises(ImportError, match="kagglehub"):
        download_kaggle_dataset("owner/my-dataset", download_root=str(tmp_path))


def test_kaggle_helper_finds_files_in_a_stable_order(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "second.pt").write_bytes(b"x")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "first.pt").write_bytes(b"x")
    found = find_kaggle_files(tmp_path, "*.pt")
    assert found == sorted(found)
    assert [path.name for path in found] == ["first.pt", "second.pt"]


def test_kaggle_helper_explains_a_missing_kagglehub(tmp_path):
    with pytest.raises(ImportError, match="kagglehub"):
        download_kaggle_dataset("owner/name", download_root=str(tmp_path))
