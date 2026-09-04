import hashlib

import pytest

from fpeval.adapters import adapter_names
from fpeval.adapters import anovl, crane, dictas, kagprompt


def test_all_four_are_registered():
    for name in (
        "craneplus", "crane-plus", "dictas", "dict-as",
        "anovl", "ano-vl", "kagprompt", "kag-prompt",
    ):
        assert name in adapter_names()


# ----------------------------------------------------------------- Crane+ ---
def test_crane_plus_uses_its_own_weights_and_never_the_base_ones():
    assert crane.CRANE_PLUS_CHECKPOINT == {
        "mvtec": "trained_on_visa_cranep",
        "visa": "trained_on_mvtec_cranep",
    }
    # Still cross-dataset, and distinct from the base Crane row's files.
    for target, name in crane.CRANE_PLUS_CHECKPOINT.items():
        assert target not in name
        assert name != crane.ZERO_SHOT_CHECKPOINT[target]


def test_crane_plus_defaults_differ_from_the_base_row():
    """test.sh gives Crane+ one feature level and no --soft_mean."""
    import inspect

    base = inspect.signature(crane.CraneAdapter.__init__).parameters
    plus = inspect.signature(crane.CranePlusAdapter.__init__).parameters
    assert base["features"].default == (6, 12, 18, 24)
    assert base["soft_mean"].default is True
    assert base["dino_model"].default == "none"
    assert plus["features"].default == (24,)
    # The base row passes --soft_mean True explicitly; Crane+ does not, so the
    # argparse default of False applies.
    assert plus["soft_mean"].default is False
    assert plus["dino_model"].default == "dinov2"


def test_crane_rejects_an_unreleased_dino_branch(tmp_path):
    with pytest.raises(ValueError, match="dinov2"):
        crane.CraneAdapter(
            repository=str(tmp_path), target_dataset="mvtec", dino_model="sam"
        )


# ----------------------------------------------------------------- DictAS ---
def test_dictas_pins_both_released_dictionaries():
    assert set(dictas.CHECKPOINTS) == {"train_visa", "train_mvtec"}
    assert dictas.ZERO_SHOT_CHECKPOINT == {"mvtec": "train_visa", "visa": "train_mvtec"}
    for target, name in dictas.ZERO_SHOT_CHECKPOINT.items():
        assert target not in name
    for filename, file_id, digest in dictas.CHECKPOINTS.values():
        assert filename.endswith(".pth")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_dictas_fusion_constants_match_calcuate_metric_pixel():
    # calcuate_metric_pixel forces alpha 0.2 for exactly mvtec and visa, and
    # BESTSEGMENTATION sets sigma 6 under its default TEST_For_BESTSEGMENTATION.
    assert dictas.ALPHA == 0.2
    assert dictas.SIGMA == 6


def test_dictas_selection_covers_every_category_and_shot():
    counts = {"mvtec": 15, "visa": 12}
    for target, shots in dictas.SELECTION.items():
        assert set(shots) == set(dictas.SHOT_VALUES)
        for k, table in shots.items():
            assert len(table) == counts[target], (target, k)
            for category, positions in table.items():
                assert len(positions) == k, (target, k, category)
                assert len(set(positions)) == k, (target, k, category)
                assert all(index >= 0 for index in positions)


def test_dictas_selection_indices_fit_the_official_train_splits():
    """Every pinned position must fall inside its class's train split."""
    train_counts = {
        "mvtec": {
            "bottle": 209, "cable": 224, "capsule": 219, "carpet": 280,
            "grid": 264, "hazelnut": 391, "leather": 245, "metal_nut": 220,
            "pill": 267, "screw": 320, "tile": 230, "toothbrush": 60,
            "transistor": 213, "wood": 247, "zipper": 240,
        },
        "visa": {
            "candle": 900, "capsules": 542, "cashew": 450, "chewinggum": 453,
            "fryum": 450, "macaroni1": 900, "macaroni2": 900, "pcb1": 904,
            "pcb2": 901, "pcb3": 905, "pcb4": 904, "pipe_fryum": 450,
        },
    }
    for target, shots in dictas.SELECTION.items():
        for k, table in shots.items():
            for category, positions in table.items():
                limit = train_counts[target][category]
                assert max(positions) < limit, (target, k, category, positions, limit)


def test_dictas_rotates_only_screw():
    assert dictas.ROTATED_CATEGORY == "screw"
    # Roate_support walks range(-180, 181, 45): nine angles plus the original.
    assert dictas.ROTATION_ANGLES == (-180, -135, -90, -45, 0, 45, 90, 135, 180)
    assert len(dictas.ROTATION_ANGLES) == 9


def test_dictas_rejects_an_unsupported_setting(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        dictas.DictASAdapter(repository=str(tmp_path), target_dataset="btad")
    with pytest.raises(ValueError, match="k_shot"):
        dictas.DictASAdapter(
            repository=str(tmp_path), target_dataset="mvtec", k_shot=8
        )
    with pytest.raises(ValueError, match="TEST_For_BESTSEGMENTATION"):
        dictas.DictASAdapter(
            repository=str(tmp_path), target_dataset="mvtec", best_segmentation=False
        )


def test_dictas_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        dictas._import_official_repository(tmp_path)


# ------------------------------------------------------------------ AnoVL ---
def test_anovl_pairs_each_dataset_with_its_official_script():
    # test_zero_shot.sh runs MVTec through vl_test.py and VisA through vis_test.py.
    assert anovl.ADAPTER_MODULE == {"mvtec": "TextAdapter", "visa": "Adapter"}
    assert anovl.OFFICIAL_SEED == {"mvtec": 111, "visa": 42}


def test_anovl_map_token_index_is_the_last_requested_layer():
    """The transformer appends two tensors per requested layer.

    With four layers that is eight entries, so the scripts' ``layer != 6`` picks
    the v-v branch of the last one. At four entries it would look like a dead
    loop, which is the trap this pins.
    """
    assert anovl.MAP_TOKEN_INDEX == 6
    layers = 4
    entries = layers * 2
    assert anovl.MAP_TOKEN_INDEX < entries
    assert anovl.MAP_TOKEN_INDEX == (layers - 1) * 2


def test_anovl_entropy_loss_matches_the_official_objective():
    import torch

    torch.manual_seed(0)
    prediction = torch.softmax(torch.randn(5, 3, 3, 2), dim=-1)
    soft = -prediction[0] * prediction[0].log()
    mask = torch.zeros(prediction[1:].shape)
    mask[..., 1] = 1
    hard = -mask * prediction[1:].log() - (1 - mask) * prediction[0].log()
    official = soft.sum(-1).mean() + 0.5 * hard.sum(-1).mean()
    assert torch.allclose(anovl.AnoVLAdapter._entropy_loss(prediction), official)


def test_anovl_rejects_an_unknown_target(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        anovl.AnoVLAdapter(repository=str(tmp_path), target_dataset="btad")


def test_anovl_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        anovl._import_official_repository(tmp_path)


# ------------------------------------------------------------- KAG-Prompt ---
def test_kagprompt_pins_its_released_files():
    assert set(kagprompt.CHECKPOINTS) == {"train_on_mvtec", "train_on_visa"}
    assert kagprompt.ZERO_SHOT_CHECKPOINT == {
        "mvtec": "train_on_visa", "visa": "train_on_mvtec"
    }
    for target, name in kagprompt.ZERO_SHOT_CHECKPOINT.items():
        assert target not in name
    for filename, file_id, digest in kagprompt.CHECKPOINTS.values():
        assert filename.endswith(".pt")
        assert len(file_id) > 20
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_kagprompt_pins_the_imagebind_backbone():
    assert kagprompt.IMAGEBIND[0] == "imagebind_huge.pth"
    assert len(kagprompt.IMAGEBIND[1]) > 20
    digest = kagprompt.IMAGEBIND_SHA256
    assert digest is not None and len(digest) == 64 and int(digest, 16) >= 0


def test_kagprompt_score_constants():
    assert kagprompt.FUSION_R == 0.1
    assert kagprompt.SCORE_TOP_K == 30
    assert kagprompt.SCORE_WEIGHT == 0.1
    assert kagprompt.MODEL_IMAGE_SIZE == 224


def test_kagprompt_prompts_match_the_official_tables():
    # The model matches the prompt against its own CLASS_NAMES, which spell
    # "metal nut" and collapse pcb1-4 and macaroni1-2 onto shared names.
    assert kagprompt.DESCRIBLES["mvtec"]["metal_nut"] == "metal nut"
    assert kagprompt.DESCRIBLES["visa"]["pipe_fryum"] == "pipe fryum"
    for name in ("pcb1", "pcb2", "pcb3", "pcb4"):
        assert kagprompt.DESCRIBLES["visa"][name] == "pcb"
    for name in ("macaroni1", "macaroni2"):
        assert kagprompt.DESCRIBLES["visa"][name] == "macaroni"
    # The official table really does misspell chewinggum.
    assert kagprompt.DESCRIBLES["visa"]["chewinggum"] == "chewinggom"
    assert len(kagprompt.DESCRIBLES["mvtec"]) == 15
    assert len(kagprompt.DESCRIBLES["visa"]) == 12


def test_kagprompt_positions_follow_round_and_fall_back_on_short_splits():
    # MVTec asks for files round..round+k-1 when they exist.
    assert kagprompt.official_positions("mvtec", 1, 209) == [195]
    assert kagprompt.official_positions("mvtec", 2, 209) == [195, 196]
    assert kagprompt.official_positions("mvtec", 4, 209) == [194, 195, 196, 197]
    # toothbrush has 60 train images, so 195.png does not exist and the script
    # takes the last k instead.
    assert kagprompt.official_positions("mvtec", 1, 60) == [59]
    assert kagprompt.official_positions("mvtec", 4, 60) == [56, 57, 58, 59]
    # VisA slices [round * 4:] after collecting round * 4 + k.
    assert kagprompt.official_positions("visa", 1, 900) == [56]
    assert kagprompt.official_positions("visa", 2, 900) == [228, 229]
    assert kagprompt.official_positions("visa", 4, 900) == [312, 313, 314, 315]
    with pytest.raises(ValueError, match="only"):
        kagprompt.official_positions("visa", 4, 100)
    with pytest.raises(ValueError, match="k_shot"):
        kagprompt.official_positions("mvtec", 8, 900)


def test_kagprompt_records_the_dead_rotation_branch():
    # if 'mvtec' in 'normal_img_paths' compares against the literal string.
    assert "mvtec" not in "normal_img_paths"


def test_kagprompt_rejects_an_unknown_target(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        kagprompt.KAGPromptAdapter(repository=str(tmp_path), target_dataset="btad")


def test_kagprompt_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        kagprompt._import_official_repository(tmp_path)


def test_kagprompt_verifies_a_supplied_checkpoint(tmp_path):
    payload = b"weights"
    path = tmp_path / "train_on_visa.pt"
    path.write_bytes(payload)
    assert kagprompt.resolve_checkpoint(str(path)) == path.resolve()
    with pytest.raises(FileNotFoundError, match="not found"):
        kagprompt.resolve_checkpoint(str(tmp_path / "absent.pt"))
    # A matching digest short-circuits the download entirely.
    digest = hashlib.sha256(payload).hexdigest()
    assert kagprompt._fetch("train_on_visa.pt", "id", digest, tmp_path) == path
