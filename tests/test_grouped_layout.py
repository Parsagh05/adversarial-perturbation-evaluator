"""The generator's grouped tree, read and mirrored.

    setups/<settings>/<scope>/ep<budget>/<frozen|learnable>/

The flat setup ID is no longer a path component, so it is read from the
manifest; the settings and the scope's budget are read from the path, because
under halfcross the ID drops the cross budget while the directory keeps it.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.attacks import (
    discover_attacks,
    layout_from_bundle,
    materialize_input,
    scope_budget_tag,
    split_setup_id,
)
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate


@register_adapter("layout_probe")
class _Probe(ModelAdapter):
    name = "layout_probe"

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

    def close(self):
        pass


SETTINGS = "eps2_ce_focal_dice_full"
SETUP_ID = "ep7p14_cross50_cat100_img100_eps2_ce_focal_dice_full"


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build(root: Path) -> Path:
    mvtec = root / "mvtec"
    for name, value in (("good/000", 0), ("crack/001", 255)):
        path = mvtec / "bottle" / "test" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)
    mask = mvtec / "bottle" / "ground_truth" / "crack" / "001_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask)

    # setups/<settings>/<scope>/ep<budget>/<prompt family>
    bundle = (root / "attacks" / "setups" / SETTINGS / "per_dataset"
              / "ep7p14" / "frozen_prompt")
    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _write(bundle / "evaluation_test_indices.csv",
           ["protocol_id", "dataset", "category", "label", "partition"],
           [{"protocol_id": "test/bottle/good/000", "dataset": "mvtec",
             "category": "bottle", "label": 0, "partition": "evaluation"},
            {"protocol_id": "test/bottle/crack/001", "dataset": "mvtec",
             "category": "bottle", "label": 1, "partition": "evaluation"}])
    _write(bundle / "attack_manifest.csv",
           ["scope", "source_dataset", "target_dataset", "direction",
            "source_label", "target_label", "loss_mode", "setup_id",
            "evaluation_attacked_image_count", "perturbation_file",
            "artifact_sha256", "image_size", "epsilon"],
           [{"scope": "per_dataset", "source_dataset": "mvtec",
             "target_dataset": "mvtec", "direction": "normal_to_abnormal",
             "source_label": 0, "target_label": 1, "loss_mode": "global",
             "setup_id": SETUP_ID, "evaluation_attacked_image_count": 1,
             "perturbation_file": "perturbations/normal.pt",
             "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2}])
    return mvtec


def test_layout_is_recognised_only_at_the_right_shape(tmp_path):
    grouped = tmp_path / "setups" / SETTINGS / "per_dataset" / "ep7p14" / "frozen_prompt"
    assert layout_from_bundle(grouped) == (SETTINGS, "ep7p14")
    # The old flat tree is not mistaken for it.
    flat = tmp_path / "setups" / "frozen_prompt" / SETUP_ID / "canonical_clip_per_dataset"
    assert layout_from_bundle(flat) is None
    # Nor is a near miss on any single level.
    for wrong in (
        tmp_path / "setups" / SETTINGS / "per_dataset" / "ep7p14" / "other",
        tmp_path / "setups" / SETTINGS / "not_a_scope" / "ep7p14" / "frozen_prompt",
        tmp_path / "setups" / SETTINGS / "per_dataset" / "800" / "frozen_prompt",
    ):
        assert layout_from_bundle(wrong) is None, wrong


def test_the_setup_id_comes_from_the_manifest(tmp_path):
    """It is no longer a path component, so the path cannot supply it."""
    _build(tmp_path)
    bundles = materialize_input(tmp_path / "attacks", tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))
    assert len(attacks) == 1
    record = attacks[0].record
    assert record["setup_id"] == SETUP_ID
    assert record["prompt_mode"] == "frozen_prompt"
    assert record["settings"] == SETTINGS
    assert record["scope_budget"] == "ep7p14"


def test_a_grouped_bundle_without_the_id_fails_at_discovery(tmp_path):
    """The seam that cost a full generation run before it was caught.

    The generator regrouped the tree but its runners did not yet write
    setup_id, so the ID reached nothing: the evaluator labelled every
    condition "unspecified_setup" and the run only died later, on an empty
    selection. Under this layout the path cannot supply the ID, so a manifest
    that omits it is refused where the cost is one traceback.
    """
    import pytest

    _build(tmp_path)
    manifest = (tmp_path / "attacks" / "setups" / SETTINGS / "per_dataset"
                / "ep7p14" / "frozen_prompt" / "attack_manifest.csv")
    rows = list(csv.DictReader(manifest.open(newline="")))
    for row in rows:
        row.pop("setup_id")
    _write(manifest, list(rows[0]), rows)

    bundles = materialize_input(tmp_path / "attacks", tmp_path / "cache")
    with pytest.raises(ValueError, match="has no setup_id"):
        discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))


def test_the_output_tree_mirrors_the_generators(tmp_path):
    mvtec = _build(tmp_path)
    evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "attacks"),
        output_root=str(tmp_path / "results"), model="layout_probe",
        model_kwargs_by_target={"mvtec": {}}, mvtec_root=str(mvtec),
        targets=("mvtec",), scopes=("per_dataset",), device="cpu", image_size=8,
        batch_size=2, gaussian_sigma=0, pixel_threshold_modes=("fixed_0_5",),
        save_qualitative_samples=False, create_output_archives=False,
    ))
    separated = tmp_path / "results" / "zero_shot" / "layout_probe_separated"
    produced = next(separated.rglob("summary.csv")).parent
    assert produced.relative_to(separated).as_posix() == (
        f"setups/{SETTINGS}/per_dataset/ep7p14/frozen_prompt"
        "/datasets/mvtec_to_mvtec/numerical"
    )


def test_a_flat_bundle_is_filed_in_the_grouped_tree_too(tmp_path):
    """One results tree has one shape, whichever tree the bundle came from.

    The levels are derived from the setup ID, which is all a flat bundle has.
    """
    mvtec = _build(tmp_path)
    flat = (tmp_path / "attacks" / "setups" / SETTINGS / "per_dataset"
            / "ep7p14" / "frozen_prompt")
    moved = tmp_path / "flat" / "setups" / "frozen_prompt" / "ep100_eps2" / "canonical_clip_per_dataset"
    moved.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(flat, moved)
    # Drop the manifest's setup_id so the path has to supply it, as it did.
    rows = list(csv.DictReader((moved / "attack_manifest.csv").open(newline="")))
    for row in rows:
        row.pop("setup_id")
    _write(moved / "attack_manifest.csv", list(rows[0]), rows)

    evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "flat"),
        output_root=str(tmp_path / "flat_results"), model="layout_probe",
        model_kwargs_by_target={"mvtec": {}}, mvtec_root=str(mvtec),
        targets=("mvtec",), scopes=("per_dataset",), device="cpu", image_size=8,
        batch_size=2, gaussian_sigma=0, pixel_threshold_modes=("fixed_0_5",),
        save_qualitative_samples=False, create_output_archives=False,
    ))
    separated = tmp_path / "flat_results" / "zero_shot" / "layout_probe_separated"
    produced = next(separated.rglob("summary.csv")).parent
    assert produced.relative_to(separated).as_posix() == (
        "setups/eps2/per_dataset/ep100/frozen_prompt"
        "/datasets/mvtec_to_mvtec/numerical"
    )


def test_the_settings_split_matches_the_generators_own(tmp_path):
    """Derivation is the fallback for flat bundles, so it has to agree."""
    assert split_setup_id(SETUP_ID) == (
        "ep7p14_cross50_cat100_img100", "eps2_ce_focal_dice_full")
    assert scope_budget_tag(SETUP_ID, "per_dataset") == "ep7p14"
    assert scope_budget_tag(SETUP_ID, "cross_dataset") == "ep50"
    assert scope_budget_tag(SETUP_ID, "per_category") == "ep100"
    assert scope_budget_tag(SETUP_ID, "per_image") == "ep100"
