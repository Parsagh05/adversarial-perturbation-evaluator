"""Scoring a perturbation on the images it was fitted on.

A universal delta that works where it was fitted and nowhere else memorised its
cohort. Reporting only the held-out number hides that, so the fitted images are
scored too and the two are kept apart by a partition column.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate


@register_adapter("train_partition_probe")
class _Probe(ModelAdapter):
    name = "train_partition_probe"

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

    def close(self):
        pass


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


PROTOCOL = ["protocol_id", "dataset", "category", "label", "partition"]
MANIFEST = ["scope", "source_dataset", "target_dataset", "direction",
            "source_label", "target_label", "loss_mode",
            "evaluation_attacked_image_count", "perturbation_file",
            "artifact_sha256", "image_size", "epsilon"]


def _row(name: str, label: int, partition: str) -> dict:
    return {"protocol_id": f"test/bottle/{name}", "dataset": "mvtec",
            "category": "bottle", "label": label, "partition": partition}


def _build(root: Path, *, scope: str = "per_dataset", with_train: bool = True) -> Path:
    """Two held-out images and two the delta was fitted on."""
    mvtec = root / "mvtec"
    images = {"good/000": 0, "crack/001": 255, "good/002": 10, "crack/003": 245}
    for name, value in images.items():
        path = mvtec / "bottle" / "test" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)
        if name.startswith("crack"):
            mask = mvtec / "bottle" / "ground_truth" / f"{name}_mask.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask)

    bundle = (root / "attacks" / "setups" / "frozen_prompt" / "ep100_eps2"
              / f"canonical_clip_{scope}")
    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()

    _write(bundle / "evaluation_test_indices.csv", PROTOCOL,
           [_row("good/000", 0, "evaluation"), _row("crack/001", 1, "evaluation")])
    if with_train:
        _write(bundle / "attack_train_indices.csv", PROTOCOL,
               [_row("good/002", 0, "attack_train"),
                _row("crack/003", 1, "attack_train")])
    _write(bundle / "attack_manifest.csv", MANIFEST,
           [{"scope": scope, "source_dataset": "mvtec", "target_dataset": "mvtec",
             "direction": "normal_to_abnormal", "source_label": 0,
             "target_label": 1, "loss_mode": "global",
             "evaluation_attacked_image_count": 1,
             "perturbation_file": "perturbations/normal.pt",
             "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2}])
    return mvtec


def _build_visa(root: Path) -> Path:
    """VisA is driven by split_csv/1cls.csv, not by walking directories."""
    visa = root / "visa"
    rows = []
    for sub, label, index in (("normal", "normal", 0), ("anomaly", "anomaly", 1)):
        rel = f"candle/test/{sub}/00{index}.png"
        path = visa / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), index * 255, dtype=np.uint8)).save(path)
        mask_rel = ""
        if sub == "anomaly":
            mask_rel = f"candle/ground_truth/{sub}/00{index}_mask.png"
            mask = visa / mask_rel
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask)
        rows.append({"object": "candle", "split": "test", "label": label,
                     "image": rel, "mask": mask_rel})
    _write(visa / "split_csv" / "1cls.csv",
           ["object", "split", "label", "image", "mask"], rows)
    return visa


def _run(tmp_path: Path, mvtec: Path, **overrides):
    settings = dict(
        attacks_root=str(tmp_path / "attacks"),
        output_root=str(tmp_path / "results"), model="train_partition_probe",
        model_kwargs_by_target={"mvtec": {}}, mvtec_root=str(mvtec),
        targets=("mvtec",), scopes=("per_dataset", "per_image"), device="cpu",
        image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",), save_qualitative_samples=False,
        create_output_archives=False,
    )
    settings.update(overrides)
    output = evaluate(EvaluationConfig(**settings))
    with (output / "summary.csv").open(newline="") as handle:
        return output, list(csv.DictReader(handle))


def test_held_out_only_by_default(tmp_path):
    mvtec = _build(tmp_path)
    _, rows = _run(tmp_path, mvtec)
    assert {row["partition"] for row in rows} == {"evaluation"}
    assert len(rows) == 1


def test_the_fitted_images_are_scored_as_their_own_rows(tmp_path):
    mvtec = _build(tmp_path)
    _, rows = _run(tmp_path, mvtec, evaluate_attack_train=True)
    partitions = [row["partition"] for row in rows]
    assert sorted(partitions) == ["attack_train", "evaluation"]
    # Same condition, different cohorts: never pooled into one row.
    held_out = next(r for r in rows if r["partition"] == "evaluation")
    fitted = next(r for r in rows if r["partition"] == "attack_train")
    assert held_out["condition_id"] == fitted["condition_id"]
    assert held_out["sample_count"] == fitted["sample_count"] == "2"


def test_both_partitions_are_scored_against_the_same_thresholds(tmp_path):
    """Calibration stays on the held-out cohort and is then frozen.

    Recalibrating on the fitted images would give them their own operating
    point and make the two numbers incomparable, which is the whole purpose.
    """
    mvtec = _build(tmp_path)
    _, plain = _run(tmp_path, mvtec)
    _, both = _run(tmp_path, mvtec, evaluate_attack_train=True, overwrite=True)
    held_out = next(r for r in both if r["partition"] == "evaluation")
    for column in ("clean_i_auroc", "adversarial_i_auroc", "attack_flip_rate"):
        assert plain[0][column] == held_out[column], column


def test_per_image_gets_no_fitted_pass(tmp_path):
    """It fits the single image it attacks, so nothing is held out.

    A real per-image bundle, so this exercises the scope check in discovery
    rather than asserting a hand-made value.
    """
    mvtec = _build(tmp_path)          # the images and the dataset-scope bundle
    bundle = (tmp_path / "attacks" / "setups" / "frozen_prompt" / "ep100_eps2"
              / "canonical_clip_per_image")
    perturbation = bundle / "perturbations" / "per_image.pt"
    perturbation.parent.mkdir(parents=True)
    # per_image payloads key their stack "deltas", not "delta".
    torch.save({"deltas": torch.full((1, 3, 8, 8), 0.1),
                "sample_ids": ["test/bottle/good/000"]}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _write(bundle / "evaluation_test_indices.csv", PROTOCOL,
           [_row("good/000", 0, "evaluation"), _row("crack/001", 1, "evaluation")])
    _write(bundle / "attack_train_indices.csv", PROTOCOL,
           [_row("good/002", 0, "attack_train"),
            _row("crack/003", 1, "attack_train")])
    _write(bundle / "attack_manifest.csv", [*MANIFEST, "category"],
           [{"scope": "per_image", "source_dataset": "mvtec",
             "target_dataset": "mvtec", "direction": "normal_to_abnormal",
             "source_label": 0, "target_label": 1, "loss_mode": "global",
             "evaluation_attacked_image_count": 1,
             "perturbation_file": "perturbations/per_image.pt",
             "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2,
             "category": "bottle"}])

    from fpeval.attacks import discover_attacks, materialize_input

    bundles = materialize_input(tmp_path / "attacks", tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset", "per_image"),
                               targets=("mvtec",))
    by_scope = {a.record["scope"]: a for a in attacks}
    assert set(by_scope) == {"per_dataset", "per_image"}
    # The dataset scope has a training half even though the same CSV sits in
    # both bundles, so the difference is the scope and not the file.
    assert by_scope["per_dataset"].train_ids == (
        "test/bottle/good/002", "test/bottle/crack/003")
    assert by_scope["per_image"].train_ids == ()
    assert by_scope["per_image"].train_attacked_ids == ()

    # And the run therefore produces held-out rows only for it.
    _, rows = _run(tmp_path, mvtec, evaluate_attack_train=True,
                   scopes=("per_image",))
    assert {row["partition"] for row in rows} == {"evaluation"}


def test_a_bundle_without_the_training_protocol_still_evaluates(tmp_path):
    """Older bundles ship no attack_train_indices.csv; held-out is unaffected."""
    mvtec = _build(tmp_path, with_train=False)
    _, rows = _run(tmp_path, mvtec, evaluate_attack_train=True)
    assert {row["partition"] for row in rows} == {"evaluation"}


def test_a_cross_dataset_delta_has_no_fitted_cohort_in_its_target(tmp_path):
    """A delta delivered to a dataset it was not fitted on.

    Its fitted images live in the source, so the target cohort contains none
    of them: there is no attack_train partition there to score. Before this
    was handled, discovery still collected the source IDs and the engine
    refused them as absent from the target, so turning the flag on crashed
    every cross-dataset condition.
    """
    mvtec = _build(tmp_path)          # the images and a same-dataset bundle
    visa = _build_visa(tmp_path)

    bundle = (tmp_path / "attacks" / "setups" / "frozen_prompt" / "ep100_eps2"
              / "canonical_clip_cross_dataset")
    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    # Held out: the VisA images the delta is delivered to. Fitted: MVTec.
    _write(bundle / "evaluation_test_indices.csv", PROTOCOL, [
        {"protocol_id": "test/visa/candle/normal/000", "dataset": "visa",
         "category": "candle", "label": 0, "partition": "evaluation"},
        {"protocol_id": "test/visa/candle/anomaly/001", "dataset": "visa",
         "category": "candle", "label": 1, "partition": "evaluation"}])
    _write(bundle / "attack_train_indices.csv", PROTOCOL,
           [_row("good/002", 0, "attack_train"),
            _row("crack/003", 1, "attack_train")])
    _write(bundle / "attack_manifest.csv", MANIFEST,
           [{"scope": "cross_dataset", "source_dataset": "mvtec",
             "target_dataset": "visa", "direction": "normal_to_abnormal",
             "source_label": 0, "target_label": 1, "loss_mode": "global",
             "evaluation_attacked_image_count": 1,
             "perturbation_file": "perturbations/normal.pt",
             "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2}])

    from fpeval.attacks import discover_attacks, materialize_input

    bundles = materialize_input(tmp_path / "attacks", tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("cross_dataset",), targets=("visa",))
    assert len(attacks) == 1
    # The MVTec IDs are never collected, so the engine is never asked for them.
    assert attacks[0].train_ids == ()
    assert attacks[0].train_attacked_ids == ()

    output = evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "attacks"),
        output_root=str(tmp_path / "results"), model="train_partition_probe",
        model_kwargs_by_target={"visa": {}}, visa_root=str(visa),
        targets=("visa",), scopes=("cross_dataset",), device="cpu",
        image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",), save_qualitative_samples=False,
        create_output_archives=False, evaluate_attack_train=True))
    with (output / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "the cross-dataset condition still has to be evaluated"
    assert {row["partition"] for row in rows} == {"evaluation"}


def test_an_alltargets_bundle_is_refused_with_its_cause(tmp_path):
    """PER_IMAGE_ATTACK_COHORT=all attacks more than this scores.

    Such a bundle attacked every retained image, the attack-train half
    included, while the evaluator scores the evaluation cohort as it does for
    every other scope. The counts cannot agree, and the bare numbers do not
    say why, so the refusal names the setting.
    """
    _build(tmp_path)
    bundle = (tmp_path / "attacks" / "setups" / "frozen_prompt" / "ep100_eps2"
              / "canonical_clip_per_image")
    perturbation = bundle / "perturbations" / "per_image.pt"
    perturbation.parent.mkdir(parents=True)
    fitted = ["test/bottle/good/000", "test/bottle/good/002"]
    torch.save({"deltas": torch.full((2, 3, 8, 8), 0.1),
                "sample_ids": fitted}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _write(bundle / "evaluation_test_indices.csv", PROTOCOL,
           [_row("good/000", 0, "evaluation"), _row("crack/001", 1, "evaluation")])
    _write(bundle / "attack_train_indices.csv", PROTOCOL,
           [_row("good/002", 0, "attack_train"),
            _row("crack/003", 1, "attack_train")])
    _write(bundle / "attack_manifest.csv",
           [*MANIFEST, "category", "per_image_attack_cohort"],
           [{"scope": "per_image", "source_dataset": "mvtec",
             "target_dataset": "mvtec", "direction": "normal_to_abnormal",
             "source_label": 0, "target_label": 1, "loss_mode": "global",
             "evaluation_attacked_image_count": 2,   # what "all" attacked
             "perturbation_file": "perturbations/per_image.pt",
             "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2,
             "category": "bottle", "per_image_attack_cohort": "all"}])

    from fpeval.attacks import discover_attacks, materialize_input

    bundles = materialize_input(tmp_path / "attacks", tmp_path / "cache")
    with pytest.raises(ValueError, match="per_image_attack_cohort=all"):
        discover_attacks(bundles, scopes=("per_image",), targets=("mvtec",))
