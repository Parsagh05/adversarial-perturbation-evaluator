import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate


@register_adapter("test_adapter")
class SyntheticAdapter(ModelAdapter):
    name = "test_adapter"

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        scores = images.mean(dim=(1, 2, 3)).numpy()
        maps = images.mean(dim=1).numpy()
        return scores, maps

    def close(self):
        pass


def _csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_end_to_end_fixed_cohort(tmp_path):
    mvtec = tmp_path / "mvtec"
    good = mvtec / "bottle" / "test" / "good" / "000.png"
    crack = mvtec / "bottle" / "test" / "crack" / "001.png"
    mask = mvtec / "bottle" / "ground_truth" / "crack" / "001_mask.png"
    for path, value in ((good, 0), (crack, 255), (mask, 255)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3) if path != mask else (8, 8), value, dtype=np.uint8)).save(path)

    bundle = tmp_path / "attacks" / "setups" / "frozen_prompt" / "steps500_eps2" / "canonical_clip_per_dataset"
    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _csv(bundle / "evaluation_test_indices.csv", ["protocol_id", "dataset", "category", "label", "partition"], [
        {"protocol_id": "test/bottle/good/000", "dataset": "mvtec", "category": "bottle", "label": 0, "partition": "evaluation"},
        {"protocol_id": "test/bottle/crack/001", "dataset": "mvtec", "category": "bottle", "label": 1, "partition": "evaluation"},
    ])
    fields = ["scope", "source_dataset", "target_dataset", "direction", "source_label", "target_label", "loss_mode", "evaluation_attacked_image_count", "perturbation_file", "artifact_sha256", "image_size", "epsilon",
              "prompt_ensemble_sha256", "prompt_checkpoint_sha256"]
    _csv(bundle / "attack_manifest.csv", fields, [{
        "scope": "per_dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
        "direction": "normal_to_abnormal", "source_label": 0, "target_label": 1,
        "loss_mode": "global", "evaluation_attacked_image_count": 1,
        "perturbation_file": "perturbations/normal.pt", "artifact_sha256": digest,
        "image_size": 8, "epsilon": 0.1,
        # The setup ID names the prompt family but not its contents, so the
        # ensemble hash is what separates two otherwise identical runs.
        "prompt_ensemble_sha256": "b6b0fa61c07f2994", "prompt_checkpoint_sha256": "",
    }])
    output = evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "attacks"), output_root=str(tmp_path / "results"),
        model="test_adapter", model_kwargs_by_target={"mvtec": {}},
        mvtec_root=str(mvtec), targets=("mvtec",), scopes=("per_dataset",),
        device="cpu", image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",),
    ))
    assert (output / "summary.csv").is_file()
    assert (output / "category_metrics.csv").is_file()
    assert (output / "thresholds.json").is_file()
    with (output / "summary.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["setup_id"] == "steps500_eps2"
    assert row["prompt_mode"] == "frozen_prompt"
    assert row["category"] == "bottle"
    assert row["category_count"] == "1"
    assert row["prompt_ensemble_sha256"] == "b6b0fa61c07f2994"
    assert row["prompt_checkpoint_sha256"] == ""
    assert "prompt_provenance" not in row
    # Pixel success is reported macro only: every image is one unit of
    # evidence, as the image-level metrics already treat it. Pooling pixels
    # instead let one image with a large defect region speak for the cohort.
    for name in ("summary.csv", "category_metrics.csv"):
        with (output / name).open(newline="") as handle:
            header = next(csv.reader(handle))
        assert not [c for c in header if c.endswith("_micro")], name
        for macro in ("target_region_pixel_flip_rate_macro",
                      "target_region_pixel_attack_success_rate_macro"):
            assert macro in header, (name, macro)
    # The counts stay: the macro ASR is computed from the per-image success
    # flags, and the pixel counts separate "nothing flipped" from "nothing
    # was eligible".
    with (output / "category_metrics.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    for count in ("target_region_pixel_eligible_count",
                  "target_region_pixel_flip_count",
                  "target_region_pixel_success_eligible_count",
                  "target_region_pixel_success_count"):
        assert count in header, count
    assert float(row["clean_i_auroc"]) == 100.0
    with (output / "per_image.csv").open(newline="") as handle:
        images = list(csv.DictReader(handle))
    assert len(images) == 2
    assert sum(int(row["attacked"]) for row in images) == 1
    results_root = tmp_path / "results" / "zero_shot"
    separated = (
        results_root / "test_adapter_separated" / "setups"
        / "frozen_prompt" / "steps500_eps2" / "datasets"
        / "mvtec_to_mvtec" / "per_dataset"
    )
    assert (separated / "numerical" / "summary.csv").is_file()
    assert (separated / "numerical" / "category_metrics.csv").is_file()
    assert (separated / "numerical" / "per_image.csv").is_file()
    separated_samples = (
        results_root / "test_adapter_samples_separated" / "setups"
        / "frozen_prompt" / "steps500_eps2" / "datasets"
        / "mvtec_to_mvtec" / "per_dataset"
    )
    manifests = list((separated_samples / "fixed_0_5").glob(
        "normal_to_abnormal__ce_focal_dice__global/selection_manifest.json"
    ))
    assert len(manifests) == 1
    sample_folders = [path for path in manifests[0].parent.iterdir() if path.is_dir()]
    assert len(sample_folders) == 1
    for filename in (
        "clean.png", "adversarial.png", "difference_x10.png",
        "clean_heatmap.png", "adversarial_heatmap.png", "clean_overlay.png",
        "adversarial_overlay.png", "ground_truth_mask.png",
        "target_region_mask.png", "clean_pixel_prediction.png",
        "adversarial_pixel_prediction.png", "successful_target_pixel_flips.png",
        "heatmap_difference.png", "metrics.json", "description.md",
    ):
        assert (sample_folders[0] / filename).is_file()
    for archive in (
        "test_adapter.zip", "test_adapter_separated.zip",
        "test_adapter_samples_separated.zip",
    ):
        assert (results_root / archive).is_file()
    # The regime folder is the only thing directly under the output root.
    assert [path.name for path in (tmp_path / "results").iterdir()] == ["zero_shot"]
    # Samples are delivered only in the input-shaped separated tree; the old
    # consolidated copy held the same folders and was never archived.
    assert not (results_root / "test_adapter_samples").exists()
    assert not (results_root / "test_adapter_samples.zip").exists()


def test_selected_metrics_only_skips_aupro_and_samples(tmp_path):
    mvtec = tmp_path / "mvtec"
    good = mvtec / "bottle" / "test" / "good" / "000.png"
    crack = mvtec / "bottle" / "test" / "crack" / "001.png"
    mask = mvtec / "bottle" / "ground_truth" / "crack" / "001_mask.png"
    for path, value in ((good, 0), (crack, 255), (mask, 255)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3) if path != mask else (8, 8), value, dtype=np.uint8)).save(path)

    bundle = tmp_path / "attacks" / "setups" / "frozen_prompt" / "steps500_eps2" / "canonical_clip_per_dataset"
    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _csv(bundle / "evaluation_test_indices.csv", ["protocol_id", "dataset", "category", "label", "partition"], [
        {"protocol_id": "test/bottle/good/000", "dataset": "mvtec", "category": "bottle", "label": 0, "partition": "evaluation"},
        {"protocol_id": "test/bottle/crack/001", "dataset": "mvtec", "category": "bottle", "label": 1, "partition": "evaluation"},
    ])
    record = {
        "scope": "per_dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
        "direction": "normal_to_abnormal", "source_label": 0, "target_label": 1,
        "loss_mode": "global", "evaluation_attacked_image_count": 1,
        "perturbation_file": "perturbations/normal.pt", "artifact_sha256": digest,
        "image_size": 8, "epsilon": 0.1,
        "prompt_ensemble_sha256": "", "prompt_checkpoint_sha256": "",
    }
    _csv(bundle / "attack_manifest.csv", list(record), [record])
    output = evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "attacks"), output_root=str(tmp_path / "results"),
        model="test_adapter", model_kwargs_by_target={"mvtec": {}},
        mvtec_root=str(mvtec), targets=("mvtec",), scopes=("per_dataset",),
        device="cpu", image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",), selected_metrics_only=True,
    ))
    with (output / "summary.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))

    def value(name):
        return float(row[name]) if row[name] != "" else float("nan")

    # AUPRO is the only metric dropped; its columns stay, as NaN.
    for name in ("clean_aupro", "adversarial_aupro", "delta_aupro"):
        assert np.isnan(value(name)), name
    for name in (
        "clean_i_auroc", "adversarial_i_auroc", "clean_i_ap", "clean_i_f1_max",
        "clean_p_auroc", "adversarial_p_auroc", "delta_p_auroc",
        "clean_p_f1_max", "adversarial_p_f1_max", "delta_p_f1_max",
        "target_region_pixel_flip_rate_macro",
        "target_region_pixel_attack_success_rate_macro",
    ):
        assert np.isfinite(value(name)), name
    assert value("clean_p_auroc") == 100.0
    results_root = tmp_path / "results" / "zero_shot"
    assert not (results_root / "test_adapter_samples_separated").exists()
    assert not (results_root / "test_adapter_samples_separated.zip").exists()


def _bidirectional_fixture(tmp_path):
    # The synthetic adapter scores an image by its mean: normals 0.0 and 0.2,
    # defects 0.8 and 1.0. N->A adds 0.7 to the normals (0.7, 0.9) and A->N
    # takes 0.7 off the defects (0.1, 0.3). Each direction alone inverts one of
    # the four normal-defect pairs; both together invert all four.
    mvtec = tmp_path / "mvtec"
    images = {
        "good/000": 0, "good/001": 51, "crack/002": 204, "crack/003": 255,
    }
    for name, value in images.items():
        kind, stem = name.split("/")
        path = mvtec / "bottle" / "test" / kind / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)
        if kind == "crack":
            mask = mvtec / "bottle" / "ground_truth" / "crack" / f"{stem}_mask.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask)
    bundle = tmp_path / "attacks" / "setups" / "frozen_prompt" / "steps500_eps2" / "canonical_clip_per_dataset"
    rows = []
    for direction, source, target, shift in (
        ("normal_to_abnormal", 0, 1, 0.7), ("abnormal_to_normal", 1, 0, -0.7),
    ):
        perturbation = bundle / "perturbations" / f"{direction}.pt"
        perturbation.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"delta": torch.full((3, 8, 8), shift)}, perturbation)
        rows.append({
            "scope": "per_dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
            "direction": direction, "source_label": source, "target_label": target,
            "loss_mode": "global", "evaluation_attacked_image_count": 2,
            "perturbation_file": f"perturbations/{direction}.pt",
            "artifact_sha256": hashlib.sha256(perturbation.read_bytes()).hexdigest(),
            "image_size": 8, "epsilon": 0.7,
            "prompt_ensemble_sha256": "", "prompt_checkpoint_sha256": "",
        })
    _csv(bundle / "evaluation_test_indices.csv", ["protocol_id", "dataset", "category", "label", "partition"], [
        {"protocol_id": f"test/bottle/{name}", "dataset": "mvtec", "category": "bottle",
         "label": int(name.startswith("crack")), "partition": "evaluation"}
        for name in images
    ])
    _csv(bundle / "attack_manifest.csv", list(rows[0]), rows)
    return mvtec


def _evaluate_pair(tmp_path, mvtec, results, **overrides):
    return evaluate(EvaluationConfig(
        attacks_root=str(tmp_path / "attacks"), output_root=str(tmp_path / results),
        model="test_adapter", model_kwargs_by_target={"mvtec": {}},
        mvtec_root=str(mvtec), targets=("mvtec",), scopes=("per_dataset",),
        device="cpu", image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",), save_qualitative_samples=False,
        create_output_archives=False, **overrides,
    ))


def test_bidirectional_scores_both_directions_as_one_cohort(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(tmp_path, mvtec, "results")
    with (output / "summary.csv").open(newline="") as handle:
        v1 = {row["direction"]: row for row in csv.DictReader(handle)}
    # Each direction alone: one pair of four inverted.
    assert float(v1["normal_to_abnormal"]["adversarial_i_auroc"]) == 75.0
    assert float(v1["abnormal_to_normal"]["adversarial_i_auroc"]) == 75.0
    with (output / "summary_v2.csv").open(newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["evaluation_variant"] == "V2"
    assert row["direction"] == "bidirectional"
    assert "__all__normal_to_abnormal__" in row["n2a_condition_id"]
    assert "__all__abnormal_to_normal__" in row["a2n_condition_id"]
    assert (row["sample_count"], row["normal_count"], row["defect_count"]) == ("4", "2", "2")
    assert row["attacked_count"] == "4"
    # Together: every normal now outranks every defect.
    assert float(row["clean_i_auroc"]) == 100.0
    assert float(row["adversarial_i_auroc"]) == 0.0
    # The clean-F1 threshold is 0.8: normal 0.9 crosses it, normal 0.7 does
    # not, and both defects fall below it.
    assert float(row["n2a_targeted_attack_success_rate"]) == 50.0
    assert float(row["a2n_targeted_attack_success_rate"]) == 100.0
    assert float(row["untargeted_attack_success_rate"]) == 75.0
    assert np.isfinite(float(row["adversarial_aupro"]))
    assert (output / "category_metrics_v2.csv").is_file()
    separated = (
        tmp_path / "results" / "zero_shot" / "test_adapter_separated" / "setups"
        / "frozen_prompt" / "steps500_eps2" / "datasets" / "mvtec_to_mvtec"
        / "per_dataset" / "numerical"
    )
    assert (separated / "summary_v2.csv").is_file()
    assert (separated / "category_metrics_v2.csv").is_file()


def test_bidirectional_off_or_unpaired_writes_no_v2(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    off = _evaluate_pair(tmp_path, mvtec, "off", evaluate_bidirectional=False)
    assert not (off / "summary_v2.csv").exists()
    # One direction selected: its partner is missing, so there is no pair.
    single = _evaluate_pair(tmp_path, mvtec, "single", directions=("normal_to_abnormal",))
    assert not (single / "summary_v2.csv").exists()
    record = json.loads((single / "run_config_test_adapter.json").read_text())
    assert record["v2_pair_count"] == 0
    assert len(record["v2_unpaired_conditions"]) == 1
    # V2 adds files; it changes nothing in the V1 ones.
    paired = _evaluate_pair(tmp_path, mvtec, "paired")
    for name in ("summary.csv", "category_metrics.csv", "per_image.csv"):
        with (off / name).open(newline="") as handle:
            without = sorted(csv.DictReader(handle), key=lambda row: row.get("condition_id", ""))
        with (paired / name).open(newline="") as handle:
            with_v2 = sorted(csv.DictReader(handle), key=lambda row: row.get("condition_id", ""))
        assert without == with_v2, name


def test_stored_v1_gets_only_v2_added(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(tmp_path, mvtec, "results", evaluate_bidirectional=False)
    v1_files = ("summary.csv", "category_metrics.csv", "per_image.csv",
                "manifest_snapshot.json", "run_config_test_adapter.json")
    before = {name: (output / name).read_bytes() for name in v1_files}
    assert not (output / "summary_v2.csv").exists()
    # Without V2 the old guard stands.
    try:
        _evaluate_pair(tmp_path, mvtec, "results", evaluate_bidirectional=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing V1 result must not be re-run")

    _evaluate_pair(tmp_path, mvtec, "results")
    # V2 is added; every V1 file is left byte for byte.
    assert {name: (output / name).read_bytes() for name in v1_files} == before
    with (output / "summary_v2.csv").open(newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert float(row["adversarial_i_auroc"]) == 0.0
    record = json.loads((output / "run_config_v2_test_adapter.json").read_text())
    assert record["v2_only"] is True and record["status"] == "completed"
    assert record["evaluate_attack_train"] is False
    assert (
        tmp_path / "results" / "zero_shot" / "test_adapter_separated" / "setups"
        / "frozen_prompt" / "steps500_eps2" / "datasets" / "mvtec_to_mvtec"
        / "per_dataset" / "numerical" / "summary_v2.csv"
    ).is_file()

    # Complete now: a further run does nothing.
    stamp = (output / "run_config_v2_test_adapter.json").read_bytes()
    _evaluate_pair(tmp_path, mvtec, "results")
    assert (output / "run_config_v2_test_adapter.json").read_bytes() == stamp


def test_v2_only_refuses_a_stored_v1_it_cannot_reproduce(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(tmp_path, mvtec, "results", evaluate_bidirectional=False)
    with (output / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["adversarial_i_auroc"] = "10.0"  # as if another model had produced it
    _csv(output / "summary.csv", list(rows[0]), rows)
    try:
        _evaluate_pair(tmp_path, mvtec, "results")
    except ValueError as error:
        assert "differs from the stored result" in str(error)
    else:
        raise AssertionError("a V1 that no longer reproduces must stop the V2 run")
    assert not (output / "summary_v2.csv").exists()


def test_max_sample_conditions_keeps_only_the_strongest_per_scope(tmp_path):
    from fpeval.engine import _SampleBudget

    budget = _SampleBudget(2)
    made: dict[str, Path] = {}
    for name, rank in (("weak", 5.0), ("best", 90.0), ("mid", 40.0), ("worst", 1.0)):
        for scope in ("per_dataset", "per_category"):
            folder = tmp_path / scope / name
            (folder / "sample").mkdir(parents=True)
            (folder / "sample" / "metrics.json").write_text("{}", encoding="utf-8")
            made[f"{scope}/{name}"] = folder
        budget.register(
            "per_dataset", rank, [tmp_path / "per_dataset" / name]
        )
        budget.register(
            "per_category", rank, [tmp_path / "per_category" / name]
        )

    # Each scope is bounded independently and keeps its two strongest.
    assert budget.summary() == {"per_category": 2, "per_dataset": 2}
    for scope in ("per_dataset", "per_category"):
        assert made[f"{scope}/best"].is_dir()
        assert made[f"{scope}/mid"].is_dir()
        assert not made[f"{scope}/weak"].exists()
        assert not made[f"{scope}/worst"].exists()


def test_sample_budget_is_inactive_without_a_limit(tmp_path):
    from fpeval.engine import _SampleBudget

    budget = _SampleBudget(None)
    folder = tmp_path / "kept"
    folder.mkdir()
    budget.register("per_image", 0.0, [folder])
    assert folder.is_dir()
    assert budget.summary() == {}


def test_condition_rank_ignores_blank_and_non_finite_rates():
    from fpeval.engine import _condition_sample_rank

    assert _condition_sample_rank([{"targeted_attack_success_rate": 12.5}]) == 12.5
    assert _condition_sample_rank(
        [{"targeted_attack_success_rate": ""},
         {"targeted_attack_success_rate": 30.0},
         {"targeted_attack_success_rate": float("nan")}]
    ) == 30.0
    assert _condition_sample_rank([{}]) == float("-inf")
