import csv
import hashlib
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
    fields = ["scope", "source_dataset", "target_dataset", "direction", "source_label", "target_label", "loss_mode", "evaluation_attacked_image_count", "perturbation_file", "artifact_sha256", "image_size", "epsilon"]
    _csv(bundle / "attack_manifest.csv", fields, [{
        "scope": "per_dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
        "direction": "normal_to_abnormal", "source_label": 0, "target_label": 1,
        "loss_mode": "global", "evaluation_attacked_image_count": 1,
        "perturbation_file": "perturbations/normal.pt", "artifact_sha256": digest,
        "image_size": 8, "epsilon": 0.1,
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
    assert float(row["clean_i_auroc"]) == 100.0
    with (output / "per_image.csv").open(newline="") as handle:
        images = list(csv.DictReader(handle))
    assert len(images) == 2
    assert sum(int(row["attacked"]) for row in images) == 1
    results_root = tmp_path / "results"
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
    assert list(
        (results_root / "test_adapter_samples" / "fixed_0_5").glob(
            "*/selection_manifest.json"
        )
    )
    for archive in (
        "test_adapter.zip", "test_adapter_separated.zip",
        "test_adapter_samples_separated.zip",
    ):
        assert (results_root / archive).is_file()
    assert not (results_root / "test_adapter_samples.zip").exists()


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
