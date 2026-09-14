"""Clean-only evaluation: scoring a cohort with no attack in the picture.

Without this the only way to get clean numbers was to generate a throwaway
perturbation purely to satisfy the manifest requirement, then ignore every
adversarial column it produced.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate


@register_adapter("clean_only_probe")
class _Probe(ModelAdapter):
    """Its own adapter: importing another test module would re-register that one."""

    name = "clean_only_probe"

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

    def close(self):
        pass


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mvtec(root: Path) -> Path:
    """Two categories so the macro row has something to average."""
    mvtec = root / "mvtec"
    for category, shade in (("bottle", 40), ("cable", 90)):
        for rel, value in (
            (f"{category}/test/good/000.png", shade),
            (f"{category}/test/crack/001.png", 255),
            (f"{category}/ground_truth/crack/001_mask.png", 255),
        ):
            path = mvtec / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (8, 8) if "ground_truth" in rel else (8, 8, 3)
            Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)
    return mvtec


def _config(tmp_path: Path, mvtec: Path, **overrides) -> EvaluationConfig:
    settings = dict(
        attacks_root=None, output_root=str(tmp_path / "results"),
        model="clean_only_probe", model_kwargs_by_target={"mvtec": {}},
        mvtec_root=str(mvtec), targets=("mvtec",), device="cpu", image_size=8,
        batch_size=2, gaussian_sigma=0, pixel_threshold_modes=("fixed_0_5",),
        clean_only=True, create_output_archives=False,
    )
    settings.update(overrides)
    return EvaluationConfig(**settings)


def test_attacks_root_is_required_unless_clean_only(tmp_path):
    mvtec = _mvtec(tmp_path)
    with pytest.raises(ValueError, match="attacks_root is required"):
        _config(tmp_path, mvtec, clean_only=False)


def test_clean_only_runs_with_no_manifest_at_all(tmp_path):
    """The whole point: no perturbation has to be invented."""
    mvtec = _mvtec(tmp_path)
    output = evaluate(_config(tmp_path, mvtec))

    with (output / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1                      # one threshold mode, one macro row
    row = rows[0]
    assert row["category"] == "__macro__"
    assert row["category_count"] == "2"
    assert row["sample_count"] == "4"
    assert row["cohort"] == "full_test_split"
    assert float(row["clean_i_auroc"]) == 100.0


def test_clean_only_writes_no_adversarial_columns(tmp_path):
    """The columns the workaround produced and had to be told to ignore."""
    mvtec = _mvtec(tmp_path)
    output = evaluate(_config(tmp_path, mvtec))
    for name in ("summary.csv", "category_metrics.csv", "per_image.csv"):
        with (output / name).open(newline="") as handle:
            header = next(csv.reader(handle))
        leaked = [column for column in header
                  if column.startswith(("adversarial_", "delta_", "attack_",
                                        "targeted_", "target_region_",
                                        "location_free_", "realized_"))]
        assert not leaked, f"{name} still carries {leaked}"
        # The attack axes describe something that does not exist here.
        for absent in ("setup_id", "prompt_mode", "direction", "loss_mode"):
            assert absent not in header, f"{name} carries {absent}"


def test_clean_only_still_calibrates_and_records_thresholds(tmp_path):
    mvtec = _mvtec(tmp_path)
    output = evaluate(_config(tmp_path, mvtec))
    payload = json.loads((output / "thresholds.json").read_text(encoding="utf-8"))
    assert set(payload["targets"]["mvtec"]) == {"bottle", "cable"}
    for category in ("bottle", "cable"):
        assert "clean_pixel_f1" in payload["targets"]["mvtec"][category]


def test_clean_only_produces_no_separated_tree(tmp_path):
    """It is filed by setup and prompt mode, neither of which exists here."""
    mvtec = _mvtec(tmp_path)
    evaluate(_config(tmp_path, mvtec))
    regime = tmp_path / "results" / "zero_shot"
    assert [path.name for path in regime.iterdir()] == ["clean_only_probe"]


def test_clean_only_over_a_protocol_cohort_is_labelled_as_such(tmp_path):
    """With a manifest the cohort is the protocol's evaluation half instead.

    Both are useful and they are not the same images, so the row says which.
    """
    mvtec = _mvtec(tmp_path)
    bundle = (tmp_path / "attacks" / "setups" / "frozen_prompt" / "ep100_eps2"
              / "canonical_clip_per_dataset")
    import hashlib

    import torch

    perturbation = bundle / "perturbations" / "normal.pt"
    perturbation.parent.mkdir(parents=True)
    torch.save({"delta": torch.full((3, 8, 8), 0.1)}, perturbation)
    digest = hashlib.sha256(perturbation.read_bytes()).hexdigest()
    _write_csv(bundle / "evaluation_test_indices.csv",
              ["protocol_id", "dataset", "category", "label", "partition"],
              [{"protocol_id": "test/bottle/good/000", "dataset": "mvtec",
                "category": "bottle", "label": 0, "partition": "evaluation"},
               {"protocol_id": "test/bottle/crack/001", "dataset": "mvtec",
                "category": "bottle", "label": 1, "partition": "evaluation"}])
    _write_csv(bundle / "attack_manifest.csv",
              ["scope", "source_dataset", "target_dataset", "direction",
               "source_label", "target_label", "loss_mode",
               "evaluation_attacked_image_count", "perturbation_file",
               "artifact_sha256", "image_size", "epsilon"],
              [{"scope": "per_dataset", "source_dataset": "mvtec",
                "target_dataset": "mvtec", "direction": "normal_to_abnormal",
                "source_label": 0, "target_label": 1, "loss_mode": "global",
                "evaluation_attacked_image_count": 1,
                "perturbation_file": "perturbations/normal.pt",
                "artifact_sha256": digest, "image_size": 8, "epsilon": 0.2}])

    output = evaluate(_config(tmp_path, mvtec,
                              attacks_root=str(tmp_path / "attacks")))
    with (output / "summary.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["cohort"] == "protocol_evaluation_split"
    # Only the protocol's two bottle images, not all four mounted ones.
    assert row["sample_count"] == "2"
    assert row["category_count"] == "1"
