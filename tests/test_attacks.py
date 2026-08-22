import csv
import hashlib
from pathlib import Path
import zipfile

import pytest
import torch

from fpeval.attacks import discover_attacks, materialize_input


def _write_bundle(root: Path) -> Path:
    bundle = root / "setups" / "learnable_prompt" / "steps500_eps2_learnable_prompt" / "canonical_clip_per_dataset"
    (bundle / "perturbations").mkdir(parents=True)
    tensor_path = bundle / "perturbations" / "delta.pt"
    torch.save({"delta": torch.zeros(3, 8, 8)}, tensor_path)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    with (bundle / "evaluation_test_indices.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol_id", "dataset", "category", "label", "partition"])
        writer.writeheader()
        writer.writerows([
            {"protocol_id": "test/bottle/good/000", "dataset": "mvtec", "category": "bottle", "label": 0, "partition": "evaluation"},
            {"protocol_id": "test/bottle/crack/001", "dataset": "mvtec", "category": "bottle", "label": 1, "partition": "evaluation"},
        ])
    with (bundle / "attack_manifest.csv").open("w", newline="") as handle:
        fields = ["scope", "source_dataset", "target_dataset", "direction", "source_label", "target_label", "loss_mode", "loss_formulation", "evaluation_attacked_image_count", "perturbation_file", "tensor_key", "artifact_sha256", "image_size", "epsilon"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "scope": "dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
            "direction": "normal_to_abnormal", "source_label": 0, "target_label": 1,
            "loss_mode": "global", "loss_formulation": "margin_topk",
            "evaluation_attacked_image_count": 1, "perturbation_file": "perturbations/delta.pt",
            "tensor_key": "delta", "artifact_sha256": digest, "image_size": 8, "epsilon": 0.1,
        })
    return bundle


def test_discovers_new_setup_contract(tmp_path):
    _write_bundle(tmp_path)
    bundles = materialize_input(tmp_path, tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))
    assert len(attacks) == 1
    assert attacks[0].record["prompt_mode"] == "learnable_prompt"
    assert attacks[0].record["setup_id"] == "steps500_eps2"
    assert attacks[0].record["loss_formulation"] == "margin_topk"
    delta, alignment = attacks[0].load()
    assert tuple(delta.shape) == (1, 3, 8, 8)
    assert alignment == {"test/bottle/good/000": 0}


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape/attack_manifest.csv", "x")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        materialize_input(archive, tmp_path / "cache")


def test_directory_and_portable_zip_are_deduplicated(tmp_path):
    bundle = _write_bundle(tmp_path)
    archive = tmp_path / "canonical_clip_per_dataset_mvtec_steps500_eps2_learnable_prompt.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for path in bundle.rglob("*"):
            if path.is_file():
                package.write(path, Path("canonical_clip_per_dataset") / path.relative_to(bundle))
    bundles = materialize_input(tmp_path, tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))
    assert len(attacks) == 1
