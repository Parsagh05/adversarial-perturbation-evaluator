import csv
import hashlib
from pathlib import Path
import zipfile

import pytest
import torch

from fpeval.attacks import _metadata, discover_attacks, materialize_input


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


def test_duplicate_condition_without_manifest_checksum_is_deduplicated(tmp_path):
    bundle = _write_bundle(tmp_path)
    # Drop the recorded checksum from both copies of the same artifact.
    manifest = bundle / "attack_manifest.csv"
    rows = list(csv.DictReader(manifest.open(newline="")))
    fields = list(rows[0])
    rows[0]["artifact_sha256"] = ""
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    archive = tmp_path / "canonical_clip_per_dataset_mvtec_steps500_eps2_learnable_prompt.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for path in bundle.rglob("*"):
            if path.is_file():
                package.write(path, Path("canonical_clip_per_dataset") / path.relative_to(bundle))
    bundles = materialize_input(tmp_path, tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))
    assert len(attacks) == 1


def test_cross_dataset_scope_uses_the_whole_target_cohort(tmp_path):
    bundle = _write_bundle(tmp_path)
    manifest = bundle / "attack_manifest.csv"
    rows = list(csv.DictReader(manifest.open(newline="")))
    fields = list(rows[0])
    rows[0]["scope"] = "cross_dataset"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    attacks = discover_attacks([bundle], scopes=("cross_dataset",), targets=("mvtec",))
    assert len(attacks) == 1
    attack = attacks[0]
    assert attack.record["scope"] == "cross_dataset"
    # A dataset-level scope must not be filtered down to one category.
    assert len(attack.evaluation_ids) == 2
    assert attack.record["tensor_key"] == "delta"


def test_setup_id_normalization_covers_the_generator_grammar(tmp_path):
    """Mirror setup_catalog.compose_setup_id.

    steps{N}_eps{E}[_margin_topk][_train{P}][_learnable_prompt], where the step
    and epsilon grids are swept and a decimal point becomes "p". _metadata reads
    only the path, so the bundle does not need to exist.
    """
    cases = [
        # (directory, expected prompt_mode, expected normalized setup_id)
        ("steps500_eps2", "frozen_prompt", "steps500_eps2"),
        ("steps800_eps4_margin_topk", "frozen_prompt", "steps800_eps4_margin_topk"),
        ("steps500_eps2_learnable_prompt", "learnable_prompt", "steps500_eps2"),
        ("steps800_eps4_margin_topk_learnable_prompt",
         "learnable_prompt", "steps800_eps4_margin_topk"),
        # swept grids: any step count, any epsilon
        ("steps1200_eps8", "frozen_prompt", "steps1200_eps8"),
        ("steps250_eps0p02", "frozen_prompt", "steps250_eps0p02"),
        # a partial attack-train fraction is its own setup, never folded away
        ("steps100_eps4_train20", "frozen_prompt", "steps100_eps4_train20"),
        ("steps100_eps4_margin_topk_train20",
         "frozen_prompt", "steps100_eps4_margin_topk_train20"),
        ("steps250_eps0p02_margin_topk_train12p5_learnable_prompt",
         "learnable_prompt", "steps250_eps0p02_margin_topk_train12p5"),
    ]
    seen = set()
    for directory, expected_mode, expected_id in cases:
        prompt_dir = ("learnable_prompt" if directory.endswith("_learnable_prompt")
                      else "frozen_prompt")
        bundle = (tmp_path / "setups" / prompt_dir / directory
                  / "canonical_clip_cross_dataset")
        assert _metadata(bundle) == (expected_mode, expected_id), directory
        seen.add((expected_mode, expected_id))
    assert len(seen) == len(cases)


def test_a_partial_train_fraction_never_collapses_onto_the_full_run(tmp_path):
    # Folding _train20 away would silently pool a 20% run with a 100% run.
    def norm(directory):
        return _metadata(tmp_path / "setups" / "frozen_prompt" / directory / "bundle")

    full = norm("steps100_eps4_margin_topk")
    partial = norm("steps100_eps4_margin_topk_train20")
    assert full != partial
    assert partial[1].endswith("_train20")
