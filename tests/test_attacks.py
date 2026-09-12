import csv
import hashlib
from pathlib import Path
import zipfile

import pytest
import torch

from fpeval.attacks import _metadata, discover_attacks, materialize_input


def _write_bundle(
    root: Path,
    setup: str = "steps500_eps2_learnable_prompt",
    prompt_mode: str = "learnable_prompt",
    loss_formulation: str | None = "margin_topk",
) -> Path:
    """``loss_formulation=None`` omits the column, so the ID has to supply it."""
    bundle = root / "setups" / prompt_mode / setup / "canonical_clip_per_dataset"
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
        fields = ["scope", "source_dataset", "target_dataset", "direction", "source_label", "target_label", "loss_mode", "evaluation_attacked_image_count", "perturbation_file", "tensor_key", "artifact_sha256", "image_size", "epsilon"]
        row = {
            "scope": "dataset", "source_dataset": "mvtec", "target_dataset": "mvtec",
            "direction": "normal_to_abnormal", "source_label": 0, "target_label": 1,
            "loss_mode": "global",
            "evaluation_attacked_image_count": 1, "perturbation_file": "perturbations/delta.pt",
            "tensor_key": "delta", "artifact_sha256": digest, "image_size": 8, "epsilon": 0.1,
        }
        if loss_formulation is not None:
            fields.insert(7, "loss_formulation")
            row["loss_formulation"] = loss_formulation
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
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

    ep{E}[_cat{C}_img{I}]_eps{E}[_margin_topk][_full][_train{P}][_learnable_prompt],
    where the budget and epsilon grids are swept and any number may be
    fractional with a decimal point written "p". The historical steps{N}
    spelling still parses. _metadata reads only the path, so the bundle does not
    need to exist.

    A missed form is not a loud failure: _metadata falls back to
    "unspecified_setup", which relabels every condition, silently empties a
    setup_ids filter, and loses the margin_topk inference.
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
        # Per-scope PGD step counts: the name carries dataset/category/image when
        # they differ, and keeps the compact form when they agree.
        ("steps800_cat200_img100_eps2", "frozen_prompt", "steps800_cat200_img100_eps2"),
        ("steps800_cat200_img100_eps4_margin_topk",
         "frozen_prompt", "steps800_cat200_img100_eps4_margin_topk"),
        ("steps500_cat150_img50_eps4_margin_topk_learnable_prompt",
         "learnable_prompt", "steps500_cat150_img50_eps4_margin_topk"),
        ("steps800_cat200_img100_eps4_margin_topk_train20",
         "frozen_prompt", "steps800_cat200_img100_eps4_margin_topk_train20"),
        # The full split protocol adds "_full"; balanced adds no component.
        ("steps800_eps4_margin_topk_full", "frozen_prompt",
         "steps800_eps4_margin_topk_full"),
        ("steps800_eps4_margin_topk_full_train20", "frozen_prompt",
         "steps800_eps4_margin_topk_full_train20"),
        ("steps800_cat200_img100_eps4_margin_topk_full_learnable_prompt",
         "learnable_prompt", "steps800_cat200_img100_eps4_margin_topk_full"),
        ("steps500_eps2_full", "frozen_prompt", "steps500_eps2_full"),
        # Epoch budgets replaced raw step counts; a budget may be fractional.
        ("ep100_eps2", "frozen_prompt", "ep100_eps2"),
        ("ep7p14_cat100_img100_eps2_margin_topk", "frozen_prompt",
         "ep7p14_cat100_img100_eps2_margin_topk"),
        ("ep100_eps4_margin_topk_full", "frozen_prompt",
         "ep100_eps4_margin_topk_full"),
        ("ep0p5_cat2_img10_eps0p02_margin_topk_train12p5_learnable_prompt",
         "learnable_prompt",
         "ep0p5_cat2_img10_eps0p02_margin_topk_train12p5"),
        # margin_topk is the default and names nothing; ce_focal_dice names
        # itself and sits where _margin_topk used to.
        ("ep7p14_cat100_img100_eps2", "frozen_prompt",
         "ep7p14_cat100_img100_eps2"),
        ("ep7p14_cat100_img100_eps2_ce_focal_dice", "frozen_prompt",
         "ep7p14_cat100_img100_eps2_ce_focal_dice"),
        ("ep7p14_cat100_img100_eps2_ce_focal_dice_full_train20_learnable_prompt",
         "learnable_prompt",
         "ep7p14_cat100_img100_eps2_ce_focal_dice_full_train20"),
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


def test_scope_specific_step_counts_stay_distinct_setups(tmp_path):
    """Differing per-scope counts must never fold onto the uniform run."""
    def norm(directory):
        return _metadata(tmp_path / "setups" / "frozen_prompt" / directory / "bundle")

    uniform = norm("steps800_eps4_margin_topk")
    split = norm("steps800_cat200_img100_eps4_margin_topk")
    other = norm("steps800_cat200_img50_eps4_margin_topk")
    assert uniform[1] == "steps800_eps4_margin_topk"
    assert len({uniform[1], split[1], other[1]}) == 3


def test_the_full_split_protocol_never_collapses_onto_the_balanced_one(tmp_path):
    """The two protocols score different cohorts and must stay separate.

    An unmatched "_full" truncates the name rather than failing, which hands the
    full run the balanced run's setup_id and pools two different cohorts.
    """
    def norm(directory):
        return _metadata(tmp_path / "setups" / "frozen_prompt" / directory / "bundle")

    balanced = norm("steps800_eps4_margin_topk")
    full = norm("steps800_eps4_margin_topk_full")
    assert balanced[1] == "steps800_eps4_margin_topk"
    assert full[1] == "steps800_eps4_margin_topk_full"
    assert balanced != full
    # The trailing components still survive alongside it.
    assert norm("steps800_eps4_margin_topk_full_train20")[1].endswith("_full_train20")


def test_epoch_budgets_and_the_historical_step_spelling_both_parse(tmp_path):
    """Existing bundles name steps; new ones name the epoch budget.

    Both have to keep working, and an epoch budget must never be confused with
    a step count - they are different setups even at the same digits.
    """
    def norm(directory):
        return _metadata(tmp_path / "setups" / "frozen_prompt" / directory / "b")[1]

    assert norm("steps800_eps4_margin_topk") == "steps800_eps4_margin_topk"
    assert norm("ep100_eps4_margin_topk") == "ep100_eps4_margin_topk"
    assert norm("steps100_eps4") != norm("ep100_eps4")
    # A fractional budget survives intact rather than being cut at the "p".
    assert norm("ep7p14_eps2") == "ep7p14_eps2"
    assert norm("ep7p14_cat100_img100_eps2") == "ep7p14_cat100_img100_eps2"


@pytest.mark.parametrize("setup, expected", [
    # Epoch budgets: margin_topk is the default and names nothing.
    ("ep100_eps2", "margin_topk"),
    ("ep100_eps2_full", "margin_topk"),
    ("ep100_eps2_ce_focal_dice", "ce_focal_dice"),
    ("ep100_eps2_ce_focal_dice_train20", "ce_focal_dice"),
    # Step counts predate the flip, so there a bare name is ce_focal_dice and
    # margin_topk is the one that says so.
    ("steps500_eps2", "ce_focal_dice"),
    ("steps800_eps4_margin_topk", "margin_topk"),
])
def test_loss_formulation_is_inferred_from_the_id_when_absent(tmp_path, setup, expected):
    """A manifest without the column falls back to the ID.

    Which formulation a bare name implies flipped with the epoch budget, so the
    same absence means different things either side of that change.
    """
    _write_bundle(tmp_path, setup=setup, prompt_mode="frozen_prompt",
                  loss_formulation=None)
    bundles = materialize_input(tmp_path, tmp_path / "cache")
    attacks = discover_attacks(bundles, scopes=("per_dataset",), targets=("mvtec",))
    assert len(attacks) == 1
    assert attacks[0].record["loss_formulation"] == expected


def test_the_two_setup_patterns_do_not_drift(tmp_path):
    """attacks and kaggle parse the same directory names, so pin them equal."""
    from fpeval import kaggle
    from fpeval.attacks import SETUP_PATTERN

    assert SETUP_PATTERN.pattern == kaggle.SETUP_PATTERN.pattern


def test_a_partial_train_fraction_never_collapses_onto_the_full_run(tmp_path):
    # Folding _train20 away would silently pool a 20% run with a 100% run.
    def norm(directory):
        return _metadata(tmp_path / "setups" / "frozen_prompt" / directory / "bundle")

    full = norm("steps100_eps4_margin_topk")
    partial = norm("steps100_eps4_margin_topk_train20")
    assert full != partial
    assert partial[1].endswith("_train20")
