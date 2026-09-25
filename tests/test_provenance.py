"""run_config_<model>.json is written before the run, not only after it.

The failure this guards against: a run that crashes used to leave no record at
all of what it had been asked to do, because every JSON file was written after
the last metric was computed. The other half is provenance that was simply
missing - the evaluator's own commit, the checkpoint hashes, the attack
manifest each result came from.

The compatibility test is the important one. Analysis scripts read
the record by key (`run_metadata.replicate`, among others), so every key
the old version wrote must still be there with the same value.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate
from fpeval import provenance


@register_adapter("provenance_probe")
class _Probe(ModelAdapter):
    """Its own adapter: importing another test module would re-register that one."""

    name = "provenance_probe"

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

    def close(self):
        pass


@register_adapter("provenance_exploding_probe")
class _Exploding(ModelAdapter):
    """Fails during inference, i.e. after the models have been loaded."""

    name = "provenance_exploding_probe"

    def __init__(self, **kwargs):
        pass

    def runtime_metadata(self):
        return {"adapter": self.name, "repository": "/nowhere/in/particular"}

    def predict(self, images, categories):
        raise RuntimeError("inference exploded")

    def close(self):
        pass


def _mvtec(root: Path) -> Path:
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
        model="provenance_probe", model_kwargs_by_target={"mvtec": {}},
        mvtec_root=str(mvtec), targets=("mvtec",), device="cpu", image_size=8,
        batch_size=2, gaussian_sigma=0, pixel_threshold_modes=("fixed_0_5",),
        clean_only=True, create_output_archives=False,
    )
    settings.update(overrides)
    return EvaluationConfig(**settings)


def _run_config(output: Path) -> dict:
    # Named for its model, inside that model's folder.
    return json.loads((output / f"run_config_{output.name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- compatibility


def test_every_key_the_old_version_wrote_is_unchanged(tmp_path):
    """asdict(config) + resolved settings, byte-for-byte what it used to be."""
    mvtec = _mvtec(tmp_path)
    config = _config(tmp_path, mvtec)
    output = evaluate(config)
    payload = _run_config(output)

    former = asdict(config)
    for key, value in former.items():
        assert key in payload, f"the per-model record lost {key}"
        # Tuples round-trip through JSON as lists, exactly as they always did.
        expected = list(value) if isinstance(value, tuple) else value
        assert payload[key] == expected, f"{key} changed value"
    assert "resolved_model_settings_by_target" in payload
    assert payload["resolved_model_settings_by_target"]["mvtec"]["adapter"] == "provenance_probe"


def test_a_completed_run_adds_the_new_sections(tmp_path):
    mvtec = _mvtec(tmp_path)
    payload = _run_config(evaluate(_config(tmp_path, mvtec)))

    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["created_at_utc"] and payload["finished_at_utc"]
    for key in ("hostname", "python_version", "torch_version"):
        assert payload[key], f"{key} was not recorded"
    assert "cuda_version" in payload and "gpu_name" in payload
    assert set(payload["code"]) == {"evaluator", "fpeval_version", "models"}
    assert set(payload["code"]["evaluator"]) == {"path", "commit", "dirty"}
    assert payload["invocation"]["argv"]
    assert "config_file" in payload["invocation"]
    assert payload["cohort"]["images"]["mvtec"]["evaluation"]["total"] == 4


# ------------------------------------------------------------------- failure


def test_a_crash_after_model_loading_still_leaves_a_full_record(tmp_path):
    mvtec = _mvtec(tmp_path)
    config = _config(tmp_path, mvtec, model="provenance_exploding_probe")
    with pytest.raises(RuntimeError, match="inference exploded"):
        evaluate(config)

    output = Path(config.output_root) / "zero_shot" / "provenance_exploding_probe"
    payload = _run_config(output)
    assert payload["status"] == "failed"
    assert "inference exploded" in payload["error"]
    assert payload["finished_at_utc"]
    # Everything the config said, despite there being no results at all.
    for key, value in asdict(config).items():
        expected = list(value) if isinstance(value, tuple) else value
        assert payload[key] == expected, f"{key} missing or changed after a crash"
    assert not (output / "summary.csv").exists()


def test_the_record_exists_before_any_inference(tmp_path):
    """Written at the start, so even a crash during the first batch is covered."""
    mvtec = _mvtec(tmp_path)
    config = _config(tmp_path, mvtec, model="provenance_exploding_probe")
    with pytest.raises(RuntimeError):
        evaluate(config)
    output = Path(config.output_root) / "zero_shot" / "provenance_exploding_probe"
    # manifest_snapshot.json is copied up front too, for the same reason.
    assert (output / "manifest_snapshot.json").exists()
    assert not list(output.glob("*.tmp")), "atomic write left a temp file"


# ------------------------------------------------------------------- helpers


def test_write_json_replaces_atomically(tmp_path):
    target = tmp_path / "nested" / "payload.json"
    provenance.write_json(target, {"a": 1})
    provenance.write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert not list(tmp_path.rglob("*.tmp"))


def test_git_state_on_a_directory_that_is_not_a_checkout(tmp_path):
    state = provenance.git_state(tmp_path)
    assert state["commit"] is None and state["path"] == str(tmp_path)
    assert provenance.git_state(None) == {"path": None, "commit": None, "dirty": None}


def test_checkpoints_are_hashed_and_resolved_against_the_repository(tmp_path):
    repository = tmp_path / "repo"
    (repository / "weights").mkdir(parents=True)
    checkpoint = repository / "weights" / "model.pt"
    checkpoint.write_bytes(b"weights")
    found = provenance._checkpoints(
        {"repository": str(repository), "weights": ["weights/model.pt", "absent.pt"]}
    )
    by_path = {entry["path"]: entry for entry in found}
    assert by_path[str(checkpoint)]["sha256"] and len(by_path[str(checkpoint)]["sha256"]) == 64
    missing = [entry for entry in found if entry["sha256"] is None]
    assert len(missing) == 1, "an unresolvable checkpoint is recorded with a null hash"


def test_cohort_counts_are_per_target_partition_and_label(tmp_path):
    rows = [
        {"target_dataset": "mvtec", "partition": "evaluation", "protocol_id": "a", "label": 0},
        {"target_dataset": "mvtec", "partition": "evaluation", "protocol_id": "b", "label": 1},
        # The same image appears once per threshold mode; it must count once.
        {"target_dataset": "mvtec", "partition": "evaluation", "protocol_id": "b", "label": 1},
        {"target_dataset": "mvtec", "partition": "attack_train", "protocol_id": "c", "label": 1},
    ]
    manifest = [{"target_dataset": "mvtec", "protocol_split_sha256": "deadbeef"}]
    payload = provenance.cohort(rows, manifest)
    assert payload["protocol_split_sha256"] == {"mvtec": ["deadbeef"]}
    assert payload["images"]["mvtec"]["evaluation"] == {"normal": 1, "abnormal": 1, "total": 2}
    assert payload["images"]["mvtec"]["attack_train"] == {"abnormal": 1, "total": 1}
