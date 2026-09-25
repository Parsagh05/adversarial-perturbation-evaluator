"""<output_root>/evaluator_config.json: one record for the whole invocation.

It is written before the first model loads and names every model of the run,
the settings they share, what each writes and where, and how every evaluated
attack bundle was set up. What only one model knows stays in its own
run_config_<model>.json.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.provenance import attack_settings
from fpeval.run import RECORD_NAME, expand, run


def _probe(name, explode=False):
    @register_adapter(name)
    class _Probe(ModelAdapter):
        def __init__(self, **kwargs):
            pass

        def predict(self, images, categories):
            if explode:
                raise RuntimeError("inference exploded")
            return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

        def close(self):
            pass

    _Probe.name = name
    return _Probe


_probe("config_probe_a")
_probe("config_probe_b")
_probe("config_probe_exploding", explode=True)


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


def _payload(tmp_path: Path, models, **overrides) -> dict:
    payload = dict(
        attacks_root=None, output_root=str(tmp_path / "results"),
        mvtec_root=str(_mvtec(tmp_path)), targets=["mvtec"], device="cpu",
        image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=["fixed_0_5"], clean_only=True,
        create_output_archives=False,
        models={name: {"model_kwargs_by_target": {"mvtec": {}}} for name in models},
    )
    payload.update(overrides)
    return payload


def _record(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "results" / RECORD_NAME).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the record


def test_the_record_sits_at_the_output_root_and_names_every_model(tmp_path):
    payload = _payload(tmp_path, ["config_probe_a", "config_probe_b"])
    payload["models"]["config_probe_b"]["batch_size"] = 1
    run(expand(payload))

    record = _record(tmp_path)
    assert record["status"] == "completed"
    assert record["models"] == ["config_probe_a", "config_probe_b"]
    # Shared settings once; what differs per model under that model.
    assert record["settings"]["image_size"] == 8
    assert record["settings"]["output_root"] == payload["output_root"]
    assert "batch_size" not in record["settings"]
    assert record["per_model"]["config_probe_a"]["batch_size"] == 2
    assert record["per_model"]["config_probe_b"]["batch_size"] == 1
    for name, entry in record["per_model"].items():
        assert entry["status"] == "completed"
        assert entry["regime"] == "zero_shot"
        assert entry["started_at_utc"] and entry["finished_at_utc"]
        # The model-specific record, next to that model's results.
        assert entry["run_config"].endswith(f"run_config_{name}.json")
        assert Path(entry["run_config"]).is_file()


def test_every_output_the_record_announces_is_actually_written(tmp_path):
    """The paths are resolved in advance; a wrong one would be a lie."""
    run(expand(_payload(tmp_path, ["config_probe_a"])))
    outputs = _record(tmp_path)["per_model"]["config_probe_a"]["outputs"]
    assert Path(outputs["directory"]).is_dir()
    for file in outputs["files"]:
        assert Path(file).is_file(), file
    # clean_only writes no separated tree, no samples and here no archives.
    assert outputs["separated_results"] is None
    assert outputs["qualitative_samples"] is None
    assert outputs["archives"] == []


def test_a_single_model_config_writes_the_same_record(tmp_path):
    payload = _payload(tmp_path, [])
    del payload["models"]
    payload.update(model="config_probe_a", model_kwargs_by_target={"mvtec": {}})
    run(expand(payload))
    record = _record(tmp_path)
    assert record["models"] == ["config_probe_a"]
    assert record["per_model"]["config_probe_a"]["status"] == "completed"


# ------------------------------------------------------------------- failures


def test_one_failing_model_is_recorded_and_the_rest_still_run(tmp_path):
    payload = _payload(tmp_path, ["config_probe_exploding", "config_probe_a"])
    with pytest.raises(RuntimeError, match="1 of 2 models failed"):
        run(expand(payload))

    record = _record(tmp_path)
    assert record["status"] == "failed"
    exploded = record["per_model"]["config_probe_exploding"]
    assert exploded["status"] == "failed"
    assert "inference exploded" in exploded["error"]
    assert record["per_model"]["config_probe_a"]["status"] == "completed"
    assert Path(record["per_model"]["config_probe_a"]["outputs"]["files"][0]).is_file()


@pytest.mark.parametrize("bad, message", [
    ({"image_size": 16}, "must be set once for the whole run"),
    ({"scopes": ["per_dataset"]}, "must be set once for the whole run"),
])
def test_a_model_may_not_change_what_makes_models_comparable(tmp_path, bad, message):
    payload = _payload(tmp_path, ["config_probe_a"])
    payload["models"]["config_probe_a"].update(bad)
    with pytest.raises(ValueError, match=message):
        expand(payload)


def test_a_bad_model_fails_before_anything_is_written(tmp_path):
    payload = _payload(tmp_path, ["config_probe_a", "no_such_model"])
    with pytest.raises(ValueError, match="no_such_model"):
        run(expand(payload))
    assert not (tmp_path / "results").exists()


def test_existing_results_for_a_later_model_stop_the_run_at_the_start(tmp_path):
    payload = _payload(tmp_path, ["config_probe_a", "config_probe_b"])
    later = tmp_path / "results" / "zero_shot" / "config_probe_b"
    later.mkdir(parents=True)
    (later / "summary.csv").write_text("done\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(expand(payload))
    assert not (tmp_path / "results" / "zero_shot" / "config_probe_a").exists()
    assert not (tmp_path / "results" / RECORD_NAME).exists()


def test_an_earlier_record_is_kept_not_overwritten(tmp_path):
    run(expand(_payload(tmp_path, ["config_probe_a"])))
    first = _record(tmp_path)
    run(expand(_payload(tmp_path, ["config_probe_b"])))
    kept = [path for path in (tmp_path / "results").glob("evaluator_config.*.json")]
    assert len(kept) == 1
    assert json.loads(kept[0].read_text(encoding="utf-8"))["models"] == first["models"]
    assert _record(tmp_path)["models"] == ["config_probe_b"]


def test_the_example_multi_model_config_expands():
    payload = json.loads(Path("configs/multi_model.example.json").read_text(encoding="utf-8"))
    # Its paths are placeholders; only the shape is under test.
    payload.update(attacks_root=None, clean_only=True)
    configs = {config.model: config for config in expand(payload)}
    assert sorted(configs) == ["afclip", "anomalyclip"]
    assert configs["afclip"].batch_size == 8
    assert configs["anomalyclip"].batch_size == 4
    assert configs["afclip"].extraction_cache.replace("\\", "/").endswith(
        "results/extracted_attacks")


# ------------------------------------------------------------ attack settings


def _attack(bundle: Path, **record):
    return SimpleNamespace(bundle=bundle, record=record)


def test_attack_settings_copy_the_generator_and_keep_k_per_direction(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "generation_config.json").write_text(json.dumps({
        "setup": {"setup_id": "ep100_eps4_sga_fullcross"},
        "hyperparameters": {"OPTIMIZER": ["sga", 2, 4], "AMP_DTYPE_NAME": "bfloat16"},
        "execution": {"snapshot_epochs": []},
        "environment": {"HOME": "/root"},        # the generator's run, not the setup
    }), encoding="utf-8")
    attacks = [
        _attack(bundle, setup_id="ep100_eps4_sga_fullcross", direction="normal_to_abnormal",
                margin_topk_fraction="0.1", category="bottle", epsilon="0.0157"),
        _attack(bundle, setup_id="ep100_eps4_sga_fullcross", direction="normal_to_abnormal",
                margin_topk_fraction="0.1", category="cable", epsilon="0.0157"),
        _attack(bundle, setup_id="ep100_eps4_sga_fullcross", direction="abnormal_to_normal",
                margin_topk_fraction="0.2", category="bottle", epsilon="0.0157"),
    ]
    [entry] = attack_settings(attacks)

    generation = entry["generation_config"]
    assert generation["sha256"] and len(generation["sha256"]) == 64
    assert generation["hyperparameters"]["OPTIMIZER"] == ["sga", 2, 4]
    assert generation["hyperparameters"]["AMP_DTYPE_NAME"] == "bfloat16"
    assert "environment" not in generation

    manifest = entry["manifest"]
    assert entry["conditions"] == 3
    assert manifest["constant"]["epsilon"] == "0.0157"
    assert manifest["varying"]["margin_topk_fraction"] == {"by_direction": {
        "abnormal_to_normal": "0.2", "normal_to_abnormal": "0.1"}}
    assert manifest["varying"]["category"] == {"values": ["bottle", "cable"]}


def test_a_bundle_without_generation_config_still_records_its_manifest(tmp_path):
    [entry] = attack_settings([_attack(tmp_path, setup_id="ep10_eps4", direction="n")])
    assert entry["generation_config"] is None
    assert entry["manifest"]["constant"]["setup_id"] == "ep10_eps4"
