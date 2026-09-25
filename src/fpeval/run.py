"""One evaluator invocation: one or more models, and the record of the whole run.

A config names either one ``model`` or several under ``models``::

    {"output_root": "...", "attacks_root": "...", "batch_size": 8, ...,
     "models": {"afclip":      {"model_kwargs_by_target": {...}},
                "anomalyclip": {"model_kwargs_by_target": {...}, "batch_size": 4}}}

Everything outside ``models`` is shared. A model entry may set only what
genuinely differs between models - how it is loaded, how large a batch fits,
which device, where its separated trees go - so every model in one run scores
the same attacks, cohort and metrics and the results stay comparable.

``<output_root>/evaluator_config.json`` is written before the first model loads
and updated as each one starts and finishes. What only one model knows stays in
``<output_root>/<regime>/<model>/run_config_<model>.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import regime
from .config import EvaluationConfig
from .engine import evaluate, output_paths, run_config_name, select_attacks
from .provenance import (
    SCHEMA_VERSION,
    _utc_now,
    attack_settings,
    code,
    environment,
    invocation,
    write_json,
)

RECORD_NAME = "evaluator_config.json"
MODEL_KEYS = (
    "model_kwargs_by_target", "batch_size", "device",
    "separated_output_root", "separated_samples_output_root",
)
# Shared paths every model would write into at once.
PER_MODEL_ONLY = ("separated_output_root", "separated_samples_output_root")


def expand(payload: dict[str, Any]) -> list[EvaluationConfig]:
    """One config per model, every one validated before anything runs."""

    if "models" not in payload:
        configs = [EvaluationConfig(**payload)]
    else:
        if "model" in payload:
            raise ValueError("Give either model or models, not both")
        models = payload["models"]
        if not isinstance(models, dict) or not models:
            raise ValueError("models must map each model name to its own settings")
        shared = {key: value for key, value in payload.items() if key != "models"}
        clashing = [key for key in PER_MODEL_ONLY if shared.get(key)]
        if clashing:
            raise ValueError(
                f"{clashing} would send every model to the same folder; "
                "set them per model instead"
            )
        # One extraction for the whole run, not one copy per model.
        if shared.get("output_root") and not shared.get("extraction_cache"):
            shared["extraction_cache"] = str(
                Path(shared["output_root"]).expanduser() / "extracted_attacks"
            )
        configs = []
        for name, own in models.items():
            own = own or {}
            extra = sorted(set(own) - set(MODEL_KEYS))
            if extra:
                raise ValueError(
                    f"{name}: {extra} must be set once for the whole run; a model "
                    f"may set only {list(MODEL_KEYS)}"
                )
            configs.append(EvaluationConfig(**{**shared, **own, "model": name}))
    roots = {Path(config.output_root).expanduser().resolve() for config in configs}
    if len(roots) != 1:
        raise ValueError("Every model of one run writes under one output_root")
    for config in configs:
        regime(config.model)  # an unknown model fails here, before any work
    return configs


def _outputs(config: EvaluationConfig) -> dict[str, Any]:
    """What this model's run will write, and where - resolved, not as configured."""

    paths = output_paths(config)
    model_dir = paths["model"]
    separated = config.write_separated_results and not config.clean_only
    samples = config.save_qualitative_samples and not config.clean_only
    archives = [model_dir.with_suffix(".zip")] if config.create_output_archives else []
    if config.create_output_archives and samples:
        archives.append(paths["samples"].with_suffix(".zip"))
    if config.create_output_archives and separated:
        archives.append(paths["separated"].with_suffix(".zip"))
    return {
        "directory": str(model_dir),
        "files": [str(model_dir / name) for name in (
            "summary.csv", "category_metrics.csv", "per_image.csv",
            "thresholds.json", run_config_name(config.model), "manifest_snapshot.json",
        )],
        "predictions": str(model_dir / "predictions") if config.save_predictions else None,
        "separated_results": str(paths["separated"]) if separated else None,
        "qualitative_samples": str(paths["samples"]) if samples else None,
        "archives": [str(path) for path in archives],
        "extraction_cache": (
            str(paths["extraction_cache"]) if config.attacks_root is not None else None
        ),
    }


def _set_aside(path: Path) -> None:
    """Keep an earlier run's record: other results under this root still need it."""

    if not path.exists():
        return
    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get("created_at_utc")
    except (OSError, ValueError, AttributeError):
        stamp = None
    stamp = str(stamp or _utc_now()).replace(":", "").replace("+", "_")
    path.replace(path.with_name(f"evaluator_config.{stamp}.json"))


def run(configs: list[EvaluationConfig]) -> list[Path]:
    """Evaluate every model in turn, recording the run before the first starts.

    A model that fails is recorded and the next one still runs: on a long
    multi-model run one broken adapter should not cost every model after it.
    The run itself fails at the end if any model did.
    """

    for config in configs:
        existing = output_paths(config)["model"] / "summary.csv"
        if existing.exists() and not config.overwrite:
            raise FileExistsError(f"Results already exist: {existing}; set overwrite=true")

    first = configs[0]
    path = Path(first.output_root).expanduser().resolve() / RECORD_NAME
    shared = {
        key: value for key, value in asdict(first).items()
        if key not in MODEL_KEYS and key != "model"
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": _utc_now(),
        "finished_at_utc": None,
        **environment(),
        "code": code(),
        "invocation": invocation(),
        "models": [config.model for config in configs],
        "settings": shared,
        "per_model": {
            config.model: {
                "regime": regime(config.model),
                **{key: getattr(config, key) for key in MODEL_KEYS},
                "run_config": str(output_paths(config)["model"] / run_config_name(config.model)),
                "outputs": _outputs(config),
                "status": "pending",
                "started_at_utc": None,
                "finished_at_utc": None,
                "error": None,
            }
            for config in configs
        },
        # Every model shares the selection, so the attacks are read once.
        "attack_settings": attack_settings(select_attacks(first)),
    }
    # Only once the new record is complete, so a failure while building it
    # never leaves the root without one.
    _set_aside(path)
    write_json(path, payload)

    outputs: list[Path] = []
    failed: dict[str, BaseException] = {}
    try:
        for config in configs:
            entry = payload["per_model"][config.model]
            entry.update(status="running", started_at_utc=_utc_now())
            write_json(path, payload)
            try:
                outputs.append(evaluate(config))
                entry["status"] = "completed"
            except Exception as error:  # noqa: BLE001 - recorded, run continues
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                failed[config.model] = error
            finally:
                entry["finished_at_utc"] = _utc_now()
                write_json(path, payload)
    except BaseException as error:  # interrupted: the record must still close
        for entry in payload["per_model"].values():
            if entry["status"] == "running":
                entry["status"] = "failed"
        payload.update(status="failed", finished_at_utc=_utc_now(),
                       error=f"{type(error).__name__}: {error}")
        write_json(path, payload)
        raise
    payload.update(status="failed" if failed else "completed", finished_at_utc=_utc_now())
    if failed:
        payload["error"] = f"{len(failed)} of {len(configs)} models failed: {sorted(failed)}"
    write_json(path, payload)
    if failed:
        raise RuntimeError(payload["error"]) from next(iter(failed.values()))
    return outputs
