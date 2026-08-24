"""Input-shaped paths and numerical exports for individual evaluation slices."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Iterable


GROUP_FIELDS = (
    "prompt_mode",
    "setup_id",
    "source_dataset",
    "target_dataset",
    "scope",
)


def safe_component(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not result or result in {".", ".."}:
        raise ValueError(f"Unsafe empty path component: {value!r}")
    return result


def separated_root(output_root: str | Path, model: str, selected: str | None) -> Path:
    if selected:
        return Path(selected).expanduser().resolve()
    return Path(output_root).expanduser().resolve() / f"{safe_component(model)}_separated"


def model_sibling_root(
    output_root: str | Path, model: str, suffix: str, selected: str | None
) -> Path:
    if selected:
        return Path(selected).expanduser().resolve()
    return Path(output_root).expanduser().resolve() / f"{safe_component(model)}{suffix}"


def compact_sample_condition_id(record: dict[str, Any]) -> str:
    """Name a condition using only context absent from the structured parents."""
    category = str(record.get("category") or "").strip()
    parts = [] if not category or category == "all" else [category]
    parts.extend(
        str(record.get(field) or "")
        for field in ("direction", "loss_formulation", "loss_mode")
    )
    return "__".join(safe_component(part) for part in parts)


def slice_root(root: Path, record: dict[str, Any]) -> Path:
    dataset_pair = (
        f"{safe_component(record['source_dataset'])}_to_"
        f"{safe_component(record['target_dataset'])}"
    )
    return (
        root
        / "setups"
        / safe_component(record["prompt_mode"])
        / safe_component(record["setup_id"])
        / "datasets"
        / dataset_pair
        / safe_component(record["scope"])
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(record.get(field, "")) for field in GROUP_FIELDS)


def write_separated_numerical(
    root: Path,
    *,
    summary_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    manifest_records: Iterable[dict[str, Any]],
    thresholds: dict[str, Any],
    config: Any,
) -> None:
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"summary": [], "category": [], "image": [], "manifest": []}
    )
    for name, rows in (
        ("summary", summary_rows),
        ("category", category_rows),
        ("image", image_rows),
        ("manifest", list(manifest_records)),
    ):
        for row in rows:
            grouped[_key(row)][name].append(row)

    config_payload = asdict(config)
    config_payload["resolved_model_settings_by_target"] = thresholds.get(
        "resolved_model_settings_by_target", {}
    )
    for values in grouped.values():
        reference = values["summary"][0]
        numerical = slice_root(root, reference) / "numerical"
        _write_csv(numerical / "summary.csv", values["summary"])
        _write_csv(numerical / "category_metrics.csv", values["category"])
        _write_csv(numerical / "per_image.csv", values["image"])
        (numerical / "manifest_snapshot.json").write_text(
            json.dumps(values["manifest"], indent=2, default=_json_value),
            encoding="utf-8",
        )
        target = str(reference["target_dataset"])
        threshold_slice = {
            "schema_version": thresholds.get("schema_version", 1),
            "model": thresholds.get("model"),
            "targets": {target: thresholds["targets"][target]},
            "resolved_model_settings_by_target": {
                target: thresholds.get("resolved_model_settings_by_target", {}).get(target, {})
            },
        }
        (numerical / "thresholds.json").write_text(
            json.dumps(threshold_slice, indent=2, default=_json_value),
            encoding="utf-8",
        )
        slice_config = dict(config_payload)
        slice_config["structured_slice"] = dict(zip(GROUP_FIELDS, _key(reference)))
        (numerical / "run_config.json").write_text(
            json.dumps(slice_config, indent=2, default=_json_value),
            encoding="utf-8",
        )
