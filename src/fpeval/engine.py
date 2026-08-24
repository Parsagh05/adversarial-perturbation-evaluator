"""End-to-end manifest-driven evaluation engine."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .adapters import create_adapter
from .archive import archive_directory
from .attacks import Attack, discover_attacks, materialize_input
from .config import EvaluationConfig
from .data import Sample, discover_dataset, load_image, load_mask, sample_index
from .metrics import (
    classification,
    optimal_f1,
    performance,
    pixel_performance,
    targeted_images,
    targeted_pixels,
    topk_region,
)
from .qualitative import export_samples
from .structured import (
    compact_sample_condition_id,
    model_sibling_root,
    separated_root,
    slice_root,
    write_separated_numerical,
)


Prediction = tuple[float, np.ndarray]


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict_clean(adapter: Any, samples: list[Sample], config: EvaluationConfig) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for batch in tqdm(list(_chunks(samples, config.batch_size)), desc="clean inference", leave=False):
        images = torch.stack([load_image(sample, config.image_size) for sample in batch])
        scores, maps = adapter.predict(images, [sample.category for sample in batch])
        if scores.shape != (len(batch),) or len(maps) != len(batch):
            raise ValueError("Adapter returned an invalid batch shape")
        for sample, score, anomaly_map in zip(batch, scores, maps):
            array = np.asarray(anomaly_map, dtype=np.float32)
            if array.ndim != 2 or not np.isfinite(array).all() or not np.isfinite(score):
                raise ValueError("Adapter predictions must be finite [B] and [B,H,W]")
            predictions[sample.protocol_id] = (float(score), array)
    return predictions


def _predict_attacked(
    adapter: Any,
    samples: list[Sample],
    delta: torch.Tensor,
    delta_index: dict[str, int],
    config: EvaluationConfig,
) -> tuple[dict[str, Prediction], dict[str, float]]:
    predictions: dict[str, Prediction] = {}
    distances: dict[str, float] = {}
    for batch in tqdm(list(_chunks(samples, config.batch_size)), desc="adversarial inference", leave=False):
        clean = torch.stack([load_image(sample, config.image_size) for sample in batch])
        perturbations = torch.stack([delta[delta_index[sample.protocol_id]] for sample in batch])
        adversarial = (clean + perturbations).clamp(0, 1)
        linf = (adversarial - clean).abs().flatten(1).amax(dim=1).numpy()
        scores, maps = adapter.predict(adversarial, [sample.category for sample in batch])
        for sample, score, anomaly_map, distance in zip(batch, scores, maps, linf):
            array = np.asarray(anomaly_map, dtype=np.float32)
            if array.ndim != 2 or not np.isfinite(array).all() or not np.isfinite(score):
                raise ValueError("Adapter predictions must be finite [B] and [B,H,W]")
            predictions[sample.protocol_id] = (float(score), array)
            distances[sample.protocol_id] = float(distance)
    return predictions, distances


def _resize_maps(maps: Sequence[np.ndarray], size: int, sigma: float) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(maps).astype(np.float32))[:, None]
    output = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    if sigma > 0:
        output = np.stack([gaussian_filter(item, sigma=sigma) for item in output])
    return output.astype(np.float32)


def _postprocess_predictions(
    adapter: Any,
    samples: list[Sample],
    predictions: dict[str, Prediction],
    *,
    reference_samples: list[Sample] | None = None,
    reference_predictions: dict[str, Prediction] | None = None,
) -> dict[str, Prediction]:
    """Run optional model-specific cohort normalization and score aggregation."""

    raw_maps = np.stack([predictions[sample.protocol_id][1] for sample in samples])
    categories = [sample.category for sample in samples]
    if reference_samples is None or reference_predictions is None:
        processed_maps = adapter.postprocess_anomaly_maps(raw_maps, categories)
    else:
        reference_raw_maps = np.stack(
            [reference_predictions[sample.protocol_id][1] for sample in reference_samples]
        )
        processed_maps = adapter.postprocess_anomaly_maps_with_reference(
            raw_maps,
            categories,
            reference_maps=reference_raw_maps,
            reference_categories=[sample.category for sample in reference_samples],
        )
    processed_maps = np.asarray(processed_maps, dtype=np.float32)
    if processed_maps.shape != raw_maps.shape or not np.isfinite(processed_maps).all():
        raise ValueError("Adapter map postprocessing must preserve shape and finite values")
    scores = np.asarray(
        [predictions[sample.protocol_id][0] for sample in samples], dtype=np.float32
    )
    maps = list(processed_maps)
    map_mins = np.asarray([item.min() for item in maps], dtype=np.float32)
    map_maxs = np.asarray([item.max() for item in maps], dtype=np.float32)
    if reference_samples is None or reference_predictions is None:
        processed = adapter.postprocess_image_scores(
            scores, map_mins, map_maxs, categories
        )
    else:
        reference_scores = np.asarray(
            [reference_predictions[sample.protocol_id][0] for sample in reference_samples],
            dtype=np.float32,
        )
        reference_maps = adapter.postprocess_anomaly_maps(
            np.stack(
                [
                    reference_predictions[sample.protocol_id][1]
                    for sample in reference_samples
                ]
            ),
            [sample.category for sample in reference_samples],
        )
        processed = adapter.postprocess_image_scores_with_reference(
            scores,
            map_mins,
            map_maxs,
            categories,
            reference_scores=reference_scores,
            reference_map_mins=np.asarray(
                [item.min() for item in reference_maps], dtype=np.float32
            ),
            reference_map_maxs=np.asarray(
                [item.max() for item in reference_maps], dtype=np.float32
            ),
            reference_categories=[sample.category for sample in reference_samples],
        )
    processed = np.asarray(processed, dtype=np.float32)
    if processed.shape != (len(samples),) or not np.isfinite(processed).all():
        raise ValueError("Adapter score postprocessing must return one finite score per sample")
    return {
        sample.protocol_id: (float(score), anomaly_map)
        for sample, score, anomaly_map in zip(samples, processed, processed_maps)
    }


def _fixed_region(record: dict[str, Any], size: int) -> np.ndarray:
    mode = str(record.get("normal_local_target") or "fixed_region")
    if mode == "full_image":
        return np.ones((size, size), dtype=bool)
    if mode != "fixed_region":
        raise ValueError(f"Unknown normal_local_target: {mode}")
    fraction = float(record.get("normal_target_region_fraction") or 0.25)
    center_x = float(record.get("normal_target_center_x") or 0.5)
    center_y = float(record.get("normal_target_center_y") or 0.5)
    if not 0 < fraction <= 1 or not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
        raise ValueError("Invalid fixed target-region metadata")
    height = max(1, round(size * fraction))
    width = max(1, round(size * fraction))
    row = round(center_y * (size - 1))
    column = round(center_x * (size - 1))
    top = min(max(row - height // 2, 0), size - height)
    left = min(max(column - width // 2, 0), size - width)
    result = np.zeros((size, size), dtype=bool)
    result[top : top + height, left : left + width] = True
    return result


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
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _condition_fields(attack: Attack) -> dict[str, Any]:
    names = (
        "prompt_mode", "setup_id", "source_dataset", "target_dataset", "scope",
        "category", "direction", "source_label", "target_label", "loss_formulation",
        "loss_mode", "epsilon", "image_size", "optimization_steps",
        "margin_topk_fraction", "prompt_provenance", "normal_local_target",
        "normal_target_region_fraction", "normal_target_center_x",
        "normal_target_center_y",
    )
    return {name: attack.record.get(name, "") for name in names}


def _calibrate_clean(
    target: str,
    samples: list[Sample],
    clean: dict[str, Prediction],
    config: EvaluationConfig,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for category in sorted({sample.category for sample in samples}):
        cohort = [sample for sample in samples if sample.category == category]
        labels = np.asarray([sample.label for sample in cohort], dtype=np.uint8)
        scores = np.asarray([clean[sample.protocol_id][0] for sample in cohort], dtype=np.float32)
        image_op = optimal_f1(labels, scores)
        maps = _resize_maps(
            [clean[sample.protocol_id][1] for sample in cohort],
            config.image_size, config.gaussian_sigma,
        )
        masks = np.stack([load_mask(sample, config.image_size) for sample in cohort])
        pixel_labels = masks.reshape(-1)
        pixel_scores = maps.reshape(-1)
        pixel_op = optimal_f1(pixel_labels, pixel_scores)
        thresholds[category] = {
            "image_f1": image_op["threshold"],
            "clean_pixel_f1": pixel_op["threshold"],
            "image_f1_max": 100 * image_op["f1"],
            "pixel_f1_max": 100 * pixel_op["f1"],
            "sample_count": len(cohort),
        }
    return thresholds


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    values = np.asarray([row.get(name, np.nan) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _evaluate_condition(
    attack: Attack,
    samples_by_id: dict[str, Sample],
    raw_clean_cache: dict[str, Prediction],
    adapter: Any,
    thresholds: dict[str, dict[str, float]],
    config: EvaluationConfig,
    clean_metrics: dict[tuple[str, ...], dict[str, dict[str, float]]] | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    dict[str, np.ndarray], torch.Tensor, dict[str, int],
]:
    cohort = [samples_by_id[sample_id] for sample_id in attack.evaluation_ids]
    # Clean predictions and their postprocessing depend only on the cohort, so
    # identical cohorts across conditions reuse one set of clean-side metrics.
    cohort_key = tuple(sample.protocol_id for sample in cohort)
    if clean_metrics is None:
        clean_metrics = {}
    clean_metric_cache = clean_metrics.setdefault(cohort_key, {})
    clean_cache = _postprocess_predictions(adapter, cohort, raw_clean_cache)
    attacked_set = set(attack.attacked_ids)
    attacked_samples = [sample for sample in cohort if sample.protocol_id in attacked_set]
    delta, delta_index = attack.load(verify_checksum=config.verify_checksums)
    adversarial_only, linf = _predict_attacked(
        adapter, attacked_samples, delta, delta_index, config
    )
    raw_adversarial = {
        sample.protocol_id: adversarial_only.get(
            sample.protocol_id, raw_clean_cache[sample.protocol_id]
        )
        for sample in cohort
    }
    adversarial = _postprocess_predictions(
        adapter,
        cohort,
        raw_adversarial,
        reference_samples=cohort,
        reference_predictions=raw_clean_cache,
    )
    linf = {sample.protocol_id: linf.get(sample.protocol_id, 0.0) for sample in cohort}
    category_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    base = _condition_fields(attack)
    base["condition_id"] = attack.condition_id

    for category in sorted({sample.category for sample in cohort}):
        category_samples = [sample for sample in cohort if sample.category == category]
        labels = np.asarray([sample.label for sample in category_samples], dtype=np.uint8)
        attacked = np.asarray([sample.protocol_id in attacked_set for sample in category_samples], dtype=bool)
        clean_scores = np.asarray([clean_cache[sample.protocol_id][0] for sample in category_samples], dtype=np.float32)
        adversarial_scores = np.asarray([adversarial[sample.protocol_id][0] for sample in category_samples], dtype=np.float32)
        clean_maps = _resize_maps(
            [clean_cache[sample.protocol_id][1] for sample in category_samples],
            config.image_size, config.gaussian_sigma,
        )
        adversarial_maps = _resize_maps(
            [adversarial[sample.protocol_id][1] for sample in category_samples],
            config.image_size, config.gaussian_sigma,
        )
        masks = np.stack([load_mask(sample, config.image_size) for sample in category_samples])
        cached = clean_metric_cache.get(category)
        if cached is None:
            cached = {
                "image": performance(labels, clean_scores),
                "pixel": pixel_performance(
                    masks, clean_maps, fpr_limit=config.aupro_fpr_limit,
                    thresholds=config.aupro_thresholds,
                ),
            }
            clean_metric_cache[category] = cached
        clean_image_perf, clean_pixel_perf = cached["image"], cached["pixel"]
        adversarial_image_perf = performance(labels, adversarial_scores)
        adversarial_pixel_perf = pixel_performance(
            masks, adversarial_maps, fpr_limit=config.aupro_fpr_limit,
            thresholds=config.aupro_thresholds,
        )
        image_threshold = thresholds[category]["image_f1"]
        clean_image_pred = (clean_scores >= image_threshold).astype(np.uint8)
        adversarial_image_pred = (adversarial_scores >= image_threshold).astype(np.uint8)
        target_image_metrics = targeted_images(
            clean_image_pred, adversarial_image_pred, attacked,
            source=int(attack.record["source_label"]), target=int(attack.record["target_label"]),
        )

        for pixel_mode in config.pixel_threshold_modes:
            if pixel_mode == "fixed_0_5":
                pixel_threshold = 0.5
            else:
                pixel_threshold = thresholds[category][pixel_mode]
            pixel_records: list[dict[str, float | int]] = []
            topk_records: list[dict[str, float | int]] = []
            for index, sample in enumerate(category_samples):
                if sample.protocol_id not in attacked_set:
                    pixel = {name: 0 for name in (
                        "pixel_count", "pixel_eligible_count", "pixel_flip_count",
                        "pixel_success_eligible", "pixel_attack_success",
                    )}
                    pixel["pixel_flip_rate"] = np.nan
                    topk = dict(pixel)
                else:
                    region = (
                        _fixed_region(attack.record, config.image_size)
                        if int(attack.record["target_label"]) == 1
                        else masks[index].astype(bool)
                    )
                    pixel = targeted_pixels(
                        clean_maps[index], adversarial_maps[index], region,
                        threshold=pixel_threshold,
                        source=int(attack.record["source_label"]),
                        target=int(attack.record["target_label"]),
                        minimum_fraction=config.pixel_success_min_flip_fraction,
                    )
                    if int(attack.record["target_label"]) == 1:
                        topk = targeted_pixels(
                            clean_maps[index], adversarial_maps[index],
                            topk_region(adversarial_maps[index], config.location_free_topk_fraction),
                            threshold=pixel_threshold, source=0, target=1,
                            minimum_fraction=config.pixel_success_min_flip_fraction,
                        )
                    else:
                        topk = {name: np.nan for name in pixel}
                pixel_records.append(pixel)
                topk_records.append(topk)
                direction_sign = 1 if int(attack.record["target_label"]) == 1 else -1
                per_image_rows.append({
                    **base, "pixel_threshold_mode": pixel_mode,
                    "pixel_threshold": pixel_threshold, "protocol_id": sample.protocol_id,
                    "dataset": sample.dataset, "category": sample.category,
                    "defect_type": sample.defect_type, "label": sample.label,
                    "attacked": int(attacked[index]), "clean_score": float(clean_scores[index]),
                    "adversarial_score": float(adversarial_scores[index]),
                    "clean_prediction": int(clean_image_pred[index]),
                    "adversarial_prediction": int(adversarial_image_pred[index]),
                    "image_flip": int(clean_image_pred[index] != adversarial_image_pred[index]),
                    "targeted_image_success": int(
                        attacked[index] and clean_image_pred[index] == int(attack.record["source_label"])
                        and adversarial_image_pred[index] == int(attack.record["target_label"])
                        and clean_image_pred[index] != adversarial_image_pred[index]
                    ),
                    "target_direction_score_shift": direction_sign * float(adversarial_scores[index] - clean_scores[index]),
                    "target_direction_map_shift": direction_sign * float((adversarial_maps[index] - clean_maps[index]).mean()),
                    "realized_linf": linf[sample.protocol_id],
                    "location_free_topk_fraction": config.location_free_topk_fraction,
                    **{f"target_region_{key}": value for key, value in pixel.items()},
                    **{f"location_free_topk_{key}": value for key, value in topk.items()},
                })

            attacked_linf = [
                {"realized_linf": linf[sample.protocol_id]}
                for sample in category_samples
                if sample.protocol_id in attacked_set
            ]
            clean_class = classification(labels, clean_image_pred)
            adversarial_class = classification(labels, adversarial_image_pred)
            region_eligible = sum(int(row["pixel_eligible_count"]) for row in pixel_records)
            region_flips = sum(int(row["pixel_flip_count"]) for row in pixel_records)
            region_success_eligible = sum(int(row["pixel_success_eligible"]) for row in pixel_records)
            region_successes = sum(int(row["pixel_attack_success"]) for row in pixel_records)
            topk_valid = [row for row in topk_records if np.isfinite(float(row["pixel_eligible_count"]))]
            topk_eligible = sum(int(row["pixel_eligible_count"]) for row in topk_valid)
            topk_flips = sum(int(row["pixel_flip_count"]) for row in topk_valid)
            topk_success_eligible = sum(int(row["pixel_success_eligible"]) for row in topk_valid)
            topk_successes = sum(int(row["pixel_attack_success"]) for row in topk_valid)
            row: dict[str, Any] = {
                **base, "category": category, "pixel_threshold_mode": pixel_mode,
                "image_threshold": image_threshold, "pixel_threshold": pixel_threshold,
                "sample_count": len(category_samples), "attacked_count": int(attacked.sum()),
                "clean_i_auroc": clean_image_perf["auroc"],
                "clean_i_ap": clean_image_perf["ap"],
                "clean_i_f1_max": clean_image_perf["f1_max"],
                "adversarial_i_auroc": adversarial_image_perf["auroc"],
                "adversarial_i_ap": adversarial_image_perf["ap"],
                "adversarial_i_f1_max": adversarial_image_perf["f1_max"],
                "delta_i_auroc": clean_image_perf["auroc"] - adversarial_image_perf["auroc"],
                "delta_i_ap": clean_image_perf["ap"] - adversarial_image_perf["ap"],
                "delta_i_f1_max": clean_image_perf["f1_max"] - adversarial_image_perf["f1_max"],
                **{f"clean_{name}": value for name, value in clean_pixel_perf.items()},
                **{f"adversarial_{name}": value for name, value in adversarial_pixel_perf.items()},
                "delta_p_auroc": clean_pixel_perf["p_auroc"] - adversarial_pixel_perf["p_auroc"],
                "delta_p_f1_max": clean_pixel_perf["p_f1_max"] - adversarial_pixel_perf["p_f1_max"],
                "delta_aupro": clean_pixel_perf["aupro"] - adversarial_pixel_perf["aupro"],
                **{f"clean_{name}": value for name, value in clean_class.items()},
                **{f"adversarial_{name}": value for name, value in adversarial_class.items()},
                **target_image_metrics,
                "target_region_pixel_flip_rate_macro": _mean(pixel_records, "pixel_flip_rate"),
                "target_region_pixel_flip_rate_micro": 100 * region_flips / region_eligible if region_eligible else np.nan,
                "target_region_pixel_attack_success_rate_macro": 100 * region_successes / region_success_eligible if region_success_eligible else np.nan,
                "target_region_pixel_eligible_count": region_eligible,
                "target_region_pixel_flip_count": region_flips,
                "target_region_pixel_success_eligible_count": region_success_eligible,
                "target_region_pixel_success_count": region_successes,
                "location_free_topk_fraction": config.location_free_topk_fraction,
                "location_free_topk_pixel_flip_rate_macro": _mean(topk_valid, "pixel_flip_rate"),
                "location_free_topk_pixel_flip_rate_micro": 100 * topk_flips / topk_eligible if topk_eligible else np.nan,
                "location_free_topk_pixel_attack_success_rate_macro": 100 * topk_successes / topk_success_eligible if topk_success_eligible else np.nan,
                "location_free_topk_pixel_eligible_count": topk_eligible,
                "location_free_topk_pixel_flip_count": topk_flips,
                "location_free_topk_pixel_success_eligible_count": topk_success_eligible,
                "location_free_topk_pixel_success_count": topk_successes,
                "realized_linf_mean": _mean(attacked_linf, "realized_linf"),
                "realized_linf_max": (
                    float(np.max([row["realized_linf"] for row in attacked_linf]))
                    if attacked_linf else np.nan
                ),
            }
            category_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for pixel_mode in config.pixel_threshold_modes:
        rows = [row for row in category_rows if row["pixel_threshold_mode"] == pixel_mode]
        summary = {**base, "category": "__macro__", "pixel_threshold_mode": pixel_mode,
                   "category_count": len(rows), "sample_count": sum(int(row["sample_count"]) for row in rows),
                   "attacked_count": sum(int(row["attacked_count"]) for row in rows)}
        metric_names = [
            "clean_i_auroc", "clean_i_ap", "clean_i_f1_max", "adversarial_i_auroc",
            "adversarial_i_ap", "adversarial_i_f1_max", "delta_i_auroc", "delta_i_ap",
            "delta_i_f1_max", "clean_p_auroc", "clean_p_f1_max", "clean_aupro",
            "adversarial_p_auroc", "adversarial_p_f1_max", "adversarial_aupro",
            "delta_p_auroc", "delta_p_f1_max", "delta_aupro", "clean_accuracy",
            "clean_fpr", "clean_fnr", "adversarial_accuracy", "adversarial_fpr",
            "adversarial_fnr", "attack_flip_rate", "targeted_attack_success_rate",
            "target_region_pixel_flip_rate_macro", "target_region_pixel_flip_rate_micro",
            "target_region_pixel_attack_success_rate_macro",
            "location_free_topk_pixel_flip_rate_macro", "location_free_topk_pixel_flip_rate_micro",
            "location_free_topk_pixel_attack_success_rate_macro", "realized_linf_mean", "realized_linf_max",
        ]
        summary.update({name: _mean(rows, name) for name in metric_names})
        region_eligible = sum(int(row["target_region_pixel_eligible_count"]) for row in rows)
        region_flips = sum(int(row["target_region_pixel_flip_count"]) for row in rows)
        region_success_eligible = sum(int(row["target_region_pixel_success_eligible_count"]) for row in rows)
        region_successes = sum(int(row["target_region_pixel_success_count"]) for row in rows)
        topk_eligible = sum(int(row["location_free_topk_pixel_eligible_count"]) for row in rows)
        topk_flips = sum(int(row["location_free_topk_pixel_flip_count"]) for row in rows)
        topk_success_eligible = sum(int(row["location_free_topk_pixel_success_eligible_count"]) for row in rows)
        topk_successes = sum(int(row["location_free_topk_pixel_success_count"]) for row in rows)
        summary.update({
            "target_region_pixel_flip_rate_micro": 100 * region_flips / region_eligible if region_eligible else np.nan,
            "target_region_pixel_attack_success_rate_micro": 100 * region_successes / region_success_eligible if region_success_eligible else np.nan,
            "location_free_topk_pixel_flip_rate_micro": 100 * topk_flips / topk_eligible if topk_eligible else np.nan,
            "location_free_topk_pixel_attack_success_rate_micro": 100 * topk_successes / topk_success_eligible if topk_success_eligible else np.nan,
            "location_free_topk_fraction": config.location_free_topk_fraction,
        })
        summary_rows.append(summary)

    predictions = {
        "sample_ids": np.asarray([sample.protocol_id for sample in cohort]),
        "labels": np.asarray([sample.label for sample in cohort], dtype=np.uint8),
        "attacked": np.asarray([sample.protocol_id in attacked_set for sample in cohort], dtype=bool),
        "clean_scores": np.asarray([clean_cache[sample.protocol_id][0] for sample in cohort], dtype=np.float32),
        "adversarial_scores": np.asarray([adversarial[sample.protocol_id][0] for sample in cohort], dtype=np.float32),
        "clean_maps": np.stack([clean_cache[sample.protocol_id][1] for sample in cohort]),
        "adversarial_maps": np.stack([adversarial[sample.protocol_id][1] for sample in cohort]),
    }
    return summary_rows, category_rows, per_image_rows, predictions, delta, delta_index


def evaluate(config: EvaluationConfig) -> Path:
    """Run all selected setup/scope conditions and return the model output root."""
    output = Path(config.output_root).expanduser().resolve() / config.model
    structured_output = separated_root(
        config.output_root, config.model, config.separated_output_root
    )
    samples_output = model_sibling_root(
        config.output_root, config.model, "_samples", config.samples_output_root
    )
    structured_samples_output = model_sibling_root(
        config.output_root,
        config.model,
        "_samples_separated",
        config.separated_samples_output_root,
    )
    existing = output / "summary.csv"
    if existing.exists() and not config.overwrite:
        raise FileExistsError(f"Results already exist: {existing}; set overwrite=true")
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(config.extraction_cache).expanduser().resolve() if config.extraction_cache else output / "extracted_attacks"
    bundles = materialize_input(config.attacks_root, cache)
    attacks = discover_attacks(
        bundles, scopes=config.scopes, targets=config.targets,
        prompt_modes=config.prompt_modes, setup_ids=config.setup_ids,
        sources=config.source_datasets, categories=config.categories,
        directions=config.directions, loss_modes=config.loss_modes,
        loss_formulations=config.loss_formulations,
    )
    if config.max_conditions:
        attacks = attacks[: config.max_conditions]
    all_summary: list[dict[str, Any]] = []
    all_categories: list[dict[str, Any]] = []
    all_images: list[dict[str, Any]] = []
    threshold_payload: dict[str, Any] = {"schema_version": 1, "model": config.model, "targets": {}}
    resolved_model_settings: dict[str, dict[str, object]] = {}

    for target in config.targets:
        target_attacks = [attack for attack in attacks if attack.record["target_dataset"] == target]
        if not target_attacks:
            continue
        discovered = discover_dataset(target, mvtec_root=config.mvtec_root, visa_root=config.visa_root)
        indexed = sample_index(discovered)
        required_ids = list(dict.fromkeys(sample_id for attack in target_attacks for sample_id in attack.evaluation_ids))
        missing = [sample_id for sample_id in required_ids if sample_id not in indexed]
        if missing:
            raise ValueError(f"Protocol IDs are absent from {target}: {missing[:5]}")
        fixed_samples = [indexed[sample_id] for sample_id in required_ids]
        kwargs = dict(config.model_kwargs_by_target[target])
        kwargs.setdefault("device", config.device)
        kwargs.setdefault("image_size", config.image_size)
        adapter = create_adapter(config.model, **kwargs)
        try:
            model_settings = adapter.runtime_metadata()
            model_settings["shared_gaussian_sigma"] = config.gaussian_sigma
            resolved_model_settings[target] = model_settings
            raw_clean = _predict_clean(adapter, fixed_samples, config)
            clean = _postprocess_predictions(adapter, fixed_samples, raw_clean)
            thresholds = _calibrate_clean(target, fixed_samples, clean, config)
            clean_metrics: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
            threshold_payload["targets"][target] = thresholds
            for attack in target_attacks:
                summary, categories, images, predictions, delta, delta_index = _evaluate_condition(
                    attack, indexed, raw_clean, adapter, thresholds, config,
                    clean_metrics=clean_metrics,
                )
                all_summary.extend(summary)
                all_categories.extend(categories)
                all_images.extend(images)
                if config.save_predictions:
                    prediction_dir = output / "predictions" / attack.record["prompt_mode"] / attack.record["setup_id"]
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(prediction_dir / f"{attack.condition_id}.npz", **predictions)
                if config.save_qualitative_samples:
                    separated_sample_slice = slice_root(
                        structured_samples_output, attack.record
                    )
                    for threshold_mode in config.qualitative_threshold_modes:
                        selected_rows = [
                            row for row in images
                            if row["pixel_threshold_mode"] == threshold_mode
                            and int(row["attacked"])
                        ]
                        sample_arguments = dict(
                            rows=selected_rows,
                            samples_by_id=indexed,
                            sample_ids=predictions["sample_ids"].tolist(),
                            clean_maps=predictions["clean_maps"],
                            adversarial_maps=predictions["adversarial_maps"],
                            delta=delta, delta_index=delta_index,
                            image_size=config.image_size,
                            gaussian_sigma=config.gaussian_sigma,
                        )
                        export_samples(
                            samples_output / threshold_mode,
                            condition_id=attack.condition_id,
                            **sample_arguments,
                        )
                        export_samples(
                            separated_sample_slice / threshold_mode,
                            condition_id=compact_sample_condition_id(attack.record),
                            **sample_arguments,
                        )
        finally:
            adapter.close()

    if not all_summary:
        raise ValueError("No selected conditions had a configured target")
    _write_csv(output / "summary.csv", all_summary)
    _write_csv(output / "category_metrics.csv", all_categories)
    _write_csv(output / "per_image.csv", all_images)
    config_payload = asdict(config)
    config_payload["resolved_model_settings_by_target"] = resolved_model_settings
    threshold_payload["resolved_model_settings_by_target"] = resolved_model_settings
    (output / "thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2, default=_json_value), encoding="utf-8"
    )
    # Checkpoint locations are provenance, but their contents are never copied.
    (output / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, default=_json_value), encoding="utf-8"
    )
    manifest_snapshot = [attack.record for attack in attacks]
    (output / "manifest_snapshot.json").write_text(
        json.dumps(manifest_snapshot, indent=2, default=_json_value), encoding="utf-8"
    )
    if config.write_separated_results:
        write_separated_numerical(
            structured_output,
            summary_rows=all_summary,
            category_rows=all_categories,
            image_rows=all_images,
            manifest_records=manifest_snapshot,
            thresholds=threshold_payload,
            config=config,
        )
    if config.create_output_archives:
        # Extraction is a disposable input cache, not an evaluation result.
        archive_directory(output, exclude_top_level=("extracted_attacks",))
        if config.save_qualitative_samples:
            archive_directory(structured_samples_output)
        if config.write_separated_results:
            archive_directory(structured_output)
    return output
