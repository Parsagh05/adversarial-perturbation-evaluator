"""Representative qualitative samples for evaluated perturbation conditions."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from .data import Sample, load_image, load_mask
from .metrics import topk_region
from .structured import safe_component


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number(value: Any) -> float:
    """Parse a CSV-round-tripped metric, treating blanks and NaN as missing."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


# Threshold-free per-image quantities. These need no pixel threshold at all, so
# they rank the same way whichever threshold mode is being exported, and each
# one is what actually moves a reported threshold-free metric.
THRESHOLD_FREE_CRITERIA = (
    (
        "largest_score_shift",
        "target_direction_score_shift",
        "image score toward the target class",
        "image AUROC, image AP, and image F1-max",
    ),
    (
        "largest_map_shift",
        "target_direction_map_shift",
        "mean anomaly-map value toward the target class",
        "pixel AUROC, pixel F1-max, and AUPRO",
    ),
)


def select_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """Pick representative attacked images and record why each was chosen.

    Two independent families are used. The threshold-based family ranks by the
    targeted pixel flip rate at the exported threshold; the threshold-free
    family ranks by the directional shifts that drive the continuous metrics.
    An image chosen by several criteria is exported once, carrying every reason.
    """
    if not rows:
        return []
    target_label = int(rows[0]["target_label"])
    prefix = "location_free_topk_" if target_label == 1 else "target_region_"
    region = "location-free top-k region" if target_label == 1 else "ground-truth defect mask"
    picks: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def rate(row: dict[str, Any]) -> float:
        return _number(row.get(f"{prefix}pixel_flip_rate"))

    eligible = [
        row for row in rows
        if int(_number(row.get(f"{prefix}pixel_success_eligible")) or 0)
    ]
    successful = sorted(
        (r for r in eligible if int(_number(r.get(f"{prefix}pixel_attack_success")) or 0)),
        key=rate,
    )
    failed = sorted(
        (r for r in eligible if not int(_number(r.get(f"{prefix}pixel_attack_success")) or 0)),
        key=rate,
    )
    pool = (
        f"{len(successful)} succeeded and {len(failed)} failed out of "
        f"{len(eligible)} attacked images with pixels eligible to flip "
        f"(of {len(rows)} attacked)"
    )
    if successful:
        best = successful[-1]
        picks.append(("strongest_success", best, {
            "criterion": "strongest_success",
            "family": "threshold-based",
            "metric": f"{prefix}pixel_flip_rate",
            "value": rate(best),
            "explanation": (
                f"Highest targeted pixel flip rate ({rate(best):.1f}% of eligible "
                f"pixels in the {region}) among successful attacks; {pool}."
            ),
        }))
        median = successful[(len(successful) - 1) // 2]
        picks.append(("median_success", median, {
            "criterion": "median_success",
            "family": "threshold-based",
            "metric": f"{prefix}pixel_flip_rate",
            "value": rate(median),
            "explanation": (
                f"Median successful attack ({rate(median):.1f}% of eligible pixels "
                f"in the {region} flipped), i.e. a typical rather than a best case; {pool}."
            ),
        }))
    if failed:
        worst = failed[0]
        picks.append(("worst_failure", worst, {
            "criterion": "worst_failure",
            "family": "threshold-based",
            "metric": f"{prefix}pixel_flip_rate",
            "value": rate(worst),
            "explanation": (
                f"Least effective attack that still had pixels eligible to flip "
                f"({rate(worst):.1f}% flipped in the {region}); {pool}."
            ),
        }))

    for name, field, moved, drives in THRESHOLD_FREE_CRITERIA:
        ranked = sorted(
            (r for r in rows if not math.isnan(_number(r.get(field)))),
            key=lambda r: _number(r.get(field)),
            reverse=True,
        )
        if not ranked:
            continue
        top = ranked[0]
        picks.append((name, top, {
            "criterion": name,
            "family": "threshold-free",
            "metric": field,
            "value": _number(top.get(field)),
            "explanation": (
                f"Largest threshold-free shift of the {moved} "
                f"({_number(top.get(field)):+.6f}) among {len(ranked)} attacked images. "
                f"This quantity needs no pixel threshold and is what drives {drives}."
            ),
        }))

    # One folder per image; the first criterion to claim it names the folder.
    merged: dict[str, tuple[str, dict[str, Any], list[dict[str, Any]]]] = {}
    order: list[str] = []
    for name, row, reason in picks:
        sample_id = str(row["protocol_id"])
        if sample_id not in merged:
            merged[sample_id] = (name, row, [reason])
            order.append(sample_id)
        else:
            merged[sample_id][2].append(reason)
    return [merged[sample_id] for sample_id in order]


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()


def _resize_map(value: np.ndarray, size: int, sigma: float) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
    result = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return gaussian_filter(result, sigma=sigma) if sigma > 0 else result


def _normalize_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = float(min(first.min(), second.min()))
    high = float(max(first.max(), second.max()))
    if high <= low:
        return np.zeros_like(first), np.zeros_like(second)
    return (first - low) / (high - low), (second - low) / (high - low)


def _heatmap(value: np.ndarray) -> np.ndarray:
    stops = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.asarray(
        [[0, 0, 128], [0, 128, 255], [255, 255, 0], [255, 0, 0], [255, 255, 255]],
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(np.clip(value, 0, 1), stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).round().astype(np.uint8)


def _target_region(row: dict[str, Any], mask: np.ndarray, size: int) -> np.ndarray:
    if int(row["target_label"]) == 0:
        return mask.astype(bool)
    # Normal-to-anomalous success is measured without assuming a location. Mirror
    # that metric in the visual by showing the adversarial map's strongest pixels.
    if "_adversarial_map" in row:
        # Reuse the scored region so the visual can never drift from the metric.
        return topk_region(
            np.asarray(row["_adversarial_map"]),
            float(row["location_free_topk_fraction"]),
        )
    mode = str(row.get("normal_local_target") or "fixed_region")
    if mode == "full_image":
        return np.ones((size, size), dtype=bool)
    fraction = float(row.get("normal_target_region_fraction") or 0.25)
    side = max(1, round(size * fraction))
    center_x = round(float(row.get("normal_target_center_x") or 0.5) * (size - 1))
    center_y = round(float(row.get("normal_target_center_y") or 0.5) * (size - 1))
    left = min(max(center_x - side // 2, 0), size - side)
    top = min(max(center_y - side // 2, 0), size - side)
    region = np.zeros((size, size), dtype=bool)
    region[top : top + side, left : left + side] = True
    return region


def _row(name: str, clean: Any, adversarial: Any, change: Any = None) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):+.6g}" if value is change else f"{float(value):.6g}"
    return f"| {name} | {cell(clean)} | {cell(adversarial)} | {cell(change)} |"


def _description(
    selection: str,
    reasons: list[dict[str, Any]],
    row: dict[str, Any],
    sample: Sample,
    extra: dict[str, Any],
) -> str:
    """Render a self-contained note explaining the pick and the before/after."""
    target_label = int(row["target_label"])
    prefix = "location_free_topk_" if target_label == 1 else "target_region_"
    region = "location-free top-k region" if target_label == 1 else "ground-truth defect mask"
    clean_score, adversarial_score = _number(row.get("clean_score")), _number(row.get("adversarial_score"))
    clean_pred, adversarial_pred = row.get("clean_prediction"), row.get("adversarial_prediction")
    names = {0: "normal", 1: "anomalous"}
    lines = [
        f"# {selection} - `{sample.protocol_id}`",
        "",
        f"- **Condition**: `{row.get('condition_id', '')}`",
        f"- **Direction**: {row.get('direction')} (source label {row.get('source_label')}"
        f" -> target label {row.get('target_label')})",
        f"- **Image**: {sample.dataset}/{sample.category}/{sample.defect_type}"
        f" (ground-truth label {row.get('label')})",
        f"- **Pixel threshold mode**: `{row.get('pixel_threshold_mode')}`"
        f" (threshold {_number(row.get('pixel_threshold')):.6g})",
        f"- **Target region**: {region}",
        "",
        "## Why this image was selected",
        "",
    ]
    for reason in reasons:
        lines.append(f"- **{reason['criterion']}** ({reason['family']}): {reason['explanation']}")
    lines += [
        "",
        "## Before vs after",
        "",
        "| quantity | clean | adversarial | change |",
        "| --- | --- | --- | --- |",
        _row("image score", clean_score, adversarial_score, adversarial_score - clean_score),
        f"| image prediction | {clean_pred} ({names.get(int(_number(clean_pred) or 0), '?')})"
        f" | {adversarial_pred} ({names.get(int(_number(adversarial_pred) or 0), '?')})"
        f" | {'flipped' if int(_number(row.get('image_flip')) or 0) else 'unchanged'} |",
        _row("anomaly map mean", extra["clean_map_mean"], extra["adversarial_map_mean"],
             extra["adversarial_map_mean"] - extra["clean_map_mean"]),
        _row("anomaly map max", extra["clean_map_max"], extra["adversarial_map_max"],
             extra["adversarial_map_max"] - extra["clean_map_max"]),
        "",
        "Directional shifts are signed so that positive always means *toward the",
        "attack target*; they use no pixel threshold.",
        "",
        f"- targeted image success: **{'yes' if int(_number(row.get('targeted_image_success')) or 0) else 'no'}**",
        f"- target-direction score shift: {_number(row.get('target_direction_score_shift')):+.6g}",
        f"- target-direction map shift: {_number(row.get('target_direction_map_shift')):+.6g}",
        "",
        f"## Targeted pixels in the {region}",
        "",
        f"- pixels in region: {row.get(f'{prefix}pixel_count')}",
        f"- eligible (clean-predicted as the source class): {row.get(f'{prefix}pixel_eligible_count')}",
        f"- flipped to the target class: {row.get(f'{prefix}pixel_flip_count')}"
        f" ({_number(row.get(f'{prefix}pixel_flip_rate')):.2f}%)",
        f"- counted as a successful attack: "
        f"**{'yes' if int(_number(row.get(f'{prefix}pixel_attack_success')) or 0) else 'no'}**"
        f" (needs at least {100 * float(row.get('pixel_success_min_flip_fraction', 0.5)):.0f}% flipped)"
        if row.get("pixel_success_min_flip_fraction") is not None else
        f"- counted as a successful attack: "
        f"**{'yes' if int(_number(row.get(f'{prefix}pixel_attack_success')) or 0) else 'no'}**",
        "",
        "## Perturbation",
        "",
        f"- realized L-inf: {_number(row.get('realized_linf')):.6g}",
        f"- mean absolute change: {extra['image_mae']:.6g}",
        f"- PSNR: {extra['image_psnr']:.4g} dB",
        "",
        f"Source image: `{sample.image_path}`",
        "",
    ]
    return "\n".join(lines)


def export_samples(
    output: Path,
    *,
    condition_id: str,
    rows: list[dict[str, Any]],
    samples_by_id: dict[str, Sample],
    sample_ids: Sequence[str],
    clean_maps: np.ndarray,
    adversarial_maps: np.ndarray,
    delta: torch.Tensor,
    delta_index: dict[str, int],
    image_size: int,
    gaussian_sigma: float,
) -> None:
    map_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    condition_root = output / safe_component(condition_id)
    if condition_root.exists():
        shutil.rmtree(condition_root)
    condition_root.mkdir(parents=True, exist_ok=True)
    selection_manifest: list[dict[str, Any]] = []
    for selection, row, reasons in select_rows(rows):
        sample_id = str(row["protocol_id"])
        sample = samples_by_id[sample_id]
        clean_tensor = load_image(sample, image_size)
        adversarial_tensor = (clean_tensor + delta[delta_index[sample_id]]).clamp(0, 1)
        clean_rgb, adversarial_rgb = _rgb(clean_tensor), _rgb(adversarial_tensor)
        index = map_index[sample_id]
        clean_map = _resize_map(clean_maps[index], image_size, gaussian_sigma)
        adversarial_map = _resize_map(adversarial_maps[index], image_size, gaussian_sigma)
        clean_norm, adversarial_norm = _normalize_pair(clean_map, adversarial_map)
        clean_heatmap, adversarial_heatmap = _heatmap(clean_norm), _heatmap(adversarial_norm)
        mask = load_mask(sample, image_size)
        region = _target_region({**row, "_adversarial_map": adversarial_map}, mask, image_size)
        threshold = float(row["pixel_threshold"])
        clean_binary = clean_map >= threshold
        adversarial_binary = adversarial_map >= threshold
        source, target = int(row["source_label"]), int(row["target_label"])
        successful = region & (clean_binary == source) & (adversarial_binary == target)

        folder = condition_root / f"{selection}__{safe_component(sample_id)}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean_rgb).save(folder / "clean.png")
        Image.fromarray(adversarial_rgb).save(folder / "adversarial.png")
        difference = np.clip(
            np.abs(adversarial_tensor.numpy() - clean_tensor.numpy()).transpose(1, 2, 0) * 10,
            0,
            1,
        )
        Image.fromarray((difference * 255).round().astype(np.uint8)).save(
            folder / "difference_x10.png"
        )
        Image.fromarray(clean_heatmap).save(folder / "clean_heatmap.png")
        Image.fromarray(adversarial_heatmap).save(folder / "adversarial_heatmap.png")
        Image.blend(Image.fromarray(clean_rgb), Image.fromarray(clean_heatmap), 0.45).save(
            folder / "clean_overlay.png"
        )
        Image.blend(
            Image.fromarray(adversarial_rgb), Image.fromarray(adversarial_heatmap), 0.45
        ).save(folder / "adversarial_overlay.png")
        for name, value in (
            ("ground_truth_mask", mask.astype(bool)),
            ("target_region_mask", region),
            ("clean_pixel_prediction", clean_binary),
            ("adversarial_pixel_prediction", adversarial_binary),
            ("successful_target_pixel_flips", successful),
        ):
            Image.fromarray((value.astype(np.uint8) * 255)).save(folder / f"{name}.png")
        difference_map = adversarial_map - clean_map
        bound = max(float(np.abs(difference_map).max()), np.finfo(np.float32).eps)
        signed = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        signed[..., 0] = (np.clip(difference_map / bound, 0, 1) * 255).astype(np.uint8)
        signed[..., 2] = (np.clip(-difference_map / bound, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(signed).save(folder / "heatmap_difference.png")
        perturbation = adversarial_tensor.numpy() - clean_tensor.numpy()
        mse = float(np.mean(perturbation ** 2))
        extra = {
            "image_mae": float(np.mean(np.abs(perturbation))),
            "image_linf": float(np.max(np.abs(perturbation))),
            "image_psnr": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
            "clean_map_mean": float(clean_map.mean()),
            "adversarial_map_mean": float(adversarial_map.mean()),
            "clean_map_max": float(clean_map.max()),
            "adversarial_map_max": float(adversarial_map.max()),
        }
        metadata = {
            **row,
            "selection": selection,
            "selection_reasons": reasons,
            "sample_image": str(sample.image_path),
            **extra,
        }
        metadata.pop("_adversarial_map", None)
        (folder / "metrics.json").write_text(
            json.dumps(_finite(metadata), indent=2), encoding="utf-8"
        )
        (folder / "description.md").write_text(
            _description(selection, reasons, row, sample, extra), encoding="utf-8"
        )
        selection_manifest.append({
            "selection": selection,
            "protocol_id": sample_id,
            "folder": folder.name,
            "reasons": _finite(reasons),
        })
    (condition_root / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )
