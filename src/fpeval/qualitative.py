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


def select_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if not rows:
        return []
    prefix = (
        "location_free_topk_"
        if int(rows[0]["target_label"]) == 1
        else "target_region_"
    )
    eligible = [
        row for row in rows
        if int(row.get(f"{prefix}pixel_success_eligible", 0) or 0)
    ]
    successful = sorted(
        (row for row in eligible if int(row.get(f"{prefix}pixel_attack_success", 0) or 0)),
        key=lambda row: float(row.get(f"{prefix}pixel_flip_rate", 0) or 0),
    )
    failed = sorted(
        (row for row in eligible if not int(row.get(f"{prefix}pixel_attack_success", 0) or 0)),
        key=lambda row: float(row.get(f"{prefix}pixel_flip_rate", 0) or 0),
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    if successful:
        selected.append(("strongest_success", successful[-1]))
        median = successful[(len(successful) - 1) // 2]
        if median["protocol_id"] != successful[-1]["protocol_id"]:
            selected.append(("median_success", median))
    if failed:
        selected.append(("worst_failure", failed[0]))
    return selected


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
        value = np.asarray(row["_adversarial_map"])
        count = max(1, int(math.ceil(value.size * float(row["location_free_topk_fraction"]))))
        indices = np.argpartition(value.reshape(-1), -count)[-count:]
        region = np.zeros(value.size, dtype=bool)
        region[indices] = True
        return region.reshape(value.shape)
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
    for selection, row in select_rows(rows):
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
        metadata = {
            **row,
            "selection": selection,
            "sample_image": str(sample.image_path),
            "image_mae": float(np.mean(np.abs(perturbation))),
            "image_linf": float(np.max(np.abs(perturbation))),
            "image_psnr": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
            "clean_map_mean": float(clean_map.mean()),
            "adversarial_map_mean": float(adversarial_map.mean()),
        }
        metadata.pop("_adversarial_map", None)
        (folder / "metrics.json").write_text(
            json.dumps(_finite(metadata), indent=2), encoding="utf-8"
        )
        selection_manifest.append(
            {"selection": selection, "protocol_id": sample_id, "folder": folder.name}
        )
    (condition_root / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )
