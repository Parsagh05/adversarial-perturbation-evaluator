"""Threshold-free performance and targeted attack diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import label as connected_components


def _curve(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.uint8)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != score.shape or not len(y) or not np.isfinite(score).all():
        raise ValueError("Labels and scores must be finite matching vectors")
    order = np.argsort(score, kind="mergesort")[::-1]
    sorted_y, sorted_score = y[order], score[order]
    ends = np.r_[np.where(np.diff(sorted_score))[0], len(y) - 1]
    tp = np.cumsum(sorted_y, dtype=np.float64)[ends]
    fp = 1 + ends - tp
    return fp, tp, sorted_score[ends]


def optimal_f1(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("F1 calibration needs both binary classes")
    fp, tp, thresholds = _curve(labels, scores)
    fn = int(labels.sum()) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    index = int(np.argmax(f1))
    return {"threshold": float(thresholds[index]), "f1": float(f1[index])}


def performance(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    if np.unique(labels).size < 2:
        return {"auroc": np.nan, "ap": np.nan, "f1_max": np.nan}
    fp, tp, thresholds = _curve(labels, scores)
    fpr, tpr = np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(trap(tpr, fpr))
    precision, recall = tp / (tp + fp), tp / tp[-1]
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    fn = int(labels.sum()) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return {"auroc": 100 * auroc, "ap": 100 * ap, "f1_max": 100 * float(f1.max())}


def aupro(masks: np.ndarray, maps: np.ndarray, *, fpr_limit: float = 0.3, thresholds: int = 200) -> float:
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float32)
    negatives = ~masks
    negative_count = int(negatives.sum())
    regions: list[tuple[int, np.ndarray]] = []
    for image_index, mask in enumerate(masks):
        components, count = connected_components(mask, structure=np.ones((3, 3)))
        regions.extend((image_index, components == component) for component in range(1, count + 1))
    if not negative_count or not regions:
        return np.nan
    flat = maps.reshape(-1)
    stride = max(1, int(np.ceil(flat.size / 1_000_000)))
    sampled = flat[::stride]
    candidates = np.unique(np.quantile(sampled, np.linspace(1, 0, min(thresholds, len(sampled)))))[::-1]
    fprs, pros = [0.0], [0.0]
    for threshold in candidates:
        prediction = maps >= threshold
        fprs.append(float((prediction & negatives).sum()) / negative_count)
        pros.append(float(np.mean([prediction[index][region].mean() for index, region in regions])))
    x_raw, y_raw = np.asarray(fprs), np.asarray(pros)
    order = np.argsort(x_raw)
    x_raw, y_raw = x_raw[order], y_raw[order]
    x = np.unique(x_raw)
    y = np.asarray([y_raw[x_raw == item].max() for item in x])
    boundary = float(np.interp(fpr_limit, x, y))
    keep = x < fpr_limit
    x, y = np.r_[x[keep], fpr_limit], np.r_[y[keep], boundary]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 100 * float(trap(y, x) / fpr_limit)


def pixel_performance(masks: np.ndarray, maps: np.ndarray, *, fpr_limit: float, thresholds: int) -> dict[str, float]:
    flat_masks = np.asarray(masks, dtype=np.uint8).reshape(-1)
    flat_maps = np.asarray(maps, dtype=np.float32).reshape(-1)
    base = performance(flat_masks, flat_maps)
    return {
        "p_auroc": base["auroc"], "p_f1_max": base["f1_max"],
        "p_f1_threshold": optimal_f1(flat_masks, flat_maps)["threshold"] if np.unique(flat_masks).size == 2 else np.nan,
        "aupro": aupro(masks, maps, fpr_limit=fpr_limit, thresholds=thresholds),
    }


def classification(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    y, pred = np.asarray(labels, dtype=np.uint8), np.asarray(predictions, dtype=np.uint8)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("Classification arrays must be non-empty matching vectors")
    normal, abnormal = y == 0, y == 1
    return {
        "accuracy": 100 * float((y == pred).mean()),
        "fpr": 100 * float((pred[normal] == 1).mean()) if normal.any() else np.nan,
        "fnr": 100 * float((pred[abnormal] == 0).mean()) if abnormal.any() else np.nan,
    }


def targeted_images(clean: np.ndarray, adversarial: np.ndarray, attacked: np.ndarray, *, source: int, target: int) -> dict[str, float | int]:
    attacked = np.asarray(attacked, dtype=bool)
    eligible = attacked & (clean == source)
    success = eligible & (adversarial == target) & (adversarial != clean)
    return {
        "attack_flip_rate": 100 * float((clean[attacked] != adversarial[attacked]).mean()) if attacked.any() else np.nan,
        "targeted_attack_success_rate": 100 * float(success[eligible].mean()) if eligible.any() else np.nan,
        "targeted_success_eligible_count": int(eligible.sum()),
    }


def targeted_pixels(clean: np.ndarray, adversarial: np.ndarray, region: np.ndarray, *, threshold: float, source: int, target: int, minimum_fraction: float) -> dict[str, float | int]:
    region = np.asarray(region, dtype=bool)
    clean_pred, adversarial_pred = clean >= threshold, adversarial >= threshold
    eligible = region & (clean_pred == bool(source))
    flipped = eligible & (adversarial_pred == bool(target)) & (adversarial_pred != clean_pred)
    count = int(eligible.sum())
    fraction = float(flipped.sum()) / count if count else np.nan
    return {
        "pixel_count": int(region.sum()), "pixel_eligible_count": count,
        "pixel_flip_count": int(flipped.sum()), "pixel_flip_rate": 100 * fraction,
        "pixel_success_eligible": int(count > 0),
        "pixel_attack_success": int(count > 0 and fraction >= minimum_fraction),
    }


def topk_region(anomaly_map: np.ndarray, fraction: float) -> np.ndarray:
    score = np.asarray(anomaly_map, dtype=np.float32)
    count = max(1, min(score.size, int(round(fraction * score.size))))
    indices = np.argpartition(score.ravel(), score.size - count)[-count:]
    region = np.zeros(score.size, dtype=bool)
    region[indices] = True
    return region.reshape(score.shape)
