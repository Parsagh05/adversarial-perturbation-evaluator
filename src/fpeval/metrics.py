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


def _summary(labels: Sequence[int], scores: Sequence[float]) -> tuple[dict[str, float], float]:
    """Return performance metrics and the F1-optimal threshold from one sort."""
    labels = np.asarray(labels, dtype=np.uint8)
    if np.unique(labels).size < 2:
        return {"auroc": np.nan, "ap": np.nan, "f1_max": np.nan}, np.nan
    fp, tp, thresholds = _curve(labels, scores)
    fpr, tpr = np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(trap(tpr, fpr))
    precision, recall = tp / (tp + fp), tp / tp[-1]
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    fn = int(labels.sum()) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    metrics = {"auroc": 100 * auroc, "ap": 100 * ap, "f1_max": 100 * float(f1.max())}
    return metrics, float(thresholds[int(np.argmax(f1))])


def performance(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    return _summary(labels, scores)[0]


def aupro(masks: np.ndarray, maps: np.ndarray, *, fpr_limit: float = 0.3, thresholds: int = 200) -> float:
    """AUPRO as FasterAUPRO's ``cal_pro_score_opp`` computes it, on a 0-100 scale.

    https://github.com/AlirezaSalehy/FasterAUPRO - the cflow-ad algorithm that
    AnomalyCLIP ships, with the region labelling taken out of the threshold
    loop, so it returns their values: ``thresholds`` evenly spaced steps from
    the lowest to the highest score, the points below FPR ``fpr_limit`` kept,
    their FPRs rescaled to [0, 1] and the PRO curve integrated over them.
    Ported line by line on scipy: ``skimage.measure.label``'s default 2-D
    connectivity is the 8-neighbourhood used here, and sklearn's ``auc`` of a
    non-increasing x is the negated trapezoid. That code reads a few tenths
    below the official MVTec AD evaluation, which uses every score as a
    threshold and integrates exactly up to the limit.
    """
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float32)
    # Each region's pixel coordinates, labelled once rather than per threshold.
    coords_list = []
    for mask in masks:
        components, count = connected_components(mask, structure=np.ones((3, 3)))
        coords_list.append([np.nonzero(components == k) for k in range(1, count + 1)])
    inverse_masks = ~masks
    tn_pixel = int(inverse_masks.sum())  # pixels that truly have the label 0
    min_th, max_th = maps.min(), maps.max()
    # The reference has no answer for these and fails on them.
    if not tn_pixel or not any(coords_list) or max_th == min_th:
        return np.nan
    delta = (max_th - min_th) / thresholds
    binary_amaps = np.zeros_like(maps, dtype=bool)
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        np.greater(maps, th, out=binary_amaps)
        pro = [
            binary_amap[coords].sum() / coords[0].size
            for binary_amap, regions_coords in zip(binary_amaps, coords_list)
            for coords in regions_coords
        ]
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fprs.append(fp_pixels / tn_pixel)
        pros.append(np.mean(pro))
    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < fpr_limit
    fprs, pros = fprs[idxes], pros[idxes]
    if not fprs.size:
        return np.nan
    if np.ptp(fprs) == 0:
        return 100 * float(np.mean(pros))
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    # The thresholds rise, so the FPRs fall: sklearn's auc negates for that.
    return 100 * float(-trap(pros, fprs))


def pixel_performance(
    masks: np.ndarray, maps: np.ndarray, *, fpr_limit: float, thresholds: int,
    with_aupro: bool = True,
) -> dict[str, float]:
    flat_masks = np.asarray(masks, dtype=np.uint8).reshape(-1)
    flat_maps = np.asarray(maps, dtype=np.float32).reshape(-1)
    # One sorted curve serves AUROC, F1-max, and the F1-optimal threshold.
    base, f1_threshold = _summary(flat_masks, flat_maps)
    return {
        "p_auroc": base["auroc"], "p_f1_max": base["f1_max"],
        "p_f1_threshold": f1_threshold,
        # AUPRO labels connected components per image and is by far the most
        # expensive metric; without it the entry is NaN, not absent.
        "aupro": (
            aupro(masks, maps, fpr_limit=fpr_limit, thresholds=thresholds)
            if with_aupro else np.nan
        ),
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
