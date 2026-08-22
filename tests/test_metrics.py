import numpy as np
import pytest

from fpeval.metrics import (
    classification,
    optimal_f1,
    performance,
    targeted_images,
    targeted_pixels,
    topk_region,
)


def test_perfect_continuous_metrics_and_f1_threshold():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    result = performance(labels, scores)
    assert result == {"auroc": 100.0, "ap": 100.0, "f1_max": 100.0}
    assert optimal_f1(labels, scores)["threshold"] == pytest.approx(0.8)


def test_targeted_metrics_exclude_clean_target_predictions():
    result = targeted_images(
        np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([1, 1, 1], dtype=bool),
        source=0, target=1,
    )
    assert result["attack_flip_rate"] == pytest.approx(100 / 3)
    assert result["targeted_attack_success_rate"] == pytest.approx(50)
    assert result["targeted_success_eligible_count"] == 2


def test_pixel_success_and_location_free_topk():
    clean = np.zeros((2, 2), dtype=np.float32)
    adversarial = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)
    region = topk_region(adversarial, 0.5)
    assert region.tolist() == [[True, True], [False, False]]
    result = targeted_pixels(
        clean, adversarial, region, threshold=0.5,
        source=0, target=1, minimum_fraction=0.5,
    )
    assert result["pixel_flip_rate"] == 100
    assert result["pixel_attack_success"] == 1


def test_classification_fpr_fnr():
    result = classification([0, 0, 1, 1], [0, 1, 0, 1])
    assert result == {"accuracy": 50.0, "fpr": 50.0, "fnr": 50.0}

