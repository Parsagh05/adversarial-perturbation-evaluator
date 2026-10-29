import csv

import numpy as np
import pytest
import torch

from fpeval.adapters.base import ModelAdapter, register_adapter

from test_engine import _bidirectional_fixture, _evaluate_pair


@register_adapter("test_set_adapter")
class SetRelativeAdapter(ModelAdapter):
    """Scores an image by its mean minus the mean of the set it is scored in."""

    name = "test_set_adapter"
    transductive = True

    def __init__(self, **kwargs):
        pass

    def predict(self, images, categories):
        raise AssertionError("a transductive adapter is never asked for single images")

    def predict_set(self, images, categories):
        stacked = torch.stack([images[index] for index in range(len(images))])
        means = stacked.mean(dim=(1, 2, 3))
        scores = (means - means.mean()).numpy()
        maps = (stacked.mean(dim=1) - means.mean()).numpy()
        return scores, maps

    def close(self):
        pass


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_attack_reaches_unattacked_images_of_a_set(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(tmp_path, mvtec, "results", model="test_set_adapter")
    images = [
        row for row in _rows(output / "per_image.csv")
        if row["direction"] == "normal_to_abnormal"
    ]
    by_id = {row["protocol_id"].rsplit("/", 1)[-1]: row for row in images}
    # Clean set means 0, 0.2, 0.8, 1.0 around 0.5. N->A lifts the normals to
    # 0.7 and 0.9, so the set mean is 0.85 and the defects, untouched, drop.
    assert float(by_id["002"]["clean_score"]) == pytest.approx(0.3, abs=1e-5)
    assert float(by_id["002"]["adversarial_score"]) == pytest.approx(-0.05, abs=1e-5)
    assert by_id["002"]["attacked"] == "0"
    assert float(by_id["002"]["realized_linf"]) == 0.0
    assert float(by_id["000"]["realized_linf"]) == pytest.approx(0.7, abs=1e-5)


def test_v2_scores_both_deltas_in_one_set(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(tmp_path, mvtec, "results", model="test_set_adapter")
    (row,) = _rows(output / "summary_v2.csv")
    # Joint set 0.7, 0.9, 0.1, 0.3: every normal outranks every defect. Merging
    # the two V1 sets instead would leave the ranking partly intact.
    assert float(row["clean_i_auroc"]) == 100.0
    assert float(row["adversarial_i_auroc"]) == 0.0


def test_attack_train_is_refused(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    with pytest.raises(ValueError, match="transductive"):
        _evaluate_pair(
            tmp_path, mvtec, "results", model="test_set_adapter",
            evaluate_attack_train=True,
        )


def test_clean_only_scores_the_split_as_one_set(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    output = _evaluate_pair(
        tmp_path, mvtec, "results", model="test_set_adapter", clean_only=True,
    )
    (summary,) = [row for row in _rows(output / "summary.csv") if row["category"] == "__macro__"]
    assert float(summary["clean_i_auroc"]) == 100.0
    assert np.isfinite(float(summary["clean_p_auroc"]))
