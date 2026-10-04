"""AUPRO must reproduce FasterAUPRO's ``cal_pro_score_opp``.

The expected values were computed with that function itself
(https://github.com/AlirezaSalehy/FasterAUPRO, commit cf7a8d6) on the inputs
built below, with ``sklearn.metrics.auc`` supplied for the ``auc`` it calls.
"""

import numpy as np
import pytest

from fpeval.metrics import aupro


def _case(seed, n, size, signal):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), bool)
    for i in range(n - 1):
        y, x = rng.integers(0, size - 6, 2)
        masks[i, y:y + 5, x:x + 4] = True
        masks[i, (y + 9) % size, (x + 7) % size] = True  # a one-pixel region
    maps = (rng.random((n, size, size)) + masks * signal).astype(np.float32)
    return masks, maps


@pytest.mark.parametrize("args, expected", [
    ((0, 6, 32, 0.3), 51.129296008869176),
    ((1, 4, 24, 0.05), 19.93868868868869),
    ((2, 8, 16, 1.0), 100.0),
])
def test_aupro_matches_fasteraupro(args, expected):
    masks, maps = _case(*args)
    assert aupro(masks, maps) == pytest.approx(expected, abs=1e-9)


def test_aupro_without_regions_or_normal_pixels_is_nan():
    maps = np.random.default_rng(0).random((2, 8, 8)).astype(np.float32)
    assert np.isnan(aupro(np.zeros((2, 8, 8), bool), maps))
    assert np.isnan(aupro(np.ones((2, 8, 8), bool), maps))
