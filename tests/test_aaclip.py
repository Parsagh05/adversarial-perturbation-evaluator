from inspect import signature

import numpy as np

from fpeval.adapters.aaclip import AACLIPAdapter


def test_aaclip_constructor_matches_official_cli_defaults():
    parameters = signature(AACLIPAdapter.__init__).parameters
    assert parameters["image_size"].default == 518
    assert parameters["backbone"].default == "ViT-L-14-336"
    assert parameters["seed"].default == 111
    assert parameters["text_adapter_weight"].default == 0.1
    assert parameters["image_adapter_weight"].default == 0.1
    assert parameters["text_adapter_layers"].default == 3
    assert parameters["image_adapter_layers"].default == 6
    assert parameters["feature_levels"].default == (6, 12, 18, 24)
    assert parameters["relu"].default is False


def test_aaclip_official_category_aggregation_and_frozen_reference():
    adapter = AACLIPAdapter.__new__(AACLIPAdapter)
    categories = ["bottle", "bottle"]
    clean_scores = np.asarray([2.0, 4.0])
    clean_mins = np.asarray([1.0, 2.0])
    clean_maxs = np.asarray([3.0, 5.0])
    clean = adapter.postprocess_image_scores(
        clean_scores, clean_mins, clean_maxs, categories
    )
    np.testing.assert_allclose(clean, [0.25, 1.0])

    adversarial = adapter.postprocess_image_scores_with_reference(
        np.asarray([2.0, 100.0]),
        np.asarray([1.0, 2.0]),
        np.asarray([3.0, 200.0]),
        categories,
        reference_scores=clean_scores,
        reference_map_mins=clean_mins,
        reference_map_maxs=clean_maxs,
        reference_categories=categories,
    )
    # The untouched first sample retains its clean score because adversarial
    # extrema never refit AA-CLIP's category-level normalization.
    assert adversarial[0] == clean[0]
    assert adversarial[1] > 1.0


def test_aaclip_map_normalization_uses_clean_range():
    adapter = AACLIPAdapter.__new__(AACLIPAdapter)
    clean = np.asarray([[[1.0, 3.0]], [[2.0, 5.0]]], dtype=np.float32)
    normalized = adapter.postprocess_anomaly_maps(clean, ["a", "a"])
    np.testing.assert_allclose(normalized, [[[0.0, 0.5]], [[0.25, 1.0]]])
    adversarial = adapter.postprocess_anomaly_maps_with_reference(
        np.asarray([[[1.0, 3.0]], [[2.0, 9.0]]], dtype=np.float32),
        ["a", "a"],
        reference_maps=clean,
        reference_categories=["a", "a"],
    )
    np.testing.assert_allclose(adversarial[0], normalized[0])
    assert adversarial.max() == 2.0
