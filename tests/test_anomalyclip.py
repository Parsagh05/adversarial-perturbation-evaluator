from inspect import signature

from fpeval.adapters.anomalyclip import AnomalyCLIPAdapter


def test_anomalyclip_constructor_matches_official_test_defaults():
    parameters = signature(AnomalyCLIPAdapter.__init__).parameters
    assert parameters["image_size"].default == 518
    assert parameters["features"].default == (6, 12, 18, 24)
    assert parameters["map_layers"].default == (0, 1, 2, 3)
    assert parameters["prompt_depth"].default == 9
    assert parameters["context_length"].default == 12
    assert parameters["compound_context_length"].default == 4
    assert parameters["dpam_layer"].default == 20
    assert parameters["backbone"].default == "ViT-L/14@336px"
    assert parameters["seed"].default == 111
