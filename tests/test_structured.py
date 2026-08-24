from fpeval.structured import compact_sample_condition_id


def test_compact_per_dataset_sample_condition_id():
    assert compact_sample_condition_id({
        "category": "",
        "direction": "abnormal_to_normal",
        "loss_formulation": "ce_focal_dice",
        "loss_mode": "combined",
    }) == "abnormal_to_normal__ce_focal_dice__combined"


def test_compact_category_sample_condition_id_keeps_category():
    assert compact_sample_condition_id({
        "category": "bottle",
        "direction": "normal_to_abnormal",
        "loss_formulation": "margin_topk",
        "loss_mode": "local",
    }) == "bottle__normal_to_abnormal__margin_topk__local"
