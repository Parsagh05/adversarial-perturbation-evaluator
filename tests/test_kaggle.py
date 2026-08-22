from pathlib import Path

from fpeval.kaggle import discover_kaggle_inputs, inventory_attack_setups


def test_discovers_screenshot_kaggle_layout_and_both_prompt_modes(tmp_path: Path):
    mvtec = tmp_path / "MVTec-AD" / "mvtec_anomaly_detection"
    (mvtec / "bottle" / "test" / "good").mkdir(parents=True)
    (mvtec / "bottle" / "ground_truth").mkdir()

    visa = tmp_path / "VisA-AD" / "VisA_20220922"
    (visa / "split_csv").mkdir(parents=True)
    (visa / "split_csv" / "1cls.csv").write_text("object,split,label,image\n")

    attacks = tmp_path / "perturbation-generated"
    (attacks / "setups" / "frozen_prompt" / "steps500_eps2").mkdir(parents=True)
    (attacks / "setups" / "learnable_prompt" / "steps500_eps2_learnable_prompt").mkdir(parents=True)

    discovered = discover_kaggle_inputs(tmp_path)
    assert discovered.mvtec_root == mvtec.resolve()
    assert discovered.visa_root == visa.resolve()
    assert discovered.attacks_root == attacks.resolve()
    assert inventory_attack_setups(attacks) == {
        "frozen_prompt": {"steps500_eps2": "steps500_eps2"},
        "learnable_prompt": {
            "steps500_eps2": "steps500_eps2_learnable_prompt"
        },
    }


def test_user_can_select_the_perturbation_input(tmp_path: Path):
    mvtec = tmp_path / "MVTec-AD" / "mvtec_anomaly_detection"
    (mvtec / "bottle" / "test" / "good").mkdir(parents=True)
    (mvtec / "bottle" / "ground_truth").mkdir()
    visa = tmp_path / "VisA-AD" / "VisA_20220922" / "split_csv"
    visa.mkdir(parents=True)
    (visa / "1cls.csv").write_text("object,split,label,image\n")
    selected = tmp_path / "my-perturbations"
    (selected / "setups" / "frozen_prompt" / "steps500_eps2").mkdir(parents=True)

    result = discover_kaggle_inputs(
        tmp_path, attacks_root=selected / "setups"
    )
    assert result.attacks_root == selected.resolve()
