from pathlib import Path

import pytest

from fpeval.adapters import base, iipad, rempad
from fpeval.data import Sample


def _sample(path: str) -> Sample:
    return Sample(dataset="mvtec", category=path.split("/")[-4], defect_type="good",
                  image_path=Path(path), mask_path=None, label=0, split="train")


def test_both_are_registered_as_few_shot():
    for name in ("rempad", "remp-ad", "iipad"):
        assert name in base.adapter_names()
        assert base.regime(name) == "few_shot"


def test_rempad_reads_k_shot_txt_in_drawn_order_and_keeps_repeats(tmp_path):
    listing = tmp_path / "k_shot.txt"
    listing.write_text("bottle/train/good/192.png\nbottle/train/good/074.png\n"
                       "bottle/train/good/074.png\ncable/train/good/168.png\n", encoding="utf-8")
    selection = rempad.read_shot_list(listing)
    assert selection == {
        "bottle": ["bottle/train/good/192.png", "bottle/train/good/074.png", "bottle/train/good/074.png"],
        "cable": ["cable/train/good/168.png"],
    }


def test_rempad_matches_the_recorded_files_by_path_suffix():
    candidates = [_sample(f"/data/mvtec/bottle/train/good/{n:03d}.png") for n in range(200)]
    picked = rempad.match_samples(candidates, ["bottle/train/good/192.png", "bottle/train/good/074.png"])
    assert [p.image_path.name for p in picked] == ["192.png", "074.png"]
    with pytest.raises(FileNotFoundError):
        rempad.match_samples(candidates, ["bottle/train/good/999.png"])


def test_rempad_resolves_each_targets_weights_and_shot_list(tmp_path):
    for target, epochs in (("mvtec", 15), ("visa", 3)):
        folder = tmp_path / "version" / target
        (folder / f"{target}_4shot").mkdir(parents=True)
        (folder / f"epoch_{epochs}.pth").write_bytes(b"")
        (folder / f"{target}_4shot" / "k_shot.txt").write_text("x/y.png\n", encoding="utf-8")
    weights, shots = rempad.resolve_release("visa", checkpoint_root=tmp_path)
    assert weights.name == "epoch_3.pth" and weights.parent.name == "visa"
    assert shots.parent.name == "visa_4shot"
    assert rempad.CONFIG == {"mvtec": {"lambd": 0.64, "epochs": 15}, "visa": {"lambd": 0.72, "epochs": 3}}


def test_iipad_resolves_best_and_last_and_rejects_others(tmp_path):
    for target in ("mvtec", "visa"):
        (tmp_path / target).mkdir()
        for kind in iipad.CHECKPOINTS:
            (tmp_path / target / f"soft_prompt_{kind}.pt").write_bytes(b"")
    assert iipad.resolve_checkpoint("mvtec", "best", checkpoint_root=tmp_path).name == "soft_prompt_best.pt"
    assert iipad.resolve_checkpoint("visa", "last", checkpoint_root=tmp_path).parent.name == "visa"
    with pytest.raises(ValueError):
        iipad.resolve_checkpoint("mvtec", "epoch3", checkpoint_root=tmp_path)


def test_iipad_shots_are_main_py_positions():
    # main.py: '00' + str(i * 3) + '.png' for i < k  ->  000, 003, 006, 009.
    assert iipad.shot_positions(4) == [0, 3, 6, 9]


@pytest.mark.parametrize("module, cls", [(rempad, rempad.ReMPADAdapter), (iipad, iipad.IIPADAdapter)])
def test_both_refuse_an_untrained_shot_count_and_an_unknown_target(module, cls, tmp_path):
    with pytest.raises(ValueError, match="k_shot"):
        cls(repository=str(tmp_path), target_dataset="mvtec", k_shot=1)
    with pytest.raises(ValueError, match="target_dataset"):
        cls(repository=str(tmp_path), target_dataset="btad", k_shot=4)


@pytest.mark.parametrize("module", [rempad, iipad])
def test_both_report_an_incomplete_repository(module, tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        module._import_official_repository(tmp_path)
