"""A target's model and predictions are released before the next target starts."""

import csv
import gc
import weakref

import numpy as np
from PIL import Image

from fpeval.adapters.base import ModelAdapter, register_adapter
from fpeval.config import EvaluationConfig
from fpeval.engine import evaluate

# Every adapter built in a run, as weak references, and whether the previous
# one was still alive when the next was constructed.
_BUILT: list[weakref.ref] = []
_PREVIOUS_ALIVE: list[bool] = []


class _Maps:
    """Stands in for the large arrays a target holds; weakly referenceable."""


@register_adapter("test_memory_adapter")
class MemoryAdapter(ModelAdapter):
    name = "test_memory_adapter"

    def __init__(self, **kwargs):
        gc.collect()
        if _BUILT:
            _PREVIOUS_ALIVE.append(_BUILT[-1]() is not None)
        _BUILT.append(weakref.ref(self))
        self.payload = _Maps()

    def predict(self, images, categories):
        return images.mean(dim=(1, 2, 3)).numpy(), images.mean(dim=1).numpy()

    def close(self):
        pass


def _image(path, value, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 8) if mask else (8, 8, 3)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)


def _datasets(tmp_path):
    mvtec = tmp_path / "mvtec"
    _image(mvtec / "bottle" / "test" / "good" / "000.png", 10)
    _image(mvtec / "bottle" / "test" / "good" / "001.png", 60)
    _image(mvtec / "bottle" / "test" / "crack" / "002.png", 200)
    _image(mvtec / "bottle" / "ground_truth" / "crack" / "002_mask.png", 255, mask=True)
    visa = tmp_path / "visa"
    rows = []
    for stem, value, label in (("000", 20, "normal"), ("001", 90, "normal"), ("002", 240, "anomaly")):
        image = f"candle/Data/Images/{stem}.JPG"
        _image(visa / image, value)
        mask = ""
        if label == "anomaly":
            mask = f"candle/Data/Masks/{stem}.png"
            _image(visa / mask, 255, mask=True)
        rows.append({"object": "candle", "split": "test", "label": label, "image": image, "mask": mask})
    (visa / "split_csv").mkdir(parents=True)
    with (visa / "split_csv" / "1cls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return mvtec, visa


def _run(tmp_path, mvtec, visa, targets, results):
    config = EvaluationConfig(
        attacks_root=None, output_root=str(tmp_path / results),
        model="test_memory_adapter", model_kwargs_by_target={target: {} for target in targets},
        mvtec_root=str(mvtec), visa_root=str(visa), targets=targets,
        device="cpu", image_size=8, batch_size=2, gaussian_sigma=0,
        pixel_threshold_modes=("fixed_0_5",), save_qualitative_samples=False,
        create_output_archives=False, clean_only=True,
    )
    output = evaluate(config)
    with (output / "summary.csv").open(newline="") as handle:
        return {row["target_dataset"]: row for row in csv.DictReader(handle)
                if row["category"] == "__macro__"}


def test_first_target_is_released_before_the_second_is_built(tmp_path):
    mvtec, visa = _datasets(tmp_path)
    _BUILT.clear()
    _PREVIOUS_ALIVE.clear()
    _run(tmp_path, mvtec, visa, ("mvtec", "visa"), "both")
    assert len(_BUILT) == 2
    assert _PREVIOUS_ALIVE == [False]


def test_releasing_does_not_change_any_result(tmp_path):
    mvtec, visa = _datasets(tmp_path)
    both = _run(tmp_path, mvtec, visa, ("mvtec", "visa"), "both")
    alone = {
        **_run(tmp_path, mvtec, visa, ("mvtec",), "mvtec_only"),
        **_run(tmp_path, mvtec, visa, ("visa",), "visa_only"),
    }
    assert set(both) == {"mvtec", "visa"}
    for target in ("mvtec", "visa"):
        assert both[target] == alone[target]
