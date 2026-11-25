"""maps_on_disk keeps the maps in files instead of memory and changes no result."""

import numpy as np

from fpeval import engine

from test_engine import _bidirectional_fixture, _evaluate_pair


def test_spilled_maps_are_file_backed_and_unchanged(tmp_path, monkeypatch):
    maps = [np.random.default_rng(i).random((5, 7), dtype=np.float32) for i in range(3)]
    monkeypatch.setattr(engine, "_MAP_DIR", tmp_path)
    spilled = engine._spill_maps(maps)
    assert all(isinstance(item, np.memmap) for item in spilled)
    for original, copy in zip(maps, spilled):
        assert copy.dtype == np.float32
        assert np.array_equal(original, copy)


def test_off_by_default_keeps_maps_in_memory():
    maps = [np.ones((2, 2), dtype=np.float32)]
    assert engine._MAP_DIR is None
    assert engine._spill_maps(maps)[0] is maps[0]


def _outputs(output):
    names = ("summary.csv", "summary_v2.csv", "category_metrics.csv",
             "category_metrics_v2.csv", "per_image.csv", "thresholds.json")
    return {name: (output / name).read_bytes() for name in names}


def test_attacked_run_is_identical_with_maps_on_disk(tmp_path):
    mvtec = _bidirectional_fixture(tmp_path)
    in_memory = _evaluate_pair(tmp_path, mvtec, "memory")
    on_disk = _evaluate_pair(tmp_path, mvtec, "disk", maps_on_disk=True)
    assert _outputs(in_memory) == _outputs(on_disk)
    # The per-target map folder is gone once the target is done.
    assert not list(on_disk.glob("_maps_*"))
    assert engine._MAP_DIR is None
