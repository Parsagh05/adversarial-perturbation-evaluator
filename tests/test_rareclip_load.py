"""RareCLIP's checkpoint load must not pass a torch.device to Storage.cuda."""

import torch

from fpeval.adapters.rareclip import _gpu_map_location


def _record(monkeypatch):
    seen = {}

    def fake_load(*args, **kwargs):
        seen.update(kwargs)
        return "weights"

    monkeypatch.setattr(torch, "load", fake_load)
    return seen


def test_callable_map_location_becomes_the_gpu(monkeypatch):
    seen = _record(monkeypatch)
    device = torch.device("cuda:0")
    with _gpu_map_location(0):
        # The exact call in rareclip.py.
        result = torch.load("ckpt.pth", map_location=lambda storage, loc: storage.cuda(device))
    assert result == "weights"
    assert seen["map_location"] == "cuda:0"


def test_other_map_locations_and_cpu_are_untouched(monkeypatch):
    seen = _record(monkeypatch)
    with _gpu_map_location(0):
        torch.load("ckpt.pth", map_location=torch.device("cpu"))
    assert seen["map_location"] == torch.device("cpu")
    with _gpu_map_location(-1):
        torch.load("ckpt.pth", map_location=lambda storage, loc: storage)
    assert callable(seen["map_location"])


def test_torch_load_is_restored(monkeypatch):
    original = torch.load
    with _gpu_map_location(0):
        assert torch.load is not original
    assert torch.load is original
