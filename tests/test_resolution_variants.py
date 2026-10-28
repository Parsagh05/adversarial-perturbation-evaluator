"""anovl_512, promptad_512, iipad_518: the native models run at the perturbation's size."""

import inspect
import json
from pathlib import Path

import pytest
import torch

from fpeval.adapters import adapter_names, base, regime
from fpeval.adapters.iipad import resize_vision_positions

# name -> (native model, constructor parameter that sets the size, its default)
VARIANTS = {
    "anovl_512": ("anovl", "model_image_size", 512),
    "promptad_512": ("promptad", "model_image_size", 512),
    "iipad_518": ("iipad", "model_image_size", 518),
    "dictas_518": ("dictas", "model_image_size", 518),
    "univad_518": ("univad", "model_image_size", 518),
    "kagprompt_518": ("kagprompt", "model_image_size", 518),
    "inctrl_512": ("inctrl", "model_image_size", 512),
    "fade_512": ("fade", "classification_img_size", 512),
    "winclip_accurate_512": ("winclip_accurate", "input_size", 512),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_is_registered_in_its_base_models_regime(name):
    native = VARIANTS[name][0]
    assert name in adapter_names()
    assert regime(name) == regime(native)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_defaults_to_its_size_and_ships_a_config(name):
    native, parameter, size = VARIANTS[name]
    parameters = inspect.signature(base._ADAPTERS[name].__init__).parameters
    assert parameters[parameter].default == size
    config = json.loads(Path(f"configs/{name}.example.json").read_text(encoding="utf-8"))
    native_config = json.loads(Path(f"configs/{native}.example.json").read_text(encoding="utf-8"))
    assert config["model"] == name
    assert config["image_size"] == native_config["image_size"] == 518
    assert config["gaussian_sigma"] == native_config["gaussian_sigma"]


def test_variant_sizes_tile_their_patches():
    # 16-pixel patches for the ViT-B/16+ models, 14 for IIPAD's ViT-L/14.
    assert 512 % 16 == 0 and 518 % 14 == 0


class _Embeddings(torch.nn.Module):
    def __init__(self, grid: int, patch: int = 14, width: int = 8) -> None:
        super().__init__()
        self.patch_size = patch
        self.image_size = grid * patch
        self.num_patches = grid * grid
        self.num_positions = grid * grid + 1
        self.position_embedding = torch.nn.Parameter(torch.randn(1, grid * grid + 1, width))


def test_blip_positions_grow_to_the_new_grid_and_keep_the_class_position():
    embeddings = _Embeddings(grid=16)
    class_position = embeddings.position_embedding.data[:, :1].clone()
    assert resize_vision_positions(embeddings, 518) == 37
    assert embeddings.position_embedding.shape == (1, 37 * 37 + 1, 8)
    assert torch.equal(embeddings.position_embedding.data[:, :1], class_position)
    assert (embeddings.image_size, embeddings.num_patches, embeddings.num_positions) == (518, 1369, 1370)


def test_blip_positions_are_untouched_at_the_native_size():
    embeddings = _Embeddings(grid=16)
    before = embeddings.position_embedding.data.clone()
    assert resize_vision_positions(embeddings, 224) == 16
    assert torch.equal(embeddings.position_embedding.data, before)


def test_anovl_512_softmax_matches_the_vendored_one_and_does_not_overflow():
    from fpeval.adapters.anovl import stable_softmax

    vendored = lambda x: x.exp() / x.exp().sum(dim=-1, keepdim=True)
    small = torch.randn(4, 32) * 5
    assert torch.allclose(stable_softmax(small), vendored(small), atol=1e-6)
    large = small + 100.0  # exp overflows to inf in float32
    assert torch.isnan(vendored(large)).any()
    assert torch.isfinite(stable_softmax(large)).all()
    assert torch.allclose(stable_softmax(large), stable_softmax(small), atol=1e-6)


def test_winclip_matrix_aggregation_matches_the_official_loop():
    from fpeval.adapters.winclip_accurate import (
        _harmonic_aggregation, harmonic_aggregation_matrix,
    )

    grid, kernel = 6, 3
    board = torch.arange(1, grid * grid + 1, dtype=torch.float32).reshape(1, 1, grid, grid)
    mask = torch.nn.functional.unfold(board, kernel_size=kernel, stride=1).squeeze()
    similarity = torch.rand(2, mask.shape[1]) * 0.9 + 0.05
    expected = _harmonic_aggregation((2, grid, grid), similarity, mask)
    assert torch.allclose(harmonic_aggregation_matrix((2, grid, grid), similarity, mask), expected)


def test_blip_positions_reject_a_size_off_the_patch_grid():
    with pytest.raises(ValueError, match="multiple"):
        resize_vision_positions(_Embeddings(grid=16), 512)
