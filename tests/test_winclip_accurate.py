"""The second WinCLIP port, and what keeps it distinct from the first.

WinCLIP ships no official code, so caoyunkang/WinClip and
zqhang/Accurate-WinCLIP-pytorch are two independent readings of one paper. They
are carried as two target models, and these tests pin the constants that make
that necessary - if a future edit quietly aligned them, the second row would
stop being an independent measurement.
"""

import inspect
import re

import pytest

from fpeval.adapters import adapter_names
from fpeval.adapters import winclip, winclip_accurate


def test_both_ports_are_registered_and_zero_shot():
    from fpeval.adapters.base import regime

    for name in ("winclip", "win-clip",
                 "winclip_accurate", "winclip-accurate", "accurate-winclip"):
        assert name in adapter_names(), name
        assert regime(name) == "zero_shot", name


def test_the_two_ports_are_separate_classes():
    """Aliases of one model, not one model under two names."""
    from fpeval.adapters.base import _ADAPTERS

    assert _ADAPTERS["winclip"] is winclip.WinCLIPAdapter
    assert _ADAPTERS["winclip_accurate"] is winclip_accurate.AccurateWinCLIPAdapter
    assert winclip.WinCLIPAdapter is not winclip_accurate.AccurateWinCLIPAdapter
    assert winclip.WinCLIPAdapter.name != winclip_accurate.AccurateWinCLIPAdapter.name


def test_the_backbone_weights_differ():
    """CLIP_AD hardcodes e31; eval_WinCLIP.py's default is e32."""
    assert winclip_accurate.PRETRAINED == "laion400m_e31"
    base = inspect.signature(winclip.WinCLIPAdapter.__init__).parameters
    assert base["pretrained_dataset"].default == "laion400m_e32"
    assert base["pretrained_dataset"].default != winclip_accurate.PRETRAINED


def test_the_window_geometry_differs():
    """Pixel kernels on a patch grid, against scale counts on a 400-pixel grid."""
    assert winclip_accurate.PATCH_SIZE == 16
    assert winclip_accurate.LARGE_KERNEL == 48        # 3x3 patches
    assert winclip_accurate.MID_KERNEL == 32          # 2x2 patches
    base = inspect.signature(winclip.WinCLIPAdapter.__init__).parameters
    assert base["scales"].default == (2, 3)
    assert base["resolution"].default == 400


def test_neither_port_blurs_its_map():
    """Both leave smoothing to the evaluator, which is told to add none."""
    for module in (winclip, winclip_accurate):
        source = inspect.getsource(module)
        assert "gaussian" not in source.lower() or "official_gaussian_sigma" in source


def test_accurate_port_rejects_settings_its_mask_cannot_serve(tmp_path):
    with pytest.raises(ValueError, match="mvtec"):
        winclip_accurate.AccurateWinCLIPAdapter(
            repository=str(tmp_path), target_dataset="btad"
        )
    with pytest.raises(ValueError, match="ViT-B-16-plus-240"):
        winclip_accurate.AccurateWinCLIPAdapter(
            repository=str(tmp_path), target_dataset="mvtec", backbone="ViT-B-16"
        )
    with pytest.raises(ValueError, match="patch"):
        winclip_accurate.AccurateWinCLIPAdapter(
            repository=str(tmp_path), target_dataset="mvtec", input_size=250
        )


def test_accurate_port_reports_an_incomplete_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        winclip_accurate._import_official_repository(tmp_path)


def test_harmonic_aggregation_matches_the_official_definition():
    """Each patch is the harmonic mean of the windows covering it.

    Reimplemented only to drop the upstream ``.cuda()`` calls, so the arithmetic
    is checked against a hand-computed case rather than trusted.
    """
    import torch

    # Two windows over a 1x2 grid: patch 1 is in window 0, patch 2 in both.
    mask = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
    similarity = torch.tensor([[0.25, 0.5]])
    score = winclip_accurate._harmonic_aggregation((1, 1, 2), similarity, mask)
    # patch 1: covered once, by the 0.25 window -> 1 / (1/0.25) = 0.25
    # patch 2: covered twice -> 2 / (1/0.25 + 1/0.5) = 2 / 6
    assert score.shape == (1, 1, 2)
    assert score[0, 0, 0].item() == pytest.approx(0.25)
    assert score[0, 0, 1].item() == pytest.approx(2.0 / 6.0)


def _upstream_compute_score(image_features, text_features):
    """``reproduce_WinCLIP.compute_score``, verbatim including the in-place /=."""
    import torch

    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    return (torch.bmm(image_features.unsqueeze(1), text_features) / 0.07).softmax(dim=-1)


def _upstream_compute_sim(image_features, text_features):
    """``reproduce_WinCLIP.compute_sim``, verbatim."""
    import torch

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    return (torch.bmm(image_features.squeeze(2), text_features) / 0.07).softmax(dim=-1)


def _adapter_with_text(dimension=8):
    """An adapter carrying only what ``_text_stack`` needs, no repository."""
    import torch

    adapter = object.__new__(winclip_accurate.AccurateWinCLIPAdapter)
    torch.manual_seed(0)
    adapter._text = {
        "bottle": (torch.randn(1, dimension), torch.randn(1, dimension))
    }
    return adapter


def test_text_stack_has_the_shape_the_upstream_scorers_consume():
    """The bug this pins: a ``[1, D]`` mean concatenated on dim 1 gives
    ``[1, 2D]``, a 2-D tensor, and the upstream permute then fails.

    Upstream holds ``[C, 1, D]`` and indexes it, so the tensor reaching
    ``compute_score`` is ``[B, D, 2]``.
    """
    adapter = _adapter_with_text(dimension=8)
    text = adapter._text_stack("bottle", 3)
    assert tuple(text.shape) == (3, 8, 2)


def test_text_stack_feeds_the_real_upstream_scorers():
    """Shapes are checked against the upstream functions, not asserted alone."""
    import torch

    adapter = _adapter_with_text(dimension=8)
    batch, windows = 3, 5

    class_tokens = torch.randn(batch, 8)
    score = _upstream_compute_score(class_tokens, adapter._text_stack("bottle", batch))
    assert tuple(score.shape) == (batch, 1, 2)
    assert tuple(score[:, 0, 1].shape) == (batch,)

    window_tokens = torch.randn(batch, windows, 8)
    similarity = _upstream_compute_sim(
        window_tokens, adapter._text_stack("bottle", batch)
    )
    assert tuple(similarity.shape) == (batch, windows, 2)
    assert tuple(similarity[:, :, 1].shape) == (batch, windows)


def test_text_stack_survives_the_in_place_normalisation():
    """Both scorers divide the text tensor in place.

    A broadcast view cannot take that, and a shared one would corrupt the cache
    that every later batch of this category reads.
    """
    import torch

    adapter = _adapter_with_text(dimension=8)
    before = tuple(item.clone() for item in adapter._text["bottle"])

    text = adapter._text_stack("bottle", 4)
    _upstream_compute_score(torch.randn(4, 8), text)      # divides text in place

    for original, current in zip(before, adapter._text["bottle"]):
        assert torch.equal(original, current), "the cached text features moved"
    # Every row is its own storage, so an in-place write cannot alias.
    text = adapter._text_stack("bottle", 2)
    text[0] += 1.0
    assert not torch.equal(text[0], text[1])


def test_the_extra_covers_every_module_level_import_of_the_upstream_entrypoint():
    """The adapter imports reproduce_WinCLIP, so its imports must resolve.

    They are needed to *load* the module even though inference never calls most
    of them, and a gap surfaces only as a ModuleNotFoundError part-way into a
    run. Listed here as ``import name -> distribution`` exactly as
    ``reproduce_WinCLIP.py``, ``dataset.py`` and ``few_shot.py`` import them.
    """
    import tomllib
    from pathlib import Path

    required = {
        "cv2": "opencv-python-headless",
        "ftfy": "ftfy",                 # via the vendored open_clip tokenizer
        "numpy": "numpy",
        "PIL": "Pillow",
        "regex": "regex",               # via the vendored open_clip tokenizer
        "skimage": "scikit-image",
        "sklearn": "scikit-learn",
        "tabulate": "tabulate",
        "torch": "torch",
        "torchvision": "torchvision",
        "tqdm": "tqdm",
    }
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = (project["project"]["optional-dependencies"]["winclip_accurate"]
                + project["project"]["dependencies"])
    names = {re.split(r"[<>=!~\[]", item)[0].strip().lower() for item in declared}
    for module, distribution in sorted(required.items()):
        assert distribution.lower() in names, (
            f"{module} is imported by the upstream entry point but "
            f"{distribution} is in neither the winclip_accurate extra nor the "
            "core dependencies"
        )


def test_the_extra_does_not_install_open_clip_over_the_vendored_copy():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["winclip_accurate"]
    assert not any("open_clip" in item or "open-clip" in item for item in extra)


def test_the_image_score_definitions_differ():
    """The load-bearing difference, stated in both adapters' metadata."""
    accurate = inspect.getsource(winclip_accurate)
    original = inspect.getsource(winclip)
    assert '"image_score": "abnormal softmax probability of the class token"' in accurate
    assert '"image_score": "max of the anomaly map"' in original
