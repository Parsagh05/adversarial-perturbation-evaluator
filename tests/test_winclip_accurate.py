"""The second WinCLIP port, and what keeps it distinct from the first.

WinCLIP ships no official code, so caoyunkang/WinClip and
zqhang/Accurate-WinCLIP-pytorch are two independent readings of one paper. They
are carried as two target models, and these tests pin the constants that make
that necessary - if a future edit quietly aligned them, the second row would
stop being an independent measurement.
"""

import inspect

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


def test_the_image_score_definitions_differ():
    """The load-bearing difference, stated in both adapters' metadata."""
    accurate = inspect.getsource(winclip_accurate)
    original = inspect.getsource(winclip)
    assert '"image_score": "abnormal softmax probability of the class token"' in accurate
    assert '"image_score": "max of the anomaly map"' in original
