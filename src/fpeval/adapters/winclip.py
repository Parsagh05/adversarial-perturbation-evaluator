"""WinCLIP adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "WinCLIP" / "model.py", root / "eval_WinCLIP.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official WinClip repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"WinCLIP", "datasets", "utils", "dataset"}
            or name.startswith(("WinCLIP.", "datasets.", "utils."))
        ):
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", []) or [])
        origin = str(
            getattr(module, "__file__", "") or (locations[0] if locations else "")
        )
        if not origin.startswith(root_text):
            sys.modules.pop(name, None)
    return importlib.import_module("WinCLIP.model")


@register_adapter("win-clip")
@register_adapter("winclip")
class WinCLIPAdapter(ModelAdapter):
    """Official WinCLIP zero-shot inference for MVTec AD and VisA.

    Follows eval_WinCLIP.py at its defaults: the ``ViT-B-16-plus-240`` backbone
    with ``laion400m_e32`` weights, 240-pixel inputs, window scales (2, 3), and
    a 400-pixel output grid. WinCLIP is training-free, so there is no checkpoint
    beyond the backbone, which open_clip downloads on first use.

    With ``k_shot`` at zero the visual gallery stays empty, so the official
    ``textual_visual`` fusion reduces to ``1 / (1/t + 1/t)``, i.e. **half** the
    textual map. That halving is monotonic and cannot change AUROC, but it does
    change the absolute values the frozen thresholds are calibrated against, so
    it is reproduced rather than simplified away. The image score is the maximum
    of the returned map, exactly as ``metric_cal`` computes it, and the official
    loop applies no Gaussian filter, so the shared evaluator adds none.

    The cohort images arrive at 518 pixels because that is the grid the
    perturbations are defined on; the adapter performs the official bicubic
    resize down to 240 itself. The L-infinity budget therefore applies at 518
    and is attenuated by that downsample, which is a property of evaluating a
    240-pixel model against a 518-pixel perturbation, not of this adapter.
    """

    name = "winclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-B-16-plus-240",
        pretrained_dataset: str = "laion400m_e32",
        scales: Sequence[int] = (2, 3),
        input_size: int = 240,
        resolution: int = 400,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("WinCLIP target_dataset must be 'mvtec' or 'visa'")
        if backbone != "ViT-B-16-plus-240":
            raise ValueError("Official WinCLIP evaluation uses ViT-B-16-plus-240")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.input_size = int(input_size)
        self.resolution = int(resolution)

        model_module = _import_official_repository(repository)
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        model = model_module.WinClipAD(
            out_size_h=self.resolution,
            out_size_w=self.resolution,
            device=str(self.device),
            backbone=backbone,
            pretrained_dataset=pretrained_dataset,
            scales=tuple(int(scale) for scale in scales),
            img_resize=self.input_size,
            img_cropsize=self.input_size,
        )
        model = model.to(self.device)
        model.eval_mode()
        if model.visual_gallery is not None:
            raise ValueError("WinCLIP zero-shot evaluation must leave the gallery empty")
        self._model = model
        self._current: str | None = None

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "WinClip/eval_WinCLIP.py",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained_dataset": pretrained_dataset,
            "scales": [int(scale) for scale in scales],
            "model_input_size": self.input_size,
            "map_resolution": self.resolution,
            "cohort_image_size": self.image_size,
            "fusion_version": model.fusion_version,
            "precision": model.precision,
            "k_shot": 0,
            "seed": int(seed),
            # The official gaussian_filter call is commented out.
            "official_gaussian_sigma": 0.0,
            "image_score": "max of the anomaly map",
            "training_free": True,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _ensure_text_gallery(self, category: str) -> None:
        # build_text_feature_gallery overwrites the gallery, so it has to run
        # again whenever the category changes.
        if self._current != category:
            self._model.build_text_feature_gallery(category)
            self._current = category

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("WinCLIP received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        # The official transform resizes to 240 with PIL bicubic and centre-crops
        # to the same size, which is a no-op on a square input.
        batch = self._resize(
            batch,
            [self.input_size, self.input_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.resolution, self.resolution), dtype=np.float32
        )
        with torch.no_grad():
            # The text gallery is per category, so images are grouped by it.
            order = sorted(range(len(categories)), key=lambda i: str(categories[i]))
            for index in order:
                category = str(categories[index])
                self._ensure_text_gallery(category)
                produced = self._model(batch[index : index + 1])
                anomaly_map = np.asarray(produced[0], dtype=np.float32)
                maps[index] = anomaly_map
                # metric_cal scores an image by the maximum of its map.
                scores[index] = float(anomaly_map.max())
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._current = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
