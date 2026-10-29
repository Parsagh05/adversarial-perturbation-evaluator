"""RareCLIP adapter matching the official test.sh (online zero-shot)."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# test.sh evaluates a dataset with the weights trained on the other one; both
# ship in the repository under weights/.
ZERO_SHOT_CHECKPOINT = {"mvtec": "visa_pretrained.pth", "visa": "mvtec_pretrained.pth"}

# test.py's argparse defaults, which test.sh does not override.
OFFICIAL_ARGS = dict(
    model="ViT-L-14-336", pretrained="openai",
    features_list_text=[12, 16, 20, 24], features_list_rare=[6, 12, 18, 24],
    keep_ftime=3, keep_fratio=0.333, keep_snum=200, keep_inum=1000, topk=3,
    LS_ratio=0.01, Rs=1, Rs_freq=20, max_Rs_num=8, Rs_temp=200, text_temp=20,
    sigma=4, rare_thd=0.3, sampler="SCS", k_shot=0, other="",
)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "rareclip.py", root / "sampler.py", root / "open_clip", root / "weights")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official RareCLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (name in {"rareclip", "sampler", "open_clip"} or name.startswith("open_clip.")):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "") or "") if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return root, importlib.import_module("rareclip")


@register_adapter("rareclip")
class RareCLIPAdapter(ModelAdapter):
    """Official RareCLIP inference: an online stream that learns from what it sees.

    Per category the memory is renewed and the test images arrive one at a
    time in a seeded random order (``--seed``, test.sh runs 0-4; the paper
    averages them). Each image is scored against the patch and image rarity of
    the images before it and then added to the memory (``--online 1``), so an
    attacked image changes the scores of every image after it in the stream.
    The order here is the cohort's, shuffled with ``random.seed``.
    """

    name = "rareclip"
    transductive = True

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        stream_seed: int = 0,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("RareCLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official RareCLIP evaluation uses image_size=518")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.stream_seed = int(stream_seed)
        root, module = _import_official_repository(repository)
        checkpoint_path = (
            Path(checkpoint).expanduser() if checkpoint
            else root / "weights" / ZERO_SHOT_CHECKPOINT[target_key]
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RareCLIP checkpoint not found: {checkpoint_path}")
        if self.device.type == "cuda":
            gpu = self.device.index if self.device.index is not None else 0
        else:
            gpu = -1
        args = SimpleNamespace(
            **OFFICIAL_ARGS, image_size=self.image_size, gpu=gpu,
            load_path=str(checkpoint_path.resolve()),
        )
        random.seed(self.stream_seed)
        np.random.seed(self.stream_seed)
        torch.manual_seed(self.stream_seed)
        # RareCLIP reads its model config and caches the CLIP weights through
        # paths relative to the repository.
        previous = os.getcwd()
        os.chdir(root)
        try:
            self._model = module.RareCLIP(args)
        finally:
            os.chdir(previous)
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "RareCLIP/test.sh",
            "mode": "zero_shot",
            "transductive": "online stream per category, update after each image",
            "target_dataset": target_key,
            "image_size": self.image_size,
            "checkpoint": checkpoint_path.name,
            "stream_seed": self.stream_seed,
            "stream_order": "cohort order, then random.seed(stream_seed) shuffle",
            **{name: value for name, value in OFFICIAL_ARGS.items() if name != "other"},
            # The model blurs its own maps (GaussianBlur, sigma 4).
            "official_gaussian_sigma": 0.0,
        }

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("RareCLIP scores streams; the engine calls predict_set")

    def _stream(self, images: Sequence[torch.Tensor], order: list[int], scores, maps) -> None:
        for index in order:
            image = images[index][None].to(self.device, dtype=torch.float32)
            image = (image - self._mean) / self._std
            anomaly_map, anomaly_score = self._model.process_image_and_update(image, update=True)
            scores[index] = float(anomaly_score)
            maps[index] = anomaly_map[0, 0].float().cpu().numpy()

    def predict_set(
        self, images: Sequence[torch.Tensor], categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("RareCLIP received mismatched images and categories")
        scores = np.empty(len(images), dtype=np.float32)
        maps = np.empty((len(images), self.image_size, self.image_size), dtype=np.float32)
        for category in dict.fromkeys(categories):
            order = [i for i, name in enumerate(categories) if name == category]
            random.seed(self.stream_seed)
            random.shuffle(order)
            self._model.renew_memory()
            self._stream(images, order, scores, maps)
        return scores, maps

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
