"""MuSc adapter matching the official scripts/musc.sh (zero-shot, training-free)."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The neighbourhood sizes RsCIN averages over (models/musc.py).
RSCIN_K = {"mvtec": [1, 2, 3], "visa": [1, 8, 9]}


def _load_file(path: Path, name: str):
    """Load one module by path; MuSc's models/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    backbone = root / "models" / "backbone"
    modules = root / "models" / "modules"
    required = (
        backbone / "open_clip", modules / "_LNAMD.py", modules / "_MSM.py", modules / "_RsCIN.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official MuSc repository is incomplete at {root}; missing {missing}"
        )
    backbone_text = str(backbone)
    if backbone_text in sys.path:
        sys.path.remove(backbone_text)
    sys.path.insert(0, backbone_text)
    importlib.invalidate_caches()
    # Other target repositories ship their own open_clip.
    for name in list(sys.modules):
        if name != "open_clip" and not name.startswith("open_clip."):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "") or "") if module else ""
        if module and not module_path.startswith(backbone_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("open_clip"),
        _load_file(modules / "_LNAMD.py", "musc_lnamd"),
        _load_file(modules / "_MSM.py", "musc_msm"),
        _load_file(modules / "_RsCIN.py", "musc_rscin"),
    )


@register_adapter("musc")
class MuScAdapter(ModelAdapter):
    """Official MuSc inference: each test image is scored by the others.

    Per category, every image's patch tokens (layers 6/12/18/24 of OpenAI CLIP
    ViT-L/14@336 at 518 px) are aggregated over neighbourhoods r = 1, 3, 5
    (LNAMD) and scored by mutual scoring against all other images of the set
    (MSM, interval average of the lowest 30%); maps are averaged over layers and
    r and upsampled. The image score is the map maximum refined by RsCIN over
    the class-token similarity graph of the same set. The set is the cohort the
    engine passes, so an attacked image changes every other image's score.
    """

    name = "musc"
    transductive = True

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        feature_layers: Sequence[int] = (5, 11, 17, 23),
        r_list: Sequence[int] = (1, 3, 5),
        batch_size: int = 4,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in RSCIN_K:
            raise ValueError("MuSc target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official MuSc evaluation uses img_resize=518")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        # musc.py adds one to the configured layers.
        self.features = [int(level) + 1 for level in feature_layers]
        self.r_list = [int(r) for r in r_list]
        self.batch_size = int(batch_size)
        self.k_list = RSCIN_K[target_key]
        self._autocast = self.device.type == "cuda"

        open_clip, self._lnamd, self._msm, self._rscin = _import_official_repository(repository)
        model, _, _ = open_clip.create_model_and_transforms(
            backbone, self.image_size, pretrained=pretrained
        )
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "MuSc/scripts/musc.sh",
            "mode": "zero_shot",
            "transductive": "per category, over the cohort scored",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": pretrained,
            "image_size": self.image_size,
            "features": self.features,
            "r_list": self.r_list,
            "msm_topmin": [0.0, 0.3],
            "rscin_k_list": self.k_list,
            "batch_size": self.batch_size,
            # musc.py upsamples the maps and applies no gaussian filter.
            "official_gaussian_sigma": 0.0,
        }

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("MuSc scores sets; the engine calls predict_set")

    def _category(self, images: Sequence[torch.Tensor], indices: list[int]):
        patch_tokens_list, class_tokens = [], []
        for start in range(0, len(indices), self.batch_size):
            batch = torch.stack([images[i] for i in indices[start : start + self.batch_size]])
            batch = (batch.to(self.device, dtype=torch.float32) - self._mean) / self._std
            with torch.no_grad(), torch.autocast("cuda", enabled=self._autocast):
                image_features, patch_tokens = self._model.encode_image(batch, self.features)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            class_tokens.extend(feature.squeeze().cpu().numpy() for feature in image_features)
            patch_tokens_list.append([tokens.cpu() for tokens in patch_tokens])
        feature_dim = patch_tokens_list[0][0].shape[-1]

        maps_r = []
        for r in self.r_list:
            lnamd = self._lnamd.LNAMD(
                device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features
            )
            layers: list[list[torch.Tensor]] = [[] for _ in self.features]
            for patch_tokens in patch_tokens_list:
                tokens = [item.to(self.device) for item in patch_tokens]
                with torch.no_grad(), torch.autocast("cuda", enabled=self._autocast):
                    features = lnamd._embed(tokens)
                    features /= features.norm(dim=-1, keepdim=True)
                for layer in range(len(self.features)):
                    layers[layer].append(features[:, :, layer, :])
            maps_l = []
            for layer in layers:
                z = torch.cat(layer, dim=0).to(self.device)
                maps_l.append(self._msm.MSM(Z=z, device=self.device, topmin_min=0, topmin_max=0.3).cpu())
                del z
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            maps_r.append(torch.stack(maps_l).mean(0))
        maps = torch.stack(maps_r).mean(0).to(self.device)
        count, length = maps.shape
        side = int(np.sqrt(length))
        maps = F.interpolate(
            maps.view(count, 1, side, side), size=self.image_size,
            mode="bilinear", align_corners=True,
        )[:, 0].cpu().numpy()
        peaks = maps.reshape(count, -1).max(-1)
        scores = self._rscin.RsCIN(peaks, class_tokens, k_list=self.k_list)
        return np.asarray(scores, dtype=np.float32), maps.astype(np.float32)

    def predict_set(
        self, images: Sequence[torch.Tensor], categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("MuSc received mismatched images and categories")
        scores = np.empty(len(images), dtype=np.float32)
        maps = np.empty((len(images), self.image_size, self.image_size), dtype=np.float32)
        for category in dict.fromkeys(categories):
            indices = [i for i, name in enumerate(categories) if name == category]
            scores[indices], maps[indices] = self._category(images, indices)
        return scores, maps

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
