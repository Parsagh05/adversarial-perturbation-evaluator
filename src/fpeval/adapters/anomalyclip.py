"""Adapter for the official AnomalyCLIP implementation."""

from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _official_modules(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    if not (root / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(f"AnomalyCLIP_lib was not found below {root}")
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # The official project uses top-level module names. Avoid accidentally
    # reusing modules imported from a different checkout.
    for module_name in ("prompt_ensemble", "utils"):
        loaded = sys.modules.get(module_name)
        loaded_path = str(getattr(loaded, "__file__", "")) if loaded else ""
        if loaded and not loaded_path.startswith(root_text):
            del sys.modules[module_name]
    return importlib.import_module("AnomalyCLIP_lib"), importlib.import_module(
        "prompt_ensemble"
    )


@register_adapter("anomalyclip")
class AnomalyCLIPAdapter(ModelAdapter):
    """Official ViT-L/14@336px model with the released prompt checkpoint."""

    name = "anomalyclip"

    def __init__(
        self,
        *,
        repository: str,
        checkpoint: str,
        device: str = "cuda",
        image_size: int = 518,
        features: Sequence[int] = (6, 12, 18, 24),
        map_layers: Sequence[int] = (0, 1, 2, 3),
        prompt_depth: int = 9,
        context_length: int = 12,
        compound_context_length: int = 4,
        dpam_layer: int = 20,
        backbone: str = "ViT-L/14@336px",
        download_root: str | None = None,
        seed: int = 111,
    ) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(value) for value in features)
        self.map_layers = frozenset(int(value) for value in map_layers)
        self.dpam_layer = int(dpam_layer)
        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "AnomalyCLIP/test.py",
            "backbone": backbone,
            "image_size": self.image_size,
            "features": list(self.features),
            "map_layers": sorted(self.map_layers),
            "prompt_depth": int(prompt_depth),
            "context_length": int(context_length),
            "compound_context_length": int(compound_context_length),
            "dpam_layer": self.dpam_layer,
            "temperature": 0.07,
            "seed": int(seed),
            "official_gaussian_sigma": 4.0,
        }
        library, prompts_module = _official_modules(repository)
        self._library = library

        design = {
            "Prompt_length": int(context_length),
            "learnabel_text_embedding_depth": int(prompt_depth),
            "learnabel_text_embedding_length": int(compound_context_length),
        }
        load_options: dict[str, object] = {
            "device": self.device,
            "design_details": design,
        }
        if download_root:
            load_options["download_root"] = download_root
        self._model, _ = library.load(backbone, **load_options)
        self._model.eval()

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AnomalyCLIP checkpoint not found: {checkpoint_path}")
        learner = prompts_module.AnomalyCLIP_PromptLearner(
            self._model.to("cpu"), design
        )
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # Compatibility with older PyTorch releases.
            state = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state, dict) or "prompt_learner" not in state:
            raise ValueError("The checkpoint does not contain 'prompt_learner'")
        learner.load_state_dict(state["prompt_learner"])
        learner.to(self.device).eval().requires_grad_(False)
        self._model.to(self.device).eval().requires_grad_(False)
        self._model.visual.DAPM_replace(DPAM_layer=self.dpam_layer)

        with torch.inference_mode():
            prompt, tokens, compound = learner(cls_id=None)
            text = self._model.encode_text_learn(prompt, tokens, compound).float()
            text = torch.stack(torch.chunk(text, chunks=2, dim=0), dim=1)
            self._text = F.normalize(text, dim=-1).detach()
        self._learner = learner

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B,3,H,W]")
        if len(categories) != len(images):
            raise ValueError("One category is required for every image")
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        image = images.to(self.device)
        mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        image = (image - mean) / std
        with torch.inference_mode():
            global_features, patch_features = self._model.encode_image(
                image, list(self.features), DPAM_layer=self.dpam_layer
            )
            global_features = F.normalize(global_features.float(), dim=-1)
            image_logits = global_features @ self._text[0].t()
            image_scores = (image_logits / 0.07).softmax(dim=-1)[:, 1]

            selected_maps: list[torch.Tensor] = []
            for index, patch in enumerate(patch_features):
                if index not in self.map_layers:
                    continue
                similarity, _ = self._library.compute_similarity(
                    F.normalize(patch.float(), dim=-1), self._text[0]
                )
                patch_similarity = similarity[:, 1:, :]
                side = int(round(patch_similarity.shape[1] ** 0.5))
                if side * side != patch_similarity.shape[1]:
                    raise ValueError("AnomalyCLIP returned a non-square patch grid")
                patch_similarity = patch_similarity.reshape(-1, side, side, 2)
                selected_maps.append(
                    (patch_similarity[..., 1] + 1.0 - patch_similarity[..., 0]) / 2.0
                )
            if not selected_maps:
                raise ValueError("map_layers did not select an AnomalyCLIP feature map")
            anomaly_maps = torch.stack(selected_maps).sum(dim=0)
        return (
            image_scores.cpu().numpy().astype(np.float32),
            anomaly_maps.cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        for name in ("_text", "_learner", "_model"):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)
