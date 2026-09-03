"""AA-CLIP adapter matching the official industrial evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OFFICIAL_DATASET_NAMES = {"mvtec": "MVTec", "visa": "VisA"}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model" / "adapter.py",
        root / "model" / "clip.py",
        root / "forward_utils.py",
        root / "utils.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official AA-CLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # The repository imports these packages by top-level names. Purge modules
    # belonging to another model checkout before resolving AA-CLIP.
    for name in list(sys.modules):
        if not (
            name in {"dataset", "forward_utils", "model", "utils"}
            or name.startswith(("dataset.", "model."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("model.adapter"),
        importlib.import_module("model.clip"),
        importlib.import_module("forward_utils"),
        importlib.import_module("utils"),
    )


def _load_component(module: torch.nn.Module, checkpoint: str | Path, key: str) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AA-CLIP {key} checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"AA-CLIP checkpoint has no {key!r} state: {path}")
    module.load_state_dict(payload[key])
    return path


@register_adapter("aa-clip")
@register_adapter("aaclip")
class AACLIPAdapter(ModelAdapter):
    """Official AA-CLIP zero-shot inference for MVTec AD and VisA."""

    name = "aaclip"

    def __init__(
        self,
        *,
        repository: str,
        image_checkpoint: str,
        target_dataset: str,
        text_checkpoint: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        seed: int = 111,
        text_adapter_weight: float = 0.1,
        image_adapter_weight: float = 0.1,
        text_adapter_layers: int = 3,
        image_adapter_layers: int = 6,
        feature_levels: Sequence[int] = (6, 12, 18, 24),
        relu: bool = False,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in OFFICIAL_DATASET_NAMES:
            raise ValueError("AA-CLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official AA-CLIP evaluation uses image_size=518")
        if backbone != "ViT-L-14-336":
            raise ValueError("Official AA-CLIP evaluation uses ViT-L-14-336")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.dataset_name = OFFICIAL_DATASET_NAMES[target_key]
        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "AA-CLIP/test.py",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": "openai",
            "image_size": self.image_size,
            "seed": int(seed),
            "text_adapter_weight": float(text_adapter_weight),
            "image_adapter_weight": float(image_adapter_weight),
            "text_adapter_layers": int(text_adapter_layers),
            "image_adapter_layers": int(image_adapter_layers),
            "feature_levels": [int(level) for level in feature_levels],
            "relu": bool(relu),
            "industrial_blur_kernel": [7, 7],
            "industrial_blur_sigma": 1.0,
            "uses_text_adapter": text_checkpoint is not None,
        }
        adapter_module, clip_module, forward_module, utils_module = (
            _import_official_repository(repository)
        )
        utils_module.setup_seed(int(seed))

        clip_model = clip_module.create_model(
            model_name=backbone,
            img_size=self.image_size,
            device=self.device,
            pretrained="openai",
            require_pretrained=True,
        )
        clip_model.eval()
        model = adapter_module.AdaptedCLIP(
            clip_model=clip_model,
            text_adapt_weight=float(text_adapter_weight),
            image_adapt_weight=float(image_adapter_weight),
            text_adapt_until=int(text_adapter_layers),
            image_adapt_until=int(image_adapter_layers),
            levels=[int(level) for level in feature_levels],
            # Official test.py exposes --relu as store_true, hence False by default.
            relu=bool(relu),
        ).to(self.device)
        _load_component(model.image_adapter, image_checkpoint, "image_adapter")

        text_model = clip_model
        self.uses_text_adapter = text_checkpoint is not None
        if text_checkpoint is not None:
            _load_component(model.text_adapter, text_checkpoint, "text_adapter")
            text_model = model
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            embeddings = forward_module.get_adapted_text_embedding(
                text_model, self.dataset_name, self.device
            )
            self._text_embeddings = {
                category: value.detach() for category, value in embeddings.items()
            }
        self._model = model
        self._forward = forward_module

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B,3,H,W]")
        if len(categories) != len(images):
            raise ValueError("One category is required for every AA-CLIP image")
        unknown = sorted(set(categories) - set(self._text_embeddings))
        if unknown:
            raise ValueError(f"Unknown {self.dataset_name} categories: {unknown}")
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        images = images.to(self.device)
        mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        images = (images - mean) / std
        with torch.inference_mode():
            patch_features, detection_features = self._model(images)
            scores = torch.empty(len(images), dtype=torch.float32, device=self.device)
            maps = torch.empty(
                len(images), self.image_size, self.image_size,
                dtype=torch.float32, device=self.device,
            )
            for category in dict.fromkeys(categories):
                positions = [
                    index for index, current in enumerate(categories)
                    if current == category
                ]
                indices = torch.as_tensor(positions, device=self.device)
                text = self._text_embeddings[category]
                raw_score = detection_features[indices] @ text
                scores[indices] = (raw_score[:, 1] + 1.0) / 2.0
                level_maps = [
                    self._forward.calculate_similarity_map(
                        features[indices], text, self.image_size,
                        test=True, domain="Industrial",
                    )
                    for features in patch_features
                ]
                maps[indices] = torch.cat(level_maps, dim=1).sum(dim=1)
        return (
            scores.cpu().numpy().astype(np.float32),
            maps.cpu().numpy().astype(np.float32),
        )

    def postprocess_image_scores(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
        *,
        maps: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.postprocess_image_scores_with_reference(
            scores,
            map_mins,
            map_maxs,
            categories,
            reference_scores=scores,
            reference_map_mins=map_mins,
            reference_map_maxs=map_maxs,
            reference_categories=categories,
        )

    def postprocess_image_scores_with_reference(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
        *,
        reference_scores: np.ndarray,
        reference_map_mins: np.ndarray,
        reference_map_maxs: np.ndarray,
        reference_categories: Sequence[str],
        maps: np.ndarray | None = None,
        reference_maps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Match official category scoring, fitted only on clean references."""

        result = np.asarray(scores, dtype=np.float64).copy()
        map_maxs = np.asarray(map_maxs, dtype=np.float64)
        category_array = np.asarray(categories)
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        reference_map_mins = np.asarray(reference_map_mins, dtype=np.float64)
        reference_map_maxs = np.asarray(reference_map_maxs, dtype=np.float64)
        reference_category_array = np.asarray(reference_categories)
        if not (
            result.shape == np.asarray(map_mins).shape == map_maxs.shape == category_array.shape
        ):
            raise ValueError("AA-CLIP score inputs must have matching vector shapes")
        if not (
            reference_scores.shape
            == reference_map_mins.shape
            == reference_map_maxs.shape
            == reference_category_array.shape
        ):
            raise ValueError("AA-CLIP clean reference inputs must have matching shapes")

        for category in dict.fromkeys(categories):
            selected = category_array == category
            reference_selected = reference_category_array == category
            if not reference_selected.any():
                raise ValueError(f"Clean reference cohort has no {category!r} samples")
            category_scores = result[selected]
            category_map_maxs = map_maxs[selected]
            clean_score_min = float(reference_scores[reference_selected].min())
            clean_score_max = float(reference_scores[reference_selected].max())
            clean_pixel_min = float(reference_map_mins[reference_selected].min())
            clean_pixel_max = float(reference_map_maxs[reference_selected].max())
            # These two min/max operations and the 50/50 aggregation reproduce
            # official metrics_eval; adversarial data never refits the ranges.
            if clean_pixel_max != 1.0:
                if clean_pixel_max == clean_pixel_min:
                    raise ValueError(f"Constant AA-CLIP maps for {category!r}")
                category_map_maxs = (
                    category_map_maxs - clean_pixel_min
                ) / (clean_pixel_max - clean_pixel_min)
            if clean_score_max != 1.0:
                if clean_score_max == clean_score_min:
                    raise ValueError(f"Constant AA-CLIP scores for {category!r}")
                category_scores = (
                    category_scores - clean_score_min
                ) / (clean_score_max - clean_score_min)
            result[selected] = 0.5 * category_map_maxs + 0.5 * category_scores
        return result.astype(np.float32)

    def postprocess_anomaly_maps(
        self, maps: np.ndarray, categories: Sequence[str]
    ) -> np.ndarray:
        return self.postprocess_anomaly_maps_with_reference(
            maps,
            categories,
            reference_maps=maps,
            reference_categories=categories,
        )

    def postprocess_anomaly_maps_with_reference(
        self,
        maps: np.ndarray,
        categories: Sequence[str],
        *,
        reference_maps: np.ndarray,
        reference_categories: Sequence[str],
    ) -> np.ndarray:
        """Apply official per-category map min/max using clean ranges."""

        result = np.asarray(maps, dtype=np.float32).copy()
        reference = np.asarray(reference_maps, dtype=np.float32)
        category_array = np.asarray(categories)
        reference_category_array = np.asarray(reference_categories)
        if result.ndim != 3 or reference.ndim != 3:
            raise ValueError("AA-CLIP anomaly maps must have shape [B,H,W]")
        if len(result) != len(category_array) or len(reference) != len(
            reference_category_array
        ):
            raise ValueError("AA-CLIP maps and categories must have matching lengths")
        for category in dict.fromkeys(categories):
            selected = category_array == category
            reference_selected = reference_category_array == category
            if not reference_selected.any():
                raise ValueError(f"Clean reference maps have no {category!r} samples")
            minimum = float(reference[reference_selected].min())
            maximum = float(reference[reference_selected].max())
            if maximum != 1.0:
                if maximum == minimum:
                    raise ValueError(f"Constant AA-CLIP maps for {category!r}")
                result[selected] = (result[selected] - minimum) / (maximum - minimum)
        return result

    def close(self) -> None:
        for name in ("_text_embeddings", "_model"):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)
