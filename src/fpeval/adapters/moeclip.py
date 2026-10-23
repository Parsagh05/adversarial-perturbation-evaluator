"""MoECLIP adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .aaclip import AACLIPAdapter, CLIP_BACKBONE_SHA256, CLIP_MEAN, CLIP_STD, _sha256
from .aaclip import resolve_clip_backbone
from .base import register_adapter


OFFICIAL_DATASET_NAMES = {"mvtec": "MVTec", "visa": "VisA"}
# test.py scores a dataset with the adapters trained on the other one (Sec. 4.1).
ZERO_SHOT_TRAINING = {"mvtec": "train_on_visa", "visa": "train_on_mvtec"}
# MoECLIP publishes no weights. train_pipeline/train_moeclip.sh trains both with
# the official train.py defaults and stores the last epoch as
# <checkpoint_root>/<train_on_*>/moe_epoch_20.pth.
CHECKPOINT_NAME = "moe_epoch_20.pth"


def resolve_checkpoint(target_dataset: str, checkpoint_root: str | Path) -> Path:
    key = str(target_dataset).strip().lower()
    if key not in ZERO_SHOT_TRAINING:
        raise ValueError("MoECLIP target_dataset must be 'mvtec' or 'visa'")
    path = Path(checkpoint_root).expanduser().resolve() / ZERO_SHOT_TRAINING[key] / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"MoECLIP checkpoint not found: {path}. MoECLIP releases no weights; "
            "train them with train_pipeline/train_moeclip.sh."
        )
    return path


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model" / "moe_adapter.py",
        root / "model" / "clip.py",
        root / "forward_utils.py",
        root / "utils.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official MoECLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Same top-level module names as AA-CLIP's checkout, which MoECLIP builds on.
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
        importlib.import_module("model.moe_adapter"),
        importlib.import_module("model.clip"),
        importlib.import_module("forward_utils"),
        importlib.import_module("utils"),
    )


@register_adapter("moe-clip")
@register_adapter("moeclip")
class MoECLIPAdapter(AACLIPAdapter):
    """Official MoECLIP zero-shot inference for MVTec AD and VisA.

    Follows test.py at its defaults: OpenAI CLIP ViT-L/14@336 at 518 pixels,
    LoRA mixture-of-experts at layers 5, 11, 17 and 23 (4 experts, top-2, rank 8,
    alpha 16, frozen orthogonal feature separation), patch average aggregation
    with a shared projector, image adapter weight 0.1 and no ReLU (``--relu`` is
    store_true). The model scoring a dataset is the one trained on the other.

    MoECLIP is built on AA-CLIP's code: test.py scores images and maps with the
    same ``calculate_similarity_map(test=True, domain="Industrial")`` and the
    same ``metrics_eval``, so the per-category min/max normalisation and the
    0.5 * map max + 0.5 * score fusion are AA-CLIP's, inherited here and fitted
    on clean references only.

    test.py evaluates every saved epoch; the last one (20) is used, so no epoch
    is selected on the test set.
    """

    name = "moeclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint_root: str | None = None,
        checkpoint: str | None = None,
        download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        seed: int = 111,
        image_adapter_weight: float = 0.1,
        moe_r: int = 8,
        moe_lora_alpha: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_layers: Sequence[int] = (5, 11, 17, 23),
        use_fofs: bool = True,
        use_paa: bool = True,
        seg_proj_sharing_strategy: str = "shared",
        relu: bool = False,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in OFFICIAL_DATASET_NAMES:
            raise ValueError("MoECLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official MoECLIP evaluation uses image_size=518")
        if backbone != "ViT-L-14-336":
            raise ValueError("Official MoECLIP evaluation uses ViT-L-14-336")
        if checkpoint is None:
            if checkpoint_root is None:
                raise ValueError("MoECLIP needs checkpoint or checkpoint_root")
            checkpoint_path = resolve_checkpoint(target_key, checkpoint_root)
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"MoECLIP checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.dataset_name = OFFICIAL_DATASET_NAMES[target_key]

        moe_module, clip_module, forward_module, utils_module = (
            _import_official_repository(repository)
        )
        utils_module.setup_seed(int(seed))

        # model/clip.py expects the OpenAI checkpoint next to it (README); use a
        # verified copy, as for AA-CLIP, which ships the same registry.
        registry = getattr(clip_module, "_MODEL_CKPT_PATHS", None)
        if not isinstance(registry, dict):
            raise RuntimeError("Official MoECLIP model checkpoint registry is missing")
        bundled = Path(registry.get(backbone, ""))
        backbone_path = (
            bundled.resolve()
            if bundled.is_file() and _sha256(bundled) == CLIP_BACKBONE_SHA256
            else resolve_clip_backbone(download_root)
        )
        registry[backbone] = backbone_path

        clip_model = clip_module.create_model(
            model_name=backbone,
            img_size=self.image_size,
            device=self.device,
            pretrained="openai",
            require_pretrained=True,
        )
        clip_model.eval()
        model = moe_module.MoECLIP(
            clip_model=clip_model,
            use_paa=bool(use_paa),
            seg_proj_sharing_strategy=seg_proj_sharing_strategy,
            image_adapt_weight=float(image_adapter_weight),
            moe_r=int(moe_r),
            moe_lora_alpha=int(moe_lora_alpha),
            moe_num_experts=int(moe_num_experts),
            moe_top_k=int(moe_top_k),
            moe_layers=[int(layer) for layer in moe_layers],
            use_fofs=bool(use_fofs),
            relu=bool(relu),
        ).to(self.device)

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        if not isinstance(payload, dict) or "image_adapter" not in payload:
            raise ValueError(f"MoECLIP checkpoint has no image_adapter state: {checkpoint_path}")
        model.image_adapter.load_state_dict(payload["image_adapter"])
        # test.py adapts the text side only when the checkpoint carries it.
        self.uses_text_adapter = "text_adapter" in payload
        if self.uses_text_adapter:
            model.text_adapter.load_state_dict(payload["text_adapter"])
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            embeddings = forward_module.get_adapted_text_embedding(
                model if self.uses_text_adapter else clip_model,
                self.dataset_name,
                self.device,
            )
            self._text_embeddings = {
                category: value.detach() for category, value in embeddings.items()
            }
        self._model = model
        self._forward = forward_module

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "MoECLIP/test.py",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": "openai",
            "image_size": self.image_size,
            "seed": int(seed),
            "image_adapter_weight": float(image_adapter_weight),
            "moe_layers": [int(layer) for layer in moe_layers],
            "moe_num_experts": int(moe_num_experts),
            "moe_top_k": int(moe_top_k),
            "moe_r": int(moe_r),
            "moe_lora_alpha": int(moe_lora_alpha),
            "use_fofs": bool(use_fofs),
            "use_paa": bool(use_paa),
            "seg_proj_sharing_strategy": seg_proj_sharing_strategy,
            "relu": bool(relu),
            "industrial_blur_kernel": [7, 7],
            "industrial_blur_sigma": 1.0,
            "uses_text_adapter": self.uses_text_adapter,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_selection": ZERO_SHOT_TRAINING[target_key] + ", last epoch",
            "clip_backbone": str(backbone_path),
            "clip_backbone_sha256": CLIP_BACKBONE_SHA256,
            "gaussian_applied_inside_adapter": True,
        }

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B,3,H,W]")
        if len(categories) != len(images):
            raise ValueError("One category is required for every MoECLIP image")
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
            # forward returns (seg tokens, detection token, two training losses).
            patch_features, detection_features, _, _ = self._model(images)
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
