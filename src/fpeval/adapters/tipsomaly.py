"""Tipsomaly adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


# TIPS is trained without image normalization: create_transforms_tips uses
# mean (0, 0, 0) and std (1, 1, 1), so inputs stay in [0, 1].
IMAGE_MEAN = (0.0, 0.0, 0.0)
IMAGE_STD = (1.0, 1.0, 1.0)

# reproduce.sh evaluates a dataset with the checkpoint that did not train on it.
ZERO_SHOT_CHECKPOINT = {
    "mvtec": "trained_on_visa_default",
    "visa": "trained_on_mvtec_default",
}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model" / "omaly",
        root / "model" / "tips" / "load_model.py",
        root / "datasets" / "input_transforms.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official Tipsomaly repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if not (
            name in {"model", "datasets", "utils", "dataset"}
            or name.startswith(("model.", "datasets.", "utils."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("model.tips.load_model"),
        importlib.import_module("model.omaly"),
    )


def resolve_checkpoint(
    repository: str | Path, checkpoint: str, epoch: int = 2
) -> Path:
    """Return the released learnable-prompt file; Tipsomaly ships them in-repo."""

    text = str(checkpoint).strip()
    direct = Path(text).expanduser()
    if direct.is_file():
        return direct.resolve()
    root = Path(repository).expanduser().resolve()
    folder = root / "workspaces" / text / "vegan-arkansas" / "checkpoints"
    path = folder / f"learnable_params_{int(epoch)}.pth"
    if not path.is_file():
        available = (
            sorted(entry.name for entry in (root / "workspaces").iterdir() if entry.is_dir())
            if (root / "workspaces").is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Tipsomaly checkpoint not found: {path}. Pass a path or one of {available}."
        )
    return path


@register_adapter("tipsomaly")
class TipsomalyAdapter(ModelAdapter):
    """Official Tipsomaly zero-shot inference for MVTec AD and VisA.

    Follows reproduce.sh: the TIPS ``l14h`` backbone, industrial fixed prompts,
    decoupled prompting (fixed prompts for the image score, learnable prompts
    for the map) and local-to-global aggregation.
    """

    name = "tipsomaly"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        models_dir: str,
        checkpoint: str | None = None,
        epoch: int = 2,
        device: str = "cuda",
        image_size: int = 518,
        model_version: str = "l14h",
        fixed_prompt_type: str = "industrial",
        prompt_learn_method: str = "concat",
        n_prompt: int = 8,
        n_deep_tokens: int = 0,
        d_deep_tokens: int = 0,
        decoupled_prompt: bool = True,
        aggregate_local2global: bool = True,
        cls_token_index: int = 0,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("Tipsomaly target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official Tipsomaly evaluation uses image_size=518")
        if cls_token_index not in (0, 1):
            raise ValueError("cls_token_index must be 0 or 1")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.decoupled_prompt = bool(decoupled_prompt)
        self.aggregate_local2global = bool(aggregate_local2global)
        self.cls_token_index = int(cls_token_index)

        load_model, omaly = _import_official_repository(repository)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(repository, selected, epoch)

        # ensure_model_files downloads the TIPS backbone into models_dir.
        vision_backbone, text_backbone, tokenizer, temperature = load_model.get_model(
            str(Path(models_dir).expanduser().resolve()), model_version
        )
        self._temperature = temperature
        self._text = omaly.text_encoder(
            tokenizer,
            text_backbone.to(self.device),
            "tips",
            text_backbone.transformer.width,
            64,
            prompt_learn_method,
            fixed_prompt_type,
            int(n_prompt),
            int(n_deep_tokens),
            int(d_deep_tokens),
        )
        self._vision = omaly.vision_encoder(vision_backbone.to(self.device), "tips")

        payload = torch.load(str(checkpoint_path), weights_only=False)
        self._text.learnable_prompts = (
            payload["learnable_prompts"] if isinstance(payload, dict) else payload
        )
        with torch.no_grad():
            learnable = self._text(["object"], self.device, learned=True)
            self._learnable_text = (
                learnable / learnable.norm(dim=-1, keepdim=True)
            ).detach()
        self._fixed_cache: dict[str, torch.Tensor] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "Tipsomaly/reproduce.sh",
            "target_dataset": target_key,
            "backbone_name": "tips",
            "model_version": model_version,
            "image_size": self.image_size,
            "image_normalization": "none (mean 0, std 1)",
            "fixed_prompt_type": fixed_prompt_type,
            "prompt_learn_method": prompt_learn_method,
            "n_prompt": int(n_prompt),
            "decoupled_prompt": self.decoupled_prompt,
            "aggregate_local2global": self.aggregate_local2global,
            # The official loop reports two image scores, one per TIPS CLS token.
            "cls_token_index": self.cls_token_index,
            "reported_image_scores": 2,
            "temperature": float(temperature),
            "epoch": int(epoch),
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            # regrid_upsample_smooth bilinear-upsamples then applies
            # gaussian_filter(sigma=4); the shared evaluator does both, so
            # gaussian_sigma stays 4.0. Its (1 - p0 + p1) / 2 reduces to p1
            # under a two-class softmax, and bilinear preserves that.
            "official_gaussian_sigma": 4.0,
        }
        mean = torch.tensor(IMAGE_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _fixed_text(self, category: str) -> torch.Tensor:
        cached = self._fixed_cache.get(category)
        if cached is None:
            with torch.no_grad():
                features = self._text(
                    [category.replace("_", " ")], self.device, learned=False
                )
            cached = (features / features.norm(dim=-1, keepdim=True)).detach()
            self._fixed_cache[category] = cached
        return cached

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("Tipsomaly received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        side = None
        collected = []
        with torch.no_grad():
            for index, category in enumerate(categories):
                features = self._vision(batch[index : index + 1])
                features = [f / f.norm(dim=-1, keepdim=True) for f in features]
                fixed = self._fixed_text(str(category))
                # decoupled_prompt: fixed prompts score the image, learnable
                # prompts score the map.
                cls_text = fixed if self.decoupled_prompt else self._learnable_text
                seg_text = self._learnable_text
                score = (
                    self._temperature * features[self.cls_token_index] @ cls_text.T
                ).softmax(dim=-1).squeeze(dim=1)
                patch = (
                    self._temperature * features[2] @ seg_text.T
                ).softmax(dim=-1)
                if self.aggregate_local2global:
                    score = score + torch.max(patch, dim=1)[0]
                scores[index] = float(score.reshape(-1)[1])
                anomaly = patch[..., 1]
                side = int(anomaly.shape[1] ** 0.5)
                collected.append(anomaly.reshape(1, side, side))
        maps = torch.cat(collected, dim=0)
        return scores, maps.float().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._vision = None
        self._text = None
        self._fixed_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
