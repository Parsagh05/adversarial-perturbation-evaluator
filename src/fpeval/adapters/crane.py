"""Crane adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# test.sh evaluates a dataset with the checkpoint that did not train on it.
ZERO_SHOT_CHECKPOINT = {
    "mvtec": "trained_on_visa_crane",
    "visa": "trained_on_mvtec_crane",
}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "models" / "__init__.py",
        root / "models" / "prompt_ensemble.py",
        root / "utils" / "similarity.py",
        root / "test.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official Crane repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Several target repositories ship modules with these names; drop any that
    # were resolved from a different checkout.
    for name in list(sys.modules):
        if not (
            name in {"models", "utils", "dataset"}
            or name.startswith(("models.", "utils.", "dataset."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("models"),
        importlib.import_module("models.prompt_ensemble"),
        importlib.import_module("models.Crane"),
        importlib.import_module("utils.similarity"),
        importlib.import_module("utils"),
    )


def resolve_checkpoint(
    repository: str | Path, checkpoint: str, epoch: int = 5
) -> Path:
    """Return the released checkpoint path; Crane ships its weights in-repo."""

    text = str(checkpoint).strip()
    direct = Path(text).expanduser()
    if direct.is_file():
        return direct.resolve()
    root = Path(repository).expanduser().resolve()
    path = root / "checkpoints" / text / f"epoch_{int(epoch)}.pth"
    if not path.is_file():
        available = sorted(
            entry.name for entry in (root / "checkpoints").iterdir() if entry.is_dir()
        ) if (root / "checkpoints").is_dir() else []
        raise FileNotFoundError(
            f"Crane checkpoint not found: {path}. Pass a path or one of {available}."
        )
    return path


@register_adapter("crane")
class CraneAdapter(ModelAdapter):
    """Official Crane zero-shot inference for MVTec AD and VisA.

    This is the base ``Crane`` row of the paper's Table 1, which test.sh runs
    with ``--dino_model none``. The ``Crane+`` variant additionally loads DINOv2
    and is not covered here.
    """

    name = "crane"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        epoch: int = 5,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L/14@336px",
        features: Sequence[int] = (6, 12, 18, 24),
        prompt_depth: int = 9,
        context_length: int = 12,
        compound_context_length: int = 4,
        attention_layer: int = 20,
        attn_type: str = "qq+kk+vv",
        soft_mean: bool = True,
        use_scorebase_pooling: bool = True,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("Crane target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official Crane evaluation uses image_size=518")
        if backbone != "ViT-L/14@336px":
            raise ValueError("Official Crane evaluation uses ViT-L/14@336px")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(level) for level in features)
        self.attention_layer = int(attention_layer)
        self.soft_mean = bool(soft_mean)
        self.use_scorebase_pooling = bool(use_scorebase_pooling)

        models, prompt_ensemble, crane_module, similarity, utils = (
            _import_official_repository(repository)
        )
        self._similarity = similarity
        utils.setup_seed(int(seed))

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(repository, selected, epoch)

        # design_details carries the argparse namespace itself, so mirror the
        # fields the official model and prompt learner read from it.
        others = SimpleNamespace(
            image_size=self.image_size,
            features_list=list(self.features),
            attn_type=attn_type,
            both_eattn_dattn=True,
            dino_model="none",
            train_with_img_cls_prob=0.0,
            train_with_img_cls_type="pad_suffix",
            use_scorebase_pooling=self.use_scorebase_pooling,
            soft_mean=self.soft_mean,
            depth=int(prompt_depth),
            n_ctx=int(context_length),
            t_n_ctx=int(compound_context_length),
        )
        design = {
            "Prompt_length": int(context_length),
            "learnabel_text_embedding_depth": int(prompt_depth),
            "learnabel_text_embedding_length": int(compound_context_length),
            "others": others,
        }
        self._args = others
        model, _ = models.load(backbone, device=str(self.device), design_details=design)
        model.visual.replace_with_EAttn(to_layer=self.attention_layer, type=attn_type)
        model = utils.turn_gradient_off(model)

        learner = prompt_ensemble.PromptLearner(model.to("cpu"), design)
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        missing, unexpected = learner.load_state_dict(
            payload["prompt_learner"], strict=True
        )
        if missing or unexpected:
            raise ValueError(
                f"Crane checkpoint {checkpoint_path.name} does not match the prompt "
                f"learner; missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}"
            )
        learner.to(self.device).eval().requires_grad_(False)
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._pooling = crane_module.ScoreBasePooling()

        # train_with_img_cls_prob is 0 in the released setup, so the text side is
        # image-independent and encoded once.
        with torch.no_grad():
            prompts, tokenized, compound, _ = learner()
            text = model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
            self._text = F.normalize(text, dim=-1).to(self.device).detach()

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "Crane/test.sh (base Crane row)",
            "variant": "crane",
            "target_dataset": target_key,
            "backbone": backbone,
            "image_size": self.image_size,
            "features": [int(level) for level in self.features],
            "prompt_depth": int(prompt_depth),
            "context_length": int(context_length),
            "compound_context_length": int(compound_context_length),
            "self_cor_attn_layers": self.attention_layer,
            "attn_type": attn_type,
            "dino_model": "none",
            "soft_mean": self.soft_mean,
            "use_scorebase_pooling": self.use_scorebase_pooling,
            "temperature": 0.07,
            "epoch": int(epoch),
            "seed": int(seed),
            "checkpoint": str(checkpoint_path.relative_to(Path(repository).resolve()))
            if checkpoint_path.is_relative_to(Path(repository).resolve())
            else checkpoint_path.name,
            "checkpoint_selection": selected,
            "official_gaussian_sigma": 4.0,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        with torch.no_grad():
            image_features, patch_list = self._model.encode_image(
                batch, list(self.features), self_cor_attn_layers=self.attention_layer
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            patch_features = [
                patch / patch.norm(dim=-1, keepdim=True) for patch in patch_list
            ]
            patch_features = torch.stack(patch_features)

            pixel_logits_list = [
                self._similarity.calc_similarity_logits(patch, self._text)
                for patch in patch_features
            ]
            if self.soft_mean:
                maps = [
                    self._similarity.regrid_upsample(
                        logits.softmax(dim=-1), self.image_size
                    )
                    for logits in pixel_logits_list
                ]
                score_map = torch.stack(maps).mean(dim=0)
            else:
                maps = [
                    self._similarity.regrid_upsample(logits, self.image_size)
                    for logits in pixel_logits_list
                ]
                score_map = torch.stack(maps).mean(dim=0).softmax(dim=-1)
            anomaly_map = score_map[..., 1]

            if self.use_scorebase_pooling:
                alpha = 0.5
                clustered = self._pooling.forward(patch_features, pixel_logits_list)
                image_features = alpha * clustered + (1 - alpha) * image_features
                image_features = F.normalize(image_features, dim=1)

            image_logits = self._similarity.calc_similarity_logits(
                image_features, self._text
            )
            anomaly_score = image_logits.softmax(dim=-1)[:, 1]
        return (
            anomaly_score.float().cpu().numpy().astype(np.float32),
            anomaly_map.float().cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        self._model = None
        self._pooling = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
