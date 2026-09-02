"""FAPrompt adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released weights are published as a Google Drive folder, so they are
# fetched by file ID and verified against a pinned digest.
CHECKPOINTS = {
    "train_on_mvtecad": (
        "train_on_mvtecad.pth",
        "1XDXtx-17JCIT-EAIpS_R_C2T2S03bzdD",
        "13497ace23f87d3d4bff7e4695b43d4257c7af646da04e8fb8996428c9a4e84f",
    ),
    "train_on_visa": (
        "train_on_visa.pth",
        "1MLZ7OPcO3JawLYG31RZLxNUbtFXqzrsC",
        "82ce5d51c97cedc08f6b27f75094967a3cd7f0b67c45bc0f8daf328638562e9f",
    ),
}
# Evaluate a dataset with the checkpoint that did not train on it.
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_on_visa", "visa": "train_on_mvtecad"}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "FAPrompt.py", root / "AnomalyCLIP_lib")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official FAPrompt repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # AnomalyCLIP ships modules of the same names; drop any resolved from
    # another checkout so FAPrompt's own copies win.
    for name in ("FAPrompt", "AnomalyCLIP_lib", "utils", "loss", "dataset"):
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    for name in list(sys.modules):
        if name.startswith("AnomalyCLIP_lib."):
            module = sys.modules[name]
            if not str(getattr(module, "__file__", "")).startswith(root_text):
                sys.modules.pop(name, None)
    return (
        importlib.import_module("AnomalyCLIP_lib"),
        importlib.import_module("FAPrompt"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    checkpoint: str | Path, *, download_root: str | Path | None = None
) -> Path:
    """Return a local checkpoint path, downloading a released name if needed."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"FAPrompt checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, file_id, expected = CHECKPOINTS[key]
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "faprompt"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading FAPrompt weights needs gdown; install the faprompt extra "
            "or pass an existing checkpoint path."
        ) from error
    temporary = destination.with_suffix(".pth.tmp")
    gdown.download(id=file_id, output=str(temporary), quiet=True)
    if not temporary.is_file():
        raise RuntimeError(
            f"Google Drive did not return {filename}. Drive rate-limits popular "
            "files; retry later or download it manually and pass the path."
        )
    actual = _sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"FAPrompt checkpoint checksum mismatch for {filename}: "
            f"expected {expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


@register_adapter("fa-prompt")
@register_adapter("faprompt")
class FAPromptAdapter(ModelAdapter):
    """Official FAPrompt zero-shot inference for MVTec AD and VisA."""

    name = "faprompt"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        download_root: str | None = None,
        clip_download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L/14@336px",
        features: Sequence[int] = (6, 12, 18, 24),
        prompt_depth: int = 9,
        context_length: int = 12,
        compound_context_length: int = 4,
        dpam_layer: int = 20,
        topk: int = 10,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("FAPrompt target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official FAPrompt evaluation uses image_size=518")
        if backbone != "ViT-L/14@336px":
            raise ValueError("Official FAPrompt evaluation uses ViT-L/14@336px")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(level) for level in features)
        self.dpam_layer = int(dpam_layer)
        self.topk = int(topk)

        library, faprompt_module = _import_official_repository(repository)
        self._library = library
        torch.manual_seed(int(seed))

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)

        design = {
            "Prompt_length": int(context_length),
            "learnabel_text_embedding_depth": int(prompt_depth),
            "learnabel_text_embedding_length": int(compound_context_length),
        }
        load_kwargs = {"device": self.device, "design_details": design}
        if clip_download_root:
            load_kwargs["download_root"] = clip_download_root
        model, _ = library.load(backbone, **load_kwargs)
        model.eval()

        learner = faprompt_module.FAPrompt(model.to("cpu"), design)
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        learner.load_state_dict(payload["prompt_learner"])
        learner.to(self.device).eval().requires_grad_(False)
        model.to(self.device).eval().requires_grad_(False)
        model.visual.DAPM_replace(DPAM_layer=self.dpam_layer)
        self._model = model
        self._learner = learner

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "FAPrompt/test.py",
            "target_dataset": target_key,
            "backbone": backbone,
            "image_size": self.image_size,
            "features": [int(level) for level in self.features],
            "prompt_depth": int(prompt_depth),
            "context_length": int(context_length),
            "compound_context_length": int(compound_context_length),
            "dpam_layer": self.dpam_layer,
            "temperature": 0.07,
            "topk_tokens": self.topk,
            "negative_prompt_count": 10,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "official_gaussian_sigma": 10.0,
            "batch_size_note": "official test.py evaluates one image at a time",
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _text_features(self, selected_tokens: torch.Tensor | None) -> torch.Tensor:
        """Encode the positive prompt and the mean of the ten negative prompts."""

        if selected_tokens is None:
            outputs = self._learner.forward()
        else:
            outputs = self._learner.forward(selected_tokens=selected_tokens)
        prompts_pos, prompts_neg, tokens_pos, tokens_neg, compound, _ = outputs
        positive = self._model.encode_text_learn(prompts_pos, tokens_pos, compound).float()
        negative = self._model.encode_text_learn(prompts_neg, tokens_neg, compound).float()
        negative = torch.mean(negative, dim=0, keepdim=True)
        return torch.cat([positive, negative])

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.image_size, self.image_size), dtype=np.float32)
        with torch.no_grad():
            # The official loop conditions the prompt learner on each image's own
            # top-k patch tokens, so images are processed one at a time.
            for index in range(len(batch)):
                image = batch[index : index + 1]
                image_features, patch_features = self._model.encode_image(
                    image, list(self.features), DPAM_layer=self.dpam_layer
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

                base_text = self._text_features(None)
                base_text = torch.stack(
                    torch.chunk(base_text, dim=0, chunks=2), dim=1
                )
                base_text = base_text / base_text.norm(dim=-1, keepdim=True)

                similarity, _ = self._library.compute_similarity_ori(
                    patch_features, base_text[0]
                )
                similarity_map1 = self._library.get_similarity_map(
                    similarity[1:, :], self.image_size
                ).squeeze(0)
                map_max_score1 = similarity[1:, :, 1].max(dim=0).values

                _, top_indices = torch.topk(
                    similarity[1:, :, 1], k=self.topk, dim=0, largest=True, sorted=True
                )
                top_indices = top_indices.permute(1, 0)
                tokens = patch_features[1:, :, :].permute(1, 0, 2)
                selected = torch.stack([
                    torch.stack([tokens[i, k, :] for k in range(top_indices.shape[1])])
                    for i in range(top_indices.shape[0])
                ])

                conditioned = torch.stack(
                    [self._text_features(selected[i]) for i in range(len(selected))]
                )
                conditioned = conditioned / conditioned.norm(dim=-1, keepdim=True)

                text_probs = image_features @ conditioned.permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)[:, 0, 1]

                similarity2, _ = self._library.compute_similarity(
                    patch_features, conditioned
                )
                map_max_score2 = similarity2[1:, :, 1].max(dim=0).values
                similarity_map2 = self._library.get_similarity_map(
                    similarity2[1:, :], self.image_size
                ).squeeze(0)

                anomaly_map = (
                    similarity_map1[1, :] + 1 - similarity_map1[0, :]
                    + similarity_map2[1, :] + 1 - similarity_map2[0, :]
                ) / 4.0
                map_max_score = (2 * map_max_score1 + map_max_score2) / 3
                score = 0.5 * (text_probs + map_max_score)

                scores[index] = float(score.reshape(-1)[0])
                maps[index] = anomaly_map.float().cpu().numpy()
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._learner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
