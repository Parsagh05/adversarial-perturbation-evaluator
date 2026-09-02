"""FB-CLIP adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released weights are published as a Google Drive folder, so they are
# fetched by file ID and verified against a pinned digest.
CHECKPOINTS = {
    "train_on_mvtec": (
        "mvtec_epoch_1_model.pth",
        "1Qw0w-5WeYcVbOlQvrJnjcAgjP9SLhMTC",
        "cb7e9a8d882cd144673f207856aec3ae355c5d043ef43f9e46db3859550b50ab",
    ),
    "train_on_visa": (
        "visa_epoch_2_model.pth",
        "1hzKUafDEpF1KUk6psKnA3anrAGR2nGj4",
        "82cb9fab074da81f94088ded4e1c1ff83ab67dcf5207a2782613f5386fc42a17",
    ),
}
# Evaluate a dataset with the checkpoint that did not train on it.
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_on_visa", "visa": "train_on_mvtec"}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "FBCLIP_lib", root / "prompt_ensemble.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official FB-CLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if not (
            name in {"prompt_ensemble", "FBCLIP_lib", "utils", "loss", "dataset", "metrics"}
            or name.startswith("FBCLIP_lib.")
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("FBCLIP_lib"),
        importlib.import_module("prompt_ensemble"),
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
                f"FB-CLIP checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, file_id, expected = CHECKPOINTS[key]
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "fbclip"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading FB-CLIP weights needs gdown; install the fbclip extra or "
            "pass an existing checkpoint path."
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
            f"FB-CLIP checkpoint checksum mismatch for {filename}: "
            f"expected {expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


@register_adapter("fb-clip")
@register_adapter("fbclip")
class FBCLIPAdapter(ModelAdapter):
    """Official FB-CLIP zero-shot inference for MVTec AD and VisA."""

    name = "fbclip"

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
        prompt_depth: int = 9,
        context_length: int = 12,
        compound_context_length: int = 4,
        feature_layers: Sequence[int] = (1, 6, 12, 18, 24),
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("FB-CLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official FB-CLIP evaluation uses image_size=518")
        if backbone != "ViT-L/14@336px":
            raise ValueError("Official FB-CLIP evaluation uses ViT-L/14@336px")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.feature_layers = [int(layer) for layer in feature_layers]

        library, prompt_module = _import_official_repository(repository)
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
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        learner = prompt_module.FBCLIP_PromptLearner(model.to("cpu"), design)
        learner.to(self.device)
        model.to(self.device)
        # FB_encode only reads args.feature_layers; test.sh overrides the
        # argparse default with 1 6 12 18 24.
        self._args = SimpleNamespace(feature_layers=self.feature_layers)
        model.FB_params(args=self._args, device=self.device)

        payload = torch.load(str(checkpoint_path), map_location=self.device)
        learner.load_state_dict(payload["prompt_learner"])
        trainable = payload.get("model_trainable_params")
        if not trainable:
            raise ValueError(
                f"FB-CLIP checkpoint {checkpoint_path.name} has no "
                "model_trainable_params; the model would keep its random "
                "initialization for those tensors"
            )
        named = dict(model.named_parameters())
        loaded = 0
        for name, data in trainable.items():
            parameter = named.get(name)
            if parameter is None:
                continue
            parameter.data.copy_(data)
            loaded += 1
        if loaded != len(trainable):
            raise ValueError(
                f"FB-CLIP checkpoint {checkpoint_path.name} carries {len(trainable)} "
                f"trainable tensors but only {loaded} matched the model"
            )
        learner.eval().requires_grad_(False)
        model.eval().requires_grad_(False)
        self._model = model
        self._learner = learner

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "FB-CLIP/test.sh",
            "target_dataset": target_key,
            "backbone": backbone,
            "image_size": self.image_size,
            "prompt_depth": int(prompt_depth),
            "context_length": int(context_length),
            "compound_context_length": int(compound_context_length),
            "feature_layers": self.feature_layers,
            "temperature": 0.07,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "trained_epoch": payload.get("epoch"),
            "source_domain": payload.get("source_domain"),
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
            prompts, tokenized, compound = self._learner(cls_id=None)
            labels, masks, _ = self._model.FB_encode(
                batch,
                args=self._args,
                prompts=prompts,
                tokenized_prompts=tokenized,
                compound_prompts_text=compound,
            )
        # masks are [B,1,h,w] at patch resolution; the shared evaluator performs
        # the same bilinear upsample and gaussian_filter the official loop does.
        return (
            labels.reshape(-1).float().cpu().numpy().astype(np.float32),
            masks.squeeze(1).float().cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        self._model = None
        self._learner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
