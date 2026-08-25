"""AdaCLIP adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released weights are distributed through the authors' HuggingFace Space.
# The README links Google Drive only, which cannot be fetched unattended.
CHECKPOINT_BASE = (
    "https://huggingface.co/spaces/Caoyunkang/AdaCLIP/resolve/main/weights/"
)
CHECKPOINTS = {
    "mvtec_colondb": (
        "pretrained_mvtec_colondb.pth",
        "be51a42c052bd4cf060e54f503a1f5d0b2a3b899bc8dc2e243042f18b215427e",
    ),
    "visa_clinicdb": (
        "pretrained_visa_clinicdb.pth",
        "3deabbbaf1e412cfdfcb42923a500b986f4b9ee96ccbc7a735d89dbc87df44c8",
    ),
    "all": (
        "pretrained_all.pth",
        "33e8d3db1cb4aab030866b8b70a46e10aa27ebf2c23b5463cb07f2574addd98c",
    ),
}
# test.sh evaluates a dataset with the checkpoint that did not train on it.
ZERO_SHOT_CHECKPOINT = {"mvtec": "visa_clinicdb", "visa": "mvtec_colondb"}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "method" / "trainer.py",
        root / "method" / "adaclip.py",
        root / "method" / "custom_clip.py",
        root / "tools" / "__init__.py",
        root / "loss.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official AdaCLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # AnomalyCLIP also ships a top-level ``loss`` module, so purge any module
    # of these names that was resolved from a different model checkout.
    for name in list(sys.modules):
        if not (
            name in {"method", "tools", "loss", "config", "dataset"}
            or name.startswith(("method.", "tools.", "dataset."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("method"),
        importlib.import_module("tools"),
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
                f"AdaCLIP checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, expected = CHECKPOINTS[key]
    root = Path(download_root).expanduser().resolve() if download_root else (
        Path.home() / ".cache" / "adaclip"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    temporary = destination.with_suffix(".pth.tmp")
    urllib.request.urlretrieve(CHECKPOINT_BASE + filename, temporary)
    actual = _sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"AdaCLIP checkpoint checksum mismatch for {filename}: "
            f"expected {expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


@register_adapter("ada-clip")
@register_adapter("adaclip")
class AdaCLIPAdapter(ModelAdapter):
    """Official AdaCLIP zero-shot inference for MVTec AD and VisA."""

    name = "adaclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        download_root: str | None = None,
        clip_cache_dir: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        seed: int = 111,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        prompting_type: str = "SD",
        prompting_branch: str = "VL",
        use_hsf: bool = True,
        k_clusters: int = 20,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("AdaCLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official AdaCLIP evaluation uses image_size=518")
        if backbone != "ViT-L-14-336":
            raise ValueError("Official AdaCLIP evaluation uses ViT-L-14-336")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            # The prompt layers cast their learned context to half precision, so
            # the official forward pass only composes under CUDA autocast.
            raise ValueError(
                "AdaCLIP requires a CUDA device: its prompt layers call .half() "
                "and the official evaluation runs under torch.cuda.amp.autocast"
            )
        self.image_size = int(image_size)
        method_module, tools_module = _import_official_repository(repository)
        tools_module.setup_seed(int(seed))

        config_path = Path(repository).expanduser().resolve() / "model_configs" / f"{backbone}.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"AdaCLIP model config not found: {config_path}")
        model_configs = json.loads(config_path.read_text(encoding="utf-8"))
        layers = int(model_configs["vision_cfg"]["layers"])
        substage = layers // 4
        features = [substage, substage * 2, substage * 3, substage * 4]

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "AdaCLIP/test.py",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": "openai",
            "image_size": self.image_size,
            "feature_levels": features,
            "seed": int(seed),
            "prompting_depth": int(prompting_depth),
            "prompting_length": int(prompting_length),
            "prompting_type": prompting_type,
            "prompting_branch": prompting_branch,
            "use_hsf": bool(use_hsf),
            "k_clusters": int(k_clusters),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "batch_size_note": "official test.py supports batch size 1 only",
            "official_gaussian_sigma": 4.0,
        }

        model = method_module.AdaCLIP_Trainer(
            backbone=backbone,
            feat_list=features,
            input_dim=int(model_configs["vision_cfg"]["width"]),
            output_dim=int(model_configs["embed_dim"]),
            learning_rate=0.0,
            device=str(self.device),
            image_size=self.image_size,
            prompting_depth=int(prompting_depth),
            prompting_length=int(prompting_length),
            prompting_branch=prompting_branch,
            prompting_type=prompting_type,
            use_hsf=bool(use_hsf),
            k_clusters=int(k_clusters),
        ).to(self.device)
        # The official ``load`` is strict=False, so a renamed module would leave
        # the prompters at their random initialization without any error. Every
        # checkpoint tensor must be consumed; only frozen CLIP weights may be
        # missing from the checkpoint.
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        incompatible = model.load_state_dict(payload, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"AdaCLIP checkpoint {checkpoint_path.name} has "
                f"{len(incompatible.unexpected_keys)} tensors the model does not "
                f"accept, e.g. {incompatible.unexpected_keys[:3]}"
            )
        model.eval().requires_grad_(False)
        model.clip_model.eval()
        self._model = model
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std
        del clip_cache_dir

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("AdaCLIP received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.image_size, self.image_size), dtype=np.float32)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # The official text-prompt layer is documented as incompatible with
            # multi-image batches, and test.py refuses any batch size above one.
            for index, category in enumerate(categories):
                anomaly_map, anomaly_score = self._model.clip_model(
                    batch[index : index + 1], [str(category)], aggregation=True
                )
                scores[index] = float(anomaly_score[0])
                maps[index] = anomaly_map[0].float().cpu().numpy()
        return scores, maps

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            self._model = None
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
