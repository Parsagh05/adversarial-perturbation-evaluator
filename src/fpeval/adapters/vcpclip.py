"""VCP-CLIP adapter matching the official zero-shot evaluation defaults."""

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
import torch.nn.functional as F
from torch import nn

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released VCP weights are individual Google Drive files.
CHECKPOINTS = {
    "train_visa": (
        "train_visa.pth",
        "1MOTaN2hf6ejraTzax6Fnr0fa_b2ZMzjN",
        "bbb15ee3e303d6036d7eb82ecf6ee2419500d12472dc82c9023069b4110a2849",
    ),
    "train_mvtec": (
        "train_mvtec.pth",
        "1uJE25wx2OgSbVPMhR2rbOO5ey0r9UkTr",
        "3e9364a149f34cb70dfed5dd493a34907659ee9955429ee0fdcf2899eb3f94e0",
    ),
}
# test.sh evaluates a dataset with the weights trained on the other one.
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_visa", "visa": "train_mvtec"}

# The README pins this exact OpenAI backbone for the released VCP weights.
CLIP_BACKBONE_SHA256 = (
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
)
CLIP_BACKBONE_NAME = "ViT-L-14-336px.pt"
CLIP_BACKBONE_URL = (
    f"https://openaipublic.azureedge.net/clip/models/{CLIP_BACKBONE_SHA256}/"
    f"{CLIP_BACKBONE_NAME}"
)


class LinearLayer(nn.Module):
    """test.py defines this projection head in the entrypoint, not under models/."""

    def __init__(self, dim_in: int, dim_out: int, k: int, model: str) -> None:
        super().__init__()
        assert "ViT" in model
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens):
        return [self.fc[i](tokens[i][:, 1:, :]) for i in range(len(tokens))]


def _load_stages(model, params, exclude_key: str) -> None:
    """Mirror test.py::_load_stages: copy every matching model parameter."""

    for name, parameter in model.named_parameters():
        if exclude_key in name:
            stored = params[name]
            if parameter.data.size() != stored.data.size():
                raise ValueError(
                    f"VCP-CLIP checkpoint tensor {name} has shape "
                    f"{tuple(stored.data.size())}, model expects "
                    f"{tuple(parameter.data.size())}"
                )
            parameter.data = stored.data.to(parameter.data.device)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "models" / "model_CLIP.py", root / "test.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official VCP-CLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "utils", "dataset"}
            or name.startswith(("models.", "utils.", "dataset."))
        ):
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", []) or [])
        origin = str(
            getattr(module, "__file__", "") or (locations[0] if locations else "")
        )
        if not origin.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("models.model_CLIP"),
        importlib.import_module("models.pre_vcp"),
        importlib.import_module("models.post_vcp"),
        importlib.import_module("models.prompt_ensemble"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_root(download_root: str | Path | None) -> Path:
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "vcpclip"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_clip_backbone(download_root: str | Path | None = None) -> Path:
    """Download and verify the exact OpenAI backbone the README pins."""

    destination = _cache_root(download_root) / CLIP_BACKBONE_NAME
    if destination.is_file() and _sha256(destination) == CLIP_BACKBONE_SHA256:
        return destination
    temporary = destination.with_suffix(".pt.tmp")
    urllib.request.urlretrieve(CLIP_BACKBONE_URL, temporary)
    actual = _sha256(temporary)
    if actual != CLIP_BACKBONE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"OpenAI backbone checksum mismatch: expected {CLIP_BACKBONE_SHA256}, "
            f"got {actual}"
        )
    temporary.replace(destination)
    return destination


def resolve_checkpoint(
    checkpoint: str | Path, *, download_root: str | Path | None = None
) -> Path:
    """Return a local VCP checkpoint, downloading a released name if needed."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"VCP-CLIP checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, file_id, expected = CHECKPOINTS[key]
    destination = _cache_root(download_root) / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading VCP-CLIP weights needs gdown; install the vcpclip extra "
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
            f"VCP-CLIP checkpoint checksum mismatch for {filename}: "
            f"expected {expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def _minmax(values: np.ndarray) -> np.ndarray:
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def _rescale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


@register_adapter("vcp-clip")
@register_adapter("vcpclip")
class VCPCLIPAdapter(ModelAdapter):
    """Official VCP-CLIP zero-shot inference for MVTec AD and VisA.

    calcuate_metric blends the two layer-averaged maps at alpha=0.2, blurs the
    blend at sigma 8 and only then min-max normalizes each category's maps and
    image scores over the whole cohort. Because the blur precedes the
    normalization, this adapter applies it internally and returns full-resolution
    maps, so the evaluator adds no blur of its own; the normalization is fitted on
    the clean cohort and frozen for the adversarial pass.
    """

    name = "vcpclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        download_root: str | None = None,
        clip_backbone_path: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        features: Sequence[int] = (6, 12, 18, 24),
        prompt_len: int = 2,
        deep_prompt_len: int = 1,
        total_d_layer_len: int = 11,
        use_global: bool = True,
        alpha: float = 0.2,
        top_pixels: int = 2000,
        sigma: float = 8.0,
        seed: int = 333,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("VCP-CLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official VCP-CLIP evaluation uses image_size=518")
        if backbone != "ViT-L-14-336":
            raise ValueError("Official VCP-CLIP evaluation uses ViT-L-14-336")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(level) for level in features)
        self.use_global = bool(use_global)
        self.alpha = float(alpha)
        self.top_pixels = int(top_pixels)
        self.sigma = float(sigma)

        clip_module, pre_module, post_module, prompt_module = (
            _import_official_repository(repository)
        )
        from scipy.ndimage import gaussian_filter

        self._gaussian_filter = gaussian_filter
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)
        clip_path = (
            Path(clip_backbone_path).expanduser().resolve()
            if clip_backbone_path
            else resolve_clip_backbone(download_root)
        )
        config_path = (
            Path(repository).expanduser().resolve()
            / "models"
            / "model_configs"
            / f"{backbone}.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"VCP-CLIP model config not found: {config_path}")
        configs = json.loads(config_path.read_text(encoding="utf-8"))

        model, _, _ = clip_module.Load_CLIP(
            self.image_size,
            str(clip_path),
            device=self.device,
            deep_prompt_len=int(deep_prompt_len),
            total_d_layer_len=int(total_d_layer_len),
        )

        linear = LinearLayer(
            configs["vision_cfg"]["width"],
            configs["embed_dim"],
            len(self.features),
            backbone,
        ).to(self.device)
        embed = pre_module.Context_Prompting(configs, cla_len=int(prompt_len)).to(
            self.device
        )
        zero = post_module.Zero_Parameter(
            dim_v=configs["vision_cfg"]["width"],
            dim_t=configs["text_cfg"]["width"],
            dim_out=configs["vision_cfg"]["width"],
        ).to(self.device)
        self._ensemble = prompt_module.Prompt_Ensemble(
            int(prompt_len), clip_module.tokenize
        )

        payload = torch.load(str(checkpoint_path), map_location=self.device)
        linear.load_state_dict(payload["trainable_linearlayer"], strict=False)
        embed.load_state_dict(payload["New_Lan_Embed"])
        zero.load_state_dict(payload["Zero_try"], strict=False)
        _load_stages(model, payload, "prompt")

        for module in (model, linear, embed, zero):
            module.to(self.device).eval().requires_grad_(False)
        self._model, self._linear, self._embed, self._zero = model, linear, embed, zero

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "VCP-CLIP/test.sh",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": "openai",
            "clip_backbone_sha256": CLIP_BACKBONE_SHA256,
            "image_size": self.image_size,
            "features": [int(level) for level in self.features],
            "prompt_len": int(prompt_len),
            "deep_prompt_len": int(deep_prompt_len),
            "total_d_layer_len": int(total_d_layer_len),
            "use_global": self.use_global,
            "map_blend_alpha": self.alpha,
            "image_score_top_pixels": self.top_pixels,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            # calcuate_metric blurs before the per-category min-max, so the blur
            # runs inside the adapter and the evaluator adds none.
            "official_gaussian_sigma": self.sigma,
            "gaussian_applied_inside_adapter": True,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _layer_maps(self, tokens, text, temperature, contextual: bool) -> np.ndarray:
        """Average the per-layer two-class softmax maps, as test.py does."""

        maps = []
        for layer in range(len(tokens)):
            # LinearLayer already drops the class token, encode_image does not.
            dense = tokens[layer][:, 1:, :] if contextual else tokens[layer]
            dense = dense.clone()
            dense = dense / dense.norm(dim=-1, keepdim=True)
            if contextual:
                _, target = self._zero(text.permute(0, 2, 1), dense)
                logits = temperature.exp() * dense @ target.permute(0, 2, 1)
            else:
                logits = temperature.exp() * dense @ text
            count, length, _ = logits.shape
            side = int(np.sqrt(length))
            logits = F.interpolate(
                logits.permute(0, 2, 1).view(count, 2, side, side),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            maps.append(torch.softmax(logits, dim=1)[:, 1, :, :].float().cpu().numpy())
        return np.mean(maps, axis=0)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories  # VCP-CLIP conditions on the image, not on a class name.
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        with torch.no_grad():
            image_features, patch_tokens = self._model.encode_image(
                batch, list(self.features)
            )
            class_token = self._embed.before_extract_feat(
                patch_tokens, image_features.clone(), use_global=self.use_global
            )
            text = self._ensemble.forward_ensemble(
                self._model, class_token, self.device
            ).permute(0, 2, 1)
            new = self._layer_maps(
                patch_tokens, text, self._zero.prompt_temp_l1, contextual=True
            )
            raw = self._layer_maps(
                self._linear(patch_tokens),
                text,
                self._embed.prompt_temp,
                contextual=False,
            )

        # calcuate_metric scores an image by the mean of each map's top-k pixels,
        # taken before the blur and before the normalization.
        flat_raw = raw.reshape(len(raw), -1)
        flat_new = new.reshape(len(new), -1)
        k = min(self.top_pixels, flat_raw.shape[1])
        top_raw = np.partition(flat_raw, kth=-k, axis=1)[:, -k:].mean(axis=1)
        top_new = np.partition(flat_new, kth=-k, axis=1)[:, -k:].mean(axis=1)
        scores = (top_raw + top_new).astype(np.float32)

        blended = self.alpha * raw + (1.0 - self.alpha) * new
        blurred = self._gaussian_filter(blended, sigma=self.sigma, axes=(1, 2))
        return scores, blurred.astype(np.float32)

    def postprocess_anomaly_maps(
        self, maps: np.ndarray, categories: Sequence[str]
    ) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float32)
        result = maps.copy()
        for category in dict.fromkeys(categories):
            selected = np.asarray([name == category for name in categories])
            result[selected] = _minmax(maps[selected])
        return result

    def postprocess_anomaly_maps_with_reference(
        self,
        maps: np.ndarray,
        categories: Sequence[str],
        *,
        reference_maps: np.ndarray,
        reference_categories: Sequence[str],
    ) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float32)
        reference = np.asarray(reference_maps, dtype=np.float32)
        result = maps.copy()
        for category in dict.fromkeys(categories):
            selected = np.asarray([name == category for name in categories])
            matched = np.asarray([name == category for name in reference_categories])
            if not matched.any():
                continue
            block = reference[matched]
            result[selected] = _rescale(
                maps[selected], float(block.min()), float(block.max())
            )
        return result

    def postprocess_image_scores(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
    ) -> np.ndarray:
        del map_mins, map_maxs
        scores = np.asarray(scores, dtype=np.float32)
        result = scores.copy()
        for category in dict.fromkeys(categories):
            selected = np.asarray([name == category for name in categories])
            result[selected] = _minmax(scores[selected])
        return result

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
    ) -> np.ndarray:
        del map_mins, map_maxs, reference_map_mins, reference_map_maxs
        scores = np.asarray(scores, dtype=np.float32)
        reference = np.asarray(reference_scores, dtype=np.float32)
        result = scores.copy()
        for category in dict.fromkeys(categories):
            selected = np.asarray([name == category for name in categories])
            matched = np.asarray([name == category for name in reference_categories])
            if not matched.any():
                continue
            block = reference[matched]
            result[selected] = _rescale(
                scores[selected], float(block.min()), float(block.max())
            )
        return result

    def close(self) -> None:
        self._model = None
        self._linear = None
        self._embed = None
        self._zero = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
