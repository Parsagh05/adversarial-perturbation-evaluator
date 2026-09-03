"""Bayes-PFL adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released Bayes-PFL weights are individual Google Drive files.
CHECKPOINTS = {
    "train_visa": (
        "train_visa.pth",
        "1rNs_rdTmrg4JshmKHotq6AN1gqTpPjvm",
        "b3d89b6a6018679e44f413ce4cb0931626bedbd480829d6fba94f2176f270fc3",
    ),
    "train_mvtec": (
        "train_mvtec.pth",
        "1EHa4jPi7r8jmRVURoZ4yH2-jqDt6K1Ni",
        "eb283cd875d997104b275f5b7a232dfefc7b733309d44a374780595cccc4058b",
    ),
}
# "train_visa" is the auxiliary training set, so it evaluates everything else.
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_visa", "visa": "train_mvtec"}

# The README requires this exact OpenAI backbone for the released weights.
CLIP_BACKBONE_SHA256 = (
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
)
CLIP_BACKBONE_NAME = "ViT-L-14-336px.pt"
CLIP_BACKBONE_URL = (
    f"https://openaipublic.azureedge.net/clip/models/{CLIP_BACKBONE_SHA256}/"
    f"{CLIP_BACKBONE_NAME}"
)

# calcuate_metric_pixel takes the mean of only the top 20 pixels for these
# fine-grained categories, and of the top 2000 for every other one.
FINE_GRAINED_CATEGORIES = frozenset(
    {
        "capsules", "macaroni1", "macaroni2", "pipe_fryum",
        "screw", "cashew", "chewinggum",
    }
)
FINE_GRAINED_TOP_PIXELS = 20
DEFAULT_TOP_PIXELS = 2000


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "models" / "VPB.py", root / "models" / "model_CLIP.py", root / "test.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official Bayes-PFL repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "datasets", "open_clip_local", "utils", "dataset"}
            or name.startswith(("models.", "open_clip_local.", "utils.", "dataset."))
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
        importlib.import_module("models.VPB"),
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
        else Path.home() / ".cache" / "bayespfl"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_clip_backbone(download_root: str | Path | None = None) -> Path:
    """Download and verify the exact OpenAI backbone the README requires."""

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
    """Return a local checkpoint, downloading a released name if needed."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Bayes-PFL checkpoint not found: {path}. Pass a path or one of "
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
            "Downloading Bayes-PFL weights needs gdown; install the bayespfl "
            "extra or pass an existing checkpoint path."
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
            f"Bayes-PFL checkpoint checksum mismatch for {filename}: expected "
            f"{expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def top_pixels_for(category: str) -> int:
    """Return the official per-category top-k used for the map-side score."""

    return (
        FINE_GRAINED_TOP_PIXELS
        if str(category) in FINE_GRAINED_CATEGORIES
        else DEFAULT_TOP_PIXELS
    )


def _top_mean(maps: np.ndarray, categories: Sequence[str]) -> np.ndarray:
    """Mean of each map's top-k pixels, with k chosen per category."""

    values = np.empty(len(maps), dtype=np.float64)
    for index, (single, category) in enumerate(zip(maps, categories)):
        flat = np.asarray(single, dtype=np.float64).reshape(-1)
        k = min(top_pixels_for(category), flat.size)
        values[index] = np.partition(flat, -k)[-k:].mean()
    return values


def _minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    # calcuate_metric_pixel divides by (max - min + 1e-8).
    return (values - low) / (high - low + 1e-8)


@register_adapter("bayes-pfl")
@register_adapter("bayespfl")
class BayesPFLAdapter(ModelAdapter):
    """Official Bayes-PFL zero-shot inference for MVTec AD and VisA.

    Follows test.sh: ViT-L-14-336 on the pinned OpenAI backbone at 518 pixels,
    features 6/12/18/24, a prompt bank of 3 prompts with 5 context and 5 state
    tokens, 10 planar flows and 10 Monte Carlo samples, giving 30 prompt draws
    per image. ``calcuate_metric_pixel`` blurs the maps at sigma 8 without
    normalizing them, and fuses two per-category min-max normalized image scores
    at 0.5/0.5: the Monte Carlo averaged text probability, and the mean of the
    unblurred map's top-k pixels, where k is 20 for the fine-grained VisA
    categories and 2000 otherwise.

    Two properties of the official code need care under a matched clean-versus-
    adversarial protocol, and both are handled here:

    * The prompt sampling is stochastic. The RNG is reset to ``seed`` before each
      image, so an image draws the same 30 prompts on the clean and the
      adversarial pass. The official loop instead lets the RNG advance across the
      test set, which would make the two passes incomparable.
    * ``forward_ensemble`` caches the image-agnostic state latents from the first
      test image it ever sees and reuses them for the rest of the run. That cache
      is deliberately not reset, so the latents are fitted on clean data and
      frozen for the adversarial pass.
    """

    name = "bayespfl"

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
        pretrained: str = "openai",
        features: Sequence[int] = (6, 12, 18, 24),
        prompt_context_len: int = 5,
        prompt_state_len: int = 5,
        prompt_num: int = 3,
        sample_num: int = 10,
        num_flows: int = 10,
        alpha: float = 0.5,
        sigma: float = 8.0,
        seed: int = 333,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("Bayes-PFL target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official Bayes-PFL evaluation uses image_size=518")
        if backbone != "ViT-L-14-336":
            raise ValueError(
                "The released Bayes-PFL weights require the ViT-L-14-336 backbone"
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.draws = int(prompt_num) * int(sample_num)

        clip_module, vpb_module = _import_official_repository(repository)

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)
        clip_path = (
            Path(clip_backbone_path).expanduser().resolve()
            if clip_backbone_path
            else resolve_clip_backbone(download_root)
        )
        config_path = (
            Path(repository).expanduser().resolve()
            / "open_clip_local"
            / "model_configs"
            / f"{backbone}.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"Bayes-PFL model config not found: {config_path}")
        configs = json.loads(config_path.read_text(encoding="utf-8"))

        # test.py reads all of these off the argparse namespace.
        self._args = SimpleNamespace(
            model=backbone,
            pretrained=pretrained,
            image_size=self.image_size,
            features_list=[int(level) for level in features],
            vision_width=configs["vision_cfg"]["width"],
            text_width=configs["text_cfg"]["width"],
            embed_dim=configs["embed_dim"],
            num_flows=int(num_flows),
            prompt_context_len=int(prompt_context_len),
            prompt_num=int(prompt_num),
            prompt_state_len=int(prompt_state_len),
            sample_num=int(sample_num),
            seed=self.seed,
        )

        self._reseed()
        model_clip, _, _ = clip_module.Load_CLIP(
            self.image_size, str(clip_path), device=self.device
        )
        model_clip.to(self.device).eval().requires_grad_(False)
        self._clip = model_clip
        self._tokenizer = clip_module.tokenize
        self._text_encoder = vpb_module.TextEncoder(model_clip, self._args)

        prompting = vpb_module.Context_Prompting(args=self._args).to(self.device)
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        prompting.load_state_dict(payload["MyModel"], strict=True)
        prompting.eval().requires_grad_(False)
        self._prompting = prompting

        # test.py derives the stage from the file name; the released weights
        # carry no "epoch_post" marker, so they run the full second stage.
        self._stage = 1 if "epoch_post" in checkpoint_path.name else 2

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "Bayes-PFL/test.sh",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": pretrained,
            "clip_backbone_sha256": CLIP_BACKBONE_SHA256,
            "image_size": self.image_size,
            "features": self._args.features_list,
            "prompt_context_len": int(prompt_context_len),
            "prompt_state_len": int(prompt_state_len),
            "prompt_num": int(prompt_num),
            "sample_num": int(sample_num),
            "num_flows": int(num_flows),
            "monte_carlo_draws_per_image": self.draws,
            "stage": self._stage,
            "score_fusion_alpha": self.alpha,
            "top_pixels_default": DEFAULT_TOP_PIXELS,
            "top_pixels_fine_grained": FINE_GRAINED_TOP_PIXELS,
            "fine_grained_categories": sorted(FINE_GRAINED_CATEGORIES),
            "seed": self.seed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "official_gaussian_sigma": sigma,
            # Deviations required by the matched clean/adversarial protocol.
            "monte_carlo_rng_reset_per_image": True,
            "state_latents_frozen_after_first_image": True,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _reseed(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("Bayes-PFL received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        draws = self.draws
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.image_size, self.image_size), dtype=np.float32
        )
        with torch.no_grad():
            # The official loop scores one image at a time; its image-score
            # softmax is taken over the flattened batch, so it is only correct
            # for a batch of one.
            for index, category in enumerate(categories):
                self._reseed()
                image = batch[index : index + 1]
                image_features, _, patch_tokens = self._clip.encode_image(
                    image, self._args.features_list
                )
                text_embeddings, _ = self._prompting.forward_ensemble(
                    self._text_encoder,
                    image_features,
                    patch_tokens,
                    [str(category)],
                    self.device,
                    self._tokenizer,
                    mode="test",
                )
                logits, layer_maps = self._prompting(
                    text_embeddings,
                    image_features,
                    patch_tokens,
                    stage=self._stage,
                    mode="test",
                )
                logits = logits.squeeze(2)
                paired = torch.stack([logits[:, :draws], logits[:, draws:]], dim=1)
                scores[index] = float(
                    torch.softmax(paired, dim=1)[:, 1].mean().item()
                )

                total = None
                for layer_map in layer_maps:
                    paired_map = torch.stack(
                        [layer_map[:, :draws], layer_map[:, draws:]], dim=1
                    )
                    averaged = torch.softmax(paired_map, dim=1)[:, 1].mean(dim=1)
                    total = averaged if total is None else total + averaged
                maps[index] = (total / len(layer_maps))[0].float().cpu().numpy()
        return scores, maps

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
            maps=maps,
            reference_maps=maps,
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
        """Fuse the two per-category min-max normalized image scores."""

        del map_mins, map_maxs, reference_map_mins, reference_map_maxs
        if maps is None or reference_maps is None:
            raise ValueError(
                "Bayes-PFL scoring needs the cohort maps to take their top-k mean"
            )
        scores = np.asarray(scores, dtype=np.float64)
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        top = _top_mean(np.asarray(maps), categories)
        reference_top = _top_mean(np.asarray(reference_maps), reference_categories)
        category_array = np.asarray(categories)
        reference_array = np.asarray(reference_categories)

        result = scores.copy()
        for category in dict.fromkeys(categories):
            selected = category_array == category
            matched = reference_array == category
            if not matched.any():
                continue
            text = _minmax(
                scores[selected],
                float(reference_scores[matched].min()),
                float(reference_scores[matched].max()),
            )
            pixel = _minmax(
                top[selected],
                float(reference_top[matched].min()),
                float(reference_top[matched].max()),
            )
            result[selected] = self.alpha * text + (1.0 - self.alpha) * pixel
        return result.astype(np.float32)

    def close(self) -> None:
        self._clip = None
        self._prompting = None
        self._text_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
