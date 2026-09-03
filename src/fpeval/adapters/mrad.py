"""MRAD adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released weights and memory banks live in two Google Drive folders.
CHECKPOINTS = {
    "test_on_mvtec": (
        "test_on_mvtec.pth",
        "1EdqTDU3NSeW1fQIkdZNdlH6iHcOjqQlC",
        "2af68c308183678bcdbe216dd62151d02294bfe726fd51ad4205f663a1a02535",
    ),
    "test_on_visa": (
        "test_on_visa.pth",
        "1Wv9ajL-_6tZcftsyamuaWrFHL-RiBptR",
        "8b46b1358218e8b43872a9d00d13206b8a213315350b57031f6da682694ca524",
    ),
}
MEMORY_BANKS = {
    "cache_model_mvtec": (
        "cache_model_mvtec.pt",
        "1dBMpWfPed0ImjkpkuhqbLWzOEhebI6MV",
        "886431afb17a35d9c72aaeecf46655bd8d126236350566842aa828d6c45fd19e",
    ),
    "cache_model_visa": (
        "cache_model_visa.pt",
        "1Z8vgexJc20TvTnMBRU-NUmmphD2PbNoJ",
        "66dfa8b32ad2c151ea3677a41f830cef0fd406234c7778644fd10bc819d1214b",
    ),
    "cache_patch_model_mvtec": (
        "cache_patch_model_mvtec.pt",
        "1sqavkro1VXZw1vSDJuUQ8Il5EE5yQP0l",
        "f5b1b081c10f6fa7a9fe0cf034be2425de45043f932e6665546b3d16123fbef3",
    ),
    "cache_patch_model_visa": (
        "cache_patch_model_visa.pt",
        "14E6bn3-pau1vg6CQhRLfZ5JPi9OP7MEm",
        "822bccd1c590b515ef49102afa631b1682131a299442fb18065c76f44138f600",
    ),
}
# test.py builds the memory bank from the dataset it is NOT evaluating.
ZERO_SHOT_MEMORY = {"mvtec": "visa", "visa": "mvtec"}
MODEL_TYPES = ("mrad-clip", "mrad-ft", "mrad-tf")


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "mrad.py", root / "test.py", root / "AnomalyCLIP_lib")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official MRAD repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "utils", "mrad", "AnomalyCLIP_lib", "dataset", "datasets"}
            or name.startswith(("models.", "utils.", "AnomalyCLIP_lib."))
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
        importlib.import_module("AnomalyCLIP_lib"),
        importlib.import_module("mrad"),
        importlib.import_module("models.prompt_learner"),
        importlib.import_module("models.mlp"),
        importlib.import_module("models.attention"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(entry, root: Path) -> Path:
    filename, file_id, expected = entry
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading MRAD weights needs gdown; install the mrad extra or "
            "point download_root at a directory that already holds them."
        ) from error
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    gdown.download(id=file_id, output=str(temporary), quiet=True)
    if not temporary.is_file():
        raise RuntimeError(
            f"Google Drive did not return {filename}. Drive rate-limits popular "
            "files; retry later or download it manually into download_root."
        )
    actual = _sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"MRAD checksum mismatch for {filename}: expected {expected}, "
            f"got {actual}"
        )
    temporary.replace(destination)
    return destination


def _cache_root(download_root: str | Path | None) -> Path:
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "mrad"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_checkpoint(
    checkpoint: str | Path, *, download_root: str | Path | None = None
) -> Path:
    """Return a local MRAD checkpoint, downloading a released name if needed."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"MRAD checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    return _download(CHECKPOINTS[key], _cache_root(download_root))


def resolve_memory_banks(
    source: str, *, download_root: str | Path | None = None
) -> tuple[Path, Path]:
    """Return the image and patch memory banks built from ``source``."""

    root = _cache_root(download_root)
    keys = (f"cache_model_{source}", f"cache_patch_model_{source}")
    missing = [key for key in keys if key not in MEMORY_BANKS]
    if missing:
        raise KeyError(f"MRAD has no released memory bank named {missing}")
    return tuple(_download(MEMORY_BANKS[key], root) for key in keys)


@register_adapter("mrad")
class MRADAdapter(ModelAdapter):
    """Official MRAD zero-shot inference for MVTec AD and VisA.

    Follows the README quick start and test.py defaults: CLIP ViT-L/14@336px at
    518 pixels with DPAM at layer 24, prompt depth/context 9/12/4, retrieval over
    memory banks built from the *other* dataset, and the ``mrad-clip`` variant
    with its released weights. The image score fuses the CLIP-side probability
    with the mean of the top 1 percent of map pixels at ``k=0.7``, and the map is
    blurred at sigma 4 before that top-k is taken, so the blur runs inside the
    adapter and the evaluator adds none.

    The memory banks matter for this protocol and are safe here: ``test.py``
    loads them with ``load_cache=True`` before the test loop and nothing writes
    to them while images are scored, so a prediction never depends on which
    images preceded it. ``feature_map_layer`` defaults to ``[0, 1, 2, 3]`` and
    the loop keeps only indices at or above its last element, so exactly one
    projection layer contributes.
    """

    name = "mrad"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        download_root: str | None = None,
        cache_dir: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_type: str = "mrad-clip",
        features: Sequence[int] = (6, 12, 18, 24),
        feature_map_layer: Sequence[int] = (0, 1, 2, 3),
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        dpam: int = 24,
        sigma: float = 4.0,
        k: float = 0.7,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_MEMORY:
            raise ValueError("MRAD target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official MRAD evaluation uses image_size=518")
        if model_type not in MODEL_TYPES:
            raise ValueError(f"MRAD model_type must be one of {MODEL_TYPES}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = [int(level) for level in features]
        self.first_map_layer = int(feature_map_layer[-1])
        self.model_type = model_type
        self.sigma = float(sigma)
        self.k = float(k)

        lib_module, mrad_module, prompt_module, mlp_module, attention_module = (
            _import_official_repository(repository)
        )
        from scipy.ndimage import gaussian_filter

        self._gaussian_filter = gaussian_filter
        self._lib = lib_module
        self._average_neighbor = mlp_module.average_neighbor
        self._compute_score = mrad_module.compute_socre
        self._compute_patch_score = mrad_module.compute_patch_socre
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        source = ZERO_SHOT_MEMORY[target_key]
        selected = checkpoint or f"test_on_{target_key}"
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)
        bank_root = cache_dir or download_root
        image_bank, patch_bank = resolve_memory_banks(source, download_root=bank_root)

        parameters = {
            "Prompt_length": int(n_ctx),
            "learnabel_text_embedding_depth": int(depth),
            "learnabel_text_embedding_length": int(t_n_ctx),
        }
        model, _ = lib_module.load(
            "ViT-L/14@336px", device=str(self.device), design_details=parameters
        )
        model.eval()

        prompt_learner = prompt_module.AnomalyCLIP_PromptLearner(
            model.to("cpu"), parameters
        )
        image_proj = mlp_module.MLP()
        patch_proj = mlp_module.Projector(1024, 768, length=2)
        prompt_proj = mlp_module.AnomalyMLP()

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        image_proj.load_state_dict(payload["image_proj"])
        patch_proj.load_state_dict(payload["patch_proj"])
        if model_type == "mrad-clip":
            prompt_learner.load_state_dict(payload["prompt_learner"])
            prompt_proj.load_state_dict(payload["prompt_proj"])

        model.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=int(dpam))
        for module in (image_proj, patch_proj, prompt_proj, prompt_learner):
            module.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._image_proj = image_proj
        self._patch_proj = patch_proj
        self._prompt_proj = prompt_proj
        self._prompt_learner = prompt_learner

        # Loaded once, before any image is scored, and never written to again.
        self._cache_key, self._cache_value = mrad_module.build_cache_model(
            load_cache=True,
            clip_model=model,
            train_loader_cache=None,
            device=str(self.device),
            dir=str(image_bank),
        )
        self._patch_keys, self._patch_values = mrad_module.build_patch_cache_model(
            load_cache=True,
            clip_model=model,
            train_loader_cache=None,
            device=str(self.device),
            dir=str(patch_bank),
        )

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "MRAD/test.py (README quick start)",
            "target_dataset": target_key,
            "model_type": model_type,
            "clip_model": "ViT-L/14@336px",
            "image_size": self.image_size,
            "features": self.features,
            "feature_map_layer": [int(level) for level in feature_map_layer],
            "contributing_projection_layers": self.first_map_layer,
            "depth": int(depth),
            "n_ctx": int(n_ctx),
            "t_n_ctx": int(t_n_ctx),
            "dpam_layer": int(dpam),
            "score_fusion_k": self.k,
            "score_top_fraction": 0.01,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "memory_bank_source": source,
            "memory_banks": [image_bank.name, patch_bank.name],
            "memory_banks_static": True,
            "official_gaussian_sigma": self.sigma,
            "gaussian_applied_inside_adapter": True,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories  # MRAD retrieves from the memory bank, not by class name.
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.image_size, self.image_size), dtype=np.float32
        )
        use_proj = self.model_type != "mrad-tf"
        with torch.no_grad():
            # The official dataloader uses batch_size 1.
            for index in range(len(batch)):
                image = batch[index : index + 1]
                image_features, patch_features, _, projections = (
                    self._model.encode_image(image, self.features, DPAM_layer=24)
                )

                patch_projection = self._average_neighbor(projections[3])
                patch_projection = patch_projection / patch_projection.norm(
                    dim=-1, keepdim=True
                )
                patch_bias = self._average_neighbor(patch_features[3])
                patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)

                seg_logit, patch_bias, _, _ = self._compute_patch_score(
                    patch_bias,
                    self._patch_keys,
                    self._patch_values,
                    device=str(self.device),
                    proj=self._patch_proj if use_proj else None,
                    need_mask=False,
                    patch_projection=patch_projection,
                    gt_mask=None,
                    is_mradft=(self.model_type != "mrad-clip"),
                    use_proj=use_proj,
                )
                seg_map = self._lib.get_similarity_map(seg_logit, self.image_size)

                if self.model_type == "mrad-clip":
                    bias = self._prompt_proj(patch_bias[:, 0, :], patch_bias[:, 1, :])
                    prompts, tokenized, compound = self._prompt_learner(
                        cls_id=None, bias=bias
                    )
                    text_features = self._model.encode_text_learn(
                        prompts, tokenized, compound
                    ).float()
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                normalized = image_features / image_features.norm(dim=-1, keepdim=True)
                cache_logits, _ = self._compute_score(
                    normalized,
                    self._cache_key,
                    self._cache_value,
                    str(self.device),
                    proj=self._image_proj if use_proj else None,
                    use_proj=use_proj,
                )
                text_probs = cache_logits[:, 1]

                layer_maps = []
                for layer, projection in enumerate(projections):
                    if layer < self.first_map_layer:
                        continue
                    if self.model_type == "mrad-clip":
                        feature = self._average_neighbor(projection)
                        feature = feature / feature.norm(dim=-1, keepdim=True)
                        similarity = self._lib.compute_similarity(
                            feature, text_features
                        )
                        similarity_map = self._lib.get_similarity_map(
                            similarity, self.image_size
                        )
                        layer_maps.append(
                            (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                        )
                    else:
                        layer_maps.append(seg_map[..., 1])

                stacked = torch.stack(layer_maps).detach().cpu().numpy()
                blurred = np.stack(
                    [self._gaussian_filter(single, sigma=self.sigma) for single in stacked]
                )
                anomaly_map = blurred.sum(axis=0)[0]
                maps[index] = anomaly_map.astype(np.float32)

                flat = anomaly_map.reshape(-1)
                top = max(1, int(flat.size * 0.01))
                top_mean = float(np.partition(flat, -top)[-top:].mean())
                scores[index] = float(
                    (1 - self.k) * top_mean + self.k * float(text_probs.item())
                )
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._prompt_learner = None
        self._image_proj = None
        self._patch_proj = None
        self._prompt_proj = None
        self._cache_key = None
        self._cache_value = None
        self._patch_keys = None
        self._patch_values = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
