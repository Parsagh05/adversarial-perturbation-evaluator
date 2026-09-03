"""CoPS adapter matching the official zero-shot evaluation defaults."""

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

# shell/test.sh lower-cases the training dataset into the checkpoint path, and
# the repository ships one epoch per training set.
ZERO_SHOT_CHECKPOINT = {
    "mvtec": ("visa", 10),
    "visa": ("mvtec", 5),
}
# test.py branches on whether "visa" appears in the *test* dataset name.
SCORE_WEIGHTS = {"visa": (0.35, 1.0), "mvtec": (0.26, 0.9)}
NEIGHBOUR_KERNEL = {"visa": 3, "mvtec": 5}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "lib" / "cops.py", root / "test.py", root / "utils" / "tools.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official CoPS repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"lib", "utils", "datasets", "dataset", "models"}
            or name.startswith(("lib.", "utils.", "datasets.", "models."))
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
        importlib.import_module("lib"),
        importlib.import_module("lib.cops"),
        importlib.import_module("utils.tools"),
    )


def resolve_checkpoint(
    repository: str | Path, source: str, epoch: int
) -> Path:
    """Return the released checkpoint; CoPS commits its weights in-repo."""

    direct = Path(str(source)).expanduser()
    if direct.is_file():
        return direct.resolve()
    root = Path(repository).expanduser().resolve()
    path = root / "results" / "models" / str(source) / f"epoch_{int(epoch)}.pth"
    if not path.is_file():
        folder = root / "results" / "models"
        available = (
            sorted(str(entry.relative_to(folder)) for entry in folder.rglob("*.pth"))
            if folder.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"CoPS checkpoint not found: {path}. Cloning the repository is the "
            f"download; it currently holds {available}."
        )
    return path


@register_adapter("cops")
class CoPSAdapter(ModelAdapter):
    """Official CoPS zero-shot inference for MVTec AD and VisA.

    Follows shell/test.sh: CLIP ViT-L/14@336px at 518 pixels with DPAM at layer
    24, prompt depth/context 8/12/4, a 6-vector prototype bank, and feature layer
    24 only. MVTec is evaluated with the VisA-trained ``epoch_10`` weights and
    VisA with the MVTec-trained ``epoch_5`` weights, both committed to the
    repository.

    Three details are easy to miss. ``get_fullsize_map`` is called without its
    ``mode`` argument, so the map is upsampled with **nearest** interpolation
    rather than bilinear. The ``alpha``/``beta`` fusion weights and the
    neighbourhood kernel depend on the *test* dataset, not the trained one. And
    the prototype distances are min-max normalized over the whole batch tensor,
    which only matches the official ``batch_size 1`` when images are scored one
    at a time, so the adapter does exactly that.

    The ICTS branch samples 10 latent vectors per image, so the RNG is reset to
    the seed before each image; an image then draws the same latents on the clean
    and the adversarial pass, which a matched comparison requires.
    """

    name = "cops"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        epoch: int | None = None,
        device: str = "cuda",
        image_size: int = 518,
        clip_model: str = "ViT-L/14@336px",
        features: Sequence[int] = (24,),
        depth: int = 8,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        prt_length: int = 6,
        vae_length: int = 1,
        dpam: int = 24,
        seed: int = 0,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("CoPS target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official CoPS evaluation uses image_size=518")
        if clip_model != "ViT-L/14@336px":
            raise ValueError("Official CoPS evaluation uses ViT-L/14@336px")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = [int(level) for level in features]
        self.dpam = int(dpam) if int(dpam) != 0 else None
        self.vae_length = int(vae_length)
        self.seed = int(seed)
        self.alpha, self.beta = SCORE_WEIGHTS[target_key]
        self.kernel = NEIGHBOUR_KERNEL[target_key]

        lib_module, cops_module, tools_module = _import_official_repository(repository)
        from scipy.ndimage import gaussian_filter  # noqa: F401  (evaluator blurs)

        self._compute_similarity = tools_module.compute_similarity
        self._get_fullsize_map = tools_module.get_fullsize_map
        self._average_neighbor = tools_module.average_neighbor
        self._target = target_key
        tools_module.setup_seed(self.seed)

        source, default_epoch = ZERO_SHOT_CHECKPOINT[target_key]
        selected = checkpoint or source
        checkpoint_path = resolve_checkpoint(
            repository, selected, default_epoch if epoch is None else epoch
        )

        hyperparameters = {
            "prompt_length": int(n_ctx),
            "learnable_text_embedding_depth": int(depth),
            "learnable_text_embedding_length": int(t_n_ctx),
            "prt_length": int(prt_length),
            "vae_length": self.vae_length,
        }
        design = hyperparameters if int(t_n_ctx) != 0 else None
        model, _ = lib_module.load(
            clip_model, device=str(self.device), design_details=design
        )
        model.visual.DPAM_replace(DPAM_layer=self.dpam)
        # PromptLearner is built on the CPU copy, exactly as test.py does.
        prompt_learner = cops_module.PromptLearner(
            model.cpu(), hyperparameters, self.features
        )
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        prompt_learner.load_state_dict(payload["prompt_learner"], strict=False)
        model = model.to(self.device).eval().requires_grad_(False)
        prompt_learner = prompt_learner.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._prompt_learner = prompt_learner

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "CoPS/shell/test.sh",
            "target_dataset": target_key,
            "clip_model": clip_model,
            "image_size": self.image_size,
            "features": self.features,
            "depth": int(depth),
            "n_ctx": int(n_ctx),
            "t_n_ctx": int(t_n_ctx),
            "prt_length": int(prt_length),
            "vae_length": self.vae_length,
            "dpam_layer": self.dpam,
            "distance_alpha": self.alpha,
            "score_beta": self.beta,
            "neighbour_kernel": self.kernel,
            "temperature": 0.07,
            "seed": self.seed,
            "checkpoint": str(checkpoint_path.relative_to(Path(repository).resolve()))
            if checkpoint_path.is_relative_to(Path(repository).resolve())
            else checkpoint_path.name,
            "checkpoint_selection": f"{source}/epoch_{default_epoch}",
            # get_fullsize_map defaults to mode="train", i.e. nearest.
            "map_upsample": "nearest",
            "official_gaussian_sigma": 4.0,
            "latent_rng_reset_per_image": True,
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
            raise ValueError("CoPS received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.image_size, self.image_size), dtype=np.float32
        )
        with torch.no_grad():
            for index, raw_category in enumerate(categories):
                self._reseed()
                image = batch[index : index + 1]
                category = str(raw_category)
                image_features, patch_features, _, _ = self._model.encode_image(
                    image, self.features, DPAM_layer=self.dpam
                )

                # ICTS: 10 latent draws decoded into a prompt bias.
                if self.vae_length > 0:
                    z = torch.randn(
                        (10,) + tuple(image_features.shape), device=self.device
                    )
                    z = z.reshape(-1, z.shape[-1])
                    bias = self._prompt_learner.vae_decoder(z).unsqueeze(1).unsqueeze(1)
                else:
                    bias = torch.zeros_like(image_features)

                # ESTS: prototype distances over the neighbourhood-averaged
                # patch features. The kernel follows the test dataset.
                patch_features = [
                    self._average_neighbor(feature, self._target, mode="test")
                    for feature in patch_features
                ]
                fused = torch.stack(patch_features, dim=1).mean(dim=1)
                prototype_n = self._prompt_learner.extractor(
                    self._prompt_learner.t_n, fused
                )
                distance_n = torch.min(
                    1.0
                    - F.cosine_similarity(
                        fused.unsqueeze(2), prototype_n.unsqueeze(1), dim=-1
                    ),
                    dim=2,
                )[0]
                prototype_a = self._prompt_learner.extractor(
                    self._prompt_learner.t_a, fused
                )
                distance_a = torch.min(
                    1.0
                    - F.cosine_similarity(
                        fused.unsqueeze(2), prototype_a.unsqueeze(1), dim=-1
                    ),
                    dim=2,
                )[0]
                prototypes = torch.stack([prototype_n, prototype_a], dim=1)

                text_features = self._prompt_learner(
                    self._model, image.shape[0], prototypes, bias, [category]
                )
                text_features = text_features.mean(dim=2)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                anomaly_map = None
                patch_sim = None
                for feature in patch_features:
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                    patch_sim = self._compute_similarity(feature, text_features)
                    patch_sim = (patch_sim / 0.07).softmax(-1)
                    # test.py reassigns distance_n/distance_a inside this loop,
                    # so the normalization compounds across feature layers.
                    distance_n = (distance_n - distance_n.min()) / (
                        distance_n.max() - distance_n.min()
                    )
                    distance_a = 1 - (distance_a - distance_a.min()) / (
                        distance_a.max() - distance_a.min()
                    )
                    distance = self.alpha * distance_n + (1 - self.alpha) * distance_a
                    patch_sim = patch_sim * distance.unsqueeze(-1)
                    similarity_map = self._get_fullsize_map(patch_sim, self.image_size)
                    layer_map = similarity_map[..., 1]
                    anomaly_map = layer_map if anomaly_map is None else anomaly_map + layer_map
                anomaly_map = anomaly_map / len(self.features)
                maps[index] = anomaly_map[0].float().cpu().numpy()

                normalized = image_features / image_features.norm(dim=-1, keepdim=True)
                image_sim = self._compute_similarity(
                    normalized.unsqueeze(1), text_features
                )
                image_sim = (image_sim / 0.07).softmax(-1)[:, 0, 1]
                peak = torch.amax(patch_sim[..., 1], dim=-1)
                scores[index] = float(
                    (self.beta * image_sim + (1 - self.beta) * peak).item()
                )
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._prompt_learner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
