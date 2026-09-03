"""SubspaceAD adapter matching the official few-shot benchmark defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


# scripts/benchmark_few_shot.sh, which is the paper's few-shot protocol.
LAYERS = (-12, -13, -14, -15, -16, -17, -18)
SHOT_VALUES = (1, 2, 4)
# main.py disables augmentation for this category.
NO_AUG_CATEGORIES = frozenset({"transistor"})


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "main.py",
        root / "src" / "subspacead" / "core" / "extractor.py",
        root / "src" / "subspacead" / "core" / "pca.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official SubspaceAD repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # The package is imported as "src.subspacead" from the repository root.
    for name in list(sys.modules):
        if not (name == "src" or name.startswith("src.")):
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
        importlib.import_module("src.subspacead.core.extractor"),
        importlib.import_module("src.subspacead.core.pca"),
        importlib.import_module("src.subspacead.post_process.scoring"),
        importlib.import_module("src.subspacead.data.transforms"),
    )


@register_adapter("subspace-ad")
@register_adapter("subspacead")
class SubspaceADAdapter(ModelAdapter):
    """Official SubspaceAD few-shot inference for MVTec AD and VisA.

    Follows scripts/benchmark_few_shot.sh: a frozen ``dinov2-with-registers-giant``
    at 672 pixels, hidden layers -12 to -18 averaged, PCA keeping 0.99 explained
    variance, reconstruction-residual scoring, and the ``mtop1p`` image score
    (the mean of the top one percent of map pixels). It is **training-free** —
    there is no checkpoint beyond the HuggingFace backbone — and the subspace is
    fitted per category on k normal training images plus 30 rotation
    augmentations each, exactly as the script specifies. Augmentation is disabled
    for ``transistor``, as main.py does.

    ``post_process_map`` resizes the 48x48 residual grid to 672 and blurs it with
    a 3x3 Gaussian at sigma 4, so the blur runs inside the adapter and the
    evaluator adds none.

    Two protocol notes. The cohort arrives at 518 pixels because that is the grid
    the perturbations are defined on, and the official run feeds the original
    image straight to a 672-pixel resize; the adapter therefore resizes 518 to
    672 and — importantly — puts the **reference images through the identical
    path**, so the fitted subspace and the scored features come from the same
    distribution. And the official k-shot draw is ``random.shuffle`` under
    ``--seed 42`` over its own file ordering, which cannot be reproduced from
    outside; the adapter shuffles the sorted normal training images under that
    seed and records the file names it used.
    """

    name = "subspacead"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_ckpt: str = "facebook/dinov2-with-registers-giant",
        image_res: int = 672,
        k_shot: int = 1,
        layers: Sequence[int] = LAYERS,
        agg_method: str = "mean",
        aug_count: int = 30,
        pca_ev: float = 0.99,
        score_method: str = "reconstruction",
        drop_k: int = 0,
        img_score_agg: str = "mtop1p",
        shot_seed: int = 42,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("SubspaceAD target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"Official SubspaceAD few-shot uses k_shot in {SHOT_VALUES}")
        if int(image_res) != 672:
            raise ValueError("Official SubspaceAD few-shot evaluation uses image_res=672")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.image_res = int(image_res)
        self.k_shot = int(k_shot)
        self.layers = [int(level) for level in layers]
        self.agg_method = agg_method
        self.aug_count = int(aug_count)
        self.pca_ev = float(pca_ev)
        self.score_method = score_method
        self.drop_k = int(drop_k)
        self.img_score_agg = img_score_agg
        self.shot_seed = int(shot_seed)

        extractor_module, pca_module, scoring_module, transforms_module = (
            _import_official_repository(repository)
        )
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        self._pca_module = pca_module
        self._scores = scoring_module.calculate_anomaly_scores
        self._post_process = scoring_module.post_process_map
        self._aggregate = scoring_module.aggregate_image_score
        self._augmentation = transforms_module.get_augmentation_transform(
            ["rotate"], self.image_res
        )

        torch.manual_seed(self.shot_seed)
        np.random.seed(self.shot_seed)
        self._extractor = extractor_module.FeatureExtractor(model_ckpt)
        self._reference = NormalReference(
            dataset=target_key,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._subspace: dict[str, dict] = {}
        self._selection: dict[str, list[str]] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "SubspaceAD/scripts/benchmark_few_shot.sh",
            "mode": "few_shot",
            "target_dataset": target_key,
            "model_ckpt": model_ckpt,
            "image_res": self.image_res,
            "cohort_image_size": self.image_size,
            "layers": self.layers,
            "agg_method": self.agg_method,
            "k_shot": self.k_shot,
            "aug_count": self.aug_count,
            "aug_list": ["rotate"],
            "no_aug_categories": sorted(NO_AUG_CATEGORIES),
            "pca_ev": self.pca_ev,
            "score_method": self.score_method,
            "drop_k": self.drop_k,
            "img_score_agg": self.img_score_agg,
            "shot_seed": self.shot_seed,
            "training_free": True,
            # post_process_map blurs with a 3x3 kernel at sigma 4 internally.
            "official_gaussian_sigma": 4.0,
            "official_gaussian_kernel_size": 3,
            "gaussian_applied_inside_adapter": True,
            "reference_path": "identical to the cohort path (518 then 672)",
        }

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data

    def _as_processor_input(self, batch: torch.Tensor) -> list[np.ndarray]:
        """Hand the processor float arrays in [0, 255] so it rescales, not quantizes."""

        arrays = (batch.clamp(0, 1) * 255.0).permute(0, 2, 3, 1).cpu().numpy()
        return [np.ascontiguousarray(single, dtype=np.float32) for single in arrays]

    def _tokens(self, batch: torch.Tensor):
        resized = self._resize(
            batch, [self.image_res, self.image_res],
            interpolation=self._bicubic, antialias=True,
        ).clamp(0, 1)
        tokens, grid, _ = self._extractor.extract_tokens(
            self._as_processor_input(resized),
            self.image_res,
            self.layers,
            self.agg_method,
        )
        return tokens, grid

    def _ensure_subspace(self, category: str) -> dict:
        cached = self._subspace.get(category)
        if cached is not None:
            return cached
        candidates = list(self._reference.candidates(category))
        # main.py shuffles under the run seed and keeps the first k.
        rng = random.Random(self.shot_seed)
        rng.shuffle(candidates)
        samples = candidates[: self.k_shot]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.load(samples).to(self.device, dtype=torch.float32)

        augment = None if category in NO_AUG_CATEGORIES else self._augmentation
        # Seeded once, then the RNG advances across the rotations exactly as the
        # official loop lets it advance under its run seed.
        torch.manual_seed(self.shot_seed)
        views: list[torch.Tensor] = []
        for index in range(len(references)):
            single = references[index]
            views.append(single)
            if augment is not None:
                views.extend(augment(single) for _ in range(self.aug_count))
        stacked = torch.stack(views)

        collected: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(len(stacked)):
                tokens, _ = self._tokens(stacked[index : index + 1])
                collected.append(tokens.reshape(-1, tokens.shape[-1]))
        feature_dim = collected[0].shape[-1]
        total_tokens = sum(item.shape[0] for item in collected)

        def feature_generator():
            for item in collected:
                yield item

        model = self._pca_module.PCAModel(k=None, ev=self.pca_ev, whiten=False)
        parameters = model.fit(
            feature_generator, feature_dim, total_tokens, len(collected)
        )
        self._subspace[category] = parameters
        return parameters

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("SubspaceAD received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.image_res, self.image_res), dtype=np.float32
        )
        with torch.no_grad():
            # The subspace is per category, so images are grouped by it.
            for index in sorted(range(len(categories)), key=lambda i: str(categories[i])):
                category = str(categories[index])
                parameters = self._ensure_subspace(category)
                tokens, (height, width) = self._tokens(batch[index : index + 1])
                residual = self._scores(
                    tokens.reshape(-1, tokens.shape[-1]),
                    parameters,
                    self.score_method,
                    self.drop_k,
                ).reshape(height, width)
                anomaly_map = self._post_process(residual, self.image_res)
                maps[index] = np.asarray(anomaly_map, dtype=np.float32)
                scores[index] = float(
                    self._aggregate(maps[index], self.img_score_agg)
                )
        return scores, maps

    def close(self) -> None:
        self._extractor = None
        self._subspace.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
