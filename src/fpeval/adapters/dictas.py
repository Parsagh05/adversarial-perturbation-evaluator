"""DictAS adapter matching the official few-shot evaluation defaults."""

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
from .reference import NormalReference


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The released DictAS weights are individual Google Drive files. The name is the
# auxiliary training set, so each one evaluates the dataset it did not see.
CHECKPOINTS = {
    "train_visa": (
        "train_visa.pth",
        "1cJGCNJGtqUs1MdGyuZ2diS4yMaOyapDC",
        "82d420868fbc2c7e5ae1cb0935b1838f5d9027faa57af3a36a78cfcb2212c74a",
    ),
    "train_mvtec": (
        "train_mvtec.pth",
        "1iUds-WgeyfU78z3SWtwqonZ8fP8YKN0Y",
        "6ece22e1fc22b6e953f8e614eb79e728644991c3642c4ada7d585fd590491b18",
    ),
}
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_visa", "visa": "train_mvtec"}

# The README requires this exact OpenAI backbone for the released weights; it is
# the same file VCP-CLIP and Bayes-PFL pin.
CLIP_BACKBONE_SHA256 = (
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
)
CLIP_BACKBONE_NAME = "ViT-L-14-336px.pt"
CLIP_BACKBONE_URL = (
    f"https://openaipublic.azureedge.net/clip/models/{CLIP_BACKBONE_SHA256}/"
    f"{CLIP_BACKBONE_NAME}"
)

SHOT_VALUES = (1, 2, 4)
# calcuate_metric_pixel overrides args.alpha for exactly these two datasets.
ALPHA = 0.2
# BESTSEGMENTATION sets this when TEST_For_BESTSEGMENTATION is True, its default.
SIGMA = 6

# The official k-shot selection, committed to the repository under fix_few_path
# as paths into the authors' own converted "mvisa" tree (mvtec_000091.bmp and
# so on). Those names do not exist in the original MVTec/VisA downloads, but the
# repository also ships meta_mvtec.json / meta_visa.json, which list every
# converted name in per-class order - so each pinned file resolves to a position
# in the sorted normal training split, which is what these tables record. Every
# index was recovered that way and lies inside its class's train count, and the
# counts match the official splits exactly (bottle 209, toothbrush 60,
# candle 900, ...). tests/test_dictas.py re-checks the bounds.
SELECTION: dict[str, dict[int, dict[str, list[int]]]] = {
    "mvtec": {
        1: {
            "bottle": [91], "cable": [4], "capsule": [175], "carpet": [244],
            "grid": [2], "hazelnut": [10], "leather": [9], "metal_nut": [98],
            "pill": [174], "screw": [274], "tile": [164], "toothbrush": [54],
            "transistor": [0], "wood": [191], "zipper": [125],
        },
        2: {
            "bottle": [91, 178], "cable": [4, 82], "capsule": [2, 206],
            "carpet": [244, 134], "grid": [2, 241], "hazelnut": [21, 149],
            "leather": [9, 131], "metal_nut": [7, 208], "pill": [174, 89],
            "screw": [133, 290], "tile": [164, 48], "toothbrush": [54, 1],
            "transistor": [0, 186], "wood": [191, 75], "zipper": [125, 98],
        },
        4: {
            "bottle": [91, 178, 97, 207], "cable": [76, 152, 199, 88],
            "capsule": [177, 207, 183, 169], "carpet": [244, 134, 229, 89],
            "grid": [2, 241, 179, 14], "hazelnut": [10, 179, 362, 310],
            "leather": [109, 76, 87, 166], "metal_nut": [107, 0, 100, 31],
            "pill": [174, 89, 168, 83], "screw": [168, 270, 55, 318],
            "tile": [59, 199, 88, 208], "toothbrush": [27, 15, 54, 37],
            "transistor": [97, 176, 115, 48], "wood": [191, 75, 0, 50],
            "zipper": [125, 98, 46, 141],
        },
    },
    "visa": {
        1: {
            "candle": [225], "capsules": [321], "cashew": [168],
            "chewinggum": [64], "fryum": [309], "macaroni1": [501],
            "macaroni2": [105], "pcb1": [489], "pcb2": [900], "pcb3": [296],
            "pcb4": [429], "pipe_fryum": [397],
        },
        2: {
            "candle": [225, 744], "capsules": [321, 202], "cashew": [168, 130],
            "chewinggum": [64, 322], "fryum": [309, 200],
            "macaroni1": [501, 607], "macaroni2": [105, 817],
            "pcb1": [489, 613], "pcb2": [900, 186], "pcb3": [296, 126],
            "pcb4": [429, 774], "pipe_fryum": [397, 343],
        },
        4: {
            "candle": [225, 744, 0, 840], "capsules": [321, 202, 160, 402],
            "cashew": [168, 130, 301, 427], "chewinggum": [64, 322, 27, 77],
            "fryum": [309, 200, 318, 354], "macaroni1": [501, 607, 141, 372],
            "macaroni2": [105, 817, 392, 332], "pcb1": [489, 613, 788, 706],
            "pcb2": [900, 186, 550, 598], "pcb3": [296, 126, 317, 604],
            "pcb4": [429, 774, 635, 10], "pipe_fryum": [397, 343, 234, 285],
        },
    },
}
# dataset.py rotates the support set for this one category at k <= 4.
ROTATED_CATEGORY = "screw"
ROTATION_ANGLES = tuple(range(-180, 181, 45))


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "models" / "DictAS.py",
        root / "models" / "model_CLIP.py",
        root / "models" / "utils.py",
        root / "test.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official DictAS repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "dataset", "open_clip_local", "utils"}
            or name.startswith(("models.", "open_clip_local.", "utils."))
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
        importlib.import_module("models.DictAS"),
        importlib.import_module("models.prompt_ensemble"),
        importlib.import_module("models.utils"),
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
        else Path.home() / ".cache" / "dictas"
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
    """Return a local DictAS checkpoint, downloading a released name if needed."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"DictAS checkpoint not found: {path}. Pass a path or one of "
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
            "Downloading DictAS weights needs gdown; install the dictas extra or "
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
            f"DictAS checkpoint checksum mismatch for {filename}: expected "
            f"{expected}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def _minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    # calcuate_metric_pixel divides by (max - min + 1e-6).
    return (values - low) / (high - low + 1e-6)


@register_adapter("dict-as")
@register_adapter("dictas")
class DictASAdapter(ModelAdapter):
    """Official DictAS few-shot inference for MVTec AD and VisA.

    Follows test.sh and the argparse defaults: the repository's own CLIP
    implementation on the pinned OpenAI ``ViT-L-14-336px.pt`` at **336 pixels**,
    the released ``train_visa.pth`` / ``train_mvtec.pth`` dictionary, and
    ``TEST_For_BESTSEGMENTATION`` left at its default ``True``. That default
    decides three things through ``BESTSEGMENTATION``: the feature levels are
    **6/12 on MVTec** but 6/12/18/24 on VisA, the neighbourhood ``scale_list``
    stays (1, 3), and the map blur is sigma **6**.

    The blur runs before both the pixel metrics and the map-maximum that feeds
    the image score, so the adapter applies it and the evaluator adds none. The
    image score is ``0.2 * text probability + 0.8 * map maximum``, each min-max
    normalized per category over the cohort - ``calcuate_metric_pixel`` forces
    that 0.2 for MVTec and VisA whatever ``--alpha`` says. Both normalizations
    are fitted on the clean cohort and frozen for the adversarial pass.

    The k-shot selection is the paper's own. The repository pins it in
    ``fix_few_path`` as paths into the authors' converted ``mvisa`` tree, whose
    file names do not exist in the original downloads; the shipped
    ``meta_*.json`` resolve each one to a position in the sorted normal training
    split, and those positions are what ``SELECTION`` records. DictAS is
    therefore one of the few few-shot adapters here whose reference set is the
    published one rather than a re-drawn sample.

    Two reproduction notes. ``screw`` at k <= 4 has its support set rotated
    through nine angles, which the adapter reproduces (the rotation runs on the
    518-pixel cohort tensor rather than the original file, since that is the grid
    the perturbation is defined on). And the official ``test.py`` computes a
    blurred map into a local variable it never stores - the stored map is the
    unblurred one and ``calcuate_metric_pixel`` blurs it again later, so the
    single blur here is the one that reaches the metrics.
    """

    name = "dictas"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        download_root: str | None = None,
        clip_backbone_path: str | None = None,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = 336,
        backbone: str = "ViT-L-14-336",
        features: Sequence[int] = (6, 12, 18, 24),
        k_shot: int = 4,
        seed: int = 222,
        best_segmentation: bool = True,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("DictAS target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"DictAS ships a pinned selection for k_shot in {SHOT_VALUES}")
        if not best_segmentation:
            raise ValueError(
                "Only TEST_For_BESTSEGMENTATION=True is wired up; it is the "
                "argparse default and the setting test.sh runs."
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.k_shot = int(k_shot)
        self.target_dataset = target_key
        self.seed = int(seed)

        clip_module, dictas_module, prompt_module, utils_module = (
            _import_official_repository(repository)
        )
        import albumentations
        from scipy.ndimage import gaussian_filter
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._albumentations = albumentations
        self._gaussian_filter = gaussian_filter
        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        self._norm_patch = utils_module.norm_patch
        self._best_segmentation = utils_module.BESTSEGMENTATION
        utils_module.setup_seed(self.seed)

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
            raise FileNotFoundError(f"DictAS model config not found: {config_path}")
        configs = json.loads(config_path.read_text(encoding="utf-8"))

        # BESTSEGMENTATION and MyDictionary both read these off the namespace,
        # and BESTSEGMENTATION mutates scale_list and sigm in place.
        self._args = SimpleNamespace(
            dataset=target_key,
            features_list=[int(level) for level in features],
            scale_list=[1, 3],
            k_shot=self.k_shot,
            image_size=self.model_image_size,
            alpha=ALPHA,
            sigm=SIGMA,
            TEST_For_BESTSEGMENTATION=True,
        )

        model_clip, _, _ = clip_module.Load_CLIP(
            self.model_image_size, str(clip_path), device=self.device
        )
        model_clip.to(self.device).eval().requires_grad_(False)
        self._clip = model_clip
        self._tokenize = clip_module.tokenize
        self._encode_text = prompt_module.encode_text_with_prompt_ensemble

        dictionary = dictas_module.MyDictionary(configs, self._args).to(self.device)
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        dictionary.load_state_dict(payload["Mymodel"])
        dictionary.eval().requires_grad_(False)
        self._dictionary = dictionary

        self._reference = NormalReference(
            dataset=target_key,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._memory: dict[str, list[torch.Tensor]] = {}
        self._selection: dict[str, list[str]] = {}
        self._text_cache: dict[str, torch.Tensor] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "DictAS/test.sh",
            "mode": "few_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": "openai",
            "clip_backbone_sha256": CLIP_BACKBONE_SHA256,
            "model_image_size": self.model_image_size,
            "cohort_image_size": self.image_size,
            "features": list(self._args.features_list),
            "scale_list": list(self._args.scale_list),
            "k_shot": self.k_shot,
            "score_fusion_alpha": ALPHA,
            "seed": self.seed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "test_for_best_segmentation": True,
            "official_gaussian_sigma": float(SIGMA),
            "gaussian_applied_inside_adapter": True,
            "reference_selection": "official fix_few_path, resolved through meta_*.json",
            "rotated_support_category": ROTATED_CATEGORY,
        }
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        data["feature_levels_by_category"] = {
            category: self._best_segmentation(self._args, [category])
            for category in sorted(self._selection)
        }
        return data

    # ---------------------------------------------------------------- inputs --
    def _prepare(self, batch: torch.Tensor) -> torch.Tensor:
        resized = self._resize(
            batch,
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        return (resized - self._mean) / self._std

    def _rotated(self, references: torch.Tensor) -> torch.Tensor:
        """dataset.Roate_support: the original plus nine albumentations rotations."""

        rotate = self._albumentations.Rotate
        views: list[torch.Tensor] = []
        for index in range(len(references)):
            single = references[index]
            views.append(single)
            array = (single * 255.0).round().clamp(0, 255).to(torch.uint8)
            array = array.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            for angle in ROTATION_ANGLES:
                transform = rotate(limit=(angle, angle), always_apply=True)
                turned = transform(image=np.ascontiguousarray(array))["image"]
                turned = np.ascontiguousarray(turned[:, :, ::-1])
                views.append(
                    torch.from_numpy(turned).to(references.device).permute(2, 0, 1)
                    .to(torch.float32) / 255.0
                )
        return torch.stack(views)

    # ----------------------------------------------------------- per category --
    def _text_features(self, category: str) -> torch.Tensor:
        cached = self._text_cache.get(category)
        if cached is None:
            with torch.no_grad():
                prompts = self._encode_text(
                    self._clip, [category], self._tokenize, self.device
                )
            cached = prompts[category].detach()
            self._text_cache[category] = cached
        return cached

    def _memory_bank(self, category: str, features_list: list[int]) -> list[torch.Tensor]:
        cached = self._memory.get(category)
        if cached is not None:
            return cached
        candidates = list(self._reference.candidates(category))
        try:
            positions = SELECTION[self.target_dataset][self.k_shot][category]
        except KeyError as error:
            raise KeyError(
                f"DictAS has no pinned {self.k_shot}-shot selection for "
                f"{category!r} in {self.target_dataset}"
            ) from error
        if max(positions) >= len(candidates):
            raise ValueError(
                f"DictAS pins index {max(positions)} for {category}, but only "
                f"{len(candidates)} normal training images were discovered; the "
                "selection is positional in the sorted train split."
            )
        samples = [candidates[index] for index in positions]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.load(samples).to(self.device, dtype=torch.float32)
        if category == ROTATED_CATEGORY and self.k_shot <= 4:
            references = self._rotated(references)

        prepared = self._prepare(references)
        collected: list[list[torch.Tensor]] = []
        with torch.no_grad():
            # test.py chunks the support set at 4 images when there are more.
            chunks = (
                torch.chunk(prepared, int(np.ceil(len(prepared) / 4.0)), dim=0)
                if len(prepared) > 4
                else [prepared]
            )
            for chunk in chunks:
                _, _, tokens = self._clip.encode_image(chunk, features_list)
                collected.append(tokens)
        memory = [
            self._norm_patch(torch.cat([part[level] for part in collected], dim=0), True)
            for level in range(len(collected[0]))
        ]
        self._memory[category] = memory
        return memory

    # -------------------------------------------------------------- inference --
    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("DictAS received mismatched images and categories")
        batch = self._prepare(images.to(self.device, dtype=torch.float32))
        size = self.model_image_size
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), size, size), dtype=np.float32)

        with torch.no_grad():
            # The memory is per category, so images are grouped by it.
            order = sorted(range(len(categories)), key=lambda i: str(categories[i]))
            for index in order:
                category = str(categories[index])
                # BESTSEGMENTATION picks the levels and mutates scale_list/sigm.
                features_list = self._best_segmentation(self._args, [category])
                memory = self._memory_bank(category, features_list)
                image_features, _, patch_tokens = self._clip.encode_image(
                    batch[index : index + 1], features_list
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                patch_tokens = [
                    self._norm_patch(token, True) for token in patch_tokens
                ]
                patch_tokens = [
                    self._dictionary.Value_Generator(token) for token in patch_tokens
                ]
                layer_maps, _ = self._dictionary(patch_tokens, memory, mode="test")

                text = torch.stack([self._text_features(category)], dim=0).float()
                probability = (
                    100.0 * image_features.unsqueeze(1) @ text
                ).softmax(dim=-1).squeeze()
                scores[index] = float(probability[1])

                resized = [
                    torch.nn.functional.interpolate(
                        layer.unsqueeze(1), size=size, mode="bilinear",
                        align_corners=True,
                    ).squeeze()
                    for layer in layer_maps
                ]
                anomaly_map = torch.mean(torch.stack(resized, dim=0), dim=0)
                maps[index] = anomaly_map.float().cpu().numpy()

        # calcuate_metric_pixel blurs before the pixel metrics and before the
        # map maximum that half the image score is taken from.
        blurred = self._gaussian_filter(maps, sigma=SIGMA, axes=(1, 2))
        return scores, blurred.astype(np.float32)

    # ------------------------------------------------------------ postprocess --
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
        """Fuse the text probability with the blurred map maximum, per category."""

        del map_mins, reference_map_mins, maps, reference_maps
        scores = np.asarray(scores, dtype=np.float64)
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        peaks = np.asarray(map_maxs, dtype=np.float64)
        reference_peaks = np.asarray(reference_map_maxs, dtype=np.float64)
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
                peaks[selected],
                float(reference_peaks[matched].min()),
                float(reference_peaks[matched].max()),
            )
            result[selected] = ALPHA * text + (1.0 - ALPHA) * pixel
        return result.astype(np.float32)

    def close(self) -> None:
        self._clip = None
        self._dictionary = None
        self._memory.clear()
        self._text_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
