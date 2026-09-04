"""KAG-Prompt adapter matching the official few-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# test_mvtec.py loads the VisA-trained weights and vice versa.
CHECKPOINTS = {
    "train_on_mvtec": (
        "train_on_mvtec.pt",
        "1WVPpRKhO-1KBgbo_2JYH67F5fglKxU-I",
        "4bff3d6a668ee334db35486dc9b01eee1b4c37052767ffbc0f8b063eeeebf597",
    ),
    "train_on_visa": (
        "train_on_visa.pt",
        "1Reig-0RUnF1yyD7wYRoioJJ04kfRndgw",
        "bdf7746cf21b4b3ab34ba64543b4939a91d9e1f49fb31fa98c72fefccefa9403",
    ),
}
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_on_visa", "visa": "train_on_mvtec"}
IMAGEBIND = ("imagebind_huge.pth", "1jLpa_YCL_bOHtSZ1FpZygfQFHJOrWe71")
# Filled in once the 4.8 GB ImageBind archive has been fetched and hashed.
IMAGEBIND_SHA256: str | None = None

MODEL_IMAGE_SIZE = 224
FEATURES = (6, 12, 18, 24)
# test_*.py pass this as the map fusion weight.
FUSION_R = 0.1
# The image score blends the CLS logit with the mean of the map's top-k pixels.
SCORE_TOP_K = 30
SCORE_WEIGHT = 0.1

# The prompt each script hands the model. The model matches it against its own
# CLASS_NAMES, which spell "metal nut" and collapse pcb1-4 and macaroni1-2.
DESCRIBLES = {
    "mvtec": {
        "bottle": "bottle", "cable": "cable", "capsule": "capsule",
        "carpet": "carpet", "grid": "grid", "hazelnut": "hazelnut",
        "leather": "leather", "metal_nut": "metal nut", "pill": "pill",
        "screw": "screw", "tile": "tile", "toothbrush": "toothbrush",
        "transistor": "transistor", "wood": "wood", "zipper": "zipper",
    },
    "visa": {
        "candle": "candle", "capsules": "capsule", "cashew": "cashew",
        # The official table really does misspell this one.
        "chewinggum": "chewinggom", "fryum": "fryum", "macaroni1": "macaroni",
        "macaroni2": "macaroni", "pcb1": "pcb", "pcb2": "pcb", "pcb3": "pcb",
        "pcb4": "pcb", "pipe_fryum": "pipe fryum",
    },
}
# --round defaults, per shot count, from the comment in each test script.
MVTEC_ROUND = {1: 195, 2: 195, 4: 194}
VISA_ROUND = {1: 14, 2: 57, 4: 78}
SHOT_VALUES = (1, 2, 4)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    code = root / "code"
    required = (
        code / "model" / "openllama.py",
        code / "model" / "ImageBind",
        code / "test_mvtec.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official KAG-prompt repository is incomplete at {root}; missing "
            f"{missing}. Its entrypoints live under code/."
        )
    code_text = str(code)
    if code_text in sys.path:
        sys.path.remove(code_text)
    sys.path.insert(0, code_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"model", "utils", "datasets", "dataset", "metrics", "header"}
            or name.startswith(("model.", "utils.", "datasets."))
        ):
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", []) or [])
        origin = str(
            getattr(module, "__file__", "") or (locations[0] if locations else "")
        )
        if not origin.startswith(code_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("model.openllama"),
        importlib.import_module("model.ImageBind.data"),
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
        else Path.home() / ".cache" / "kagprompt"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fetch(filename: str, file_id: str, expected: str | None, root: Path) -> Path:
    destination = root / filename
    if destination.is_file() and (expected is None or _sha256(destination) == expected):
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading KAG-prompt weights needs gdown; install the kagprompt "
            "extra or pass existing paths."
        ) from error
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    gdown.download(id=file_id, output=str(temporary), quiet=True, resume=True)
    if not temporary.is_file():
        raise RuntimeError(
            f"Google Drive did not return {filename}. Drive rate-limits popular "
            "files, and the ImageBind archive is 4.8 GB; retry later or download "
            "it manually and pass its path."
        )
    actual = _sha256(temporary)
    if expected is not None and actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"KAG-prompt checksum mismatch for {filename}: expected {expected}, "
            f"got {actual}"
        )
    temporary.replace(destination)
    return destination


def resolve_checkpoint(
    checkpoint: str | Path, *, download_root: str | Path | None = None
) -> Path:
    """Return a local KAG-prompt checkpoint, downloading a released name."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"KAG-prompt checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, file_id, expected = CHECKPOINTS[key]
    return _fetch(filename, file_id, expected, _cache_root(download_root))


def resolve_imagebind(
    imagebind: str | None = None, *, download_root: str | Path | None = None
) -> Path:
    """Return the ImageBind-huge backbone the model is built on."""

    if imagebind:
        path = Path(imagebind).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ImageBind checkpoint not found: {path}")
        return path
    filename, file_id = IMAGEBIND
    return _fetch(filename, file_id, IMAGEBIND_SHA256, _cache_root(download_root))


def official_positions(target: str, k_shot: int, available: int) -> list[int]:
    """Positions in the sorted normal training split the official scripts use.

    MVTec names its files ``{index:03d}.png`` and the script asks for
    ``round + i``, falling back to the last k when that file does not exist -
    which happens for ``toothbrush``, whose train split has only 60 images. VisA
    walks ``split_csv/1cls.csv`` and slices ``[round * 4:]`` after collecting
    ``round * 4 + k`` train normals, i.e. the same contiguous window.
    """

    if int(k_shot) not in SHOT_VALUES:
        raise ValueError(f"KAG-prompt covers k_shot in {SHOT_VALUES}")
    if target == "mvtec":
        start = MVTEC_ROUND[int(k_shot)]
        if start + int(k_shot) <= available:
            return list(range(start, start + int(k_shot)))
        # The script's Path.is_file() check fails and it takes the tail instead.
        return list(range(available - int(k_shot), available))
    start = VISA_ROUND[int(k_shot)] * 4
    if start + int(k_shot) > available:
        raise ValueError(
            f"KAG-prompt asks for VisA positions {start}..{start + k_shot - 1} "
            f"but only {available} normal training images were discovered"
        )
    return list(range(start, start + int(k_shot)))


@register_adapter("kag-prompt")
@register_adapter("kagprompt")
class KAGPromptAdapter(ModelAdapter):
    """Official KAG-Prompt few-shot inference for MVTec AD and VisA.

    Follows test_mvtec.py and test_visa.py: an ImageBind-huge visual encoder with
    the released ``train_on_*.pt`` heads (a linear decoder, an adapter, the
    kernel-aware graph module and MMCI), **224-pixel** inputs, feature levels
    6/12/18/24, and the fusion weight ``r = 0.1``. Despite the
    ``OpenLLAMAPEFTModel`` class name inherited from AnomalyGPT, **no language
    model is loaded** - the constructor builds ImageBind and the small heads and
    nothing else. The weights are cross-dataset as usual: MVTec is scored with
    the VisA-trained file.

    The image score is ``0.1 * cls_logit + 0.9 * mean(top-30 map pixels)``, taken
    per image, so no cohort statistic enters a prediction and the postprocess
    hooks stay at their defaults.

    Two things needed care. The official model loads its images **from disk by
    path** inside ``extract_multimodal_feature``; feeding it file paths here
    would bypass the perturbation entirely, so the adapter serves the evaluated
    tensors through the repository's own ``load_and_transform_vision_data`` entry
    point, keyed by a sentinel path, leaving the official code unmodified. And
    the reference branch contains ``if 'mvtec' in 'normal_img_paths'`` - a
    comparison against the *literal string*, which is always false - so the
    rotation-augmented path is dead and every dataset takes the plain branch.
    That is reproduced rather than corrected, and recorded in the metadata.

    The k-shot selection is positional and reproducible: MVTec asks for files
    ``round + i`` and falls back to the last k when they do not exist (only
    ``toothbrush``, whose train split has 60 images), while VisA takes a
    contiguous window starting at ``round * 4``. The VisA window indexes the
    repository's ``split_csv/1cls.csv`` row order, which coincides with the
    sorted train split when that file lists each object's normals in filename
    order.
    """

    name = "kagprompt"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        imagebind: str | None = None,
        download_root: str | None = None,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = MODEL_IMAGE_SIZE,
        features: Sequence[int] = FEATURES,
        k_shot: int = 1,
        fusion_r: float = FUSION_R,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("KAG-prompt target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"Official KAG-prompt evaluation uses k_shot in {SHOT_VALUES}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.k_shot = int(k_shot)
        self.fusion_r = float(fusion_r)
        self.target_dataset = target_key
        self.seed = int(seed)

        openllama_module, data_module = _import_official_repository(repository)
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(selected, download_root=download_root)
        imagebind_path = resolve_imagebind(imagebind, download_root=download_root)

        # The evaluated tensors are handed to the official loader by key, so
        # extract_multimodal_feature runs unmodified and never reads the disk.
        self._tensors: dict[str, torch.Tensor] = {}

        def load_by_key(image_paths, device):
            if image_paths is None:
                return None
            return torch.stack([self._tensors[str(key)] for key in image_paths]).to(
                device
            )

        data_module.load_and_transform_vision_data = load_by_key
        openllama_module.data.load_and_transform_vision_data = load_by_key

        model = openllama_module.OpenLLAMAPEFTModel(
            model="openllama_peft",
            imagebind_ckpt_path=str(imagebind_path),
            anomalygpt_ckpt_path=str(checkpoint_path),
            stage=2,
            features_list=[int(level) for level in features],
        )
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        incompatible = model.load_state_dict(payload, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"KAG-prompt checkpoint {checkpoint_path.name} has "
                f"{len(incompatible.unexpected_keys)} tensors the model does not "
                f"accept, e.g. {incompatible.unexpected_keys[:3]}"
            )
        model = model.to(self.device).eval()
        self._model = model

        self._reference = NormalReference(
            dataset=target_key,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._describles = DESCRIBLES[target_key]
        self._reference_keys: dict[str, list[str]] = {}
        self._selection: dict[str, list[str]] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": (
                f"KAG-prompt/code/test_{target_key}.py"
            ),
            "mode": "few_shot",
            "target_dataset": target_key,
            "visual_encoder": "imagebind_huge",
            "language_model": None,
            "model_image_size": self.model_image_size,
            "cohort_image_size": self.image_size,
            "features": [int(level) for level in features],
            "k_shot": self.k_shot,
            "round": (
                MVTEC_ROUND[self.k_shot] if target_key == "mvtec"
                else VISA_ROUND[self.k_shot]
            ),
            "fusion_r": self.fusion_r,
            "score_top_k": SCORE_TOP_K,
            "score_weight": SCORE_WEIGHT,
            "seed": self.seed,
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            "imagebind_checkpoint": imagebind_path.name,
            "images_supplied_in_memory": True,
            # if 'mvtec' in 'normal_img_paths' compares against the literal
            # string, so the rotation-augmented reference branch never runs.
            "rotation_augmented_references": False,
            "official_gaussian_sigma": 0.0,
        }
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data

    def _prepare(self, batch: torch.Tensor) -> torch.Tensor:
        """Resize(224) then CenterCrop(224), which is a no-op on a square input."""

        resized = self._resize(
            batch,
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        return (resized - self._mean) / self._std

    def _reference_for(self, category: str) -> list[str]:
        cached = self._reference_keys.get(category)
        if cached is not None:
            return cached
        candidates = list(self._reference.candidates(category))
        positions = official_positions(
            self.target_dataset, self.k_shot, len(candidates)
        )
        samples = [candidates[index] for index in positions]
        self._selection[category] = NormalReference.describe(samples)
        references = self._prepare(
            self._reference.load(samples).to(self.device, dtype=torch.float32)
        )
        keys = []
        for offset in range(len(references)):
            key = f"<reference>/{category}/{offset}"
            self._tensors[key] = references[offset]
            keys.append(key)
        self._reference_keys[category] = keys
        return keys

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("KAG-prompt received mismatched images and categories")
        batch = self._prepare(images.to(self.device, dtype=torch.float32))
        size = self.model_image_size
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), size, size), dtype=np.float32)

        with torch.no_grad():
            for index in range(len(batch)):
                category = str(categories[index])
                if category not in self._describles:
                    raise KeyError(
                        f"KAG-prompt has no prompt for {category!r} in "
                        f"{self.target_dataset}; known: {sorted(self._describles)}"
                    )
                reference_keys = self._reference_for(category)
                key = "<query>/current"
                self._tensors[key] = batch[index]
                anomaly_map, image_score = self._model.generate(
                    {
                        "prompt": self._describles[category],
                        "image_paths": [key],
                        "normal_img_paths": reference_keys,
                        "r": self.fusion_r,
                    }
                )
                single = anomaly_map.reshape(size, size).float().cpu().numpy()
                maps[index] = single
                flat = single.reshape(-1)
                top = np.partition(flat, -SCORE_TOP_K)[-SCORE_TOP_K:].mean()
                scores[index] = float(
                    SCORE_WEIGHT * float(image_score) + (1 - SCORE_WEIGHT) * top
                )
                self._tensors.pop(key, None)
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._tensors.clear()
        self._reference_keys.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
