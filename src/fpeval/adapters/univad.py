"""UniVAD adapter matching the official few-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


# test.sh runs every dataset at --image_size 448 --k_shot 1 --round 0.
MODEL_IMAGE_SIZE = 448
# UniVAD.__init__ fixes these; they are not command-line options.
CLIP_MODEL = "ViT-L-14-336"
CLIP_PRETRAINED = "openai"
OUT_LAYERS = (6, 12, 18, 24)
DINOV2_MODEL = "dinov2_vitg14"
# The two files the README's wget lines fetch into pretrained_ckpts/.
# 662 MB and 2.40 GB; both digests taken from the downloaded files.
CHECKPOINTS = {
    "groundingdino_swint_ogc.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    ),
    "sam_hq_vit_h.pth": (
        "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
    ),
}
GROUNDING_DINO_CONFIG = (
    "./models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "UniVAD.py",
        root / "modules.py",
        root / "models" / "component_segmentaion.py",
        root / "models" / "grounded_sam.py",
        root / "models" / "component_feature_extractor.py",
        root / "models" / "dinov2" / "hubconf.py",
        root / "models" / "GroundingDINO" / "groundingdino" / "config"
        / "GroundingDINO_SwinT_OGC.py",
        root / "configs" / "class_histogram" / "bottle.yaml",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official UniVAD repository is incomplete at {root}; missing {missing}. "
            "It needs --recurse-submodules: models/dinov2 is a submodule, and "
            "models/GroundingDINO must be installed with `pip install -e .` from "
            "inside it."
        )
    root_text = str(root)
    for entry in (root_text, str(root / "models" / "GroundingDINO")):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "modules", "utils", "datasets", "dataset", "UniVAD"}
            or name.startswith(("models.", "utils.", "datasets."))
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
        importlib.import_module("UniVAD"),
        importlib.import_module("models.component_segmentaion"),
        importlib.import_module("models.grounded_sam"),
    )


@contextmanager
def _working_directory(path: Path):
    """Run inside the repository.

    UniVAD addresses everything by relative path - the Grounding DINO config,
    ``./pretrained_ckpts``, ``torch.hub.load('./models/dinov2', source='local')``,
    and the ``./masks`` and ``./heat_masks`` folders it reads and writes per
    image - so the official code only resolves correctly from its own root.
    """

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pretrained(repository: str | Path, *, source: str | Path | None = None) -> dict:
    """Put both released checkpoints in the repository's pretrained_ckpts/."""

    import urllib.request

    destination = Path(repository).expanduser().resolve() / "pretrained_ckpts"
    destination.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for filename, (url, expected) in CHECKPOINTS.items():
        target = destination / filename
        if target.is_file() and _sha256(target) == expected:
            resolved[filename] = target
            continue
        if source is not None:
            supplied = Path(source).expanduser().resolve() / filename
            if not supplied.is_file():
                raise FileNotFoundError(f"UniVAD checkpoint not found: {supplied}")
            actual = _sha256(supplied)
            if actual != expected:
                raise ValueError(
                    f"UniVAD checksum mismatch for {filename}: expected {expected}, "
                    f"got {actual}"
                )
            target.write_bytes(supplied.read_bytes())
            resolved[filename] = target
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        urllib.request.urlretrieve(url, temporary)
        actual = _sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"UniVAD checksum mismatch for {filename}: expected {expected}, "
                f"got {actual}"
            )
        temporary.replace(target)
        resolved[filename] = target
    return resolved


class _ComponentSegmenter:
    """The grounding half of Contextual Component Clustering, run in memory.

    ``segment_components.py`` precomputes these masks offline from the file on
    disk and ``UniVAD.forward`` then looks them up by image path. That is fine
    for the official benchmark and wrong here: the mask would always come from
    the clean image, so the attack could never reach the component segmentation
    even though it is one of the three parts of the method. This runs the same
    Grounding DINO and SAM-HQ pass on the image actually being scored. The body
    is ``grounding_segmentation``'s loop, calling the repository's own helpers.
    """

    def __init__(self, segmentation_module, grounded_sam_module, device) -> None:
        from groundingdino.datasets import transforms as T

        self._segmentation = segmentation_module
        self._grounded_sam = grounded_sam_module
        self._device = device
        # Exactly grounded_sam.load_image's transform, minus the file open.
        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._model = grounded_sam_module.load_model(
            GROUNDING_DINO_CONFIG,
            "./pretrained_ckpts/groundingdino_swint_ogc.pth",
            "cuda",
        )
        sam = segmentation_module.sam_hq_model_registry["vit_h"](
            "./pretrained_ckpts/sam_hq_vit_h.pth"
        ).to(device)
        self._predictor = segmentation_module.SamPredictor(sam)

    def segment(self, image_pil, config: dict) -> np.ndarray:
        module = self._segmentation
        image, _ = self._transform(image_pil, None)
        boxes_filt, pred_phrases = self._grounded_sam.get_grounding_output(
            self._model,
            image,
            config["text_prompt"],
            config["box_threshold"],
            config["text_threshold"],
            device="cuda",
        )
        background_box = []
        for index, text in enumerate(pred_phrases):
            for token in str(config.get("background_prompt") or "").split("."):
                if token in text.replace(" - ", "-") and token not in (" ", ""):
                    background_box.append(index)

        # cv2.imread + BGR2RGB on an RGB PIL image is just its array.
        image_rgb = np.array(image_pil)
        self._predictor.set_image(image_rgb)
        width, height = image_pil.size
        H, W = height, width
        for index in range(boxes_filt.size(0)):
            boxes_filt[index] = boxes_filt[index] * torch.Tensor([W, H, W, H])
            boxes_filt[index][:2] -= boxes_filt[index][2:] / 2
            boxes_filt[index][2:] += boxes_filt[index][:2]
        boxes_filt = boxes_filt.cpu()
        if len(boxes_filt) == 0:
            return np.ones((W, H))

        transformed = self._predictor.transform.apply_boxes_torch(
            boxes_filt, image_rgb.shape[:2]
        ).to(self._device)
        masks, _, _ = self._predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed.to(self._device),
            multimask_output=False,
        )
        masks = torch.stack(
            [mask for index, mask in enumerate(masks) if index not in background_box]
        )
        masks = module.turn_binary_to_int(masks[:, 0, :, :].cpu().numpy())
        if config.get("filter_by_combine", False):
            masks = module.filter_by_combine(masks)
        return module.merge_masks(masks)


@register_adapter("uni-vad")
@register_adapter("univad")
class UniVADAdapter(ModelAdapter):
    """Official UniVAD few-shot inference for MVTec AD and VisA.

    Follows test.sh at its defaults - ``--image_size 448 --k_shot 1 --round 0`` -
    and UniVAD.py's own fixed settings: CLIP ``ViT-L-14-336`` with OpenAI weights
    at layers 6/12/18/24, DINOv2-giant, DINO ViT-S/8 for the clustering heatmaps,
    Grounding DINO SwinT-OGC and SAM-HQ ViT-H for component segmentation, and the
    per-category ``configs/class_histogram/<category>.yaml`` box and text
    thresholds. It is **training-free**: every weight is a public backbone, and
    ``round`` picks the k normal images by their position in the sorted training
    split exactly as `test_univad.py`'s ``range(round, round + k_shot)`` does.

    The gate UniVAD selects from the reference masks - TEXTURE, SINGLE or MULTI -
    decides which branches run, and the returned map and score come straight from
    ``forward``: the map is ``anomaly_map_ret_all`` at 448 and the score is its
    maximum plus the global CLIP residual. No blur is applied - test_univad.py
    builds a ``GaussianBlur(3, 4.0)`` and never uses it - so the evaluator's own
    smoothing is the only one.

    **Three things to know before reading its numbers.**

    First, cost. ``segment_components.py`` precomputes the component masks offline
    and ``forward`` looks them up by file path. Reusing those here would mean the
    mask always came from the clean image, so the attack could never reach
    Contextual Component Clustering - one of the three parts of the method - and
    UniVAD would look robust for a reason that is an artifact of the harness.
    This adapter therefore runs Grounding DINO and SAM-HQ on the image actually
    being scored. That is the right thing to measure, and it is expensive: a
    SAM-HQ ViT-H encode plus a Grounding DINO pass per image, on top of the three
    to five backbones ``forward`` already runs. It has not been timed on this
    project's hardware, but it is the slowest adapter here by a wide margin -
    time a small ``max_conditions`` run before committing to a sweep.

    Second, resolution. The official masks are segmented from the original
    full-resolution file; the perturbation is only defined on the evaluation
    grid, so these are segmented at 448. ``runtime_metadata`` records that as
    ``segmentation_resolution``.

    Third, determinism. The MULTI gate's setup fits ``KMeans(init="k-means++")``
    with no ``random_state`` inside a ``while`` loop, so the reference side would
    otherwise differ run to run. The adapter pins the seed of each successive
    attempt from ``kmeans_seed``, which keeps a rerun reproducible while still
    letting the loop advance - the same fix AdaCLIP's HSF clustering needed.

    The official code addresses everything relatively and round-trips masks
    through ``./masks`` and ``./heat_masks`` for every image, so the adapter runs
    with the repository as its working directory and materializes the reference
    images under ``<repository>/data`` where ``setup`` expects to reopen them.
    That makes the repository checkout **writable state**, not a read-only
    dependency.
    """

    name = "univad"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = MODEL_IMAGE_SIZE,
        k_shot: int = 1,
        round_index: int = 0,
        kmeans_seed: int = 0,
        checkpoint_source: str | None = None,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("UniVAD target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) < 1:
            raise ValueError("UniVAD few-shot needs at least one reference image")
        if int(round_index) < 0:
            raise ValueError("UniVAD round_index is an offset into the training split")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.k_shot = int(k_shot)
        self.round_index = int(round_index)
        self.kmeans_seed = int(kmeans_seed)
        self.target_dataset = target_key
        self._root = Path(repository).expanduser().resolve()

        univad_module, segmentation_module, grounded_sam_module = (
            _import_official_repository(repository)
        )
        import cv2
        import yaml
        from PIL import Image
        from sklearn.cluster import KMeans
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._cv2 = cv2
        self._yaml = yaml
        self._image = Image
        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        self._segmentation = segmentation_module

        # UniVAD.py does `from sklearn.cluster import KMeans` and constructs it
        # without a random_state, so the MULTI reference side is not reproducible
        # as written. Seeding each attempt from a counter fixes that without
        # freezing the retry loop, which needs the clustering to keep changing.
        self._kmeans_attempts = 0

        def seeded_kmeans(**kwargs):
            kwargs.setdefault("random_state", self.kmeans_seed + self._kmeans_attempts)
            self._kmeans_attempts += 1
            return KMeans(**kwargs)

        univad_module.KMeans = seeded_kmeans

        fetch_pretrained(self._root, source=checkpoint_source)

        with _working_directory(self._root):
            torch.manual_seed(self.kmeans_seed)
            np.random.seed(self.kmeans_seed)
            self._model = univad_module.UniVAD(image_size=self.model_image_size).to(
                self.device
            )
            self._model.eval()
            self._segmenter = _ComponentSegmenter(
                segmentation_module, grounded_sam_module, self.device
            )

        self._reference = NormalReference(
            dataset=target_key,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._configs: dict[str, dict] = {}
        self._selection: dict[str, list[str]] = {}
        self._gates: dict[str, str] = {}
        self._prepared = None

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "UniVAD/test.sh + test_univad.py",
            "mode": "few_shot",
            "target_dataset": target_key,
            "clip_model": CLIP_MODEL,
            "clip_pretrained": CLIP_PRETRAINED,
            "out_layers": list(OUT_LAYERS),
            "dinov2_model": DINOV2_MODEL,
            "model_image_size": self.model_image_size,
            "k_shot": self.k_shot,
            "round": self.round_index,
            "kmeans_seed": self.kmeans_seed,
            "training_free": True,
            "component_segmentation": "live, on the image being scored",
            "segmentation_resolution": self.model_image_size,
            "official_segmentation_resolution": "original file resolution",
            "grounding_dino": "swint_ogc",
            "sam": "hq_vit_h",
            # test_univad.py builds a GaussianBlur(3, 4.0) and never applies it.
            "gaussian_applied_inside_adapter": False,
            "cohort_image_size": self.image_size,
            "writes_into_repository": ["data", "masks", "heat_masks"],
        }

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        data["gate"] = dict(sorted(self._gates.items()))
        return data

    # --------------------------------------------------------------- helpers --
    def _grounding_config(self, category: str) -> dict:
        cached = self._configs.get(category)
        if cached is not None:
            return cached
        path = self._root / "configs" / "class_histogram" / f"{category}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"UniVAD has no grounding config for {category!r}; expected {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            settings = self._yaml.load(handle, Loader=self._yaml.SafeLoader)
        config = settings["grounding_config"]
        self._configs[category] = config
        return config

    def _to_model_size(self, batch: torch.Tensor) -> torch.Tensor:
        return self._resize(
            batch,
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)

    def _as_pil(self, single: torch.Tensor):
        array = (single * 255.0).round().clamp(0, 255).to(torch.uint8)
        return self._image.fromarray(
            array.permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )

    def _write_mask(self, relative: str, mask: np.ndarray) -> None:
        """Put a mask exactly where UniVAD's path arithmetic will look for it."""

        folder = self._root / "masks" / relative
        folder.mkdir(parents=True, exist_ok=True)
        self._cv2.imwrite(str(folder / "grounding_mask.png"), mask)

    def _segment_and_store(self, image_pil, config: dict, relative: str) -> None:
        self._write_mask(relative, self._segmenter.segment(image_pil, config))

    # ---------------------------------------------------------- per category --
    def _ensure_setup(self, category: str) -> None:
        if self._prepared == category:
            return
        config = self._grounding_config(category)
        candidates = list(self._reference.candidates(category))
        samples = candidates[self.round_index : self.round_index + self.k_shot]
        if len(samples) != self.k_shot:
            raise ValueError(
                f"UniVAD needs {self.k_shot} normal images from position "
                f"{self.round_index} of {category}, but only {len(samples)} remain"
            )
        self._selection[category] = NormalReference.describe(samples)
        references = self._to_model_size(
            self._reference.load(samples).to(self.device, dtype=torch.float32)
        )

        # setup() reopens each reference by path for the MULTI gate, so the file
        # has to exist; "/data/" is also the token its mask lookup splits on.
        folder = self._root / "data" / self.target_dataset / category / "train" / "good"
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(len(references)):
            image_pil = self._as_pil(references[index])
            name = f"{index:03d}"
            image_pil.save(folder / f"{name}.png")
            relative = f"{self.target_dataset}/{category}/train/good/{name}"
            self._segment_and_store(image_pil, config, relative)
            paths.append(str(folder / f"{name}.png"))

        self._kmeans_attempts = 0
        self._model.setup(
            {
                "few_shot_samples": references,
                "dataset_category": category,
                "image_path": paths,
            }
        )
        self._gates[category] = self._model.gate.name
        self._prepared = category

    # -------------------------------------------------------------- inference --
    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("UniVAD received mismatched images and categories")
        batch = self._to_model_size(images.to(self.device, dtype=torch.float32))
        size = self.model_image_size
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), size, size), dtype=np.float32)

        with _working_directory(self._root), torch.no_grad():
            # setup() is per category and resets the whole reference state, so
            # the batch is walked in category order to build each one once.
            order = sorted(range(len(categories)), key=lambda i: str(categories[i]))
            for index in order:
                category = str(categories[index])
                self._ensure_setup(category)
                config = self._grounding_config(category)
                image_pil = self._as_pil(batch[index])
                # One reused query slot, mirroring the single ./heat_masks/.../test/0
                # slot the official forward overwrites for every image.
                relative = f"{self.target_dataset}/{category}/query/000"
                self._segment_and_store(image_pil, config, relative)
                query_path = str(
                    self._root / "data" / self.target_dataset / category
                    / "query" / "000.png"
                )
                result = self._model(
                    batch[index : index + 1], query_path, [np.array(image_pil)]
                )
                scores[index] = float(result["pred_score"].item())
                maps[index] = (
                    result["pred_mask"].reshape(size, size).float().cpu().numpy()
                )
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._segmenter = None
        self._configs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
