"""FADE adapter matching the official few-shot evaluation defaults."""

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


# scripts/run_fade.py hardcodes these two prompt sets; only the directory comes
# from a variable, and that variable is itself hardcoded to "prompts".
CLASSIFICATION_PROMPTS = ("winclip_prompt.json",)
SEGMENTATION_PROMPTS = (
    "winclip_prompt.json",
    "chatgpt3.5_prompt1.json",
    "chatgpt3.5_prompt2.json",
    "chatgpt3.5_prompt3.json",
    "chatgpt3.5_prompt4.json",
    "chatgpt3.5_prompt5.json",
)
# The classification prompts take the category name; the segmentation prompts
# are deliberately class-agnostic and always resolve to "object".
SEGMENTATION_CLASSNAME = "object"
SEGMENTATION_IMG_SIZES = (240, 448, 896)
# datasets/base.py names these IMAGENET_MEAN/STD but assigns OpenCLIP's values.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "scripts" / "run_fade.py",
        root / "utils" / "text_model.py",
        root / "utils" / "image_model.py",
        root / "utils" / "anomaly_detection.py",
        root / "utils" / "embeddings.py",
        root / "prompts" / "winclip_prompt.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official FADE repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"utils", "datasets", "evaluation", "models", "dataset"}
            or name.startswith(("utils.", "datasets.", "evaluation.", "models."))
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
        importlib.import_module("utils.text_model"),
        importlib.import_module("utils.image_model"),
        importlib.import_module("utils.anomaly_detection"),
        importlib.import_module("utils.embeddings"),
    )


@register_adapter("fade")
class FADEAdapter(ModelAdapter):
    """Official FADE few-shot inference for MVTec AD and VisA.

    Follows scripts/run_fade.py at its defaults, which are the same settings its
    ``--experiment-name`` default spells out (``cm_both_sm_both/img_size_448/
    1shot``): a GEM-wrapped ``ViT-B/16-plus-240`` with ``laion400m_e31`` weights,
    language **and** vision guidance for both classification and segmentation,
    CLIP features for language classification at 240 pixels, GEM features for
    everything else, and segmentation run at 240, 448 and 896 pixels then
    averaged on the 56x56 grid of the largest.

    The two branches combine exactly as the script does. The image score is the
    mean of the language score and the vision score, where the vision score is
    the maximum of the vision segmentation map **before** it is fused - so the
    score is not recoverable from the returned map and is computed alongside it.
    The map is ``0.15 * language + 0.85 * vision``, the vision half having first
    been multiplied by 3.5 to bring its upper bound near one.

    It is **training-free**: the only weights are the public open_clip
    checkpoint, and the k-shot reference set is a plain patch memory bank scored
    by ``0.5 * (1 - cosine)`` to its nearest neighbour.

    Four protocol notes. The script clips the fused map to [0, 1], quantizes it
    to ``uint8`` and only then resizes it to 448 for evaluation, so the published
    numbers come from a 256-level map; the adapter keeps that quantization and
    reports it as ``official_uint8_quantization``. ``--normalize-segmentations``
    is off by default, so no cohort statistic enters a prediction and the
    postprocess hooks stay at their defaults. The official prompts are wrapped
    once more by GEM's ``encode_text``, which prefixes every string with "a photo
    of a" - odd, but it is what produced the paper's numbers, so the adapter
    calls that same method rather than embedding the prompts itself. And the
    official k-shot draw is a ``shuffle=True`` DataLoader seeded mid-iteration by
    ``--seed``, which cannot be reproduced from outside; the adapter shuffles the
    sorted normal training images under that seed and records the file names it
    used.

    The cohort arrives at 518 pixels because that is the grid the perturbations
    are defined on, while the official run resizes the original image to each
    scale; the adapter therefore resizes 518 to 240, 448 and 896 and puts the
    **reference images through the identical path**, so the memory bank and the
    patches scored against it come from the same distribution.
    """

    name = "fade"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_name: str = "ViT-B/16-plus-240",
        pretrained: str = "laion400m_e31",
        classification_mode: str = "both",
        segmentation_mode: str = "both",
        language_classification_feature: str = "clip",
        language_segmentation_feature: str = "gem",
        vision_feature: str = "gem",
        vision_segmentation_multiplier: float = 3.5,
        vision_segmentation_weight: float = 0.85,
        classification_img_size: int = 240,
        segmentation_img_sizes: Sequence[int] = SEGMENTATION_IMG_SIZES,
        eval_img_size: int = 448,
        text_model_type: str = "average",
        k_shot: int = 1,
        shot_seed: int = 0,
        normalize_segmentations: bool = False,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("FADE target_dataset must be 'mvtec' or 'visa'")
        if classification_mode != "both" or segmentation_mode != "both":
            raise ValueError(
                "This adapter reproduces the paper's few-shot setting, which is "
                "cm_both_sm_both; the language-only and vision-only ablations are "
                "not wired up."
            )
        if int(k_shot) < 1:
            raise ValueError("FADE few-shot needs at least one reference image")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.classification_img_size = int(classification_img_size)
        self.segmentation_img_sizes = [int(size) for size in segmentation_img_sizes]
        self.eval_img_size = int(eval_img_size)
        self.vision_segmentation_multiplier = float(vision_segmentation_multiplier)
        self.vision_segmentation_weight = float(vision_segmentation_weight)
        self.language_classification_feature = language_classification_feature
        self.language_segmentation_feature = language_segmentation_feature
        self.vision_feature = vision_feature
        self.text_model_type = text_model_type
        self.k_shot = int(k_shot)
        self.shot_seed = int(shot_seed)
        self.normalize_segmentations = bool(normalize_segmentations)

        text_module, image_module, detection_module, embedding_module = (
            _import_official_repository(repository)
        )
        import cv2
        import gem
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._cv2 = cv2
        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        self._build_text_model = text_module.build_text_model
        self._build_image_models = image_module.build_image_models
        self._extract_image_embeddings = embedding_module.extract_image_embeddings
        self._retrieve = embedding_module.retrieve_image_embeddings
        self._predict_classification = detection_module.predict_classification
        self._predict_segmentation = detection_module.predict_segmentation

        prompts = Path(repository).expanduser().resolve() / "prompts"
        self._classification_prompts = [
            str(prompts / name) for name in CLASSIFICATION_PROMPTS
        ]
        self._segmentation_prompts = [
            str(prompts / name) for name in SEGMENTATION_PROMPTS
        ]

        self._gem = gem.create_gem_model(
            model_name=model_name, pretrained=pretrained, device=str(self.device)
        )
        self._patch_size = tuple(self._gem.model.visual.patch_size)
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

        self._reference = NormalReference(
            dataset=target_key,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._classification_text: dict[str, object] = {}
        self._segmentation_text: dict[str, object] = {}
        self._image_models: dict[str, dict] = {}
        self._selection: dict[str, list[str]] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "BMVC-FADE/scripts/run_fade.py",
            "mode": "few_shot",
            "target_dataset": target_key,
            "model_name": model_name,
            "pretrained": pretrained,
            "gem_depth": 7,
            "classification_mode": classification_mode,
            "segmentation_mode": segmentation_mode,
            "language_classification_feature": self.language_classification_feature,
            "language_segmentation_feature": self.language_segmentation_feature,
            "vision_feature": self.vision_feature,
            "vision_segmentation_multiplier": self.vision_segmentation_multiplier,
            "vision_segmentation_weight": self.vision_segmentation_weight,
            "classification_img_size": self.classification_img_size,
            "segmentation_img_sizes": list(self.segmentation_img_sizes),
            "eval_img_size": self.eval_img_size,
            "segmentation_grid": max(self.segmentation_img_sizes) // self._patch_size[0],
            "text_model_type": self.text_model_type,
            "classification_prompts": list(CLASSIFICATION_PROMPTS),
            "segmentation_prompts": list(SEGMENTATION_PROMPTS),
            "use_classname_in_prompt_classification": True,
            "use_classname_in_prompt_segmentation": False,
            "memory_bank_neighbour": 1,
            "use_query_img_in_vision_memory_bank": False,
            "k_shot": self.k_shot,
            "shot_seed": self.shot_seed,
            "normalize_segmentations": self.normalize_segmentations,
            "official_uint8_quantization": True,
            "training_free": True,
            # run_fade.py applies no Gaussian blur, so the evaluator's own blur
            # is the only one and stays at its default.
            "gaussian_applied_inside_adapter": False,
            "cohort_image_size": self.image_size,
            "reference_path": "identical to the cohort path (518 then each scale)",
        }

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data

    # --------------------------------------------------------------- inputs ---
    def _scaled(self, batch: torch.Tensor) -> dict[int, torch.Tensor]:
        """Resize [B,3,518,518] in [0,1] to every scale FADE reads, normalized."""

        sizes = sorted({self.classification_img_size, *self.segmentation_img_sizes})
        scaled = {}
        for size in sizes:
            resized = self._resize(
                batch, [size, size], interpolation=self._bicubic, antialias=True
            ).clamp(0, 1)
            scaled[size] = (resized - self._mean) / self._std
        return scaled

    def _embeddings(self, batch: torch.Tensor) -> dict:
        return self._extract_image_embeddings(
            self._scaled(batch), self._gem, str(self.device)
        )

    # --------------------------------------------------------- per category ---
    def _text_models(self, category: str):
        classname = category.replace("_", " ")
        if classname not in self._classification_text:
            self._classification_text[classname] = self._build_text_model(
                gem_model=self._gem,
                prompt_paths=self._classification_prompts,
                classname=classname,
                text_model_type=self.text_model_type,
            )
        # Class-agnostic, so this is built once and shared by every category.
        if SEGMENTATION_CLASSNAME not in self._segmentation_text:
            self._segmentation_text[SEGMENTATION_CLASSNAME] = self._build_text_model(
                gem_model=self._gem,
                prompt_paths=self._segmentation_prompts,
                classname=SEGMENTATION_CLASSNAME,
                text_model_type=self.text_model_type,
            )
        return (
            self._classification_text[classname],
            self._segmentation_text[SEGMENTATION_CLASSNAME],
        )

    def _memory_bank(self, category: str) -> dict:
        cached = self._image_models.get(category)
        if cached is not None:
            return cached
        candidates = list(self._reference.candidates(category))
        rng = random.Random(self.shot_seed)
        rng.shuffle(candidates)
        samples = candidates[: self.k_shot]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.load(samples).to(self.device, dtype=torch.float32)

        collected: dict[int, list[np.ndarray]] = {
            size: [] for size in self.segmentation_img_sizes
        }
        with torch.no_grad():
            for index in range(len(references)):
                embeddings = self._embeddings(references[index : index + 1])
                for size in self.segmentation_img_sizes:
                    collected[size].append(
                        self._retrieve(
                            embeddings,
                            img_size=size,
                            feature_type=self.vision_feature,
                            token_type="patch",
                        )
                    )
        patches = {
            size: np.concatenate(collected[size])
            for size in self.segmentation_img_sizes
        }
        # use_query_img_in_vision_memory_bank is False, so this is a 1-NN bank.
        models = self._build_image_models(patches, False)
        self._image_models[category] = models
        return models

    # ------------------------------------------------------------ inference ---
    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("FADE received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.eval_img_size, self.eval_img_size), dtype=np.float32
        )
        with torch.no_grad():
            # One image at a time, as run_fade.py's batch_size=1 loop does: the
            # 896-pixel pass alone is 3137 tokens, and the memory bank is per
            # category so a mixed batch would have to be regrouped anyway.
            for index in range(len(batch)):
                category = str(categories[index])
                classification_text, segmentation_text = self._text_models(category)
                image_models = self._memory_bank(category)
                embeddings = self._embeddings(batch[index : index + 1])

                language_score = self._predict_classification(
                    text_model=classification_text,
                    image_embeddings=embeddings,
                    img_size=self.classification_img_size,
                    feature_type=self.language_classification_feature,
                )
                language_map = self._predict_segmentation(
                    model=segmentation_text,
                    image_embeddings=embeddings,
                    img_sizes=self.segmentation_img_sizes,
                    feature_type=self.language_segmentation_feature,
                    patch_size=self._patch_size,
                    segmentation_mode="language",
                )
                vision_map = self._predict_segmentation(
                    model=image_models,
                    image_embeddings=embeddings,
                    img_sizes=self.segmentation_img_sizes,
                    feature_type=self.vision_feature,
                    patch_size=self._patch_size,
                    segmentation_mode="vision",
                )
                vision_map = vision_map * self.vision_segmentation_multiplier
                # The vision score is the max of the unfused, unclipped map.
                vision_score = np.max(vision_map, axis=(1, 2))

                score = np.clip((language_score + vision_score) / 2, 0, 1)
                fused = (
                    (1.0 - self.vision_segmentation_weight) * language_map
                    + self.vision_segmentation_weight * vision_map
                )
                fused = np.clip(fused, 0, 1)
                # run_fade.py quantizes before resizing, and evaluates what comes
                # out of that, so the published maps carry 256 levels.
                quantized = (fused * 255).astype("uint8")
                resized = self._cv2.resize(
                    quantized[0], (self.eval_img_size, self.eval_img_size)
                )
                scores[index] = float(score[0])
                maps[index] = resized.astype(np.float32) / 255.0
        return scores, maps

    def close(self) -> None:
        self._gem = None
        self._classification_text.clear()
        self._segmentation_text.clear()
        self._image_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
