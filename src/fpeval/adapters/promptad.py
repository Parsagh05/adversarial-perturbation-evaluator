"""PromptAD adapter matching the official few-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import re
import sys

import numpy as np
import torch

from ..kaggle import download_kaggle_dataset, find_kaggle_files
from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# PromptAD trains prompts per class, per shot and per task, so the authors
# release no weights at all. These are the checkpoints retrained for this
# project and mirrored as a public Kaggle dataset: 162 files, being
# 15 MVTec classes and 12 VisA classes x {1, 2, 4} shots x {CLS, SEG}.
KAGGLE_DATASET = "parsagholami/promptad-few-shot-checkpoints-mvtec-ad-and-visa"
CHECKPOINT_SEED = 111
SHOT_VALUES = (1, 2, 4)
TASKS = ("CLS", "SEG")
# utils/training_utils.get_dir_from_args builds exactly this path.
CHECKPOINT_TEMPLATE = "{task}-Seed_{seed}-{category}-check_point.pt"
# save_check_point stores only these three buffers; everything else in the
# state dict is the frozen backbone or the prompt learner, which inference
# never touches.
CHECKPOINT_KEYS = ("feature_gallery1", "feature_gallery2", "text_features")

MODEL_IMAGE_SIZE = 240
RESOLUTION = 400


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "PromptAD" / "model.py",
        root / "PromptAD" / "ad_prompts.py",
        root / "datasets" / "__init__.py",
        root / "test_cls.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official PromptAD repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"PromptAD", "datasets", "utils", "dataset", "model"}
            or name.startswith(("PromptAD.", "datasets.", "utils."))
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
        importlib.import_module("PromptAD.model"),
        importlib.import_module("datasets"),
    )


def _mentions_shot(path: Path, k_shot: int) -> bool:
    """Whether a path identifies itself as belonging to one shot count.

    The released package wraps each configuration in its own directory and keeps
    PromptAD's ``<dataset>/k_<shot>/checkpoint/`` tree inside it, so the shot
    count shows up either as a ``k_<n>`` component or as a ``<n>shot`` token in
    the wrapper name. Accepting both keeps the lookup working whether the inner
    tree survived packaging or was flattened.
    """

    if f"k_{k_shot}" in {part.lower() for part in path.parts}:
        return True
    token = re.compile(rf"(?:^|[^0-9]){k_shot}[ _-]?shot(?:$|[^0-9])")
    return any(token.search(part.lower()) for part in path.parts)


def resolve_checkpoint(
    target_dataset: str,
    k_shot: int,
    category: str,
    task: str,
    *,
    seed: int = CHECKPOINT_SEED,
    checkpoint_root: str | Path | None = None,
    download_root: str | None = None,
) -> Path:
    """Return one released checkpoint, fetching the Kaggle dataset if needed.

    The file name is PromptAD's own,
    ``<TASK>-Seed_<seed>-<class>-check_point.pt``, but the directories above it
    depend on how the package was assembled, so the file is searched for and
    then narrowed by dataset and shot count rather than addressed by a fixed
    path. The narrowing only has to separate the six configurations that share a
    file name; it is applied in order and each step must leave the answer
    unambiguous.
    """

    task = task.upper()
    if task not in TASKS:
        raise ValueError(f"PromptAD task must be one of {TASKS}")
    if int(k_shot) not in SHOT_VALUES:
        raise ValueError(f"PromptAD checkpoints cover k_shot in {SHOT_VALUES}")
    root = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root
        else download_kaggle_dataset(KAGGLE_DATASET, download_root=download_root)
    )
    filename = CHECKPOINT_TEMPLATE.format(task=task, seed=int(seed), category=category)
    found = find_kaggle_files(root, filename)
    if not found:
        raise FileNotFoundError(
            f"No {filename} below {root}. The released package keeps PromptAD's "
            f"own <TASK>-Seed_<seed>-<class>-check_point.pt naming."
        )
    dataset = str(target_dataset).lower()
    matches = [path for path in found if dataset in str(path).lower()]
    if len(matches) > 1:
        matches = [path for path in matches if _mentions_shot(path, int(k_shot))]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} for {target_dataset} at "
            f"{k_shot}-shot below {root}, found "
            f"{[str(path) for path in matches] or [str(path) for path in found]}"
        )
    return matches[0]


@register_adapter("prompt-ad")
@register_adapter("promptad")
class PromptADAdapter(ModelAdapter):
    """Official PromptAD few-shot inference for MVTec AD and VisA.

    Follows train_cls.py / train_seg.py and their test counterparts at the
    argparse defaults: a ``ViT-B-16-plus-240`` backbone with ``laion400m_e32``
    weights at **240 pixels** in fp16, 4 normal context tokens, 1 anomaly
    context token, 4 learnable anomaly suffixes, and a 400-pixel output grid.

    PromptAD learns its prompts **per class, per shot and per task**, so the
    authors release no weights. The checkpoints here were retrained for this
    project at seed 111 for 100 epochs per class and task and are mirrored as a
    public Kaggle dataset, which the adapter downloads automatically.

    **The image score and the anomaly map come from different checkpoints**, and
    that is the official protocol rather than a shortcut: ``test_cls.py`` loads a
    CLS checkpoint and reports image metrics from ``model(data, 'cls')``, while
    ``test_seg.py`` loads a SEG checkpoint and reports pixel metrics from
    ``model(data, 'seg')``. The adapter runs both per category - the score from
    the CLS buffers, the map from the SEG buffers - so each half reproduces the
    number the paper reports for it. It costs two backbone passes per image.

    A checkpoint holds only three tensors: ``feature_gallery1``,
    ``feature_gallery2`` and ``text_features``. Those are the k-shot memory bank
    and the learned text anchors; the prompt learner that produced them is not
    saved and inference never reads it, which is why the official test scripts
    can load with ``strict=False``. The reference images are therefore **baked
    into the checkpoint**, like INP-Former, and no reference set is drawn at
    runtime. The gallery is sized ``k_shot * 15 * 15``, so a checkpoint only
    loads at the shot count it was trained for.

    Two further notes. ``model(data, 'seg')`` applies its own
    ``gaussian_filter(sigma=4)``, so the evaluator adds none. And the released
    checkpoints were selected on **best test AUROC over 100 epochs**, which is an
    oracle selection rule the upstream training script uses; it is recorded in
    the metadata because it inflates absolute numbers relative to methods that
    select on validation or take the final epoch.
    """

    name = "promptad"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        k_shot: int = 1,
        checkpoint_root: str | None = None,
        download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = MODEL_IMAGE_SIZE,
        resolution: int = RESOLUTION,
        backbone: str = "ViT-B-16-plus-240",
        pretrained_dataset: str = "laion400m_e32",
        n_ctx: int = 4,
        n_ctx_ab: int = 1,
        n_pro: int = 1,
        n_pro_ab: int = 4,
        seed: int = CHECKPOINT_SEED,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("PromptAD target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"PromptAD checkpoints cover k_shot in {SHOT_VALUES}")
        if backbone != "ViT-B-16-plus-240":
            raise ValueError("The released checkpoints use ViT-B-16-plus-240")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.resolution = int(resolution)
        self.k_shot = int(k_shot)
        self.target_dataset = target_key
        self.seed = int(seed)
        self._checkpoint_root = checkpoint_root
        self._download_root = download_root

        model_module, datasets_module = _import_official_repository(repository)
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._categories = list(datasets_module.dataset_classes[target_key])
        # The prompt learner is built per class but never read at inference - the
        # three saved buffers are the whole model - so one instance serves every
        # category and the backbone is built once instead of 27 times.
        anchor = self._categories[0]
        model = model_module.PromptAD(
            out_size_h=self.resolution,
            out_size_w=self.resolution,
            device=str(self.device),
            backbone=backbone,
            pretrained_dataset=pretrained_dataset,
            n_ctx=int(n_ctx),
            n_pro=int(n_pro),
            n_ctx_ab=int(n_ctx_ab),
            n_pro_ab=int(n_pro_ab),
            class_name=anchor,
            k_shot=self.k_shot,
            img_resize=self.model_image_size,
            img_cropsize=self.model_image_size,
        ).to(self.device)
        model.eval_mode()
        self._model = model
        self._loaded: tuple[str, str] | None = None
        self._checkpoints: dict[tuple[str, str], Path] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "PromptAD/test_cls.py + test_seg.py",
            "mode": "few_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained_dataset": pretrained_dataset,
            "precision": model.precision,
            "model_image_size": self.model_image_size,
            "cohort_image_size": self.image_size,
            "map_resolution": self.resolution,
            "k_shot": self.k_shot,
            "n_ctx": int(n_ctx),
            "n_ctx_ab": int(n_ctx_ab),
            "n_pro": int(n_pro),
            "n_pro_ab": int(n_pro_ab),
            "seed": self.seed,
            "checkpoint_source": KAGGLE_DATASET,
            "checkpoint_keys": list(CHECKPOINT_KEYS),
            "image_score_checkpoint": "CLS",
            "anomaly_map_checkpoint": "SEG",
            "reference_set": "baked into the released checkpoint",
            "released_checkpoints_are_retrained": True,
            "checkpoint_selection_rule": "best test AUROC over 100 epochs",
            # model.forward('seg') blurs with gaussian_filter(sigma=4).
            "official_gaussian_sigma": 4.0,
            "gaussian_applied_inside_adapter": True,
        }
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["checkpoints"] = {
            f"{category}/{task}": path.name
            for (category, task), path in sorted(self._checkpoints.items())
        }
        return data

    def _load(self, category: str, task: str) -> None:
        """Swap in one checkpoint's three buffers."""

        if self._loaded == (category, task):
            return
        key = (category, task)
        path = self._checkpoints.get(key)
        if path is None:
            path = resolve_checkpoint(
                self.target_dataset,
                self.k_shot,
                category,
                task,
                seed=self.seed,
                checkpoint_root=self._checkpoint_root,
                download_root=self._download_root,
            )
            self._checkpoints[key] = path
        state = torch.load(str(path), map_location=self.device)
        unexpected = sorted(set(state) - set(CHECKPOINT_KEYS))
        if unexpected:
            raise ValueError(
                f"PromptAD checkpoint {path.name} carries unexpected tensors "
                f"{unexpected}; only {list(CHECKPOINT_KEYS)} are inference state."
            )
        missing = sorted(set(CHECKPOINT_KEYS) - set(state))
        if missing:
            raise ValueError(
                f"PromptAD checkpoint {path.name} is missing {missing}"
            )
        # strict=False, exactly as the official test scripts load it: the frozen
        # backbone and the prompt learner are absent from the file by design.
        self._model.load_state_dict(state, strict=False)
        self._loaded = key

    def _prepare(self, batch: torch.Tensor) -> torch.Tensor:
        """Resize(240) then CenterCrop(240), a no-op crop on a square input."""

        resized = self._resize(
            batch,
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        return (resized - self._mean) / self._std

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("PromptAD received mismatched images and categories")
        batch = self._prepare(images.to(self.device, dtype=torch.float32))
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.resolution, self.resolution), dtype=np.float32
        )

        # Group by category, and within a category by task, so each checkpoint is
        # loaded once rather than once per image.
        order: dict[str, list[int]] = {}
        for index, raw in enumerate(categories):
            order.setdefault(str(raw), []).append(index)

        with torch.no_grad():
            for category, indices in sorted(order.items()):
                if category not in self._categories:
                    raise KeyError(
                        f"PromptAD has no checkpoint for {category!r} in "
                        f"{self.target_dataset}; known: {sorted(self._categories)}"
                    )
                selected = batch[indices]

                # test_cls.py: the image score comes from the CLS checkpoint.
                self._load(category, "CLS")
                image_scores, _ = self._model(selected, "cls")
                for position, value in zip(indices, image_scores):
                    scores[position] = float(value)

                # test_seg.py: the map comes from the SEG checkpoint, already
                # blurred at sigma 4 inside the forward.
                self._load(category, "SEG")
                anomaly_maps = self._model(selected, "seg")
                for position, single in zip(indices, anomaly_maps):
                    maps[position] = np.asarray(single, dtype=np.float32)
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._checkpoints.clear()
        self._loaded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
