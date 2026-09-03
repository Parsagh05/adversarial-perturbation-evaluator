"""WinCLIP adapter matching the official zero-shot evaluation defaults."""

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


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "WinCLIP" / "model.py", root / "eval_WinCLIP.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official WinClip repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"WinCLIP", "datasets", "utils", "dataset"}
            or name.startswith(("WinCLIP.", "datasets.", "utils."))
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
    return importlib.import_module("WinCLIP.model")


@register_adapter("win-clip")
@register_adapter("winclip")
class WinCLIPAdapter(ModelAdapter):
    """Official WinCLIP zero-shot inference for MVTec AD and VisA.

    Follows eval_WinCLIP.py at its defaults: the ``ViT-B-16-plus-240`` backbone
    with ``laion400m_e32`` weights, 240-pixel inputs, window scales (2, 3), and
    a 400-pixel output grid. WinCLIP is training-free, so there is no checkpoint
    beyond the backbone, which open_clip downloads on first use.

    With ``k_shot`` at zero the visual gallery stays empty, so the official
    ``textual_visual`` fusion reduces to ``1 / (1/t + 1/t)``, i.e. **half** the
    textual map. That halving is monotonic and cannot change AUROC, but it does
    change the absolute values the frozen thresholds are calibrated against, so
    it is reproduced rather than simplified away. The image score is the maximum
    of the returned map, exactly as ``metric_cal`` computes it, and the official
    loop applies no Gaussian filter, so the shared evaluator adds none.

    The cohort images arrive at 518 pixels because that is the grid the
    perturbations are defined on; the adapter performs the official bicubic
    resize down to 240 itself. The L-infinity budget therefore applies at 518
    and is attenuated by that downsample, which is a property of evaluating a
    240-pixel model against a 518-pixel perturbation, not of this adapter.
    """

    name = "winclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-B-16-plus-240",
        pretrained_dataset: str = "laion400m_e32",
        scales: Sequence[int] = (2, 3),
        input_size: int = 240,
        resolution: int = 400,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("WinCLIP target_dataset must be 'mvtec' or 'visa'")
        if backbone != "ViT-B-16-plus-240":
            raise ValueError("Official WinCLIP evaluation uses ViT-B-16-plus-240")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.input_size = int(input_size)
        self.resolution = int(resolution)

        model_module = _import_official_repository(repository)
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        model = model_module.WinClipAD(
            out_size_h=self.resolution,
            out_size_w=self.resolution,
            device=str(self.device),
            backbone=backbone,
            pretrained_dataset=pretrained_dataset,
            scales=tuple(int(scale) for scale in scales),
            img_resize=self.input_size,
            img_cropsize=self.input_size,
        )
        model = model.to(self.device)
        model.eval_mode()
        if type(self) is WinCLIPAdapter and model.visual_gallery is not None:
            raise ValueError("WinCLIP zero-shot evaluation must leave the gallery empty")
        self._model = model
        self._current: str | None = None

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "WinClip/eval_WinCLIP.py",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained_dataset": pretrained_dataset,
            "scales": [int(scale) for scale in scales],
            "model_input_size": self.input_size,
            "map_resolution": self.resolution,
            "cohort_image_size": self.image_size,
            "fusion_version": model.fusion_version,
            "precision": model.precision,
            "k_shot": 0,
            "seed": int(seed),
            # The official gaussian_filter call is commented out.
            "official_gaussian_sigma": 0.0,
            "image_score": "max of the anomaly map",
            "training_free": True,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _ensure_text_gallery(self, category: str) -> None:
        # build_text_feature_gallery overwrites the gallery, so it has to run
        # again whenever the category changes.
        if self._current != category:
            self._model.build_text_feature_gallery(category)
            self._current = category

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("WinCLIP received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        # The official transform resizes to 240 with PIL bicubic and centre-crops
        # to the same size, which is a no-op on a square input.
        batch = self._resize(
            batch,
            [self.input_size, self.input_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.resolution, self.resolution), dtype=np.float32
        )
        with torch.no_grad():
            # The text gallery is per category, so images are grouped by it.
            order = sorted(range(len(categories)), key=lambda i: str(categories[i]))
            for index in order:
                category = str(categories[index])
                self._ensure_text_gallery(category)
                produced = self._model(batch[index : index + 1])
                anomaly_map = np.asarray(produced[0], dtype=np.float32)
                maps[index] = anomaly_map
                # metric_cal scores an image by the maximum of its map.
                scores[index] = float(anomaly_map.max())
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._current = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# The repository pins the exact k-shot selection per category in
# datasets/seeds_mvtec/<category>/selected_samples_per_run.txt, as lines
# "<experiment_indx>-<k_shot>: <stem> <stem> ...".
SEED_FILE = "datasets/seeds_mvtec/{category}/selected_samples_per_run.txt"
SHOT_VALUES = (1, 5, 10)
EXPERIMENT_SEEDS = (111, 333, 999)


def read_seed_selection(
    repository: str | Path, category: str, k_shot: int, experiment_indx: int
) -> list[str]:
    """Return the committed MVTec file stems for one (run, k) pair."""

    path = Path(repository).expanduser().resolve() / SEED_FILE.format(category=category)
    if not path.is_file():
        raise FileNotFoundError(
            f"WinCLIP seed file not found: {path}. It ships with the repository "
            "and pins which normal images each run uses."
        )
    prefix = f"{int(experiment_indx)}-{int(k_shot)}: "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].split()
    raise ValueError(f"WinCLIP seed file {path} has no entry for {prefix.strip()}")


@register_adapter("winclip-fewshot")
@register_adapter("winclip_fewshot")
class WinCLIPFewShotAdapter(WinCLIPAdapter):
    """Official WinCLIP few-shot inference for MVTec AD and VisA.

    Adds the image feature gallery that ``eval_WinCLIP.py`` builds when
    ``k_shot`` is positive. With the gallery present the ``textual_visual``
    fusion becomes the real harmonic mean of the textual and visual maps rather
    than the halved textual map the zero-shot path reduces to.

    ``build_image_feature_gallery`` resets the gallery on every call, so the
    official loop only works because its training batch holds every shot at
    once; the adapter builds each category's gallery in a single call for the
    same reason.

    On MVTec the selection is the one committed to the repository under
    ``datasets/seeds_mvtec``, which pins the exact file stems per run and shot
    count, so it reproduces the paper's runs. VisA ships no such file - the
    official loader draws ``random.sample`` under the run seed - so the adapter
    draws deterministically from the sorted normal training images with that
    same seed and records the file names it used.
    """

    name = "winclip_fewshot"

    def __init__(
        self,
        *,
        k_shot: int = 1,
        experiment_indx: int = 0,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        **kwargs,
    ) -> None:
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"Official WinCLIP evaluation uses k_shot in {SHOT_VALUES}")
        if int(experiment_indx) not in (0, 1, 2):
            raise ValueError("WinCLIP experiment_indx must be 0, 1 or 2")
        repository = kwargs["repository"]
        target = str(kwargs["target_dataset"]).strip().lower()
        super().__init__(**kwargs)
        self.k_shot = int(k_shot)
        self.experiment_indx = int(experiment_indx)
        self.shot_seed = EXPERIMENT_SEEDS[self.experiment_indx]
        self._repository = repository
        self._reference = NormalReference(
            dataset=target,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._selection: dict[str, list[str]] = {}
        self._runtime_metadata.update(
            {
                "adapter": self.name,
                "mode": "few_shot",
                "k_shot": self.k_shot,
                "experiment_indx": self.experiment_indx,
                "shot_seed": self.shot_seed,
                "reference_selection": (
                    "datasets/seeds_mvtec (committed)"
                    if target == "mvtec"
                    else "seeded draw from the sorted normal training images"
                ),
                "training_free": True,
            }
        )

    def _select(self, category: str) -> list:
        candidates = self._reference.candidates(category)
        if self._reference.dataset == "mvtec":
            stems = set(
                read_seed_selection(
                    self._repository, category, self.k_shot, self.experiment_indx
                )
            )
            chosen = [s for s in candidates if s.image_path.stem in stems]
            if len(chosen) != self.k_shot:
                raise ValueError(
                    f"WinCLIP seed file names {len(stems)} stems for {category} but "
                    f"{len(chosen)} matched the {len(candidates)} training images"
                )
            return chosen
        rng = random.Random(self.shot_seed)
        return [candidates[i] for i in sorted(rng.sample(range(len(candidates)), self.k_shot))]

    def _ensure_text_gallery(self, category: str) -> None:
        if self._current == category:
            return
        super()._ensure_text_gallery(category)
        samples = self._select(category)
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.cached(category, samples).to(self.device)
        references = self._resize(
            references,
            [self.input_size, self.input_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        references = (references - self._mean) / self._std
        # One call: the official builder resets the gallery each time.
        self._model.build_image_feature_gallery(references)

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data
