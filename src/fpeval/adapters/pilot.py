"""PILOT adapter matching the official tta.py (zero-shot, test-time adaptation)."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Modules of the repository whose names other target repositories reuse.
OFFICIAL_MODULES = {
    "tta", "PILOT_lib", "prompt_ensemble", "utils", "dataset", "logger", "loss", "metrics",
}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "tta.py", root / "prompt_ensemble.py", root / "PILOT_lib")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official PILOT repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name.split(".")[0] not in OFFICIAL_MODULES:
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "") or "") if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return importlib.import_module("tta")


class _CohortDataset(torch.utils.data.Dataset):
    """The cohort in the item layout tta.py reads; labels and masks are unused."""

    def __init__(self, images, categories, mean, std) -> None:
        self._images, self._categories = images, categories
        self._mean, self._std = mean, std

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, object]:
        image = (self._images[index].float() - self._mean) / self._std
        return {
            "img": image, "img_mask": torch.zeros(1, 1, 1),
            "cls_name": str(self._categories[index]), "cls_id": 0, "anomaly": 0,
        }


@register_adapter("pilot")
class PILOTAdapter(ModelAdapter):
    """Official PILOT inference with its label-free test-time adaptation.

    PILOT releases no weights; ``checkpoint`` is the ``epoch_N.pth`` that the
    official train.py writes when trained on the other dataset (the paper's
    protocol, prompt length 12 and 5 epochs). tta.py then alternates, over the
    whole test set, a scoring pass with one round of prompt training on the
    most confident pseudo-labels (top and bottom quarter after per-class
    percentile normalisation). The scores and maps are those of the scoring
    pass after ``tta_rounds`` adaptation rounds, each prediction restarting
    from the trained checkpoint. tta.py itself keeps the pass with the best
    image AUROC + AP against the test labels; a label-free evaluation cannot,
    so the round count is fixed instead (default 4, the last pass tta.py's
    5-epoch loop scores). The set is the cohort the engine passes, so an
    attacked image changes the pseudo-labels and the prompts every other image
    is scored with.
    """

    name = "pilot"
    transductive = True

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str,
        device: str = "cuda",
        image_size: int = 518,
        tta_rounds: int = 4,
        n_ctx: int = 12,
        depth: int = 9,
        t_n_ctx: int = 4,
        pool_size: int = 5,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        sigma: int = 4,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("PILOT target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official PILOT evaluation uses image_size=518")
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PILOT checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.tta_rounds = int(tta_rounds)
        self.seed = int(seed)
        self._tta = _import_official_repository(repository)
        # tta.py's argparse defaults, with the prompt length the checkpoint has.
        self._args = SimpleNamespace(
            features_list=[6, 12, 18, 24], image_size=self.image_size, depth=int(depth),
            n_ctx=int(n_ctx), t_n_ctx=int(t_n_ctx), feature_map_layer=[0, 1, 2, 3],
            sigma=int(sigma), pool_size=int(pool_size), batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )
        design = {
            "PromptPool_size": self._args.pool_size, "Prompt_length": self._args.n_ctx,
            "learnabel_text_embedding_depth": self._args.depth,
            "learnabel_text_embedding_length": self._args.t_n_ctx,
        }
        library, prompts = self._tta.PILOT_lib, self._tta
        model, _ = library.load("ViT-L/14@336px", device=self.device, design_details=design)
        model.visual.DAPM_replace(DPAM_layer=20)
        self._checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self._prompt_learner = prompts.PromptLearner(model.to("cpu"), design).to(self.device)
        self._attribute_hint = prompts.AttributePromptHint(model, design).to(self.device)
        self._attribute_hint.load_state_dict(self._checkpoint["attribute_hint_state_dict"])
        model.load_state_dict(self._checkpoint["trainable_state_dict"], strict=False)
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self._std = torch.tensor(CLIP_STD).view(3, 1, 1)
        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "PILOT/tta.py",
            "mode": "zero_shot",
            "transductive": "test-time adaptation over the cohort scored",
            "target_dataset": target_key,
            "image_size": self.image_size,
            "checkpoint": str(checkpoint_path),
            "tta_rounds": self.tta_rounds,
            "tta_selection": "fixed round count (tta.py selects by test labels)",
            "seed": self.seed,
            **{name: value for name, value in vars(self._args).items()},
            # tta.py blurs each map with gaussian_filter(sigma=4) itself.
            "official_gaussian_sigma": 0.0,
        }

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("PILOT adapts on sets; the engine calls predict_set")

    def predict_set(
        self, images: Sequence[torch.Tensor], categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("PILOT received mismatched images and categories")
        # Every set adapts from the trained prompts, under the same seed (tta.py
        # sets none; its shuffled batches and Gumbel noise draw from it).
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._prompt_learner.load_state_dict(self._checkpoint["prompt_learner_state_dict"])
        dataset = _CohortDataset(images, categories, self._mean, self._std)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        objects = list(dict.fromkeys(str(name) for name in categories))
        prompt_learner = self._prompt_learner
        for _ in range(self.tta_rounds):
            results = self._tta.run_test_loop(
                self._args, self._model, prompt_learner, self._attribute_hint,
                self.device, objects, loader,
            )
            selected = self._tta.create_tta_dataset_from_results(results, dataset)
            if len(selected) == 0:
                break
            prompt_learner = self._tta.run_training_loop(
                self._args, self._model, prompt_learner, self._attribute_hint,
                selected, self.device,
            )
        results = self._tta.run_test_loop(
            self._args, self._model, prompt_learner, self._attribute_hint,
            self.device, objects, loader,
        )
        scores = np.empty(len(images), dtype=np.float32)
        maps = np.empty((len(images), self.image_size, self.image_size), dtype=np.float32)
        for name in objects:
            entry = results[name]
            for index, score, anomaly_map in zip(entry["indices"], entry["pr_sp"], entry["anomaly_maps"]):
                scores[index] = float(score)
                maps[index] = anomaly_map[0].float().numpy()
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._prompt_learner = None
        self._attribute_hint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
