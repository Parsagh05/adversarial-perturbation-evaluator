"""AF-CLIP adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The weight files are named after the dataset they were TRAINED on, and
# test.sh evaluates a dataset with the other one's weights.
ZERO_SHOT_SOURCE = {"mvtec": "visa", "visa": "mvtec"}
WEIGHT_FILES = ("{source}_prompt.pt", "{source}_adaptor.pt")


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "clip" / "clip.py", root / "clip" / "adaptor.py", root / "main.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official AF-CLIP repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # "clip" is also the name of the OpenAI package, and other target
    # repositories ship modules called dataset/util.
    for name in list(sys.modules):
        if not (
            name in {"clip", "dataset", "util"}
            or name.startswith(("clip.", "dataset.", "util."))
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
    return importlib.import_module("clip.clip")


def resolve_weights(repository: str | Path, source: str, weight_dir: str | None = None):
    """Return the prompt and adaptor files; AF-CLIP commits them in-repo."""

    folder = (
        Path(weight_dir).expanduser().resolve()
        if weight_dir
        else Path(repository).expanduser().resolve() / "weight"
    )
    paths = tuple(folder / name.format(source=source) for name in WEIGHT_FILES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        available = (
            sorted(entry.name for entry in folder.glob("*.pt"))
            if folder.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"AF-CLIP weights not found: {missing}. Cloning the repository is the "
            f"download; {folder} currently holds {available}."
        )
    return paths


@register_adapter("af-clip")
@register_adapter("afclip")
class AFCLIPAdapter(ModelAdapter):
    """Official AF-CLIP zero-shot inference for MVTec AD and VisA.

    Follows the zero-shot rows of test.sh: CLIP ViT-L/14@336px at 518 pixels,
    feature layers 6/12/18/24 each aggregated over 1x1, 3x3 and 5x5 Gaussian
    neighbourhoods, a 12-token learned state prompt in front of the fixed
    "without defect." / "with defect." pair, and the trained adaptor on the patch
    tokens. The zero-shot branch keeps ``memorybank`` unset, so ``detect_forward``
    reduces to ``detect_forward_seg`` and ``alpha`` never applies. Scores and maps
    are used raw: the official loop performs no normalization at all.
    """

    name = "afclip"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        weight_dir: str | None = None,
        clip_download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        clip_model: str = "ViT-L/14@336px",
        feature_layers: Sequence[int] = (6, 12, 18, 24),
        prompt_len: int = 12,
        seed: int = 122,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_SOURCE:
            raise ValueError("AF-CLIP target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official AF-CLIP evaluation uses image_size=518")
        if clip_model != "ViT-L/14@336px":
            raise ValueError("Official AF-CLIP evaluation uses ViT-L/14@336px")
        self.device = torch.device(device)
        self.image_size = int(image_size)

        clip_module = _import_official_repository(repository)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        source = ZERO_SHOT_SOURCE[target_key]
        prompt_path, adaptor_path = resolve_weights(repository, source, weight_dir)

        # main.py picks jit only for names outside available_models().
        load_kwargs = {"device": self.device, "jit": False}
        if clip_download_root:
            load_kwargs["download_root"] = clip_download_root
        model, _ = clip_module.load(name=clip_model, **load_kwargs)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        # detect_forward reads feature_layers; memory_layers and alpha only
        # matter for the few-shot memory bank, which zero-shot leaves unset.
        self._args = SimpleNamespace(
            feature_layers=[int(layer) for layer in feature_layers],
            memory_layers=[int(layer) for layer in feature_layers],
            prompt_len=int(prompt_len),
            alpha=0.1,
        )
        model.insert(args=self._args, tokenizer=clip_module.tokenize, device=self.device)

        # The released files are pickled tensors and modules, not state dicts.
        model.state_prompt_embedding = torch.load(
            str(prompt_path), map_location=self.device, weights_only=False
        )
        model.adaptor = torch.load(
            str(adaptor_path), map_location=self.device, weights_only=False
        )
        model.state_prompt_embedding.requires_grad_(False)
        model.adaptor.to(self.device).eval().requires_grad_(False)
        if type(self) is AFCLIPAdapter and model.memorybank is not None:
            raise ValueError("AF-CLIP zero-shot evaluation must leave memorybank unset")
        self._model = model

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "AF-CLIP/test.sh (zero-shot rows)",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "clip_model": clip_model,
            "image_size": self.image_size,
            "feature_layers": self._args.feature_layers,
            "neighbourhood_scales": [1, 3, 5],
            "prompt_len": int(prompt_len),
            "normal_prompt": model.normal_cls_prompt,
            "abnormal_prompt": model.anomaly_cls_prompt,
            "temperature": 0.07,
            "seed": int(seed),
            "weights": [prompt_path.name, adaptor_path.name],
            "checkpoint_selection": source,
            # evaluation_pixel upsamples bilinearly then applies
            # gaussian_filter(sigma=4); the shared evaluator does both.
            "official_gaussian_sigma": 4.0,
            # alpha only combines the few-shot memory branch, which is unused.
            "alpha_unused_in_zero_shot": True,
            "model_dtype": str(model.dtype),
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories  # AF-CLIP prompts name a defect, never the category.
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        with torch.no_grad():
            labels, maps = self._model.detect_forward(batch, self._args)
        # maps are [B,1,h,w] at patch resolution; the shared evaluator performs
        # the same bilinear upsample and gaussian_filter the official loop does.
        return (
            labels.reshape(-1).float().cpu().numpy().astype(np.float32),
            maps.squeeze(1).float().cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@register_adapter("af-clip-fewshot")
@register_adapter("afclip_fewshot")
class AFCLIPFewShotAdapter(AFCLIPAdapter):
    """Official AF-CLIP few-shot inference for MVTec AD and VisA.

    Adds the memory bank ``eval_all_class`` builds when ``--fewshot`` is
    positive. With it present ``detect_forward`` stops being the zero-shot
    branch and returns ``memory + alpha * segmentation`` for both the map and
    the image score, which is where the otherwise-dead ``alpha`` of 0.1 finally
    applies.

    ``store_memory`` resets the bank on every call, and the official loop only
    works because its dataloader batch holds every shot at once, so the adapter
    stores each category's bank in a single call.

    The official selection is ``np.random.choice(n, k, replace=False)`` under
    the process seed, and its few-shot script passes ``--seed -1``, i.e. a fresh
    random seed per repeat with five repeats averaged. There is therefore no
    canonical selection to reproduce; the adapter pins ``shot_seed`` and records
    which files it drew.
    """

    name = "afclip_fewshot"

    def __init__(
        self,
        *,
        k_shot: int = 4,
        shot_seed: int = 122,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        **kwargs,
    ) -> None:
        if int(k_shot) < 1:
            raise ValueError("AF-CLIP few-shot evaluation needs k_shot >= 1")
        target = str(kwargs["target_dataset"]).strip().lower()
        super().__init__(**kwargs)
        self.k_shot = int(k_shot)
        self.shot_seed = int(shot_seed)
        self._args.fewshot = self.k_shot
        self._reference = NormalReference(
            dataset=target,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._current: str | None = None
        self._selection: dict[str, list[str]] = {}
        self._runtime_metadata.update(
            {
                "adapter": self.name,
                "mode": "few_shot",
                "k_shot": self.k_shot,
                "shot_seed": self.shot_seed,
                "memory_layers": self._args.memory_layers,
                # alpha is only reachable once the memory bank exists.
                "alpha": self._args.alpha,
                "alpha_unused_in_zero_shot": False,
                "reference_selection": "np.random.choice without replacement",
            }
        )

    def _ensure_memory(self, category: str) -> None:
        if self._current == category:
            return
        candidates = self._reference.candidates(category)
        rng = np.random.default_rng(self.shot_seed)
        picked = sorted(
            rng.choice(len(candidates), size=self.k_shot, replace=False).tolist()
        )
        samples = [candidates[index] for index in picked]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.cached(category, samples).to(
            self.device, dtype=torch.float32
        )
        references = (references - self._mean) / self._std
        # One call: the official store_memory resets the bank each time.
        self._model.store_memory(references, self._args)
        self._current = category

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("AF-CLIP received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps: list[np.ndarray] = [None] * len(batch)
        with torch.no_grad():
            # The memory bank is per category, so images are grouped by it.
            for index in sorted(range(len(categories)), key=lambda i: str(categories[i])):
                self._ensure_memory(str(categories[index]))
                label, anomaly_map = self._model.detect_forward(
                    batch[index : index + 1], self._args
                )
                scores[index] = float(label.reshape(-1)[0])
                maps[index] = anomaly_map.squeeze(1)[0].float().cpu().numpy()
        return scores, np.stack(maps).astype(np.float32)

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data
