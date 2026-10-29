"""MultiADS adapter matching the official test.py (zero-shot and few-shot)."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# test.py picks the prompt ensemble by the test dataset: every defect type of
# the dataset is one text class beside the normal one.
PROMPT_MODULE = {
    "mvtec": "prompts.prompt_ensemble_mvtec_20cls",
    "visa": "prompts.prompt_ensemble_visa_19cls_test",
}


def _import_official_repository(repository: str | Path, target: str):
    root = Path(repository).expanduser().resolve()
    required = (root / "model.py", root / "test.py", root / "open_clip", root / "prompts")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official MultiADS repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"model", "prompts", "open_clip", "dataset"}
            or name.startswith(("open_clip.", "prompts."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "") or "") if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("open_clip"),
        importlib.import_module("model"),
        importlib.import_module(PROMPT_MODULE[target]),
    )


@register_adapter("multiads")
class MultiADSAdapter(ModelAdapter):
    """Official MultiADS zero-shot inference for MVTec AD and VisA.

    MultiADS releases no weights; ``checkpoint`` is the ``epoch_N.pth`` that the
    official train.py writes when trained on the other dataset. The map is the
    sum over layers 6/12/18/24 of the probability mass on the defect-type text
    classes. test.py overwrites the text-based image score with the mean of the
    three highest map values, min-max normalised per class over the test set;
    that normalisation is monotonic within a category, which every metric and
    threshold here is computed within, so the adapter reports the top-3 mean
    as it is.
    """

    name = "multiads"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        features: Sequence[int] = (6, 12, 18, 24),
        seed: int = 42,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in PROMPT_MODULE:
            raise ValueError("MultiADS target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official MultiADS evaluation uses image_size=518")
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MultiADS checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = [int(level) for level in features]
        self._autocast = self.device.type == "cuda"

        open_clip, model_module, prompt_module = _import_official_repository(
            repository, target_key
        )
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        config_path = (
            Path(repository).expanduser().resolve()
            / "open_clip" / "model_configs" / f"{backbone}.json"
        )
        configs = json.loads(config_path.read_text(encoding="utf-8"))

        model, _, _ = open_clip.create_model_and_transforms(
            backbone, self.image_size, pretrained=pretrained
        )
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(backbone)
        self._prompt_module = prompt_module

        linear = model_module.LinearLayer(
            configs["vision_cfg"]["width"], configs["embed_dim"],
            len(self.features), backbone,
        ).to(self.device)
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        linear.load_state_dict(payload["trainable_linearlayer"])
        linear.eval().requires_grad_(False)
        self._linear = linear
        self._text_cache: dict[str, torch.Tensor] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "MultiADS/test.py --mode zero_shot",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": pretrained,
            "image_size": self.image_size,
            "features": self.features,
            "prompt_ensemble": PROMPT_MODULE[target_key],
            "seed": int(seed),
            "checkpoint": str(checkpoint_path),
            "image_score": "mean of the 3 highest map values",
            # test.py sums the per-layer maps and applies no gaussian filter.
            "official_gaussian_sigma": 0.0,
        }
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _text_features(self, category: str) -> torch.Tensor:
        cached = self._text_cache.get(category)
        if cached is None:
            with torch.no_grad(), torch.autocast("cuda", enabled=self._autocast):
                prompts = self._prompt_module.encode_text_with_prompt_ensemble(
                    self._model, [category], self._tokenizer, self.device
                )
            cached = prompts[category].detach()
            self._text_cache[category] = cached
        return cached

    def _upsample(self, patch_map: torch.Tensor) -> torch.Tensor:
        """[B, L, C] patch values to [B, C, S, S], as test.py interpolates."""
        count, length, channels = patch_map.shape
        side = int(np.sqrt(length))
        return F.interpolate(
            patch_map.permute(0, 2, 1).reshape(count, channels, side, side),
            size=self.image_size, mode="bilinear", align_corners=True,
        )

    def _maps(self, batch: torch.Tensor, categories: Sequence[str]) -> np.ndarray:
        maps = np.empty((len(batch), self.image_size, self.image_size), dtype=np.float32)
        with torch.no_grad(), torch.autocast("cuda", enabled=self._autocast):
            for index in range(len(batch)):
                text_features = self._text_features(str(categories[index]))
                _, patch_tokens = self._model.encode_image(
                    batch[index : index + 1], self.features
                )
                projected = self._linear(list(patch_tokens))
                total = 0
                for tokens in projected:
                    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
                    logits = self._upsample(100.0 * tokens @ text_features)
                    total = total + torch.softmax(logits, dim=1)[:, 1:].sum(dim=1)
                maps[index] = total[0].float().cpu().numpy()
        return maps

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("MultiADS received mismatched images and categories")
        batch = (images.to(self.device, dtype=torch.float32) - self._mean) / self._std
        maps = self._maps(batch, categories)
        return self._scores(maps), maps

    @staticmethod
    def _scores(maps: np.ndarray) -> np.ndarray:
        flat = maps.reshape(len(maps), -1)
        return np.partition(flat, -3, axis=1)[:, -3:].mean(axis=1).astype(np.float32)

    def close(self) -> None:
        self._model = None
        self._linear = None
        self._text_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@register_adapter("multiads_fewshot")
class MultiADSFewShotAdapter(MultiADSAdapter):
    """Official MultiADS few-shot inference (test.py --mode few_shot).

    Adds a retrieval map to the zero-shot one: per layer, the minimum cosine
    distance of each patch to the k normal shots' patches, upsampled and summed
    over layers 6/12/18/24. The image score is again the top-3 map mean.

    The official loader draws the shots with ``torch.randint`` (with
    replacement) from meta.json order after one global seed. As in the
    APRIL-GAN few-shot adapter, the draw is kept with replacement and seeded
    freshly per category over the sorted training images, and the file names
    used are recorded.
    """

    name = "multiads_fewshot"

    def __init__(
        self,
        *,
        k_shot: int = 4,
        shot_seed: int = 42,
        few_shot_features: Sequence[int] = (6, 12, 18, 24),
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        **kwargs,
    ) -> None:
        if int(k_shot) < 1:
            raise ValueError("MultiADS few-shot evaluation needs k_shot >= 1")
        target = str(kwargs["target_dataset"]).strip().lower()
        super().__init__(**kwargs)
        self.k_shot = int(k_shot)
        self.shot_seed = int(shot_seed)
        self.few_shot_features = [int(level) for level in few_shot_features]
        self._reference = NormalReference(
            dataset=target, mvtec_root=mvtec_root, visa_root=visa_root,
            image_size=self.image_size,
        )
        self._memory: dict[str, list[torch.Tensor]] = {}
        self._selection: dict[str, list[str]] = {}
        self._runtime_metadata.update({
            "adapter": self.name,
            "official_entrypoint_defaults": "MultiADS/test.py --mode few_shot",
            "mode": "few_shot",
            "k_shot": self.k_shot,
            "shot_seed": self.shot_seed,
            "few_shot_features": self.few_shot_features,
            "reference_selection": "torch.randint, with replacement",
        })

    def _patch_tokens(self, image: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad(), torch.autocast("cuda", enabled=self._autocast):
            _, tokens = self._model.encode_image(image, self.few_shot_features)
        return [F.normalize(token[0, 1:, :].float(), dim=-1) for token in tokens]

    def _ensure_memory(self, category: str) -> list[torch.Tensor]:
        cached = self._memory.get(category)
        if cached is not None:
            return cached
        candidates = self._reference.candidates(category)
        generator = torch.Generator().manual_seed(self.shot_seed)
        picked = torch.randint(0, len(candidates), (self.k_shot,), generator=generator).tolist()
        samples = [candidates[index] for index in picked]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.load(samples).to(self.device, dtype=torch.float32)
        references = (references - self._mean) / self._std
        per_shot = [self._patch_tokens(references[i : i + 1]) for i in range(len(references))]
        memory = [
            torch.cat([shot[layer] for shot in per_shot], dim=0)
            for layer in range(len(per_shot[0]))
        ]
        self._memory[category] = memory
        return memory

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("MultiADS received mismatched images and categories")
        batch = (images.to(self.device, dtype=torch.float32) - self._mean) / self._std
        maps = self._maps(batch, categories)
        for index, category in enumerate(categories):
            memory = self._ensure_memory(str(category))
            retrieval = 0
            for stored, patch in zip(memory, self._patch_tokens(batch[index : index + 1])):
                distance = (1.0 - stored @ patch.T).min(dim=0).values
                retrieval = retrieval + self._upsample(distance[None, :, None])[0, 0]
            maps[index] = maps[index] + retrieval.cpu().numpy()
        return self._scores(maps), maps

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data
