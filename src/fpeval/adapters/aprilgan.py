"""APRIL-GAN adapter matching the official zero-shot evaluation defaults."""

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

# test_zero_shot.sh evaluates a dataset with the checkpoint that did not train
# on it. Both files ship inside the repository under exps/pretrained.
ZERO_SHOT_CHECKPOINT = {"mvtec": "visa_pretrained", "visa": "mvtec_pretrained"}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model.py",
        root / "prompt_ensemble.py",
        root / "open_clip",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official APRIL-GAN repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"model", "prompt_ensemble", "open_clip", "dataset", "utils"}
            or name.startswith(("open_clip.", "dataset."))
        ):
            continue
        module = sys.modules.get(name)
        module_path = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_path.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("open_clip"),
        importlib.import_module("model"),
        importlib.import_module("prompt_ensemble"),
    )


def resolve_checkpoint(repository: str | Path, checkpoint: str) -> Path:
    """Return the released checkpoint path; APRIL-GAN ships weights in-repo."""

    text = str(checkpoint).strip()
    direct = Path(text).expanduser()
    if direct.is_file():
        return direct.resolve()
    root = Path(repository).expanduser().resolve()
    folder = root / "exps" / "pretrained"
    path = folder / (text if text.endswith(".pth") else f"{text}.pth")
    if not path.is_file():
        available = (
            sorted(entry.name for entry in folder.glob("*.pth"))
            if folder.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"APRIL-GAN checkpoint not found: {path}. Pass a path or one of {available}."
        )
    return path


@register_adapter("april-gan")
@register_adapter("aprilgan")
class AprilGANAdapter(ModelAdapter):
    """Official APRIL-GAN zero-shot inference for MVTec AD and VisA."""

    name = "aprilgan"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        features: Sequence[int] = (6, 12, 18, 24),
        seed: int = 10,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("APRIL-GAN target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError(
                "Official APRIL-GAN zero-shot evaluation uses image_size=518"
            )
        if backbone != "ViT-L-14-336":
            raise ValueError(
                "Official APRIL-GAN zero-shot evaluation uses ViT-L-14-336"
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.features = tuple(int(level) for level in features)

        open_clip, model_module, prompt_module = _import_official_repository(repository)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        selected = checkpoint or ZERO_SHOT_CHECKPOINT[target_key]
        checkpoint_path = resolve_checkpoint(repository, selected)
        config_path = (
            Path(repository).expanduser().resolve()
            / "open_clip"
            / "model_configs"
            / f"{backbone}.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"APRIL-GAN model config not found: {config_path}")
        configs = json.loads(config_path.read_text(encoding="utf-8"))

        model, _, _ = open_clip.create_model_and_transforms(
            backbone, self.image_size, pretrained=pretrained
        )
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(backbone)
        self._prompt_module = prompt_module

        linear = model_module.LinearLayer(
            configs["vision_cfg"]["width"],
            configs["embed_dim"],
            len(self.features),
            backbone,
        ).to(self.device)
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        linear.load_state_dict(payload["trainable_linearlayer"])
        linear.eval().requires_grad_(False)
        self._linear = linear
        self._text_cache: dict[str, torch.Tensor] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "APRIL-GAN/test_zero_shot.sh",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": pretrained,
            "image_size": self.image_size,
            "features": [int(level) for level in self.features],
            "logit_scale": 100.0,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            "checkpoint_selection": selected,
            # test.py sums the per-layer maps and applies no gaussian filter.
            "official_gaussian_sigma": 0.0,
        }
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _text_features(self, category: str) -> torch.Tensor:
        cached = self._text_cache.get(category)
        if cached is None:
            with torch.no_grad():
                prompts = self._prompt_module.encode_text_with_prompt_ensemble(
                    self._model, [category], self._tokenizer, self.device
                )
            cached = prompts[category].detach()
            self._text_cache[category] = cached
        return cached

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("APRIL-GAN received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.image_size, self.image_size), dtype=np.float32)
        with torch.no_grad():
            for start in range(len(batch)):
                category = str(categories[start])
                image = batch[start : start + 1]
                image_features, patch_tokens = self._model.encode_image(
                    image, list(self.features)
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = torch.stack([self._text_features(category)], dim=0)

                text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
                scores[start] = float(text_probs[0][1])

                projected = self._linear(list(patch_tokens))
                layer_maps = []
                for tokens in projected:
                    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
                    logits = 100.0 * tokens @ text_features
                    count, length, _ = logits.shape
                    side = int(np.sqrt(length))
                    logits = F.interpolate(
                        logits.permute(0, 2, 1).view(count, 2, side, side),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    layer_maps.append(torch.softmax(logits, dim=1)[:, 1, :, :])
                maps[start] = torch.stack(layer_maps).sum(dim=0)[0].float().cpu().numpy()
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._linear = None
        self._text_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@register_adapter("april-gan-fewshot")
@register_adapter("aprilgan_fewshot")
class AprilGANFewShotAdapter(AprilGANAdapter):
    """Official APRIL-GAN few-shot inference for MVTec AD and VisA.

    Follows test_few_shot.sh, which differs from the zero-shot row in three ways
    beyond the memory bank itself. It overrides ``few_shot_features`` to
    6/12/18/24, where the argparse default is the few-shot ViT-B 3/6/9. It adds a
    retrieval map - the per-patch minimum cosine distance to the memory, summed
    over layers - to the zero-shot map. And it changes the image score to
    ``0.5 * (text probability + per-category min-max normalized map maximum)``,
    which the zero-shot row does not do; that normalization is fitted on the
    clean cohort and frozen for the adversarial pass.

    The official selection is ``torch.randint(0, n, (k,))`` under ``--seed 42``,
    which samples **with replacement**, so a shot may legitimately repeat. That
    is reproduced rather than corrected, and the drawn file names are recorded.
    """

    name = "aprilgan_fewshot"

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
            raise ValueError("APRIL-GAN few-shot evaluation needs k_shot >= 1")
        target = str(kwargs["target_dataset"]).strip().lower()
        super().__init__(**kwargs)
        self.k_shot = int(k_shot)
        self.shot_seed = int(shot_seed)
        self.few_shot_features = [int(level) for level in few_shot_features]
        self._reference = NormalReference(
            dataset=target,
            mvtec_root=mvtec_root,
            visa_root=visa_root,
            image_size=self.image_size,
        )
        self._memory: dict[str, list[torch.Tensor]] = {}
        self._selection: dict[str, list[str]] = {}
        self._runtime_metadata.update(
            {
                "adapter": self.name,
                "official_entrypoint_defaults": "APRIL-GAN/test_few_shot.sh",
                "mode": "few_shot",
                "k_shot": self.k_shot,
                "shot_seed": self.shot_seed,
                "few_shot_features": self.few_shot_features,
                "reference_selection": "torch.randint, with replacement",
                "image_score": "0.5 * (text probability + normalized map maximum)",
            }
        )

    def _ensure_memory(self, category: str) -> list[torch.Tensor]:
        cached = self._memory.get(category)
        if cached is not None:
            return cached
        candidates = self._reference.candidates(category)
        generator = torch.Generator().manual_seed(self.shot_seed)
        # torch.randint samples with replacement, exactly as the official loader.
        picked = torch.randint(
            0, len(candidates), (self.k_shot,), generator=generator
        ).tolist()
        samples = [candidates[index] for index in picked]
        self._selection[category] = NormalReference.describe(samples)
        references = self._reference.load(samples).to(self.device, dtype=torch.float32)
        references = (references - self._mean) / self._std
        per_shot: list[list[torch.Tensor]] = []
        with torch.no_grad():
            for index in range(len(references)):
                _, tokens = self._model.encode_image(
                    references[index : index + 1], self.few_shot_features
                )
                per_shot.append([token[0, 1:, :] for token in tokens])
        memory = [
            torch.cat([shot[layer] for shot in per_shot], dim=0)
            for layer in range(len(per_shot[0]))
        ]
        self._memory[category] = memory
        return memory

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, maps = super().predict(images, categories)
        batch = images.to(self.device, dtype=torch.float32)
        batch = (batch - self._mean) / self._std
        with torch.no_grad():
            for index, raw_category in enumerate(categories):
                memory = self._ensure_memory(str(raw_category))
                _, tokens = self._model.encode_image(
                    batch[index : index + 1], self.few_shot_features
                )
                retrieval = None
                for layer, token in enumerate(tokens):
                    patch = torch.nn.functional.normalize(token[0, 1:, :], dim=-1)
                    stored = torch.nn.functional.normalize(memory[layer], dim=-1)
                    distance = (1.0 - stored @ patch.T).min(dim=0).values
                    side = int(np.sqrt(distance.shape[0]))
                    layer_map = F.interpolate(
                        distance.reshape(1, 1, side, side).float(),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=True,
                    )[0, 0]
                    retrieval = layer_map if retrieval is None else retrieval + layer_map
                maps[index] = maps[index] + retrieval.cpu().numpy()
        return scores, maps

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
            maps=maps,
            reference_maps=maps,
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
        """Blend the text probability with the normalized map maximum."""

        del map_mins, reference_map_mins, reference_scores, maps, reference_maps
        scores = np.asarray(scores, dtype=np.float64)
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
            low = float(reference_peaks[matched].min())
            high = float(reference_peaks[matched].max())
            normalized = (
                np.zeros_like(peaks[selected])
                if high == low
                else (peaks[selected] - low) / (high - low)
            )
            result[selected] = 0.5 * (scores[selected] + normalized)
        return result.astype(np.float32)

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data
