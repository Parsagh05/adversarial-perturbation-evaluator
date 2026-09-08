"""AnoVL adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# test_zero_shot.sh runs MVTec through vl_test.py and VisA through vis_test.py.
# Besides selecting a different adapter and seed, vis_test.py tiles each resized
# image into two overlapping square views and recovers their predictions.
ADAPTER_MODULE = {"mvtec": "TextAdapter", "visa": "Adapter"}
OFFICIAL_SEED = {"mvtec": 111, "visa": 42}
# Both scripts take the 7th entry of the returned token list. The transformer
# appends two tensors per requested layer - the surgical v-v branch and the
# original - so four requested layers give eight entries and index 6 is the
# **v-v branch of the last requested layer** (12).
MAP_TOKEN_INDEX = 6


def _official_weight_reset(module: torch.nn.Module) -> None:
    """Reset the layers reset by AnoVL's official evaluation scripts.

    ``weight_reset`` is defined in ``vl_test.py`` and ``vis_test.py``, not in
    the repository's ``model.py``.  Keep the tiny callback here instead of
    importing either executable evaluation script, which would also import its
    dataset and metric stack as a side effect.
    """

    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        module.reset_parameters()


def _image_tiling(image: torch.Tensor) -> tuple[torch.Tensor, int]:
    """AnoVL ``vis_test.image_tiling``, generalized to preserve dtype/device."""

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("AnoVL VisA tiling expects one BCHW image")
    height, width = image.shape[-2:]
    if width < height:
        raise ValueError("AnoVL VisA tiling expects image width >= height")
    difference = width - height
    first = image[..., :height]
    second = image[..., difference:]
    return torch.cat((first, second), dim=0), difference


def _image_recover(tiles: torch.Tensor, difference: int) -> torch.Tensor:
    """AnoVL ``vis_test.image_recover`` with device-safe allocation."""

    if tiles.ndim != 4 or tiles.shape[0] != 2:
        raise ValueError("AnoVL VisA recovery expects exactly two NCHW tiles")
    _, channels, height, width = tiles.shape
    if not 0 <= difference <= width:
        raise ValueError("AnoVL VisA tile overlap is invalid")
    recovered = tiles.new_zeros((1, channels, height, width + difference))
    recovered[..., :difference] = tiles[:1, ..., :difference]
    recovered[..., difference:width] = (
        tiles[:1, ..., difference:] + tiles[1:2, ..., : width - difference]
    ) / 2
    recovered[..., width:] = tiles[1:2, ..., width - difference :]
    return recovered


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model.py",
        root / "utils.py",
        root / "prompt_ensemble.py",
        root / "open_clip",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official AnoVL repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"model", "utils", "prompt_ensemble", "open_clip", "dataset", "clip"}
            or name.startswith(("open_clip.", "clip.", "dataset."))
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
        importlib.import_module("open_clip"),
        importlib.import_module("model"),
        importlib.import_module("prompt_ensemble"),
        importlib.import_module("utils"),
    )


@register_adapter("ano-vl")
@register_adapter("anovl")
class AnoVLAdapter(ModelAdapter):
    """Official AnoVL zero-shot inference for MVTec AD and VisA.

    Follows test_zero_shot.sh: a ``ViT-B-16-plus-240`` backbone with
    ``laion400m_e32`` weights at **240 pixels**, token layers 3/6/9/12, the
    architecture surgery that swaps the last blocks for v-v attention, and
    ``--adapter True --epoch 5``. It is **training-free** - there is no released
    checkpoint, and the projection ``LinearLayer`` turns out to be CLIP's own
    ``ln_post @ proj`` rather than a learned head, so nothing is loaded beyond the
    open_clip backbone.

    Its one unusual property is a **per-image test-time adaptation**: for each
    image the class's adapter is re-initialized and then trained for five AdamW
    steps at 1e-3 on an entropy objective computed from that image alone. As in
    the upstream scripts, the category optimizer is retained (including its AdamW
    state) while the Linear/Conv weights are reset. The adapter reseeds before
    each image so an image sees the same initialization and stochastic feature
    augmentation on the clean and adversarial passes.

    Two indexing notes, both easy to misread. The token list holds **two** tensors
    per requested layer (the v-v branch and the original), so four layers give
    eight entries and the scripts' ``if layer != 6: continue`` selects the v-v
    branch of layer 12 - not a dead loop, which is what it looks like at four
    entries. And the image score is the CLS row of the token-wise similarity,
    averaged over the 22 augmented views, so it is read off the augmented batch
    rather than the plain image. VisA instead follows ``vis_test.py``: its two
    overlapping square tiles are used for both image scoring and adaptation, and
    their maps are blended back together before the loss/final prediction.

    The official loop applies no Gaussian filter - its call is commented out and
    a 3x3 average pool inside the map takes its place - so the evaluator adds
    none.
    """

    name = "anovl"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = 240,
        backbone: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        features: Sequence[int] = (3, 6, 9, 12),
        adapter_epochs: int = 5,
        learning_rate: float = 1e-3,
        adapter_module: str | None = None,
        seed: int | None = None,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ADAPTER_MODULE:
            raise ValueError("AnoVL target_dataset must be 'mvtec' or 'visa'")
        if backbone != "ViT-B-16-plus-240":
            raise ValueError("Official AnoVL evaluation uses ViT-B-16-plus-240")
        module_name = adapter_module or ADAPTER_MODULE[target_key]
        if module_name not in {"TextAdapter", "Adapter"}:
            raise ValueError("AnoVL adapter_module must be 'TextAdapter' or 'Adapter'")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.features = [int(level) for level in features]
        self.adapter_epochs = int(adapter_epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(OFFICIAL_SEED[target_key] if seed is None else seed)
        self.adapter_module = module_name
        self.target_dataset = target_key

        open_clip, model_module, prompt_module, utils_module = (
            _import_official_repository(repository)
        )
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        self._aug = utils_module.aug
        self._adapter_class = getattr(model_module, module_name)
        self._reseed()

        model, _, _ = open_clip.create_model_and_transforms(
            backbone, self.model_image_size, pretrained=pretrained
        )
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(backbone)
        self._encode_text = prompt_module.encode_text_with_prompt_ensemble

        config_path = (
            Path(repository).expanduser().resolve()
            / "open_clip"
            / "model_configs"
            / f"{backbone}.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"AnoVL model config not found: {config_path}")
        import json

        configs = json.loads(config_path.read_text(encoding="utf-8"))
        # LinearLayer holds no trained weights: for a ViT it applies the frozen
        # ln_post and proj of the backbone itself.
        self._linear = model_module.LinearLayer(
            configs["vision_cfg"]["width"],
            configs["embed_dim"],
            len(self.features),
            backbone,
            model,
        ).to(self.device)

        self._text: dict[str, torch.Tensor] = {}
        self._text_list: dict[str, torch.Tensor] = {}
        self._adapters: dict[str, torch.nn.Module] = {}
        self._optimizers: dict[str, torch.optim.Optimizer] = {}

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": (
                "AnoVL/test_zero_shot.sh "
                f"({'vl_test.py' if target_key == 'mvtec' else 'vis_test.py'})"
            ),
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained": pretrained,
            "model_image_size": self.model_image_size,
            "cohort_image_size": self.image_size,
            "features": self.features,
            "map_token_index": MAP_TOKEN_INDEX,
            "map_token_meaning": "v-v branch of the last requested layer",
            "adapter_module": module_name,
            "adapter_epochs": self.adapter_epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "training_free": True,
            "test_time_adaptation": "per image, adapter reset before each image",
            "rng_reset_per_image": True,
            "input_views": (
                "22-view MVTec augmentation"
                if target_key == "mvtec"
                else "two-tile VisA overlap and recovery"
            ),
            # The gaussian_filter call is commented out; a 3x3 average pool
            # inside the map replaces it.
            "official_gaussian_sigma": 0.0,
        }
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _reseed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _text_features(self, category: str) -> tuple[torch.Tensor, torch.Tensor]:
        if category not in self._text:
            with torch.no_grad():
                prompts, prompts_list = self._encode_text(
                    self._model, [category], self._tokenizer, self.device
                )
            self._text[category] = prompts[category].detach()
            self._text_list[category] = prompts_list[category].detach()
        return self._text[category], self._text_list[category]

    def _adapter_for(self, category: str) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        if category not in self._adapters:
            _, prompts_list = self._text_features(category)
            module = self._adapter_class(prompts_list).to(self.device)
            self._adapters[category] = module
            self._optimizers[category] = torch.optim.AdamW(
                module.parameters(), self.learning_rate
            )
        return self._adapters[category], self._optimizers[category]

    @staticmethod
    def _entropy_loss(prediction: torch.Tensor) -> torch.Tensor:
        """vl_test.loss_func: soft entropy plus a hard abnormal-target term."""

        soft = -prediction[0] * prediction[0].log()
        mask = torch.zeros(prediction[1:].shape, device=prediction.device)
        mask[..., 1] = 1
        hard = -mask * prediction[1:].log() - (1 - mask) * prediction[0].log()
        return soft.sum(-1).mean() + 0.5 * hard.sum(-1).mean()

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("AnoVL received mismatched images and categories")
        batch = self._resize(
            images.to(self.device, dtype=torch.float32),
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        batch = (batch - self._mean) / self._std
        size = self.model_image_size
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), size, size), dtype=np.float32)

        for index in range(len(batch)):
            category = str(categories[index])
            # Every image starts from the same RNG state, so the adapter reset,
            # the aug draw and mask_aug do not depend on evaluation order.
            self._reseed()
            image = batch[index : index + 1]
            text, _ = self._text_features(category)
            text_features = torch.stack([text], dim=0)
            module, optimizer = self._adapter_for(category)
            # The official vl_test.py and vis_test.py loops reset every Linear
            # and Conv2d layer immediately before adapting each image.
            module.apply(_official_weight_reset)

            with torch.no_grad():
                if self.target_dataset == "mvtec":
                    # vl_test.py scores 22 augmented views, but obtains patch
                    # tokens from the original image for adaptation.
                    views = self._aug(image.cpu()).to(self.device)
                    image_features, _ = self._model.encode_image(views, self.features)
                    _, patch_tokens = self._model.encode_image(image, self.features)
                    difference = 0
                else:
                    # vis_test.py scores and adapts the same two overlapping
                    # tiles. The evaluator supplies a common square attack
                    # tensor, for which the official overlap is complete.
                    views, difference = _image_tiling(image)
                    image_features, patch_tokens = self._model.encode_image(
                        views, self.features
                    )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                probability = (
                    image_features @ text_features[0]
                ).softmax(dim=-1).mean(dim=0)
                # Row 0 is the class token; column 1 is the abnormal state.
                scores[index] = float(probability[0][1])

                projected = self._linear(list(patch_tokens))
            tokens = projected[MAP_TOKEN_INDEX].detach()

            # Five steps of entropy minimization on this image alone.
            for _ in range(self.adapter_epochs):
                adapted = module(_reshape_tokens(tokens))
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)
                logits = adapted @ text_features
                if self.target_dataset == "visa":
                    logits = torch.nn.functional.interpolate(
                        logits.permute(0, 3, 1, 2),
                        size=size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    normal = _image_recover(logits[:2], difference)
                    abnormal = _image_recover(logits[2:], difference)
                    logits = torch.cat((normal, abnormal), dim=0).permute(0, 2, 3, 1)
                prediction = torch.softmax(logits, dim=-1)
                loss = self._entropy_loss(prediction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                adapted = module(_reshape_tokens(tokens), is_test=True)
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)
                anomaly = torch.softmax(adapted @ text_features, dim=-1)[..., 1]
                anomaly = anomaly.unsqueeze(1)
                anomaly = torch.nn.functional.pad(anomaly, (1, 1, 1, 1), "replicate")
                anomaly = torch.nn.functional.avg_pool2d(
                    anomaly, 3, stride=1, padding=0, count_include_pad=False
                )
                anomaly = torch.nn.functional.interpolate(
                    anomaly, size=size, mode="bilinear", align_corners=True
                )
                if self.target_dataset == "visa":
                    anomaly = _image_recover(anomaly, difference)
                maps[index] = anomaly[0, 0].float().cpu().numpy()
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._linear = None
        self._adapters.clear()
        self._optimizers.clear()
        self._text.clear()
        self._text_list.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _reshape_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """vl_test.resize_tokens: [B, N, C] to [B, sqrt(N), sqrt(N), C]."""

    count, length, channels = tokens.shape
    side = int(round(length ** 0.5))
    if side * side != length:
        raise ValueError("AnoVL returned a non-square patch grid")
    return tokens.view(count, side, side, channels)
