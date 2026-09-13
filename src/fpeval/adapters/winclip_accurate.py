"""WinCLIP as reproduced by zqhang/Accurate-WinCLIP-pytorch.

This is a second, independent WinCLIP port and is kept separate from the
``winclip`` adapter, which follows caoyunkang/WinClip. WinCLIP is training-free
and both ports read the same paper, but they do not agree, so they are two
target models rather than two spellings of one:

============ ===================================== ================================
              caoyunkang/WinClip (``winclip``)      zqhang (``winclip_accurate``)
============ ===================================== ================================
prompts       22 templates, no trailing period      21 templates with trailing
                                                    periods, one duplicated, so
                                                    147 normal / 84 abnormal
                                                    against 154 / 88
weights       ``laion400m_e32``                     ``laion400m_e31``, hardcoded in
                                                    ``CLIP_AD`` and not reachable
                                                    from ``--pretrained``
windows       scales (2, 3) over a 400-pixel grid   48- and 32-pixel kernels over a
                                                    16-pixel patch grid
image score   maximum of the anomaly map            softmax probability of the
                                                    abnormal text over the class
                                                    token, independent of the map
map fusion    textual/visual harmonic fusion        harmonic mean of the two window
                                                    scales and the image score
============ ===================================== ================================

The image score is the load-bearing difference. Taking the map maximum ties the
image decision to the segmentation, while the class-token probability is a
separate head, so the two ports can disagree on an image while agreeing on where
the defect is.

Zero-shot only: ``reproduce_WinCLIP.py`` takes ``--k_shot 0`` for the zero-shot
row of ``zero_shot.sh``, and the few-shot memory bank is never built here.
"""

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
# CLIP_AD hardcodes this; --pretrained is accepted and then ignored, so the
# published numbers are the e31 weights whatever zero_shot.sh passes.
PRETRAINED = "laion400m_e31"
# reproduce_WinCLIP.py fixes both: the ViT-B-16-plus-240 patch grid and the
# window kernels measured in pixels on it.
PATCH_SIZE = 16
LARGE_KERNEL = 48
MID_KERNEL = 32


def _import_official_repository(repository: str | Path):
    """Import ``reproduce_WinCLIP`` with the repository's vendored open_clip.

    The module mixes ``from src import open_clip`` with ``from open_clip import
    tokenizer``, so both the root and ``src`` have to be importable, and the
    vendored copy must win over any other repository's ``open_clip``.
    """
    root = Path(repository).expanduser().resolve()
    required = (root / "reproduce_WinCLIP.py", root / "src" / "open_clip",
                root / "dataset.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Accurate-WinCLIP-pytorch repository is incomplete at "
            f"{root}; missing {missing}"
        )
    roots = [str(root), str(root / "src")]
    for entry in roots:
        if entry in sys.path:
            sys.path.remove(entry)
    sys.path[:0] = roots
    importlib.invalidate_caches()
    # Several target repositories ship modules under these names; drop any whose
    # file does not live in this one so the vendored copies are the ones loaded.
    for name in list(sys.modules):
        head = name.split(".")[0]
        if head not in {"open_clip", "src", "dataset", "few_shot",
                        "reproduce_WinCLIP"}:
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", []) or [])
        origin = str(
            getattr(module, "__file__", "") or (locations[0] if locations else "")
        )
        if not origin.startswith(str(root)):
            sys.modules.pop(name, None)
    return importlib.import_module("reproduce_WinCLIP")


def _harmonic_aggregation(
    shape: tuple[int, int, int], similarity: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``reproduce_WinCLIP.harmonic_aggregation``, without its ``.cuda()`` calls.

    Each patch takes the harmonic mean of every window covering it. The upstream
    mask holds 1-based patch indices, which is why the comparison is ``idx + 1``.
    """
    batch, height, width = shape
    similarity = similarity.double()
    score = torch.zeros((batch, height * width), dtype=torch.float64,
                        device=similarity.device)
    mask = mask.T
    for index in range(height * width):
        covering = [bool(torch.isin(index + 1, window)) for window in mask]
        count = sum(covering)
        harmonic_sum = torch.sum(1.0 / similarity[:, covering], dim=-1)
        score[:, index] = count / harmonic_sum
    return score.reshape(batch, height, width)


@register_adapter("winclip-accurate")
@register_adapter("accurate-winclip")
@register_adapter("winclip_accurate")
class AccurateWinCLIPAdapter(ModelAdapter):
    """zqhang/Accurate-WinCLIP-pytorch zero-shot inference for MVTec AD and VisA.

    Follows ``reproduce_WinCLIP.py`` with the arguments ``zero_shot.sh`` passes:
    ``ViT-B-16-plus-240`` at 240 pixels with ``k_shot`` 0. The anomaly map is the
    harmonic mean of the two window scales and the image score, interpolated
    bilinearly to the model input; the image score is the class token's abnormal
    probability, so it is *not* the maximum of that map.

    Cohort images arrive at 518 pixels because that is the grid the perturbations
    are defined on, and the adapter resizes to 240 itself. The L-infinity budget
    therefore applies at 518 and is attenuated by the downsample - a property of
    evaluating a 240-pixel model against a 518-pixel perturbation, shared with
    the ``winclip`` adapter.
    """

    name = "winclip_accurate"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-B-16-plus-240",
        input_size: int = 240,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError(
                "Accurate-WinCLIP target_dataset must be 'mvtec' or 'visa'"
            )
        if backbone != "ViT-B-16-plus-240":
            raise ValueError(
                "zero_shot.sh evaluates ViT-B-16-plus-240; CLIP_AD's mask is "
                "built for its 240-pixel grid"
            )
        if input_size % PATCH_SIZE:
            raise ValueError(
                f"input_size must be a multiple of the {PATCH_SIZE}-pixel patch"
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.input_size = int(input_size)

        module = _import_official_repository(repository)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self._module = module
        self._model = module.CLIP_AD(backbone).to(self.device).eval()
        self._prompts = module.prompt_order()
        self._tokenize = module.tokenizer.tokenize
        # CLIP_AD builds its mask for a fixed 240-pixel image, so a different
        # input size needs its own; both are the official make_mask.
        self._scale = module.patch_scale((self.input_size, self.input_size))
        self._text: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

        self._runtime_metadata = {
            "adapter": self.name,
            "implementation": "zqhang/Accurate-WinCLIP-pytorch",
            "official_entrypoint_defaults": "reproduce_WinCLIP.py (zero_shot.sh)",
            "mode": "zero_shot",
            "target_dataset": target_key,
            "backbone": backbone,
            "pretrained_dataset": PRETRAINED,
            "model_input_size": self.input_size,
            "cohort_image_size": self.image_size,
            "patch_size": PATCH_SIZE,
            "window_kernels": [LARGE_KERNEL, MID_KERNEL],
            "prompt_templates": len(self._prompts.template_list),
            "k_shot": 0,
            "seed": int(seed),
            # No blur anywhere in reproduce_WinCLIP.py's map path.
            "official_gaussian_sigma": 0.0,
            "image_score": "abnormal softmax probability of the class token",
            "training_free": True,
        }

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _text_features(self, category: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``prepare_text_future`` for one class, cached.

        Upstream encodes every class up front; caching per category gives the
        same tensors while only paying for the classes actually evaluated.
        """
        cached = self._text.get(category)
        if cached is not None:
            return cached
        normal, abnormal = self._prompts.prompt(category)
        features = []
        for prompts in (normal, abnormal):
            tokens = self._tokenize(prompts).to(self.device)
            encoded = self._model.encode_text(tokens).float()
            features.append(encoded.mean(dim=0, keepdim=True))
        self._text[category] = (features[0], features[1])
        return self._text[category]

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError(
                "Accurate-WinCLIP received mismatched images and categories"
            )
        batch = images.to(self.device, dtype=torch.float32)
        batch = self._resize(
            batch,
            [self.input_size, self.input_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        batch = (batch - self._mean) / self._std

        grid = self.input_size // PATCH_SIZE
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.input_size, self.input_size),
                        dtype=np.float32)
        large_mask = self._scale.make_mask(
            kernel_size=LARGE_KERNEL, patch_size=PATCH_SIZE
        ).squeeze().to(self.device)
        mid_mask = self._scale.make_mask(
            kernel_size=MID_KERNEL, patch_size=PATCH_SIZE
        ).squeeze().to(self.device)

        with torch.no_grad():
            # One category at a time: the text features are per class and the
            # upstream loop indexes them by the batch's class ids.
            for category in sorted({str(name) for name in categories}):
                members = [i for i, name in enumerate(categories)
                           if str(name) == category]
                normal, abnormal = self._text_features(category)
                slab = batch[members]
                count = len(members)
                text = torch.cat((normal, abnormal), dim=1).permute(0, 2, 1)
                text = text.expand(count, -1, -1)

                tokens, class_tokens, patch_tokens = self._model.model.encode_image(
                    slab, [large_mask, mid_mask], proj=False
                )
                large_tokens, mid_tokens = tokens[0], tokens[1]

                # compute_score: abnormal probability of the class token.
                image_score = self._module.compute_score(class_tokens, text)[:, 0, 1]
                large_similarity = self._module.compute_sim(large_tokens, text)[:, :, 1]
                mid_similarity = self._module.compute_sim(mid_tokens, text)[:, :, 1]

                large_score = _harmonic_aggregation(
                    (count, grid, grid), large_similarity, large_mask
                )
                mid_score = _harmonic_aggregation(
                    (count, grid, grid), mid_similarity, mid_mask
                )
                fused = torch.nan_to_num(
                    3.0 / (1.0 / large_score
                           + 1.0 / mid_score
                           + 1.0 / image_score.unsqueeze(1).unsqueeze(1)),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                fused = torch.nn.functional.interpolate(
                    fused.unsqueeze(1).float(),
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                )[:, 0]
                for position, index in enumerate(members):
                    maps[index] = fused[position].detach().cpu().numpy()
                    # pr_sp is z0score, never the map maximum.
                    scores[index] = float(image_score[position])
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._module = None
        self._text.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
