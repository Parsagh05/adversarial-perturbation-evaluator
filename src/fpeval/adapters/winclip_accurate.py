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

``winclip_accurate`` is the zero-shot row (``zero_shot.sh``, ``--k_shot 0``);
``winclip_accurate_fewshot`` adds the few-shot memory of ``few_shot.sh``.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter
from .reference import NormalReference


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
        # Window kernels in pixels and the per-patch aggregation; hooks for a
        # subclass that runs at another size.
        self._large_kernel = LARGE_KERNEL
        self._mid_kernel = MID_KERNEL
        self._aggregate = _harmonic_aggregation

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
            "window_kernels": [self._large_kernel, self._mid_kernel],
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

    def _text_stack(self, category: str, count: int) -> torch.Tensor:
        """The ``[B, D, 2]`` text argument ``compute_score``/``compute_sim`` take.

        Upstream keeps one ``[1, D]`` mean per class, stacks them into
        ``[C, 1, D]`` and indexes that by the batch's class ids, so a batch of
        one class becomes ``[B, 1, D]``; concatenating normal with abnormal then
        gives ``[B, 2, D]`` and the permute ``[B, D, 2]``. The rows are repeated
        rather than expanded because both callers divide this tensor in place,
        which a broadcast view cannot support.
        """
        normal, abnormal = self._text_features(category)
        stacked = torch.stack((normal, abnormal), dim=1)      # [1, 2, D]
        return stacked.permute(0, 2, 1).repeat(count, 1, 1)   # [count, D, 2]

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
            kernel_size=self._large_kernel, patch_size=PATCH_SIZE
        ).squeeze().to(self.device)
        mid_mask = self._scale.make_mask(
            kernel_size=self._mid_kernel, patch_size=PATCH_SIZE
        ).squeeze().to(self.device)

        with torch.no_grad():
            # One category at a time: the text features are per class and the
            # upstream loop indexes them by the batch's class ids.
            for category in sorted({str(name) for name in categories}):
                members = [i for i, name in enumerate(categories)
                           if str(name) == category]
                slab = batch[members]
                count = len(members)

                tokens, class_tokens, patch_tokens = self._model.model.encode_image(
                    slab, [large_mask, mid_mask], proj=False
                )
                large_tokens, mid_tokens = tokens[0], tokens[1]

                # compute_score and compute_sim normalise their text argument
                # in place, and upstream hands each call its own ``torch.cat``,
                # so each gets a freshly materialised tensor here too.
                # compute_score: abnormal probability of the class token.
                image_score = self._module.compute_score(
                    class_tokens, self._text_stack(category, count)
                )[:, 0, 1]
                large_similarity = self._module.compute_sim(
                    large_tokens, self._text_stack(category, count)
                )[:, :, 1]
                mid_similarity = self._module.compute_sim(
                    mid_tokens, self._text_stack(category, count)
                )[:, :, 1]

                large_score = self._aggregate(
                    (count, grid, grid), large_similarity, large_mask
                )
                mid_score = self._aggregate(
                    (count, grid, grid), mid_similarity, mid_mask
                )
                fused = torch.nan_to_num(
                    3.0 / (1.0 / large_score
                           + 1.0 / mid_score
                           + 1.0 / image_score.unsqueeze(1).unsqueeze(1)),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                fused, image_score = self._add_few_shot(
                    category, large_tokens, mid_tokens, patch_tokens.unsqueeze(2),
                    large_mask, mid_mask, fused, image_score,
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

    def _add_few_shot(
        self, category, large_tokens, mid_tokens, patch_tokens,
        large_mask, mid_mask, fused, image_score,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-shot: the map and score stand as they are."""
        return fused, image_score

    def close(self) -> None:
        self._model = None
        self._module = None
        self._text.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# few_shot.sh evaluates these shot counts; reproduce_WinCLIP.py's argparse seed
# default, which few_shot.sh does not override.
FEW_SHOT_VALUES = (1, 2, 4)
FEW_SHOT_SEED = 10
# The order reproduce_WinCLIP.py builds the memory in (its obj_list). The k-shot
# draws come from one generator in this order, so a category's selection does
# not depend on which category the cohort reaches first.
OBJ_LIST = {
    "mvtec": ("carpet", "bottle", "hazelnut", "leather", "cable", "capsule", "grid",
              "pill", "transistor", "metal_nut", "screw", "toothbrush", "zipper",
              "tile", "wood"),
    "visa": ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"),
}


def _few_shot_distance(memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """``reproduce_WinCLIP.few_shot`` for one class.

    Half the cosine distance from each token to its nearest memory token, over
    every token of every reference image. ``tokens`` is ``[B, N, 1, D]``;
    ``memory`` is the references' tokens concatenated along dim 0, ``[kN, 1, D]``.
    """
    dims = memory.shape[-1]
    retrieve = memory.permute(2, 1, 0).reshape(dims, -1)            # [D, kN]
    retrieve = retrieve.unsqueeze(0).expand(len(tokens), -1, -1)    # [B, D, kN]
    similarity = torch.bmm(
        torch.nn.functional.normalize(tokens.squeeze(2), dim=-1),
        torch.nn.functional.normalize(retrieve, dim=1),
    )
    return 0.5 * torch.min(1.0 - similarity, dim=-1)[0]             # [B, N]


@register_adapter("winclip-accurate-fewshot")
@register_adapter("winclip_accurate_fewshot")
class AccurateWinCLIPFewShotAdapter(AccurateWinCLIPAdapter):
    """zqhang/Accurate-WinCLIP-pytorch few-shot inference (``few_shot.sh``).

    ``reproduce_WinCLIP.py`` with ``--k_shot`` 1, 2 or 4, ``ViT-B-16-plus-240``
    at 240 pixels and the default seed 10. Per class it keeps a memory of the
    k reference images' large-window, mid-window and patch tokens. Each test
    token's few-shot score is half the cosine distance to its nearest memory
    token; the window scores are harmonically aggregated onto the patch grid,
    the three averaged, and the result *added* to the zero-shot map. The image
    score becomes the mean of the zero-shot class-token probability and the
    maximum of that few-shot map.

    Reference selection follows ``dataset.py``: per class ``torch.randint`` draws
    k indices, with replacement, from the normal training images, one generator
    seeded with 10 and the classes taken in ``obj_list`` order. Two things
    cannot be byte-identical: upstream indexes ``meta.json``'s list (here the
    sorted training images) and its global generator has already been advanced
    by model construction. The selected file names are recorded in
    ``runtime_metadata``. References are always clean.
    """

    name = "winclip_accurate_fewshot"

    def __init__(
        self,
        *,
        k_shot: int = 1,
        shot_seed: int = FEW_SHOT_SEED,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        **kwargs,
    ) -> None:
        if int(k_shot) not in FEW_SHOT_VALUES:
            raise ValueError(f"few_shot.sh evaluates k_shot in {FEW_SHOT_VALUES}")
        kwargs.setdefault("seed", FEW_SHOT_SEED)
        super().__init__(**kwargs)
        target = str(kwargs["target_dataset"]).strip().lower()
        self.k_shot = int(k_shot)
        self.shot_seed = int(shot_seed)
        self._reference = NormalReference(
            dataset=target, mvtec_root=mvtec_root, visa_root=visa_root,
            image_size=self.image_size,
        )
        self._selection = self._draw_selection(target)
        self._memory: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._runtime_metadata.update({
            "adapter": self.name,
            "official_entrypoint_defaults": "reproduce_WinCLIP.py (few_shot.sh)",
            "mode": "few_shot",
            "k_shot": self.k_shot,
            "shot_seed": self.shot_seed,
            "reference_selection": (
                "torch.randint with replacement per class, one generator seeded "
                "with shot_seed, classes in obj_list order, over the sorted normal "
                "training images"
            ),
            "image_score": "mean of the class-token abnormal probability and "
                           "the maximum of the few-shot map",
        })

    def _draw_selection(self, target: str) -> dict[str, list]:
        generator = torch.Generator().manual_seed(self.shot_seed)
        known = set(self._reference.categories())
        selection = {}
        for category in OBJ_LIST[target]:
            if category not in known:
                continue
            candidates = self._reference.candidates(category)
            indices = torch.randint(
                0, len(candidates), (self.k_shot,), generator=generator
            )
            selection[category] = [candidates[int(i)] for i in indices]
        return selection

    def _category_memory(
        self, category: str, large_mask: torch.Tensor, mid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._memory.get(category)
        if cached is not None:
            return cached
        try:
            samples = self._selection[category]
        except KeyError:
            raise KeyError(f"No few-shot references drawn for {category!r}") from None
        references = self._reference.cached(category, samples).to(self.device)
        references = self._resize(
            references, [self.input_size, self.input_size],
            interpolation=self._bicubic, antialias=True,
        ).clamp(0, 1)
        references = (references - self._mean) / self._std
        tokens, _, patch_tokens = self._model.model.encode_image(
            references, [large_mask, mid_mask], proj=False
        )
        # memory() appends each image's tokens and concatenates along dim 0.
        memory = tuple(
            torch.cat(list(item), dim=0)
            for item in (tokens[0], tokens[1], patch_tokens.unsqueeze(2))
        )
        self._memory[category] = memory
        return memory

    def _add_few_shot(
        self, category, large_tokens, mid_tokens, patch_tokens,
        large_mask, mid_mask, fused, image_score,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count, grid = len(fused), fused.shape[-1]
        large_memory, mid_memory, patch_memory = self._category_memory(
            category, large_mask, mid_mask
        )
        large = _harmonic_aggregation(
            (count, grid, grid), _few_shot_distance(large_memory, large_tokens),
            large_mask,
        )
        mid = _harmonic_aggregation(
            (count, grid, grid), _few_shot_distance(mid_memory, mid_tokens),
            mid_mask,
        )
        patch = _few_shot_distance(patch_memory, patch_tokens).reshape(
            count, grid, grid
        ).double()
        few_shot_score = torch.nan_to_num(
            (large + mid + patch) / 3.0, nan=0.0, posinf=0.0, neginf=0.0
        )
        fused = fused + few_shot_score
        peak = torch.max(torch.max(few_shot_score, dim=1)[0], dim=1)[0]
        return fused, (image_score + peak) / 2.0

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = {
            category: NormalReference.describe(samples)
            for category, samples in sorted(self._selection.items())
        }
        return data

    def close(self) -> None:
        self._memory.clear()
        super().close()


def harmonic_aggregation_matrix(
    shape: tuple[int, int, int], similarity: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``_harmonic_aggregation`` as one matrix product.

    Each patch takes the harmonic mean of the windows covering it; ``cover[p, w]``
    is 1 where window ``w`` holds patch ``p``. The patch-by-window Python loop
    upstream costs patches x windows ``isin`` calls, which at a 32x32 grid is
    about twenty times the 15x15 one; the result is the same up to float64
    summation order.
    """

    batch, height, width = shape
    similarity = similarity.double()
    windows = mask.T.long() - 1  # [W, k*k], 0-based patch indices
    cover = torch.zeros(height * width, windows.shape[0], dtype=torch.float64,
                        device=similarity.device)
    columns = torch.arange(windows.shape[0], device=similarity.device)
    cover[windows, columns.unsqueeze(1).expand_as(windows)] = 1.0
    harmonic_sum = (1.0 / similarity) @ cover.T
    return (cover.sum(dim=1) / harmonic_sum).reshape(batch, height, width)


@register_adapter("winclip_accurate_512")
class AccurateWinCLIP512Adapter(AccurateWinCLIPAdapter):
    """Accurate-WinCLIP run at 512 pixels instead of its native 240.

    The perturbation is made at 518 pixels, so at 240 WinCLIP downsamples it
    with the image. 512 is the nearest size its 16-pixel patches tile (32x32).
    ``CLIP_AD`` builds its backbone without a size, so it is built here through
    the vendored factory's ``force_image_size``, which resizes the position
    embeddings on load. The window kernels are in pixels and scale with the
    image, 48/32 px at 240 to 96/64 px at 512, so each window covers the same
    share of the image; and the per-patch harmonic aggregation is computed as a
    matrix product with identical results. The method is training-free.
    """

    name = "winclip_accurate_512"

    def __init__(self, *, repository: str, input_size: int = 512, **kwargs: object) -> None:
        module = _import_official_repository(repository)
        factory = module.open_clip.create_customer_model_and_transforms

        def at_size(*args: object, **options: object):
            options.setdefault("force_image_size", int(input_size))
            return factory(*args, **options)

        # CLIP_AD passes no size, so its 512 backbone is built through this.
        module.open_clip.create_customer_model_and_transforms = at_size
        try:
            super().__init__(repository=repository, input_size=input_size, **kwargs)
        finally:
            module.open_clip.create_customer_model_and_transforms = factory
        scale = self.input_size / 240
        self._large_kernel = int(round(LARGE_KERNEL * scale / PATCH_SIZE)) * PATCH_SIZE
        self._mid_kernel = int(round(MID_KERNEL * scale / PATCH_SIZE)) * PATCH_SIZE
        self._aggregate = harmonic_aggregation_matrix
        self._runtime_metadata["window_kernels"] = [self._large_kernel, self._mid_kernel]
        self._runtime_metadata["resolution_variant"] = (
            f"{self.input_size} px instead of the native 240; backbone built with "
            "force_image_size; window kernels scaled with the image; harmonic "
            "aggregation as a matrix product"
        )
