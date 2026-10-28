"""InCTRL adapter matching the official few-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# The released checkpoints exist for these shot counts and no others.
SHOT_VALUES = (2, 4, 8)
# InCTRL is a *generalist* detector: it is trained on one dataset and applied to
# the others, so the checkpoint that scores MVTec is the one trained on VisA.
AUXILIARY_OF = {"mvtec": "visa", "visa": "mvtec"}

# Drive folder 1McmfxF8_H0BeRvcJ_poGIB-ATQCDDEIa, one zip per training dataset.
# Each holds checkpoints/{2,4,8}/checkpoint.pyth.
MODEL_ARCHIVES = {
    "mvtec": ("traind_on_mvtecad.zip", "1zEHsbbuUgBC4yuDu3g23wbUGmWmVyDRQ"),
    "visa": ("trained_on_visa.zip", "1uDOnyRAlwtDukfhglR8YxidsnlezBD6I"),
}
# Drive folder 1_RvmTqiCc4ZGa-Oq-uF7SOVotE1RW5QZ, the published few-shot prompts.
SAMPLE_ARCHIVES = {
    "mvtec": ("mvtecad.zip", "1DfK1zhDeC2VV1PsoQqmKjwBg_FQeGp55"),
    "visa": ("visa.zip", "1JwU7n1FoKAeryRpwj01WsFQVoevm50gX"),
}
# sha256 of each released archive, verified after download. Both model archives
# are 2.32 GB and hold one 835 MB checkpoint.pyth per shot count.
MODEL_DIGESTS: dict[str, str] = {
    "mvtec": "2f87ca61b7b3442e2046591cd0cb8470d086c084eea90b146c678cc2fe70c4e7",
    "visa": "e0ea17c87c150a077b9c25dd3f9168a6a139863b99dd83bc25538eb346c9dcbc",
}
SAMPLE_DIGESTS: dict[str, str] = {
    "mvtec": "462fc29f37daefbdbdfab0d6653994dab3d3fd19c46135de3fa5a059641ea760",
    "visa": "ff0802f7dd57de9a6c829247829ed5c9ee880848e49756059d7852fdc58b3440",
}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "open_clip" / "model.py",
        root / "open_clip" / "utils" / "checkpoint.py",
        root / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json",
        root / "engine_test.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official InCTRL repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # InCTRL vendors a patched fork of open_clip under that exact name, so any
    # other open_clip already imported has to go.
    for name in list(sys.modules):
        if not (
            name in {"open_clip", "datasets", "utils", "dataset", "models"}
            or name.startswith(("open_clip.", "datasets.", "utils.", "models."))
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
        importlib.import_module("open_clip.model"),
        importlib.import_module("open_clip.utils.checkpoint"),
    )


def record_forward(module, inputs: list, outputs: list) -> None:
    """Record what a module is called with, even when ``forward`` is called directly.

    ``register_forward_hook`` only fires through ``Module.__call__``. InCTRL's
    forward invokes ``self.diff_head.forward(...)``, so the bound method is
    replaced instead.
    """

    original = module.forward

    def recording(argument, *rest, **keywords):
        inputs.append(argument.detach())
        result = original(argument, *rest, **keywords)
        outputs.append(result.detach())
        return result

    module.forward = recording


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_archive(
    filename: str, file_id: str, expected: str | None, root: Path
) -> Path:
    """Download one Drive archive and check it against its pinned digest."""

    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and (expected is None or _sha256(destination) == expected):
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading InCTRL weights needs gdown; install the inctrl extra or "
            "pass existing paths for checkpoint and few_shot_dir."
        ) from error
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    gdown.download(id=file_id, output=str(temporary), quiet=True)
    if not temporary.is_file():
        raise RuntimeError(
            f"Google Drive did not return {filename}. Drive rate-limits popular "
            "files; retry later or download it manually and pass its path."
        )
    actual = _sha256(temporary)
    if expected is not None and actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"InCTRL checksum mismatch for {filename}: expected {expected}, "
            f"got {actual}"
        )
    temporary.replace(destination)
    return destination


def _extract_member(archive: Path, member: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(member)]
        if not names:
            raise KeyError(f"{archive.name} has no member ending in {member!r}")
        with bundle.open(sorted(names, key=len)[0]) as source:
            destination.write_bytes(source.read())
    return destination


def resolve_checkpoint(
    train_dataset: str,
    shot: int,
    *,
    checkpoint: str | None = None,
    download_root: str | Path | None = None,
) -> Path:
    """Return the released model trained on ``train_dataset`` for one shot count."""

    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"InCTRL checkpoint not found: {path}")
        return path
    key = str(train_dataset).strip().lower()
    if key not in MODEL_ARCHIVES:
        raise KeyError(
            f"InCTRL has no released model trained on {key!r}; "
            f"available: {sorted(MODEL_ARCHIVES)}"
        )
    if int(shot) not in SHOT_VALUES:
        raise ValueError(f"Official InCTRL checkpoints cover shot in {SHOT_VALUES}")
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "inctrl"
    )
    filename, file_id = MODEL_ARCHIVES[key]
    archive = _fetch_archive(filename, file_id, MODEL_DIGESTS.get(key), root)
    return _extract_member(
        archive,
        f"checkpoints/{int(shot)}/checkpoint.pyth",
        root / f"{key}_{int(shot)}shot_checkpoint.pyth",
    )


def resolve_few_shot_dir(
    target_dataset: str,
    shot: int,
    *,
    few_shot_dir: str | None = None,
    download_root: str | Path | None = None,
) -> Path:
    """Return the directory of published ``<category>.pt`` sample prompts."""

    if few_shot_dir:
        path = Path(few_shot_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"InCTRL few-shot directory not found: {path}")
        return path
    key = str(target_dataset).strip().lower()
    if key not in SAMPLE_ARCHIVES:
        raise KeyError(
            f"InCTRL publishes no few-shot samples for {key!r}; "
            f"available: {sorted(SAMPLE_ARCHIVES)}"
        )
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "inctrl"
    )
    filename, file_id = SAMPLE_ARCHIVES[key]
    archive = _fetch_archive(filename, file_id, SAMPLE_DIGESTS.get(key), root)
    extracted = root / f"fs_{key}"
    if not extracted.is_dir():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    candidates = [
        folder
        for folder in extracted.rglob(str(int(shot)))
        if folder.is_dir() and any(folder.glob("*.pt"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"{filename} has no {shot}-shot folder of .pt sample prompts under "
            f"{extracted}"
        )
    return sorted(candidates, key=lambda folder: len(folder.parts))[0]


@register_adapter("in-ctrl")
@register_adapter("inctrl")
class InCTRLAdapter(ModelAdapter):
    """Official InCTRL few-shot inference for MVTec AD and VisA.

    Follows test.py and engine_test.py at their defaults: the ``ViT-B-16-plus-240``
    configuration wrapped in the repository's ``InCTRL`` module, 240-pixel inputs,
    patch tokens taken from layers 7, 9 and 11, and the released ``checkpoint.pyth``
    for the requested shot count. InCTRL is a **generalist** detector - it is
    trained on one dataset and applied to the others - so the model that scores
    MVTec is the one trained on VisA and vice versa; ``train_dataset`` defaults to
    that pairing and is recorded in the metadata.

    The k normal prompts are the repository's own published ``<category>.pt``
    tensors rather than a re-drawn sample, so the shot selection is the paper's
    exactly. They are already preprocessed at 240 pixels and are never perturbed.

    **InCTRL publishes no pixel-level results.** ``engine_test.py`` reports only
    ``roc_auc_score`` and ``average_precision_score`` over image scores, and
    ``forward`` returns two image-level numbers. It does compute a per-patch
    residual internally - ``patch_ref_map``, the mean over the three layers of
    ``0.5 * (1 - cosine)`` to the nearest normal patch - and half of the final
    image score is precisely that map's maximum. This adapter exposes that 15x15
    map as the localization output, recovering it exactly through a forward hook
    on ``diff_head`` without touching the official code. It is the model's own
    map and it is tied to the model's own score, but it is **not a published
    result**: InCTRL's pixel AUROC, pixel F1 and AUPRO here have no paper number
    to be compared against, while the image-level metrics do.

    Two protocol notes. The official forward reshapes patch features with
    ``reshape(b, 3, 225, -1)`` from a layer-major tensor, which only lines up when
    ``b == 1``; the adapter therefore scores one image at a time, as AdaCLIP and
    VCP-CLIP do for the same reason. And the official transform is
    ``Resize(240)`` on the shorter side followed by ``CenterCrop(240)``, which for
    the square 518-pixel cohort is exactly a 518-to-240 resize with nothing
    cropped, so no perturbed pixel is discarded.
    """

    name = "inctrl"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        train_dataset: str | None = None,
        shot: int = 2,
        device: str = "cuda",
        image_size: int = 518,
        model_image_size: int = 240,
        model_config: str = "ViT-B-16-plus-240",
        out_layers: Sequence[int] = (7, 9, 11),
        checkpoint: str | None = None,
        few_shot_dir: str | None = None,
        download_root: str | None = None,
        rng_seed: int = 10,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in AUXILIARY_OF:
            raise ValueError("InCTRL target_dataset must be 'mvtec' or 'visa'")
        train_key = (train_dataset or AUXILIARY_OF[target_key]).strip().lower()
        if train_key == target_key:
            raise ValueError(
                "InCTRL is a generalist model evaluated on datasets it was not "
                f"trained on; training and scoring both on {target_key!r} would "
                "not be its published protocol."
            )
        if int(shot) not in SHOT_VALUES:
            raise ValueError(f"Official InCTRL evaluation uses shot in {SHOT_VALUES}")

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model_image_size = int(model_image_size)
        self.shot = int(shot)
        self.out_layers = [int(layer) for layer in out_layers]
        self.target_dataset = target_key
        self.train_dataset = train_key
        self.rng_seed = int(rng_seed)
        # 240 / 16 = 15, so the internal residual map is 225 patches.
        self.grid = self.model_image_size // 16

        package, model_module, checkpoint_module = _import_official_repository(
            repository
        )
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._bicubic = InterpolationMode.BICUBIC

        config_path = (
            Path(repository).expanduser().resolve()
            / "open_clip"
            / "model_configs"
            / f"{model_config}.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)

        np.random.seed(self.rng_seed)
        torch.manual_seed(self.rng_seed)
        # InCTRL.__init__ takes an ``args`` it never reads; test.py hands it the
        # whole cfg, so a namespace carrying the same fields is equivalent.
        arguments = SimpleNamespace(
            shot=self.shot,
            image_size=self.model_image_size,
            category=None,
            RNG_SEED=self.rng_seed,
        )
        self._model = model_module.InCTRL(
            arguments,
            settings["embed_dim"],
            settings["vision_cfg"],
            settings["text_cfg"],
            False,
            cast_dtype=model_module.get_cast_dtype("fp32"),
        ).to(self.device)

        self._checkpoint = resolve_checkpoint(
            self.train_dataset,
            self.shot,
            checkpoint=checkpoint,
            download_root=download_root,
        )
        checkpoint_module.load_checkpoint(
            str(self._checkpoint), self._model, False, None
        )
        self._model.eval()
        self._tokenizer = package.get_tokenizer(model_config)

        self._few_shot_dir = resolve_few_shot_dir(
            self.target_dataset,
            self.shot,
            few_shot_dir=few_shot_dir,
            download_root=download_root,
        )
        self._prompts: dict[str, list[torch.Tensor]] = {}

        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

        # diff_head's input is text_score + img_ref_score + patch_ref_map, the
        # first two being per-image scalars broadcast over the 225 patches.
        # The official forward calls ``self.diff_head.forward(...)`` directly
        # rather than the module, so register_forward_hook would never fire and
        # the method itself has to be wrapped.
        self._holistic: list[torch.Tensor] = []
        self._head_score: list[torch.Tensor] = []
        record_forward(self._model.diff_head, self._holistic, self._head_score)

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "InCTRL/test.py + engine_test.py",
            "mode": "few_shot",
            "target_dataset": self.target_dataset,
            "train_dataset": self.train_dataset,
            "generalist_protocol": "trained on one dataset, evaluated on the other",
            "model_config": model_config,
            "model_image_size": self.model_image_size,
            "out_layers": self.out_layers,
            "patch_grid": self.grid,
            "shot": self.shot,
            "checkpoint": self._checkpoint.name,
            "few_shot_dir": str(self._few_shot_dir),
            "few_shot_source": "official published <category>.pt sample prompts",
            "rng_seed": self.rng_seed,
            "batch_size_forced_to_one": True,
            "image_level_only_in_the_paper": True,
            "map_source": "patch_ref_map recovered from the diff_head input",
            "map_is_published_result": False,
            "gaussian_applied_inside_adapter": False,
            "cohort_image_size": self.image_size,
        }

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["prompt_categories"] = sorted(self._prompts)
        return data

    def _normal_list(self, category: str) -> list[torch.Tensor]:
        cached = self._prompts.get(category)
        if cached is not None:
            return cached
        path = self._few_shot_dir / f"{category}.pt"
        if not path.is_file():
            available = sorted(item.stem for item in self._few_shot_dir.glob("*.pt"))
            raise FileNotFoundError(
                f"InCTRL has no {self.shot}-shot prompt for {category!r} in "
                f"{self._few_shot_dir}; it holds {available}"
            )
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        tensors = [torch.as_tensor(item).to(self.device) for item in loaded]
        if len(tensors) != self.shot:
            raise ValueError(
                f"{path.name} holds {len(tensors)} prompts but shot is {self.shot}"
            )
        self._prompts[category] = tensors
        return tensors

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("InCTRL received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        # Resize(240) then CenterCrop(240) on a square input is just the resize.
        resized = self._resize(
            batch,
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic,
            antialias=True,
        ).clamp(0, 1)
        normalized = (resized - self._mean) / self._std

        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.grid, self.grid), dtype=np.float32)
        with torch.no_grad():
            for index in range(len(batch)):
                category = str(categories[index])
                self._holistic.clear()
                self._head_score.clear()
                final_score, _ = self._model(
                    self._tokenizer,
                    [normalized[index : index + 1]],
                    [category],
                    self._normal_list(category),
                )
                holistic = self._holistic[-1]            # [1, 225]
                head_score = self._head_score[-1]        # [1, 1]
                # final_score = (hl_score + fg_score) / 2, and fg_score is the
                # maximum of patch_ref_map, so the per-image constant that
                # holistic adds on top of that map follows exactly.
                foreground = 2.0 * final_score.reshape(-1) - head_score.reshape(-1)
                offset = holistic.max(dim=1).values - foreground
                patch_map = holistic - offset.unsqueeze(1)
                scores[index] = float(final_score.reshape(-1)[0])
                maps[index] = (
                    patch_map.reshape(self.grid, self.grid)
                    .float()
                    .cpu()
                    .numpy()
                )
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._prompts.clear()
        self._holistic.clear()
        self._head_score.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# The published normal prompts are 240-pixel tensors; a match must be this much
# closer than the next candidate image before it is accepted.
REFERENCE_MATCH_MARGIN = 4.0
# diff_head is Linear(225, 128) over the 15x15 patches of a 240-pixel input.
NATIVE_GRID = 15


@register_adapter("inctrl_512")
class InCTRL512Adapter(InCTRLAdapter):
    """InCTRL run at 512 pixels instead of its native 240.

    The perturbation is made at 518 pixels, so at 240 InCTRL downsamples it with
    the image. 512 is the nearest size its 16-pixel patches tile (32x32). Three
    things in InCTRL are tied to 240 and are handled here, leaving the rest of
    the official forward as it is:

    * The backbone is built at 512. ``load_checkpoint`` only loads weights whose
      shape matches, so it would leave the position embeddings at their random
      initialization; the checkpoint is loaded here instead, with InCTRL's own
      ``resize_pos_embed``, and every weight must load.
    * ``diff_head`` is ``Linear(225, 128)`` over the 15x15 holistic map, one
      weight per patch position. At 512 the map is 32x32, so it is area-pooled
      to 15x15 for that head only - each pooled cell covers the image area of
      one native patch. The patch residual map, its maximum (half the score)
      and the returned map stay at 32x32.
    * The published normal prompts exist only as 240-pixel tensors. Each is
      matched to its normal training image, which is then prepared at 512 the
      same way (shorter side resized, centre crop); an ambiguous match is an
      error.
    """

    name = "inctrl_512"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        shot: int = 4,
        model_image_size: int = 512,
        model_config: str = "ViT-B-16-plus-240",
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            repository=repository, target_dataset=target_dataset, shot=shot,
            model_image_size=model_image_size, model_config=model_config, **kwargs,
        )
        model_module = sys.modules["open_clip.model"]
        self._official = model_module
        config_path = (
            Path(repository).expanduser().resolve() / "open_clip" / "model_configs"
            / f"{model_config}.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
        vision = dict(settings["vision_cfg"], image_size=self.model_image_size)
        np.random.seed(self.rng_seed)
        torch.manual_seed(self.rng_seed)
        arguments = SimpleNamespace(
            shot=self.shot, image_size=self.model_image_size, category=None,
            RNG_SEED=self.rng_seed,
        )
        model = model_module.InCTRL(
            arguments, settings["embed_dim"], vision, settings["text_cfg"], False,
            cast_dtype=model_module.get_cast_dtype("fp32"),
        )
        checkpoint = torch.load(str(self._checkpoint), map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model_module.resize_pos_embed(state, model)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise ValueError(
                f"InCTRL at {self.model_image_size} px did not load {result.missing_keys}"
            )
        self._model = model.to(self.device).eval()
        if tuple(self._model.visual.grid_size) != (self.grid, self.grid):
            raise ValueError(
                f"InCTRL grid {self._model.visual.grid_size} at {self.model_image_size} px"
            )

        from .reference import NormalReference

        self._reference = NormalReference(
            dataset=self.target_dataset, mvtec_root=mvtec_root, visa_root=visa_root
        )
        self._matched: dict[str, list[str]] = {}
        self._runtime_metadata["patch_grid"] = self.grid
        self._runtime_metadata["resolution_variant"] = (
            f"{self.model_image_size} px instead of the native 240; position embeddings "
            "resized at load; holistic map area-pooled to 15x15 for diff_head only; "
            "published prompts matched to their images and prepared at this size"
        )

    def runtime_metadata(self) -> dict[str, object]:
        data = super().runtime_metadata()
        data["reference_images"] = dict(sorted(self._matched.items()))
        return data

    def _prepare_reference(self, path: Path, size: int) -> torch.Tensor:
        """Shorter side resized, centre crop, CLIP normalization, as test.py prepares images."""

        from PIL import Image
        from torchvision.transforms import functional as TF

        image = Image.open(path).convert("RGB")
        tensor = TF.to_tensor(image).to(self.device)
        tensor = TF.resize(tensor, size, interpolation=self._bicubic, antialias=True)
        tensor = TF.center_crop(tensor, [size, size]).clamp(0, 1)
        return ((tensor - self._mean[0]) / self._std[0]).float()

    def _normal_list(self, category: str) -> list[torch.Tensor]:
        cached = self._prompts.get(category)
        if cached is not None:
            return cached
        published = super()._normal_list(category)  # the k 240-pixel tensors
        self._prompts.pop(category, None)
        candidates = [sample.image_path for sample in self._reference.candidates(category)]
        small = 60
        with torch.no_grad():
            pool = torch.stack([
                torch.nn.functional.interpolate(
                    self._prepare_reference(path, 240)[None], size=(small, small), mode="area"
                )[0]
                for path in candidates
            ]).flatten(1)
            chosen = []
            for tensor in published:
                target = torch.nn.functional.interpolate(
                    tensor.float().reshape(1, 3, 240, 240), size=(small, small), mode="area"
                ).flatten(1)
                error = ((pool - target) ** 2).mean(dim=1)
                best, second = torch.topk(error, k=2, largest=False).values.tolist()
                if best * REFERENCE_MATCH_MARGIN > second:
                    raise ValueError(
                        f"InCTRL {category}: a published prompt matches no single normal "
                        f"training image clearly (best {best:.3g}, next {second:.3g})"
                    )
                chosen.append(candidates[int(error.argmin())])
        self._matched[category] = [Path(path).name for path in chosen]
        tensors = [self._prepare_reference(path, self.model_image_size) for path in chosen]
        self._prompts[category] = tensors
        return tensors

    def _forward(self, image: torch.Tensor, category: str) -> tuple[float, torch.Tensor]:
        """InCTRL.forward for one image, with the grid read off the tokens."""

        F = torch.nn.functional
        model = self._model
        normal_image = torch.stack(self._normal_list(category))
        shot = normal_image.shape[0]
        token, Fp_list, _ = model.encode_image(image, normalize=False)
        token_n, Fp_list_n, _ = model.encode_image(normal_image, normalize=False)
        Fp_list = torch.stack(Fp_list)[:, :, 1:, :]      # [3, 1, N, C]
        Fp_list_n = torch.stack(Fp_list_n)[:, :, 1:, :]  # [3, shot, N, C]
        patches = Fp_list.shape[2]
        Fp = Fp_list.reshape(3, patches, -1)
        Fp_n = Fp_list_n.reshape(3, shot * patches, -1)

        token_ad = model.adapter.forward(token)
        token_n = model.adapter.forward(token_n.reshape(1, shot, -1))
        token_ref = torch.mean(token_n, dim=1) - token_ad

        # 0.5 * (1 - cosine) to the nearest normal patch, mean over the layers.
        query = Fp / Fp.norm(dim=-1, keepdim=True)
        normal = Fp_n / Fp_n.norm(dim=-1, keepdim=True)
        distance = (0.5 * (1 - query @ normal.transpose(1, 2))).min(dim=2).values
        patch_ref_map = distance.mean(dim=0)[None]       # [1, N]
        fg_score = patch_ref_map.max(dim=1).values

        image_feature = token[0][None]
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        normal_texts, anomaly_texts = self._official.get_texts(category.replace("_", " "))
        texts = []
        for prompts in (normal_texts, anomaly_texts):
            features = model.encode_text(self._tokenizer(prompts).to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0, keepdim=True)
            texts.append(features / features.norm(dim=-1, keepdim=True))
        text_score = (100 * image_feature @ torch.cat(texts, dim=0).T).softmax(dim=-1)[0, 1]

        img_ref_score = model.diff_head_ref.forward(token_ref)
        holistic_map = text_score + img_ref_score + patch_ref_map
        side = int(round(patches ** 0.5))
        pooled = F.adaptive_avg_pool2d(holistic_map.view(1, 1, side, side), NATIVE_GRID)
        hl_score = model.diff_head.forward(pooled.view(1, NATIVE_GRID * NATIVE_GRID)).squeeze(1)
        final_score = (hl_score + fg_score) / 2
        return float(final_score.reshape(-1)[0]), patch_ref_map.view(side, side)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("InCTRL received mismatched images and categories")
        resized = self._resize(
            images.to(self.device, dtype=torch.float32),
            [self.model_image_size, self.model_image_size],
            interpolation=self._bicubic, antialias=True,
        ).clamp(0, 1)
        normalized = (resized - self._mean) / self._std
        scores = np.empty(len(images), dtype=np.float32)
        maps = np.empty((len(images), self.grid, self.grid), dtype=np.float32)
        with torch.no_grad():
            for index in range(len(images)):
                score, patch_map = self._forward(
                    normalized[index : index + 1], str(categories[index])
                )
                scores[index] = score
                maps[index] = patch_map.float().cpu().numpy()
        return scores, maps
