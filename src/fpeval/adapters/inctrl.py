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
