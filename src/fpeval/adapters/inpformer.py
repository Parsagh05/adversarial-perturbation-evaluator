"""INP-Former adapter matching the official few-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


# INP-Former is trained on ImageNet statistics, not CLIP's.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SHOT_VALUES = (1, 2, 4)
# The README's few-shot rows, one released model per (dataset, shot).
CHECKPOINTS = {
    ("mvtec", 1): ("inp_former_mvtec_1shot.pth", "1ymAywov3JFFVzwDpcdt9Tj_iFv-mk32c"),
    ("mvtec", 2): ("inp_former_mvtec_2shot.pth", "1K9X8-v1bSy_mgrbVSK0w6Fx525clSTtz"),
    ("mvtec", 4): ("inp_former_mvtec_4shot.pth", "15UtpeFveG2azUQmhogoET2HifEyIKSvX"),
    ("visa", 1): ("inp_former_visa_1shot.pth", "1mwpzXjLmjYLWFDx4dUF1yuErzL37K21p"),
    ("visa", 2): ("inp_former_visa_2shot.pth", "1_vlO4OSQSze095ddhkkyRWCOA2IRVLia"),
    ("visa", 4): ("inp_former_visa_4shot.pth", "1MFZcRNwALdPPv1Wemk5_1WLq76BINdky"),
}
# sha256 of each released model.pth. A key that is absent here is downloaded
# WITHOUT an integrity check, so fill the rest in before relying on them:
#   python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
DIGESTS: dict[tuple[str, int], str] = {
    ("mvtec", 4): "518d7a57587583251313edf81dfd44abd79ffd86c50f9bb8620af3716b862667",
    ("visa", 4): "930e80ceccb53c66d3db1dd0a4b3af0c435c6325b7da1e61e1dec563fa19834a",
}


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "utils.py",
        root / "models" / "uad.py",
        root / "models" / "vit_encoder.py",
        root / "INP_Former_Few_Shot.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official INP-Former repository is incomplete at {root}; missing "
            f"{missing}. On Windows the checkout needs core.longpaths=true, "
            "because saved_results/ holds very long directory names."
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "utils", "dataset", "datasets", "optimizers", "dinov2"}
            or name.startswith(("models.", "utils.", "dinov2.", "optimizers."))
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
        importlib.import_module("utils"),
        importlib.import_module("models.vit_encoder"),
        importlib.import_module("models.uad"),
        importlib.import_module("models.vision_transformer"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    target_dataset: str,
    shot: int,
    *,
    checkpoint: str | None = None,
    download_root: str | Path | None = None,
) -> Path:
    """Return the released model for one (dataset, shot) pair."""

    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"INP-Former checkpoint not found: {path}")
        return path
    key = (str(target_dataset).strip().lower(), int(shot))
    if key not in CHECKPOINTS:
        raise KeyError(
            f"INP-Former has no released few-shot model for {key}; "
            f"available: {sorted(CHECKPOINTS)}"
        )
    filename, file_id = CHECKPOINTS[key]
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "inpformer"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    expected = DIGESTS.get(key)
    if destination.is_file() and (expected is None or _sha256(destination) == expected):
        return destination
    try:
        import gdown
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading INP-Former weights needs gdown; install the inpformer "
            "extra or pass an existing checkpoint path."
        ) from error
    temporary = destination.with_suffix(".pth.tmp")
    gdown.download(id=file_id, output=str(temporary), quiet=True)
    if not temporary.is_file():
        raise RuntimeError(
            f"Google Drive did not return {filename}. Drive rate-limits popular "
            "files; retry later or download it manually and pass its path."
        )
    if expected is not None:
        actual = _sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"INP-Former checksum mismatch for {filename}: expected "
                f"{expected}, got {actual}"
            )
    temporary.replace(destination)
    return destination


@register_adapter("inp-former")
@register_adapter("inpformer")
class INPFormerAdapter(ModelAdapter):
    """Official INP-Former few-shot inference for MVTec AD and VisA.

    Follows INP_Former_Few_Shot.py at its defaults: the ``dinov2reg_vit_base_14``
    encoder, 448-pixel resize with a 392-pixel centre crop, 6 intrinsic normal
    prototypes, target layers 2-9 fused as two groups of four, and the
    ``evaluation_batch`` path with ``max_ratio=0.01`` and ``resize_mask=256``.
    The anomaly map is the mean of ``1 - cosine`` over the two fused groups, and
    the image score is the mean of the top one percent of blurred map pixels.

    This is the one few-shot adapter that needs **no runtime reference set**: the
    released weights were trained on k normal images per category of the target
    dataset, so the shots are baked into the checkpoint. That also means there is
    no cross-dataset variant - each checkpoint belongs to the dataset it was
    trained on - and the shot count selects the file rather than a sampling seed.

    Two protocol notes. Its ``get_gaussian_kernel(5, 4)`` blur runs inside
    ``evaluation_batch``, so the adapter applies it and the evaluator adds none.
    And the official transform resizes to 448 then centre-crops to 392, which
    discards the outer border; the cohort arrives at 518 because that is the grid
    the perturbations live on, so the adapter resizes 518 to 448 and crops, and
    the perturbation in the discarded border never reaches the model.
    """

    name = "inpformer"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        shot: int = 4,
        checkpoint: str | None = None,
        download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        encoder: str = "dinov2reg_vit_base_14",
        input_size: int = 448,
        crop_size: int = 392,
        inp_num: int = 6,
        resize_mask: int = 256,
        max_ratio: float = 0.01,
        seed: int = 1,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in {"mvtec", "visa"}:
            raise ValueError("INP-Former target_dataset must be 'mvtec' or 'visa'")
        if int(shot) not in SHOT_VALUES:
            raise ValueError(f"INP-Former releases few-shot models for {SHOT_VALUES}")
        if (int(input_size), int(crop_size)) != (448, 392):
            raise ValueError(
                "The released INP-Former weights use input_size=448 and crop_size=392"
            )
        if "base" not in encoder:
            raise ValueError(
                "The released INP-Former few-shot weights use dinov2reg_vit_base_14"
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.input_size = int(input_size)
        self.crop_size = int(crop_size)
        self.resize_mask = int(resize_mask)
        self.max_ratio = float(max_ratio)
        self.shot = int(shot)

        utils_module, encoder_module, uad_module, vit_module = (
            _import_official_repository(repository)
        )
        from torch import nn
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        self._resize = TF.resize
        self._crop = TF.center_crop
        # transforms.Resize defaults to bilinear, which is what the official
        # data transform uses.
        self._bilinear = InterpolationMode.BILINEAR
        self._cal_anomaly_maps = utils_module.cal_anomaly_maps
        utils_module.setup_seed(int(seed))

        checkpoint_path = resolve_checkpoint(
            target_key, self.shot, checkpoint=checkpoint, download_root=download_root
        )

        # Grouping-based reconstruction, exactly as the script sets it up.
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        embed_dim, num_heads = 768, 12

        backbone = encoder_module.load(encoder)
        bottleneck = nn.ModuleList(
            [vit_module.Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)]
        )
        prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(int(inp_num), embed_dim))]
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-8)
        aggregation = nn.ModuleList(
            [
                vit_module.Aggregation_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                    qkv_bias=True, norm_layer=norm_layer,
                )
            ]
        )
        decoder = nn.ModuleList(
            [
                vit_module.Prototype_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                    qkv_bias=True, norm_layer=norm_layer,
                )
                for _ in range(8)
            ]
        )
        model = uad_module.INP_Former(
            encoder=backbone,
            bottleneck=bottleneck,
            aggregation=aggregation,
            decoder=decoder,
            target_layers=target_layers,
            remove_class_token=True,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            prototype_token=prototypes,
        ).to(self.device)
        model.load_state_dict(
            torch.load(str(checkpoint_path), map_location=self.device), strict=True
        )
        model.eval().requires_grad_(False)
        self._model = model
        self._blur = utils_module.get_gaussian_kernel(kernel_size=5, sigma=4).to(
            self.device
        )

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "INP-Former/INP_Former_Few_Shot.py",
            "mode": "few_shot",
            "target_dataset": target_key,
            "encoder": encoder,
            "input_size": self.input_size,
            "crop_size": self.crop_size,
            "cohort_image_size": self.image_size,
            "inp_num": int(inp_num),
            "target_layers": target_layers,
            "fuse_layer_encoder": fuse_layer_encoder,
            "fuse_layer_decoder": fuse_layer_decoder,
            "shot": self.shot,
            "resize_mask": self.resize_mask,
            "max_ratio": self.max_ratio,
            "seed": int(seed),
            "checkpoint": checkpoint_path.name,
            # The shots live in the trained weights; nothing is fitted at runtime.
            "reference_set": "baked into the released checkpoint",
            "cross_dataset_variant": False,
            # get_gaussian_kernel(5, 4) runs inside evaluation_batch.
            "official_gaussian_sigma": 4.0,
            "official_gaussian_kernel_size": 5,
            "gaussian_applied_inside_adapter": True,
        }
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        del categories  # One released model covers every category of a dataset.
        batch = images.to(self.device, dtype=torch.float32)
        # Resize then centre-crop, matching get_data_transforms.
        batch = self._resize(
            batch, [self.input_size, self.input_size],
            interpolation=self._bilinear, antialias=True,
        ).clamp(0, 1)
        batch = self._crop(batch, [self.crop_size, self.crop_size])
        batch = (batch - self._mean) / self._std
        with torch.no_grad():
            output = self._model(batch)
            anomaly_map, _ = self._cal_anomaly_maps(
                output[0], output[1], batch.shape[-1]
            )
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map, size=self.resize_mask, mode="bilinear", align_corners=False
            )
            anomaly_map = self._blur(anomaly_map)
            flat = anomaly_map.flatten(1)
            keep = int(flat.shape[1] * self.max_ratio)
            scores = (
                torch.sort(flat, dim=1, descending=True)[0][:, :keep].mean(dim=1)
                if keep > 0
                else flat.max(dim=1)[0]
            )
        return (
            scores.float().cpu().numpy().astype(np.float32),
            anomaly_map[:, 0].float().cpu().numpy().astype(np.float32),
        )

    def close(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
