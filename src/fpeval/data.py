"""Dataset discovery and shared image geometry for MVTec AD and VisA."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    dataset: str
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    label: int
    split: str = "test"

    @property
    def protocol_id(self) -> str:
        dataset_prefix = "visa/" if self.dataset == "visa" else ""
        return (
            f"{self.split}/{dataset_prefix}{self.category}/"
            f"{self.defect_type}/{self.image_path.stem}"
        )


def _images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_mvtec(root: str | Path) -> list[Sample]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec root not found: {root}")
    samples: list[Sample] = []
    for category_dir in sorted(path for path in root.iterdir() if (path / "test").is_dir()):
        for defect_dir in sorted(path for path in (category_dir / "test").iterdir() if path.is_dir()):
            normal = defect_dir.name.lower() == "good"
            for image_path in _images(defect_dir):
                mask_path: Path | None = None
                if not normal:
                    mask_dir = category_dir / "ground_truth" / defect_dir.name
                    masks = sorted(
                        item for item in mask_dir.glob(f"{image_path.stem}_mask.*")
                        if item.suffix.lower() in IMAGE_SUFFIXES
                    )
                    if not masks:
                        raise FileNotFoundError(f"Mask missing for {image_path}")
                    mask_path = masks[0]
                samples.append(Sample(
                    dataset="mvtec", category=category_dir.name,
                    defect_type=defect_dir.name, image_path=image_path,
                    mask_path=mask_path, label=0 if normal else 1,
                ))
    if not samples:
        raise ValueError(f"No MVTec test samples found below {root}")
    return samples


def _visa_manifest(root: Path) -> Path:
    for candidate in (
        root / "split_csv" / "1cls.csv",
        root / "split_csv" / "1cls.csv.csv",
        root / "1cls.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"VisA 1cls.csv was not found below {root}")


def _visa_file(root: Path, value: str) -> Path | None:
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else root / path


def discover_visa(root: str | Path) -> list[Sample]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VisA root not found: {root}")
    with _visa_manifest(root).open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            {str(key).strip().lower(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    required = {"object", "split", "label", "image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"VisA manifest must contain {sorted(required)}")
    samples: list[Sample] = []
    for row in rows:
        if row["split"].lower() != "test":
            continue
        normal = row["label"].lower() in {"normal", "good", "0"}
        image_path = _visa_file(root, row["image"])
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"VisA image missing: {image_path}")
        mask_path = None if normal else _visa_file(root, row.get("mask", ""))
        if not normal and (mask_path is None or not mask_path.is_file()):
            raise FileNotFoundError(f"VisA mask missing: {mask_path}")
        samples.append(Sample(
            dataset="visa", category=row["object"],
            defect_type="normal" if normal else "anomaly",
            image_path=image_path, mask_path=mask_path,
            label=0 if normal else 1,
        ))
    samples.sort(key=lambda item: item.protocol_id)
    if not samples:
        raise ValueError(f"No VisA test samples found below {root}")
    return samples


def discover_dataset(
    name: str, *, mvtec_root: str | None, visa_root: str | None
) -> list[Sample]:
    if name == "mvtec" and mvtec_root:
        return discover_mvtec(mvtec_root)
    if name == "visa" and visa_root:
        return discover_visa(visa_root)
    raise ValueError(f"Missing root or unsupported dataset: {name}")


def sample_index(samples: Iterable[Sample]) -> dict[str, Sample]:
    indexed: dict[str, Sample] = {}
    for sample in samples:
        if sample.protocol_id in indexed:
            raise ValueError(f"Duplicate protocol ID: {sample.protocol_id}")
        indexed[sample.protocol_id] = sample
    return indexed


def load_image(sample: Sample, size: int) -> torch.Tensor:
    array = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None] / 255.0
    return F.interpolate(
        tensor, size=(size, size), mode="bicubic",
        align_corners=False, antialias=True,
    )[0].clamp(0, 1)


def load_mask(sample: Sample, size: int) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    raw = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
    tensor = torch.from_numpy((raw > 0).astype(np.float32))[None, None]
    return F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0].numpy().astype(np.uint8)

