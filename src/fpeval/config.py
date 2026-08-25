"""Validated configuration for one evaluation invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationConfig:
    attacks_root: str
    output_root: str
    model_kwargs_by_target: dict[str, dict[str, Any]]
    model: str = "anomalyclip"
    mvtec_root: str | None = None
    visa_root: str | None = None
    targets: tuple[str, ...] = ("mvtec", "visa")
    scopes: tuple[str, ...] = (
        "per_dataset", "cross_dataset", "per_category", "per_image",
    )
    prompt_modes: tuple[str, ...] | None = None
    setup_ids: tuple[str, ...] | None = None
    source_datasets: tuple[str, ...] | None = None
    categories: tuple[str, ...] | None = None
    directions: tuple[str, ...] | None = None
    loss_modes: tuple[str, ...] | None = None
    loss_formulations: tuple[str, ...] | None = None
    device: str = "cuda"
    batch_size: int = 2
    image_size: int = 518
    gaussian_sigma: float = 4.0
    aupro_fpr_limit: float = 0.30
    aupro_thresholds: int = 200
    pixel_threshold_modes: tuple[str, ...] = (
        "fixed_0_5",
        "image_f1",
        "clean_pixel_f1",
    )
    pixel_success_min_flip_fraction: float = 0.50
    location_free_topk_fraction: float = 0.20
    verify_checksums: bool = True
    extraction_cache: str | None = None
    save_predictions: bool = False
    save_qualitative_samples: bool = True
    qualitative_threshold_modes: tuple[str, ...] | None = None
    samples_output_root: str | None = None
    write_separated_results: bool = True
    separated_output_root: str | None = None
    separated_samples_output_root: str | None = None
    create_output_archives: bool = True
    overwrite: bool = False
    max_conditions: int | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targets = tuple(self.targets)
        self.scopes = tuple(self.scopes)
        self.pixel_threshold_modes = tuple(self.pixel_threshold_modes)
        if self.qualitative_threshold_modes is None:
            self.qualitative_threshold_modes = self.pixel_threshold_modes
        else:
            self.qualitative_threshold_modes = tuple(self.qualitative_threshold_modes)
        for name in (
            "prompt_modes", "setup_ids", "source_datasets", "categories",
            "directions", "loss_modes", "loss_formulations",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, tuple(value))
        if not self.targets or set(self.targets) - {"mvtec", "visa"}:
            raise ValueError("targets must contain mvtec and/or visa")
        valid_scopes = {"per_dataset", "cross_dataset", "per_category", "per_image"}
        if not self.scopes or set(self.scopes) - valid_scopes:
            raise ValueError(f"scopes must be selected from {sorted(valid_scopes)}")
        valid_thresholds = {"fixed_0_5", "image_f1", "clean_pixel_f1"}
        if not self.pixel_threshold_modes or set(self.pixel_threshold_modes) - valid_thresholds:
            raise ValueError("Unknown or empty pixel_threshold_modes")
        if set(self.qualitative_threshold_modes) - set(self.pixel_threshold_modes):
            raise ValueError(
                "qualitative_threshold_modes must be selected pixel_threshold_modes"
            )
        if self.batch_size < 1 or self.image_size < 1:
            raise ValueError("batch_size and image_size must be positive")
        if not 0 < self.aupro_fpr_limit <= 1:
            raise ValueError("aupro_fpr_limit must be in (0,1]")
        if self.aupro_thresholds < 2:
            raise ValueError("aupro_thresholds must be at least 2")
        if not 0 <= self.pixel_success_min_flip_fraction <= 1:
            raise ValueError("pixel_success_min_flip_fraction must be in [0,1]")
        if not 0 < self.location_free_topk_fraction <= 1:
            raise ValueError("location_free_topk_fraction must be in (0,1]")
        if self.max_conditions is not None and self.max_conditions < 1:
            raise ValueError("max_conditions must be positive")
        if not Path(self.attacks_root).expanduser().exists():
            raise FileNotFoundError(f"attacks_root does not exist: {self.attacks_root}")
        if "mvtec" in self.targets and not self.mvtec_root:
            raise ValueError("mvtec_root is required when evaluating MVTec")
        if "visa" in self.targets and not self.visa_root:
            raise ValueError("visa_root is required when evaluating VisA")
        missing_models = set(self.targets) - set(self.model_kwargs_by_target)
        if missing_models:
            raise ValueError(f"Missing model kwargs for targets: {sorted(missing_models)}")
