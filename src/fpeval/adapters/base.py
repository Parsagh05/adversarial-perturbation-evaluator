"""Small, stable extension point for target anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
import torch


class ModelAdapter(ABC):
    """Accept shared RGB tensors in ``[0, 1]`` and return anomaly outputs."""

    name: str

    def runtime_metadata(self) -> dict[str, object]:
        """Return resolved model settings recorded with evaluation outputs."""

        return {"adapter": self.name}

    @abstractmethod
    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return image scores ``[B]`` and low-resolution maps ``[B,H,W]``."""

    def postprocess_image_scores(
        self,
        scores: np.ndarray,
        map_mins: np.ndarray,
        map_maxs: np.ndarray,
        categories: Sequence[str],
        *,
        maps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply optional clean-cohort image-score aggregation.

        ``maps`` carries the postprocessed cohort maps for models whose image
        score needs a map statistic other than the min and max, such as the mean
        of the top-k pixels.
        """

        del map_mins, map_maxs, categories, maps
        return np.asarray(scores, dtype=np.float32)

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
        """Postprocess using normalization fitted on frozen clean predictions."""

        del (
            reference_scores,
            reference_map_mins,
            reference_map_maxs,
            reference_categories,
            reference_maps,
        )
        return self.postprocess_image_scores(
            scores, map_mins, map_maxs, categories, maps=maps
        )

    def postprocess_anomaly_maps(
        self, maps: np.ndarray, categories: Sequence[str]
    ) -> np.ndarray:
        """Apply optional clean-cohort map normalization."""

        del categories
        return np.asarray(maps, dtype=np.float32)

    def postprocess_anomaly_maps_with_reference(
        self,
        maps: np.ndarray,
        categories: Sequence[str],
        *,
        reference_maps: np.ndarray,
        reference_categories: Sequence[str],
    ) -> np.ndarray:
        """Normalize maps with parameters fitted on frozen clean maps."""

        del reference_maps, reference_categories
        return self.postprocess_anomaly_maps(maps, categories)

    @abstractmethod
    def close(self) -> None:
        """Release device memory and external resources."""


_T = TypeVar("_T", bound=type[ModelAdapter])
_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(name: str) -> Callable[[_T], _T]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name cannot be empty")

    def decorate(adapter: _T) -> _T:
        if key in _ADAPTERS:
            raise ValueError(f"Adapter is already registered: {key}")
        _ADAPTERS[key] = adapter
        return adapter

    return decorate


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def create_adapter(name: str, **kwargs: object) -> ModelAdapter:
    key = name.strip().lower()
    try:
        adapter = _ADAPTERS[key]
    except KeyError as error:
        raise ValueError(f"Unknown adapter {name!r}; available: {adapter_names()}") from error
    return adapter(**kwargs)
