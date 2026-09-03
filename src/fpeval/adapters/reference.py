"""Shared k-shot normal reference handling for the few-shot adapters.

Few-shot models score a cohort against a small set of normal training images.
That set is built once and never touched while the cohort is scored, so it does
not make a prediction depend on which cohort images came before it; but it does
add two settings every few-shot adapter has to pin and report, the shot count
and whatever selects which normal images are drawn.

The reference images are loaded through the same ``load_image`` the evaluator
uses for the cohort, so both arrive at the adapter on the identical grid and in
the identical range. They are always the clean originals: perturbing them would
change what "normal" means between the clean and the adversarial pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from ..data import Sample, discover_normal_reference, load_image


class NormalReference:
    """Per-category normal training images, loaded on first use."""

    def __init__(
        self,
        *,
        dataset: str,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        image_size: int = 518,
    ) -> None:
        self.dataset = dataset
        self.image_size = int(image_size)
        self._by_category = discover_normal_reference(
            dataset, mvtec_root=mvtec_root, visa_root=visa_root
        )
        self._cache: dict[str, torch.Tensor] = {}

    def categories(self) -> list[str]:
        return sorted(self._by_category)

    def candidates(self, category: str) -> list[Sample]:
        """Every normal training image of a category, in a stable order."""

        try:
            return self._by_category[category]
        except KeyError:
            raise KeyError(
                f"No {self.dataset} normal training images for {category!r}; "
                f"known categories are {self.categories()}"
            ) from None

    def load(self, samples: Sequence[Sample]) -> torch.Tensor:
        """Stack the selected references as ``[k,3,S,S]`` in ``[0, 1]``."""

        if not samples:
            raise ValueError("A few-shot reference set cannot be empty")
        return torch.stack([load_image(sample, self.image_size) for sample in samples])

    def cached(self, category: str, samples: Sequence[Sample]) -> torch.Tensor:
        if category not in self._cache:
            self._cache[category] = self.load(samples)
        return self._cache[category]

    @staticmethod
    def describe(samples: Sequence[Sample]) -> list[str]:
        """File names of the selection, for runtime_metadata."""

        return [Path(sample.image_path).name for sample in samples]
