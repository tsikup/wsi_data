"""Synthetic dataset for smoke-testing training loops without real slides."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

__all__ = ["FakeDataset"]


class FakeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Yields random images and labels of fixed shape.

    Note:
        Two fixes relative to pre-1.0. Its ``input_shape``, ``output_shape``
        and ``classes`` defaults were mutable lists shared across every
        instance, and it returned a ``torch.Tensor`` image alongside a
        ``numpy`` label, which the default collate then handled
        inconsistently; both are now tensors. Its length was also hardcoded
        to ``1e6``, and is a parameter here.

    Args:
        input_shape: Shape of each generated image.
        output_shape: Shape of each generated label.
        classes: Inclusive range of label values to draw from.
        length: Number of items the dataset reports.
        seed: Seed for reproducible generation. ``None`` uses global randomness.

    Raises:
        ValueError: If ``classes`` is empty or ``length`` is negative.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...] = (3, 512, 512),
        output_shape: tuple[int, ...] = (1, 512, 512),
        classes: tuple[int, ...] = (0, 1),
        length: int = 1_000_000,
        seed: int | None = None,
    ) -> None:
        if not classes:
            msg = "classes must not be empty"
            raise ValueError(msg)
        if length < 0:
            msg = f"length must not be negative, got {length}"
            raise ValueError(msg)

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.classes = tuple(classes)
        self.length = length
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a random ``(image, label)`` pair; ``index`` is ignored."""
        image = torch.rand(self.input_shape, generator=self._generator)
        label = torch.randint(
            low=min(self.classes),
            high=max(self.classes) + 1,
            size=self.output_shape,
            generator=self._generator,
        )
        return image, label
