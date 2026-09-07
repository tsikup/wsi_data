"""The :class:`LabelDistribution` value type shared by datasets, samplers and plots.

A label distribution is just the unique label combinations and how often each
occurs, which ``numpy.unique`` computes in a single call. Representing that as
one small value type -- rather than returning a bare tuple, a pandas
``value_counts`` Series, or a rendered matplotlib figure depending on the
arguments, as the pre-1.0 code did -- means callers get the same object every
time and no plotting stack is pulled into a data path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LabelDistribution"]

#: Rank `labels` must have after being reshaped to `(n_samples, n_keys)`.
_LABELS_RANK = 2


@dataclass(frozen=True)
class LabelDistribution:
    """Counts of each distinct label combination in a dataset.

    Supports joint distributions over several label columns at once: with
    ``keys=("labels", "labels_group")`` each row of :attr:`values` is one
    observed ``(label, group)`` pair and :attr:`counts` says how often it
    occurs. Build instances with :meth:`from_labels`.

    Attributes:
        keys: The label column names described, in column order.
        values: ``(n_classes, n_keys)`` array of distinct label combinations,
            sorted lexicographically.
        counts: ``(n_classes,)`` array of occurrences of each combination.
        labels: ``(n_samples, n_keys)`` array of the raw per-sample labels.
        positions: ``(n_samples,)`` index into :attr:`values`/:attr:`counts`
            for each sample -- the class it belongs to.
    """

    keys: tuple[str, ...]
    values: np.ndarray
    counts: np.ndarray
    labels: np.ndarray
    positions: np.ndarray

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        keys: str | Sequence[str],
    ) -> LabelDistribution:
        """Build a distribution by counting distinct rows of ``labels``.

        One ``np.unique`` call yields the distinct combinations, their counts,
        and -- via ``return_inverse`` -- each sample's class index, so no
        second lookup pass is needed.

        Args:
            labels: ``(n_samples,)`` for a single key, or ``(n_samples, n_keys)``
                for a joint distribution.
            keys: The label column name, or names in column order.

        Returns:
            The corresponding distribution.

        Raises:
            ValueError: If ``labels``' shape does not match the number of keys.
        """
        key_tuple = (keys,) if isinstance(keys, str) else tuple(keys)
        array = np.asarray(labels)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != _LABELS_RANK or array.shape[1] != len(key_tuple):
            msg = (
                f"labels must be (n_samples,) or (n_samples, {len(key_tuple)}) "
                f"for keys {key_tuple}, got shape {array.shape}"
            )
            raise ValueError(msg)

        values, inverse, counts = np.unique(
            array, axis=0, return_inverse=True, return_counts=True
        )
        return cls(
            keys=key_tuple,
            values=values,
            counts=counts,
            labels=array,
            positions=np.ravel(inverse),
        )

    @property
    def n_classes(self) -> int:
        """Number of distinct label combinations."""
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples counted."""
        return int(self.labels.shape[0])

    @property
    def fractions(self) -> np.ndarray:
        """:attr:`counts` normalised to sum to one."""
        total = self.counts.sum()
        if total == 0:
            return np.zeros(self.n_classes, dtype=np.float64)
        return np.asarray(self.counts.astype(np.float64) / total)

    @property
    def class_names(self) -> list[str]:
        """A short display label per class, for plots and text output."""
        return [", ".join(str(v.item()) for v in row) for row in self.values]

    def as_mapping(self) -> dict[object, int]:
        """Return the distribution as a ``{label: count}`` mapping.

        Single-key distributions are keyed by the scalar label; joint
        distributions by a tuple of labels.
        """
        if len(self.keys) == 1:
            return {
                row[0].item(): int(count)
                for row, count in zip(self.values, self.counts, strict=True)
            }
        return {
            tuple(v.item() for v in row): int(count)
            for row, count in zip(self.values, self.counts, strict=True)
        }

    def describe(self, width: int = 32) -> str:
        """Render the distribution as a plain-text bar chart.

        A label histogram is a handful of numbers, so a text table conveys it
        as well as a rendered plot while needing no plotting dependency. Use
        :func:`wsi_data.viz.plot_label_distribution` when you want a figure.

        Args:
            width: Width in characters of the longest bar.

        Returns:
            A multi-line string with a header and one row per class.

        Example:
            >>> import numpy as np
            >>> dist = LabelDistribution.from_labels(np.array([0, 0, 0, 1]), "labels")
            >>> print(dist.describe(width=4))
            labels (4 samples, 2 classes)
                 0    3  75.0%  ####
                 1    1  25.0%  #
        """
        header = (
            f"{'/'.join(self.keys)} "
            f"({self.n_samples} samples, {self.n_classes} classes)"
        )
        if self.n_classes == 0:
            return header

        largest = int(self.counts.max())
        lines = [header]
        for name, count, fraction in zip(
            self.class_names, self.counts, self.fractions, strict=True
        ):
            bar = "#" * max(1, round(width * int(count) / largest))
            lines.append(f"{name:>6} {int(count):4d} {fraction:5.1%}  {bar}")
        return "\n".join(lines)
