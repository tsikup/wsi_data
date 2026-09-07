"""Class-balanced sampling for imbalanced slide-level datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from wsi_data.labels import LabelDistribution

__all__ = ["LabelDistributionDataset", "get_weighted_random_sampler"]


class LabelDistributionDataset(Protocol):
    """A dataset that can report its label distribution.

    Structural type for exactly what :func:`get_weighted_random_sampler`
    needs, so it works with any object exposing this method rather than only
    the concrete dataset classes in this package.
    """

    def get_label_distribution(self, label_key: str = ...) -> LabelDistribution:
        """Return the dataset's label distribution."""
        ...


def get_weighted_random_sampler(
    dataset: LabelDistributionDataset,
    label_key: str = "labels",
    *,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a sampler that draws each class with equal probability.

    Each sample's weight is the inverse frequency of its class, so rare
    classes are drawn proportionally more often.

    Note:
        The pre-1.0 implementation indexed the per-class weight array by the
        raw *label value* (``weight[int(t)]``), which is only correct when the
        labels happen to be exactly ``0..K-1``. With labels such as ``{0, 2}``
        ``np.unique`` yields two counts and ``weight[2]`` raised
        ``IndexError``; with labels ``{1, 2}`` it silently assigned both
        classes the wrong weights. Weights are now indexed by
        :attr:`~wsi_data.labels.LabelDistribution.positions`, so any label
        values work.

    Args:
        dataset: Dataset exposing
            :meth:`~LabelDistributionDataset.get_label_distribution`.
        label_key: Which label column to balance on.
        replacement: Sample with replacement. With ``False`` balancing is only
            approximate, since a class cannot be drawn more often than it occurs.

    Returns:
        A :class:`torch.utils.data.WeightedRandomSampler` over the dataset.

    Raises:
        ValueError: If the dataset reports no labels.
    """
    distribution = dataset.get_label_distribution(label_key=label_key)
    if distribution.n_samples == 0 or distribution.n_classes == 0:
        msg = f"dataset reports no labels for label_key={label_key!r}"
        raise ValueError(msg)

    weight_per_class = 1.0 / distribution.counts.astype("float64")
    return WeightedRandomSampler(
        # `WeightedRandomSampler` type-hints `weights` as `Sequence[float]`; a
        # plain list satisfies that exactly, where a `Tensor` (fine at
        # runtime) would not.
        weights=weight_per_class[distribution.positions].tolist(),
        num_samples=distribution.n_samples,
        replacement=replacement,
    )
