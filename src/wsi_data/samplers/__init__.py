"""Samplers for whole-slide and HDF5-backed datasets.

Note:
    This module was empty before 1.0 while :mod:`wsi_data.loaders` (then
    ``wsi_data.utils``) imported three names from it, so importing that module
    -- and therefore :mod:`wsi_data.augmentations`, which imports from it in
    turn -- always failed with ``ImportError``. The weak-shuffling feature was
    unreachable as shipped.
"""

from wsi_data.samplers.balanced import (
    LabelDistributionDataset,
    get_weighted_random_sampler,
)
from wsi_data.samplers.weak import (
    DistributedWeakShufflingBatchSampler,
    WeakShufflingBatchSampler,
    WeakShufflingDistributedSampler,
    WeakShufflingSampler,
)

__all__ = [
    "DistributedWeakShufflingBatchSampler",
    "LabelDistributionDataset",
    "WeakShufflingBatchSampler",
    "WeakShufflingDistributedSampler",
    "WeakShufflingSampler",
    "get_weighted_random_sampler",
]
