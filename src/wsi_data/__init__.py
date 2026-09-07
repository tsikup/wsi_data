"""Dataset, sampler and augmentation library for whole-slide image deep learning.

The top level re-exports everything most callers need::

    from wsi_data import ImageDatasetHDF5, get_augmentor

Heavier or more specialised pieces stay in their own modules:
:mod:`wsi_data.wholeslidedata` for slide access and the ``wholeslidedata``
sampler stack, :mod:`wsi_data.tissue_segmentation` for CNN tissue
segmentation, and :mod:`wsi_data.viz` for plotting.

Note:
    Before 1.0 every ``__init__.py`` here was empty, so there was no declared
    public API and callers imported from deep module paths. Those paths have
    changed; see ``MIGRATION.md``.
"""

from wsi_data.augmentations import (
    HEDAugmentor,
    ReplaceBackgroundColor,
    get_augmentor,
    multires_additional_targets,
)
from wsi_data.datasets import (
    MISSING_LABEL,
    BlurrinessMode,
    DatasetMode,
    FakeDataset,
    FeatureDatasetHDF5,
    ImageDatasetHDF5,
    SlideTileDataset,
    TileDatasetHDF5,
)
from wsi_data.labels import LabelDistribution
from wsi_data.loaders import weak_shuffling_h5_fast_loader
from wsi_data.normalization import (
    calculate_mean_and_std,
    get_channels_sums_from_ndarray,
)
from wsi_data.samplers import (
    DistributedWeakShufflingBatchSampler,
    WeakShufflingBatchSampler,
    WeakShufflingDistributedSampler,
    WeakShufflingSampler,
    get_weighted_random_sampler,
)
from wsi_data.transforms import (
    crop_data,
    transform_image_and_mask,
    transform_multires_image_and_mask,
)

__version__ = "1.0.0"

__all__ = [
    "MISSING_LABEL",
    "BlurrinessMode",
    "DatasetMode",
    "DistributedWeakShufflingBatchSampler",
    "FakeDataset",
    "FeatureDatasetHDF5",
    "HEDAugmentor",
    "ImageDatasetHDF5",
    "LabelDistribution",
    "ReplaceBackgroundColor",
    "SlideTileDataset",
    "TileDatasetHDF5",
    "WeakShufflingBatchSampler",
    "WeakShufflingDistributedSampler",
    "WeakShufflingSampler",
    "__version__",
    "calculate_mean_and_std",
    "crop_data",
    "get_augmentor",
    "get_channels_sums_from_ndarray",
    "get_weighted_random_sampler",
    "multires_additional_targets",
    "transform_image_and_mask",
    "transform_multires_image_and_mask",
    "weak_shuffling_h5_fast_loader",
]
