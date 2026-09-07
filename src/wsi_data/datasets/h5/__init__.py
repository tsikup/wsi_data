"""Datasets backed by HDF5 files.

Note:
    These three modules were one 767-line ``datasets/h5_datasets.py`` before
    1.0.
"""

from wsi_data.datasets.h5.features import MISSING_LABEL, FeatureDatasetHDF5
from wsi_data.datasets.h5.images import DatasetMode, ImageDatasetHDF5
from wsi_data.datasets.h5.tiles import TileDatasetHDF5

__all__ = [
    "MISSING_LABEL",
    "DatasetMode",
    "FeatureDatasetHDF5",
    "ImageDatasetHDF5",
    "TileDatasetHDF5",
]
