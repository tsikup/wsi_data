"""The public API imports, and every name in `__all__` resolves."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "wsi_data",
    "wsi_data.augmentations",
    "wsi_data.datasets",
    "wsi_data.datasets.h5",
    "wsi_data.labels",
    "wsi_data.loaders",
    "wsi_data.normalization",
    "wsi_data.samplers",
    "wsi_data.tissue_segmentation",
    "wsi_data.transforms",
    "wsi_data.viz",
    "wsi_data.wholeslidedata",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_and_all_resolves(module_name):
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", []):
        assert hasattr(module, name), f"{module_name}.{name} is missing"


def test_weak_shuffling_names_are_exported_from_samplers():
    """Regression: `wsi_data.samplers` was empty, so `loaders` could not import.

    The empty `samplers/__init__.py` made `wsi_data.utils` (now
    `wsi_data.loaders`) raise ImportError on import, which also took down
    `wsi_data.augmentations`.
    """
    from wsi_data.samplers import (
        DistributedWeakShufflingBatchSampler,
        WeakShufflingBatchSampler,
        WeakShufflingSampler,
    )

    assert DistributedWeakShufflingBatchSampler is not None
    assert WeakShufflingBatchSampler is not None
    assert WeakShufflingSampler is not None


def test_loaders_and_augmentations_import():
    """Both modules were unimportable at HEAD before 1.0."""
    from wsi_data.augmentations import get_augmentor
    from wsi_data.loaders import weak_shuffling_h5_fast_loader

    assert callable(get_augmentor)
    assert callable(weak_shuffling_h5_fast_loader)


def test_package_is_typed():
    import wsi_data

    marker = importlib.resources.files(wsi_data) / "py.typed"
    assert marker.is_file()
