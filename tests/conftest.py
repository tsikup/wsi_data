"""Shared fixtures: small on-disk HDF5 files matching each dataset's layout."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

N_TILES = 6
TILE = 8


@pytest.fixture
def rng():
    return np.random.default_rng(20260907)


@pytest.fixture
def tile_images(rng):
    return rng.integers(0, 256, (N_TILES, TILE, TILE, 3), dtype=np.uint8)


@pytest.fixture
def tile_masks(rng):
    return rng.integers(0, 3, (N_TILES, TILE, TILE), dtype=np.uint8)


@pytest.fixture
def singleres_seg_h5(tmp_path, tile_images, tile_masks):
    """One file with `x`/`y` for a single-resolution segmentation dataset."""
    path = tmp_path / "seg.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=tile_images)
        handle.create_dataset("y", data=tile_masks)
    return path


@pytest.fixture
def multires_seg_h5(tmp_path, tile_images, tile_masks):
    """One file with target and context images and masks."""
    path = tmp_path / "multi.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x_target", data=tile_images)
        handle.create_dataset("x_context", data=tile_images[::-1])
        handle.create_dataset("y_target", data=tile_masks)
        handle.create_dataset("y_context", data=tile_masks[::-1])
    return path


@pytest.fixture
def classification_h5(tmp_path, tile_images):
    """One file with images and one scalar label per tile."""
    path = tmp_path / "cls.h5"
    labels = np.array([0, 2, 0, 2, 2, 0], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=tile_images)
        handle.create_dataset("y", data=labels)
    return path, labels


def _write_feature_slide(path, *, features, label, group=None, survival=False):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("features_context", data=features * 2)
        if label is not None:
            handle.create_dataset("labels", data=np.array([label]))
        if group is not None:
            handle.create_dataset("labels_group", data=np.array([group]))
        if survival:
            handle.create_dataset("survtime", data=np.array([12.5]))
            handle.create_dataset("event", data=np.array([1]))
        handle.create_dataset("coords_x", data=np.arange(len(features)).reshape(-1, 1))
        handle.create_dataset(
            "coords_y", data=np.arange(len(features)).reshape(-1, 1) * 2
        )


@pytest.fixture
def feature_dir(tmp_path, rng):
    """A directory of per-slide feature bags with labels 0 and 2, plus groups."""
    directory = tmp_path / "features"
    directory.mkdir()
    labels = [0, 2, 0, 2]
    groups = [0, 1, 1, 0]
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        _write_feature_slide(
            directory / f"slide_{index}.h5",
            features=rng.random((3 + index, 16)).astype(np.float32),
            label=label,
            group=group,
        )
    return directory, np.array(labels), np.array(groups)


@pytest.fixture
def feature_dir_unlabelled(tmp_path, rng):
    """A feature bag with no `labels` dataset, to exercise the missing-label path."""
    directory = tmp_path / "unlabelled"
    directory.mkdir()
    _write_feature_slide(
        directory / "slide.h5",
        features=rng.random((4, 16)).astype(np.float32),
        label=None,
    )
    return directory
