"""Slide-level dataset over per-slide HDF5 feature bags.

One HDF5 file per slide, each holding the tile feature vectors for that slide
plus its slide-level labels. One dataset item is therefore one whole slide --
the shape multiple-instance-learning models consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from wsi_data.labels import LabelDistribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["MISSING_LABEL", "FeatureDatasetHDF5"]

logger = logging.getLogger(__name__)

#: Value used when a slide has no label. Matches PyTorch's default
#: ``ignore_index``, so such slides are skipped by the standard losses.
MISSING_LABEL = -100

#: Column names that mark a dataset as carrying survival targets.
_SURVIVAL_KEYS = frozenset({"survtime", "event", "status"})

#: Required feature column.
_PRIMARY_FEATURE_KEY = "features_target"


class FeatureDatasetHDF5(Dataset[dict[str, Any]]):
    """A dataset of per-slide HDF5 feature bags.

    Each item is a dict with ``features`` (a mapping of resolution name to a
    ``(n_tiles, n_features)`` tensor), ``labels``, ``labels_group``,
    ``slide_name``, ``coords`` and ``index``, plus ``survtime`` and ``event``
    for survival datasets.

    Note:
        The pre-1.0 ``load_ram`` parameter is gone. With ``load_ram=False`` the
        class returned ``h5py.Dataset`` handles created inside a ``with
        h5py.File(...)`` block that had already closed by the time they were
        returned, so reading one raised ``RuntimeError: Unable to
        synchronously get dataspace (invalid dataset identifier)``. Since one
        item is one whole slide's features, which the caller needs in full
        anyway, features are now always materialised.

    Args:
        data_dir: Directory of ``.h5`` files, one per slide.
        data_cols: Mapping of logical name to HDF5 dataset name. Must contain
            ``"features_target"``. Keys beginning with ``"labels"`` are
            treated as labels; ``"survtime"``, ``"event"`` and ``"status"``
            mark a survival dataset. For example::

                {"features_target": "features",
                 "features_context": "features_context",
                 "labels": "labels"}
        base_label: Subtracted from every label, for shifting 1-based labels
            to 0-based.

    Raises:
        ValueError: If ``data_cols`` lacks ``"features_target"``.
        NotADirectoryError: If ``data_dir`` is not a directory.
    """

    def __init__(
        self,
        data_dir: Path | str,
        data_cols: Mapping[str, str],
        base_label: int = 0,
    ) -> None:
        if not data_cols.get(_PRIMARY_FEATURE_KEY):
            msg = f"data_cols must contain a non-empty {_PRIMARY_FEATURE_KEY!r} entry"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            msg = f"{self.data_dir} is not a directory"
            raise NotADirectoryError(msg)

        self.data_cols = dict(data_cols)
        self.base_label = base_label
        self.survival = bool(_SURVIVAL_KEYS & set(self.data_cols))

        self.slides = sorted(self.data_dir.glob("*.h5"))
        if not self.slides:
            logger.warning("no HDF5 files found in %s", self.data_dir)

        self.features_shape: int | None = None
        self.labels_shape: int | None = None
        if self.slides:
            with h5py.File(self.slides[0], "r") as handle:
                self.features_shape = int(
                    handle[self.data_cols[_PRIMARY_FEATURE_KEY]].shape[1]
                )
                self.labels_shape = 1

    def __len__(self) -> int:
        return len(self.slides)

    @property
    def shape(self) -> list[int | None]:
        """``[n_slides, n_features]``."""
        return [len(self.slides), self.features_shape]

    @property
    def n_groups(self) -> int | None:
        """Number of distinct ``labels_group`` values, or ``None`` if absent."""
        if "labels_group" not in self.data_cols:
            return None
        return int(np.unique(self.get_labels("labels_group")).size)

    # ------------------------------------------------------------------ reading
    def _read_label(
        self, handle: h5py.File, key: str, *, offset: int = 0
    ) -> torch.Tensor:
        """Read a scalar label, falling back to :data:`MISSING_LABEL`.

        Note:
            The pre-1.0 code built its missing-label sentinel as
            ``np.array([-100], dtype=np.uint8)``. Under NumPy 1.x that silently
            wrapped to ``156`` -- a plausible-looking class index; under
            NumPy 2 (which this package requires) it raises ``OverflowError``.
            Labels are read as ``int64`` here, which represents both real
            labels and the sentinel exactly.
        """
        column = self.data_cols.get(key)
        if column is not None and column in handle:
            value = int(np.asarray(handle[column][0]).item()) - offset
        else:
            value = MISSING_LABEL
        return torch.tensor([value], dtype=torch.int64)

    def read_hdf5(self, h5_path: Path | str) -> dict[str, Any]:
        """Read one slide's features, labels and coordinates.

        Args:
            h5_path: Path to the slide's HDF5 file.

        Returns:
            The item dict, without its ``index`` entry.

        Raises:
            FileNotFoundError: If ``h5_path`` does not exist.
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            msg = f"{h5_path} does not exist"
            raise FileNotFoundError(msg)

        item: dict[str, Any] = {}
        with h5py.File(h5_path, "r") as handle:
            features = {
                key: torch.from_numpy(np.asarray(handle[column][...]))
                for key, column in self.data_cols.items()
                if not key.startswith("labels") and key not in _SURVIVAL_KEYS
            }
            # Downstream models address the target resolution as "features".
            features["features"] = features.pop(_PRIMARY_FEATURE_KEY)
            item["features"] = features

            item["labels"] = self._read_label(handle, "labels", offset=self.base_label)
            item["labels_group"] = self._read_label(handle, "labels_group")

            if self.survival:
                item["survtime"] = torch.tensor(
                    [float(np.asarray(handle[self.data_cols["survtime"]][0]).item())],
                    dtype=torch.float64,
                )
                event_key = next(
                    (k for k in ("event", "status") if k in self.data_cols), None
                )
                item["event"] = self._read_label(handle, event_key or "event")

            if "coords_x" in handle and "coords_y" in handle:
                coords = np.concatenate(
                    [handle["coords_x"][...], handle["coords_y"][...]], axis=1
                )
                item["coords"] = torch.from_numpy(coords.astype(np.float32))
            else:
                # Sentinel for "this file stores no tile coordinates"; callers
                # that need them should check for it.
                item["coords"] = -torch.ones(1, 2)

        item["slide_name"] = h5_path.name
        return item

    def get_item_on_slide_name(
        self, slide_name: str | Path, data_dir: Path | str | None = None
    ) -> dict[str, Any]:
        """Read one slide by file name rather than index."""
        base = self.data_dir if data_dir is None else Path(data_dir)
        return self.read_hdf5(base / slide_name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.read_hdf5(self.slides[index])
        item["index"] = index
        return item

    # ------------------------------------------------------------------- labels
    def get_labels(self, label_key: str = "labels") -> np.ndarray:
        """Read one label column across every slide.

        Args:
            label_key: Logical label column name.

        Returns:
            A ``(n_slides,)`` array.

        Raises:
            KeyError: If ``label_key`` is not in ``data_cols``.
        """
        if label_key not in self.data_cols:
            msg = f"{label_key!r} is not in data_cols ({sorted(self.data_cols)})"
            raise KeyError(msg)
        column = self.data_cols[label_key]
        labels = []
        for slide in self.slides:
            with h5py.File(slide, "r") as handle:
                labels.append(np.asarray(handle[column][0]).item())
        return np.array(labels)

    def get_label_distribution(
        self, label_key: str | Sequence[str] = "labels"
    ) -> LabelDistribution:
        """Count how often each label, or label combination, occurs.

        Note:
            The pre-1.0 signature returned one of three different things --
            a ``(np.unique, labels)`` tuple, a pandas ``value_counts`` Series,
            or a rendered seaborn figure -- depending on ``as_figure`` and
            whether ``label_key`` was a list. It now always returns a
            :class:`~wsi_data.labels.LabelDistribution`; use
            :meth:`~wsi_data.labels.LabelDistribution.describe` for a text
            summary or :func:`wsi_data.viz.plot_label_distribution` for a plot.

        Args:
            label_key: One label column, or several for a joint distribution.

        Returns:
            The distribution over ``label_key``.
        """
        if isinstance(label_key, str):
            return LabelDistribution.from_labels(self.get_labels(label_key), label_key)
        keys = tuple(label_key)
        stacked = np.stack([self.get_labels(key) for key in keys], axis=1)
        return LabelDistribution.from_labels(stacked, keys)

    # ------------------------------------------------------------------ collate
    @staticmethod
    def _collate(
        batch: Sequence[Mapping[str, Any]], keys: Sequence[str]
    ) -> dict[str, Any]:
        """Stack the scalar entries of a batch, leaving variable-length ones as lists.

        Feature bags and coordinates have a different number of tiles per
        slide, so they cannot be stacked and are returned as lists.
        """
        collated: dict[str, Any] = {
            "features": [item["features"] for item in batch],
            "slide_name": [item["slide_name"] for item in batch],
            "coords": [item["coords"] for item in batch],
            "index": torch.vstack([torch.tensor(item["index"]) for item in batch]),
        }
        for key in keys:
            collated[key] = torch.vstack([item[key] for item in batch])
        return collated

    @staticmethod
    def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate a classification batch."""
        return FeatureDatasetHDF5._collate(batch, ("labels",))

    @staticmethod
    def collate_fair(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate a batch carrying a group label, for fairness-aware training."""
        return FeatureDatasetHDF5._collate(batch, ("labels", "labels_group"))

    @staticmethod
    def surv_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate a survival batch."""
        return FeatureDatasetHDF5._collate(batch, ("labels", "event", "survtime"))
