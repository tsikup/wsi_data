"""Tile-level dataset over a single HDF5 file of images and labels.

Note:
    Replaces the pre-1.0 ``DatasetHDF5`` plus its three subclasses
    (``ImageOnlyDatasetHDF5``, ``SegmentationDatasetHDF5``,
    ``ClassificationDatasetHDF5``), which only forwarded constructor
    arguments. The ``segmentation``/``image_only`` boolean pair they set is
    now a single :data:`DatasetMode`, which removes the meaningless
    ``segmentation=True, image_only=True`` combination and the six scattered
    ``segmentation and not image_only`` conditions it required.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from wsi_data.labels import LabelDistribution
from wsi_data.transforms import (
    MULTIRES_PRIMARY_KEY,
    transform_image_and_mask,
    transform_multires_image_and_mask,
)

if TYPE_CHECKING:
    import albumentations as A

__all__ = ["DatasetMode", "ImageDatasetHDF5"]

logger = logging.getLogger(__name__)

#: What an item consists of. ``"image_only"`` yields just the image,
#: ``"classification"`` an ``(image, scalar_label)`` pair, and
#: ``"segmentation"`` an ``(image, mask)`` pair.
DatasetMode = Literal["classification", "segmentation", "image_only"]

_SINGLERES_KEY = "images"
_LABELS_KEY = "labels"

#: Rank of a mask with no channel axis, e.g. `(H, W)` rather than `(H, W, 1)`.
_UNCHANNELLED_RANK = 2


def _ensure_channel_axis(mask: np.ndarray) -> np.ndarray:
    """Add a trailing channel axis if `mask` is bare `(H, W)`.

    Albumentations expects a trailing channel axis on every mask.
    """
    return mask[:, :, None] if mask.ndim == _UNCHANNELLED_RANK else mask


class ImageDatasetHDF5(Dataset[Any]):
    """A dataset of tiles stored in one HDF5 file.

    Multi-resolution mode is inferred from ``data_cols``: more than one
    non-label column means each item is a resolution-keyed dict of tensors
    rather than a single tensor.

    Args:
        data_dir: Directory holding the file.
        data_name: File name within ``data_dir``.
        data_cols: Mapping of logical name to HDF5 dataset name. Single
            resolution uses the key ``"images"``; multi-resolution uses
            ``"target"`` plus one key per extra resolution. For
            ``mode="segmentation"`` the ``"labels"`` entry must itself be a
            mapping of resolution name to mask dataset name; for
            ``mode="classification"`` it is a single dataset name.
        transform: Albumentations pipeline applied jointly to image and mask.
            For multi-resolution input it must be built with
            ``additional_targets`` -- see
            :func:`~wsi_data.augmentations.multires_additional_targets`.
        mask_transform: Image-only pipeline applied to each mask afterwards.
        mode: What each item contains. See :data:`DatasetMode`.
        channels_last: Return tensors as ``(H, W, C)`` instead of ``(C, H, W)``.
        merge_labels: Mask value remapping applied last, as ``{from: to}``.
            Used to collapse several annotation classes into one.

    Raises:
        NotADirectoryError: If ``data_dir`` is not a directory.
        FileNotFoundError: If the file does not exist.
        ValueError: If ``data_cols`` does not match ``mode``.
    """

    def __init__(
        self,
        data_dir: Path | str,
        data_name: str,
        data_cols: Mapping[str, Any],
        transform: A.Compose | None = None,
        mask_transform: A.Compose | None = None,
        *,
        mode: DatasetMode = "segmentation",
        channels_last: bool = False,
        merge_labels: Mapping[int, int] | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            msg = f"{data_dir} is not a directory"
            raise NotADirectoryError(msg)
        self.h5_path = data_dir / data_name
        if not self.h5_path.exists():
            msg = f"{self.h5_path} does not exist"
            raise FileNotFoundError(msg)

        self.data_cols = dict(data_cols)
        self.mode: DatasetMode = mode
        self.channels_last = channels_last
        # `None` rather than a `{}` default: a mutable default argument is
        # shared by every instance that omits it.
        self.merge_labels = dict(merge_labels or {})
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_keys = [key for key in self.data_cols if key != _LABELS_KEY]
        self.multiresolution = len(self.image_keys) > 1
        self.base_key = MULTIRES_PRIMARY_KEY if self.multiresolution else _SINGLERES_KEY
        if self.base_key not in self.data_cols:
            msg = (
                f"data_cols must contain {self.base_key!r} for a "
                f"{'multi' if self.multiresolution else 'single'}-resolution dataset, "
                f"got {sorted(self.data_cols)}"
            )
            raise ValueError(msg)

        if self.mode != "image_only":
            label_cols = self.data_cols.get(_LABELS_KEY)
            if label_cols is None:
                msg = f"data_cols must contain {_LABELS_KEY!r} for mode={mode!r}"
                raise ValueError(msg)
            if self.mode == "segmentation" and not isinstance(label_cols, Mapping):
                msg = (
                    "mode='segmentation' needs data_cols['labels'] to be a mapping "
                    f"of resolution name to mask dataset name, got {label_cols!r}"
                )
                raise ValueError(msg)
            if self.mode == "classification" and isinstance(label_cols, Mapping):
                msg = (
                    "mode='classification' needs data_cols['labels'] to be a single "
                    f"dataset name, got {label_cols!r}"
                )
                raise ValueError(msg)

        self._handle: h5py.File | None = None
        self._images: Any = None
        self._labels: Any = None

        with h5py.File(self.h5_path, "r") as handle:
            base = handle[self.data_cols[self.base_key]]
            self.dataset_size = int(base.shape[0])
            self.image_shape = tuple(base.shape[1:])
            self.labels_shape: Any = None
            if self.mode == "segmentation":
                self.labels_shape = tuple(
                    handle[self.data_cols[_LABELS_KEY][self.base_key]].shape[1:]
                )
            elif self.mode == "classification":
                self.labels_shape = 1

    def __len__(self) -> int:
        return self.dataset_size

    # -------------------------------------------------------------- file handle
    def open_hdf5(self) -> None:
        """Open the file and bind the image and label datasets."""
        self._handle = h5py.File(self.h5_path, "r")
        if self.multiresolution:
            self._images = {
                key: self._handle[self.data_cols[key]] for key in self.image_keys
            }
        else:
            self._images = self._handle[self.data_cols[self.base_key]]

        if self.mode == "segmentation":
            label_cols = self.data_cols[_LABELS_KEY]
            self._labels = {
                key: self._handle[label_cols[key]]
                for key in self.image_keys
                if key in label_cols
            }
        elif self.mode == "classification":
            self._labels = self._handle[self.data_cols[_LABELS_KEY]]

    def close(self) -> None:
        """Close the file if it is open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._images = None
            self._labels = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open HDF5 handle so the dataset can be sent to workers.

        An ``h5py.File`` cannot be pickled. Each ``DataLoader`` worker reopens
        the file on its first item.
        """
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_images"] = None
        state["_labels"] = None
        return state

    def _ensure_open(self) -> None:
        if self._handle is None:
            self.open_hdf5()

    # ------------------------------------------------------------------ reading
    def _read_image(self, index: int) -> Any:
        if self.multiresolution:
            return {key: self._images[key][index] for key in self.image_keys}
        return self._images[index]

    def _read_mask(self, index: int) -> Any:
        if self.multiresolution:
            return {
                key: _ensure_channel_axis(np.asarray(value[index]))
                for key, value in self._labels.items()
            }
        # Single resolution: `open_hdf5` still stores `_labels` as a
        # `{resolution_name: dataset}` dict (there is exactly one resolution,
        # `self.base_key`), for a uniform shape across both modes.
        return _ensure_channel_axis(np.asarray(self._labels[self.base_key][index]))

    def _read_scalar_label(self, index: int) -> torch.Tensor:
        """Read a scalar label, broadcasting a single slide-level label if needed.

        Note:
            The pre-1.0 code wrapped this in ``except IndexError: label =
            self.labels[0]``, which silently substituted the *first* label
            whenever indexing failed -- so a genuinely truncated label array
            mislabelled every tile past its end instead of failing. A
            length-one label array is a real case (one slide-level label for
            every tile) and is broadcast explicitly; any other mismatch now
            raises.
        """
        length = len(self._labels)
        if length == self.dataset_size:
            value = self._labels[index]
        elif length == 1:
            value = self._labels[0]
        else:
            msg = (
                f"label column has {length} entries but the dataset has "
                f"{self.dataset_size} images; expected either a matching count "
                f"or exactly one label to broadcast"
            )
            raise ValueError(msg)
        return torch.as_tensor(np.asarray(value).reshape(-1))

    # ------------------------------------------------------------------- labels
    def get_label_distribution(self, label_key: str = _LABELS_KEY) -> LabelDistribution:
        """Count how often each label value occurs.

        For ``mode="segmentation"`` this counts *pixel* values across every
        mask at the base resolution.

        Note:
            The pre-1.0 version built a per-resolution dict for segmentation
            datasets and then unconditionally overwrote it with
            ``f[self.data_cols["labels"]][...]``. Since ``data_cols["labels"]``
            is a *mapping* in segmentation mode, indexing the HDF5 file with it
            raised ``TypeError`` -- so this was broken for every segmentation
            dataset, and the dict it had just built was dead code.

        Args:
            label_key: Accepted for interface compatibility with
                :class:`~wsi_data.datasets.h5.features.FeatureDatasetHDF5`;
                must be ``"labels"``.

        Returns:
            The distribution over label values.

        Raises:
            ValueError: If the dataset has no labels, or ``label_key`` is not
                ``"labels"``.
        """
        if label_key != _LABELS_KEY:
            msg = f"this dataset only has a {_LABELS_KEY!r} column, got {label_key!r}"
            raise ValueError(msg)
        if self.mode == "image_only":
            msg = "mode='image_only' datasets have no labels"
            raise ValueError(msg)

        label_cols = self.data_cols[_LABELS_KEY]
        column = (
            label_cols[self.base_key] if isinstance(label_cols, Mapping) else label_cols
        )
        with h5py.File(self.h5_path, "r") as handle:
            values = np.asarray(handle[column][...])
        return LabelDistribution.from_labels(values.reshape(-1), _LABELS_KEY)

    # -------------------------------------------------------------------- items
    def _finalise(self, image: Any, label: Any) -> Any:
        """Apply channels_last and merge_labels, then shape the return value."""
        if self.channels_last:
            if self.multiresolution:
                image = {key: value.permute(1, 2, 0) for key, value in image.items()}
            else:
                image = image.permute(1, 2, 0)
            if self.mode == "segmentation":
                if self.multiresolution:
                    label = {
                        key: value.permute(1, 2, 0) for key, value in label.items()
                    }
                else:
                    label = label.permute(1, 2, 0)

        if self.mode == "image_only":
            return image

        if self.mode == "segmentation" and self.merge_labels:
            masks = label.values() if self.multiresolution else [label]
            for mask in masks:
                for source, destination in self.merge_labels.items():
                    mask[mask == source] = destination

        return image, label

    def __getitem__(self, index: int) -> Any:
        """Read, transform and return one item."""
        self._ensure_open()

        image = self._read_image(index)
        mask = self._read_mask(index) if self.mode == "segmentation" else None

        image_out: dict[str, torch.Tensor] | torch.Tensor
        mask_out: dict[str, torch.Tensor] | torch.Tensor | None
        if self.multiresolution:
            image_out, mask_out = transform_multires_image_and_mask(
                image,
                mask,
                transform=self.transform,
                mask_transform=self.mask_transform,
            )
        else:
            # The pre-1.0 code assigned this 2-tuple straight to `image` in the
            # non-segmentation branch, so `image_only` and `classification`
            # datasets returned `(tensor, None)` where an image tensor was
            # expected -- and `channels_last=True` then hit `tuple.permute`.
            image_out, mask_out = transform_image_and_mask(
                image,
                mask,
                transform=self.transform,
                mask_transform=self.mask_transform,
            )

        if self.mode == "classification":
            return self._finalise(image_out, self._read_scalar_label(index))
        return self._finalise(image_out, mask_out)
