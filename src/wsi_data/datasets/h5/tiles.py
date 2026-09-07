"""Unlabelled multi-resolution tile dataset over one HDF5 file.

Note:
    Named ``Single_H5_Image_Dataset`` before 1.0.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    import albumentations as A

__all__ = ["TileDatasetHDF5"]


class TileDatasetHDF5(Dataset[dict[str, np.ndarray]]):
    """Tiles from one HDF5 file, with image columns discovered by regex.

    Each item is a mapping of column name to array. Where
    :class:`~wsi_data.datasets.h5.images.ImageDatasetHDF5` needs its columns
    named explicitly, this class finds every dataset whose name matches
    ``image_regex``, which suits files written with one dataset per resolution
    under a shared prefix.

    Args:
        h5_file: Path to the HDF5 file.
        image_regex: Pattern matched against dataset names to find image
            columns. Matched with :meth:`re.Pattern.match`, so it anchors at
            the start.
        data_cols: Columns to read. Defaults to every column matching
            ``image_regex``. Any extra columns listed here are read but not
            passed through ``transform``.
        transform: Albumentations pipeline. Applied with the first image
            column as the primary target and the rest as additional targets,
            so one parameter draw is shared across resolutions; it must be
            built with matching ``additional_targets``.
        channels_last: Retained for interface compatibility. Arrays are
            returned exactly as stored, so this has no effect.

    Raises:
        FileNotFoundError: If ``h5_file`` does not exist.
        ValueError: If no column matches ``image_regex``, or a requested
            column is missing from the file.
    """

    def __init__(
        self,
        h5_file: Path | str,
        image_regex: str = "^x_",
        data_cols: Sequence[str] | None = None,
        transform: A.Compose | None = None,
        *,
        channels_last: bool = False,
    ) -> None:
        self.h5_file = Path(h5_file)
        if not self.h5_file.exists():
            msg = f"{self.h5_file} does not exist"
            raise FileNotFoundError(msg)

        self.channels_last = channels_last
        self.transform = transform
        self._handle: h5py.File | None = None
        self._data: dict[str, Any] = {}

        pattern = re.compile(image_regex)
        with h5py.File(self.h5_file, "r") as handle:
            available = list(handle.keys())
            self.image_keys = [name for name in available if pattern.match(name)]
            if not self.image_keys:
                msg = (
                    f"no dataset in {self.h5_file} matches {image_regex!r}; "
                    f"available: {available}"
                )
                raise ValueError(msg)

            self.data_cols = (
                list(self.image_keys) if data_cols is None else list(data_cols)
            )
            missing = [name for name in self.data_cols if name not in available]
            if missing:
                msg = f"columns {missing} are not in {self.h5_file}"
                raise ValueError(msg)
            # `transform` addresses image_keys, so they must be readable.
            unread = [name for name in self.image_keys if name not in self.data_cols]
            if unread and transform is not None:
                msg = (
                    f"image columns {unread} match image_regex but are not in "
                    f"data_cols, so the transform cannot be applied to them; "
                    f"add them to data_cols or narrow image_regex"
                )
                raise ValueError(msg)

            first = handle[self.data_cols[0]]
            self.dataset_size = int(first.shape[0])
            self.image_shape = tuple(first.shape[1:])

    def __len__(self) -> int:
        return self.dataset_size

    def open_hdf5(self) -> None:
        """Open the file and bind the requested columns."""
        self._handle = h5py.File(self.h5_file, "r")
        self._data = {name: self._handle[name] for name in self.data_cols}

    def close(self) -> None:
        """Close the file if it is open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._data = {}

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open HDF5 handle so the dataset can be sent to workers."""
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_data"] = {}
        return state

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Read and transform one item."""
        if self._handle is None:
            self.open_hdf5()

        data = {name: np.asarray(self._data[name][index]) for name in self.data_cols}

        if self.transform is not None:
            primary, *extra = self.image_keys
            kwargs = {"image": data[primary]}
            kwargs.update({name: data[name] for name in extra})
            transformed = self.transform(**kwargs)
            data[primary] = transformed["image"]
            for name in extra:
                data[name] = transformed[name]

        return data
