"""File wrappers that open slides as this package's whole-slide image classes.

``wholeslidedata`` decides which image class a discovered file becomes via the
file wrapper's :meth:`open`, so each image class needs a matching file class.
"""

from __future__ import annotations

from wholeslidedata.data.files import WholeSlideImageFile

from wsi_data.wholeslidedata.wholeslideimage import (
    MultiResWholeSlideImage,
    SingleResWholeSlideImage,
)

__all__ = ["MultiResWholeSlideImageFile", "SingleResWholeSlideImageFile"]


class MultiResWholeSlideImageFile(WholeSlideImageFile):
    """A slide file that opens as a :class:`MultiResWholeSlideImage`."""

    def open(self) -> MultiResWholeSlideImage:
        """Open the slide for multi-resolution reads."""
        return MultiResWholeSlideImage(self.path, self._image_backend)


class SingleResWholeSlideImageFile(WholeSlideImageFile):
    """A slide file that opens as a :class:`SingleResWholeSlideImage`.

    Note:
        Named ``MyWholeSlideImageFile`` before 1.0.
    """

    def open(self) -> SingleResWholeSlideImage:
        """Open the slide for single-resolution reads."""
        return SingleResWholeSlideImage(self.path, self._image_backend)
