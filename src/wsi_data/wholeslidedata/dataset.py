"""A :class:`~wholeslidedata.data.dataset.WholeSlideDataSet` for multi-res slides.

Note:
    This class was 182 lines before 1.0, of which roughly 120 duplicated its
    upstream parent. ``_init_labels`` and ``_init_samples`` were byte-identical
    to the inherited versions; ``pixels_count``, ``pixels_per_label``,
    ``pixels_per_key`` and ``pixels_per_label_per_key`` reimplemented inherited
    properties; and ``annotations_per_label``/``annotations_per_key``/
    ``annotations_per_label_per_key`` were the inherited
    ``annotation_counts_per_*`` properties under different names. All of that
    is now simply inherited -- use the upstream ``annotation_counts_per_label``
    name for what used to be ``annotations_per_label``.

    What genuinely differs from the parent, and so remains here, is annotation
    spacing: the parent opens each image to read ``spacings[0]`` and passes it
    to the annotation parser, whereas this class opens annotations with no
    spacing, since QuPath GeoJSON coordinates are already stored at level 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wholeslidedata.data.dataset import WholeSlideDataSet
from wholeslidedata.data.files import WholeSlideAnnotationFile, WholeSlideImageFile

if TYPE_CHECKING:
    from sourcelib.associations import Associations
    from wholeslidedata.annotation.labels import Labels

    from wsi_data.wholeslidedata.files import MultiResWholeSlideImageFile

__all__ = ["MultiResWholeSlideDataSet"]


class MultiResWholeSlideDataSet(WholeSlideDataSet):
    """A dataset of slides opened for multi-resolution patch reading.

    Args:
        mode: Dataset mode, e.g. ``"default"``.
        associations: Slide-to-annotation associations from
            :func:`sourcelib.associations.associate_files`.
        labels: Labels to restrict annotations to.
        load_images: Open each slide eagerly. When ``False`` the file wrapper
            is stored instead and opened on demand by the sampler.
        copy_path: Directory to copy slides and annotations into before
            opening, for staging onto fast local storage.
    """

    def __init__(
        self,
        mode: Any,
        associations: Associations,
        labels: Labels | None = None,
        *,
        load_images: bool = True,
        copy_path: str | None = None,
    ) -> None:
        # Forwarded rather than assigned before `super().__init__`: the parent
        # sets `_load_images`/`_copy_path` from its own defaults and only then
        # calls `_open`, so the pre-1.0 code -- which assigned them first and
        # called `super().__init__(mode, associations, labels)` without them --
        # had both silently overwritten with `True`/`None` before any use.
        super().__init__(
            mode,
            associations,
            labels,
            load_images=load_images,
            copy_path=copy_path,
        )

    def _open(
        self, associations: Associations, labels: Labels | None
    ) -> dict[str, Any]:
        """Open every associated slide and annotation, keyed by file key."""
        # `IDENTIFIER` rather than the literals "wsi"/"wsa" the pre-1.0 code
        # hardcoded, so a rename upstream is a clean error instead of a KeyError.
        data: dict[str, Any] = {}
        for file_key, associated_files in associations.items():
            data[file_key] = {
                self.__class__.IMAGES_KEY: {},
                self.__class__.ANNOTATIONS_KEY: {},
            }
            for wsi_index, wsi_file in enumerate(
                associated_files[WholeSlideImageFile.IDENTIFIER]
            ):
                data[file_key][self.__class__.IMAGES_KEY][wsi_index] = self._open_image(
                    wsi_file
                )
            for wsa_index, wsa_file in enumerate(
                associated_files[WholeSlideAnnotationFile.IDENTIFIER]
            ):
                data[file_key][self.__class__.ANNOTATIONS_KEY][wsa_index] = (
                    self._open_annotation(wsa_file, labels=labels)
                )
        return data

    def _open_image(self, wsi_file: MultiResWholeSlideImageFile) -> Any:
        """Open one slide, or return the file wrapper when not loading eagerly.

        Unlike the parent, returns the image alone rather than an
        ``(image, spacing)`` pair, since this class does not feed a spacing to
        the annotation parser.
        """
        if self._copy_path:
            wsi_file.copy(self._copy_path)
        if self._load_images:
            return wsi_file.open()
        return wsi_file

    def _open_annotation(
        self, wsa_file: WholeSlideAnnotationFile, labels: Labels | None = None
    ) -> Any:
        """Open one annotation file without a spacing.

        QuPath GeoJSON coordinates are already at level 0, so no rescaling
        spacing is passed -- the one intentional divergence from the parent.
        """
        if self._copy_path:
            wsa_file.copy(self._copy_path)
        return wsa_file.open(labels=labels)

    def close_images(self) -> None:
        """Close every opened slide and drop the references.

        Note:
            The pre-1.0 version iterated ``self._images``, an attribute neither
            this class nor its parent ever defined, so it always raised
            ``AttributeError``.
        """
        for values in self._data.values():
            images = values[self.__class__.IMAGES_KEY]
            for image in images.values():
                close = getattr(image, "close", None)
                if close is not None:
                    close()
            images.clear()
