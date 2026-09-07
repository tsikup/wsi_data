"""Whole-slide image classes with annotation, tissue-mask and spacing helpers.

:class:`BaseWholeSlideImage` holds everything that does not depend on how many
resolutions a patch is read at; :class:`SingleResWholeSlideImage` and
:class:`MultiResWholeSlideImage` are siblings that differ only in what
:meth:`~BaseWholeSlideImage.get_data` returns.

Note:
    Before 1.0 the single-resolution class *subclassed* the multi-resolution
    one and overrode ``get_data`` to return a bare array where its parent
    returned a dict of arrays -- a subclass narrowing its parent's contract,
    so it could not be used anywhere the parent was expected. The two now
    share a base instead, and the ~80 lines of near-identical patch-extraction
    logic they each carried live in one place
    (:meth:`BaseWholeSlideImage.extract_patch`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import cv2
import numpy as np
from he_preprocessing.slide.io import read_scaled_region
from he_preprocessing.tissue.detect import detect_tissue
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.annotation import utils as annotation_utils
from wholeslidedata.image.spacings import take_closest_level
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.interoperability.qupath.parser import QuPathAnnotationParser
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from wholeslidedata.image.backend import WholeSlideImageBackend

    from wsi_data.tissue_segmentation import CNNTissueSegmentor

__all__ = [
    "BaseWholeSlideImage",
    "DetailTiling",
    "MultiResWholeSlideImage",
    "SingleResWholeSlideImage",
    "TissueMaskResult",
]

logger = logging.getLogger(__name__)

#: Default thumbnail spacing (microns per pixel) per tissue-detection method.
_TISSUE_MASK_SPACINGS = {"he_preprocessing": 32.0, "cnn": 8.0}


@dataclass(frozen=True)
class TissueMaskResult:
    """A tissue mask together with the downsample factor it was computed at.

    Note:
        Replaces the pre-1.0 ``get_tissue_mask`` return value, which was a
        2-tuple or a 3-tuple depending on ``return_contours`` -- so callers
        had to unpack differently based on an argument they passed.
        ``contours`` is simply ``None`` when they were not requested.

    Attributes:
        mask: ``(H, W)`` ``uint8`` mask, ``1`` where tissue was detected.
        downsample: Factor the thumbnail was downsampled by relative to level 0.
        contours: External contours of ``mask`` as returned by
            :func:`cv2.findContours`, or ``None`` if not requested.
    """

    mask: np.ndarray
    downsample: float
    contours: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True)
class DetailTiling:
    """How a target-resolution tile subdivides into higher-resolution tiles.

    Note:
        Replaces the pre-1.0 ``get_num_details``, which returned a 5-tuple
        when the ``"details"`` spacing was present and a 3-tuple of ``None``
        when it was not, so unpacking its result raised ``ValueError``
        whenever details were absent.

    Attributes:
        n_patches: Total detail patches covering one target tile.
        tiles_x: Detail patches along the x axis.
        tiles_y: Detail patches along the y axis.
        downsampling_target: Downsample factor of the target spacing.
        downsampling_details: Downsample factor of the details spacing.
    """

    n_patches: int
    tiles_x: int
    tiles_y: int
    downsampling_target: float
    downsampling_details: float


class BaseWholeSlideImage(WholeSlideImage):
    """A whole-slide image with annotations, tissue detection and spacing helpers.

    Not meant to be instantiated directly -- use
    :class:`SingleResWholeSlideImage` or :class:`MultiResWholeSlideImage`,
    which add the matching :meth:`get_data`.

    Args:
        path: Path to the slide.
        backend: Backend name or class used to read the slide.
        annotation_path: Optional QuPath ``.geojson`` annotation file.
        labels: Label mapping passed to the annotation parser.

    Raises:
        ValueError: If ``annotation_path`` is not a ``.geojson`` file.
    """

    def __init__(
        self,
        path: Path | str,
        backend: WholeSlideImageBackend | str = "openslide",
        annotation_path: Path | str | None = None,
        labels: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(path=path, backend=backend)

        self.annotation: WholeSlideAnnotation | None = None
        if annotation_path is not None:
            if not str(annotation_path).endswith(".geojson"):
                msg = f"annotation must be a .geojson file, got {annotation_path!r}"
                raise ValueError(msg)
            self.annotation = WholeSlideAnnotation(
                annotation_path=annotation_path,
                labels=labels,
                parser=QuPathAnnotationParser(),
            )

        self.mask_sampler: SegmentationPatchLabelSampler | None = None

    # ------------------------------------------------------------- annotations
    @property
    def labels(self) -> Any:
        """The annotation's labels, or an empty list when unannotated."""
        return self.annotation.labels if self.annotation is not None else []

    def _require_annotation(self) -> WholeSlideAnnotation:
        if self.annotation is None:
            msg = (
                f"{type(self).__name__} has no annotation; pass annotation_path "
                f"to the constructor or assign `.annotation`"
            )
            raise ValueError(msg)
        return self.annotation

    @property
    def annotation_counts(self) -> Any:
        """Total number of annotations."""
        return annotation_utils.get_counts_in_annotations(
            self._require_annotation().annotations
        )

    @property
    def annotations_per_label(self) -> dict[str, int]:
        """Number of annotations per label name."""
        counts: dict[str, int] = annotation_utils.get_counts_in_annotations(
            self._require_annotation().annotations, labels=self.labels
        )
        return counts

    @property
    def pixels_count(self) -> Any:
        """Total area, in pixels, covered by annotations."""
        return annotation_utils.get_pixels_in_annotations(
            self._require_annotation().annotations
        )

    @property
    def pixels_per_label(self) -> dict[str, int]:
        """Area, in pixels, covered by annotations of each label."""
        pixels: dict[str, int] = annotation_utils.get_pixels_in_annotations(
            self._require_annotation().annotations, labels=self.labels
        )
        return pixels

    def create_mask_sampler(self) -> None:
        """Create the segmentation mask sampler used by ``get_data(with_mask=True)``."""
        self.mask_sampler = SegmentationPatchLabelSampler()

    # ---------------------------------------------------------------- spacings
    def _level_and_rescale_from_spacing(self, spacing: float) -> tuple[int, bool]:
        """Find the pyramid level closest to ``spacing``, and whether to rescale."""
        closest_level = take_closest_level(self.spacings, spacing)
        # WholeSlideImage.SPACING_MARGIN is a fraction, not a percentage.
        spacing_margin = spacing * WholeSlideImage.SPACING_MARGIN
        rescale = False

        if abs(self.spacings[closest_level] - spacing) > spacing_margin:
            if self.spacings[closest_level] > spacing:
                # Step to the next finer level so the patch can be downscaled
                # to the requested spacing rather than upscaled.
                closest_level -= 1
            rescale = True
            logger.debug(
                "spacing %s is outside the %.0f%% margin of available spacings %s; "
                "using level %d (%s) and rescaling",
                spacing,
                WholeSlideImage.SPACING_MARGIN * 100,
                self.spacings,
                max(closest_level, 0),
                self.spacings[max(closest_level, 0)],
            )

        return max(closest_level, 0), rescale

    @overload
    def get_level_from_spacing(
        self, spacing: float, *, return_rescaling: Literal[False] = False
    ) -> int: ...
    @overload
    def get_level_from_spacing(
        self, spacing: float, *, return_rescaling: Literal[True]
    ) -> tuple[int, bool]: ...

    def get_level_from_spacing(
        self, spacing: float, *, return_rescaling: bool = False
    ) -> tuple[int, bool] | int:
        """Find the pyramid level closest to ``spacing``.

        Args:
            spacing: Desired spacing in microns per pixel.
            return_rescaling: Also return whether the chosen level's spacing
                differs enough from ``spacing`` that patches need resizing.

        Returns:
            The level index, or ``(level, rescale)`` when ``return_rescaling``.
        """
        level, rescale = self._level_and_rescale_from_spacing(spacing)
        if return_rescaling:
            return level, rescale
        return level

    @overload
    def get_real_spacing(
        self, spacing: float, *, return_rescaling: Literal[False] = False
    ) -> float: ...
    @overload
    def get_real_spacing(
        self, spacing: float, *, return_rescaling: Literal[True]
    ) -> tuple[float, bool]: ...

    def get_real_spacing(
        self, spacing: float, *, return_rescaling: bool = False
    ) -> tuple[float, bool] | float:
        """Return the actual spacing of the level closest to ``spacing``.

        Args:
            spacing: Desired spacing in microns per pixel.
            return_rescaling: Also return whether patches need resizing.

        Returns:
            The real spacing, or ``(spacing, rescale)`` when ``return_rescaling``.
        """
        level, rescale = self._level_and_rescale_from_spacing(spacing)
        real_spacing = float(self.spacings[level])
        if return_rescaling:
            return real_spacing, rescale
        return real_spacing

    def get_thumbnail(self, spacing: float = 8) -> tuple[np.ndarray, float, float]:
        """Read the whole slide at approximately ``spacing``.

        Args:
            spacing: Desired thumbnail spacing in microns per pixel.

        Returns:
            An ``(image, real_spacing, downsample)`` triple.
        """
        real_spacing = self.get_real_spacing(spacing)
        downsample = self.get_downsampling_from_spacing(spacing=real_spacing)
        return self.get_slide(spacing=real_spacing), real_spacing, downsample

    def get_detail_tiling(
        self, width: int, height: int, spacings: Mapping[str, float]
    ) -> DetailTiling | None:
        """Work out how a target tile subdivides into higher-resolution tiles.

        Args:
            width: Target tile width in pixels.
            height: Target tile height in pixels.
            spacings: Spacings by name; must contain ``"target"``, and
                ``"details"`` for a non-``None`` result.

        Returns:
            The tiling, or ``None`` when no ``"details"`` spacing is given.

        Raises:
            ValueError: If the detail tile size does not divide the target tile
                size exactly.
        """
        if "details" not in spacings:
            return None

        downsampling_target = self.get_downsampling_from_spacing(
            spacing=spacings["target"]
        )
        downsampling_details = self.get_downsampling_from_spacing(
            spacing=spacings["details"]
        )
        # Size of a detail patch expressed in target-resolution pixels.
        ratio = int(downsampling_target) / int(downsampling_details)
        details_width = int(width / ratio)
        details_height = int(height / ratio)

        if (
            details_width == 0
            or details_height == 0
            or width % details_width
            or height % details_height
        ):
            msg = (
                f"detail patch size ({details_width}, {details_height}) must divide "
                f"tile size ({width}, {height}) exactly; e.g. for a target size of "
                f"512 use a detail size of 256, 128, 64, ..."
            )
            raise ValueError(msg)

        tiles_x = width // details_width
        tiles_y = height // details_height
        return DetailTiling(
            n_patches=tiles_x * tiles_y,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            downsampling_target=downsampling_target,
            downsampling_details=downsampling_details,
        )

    # ------------------------------------------------------------ tissue masks
    def get_tissue_mask(
        self,
        method: str = "he_preprocessing",
        spacing: float | None = None,
        segmentor: CNNTissueSegmentor | None = None,
        *,
        return_contours: bool = True,
    ) -> TissueMaskResult:
        """Detect tissue in a downsampled thumbnail of the slide.

        Args:
            method: ``"he_preprocessing"`` for optical-density-based detection,
                or ``"cnn"`` for a pretrained CNN segmentor.
            spacing: Thumbnail spacing in microns per pixel. Defaults to 32 for
                ``"he_preprocessing"`` and 8 for ``"cnn"`` (the resolution the
                CNN checkpoint expects).
            segmentor: Required for ``method="cnn"``: a
                :class:`~wsi_data.tissue_segmentation.CNNTissueSegmentor`,
                built once and reused across slides since it holds a model.
            return_contours: Also compute the mask's external contours.

        Returns:
            A :class:`TissueMaskResult`.

        Raises:
            ValueError: If ``method`` is unknown, or ``"cnn"`` is requested
                without a ``segmentor``.
        """
        if method not in _TISSUE_MASK_SPACINGS:
            msg = (
                f"method must be one of {sorted(_TISSUE_MASK_SPACINGS)}, got {method!r}"
            )
            raise ValueError(msg)
        target_spacing = _TISSUE_MASK_SPACINGS[method] if spacing is None else spacing

        if method == "cnn":
            if segmentor is None:
                msg = "method='cnn' requires a CNNTissueSegmentor via `segmentor`"
                raise ValueError(msg)
            real_spacing = self.get_real_spacing(target_spacing)
            downsample = self.get_downsampling_from_spacing(real_spacing)
            thumbnail = self.get_slide(spacing=real_spacing)
            mask = (segmentor.predict(thumbnail) > 0).astype(np.uint8)
        else:
            thumbnail, downsample = read_scaled_region(
                self.path,
                downsample=self.get_downsampling_from_spacing(target_spacing),
            )
            mask = detect_tissue(thumbnail).astype(np.uint8)

        contours = None
        if return_contours:
            found, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = tuple(found)
        return TissueMaskResult(mask=mask, downsample=downsample, contours=contours)

    # ------------------------------------------------------------ patch reading
    def extract_patch(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Read one patch at ``spacing``, resizing if no exact level exists.

        The shared core of both subclasses' ``get_data``: resolve the requested
        spacing to a real pyramid level, read an accordingly larger patch when
        the level is finer than requested, optionally sample the annotation
        mask, and resize both back to ``(width, height)``.

        Args:
            x: x coordinate, at ``spacing`` if ``relative`` else at level 0.
            y: y coordinate, likewise.
            width: Output patch width in pixels.
            height: Output patch height in pixels.
            spacing: Requested spacing in microns per pixel.
            center: Treat ``(x, y)`` as the patch centre rather than its
                top-left corner.
            relative: Treat ``(x, y)`` as relative to ``spacing``'s own
                dimensions rather than level 0.
            with_mask: Also sample the annotation mask for this patch.

        Returns:
            A ``(patch, mask)`` pair; ``mask`` is ``None`` unless ``with_mask``.

        Raises:
            ValueError: If ``with_mask`` is set without ``center``, or the
                image has no annotation to sample a mask from.
        """
        real_spacing, rescale = self.get_real_spacing(spacing, return_rescaling=True)

        if rescale:
            # The chosen level is finer than requested, so read proportionally
            # more pixels and downscale to the requested size.
            scaling_factor = spacing / real_spacing
            read_width = int(width * scaling_factor)
            read_height = int(height * scaling_factor)
        else:
            read_width, read_height = width, height

        patch = self.get_patch(
            x,
            y,
            read_width,
            read_height,
            spacing=real_spacing,
            center=center,
            relative=relative,
        )

        mask = None
        if with_mask:
            if not center:
                msg = "mask sampling requires center=True"
                raise ValueError(msg)
            annotation = self._require_annotation()
            if self.mask_sampler is None:
                # Was missing from the pre-1.0 single-resolution override, so
                # `get_data(with_mask=True)` there always hit `None.sample`.
                self.create_mask_sampler()
            assert self.mask_sampler is not None
            mask = self.mask_sampler.sample(
                annotation,
                (x, y),
                size=(read_width, read_height),
                # Annotation coordinates are stored at level 0, so the sampler
                # needs the patch spacing relative to the finest level.
                ratio=real_spacing / self.spacings[0],
            )

        if rescale:
            patch = cv2.resize(patch, (width, height))
            if mask is not None:
                mask = cv2.resize(mask, (width, height))

        return patch, mask


class SingleResWholeSlideImage(BaseWholeSlideImage):
    """A whole-slide image read at one resolution.

    Note:
        Named ``MyWholeSlideImage`` before 1.0.
    """

    @overload
    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: Literal[False] = False,
    ) -> np.ndarray: ...
    @overload
    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a single patch from the slide.

        Note:
            ``spacing`` must be a number. The pre-1.0 version also accepted a
            one-entry dict and silently unwrapped it (dropping ``None``
            values); normalise spacing in the caller instead --
            :class:`~wsi_data.datasets.slide.SlideTileDataset` does.

        Args:
            x: x coordinate, at ``spacing`` if ``relative`` else at level 0.
            y: y coordinate, likewise.
            width: Output patch width in pixels.
            height: Output patch height in pixels.
            spacing: Spacing in microns per pixel.
            center: Treat ``(x, y)`` as the patch centre.
            relative: Treat ``(x, y)`` as relative to ``spacing``'s dimensions.
            with_mask: Also return the annotation mask.

        Returns:
            The patch, or a ``(patch, mask)`` pair when ``with_mask``.
        """
        patch, mask = self.extract_patch(
            x,
            y,
            width,
            height,
            spacing,
            center=center,
            relative=relative,
            with_mask=with_mask,
        )
        if with_mask:
            assert mask is not None
            return patch, mask
        return patch


class MultiResWholeSlideImage(BaseWholeSlideImage):
    """A whole-slide image read at several resolutions at the same location.

    Reading the same ``(x, y)`` at, say, ``{"target": 0.5, "context": 2.0}``
    yields a detail patch and a wider-field-of-view patch of identical pixel
    size, which is what context-aware models consume.
    """

    @overload
    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: Mapping[str, float] | float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: Literal[False] = False,
    ) -> dict[str, np.ndarray]: ...
    @overload
    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: Mapping[str, float] | float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: Literal[True],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...

    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: Mapping[str, float] | float,
        *,
        center: bool = True,
        relative: bool = False,
        with_mask: bool = False,
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read one patch per requested resolution.

        Args:
            x: x coordinate, at the given spacing if ``relative`` else at level 0.
            y: y coordinate, likewise.
            width: Output patch width in pixels, the same for every resolution.
            height: Output patch height in pixels.
            spacing: Mapping of resolution name to spacing in microns per
                pixel. A bare number is accepted and treated as
                ``{"target": spacing}``. A ``"details"`` entry is skipped --
                see :meth:`~BaseWholeSlideImage.get_detail_tiling`.
            center: Treat ``(x, y)`` as the patch centre.
            relative: Treat ``(x, y)`` as relative to the spacing's dimensions.
            with_mask: Also return a mask per resolution.

        Returns:
            A resolution-keyed dict of patches, or a ``(patches, masks)`` pair
            of such dicts when ``with_mask``.
        """
        spacings = (
            {"target": spacing} if isinstance(spacing, (int, float)) else dict(spacing)
        )

        data: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for key, value in spacings.items():
            if key == "details":
                continue
            patch, mask = self.extract_patch(
                x,
                y,
                width,
                height,
                value,
                center=center,
                relative=relative,
                with_mask=with_mask,
            )
            data[key] = patch
            if mask is not None:
                masks[key] = mask

        if with_mask:
            return data, masks
        return data
