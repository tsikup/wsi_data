import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.annotation import utils as annotation_utils
from wholeslidedata.image.backend import WholeSlideImageBackend
from wholeslidedata.image.spacings import take_closest_level
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.interoperability.qupath.parser import QuPathAnnotationParser
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler

import he_preprocessing.utils.image as ut_image


class MultiResWholeSlideImage(WholeSlideImage):
    def __init__(
        self,
        path: Union[Path, str],
        backend: Union[WholeSlideImageBackend, str] = "openslide",
        annotation_path=None,
        labels=None,
    ):
        super(MultiResWholeSlideImage, self).__init__(path=path, backend=backend)

        self.annotation = None
        if annotation_path:
            assert annotation_path.endswith(
                ".geojson"
            ), "Annotation must be a geojson file."
            self._annotation_parser = QuPathAnnotationParser()
            self.annotation = WholeSlideAnnotation(
                annotation_path=annotation_path,
                labels=labels,
                parser=self._annotation_parser,
            )
        else:
            self.annotation = None

        self.mask_sampler: SegmentationPatchLabelSampler = None

    @property
    def labels(self):
        if self.annotation is not None:
            return self.annotation.labels
        return []

    @property
    def annotation_counts(self):
        return annotation_utils.get_counts_in_annotations(self.annotation.annotations)

    @property
    def annotations_per_label(self) -> Dict[str, int]:
        return annotation_utils.get_counts_in_annotations(
            self.annotation.annotations, labels=self.labels
        )

    @property
    def pixels_count(self):
        return annotation_utils.get_pixels_in_annotations(self.annotation.annotations)

    @property
    def pixels_per_label(self) -> Dict[str, int]:
        return annotation_utils.get_pixels_in_annotations(
            self.annotation.annotations, labels=self.labels
        )

    def create_mask_sampler(self):
        self.mask_sampler = SegmentationPatchLabelSampler()

    def get_tissue_mask(self, spacing=32, return_contours=True):
        downsample = self.get_downsampling_from_spacing(spacing)
        return ut_image.get_slide_tissue_mask(
            slide_path=self.path,
            downsample=downsample,
            return_contours=return_contours,
        )

    def get_thumbnail(self, spacing=8):
        spacing = self.get_real_spacing(spacing)
        downsample = self.get_downsampling_from_spacing(spacing)
        return self.get_slide(spacing=spacing), spacing, downsample

    def get_num_details(self, width: int, height: int, spacings: Dict[str, float]):
        if "details" in spacings:
            downsampling_target = self.get_downsampling_from_spacing(
                spacing=spacings["target"]
            )
            downsampling_details = self.get_downsampling_from_spacing(
                spacing=spacings["details"]
            )
            # This is the relative image size of detail patches with respect to the target resolution.
            details_width = int(
                width / (int(downsampling_target) / int(downsampling_details))
            )
            details_height = int(
                height / (int(downsampling_target) / int(downsampling_details))
            )

            assert width % details_width == 0 and height % details_height == 0, (
                "The relative size of detail patches must divide tile size "
                "perfectly. E.g. target size -> 512, then detail_size -> ["
                "256, 128, 64, ...] "
            )

            # Number of detail patches that will be extracted from the target patch at a higher resolution.
            num_details_patches = int(width // details_width) * int(
                height // details_height
            )
            # Total iterations over the x-axis to get all detail patches
            total_i = int(width / details_width)
            # Total iterations over the y-axis to get all detail patches
            total_j = int(height / details_height)

            return (
                num_details_patches,
                total_i,
                total_j,
                downsampling_target,
                downsampling_details,
            )
        return None, None, None

    def get_level_from_spacing(
        self, spacing: float, return_rescaling: bool = False
    ) -> Union[tuple[int, bool], int]:
        closest_level = take_closest_level(self._spacings, spacing)
        spacing_margin = spacing * WholeSlideImage.SPACING_MARGIN
        rescale = False

        if abs(self.spacings[closest_level] - spacing) > spacing_margin:
            if self.spacings[closest_level] > spacing:
                closest_level -= 1
            rescale = True
            warnings.warn(
                f"spacing {spacing} outside margin (0.3%) for {self._spacings}, returning left bisected spacing."
            )

        closest_level = max(closest_level, 0)

        if return_rescaling:
            return closest_level, rescale
        return closest_level

    def get_real_spacing(self, spacing, return_rescaling: bool = False):
        level_rescaling = self.get_level_from_spacing(
            spacing, return_rescaling=return_rescaling
        )
        if return_rescaling:
            level, rescale = level_rescaling
            return self._spacings[level], rescale
        return self._spacings[level_rescaling]

    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: Dict[str, float],
        center: bool = True,
        relative: bool = False,
        with_mask: bool = False,
    ) -> Union[
        tuple[dict[Any, Union[Union[ndarray, ndarray], Any]], dict[Any, Any]],
        dict[Any, Union[Union[ndarray, ndarray], Any]],
    ]:
        """Extracts multi-resolution patches/regions from the wholeslideimage
        Args:
            x (int): x value
            y (int): y value
            width (int): width of region
            height (int): height of region
            spacing (list of float): spacing/resolution of the patch at target, context and details level
            center (bool, optional): if x,y values are centres or top left coordinated. Defaults to True.
            relative (bool, optional): if x,y values are a reference to the dimensions of the specified spacing. Defaults to False.
        Returns:
            np.ndarray: numpy patch
        """

        if with_mask and self.mask_sampler is None:
            self.create_mask_sampler()

        if not isinstance(spacing, dict):
            assert isinstance(spacing, float) or isinstance(
                spacing, int
            ), "Spacings must be a dictionary or float/int for uniresolution patches"
            spacing = {"target": spacing}

        assert isinstance(spacing, dict), "Spacings must be a dictionary"

        _original_spacing = spacing.copy()
        _spacings = spacing.copy()
        _rescale = dict()

        data = dict()
        mask = dict()

        for key, value in _spacings.items():
            _spacings[key], _rescale[key] = self.get_real_spacing(
                value, return_rescaling=True
            )

        for key, spacing in _spacings.items():
            if key == "details":
                continue

            if _rescale[key]:
                _scaling_factor = _original_spacing[key] / spacing
                assert _scaling_factor > 1, "Scaling factor must be > 1"

                _width = int(width * _scaling_factor)
                _height = int(height * _scaling_factor)
            else:
                _width = width
                _height = height

            _patch = self.get_patch(
                x,
                y,
                _width,
                _height,
                spacing=spacing,
                center=center,
                relative=relative,
            )

            if with_mask:
                assert center, "Mask sampling only works with center=True"
                _mask = self.mask_sampler.sample(
                    self.annotation,
                    (x, y),
                    size=(_width, _height),
                    ratio=spacing
                    / self.spacings[
                        0
                    ],  # Test if this is correct by manual annotating a region in qupath
                )

            if _rescale[key]:
                _patch = cv2.resize(_patch, (width, height))
                if with_mask:
                    _mask = cv2.resize(_mask, (width, height))

            data[key] = _patch
            if with_mask:
                mask[key] = _mask

        # # Details
        # x_details = None
        # if "details" in _spacings and not _rescale["details"]:
        #     x_details = []
        #
        #     (
        #         num_details_patches,
        #         total_i,
        #         total_j,
        #         downsampling_target,
        #         downsampling_details,
        #     ) = self.get_num_details(width, height, _spacings)
        #
        #     if center:
        #         # Get top left coords of target resolution
        #         downsampling = int(
        #             self.get_downsampling_from_spacing(_spacings["target"])
        #         )
        #         x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)
        #
        #     for idx in range(num_details_patches):
        #         i = int(idx // total_i)
        #         j = int(idx % total_j)
        #
        #         rel_coord_x = j * width * downsampling_target
        #         rel_coord_y = i * height * downsampling_target
        #
        #         coord_x = x + rel_coord_x
        #         coord_y = y + rel_coord_y
        #         x_details.append(
        #             self.get_patch(
        #                 coord_x,
        #                 coord_y,
        #                 width,
        #                 height,
        #                 spacing=_spacings["details"],
        #                 center=False,
        #                 relative=relative,
        #             )
        #         )
        #
        #     data["details"] = x_details
        # elif "details" in _spacings:
        #     raise NotImplementedError(
        #         "Details extraction with rescaling not implemented yet"
        #     )

        if with_mask:
            return data, mask
        return data


class MyWholeSlideImage(MultiResWholeSlideImage):
    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacing: float,
        center: bool = True,
        relative: bool = False,
        with_mask: bool = False,
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Extracts a patch/region from the wholeslideimage

        Args:
            x (int): x value
            y (int): y value
            width (int): width of region
            height (int): height of region
            spacing (float): spacing/resolution of the patch
            center (bool, optional): if x,y values are centres or top left coordinated. Defaults to True.
            relative (bool, optional): if x,y values are a reference to the dimensions of the specified spacing. Defaults to False.

        Returns:
            np.ndarray: numpy patch
        """

        _spacing = copy(spacing)

        if isinstance(_spacing, dict):
            _spacing = {k: v for k, v in _spacing.items() if v is not None}
            assert (
                len(_spacing) == 1
            ), "Spacings must be a dictionary with only one key, e.g. {'target': 0.5} or float or int for uniresolution patches"
            _spacing = _spacing[list(_spacing.keys())[0]]

        assert isinstance(_spacing, float) or isinstance(_spacing, int)

        _original_spacing = _spacing

        _spacing, _rescale = self.get_real_spacing(_spacing, return_rescaling=True)

        if _rescale:
            _scaling_factor = _original_spacing / _spacing
            assert _scaling_factor > 1, "Scaling factor must be > 1"

            _width = int(width * _scaling_factor)
            _height = int(height * _scaling_factor)
        else:
            _width = width
            _height = height

        _patch = self.get_patch(
            x,
            y,
            _width,
            _height,
            spacing=_spacing,
            center=center,
            relative=relative,
        )

        if with_mask:
            assert center, "Mask sampling only works with center=True"
            _mask = self.mask_sampler.sample(
                self.annotation,
                (x, y),
                size=(_width, _height),
                ratio=_spacing
                / self.spacings[
                    0
                ],  # Test if this is correct by manual annotating a region in qupath
            )

        if _rescale:
            _patch = cv2.resize(_patch, (width, height))
            if with_mask:
                _mask = cv2.resize(_mask, (width, height))

        if with_mask:
            return _patch, _mask
        return _patch


if __name__ == "__main__":
    wsi = MultiResWholeSlideImage(
        "/Volumes/MyPassport/PhD/Cohorts/TCGA/SlidesPerPatient/TCGA-A8-A07C/TCGA-A8-A07C-01Z-00-DX1.1F069BCA-D2B3-49CF-81FD-9EBA49A3439F.svs"
    )

    data = wsi.get_data(
        29000, 17000, 512, 512, {"target": 0.5, "context": 2.0, "10x": 1.0}
    )
    print(data["target"].shape)
    print(data["context"].shape)
    print(data["10x"].shape)

    wsi = MyWholeSlideImage(
        "/Volumes/MyPassport/PhD/Cohorts/TCGA/SlidesPerPatient/TCGA-A8-A07C/TCGA-A8-A07C-01Z-00-DX1.1F069BCA-D2B3-49CF-81FD-9EBA49A3439F.svs"
    )

    data = wsi.get_data(29000, 17000, 512, 512, 0.5)
    print(data.shape)
