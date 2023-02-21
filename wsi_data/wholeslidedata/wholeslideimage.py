import numpy as np
from pathlib import Path
from typing import Union, Dict
from numpy import ndarray
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.image.backend import WholeSlideImageBackend
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation import utils as annotation_utils

from .annotation_parser import QuPathAnnotationParser
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
            self._annotation_parser = QuPathAnnotationParser()
            self.annotation = WholeSlideAnnotation(
                annotation_path=annotation_path,
                labels=labels,
                parser=self._annotation_parser,
            )

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

    def get_tissue_mask(self, spacing=32, return_contours=True):
        downsample = self.get_downsampling_from_spacing(spacing)
        return ut_image.get_slide_tissue_mask(
            slide_path=self.path,
            downsample=downsample,
            return_contours=return_contours,
        )

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

    def get_data(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        spacings: Dict[str, float],
        center: bool = True,
        relative: bool = False,
    ) -> Union[
        dict[str, Union[ndarray, ndarray, list[Union[ndarray, ndarray]]]],
        dict[str, Union[ndarray, ndarray]],
    ]:
        """Extracts multi-resolution patches/regions from the wholeslideimage
        Args:
            x (int): x value
            y (int): y value
            width (int): width of region
            height (int): height of region
            spacings (list of float): spacing/resolution of the patch at target, context and details level
            center (bool, optional): if x,y values are centres or top left coordinated. Defaults to True.
            relative (bool, optional): if x,y values are a reference to the dimensions of the specified spacing. Defaults to False.
        Returns:
            np.ndarray: numpy patch
        """

        assert isinstance(spacings, dict), "Spacings must be a dictionary"

        _spacings = spacings.copy()

        data = dict()

        for key, value in _spacings.items():
            _spacings[key] = self.get_real_spacing(value)

        for key, spacing in _spacings.items():
            if key == "details":
                continue
            data[key] = self.get_patch(
                x,
                y,
                width,
                height,
                spacing=spacing,
                center=center,
                relative=relative,
            )

        # Details
        x_details = None
        if "details" in _spacings:
            x_details = []

            (
                num_details_patches,
                total_i,
                total_j,
                downsampling_target,
                downsampling_details,
            ) = self.get_num_details(width, height, _spacings)

            if center:
                # Get top left coords of target resolution
                downsampling = int(
                    self.get_downsampling_from_spacing(_spacings["target"])
                )
                x, y = x - downsampling * (width // 2), y - downsampling * (height // 2)

            for idx in range(num_details_patches):
                i = int(idx // total_i)
                j = int(idx % total_j)

                rel_coord_x = j * width * downsampling_target
                rel_coord_y = i * height * downsampling_target

                coord_x = x + rel_coord_x
                coord_y = y + rel_coord_y
                x_details.append(
                    self.get_patch(
                        coord_x,
                        coord_y,
                        width,
                        height,
                        spacing=_spacings["details"],
                        center=False,
                        relative=relative,
                    )
                )

            data["details"] = x_details

        return data
