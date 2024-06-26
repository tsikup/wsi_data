from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from wholeslidedata import WholeSlideAnnotation
from wholeslidedata.annotation.types import PolygonAnnotation as WSDPolygon
from wholeslidedata.data.files import WholeSlideAnnotationFile
from wsi_data.wholeslidedata.files import (
    MultiResWholeSlideImageFile,
    MyWholeSlideImageFile,
)
from wsi_data.wholeslidedata.wholeslideimage import (
    MultiResWholeSlideImage,
    MyWholeSlideImage,
)

from he_preprocessing.filter.filter import apply_filters_to_image
from he_preprocessing.utils.image import is_blurry, keep_tile, pad_image


class Single_WSI_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class is used to create a dataset for a single WSI based on wholeslidedata package annotations.
    """

    @staticmethod
    def collate_fn(batch):
        key = "target"
        batch = list(filter(lambda x: x[key]["img_array"] is not None, batch))
        if len(batch) == 0:
            return []
        return torch.utils.data.dataloader.default_collate(batch)

    def __init__(
        self,
        image_file: Union[
            MultiResWholeSlideImageFile,
            MyWholeSlideImageFile,
        ],
        annotations: List[WSDPolygon],
        tile_size: int = 512,
        spacing: Union[Dict[str, float], float] = 0.5,
        transform: Union[T.Compose, None] = None,
        filters2apply: Union[dict, None] = None,
        blurriness_threshold: Union[dict, int, None] = None,
        bluriness_mode: str = None,
        tissue_percentage: Union[dict, int, None] = None,
        constant_pad_value: int = 230,
        segmentation: bool = False,
        wsa: Union[WholeSlideAnnotation, WholeSlideAnnotationFile] = None,
    ):
        """
        :param image_file: A WholeSlideImageFile object
        :param annotations: A list of WSDPolygon objects
        """
        assert bluriness_mode in [
            "masked",
            "normalized",
            None,
        ], "Invalid bluriness mode. Valid options are `None`, `masked` and `normalized`."

        self.image_name = image_file.path.stem
        self.annotations = [ann.center for ann in annotations]
        self.dataset_size = len(annotations)
        self.image_file = image_file
        self.wsi: Union[MyWholeSlideImage, MultiResWholeSlideImage] = None
        self.wsa: Union[WholeSlideAnnotation, WholeSlideAnnotationFile] = wsa
        self.tile_size = tile_size

        if isinstance(spacing, dict):
            _spacing = {k: v for k, v in spacing.items() if v is not None}
            assert len(_spacing) > 0
            if len(_spacing) == 1:
                _spacing = _spacing[list(_spacing.keys())[0]]
            self.spacing = _spacing
        else:
            self.spacing = spacing

        self.multires = isinstance(self.spacing, dict)

        self.transform = transform
        self.filters2apply = filters2apply
        self.blurriness_threshold = blurriness_threshold
        self.bluriness_mode = bluriness_mode
        self.tissue_percentage = tissue_percentage
        self.constant_pad_value = constant_pad_value
        self.segmentation = segmentation

        self.assert_validity()

    def __len__(self):
        return self.dataset_size

    def assert_validity(self):
        if self.multires:
            assert isinstance(
                self.spacing, dict
            ), "Multiresolution spacing is required to be a dictionary."
            assert "target" in self.spacing.keys(), "Target spacing is required."
            assert "context" in self.spacing.keys(), "Context spacing is required."

            if self.blurriness_threshold is not None:
                assert isinstance(
                    self.blurriness_threshold, dict
                ), "Multiresolution blurriness is required to be a dictionary."
                assert (
                    "target" in self.blurriness_threshold.keys()
                ), "Target blurriness is required."
                assert (
                    "context" in self.blurriness_threshold.keys()
                ), "Context blurriness is required."
        else:
            assert isinstance(
                self.spacing, float
            ), "Only one spacing is allowed for Uniresolution-WSI."

            if self.blurriness_threshold is not None:
                assert isinstance(
                    self.blurriness_threshold, int
                ), "Blurriness threshold is required to be an integer for Uniresolution-WSI."

    def get_item(self, i):
        return self.__getitem__(i)

    def _preprocess(self, patch, blurriness_threshold=None, tissue_percentage=None):
        """Apply preprocessing to the patch."""
        patch = pad_image(
            patch,
            self.tile_size,
            value=self.constant_pad_value,
        )

        if tissue_percentage is not None:
            if not keep_tile(
                patch,
                self.tile_size,
                tissue_threshold=tissue_percentage,
                pad=True,
            ):
                return None

        if blurriness_threshold is not None and is_blurry(
            patch,
            threshold=blurriness_threshold,
            normalize=self.bluriness_mode == "normalized",
            masked=self.bluriness_mode == "masked",
        ):
            return None

        if self.filters2apply is not None:
            patch, _ = apply_filters_to_image(
                patch,
                roi_f=None,
                slide=self.image_name,
                filters2apply=self.filters2apply,
                save=False,
            )

        if self.transform is not None:
            patch = self.transform(patch)

        return patch

    def open_wsi(self):
        self.wsi = self.image_file.open()
        if isinstance(self.wsa, WholeSlideAnnotationFile):
            self.wsa = self.wsa.open()
            self.wsi.annotation = self.wsa

    def __getitem__(self, i):
        """
        :param i: The index of the item to be returned
        :return: A tuple of (target, context) patches if multiresolution is True, otherwise a single patch.
        """
        if self.wsi is None:
            self.open_wsi()

        annotation = self.annotations[i]

        if self.segmentation:
            data, mask = self.wsi.get_data(
                x=annotation[0],
                y=annotation[1],
                width=self.tile_size,
                height=self.tile_size,
                spacing=self.spacing,
                center=True,
                with_mask=True,
            )
        else:
            data = self.wsi.get_data(
                x=annotation[0],
                y=annotation[1],
                width=self.tile_size,
                height=self.tile_size,
                spacing=self.spacing,
                center=True,
                with_mask=False,
            )

        if self.multires:
            o_data = dict()
            for key in data.keys():
                o_data[key] = dict()
                o_data[key]["img_array"] = self._preprocess(
                    data[key],
                    blurriness_threshold=self.blurriness_threshold[key]
                    if self.blurriness_threshold is not None
                    else None,
                    tissue_percentage=self.tissue_percentage[key]
                    if self.tissue_percentage is not None
                    else None,
                )
                o_data[key]["x"] = annotation[0]
                o_data[key]["y"] = annotation[1]
                o_data[key]["spacing"] = self.spacing[key]
                if self.segmentation:
                    o_data[key]["mask_array"] = mask[key]
            return o_data
        else:
            o_data = {
                "target": {
                    "img_array": self._preprocess(
                        data,
                        blurriness_threshold=self.blurriness_threshold,
                        tissue_percentage=self.tissue_percentage,
                    ),
                    "x": annotation[0],
                    "y": annotation[1],
                    "spacing": self.spacing,
                }
            }
            if self.segmentation:
                o_data["target"]["mask_array"] = mask
            return o_data
