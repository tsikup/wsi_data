import torch
from torch.utils.data import Dataset
from typing import Dict, Union, List
from torchvision import transforms as T
from wholeslidedata.annotation.structures import Polygon as WSDPolygon

from he_preprocessing.filter.filter import apply_filters_to_image
from he_preprocessing.utils.image import keep_tile, is_blurry, pad_image
from wholeslidedata.source.files import WholeSlideImageFile

from wsi_data import MultiResWholeSlideImageFile


class Single_WSI_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    This class is used to create a dataset for a single WSI based on wholeslidedata package annotations.
    """

    @staticmethod
    def collate_fn(batch):
        key = "target"
        batch = list(filter(lambda x: x[key] is not None, batch))
        if len(batch) == 0:
            return []
        return torch.utils.data.dataloader.default_collate(batch)

    def __init__(
        self,
        image_file: Union[MultiResWholeSlideImageFile, WholeSlideImageFile],
        annotations: List[WSDPolygon],
        tile_size: int = 512,
        spacing: Union[Dict[str, float], float] = 0.5,
        transform: Union[T.Compose, None] = None,
        filters2apply: Union[dict, None] = None,
        blurriness_threshold: Union[dict, int, None] = None,
        tissue_percentage: Union[dict, int, None] = None,
        constant_pad_value: int = 230,
    ):
        """
        :param image_file: A WholeSlideImageFile object
        :param annotations: A list of WSDPolygon objects
        """
        self.image_name = image_file.path.stem
        self.annotations = [ann.center for ann in annotations]
        self.dataset_size = len(annotations)
        self.image_file = image_file
        self.wsi = None

        self.tile_size = tile_size
        self.spacing = spacing
        self.multires = isinstance(self.spacing, dict)

        self.transform = transform
        self.filters2apply = filters2apply
        self.blurriness_threshold = blurriness_threshold
        self.tissue_percentage = tissue_percentage
        self.constant_pad_value = constant_pad_value

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
            patch, threshold=blurriness_threshold, normalize=True
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

    def __getitem__(self, i):
        """
        :param i: The index of the item to be returned
        :return: A tuple of (target, context) patches if multiresolution is True, otherwise a single patch.
        """
        if self.wsi is None:
            self.open_wsi()

        annotation = self.annotations[i]
        if self.multires:
            data = self.wsi.get_data(
                x=annotation[0],
                y=annotation[1],
                width=self.tile_size,
                height=self.tile_size,
                spacings=self.spacing,
                center=True,
            )

            o_data = dict()
            for key in data.keys():
                o_data[key] = self._preprocess(
                    data[key],
                    blurriness_threshold=self.blurriness_threshold[key],
                    tissue_percentage=self.tissue_percentage[key],
                )
                if key == "target" and o_data[key] is None:
                    return dict.fromkeys(self.spacing.keys(), None)

            return o_data
        else:
            data = self.wsi.get_patch(
                x=annotation[0],
                y=annotation[1],
                width=self.tile_size,
                height=self.tile_size,
                spacing=self.spacing,
                center=True,
            )

            return {
                "target": self._preprocess(
                    data,
                    blurriness_threshold=self.blurriness_threshold,
                    tissue_percentage=self.tissue_percentage,
                )
            }
