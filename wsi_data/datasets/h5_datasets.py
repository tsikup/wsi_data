import glob
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Union

import albumentations as A
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class FeatureDatasetHDF5(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_cols: Dict[str, str],
        base_label: int = 0,
        load_ram: bool = True,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_cols: hdf5 dataset name, e.g.
                {
                    "features": "features",
                    "features_context": "features_context",
                    "labels": "labels"
                }
        :param base_label: label offset
        :param load_ram: load data into RAM
        """
        assert data_cols is not None

        self.data_dir = data_dir
        self.data_cols = data_cols
        self.base_label = base_label
        self.load_ram = load_ram

        self.multiresolution = len(self.data_cols) > 2

        assert (
            "features_target" in self.data_cols
            and self.data_cols["features_target"] is not None
        ), "`features_target` is required in `data_cols`"

        assert os.path.isdir(data_dir), f"{data_dir} is not a directory"

        # self.h5_dataset = None
        # self.labels = None
        self.slides = glob.glob(os.path.join(data_dir, "*.h5"))

        if len(self.slides) == 0:
            warnings.warn(f"No hdf5 files found in {data_dir}")

        # determine dataset length and shape
        self.dataset_size = len(self.slides)
        if self.dataset_size > 0:
            with h5py.File(self.slides[0], "r") as f:
                # Total number of datapoints
                self.features_shape = f[self.data_cols["features_target"]].shape[1]
                self.labels_shape = 1
        else:
            self.features_shape = None
            self.labels_shape = None

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.vstack(target)
        return [data, target]

    def read_hdf5(self, h5_path, load_ram=False):
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        # Open hdf5 file where images and labels are stored
        with h5py.File(h5_path, "r") as h5_dataset:
            features = h5_dataset[self.data_cols["features_target"]]
            features = torch.from_numpy(features[...]) if load_ram else features

            label = h5_dataset[self.data_cols["labels"]][0] - self.base_label
            label = torch.from_numpy(np.array([label], dtype=label.dtype))

            features = dict(features=features)

            if self.multiresolution:
                for key in self.data_cols:
                    if key not in ["features_target", "labels"]:
                        _tmp = h5_dataset[self.data_cols[key]]
                        _tmp = torch.from_numpy(_tmp[...]) if load_ram else _tmp
                        features[key] = _tmp

        return features, label

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        import pandas as pd
        import seaborn as sns

        labels = []
        for slide in self.slides:
            with h5py.File(slide, "r") as f:
                label = f[self.data_cols["labels"]][0]
                labels.append(label)
        if as_figure:
            labels = pd.DataFrame(labels, columns=["label"])
            if replace_names is not None:
                labels.replace(replace_names, inplace=True)
            fig = sns.displot(labels, x="label", shrink=0.8)
            label_dist = fig.figure
        else:
            label_dist = np.unique(labels, return_counts=True)
        return label_dist

    def get_item_on_slide_name(self, slide_name: Union[str, Path], _data_dir=None):
        if _data_dir is None:
            _data_dir = self.data_dir
        h5_path = os.path.join(_data_dir, slide_name)
        return self.read_hdf5(h5_path, load_ram=self.load_ram)

    def get_item(self, i: int):
        return self.__getitem__(i)

    def __getitem__(self, i: int):
        h5_path = self.slides[i]
        features, label = self.read_hdf5(h5_path, load_ram=self.load_ram)
        return {
            "features": features,
            "labels": label,
            "slide_name": Path(h5_path).name,
        }

    def __len__(self):
        return self.dataset_size

    @property
    def shape(self):
        return [self.dataset_size, self.features_shape]


class Single_H5_Image_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        h5_file: str = None,
        image_regex: str = "^x_",
        transform: Union[A.Compose, None] = None,
        pytorch_transform: Union[T.Compose, None] = None,
        channels_last: bool = False,
    ):
        """
        :param h5_file: hdf5 filename
        :param image_regex: regex to match image columns
        :param transform: albumentations image transform pipeline
        :param pytorch_transform: pytorch image transform pipeline
        :param channels_last: return tensor with channels at last index
        """
        if isinstance(h5_file, str):
            assert os.path.exists(h5_file)

        self.h5_file = h5_file
        self.h5_dataset = None
        self.images = None

        # Return tensor with channels at last index
        self.channels_last = channels_last

        # determine dataset length and shape
        with h5py.File(h5_file, "r") as f:
            # Total number of datapoints
            spacings = f.keys()
            p = re.compile(image_regex)
            self.data_cols = [s for s in spacings if p.match(s)]
            self.dataset_size = f[self.data_cols[0]].shape[0]
            self.image_shape = f[self.data_cols[0]].shape[1:]

        # Albumentations transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        # Pytorch transformation pipeline for the image
        self.pytorch_transform = pytorch_transform

    def open_hdf5(self):
        self.h5_dataset = h5py.File(self.h5_file, "r")
        self.images = dict()
        for res in self.data_cols:
            self.images[res] = self.h5_dataset[res]

    def close(self):
        if self.h5_dataset is not None:
            self.h5_dataset.close()

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if not hasattr(self, "h5_dataset") or self.h5_dataset is None:
            self.open_hdf5()

        images = dict()
        for res in self.data_cols:
            image = self.images[res][i]

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            if self.pytorch_transform is not None:
                image = self.pytorch_transform(image)

            if self.channels_last:
                image = image.permute(1, 2, 0)

            images[res] = image

        return images

    def __len__(self):
        return self.dataset_size


class DatasetHDF5(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last: bool = False,
        segmentation: bool = True,
        image_only: bool = False,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_name: hdf5 filename
        :param data_cols: dictionary with h5 dataset names for images, labels
        :param transform: albumentations image transform pipeline
        :param pytorch_transform: pytorch image transform pipeline
        :param channels_last: return tensor with channels at last index
        :param segmentation: segmentation dataset
        :param image_only: return only images
        """
        self.h5_dataset = None
        self.images = None
        self.labels = None

        self.data_cols = data_cols
        self.image_only = image_only
        self.segmentation = segmentation
        self.channels_last = channels_last
        self.h5_path = os.path.join(data_dir, data_name)

        assert os.path.isdir(data_dir), f"Directory {data_dir} does not exist"
        assert os.path.exists(self.h5_path), f"File {self.h5_path} does not exist"

        # determine dataset length and shape
        with h5py.File(self.h5_path, "r") as f:
            assert "images" in self.data_cols, "key `images` not in data_cols"
            self.dataset_size = f[self.data_cols["images"]].shape[0]
            self.image_shape = f[self.data_cols["images"]].shape[1:]
            if not self.image_only:
                assert "labels" in self.data_cols, "key `labels` not in data_cols"
                self.labels_shape = (
                    f[self.data_cols["labels"]].shape[1:] if self.segmentation else 1
                )

        self.transform = transform
        self.pytorch_transform = pytorch_transform

    def open_hdf5(self):
        self.h5_dataset = h5py.File(self.h5_path, "r")
        self.images = self.h5_dataset[self.data_cols["images"]]
        if not self.image_only:
            if self.data_cols["labels"] is not None:
                self.labels = self.h5_dataset[self.data_cols["labels"]]

    def get_label_distribution(self, replace_names: Dict = None, as_figure=False):
        import pandas as pd
        import seaborn as sns

        if self.data_cols["labels"] is not None:
            with h5py.File(self.h5_path, "r") as f:
                labels = f[self.data_cols["labels"]][...]
                if as_figure:
                    labels = pd.DataFrame(labels, columns=["label"])
                    if replace_names is not None:
                        labels.replace(replace_names, inplace=True)
                    fig = sns.displot(labels, x="label", shrink=0.8)
                    label_dist = fig.figure
                else:
                    label_dist = np.unique(labels, return_counts=True)
            return label_dist
        return None

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if not hasattr(self, "h5_dataset") or self.h5_dataset is None:
            self.open_hdf5()

        image = self.images[i]
        if not self.image_only:
            try:
                label = np.array([self.labels[i]])
            except IndexError:
                label = np.array([self.labels[0]])

        if not self.image_only and self.segmentation and len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)

        if self.segmentation and not self.image_only:
            if self.transform is not None:
                transformed = self.transform(image=image, mask=label)
                image = transformed["image"]
                label = transformed["mask"]
            image, label = self.pytorch_transform(img=image, mask=label)
        else:
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            if not self.image_only:
                label = torch.from_numpy(label)
            if self.pytorch_transform is not None:
                image = self.pytorch_transform(image)

        # By default, ToTensorV2 returns C,H,W image and H,W,C mask
        if self.channels_last:
            image = image.permute(1, 2, 0)
        else:
            if self.segmentation and not self.image_only:
                label = label.permute(2, 0, 1)

        if self.image_only:
            return image

        return image, label.to(torch.uint8)

    def __len__(self):
        return self.dataset_size


class ImageOnlyDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(ImageOnlyDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            image_only=True,
        )


class SegmentationDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(SegmentationDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            segmentation=True,
        )


class ClassificationDatasetHDF5(DatasetHDF5):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        data_cols: Dict[str, str],
        transform: Union[A.Compose, None],
        pytorch_transform: Union[T.Compose, None],
        channels_last=False,
    ):
        super(ClassificationDatasetHDF5, self).__init__(
            data_dir=data_dir,
            data_name=data_name,
            data_cols=data_cols,
            transform=transform,
            pytorch_transform=pytorch_transform,
            channels_last=channels_last,
            segmentation=False,
        )
