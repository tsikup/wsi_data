import glob
import os
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


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

        self.h5_dataset = None
        self.labels = None
        self.slides = glob.glob(os.path.join(data_dir, "*.h5"))

        assert len(self.slides) > 0, f"No hdf5 files found in {data_dir}"

        # determine dataset length and shape
        self.dataset_size = len(self.slides)
        with h5py.File(self.slides[0], "r") as f:
            # Total number of datapoints
            self.features_shape = f[self.data_cols["features_target"]].shape[1]
            self.labels_shape = 1

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.vstack(target)
        return [data, target]

    def read_hdf5(self, h5_path, load_ram=False):
        # Open hdf5 file where images and labels are stored
        h5_dataset = h5py.File(h5_path, "r")

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

    def get_item_on_slide_name(self, slide_name, _data_dir=None):
        if _data_dir is None:
            _data_dir = self.data_dir
        h5_path = os.path.join(_data_dir, slide_name)
        return self.read_hdf5(h5_path, load_ram=self.load_ram)

    def get_item(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        h5_path = self.slides[i]
        return self.read_hdf5(h5_path, load_ram=self.load_ram)

    def __len__(self):
        return self.dataset_size

    @property
    def shape(self):
        return [self.dataset_size, self.features_shape]
