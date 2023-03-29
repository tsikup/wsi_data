import numpy as np
import torch
from torch.utils.data import Dataset


class FakeDataset(Dataset):
    def __init__(
        self, input_shape=[3, 512, 512], output_shape=[1, 512, 512], classes=[0, 1]
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.classes = classes

    def __len__(self):
        return int(1e6)

    def __getitem__(self, item):
        img = torch.rand(*self.input_shape)
        label = np.random.randint(
            low=min(self.classes), high=max(self.classes) + 1, size=self.output_shape
        )
        return img, label
