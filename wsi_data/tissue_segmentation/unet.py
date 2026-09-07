"""U-Net used by :class:`~wsi_data.tissue_segmentation.segmentor.CNNTissueSegmentor`.

A 7-level encoder/decoder U-Net with instance normalisation and a 2-class
(background, tissue) output head. The layer attribute names mirror a
pretrained checkpoint's ``state_dict`` keys exactly (after stripping the
``module.`` prefix left over from ``DataParallel`` training) -- renaming any
of ``down_block1``..``down_block7``, ``mid_conv1``/``bn1``/``mid_conv2``/``bn2``,
``up_block1``..``up_block6`` or ``last_conv1``/``last_bn``/``last_conv2`` (or
their nested ``conv1``/``bn1``/``conv2``/``bn2`` submodule attributes) would
break loading such a checkpoint.
"""

import torch
from torch import nn


class _UNetDownBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, down_size: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class _UNetUpBlock(nn.Module):
    def __init__(self, prev_channel: int, input_channel: int, output_channel: int):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            prev_channel + input_channel, output_channel, 3, padding=1
        )
        self.bn1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, prev_feature_map: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sampling(x)
        x = torch.cat((x, self.dropout(prev_feature_map)), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """7-level U-Net with instance norm, producing 2 output channels per pixel."""

    def __init__(self):
        super().__init__()
        self.down_block1 = _UNetDownBlock(3, 16, False)
        self.down_block2 = _UNetDownBlock(16, 32, True)
        self.down_block3 = _UNetDownBlock(32, 64, True)
        self.down_block4 = _UNetDownBlock(64, 128, True)
        self.down_block5 = _UNetDownBlock(128, 256, True)
        self.down_block6 = _UNetDownBlock(256, 512, True)
        self.down_block7 = _UNetDownBlock(512, 1024, True)

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(1024)

        self.up_block1 = _UNetUpBlock(512, 1024, 512)
        self.up_block2 = _UNetUpBlock(256, 512, 256)
        self.up_block3 = _UNetUpBlock(128, 256, 128)
        self.up_block4 = _UNetUpBlock(64, 128, 64)
        self.up_block5 = _UNetUpBlock(32, 64, 32)
        self.up_block6 = _UNetUpBlock(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.InstanceNorm2d(16)
        self.last_conv2 = nn.Conv2d(16, 2, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, an ``(N, 3, H, W)`` batch. Returns raw ``(N, 2, H, W)`` logits."""
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)
        x7 = self.relu(self.bn1(self.mid_conv1(x7)))
        x7 = self.relu(self.bn2(self.mid_conv2(x7)))
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x
