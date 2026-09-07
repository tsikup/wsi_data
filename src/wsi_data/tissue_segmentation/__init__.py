"""CNN-based whole-slide tissue segmentation."""

from wsi_data.tissue_segmentation.segmentor import (
    CNNTissueSegmentor,
    CNNTissueSegmentorConfig,
)
from wsi_data.tissue_segmentation.unet import UNet

__all__ = [
    "CNNTissueSegmentor",
    "CNNTissueSegmentorConfig",
    "UNet",
]
