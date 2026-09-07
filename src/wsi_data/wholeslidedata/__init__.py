"""Extensions to the ``wholeslidedata`` package for multi-resolution sampling.

Note:
    The pre-1.0 ``wholeslidedata/utils.py`` grab-bag is now three modules:
    :mod:`~wsi_data.wholeslidedata.sources` for file discovery,
    :mod:`~wsi_data.wholeslidedata.batch` for sampler assembly, and
    :mod:`wsi_data.viz` for drawing.
"""

from wsi_data.wholeslidedata.batch import create_batch_sampler
from wsi_data.wholeslidedata.callbacks import MaskedTiledAnnotationCallback
from wsi_data.wholeslidedata.dataset import MultiResWholeSlideDataSet
from wsi_data.wholeslidedata.files import (
    MultiResWholeSlideImageFile,
    SingleResWholeSlideImageFile,
)
from wsi_data.wholeslidedata.samplers import (
    BatchOneTimeReferenceSampler,
    MultiResPatchSampler,
    MultiResSampleSampler,
    OrderedLabelOneTimeSampler,
    RandomOneTimeAnnotationSampler,
)
from wsi_data.wholeslidedata.sources import (
    FileType,
    get_files,
    whole_slide_files_from_folder_factory,
)
from wsi_data.wholeslidedata.wholeslideimage import (
    BaseWholeSlideImage,
    DetailTiling,
    MultiResWholeSlideImage,
    SingleResWholeSlideImage,
    TissueMaskResult,
)

__all__ = [
    "BaseWholeSlideImage",
    "BatchOneTimeReferenceSampler",
    "DetailTiling",
    "FileType",
    "MaskedTiledAnnotationCallback",
    "MultiResPatchSampler",
    "MultiResSampleSampler",
    "MultiResWholeSlideDataSet",
    "MultiResWholeSlideImage",
    "MultiResWholeSlideImageFile",
    "OrderedLabelOneTimeSampler",
    "RandomOneTimeAnnotationSampler",
    "SingleResWholeSlideImage",
    "SingleResWholeSlideImageFile",
    "TissueMaskResult",
    "create_batch_sampler",
    "get_files",
    "whole_slide_files_from_folder_factory",
]
