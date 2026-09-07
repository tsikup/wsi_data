"""CNN-based whole-slide tissue segmentation.

An alternative to he_preprocessing's optical-density/edge-based tissue
detection: runs a pretrained :class:`~wsi_data.tissue_segmentation.unet.UNet`
over a whole-slide thumbnail using overlap-tile inference with spline-window
blending ("smooth tiled predictions") for seam-free stitching across patch
boundaries.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from scipy.signal.windows import triang
from skimage.morphology import remove_small_objects
from torch.utils.data import DataLoader, Dataset

from wsi_data.tissue_segmentation.unet import UNet


@dataclass
class CNNTissueSegmentorConfig:
    """Configuration for :class:`CNNTissueSegmentor`.

    Attributes:
        tile_size: Side length, in pixels, of the sliding-window patches fed to the model.
        subdivisions: Overlap factor; the stride between patch origins is
            ``tile_size / subdivisions`` (``2.0`` means 50% overlap in both directions).
        batch_size: Number of patches per forward pass.
        min_object_size: Minimum connected-component size, in pixels, kept in the
            final mask; smaller tissue islands are discarded as noise.
        num_workers: Subprocesses used to extract and preprocess (CLAHE) patches in
            parallel with the GPU/CPU forward pass. ``0`` runs preprocessing in the
            main process (matches the original script's default, but serialises
            preprocessing and inference).
        pin_memory: Use pinned host memory for the patch batches, which speeds up
            the host-to-device copy. Only takes effect when running on CUDA.
        prefetch_factor: Batches each worker prepares ahead of time. Only used
            when ``num_workers > 0``.
    """

    tile_size: int = 512
    subdivisions: float = 2.0
    batch_size: int = 32
    min_object_size: int = 2500
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2


class _PatchDataset(Dataset):
    """Extracts and preprocesses one overlap-tile patch per index.

    Note:
        CLAHE reproduces the pretrained checkpoint's training-time
        preprocessing exactly, including applying it to HSV channel 0 (hue),
        not value/luminance -- a bug in the original training pipeline. The
        checkpoint was trained on images preprocessed this way, so it is
        reproduced as-is; "fixing" it would shift the input distribution away
        from what the model was trained on.

        ``cv2.CLAHE`` objects are not picklable, so the CLAHE instance is
        built lazily on first use rather than in ``__init__`` -- this lets
        the dataset be handed to ``DataLoader`` worker subprocesses (which
        pickle it under the ``spawn`` start method) without error; each
        worker builds its own instance on first access.
    """

    def __init__(self, padded_image: np.ndarray, origins: list, tile_size: int):
        self.padded_image = padded_image
        self.origins = origins
        self.tile_size = tile_size
        self._clahe = None

    def __len__(self) -> int:
        return len(self.origins)

    def _get_clahe(self) -> cv2.CLAHE:
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return self._clahe

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        y, x = self.origins[index]
        tile_size = self.tile_size
        patch = self.padded_image[y : y + tile_size, x : x + tile_size]

        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = self._get_clahe().apply(hsv[:, :, 0])
        patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        patch = patch.astype(np.float32) / 255.0
        patch = (patch - 0.5) / 0.5
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1))
        return torch.from_numpy(patch), y, x


def _spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """1-D squared-triangular blending window ("smooth tiled predictions" trick)."""
    intersection = window_size // 4
    wind_outer = (abs(2 * triang(window_size)) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    window = wind_inner + wind_outer
    return window / np.average(window)


def _window_2d(window_size: int, power: int = 2) -> np.ndarray:
    """2-D blending window, shape ``(window_size, window_size, 1)``."""
    window = _spline_window(window_size, power)
    window = np.expand_dims(np.expand_dims(window, 1), 1)
    return window * window.transpose(1, 0, 2)


def _reflect_pad(
    image: np.ndarray, tile_size: int, stride: int
) -> Tuple[np.ndarray, int]:
    """Reflect-pad ``image`` so the sliding window covers it with no leftover edge."""
    margin = tile_size - stride
    height, width = image.shape[:2]
    extra_height = (stride - (height + 2 * margin - tile_size) % stride) % stride
    extra_width = (stride - (width + 2 * margin - tile_size) % stride) % stride
    padded = np.pad(
        image,
        ((margin, margin + extra_height), (margin, margin + extra_width), (0, 0)),
        mode="reflect",
    )
    return padded, margin


class CNNTissueSegmentor:
    """Overlap-tile CNN tissue segmentation using a pretrained :class:`UNet`.

    Instantiate once (it loads and holds the model in memory/on-device) and
    reuse across slides via :meth:`predict`.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[CNNTissueSegmentorConfig] = None,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        self.config = config or CNNTissueSegmentorConfig()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UNet().to(self.device)
        self.model.load_state_dict(self._load_state_dict(model_path))
        self.model.eval()
        self._window_cache: Dict[int, np.ndarray] = {}

        if self.device.type == "cuda":
            # All patches share one fixed shape (config.tile_size), so letting
            # cuDNN benchmark convolution algorithms once and reuse them pays off.
            torch.backends.cudnn.benchmark = True

    def _load_state_dict(self, model_path: Union[str, Path]) -> dict:
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Older torch releases do not accept `weights_only`.
            checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = (
            checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        )
        # Strip the `module.` prefix left by DataParallel training.
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    def _window(self) -> np.ndarray:
        tile_size = self.config.tile_size
        if tile_size not in self._window_cache:
            self._window_cache[tile_size] = _window_2d(tile_size)
        return self._window_cache[tile_size]

    def _make_loader(self, dataset: _PatchDataset) -> DataLoader:
        kwargs = dict(
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
        )
        if self.config.num_workers > 0:
            # Only valid alongside multiprocessing workers.
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **kwargs)

    @torch.no_grad()
    def _predict_logits(self, image: np.ndarray) -> np.ndarray:
        """Run overlap-tile inference, returning per-pixel (background, tissue) logits.

        Patch extraction and CLAHE preprocessing run in ``DataLoader`` worker
        subprocesses (when ``config.num_workers > 0``), overlapping with the
        forward pass on the current batch instead of blocking on it.
        """
        tile_size = self.config.tile_size
        stride = int(round(tile_size / self.config.subdivisions))
        padded, margin = _reflect_pad(image, tile_size, stride)
        window = self._window()

        origins = [
            (y, x)
            for y in range(0, padded.shape[0] - tile_size + 1, stride)
            for x in range(0, padded.shape[1] - tile_size + 1, stride)
        ]
        loader = self._make_loader(_PatchDataset(padded, origins, tile_size))

        canvas = np.zeros((*padded.shape[:2], 2), dtype=np.float32)
        for batch, ys, xs in loader:
            batch = batch.to(self.device, non_blocking=True)
            logits = self.model(batch)
            logits = logits.permute(0, 2, 3, 1).cpu().numpy()
            for i in range(logits.shape[0]):
                y, x = int(ys[i]), int(xs[i])
                canvas[y : y + tile_size, x : x + tile_size] += logits[i] * window

        canvas /= self.config.subdivisions**2
        height, width = image.shape[:2]
        return canvas[margin : margin + height, margin : margin + width]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Segment tissue in an RGB whole-slide thumbnail.

        Args:
            image: An ``(H, W, 3)`` ``uint8`` RGB image -- a downsampled
                whole-slide thumbnail, not a single small tile.

        Returns:
            An ``(H, W)`` ``uint8`` mask, ``255`` for tissue and ``0`` for
            background, with connected components smaller than
            ``config.min_object_size`` removed.
        """
        logits = self._predict_logits(image)
        mask = np.argmax(logits, axis=-1) == 1
        mask = remove_small_objects(mask, min_size=self.config.min_object_size)
        return (mask * 255).astype(np.uint8)
