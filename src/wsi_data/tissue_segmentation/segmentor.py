"""CNN-based whole-slide tissue segmentation.

An alternative to he_preprocessing's optical-density/edge-based tissue
detection: runs a pretrained :class:`~wsi_data.tissue_segmentation.unet.UNet`
over a whole-slide thumbnail using overlap-tile inference with spline-window
blending ("smooth tiled predictions") for seam-free stitching across patch
boundaries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from scipy.signal.windows import triang
from skimage.morphology import remove_small_objects
from torch.utils.data import DataLoader, Dataset

from wsi_data.tissue_segmentation.unet import UNet

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

__all__ = ["CNNTissueSegmentor", "CNNTissueSegmentorConfig"]

#: `_spline_window` needs a distinguishable inner and outer region.
_MIN_WINDOW_SIZE = 4

#: `predict` only accepts a 3-D `(H, W, C)` array...
_HWC_RANK = 3
#: ...with exactly 3 (RGB) channels. Coincidentally the same number as
#: `_HWC_RANK`, but a distinct constant since they check different things.
_RGB_CHANNELS = 3


@dataclass
class CNNTissueSegmentorConfig:
    """Configuration for :class:`CNNTissueSegmentor`.

    Attributes:
        tile_size: Side length, in pixels, of the sliding-window patches fed to
            the model.
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
        cudnn_benchmark: Let cuDNN benchmark convolution algorithms. Every
            patch shares one shape, so benchmarking once and reusing the
            result pays off. Applied only for the duration of a
            :meth:`CNNTissueSegmentor.predict` call and then restored, rather
            than left switched on globally as before 1.0. Has no effect off CUDA.
    """

    tile_size: int = 512
    subdivisions: float = 2.0
    batch_size: int = 32
    min_object_size: int = 2500
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    cudnn_benchmark: bool = True


class _PatchDataset(Dataset[tuple[torch.Tensor, int, int]]):
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

    def __init__(
        self,
        padded_image: np.ndarray,
        origins: Sequence[tuple[int, int]],
        tile_size: int,
    ) -> None:
        self.padded_image = padded_image
        self.origins = origins
        self.tile_size = tile_size
        self._clahe: cv2.CLAHE | None = None

    def __len__(self) -> int:
        return len(self.origins)

    def _get_clahe(self) -> cv2.CLAHE:
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return self._clahe

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
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
    """1-D squared-triangular blending window ("smooth tiled predictions" trick).

    Raises:
        ValueError: If ``window_size`` is under 4, where the window has no
            distinguishable inner and outer region. The pre-1.0 code computed
            ``intersection = window_size // 4`` and then sliced
            ``wind_inner[-intersection:]``; with ``intersection == 0`` that is
            ``[-0:]``, i.e. the *whole* array, so the window came out all
            zeros and blended every prediction to nothing.
    """
    if window_size < _MIN_WINDOW_SIZE:
        msg = f"window_size must be at least {_MIN_WINDOW_SIZE}, got {window_size}"
        raise ValueError(msg)

    intersection = window_size // 4
    wind_outer = (abs(2 * triang(window_size)) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    window = wind_inner + wind_outer
    # `scipy.signal.windows.triang` (untyped) contaminates the whole
    # computation above with `Any`; `np.asarray` recovers a concrete type.
    return np.asarray(window / np.average(window))


def _window_2d(window_size: int, power: int = 2) -> np.ndarray:
    """2-D blending window, shape ``(window_size, window_size, 1)``."""
    window = _spline_window(window_size, power)
    window = np.expand_dims(np.expand_dims(window, 1), 1)
    return np.asarray(window * window.transpose(1, 0, 2))


def _reflect_pad(
    image: np.ndarray, tile_size: int, stride: int
) -> tuple[np.ndarray, int]:
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


@contextmanager
def _cudnn_benchmark(*, enabled: bool) -> Iterator[None]:
    """Set ``torch.backends.cudnn.benchmark`` for a block, then restore it.

    Note:
        Before 1.0 the segmentor set this flag globally in its constructor and
        never restored it, so merely constructing one changed convolution
        behaviour for every other model in the process.
    """
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


class CNNTissueSegmentor:
    """Overlap-tile CNN tissue segmentation using a pretrained :class:`UNet`.

    Instantiate once -- it loads and holds the model on-device -- and reuse
    across slides via :meth:`predict`.

    Args:
        model_path: Path to the checkpoint, either a bare ``state_dict`` or a
            dict with a ``"state_dict"`` entry.
        config: Inference settings. Defaults to
            :class:`CNNTissueSegmentorConfig`'s own defaults.
        device: Device to run on. Defaults to CUDA when available.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: CNNTissueSegmentorConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or CNNTissueSegmentorConfig()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UNet().to(self.device)
        self.model.load_state_dict(self._load_state_dict(model_path))
        self.model.eval()

    def _load_state_dict(self, model_path: str | Path) -> dict[str, Any]:
        """Load a checkpoint's tensors, stripping any ``DataParallel`` prefix.

        Note:
            Loads with ``weights_only=True``. The pre-1.0 code passed
            ``weights_only=False``, which unpickles arbitrary objects and so
            executes arbitrary code from the checkpoint file; the payload here
            is a plain tensor ``state_dict``, which the safe loader handles.
            The ``except TypeError`` fallback for torch releases without the
            argument is also gone -- it has existed since torch 1.13, well
            below this package's floor.
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        return {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }

    @cached_property
    def _window(self) -> np.ndarray:
        """The blending window for this instance's tile size."""
        return _window_2d(self.config.tile_size)

    def _make_loader(self, dataset: _PatchDataset) -> DataLoader[Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": False,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory and self.device.type == "cuda",
        }
        if self.config.num_workers > 0:
            # Only valid alongside multiprocessing workers.
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **kwargs)

    @torch.no_grad()
    def _predict_logits(self, image: np.ndarray) -> np.ndarray:
        """Run overlap-tile inference, returning per-pixel (background, tissue) logits.

        Patch extraction and CLAHE preprocessing run in ``DataLoader`` worker
        subprocesses when ``config.num_workers > 0``, overlapping with the
        forward pass instead of blocking on it.
        """
        tile_size = self.config.tile_size
        # `round()` of a float with no `ndigits` already returns `int`.
        stride = round(tile_size / self.config.subdivisions)
        padded, margin = _reflect_pad(image, tile_size, stride)
        window = self._window

        origins = [
            (y, x)
            for y in range(0, padded.shape[0] - tile_size + 1, stride)
            for x in range(0, padded.shape[1] - tile_size + 1, stride)
        ]
        loader = self._make_loader(_PatchDataset(padded, origins, tile_size))

        canvas = np.zeros((*padded.shape[:2], 2), dtype=np.float32)
        for batch, ys, xs in loader:
            logits = self.model(batch.to(self.device, non_blocking=True))
            logits = logits.permute(0, 2, 3, 1).cpu().numpy()
            for i in range(logits.shape[0]):
                y, x = int(ys[i]), int(xs[i])
                canvas[y : y + tile_size, x : x + tile_size] += logits[i] * window

        # No normalisation by `subdivisions ** 2` here: the pre-1.0 code
        # divided the accumulated logits by it, but the only consumer is the
        # `argmax` in `predict`, which is invariant to a positive scale factor.
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

        Raises:
            ValueError: If ``image`` is not an ``(H, W, 3)`` array, or is
                smaller than one tile.
        """
        if image.ndim != _HWC_RANK or image.shape[2] != _RGB_CHANNELS:
            msg = f"image must be (H, W, 3) RGB, got shape {image.shape}"
            raise ValueError(msg)

        with _cudnn_benchmark(
            enabled=self.config.cudnn_benchmark and self.device.type == "cuda"
        ):
            logits = self._predict_logits(image)

        mask = np.argmax(logits, axis=-1) == 1
        # scikit-image 0.26 renamed `min_size` to `max_size` and moved the
        # boundary: `min_size=N` kept components with area >= N, while
        # `max_size=N` *removes* components with area <= N. `max_size=N - 1`
        # reproduces the old `min_size=N` exactly (area >= N <=> area > N - 1).
        mask = remove_small_objects(mask, max_size=self.config.min_object_size - 1)
        return np.asarray((mask * 255).astype(np.uint8))
