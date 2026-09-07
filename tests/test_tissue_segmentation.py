"""CNNTissueSegmentor: window blending, checkpoint loading, and prediction shape."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wsi_data.tissue_segmentation import CNNTissueSegmentor, CNNTissueSegmentorConfig
from wsi_data.tissue_segmentation.segmentor import (
    _reflect_pad,
    _spline_window,
    _window_2d,
)
from wsi_data.tissue_segmentation.unet import UNet


class TestSplineWindow:
    def test_rejects_too_small_a_window(self):
        """Regression: `wind_inner[-0:] = 0` zeroed the whole window for size < 4."""
        with pytest.raises(ValueError, match="at least 4"):
            _spline_window(2)

    @pytest.mark.parametrize("size", [4, 8, 16, 512])
    def test_window_is_not_all_zero(self, size):
        window = _spline_window(size)
        assert window.max() > 0

    def test_window_2d_is_symmetric(self):
        window = _window_2d(16)
        np.testing.assert_allclose(window, window.transpose(1, 0, 2))


class TestReflectPad:
    def test_covers_the_image_with_no_leftover_edge(self):
        image = np.zeros((100, 130, 3), np.uint8)
        tile_size, stride = 64, 32
        padded, margin = _reflect_pad(image, tile_size, stride)
        usable = padded.shape[0] - tile_size
        assert usable % stride == 0
        usable_w = padded.shape[1] - tile_size
        assert usable_w % stride == 0
        assert margin == tile_size - stride


@pytest.fixture
def tiny_checkpoint(tmp_path):
    path = tmp_path / "unet.pt"
    torch.save(UNet().state_dict(), path)
    return path


@pytest.fixture
def dataparallel_checkpoint(tmp_path):
    """A checkpoint with the `module.` prefix DataParallel training leaves behind."""
    path = tmp_path / "unet_dp.pt"
    state = {f"module.{k}": v for k, v in UNet().state_dict().items()}
    torch.save({"state_dict": state}, path)
    return path


class TestCheckpointLoading:
    def test_loads_a_bare_state_dict(self, tiny_checkpoint):
        segmentor = CNNTissueSegmentor(tiny_checkpoint, device="cpu")
        assert next(segmentor.model.parameters()).device.type == "cpu"

    def test_strips_the_dataparallel_prefix(self, dataparallel_checkpoint):
        # Loading only succeeds if `module.` was stripped so keys line up.
        segmentor = CNNTissueSegmentor(dataparallel_checkpoint, device="cpu")
        assert isinstance(segmentor.model, UNet)

    def test_uses_weights_only_loading(self, tiny_checkpoint, monkeypatch):
        """Regression: `weights_only=False` executes arbitrary pickled code."""
        seen = {}
        real_load = torch.load

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", spy)
        CNNTissueSegmentor(tiny_checkpoint, device="cpu")
        assert seen.get("weights_only") is True


class TestPredict:
    def test_output_shape_and_dtype(self, tiny_checkpoint, rng):
        # tile_size must be >= 128 for this UNet: it has 6 downsampling stages
        # (2**6 = 64), and InstanceNorm needs > 1 spatial element at the deepest,
        # 1/64-scale bottleneck.
        config = CNNTissueSegmentorConfig(tile_size=128, subdivisions=2.0, batch_size=2)
        segmentor = CNNTissueSegmentor(tiny_checkpoint, config=config, device="cpu")
        image = rng.integers(0, 256, (150, 180, 3), dtype=np.uint8)
        mask = segmentor.predict(image)
        assert mask.shape == (150, 180)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 255}

    def test_rejects_non_rgb_input(self, tiny_checkpoint):
        segmentor = CNNTissueSegmentor(tiny_checkpoint, device="cpu")
        with pytest.raises(ValueError, match="H, W, 3"):
            segmentor.predict(np.zeros((10, 10), np.uint8))

    def test_cudnn_benchmark_is_restored_after_predict(self, tiny_checkpoint, rng):
        """Regression: the pre-1.0 constructor set this flag globally forever."""
        before = torch.backends.cudnn.benchmark
        config = CNNTissueSegmentorConfig(tile_size=128, batch_size=2)
        segmentor = CNNTissueSegmentor(tiny_checkpoint, config=config, device="cpu")
        segmentor.predict(rng.integers(0, 256, (150, 150, 3), dtype=np.uint8))
        assert torch.backends.cudnn.benchmark == before
