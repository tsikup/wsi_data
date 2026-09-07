"""crop_data geometry and the albumentations/tensor pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import transforms as T

from wsi_data.transforms import (
    crop_data,
    to_chw_float_tensor,
    to_mask_tensor,
    transform_image_and_mask,
    transform_multires_image_and_mask,
)


class TestCropData:
    def test_axis_already_correct_is_not_emptied(self):
        """Regression: `data[0:-0]` is `data[0:0]`, which returned an empty array."""
        data = np.arange(12).reshape(4, 3)
        assert crop_data(data, (4, 1)).shape == (4, 1)
        assert crop_data(data, (4, 1)).size > 0

    def test_exact_size_returned_unchanged(self):
        data = np.arange(12).reshape(4, 3)
        assert crop_data(data, (4, 3)) is data

    def test_odd_difference_gives_exact_output_shape(self):
        """Regression: an odd size difference used to leave one extra row."""
        data = np.arange(12).reshape(4, 3)
        assert crop_data(data, (3, 3)).shape == (3, 3)

    @pytest.mark.parametrize(
        ("shape", "out", "expected"),
        [
            ((8, 8), (4, 4), (4, 4)),
            ((8, 8, 3), (4, 4), (4, 4, 3)),
            ((2, 8, 8, 3), (4, 4), (2, 4, 4, 3)),
            ((2, 5, 8, 8, 3), (4, 4), (2, 5, 4, 4, 3)),
        ],
    )
    def test_spatial_axes_per_rank(self, shape, out, expected):
        assert crop_data(np.zeros(shape), out).shape == expected

    def test_centres_the_crop(self):
        data = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(crop_data(data, (2, 2)), data[1:3, 1:3])

    def test_rejects_unsupported_rank(self):
        with pytest.raises(ValueError, match="rank 2-5"):
            crop_data(np.zeros(6), (2, 2))

    def test_rejects_upscaling(self):
        with pytest.raises(ValueError, match="only shrinks"):
            crop_data(np.zeros((2, 2)), (4, 4))


class TestTensorConversion:
    def test_matches_totensor_for_uint8(self, rng):
        image = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
        assert torch.equal(to_chw_float_tensor(image), T.ToTensor()(image))

    def test_matches_totensor_for_float(self, rng):
        image = rng.random((5, 7, 3)).astype(np.float32)
        assert torch.equal(to_chw_float_tensor(image), T.ToTensor()(image))

    def test_uint8_is_scaled_to_unit_range(self):
        image = np.full((2, 2, 3), 255, np.uint8)
        assert to_chw_float_tensor(image).max().item() == pytest.approx(1.0)

    def test_float_is_not_rescaled(self):
        image = np.full((2, 2, 3), 11.0, np.float32)
        assert to_chw_float_tensor(image).max().item() == pytest.approx(11.0)

    def test_greyscale_gains_a_channel_axis(self):
        assert to_chw_float_tensor(np.zeros((4, 5), np.uint8)).shape == (1, 4, 5)

    def test_mask_keeps_axes_and_becomes_uint8(self):
        mask = np.arange(6, dtype=np.int64).reshape(2, 3)
        out = to_mask_tensor(mask)
        assert out.shape == (2, 3)
        assert out.dtype == torch.uint8


class TestPipeline:
    def test_single_resolution_without_transform(self, rng):
        image = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
        out_image, out_mask = transform_image_and_mask(image)
        assert out_image.shape == (3, 4, 4)
        assert out_mask is None

    def test_single_resolution_with_mask(self, rng):
        image = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
        mask = rng.integers(0, 2, (4, 4), dtype=np.uint8)
        out_image, out_mask = transform_image_and_mask(image, mask)
        assert out_image.shape == (3, 4, 4)
        assert out_mask is not None
        assert out_mask.dtype == torch.uint8

    def test_multires_shares_one_geometric_draw(self):
        """A flip must hit every resolution identically, or they desynchronise."""
        import albumentations as A

        from wsi_data.augmentations import multires_additional_targets

        image = np.zeros((4, 4, 3), np.uint8)
        image[0, 0] = 255
        pipeline = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            additional_targets=multires_additional_targets(["target", "context"]),
            seed=1,
        )
        out, _ = transform_multires_image_and_mask(
            {"target": image, "context": image.copy()}, transform=pipeline
        )
        assert torch.equal(out["target"], out["context"])
        # The lit corner moved to the right-hand side in both.
        assert out["target"][0, 0, -1].item() == pytest.approx(1.0)

    def test_multires_requires_target_key(self):
        with pytest.raises(KeyError, match="target"):
            transform_multires_image_and_mask({"context": np.zeros((4, 4, 3))})

    def test_multires_mask_keys_lose_the_suffix(self, rng):
        images = {
            "target": rng.integers(0, 256, (4, 4, 3), dtype=np.uint8),
            "context": rng.integers(0, 256, (4, 4, 3), dtype=np.uint8),
        }
        masks = {
            "target": rng.integers(0, 2, (4, 4), dtype=np.uint8),
            "context": rng.integers(0, 2, (4, 4), dtype=np.uint8),
        }
        _, out_masks = transform_multires_image_and_mask(images, masks)
        assert out_masks is not None
        assert set(out_masks) == {"target", "context"}
