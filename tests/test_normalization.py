"""Streaming per-channel mean/std estimation."""

from __future__ import annotations

import numpy as np
import pytest

from wsi_data.normalization import (
    calculate_mean_and_std,
    get_channels_sums_from_ndarray,
)


class TestChannelSums:
    def test_matches_direct_statistics_for_one_image(self, rng):
        image = rng.random((4, 5, 3)).astype(np.float64)
        sums, squares = get_channels_sums_from_ndarray(image, channels_last=True)
        np.testing.assert_allclose(sums, image.mean(axis=(0, 1)))
        np.testing.assert_allclose(squares, (image**2).mean(axis=(0, 1)))

    def test_channels_first_uses_the_other_axes(self, rng):
        image = rng.random((3, 4, 5))
        sums, _ = get_channels_sums_from_ndarray(image, channels_last=False)
        np.testing.assert_allclose(sums, image.mean(axis=(1, 2)))

    def test_batch_totals_equal_the_sum_of_per_image_totals(self, rng):
        batch = rng.random((4, 6, 6, 3))
        batch_sums, batch_squares = get_channels_sums_from_ndarray(
            batch, channels_last=True
        )
        per_image = [
            get_channels_sums_from_ndarray(image, channels_last=True) for image in batch
        ]
        np.testing.assert_allclose(
            batch_sums, np.sum([s for s, _ in per_image], axis=0)
        )
        np.testing.assert_allclose(
            batch_squares, np.sum([q for _, q in per_image], axis=0)
        )

    def test_uint8_is_scaled_but_float_is_not(self):
        as_uint8 = np.full((2, 2, 3), 255, np.uint8)
        sums, _ = get_channels_sums_from_ndarray(as_uint8, channels_last=True)
        np.testing.assert_allclose(sums, [1.0, 1.0, 1.0])

        as_float = np.full((2, 2, 3), 255.0)
        sums, _ = get_channels_sums_from_ndarray(as_float, channels_last=True)
        np.testing.assert_allclose(sums, [255.0, 255.0, 255.0])

    @pytest.mark.parametrize("shape", [(4,), (2, 2), (2, 2, 2, 2, 2)])
    def test_rejects_unsupported_rank(self, shape):
        """Regression: any other rank left the axis variable unbound."""
        with pytest.raises(ValueError, match="3-D image or a 4-D batch"):
            get_channels_sums_from_ndarray(np.zeros(shape))


class TestMeanAndStd:
    def test_recovers_the_true_statistics_over_a_corpus(self, rng):
        batch = rng.random((32, 8, 8, 3))
        sums, squares = get_channels_sums_from_ndarray(batch, channels_last=True)
        mean, std = calculate_mean_and_std(sums, squares, count=len(batch))
        np.testing.assert_allclose(mean, batch.mean(axis=(0, 1, 2)), atol=1e-12)
        # Per-image means are averaged, so this matches the pooled std only
        # approximately; a loose tolerance is the point of the check.
        np.testing.assert_allclose(std, batch.std(axis=(0, 1, 2)), atol=0.02)

    def test_constant_channel_gives_zero_not_nan(self):
        """Regression: float error made the variance slightly negative -> nan std."""
        image = np.full((64, 64, 3), 0.4)
        sums, squares = get_channels_sums_from_ndarray(image, channels_last=True)
        _, std = calculate_mean_and_std(sums, squares, count=1)
        assert not np.isnan(std).any()
        np.testing.assert_allclose(std, 0.0, atol=1e-8)

    def test_rejects_non_positive_count(self):
        with pytest.raises(ValueError, match="count must be positive"):
            calculate_mean_and_std(np.zeros(3), np.zeros(3), count=0)
