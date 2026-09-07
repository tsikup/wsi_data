"""Streaming per-channel mean/standard-deviation estimation over image batches.

Two-pass statistics over a large tile corpus are impractical, so
:func:`get_channels_sums_from_ndarray` accumulates per-channel sums and sums
of squares one batch at a time and :func:`calculate_mean_and_std` reduces the
running totals at the end.
"""

from __future__ import annotations

import numpy as np

__all__ = ["calculate_mean_and_std", "get_channels_sums_from_ndarray"]

#: Rank of a single `(H, W, C)`/`(C, H, W)` image, versus `_BATCH_RANK` for a batch.
_IMAGE_RANK = 3
_BATCH_RANK = 4


def calculate_mean_and_std(
    channels_sum: np.ndarray,
    channels_squared_sum: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce accumulated per-channel sums to a mean and standard deviation.

    Uses ``std = sqrt(E[X^2] - E[X]^2)``.

    Note:
        The variance is clamped at zero before the square root. With
        near-constant channels the two large accumulated terms can differ by
        a few ULP in the wrong direction, and the pre-1.0 code took
        ``(...) ** 0.5`` of that negative value, yielding a silent ``nan``
        standard deviation.

    Args:
        channels_sum: Running total from :func:`get_channels_sums_from_ndarray`.
        channels_squared_sum: Running total of squares from the same function.
        count: Number of *images* accumulated into the running totals -- not
            the number of pixels, since the accumulators hold per-image means.

    Returns:
        A ``(mean, std)`` pair, each of shape ``(channels,)``.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        msg = f"count must be positive, got {count}"
        raise ValueError(msg)
    mean = channels_sum / count
    variance = channels_squared_sum / count - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean, std


def get_channels_sums_from_ndarray(
    data: np.ndarray,
    *,
    channels_last: bool = False,
    max_value: float = 255.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate per-channel means and mean-of-squares for one image or batch.

    A 3-D input is treated as a single image and reduced over its spatial
    axes. A 4-D input is treated as a batch: each image is reduced over its
    spatial axes and the per-image results are then **summed** over the batch,
    so the returned totals are directly addable across calls and
    :func:`calculate_mean_and_std` divides by the accumulated image count.

    Note:
        Only ``uint8`` input is divided by ``max_value``. Floating-point input
        is assumed to be normalised already and is passed through untouched --
        matching pre-1.0 behaviour, which this preserves deliberately so that
        existing corpus statistics remain reproducible.

    Args:
        data: A ``(H, W, C)``/``(C, H, W)`` image or a ``(N, H, W, C)``/
            ``(N, C, H, W)`` batch.
        channels_last: Whether the channel axis is last.
        max_value: Divisor applied to ``uint8`` input to bring it into ``[0, 1]``.

    Returns:
        A ``(channels_sum, channels_squared_sum)`` pair, each of shape
        ``(channels,)``.

    Raises:
        ValueError: If ``data`` is neither 3-D nor 4-D. The pre-1.0 code left
            its axis variable unbound on any other rank and failed with an
            ``UnboundLocalError`` from inside numpy instead.
    """
    if data.ndim == _IMAGE_RANK:
        axes = (0, 1) if channels_last else (1, 2)
        aggregate = False
    elif data.ndim == _BATCH_RANK:
        axes = (1, 2) if channels_last else (2, 3)
        aggregate = True
    else:
        msg = (
            f"data must be a 3-D image or a 4-D batch, got rank {data.ndim} "
            f"with shape {data.shape}"
        )
        raise ValueError(msg)

    if data.dtype == np.uint8:
        data = data / max_value

    # Mean over height and width only, keeping channels (and batch) separate.
    channels_sum = data.mean(axis=axes)
    channels_squared_sum = (data**2).mean(axis=axes)

    if aggregate:
        channels_sum = channels_sum.sum(axis=0)
        channels_squared_sum = channels_squared_sum.sum(axis=0)

    return channels_sum, channels_squared_sum
