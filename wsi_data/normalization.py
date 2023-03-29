import numpy as np


def calculate_mean_and_std(channels_sum, channels_squared_sum, count):
    mean = channels_sum / count
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / count - mean**2) ** 0.5
    return mean, std


def get_channels_sums_from_ndarray(
    data: np.ndarray, channels_last=False, max_value=255.0
):
    aggregate = False
    if len(data.shape) == 3:
        if channels_last:
            channels = (0, 1)
        else:
            channels = (1, 2)
    elif len(data.shape) == 4:
        aggregate = True
        if channels_last:
            channels = (1, 2)
        else:
            channels = (2, 3)

    if data.dtype == np.uint8:
        data = data / max_value

    # Mean over height and width, but not over the channels and batch
    channels_sums = data.mean(axis=channels)
    channels_squared_sum = (data**2).mean(axis=channels)

    # Aggregate over batch
    if aggregate:
        channels_sums = channels_sums.sum(axis=0)
        channels_squared_sum = channels_squared_sum.sum(axis=0)

    return channels_sums, channels_squared_sum
