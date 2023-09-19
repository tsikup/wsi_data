from torch.utils.data import DataLoader
from wsi_data.samplers import (
    DistributedWeakShufflingBatchSampler,
    WeakShufflingBatchSampler,
    WeakShufflingSampler,
)


def to_tuple(x, shape=3):
    if isinstance(x, (tuple, list)):
        return list(x)
    else:
        return [
            x,
        ] * shape


def setall(d, keys, value):
    for k in keys:
        d[k] = value


def weak_shuffling_h5_fast_loader(
    dataset,
    batch_size=32,
    num_replicas=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=False,
    seed=42,
):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :param shuffle: flag to indicate if data will be shuffled
    :type shuffle: bool
    :param num_workers: number of workers to use for data loading
    :type num_workers: int
    :param pin_memory: flag to indicate if data will be pinned in memory
    :type pin_memory: bool
    :param prefetch_factor: number of samples loaded in advance by each worker
    :type prefetch_factor: int
    :param persistent_workers: flag to indicate if the data loader will keep the workers
        alive after the dataloader has been destroyed
    :type persistent_workers: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    sampler = WeakShufflingSampler(dataset, batch_size, seed=seed)

    batch_sampler = WeakShufflingBatchSampler(
        sampler,
        batch_size=batch_size * num_replicas,
        drop_last=drop_last,
    )

    distributed_batch_sampler = DistributedWeakShufflingBatchSampler(batch_sampler)

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=distributed_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


def crop_data(data, output_shape):

    if (data.shape[0] == output_shape[0]) and (data.shape[1] == output_shape[1]):
        return data

    cropx = (data.shape[0] - output_shape[0]) // 2
    cropy = (data.shape[1] - output_shape[1]) // 2

    if len(data.shape) == 2:
        return data[cropx:-cropx, cropy:-cropy]
    if len(data.shape) == 3:
        return data[cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 4:
        cropx = (data.shape[1] - output_shape[0]) // 2
        cropy = (data.shape[2] - output_shape[1]) // 2
        return data[:, cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 5:
        cropx = (data.shape[2] - output_shape[0]) // 2
        cropy = (data.shape[3] - output_shape[1]) // 2
        return data[:, :, cropx:-cropx, cropy:-cropy, :]
