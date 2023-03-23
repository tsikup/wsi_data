from torch.utils.data import BatchSampler, DataLoader
from wsi_data.samplers import RandomHDF5BatchSampler


def fast_loader(
    dataset,
    batch_size=32,
    drop_last=False,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    worker_init_fn=None,
    prefetch_factor=None,
    persistent_workers=False,
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
    :param worker_init_fn: function to be called on each worker to initialize it
    :type worker_init_fn: function
    :param prefetch_factor: number of samples loaded in advance by each worker
    :type prefetch_factor: int
    :param persistent_workers: flag to indicate if the data loader will keep the workers
        alive after the dataloader has been destroyed
    :type persistent_workers: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=BatchSampler(
            RandomHDF5BatchSampler(dataset, batch_size),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
