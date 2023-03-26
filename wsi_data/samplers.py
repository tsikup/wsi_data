import math
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DistributedSampler, Sampler


class WeakShufflingSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2.
    Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """

    def __init__(self, dataset, batch_size, seed=42):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = None
        self.seed = seed
        self.shuffle_indices()

    def __len__(self):
        return self.dataset_length

    def shuffle_indices(self, epoch=None):
        # deterministically shuffle based on seed and possibly epoch
        _seed = self.seed + epoch if epoch is not None else self.seed
        g = torch.Generator()
        g.manual_seed(_seed)
        self.batch_ids = torch.randperm(int(self.n_batches), generator=g).tolist()

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(
                int(self.n_batches) * self.batch_size, self.dataset_length
            )
            for index in idx:
                yield int(index)


class WeakShufflingBatchSampler(BatchSampler):
    def __init__(self, sampler: WeakShufflingSampler, batch_size: int, drop_last: bool):
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.sampler = sampler

    def shuffle_indices(self, epoch=None):
        # deterministically shuffle based on seed and possibly epoch
        self.sampler.shuffle_indices(epoch=epoch)


class WeakShufflingDistributedSampler(object):
    def __init__(
        self,
        indices: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.indices = indices
        self.epoch = 0
        self.drop_last = drop_last
        self.total_size = None
        self.num_samples = None
        self.num_replicas = num_replicas
        self.rank = rank
        self.setup_dist()

    def setup_dist(self):
        if self.num_replicas is None:
            self.num_replicas = dist.get_world_size()
        if self.rank is None:
            self.rank = dist.get_rank()
        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(self.rank, self.num_replicas - 1)
            )
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.indices) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = self.indices

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        _rank_size = len(indices) // self.num_replicas
        _rank_start = self.rank * _rank_size
        _rank_end = _rank_start + _rank_size

        # subsample
        indices = indices[_rank_start:_rank_end]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedWeakShufflingBatchSampler(object):
    def __init__(self, batch_sampler: WeakShufflingBatchSampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.epoch = 0
        self.kwargs = kwargs
        self.num_replicas = kwargs.get("num_replicas", None)
        self.rank = kwargs.get("rank", None)
        self.setup_dist()

    def setup_dist(self):
        if self.num_replicas is None:
            self.num_replicas = dist.get_world_size()
            self.kwargs["num_replicas"] = self.num_replicas
        if self.rank is None:
            self.rank = dist.get_rank()
            self.kwargs["rank"] = self.rank
        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(self.rank, self.num_replicas - 1)
            )

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(WeakShufflingDistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.batch_sampler.shuffle_indices(epoch=self.epoch)
