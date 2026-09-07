"""Weak-shuffling samplers for HDF5-backed datasets.

Reading an HDF5 dataset costs roughly one index lookup per *call*, not per
element, so ``data[i : i + batch_size]`` is nearly as cheap as ``data[i]``.
Fully shuffling indices destroys that, but loading the whole corpus into RAM
to shuffle is not an option either. Weak shuffling is the middle ground:
elements stay in their original contiguous blocks and the *blocks* are
shuffled.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sized

__all__ = [
    "DistributedWeakShufflingBatchSampler",
    "WeakShufflingBatchSampler",
    "WeakShufflingDistributedSampler",
    "WeakShufflingSampler",
]


def _resolve_dist(num_replicas: int | None, rank: int | None) -> tuple[int, int]:
    """Fill in ``num_replicas``/``rank`` from the process group and validate them."""
    if num_replicas is None or rank is None:
        if not dist.is_available() or not dist.is_initialized():
            msg = (
                "num_replicas and rank must be given explicitly when "
                "torch.distributed is not initialised"
            )
            raise RuntimeError(msg)
        num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        rank = dist.get_rank() if rank is None else rank
    if num_replicas < 1:
        msg = f"num_replicas must be at least 1, got {num_replicas}"
        raise ValueError(msg)
    if not 0 <= rank < num_replicas:
        msg = f"rank must be in [0, {num_replicas - 1}], got {rank}"
        raise ValueError(msg)
    return num_replicas, rank


class WeakShufflingSampler(Sampler[int]):
    """Yield indices in shuffled contiguous blocks of ``batch_size``.

    For a dataset ``[0, 1, 2, 3]`` with ``batch_size=2`` the blocks are
    ``[0, 1]`` and ``[2, 3]``; the blocks are shuffled but each block's
    contents stay contiguous and in order, so downstream HDF5 reads remain
    sequential.

    A trailing partial block, when the dataset length is not a multiple of
    ``batch_size``, is always yielded last and is never shuffled.

    Args:
        dataset: The dataset to sample from; only its length is used.
        batch_size: Block size to keep contiguous.
        seed: Base seed for block shuffling.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    def __init__(self, dataset: Sized, batch_size: int, seed: int = 42) -> None:
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_full_blocks = self.dataset_length // batch_size
        self.remainder = self.dataset_length - self.n_full_blocks * batch_size
        self.seed = seed
        self.block_ids: list[int] = []
        self.shuffle_indices()

    def __len__(self) -> int:
        return self.dataset_length

    def shuffle_indices(self, epoch: int | None = None) -> None:
        """Reshuffle the block order deterministically from ``seed`` and ``epoch``."""
        generator = torch.Generator()
        generator.manual_seed(self.seed if epoch is None else self.seed + epoch)
        self.block_ids = torch.randperm(
            self.n_full_blocks, generator=generator
        ).tolist()

    def __iter__(self) -> Iterator[int]:
        for block_id in self.block_ids:
            start = block_id * self.batch_size
            yield from range(start, start + self.batch_size)
        # The trailing partial block cannot be shuffled with the rest without
        # changing block sizes, so it is emitted last, in order.
        if self.remainder:
            yield from range(self.n_full_blocks * self.batch_size, self.dataset_length)


class WeakShufflingBatchSampler(BatchSampler):
    """Group a :class:`WeakShufflingSampler`'s output into batches.

    Adds only :meth:`shuffle_indices`, forwarding to the wrapped sampler so an
    epoch reshuffle can be triggered through the batch sampler.
    """

    def __init__(
        self,
        sampler: WeakShufflingSampler,
        batch_size: int,
        *,
        drop_last: bool,
    ) -> None:
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.sampler: WeakShufflingSampler = sampler

    def shuffle_indices(self, epoch: int | None = None) -> None:
        """Reshuffle the underlying sampler's blocks."""
        self.sampler.shuffle_indices(epoch=epoch)


class WeakShufflingDistributedSampler:
    """Split one batch of indices across distributed ranks.

    Pads the batch to a multiple of ``num_replicas`` (or truncates it, with
    ``drop_last``) and returns this rank's contiguous slice, so every rank
    gets the same count.

    Note:
        The pre-1.0 implementation bound ``indices = self.indices`` and then
        did ``indices += ...``, which mutates a list in place. Since the list
        passed in is owned by the caller -- normally the batch that
        :class:`DistributedWeakShufflingBatchSampler` just yielded -- iterating
        grew the caller's batch, and iterating twice produced different
        results. This copies before padding.

    Args:
        indices: Indices making up a single batch.
        num_replicas: World size; read from the process group when ``None``.
        rank: This process's rank; read from the process group when ``None``.
        drop_last: Truncate rather than pad to an even split.

    Raises:
        ValueError: If ``indices`` is empty, or ``rank``/``num_replicas`` are
            inconsistent.
        RuntimeError: If ranks must be inferred but ``torch.distributed`` is
            not initialised.
    """

    def __init__(
        self,
        indices: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        *,
        drop_last: bool = False,
    ) -> None:
        if not indices:
            msg = "indices must not be empty"
            raise ValueError(msg)
        self.indices = indices
        self.epoch = 0
        self.drop_last = drop_last
        self.num_replicas, self.rank = _resolve_dist(num_replicas, rank)

        if self.drop_last and len(indices) % self.num_replicas != 0:
            # Truncate to the nearest evenly divisible length so each rank
            # receives the same amount of data.
            self.num_samples = math.ceil(
                (len(indices) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        indices = list(self.indices)  # never mutate the caller's list

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                repeats = math.ceil(padding_size / len(indices))
                indices += (indices * repeats)[:padding_size]

        rank_size = len(indices) // self.num_replicas
        start = self.rank * rank_size
        return iter(indices[start : start + rank_size])


class DistributedWeakShufflingBatchSampler:
    """Yield each weak-shuffled batch already split down to this rank's share.

    Args:
        batch_sampler: The batch sampler whose batches are to be split.
        num_replicas: World size; read from the process group when ``None``.
        rank: This process's rank; read from the process group when ``None``.
        drop_last: Passed through to :class:`WeakShufflingDistributedSampler`.
    """

    def __init__(
        self,
        batch_sampler: WeakShufflingBatchSampler,
        num_replicas: int | None = None,
        rank: int | None = None,
        *,
        drop_last: bool = False,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.epoch = 0
        self.drop_last = drop_last
        self.num_replicas, self.rank = _resolve_dist(num_replicas, rank)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batch_sampler:
            yield list(
                WeakShufflingDistributedSampler(
                    batch,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    drop_last=self.drop_last,
                )
            )

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle blocks for ``epoch``, as PyTorch's distributed samplers do."""
        self.epoch = epoch
        self.batch_sampler.shuffle_indices(epoch=epoch)
