"""DataLoader construction for weak-shuffled HDF5 datasets.

Note:
    Named ``wsi_data.utils`` before 1.0. The dead ``setall`` helper is gone,
    and ``to_tuple`` moved into :mod:`wsi_data.augmentations`, its only user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from torch.utils.data import DataLoader

from wsi_data.samplers import (
    DistributedWeakShufflingBatchSampler,
    WeakShufflingBatchSampler,
    WeakShufflingSampler,
)

if TYPE_CHECKING:
    from collections.abc import Sized

    from torch.utils.data import Dataset

__all__ = ["weak_shuffling_h5_fast_loader"]


def weak_shuffling_h5_fast_loader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    *,
    num_replicas: int = 1,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    seed: int = 42,
    rank: int | None = None,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader[Any]:
    """Build a loader that reads an HDF5 dataset in shuffled contiguous blocks.

    Reading an HDF5 dataset costs roughly one index lookup per *call*, so
    ``data[i : i + batch_size]`` is nearly as cheap as ``data[i]`` while a
    fully shuffled access pattern pays that cost per element. This loader
    keeps reads contiguous by shuffling blocks of ``batch_size`` rather than
    individual indices -- "weak shuffling" -- which recovers most of the
    sequential read throughput without loading the corpus into RAM.

    The returned loader yields already-batched items, so its ``batch_size``
    is ``None`` and the batching happens in the sampler.

    Note:
        This function could not be imported at all before 1.0: it lives in a
        module whose sampler imports failed because ``wsi_data.samplers``
        exported nothing.

    Args:
        dataset: A dataset backed by ``.h5`` files.
        batch_size: Items per batch, per replica.
        num_replicas: Distributed world size. The sampler groups
            ``batch_size * num_replicas`` items and each rank takes its share.
        num_workers: Worker processes for loading.
        prefetch_factor: Batches prefetched per worker. Only valid with
            ``num_workers > 0``.
        seed: Seed for block shuffling.
        rank: This process's rank. Read from the process group when ``None``;
            with ``num_replicas=1`` it defaults to ``0``.
        drop_last: Drop a trailing partial batch.
        pin_memory: Use pinned host memory, which speeds up transfers to CUDA.
        persistent_workers: Keep workers alive between epochs.

    Returns:
        The configured loader. Call ``loader.sampler.set_epoch(epoch)`` each
        epoch to reshuffle.

    Raises:
        ValueError: If ``prefetch_factor`` is set without workers.
    """
    if prefetch_factor is not None and num_workers == 0:
        msg = "prefetch_factor requires num_workers > 0"
        raise ValueError(msg)

    # `Dataset`'s stub does not declare `__len__` (it's only a map-style-dataset
    # convention, not part of its type), but every HDF5-backed dataset this
    # loader is meant for implements it -- which is exactly what
    # `WeakShufflingSampler` needs.
    sampler = WeakShufflingSampler(cast("Sized", dataset), batch_size, seed=seed)
    batch_sampler = WeakShufflingBatchSampler(
        sampler,
        batch_size=batch_size * num_replicas,
        drop_last=drop_last,
    )
    distributed_batch_sampler = DistributedWeakShufflingBatchSampler(
        batch_sampler,
        num_replicas=num_replicas,
        # With a single replica there is no process group to query.
        rank=0 if (rank is None and num_replicas == 1) else rank,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=distributed_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
