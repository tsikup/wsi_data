"""Weak-shuffling and class-balanced samplers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wsi_data.labels import LabelDistribution
from wsi_data.samplers import (
    DistributedWeakShufflingBatchSampler,
    WeakShufflingBatchSampler,
    WeakShufflingDistributedSampler,
    WeakShufflingSampler,
    get_weighted_random_sampler,
)


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class TestWeakShufflingSampler:
    @pytest.mark.parametrize("length", [8, 9, 10, 11, 12])
    def test_yields_every_index_exactly_once(self, length):
        sampler = WeakShufflingSampler(_Sized(length), batch_size=4, seed=1)
        assert sorted(sampler) == list(range(length))
        assert len(sampler) == length

    def test_blocks_stay_contiguous(self):
        sampler = WeakShufflingSampler(_Sized(12), batch_size=4, seed=1)
        indices = list(sampler)
        for start in range(0, 12, 4):
            block = indices[start : start + 4]
            assert block == list(range(block[0], block[0] + 4))

    def test_blocks_are_actually_shuffled(self):
        sampler = WeakShufflingSampler(_Sized(64), batch_size=4, seed=1)
        assert list(sampler) != list(range(64))

    def test_partial_tail_comes_last_in_order(self):
        sampler = WeakShufflingSampler(_Sized(10), batch_size=4, seed=1)
        assert list(sampler)[-2:] == [8, 9]

    def test_seed_is_deterministic(self):
        a = list(WeakShufflingSampler(_Sized(32), 4, seed=5))
        b = list(WeakShufflingSampler(_Sized(32), 4, seed=5))
        assert a == b

    def test_epoch_changes_the_order(self):
        sampler = WeakShufflingSampler(_Sized(32), 4, seed=5)
        first = list(sampler)
        sampler.shuffle_indices(epoch=1)
        assert list(sampler) != first

    def test_rejects_non_positive_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            WeakShufflingSampler(_Sized(4), 0)


class TestDistributedSplit:
    def test_does_not_mutate_the_caller_list(self):
        """Regression: `indices += ...` mutated the caller's batch in place."""
        batch = [1, 2, 3]
        sampler = WeakShufflingDistributedSampler(batch, num_replicas=2, rank=0)
        list(sampler)
        assert batch == [1, 2, 3]

    def test_iteration_is_repeatable(self):
        """Regression: the in-place growth made a second pass differ."""
        sampler = WeakShufflingDistributedSampler([1, 2, 3], num_replicas=2, rank=0)
        assert list(sampler) == list(sampler)

    def test_ranks_partition_an_even_batch(self):
        batch = [0, 1, 2, 3]
        left = list(WeakShufflingDistributedSampler(batch, 2, 0))
        right = list(WeakShufflingDistributedSampler(batch, 2, 1))
        assert sorted(left + right) == batch

    def test_pads_an_uneven_batch_so_ranks_match(self):
        batch = [0, 1, 2]
        left = list(WeakShufflingDistributedSampler(batch, 2, 0))
        right = list(WeakShufflingDistributedSampler(batch, 2, 1))
        assert len(left) == len(right) == 2

    def test_drop_last_truncates(self):
        batch = [0, 1, 2]
        out = list(WeakShufflingDistributedSampler(batch, 2, 0, drop_last=True))
        assert len(out) == 1

    def test_rejects_empty_batch(self):
        with pytest.raises(ValueError, match="must not be empty"):
            WeakShufflingDistributedSampler([], 1, 0)

    @pytest.mark.parametrize(("replicas", "rank"), [(2, 2), (2, -1)])
    def test_rejects_bad_rank(self, replicas, rank):
        with pytest.raises(ValueError, match="rank must be"):
            WeakShufflingDistributedSampler([1, 2], replicas, rank)

    def test_requires_explicit_ranks_without_a_process_group(self):
        with pytest.raises(RuntimeError, match=r"torch\.distributed"):
            WeakShufflingDistributedSampler([1, 2])


class TestDistributedBatchSampler:
    def test_yields_this_ranks_share_of_each_batch(self):
        sampler = WeakShufflingSampler(_Sized(16), batch_size=2, seed=1)
        batch_sampler = WeakShufflingBatchSampler(
            sampler, batch_size=4, drop_last=False
        )
        distributed = DistributedWeakShufflingBatchSampler(
            batch_sampler, num_replicas=2, rank=0
        )
        batches = list(distributed)
        assert len(batches) == len(distributed)
        assert all(len(batch) == 2 for batch in batches)

    def test_set_epoch_reshuffles(self):
        sampler = WeakShufflingSampler(_Sized(32), batch_size=2, seed=1)
        batch_sampler = WeakShufflingBatchSampler(
            sampler, batch_size=4, drop_last=False
        )
        distributed = DistributedWeakShufflingBatchSampler(
            batch_sampler, num_replicas=1, rank=0
        )
        first = list(distributed)
        distributed.set_epoch(3)
        assert list(distributed) != first


class _LabelledDataset:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def get_label_distribution(self, label_key="labels"):
        return LabelDistribution.from_labels(self.labels, label_key)


class TestWeightedRandomSampler:
    def test_handles_non_contiguous_labels(self):
        """Regression: weights were indexed by label value, not class position.

        With labels {0, 2}, `np.unique` yields two counts and the pre-1.0
        `weight[2]` raised IndexError.
        """
        sampler = get_weighted_random_sampler(_LabelledDataset([0, 0, 2, 2, 2]))
        assert len(sampler) == 5

    def test_weights_are_inverse_class_frequency(self):
        sampler = get_weighted_random_sampler(_LabelledDataset([0, 0, 0, 1]))
        weights = torch.as_tensor(sampler.weights)
        # Three samples of class 0 at 1/3, one of class 1 at 1/1.
        assert weights[:3].tolist() == pytest.approx([1 / 3] * 3)
        assert weights[3].item() == pytest.approx(1.0)

    def test_balances_draws(self):
        torch.manual_seed(0)
        sampler = get_weighted_random_sampler(_LabelledDataset([0] * 90 + [1] * 10))
        labels = np.array([0] * 90 + [1] * 10)
        drawn = labels[list(sampler)]
        # Rare class should be drawn far more often than its 10% base rate.
        assert 0.3 < drawn.mean() < 0.7

    def test_rejects_empty_dataset(self):
        with pytest.raises(ValueError, match="no labels"):
            get_weighted_random_sampler(_LabelledDataset([]))
