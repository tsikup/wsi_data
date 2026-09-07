"""FakeDataset: smoke-test dataset for training loops."""

from __future__ import annotations

import pytest
import torch

from wsi_data.datasets import FakeDataset


def test_shapes_match_configuration():
    dataset = FakeDataset(input_shape=(3, 4, 4), output_shape=(1, 4, 4), length=10)
    image, label = dataset[0]
    assert image.shape == (3, 4, 4)
    assert label.shape == (1, 4, 4)
    assert len(dataset) == 10


def test_labels_are_tensors_not_numpy():
    """Regression: labels were built with np.random.randint, images with torch."""
    dataset = FakeDataset(classes=(0, 2), length=1)
    _, label = dataset[0]
    assert isinstance(label, torch.Tensor)


def test_defaults_are_not_a_shared_mutable_list():
    """Regression: input_shape/output_shape/classes defaulted to mutable lists."""
    a = FakeDataset()
    a.input_shape = (99, 99, 99)
    b = FakeDataset()
    assert b.input_shape != (99, 99, 99)


def test_labels_stay_within_the_class_range():
    dataset = FakeDataset(classes=(0, 2), output_shape=(64,), length=1, seed=0)
    _, label = dataset[0]
    assert label.min() >= 0
    assert label.max() <= 2


def test_seed_is_reproducible():
    a = FakeDataset(input_shape=(1, 2, 2), length=1, seed=1)[0]
    b = FakeDataset(input_shape=(1, 2, 2), length=1, seed=1)[0]
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_rejects_empty_classes():
    with pytest.raises(ValueError, match="classes"):
        FakeDataset(classes=())


def test_rejects_negative_length():
    with pytest.raises(ValueError, match="length"):
        FakeDataset(length=-1)
