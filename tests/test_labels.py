"""LabelDistribution: counting, positions and text rendering."""

from __future__ import annotations

import numpy as np
import pytest

from wsi_data.labels import LabelDistribution


class TestFromLabels:
    def test_counts_single_key(self):
        dist = LabelDistribution.from_labels(np.array([0, 0, 0, 1]), "labels")
        np.testing.assert_array_equal(dist.values.ravel(), [0, 1])
        np.testing.assert_array_equal(dist.counts, [3, 1])
        assert dist.n_classes == 2
        assert dist.n_samples == 4

    def test_handles_non_contiguous_labels(self):
        dist = LabelDistribution.from_labels(np.array([0, 2, 2]), "labels")
        np.testing.assert_array_equal(dist.values.ravel(), [0, 2])
        np.testing.assert_array_equal(dist.counts, [1, 2])

    def test_positions_index_into_values(self):
        labels = np.array([0, 2, 0, 2, 2])
        dist = LabelDistribution.from_labels(labels, "labels")
        np.testing.assert_array_equal(dist.values[dist.positions].ravel(), labels)

    def test_joint_distribution(self):
        labels = np.array([[0, 1], [2, 0], [0, 1]])
        dist = LabelDistribution.from_labels(labels, ("labels", "labels_group"))
        assert dist.n_classes == 2
        assert dist.as_mapping() == {(0, 1): 2, (2, 0): 1}

    def test_fractions_sum_to_one(self):
        dist = LabelDistribution.from_labels(np.array([0, 0, 1, 2]), "labels")
        assert dist.fractions.sum() == pytest.approx(1.0)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="labels must be"):
            LabelDistribution.from_labels(np.zeros((3, 2)), "labels")


class TestRendering:
    def test_describe_has_one_row_per_class(self):
        dist = LabelDistribution.from_labels(np.array([0, 0, 0, 1]), "labels")
        lines = dist.describe(width=4).splitlines()
        assert lines[0].startswith("labels")
        assert "4 samples, 2 classes" in lines[0]
        assert len(lines) == 3
        assert "75.0%" in lines[1]

    def test_describe_needs_no_plotting_dependency(self, monkeypatch):
        """The text summary must not import matplotlib or seaborn."""
        import sys

        monkeypatch.setitem(sys.modules, "seaborn", None)
        dist = LabelDistribution.from_labels(np.array([1, 1, 2]), "labels")
        assert "labels" in dist.describe()

    def test_class_names_are_readable(self):
        dist = LabelDistribution.from_labels(np.array([[0, 1]]), ("a", "b"))
        assert dist.class_names == ["0, 1"]


class TestMapping:
    def test_single_key_mapping_uses_scalars(self):
        dist = LabelDistribution.from_labels(np.array([5, 5, 7]), "labels")
        assert dist.as_mapping() == {5: 2, 7: 1}
