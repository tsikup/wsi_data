"""HDF5-backed datasets: feature bags, tile images and tile stores."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from wsi_data.datasets import (
    MISSING_LABEL,
    FeatureDatasetHDF5,
    ImageDatasetHDF5,
    TileDatasetHDF5,
)

FEATURE_COLS = {"features_target": "features", "features_context": "features_context"}


class TestFeatureDataset:
    def test_reads_features_and_labels(self, feature_dir):
        directory, labels, _ = feature_dir
        dataset = FeatureDatasetHDF5(directory, {**FEATURE_COLS, "labels": "labels"})
        assert len(dataset) == len(labels)
        item = dataset[0]
        assert set(item) >= {"features", "labels", "slide_name", "coords", "index"}
        assert item["labels"].item() == labels[0]

    def test_features_are_materialised_tensors(self, feature_dir):
        """Regression: `load_ram=False` returned handles from a closed file.

        Reading one raised `RuntimeError: Unable to synchronously get
        dataspace (invalid dataset identifier)`.
        """
        directory, _, _ = feature_dir
        dataset = FeatureDatasetHDF5(directory, {**FEATURE_COLS, "labels": "labels"})
        features = dataset[0]["features"]
        assert isinstance(features["features"], torch.Tensor)
        # Readable outside any open-file context.
        assert features["features"].sum() is not None
        assert features["features"].shape[1] == 16

    def test_target_column_is_renamed_to_features(self, feature_dir):
        directory, _, _ = feature_dir
        dataset = FeatureDatasetHDF5(directory, {**FEATURE_COLS, "labels": "labels"})
        keys = set(dataset[0]["features"])
        assert keys == {"features", "features_context"}

    def test_missing_label_uses_the_ignore_sentinel(self, feature_dir_unlabelled):
        """Regression: the sentinel was built as uint8(-100).

        Under NumPy 1.x that silently wrapped to 156 -- a plausible class
        index -- and under NumPy 2 it raises OverflowError.
        """
        dataset = FeatureDatasetHDF5(
            feature_dir_unlabelled, {**FEATURE_COLS, "labels": "labels"}
        )
        label = dataset[0]["labels"]
        assert label.item() == MISSING_LABEL
        assert label.dtype == torch.int64

    def test_base_label_offsets_labels(self, feature_dir):
        directory, labels, _ = feature_dir
        dataset = FeatureDatasetHDF5(
            directory, {**FEATURE_COLS, "labels": "labels"}, base_label=1
        )
        assert dataset[0]["labels"].item() == labels[0] - 1

    def test_label_distribution(self, feature_dir):
        directory, labels, _ = feature_dir
        dataset = FeatureDatasetHDF5(directory, {**FEATURE_COLS, "labels": "labels"})
        dist = dataset.get_label_distribution()
        assert dist.as_mapping() == {0: 2, 2: 2}
        np.testing.assert_array_equal(np.sort(dist.labels.ravel()), np.sort(labels))

    def test_joint_label_distribution(self, feature_dir):
        directory, _, _ = feature_dir
        dataset = FeatureDatasetHDF5(
            directory,
            {**FEATURE_COLS, "labels": "labels", "labels_group": "labels_group"},
        )
        dist = dataset.get_label_distribution(["labels", "labels_group"])
        assert dist.keys == ("labels", "labels_group")
        assert sum(dist.counts) == 4

    def test_n_groups(self, feature_dir):
        directory, _, groups = feature_dir
        dataset = FeatureDatasetHDF5(
            directory,
            {**FEATURE_COLS, "labels": "labels", "labels_group": "labels_group"},
        )
        assert dataset.n_groups == len(np.unique(groups))

    def test_collate_keeps_variable_length_bags_as_lists(self, feature_dir):
        directory, _, _ = feature_dir
        dataset = FeatureDatasetHDF5(directory, {**FEATURE_COLS, "labels": "labels"})
        batch = FeatureDatasetHDF5.collate([dataset[0], dataset[1]])
        assert isinstance(batch["features"], list)
        assert batch["labels"].shape == (2, 1)
        assert batch["index"].shape == (2, 1)

    def test_collate_fair_adds_the_group(self, feature_dir):
        directory, _, _ = feature_dir
        dataset = FeatureDatasetHDF5(
            directory,
            {**FEATURE_COLS, "labels": "labels", "labels_group": "labels_group"},
        )
        batch = FeatureDatasetHDF5.collate_fair([dataset[0], dataset[1]])
        assert batch["labels_group"].shape == (2, 1)

    def test_requires_the_target_feature_column(self, feature_dir):
        directory, _, _ = feature_dir
        with pytest.raises(ValueError, match="features_target"):
            FeatureDatasetHDF5(directory, {"labels": "labels"})

    def test_rejects_a_non_directory(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            FeatureDatasetHDF5(tmp_path / "nope", FEATURE_COLS)

    def test_warns_on_an_empty_directory(self, tmp_path, caplog):
        empty = tmp_path / "empty"
        empty.mkdir()
        FeatureDatasetHDF5(empty, FEATURE_COLS)
        assert "no HDF5 files found" in caplog.text


class TestImageDatasetClassification:
    def test_returns_an_image_and_a_label(self, classification_h5):
        path, labels = classification_h5
        dataset = ImageDatasetHDF5(
            path.parent,
            path.name,
            {"images": "x", "labels": "y"},
            mode="classification",
        )
        image, label = dataset[1]
        assert isinstance(image, torch.Tensor), "must not be an (image, mask) tuple"
        assert image.shape == (3, 8, 8)
        assert label.item() == labels[1]

    def test_image_is_not_a_tuple(self, classification_h5):
        """Regression: the 2-tuple from the transform was assigned to `image`.

        `ImageOnlyDatasetHDF5` returned `(tensor, None)` and
        `ClassificationDatasetHDF5` returned `((tensor, None), label)`.
        """
        path, _ = classification_h5
        dataset = ImageDatasetHDF5(
            path.parent,
            path.name,
            {"images": "x", "labels": "y"},
            mode="classification",
        )
        assert not isinstance(dataset[0][0], tuple)

    def test_channels_last_permutes_the_image(self, classification_h5):
        path, _ = classification_h5
        dataset = ImageDatasetHDF5(
            path.parent,
            path.name,
            {"images": "x", "labels": "y"},
            mode="classification",
            channels_last=True,
        )
        assert dataset[0][0].shape == (8, 8, 3)

    def test_label_distribution(self, classification_h5):
        path, labels = classification_h5
        dataset = ImageDatasetHDF5(
            path.parent,
            path.name,
            {"images": "x", "labels": "y"},
            mode="classification",
        )
        assert dataset.get_label_distribution().as_mapping() == {
            0: int((labels == 0).sum()),
            2: int((labels == 2).sum()),
        }

    def test_rejects_a_mapping_label_column(self, classification_h5):
        path, _ = classification_h5
        with pytest.raises(ValueError, match="single"):
            ImageDatasetHDF5(
                path.parent,
                path.name,
                {"images": "x", "labels": {"images": "y"}},
                mode="classification",
            )


class TestImageDatasetImageOnly:
    def test_returns_a_bare_image(self, singleres_seg_h5):
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x"},
            mode="image_only",
        )
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 8, 8)

    def test_has_no_label_distribution(self, singleres_seg_h5):
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x"},
            mode="image_only",
        )
        with pytest.raises(ValueError, match="no labels"):
            dataset.get_label_distribution()


class TestImageDatasetSegmentation:
    def test_single_resolution(self, singleres_seg_h5):
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x", "labels": {"images": "y"}},
        )
        image, mask = dataset[0]
        assert image.shape == (3, 8, 8)
        assert mask.dtype == torch.uint8

    def test_label_distribution_works_for_segmentation(self, singleres_seg_h5):
        """Regression: the dict of label columns was used to index the HDF5 file.

        `get_label_distribution` unconditionally overwrote its per-resolution
        dict with `f[data_cols["labels"]]`, so it raised TypeError for every
        segmentation dataset.
        """
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x", "labels": {"images": "y"}},
        )
        dist = dataset.get_label_distribution()
        assert dist.n_classes >= 1
        assert dist.counts.sum() > 0

    def test_multi_resolution(self, multires_seg_h5):
        dataset = ImageDatasetHDF5(
            multires_seg_h5.parent,
            multires_seg_h5.name,
            {
                "target": "x_target",
                "context": "x_context",
                "labels": {"target": "y_target", "context": "y_context"},
            },
        )
        assert dataset.multiresolution
        images, masks = dataset[0]
        assert set(images) == {"target", "context"}
        assert set(masks) == {"target", "context"}
        assert images["target"].shape == (3, 8, 8)

    def test_merge_labels_remaps_mask_values(self, singleres_seg_h5):
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x", "labels": {"images": "y"}},
            merge_labels={2: 1},
        )
        _, mask = dataset[0]
        assert 2 not in set(mask.flatten().tolist())

    def test_merge_labels_default_is_not_shared(self, singleres_seg_h5):
        """Regression: `merge_labels: dict = {}` was a shared mutable default."""
        cols = {"images": "x", "labels": {"images": "y"}}
        first = ImageDatasetHDF5(singleres_seg_h5.parent, singleres_seg_h5.name, cols)
        first.merge_labels[9] = 9
        second = ImageDatasetHDF5(singleres_seg_h5.parent, singleres_seg_h5.name, cols)
        assert second.merge_labels == {}

    def test_requires_a_mapping_label_column(self, singleres_seg_h5):
        with pytest.raises(ValueError, match="mapping"):
            ImageDatasetHDF5(
                singleres_seg_h5.parent,
                singleres_seg_h5.name,
                {"images": "x", "labels": "y"},
            )


class TestImageDatasetHandles:
    def test_is_picklable_after_reading(self, singleres_seg_h5):
        """An open h5py.File cannot be pickled to DataLoader workers."""
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x"},
            mode="image_only",
        )
        _ = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        assert restored[0].shape == (3, 8, 8)

    def test_close_is_idempotent(self, singleres_seg_h5):
        dataset = ImageDatasetHDF5(
            singleres_seg_h5.parent,
            singleres_seg_h5.name,
            {"images": "x"},
            mode="image_only",
        )
        _ = dataset[0]
        dataset.close()
        dataset.close()
        assert dataset[0].shape == (3, 8, 8)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageDatasetHDF5(tmp_path, "nope.h5", {"images": "x"}, mode="image_only")


class TestTileDataset:
    def test_discovers_image_columns_by_regex(self, multires_seg_h5):
        dataset = TileDatasetHDF5(multires_seg_h5, image_regex="^x_")
        assert set(dataset.image_keys) == {"x_target", "x_context"}
        assert len(dataset) == 6
        item = dataset[0]
        assert set(item) == {"x_target", "x_context"}

    def test_rejects_a_regex_matching_nothing(self, multires_seg_h5):
        with pytest.raises(ValueError, match="matches"):
            TileDatasetHDF5(multires_seg_h5, image_regex="^nope")

    def test_rejects_unknown_columns(self, multires_seg_h5):
        with pytest.raises(ValueError, match="not in"):
            TileDatasetHDF5(multires_seg_h5, data_cols=["missing"])

    def test_rejects_image_columns_absent_from_data_cols(self, multires_seg_h5):
        """The transform addresses every image key, so all must be readable."""
        import albumentations as A

        with pytest.raises(ValueError, match="cannot be applied"):
            TileDatasetHDF5(
                multires_seg_h5,
                image_regex="^x_",
                data_cols=["x_target"],
                transform=A.Compose([A.HorizontalFlip(p=1)]),
            )

    def test_is_picklable_after_reading(self, multires_seg_h5):
        dataset = TileDatasetHDF5(multires_seg_h5)
        _ = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        assert set(restored[0]) == {"x_target", "x_context"}
