"""Stain augmentation correctness, and the albumentations 2.x API regressions."""

from __future__ import annotations

import warnings

import albumentations as A
import numpy as np
import pytest
from skimage.color import rgb2hed

from wsi_data.augmentations import (
    HEDAugmentor,
    ReplaceBackgroundColor,
    get_augmentor,
    multires_additional_targets,
)


@pytest.fixture
def he_tile(rng):
    """A realistic bright, low-saturation H&E-like tile."""
    return rng.integers(120, 250, (24, 24, 3), dtype=np.uint8)


class TestProbabilityRegression:
    def test_hed_honours_p(self):
        """Regression: `super().__init__(always_apply, p)` gave an effective p of 0.

        albumentations >= 2.0 has `ImageOnlyTransform.__init__(self, p=0.5)`,
        so the pre-1.0 positional call passed `always_apply=False` as `p` and
        stain augmentation silently never ran.
        """
        assert HEDAugmentor(p=0.25).p == pytest.approx(0.25)

    def test_replace_background_honours_p(self):
        assert ReplaceBackgroundColor(p=1.0).p == pytest.approx(1.0)

    def test_hed_actually_applies_under_compose(self, he_tile):
        out = A.Compose([HEDAugmentor(h_sigma=0.3, e_sigma=0.3, p=1.0)], seed=3)(
            image=he_tile
        )["image"]
        assert not np.array_equal(out, he_tile)


class TestHEDNumerics:
    def test_zero_sigma_is_an_exact_identity(self, he_tile):
        """Deconvolution must be invertible, or sigma=0 would still distort."""
        out = A.Compose([HEDAugmentor(h_sigma=0.0, e_sigma=0.0, p=1.0)], seed=1)(
            image=he_tile
        )["image"]
        np.testing.assert_array_equal(out, he_tile)

    def test_matches_skimage_where_skimage_does_not_clip(self, he_tile):
        """Stain concentrations are on skimage's scale for unclipped values.

        `skimage.color.separate_stains` clamps negative concentrations with
        `np.maximum(stains, 0)`, which is not invertible; this implementation
        omits that clamp but is otherwise identical.
        """
        from skimage.color.colorconv import hed_from_rgb

        scale = np.log(1e-6)
        mine = (
            np.log(np.clip(he_tile.astype(np.float64) / 255.0, 1e-6, 1.0)) / scale
        ) @ hed_from_rgb
        theirs = rgb2hed(he_tile)
        unclipped = theirs > 0
        assert unclipped.any()
        np.testing.assert_allclose(mine[unclipped], theirs[unclipped], atol=1e-12)

    def test_output_is_uint8_in_range(self, he_tile):
        out = A.Compose([HEDAugmentor(h_sigma=0.9, e_sigma=0.9, p=1.0)], seed=5)(
            image=he_tile
        )["image"]
        assert out.dtype == np.uint8
        assert out.shape == he_tile.shape

    def test_seeded_runs_are_reproducible(self, he_tile):
        def run(seed):
            return A.Compose([HEDAugmentor(p=1.0)], seed=seed)(image=he_tile)["image"]

        np.testing.assert_array_equal(run(7), run(7))
        assert not np.array_equal(run(7), run(8))

    @pytest.mark.parametrize("fill", [250, 4])
    def test_cutoff_range_skips_near_empty_tiles(self, fill):
        tile = np.full((8, 8, 3), fill, np.uint8)
        out = A.Compose([HEDAugmentor(h_sigma=0.5, p=1.0)], seed=1)(image=tile)["image"]
        np.testing.assert_array_equal(out, tile)

    def test_dab_is_untouched_by_default(self):
        """H&E slides carry no DAB, so its default jitter is zero."""
        assert HEDAugmentor().d_sigma == 0.0
        assert HEDAugmentor().d_bias == 0.0


class TestReplaceBackgroundColor:
    def test_replaces_only_the_matching_colour(self):
        image = np.zeros((3, 3, 3), np.uint8)
        image[0, 0] = (10, 20, 30)
        out = A.Compose([ReplaceBackgroundColor(0, 230, p=1.0)])(image=image)["image"]
        assert out[1, 1].tolist() == [230, 230, 230]
        assert out[0, 0].tolist() == [10, 20, 30]

    def test_serialises_rgb_flag(self):
        """Regression: `rgb` was omitted from the serialised init args."""
        assert "rgb" in ReplaceBackgroundColor().get_transform_init_args_names()

    def test_rejects_mismatched_colour_length(self):
        with pytest.raises(ValueError, match="3 components"):
            ReplaceBackgroundColor(old_color=(1, 2), new_color=0)


class TestGetAugmentor:
    def test_train_pipeline_emits_no_warnings(self, rng):
        """Regression: `ElasticTransform(alpha_affine=...)` warned and was ignored."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline = get_augmentor(patch_size=16, split="train", seed=2)
            out = pipeline(image=rng.integers(0, 256, (24, 24, 3), dtype=np.uint8))
        assert out["image"].shape[:2] == (16, 16)

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_every_split_resizes(self, split, rng):
        pipeline = get_augmentor(patch_size=16, split=split, seed=2)
        out = pipeline(image=rng.integers(0, 256, (24, 24, 3), dtype=np.uint8))
        assert out["image"].shape[:2] == (16, 16)

    def test_eval_splits_have_no_random_geometry(self, rng):
        """val/test must be deterministic regardless of seed."""
        image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        first = get_augmentor(16, split="val", seed=1)(image=image)["image"]
        second = get_augmentor(16, split="val", seed=999)(image=image)["image"]
        np.testing.assert_array_equal(first, second)

    def test_rejects_unknown_split(self):
        with pytest.raises(ValueError, match="split must be"):
            get_augmentor(split="holdout")  # type: ignore[arg-type]  # deliberately invalid


class TestMultiresTargets:
    def test_builds_image_and_mask_targets(self):
        assert multires_additional_targets(["target", "context"]) == {
            "context": "image"
        }
        assert multires_additional_targets(["target", "context"], with_masks=True) == {
            "context": "image",
            "target_mask": "mask",
            "context_mask": "mask",
        }
