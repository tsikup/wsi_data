"""Albumentations pipeline and H&E-specific image augmentations.

Provides :func:`get_augmentor` -- the standard geometric + photometric +
stain-jitter pipeline for H&E tiles -- plus the two custom transforms it
uses, and :func:`multires_additional_targets` for wiring a pipeline up to
multi-resolution input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from he_preprocessing.transform import replace_color
from skimage.color.colorconv import hed_from_rgb, rgb_from_hed

from wsi_data.transforms import MULTIRES_PRIMARY_KEY

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "HEDAugmentor",
    "ReplaceBackgroundColor",
    "get_augmentor",
    "multires_additional_targets",
]

Split = Literal["train", "val", "test"]

# skimage's `separate_stains` scales optical density by log(1e-6) and floors
# RGB at 1e-6 to keep the logarithm finite. Reused here so that the stain
# concentrations produced below are on the same scale as `skimage.color.rgb2hed`.
_OD_SCALE = np.log(1e-6)
_RGB_FLOOR = 1e-6


def _to_color_tuple(value: int | Sequence[int], channels: int) -> tuple[int, ...]:
    """Broadcast a scalar colour to ``channels`` components, or pass one through."""
    if isinstance(value, int):
        return (value,) * channels
    color = tuple(int(v) for v in value)
    if len(color) != channels:
        msg = f"expected a colour with {channels} components, got {len(color)}"
        raise ValueError(msg)
    return color


def multires_additional_targets(
    resolutions: Iterable[str],
    *,
    with_masks: bool = False,
) -> dict[str, str]:
    """Build the ``additional_targets`` mapping for a multi-resolution pipeline.

    A single :class:`albumentations.Compose` samples its parameters once per
    call and applies them to every declared target, which is what keeps the
    resolutions of one tile geometrically aligned. Each non-primary resolution
    must therefore be declared as an extra ``"image"`` target (and, for
    segmentation, its ``<name>_mask`` counterpart as a ``"mask"`` target).

    Args:
        resolutions: Resolution names, e.g. ``["target", "context"]``. The
            primary :data:`~wsi_data.transforms.MULTIRES_PRIMARY_KEY` is
            skipped if present, since albumentations addresses it as
            ``"image"``/``"mask"`` already.
        with_masks: Also declare a mask target per resolution.

    Returns:
        A mapping suitable for ``A.Compose(..., additional_targets=...)``.

    Example:
        >>> multires_additional_targets(["target", "context"], with_masks=True)
        {'context': 'image', 'target_mask': 'mask', 'context_mask': 'mask'}
    """
    names = list(resolutions)
    targets = {n: "image" for n in names if n != MULTIRES_PRIMARY_KEY}
    if with_masks:
        targets.update({f"{n}_mask": "mask" for n in names})
    return targets


class ReplaceBackgroundColor(ImageOnlyTransform):
    """Replace one exact colour with another, e.g. zero-padding with tissue white.

    Geometric transforms pad with black by default, which downstream tissue
    detectors read as dense tissue. This recolours that padding to a
    background-like value.

    Args:
        old_color: Colour to replace, as a scalar (broadcast over channels) or
            a per-channel sequence.
        new_color: Replacement colour, in the same form.
        rgb: Whether images have 3 channels; ``False`` means 2.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        old_color: int | Sequence[int] = 0,
        new_color: int | Sequence[int] = 230,
        *,
        rgb: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        channels = 3 if rgb else 2
        self.rgb = rgb
        self.old_color = _to_color_tuple(old_color, channels)
        self.new_color = _to_color_tuple(new_color, channels)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor arguments to serialise."""
        return ("old_color", "new_color", "rgb")

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:  # noqa: ARG002
        """Replace ``old_color`` with ``new_color`` in ``img``."""
        return replace_color(
            img, old_color=self.old_color, new_color=self.new_color
        ).astype(np.uint8)


class HEDAugmentor(ImageOnlyTransform):
    """Jitter haematoxylin/eosin/DAB stain concentrations independently.

    Deconvolves RGB into the three HED stain channels, scales and shifts each
    by a random amount, and recomposes. Because it perturbs the stains
    themselves rather than RGB directly, it models inter-scanner and
    inter-laboratory staining variation far better than a generic
    hue/saturation jitter.

    Each channel ``c`` is mapped to ``c * (1 + sigma_c) + bias_c``, with
    ``sigma_c`` drawn uniformly from ``[-<x>_sigma, +<x>_sigma]`` and
    ``bias_c`` from ``[-<x>_bias, +<x>_bias]``.

    Note:
        The deconvolution uses scikit-image's published HED stain matrices and
        optical-density scaling, so the stain concentrations match
        :func:`skimage.color.rgb2hed`, but it deliberately does **not** call
        that function. ``skimage.color.separate_stains`` applies
        ``np.maximum(stains, 0)``, clamping physically-negative
        concentrations; that step is not invertible, and on realistic H&E
        tiles it zeroes roughly a third of all stain values. Round-tripping
        through it distorts the image even with zero sigma and bias. Without
        the clamp the transform is exactly invertible, so zero sigma and bias
        is an exact identity (verified bit-exact through ``uint8``).

        This replaces ``stainlib.augmentation.augmenter.HedColorAugmenter``,
        which is unpublished and unmaintained. Randomness now comes from
        albumentations' own generator instead of an internal unseeded one, so
        pipelines are reproducible via ``A.Compose(..., seed=...)``.

    Args:
        h_sigma: Haematoxylin multiplicative jitter magnitude, in ``[0, 1]``.
        e_sigma: Eosin multiplicative jitter magnitude.
        d_sigma: DAB multiplicative jitter magnitude. Zero by default, since
            H&E slides carry no DAB.
        h_bias: Haematoxylin additive jitter magnitude.
        e_bias: Eosin additive jitter magnitude.
        d_bias: DAB additive jitter magnitude.
        cutoff_range: Skip augmentation when the image's mean intensity,
            normalised to ``[0, 1]``, falls outside this range. Near-empty or
            near-saturated tiles carry no stain signal worth perturbing, and
            jittering them mostly amplifies noise.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        h_sigma: float = 0.1,
        e_sigma: float = 0.1,
        *,
        d_sigma: float = 0.0,
        h_bias: float = 0.0,
        e_bias: float = 0.0,
        d_bias: float = 0.0,
        cutoff_range: tuple[float, float] = (0.05, 0.95),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.h_sigma = h_sigma
        self.e_sigma = e_sigma
        self.d_sigma = d_sigma
        self.h_bias = h_bias
        self.e_bias = e_bias
        self.d_bias = d_bias
        self.cutoff_range = cutoff_range

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the constructor arguments to serialise."""
        return (
            "h_sigma",
            "e_sigma",
            "d_sigma",
            "h_bias",
            "e_bias",
            "d_bias",
            "cutoff_range",
        )

    def get_params(self) -> dict[str, Any]:
        """Draw per-channel stain factors and biases from albumentations' RNG."""
        sigmas = (self.h_sigma, self.e_sigma, self.d_sigma)
        biases = (self.h_bias, self.e_bias, self.d_bias)
        return {
            "factors": np.array(
                [1.0 + self.py_random.uniform(-s, s) for s in sigmas], dtype=np.float64
            ),
            "biases": np.array(
                [self.py_random.uniform(-b, b) for b in biases], dtype=np.float64
            ),
        }

    def apply(
        self,
        img: np.ndarray,
        factors: np.ndarray | None = None,
        biases: np.ndarray | None = None,
        **params: Any,  # noqa: ARG002
    ) -> np.ndarray:
        """Jitter ``img``'s stain concentrations and recompose to ``uint8`` RGB."""
        if factors is None or biases is None:  # pragma: no cover - defensive
            drawn = self.get_params()
            factors, biases = drawn["factors"], drawn["biases"]

        scaled = img.astype(np.float64) / 255.0
        low, high = self.cutoff_range
        if not low < float(scaled.mean()) < high:
            return img

        clipped = np.clip(scaled, _RGB_FLOOR, 1.0)
        stains = (np.log(clipped) / _OD_SCALE) @ hed_from_rgb
        stains = stains * factors + biases
        recomposed = np.exp((stains * _OD_SCALE) @ rgb_from_hed)
        result = (np.clip(recomposed, 0.0, 1.0) * 255.0).round()
        return np.asarray(result).astype(np.uint8)


def get_augmentor(
    patch_size: int = 512,
    *,
    enable_augmentation: bool = True,
    enable_stain_augmentation: bool = True,
    replace_background: bool = True,
    constant_pad_value: int = 230,
    split: Split = "train",
    additional_targets: dict[str, str] | None = None,
    seed: int | None = None,
) -> A.Compose:
    """Build the standard H&E tile augmentation pipeline.

    Geometric and photometric augmentations are applied only for
    ``split="train"``; the resizing and background-recolouring tail runs for
    every split so that train and evaluation tiles are shaped identically.

    Args:
        patch_size: Longest-side size every tile is resized to.
        enable_augmentation: Master switch for the training-only augmentations.
        enable_stain_augmentation: Include the H&E stain-jitter block.
        replace_background: Recolour pure-black padding to
            ``constant_pad_value``.
        constant_pad_value: Replacement value for black padding.
        split: Which split this pipeline is for.
        additional_targets: Extra albumentations targets, for multi-resolution
            or segmentation input. See :func:`multires_additional_targets`.
        seed: Seed for the pipeline's random generator, for reproducible runs.

    Returns:
        The composed pipeline.

    Raises:
        ValueError: If ``split`` is not ``"train"``, ``"val"`` or ``"test"``.
    """
    if split not in ("train", "val", "test"):
        msg = f"split must be 'train', 'val' or 'test', got {split!r}"
        raise ValueError(msg)

    transforms: list[A.BasicTransform] = []

    if enable_augmentation and split == "train":
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
            ),
            A.Sharpen(alpha=(0.0, 0.5), p=0.3),
            A.GaussNoise(p=0.3),
        ]

        if enable_stain_augmentation:
            transforms.append(
                A.OneOf(
                    [
                        HEDAugmentor(p=0.25),
                        A.HueSaturationValue(
                            hue_shift_limit=(-10, 10),
                            sat_shift_limit=(-10, 10),
                            val_shift_limit=0,
                            p=0.25,
                        ),
                    ],
                    p=0.3,
                )
            )

        # Elastic warp or an affine jitter, not both. `ElasticTransform` used
        # to take an `alpha_affine` argument that folded a small affine jitter
        # into the warp; albumentations removed it, and passing it now only
        # raises a warning and is ignored -- the `A.Affine` alternative in
        # this `OneOf` covers that case explicitly.
        transforms.append(
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=0.1,
                        rotate=(-30, 30),
                        shear=(-9, 9),
                        p=0.3,
                    ),
                ],
                p=0.3,
            )
        )

    transforms.append(A.LongestMaxSize(max_size=patch_size, p=1.0))

    if replace_background:
        transforms.append(
            ReplaceBackgroundColor(old_color=0, new_color=constant_pad_value, p=1.0)
        )

    return A.Compose(transforms, additional_targets=additional_targets, seed=seed)
