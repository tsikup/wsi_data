"""Array cropping and the albumentations transform pipeline shared by the HDF5 datasets.

The transform helpers here are deliberately free functions taking explicit
inputs rather than methods reading ``self.data_cols``: the multi-resolution
key set is derivable from the image mapping itself, which makes each step
testable in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    import albumentations as A

__all__ = [
    "MULTIRES_PRIMARY_KEY",
    "crop_data",
    "to_chw_float_tensor",
    "to_mask_tensor",
    "transform_image_and_mask",
    "transform_multires_image_and_mask",
]

#: Resolution key treated as albumentations' primary ``image``/``mask`` target.
MULTIRES_PRIMARY_KEY = "target"

_MASK_SUFFIX = "_mask"

#: Rank of a bare `(H, W)` image with no channel axis.
_UNCHANNELLED_RANK = 2

# Spatial (height, width) axis positions by array rank. Ranks 2 and 3 are a
# bare or channelled image, 4 adds a batch axis, 5 a batch and a sequence axis.
_SPATIAL_AXES: dict[int, tuple[int, int]] = {
    2: (0, 1),
    3: (0, 1),
    4: (1, 2),
    5: (2, 3),
}


def crop_data(data: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Centre-crop an array's spatial axes to ``output_shape``.

    Which axes are spatial depends on the rank: ``(0, 1)`` for 2-D ``(H, W)``
    and 3-D ``(H, W, C)``, ``(1, 2)`` for 4-D ``(N, H, W, C)``, and ``(2, 3)``
    for 5-D ``(N, T, H, W, C)``.

    Note:
        Two behaviour fixes relative to the pre-1.0 implementation. It sliced
        as ``data[crop:-crop]``, so whenever an axis already had the requested
        length the crop was ``0`` and ``data[0:-0]`` -- which Python reads as
        ``data[0:0]`` -- returned an **empty** array. It also left an extra
        row or column whenever the size difference was odd, returning
        ``output_shape + 1`` on that axis. Both are corrected here: the result
        is always exactly ``output_shape`` on the spatial axes, and an
        already-correct axis is passed through untouched.

    Args:
        data: Array of rank 2 to 5.
        output_shape: Target ``(height, width)``.

    Returns:
        A view of ``data`` centre-cropped on its spatial axes. Returned
        unchanged when it already has the requested spatial size.

    Raises:
        ValueError: If ``data`` has an unsupported rank, or is smaller than
            ``output_shape`` on either spatial axis.
    """
    axes = _SPATIAL_AXES.get(data.ndim)
    if axes is None:
        msg = (
            f"crop_data supports arrays of rank 2-5, got rank {data.ndim} "
            f"with shape {data.shape}"
        )
        raise ValueError(msg)

    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    cur_h, cur_w = data.shape[axes[0]], data.shape[axes[1]]
    if cur_h < out_h or cur_w < out_w:
        msg = (
            f"cannot crop spatial size ({cur_h}, {cur_w}) up to "
            f"({out_h}, {out_w}); crop_data only shrinks"
        )
        raise ValueError(msg)
    if (cur_h, cur_w) == (out_h, out_w):
        return data

    start_h = (cur_h - out_h) // 2
    start_w = (cur_w - out_w) // 2
    index: list[slice] = [slice(None)] * data.ndim
    index[axes[0]] = slice(start_h, start_h + out_h)
    index[axes[1]] = slice(start_w, start_w + out_w)
    return data[tuple(index)]


def to_chw_float_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an ``(H, W[, C])`` image to a ``(C, H, W)`` ``float32`` tensor.

    Reproduces ``torchvision.transforms.ToTensor`` exactly, including its
    dtype-dependent scaling: ``uint8`` input is divided by 255 to land in
    ``[0, 1]``, while floating-point input is passed through **unscaled**.
    Written out explicitly so the scaling rule is visible at the call site
    rather than hidden in a transform object constructed per item.

    Args:
        image: An ``(H, W)`` or ``(H, W, C)`` array.

    Returns:
        A ``(C, H, W)`` ``float32`` tensor.
    """
    if image.ndim == _UNCHANNELLED_RANK:
        image = image[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div(255.0)
    return tensor.to(torch.float32)


def to_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a segmentation mask to a ``uint8`` tensor, leaving its axes alone."""
    if isinstance(mask, torch.Tensor):
        return mask.to(torch.uint8)
    return torch.from_numpy(np.ascontiguousarray(mask)).to(torch.uint8)


def _multires_kwargs(
    image: Mapping[str, np.ndarray],
    mask: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Map a resolution dict onto albumentations' primary/additional target names."""
    kwargs: dict[str, np.ndarray] = {"image": image[MULTIRES_PRIMARY_KEY]}
    if mask is not None:
        kwargs["mask"] = mask[MULTIRES_PRIMARY_KEY]
    for key in image:
        if key == MULTIRES_PRIMARY_KEY:
            continue
        kwargs[key] = image[key]
        if mask is not None:
            kwargs[f"{key}{_MASK_SUFFIX}"] = mask[key]
    return kwargs


def _split_multires_output(
    transformed: Mapping[str, np.ndarray],
    mask_transform: A.Compose | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Split an albumentations multi-target result back into image and mask dicts."""
    out = dict(transformed)
    # Normalise the primary mask to the `<key>_mask` form so every mask is
    # handled by one branch below.
    if "mask" in out:
        out[f"{MULTIRES_PRIMARY_KEY}{_MASK_SUFFIX}"] = out.pop("mask")

    images: dict[str, np.ndarray] = {MULTIRES_PRIMARY_KEY: out.pop("image")}
    masks: dict[str, np.ndarray] = {}
    for key, value in out.items():
        if key.endswith(_MASK_SUFFIX):
            name = key[: -len(_MASK_SUFFIX)]
            masks[name] = (
                mask_transform(image=value)["image"]
                if mask_transform is not None
                else value
            )
        else:
            images[key] = value
    return images, (masks or None)


def transform_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    transform: A.Compose | None = None,
    mask_transform: A.Compose | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the single-resolution albumentations pipeline and convert to tensors.

    Args:
        image: An ``(H, W, C)`` array.
        mask: A matching ``(H, W[, C])`` mask, or ``None``.
        transform: Pipeline applied jointly to image and mask.
        mask_transform: Image-only pipeline applied to the mask afterwards.

    Returns:
        An ``(image, mask)`` tensor pair; ``mask`` is ``None`` when none was given.
    """
    if transform is not None:
        if mask is None:
            image = transform(image=image)["image"]
        else:
            out = transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            if mask_transform is not None:
                mask = mask_transform(image=mask)["image"]
    elif mask_transform is not None and mask is not None:
        # No joint transform, so the mask-only pipeline still has to run.
        mask = mask_transform(image=mask)["image"]

    tensor = to_chw_float_tensor(image)
    return tensor, (None if mask is None else to_mask_tensor(mask))


def transform_multires_image_and_mask(
    images: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray] | None = None,
    *,
    transform: A.Compose | None = None,
    mask_transform: A.Compose | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Run the multi-resolution albumentations pipeline and convert to tensors.

    Every resolution is passed to ``transform`` as an albumentations
    additional target, so one sampled parameter set is shared across all of
    them and the resolutions stay geometrically aligned.

    Args:
        images: Mapping of resolution name to ``(H, W, C)`` array. Must
            contain :data:`MULTIRES_PRIMARY_KEY`.
        masks: Matching mapping of masks, or ``None``.
        transform: Pipeline applied jointly to every image and mask. Must have
            been built with ``additional_targets`` covering the non-primary
            resolution keys and their ``_mask`` counterparts -- see
            :func:`wsi_data.augmentations.multires_additional_targets`.
        mask_transform: Image-only pipeline applied to each mask afterwards.

    Returns:
        An ``(images, masks)`` pair of resolution-keyed tensor dicts; ``masks``
        is ``None`` when none were given.

    Raises:
        KeyError: If ``images`` has no :data:`MULTIRES_PRIMARY_KEY` entry.
    """
    if MULTIRES_PRIMARY_KEY not in images:
        msg = (
            f"multi-resolution input must contain the "
            f"{MULTIRES_PRIMARY_KEY!r} resolution, got {sorted(images)}"
        )
        raise KeyError(msg)

    if transform is not None:
        images, masks = _split_multires_output(
            transform(**_multires_kwargs(images, masks)), mask_transform
        )
    elif mask_transform is not None and masks is not None:
        masks = {
            key: mask_transform(image=value)["image"] for key, value in masks.items()
        }

    image_tensors = {key: to_chw_float_tensor(v) for key, v in images.items()}
    if masks is None:
        return image_tensors, None
    return image_tensors, {key: to_mask_tensor(v) for key, v in masks.items()}
