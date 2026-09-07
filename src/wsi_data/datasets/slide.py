"""Tile dataset reading directly from a whole-slide image.

Note:
    Named ``Single_WSI_Dataset`` before 1.0.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from he_preprocessing.pipeline.tile import TilePipelineConfig, preprocess_tile
from he_preprocessing.tissue.detect import is_blurry, keep_tile
from he_preprocessing.transform import pad_to_size
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from wholeslidedata.data.files import WholeSlideAnnotationFile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchvision.transforms import Compose
    from wholeslidedata import WholeSlideAnnotation
    from wholeslidedata.annotation.types import PolygonAnnotation

    from wsi_data.wholeslidedata.files import (
        MultiResWholeSlideImageFile,
        SingleResWholeSlideImageFile,
    )

__all__ = ["BlurrinessMode", "SlideTileDataset"]

logger = logging.getLogger(__name__)

#: How blur is measured. ``"normalized"`` scales the Laplacian variance by
#: tissue area, ``"masked"`` measures only inside the tissue mask, and
#: ``None`` uses the raw variance.
BlurrinessMode = Literal["masked", "normalized"]

_PRIMARY_KEY = "target"


class SlideTileDataset(Dataset[dict[str, dict[str, Any]]]):
    """Reads one tile per annotation from a single whole-slide image.

    Each item is keyed by resolution name; every entry holds the tile under
    ``img_array``, its centre under ``x``/``y``, its ``spacing``, and -- for
    segmentation -- its ``mask_array``. A tile rejected by the tissue or blur
    filters has ``img_array`` set to ``None``; use :meth:`collate_fn` to drop
    those from a batch.

    The slide is opened lazily on first access so that the dataset can be
    pickled to ``DataLoader`` worker processes.

    Args:
        image_file: The slide file to read from.
        annotations: Annotations whose centres locate the tiles.
        tile_size: Tile side length in pixels.
        spacing: A single spacing, or a mapping of resolution name to spacing
            for multi-resolution reads. ``None`` values in a mapping are
            dropped, and a mapping left with one entry is treated as
            single-resolution.
        transform: Optional tile transform, applied last.
        filters2apply: he-preprocessing tile pipeline configuration.
        blurriness_threshold: Laplacian-variance threshold below which a tile
            is discarded; a mapping per resolution for multi-resolution reads.
        blurriness_mode: How blur is measured. Spelled ``bluriness_mode``
            before 1.0.
        tissue_percentage: Minimum tissue fraction to keep a tile; a mapping
            per resolution for multi-resolution reads.
        constant_pad_value: Value used to pad tiles up to ``tile_size``.
        segmentation: Also read the annotation mask for each tile.
        wsa: Annotations to attach to the slide for mask sampling. Required
            when ``segmentation`` is set and the slide has none.

    Raises:
        ValueError: If ``spacing`` is empty, or if a multi-resolution
            ``spacing`` lacks ``"target"``.
        TypeError: If the per-resolution arguments do not match ``spacing``'s
            shape (a mapping given for a single-resolution dataset, or a bare
            number given for a multi-resolution one).
    """

    def __init__(
        self,
        image_file: MultiResWholeSlideImageFile | SingleResWholeSlideImageFile,
        annotations: Sequence[PolygonAnnotation],
        tile_size: int = 512,
        spacing: Mapping[str, float | None] | float = 0.5,
        *,
        transform: Compose | None = None,
        filters2apply: Mapping[str, Any] | TilePipelineConfig | None = None,
        blurriness_threshold: Mapping[str, int | None] | int | None = None,
        blurriness_mode: BlurrinessMode | None = None,
        tissue_percentage: Mapping[str, float | None] | float | None = None,
        constant_pad_value: int = 230,
        segmentation: bool = False,
        wsa: WholeSlideAnnotation | WholeSlideAnnotationFile | None = None,
    ) -> None:
        if blurriness_mode not in ("masked", "normalized", None):
            msg = (
                "blurriness_mode must be 'masked', 'normalized' or None, "
                f"got {blurriness_mode!r}"
            )
            raise ValueError(msg)

        self.image_name = image_file.path.stem
        self.image_file = image_file
        self.annotations = [annotation.center for annotation in annotations]
        self.dataset_size = len(annotations)
        self.tile_size = tile_size
        self.spacing = self._normalise_spacing(spacing)
        self.multires = isinstance(self.spacing, dict)

        self.transform = transform
        self.filters2apply = filters2apply
        self.blurriness_threshold = blurriness_threshold
        self.blurriness_mode = blurriness_mode
        self.tissue_percentage = tissue_percentage
        self.constant_pad_value = constant_pad_value
        self.segmentation = segmentation

        self.wsa = wsa
        self.wsi: Any = None

        self._validate()

    @staticmethod
    def _normalise_spacing(
        spacing: Mapping[str, float | None] | float,
    ) -> dict[str, float] | float:
        """Drop ``None`` entries and unwrap a one-entry mapping to a scalar.

        Note:
            The pre-1.0 whole-slide image class did this unwrapping itself, so
            that ``get_data`` silently accepted a dict where it documented a
            float. Normalising here keeps the image classes' signatures honest.
        """
        if not isinstance(spacing, Mapping):
            return float(spacing)
        present = {key: float(v) for key, v in spacing.items() if v is not None}
        if not present:
            msg = "spacing mapping has no non-None entries"
            raise ValueError(msg)
        if len(present) == 1:
            return next(iter(present.values()))
        return present

    def _validate(self) -> None:
        """Check that per-resolution arguments match the spacing's shape."""
        if not self.multires:
            for name, value in (
                ("blurriness_threshold", self.blurriness_threshold),
                ("tissue_percentage", self.tissue_percentage),
            ):
                if isinstance(value, Mapping):
                    msg = (
                        f"{name} must be a number for a single-resolution dataset, "
                        f"got {value!r}"
                    )
                    raise TypeError(msg)
            return

        assert isinstance(self.spacing, dict)
        if _PRIMARY_KEY not in self.spacing:
            msg = (
                f"a multi-resolution spacing must include {_PRIMARY_KEY!r}, "
                f"got {sorted(self.spacing)}"
            )
            raise ValueError(msg)
        for name, value in (
            ("blurriness_threshold", self.blurriness_threshold),
            ("tissue_percentage", self.tissue_percentage),
        ):
            if value is None:
                continue
            if not isinstance(value, Mapping):
                msg = (
                    f"{name} must be a mapping keyed like spacing "
                    f"({sorted(self.spacing)}) for a multi-resolution dataset, "
                    f"got {value!r}"
                )
                raise TypeError(msg)
            missing = set(self.spacing) - set(value)
            if missing:
                msg = f"{name} is missing entries for {sorted(missing)}"
                raise ValueError(msg)

    def __len__(self) -> int:
        return self.dataset_size

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Any:
        """Drop filtered-out tiles from a batch, then collate the rest.

        Returns an empty list when every tile in the batch was filtered out,
        which the training loop must skip.
        """
        kept = [item for item in batch if item[_PRIMARY_KEY]["img_array"] is not None]
        if not kept:
            return []
        return default_collate(kept)

    def open_wsi(self) -> None:
        """Open the slide, and its annotations if mask sampling is needed."""
        self.wsi = self.image_file.open()
        if isinstance(self.wsa, WholeSlideAnnotationFile):
            self.wsa = self.wsa.open()
        if self.wsa is not None:
            self.wsi.annotation = self.wsa

    def _preprocess(
        self,
        patch: Any,
        blurriness_threshold: int | None = None,
        tissue_percentage: float | None = None,
    ) -> Any:
        """Pad, filter and transform one tile, or return ``None`` if rejected."""
        patch = pad_to_size(patch, self.tile_size, value=self.constant_pad_value)

        if tissue_percentage is not None and not keep_tile(
            patch, self.tile_size, tissue_threshold=tissue_percentage
        ):
            return None

        if (
            blurriness_threshold is not None
            and is_blurry(
                patch,
                threshold=blurriness_threshold,
                normalize=self.blurriness_mode == "normalized",
                tissue_only=self.blurriness_mode == "masked",
            )[0]
        ):
            return None

        if self.filters2apply is not None:
            config = (
                self.filters2apply
                if isinstance(self.filters2apply, TilePipelineConfig)
                else TilePipelineConfig(**self.filters2apply)
            )
            patch = preprocess_tile(patch, config)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch

    def _threshold_for(self, attribute: Any, key: str) -> Any:
        """Look up a per-resolution filter threshold."""
        if attribute is None:
            return None
        if isinstance(attribute, Mapping):
            return attribute[key]
        return attribute

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        """Read, filter and return one tile per resolution."""
        if self.wsi is None:
            self.open_wsi()

        x, y = self.annotations[index]
        result = self.wsi.get_data(
            x=x,
            y=y,
            width=self.tile_size,
            height=self.tile_size,
            spacing=self.spacing,
            center=True,
            with_mask=self.segmentation,
        )

        # Narrowed with an inline `isinstance` rather than the pre-computed
        # `self.multires`: mypy cannot correlate a separate bool attribute
        # with `self.spacing`'s type, so a per-branch dict is built here for
        # each of the two shapes instead of a single expression mixing both.
        spacings: dict[str, float]
        patches: dict[str, Any]
        masks: dict[str, Any]
        if isinstance(self.spacing, dict):
            spacings = self.spacing
            patches, masks = result if self.segmentation else (result, {})
        else:
            spacings = {_PRIMARY_KEY: self.spacing}
            patch, mask = result if self.segmentation else (result, None)
            patches = {_PRIMARY_KEY: patch}
            masks = {_PRIMARY_KEY: mask}

        item: dict[str, dict[str, Any]] = {}
        for key, spacing in spacings.items():
            entry: dict[str, Any] = {
                "img_array": self._preprocess(
                    patches[key],
                    blurriness_threshold=self._threshold_for(
                        self.blurriness_threshold, key
                    ),
                    tissue_percentage=self._threshold_for(self.tissue_percentage, key),
                ),
                "x": x,
                "y": y,
                "spacing": spacing,
            }
            if self.segmentation:
                entry["mask_array"] = masks[key]
            item[key] = entry
        return item
