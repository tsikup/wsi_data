"""Annotation callbacks run by ``wholeslidedata``'s parsers after parsing.

:class:`MaskedTiledAnnotationCallback` turns each large region annotation into
a grid of tile-sized annotations, so that a sampler can iterate tiles rather
than sample random points inside one big polygon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import geometry
from wholeslidedata.annotation.callbacks import AnnotationCallback
from wholeslidedata.annotation.types import Annotation

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

__all__ = ["MaskedTiledAnnotationCallback"]


def _polygons(shape: geometry.base.BaseGeometry) -> list[geometry.Polygon]:
    """Flatten any geometry into its constituent polygons.

    Note:
        Shapely 2.0 removed iteration over multi-part geometries, so the
        pre-1.0 ``for poly in multipolygon`` raised
        ``TypeError: 'MultiPolygon' object is not iterable``. Recursing through
        ``.geoms`` also handles a ``GeometryCollection`` that itself contains a
        ``MultiPolygon``, which the pre-1.0 single-level checks missed.
        Zero-area parts (points, line strings) are dropped.
    """
    if shape.is_empty:
        return []
    if isinstance(shape, geometry.Polygon):
        return [shape]
    if hasattr(shape, "geoms"):
        return [poly for part in shape.geoms for poly in _polygons(part)]
    return []


class MaskedTiledAnnotationCallback(AnnotationCallback):
    """Replace each matching annotation with a grid of tile-sized annotations.

    Every annotation whose label is in ``label_names`` is covered with a grid
    of ``tile_size`` boxes stepping by ``tile_size - overlap``. Annotations
    with other labels pass through re-indexed but otherwise untouched.

    Each emitted tile also carries a ``mask_coordinates`` attribute: the
    polygon rings of its intersection with the parent annotation, which
    :func:`wsi_data.viz.draw_relative_annotations` can rasterise into a
    segmentation mask for that tile.

    Args:
        tile_size: Tile side length in pixels, before ``ratio`` is applied.
        label_names: Labels whose annotations should be tiled.
        ratio: Scale applied to ``tile_size`` and ``overlap``, for expressing
            them at a different resolution than the annotations.
        overlap: Overlap between neighbouring tiles in pixels, before ``ratio``.
        intersection_percentage: Minimum fraction of a tile that must fall
            inside the annotation for the tile to be kept. Only enforced when
            ``full_coverage`` is set.
        full_coverage: Enforce ``intersection_percentage``. With the default
            ``intersection_percentage=1.0`` this means "keep only tiles lying
            entirely within the annotation", hence the name; when unset, every
            tile in the bounding box is kept regardless of overlap.
        only_intersection: Emit the tile clipped to the annotation rather than
            the full square tile.

    Raises:
        ValueError: If ``tile_size`` is not positive or ``overlap`` is not
            smaller than ``tile_size`` (which would make the grid step zero or
            negative and never terminate).
    """

    def __init__(
        self,
        tile_size: int,
        label_names: Collection[str],
        ratio: float = 1,
        overlap: int = 0,
        intersection_percentage: float = 0.2,
        *,
        full_coverage: bool = False,
        only_intersection: bool = False,
    ) -> None:
        self._tile_size = int(tile_size * ratio)
        self._overlap = int(overlap * ratio)
        if self._tile_size <= 0:
            msg = f"tile_size * ratio must be positive, got {self._tile_size}"
            raise ValueError(msg)
        if self._overlap >= self._tile_size:
            msg = (
                f"overlap ({self._overlap}) must be smaller than tile_size "
                f"({self._tile_size}); otherwise the tile grid never advances"
            )
            raise ValueError(msg)
        self._full_coverage = full_coverage
        self._only_intersection = only_intersection
        self._label_names = set(label_names)
        self._intersection_percentage = intersection_percentage

    @property
    def _step(self) -> int:
        return self._tile_size - self._overlap

    def __call__(self, annotations: Sequence[Annotation]) -> list[Annotation]:
        """Tile every annotation whose label is selected, re-indexing all of them."""
        new_annotations: list[Annotation] = []
        index = 0

        for annotation in annotations:
            if annotation.label.name not in self._label_names:
                annotation._index = index  # noqa: SLF001 - upstream has no setter
                new_annotations.append(annotation)
                index += 1
                continue

            for tile in self._tiles_for(annotation):
                new_annotations.append(self._make_annotation(tile, annotation, index))
                index += 1

        return new_annotations

    def _tiles_for(self, annotation: Annotation) -> list[geometry.Polygon]:
        """Build the kept tile polygons covering one annotation's bounding box."""
        x1, y1, x2, y2 = annotation.bounds
        tiles: list[geometry.Polygon] = []

        for x in range(int(x1), int(x2), self._step):
            for y in range(int(y1), int(y2), self._step):
                box = geometry.box(x, y, x + self._tile_size, y + self._tile_size)
                if self._full_coverage:
                    covered = box.intersection(annotation.geometry).area / box.area
                    if covered < self._intersection_percentage:
                        continue

                if self._only_intersection:
                    # The pre-1.0 code computed the intersection here but then
                    # appended `box_poly` -- the full tile -- so
                    # `only_intersection` silently had no effect for the common
                    # single-polygon case, and its loop variable shadowed the
                    # tile used by the mask computation below.
                    tiles.extend(_polygons(box.intersection(annotation.geometry)))
                else:
                    tiles.append(box)

        return tiles

    def _make_annotation(
        self, tile: geometry.Polygon, parent: Annotation, index: int
    ) -> Annotation:
        """Create one tile annotation, attaching its mask rings."""
        new_annotation = Annotation.create(
            index=index,
            coordinates=list(tile.exterior.coords),
            label=parent.label.todict(),
        )
        new_annotation.mask_coordinates = [
            np.asarray(poly.exterior.coords)
            for poly in _polygons(tile.intersection(parent.geometry))
        ]
        return new_annotation
