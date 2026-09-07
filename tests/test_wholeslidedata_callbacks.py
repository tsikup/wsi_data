"""MaskedTiledAnnotationCallback: tiling geometry and shapely 2.x compatibility."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import geometry
from wholeslidedata.annotation.labels import Label
from wholeslidedata.annotation.types import Annotation

from wsi_data.wholeslidedata.callbacks import MaskedTiledAnnotationCallback, _polygons


def _label(name, value=1):
    return Label.create({"name": name, "value": value}).todict()


def _square_annotation(label_name="tumor", size=20):
    return Annotation.create(
        index=0,
        coordinates=[(0, 0), (size, 0), (size, size), (0, size), (0, 0)],
        label=_label(label_name),
    )


class TestPolygonFlattening:
    def test_multipolygon_no_longer_raises_typeerror(self):
        """Regression: shapely 2.x removed iteration over MultiPolygon."""
        multi = geometry.MultiPolygon(
            [geometry.box(0, 0, 1, 1), geometry.box(2, 2, 3, 3)]
        )
        assert len(_polygons(multi)) == 2

    def test_geometry_collection_is_flattened_recursively(self):
        collection = geometry.GeometryCollection(
            [
                geometry.MultiPolygon([geometry.box(0, 0, 1, 1)]),
                geometry.Point(5, 5),  # zero-area, dropped
            ]
        )
        polys = _polygons(collection)
        assert len(polys) == 1
        assert isinstance(polys[0], geometry.Polygon)

    def test_empty_geometry_yields_nothing(self):
        assert _polygons(geometry.Polygon()) == []


class TestTiling:
    def test_tiles_every_selected_label(self):
        callback = MaskedTiledAnnotationCallback(tile_size=10, label_names=["tumor"])
        out = callback([_square_annotation(size=20)])
        assert len(out) > 1
        assert all(a.label.name == "tumor" for a in out)

    def test_unselected_labels_pass_through_reindexed(self):
        callback = MaskedTiledAnnotationCallback(tile_size=10, label_names=["tumor"])
        stroma = _square_annotation("stroma", size=20)
        out = callback([stroma, _square_annotation("tumor", size=20)])
        assert out[0].label.name == "stroma"
        assert out[0]._index == 0

    def test_only_intersection_clips_to_the_annotation(self):
        """Regression: `only_intersection` appended the full tile, not the clip.

        The pre-1.0 code built the intersection polygon but then appended
        `box_poly` (the unclipped tile square) to the output list.
        """
        callback = MaskedTiledAnnotationCallback(
            tile_size=16, label_names=["tumor"], only_intersection=True
        )
        # A triangle: several boundary tiles will only partially overlap it.
        triangle = Annotation.create(
            index=0,
            coordinates=[(0, 0), (40, 0), (0, 40), (0, 0)],
            label=_label("tumor"),
        )
        out = callback([triangle])
        full_tile_area = 16 * 16
        assert any(
            geometry.Polygon(a.coordinates).area < full_tile_area - 1e-6 for a in out
        )

    def test_full_coverage_drops_low_overlap_tiles(self):
        callback = MaskedTiledAnnotationCallback(
            tile_size=16,
            label_names=["tumor"],
            full_coverage=True,
            intersection_percentage=0.99,
        )
        triangle = Annotation.create(
            index=0,
            coordinates=[(0, 0), (40, 0), (0, 40), (0, 0)],
            label=_label("tumor"),
        )
        without_filter = MaskedTiledAnnotationCallback(
            tile_size=16, label_names=["tumor"], full_coverage=False
        )([triangle])
        with_filter = callback([triangle])
        assert len(with_filter) < len(without_filter)

    def test_mask_coordinates_are_attached(self):
        callback = MaskedTiledAnnotationCallback(tile_size=10, label_names=["tumor"])
        out = callback([_square_annotation(size=20)])
        assert all(hasattr(a, "mask_coordinates") for a in out)
        assert all(isinstance(c, np.ndarray) for c in out[0].mask_coordinates)

    def test_rejects_overlap_at_least_tile_size(self):
        with pytest.raises(ValueError, match="overlap"):
            MaskedTiledAnnotationCallback(tile_size=10, overlap=10, label_names=[])

    def test_rejects_non_positive_tile_size(self):
        with pytest.raises(ValueError, match="tile_size"):
            MaskedTiledAnnotationCallback(tile_size=0, label_names=[])
