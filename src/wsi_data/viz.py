"""Optional visualisation helpers: annotation overlays and label-distribution plots.

Every function here is a debugging or reporting aid, never part of a data
path, which is why plotting lives in its own module: importing a dataset class
must not drag in a plotting stack. Pillow is a hard dependency (the overlay
helpers draw with it), but matplotlib is only needed by
:func:`plot_label_distribution` and is imported lazily -- install it with the
``viz`` extra.

Note:
    Before 1.0, label-distribution plotting lived *inside* the dataset classes
    as ``get_label_distribution(as_figure=True)``, which made ``seaborn`` and
    ``pandas`` effectively mandatory. Datasets now return a
    :class:`~wsi_data.labels.LabelDistribution` and plotting happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from wholeslidedata.annotation.types import Annotation

    from wsi_data.labels import LabelDistribution

__all__ = [
    "draw_relative_annotations",
    "draw_tiles",
    "get_annotation_mask",
    "plot_label_distribution",
]

#: An RGB or RGBA colour, matching what Pillow's ``ImageDraw`` accepts.
Color = tuple[int, int, int] | tuple[int, int, int, int]


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Accept either a PIL image or an array."""
    return Image.fromarray(image) if isinstance(image, np.ndarray) else image


def draw_tiles(
    image: Image.Image | np.ndarray,
    annotation_centers: Sequence[Sequence[int]],
    tile_size: int,
    *,
    outline_color: Color | None = (255, 0, 0),
    fill_color: Color | None = None,
    width: int = 3,
    spacings_ratio: float = 1.0,
    view_scale: float = 1.0,
) -> Image.Image:
    """Draw a square per tile centre, for checking tile placement on a thumbnail.

    Args:
        image: Slide thumbnail to draw on. Never modified.
        annotation_centers: ``(x, y)`` tile centres, in the coordinate system
            that ``spacings_ratio`` converts to the thumbnail's.
        tile_size: Tile side length at the centres' own resolution.
        outline_color: Square outline colour, or ``None`` for no outline.
        fill_color: Square fill colour, or ``None`` for unfilled.
        width: Outline width in pixels.
        spacings_ratio: Ratio converting centre coordinates to the
            thumbnail's resolution.
        view_scale: Extra scale applied to both coordinates and tile size.

    Returns:
        A new image with the squares drawn on it.
    """
    canvas = _as_pil(image).copy()
    draw = ImageDraw.Draw(canvas)
    scaled_size = int(tile_size * view_scale)

    for center in annotation_centers:
        x = int(center[0] * spacings_ratio * view_scale) - scaled_size // 2
        y = int(center[1] * spacings_ratio * view_scale) - scaled_size // 2
        draw.rectangle(
            (x, y, x + scaled_size, y + scaled_size),
            outline=outline_color,
            fill=fill_color,
            width=width,
        )
    return canvas


def draw_relative_annotations(
    image: Image.Image | np.ndarray,
    annotations: Sequence[Annotation],
    *,
    use_base_coordinates: bool = False,
    scale: float = 1.0,
    relative_bounds: Sequence[int] = (0, 0),
    plot_mask: bool = False,
    outline_color: Color | None = (255, 0, 0),
    fill_color: Color | None = None,
    width: int = 3,
    mask_only: bool = False,
) -> Image.Image:
    """Draw annotation geometries onto an image, rescaled and offset to fit it.

    Args:
        image: Image to draw on. Never modified.
        annotations: Annotations to draw. Point and polygon types are
            supported.
        use_base_coordinates: Use each annotation's level-0 coordinates rather
            than its own resolution's.
        scale: Factor applied to coordinates after subtracting
            ``relative_bounds``.
        relative_bounds: ``(x, y)`` origin subtracted from every coordinate,
            for drawing onto a crop rather than the whole slide.
        plot_mask: Draw each annotation's ``mask_coordinates`` -- the
            intersection with its parent annotation attached by
            :class:`~wsi_data.wholeslidedata.callbacks.MaskedTiledAnnotationCallback`
            -- instead of its own outline.
        outline_color: Polygon outline colour, or ``None``.
        fill_color: Polygon/point fill colour, or ``None``.
        width: Outline width in pixels.
        mask_only: Draw onto a fresh black canvas instead of a copy of
            ``image``, producing a mask rather than an overlay.

    Returns:
        A new image with the annotations drawn on it.

    Raises:
        ValueError: If an annotation is neither a point nor a polygon.
    """
    source = _as_pil(image)
    canvas = Image.new("RGB", source.size) if mask_only else source.copy()
    origin = np.asarray(relative_bounds)
    draw = ImageDraw.Draw(canvas)

    for annotation in annotations:
        if plot_mask:
            coordinate_sets = list(annotation.mask_coordinates)
        elif use_base_coordinates:
            coordinate_sets = [annotation.base_coordinates]
        else:
            coordinate_sets = [annotation.coordinates]

        if plot_mask and use_base_coordinates:
            offset = np.asarray(annotation.bounds[:2])
            coordinate_sets = [np.asarray(c) - offset for c in coordinate_sets]
        else:
            coordinate_sets = [
                (np.asarray(c) - origin) * scale for c in coordinate_sets
            ]

        if annotation.type == "point":
            for coordinates in coordinate_sets:
                points = [tuple(xy) for xy in np.atleast_2d(coordinates)]
                draw.point(points, fill=fill_color)
        elif annotation.type == "polygon":
            for coordinates in coordinate_sets:
                draw.polygon(
                    [tuple(xy) for xy in coordinates],
                    outline=outline_color,
                    fill=fill_color,
                    width=width,
                )
        else:
            msg = f"cannot draw annotation of type {annotation.type!r}"
            raise ValueError(msg)

    return canvas


def get_annotation_mask(
    image: Image.Image | np.ndarray,
    annotations: Sequence[Annotation],
    *,
    scale: float = 1.0,
    relative_bounds: Sequence[int] = (0, 0),
) -> Image.Image:
    """Rasterise annotations into a white-on-black mask the size of ``image``.

    Args:
        image: Image whose size the mask should match. Only its size is used.
        annotations: Annotations to rasterise.
        scale: Factor applied to coordinates after subtracting
            ``relative_bounds``.
        relative_bounds: ``(x, y)`` origin subtracted from every coordinate.

    Returns:
        An RGB mask image, white inside the annotations and black outside.
    """
    return draw_relative_annotations(
        image,
        annotations,
        scale=scale,
        relative_bounds=relative_bounds,
        outline_color=(255, 255, 255),
        fill_color=(255, 255, 255),
        width=1,
        mask_only=True,
    )


def plot_label_distribution(
    distribution: LabelDistribution,
    *,
    ax: Axes | None = None,
    **bar_kwargs: Any,
) -> Axes:
    """Plot a label distribution as a bar chart.

    Note:
        This replaces the pre-1.0 ``seaborn.displot`` call. The counts are
        already computed by
        :meth:`~wsi_data.labels.LabelDistribution.from_labels`, so re-deriving
        a histogram from raw labels through seaborn and pandas was
        unnecessary; a plain matplotlib bar chart of ``counts`` is equivalent.
        For a quick look with no plotting dependency at all, use
        :meth:`~wsi_data.labels.LabelDistribution.describe`.

    Args:
        distribution: The distribution to plot.
        ax: Axes to draw on. A new figure and axes are created when ``None``.
        **bar_kwargs: Forwarded to :meth:`matplotlib.axes.Axes.bar`.

    Returns:
        The axes drawn on; reach its figure via ``ax.figure``.

    Raises:
        ImportError: If matplotlib is not installed. Install the ``viz`` extra.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        msg = (
            "plot_label_distribution requires matplotlib; "
            "install it with `pip install 'wsi-data[viz]'`"
        )
        raise ImportError(msg) from exc

    if ax is None:
        _, ax = plt.subplots()
    ax.bar(distribution.class_names, distribution.counts, **bar_kwargs)
    ax.set_xlabel("/".join(distribution.keys))
    ax.set_ylabel("count")
    return ax
