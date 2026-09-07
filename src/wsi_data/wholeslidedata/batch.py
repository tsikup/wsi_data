"""Assembly of the ``wholeslidedata`` batch-sampler stack.

``wholeslidedata`` composes batching out of six collaborating samplers.
:func:`create_batch_sampler` wires up the combination this package needs --
one deterministic pass over every tile annotation in a dataset -- so callers
do not have to.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from sourcelib.associations import associate_files
from wholeslidedata.data.dataset import WholeSlideDataSet
from wholeslidedata.samplers.annotationsampler import OrderedAnnotationSampler
from wholeslidedata.samplers.batchsampler import BatchSampler
from wholeslidedata.samplers.batchshape import BatchShape
from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler
from wholeslidedata.samplers.patchsampler import PatchSampler
from wholeslidedata.samplers.pointsampler import CenterPointSampler
from wholeslidedata.samplers.samplesampler import SampleSampler

from wsi_data.wholeslidedata.dataset import MultiResWholeSlideDataSet
from wsi_data.wholeslidedata.samplers import (
    BatchOneTimeReferenceSampler,
    MultiResPatchSampler,
    MultiResSampleSampler,
    OrderedLabelOneTimeSampler,
    RandomOneTimeAnnotationSampler,
)
from wsi_data.wholeslidedata.sources import FileType, get_files

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["create_batch_sampler"]

_DEFAULT_LABELS = {"tissue": 0, "tumor": 1}
_DEFAULT_MULTIRES_SPACING = {"target": 0.5, "context": 2.0}
_DEFAULT_SINGLERES_SPACING = 0.5


def _batch_spacing_and_shape(
    spacing: Mapping[str, float] | float, tile_size: int
) -> tuple[list[tuple[str, float]] | float, list[list[int]] | list[int]]:
    """Build the ``spacing``/``shape`` arguments :class:`BatchShape` expects.

    Note:
        Narrowed with an inline ``isinstance`` rather than a separate
        ``multires`` bool: mypy cannot correlate that flag with ``spacing``'s
        type, even where a caller has already checked they agree.
    """
    if isinstance(spacing, Mapping):
        return list(spacing.items()), [[tile_size, tile_size, 3] for _ in spacing]
    return spacing, [tile_size, tile_size, 3]


def create_batch_sampler(
    slides_dir: Path | str | None = None,
    annotations_dir: Path | str | None = None,
    image_files: Sequence[Any] | None = None,
    annotation_files: Sequence[Any] | None = None,
    *,
    slide_extension: str = ".ndpi",
    ann_extension: str = ".geojson",
    file_type: FileType | str = FileType.MULTIRES_IMAGE,
    tile_size: int = 512,
    tissue_percentage: float = 0.5,
    stride_overlap_percentage: float = 0.0,
    intersection_percentage: float = 1.0,
    blurriness_threshold: Mapping[str, int | None] | None = None,
    batch_size: int = 1,
    labels: Mapping[str, int] | None = None,
    spacing: Mapping[str, float] | float | None = None,
    ratio: float = 1,
    seed: int = 123,
    random_annotations: bool = False,
) -> tuple[BatchSampler, BatchOneTimeReferenceSampler, BatchShape]:
    """Build a sampler stack that walks every tile annotation exactly once.

    Either pass ``slides_dir`` and ``annotations_dir`` to discover files, or
    pass ``image_files`` and ``annotation_files`` directly.

    Args:
        slides_dir: Directory of slides, used when ``image_files`` is ``None``.
        annotations_dir: Directory of annotations, likewise.
        image_files: Pre-discovered slide files, e.g. from
            :func:`~wsi_data.wholeslidedata.sources.get_files`.
        annotation_files: Pre-discovered annotation files. Required alongside
            ``image_files``.
        slide_extension: Substring filter for discovered slides.
        ann_extension: Substring filter for discovered annotations; also
            selects the parser.
        file_type: ``"mrwsi"`` for multi-resolution or ``"wsi"`` for
            single-resolution reads.
        tile_size: Tile side length in pixels.
        tissue_percentage: Minimum tissue fraction for a sampled patch.
        stride_overlap_percentage: Tile overlap as a fraction of ``tile_size``.
        intersection_percentage: Minimum tile coverage to keep a tile.
        blurriness_threshold: Per-resolution blur thresholds, or ``None``.
        batch_size: Annotations per batch.
        labels: Label name to value mapping. Defaults to
            ``{"tissue": 0, "tumor": 1}``.
        spacing: Resolution mapping for ``"mrwsi"`` (default
            ``{"target": 0.5, "context": 2.0}``), or a single spacing for
            ``"wsi"`` (default ``0.5``).
        ratio: Scale applied to tile size and overlap by the tiling callback.
        seed: Seed for the label and annotation samplers.
        random_annotations: Visit each label's annotations in random rather
            than sequential order. Every annotation is still visited once.

    Returns:
        A ``(batch_sampler, batch_reference_sampler, batch_shape)`` triple.

    Raises:
        ValueError: If neither directories nor file lists are supplied, if
            only one of the two file lists is given, or if ``spacing`` does not
            match ``file_type``.
    """
    file_type = FileType(file_type)
    multires = file_type is FileType.MULTIRES_IMAGE
    if labels is None:
        labels = dict(_DEFAULT_LABELS)

    # Defaulted per file type, since a single-resolution dataset cannot use
    # the multi-resolution default. The pre-1.0 code defaulted `spacing` to a
    # dict unconditionally and then asserted it was a float for `wsi`, so
    # single-resolution use failed on the default arguments.
    if spacing is None:
        spacing = (
            dict(_DEFAULT_MULTIRES_SPACING) if multires else _DEFAULT_SINGLERES_SPACING
        )
    if multires and not isinstance(spacing, Mapping):
        msg = (
            f"file_type={file_type.value!r} needs a mapping of spacings, "
            f"got {spacing!r}"
        )
        raise ValueError(msg)
    if not multires and isinstance(spacing, Mapping):
        msg = (
            f"file_type={file_type.value!r} reads one resolution, so spacing must be "
            f"a number, got {spacing!r}"
        )
        raise ValueError(msg)

    if image_files is None:
        if slides_dir is None or annotations_dir is None:
            msg = (
                "provide either image_files and annotation_files, or both "
                "slides_dir and annotations_dir"
            )
            raise ValueError(msg)
        image_files, annotation_files = get_files(
            slides_dir=slides_dir,
            annotations_dir=annotations_dir,
            tile_size=tile_size,
            labels=labels,
            stride_overlap_percentage=stride_overlap_percentage,
            intersection_percentage=intersection_percentage,
            ratio=ratio,
            file_type=file_type,
            slide_extension=slide_extension,
            ann_extension=ann_extension,
        )
    elif annotation_files is None:
        msg = "annotation_files is required when image_files is given"
        raise ValueError(msg)

    associations = associate_files(image_files, annotation_files, exact_match=True)

    dataset: WholeSlideDataSet
    if multires:
        dataset = MultiResWholeSlideDataSet(
            mode="default", associations=associations, labels=list(labels)
        )
    else:
        dataset = WholeSlideDataSet(
            mode="default", associations=associations, labels=list(labels)
        )

    # `annotation_counts_per_label` is the upstream name; the pre-1.0 subclass
    # re-exposed it as `annotations_per_label`.
    counts_per_label = dataset.annotation_counts_per_label
    batch_ref_sampler = BatchOneTimeReferenceSampler(
        dataset=dataset,
        batch_size=batch_size,
        label_sampler=OrderedLabelOneTimeSampler(
            annotations_per_label=counts_per_label, seed=seed
        ),
        annotation_sampler=(
            RandomOneTimeAnnotationSampler(counts_per_label, seed=seed)
            if random_annotations
            else OrderedAnnotationSampler(counts_per_label, seed=seed)
        ),
        point_sampler=CenterPointSampler(),
    )

    batch_spacing, batch_tile_shape = _batch_spacing_and_shape(spacing, tile_size)
    batch_shape = BatchShape(
        batch_size,
        spacing=batch_spacing,
        shape=batch_tile_shape,
        labels=dataset.sample_labels,
    )

    sample_sampler: SampleSampler
    if multires:
        sample_sampler = MultiResSampleSampler(
            patch_sampler=MultiResPatchSampler(
                tissue_percentage=tissue_percentage,
                blurriness_threshold=blurriness_threshold,
            ),
            patch_label_sampler=SegmentationPatchLabelSampler(),
            batch_shape=batch_shape,
        )
    else:
        sample_sampler = SampleSampler(
            patch_sampler=PatchSampler(center=True, relative=False),
            patch_label_sampler=SegmentationPatchLabelSampler(),
            batch_shape=batch_shape,
        )

    return (
        BatchSampler(dataset=dataset, sampler=sample_sampler),
        batch_ref_sampler,
        batch_shape,
    )
