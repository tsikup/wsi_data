"""Discovery of slide and annotation files on disk.

One of the three pieces the pre-1.0 ``wholeslidedata/utils.py`` grab-bag was
split into: file discovery here, batch-sampler assembly in
:mod:`wsi_data.wholeslidedata.batch`, and drawing in :mod:`wsi_data.viz`.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from natsort import os_sorted
from sourcelib.collect import NoSourceFilesInFolderError
from wholeslidedata.annotation.parser import AnnotationParser
from wholeslidedata.data.files import WholeSlideAnnotationFile
from wholeslidedata.interoperability.qupath.parser import QuPathAnnotationParser

from wsi_data.wholeslidedata.callbacks import MaskedTiledAnnotationCallback
from wsi_data.wholeslidedata.files import (
    MultiResWholeSlideImageFile,
    SingleResWholeSlideImageFile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "FileType",
    "factory_sources_from_paths",
    "get_files",
    "whole_slide_files_from_folder_factory",
]


class FileType(StrEnum):
    """Which file wrapper class a discovery call should produce.

    Note:
        The pre-1.0 code took a bare string and matched it with an
        ``if``/``elif`` chain that had no ``else``, so any unrecognised value
        left the class variable unbound and failed with ``UnboundLocalError``
        further down instead of reporting the bad argument.
    """

    MULTIRES_IMAGE = "mrwsi"
    SINGLERES_IMAGE = "wsi"
    ANNOTATION = "wsa"


_FILE_CLASSES: dict[FileType, type[Any]] = {
    FileType.MULTIRES_IMAGE: MultiResWholeSlideImageFile,
    FileType.SINGLERES_IMAGE: SingleResWholeSlideImageFile,
    FileType.ANNOTATION: WholeSlideAnnotationFile,
}


def _file_class(file_type: FileType | str | type[Any]) -> type[Any]:
    """Resolve a file-type name, enum member or class to a file class."""
    if isinstance(file_type, type):
        return file_type
    try:
        return _FILE_CLASSES[FileType(file_type)]
    except ValueError:
        valid = ", ".join(repr(member.value) for member in FileType)
        msg = f"file_type must be one of {valid} or a class, got {file_type!r}"
        raise ValueError(msg) from None


def _annotation_parser(extension: str) -> type[Any]:
    """Pick the annotation parser matching an annotation file extension."""
    parser = QuPathAnnotationParser if extension == ".geojson" else AnnotationParser
    return cast("type[Any]", parser)


def factory_sources_from_paths(
    cls: type[Any],
    mode: str,
    paths: Sequence[Path | str],
    filters: Sequence[str],
    excludes: Sequence[str],
    **kwargs: Any,
) -> list[Any]:
    """Wrap each path in ``cls``, dropping ones excluded or not matched.

    Args:
        cls: File wrapper class to construct.
        mode: Dataset mode passed to each file.
        paths: Candidate paths. Duplicates are collapsed.
        filters: Substrings a path must contain at least one of, if non-empty.
        excludes: Substrings that disqualify a path.
        **kwargs: Forwarded to ``cls``.

    Returns:
        The constructed file wrappers, sorted for determinism.
    """
    files = []
    for path in sorted({str(p) for p in paths}):
        if any(exclude in path for exclude in excludes):
            continue
        if filters and not any(pattern in path for pattern in filters):
            continue
        files.append(cls(mode=mode, path=path, **kwargs))
    return files


def whole_slide_files_from_folder_factory(
    folder: Path | str,
    file_type: FileType | str | type[Any],
    mode: str = "default",
    filters: Sequence[str] = (),
    excludes: Sequence[str] = (),
    *,
    recursive: bool = False,
    **kwargs: Any,
) -> list[Any]:
    """Find every file of a type in a folder and wrap it.

    Args:
        folder: Directory to search.
        file_type: A :class:`FileType`, its string value, or a file class.
        mode: Dataset mode passed to each file.
        filters: Substrings a path must contain at least one of.
        excludes: Substrings that disqualify a path.
        recursive: Search subdirectories too.
        **kwargs: Forwarded to the file class, e.g. ``image_backend`` or
            ``annotation_parser``.

    Returns:
        The discovered file wrappers.

    Raises:
        ValueError: If ``file_type`` is not recognised.
        NoSourceFilesInFolderError: If nothing matched.
    """
    class_type = _file_class(file_type)
    folder = Path(folder)
    all_sources: list[Any] = []

    for extension in class_type.EXTENSIONS:
        pattern = f"*{extension}"
        paths = os_sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
        all_sources.extend(
            factory_sources_from_paths(
                class_type, mode, paths, filters, excludes, **kwargs
            )
        )

    if not all_sources:
        raise NoSourceFilesInFolderError(class_type, filters, excludes, folder)
    return all_sources


def get_files(
    slides_dir: Path | str | None = None,
    annotations_dir: Path | str | None = None,
    *,
    tile_size: int = 512,
    labels: Mapping[str, int] | None = None,
    stride_overlap_percentage: float = 0.0,
    intersection_percentage: float = 1.0,
    ratio: float = 1,
    file_type: FileType | str | type[Any] = FileType.MULTIRES_IMAGE,
    slide_extension: str = ".ndpi",
    ann_extension: str = ".geojson",
    tiled: bool = True,
    return_raw_annotation: bool = False,
    segmentation_labels: Mapping[str, int] | None = None,
) -> tuple[list[Any] | None, ...]:
    """Discover slides and annotations, wiring up the tiling callback.

    Args:
        slides_dir: Directory of slides, or ``None`` to skip slides.
        annotations_dir: Directory of annotations, or ``None`` to skip them.
        tile_size: Tile side length for the tiling callback.
        labels: Label name to value mapping. Defaults to ``{"tissue": 1}``.
        stride_overlap_percentage: Tile overlap as a fraction of ``tile_size``.
        intersection_percentage: Minimum tile coverage to keep a tile.
        ratio: Scale applied to tile size and overlap by the callback.
        file_type: Which image file class to produce.
        slide_extension: Substring filter for slide files.
        ann_extension: Substring filter for annotation files; also selects the
            parser.
        tiled: Attach the tiling callback. When ``False``, annotations are
            returned whole.
        return_raw_annotation: Also return the annotations parsed with
            ``segmentation_labels`` and no tiling, for full-region masks.
        segmentation_labels: Labels for the raw annotation pass. Defaults to
            ``{"tissue": 1, "tumor": 2}``.

    Returns:
        ``(image_files, annotation_files)``, or
        ``(image_files, annotation_files, raw_annotation_files)`` when
        ``return_raw_annotation``. Entries are ``None`` when the matching
        directory was not given.
    """
    image_files = None
    annotation_files = None
    raw_annotation_files = None

    if slides_dir is not None:
        image_files = whole_slide_files_from_folder_factory(
            slides_dir,
            file_type,
            excludes=["mask"],
            filters=[slide_extension],
            image_backend="openslide",
        )

    if annotations_dir is not None:
        if labels is None:
            labels = {"tissue": 1}
        parser_class = _annotation_parser(ann_extension)
        callbacks = (
            (
                MaskedTiledAnnotationCallback(
                    tile_size=tile_size,
                    label_names=list(labels),
                    ratio=ratio,
                    overlap=int(tile_size * stride_overlap_percentage),
                    intersection_percentage=intersection_percentage,
                    full_coverage=True,
                ),
            )
            if tiled
            else None
        )
        annotation_files = whole_slide_files_from_folder_factory(
            annotations_dir,
            FileType.ANNOTATION,
            excludes=["tif"],
            filters=[ann_extension],
            annotation_parser=parser_class(labels=labels, callbacks=callbacks),
        )

        if return_raw_annotation:
            if segmentation_labels is None:
                segmentation_labels = {"tissue": 1, "tumor": 2}
            raw_annotation_files = whole_slide_files_from_folder_factory(
                annotations_dir,
                FileType.ANNOTATION,
                excludes=["tif"],
                filters=[ann_extension],
                annotation_parser=parser_class(
                    labels=segmentation_labels, callbacks=None
                ),
            )

    if return_raw_annotation:
        return image_files, annotation_files, raw_annotation_files
    return image_files, annotation_files
