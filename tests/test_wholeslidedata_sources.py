"""File discovery: extension filtering, excludes, and the unbound file_type fix.

These exercise real path discovery on empty placeholder files -- construction
of a ``WholeSlideImageFile``/``WholeSlideAnnotationFile`` does not read the
file's content, so no real slide or an openslide install is needed here.
"""

from __future__ import annotations

import pytest
from sourcelib.collect import NoSourceFilesInFolderError

from wsi_data.wholeslidedata.files import (
    MultiResWholeSlideImageFile,
    SingleResWholeSlideImageFile,
)
from wsi_data.wholeslidedata.sources import (
    FileType,
    factory_sources_from_paths,
    get_files,
    whole_slide_files_from_folder_factory,
)


@pytest.fixture
def slide_dir(tmp_path):
    for name in ("a.ndpi", "b.ndpi", "a_mask.ndpi", "notes.txt"):
        (tmp_path / name).touch()
    return tmp_path


@pytest.fixture
def annotation_dir(tmp_path):
    (tmp_path / "a.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "b.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    return tmp_path


class TestFactorySourcesFromPaths:
    def test_excludes_matching_paths(self, tmp_path):
        paths = [tmp_path / "keep.ndpi", tmp_path / "mask.ndpi"]
        files = factory_sources_from_paths(
            MultiResWholeSlideImageFile, "default", paths, filters=[], excludes=["mask"]
        )
        assert [f.path.name for f in files] == ["keep.ndpi"]

    def test_filters_require_a_match(self, tmp_path):
        paths = [tmp_path / "a.ndpi", tmp_path / "a.svs"]
        files = factory_sources_from_paths(
            MultiResWholeSlideImageFile, "default", paths, filters=[".svs"], excludes=[]
        )
        assert [f.path.name for f in files] == ["a.svs"]

    def test_deduplicates_paths(self, tmp_path):
        path = tmp_path / "a.ndpi"
        files = factory_sources_from_paths(
            MultiResWholeSlideImageFile,
            "default",
            [path, path],
            filters=[],
            excludes=[],
        )
        assert len(files) == 1

    def test_result_is_sorted(self, tmp_path):
        paths = [tmp_path / "b.ndpi", tmp_path / "a.ndpi"]
        files = factory_sources_from_paths(
            MultiResWholeSlideImageFile, "default", paths, filters=[], excludes=[]
        )
        assert [f.path.name for f in files] == ["a.ndpi", "b.ndpi"]


class TestWholeSlideFilesFromFolderFactory:
    def test_discovers_by_extension_and_excludes(self, slide_dir):
        files = whole_slide_files_from_folder_factory(
            slide_dir,
            FileType.MULTIRES_IMAGE,
            excludes=["mask"],
            filters=[".ndpi"],
            image_backend="openslide",
        )
        assert sorted(f.path.name for f in files) == ["a.ndpi", "b.ndpi"]
        assert all(isinstance(f, MultiResWholeSlideImageFile) for f in files)

    def test_singleres_file_type_produces_singleres_files(self, slide_dir):
        files = whole_slide_files_from_folder_factory(
            slide_dir, FileType.SINGLERES_IMAGE, filters=[".ndpi"], excludes=["mask"]
        )
        assert all(isinstance(f, SingleResWholeSlideImageFile) for f in files)

    def test_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(NoSourceFilesInFolderError):
            whole_slide_files_from_folder_factory(
                tmp_path, FileType.MULTIRES_IMAGE, filters=[".ndpi"]
            )

    def test_rejects_unknown_file_type(self, slide_dir):
        """Regression: an unrecognised file_type left `class_type` unbound.

        The pre-1.0 `if`/`elif` chain had no `else`, so this raised
        `UnboundLocalError` deep inside the function instead of reporting the
        bad argument.
        """
        with pytest.raises(ValueError, match="file_type must be one of"):
            whole_slide_files_from_folder_factory(slide_dir, "not-a-real-type")

    def test_accepts_a_file_class_directly(self, slide_dir):
        files = whole_slide_files_from_folder_factory(
            slide_dir,
            MultiResWholeSlideImageFile,
            filters=[".ndpi"],
            excludes=["mask"],
        )
        assert len(files) == 2


class TestGetFiles:
    def test_discovers_slides_and_annotations(self, slide_dir, annotation_dir):
        image_files, annotation_files = get_files(
            slides_dir=slide_dir, annotations_dir=annotation_dir, tiled=False
        )
        assert image_files is not None
        assert annotation_files is not None
        assert sorted(f.path.name for f in image_files) == ["a.ndpi", "b.ndpi"]
        expected = ["a.geojson", "b.geojson"]
        assert sorted(f.path.name for f in annotation_files) == expected

    def test_slides_only(self, slide_dir):
        image_files, annotation_files = get_files(slides_dir=slide_dir)
        assert image_files is not None
        assert annotation_files is None

    def test_annotations_only(self, annotation_dir):
        image_files, annotation_files = get_files(
            annotations_dir=annotation_dir, tiled=False
        )
        assert image_files is None
        assert annotation_files is not None

    def test_neither_directory_returns_none_for_both(self):
        image_files, annotation_files = get_files()
        assert image_files is None
        assert annotation_files is None

    def test_return_raw_annotation_adds_a_third_element(self, annotation_dir):
        result = get_files(
            annotations_dir=annotation_dir, tiled=False, return_raw_annotation=True
        )
        assert len(result) == 3
        _, _, raw = result
        assert raw is not None
