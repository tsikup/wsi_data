"""MultiResWholeSlideDataSet against lightweight fake file wrappers.

Fakes stand in for the real ``wholeslidedata`` file/annotation types --
opening a real slide needs openslide and an actual pyramidal image, whereas
this dataset's own logic only calls ``.open()``/``.copy()`` on its inputs, so
duck-typed fakes exercise the same code paths.
"""

from __future__ import annotations

import pytest
from wholeslidedata.annotation.labels import Label
from wholeslidedata.data.mode import WholeSlideMode

from wsi_data.wholeslidedata.dataset import MultiResWholeSlideDataSet


class _FakeAnnotation:
    def __init__(self, label_name: str, index: int = 0):
        self.label = Label.create({"name": label_name, "value": 1})
        self.index = index


class _FakeWSA:
    def __init__(self, label_names):
        self._annotations = [
            _FakeAnnotation(name, i) for i, name in enumerate(label_names)
        ]
        self.sampling_annotations = self._annotations

    @property
    def annotations(self):
        return self._annotations


class _FakeWSAFile:
    def __init__(self, label_names):
        self._label_names = label_names
        self.copy_called_with = None

    def copy(self, destination):
        self.copy_called_with = destination

    def open(self, labels=None):  # noqa: ARG002 - matches the real signature
        return _FakeWSA(self._label_names)


class _FakeOpenedImage:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeWSIFile:
    def __init__(self):
        self.copy_called_with = None

    def copy(self, destination):
        self.copy_called_with = destination

    def open(self):
        return _FakeOpenedImage()


def _associations(wsi_files=None, wsa_files=None):
    return {"slide1": {"wsi": wsi_files or [], "wsa": wsa_files or []}}


class TestMultiResWholeSlideDataSet:
    def test_builds_sample_references_from_annotations(self):
        dataset = MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations(
                [_FakeWSIFile()], [_FakeWSAFile(["tumor", "tumor"])]
            ),
            labels=["tumor"],
        )
        assert list(dataset.sample_references) == ["tumor"]
        assert len(dataset.sample_references["tumor"]) == 2

    def test_load_images_true_opens_the_slide(self):
        """Regression: load_images/copy_path were overwritten by super().__init__.

        Assigning `self._load_images` before calling `super().__init__()`
        (which sets its own default and only then calls `_open`) meant the
        pre-1.0 subclass's `load_images` argument had no effect at all.
        """
        dataset = MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations([_FakeWSIFile()], [_FakeWSAFile(["tumor"])]),
            labels=["tumor"],
            load_images=True,
        )
        image = next(iter(dataset._data["slide1"][dataset.IMAGES_KEY].values()))
        assert isinstance(image, _FakeOpenedImage)

    def test_load_images_false_keeps_the_file_wrapper(self):
        wsi_file = _FakeWSIFile()
        dataset = MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations([wsi_file], [_FakeWSAFile(["tumor"])]),
            labels=["tumor"],
            load_images=False,
        )
        image = next(iter(dataset._data["slide1"][dataset.IMAGES_KEY].values()))
        assert image is wsi_file

    def test_copy_path_copies_both_images_and_annotations(self):
        wsi_file = _FakeWSIFile()
        wsa_file = _FakeWSAFile(["tumor"])
        MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations([wsi_file], [wsa_file]),
            labels=["tumor"],
            copy_path="/staging",
        )
        assert wsi_file.copy_called_with == "/staging"
        assert wsa_file.copy_called_with == "/staging"

    def test_close_images_closes_and_clears(self):
        """Regression: `close_images` iterated `self._images`, which never existed.

        It always raised `AttributeError` -- this checks it now actually
        closes every opened image and empties the images dict.
        """
        dataset = MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations([_FakeWSIFile()], [_FakeWSAFile(["tumor"])]),
            labels=["tumor"],
        )
        image = next(iter(dataset._data["slide1"][dataset.IMAGES_KEY].values()))

        dataset.close_images()

        assert image.closed
        assert dataset._data["slide1"][dataset.IMAGES_KEY] == {}

    def test_close_images_tolerates_unopened_slides(self):
        """`load_images=False` slides have no `.close()`; closing must not raise."""
        dataset = MultiResWholeSlideDataSet(
            mode="default",
            associations=_associations([_FakeWSIFile()], [_FakeWSAFile(["tumor"])]),
            labels=["tumor"],
            load_images=False,
        )
        dataset.close_images()  # must not raise
        assert dataset._data["slide1"][dataset.IMAGES_KEY] == {}

    def test_raises_when_no_samples_found(self):
        # `mode` must be the real enum here, not the "default" string every
        # other test in this module uses: upstream's own "no samples" message
        # does `self.mode.name`, which raises AttributeError on a plain str
        # before it ever gets to raising the ValueError under test.
        with pytest.raises(ValueError, match="No samples found"):
            MultiResWholeSlideDataSet(
                mode=WholeSlideMode.default,
                associations=_associations([_FakeWSIFile()], [_FakeWSAFile([])]),
                labels=[],
            )
