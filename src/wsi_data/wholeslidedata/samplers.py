"""Multi-resolution and single-pass variants of ``wholeslidedata``'s samplers.

The upstream samplers are built for endless random sampling during training.
The ``OneTime`` variants here instead walk every annotation exactly once,
which is what inference and feature extraction need, and the ``MultiRes``
variants extend patch sampling to several resolutions per point.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from wholeslidedata.samplers.annotationsampler import AnnotationSampler
from wholeslidedata.samplers.batchreferencesampler import BatchReferenceSampler
from wholeslidedata.samplers.labelsampler import LabelSampler
from wholeslidedata.samplers.samplesampler import SampleSampler

from wsi_data.wholeslidedata.wholeslideimage import MultiResWholeSlideImage

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from wholeslidedata import WholeSlideAnnotation
    from wholeslidedata.samplers.batchshape import BatchShape
    from wholeslidedata.samplers.patchlabelsampler import PatchLabelSampler

    from wsi_data.wholeslidedata.files import MultiResWholeSlideImageFile

__all__ = [
    "BatchOneTimeReferenceSampler",
    "MultiResPatchSampler",
    "MultiResSampleSampler",
    "OrderedLabelOneTimeSampler",
    "RandomOneTimeAnnotationSampler",
]


@contextmanager
def _opened(
    image: MultiResWholeSlideImage | MultiResWholeSlideImageFile,
) -> Iterator[MultiResWholeSlideImage]:
    """Yield an open slide, closing it again only if it was opened here.

    A dataset built with ``load_images=False`` hands samplers file wrappers
    rather than open slides, so each sampled point has to open and close its
    own handle; an already-open slide must be left open for the next point.
    """
    if isinstance(image, MultiResWholeSlideImage):
        yield image
        return
    wsi = image.open()
    try:
        yield wsi
    finally:
        wsi.close()


class MultiResPatchSampler:
    """Read one patch per resolution at a point, plus each one's downsampling.

    Args:
        center: Treat sampled points as patch centres.
        relative: Treat sampled points as relative to their spacing's
            dimensions rather than level 0.
        tissue_percentage: Minimum tissue fraction, recorded for downstream
            quality-control callbacks to apply.
        blurriness_threshold: Per-resolution blur thresholds, likewise.
    """

    def __init__(
        self,
        tissue_percentage: float = 0.5,
        blurriness_threshold: Mapping[str, int | None] | None = None,
        *,
        center: bool = True,
        relative: bool = False,
    ) -> None:
        self._center = center
        self._relative = relative
        self.tissue_percentage = tissue_percentage
        self.blurriness_threshold = blurriness_threshold

    def sample(
        self,
        image: MultiResWholeSlideImage | MultiResWholeSlideImageFile,
        point: Sequence[int],
        size: Sequence[int],
        pixel_spacings: Mapping[str, float],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Sample every resolution at one point.

        Args:
            image: An open multi-resolution slide, or a file wrapper to open.
            point: ``(x, y)`` location.
            size: ``(width, height)`` of each patch.
            pixel_spacings: Resolution name to spacing mapping.

        Returns:
            A ``(patches, downsamplings)`` pair, both keyed by resolution name.
        """
        with _opened(image) as wsi:
            patch = wsi.get_data(
                point[0],
                point[1],
                size[0],
                size[1],
                pixel_spacings,
                center=self._center,
                relative=self._relative,
            )
            ratio = {
                key: wsi.get_downsampling_from_spacing(value)
                for key, value in pixel_spacings.items()
            }
        return patch, ratio


class MultiResSampleSampler(SampleSampler):
    """Assemble one multi-resolution sample and its per-resolution masks.

    Args:
        patch_sampler: Reads the patches.
        patch_label_sampler: Rasterises annotation masks per resolution.
        batch_shape: Declares the resolutions and patch shapes to produce.
        sample_callbacks: Callbacks applied to each ``(patch, mask)`` pair.
    """

    def __init__(
        self,
        patch_sampler: MultiResPatchSampler,
        patch_label_sampler: PatchLabelSampler,
        batch_shape: BatchShape,
        sample_callbacks: Sequence[Any] | None = None,
    ) -> None:
        self._batch_shape = batch_shape
        self._patch_sampler = patch_sampler
        self._patch_label_sampler = patch_label_sampler
        self._sample_callbacks = sample_callbacks

    def sample(
        self,
        wsi: MultiResWholeSlideImage,
        wsa: WholeSlideAnnotation,
        point: Sequence[int],
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """Sample one point into the batch shape's slots.

        Args:
            wsi: The slide to read from.
            wsa: Its annotations, for mask rasterisation.
            point: ``(x, y)`` location.

        Returns:
            An ``(x_samples, y_samples)`` pair keyed by ``(name, spacing)``.

        Raises:
            ValueError: If the batch shape declares no ``"target"`` resolution.
        """
        x_samples = self._init_samples()
        y_samples = self._init_samples()

        spacings = {key[0]: key[1] for key in x_samples}
        if "target" not in spacings:
            msg = (
                "the BatchShape must declare a 'target' resolution, "
                f"got {sorted(spacings)}"
            )
            raise ValueError(msg)

        patch_shape = next(iter(x_samples[("target", spacings["target"])]))
        x_sample, y_sample = self._sample(point, wsi, wsa, patch_shape, spacings)

        for key, value in spacings.items():
            if key in x_sample:
                x_samples[(key, value)][tuple(patch_shape)] = x_sample[key]
            if key in y_sample:
                y_samples[(key, value)][tuple(patch_shape)] = y_sample[key]

        self._reset_sample_callbacks()
        return x_samples, y_samples

    def _sample(
        self,
        point: Sequence[int],
        wsi: MultiResWholeSlideImage,
        wsa: WholeSlideAnnotation,
        patch_shape: Sequence[int],
        pixel_spacings: Mapping[str, float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read patches and masks, then apply the sample callbacks."""
        data, ratio = self._patch_sampler.sample(
            wsi, point, patch_shape[:2], pixel_spacings
        )

        # "graph" is a pseudo-resolution consumed elsewhere; it has no pixels
        # and so no mask.
        pixel_keys = [key for key in pixel_spacings if key != "graph"]

        label = {
            key: self._patch_label_sampler.sample(
                wsa=wsa, point=point, size=patch_shape[:2], ratio=ratio[key]
            )
            for key in pixel_keys
        }
        for key in pixel_keys:
            data[key], label = self._apply_sample_callbacks(data[key], label)

        return {key: data[key] for key in data if key in pixel_spacings}, label

    def _init_samples(self) -> dict[Any, dict[Any, list[Any]]]:
        return {
            tuple(spacing): {tuple(input_size): [] for input_size in sizes}
            for spacing, sizes in self._batch_shape.items()
        }

    def _apply_sample_callbacks(self, patch: Any, mask: Any) -> tuple[Any, Any]:
        if self._sample_callbacks:
            for callback in self._sample_callbacks:
                patch, mask = callback(patch, mask)
        return patch, mask


class RandomOneTimeAnnotationSampler(AnnotationSampler):
    """Draw each annotation of a label once, in random order.

    Args:
        counts_per_label: Number of annotations per label name.
        seed: Seed for the draw order.
    """

    def __init__(self, counts_per_label: Mapping[str, int], seed: int) -> None:
        super().__init__(counts_per_label=counts_per_label, seed=seed)
        self._remaining = {
            label: list(range(count)) for label, count in counts_per_label.items()
        }
        self.reset()

    def _next(self, label: str) -> int | None:
        """Return an unvisited annotation index, or ``None`` when exhausted."""
        remaining = self._remaining[label]
        if not remaining:
            return None
        index = int(self._rng.choice(remaining))
        remaining.remove(index)
        return index

    def _reset_label(self, label: str) -> None:
        """No-op: exhausted labels are not refilled, by design."""

    def update(self, data: Any) -> None:
        """No-op: this sampler's order does not depend on sampled data."""

    def reset(self) -> None:
        """Reset the random generator to its seed."""
        self.set_seed()


class OrderedLabelOneTimeSampler(LabelSampler):
    """Yield each label as many times as it has annotations, then stop.

    Args:
        annotations_per_label: Number of annotations per label name.
        seed: Accepted for interface compatibility; the order is deterministic.
    """

    def __init__(
        self, annotations_per_label: Mapping[str, int], seed: int = 123
    ) -> None:
        labels = [
            label
            for label, count in annotations_per_label.items()
            for _ in range(count)
        ]
        super().__init__(labels, seed=seed)
        self._labels_cycle: Iterator[str] = iter(self._labels)
        self.reset()

    def __len__(self) -> int:
        return len(self._labels)

    def __next__(self) -> str | None:
        """Return the next label, or ``None`` once every one has been yielded."""
        return next(self._labels_cycle, None)

    def reset(self) -> None:
        """Restart from the first label."""
        self._labels_cycle = iter(self._labels)

    def update(self, batch: Any) -> None:
        """No-op: this sampler's order does not depend on sampled data."""


class BatchOneTimeReferenceSampler(BatchReferenceSampler):
    """Build batches of sample references, stopping once samplers are exhausted.

    Batches near the end may be shorter than ``batch_size``, and the final
    batch may be empty.
    """

    def __len__(self) -> int:
        try:
            return len(self._label_sampler)
        except (AttributeError, TypeError):
            return 0

    def batch(self) -> list[dict[str, Any]]:
        """Return the next batch of ``{"reference", "point"}`` entries."""
        batch = []
        for _ in range(self._batch_size):
            label = next(self._label_sampler)
            if label is None:
                continue

            index = next(self._annotation_sampler)(label)
            if index is None:
                # A one-time annotation sampler returns None once a label is
                # exhausted; indexing sample_references with it would raise.
                continue

            reference = self._dataset.sample_references[label][index]
            annotation = self._dataset.get_annotation_from_reference(reference)
            batch.append(
                {
                    "reference": reference,
                    "point": self._point_sampler.sample(annotation),
                }
            )
        return batch
