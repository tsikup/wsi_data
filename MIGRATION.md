# Migrating from pre-1.0

Version 1.0 is a full rewrite: a `src/` layout, `pyproject.toml` packaging,
type hints checked with `mypy --strict`, a real test suite, and fixes for a
long list of latent bugs. The public API changed on every import path —
there are no compatibility shims. This document maps the old surface to the
new one and explains every behavioural change.

## Why a clean break

The pre-1.0 package could not run at all in a modern environment: its
`samplers/__init__.py` was empty while `wsi_data.utils` imported three names
from it, so `import wsi_data.utils` — and therefore `wsi_data.augmentations`,
which imports from `utils` in turn — always raised `ImportError`. Its
declared dependency on `wholeslidedata` pointed at a PyPI release
(`0.0.16`, from 2023) that predates the package layout the code actually
imports; `albumentations`, unpinned, resolved to a 2.x release whose API
silently broke two of the library's own custom transforms; and `stainlib`,
one of its dependencies, was never published anywhere. Fixing all of that in
place while keeping the old names and shapes would have preserved a design —
no declared public API (every `__init__.py` was empty), `wsi_data.datasets.h5_datasets`
as a 767-line monolith, three dataset classes that only forwarded constructor
arguments — that a rewrite was the better fix for.

## Package layout

| Old | New |
|---|---|
| `wsi_data.utils` | `wsi_data.loaders` (`weak_shuffling_h5_fast_loader`); `to_tuple` moved into `wsi_data.augmentations`, its only caller |
| `wsi_data.graphs` (`CellGraphExtractor`) | Dropped. Depended on `histocartography` (last released 2022-01) and `dgl` (last released 2024-05, wheels pinned to specific torch/CUDA builds); neither installs against current torch. No replacement is provided — reimplementing cell-graph extraction was out of scope for this rewrite. |
| `wsi_data.datasets.h5_datasets` | Split into `wsi_data.datasets.h5.{features,images,tiles}` |
| `wsi_data.datasets.wsi_datasets` (`Single_WSI_Dataset`) | `wsi_data.datasets.slide` (`SlideTileDataset`) |
| `wsi_data.datasets.fake_dataset` | `wsi_data.datasets.fake` |
| `wsi_data.wholeslidedata.utils` | Split into `wsi_data.wholeslidedata.sources` (file discovery), `wsi_data.wholeslidedata.batch` (`create_batch_sampler`), and `wsi_data.viz` (drawing) |
| `wsi_data.wholeslidedata.wholeslideimage.MyWholeSlideImage` | `wsi_data.wholeslidedata.SingleResWholeSlideImage` |
| `wsi_data.wholeslidedata.wholeslideimage.MultiResWholeSlideImage` | Same name, now a sibling of `SingleResWholeSlideImage` rather than its parent — see below |
| `wsi_data.wholeslidedata.files.MyWholeSlideImageFile` | `wsi_data.wholeslidedata.SingleResWholeSlideImageFile` |

Everything most callers need is re-exported from the top-level `wsi_data`
package now; see its docstring for the full list.

## The whole-slide-image inheritance was inverted

Before 1.0, `MyWholeSlideImage` (single resolution) *subclassed*
`MultiResWholeSlideImage` and overrode `get_data` to return a bare array
where its parent returned a dict of arrays — a subclass narrowing its
parent's contract, which is unsound (it could not be used anywhere the
parent was expected) and made `isinstance` checks against the multi-res
class silently match single-res instances too.

`SingleResWholeSlideImage` and `MultiResWholeSlideImage` are now siblings
sharing a `BaseWholeSlideImage` base. If you subclassed
`MultiResWholeSlideImage` expecting to get single-resolution behaviour for
free, subclass `SingleResWholeSlideImage` instead.

## `wsi_data.wholeslidedata.dataset.MultiResWholeSlideDataSet`

About 120 of this class's 182 lines duplicated its upstream
`wholeslidedata.data.dataset.WholeSlideDataSet` parent byte-for-byte or under
a different name:

| Old (this class) | Upstream (now simply inherited) |
|---|---|
| `annotations_per_label` | `annotation_counts_per_label` |
| `annotations_per_key` | `annotation_counts_per_key` |
| `annotations_per_label_per_key` | `annotation_counts_per_label_per_key` |
| `pixels_count`, `pixels_per_label`, `pixels_per_key`, `pixels_per_label_per_key` | identical properties, inherited directly |
| `_init_labels`, `_init_samples` | identical methods, inherited directly |

Use the upstream names. What genuinely differed from the parent — opening
annotations without a rescaling spacing, since QuPath GeoJSON coordinates are
already at level 0 — is the only thing this subclass still overrides.

## HDF5 datasets

`wsi_data.datasets.h5_datasets.DatasetHDF5` plus its three subclasses
(`ImageOnlyDatasetHDF5`, `SegmentationDatasetHDF5`, `ClassificationDatasetHDF5`,
which only forwarded constructor arguments) are now one class,
`wsi_data.datasets.h5.ImageDatasetHDF5`, selected by a `mode` argument:

```python
# Old
from wsi_data.datasets.h5_datasets import SegmentationDatasetHDF5

dataset = SegmentationDatasetHDF5(data_dir, data_name, data_cols, transform)

# New
from wsi_data.datasets.h5 import ImageDatasetHDF5

dataset = ImageDatasetHDF5(
    data_dir, data_name, data_cols, transform, mode="segmentation"
)
```

`mode="image_only"` replaces `ImageOnlyDatasetHDF5`; `mode="classification"`
(the default `image_only=False, segmentation=False`) replaces
`ClassificationDatasetHDF5`. The removed `segmentation=True, image_only=True`
combination was never a meaningful configuration.

`FeatureDatasetHDF5`'s `load_ram` parameter is gone: with `load_ram=False` the
pre-1.0 class returned `h5py.Dataset` handles opened inside a closed `with
h5py.File(...)` block, so reading one raised `RuntimeError: Unable to
synchronously get dataspace (invalid dataset identifier)`. Since one item is
one whole slide's features — which callers need in full regardless — features
are now always materialised into tensors.

`get_label_distribution()` on every dataset class now returns one consistent
type, `wsi_data.labels.LabelDistribution`, instead of a bare tuple, a pandas
`Series`, or a rendered `seaborn` figure depending on arguments:

```python
# Old
dist, labels = dataset.get_label_distribution()  # tuple
fig = dataset.get_label_distribution(as_figure=True)  # or a figure

# New
dist = dataset.get_label_distribution()
dist.counts, dist.values, dist.positions  # what you need, always present
print(dist.describe())  # zero-dependency text summary
from wsi_data.viz import plot_label_distribution

plot_label_distribution(dist)  # needs the `viz` extra
```

## Augmentations

`get_augmentor` and its custom transforms now target `albumentations >= 2.0`
(the pre-1.0 code required `< 2.0`, unpinned, so it already silently
miscompiled on a modern install — see "Confirmed bugs" below).

`HEDAugmentor` no longer depends on `stainlib` (unpublished, unmaintained).
It reimplements the same haematoxylin/eosin/DAB stain-jitter transform on
`scikit-image`'s published stain matrices, with one numerical improvement:
unlike `skimage.color.rgb2hed`, it does not clamp negative stain
concentrations to zero, which makes the transform exactly invertible (zero
sigma and bias is a verified-exact identity through `uint8`); `rgb2hed`
distorts even at zero jitter on realistic H&E tiles, since its clamp zeroes
roughly a third of stain values on those images.

`StainAugmentor` (Macenko/Vahadane augmentation via `stainlib`) is removed.
It called methods that only existed on the transform's own `augmentor`
attribute, so it always raised `AttributeError`, and both its call sites in
`get_augmentor` were already commented out.

## Samplers

`wsi_data.samplers` now actually exports its weak-shuffling classes (see
"Confirmed bugs"). `get_weighted_random_sampler` fixes a real correctness
bug: it now maps labels to weights by their position among the dataset's
distinct class values, not by the raw label value, so it works for any label
values rather than only `0..K-1`.

## Dependencies

- **`wholeslidedata`** now resolves to `github.com/tsikup/pathology-whole-slide-data`
  rather than the PyPI package: the PyPI release (`0.0.16`, 2023-01) ships
  `wholeslidedata.accessories.qupath`, but this library imports
  `wholeslidedata.interoperability.qupath` — a later, unreleased layout. If
  you install with plain `pip`, add `--find-links` or install the fork
  directly: `pip install "wholeslidedata @ git+https://github.com/tsikup/pathology-whole-slide-data.git"`.
- **`he-preprocessing`** is likewise a first-party package not on PyPI;
  install `pip install "he-preprocessing[slide] @ git+https://github.com/tsikup/he_preprocessing.git"`.
  Version 1.0 of it is itself a from-scratch rewrite — see its own
  `MIGRATION.md` if you called it directly.
- **`stainlib`**, **`histocartography`** and **`dgl`** are no longer
  dependencies at all (see above).
- **`dotmap`**, **`tqdm`** and **`networkx`** are dropped: none were imported
  anywhere in the package.
- **`opencv-python`** is now **`opencv-python-headless`** for `wsi_data`'s
  own direct dependency: both `he-preprocessing` and `wholeslidedata`
  already require the headless build. This does not fully close the door,
  though — `he-preprocessing[slide]`'s own `tissueloc` dependency hard-requires
  plain `opencv-python`, and there is no packaging-level way to override a
  transitive dependency's distribution name. Both can still end up
  installed; this is usually harmless in an ordinary virtualenv, but can
  duplicate the bundled OpenMP runtime and abort the interpreter at import
  (`OMP: Error #15`) in an environment where both are simultaneously
  importable (e.g. a venv created with `--system-site-packages` over a conda
  base environment that already has OpenCV). If you hit that,
  `pip uninstall opencv-python` (keeping the headless build) resolves it.
- **`h5py`**, **`sourcelib`** and **`natsort`** were used but never declared
  as dependencies; they are now.
- **`seaborn`** and **`pandas`** are no longer required at all: label-count
  plotting moved to `wsi_data.viz.plot_label_distribution`, a thin
  `matplotlib`-only wrapper behind the `viz` extra — the counts themselves
  come from one `numpy.unique` call in `LabelDistribution`.

## Confirmed bugs fixed

Each of these was verified by execution, not just by inspection, against
albumentations 2.0.8, numpy 2.5, shapely 2.1, h5py 3.16 and scikit-image
0.26 — the versions this rewrite targets.

- `wsi_data.utils`/`wsi_data.augmentations` were unimportable at HEAD: the
  empty `samplers/__init__.py` broke both.
- `HEDAugmentor(p=0.25)` silently ran with an effective `p=0.0` under
  albumentations ≥2.0, because its `__init__` called
  `super().__init__(always_apply, p)` positionally against a signature that
  had become `__init__(self, p=0.5)`.
- `ElasticTransform(..., alpha_affine=50)` silently dropped the affine
  component of that transform (a `UserWarning`, ignored).
- `crop_data` returned an **empty** array whenever a spatial axis already had
  the requested size (`data[0:-0]` is `data[0:0]`), and left one extra
  row/column whenever the size difference was odd.
- `MaskedTiledAnnotationCallback(only_intersection=True)` was a no-op: it
  computed the clipped intersection polygon but then appended the
  unclipped tile square to the result. Iterating a `MultiPolygon`
  intersection also raised `TypeError` under shapely ≥2.0.
- The missing-label sentinel in `FeatureDatasetHDF5` was built as
  `np.array([-100], dtype=np.uint8)`; under NumPy 2 (this package's floor)
  that raises `OverflowError` outright, and under NumPy 1.x it silently
  wrapped to `156` — a plausible-looking class index.
- `ImageOnlyDatasetHDF5`/`ClassificationDatasetHDF5` returned a 2-tuple
  where an image tensor was expected, because the non-segmentation branch
  assigned the transform's `(image, mask)` return straight to `image`.
- `get_label_distribution()` on every segmentation `DatasetHDF5` raised
  `TypeError`: it built a per-resolution label dict, then unconditionally
  indexed the HDF5 file with `data_cols["labels"]` — itself a dict in
  segmentation mode.
- `get_weighted_random_sampler` raised `IndexError` (or silently
  mis-weighted classes) for any label set other than `0..K-1`.
- `MultiResWholeSlideDataSet.__init__` set `load_images`/`copy_path` and
  then called `super().__init__()` without forwarding them, so the parent
  silently reset both to its own defaults before `_open` ever ran.
- `MultiResWholeSlideDataSet.close_images()` always raised `AttributeError`
  (it iterated `self._images`, an attribute neither it nor its parent ever
  defined).
- `MyWholeSlideImage.get_data(with_mask=True)` always raised
  `AttributeError`: its override omitted the parent's
  "create the mask sampler on first use" guard.
- `CNNTissueSegmentor` loaded checkpoints with `torch.load(...,
  weights_only=False)` — arbitrary code execution from a checkpoint file.
- `CNNTissueSegmentor`'s blending window was silently all-zero for
  `tile_size < 4 * subdivisions`: `wind_inner[-intersection:] = 0` with
  `intersection == 0` slices as `[-0:]`, i.e. the whole array.
- `CNNTissueSegmentor` set `torch.backends.cudnn.benchmark = True` globally
  and permanently in its constructor, changing convolution behaviour for
  every other model in the process for the rest of its lifetime.

## Python floor and tooling

The minimum supported Python is now **3.11** (previously undeclared).
Tooling is Ruff, `mypy --strict`, pytest and pre-commit, matching
`he-preprocessing`.

CI (`.github/workflows/ci.yml`) runs on `ubuntu-latest` only. This package's
dependency chain includes two packages with native components (`rtree`,
`lxml`) plus two git-only dependencies; cross-platform (macOS/Windows)
behaviour has not been verified and may need adjustment before enabling it.
