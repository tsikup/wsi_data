# wsi_data

Dataset, sampler and augmentation library for whole-slide image (WSI) deep
learning: PyTorch datasets over HDF5 feature/tile stores and whole-slide
images, H&E-specific albumentations transforms, class-balanced and
weak-shuffling samplers, and a CNN tissue segmentor — as a typed, tested
Python library.

Version 1.0 is a full rewrite. If you used this package before, see
[`MIGRATION.md`](MIGRATION.md) — every import path changed, and a long list
of latent bugs (silently-disabled stain augmentation, an unimportable
sampler module, a broken `close_images`, an `IndexError` in the class-balanced
sampler, ...) is fixed. `MIGRATION.md` documents each one.

## Install

Two of this package's dependencies are first-party packages not published on
PyPI, so a plain `pip install wsi-data` needs a bit of help resolving them:

```bash
pip install "wsi-data @ git+https://github.com/tsikup/wsi_data.git" \
  "he-preprocessing[slide] @ git+https://github.com/tsikup/he_preprocessing.git" \
  "wholeslidedata @ git+https://github.com/tsikup/pathology-whole-slide-data.git"

pip install "wsi-data[viz]"  # + matplotlib, for wsi_data.viz.plot_label_distribution
```

With [uv](https://docs.astral.sh/uv/), the `[tool.uv.sources]` entries in
`pyproject.toml` resolve both automatically:

```bash
uv add wsi-data
```

Requires Python 3.11+.

## Quick start

```python
from wsi_data.datasets.h5 import ImageDatasetHDF5
from wsi_data.augmentations import get_augmentor

transform = get_augmentor(patch_size=512, split="train")
dataset = ImageDatasetHDF5(
    data_dir="tiles/",
    data_name="train.h5",
    data_cols={"images": "x", "labels": "y"},
    transform=transform,
    mode="segmentation",
)
image, mask = dataset[0]
```

```python
from wsi_data.labels import LabelDistribution
from wsi_data.samplers import get_weighted_random_sampler

sampler = get_weighted_random_sampler(dataset)  # draws every class equally often
print(dataset.get_label_distribution().describe())  # a quick text histogram
```

## Package layout

- `wsi_data.datasets` — `ImageDatasetHDF5`, `FeatureDatasetHDF5`,
  `TileDatasetHDF5` (HDF5-backed), `SlideTileDataset` (reads tiles directly
  from a whole-slide image), `FakeDataset` (synthetic, for smoke tests).
- `wsi_data.augmentations` — `get_augmentor` (the standard H&E pipeline),
  `HEDAugmentor` (stain-concentration jitter), `ReplaceBackgroundColor`.
- `wsi_data.samplers` — `get_weighted_random_sampler` (class-balanced),
  weak-shuffling samplers for fast sequential HDF5 reads.
- `wsi_data.labels` — `LabelDistribution`, the value type every dataset's
  `get_label_distribution()` returns.
- `wsi_data.transforms` — `crop_data` and the tensor-conversion pipeline the
  HDF5 datasets share.
- `wsi_data.normalization` — streaming per-channel mean/std over a tile corpus.
- `wsi_data.wholeslidedata` — `MultiResWholeSlideImage`/
  `SingleResWholeSlideImage`, `create_batch_sampler`, file discovery.
- `wsi_data.tissue_segmentation` — `CNNTissueSegmentor`, a pretrained U-Net
  for overlap-tile tissue detection.
- `wsi_data.viz` — annotation-overlay drawing and label-distribution plots
  (needs the `viz` extra).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

CI (`.github/workflows/ci.yml`) runs these on `ubuntu-latest` across
Python 3.11–3.13; see [`MIGRATION.md`](MIGRATION.md#python-floor-and-tooling)
for why other platforms aren't covered yet.
