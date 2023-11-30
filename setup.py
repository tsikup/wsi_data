from setuptools import setup

EXTRA_REQUIREMENTS = {
    "graphs": ["networkx", "dgl", "histocartography"],
}

setup(
    name="wsi_data",
    version="0.1.0",
    packages=[
        "wsi_data",
        "wsi_data.wholeslidedata",
        "wsi_data.datasets",
        "wsi_data.samplers",
    ],
    url="https://github.com/tsikup/wsi_data",
    license="MIT",
    author="Nikos Tsiknakis",
    author_email="tsiknakisn@gmail.com",
    description="WSI data loading library",
    extras_require=EXTRA_REQUIREMENTS,
    install_requires=[
        "dotmap",
        "natsort",
        "numpy",
        "pandas",
        "torch",
        "tqdm",
        "opencv-python",
        "scikit-image",
        "Pillow",
        "shapely",
        "torch",
        "torchvision",
        "albumentations",
        "wholeslidedata",
    ],
)
