import sys
from setuptools import setup

setup(
    name="wsi_data",
    version="0.1.0",
    packages=[""],
    url="https://github.com/tsikup/wsi_data",
    license="MIT",
    author="Nikos Tsiknakis",
    author_email="tsiknakisn@gmail.com",
    description="WSI data loading library",
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
        "topk",
        "albumentations",
        "wholeslidedata",
    ]
    + (["spams"] if sys.platform == "darwin" else ["spams-bin"]),
)
