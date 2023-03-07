from pathlib import Path
from typing import Union
from wholeslidedata.extensions import (
    FolderCoupledExtension,
    WholeSlideImageExtension,
)
from wholeslidedata.mode import Mode
from wholeslidedata.source.copy import copy as copy_source
from wholeslidedata.source.files import WholeSlideFile, ImageFile

from .wholeslideimage import MultiResWholeSlideImage, MyWholeSlideImage


class BaseWholeSlideImageFile(WholeSlideFile, ImageFile):
    EXTENSIONS = WholeSlideImageExtension

    def __init__(
        self, mode: Union[str, Mode], path: Union[str, Path], image_backend: str = None
    ):
        super().__init__(mode, path, image_backend)

    def copy(self, destination_folder) -> None:
        destination_folder = Path(destination_folder) / "images"
        extension_name = self.path.suffix
        if WholeSlideImageExtension.is_extension(
            extension_name, FolderCoupledExtension
        ):
            folder = self.path.with_suffix("")
            copy_source(folder, destination_folder)
        super().copy(destination_folder=destination_folder)

    def open(self):
        raise NotImplementedError


@WholeSlideFile.register(
    ("mrwsi", "multires_wsi", "multiresolutionwsi", "multiresolutionwholeslideimage")
)
class MultiResWholeSlideImageFile(BaseWholeSlideImageFile):
    def open(self):
        return MultiResWholeSlideImage(
            self.path,
            self._image_backend,
        )


@WholeSlideFile.register(("mywsi", "mywholeslideimage"))
class MyWholeSlideImageFile(BaseWholeSlideImageFile):
    def open(self):
        return MyWholeSlideImage(
            self.path,
            self._image_backend,
        )
