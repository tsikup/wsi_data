from wholeslidedata.data.files import WholeSlideImageFile
from .wholeslideimage import MultiResWholeSlideImage, MyWholeSlideImage


class MultiResWholeSlideImageFile(WholeSlideImageFile):
    def open(self):
        return MultiResWholeSlideImage(
            self.path,
            self._image_backend,
        )


class MyWholeSlideImageFile(WholeSlideImageFile):
    def open(self):
        return MyWholeSlideImage(
            self.path,
            self._image_backend,
        )
