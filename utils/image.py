from pathlib import Path
from typing import Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import cv2
import numpy as np

__all__ = [
    'Image',
    'RGB',
    'BGR',
    'get_wh',
]

RGB = Literal['RGB']
BGR = Literal['BGR']
Mode = Union[BGR, RGB]


class Image:
    data: np.ndarray
    path: Path
    mode: Mode

    def __init__(self, data: np.ndarray, mode: Mode = RGB, path: Path = None):
        self.data = data
        self.mode = mode
        self.path = path

    @property
    def w(self) -> int:
        return self.data.shape[1]

    @property
    def h(self) -> int:
        return self.data.shape[0]

    @classmethod
    def open(cls, path: Path, mode: Mode = RGB) -> 'Image':
        data = cv2.imread(str(path))
        if mode == RGB:
            return cls(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), mode, path)
        return cls(data, mode, path)

    def save(self, path: Path):
        if self.mode == RGB:
            cv2.imwrite(str(path), cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), self.data)

    def convert(self, mode: Mode) -> 'Image':
        if mode == self.mode:
            data = self.data
        elif mode == BGR:
            data = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        else:
            data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)

        return Image(data, mode, self.path)


def get_wh(img: np.ndarray) -> Tuple[int, int]:
    h, w, _ = img.shape
    return w, h
