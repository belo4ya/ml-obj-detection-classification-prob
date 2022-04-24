from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.image import get_wh
from utils.label import BBox

__all__ = [
    'ColorPalette',
    'Mosaic',
    'visualize_bbox',
    'reset_axes',
]

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


class ColorPalette:

    def __init__(self, colors: List[Tuple[int, int, int]], ids: List[Any]):
        self._palette = dict(zip(ids, colors))

    def get(self, id_: Any) -> Tuple[int, int, int]:
        return self._palette[id_]


class Mosaic:

    def __init__(self, layout: str):
        self.layout = layout
        self.nimages = len(set(layout)) - 2

        _layout = layout.strip().replace(' ', '').split('\n')
        self.w = len(_layout[0])
        self.h = len(_layout)

        self.fig = None

    def plot(self, images: List[np.ndarray], fig_width: float = 15, title_kwargs: dict = None):
        self.fig: Figure = plt.figure(figsize=(fig_width, fig_width / (self.w / self.h)))
        self.fig.subplots_adjust(hspace=0.00, wspace=0.00)

        if title_kwargs:
            self.fig.suptitle(title_kwargs['title'], y=title_kwargs['y'])

        axes: List[Axes] = list(self.fig.subplot_mosaic(self.layout).values())
        for i in range(self.nimages):
            ax = axes[i]
            reset_axes(ax)
            ax.imshow(images[i], aspect='auto')

    def __repr__(self):
        return f'Mosaic: w={self.w}, h={self.h}, nimages={self.nimages}{self.layout}'

    __str__ = __repr__


def visualize_bbox(image: np.ndarray, bbox: BBox, name: str, color: Tuple[int, int, int] = BOX_COLOR, thickness=2):
    img = image.copy()
    img_w, img_h = get_wh(img)

    x_min, y_min, x_max, y_max = map(int, bbox.xyxy(w=img_w, h=img_h))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def reset_axes(ax: Axes):
    ax.grid(visible=False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    ax.spines[:].set_visible(False)
    ax.set_xmargin(20)
    ax.set_ymargin(20)
    return ax
