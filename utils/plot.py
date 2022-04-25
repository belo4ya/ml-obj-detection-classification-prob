from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from utils.image import get_wh
from utils.label import BBox

__all__ = [
    'ColorPalette',
    'Mosaic',
    'visualize_bbox',
    'reset_axes',
    'plot_grid',
    'plot_probability_grid'
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


def visualize_bbox(
        image: np.ndarray,
        bbox: BBox, name: str = None,
        color: Tuple[int, int, int] = BOX_COLOR,
        thickness=2
) -> np.ndarray:
    img = image.copy()
    img_w, img_h = get_wh(img)

    x_min, y_min, x_max, y_max = map(int, bbox.xyxy(w=img_w, h=img_h))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    if name:
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
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        labelright=False,
        labeltop=False
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax


def plot_grid():
    pass


def plot_probability_grid(
        nrows: int,
        ncols: int,
        idxs: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
        decision_f: np.ndarray,
        images: List[np.ndarray],
        labels: List[str],
        figsize: Tuple[float, float],
        title: str,
        facecolor=None,
        style: str = None
):
    fig: Figure = plt.figure(figsize=figsize)
    fig.subplots_adjust(0.1, 0.1, 0.85, 0.9, wspace=0.05, hspace=0)
    fig.suptitle(title, y=0.95)

    subfigs = fig.subfigures(nrows, ncols, facecolor=facecolor)

    layout = """
        AAB
        AAC
    """
    for i, j in enumerate(idxs):
        cls_true = y_true[j]
        cls_pred = y_pred[j]
        prob = proba[j]
        decision = decision_f[j]
        img = images[j]

        row = i // ncols
        col = i % ncols
        subfig: SubFigure = subfigs[row, col]

        if style:
            with sns.axes_style('dark'):
                axes: List[Axes] = list(subfig.subplot_mosaic(layout).values())
        else:
            axes: List[Axes] = list(subfig.subplot_mosaic(layout).values())

        # img
        if cls_true == cls_pred:
            title = 'True'
            title_color = sns.color_palette('deep')[2]
        else:
            title = 'False'
            title_color = sns.color_palette('deep')[3]

        subfig.suptitle(title, size=12, y=0.98, color=title_color)
        img_ax = axes[0]
        reset_axes(img_ax)
        img_ax.imshow(img, aspect='auto')

        # probability
        prob_ax = axes[1]
        twinx = reset_axes(prob_ax.twinx())
        twinx.set_ylabel('probability', labelpad=15, rotation=270)

        prob_data = pd.DataFrame({'prob': prob, 'mob': labels})
        sns.barplot(data=prob_data, x='prob', y='mob', orient='h', hue_order=labels, ax=prob_ax)
        prob_ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        prob_ax.set_xlabel('')
        prob_ax.set_ylabel('')

        # decision function
        decision_ax = axes[2]
        twinx = reset_axes(decision_ax.twinx())
        twinx.set_ylabel('decision f', labelpad=15, rotation=270)

        decision_data = pd.DataFrame({'decision': decision, 'mob': labels})
        sns.barplot(data=decision_data, x='decision', y='mob', orient='h', hue_order=labels, ax=decision_ax)
        decision_ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        decision_ax.set_xlabel('')
        decision_ax.set_ylabel('')

    return fig
