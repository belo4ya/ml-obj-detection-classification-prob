from pathlib import Path
from typing import Any, List, Tuple, Union

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
    'savefig',
    'rectsize',
    'ColorPalette',
    'Mosaic',
    'visualize_bbox',
    'reset_axes',
    'plot_grid',
    'plot_probability_grid',
]

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def rectsize(width: float, nrows: int, ncols: int) -> Tuple[float, float]:
    return width, width / (ncols / nrows)


def savefig(path: Union[str, Path], fig: Figure, bbox_inches='tight', pad_inches=1 / 3, **kwargs):
    kwargs['bbox_inches'] = bbox_inches
    kwargs['pad_inches'] = pad_inches
    fig.savefig(str(path), **kwargs)


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

    def plot(self, images: List[np.ndarray], fig_width: float = 15, title: str = None, title_kw: dict = None) -> Figure:
        fig: Figure = plt.figure(figsize=rectsize(fig_width, self.h, self.w))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.00, wspace=0.00)

        if title:
            title_kw = title_kw or {}
            fig.suptitle(title, **title_kw)

        axes: List[Axes] = list(fig.subplot_mosaic(self.layout).values())
        for i in range(self.nimages):
            ax = axes[i]
            reset_axes(ax)
            ax.imshow(images[i])

        return fig

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
        axis='both',
        which='both',
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        labelright=False,
        labeltop=False,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax


def plot_grid(nrows: int, ncols: int, images: List[np.ndarray], title: str = None, titles: List[str] = None,
              fig_width: float = 15) -> Figure:
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_width / (ncols / nrows)))
    fig.subplots_adjust(hspace=0.12, wspace=0.1)
    if title:
        fig.suptitle(title, y=0.93)
    for i, image in enumerate(images):
        row = i // ncols
        col = i % ncols
        ax: Axes = axes[row, col]
        reset_axes(ax)
        if titles:
            ax.set_title(titles[i][:-4], size=9)

        ax.imshow(image, aspect='auto')

    reset_axes(axes[3, 3])
    return fig


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
):
    fig: Figure = plt.figure(figsize=figsize)

    subfigs = fig.subfigures(nrows, ncols)

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

        with sns.axes_style('white'):
            axes: List[Axes] = list(subfig.subplot_mosaic(layout).values())

        # img
        subfig.suptitle(str(cls_true == cls_pred), size=12, y=0.98)
        img_ax = axes[0]
        reset_axes(img_ax)
        img_ax.imshow(img)

        # probability
        prob_ax = axes[1]
        prob_data = pd.DataFrame({'prob': prob, 'mob': labels})

        sns.barplot(data=prob_data, x='prob', y='mob', orient='h', hue_order=labels, ax=prob_ax)
        reset_axes(prob_ax)
        prob_ax.set_xlabel('')
        prob_ax.yaxis.set_label_position('right')
        prob_ax.set_ylabel('probability', size=8, labelpad=15, rotation=270)

        # decision function
        decision_ax = axes[2]
        decision_data = pd.DataFrame({'decision': decision, 'mob': labels})

        sns.barplot(data=decision_data, x='decision', y='mob', orient='h', hue_order=labels, ax=decision_ax)
        reset_axes(decision_ax)
        decision_ax.set_xlabel('')
        decision_ax.yaxis.set_label_position('right')
        decision_ax.set_ylabel('decision f', size=8, labelpad=15, rotation=270)

    fig.legend(fig.axes[-1].patches, labels, bbox_to_anchor=(0.65, 0), ncol=len(labels) // 2)
    return fig
