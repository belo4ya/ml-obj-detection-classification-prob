from __future__ import annotations

from itertools import chain
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split

Image = np.ndarray
BBox = list[float]
Label = tuple[int, BBox]

BBOX_PARAMS = A.BboxParams(format='yolo', label_fields=['labels'])
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def read_label(path: Path) -> Label:
    with open(path) as f:
        label = f.read().split()
    return int(label[0]), [float(i) for i in label[1:]]


def write_label(path: Path, label: Label):
    with open(path, 'w') as f:
        f.write(' '.join(map(str, [label[0], *label[1]])))


def read_image(path: Path) -> Image:
    return cv2.imread(str(path))


def write_image(path: Path, img: Image):
    cv2.imwrite(str(path), img)


def bgr2rgb(img: Image) -> Image:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img: Image) -> Image:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def a_resize(height, width) -> A.Compose:
    return A.Compose([A.Resize(height, width, always_apply=True)], bbox_params=BBOX_PARAMS)


def a_horizontal_flip() -> A.Compose:
    return A.Compose([A.HorizontalFlip(always_apply=True)], bbox_params=BBOX_PARAMS)


def a_rotate(angle) -> A.Compose:
    return A.Compose([A.Affine(rotate=angle, always_apply=True)], bbox_params=BBOX_PARAMS)


def a_rain() -> A.Compose:
    return A.Compose([A.RandomRain(
        drop_width=2,
        drop_color=(100, 119, 188),
        blur_value=2,
        brightness_coefficient=0.9,
        rain_type='drizzle',
        always_apply=True
    )], bbox_params=BBOX_PARAMS)


def transform(transform_: A.Compose, img: Image, label: Label) -> (Image, Label):
    transformed = transform_(
        image=bgr2rgb(img),
        labels=[label[0]],
        bboxes=[label[1]],
    )
    new_img = transformed['image']
    new_label = [transformed['labels'][0], transformed['bboxes'][0]]

    return rgb2bgr(new_img), new_label


def train_valid_test_split(*arrays, train_size, valid_size, **kwargs):
    """Разделят выборку на тренировочную, валидационную и тестовую"""
    split = train_test_split(*arrays, train_size=train_size, **kwargs)
    train = [split[i] for i in range(0, len(split), 2)]
    rest = [split[i] for i in range(1, len(split), 2)]

    valid_size = valid_size / (1 - train_size)
    split = train_test_split(*rest, train_size=valid_size)
    return list(chain.from_iterable(zip(
        train,
        [split[i] for i in range(0, len(split), 2)],
        [split[i] for i in range(1, len(split), 2)]
    )))


def visualize_bbox(image: Image, bbox: BBox, name: str, color: tuple[int, int, int] = BOX_COLOR, thickness=2):
    img = image.copy()
    img_h, img_w = img.shape[0], img.shape[1]
    bbox = A.convert_bbox_from_albumentations(
        A.convert_bbox_to_albumentations(bbox, source_format='yolo', rows=img_h, cols=img_w),
        target_format='coco', rows=img_h, cols=img_w
    )

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
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
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    ax.spines[:].set_visible(False)
    return ax
