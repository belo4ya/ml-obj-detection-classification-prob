from typing import Tuple

import albumentations as A
import numpy as np
from yolov5.utils import plots

from utils.image import get_wh
from utils.label import BBox

__all__ = [
    'a_resize',
    'a_horizontal_flip',
    'a_rotate',
    'a_shear',
    'transform',
    'crop',
    'crop_bbox_rect',
    'crop_bbox_constant',
]

# ---- Albumentations

BBOX_PARAMS = A.BboxParams(format='yolo', label_fields=['labels'])


def a_resize(height, width) -> A.Compose:
    return A.Compose([A.Resize(height, width, always_apply=True)], bbox_params=BBOX_PARAMS)


def a_horizontal_flip() -> A.Compose:
    return A.Compose([A.HorizontalFlip(always_apply=True)], bbox_params=BBOX_PARAMS)


def a_rotate(angle) -> A.Compose:
    return A.Compose([A.Affine(rotate=angle, always_apply=True)], bbox_params=BBOX_PARAMS)


def a_shear(angle_x, angle_y) -> A.Compose:
    return A.Compose([A.Affine(shear={'x': angle_x, 'y': angle_y}, always_apply=True)], bbox_params=BBOX_PARAMS)


def transform(transform_: A.Compose, img: np.ndarray, cls: int, bbox: BBox) -> Tuple[np.ndarray, int, BBox]:
    transformed = transform_(
        image=img,
        labels=[cls],
        bboxes=[bbox.xywhn],
    )
    new_img = transformed['image']
    h, w, _ = new_img.shape
    return new_img, transformed['labels'][0], BBox(*transformed['bboxes'][0], img_h=h, img_w=w)


# ---- bbox

def crop(img: np.ndarray, bbox: BBox) -> np.ndarray:
    xyxy = [int(i) for i in bbox.xyxy(*get_wh(img))]
    return img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]


def crop_bbox_rect(img: np.ndarray, bbox: BBox, pad=0) -> np.ndarray:
    w, h = get_wh(img)
    xyxy = bbox.xyxy(w=w, h=h)
    return plots.save_one_box(xyxy, img, pad=pad, gain=1, save=False, BGR=True, square=True)


def crop_bbox_constant(img: np.ndarray, bbox: BBox, pad: float = 0, constant: int = 0) -> np.ndarray:
    w, h = get_wh(img)
    xyxy = bbox.xyxy(w=w, h=h)

    img = plots.save_one_box(xyxy, img, pad=pad, gain=1, save=False, BGR=True, square=False)
    w, h = get_wh(img)

    a = max(h, w)
    lpad = (a - w) // 2
    tpad = (a - h) // 2
    pads = ((tpad, (a - h) - tpad), (lpad, (a - w) - lpad), (0, 0))

    return np.pad(img, pad_width=pads, constant_values=constant)
