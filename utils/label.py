from pathlib import Path
from typing import Tuple

import albumentations as A

__all__ = [
    'Label',
    'BBox',
    'xywh2xywhn',
    'xyxy2xywhn',
]


class Label:

    def __init__(self, cls: int, bbox: 'BBox', path: Path = None):
        self.cls = cls
        self.bbox = bbox
        self.path = path

    @classmethod
    def open(cls, path: Path) -> 'Label':
        with open(path) as f:
            label = f.readline().strip().split()
        return Label(int(label[0]), BBox(*map(float, label[1:5])), path=path)

    def save(self, path: Path):
        x, y, w, h = self.bbox.xywhn
        with open(path, 'w') as f:
            f.write(f'{self.cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}')


class BBox:
    """Bounding box в формате YOLO: (xn, yn, wn, hn), например (0.3, 0.1, 0.05, 0.07)"""

    def __init__(self, x: float, y: float, w: float, h: float, img_h: int = None, img_w: int = None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xywhn = (x, y, w, h)

        self.rows = img_h
        self.cols = img_w

    def xyxy(self, w: int = None, h: int = None) -> Tuple[float, float, float, float]:
        rows = h or self.rows
        cols = w or self.cols
        return A.convert_bbox_from_albumentations(
            A.convert_bbox_to_albumentations(self.xywhn, source_format='yolo', rows=rows, cols=cols),
            target_format='pascal_voc', rows=rows, cols=cols
        )

    def xywh(self, w: int = None, h: int = None) -> Tuple[float, float, float, float]:
        rows = h or self.rows
        cols = w or self.cols
        return A.convert_bbox_from_albumentations(
            A.convert_bbox_to_albumentations(self.xywhn, source_format='yolo', rows=rows, cols=cols),
            target_format='coco', rows=rows, cols=cols
        )


def xywh2xywhn(x: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    return A.convert_bbox_from_albumentations(
        A.convert_bbox_to_albumentations(x, source_format='coco', rows=h, cols=w),
        target_format='yolo',
        rows=h,
        cols=w,
    )


def xyxy2xywhn(x: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    return A.convert_bbox_from_albumentations(
        A.convert_bbox_to_albumentations(x, source_format='pascal_voc', rows=h, cols=w),
        target_format='yolo',
        rows=h,
        cols=w,
    )
