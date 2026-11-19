"""Crop an image to the object a detector found, before it is embedded.

A product photo from a user can show the product small in a cluttered scene.
Detector boxes are stored once by `visual_search.detect`; this module turns the
boxes of one image into a crop. The functions here work on one image at a time,
so the same code can serve a single uploaded photo.

The detector's class labels are ignored. A COCO detector has no class for most
products and mislabels them, but its box is still a useful "object here".
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

STRATEGIES = ("top", "union")


def select_box(boxes: np.ndarray, strategy: str, min_confidence: float) -> tuple | None:
    """Pick one box from an (n, 5) array of x1, y1, x2, y2, confidence rows.

    "top" takes the most confident box. "union" takes the box around all boxes
    that clear `min_confidence`. Returns None when no box clears it.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    kept = boxes[boxes[:, 4] >= min_confidence] if len(boxes) else boxes
    if len(kept) == 0:
        return None
    if strategy == "top":
        x1, y1, x2, y2 = kept[kept[:, 4].argmax(), :4]
    else:
        x1, y1 = kept[:, 0].min(), kept[:, 1].min()
        x2, y2 = kept[:, 2].max(), kept[:, 3].max()
    return float(x1), float(y1), float(x2), float(y2)


def square_crop_box(
    box: tuple, image_size: tuple[int, int], margin: float, min_area_fraction: float
) -> tuple[int, int, int, int] | None:
    """Grow a box by `margin` on each side, make it square, and keep it inside the image.

    The embedders resize the short side and then centre-crop a square, so a wide
    box would lose its ends. A square box keeps the whole object. Returns None
    when the box is smaller than `min_area_fraction` of the image: a tiny box is
    more likely a part of the product than the product. Also returns None for a box
    with no area, an inverted box or NaN coordinates.
    """
    width, height = image_size
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):  # false for NaN as well
        return None
    if (x2 - x1) * (y2 - y1) < min_area_fraction * width * height:
        return None
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    left, right = _fit_interval((x1 + x2) / 2, side, width)
    top, bottom = _fit_interval((y1 + y2) / 2, side, height)
    crop_box = round(left), round(top), round(right), round(bottom)
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None  # a very thin box can round down to no pixels
    return crop_box


def _fit_interval(center: float, length: float, limit: int) -> tuple[float, float]:
    """An interval of `length` around `center`, shifted to fit in [0, limit] or cut to it."""
    if length >= limit:
        return 0, limit
    start = min(max(center - length / 2, 0), limit - length)
    return start, start + length


def crop_to_object(
    image: Image.Image,
    boxes: np.ndarray,
    strategy: str = "top",
    min_confidence: float = 0.25,
    margin: float = 0.1,
    min_area_fraction: float = 0.1,
) -> Image.Image:
    """The image cropped to its detected object, or the whole image when there is none."""
    box = select_box(boxes, strategy, min_confidence)
    if box is None:
        return image
    crop_box = square_crop_box(box, image.size, margin, min_area_fraction)
    return image if crop_box is None else image.crop(crop_box)


class BoxCropper:
    """Crops dataset images using boxes cached by `visual_search.detect`.

    It is a plain object with a `__call__`, so it can be sent to DataLoader workers.
    """

    def __init__(self, boxes_by_path: dict, root: Path, **crop_options):
        self.boxes_by_path = boxes_by_path
        self.root = Path(root)
        self.crop_options = crop_options

    @classmethod
    def from_file(cls, cache_path: Path, root: Path, **crop_options) -> "BoxCropper":
        raw = json.loads(Path(cache_path).read_text())
        boxes = {
            key: np.asarray(rows, dtype=np.float32).reshape(-1, 5) for key, rows in raw.items()
        }
        return cls(boxes, root, **crop_options)

    def __call__(self, path: Path, image: Image.Image) -> Image.Image:
        key = Path(path).relative_to(self.root).as_posix()
        if key not in self.boxes_by_path:
            raise KeyError(f"no boxes for {key}; was the box file made for this split?")
        return crop_to_object(image, self.boxes_by_path[key], **self.crop_options)
