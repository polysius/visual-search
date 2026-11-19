import numpy as np
import pytest
from PIL import Image

from visual_search.crop import BoxCropper, crop_to_object, select_box, square_crop_box

# Rows are x1, y1, x2, y2, confidence.
BOXES = np.array(
    [
        [10, 20, 50, 60, 0.9],
        [30, 10, 90, 40, 0.4],
        [0, 0, 5, 5, 0.1],
    ],
    dtype=np.float32,
)


def test_select_box_top_takes_the_most_confident_box():
    assert select_box(BOXES, "top", min_confidence=0.25) == (10.0, 20.0, 50.0, 60.0)


def test_select_box_union_surrounds_every_box_above_the_threshold():
    assert select_box(BOXES, "union", min_confidence=0.25) == (10.0, 10.0, 90.0, 60.0)


def test_select_box_ignores_boxes_below_the_threshold():
    assert select_box(BOXES, "union", min_confidence=0.95) is None
    assert select_box(np.zeros((0, 5), dtype=np.float32), "top", min_confidence=0.25) is None


def test_select_box_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        select_box(BOXES, "largest", min_confidence=0.25)


def test_square_crop_box_is_square_and_centred_on_the_object():
    # A 40 x 20 box centred at (100, 50) in a 200 x 100 image, margin 0.25: side = 40 * 1.5 = 60.
    crop = square_crop_box((80, 40, 120, 60), (200, 100), margin=0.25, min_area_fraction=0.0)
    assert crop == (70, 20, 130, 80)


def test_square_crop_box_shifts_inside_the_image():
    # The same object near the left edge: the square slides right instead of leaving the image.
    crop = square_crop_box((0, 40, 40, 60), (200, 100), margin=0.25, min_area_fraction=0.0)
    assert crop == (0, 20, 60, 80)


def test_square_crop_box_is_cut_to_the_image_when_too_large():
    crop = square_crop_box((10, 10, 190, 90), (200, 100), margin=0.5, min_area_fraction=0.0)
    assert crop == (0, 0, 200, 100)


def test_square_crop_box_rejects_boxes_that_are_too_small():
    # A 10 x 10 box in a 100 x 100 image is 1% of the area.
    assert square_crop_box((0, 0, 10, 10), (100, 100), margin=0.1, min_area_fraction=0.1) is None


@pytest.mark.parametrize(
    "box",
    [
        (100, 80, 50, 20),  # inverted: both sides negative, so the area looks positive
        (50, 50, 50, 50),  # no area
        (10, 10, float("nan"), 60),
        (18.15, 114.0, 18.45, 114.3),  # under a pixel across, so it rounds to no pixels
    ],
)
def test_square_crop_box_rejects_boxes_that_cannot_be_cropped(box):
    assert square_crop_box(box, (200, 120), margin=0.0, min_area_fraction=0.0) is None


def test_box_cropper_names_the_missing_path(tmp_path):
    cropper = BoxCropper({}, tmp_path)
    with pytest.raises(KeyError, match="mug_final/1.JPG"):
        cropper(tmp_path / "mug_final" / "1.JPG", Image.new("RGB", (10, 10)))


def test_crop_to_object_returns_the_whole_image_without_boxes():
    image = Image.new("RGB", (100, 80))
    result = crop_to_object(image, np.zeros((0, 5), dtype=np.float32))
    assert result is image


def test_crop_to_object_crops_to_the_object():
    image = Image.new("RGB", (200, 100))
    boxes = np.array([[80, 40, 120, 60, 0.9]], dtype=np.float32)
    result = crop_to_object(image, boxes, margin=0.25, min_area_fraction=0.0)
    assert result.size == (60, 60)


def test_box_cropper_looks_up_boxes_by_path_relative_to_the_root(tmp_path):
    image = Image.new("RGB", (200, 100))
    boxes = {"mug_final/1.JPG": np.array([[80, 40, 120, 60, 0.9]], dtype=np.float32)}
    cropper = BoxCropper(boxes, tmp_path, margin=0.25, min_area_fraction=0.0)

    assert cropper(tmp_path / "mug_final" / "1.JPG", image).size == (60, 60)
