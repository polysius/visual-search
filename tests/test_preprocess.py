import numpy as np
import pytest
from PIL import Image

from visual_search.preprocess import (
    PreprocessConfig,
    UnsupportedImageError,
    preprocess,
    sidecar_path,
)

CONFIG = PreprocessConfig(
    model="test-model",
    shortest_edge=32,
    crop_size=32,
    resample=3,
    mean=(0.5, 0.5, 0.5),
    std=(0.25, 0.25, 0.25),
)


@pytest.mark.parametrize("size", [(100, 50), (50, 100), (64, 64), (33, 700), (31, 31)])
def test_output_is_a_square_channel_first_float_array(size):
    array = preprocess(Image.new("RGB", size), CONFIG)

    assert array.shape == (3, 32, 32)
    assert array.dtype == np.float32


def test_pixels_are_scaled_and_normalised():
    # A white pixel is 1.0, so (1.0 - 0.5) / 0.25 = 2.0. A black pixel gives -2.0.
    white = preprocess(Image.new("RGB", (40, 40), "white"), CONFIG)
    black = preprocess(Image.new("RGB", (40, 40), "black"), CONFIG)

    assert np.allclose(white, 2.0)
    assert np.allclose(black, -2.0)


def test_the_short_side_is_resized_and_the_long_side_is_centre_cropped():
    # Left third red, middle third green, right third blue, in a 3:1 wide image.
    image = Image.new("RGB", (192, 64))
    for index, color in enumerate(["red", "lime", "blue"]):
        image.paste(Image.new("RGB", (64, 64), color), (index * 64, 0))

    array = preprocess(image, CONFIG)

    # The 32 x 32 centre crop of the resized 96 x 32 image is the green third. Its edges sit on
    # the colour boundaries, where bicubic resizing blends neighbours, so check the interior.
    interior = array[:, :, 4:-4]
    assert np.allclose(interior[1], 2.0)  # green channel is 1.0 for lime
    assert np.allclose(interior[0], -2.0)  # red channel is 0 for lime
    assert np.allclose(interior[2], -2.0)  # blue channel is 0 for lime


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "LA", "1"])
def test_other_image_modes_are_accepted(mode):
    array = preprocess(Image.new(mode, (40, 40)), CONFIG)

    assert array.shape == (3, 32, 32)


@pytest.mark.parametrize("size", [(101, 1), (1, 101), (10_000, 1), (3, 5000), (10_000_000, 1)])
def test_images_that_are_far_too_thin_are_rejected_before_resizing(size):
    # Without the limit, a 20000 x 1 image would be resized to 4.5 million x 224 pixels.
    with pytest.raises(UnsupportedImageError, match="at most"):
        preprocess(Image.new("1", size), CONFIG)


def test_images_at_the_aspect_ratio_limit_are_accepted():
    assert preprocess(Image.new("RGB", (100, 1)), CONFIG).shape == (3, 32, 32)
    assert preprocess(Image.new("RGB", (1, 100)), CONFIG).shape == (3, 32, 32)


def test_the_error_for_a_thin_image_can_be_caught_as_a_value_error():
    assert issubclass(UnsupportedImageError, ValueError)


def test_config_survives_a_round_trip_through_json(tmp_path):
    path = tmp_path / "model.preprocess.json"
    CONFIG.save(path)

    assert PreprocessConfig.load(path) == CONFIG


def test_a_config_saved_before_the_adapter_hash_existed_still_loads(tmp_path):
    path = tmp_path / "old.preprocess.json"
    path.write_text(
        '{"model": "m", "shortest_edge": 32, "crop_size": 32, "resample": 3, '
        '"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}'
    )

    assert PreprocessConfig.load(path).adapter_sha256 is None


def test_sidecar_sits_next_to_the_model():
    assert sidecar_path("models/clip.onnx").as_posix() == "models/clip.preprocess.json"
