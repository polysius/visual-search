"""Turn a PIL image into the array the exported model expects, using only PIL and numpy.

The serving path must not import transformers, so the settings that
`AutoImageProcessor` would apply are read once at export time, saved in a small
JSON file next to the ONNX model, and applied here: resize the short side, centre-crop
a square, scale to [0, 1] and normalise with the model's mean and standard deviation.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# The resize makes the long side `shortest_edge * long / short` pixels, so a very thin
# image would become enormous before it is cropped: a 20000 x 1 image becomes
# 4,480,000 x 224 pixels, several gigabytes. No real photo is anywhere near this thin.
MAX_ASPECT_RATIO = 100


class UnsupportedImageError(ValueError):
    """The image cannot be preprocessed, so a caller can report it as bad input."""


@dataclass(frozen=True)
class PreprocessConfig:
    model: str  # the base model the settings belong to
    shortest_edge: int
    crop_size: int
    resample: int  # a PIL resampling filter, 3 is bicubic
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # Which fine-tuned weights are inside the exported model. A gallery embedded with other
    # weights lives in a different vector space, so the server compares this when it loads.
    adapter_sha256: str | None = None

    @classmethod
    def from_processor(
        cls, model_name: str, processor, adapter_sha256: str | None = None
    ) -> "PreprocessConfig":
        """Read the settings from a Hugging Face image processor."""
        return cls(
            model=model_name,
            adapter_sha256=adapter_sha256,
            shortest_edge=int(_dimension(processor.size, "shortest_edge")),
            crop_size=int(_dimension(processor.crop_size, "height")),
            resample=int(processor.resample),
            mean=tuple(float(value) for value in processor.image_mean),
            std=tuple(float(value) for value in processor.image_std),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PreprocessConfig":
        raw = json.loads(Path(path).read_text())
        return cls(**{**raw, "mean": tuple(raw["mean"]), "std": tuple(raw["std"])})


def sidecar_path(onnx_path: Path) -> Path:
    """Where `visual_search.export` writes the preprocessing settings for an ONNX model."""
    return Path(onnx_path).with_suffix(".preprocess.json")


def _dimension(size, key: str) -> int:
    """Processor sizes are dicts in some transformers versions and objects in others."""
    return size[key] if isinstance(size, dict) else getattr(size, key)


def preprocess(image: Image.Image, config: PreprocessConfig) -> np.ndarray:
    """Return a (3, crop, crop) float32 array for one image.

    Raises `UnsupportedImageError` for an image more than `MAX_ASPECT_RATIO` times longer
    than it is wide.
    """
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if long > MAX_ASPECT_RATIO * short:
        raise UnsupportedImageError(
            f"image is {width} x {height}; the long side may be at most "
            f"{MAX_ASPECT_RATIO} times the short side"
        )
    image = image.convert("RGB")
    new_short = config.shortest_edge
    new_long = int(config.shortest_edge * long / short)  # rounds down, like the HF processor
    new_size = (new_short, new_long) if width <= height else (new_long, new_short)
    image = image.resize(new_size, resample=config.resample)

    width, height = image.size
    left = (width - config.crop_size) // 2
    top = (height - config.crop_size) // 2
    image = image.crop((left, top, left + config.crop_size, top + config.crop_size))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(config.mean, dtype=np.float32)) / np.asarray(
        config.std, dtype=np.float32
    )
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))
