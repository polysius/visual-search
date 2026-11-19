"""Embed images with an exported ONNX model, using only onnxruntime, PIL and numpy."""

from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.embedder import PathEncoder
from visual_search.preprocess import PreprocessConfig, preprocess, sidecar_path


class OnnxEmbedder(PathEncoder):
    def __init__(self, onnx_path: Path, intra_op_threads: int | None = None):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        self.session = onnxruntime.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.config = PreprocessConfig.load(sidecar_path(onnx_path))
        self.model_name = self.config.model
        self.adapter_sha256 = self.config.adapter_sha256

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images; returns a (batch, dim) float32 array."""
        pixels = np.stack([preprocess(image, self.config) for image in images])
        return self.session.run(None, {"pixel_values": pixels})[0].astype(np.float32)
