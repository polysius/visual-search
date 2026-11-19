"""Turn images into embedding vectors with a pretrained vision encoder.

CLIP and DINOv2 are loaded through Hugging Face transformers. Their attention
projections are plain `nn.Linear` layers, which LoRA (M2) and ONNX export (M4)
both handle well, so baseline, fine-tuned and exported models share this wrapper.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
DINOV2_MODEL = "facebook/dinov2-small"


def adapter_fingerprint(adapter_path: Path | None) -> str | None:
    """A hash of a LoRA adapter's weights, or None when there is no adapter.

    Exported models and gallery indexes record it, so the server can refuse to search a
    gallery that was embedded with different weights.
    """
    if adapter_path is None:
        return None
    weights = Path(adapter_path) / "adapter_model.safetensors"
    return hashlib.sha256(weights.read_bytes()).hexdigest()


def is_clip_model(model_name: str) -> bool:
    return "clip" in model_name.lower()


def pooled_embedding(model, is_clip: bool, pixel_values):
    """Run the encoder and return one vector per image (a torch tensor).

    Shared by embedding and training so both use the same output head.
    CLIP: the projected image embedding. DINOv2: the CLS token of the last layer.
    """
    outputs = model(pixel_values=pixel_values)
    return outputs.image_embeds if is_clip else outputs.last_hidden_state[:, 0]


class PathEncoder:
    """Embeds image files in batches, for any class that provides `encode_images`."""

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def encode_paths(
        self,
        paths: list[Path],
        batch_size: int = 64,
        num_workers: int = 4,
        crop: Callable[[Path, Image.Image], Image.Image] | None = None,
    ) -> np.ndarray:
        """Embed image files. Decoding runs in worker processes while the model runs.

        `crop(path, image)` can return a cropped image, such as `BoxCropper`; it runs in the workers.
        """
        from torch.utils.data import DataLoader  # not needed by the serving path

        loader = DataLoader(
            _ImageFiles(paths, crop),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=list,  # keep PIL images; the embedder does the resizing
        )
        return np.concatenate([self.encode_images(batch) for batch in loader])


class ImageEmbedder(PathEncoder):
    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        device: str = "cpu",
        adapter_path: Path | None = None,
    ):
        """`adapter_path` is a LoRA adapter saved by `visual_search.train`; it is merged into the base."""
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        self.model_name = model_name
        self.device = device
        self.is_clip = is_clip_model(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model_class = CLIPVisionModelWithProjection if self.is_clip else AutoModel
        self.model = model_class.from_pretrained(model_name)
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
        self.model = self.model.to(device).eval()
        self._torch = torch

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images; returns a (batch, dim) float32 array."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            vectors = pooled_embedding(self.model, self.is_clip, inputs["pixel_values"])
        return vectors.cpu().numpy().astype(np.float32)


class _ImageFiles:
    def __init__(self, paths: list[Path], crop: Callable | None = None):
        self.paths = paths
        self.crop = crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        image = Image.open(self.paths[index]).convert("RGB")
        return image if self.crop is None else self.crop(self.paths[index], image)
