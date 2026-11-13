"""Turn images into embedding vectors with a pretrained vision encoder.

CLIP and DINOv2 are loaded through Hugging Face transformers. Their attention
projections are plain `nn.Linear` layers, which LoRA (M2) and ONNX export (M4)
both handle well, so baseline, fine-tuned and exported models share this wrapper.
"""

from pathlib import Path

import numpy as np
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"
DINOV2_MODEL = "facebook/dinov2-small"


class ImageEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        self.model_name = model_name
        self.device = device
        self.is_clip = "clip" in model_name.lower()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model_class = CLIPVisionModelWithProjection if self.is_clip else AutoModel
        self.model = model_class.from_pretrained(model_name).to(device).eval()
        self._torch = torch

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images; returns a (batch, dim) float32 array."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(pixel_values=inputs["pixel_values"])
        # CLIP: projected image embedding. DINOv2: the CLS token of the last layer.
        vectors = outputs.image_embeds if self.is_clip else outputs.last_hidden_state[:, 0]
        return vectors.cpu().numpy().astype(np.float32)

    def encode_paths(
        self, paths: list[Path], batch_size: int = 64, num_workers: int = 4
    ) -> np.ndarray:
        """Embed image files. Decoding runs in worker processes while the model runs."""
        loader = self._torch.utils.data.DataLoader(
            _ImageFiles(paths),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=list,  # keep PIL images; the processor does the resizing
        )
        batches = [self.encode_images(batch) for batch in loader]
        return np.concatenate(batches)


class _ImageFiles:
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert("RGB")
