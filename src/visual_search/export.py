"""Export a (fine-tuned) encoder to ONNX for serving.

Example:
    python -m visual_search.export --adapter artifacts/m2_clip/best --out artifacts/onnx/clip_lora.onnx

Writes two files: the model, and `<name>.preprocess.json` with the image
preprocessing settings, so the serving code needs neither transformers nor the
hard-coded constants of one model. The LoRA adapter is merged into the weights
first, so the exported graph is a plain encoder.

Needs the optional extra: pip install -e .[serve]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from visual_search.embedder import (
    CLIP_MODEL,
    ImageEmbedder,
    adapter_fingerprint,
    pooled_embedding,
)
from visual_search.preprocess import PreprocessConfig, sidecar_path

OPSET = 17
PARITY_TOLERANCE = 1e-3  # largest allowed difference between torch and ONNX outputs


class _EmbeddingModule(torch.nn.Module):
    """The encoder with the same output head that evaluation and training use."""

    def __init__(self, model, is_clip: bool):
        super().__init__()
        self.model = model
        self.is_clip = is_clip

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pooled_embedding(self.model, self.is_clip, pixel_values)


def fix_position_embeddings(model, crop_size: int) -> None:
    """Compute DINOv2's position embeddings for the fixed input size once, up front.

    DINOv2 was trained on 518 pixel images, so its transformers implementation resizes the
    stored position embeddings to the input size on every forward pass. The exported graph
    always gets the same crop_size input, so that resize gives the same result every time,
    yet in the ONNX graph it was half of the run time (62 of 123 ms on the CPU). This replaces
    it with the precomputed result. A model whose top level has no `embeddings` object with such
    a method is left alone. CLIP is one: its embeddings sit one level down, in `vision_model`,
    and its position embeddings are not resized unless the caller asks for it.
    """
    embeddings = getattr(model, "embeddings", None)
    if not hasattr(embeddings, "interpolate_pos_encoding"):
        return
    patches = (crop_size // model.config.patch_size) ** 2
    placeholder = torch.zeros(1, patches + 1, model.config.hidden_size)  # +1 for the CLS token
    with torch.no_grad():
        fixed = embeddings.interpolate_pos_encoding(placeholder, crop_size, crop_size)
    embeddings.interpolate_pos_encoding = lambda embeddings, height, width: fixed


def export_onnx(model_name: str, adapter_path: Path | None, out_path: Path) -> float:
    """Export the model, check it against torch, and return the largest output difference."""
    embedder = ImageEmbedder(model_name, device="cpu", adapter_path=adapter_path)
    config = PreprocessConfig.from_processor(
        model_name, embedder.processor, adapter_fingerprint(adapter_path)
    )
    fix_position_embeddings(embedder.model, config.crop_size)
    module = _EmbeddingModule(embedder.model, embedder.is_clip).eval()
    example = torch.randn(2, 3, config.crop_size, config.crop_size)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The legacy TorchScript exporter is the one checked against torch here. It prints a
    # deprecation warning and bakes the 224 x 224 input size into the graph, so the
    # preprocessing must always produce exactly crop_size x crop_size. Only the batch
    # dimension is dynamic.
    torch.onnx.export(
        module,
        (example,),
        str(out_path),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    config.save(sidecar_path(out_path))
    return _largest_difference(module, out_path, config.crop_size)


def _largest_difference(module: torch.nn.Module, onnx_path: Path, crop_size: int) -> float:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    differences = []
    for batch_size in (1, 3):  # a batch size other than the one used for the export
        pixels = torch.randn(batch_size, 3, crop_size, crop_size)
        with torch.no_grad():
            expected = module(pixels).numpy()
        actual = session.run(None, {"pixel_values": pixels.numpy()})[0]
        differences.append(float(np.abs(expected - actual).max()))
    return max(differences)


def main() -> None:
    args = parse_args()
    difference = export_onnx(args.model, args.adapter, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    print(f"largest difference between torch and ONNX outputs: {difference:.2e}")
    if difference > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output differs from torch by more than {PARITY_TOLERANCE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CLIP_MODEL)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter to merge in")
    parser.add_argument("--out", type=Path, required=True, help="where to write the .onnx file")
    return parser.parse_args()


if __name__ == "__main__":
    main()
