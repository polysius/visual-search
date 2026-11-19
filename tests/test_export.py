import pytest
import torch

pytest.importorskip("onnx")  # the serving extra is optional
pytest.importorskip("onnxruntime")

import onnx
from transformers import Dinov2Config, Dinov2Model

from visual_search import export


def tiny_dinov2() -> Dinov2Model:
    """A small randomly initialised DINOv2, so the test needs no download."""
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=518,  # as in the real model: smaller inputs need interpolated embeddings
        patch_size=14,
    )
    return Dinov2Model(config).eval()


def test_fixing_the_position_embeddings_does_not_change_the_output():
    torch.manual_seed(0)
    model = tiny_dinov2()
    pixels = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        before = model(pixel_values=pixels).last_hidden_state

    export.fix_position_embeddings(model, crop_size=224)
    with torch.no_grad():
        after = model(pixel_values=pixels).last_hidden_state

    assert torch.allclose(before, after, atol=1e-6)


def test_the_exported_graph_has_no_resize_for_the_position_embeddings(tmp_path):
    model = tiny_dinov2()
    export.fix_position_embeddings(model, crop_size=224)
    module = export._EmbeddingModule(model, is_clip=False).eval()
    path = tmp_path / "tiny.onnx"

    torch.onnx.export(
        module,
        (torch.randn(1, 3, 224, 224),),
        str(path),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=export.OPSET,
        dynamo=False,
    )

    operators = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert "Resize" not in operators


def test_a_model_without_position_embedding_interpolation_is_left_alone():
    class NoEmbeddings(torch.nn.Module):
        pass

    export.fix_position_embeddings(NoEmbeddings(), crop_size=224)  # must not raise
