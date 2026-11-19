import numpy as np
import pytest
from PIL import Image

from visual_search.index import GalleryIndex
from visual_search.search import GalleryItems, Searcher, save_gallery

# Three gallery vectors pointing along different axes, so the nearest item is obvious.
GALLERY_VECTORS = np.eye(3, dtype=np.float32)
ITEMS = GalleryItems(
    images=["mug_final/1.JPG", "lamp_final/2.JPG", "sofa_final/3.JPG"], class_ids=[11, 22, 33]
)


class FakeEmbedder:
    """Embeds an image as the axis named by its red channel, 0, 1 or 2."""

    model_name = "test-model"

    def encode_images(self, images):
        axes = [image.getpixel((0, 0))[0] for image in images]
        return np.eye(3, dtype=np.float32)[axes]


def query(axis: int) -> Image.Image:
    return Image.new("RGB", (4, 4), (axis, 0, 0))


def test_search_returns_matches_best_first():
    searcher = Searcher(FakeEmbedder(), GalleryIndex(GALLERY_VECTORS), ITEMS)

    matches = searcher.search(query(1), k=2)

    assert [match.rank for match in matches] == [1, 2]
    assert matches[0].image == "lamp_final/2.JPG"
    assert matches[0].class_id == 22
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(0.0)


def test_search_with_k_above_the_gallery_size_returns_every_item():
    searcher = Searcher(FakeEmbedder(), GalleryIndex(GALLERY_VECTORS), ITEMS)

    assert len(searcher.search(query(0), k=50)) == 3


def test_searcher_rejects_an_index_that_does_not_match_its_items():
    with pytest.raises(ValueError, match="3 vectors but 2 items"):
        Searcher(FakeEmbedder(), GalleryIndex(GALLERY_VECTORS), GalleryItems(["a", "b"], [1, 2]))


def test_a_saved_gallery_loads_and_answers_the_same_queries(tmp_path):
    meta = {"model": "test-model"}
    save_gallery(tmp_path, GalleryIndex(GALLERY_VECTORS), ITEMS, meta)

    searcher = Searcher.load(FakeEmbedder(), tmp_path)

    assert searcher.gallery_size == 3
    assert searcher.search(query(2), k=1)[0].image == "sofa_final/3.JPG"


class FineTunedEmbedder(FakeEmbedder):
    def __init__(self, adapter_sha256):
        self.adapter_sha256 = adapter_sha256


def test_a_gallery_built_with_different_fine_tuned_weights_is_rejected(tmp_path):
    meta = {"model": "test-model", "adapter_sha256": "aaa"}
    save_gallery(tmp_path, GalleryIndex(GALLERY_VECTORS), ITEMS, meta)

    assert Searcher.load(FineTunedEmbedder("aaa"), tmp_path).gallery_size == 3
    with pytest.raises(ValueError, match="different fine-tuned weights"):
        Searcher.load(FineTunedEmbedder("bbb"), tmp_path)
    with pytest.raises(ValueError, match="different fine-tuned weights"):
        Searcher.load(FineTunedEmbedder(None), tmp_path)


def test_a_gallery_built_without_an_adapter_is_rejected_by_a_fine_tuned_model(tmp_path):
    save_gallery(tmp_path, GalleryIndex(GALLERY_VECTORS), ITEMS, {"model": "test-model"})

    assert Searcher.load(FineTunedEmbedder(None), tmp_path).gallery_size == 3
    with pytest.raises(ValueError, match="different fine-tuned weights"):
        Searcher.load(FineTunedEmbedder("aaa"), tmp_path)


def test_loading_a_gallery_built_with_another_model_fails(tmp_path):
    save_gallery(tmp_path, GalleryIndex(GALLERY_VECTORS), ITEMS, {"model": "other-model"})

    with pytest.raises(ValueError, match="other-model"):
        Searcher.load(FakeEmbedder(), tmp_path)
