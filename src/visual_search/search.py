"""Search a gallery with one query image: embed it, then look up its nearest gallery items."""

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from visual_search.index import GalleryIndex

INDEX_FILE = "index.faiss"
ITEMS_FILE = "items.json"
META_FILE = "meta.json"


@dataclass
class Match:
    rank: int
    score: float  # cosine similarity, 1.0 is identical
    class_id: int  # the product the gallery image shows
    image: str  # path of the gallery image, relative to the dataset folder


@dataclass
class GalleryItems:
    images: list[str]
    class_ids: list[int]


def save_gallery(out_dir: Path, index: GalleryIndex, items: GalleryItems, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save(out_dir / INDEX_FILE)
    (out_dir / ITEMS_FILE).write_text(
        json.dumps({"images": items.images, "class_ids": items.class_ids})
    )
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


class Searcher:
    """An embedder plus a gallery. `embedder` needs `encode_images(list[PIL]) -> (n, dim)`."""

    def __init__(self, embedder, index: GalleryIndex, items: GalleryItems):
        if index.size != len(items.images):
            raise ValueError(f"index has {index.size} vectors but {len(items.images)} items")
        self.embedder = embedder
        self.index = index
        self.items = items

    @classmethod
    def load(cls, embedder, gallery_dir: Path) -> "Searcher":
        gallery_dir = Path(gallery_dir)
        meta = json.loads((gallery_dir / META_FILE).read_text())
        embedder_model = getattr(embedder, "model_name", None)
        if embedder_model is not None and meta["model"] != embedder_model:
            raise ValueError(
                f"gallery was built with {meta['model']} but the embedder is {embedder_model}"
            )
        # A gallery embedded with other fine-tuned weights is in a different vector space,
        # and searching it gives worse results without any error, so check it here.
        if hasattr(embedder, "adapter_sha256") and embedder.adapter_sha256 != meta.get(
            "adapter_sha256"
        ):
            raise ValueError(
                "the gallery was embedded with different fine-tuned weights than this model "
                f"(gallery: {meta.get('adapter_sha256')}, model: {embedder.adapter_sha256}); "
                "rebuild the gallery with visual_search.build_index, or export the model again "
                "with visual_search.export if it was made before weights were recorded"
            )
        raw = json.loads((gallery_dir / ITEMS_FILE).read_text())
        items = GalleryItems(raw["images"], raw["class_ids"])
        return cls(embedder, GalleryIndex.load(gallery_dir / INDEX_FILE), items)

    @property
    def gallery_size(self) -> int:
        return self.index.size

    def search(self, image: Image.Image, k: int) -> list[Match]:
        """The `k` gallery items most similar to the image, best first."""
        scores, ids = self.index.search(self.embedder.encode_images([image]), k)
        return [
            Match(
                rank=rank,
                score=float(score),
                class_id=self.items.class_ids[item],
                image=self.items.images[item],
            )
            for rank, (score, item) in enumerate(zip(scores[0], ids[0]), start=1)
        ]
