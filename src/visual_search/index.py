"""Exact inner-product search over L2-normalised embeddings, backed by FAISS."""

import numpy as np


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


class GalleryIndex:
    """Flat index: with unit vectors, inner product equals cosine similarity."""

    def __init__(self, gallery_embeddings: np.ndarray):
        import faiss  # imported here so the metrics tests do not need faiss

        vectors = np.ascontiguousarray(l2_normalize(gallery_embeddings), dtype=np.float32)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return self._index.ntotal

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, gallery_ids), each of shape (num_queries, k), best first."""
        queries = np.ascontiguousarray(l2_normalize(query_embeddings), dtype=np.float32)
        return self._index.search(queries, min(k, self.size))
