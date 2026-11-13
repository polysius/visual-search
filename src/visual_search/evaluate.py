"""Evaluate retrieval quality of a set of embeddings."""

from dataclasses import dataclass

import numpy as np

from visual_search.index import GalleryIndex
from visual_search.metrics import average_precision, recall_at_k

DEFAULT_KS = (1, 5, 10)
QUERY_CHUNK = 128  # queries ranked at once; 128 x 60k gallery keeps each matrix near 60 MB


@dataclass
class RetrievalReport:
    recall: dict[int, float]
    map: float
    num_queries: int
    num_skipped: int  # queries with no matching gallery item; excluded from the means

    def as_dict(self) -> dict:
        return {
            **{f"recall@{k}": round(value, 4) for k, value in self.recall.items()},
            "mAP": round(self.map, 4),
            "num_queries": self.num_queries,
            "num_skipped": self.num_skipped,
        }


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    ks: tuple[int, ...] = DEFAULT_KS,
    query_gallery_ids: np.ndarray | None = None,
) -> RetrievalReport:
    """Rank the whole gallery for every query and score the rankings.

    `query_gallery_ids` is for protocols where queries are drawn from the gallery
    itself (Stanford Online Products): entry i is the gallery id of query i, and
    that item is removed from the query's own ranking.
    """
    index = GalleryIndex(gallery_embeddings)
    gallery_labels = np.asarray(gallery_labels)
    query_labels = np.asarray(query_labels)

    hits_per_k = {k: [] for k in ks}
    average_precisions = []
    num_skipped = 0

    for start in range(0, len(query_embeddings), QUERY_CHUNK):
        chunk = slice(start, start + QUERY_CHUNK)
        _, ranked_ids = index.search(query_embeddings[chunk], k=index.size)
        if query_gallery_ids is not None:
            ranked_ids = _drop_self(ranked_ids, query_gallery_ids[chunk])

        relevance = gallery_labels[ranked_ids] == query_labels[chunk, None]
        has_match = relevance.any(axis=1)
        num_skipped += int((~has_match).sum())
        relevance = relevance[has_match]

        for k in ks:
            hits_per_k[k].append(recall_at_k(relevance, k))
        average_precisions.append(average_precision(relevance))

    return RetrievalReport(
        recall={k: float(np.concatenate(hits).mean()) for k, hits in hits_per_k.items()},
        map=float(np.concatenate(average_precisions).mean()),
        num_queries=len(query_embeddings) - num_skipped,
        num_skipped=num_skipped,
    )


def _drop_self(ranked_ids: np.ndarray, self_ids: np.ndarray) -> np.ndarray:
    """Remove each query's own gallery id from its row, keeping the order of the rest."""
    keep = ranked_ids != self_ids[:, None]
    return ranked_ids[keep].reshape(len(ranked_ids), -1)
