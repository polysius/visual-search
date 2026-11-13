"""Retrieval metrics computed from ranked relevance.

Every function takes `relevance`, a boolean matrix of shape (num_queries, num_ranked)
where entry [q, r] is True when the item at rank r (0 is best) matches query q.
Only numpy is needed, so these run anywhere.
"""

import numpy as np


def recall_at_k(relevance: np.ndarray, k: int) -> np.ndarray:
    """Per query: 1.0 when at least one match sits in the top k, else 0.0.

    This is the metric-learning definition of recall@k (a hit rate), not the
    fraction of all matches that were retrieved.
    """
    return relevance[:, :k].any(axis=1).astype(np.float64)


def average_precision(relevance: np.ndarray) -> np.ndarray:
    """Per query: average of precision@r taken at each rank r that holds a match.

    The whole ranking is used, so a match ranked 500th still counts. Queries with
    no match anywhere get 0.0; callers should drop them before averaging.
    """
    hits = relevance.astype(np.float64)
    ranks = np.arange(1, relevance.shape[1] + 1)
    precision_at_rank = np.cumsum(hits, axis=1) / ranks
    num_matches = hits.sum(axis=1)
    summed = (precision_at_rank * hits).sum(axis=1)
    return np.divide(summed, num_matches, out=np.zeros_like(summed), where=num_matches > 0)
