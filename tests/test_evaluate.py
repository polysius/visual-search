import numpy as np
import pytest

pytest.importorskip("faiss")

from visual_search.evaluate import evaluate_retrieval

# Two well separated clusters, so nearest neighbours are the same-label items.
GALLERY = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
LABELS = np.array([0, 0, 1, 1])


def test_separate_query_set():
    queries = np.array([[1.0, 0.05], [0.05, 1.0]])
    report = evaluate_retrieval(queries, np.array([0, 1]), GALLERY, LABELS, ks=(1, 2))
    assert report.recall == {1: 1.0, 2: 1.0}
    assert report.map == 1.0
    assert report.num_skipped == 0


def test_queries_from_the_gallery_exclude_themselves():
    # Each item's only other match is its cluster mate. Without dropping self,
    # recall@1 would be trivially 1.0 for every query.
    report = evaluate_retrieval(
        GALLERY, LABELS, GALLERY, LABELS, ks=(1,), query_gallery_ids=np.arange(4)
    )
    assert report.recall[1] == 1.0
    assert report.map == 1.0


def test_self_exclusion_matters_when_neighbours_are_wrong():
    # Item 1 sits next to item 2 (other label), so its cluster mate is not the nearest.
    gallery = np.array([[1.0, 0.0], [0.5, 0.5], [0.4, 0.6], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    report = evaluate_retrieval(
        gallery, labels, gallery, labels, ks=(1,), query_gallery_ids=np.arange(4)
    )
    # Nearest other item: 0 -> item 1 (hit), 1 -> item 2 (miss), 2 -> item 1 (miss), 3 -> item 2 (hit).
    assert report.recall[1] == 0.5


def test_queries_without_a_match_are_skipped():
    queries = np.array([[1.0, 0.0], [1.0, 0.0]])
    report = evaluate_retrieval(queries, np.array([0, 7]), GALLERY, LABELS, ks=(1,))
    assert report.num_skipped == 1
    assert report.num_queries == 1
