import numpy as np
import pytest

from visual_search.metrics import average_precision, recall_at_k

# Three queries over a ranking of five gallery items (True = matches the query).
RELEVANCE = np.array(
    [
        [True, False, False, False, False],  # match at rank 1
        [False, False, True, False, True],  # matches at ranks 3 and 5
        [False, False, False, False, False],  # no match at all
    ]
)


def test_recall_at_k_is_a_hit_rate():
    assert recall_at_k(RELEVANCE, 1).tolist() == [1.0, 0.0, 0.0]
    assert recall_at_k(RELEVANCE, 3).tolist() == [1.0, 1.0, 0.0]


def test_average_precision_by_hand():
    # Query 1: precision@1 = 1 -> AP 1.
    # Query 2: precision@3 = 1/3, precision@5 = 2/5 -> AP (1/3 + 2/5) / 2.
    expected = [1.0, (1 / 3 + 2 / 5) / 2, 0.0]
    assert average_precision(RELEVANCE) == pytest.approx(expected)
