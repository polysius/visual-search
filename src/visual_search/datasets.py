"""Loaders for the retrieval datasets.

Each loader returns a `RetrievalSplit`: image paths and integer labels for the
queries and the gallery. The two datasets use different protocols.

* Stanford Online Products (SOP): the test set is both query set and gallery.
  Each query is ranked against every other test image, so `query_gallery_ids`
  records which gallery item a query is, and the evaluator drops it.
* DeepFashion In-Shop: separate query and gallery images, matched by item id.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOP_TEST_LIST = "Stanford_Online_Products/Ebay_test.txt"
INSHOP_PARTITION_FILE = "Eval/list_eval_partition.txt"


@dataclass
class RetrievalSplit:
    query_paths: list[Path]
    query_labels: np.ndarray
    gallery_paths: list[Path]
    gallery_labels: np.ndarray
    query_gallery_ids: np.ndarray | None = None


def load_sop(data_dir: Path, limit: int | None = None) -> RetrievalSplit:
    """Stanford Online Products test split. `limit` keeps the first N images (for smoke runs)."""
    data_dir = Path(data_dir)
    paths, labels = [], []
    # Header: image_id class_id super_class_id path
    for line in (data_dir / SOP_TEST_LIST).read_text().splitlines()[1:]:
        _, class_id, _, relative_path = line.split()
        paths.append(data_dir / "Stanford_Online_Products" / relative_path)
        labels.append(int(class_id))
    if limit is not None:
        paths, labels = paths[:limit], labels[:limit]
    labels = np.asarray(labels)
    return RetrievalSplit(
        query_paths=paths,
        query_labels=labels,
        gallery_paths=paths,
        gallery_labels=labels,
        query_gallery_ids=np.arange(len(paths)),
    )


def load_inshop(data_dir: Path, limit: int | None = None) -> RetrievalSplit:
    """DeepFashion In-Shop standard split. `limit` keeps the first N queries and galleries."""
    data_dir = Path(data_dir)
    queries: list[tuple[Path, int]] = []
    gallery: list[tuple[Path, int]] = []
    # Two header lines, then: image_name item_id evaluation_status
    for line in (data_dir / INSHOP_PARTITION_FILE).read_text().splitlines()[2:]:
        image_name, item_id, status = line.split()
        entry = (data_dir / image_name, int(item_id.removeprefix("id_")))
        if status == "query":
            queries.append(entry)
        elif status == "gallery":
            gallery.append(entry)
    if limit is not None:
        queries, gallery = queries[:limit], gallery[:limit]
    return RetrievalSplit(
        query_paths=[path for path, _ in queries],
        query_labels=np.asarray([label for _, label in queries]),
        gallery_paths=[path for path, _ in gallery],
        gallery_labels=np.asarray([label for _, label in gallery]),
    )


LOADERS = {"sop": load_sop, "inshop": load_inshop}
