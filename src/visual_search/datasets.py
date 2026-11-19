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

SOP_ROOT = "Stanford_Online_Products"
SOP_TEST_LIST = f"{SOP_ROOT}/Ebay_test.txt"
SOP_TRAIN_LIST = f"{SOP_ROOT}/Ebay_train.txt"
INSHOP_PARTITION_FILE = "Eval/list_eval_partition.txt"


@dataclass
class RetrievalSplit:
    query_paths: list[Path]
    query_labels: np.ndarray
    gallery_paths: list[Path]
    gallery_labels: np.ndarray
    query_gallery_ids: np.ndarray | None = None


def _read_sop_list(data_dir: Path, list_file: str) -> tuple[list[Path], np.ndarray]:
    paths, labels = [], []
    # Header: image_id class_id super_class_id path
    for line in (data_dir / list_file).read_text().splitlines()[1:]:
        _, class_id, _, relative_path = line.split()
        paths.append(data_dir / SOP_ROOT / relative_path)
        labels.append(int(class_id))
    return paths, np.asarray(labels)


def _self_retrieval_split(paths: list[Path], labels: np.ndarray) -> RetrievalSplit:
    """Every image is both a query and a gallery item, as in the SOP protocol."""
    return RetrievalSplit(
        query_paths=paths,
        query_labels=labels,
        gallery_paths=paths,
        gallery_labels=labels,
        query_gallery_ids=np.arange(len(paths)),
    )


def load_sop(data_dir: Path, limit: int | None = None) -> RetrievalSplit:
    """Stanford Online Products test split. `limit` keeps the first N images (for smoke runs)."""
    paths, labels = _read_sop_list(Path(data_dir), SOP_TEST_LIST)
    if limit is not None:
        paths, labels = paths[:limit], labels[:limit]
    return _self_retrieval_split(paths, labels)


@dataclass
class TrainingData:
    train_paths: list[Path]
    train_labels: np.ndarray
    validation: RetrievalSplit  # held-out classes, scored the same way as the test split


def load_sop_training(
    data_dir: Path, validation_classes: int = 1000, seed: int = 0, limit: int | None = None
) -> TrainingData:
    """SOP train split, with some classes held out for validation.

    Whole classes are held out, as in the real train/test split, so validation
    measures retrieval of products the model never saw. The test split is not touched.
    `limit` keeps the first N images of the list (for smoke runs).
    """
    paths, labels = _read_sop_list(Path(data_dir), SOP_TRAIN_LIST)
    if limit is not None:
        paths, labels = paths[:limit], labels[:limit]
    classes = np.unique(labels)
    held_out = np.random.default_rng(seed).choice(classes, size=validation_classes, replace=False)
    is_validation = np.isin(labels, held_out)
    return TrainingData(
        train_paths=[path for path, flag in zip(paths, is_validation) if not flag],
        train_labels=labels[~is_validation],
        validation=_self_retrieval_split(
            [path for path, flag in zip(paths, is_validation) if flag], labels[is_validation]
        ),
    )


def load_sop_validation(data_dir: Path, limit: int | None = None) -> RetrievalSplit:
    """The classes held out of training in M2, used to choose settings without the test split.

    It uses the same class count and seed as `visual_search.train`, so a model trained
    there has never seen these classes. `limit` keeps the first N images.
    """
    validation = load_sop_training(data_dir).validation
    if limit is None:
        return validation
    return _self_retrieval_split(
        validation.gallery_paths[:limit], validation.gallery_labels[:limit]
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


LOADERS = {"sop": load_sop, "sop-val": load_sop_validation, "inshop": load_inshop}
