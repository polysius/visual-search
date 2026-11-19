"""Batch sampling and loss for contrastive fine-tuning.

Retrieval wants images of the same product close together and images of other
products far apart. A batch therefore needs several images per product, which
`ClassBalancedBatchSampler` provides, and a loss that pulls those images
together, which `supervised_contrastive_loss` provides.
"""

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F


class ClassBalancedBatchSampler:
    """Yields batches of `classes_per_batch` classes with `images_per_class` images each.

    A class with fewer images than `images_per_class` is sampled with repeats.
    One epoch visits each class at most once, so it has `num_classes // classes_per_batch` batches.
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        images_per_class: int,
        seed: int = 0,
    ):
        self.indices_by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.indices_by_class[int(label)].append(index)
        if len(self.indices_by_class) < classes_per_batch:
            raise ValueError("fewer classes than classes_per_batch")
        self.classes_per_batch = classes_per_batch
        self.images_per_class = images_per_class
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices_by_class) // self.classes_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        classes = list(self.indices_by_class)
        self.random.shuffle(classes)
        for start in range(0, len(self) * self.classes_per_batch, self.classes_per_batch):
            batch = []
            for label in classes[start : start + self.classes_per_batch]:
                batch.extend(self._pick(self.indices_by_class[label]))
            yield batch

    def _pick(self, indices: list[int]) -> list[int]:
        if len(indices) >= self.images_per_class:
            return self.random.sample(indices, self.images_per_class)
        return self.random.choices(indices, k=self.images_per_class)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalised embeddings.

    For each image, every other image of the same class is a positive and every
    remaining image in the batch is a negative. Images with no positive in the
    batch are skipped.
    """
    vectors = F.normalize(embeddings.float(), dim=1)
    similarity = vectors @ vectors.T / temperature
    is_self = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    similarity = similarity.masked_fill(is_self, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    is_positive = (labels[:, None] == labels[None, :]) & ~is_self
    positive_count = is_positive.sum(dim=1)
    has_positive = positive_count > 0
    positive_log_probability = log_probability.masked_fill(~is_positive, 0.0).sum(dim=1)
    return -(positive_log_probability[has_positive] / positive_count[has_positive]).mean()
