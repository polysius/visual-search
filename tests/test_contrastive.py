import math

import pytest
import torch

from visual_search.contrastive import ClassBalancedBatchSampler, supervised_contrastive_loss


def test_sampler_batches_have_the_requested_classes_and_images():
    labels = [0] * 4 + [1] * 3 + [2] * 5 + [3] * 2 + [4] * 6 + [5] * 2
    sampler = ClassBalancedBatchSampler(labels, classes_per_batch=3, images_per_class=2)

    batches = list(sampler)

    assert len(batches) == len(sampler) == 2
    for batch in batches:
        batch_labels = [labels[i] for i in batch]
        assert len(batch) == 6
        assert len(set(batch_labels)) == 3
        assert all(batch_labels.count(label) == 2 for label in set(batch_labels))


def test_sampler_repeats_images_of_small_classes():
    sampler = ClassBalancedBatchSampler([0, 1, 1], classes_per_batch=2, images_per_class=3)

    (batch,) = list(sampler)

    assert len(batch) == 6
    assert sorted(set(batch)) == [0, 1, 2]


def test_sampler_rejects_too_few_classes():
    with pytest.raises(ValueError):
        ClassBalancedBatchSampler([0, 0, 1, 1], classes_per_batch=3, images_per_class=2)


def test_loss_matches_hand_computed_case():
    # Two products with two identical images each, and the products orthogonal.
    # With temperature 1, an anchor sees similarities (1, 0, 0): one positive, two negatives.
    # Its loss is -log(e / (e + 2)).
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])

    loss = supervised_contrastive_loss(embeddings, labels, temperature=1.0)

    assert loss.item() == pytest.approx(math.log(math.e + 2) - 1)


def test_loss_is_lower_when_same_class_images_are_closer():
    labels = torch.tensor([0, 0, 1, 1])
    close = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]])

    assert supervised_contrastive_loss(close, labels, 0.1) < supervised_contrastive_loss(
        mixed, labels, 0.1
    )


def test_loss_skips_images_without_a_positive():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])

    loss = supervised_contrastive_loss(embeddings, labels, temperature=1.0)

    assert torch.isfinite(loss)
