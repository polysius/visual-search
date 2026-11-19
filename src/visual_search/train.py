"""LoRA contrastive fine-tuning of a pretrained vision encoder on Stanford Online Products.

Example:
    python -m visual_search.train --data-dir ~/vs-data --device cuda \
        --model openai/clip-vit-base-patch32 --out-dir artifacts/m2_clip

Training uses the SOP train split only. A set of whole classes is held out for
validation, which picks the best epoch. The SOP test split is scored separately by
`run_baseline --adapter <out-dir>/best`, so it never influences any choice made here.
"""

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from visual_search.contrastive import ClassBalancedBatchSampler, supervised_contrastive_loss
from visual_search.datasets import RetrievalSplit, load_sop_training
from visual_search.embedder import CLIP_MODEL, ImageEmbedder, pooled_embedding
from visual_search.evaluate import evaluate_retrieval

# Attention query and value projections have the same names in the CLIP and DINOv2 vision models.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
IMAGE_SIZE = 224
WARMUP_FRACTION = 0.05


class AugmentedImages(Dataset):
    """Training images with random crop and flip, already normalised tensors."""

    def __init__(self, paths: list[Path], labels: np.ndarray, mean: list[float], std: list[float]):
        self.paths = paths
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def main() -> None:
    args = parse_args()
    if args.images_per_class < 2:
        raise SystemExit("--images-per-class must be at least 2, or no image has a positive pair")
    seed_everything(args.seed)
    data = load_sop_training(args.data_dir, args.validation_classes, args.seed)
    embedder = ImageEmbedder(args.model, device=args.device)
    add_lora(embedder, args.rank)

    sampler = ClassBalancedBatchSampler(
        data.train_labels, args.classes_per_batch, args.images_per_class, args.seed
    )
    loader = DataLoader(
        AugmentedImages(
            data.train_paths,
            data.train_labels,
            embedder.processor.image_mean,
            embedder.processor.image_std,
        ),
        batch_sampler=sampler,
        num_workers=args.workers,
    )
    steps_per_epoch = min(args.max_steps_per_epoch or len(sampler), len(sampler))
    total_steps = steps_per_epoch * args.epochs
    parameters = [p for p in embedder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_images": len(data.train_paths),
        "validation_images": len(data.validation.gallery_paths),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "epochs": [],
    }
    # LoRA starts as a no-op, so this is the pretrained model's validation score.
    log["validation_before_training"] = validate(embedder, data.validation, args)
    print("before training", log["validation_before_training"], flush=True)

    best_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        loss = train_one_epoch(embedder, loader, optimizer, scheduler, steps_per_epoch, args)
        scores = validate(embedder, data.validation, args)
        entry = {
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "validation": scores,
            "seconds": round(time.perf_counter() - started, 1),
        }
        log["epochs"].append(entry)
        print(entry, flush=True)
        embedder.model.save_pretrained(args.out_dir / "last")
        if scores["recall@1"] > best_recall:
            best_recall = scores["recall@1"]
            embedder.model.save_pretrained(args.out_dir / "best")
            log["best_epoch"] = epoch
        (args.out_dir / "log.json").write_text(json.dumps(log, indent=2) + "\n")


def add_lora(embedder: ImageEmbedder, rank: int) -> None:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        # CLIP's final projection is small and sits right before the loss, so train it fully.
        modules_to_save=["visual_projection"] if embedder.is_clip else None,
    )
    embedder.model = get_peft_model(embedder.model, config)


def train_one_epoch(embedder, loader, optimizer, scheduler, steps: int, args) -> float:
    embedder.model.train()
    losses = []
    for images, labels in itertools.islice(loader, steps):
        images, labels = images.to(args.device), labels.to(args.device)
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
            embeddings = pooled_embedding(embedder.model, embedder.is_clip, images)
        loss = supervised_contrastive_loss(embeddings, labels, args.temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def validate(embedder: ImageEmbedder, split: RetrievalSplit, args) -> dict:
    embedder.model.eval()
    embeddings = embedder.encode_paths(split.gallery_paths, args.eval_batch_size, args.workers)
    report = evaluate_retrieval(
        embeddings,
        split.query_labels,
        embeddings,
        split.gallery_labels,
        query_gallery_ids=split.query_gallery_ids,
    )
    return report.as_dict()


def warmup_cosine(total_steps: int):
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return scale


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=CLIP_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--classes-per-batch", type=int, default=64)
    parser.add_argument("--images-per-class", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--validation-classes", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps-per-epoch", type=int, default=None, help="cut epochs short, for smoke runs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
