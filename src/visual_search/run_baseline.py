"""Command line entry point: embed a dataset with a pretrained model and report retrieval metrics.

Example:
    python -m visual_search.run_baseline --dataset sop --data-dir ~/vs-data \
        --model openai/clip-vit-base-patch32 --out artifacts/m1_clip.json
"""

import argparse
import json
import time
from pathlib import Path

from visual_search.datasets import LOADERS
from visual_search.embedder import CLIP_MODEL, ImageEmbedder
from visual_search.evaluate import evaluate_retrieval


def main() -> None:
    args = parse_args()
    split = LOADERS[args.dataset](args.data_dir, limit=args.limit)
    embedder = ImageEmbedder(args.model, device=args.device)

    started = time.perf_counter()
    gallery_embeddings = embedder.encode_paths(split.gallery_paths, args.batch_size, args.workers)
    if split.query_gallery_ids is not None:
        query_embeddings = gallery_embeddings  # queries are gallery images; embed once
    else:
        query_embeddings = embedder.encode_paths(split.query_paths, args.batch_size, args.workers)
    embed_seconds = time.perf_counter() - started

    report = evaluate_retrieval(
        query_embeddings,
        split.query_labels,
        gallery_embeddings,
        split.gallery_labels,
        query_gallery_ids=split.query_gallery_ids,
    )
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "device": args.device,
        "limit": args.limit,
        "gallery_size": len(split.gallery_paths),
        "embed_seconds": round(embed_seconds, 1),
        **report.as_dict(),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="sop")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default=CLIP_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="use only the first N images (smoke runs)")
    parser.add_argument("--out", type=Path, help="write the result as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    main()
