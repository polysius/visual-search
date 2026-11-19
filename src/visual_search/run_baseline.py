"""Command line entry point: embed a dataset with a pretrained model and report retrieval metrics.

Example:
    python -m visual_search.run_baseline --dataset sop --data-dir ~/vs-data \
        --model openai/clip-vit-base-patch32 --out artifacts/m1_clip.json
"""

import argparse
import json
import time
from pathlib import Path

from visual_search.crop import STRATEGIES, BoxCropper
from visual_search.datasets import LOADERS, SOP_ROOT
from visual_search.embedder import CLIP_MODEL, ImageEmbedder
from visual_search.evaluate import evaluate_retrieval
from visual_search.onnx_embedder import OnnxEmbedder


def main() -> None:
    args = parse_args()
    if args.crop is not None and args.dataset == "inshop":
        raise SystemExit("--crop works with the SOP datasets only (sop, sop-val)")
    if args.onnx is not None and args.adapter is not None:
        raise SystemExit("--onnx already contains the merged adapter; do not pass --adapter")
    split = LOADERS[args.dataset](args.data_dir, limit=args.limit)
    if args.onnx is not None:
        embedder = OnnxEmbedder(args.onnx)
    else:
        embedder = ImageEmbedder(args.model, device=args.device, adapter_path=args.adapter)

    crop = make_cropper(args)

    started = time.perf_counter()
    gallery_embeddings = embedder.encode_paths(
        split.gallery_paths, args.batch_size, args.workers, crop
    )
    if split.query_gallery_ids is not None:
        query_embeddings = gallery_embeddings  # queries are gallery images; embed once
    else:
        query_embeddings = embedder.encode_paths(
            split.query_paths, args.batch_size, args.workers, crop
        )
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
        # An ONNX file knows which model it came from, and --model is ignored for it.
        "model": embedder.model_name if args.onnx else args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "onnx": str(args.onnx) if args.onnx else None,
        "crop": crop_settings(args),
        "device": "cpu (onnxruntime)" if args.onnx else args.device,
        "limit": args.limit,
        "gallery_size": len(split.gallery_paths),
        "embed_seconds": round(embed_seconds, 1),
        **report.as_dict(),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


def crop_settings(args: argparse.Namespace) -> dict | None:
    if args.crop is None:
        return None
    return {
        "boxes": str(args.crop),
        "strategy": args.crop_strategy,
        "min_confidence": args.crop_min_confidence,
        "margin": args.crop_margin,
        "min_area_fraction": args.crop_min_area,
    }


def make_cropper(args: argparse.Namespace) -> BoxCropper | None:
    settings = crop_settings(args)
    if settings is None:
        return None
    return BoxCropper.from_file(
        args.crop,
        root=args.data_dir / SOP_ROOT,
        strategy=settings["strategy"],
        min_confidence=settings["min_confidence"],
        margin=settings["margin"],
        min_area_fraction=settings["min_area_fraction"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="sop")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default=CLIP_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter folder saved by visual_search.train",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="embed with an ONNX model made by visual_search.export (CPU); replaces --model/--adapter",
    )
    parser.add_argument(
        "--crop",
        type=Path,
        default=None,
        help="boxes saved by visual_search.detect; crop each image to its object before embedding",
    )
    parser.add_argument("--crop-strategy", choices=STRATEGIES, default="top")
    parser.add_argument("--crop-min-confidence", type=float, default=0.25)
    parser.add_argument("--crop-margin", type=float, default=0.1)
    parser.add_argument(
        "--crop-min-area", type=float, default=0.1, help="smaller boxes keep the full image"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="use only the first N images (smoke runs)")
    parser.add_argument("--out", type=Path, help="write the result as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    main()
