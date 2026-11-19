"""Embed a gallery of images once and save it as a FAISS index the API can load.

Example:
    python -m visual_search.build_index --data-dir ~/vs-data --split test \
        --adapter artifacts/m2_clip/best --out-dir artifacts/gallery_test

The gallery is embedded with the torch model on the GPU because that is fast. The
ONNX model gives the same vectors to within preprocessing noise (see RESULTS.md),
so the API can search this index with the ONNX embedder.
"""

import argparse
import json
import time
from pathlib import Path

from visual_search.datasets import SOP_ROOT, load_sop, load_sop_validation
from visual_search.embedder import CLIP_MODEL, ImageEmbedder, adapter_fingerprint
from visual_search.index import GalleryIndex
from visual_search.search import GalleryItems, save_gallery

SPLITS = {"test": load_sop, "validation": load_sop_validation}


def main() -> None:
    args = parse_args()
    split = SPLITS[args.split](args.data_dir, limit=args.limit)
    embedder = ImageEmbedder(args.model, device=args.device, adapter_path=args.adapter)

    started = time.perf_counter()
    embeddings = embedder.encode_paths(split.gallery_paths, args.batch_size, args.workers)
    seconds = time.perf_counter() - started

    root = Path(args.data_dir) / SOP_ROOT
    items = GalleryItems(
        images=[path.relative_to(root).as_posix() for path in split.gallery_paths],
        class_ids=[int(label) for label in split.gallery_labels],
    )
    meta = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": adapter_fingerprint(args.adapter),
        "split": args.split,
        "size": len(items.images),
        "dim": int(embeddings.shape[1]),
        "built_with": f"torch on {args.device}",
        "embed_seconds": round(seconds, 1),
    }
    save_gallery(args.out_dir, GalleryIndex(embeddings), items, meta)
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=CLIP_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first N images (smoke runs)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
