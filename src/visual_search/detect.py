"""Run a YOLO detector over a dataset split once and save every box it finds.

Example:
    python -m visual_search.detect --data-dir ~/vs-data --split test \
        --out artifacts/boxes_test.json

All boxes above a low confidence are saved with their scores, so crop strategies
can be compared later without running the detector again. Boxes are keyed by the
image path relative to the SOP folder, for example "bicycle_final/251952414262_0.JPG".

Needs the optional extra: pip install -e .[detect]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from visual_search.datasets import SOP_ROOT, load_sop, load_sop_validation

SPLITS = {"validation": load_sop_validation, "test": load_sop}
YOLO_INPUT_SIZE = 640
CHUNK_SIZE = 256  # images decoded and detected at a time


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    paths = SPLITS[args.split](args.data_dir, limit=args.limit).gallery_paths
    detector = YOLO(args.weights)
    root = Path(args.data_dir) / SOP_ROOT
    boxes_by_path = {}
    # Images go through in chunks so that only a few hundred decoded images and their
    # results are held at once. Holding all 60,000 would not fit in memory.
    for start in range(0, len(paths), CHUNK_SIZE):
        chunk = paths[start : start + CHUNK_SIZE]
        small_images, scales = zip(*(load_small(path) for path in chunk))
        results = detector.predict(
            list(small_images),
            batch=args.batch_size,
            conf=args.min_confidence,
            device=args.device,
            verbose=False,
        )
        for path, scale, result in zip(chunk, scales, results):
            boxes_by_path[path.relative_to(root).as_posix()] = boxes_in_original_pixels(
                result, scale
            )
        if (start // CHUNK_SIZE) % 20 == 19:
            print(f"{len(boxes_by_path)} / {len(paths)} images", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(boxes_by_path))
    with_boxes = sum(1 for rows in boxes_by_path.values() if rows)
    print(f"{with_boxes} of {len(boxes_by_path)} images have at least one box -> {args.out}")


def load_small(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Decode an image no larger than YOLO's input size, with the factors to undo the shrink.

    Some SOP photos are 3600 x 2400. Ultralytics copies the full-size batch to the GPU
    before resizing it, which runs out of memory on a 6 GB card, and YOLO shrinks the
    image to 640 pixels anyway. Shrinking here first gives the detector the same view.
    """
    image = Image.open(path).convert("RGB")
    original_width, original_height = image.size
    image.thumbnail((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    scale = (original_width / image.width, original_height / image.height)
    return np.asarray(image)[:, :, ::-1], scale  # ultralytics expects BGR arrays


def boxes_in_original_pixels(result, scale: tuple[float, float]) -> list[list[float]]:
    x_scale, y_scale = scale
    rows = result.boxes.xyxy.cpu().numpy() * [x_scale, y_scale, x_scale, y_scale]
    scores = result.boxes.conf.cpu().numpy()
    return [[*row.round(1).tolist(), round(float(score), 3)] for row, score in zip(rows, scores)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N images (smoke runs)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
