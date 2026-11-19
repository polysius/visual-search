# visual-search

Visual search over product images with fine-tuned embeddings. A query photo is embedded, matched against a FAISS index of gallery images, and the closest products are returned.

Planned pipeline:

```
query image -> [YOLO crop] -> vision encoder (CLIP or DINOv2, LoRA fine-tuned) -> L2 normalise -> FAISS flat index -> ranked gallery
```

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | Baseline retrieval: pretrained embeddings, FAISS, recall@1/5/10 and mAP | Done. Baseline numbers for CLIP and DINOv2 small are in `RESULTS.md` |
| M2 | LoRA contrastive fine-tuning | Done for CLIP: recall@1 0.604 to 0.831 on the SOP test split, see `RESULTS.md`. DINOv2 not yet fine-tuned |
| M3 | YOLO crop before embedding | Built and measured. No crop setting beat full images on recall@1 or mAP on SOP validation, so it is off by default, see `RESULTS.md` |
| M4 | ONNX export, FastAPI upload endpoint, latency table | Not started |

## Dataset

The spec default, DeepFashion In-Shop, is released under a signed agreement and cannot be fetched by a script. This project uses Stanford Online Products (SOP), also one of the spec options. Both loaders exist in `src/visual_search/datasets.py`, so In-Shop can be used once access is granted.

SOP protocol: the test split is both query set and gallery. Each query is ranked against every other test image, and the query itself is removed from its own ranking.

## Metrics

`src/visual_search/metrics.py` holds the metrics, using numpy only.

* recall@k is a hit rate: the share of queries with at least one correct match in the top k.
* mAP averages precision over every correct match in the full ranked gallery.

Both are checked against hand-computed cases in `tests/test_metrics.py`.

## Run

```
pip install -e .[dev]
DATA_DIR=~/vs-data scripts/download_sop.sh
python -m visual_search.run_baseline --dataset sop --data-dir ~/vs-data \
    --model openai/clip-vit-base-patch32 --out artifacts/m1_clip.json
pytest
```

`--limit N` embeds only the first N images, for a quick smoke run.

Fine-tune with LoRA, then score the best epoch on the test split:

```
python -m visual_search.train --data-dir ~/vs-data --device cuda --out-dir artifacts/m2_clip
python -m visual_search.run_baseline --dataset sop --data-dir ~/vs-data --device cuda \
    --adapter artifacts/m2_clip/best --out artifacts/m2_clip_test.json
```

Crop each image to its detected object before embedding. This needs the `detect` extra (`pip install -e .[detect]`). The detector saves its boxes once, and `run_baseline` reads them:

```
python -m visual_search.detect --data-dir ~/vs-data --split validation --out artifacts/boxes_val.json
python -m visual_search.run_baseline --dataset sop-val --data-dir ~/vs-data --device cuda \
    --crop artifacts/boxes_val.json --crop-strategy top
```

`scripts/sweep_crop.ps1` compares several crop settings on the validation classes. Cropping is off unless `--crop` is given, and it works with the SOP datasets only.

Training holds out 1,000 whole classes of the SOP train split for validation and writes `best`, `last` and `log.json` to the output folder. The test split is only used by the second command.

The model wrapper uses Hugging Face transformers rather than open_clip. Its attention projections are plain linear layers, which LoRA and ONNX export both support, and the baseline, fine-tuned and exported models share one code path.

M2 trained locally on a 6 GB RTX 3060 in about 17 minutes, so Kaggle is not needed for CLIP.
