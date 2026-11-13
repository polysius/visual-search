# visual-search

Visual search over product images with fine-tuned embeddings. A query photo is embedded, matched against a FAISS index of gallery images, and the closest products are returned.

Planned pipeline:

```
query image -> [YOLO crop] -> vision encoder (CLIP or DINOv2, LoRA fine-tuned) -> L2 normalise -> FAISS flat index -> ranked gallery
```

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | Baseline retrieval: pretrained embeddings, FAISS, recall@1/5/10 and mAP | Code and tests done. Not yet run on real images, see `RESULTS.md` |
| M2 | LoRA contrastive fine-tuning | Not started |
| M3 | YOLO crop before embedding | Not started |
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

The model wrapper uses Hugging Face transformers rather than open_clip. Its attention projections are plain linear layers, which LoRA and ONNX export both support, and the baseline, fine-tuned and exported models share one code path.

GPU work for M2 is planned for Kaggle.
