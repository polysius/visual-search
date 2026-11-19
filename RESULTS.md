# Results

## M1 baseline (Stanford Online Products test split)

Pretrained encoders, no fine-tuning. The gallery is the full test split of 60,502 images, and each image is also a query. A query is removed from its own ranking.

| Model | recall@1 | recall@5 | recall@10 | mAP |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | 0.6041 | 0.7289 | 0.7721 | 0.3479 |
| DINOv2 small | 0.5503 | 0.6726 | 0.7171 | 0.3111 |

Run on an RTX 3060 Laptop GPU with batch size 64. Embedding the full split took 330 s for CLIP and 392 s for DINOv2 small.

CLIP is ahead of DINOv2 small on every metric. The two models differ in size as well as in training, so this is a starting point for M2 and not a verdict on the model families.
