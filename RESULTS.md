# Results

## M1 baseline (Stanford Online Products test split)

Pretrained encoders, no fine-tuning. The gallery is the full test split of 60,502 images, and each image is also a query. A query is removed from its own ranking.

| Model | recall@1 | recall@5 | recall@10 | mAP |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | 0.6041 | 0.7289 | 0.7721 | 0.3479 |
| DINOv2 small | 0.5503 | 0.6726 | 0.7171 | 0.3111 |

Run on an RTX 3060 Laptop GPU with batch size 64. Embedding the full split took 330 s for CLIP and 392 s for DINOv2 small.

## M2 LoRA fine-tuning (Stanford Online Products test split)

CLIP ViT-B/32 with a LoRA adapter, trained on the SOP train split with a supervised contrastive loss. The test split was scored once, with the epoch chosen on validation.

| Model | recall@1 | recall@5 | recall@10 | mAP |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32, pretrained (M1) | 0.6041 | 0.7289 | 0.7721 | 0.3479 |
| CLIP ViT-B/32, LoRA fine-tuned | 0.8307 | 0.9171 | 0.9389 | 0.6660 |

Setup: rank 16 LoRA on the attention query and value projections, plus the CLIP visual projection trained in full (983,040 trainable parameters). Batches hold 64 products with 2 images each. AdamW at learning rate 1e-4 with warmup and cosine decay, temperature 0.05, bf16 autocast, 10 epochs, seed 0. Training took about 17 minutes on an RTX 3060 Laptop GPU.

Validation used 1,000 whole classes held out of the train split. On those classes recall@1 rose from 0.7465 before training to 0.9306 after epoch 10, and epoch 9 scored highest at 0.9308, so epoch 9 was used for the test run. Validation recall@1 was flat over the last three epochs (0.9301, 0.9308, 0.9306) while mAP crept up from 0.8328 to 0.8350, so the run was close to converged at this rank.

## M1 notes

CLIP is ahead of DINOv2 small on every metric. The two models differ in size as well as in training, so this is a starting point for M2 and not a verdict on the model families.
