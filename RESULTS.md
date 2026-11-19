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

## M3 YOLO crop before embedding

Each image is cropped to the object a YOLO detector finds, then embedded. The detector is the COCO-pretrained YOLO11n. Its class labels are ignored. COCO has no class for most SOP products, and in two spot checks a stapler photo came back as apple and scissors and a toaster photo as book, frisbee and mouse, so its boxes only say "an object is here". The crop is grown by a margin and made square, because the embedders centre-crop a square and would otherwise cut the ends off a wide object. When there is no box, or the box covers less than a set share of the image, the full image is used.

Settings were compared on the validation classes (5,247 images held out of the SOP train split, the same classes M2 validated on). The test split was not used.

| Setting | CLIP recall@1 | CLIP mAP | CLIP + LoRA recall@1 | CLIP + LoRA mAP |
| --- | --- | --- | --- | --- |
| Full image, no crop | 0.7465 | 0.4991 | 0.9308 | 0.8349 |
| Top box, confidence 0.25 | 0.7296 | 0.4747 | 0.9211 | 0.8197 |
| Top box, confidence 0.5 | 0.7374 | 0.4863 | 0.9243 | 0.8258 |
| Union of boxes, confidence 0.25 | 0.7383 | 0.4838 | 0.9238 | 0.8237 |
| Top box, margin 0.25 | 0.7456 | 0.4888 | 0.9272 | 0.8282 |
| Top box, ignore boxes under 30% of the image | 0.7452 | 0.4964 | 0.9287 | 0.8338 |

No crop setting beats the full image on recall@1 or mAP, for either model. Settings that remove more of the image score lower on mAP. A few settings are higher than full images on CLIP + LoRA recall@5 and recall@10, by at most 0.0006, which is within noise for 5,247 queries. Ignoring small boxes loses the least on mAP for both models. Growing the margin to 0.25 is slightly closer on pretrained recall@1.

This is plausible for SOP. Its photos are about 400 pixels wide with the product centred on a white background, so there is little to remove, and a crop can cut off part of the product or drop context that separates two similar items. Cropping stays off by default and the test split was not scored.

Cropping is more likely to help on user photos where the product is small in a cluttered scene, which is the case for the upload endpoint in M4. The crop code works on a single image so that endpoint can use it.

## M1 notes

CLIP is ahead of DINOv2 small on every metric. The two models differ in size as well as in training, so this is a starting point for M2 and not a verdict on the model families.
