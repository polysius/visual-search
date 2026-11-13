# Results

No retrieval numbers have been produced yet.

## M1 baseline (Stanford Online Products test split)

| Model | recall@1 | recall@5 | recall@10 | mAP |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | not run | not run | not run | not run |
| DINOv2 small | not run | not run | not run | not run |

Reason: the dataset download and torch install were not possible in the environment used to write M1. Native ARM64 Windows has no torch wheel, and the WSL network could not resolve hosts. The metric and index code is covered by unit tests on small hand-built cases.
