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

## M4 ONNX export and the search API

The M2 model (CLIP with its LoRA adapter merged in) is exported to ONNX and served by a FastAPI endpoint. `POST /search` takes an uploaded photo and returns the closest gallery images. The gallery is the 60,502 SOP test images, embedded once and stored in a FAISS flat index.

### The served path matches the evaluated path

Preprocessing in the server uses PIL and numpy only, with the resize, crop and normalisation settings read from a file saved at export time. It differs from the Hugging Face processor by at most about 2 of 255 gray levels per pixel, because the two resize implementations round slightly differently. To check that this does not matter, the same 500 SOP test images were scored through both paths:

| Path | recall@1 | recall@5 | recall@10 | mAP |
| --- | --- | --- | --- | --- |
| torch, GPU, Hugging Face processor | 0.898 | 0.966 | 0.980 | 0.7989 |
| ONNX, CPU, PIL preprocessing | 0.898 | 0.968 | 0.980 | 0.7996 |

The ONNX output differs from torch by a few times 1e-06 on random input, and the value changes a little with each random input. The export step fails if the difference passes 1e-3.

`scripts/smoke_check_server.py` queries the running server over HTTP with 100 test images spread through the gallery. The nearest other image shared the query's product in 81 of 100 cases, against about 83 expected from the test recall@1 of 0.8307. That is within sampling noise for 100 queries. Two of the 100 queries did not return themselves at rank 1. I assume the gallery holds duplicate images, and one random query did return two images with identical scores, but I did not check these two cases.

### Limits on uploads

The endpoint refuses what it cannot handle safely, with a 4xx and no server error:

* A file over 10 MB gets a 413. A chunked upload with no declared length is counted as it arrives and cut off at the limit, so the server does not write the whole body to disk first. A unit test shows the read stops once the limit is passed. The smoke script offers 400 MB this way, the server stops reading at about 10 MB, and the client gets a 413 in under half a second. After a 413 the server leaves the connection open, so a proxy in front of it should close connections that stall.
* An image over 64 million pixels, a file that is not an image, and a truncated image each get a 400.
* An image more than 100 times longer than it is wide gets a 400, and the shape is checked from the file header before any pixels are decoded. The resize makes the long side 224 times the aspect ratio, so without this limit a tiny 20000 by 1 image expanded to several gigabytes inside the server, and a 10 million by 1 image made the request fail with a server error. A 1 by 64 million image, which is inside the pixel limit, cost the server about 2 GB to decode before the shape was checked first. In the smoke script six thin images, from 6000 by 1 up to 1 by 64 million and 64 million by 1, all get a 400 in about 10 ms, and the server's memory stays at 612 MB, which is the model and the index.
* An image that passes the limits is decoded in memory. An 8000 by 8000 image, which is 64 million pixels, is accepted and takes 0.75 s, and the server's peak memory rises from 612 to 1,378 MB while it handles it. A server that takes many such uploads at once needs memory for each of them.
* The server also refuses to start if the gallery was embedded with different fine-tuned weights than the ONNX model. Searching such a gallery would give worse results with no error. The check compares a hash of the adapter weights stored with both.

The ONNX model was not scored on the full test split. The gallery was embedded with the torch model on the GPU, and the API searches it with ONNX query vectors. The 500-image comparison above is the evidence that the two agree.

### Latency

One search request at batch size 1, on a laptop with an Intel Core i7-12700H CPU (Windows reports 14 cores and 20 logical processors) and an RTX 3060 Laptop GPU. Each row is the median (p50) and 95th percentile (p95) in milliseconds over 200 runs after 10 warm-up runs. The HTTP rows use 100 runs against a real uvicorn server on localhost, with the ONNX CPU model and the 60,502 image index. The server and the client share this machine, so the HTTP rows include a small amount of client time.

| Stage | Typical photo, 400 x 400, 22 KB | Large photo, 3600 x 2400, 1.9 MB |
| --- | --- | --- |
| Decode and preprocess | 2.6 / 2.8 | 107.7 / 123.7 |
| Forward pass, torch CPU | 45.8 / 49.4 | 44.0 / 48.7 |
| Forward pass, torch GPU | 5.2 / 5.7 | 5.2 / 5.7 |
| Forward pass, ONNX CPU | 29.1 / 30.5 | 29.0 / 30.2 |
| FAISS search over 60,502 vectors, k = 10 | 5.4 / 5.9 | 5.3 / 5.8 |
| Whole request over HTTP, ONNX CPU | 67.9 / 92.3 | 216.9 / 263.7 |

The CPU timings on this laptop move between runs. An earlier run of the same benchmark gave 32.6 ms for torch CPU, 20.8 ms for ONNX CPU and 48.5 ms for the whole typical request, about 30 to 40 percent faster than the table above. Between those two runs the benchmark script was refactored without changing what it times. The GPU rows moved by under 12 percent across the runs and the FAISS rows by under 17 percent, comparing both the median and the 95th percentile. I did not find the cause of the CPU difference, so treat the CPU rows as accurate to roughly 40 percent, and the ratios between rows as more reliable than the absolute values. A typical request therefore takes about 50 to 70 ms on this machine.

The forward pass does the same work for both photos, since both are resized to 224 x 224 first, and in this run the two columns agree.

What the table says:

* ONNX on the CPU is 1.5 to 1.6 times faster than torch on the CPU for the same model in this run, and 1.2 to 1.6 times across both runs. The GPU forward pass is 4 to 6 times faster than ONNX on the CPU. The served endpoint uses the CPU ONNX model, since the installed ONNX Runtime is the CPU build.
* For a typical photo the model forward pass is the largest part of the request, about 29 of the 68 ms. The rest is FAISS search at about 5 ms, preprocessing at about 3 ms, and about 31 ms of HTTP, multipart parsing and JSON that this table does not split further.
* For a large photo, decoding the JPEG and preprocessing it together are the largest part, about 108 of the 217 ms. The benchmark times those two steps as one, so it does not say how the 108 ms divides between them. A client that shrinks the photo before uploading would likely save much of it, though I did not measure that.
* A flat FAISS index scans every vector, so the search time grows with the gallery size. At 60,502 vectors it is a small part of the request.

No quantized or half precision model was built, so there is no speed and accuracy trade-off to report.

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

Cropping is more likely to help on user photos where the product is small in a cluttered scene. The crop code works on a single image, so an endpoint option could use it later. The M4 endpoint does not crop, and nothing measured here says it should.

## M1 notes

CLIP is ahead of DINOv2 small on every metric. The two models differ in size as well as in training, so this is a starting point for M2 and not a verdict on the model families.
