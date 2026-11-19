"""Time each stage of one search request, and the whole request over HTTP.

Usage:
    python scripts/benchmark_latency.py --onnx artifacts/onnx/clip_lora.onnx \
        --adapter artifacts/m2_clip/best --gallery-dir artifacts/gallery_test \
        --typical-image <a 400 px photo> --large-image <a 3600 x 2400 photo> \
        --out artifacts/m4_latency.json

Everything runs at batch size 1, which is what one upload looks like. Each timing
does warm-up runs first, then many timed runs, and reports the median (p50) and the
95th percentile (p95) in milliseconds.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WARMUP_RUNS = 10
TIMED_RUNS = 200
HTTP_TIMED_RUNS = 100  # the large image takes long enough that fewer runs are fine
SERVER_START_TIMEOUT_SECONDS = 120


def time_ms(function, runs: int = TIMED_RUNS) -> list[float]:
    for _ in range(WARMUP_RUNS):
        function()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def summarize(samples: list[float]) -> dict:
    return {
        "p50_ms": round(float(np.percentile(samples, 50)), 2),
        "p95_ms": round(float(np.percentile(samples, 95)), 2),
        "runs": len(samples),
    }


def main() -> None:
    args = parse_args()
    import faiss
    import onnxruntime
    import torch
    from PIL import Image

    from visual_search.embedder import ImageEmbedder, pooled_embedding
    from visual_search.index import GalleryIndex
    from visual_search.onnx_embedder import OnnxEmbedder
    from visual_search.preprocess import preprocess
    from visual_search.serve import decode_upload

    onnx_embedder = OnnxEmbedder(args.onnx)
    torch_cpu = ImageEmbedder(onnx_embedder.model_name, "cpu", args.adapter)
    torch_gpu = ImageEmbedder(onnx_embedder.model_name, "cuda", args.adapter)
    gallery_index = GalleryIndex.load(args.gallery_dir / "index.faiss")
    dim = onnx_embedder.session.get_outputs()[0].shape[1]
    query_vector = np.random.default_rng(0).standard_normal((1, dim)).astype(np.float32)

    def stage_functions(data: bytes) -> dict:
        """The functions to time for one uploaded photo, each taking no arguments."""
        pixels = preprocess(decode_upload(data), onnx_embedder.config)[None]
        tensor = torch.from_numpy(pixels)

        def torch_forward(embedder, device):
            def run():
                with torch.no_grad():
                    vectors = pooled_embedding(embedder.model, embedder.is_clip, tensor.to(device))
                return vectors.cpu().numpy()  # copying to the CPU also waits for the GPU

            return run

        return {
            "decode_and_preprocess": lambda: preprocess(decode_upload(data), onnx_embedder.config),
            "forward_torch_cpu": torch_forward(torch_cpu, "cpu"),
            "forward_torch_gpu": torch_forward(torch_gpu, "cuda"),
            "forward_onnx_cpu": lambda: onnx_embedder.session.run(None, {"pixel_values": pixels}),
            "faiss_search_k10": lambda: gallery_index.search(query_vector, 10),
        }

    results = {"environment": environment(torch, onnxruntime, faiss), "images": {}}
    for label, path in (("typical", args.typical_image), ("large", args.large_image)):
        data = Path(path).read_bytes()
        results["images"][label] = {
            "file": Path(path).name,
            "size": list(Image.open(path).size),
            "bytes": len(data),
            "stages": {
                name: summarize(time_ms(run)) for name, run in stage_functions(data).items()
            },
        }
        results["images"][label]["http_request"] = summarize(
            http_request_times(args, data, HTTP_TIMED_RUNS)
        )
        print(label, json.dumps(results["images"][label]["stages"]), flush=True)
        print(label, "http", results["images"][label]["http_request"], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


def http_request_times(args, data: bytes, runs: int) -> list[float]:
    """Start the real server in a subprocess and time POST /search against it."""
    import httpx

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "visual_search.serve",
            "--onnx",
            str(args.onnx),
            "--gallery-dir",
            str(args.gallery_dir),
            "--port",
            str(args.port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=60) as client:
            wait_until_healthy(client, server)

            def request():
                response = client.post("/search", files={"file": ("q.jpg", data)}, params={"k": 10})
                response.raise_for_status()

            return time_ms(request, runs)
    finally:
        server.terminate()
        server.wait(timeout=30)


def is_healthy(client) -> bool:
    import httpx

    try:
        return client.get("/health").status_code == 200
    except httpx.TransportError:  # the server is still starting, so nothing is listening yet
        return False


def wait_until_healthy(client, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("the server exited before it became healthy")
        if is_healthy(client):
            return
        time.sleep(0.5)
    raise TimeoutError("the server did not become healthy in time")


def environment(torch, onnxruntime, faiss) -> dict:
    import fastapi
    import PIL
    import uvicorn

    return {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "faiss": faiss.__version__ if hasattr(faiss, "__version__") else None,
        "pillow": PIL.__version__,
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path, default=None, help="LoRA adapter, for the torch rows"
    )
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--typical-image", type=Path, required=True)
    parser.add_argument("--large-image", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
