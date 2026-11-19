"""Query a running server over HTTP and check that its answers and its limits hold up.

Usage:
    python scripts/smoke_check_server.py --onnx artifacts/onnx/clip_lora.onnx \
        --gallery-dir artifacts/gallery_test --data-dir ~/vs-data --out artifacts/m4_smoke.json

It starts the real server in a subprocess and checks:

* 100 gallery images, spread through the gallery, come back at rank 1 and their nearest
  other image shows the same product about as often as the test recall@1 says it should;
* images that are far too thin are refused with a 400, without the server's memory growing;
* an upload with no Content-Length is refused after about the size limit, not after all of it.
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import psutil
from PIL import Image

from visual_search.datasets import SOP_ROOT
from visual_search.serve import MAX_UPLOAD_BYTES

QUERIES = 100
CHUNK_BYTES = 1024 * 1024
CHUNKS_OFFERED = 400  # 400 MB, far above the limit


def main() -> None:
    args = parse_args()
    items = json.loads((args.gallery_dir / "items.json").read_text())
    command = [
        sys.executable,
        "-m",
        "visual_search.serve",
        "--onnx",
        str(args.onnx),
        "--gallery-dir",
        str(args.gallery_dir),
        "--port",
        str(args.port),
    ]
    # The server's output goes to a file: a pipe that nobody reads fills up and blocks it.
    with tempfile.TemporaryFile("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=120) as client:
                wait_until_healthy(client, server)
                results = {
                    "gallery_size": client.get("/health").json()["gallery_size"],
                    "queries": check_queries(client, items, args.data_dir / SOP_ROOT),
                    "thin_images": check_thin_images(client, psutil.Process(server.pid)),
                    "largest_accepted_image": check_largest_accepted_image(
                        client, psutil.Process(server.pid)
                    ),
                    "chunked_upload": check_chunked_upload(client),
                }
        finally:
            server.terminate()
            server.wait(timeout=30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def check_queries(client, items: dict, root: Path) -> dict:
    images, classes = items["images"], items["class_ids"]
    step = len(images) // QUERIES
    self_at_rank_one = same_product_at_rank_two = 0
    for index in range(0, step * QUERIES, step):
        data = (root / images[index]).read_bytes()
        response = client.post("/search", files={"file": ("q.jpg", data)}, params={"k": 5})
        matches = response.json()["matches"]
        self_at_rank_one += matches[0]["image"] == images[index]
        same_product_at_rank_two += matches[1]["class_id"] == classes[index]
    return {
        "queries": QUERIES,
        "query_is_rank_1": self_at_rank_one,
        "rank_2_same_product": same_product_at_rank_two,
    }


def png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("1", size).save(buffer, "PNG")
    return buffer.getvalue()


def peak_memory_bytes(server_process: psutil.Process) -> int:
    """Peak memory of the server. On Windows the process we start is a small launcher for the
    real server, so take the largest of it and its children."""
    processes = [server_process, *server_process.children(recursive=True)]
    peaks = []
    for process in processes:
        info = process.memory_info()
        peaks.append(getattr(info, "peak_wset", info.rss))
    return max(peaks)


def check_thin_images(client, server_process: psutil.Process) -> list[dict]:
    """Each of these used to cost gigabytes of memory, or a 500, before the aspect limit."""
    outcomes = []
    # The last two are within the 64 million pixel limit but are as thin as an image can be,
    # so decoding them is the expensive part.
    for size in (
        (6000, 1),
        (20_000, 1),
        (1, 20_000),
        (10_000_000, 1),
        (1, 64_000_000),
        (64_000_000, 1),
    ):
        data = png_bytes(size)
        started = time.perf_counter()
        response = client.post("/search", files={"file": ("q.png", data)})
        peak = peak_memory_bytes(server_process)
        outcomes.append(
            {
                "size": list(size),
                "upload_bytes": len(data),
                "status": response.status_code,
                "seconds": round(time.perf_counter() - started, 2),
                "server_peak_memory_mb": round(peak / 1e6),
            }
        )
    return outcomes


def check_largest_accepted_image(client, server_process: psutil.Process) -> dict:
    """An 8000 x 8000 RGB image is the biggest square one inside the 64 million pixel limit."""
    buffer = io.BytesIO()
    Image.new("RGB", (8000, 8000)).save(buffer, "PNG")  # one colour, so the file is small
    data = buffer.getvalue()
    before = peak_memory_bytes(server_process)
    started = time.perf_counter()
    response = client.post("/search", files={"file": ("big.png", data)})
    return {
        "size": [8000, 8000],
        "upload_bytes": len(data),
        "status": response.status_code,
        "seconds": round(time.perf_counter() - started, 2),
        "server_peak_memory_mb_before": round(before / 1e6),
        "server_peak_memory_mb_after": round(peak_memory_bytes(server_process) / 1e6),
    }


def check_chunked_upload(client) -> dict:
    """Send far more than the limit as a valid multipart body with no Content-Length.

    The body starts like a real file upload and never ends, so the multipart parser keeps
    reading until the size limit stops it. The status says whether the limit applied. How
    much the server read before stopping is covered by a unit test, because the client
    cannot see it.
    """
    start = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
    )

    def body():
        yield start
        for _ in range(CHUNKS_OFFERED):
            yield b"0" * CHUNK_BYTES

    started = time.perf_counter()
    try:
        response = client.post(
            "/search", content=body(), headers={"content-type": "multipart/form-data; boundary=x"}
        )
        status = response.status_code
    except httpx.TransportError as error:  # the server may close the connection once it has refused
        status = type(error).__name__
    return {
        "limit_mb": round(MAX_UPLOAD_BYTES / 1e6, 1),
        "offered_mb": CHUNKS_OFFERED,
        "status": status,
        "seconds": round(time.perf_counter() - started, 2),
    }


def wait_until_healthy(client, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("the server exited before it became healthy")
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:  # nothing is listening yet
            time.sleep(0.5)
    raise TimeoutError("the server did not become healthy in time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
