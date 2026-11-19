"""HTTP API: upload a photo, get the most similar gallery products back.

Example:
    python -m visual_search.serve --onnx artifacts/onnx/clip_lora.onnx \
        --gallery-dir artifacts/gallery_test
    curl -F file=@photo.jpg "http://127.0.0.1:8000/search?k=5"

The server binds to 127.0.0.1 by default. Nothing here authenticates callers or
rate-limits them, so put it behind a proxy that does before exposing it. A proxy should
also close connections that stall: after a 413 the server leaves the connection open, and a
client that never finishes sending its body keeps it open. Closing the connection instead
would make a user with a slightly oversized photo see a connection error, not the 413.

A query that is itself a gallery image matches itself at rank 1 with a score near 1.
That says nothing about retrieval quality; see RESULTS.md for measured numbers.

Needs the optional extra: pip install -e .[serve]
"""

import argparse
import io
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from visual_search.preprocess import MAX_ASPECT_RATIO, UnsupportedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 64_000_000  # checked before decoding, so a small file cannot expand to gigabytes
DEFAULT_K = 10
MAX_K = 50
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form boundaries and headers around the file


class MatchResponse(BaseModel):
    rank: int
    score: float
    class_id: int
    image: str


class SearchResponse(BaseModel):
    matches: list[MatchResponse]


def decode_upload(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB image, or raise ValueError with a message for the caller."""
    try:
        image = Image.open(io.BytesIO(data))  # reads the header only, so the size is known here
        if image.width * image.height > MAX_PIXELS:
            raise ValueError(f"image is larger than {MAX_PIXELS:,} pixels")
        # Check the shape before decoding: a 1 x 64,000,000 image is within the pixel limit
        # but would still cost gigabytes to decode and copy, only to be refused later.
        short, long = sorted(image.size)
        if long > MAX_ASPECT_RATIO * short:
            raise ValueError(
                f"image is {image.width} x {image.height}; the long side may be at most "
                f"{MAX_ASPECT_RATIO} times the short side"
            )
        image.load()  # a truncated file only fails here, not in open()
        # Phone photos store rotation in EXIF metadata, so apply it before embedding.
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("the file is not a readable image") from error


class LimitBodySize:
    """Rejects a request body above `max_bytes` with a 413, without reading all of it.

    The check uses the declared Content-Length when there is one. A chunked upload has
    none, so the bytes are also counted as they arrive and the request is cut off at the
    limit. Without that, the framework would write the whole body to a temporary file
    before the endpoint could look at its size.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": "upload is too large"}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI turns any other exception raised while it reads the body into
                    # a 400 "error parsing the body", but passes an HTTPException through.
                    raise HTTPException(413, "upload is too large")
            return message

        await self.app(scope, counting_receive, send)


def create_app(searcher) -> FastAPI:
    """Build the API around a `Searcher`. Tests pass a fake with the same two members."""
    app = FastAPI(title="visual-search")
    app.add_middleware(LimitBodySize, max_bytes=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "gallery_size": searcher.gallery_size}

    # A plain `def`, not `async def`: FastAPI runs it in a worker thread, so the slow
    # model call does not block the event loop.
    @app.post("/search", response_model=SearchResponse)
    def search(
        file: Annotated[UploadFile, File(description="query photo")],
        k: Annotated[int, Query(ge=1, le=MAX_K)] = DEFAULT_K,
    ) -> SearchResponse:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "upload is too large")
        try:
            image = decode_upload(data)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        try:
            matches = searcher.search(image, k)
        except UnsupportedImageError as error:  # for example an image 20000 pixels wide and 1 high
            raise HTTPException(400, str(error)) from error
        return SearchResponse(matches=[MatchResponse(**vars(match)) for match in matches])

    return app


def main() -> None:
    args = parse_args()
    import uvicorn

    from visual_search.onnx_embedder import OnnxEmbedder
    from visual_search.search import Searcher

    searcher = Searcher.load(OnnxEmbedder(args.onnx), args.gallery_dir)
    uvicorn.run(create_app(searcher), host=args.host, port=args.port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="model from visual_search.export")
    parser.add_argument(
        "--gallery-dir", type=Path, required=True, help="folder from visual_search.build_index"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
