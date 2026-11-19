import asyncio
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")  # the serving extra is optional
pytest.importorskip("httpx")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from visual_search import serve
from visual_search.preprocess import UnsupportedImageError
from visual_search.search import Match


class FakeSearcher:
    gallery_size = 3

    def __init__(self):
        self.queries = []

    def search(self, image, k):
        self.queries.append((image, k))
        return [
            Match(rank=i + 1, score=1.0 - 0.1 * i, class_id=i, image=f"cat/{i}.JPG")
            for i in range(min(k, 3))
        ]


@pytest.fixture
def searcher():
    return FakeSearcher()


@pytest.fixture
def client(searcher):
    return TestClient(serve.create_app(searcher))


def image_bytes(size=(40, 20), mode="RGB", format="JPEG", **save_options) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format, **save_options)
    return buffer.getvalue()


def post(client, data: bytes, **params):
    return client.post("/search", files={"file": ("photo.jpg", data)}, params=params)


def test_health_reports_the_gallery_size(client):
    assert client.get("/health").json() == {"status": "ok", "gallery_size": 3}


def test_search_returns_the_matches(client, searcher):
    response = post(client, image_bytes(), k=2)

    assert response.status_code == 200
    assert response.json() == {
        "matches": [
            {"rank": 1, "score": 1.0, "class_id": 0, "image": "cat/0.JPG"},
            {"rank": 2, "score": 0.9, "class_id": 1, "image": "cat/1.JPG"},
        ]
    }
    assert searcher.queries[0][1] == 2


def test_k_defaults_to_ten(client, searcher):
    post(client, image_bytes())

    assert searcher.queries[0][1] == serve.DEFAULT_K


@pytest.mark.parametrize("k", [0, -1, serve.MAX_K + 1, "many"])
def test_k_outside_the_allowed_range_is_rejected(client, k):
    assert post(client, image_bytes(), k=k).status_code == 422


def test_a_request_without_a_file_is_rejected(client):
    assert client.post("/search").status_code == 422


def test_a_file_that_is_not_an_image_is_rejected(client):
    response = post(client, b"this is not an image")

    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_a_truncated_image_is_rejected(client):
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, "JPEG")
    data = buffer.getvalue()

    assert post(client, data[: len(data) // 2]).status_code == 400


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_images_that_are_not_rgb_are_accepted(client, searcher, mode):
    response = post(client, image_bytes(mode=mode, format="PNG"))

    assert response.status_code == 200
    assert searcher.queries[0][0].mode == "RGB"


def test_exif_rotation_is_applied_before_searching(client, searcher):
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"

    post(client, image_bytes(size=(40, 20), exif=exif))

    assert searcher.queries[0][0].size == (20, 40)


def test_an_upload_far_over_the_limit_is_rejected_from_its_declared_length(client, searcher):
    response = post(client, b"0" * (serve.MAX_UPLOAD_BYTES + serve.MULTIPART_OVERHEAD_BYTES + 1))

    assert response.status_code == 413
    assert searcher.queries == []


def test_the_body_limit_stops_reading_as_soon_as_it_is_exceeded():
    # Feed 1 MB chunks forever to an app that reads them all, and count what was pulled.
    chunks_pulled = 0

    async def receive():
        nonlocal chunks_pulled
        chunks_pulled += 1
        return {"type": "http.request", "body": b"0" * 1024 * 1024, "more_body": True}

    async def app_that_reads_everything(scope, receive, send):
        while True:
            await receive()

    limiter = serve.LimitBodySize(app_that_reads_everything, max_bytes=5 * 1024 * 1024)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(limiter({"type": "http", "headers": []}, receive, None))

    assert refused.value.status_code == 413
    assert chunks_pulled == 6  # five chunks fit; the sixth goes over and stops the read


def test_a_chunked_upload_without_a_length_is_rejected(client, searcher):
    # A generator body is sent without a Content-Length, so only the byte count can catch it.
    # The test client hands the app the whole body, so it cannot show that the server stops
    # reading early; that was checked against a running server.
    chunk = b"0" * (1024 * 1024)

    response = client.post(
        "/search",
        content=(chunk for _ in range(30)),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )

    assert response.status_code == 413
    assert searcher.queries == []


def test_an_image_the_searcher_cannot_handle_is_a_400_not_a_500(client, searcher):
    def refuse(image, k):
        raise UnsupportedImageError("image is far too thin")

    searcher.search = refuse

    response = post(client, image_bytes())

    assert response.status_code == 400
    assert "too thin" in response.json()["detail"]


def test_a_file_just_over_the_limit_is_rejected_by_the_endpoint(client, searcher):
    response = post(client, b"0" * (serve.MAX_UPLOAD_BYTES + 1000))

    assert response.status_code == 413
    assert searcher.queries == []


@pytest.mark.parametrize("size", [(1, 10_000), (10_000, 1), (3, 5_000)])
def test_a_thin_image_is_refused_before_it_is_decoded(client, searcher, monkeypatch, size):
    # An image can be inside the pixel limit and still need gigabytes to decode and resize.
    def fail(self, *args, **kwargs):
        raise AssertionError("the image was decoded")

    monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.load", fail)

    response = post(client, image_bytes(size=size, mode="1", format="PNG"))

    assert response.status_code == 400
    assert "times the short side" in response.json()["detail"]
    assert searcher.queries == []


def test_an_image_with_too_many_pixels_is_rejected_before_decoding(client, monkeypatch):
    monkeypatch.setattr(serve, "MAX_PIXELS", 100)

    response = post(client, image_bytes(size=(20, 20)))

    assert response.status_code == 400
    assert "larger than" in response.json()["detail"]
