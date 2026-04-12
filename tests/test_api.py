from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cag.api.upload import app
from cag.config import settings


client = TestClient(app)


def test_query_requires_api_key_when_configured():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post("/query", json={"query": "Hello"})
        assert response.status_code == 401
    finally:
        settings.cag_api_key = original_key


def test_query_rejects_invalid_api_key():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post("/query", headers={"X-API-Key": "wrong"}, json={"query": "Hello"})
        assert response.status_code == 403
    finally:
        settings.cag_api_key = original_key


def test_query_accepts_valid_api_key_and_does_not_mutate_settings():
    original_api_key = settings.cag_api_key
    original_relevance = settings.relevance_threshold
    original_confidence = settings.confidence_threshold
    original_hallucination = settings.hallucination_threshold
    settings.cag_api_key = "secret-key"
    try:
        with patch("cag.api.upload.run_query") as mock_run_query:
            mock_run_query.return_value = {"answer": "ok", "fallback_used": False, "fallback_reason": ""}
            response = client.post(
                "/query",
                headers={"X-API-Key": "secret-key"},
                json={
                    "query": "Hello",
                    "relevance_threshold": 0.8,
                    "confidence_threshold": 0.7,
                    "hallucination_threshold": 0.2,
                },
            )

        assert response.status_code == 200
        kwargs = mock_run_query.call_args.kwargs
        runtime_config = kwargs["runtime_config"]
        assert runtime_config.relevance_threshold == 0.8
        assert runtime_config.confidence_threshold == 0.7
        assert runtime_config.hallucination_threshold == 0.2
        assert settings.relevance_threshold == original_relevance
        assert settings.confidence_threshold == original_confidence
        assert settings.hallucination_threshold == original_hallucination
    finally:
        settings.cag_api_key = original_api_key


def test_query_rejects_out_of_range_thresholds():
    response = client.post("/query", json={"query": "Hello", "relevance_threshold": 1.2})
    assert response.status_code == 422


def test_upload_requires_api_key_when_configured():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            files={"files": ("demo.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 401
    finally:
        settings.cag_api_key = original_key


def test_upload_rejects_invalid_extension():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            headers={"X-API-Key": "secret-key"},
            files={"files": ("malware.exe", BytesIO(b"boom"), "application/octet-stream")},
        )
        assert response.status_code == 400
    finally:
        settings.cag_api_key = original_key


def test_upload_rejects_large_file():
    original_key = settings.cag_api_key
    settings.cag_api_key = "secret-key"
    try:
        response = client.post(
            "/upload",
            headers={"X-API-Key": "secret-key"},
            files={"files": ("large.txt", BytesIO(b"a" * (10 * 1024 * 1024 + 1)), "text/plain")},
        )
        assert response.status_code == 413
    finally:
        settings.cag_api_key = original_key


def test_root_frontend_serves_built_index_and_assets():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    dist_assets = Path.cwd() / "frontend" / "dist" / "assets"
    asset_name = next(path.name for path in dist_assets.iterdir() if path.is_file())
    asset_response = client.get(f"/assets/{asset_name}")

    assert asset_response.status_code == 200
