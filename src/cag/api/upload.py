"""
FastAPI endpoints for upload, ingestion, and querying.
"""
from __future__ import annotations

import logging
import os
from hmac import compare_digest
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from cag.config import settings
from cag.graph.graph import run_query
from cag.graph.runtime import RuntimeConfig
from cag.ingestion.chunker import chunk_documents
from cag.ingestion.embedder import get_embeddings, get_vector_store, upsert_chunks
from cag.ingestion.loader import SUPPORTED_EXTENSIONS, load_documents

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 25 * 1024 * 1024
HTTP_413_TOO_LARGE = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}
_warned_open_api = False

app = FastAPI(title="CAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
        "http://localhost:5178",
        "http://127.0.0.1:5178",
        "http://localhost:5179",
        "http://127.0.0.1:5179",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_raw_dir() -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _frontend_dist_dir() -> Path:
    return PROJECT_ROOT / "frontend" / "dist"


def _frontend_assets_dir() -> Path:
    return _frontend_dist_dir() / "assets"


def _require_api_key(x_api_key: str | None) -> None:
    global _warned_open_api

    configured_key = settings.cag_api_key.strip()
    if not configured_key:
        if not _warned_open_api:
            logger.warning("CAG_API_KEY is not configured; /upload and /query remain open for local preview.")
            _warned_open_api = True
        return

    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header.")
    if not compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")


def _sanitize_upload_name(filename: str | None) -> str:
    sanitized = Path(filename or "").name.strip()
    if not sanitized or sanitized in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload filename.")
    return sanitized


def _validate_upload(upload: UploadFile, content: bytes) -> str:
    filename = _sanitize_upload_name(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {extension or '[none]'}.",
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=f"File exceeds max size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
        )

    content_type = (upload.content_type or "").strip().lower()
    allowed_types = ALLOWED_CONTENT_TYPES.get(extension, {"application/octet-stream"})
    if content_type and content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unexpected content type {content_type!r} for {extension} file.",
        )

    return filename


def _ingest_dir(data_dir: str | Path) -> None:
    try:
        logger.info("Starting ingestion for %s", data_dir)
        documents = load_documents(data_dir)
        if not documents:
            logger.warning("No documents found in %s", data_dir)
            return

        chunks = chunk_documents(documents)
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)
        upsert_chunks(chunks, vector_store)
        logger.info("Ingestion completed: %s chunks indexed", len(chunks))
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)


@app.post("/upload")
async def upload_files(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    ingest: bool = True,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Save uploaded files to `data/raw/` and optionally trigger ingestion."""

    _require_api_key(x_api_key)
    raw_dir = _ensure_raw_dir()
    saved_files = []
    total_bytes = 0

    for upload in files:
        content = await upload.read()
        total_bytes += len(content)
        if total_bytes > MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=HTTP_413_TOO_LARGE,
                detail=f"Request exceeds max total upload size of {MAX_REQUEST_BYTES // (1024 * 1024)} MiB.",
            )
        filename = _validate_upload(upload, content)
        destination = raw_dir / filename
        with destination.open("wb") as output:
            output.write(content)
        saved_files.append(str(destination))
        logger.info("Saved uploaded file: %s", destination)

    if ingest:
        background.add_task(_ingest_dir, str(raw_dir))

    return {"status": "ok", "saved": saved_files, "ingest_started": ingest}


@app.get("/files")
async def list_files(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Return the list of documents currently present in data/raw/."""
    _require_api_key(x_api_key)
    raw_dir = _ensure_raw_dir()
    files = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            })
    return {"files": files, "total": len(files)}


@app.delete("/files/{filename}")
async def delete_file(
    filename: str,
    background: BackgroundTasks,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Delete a document from data/raw/ and re-index the remaining files."""
    _require_api_key(x_api_key)

    # Sanitize and validate
    safe_name = _sanitize_upload_name(filename)
    raw_dir = _ensure_raw_dir()
    target = (raw_dir / safe_name).resolve()

    # Path traversal guard
    if raw_dir.resolve() not in target.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{safe_name}' not found.")

    target.unlink()
    logger.info("Deleted file: %s", target)

    # Re-index remaining files in background (reset + re-ingest)
    background.add_task(_reindex_after_delete, str(raw_dir))

    return {"status": "ok", "deleted": safe_name, "reindex_started": True}


def _reindex_after_delete(data_dir: str | Path) -> None:
    """Reset the vector store and re-ingest all remaining documents."""
    try:
        from cag.ingestion.chunker import chunk_documents
        from cag.ingestion.loader import load_documents

        logger.info("Re-indexing after deletion: %s", data_dir)
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)

        # Reset Chroma collection to avoid stale vectors
        if hasattr(vector_store, "delete_collection"):
            vector_store.delete_collection()
            vector_store = get_vector_store(embeddings)

        documents = load_documents(data_dir)
        if not documents:
            logger.info("No documents remaining after deletion — vector store cleared.")
            return

        chunks = chunk_documents(documents)
        upsert_chunks(chunks, vector_store)
        logger.info("Re-indexing completed: %s chunks indexed.", len(chunks))
    except Exception as exc:
        logger.exception("Re-indexing after deletion failed: %s", exc)



class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    conversation_history: list[Any] | None = None
    relevance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@app.post("/query")
async def query_endpoint(payload: QueryRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Query the CAG pipeline from the frontend."""

    _require_api_key(x_api_key)
    try:
        runtime_config = RuntimeConfig(
            relevance_threshold=(
                payload.relevance_threshold
                if payload.relevance_threshold is not None
                else settings.relevance_threshold
            ),
            confidence_threshold=(
                payload.confidence_threshold
                if payload.confidence_threshold is not None
                else settings.confidence_threshold
            ),
            hallucination_threshold=(
                payload.hallucination_threshold
                if payload.hallucination_threshold is not None
                else settings.hallucination_threshold
            ),
        )
        return run_query(
            query=payload.query,
            conversation_history=payload.conversation_history or [],
            runtime_config=runtime_config,
        )
    except Exception as exc:
        logger.exception("Query endpoint failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.get("/")
async def root_frontend():
    """Serve the built frontend when available, otherwise redirect to the dev server."""

    dist_index = _frontend_dist_dir() / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index), media_type="text/html")

    frontend_url = os.environ.get("FRONTEND_URL") or os.environ.get("FRONTEND_PORT")
    if frontend_url and frontend_url.isdigit():
        url = f"http://localhost:{frontend_url}/"
    else:
        url = os.environ.get("FRONTEND_URL", "http://localhost:5174/")

    return RedirectResponse(url)


@app.get("/assets/{asset_path:path}")
async def frontend_asset(asset_path: str):
    """Serve built frontend assets when the production bundle is available."""

    asset_file = (_frontend_assets_dir() / asset_path).resolve()
    assets_dir = _frontend_assets_dir().resolve()
    if assets_dir not in asset_file.parents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    if not asset_file.exists() or not asset_file.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    return FileResponse(str(asset_file))
