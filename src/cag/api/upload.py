"""
FastAPI endpoints for upload, ingestion, and querying.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from cag.config import settings
from cag.graph.graph import run_query
from cag.ingestion.chunker import chunk_documents
from cag.ingestion.embedder import get_embeddings, get_vector_store, upsert_chunks
from cag.ingestion.loader import load_documents

logger = logging.getLogger(__name__)

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_raw_dir() -> Path:
    raw_dir = Path.cwd() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


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
):
    """Save uploaded files to `data/raw/` and optionally trigger ingestion."""

    raw_dir = _ensure_raw_dir()
    saved_files = []

    for upload in files:
        destination = raw_dir / upload.filename
        with destination.open("wb") as output:
            content = await upload.read()
            output.write(content)
        saved_files.append(str(destination))
        logger.info("Saved uploaded file: %s", destination)

    if ingest:
        background.add_task(_ingest_dir, str(raw_dir))

    return {"status": "ok", "saved": saved_files, "ingest_started": ingest}


class QueryRequest(BaseModel):
    query: str
    conversation_history: list[Any] | None = None
    relevance_threshold: float | None = None
    confidence_threshold: float | None = None
    hallucination_threshold: float | None = None


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    """Query the CAG pipeline from the frontend."""

    try:
        if payload.relevance_threshold is not None:
            settings.relevance_threshold = payload.relevance_threshold
        if payload.confidence_threshold is not None:
            settings.confidence_threshold = payload.confidence_threshold
        if payload.hallucination_threshold is not None:
            settings.hallucination_threshold = payload.hallucination_threshold

        return run_query(
            query=payload.query,
            conversation_history=payload.conversation_history or [],
        )
    except Exception as exc:
        logger.exception("Query endpoint failed")
        return {"error": str(exc)}


@app.get("/")
async def root_frontend():
    """Serve the built frontend when available, otherwise redirect to the dev server."""

    dist_index = Path.cwd() / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index), media_type="text/html")

    frontend_url = os.environ.get("FRONTEND_URL") or os.environ.get("FRONTEND_PORT")
    if frontend_url and frontend_url.isdigit():
        url = f"http://localhost:{frontend_url}/"
    else:
        url = os.environ.get("FRONTEND_URL", "http://localhost:5174/")

    return RedirectResponse(url)
