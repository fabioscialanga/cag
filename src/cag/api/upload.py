"""
FastAPI endpoints for upload, ingestion, and querying.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from hmac import compare_digest
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cag.config import settings
from cag.graph.graph import run_query
from cag.graph.nodes import entry_node, retrieve_node, select_context_node
from cag.graph.runtime import RuntimeConfig
from cag.ingestion.chunker import chunk_documents
from cag.ingestion.embedder import get_embeddings, get_vector_store, upsert_chunks
from cag.ingestion.loader import SUPPORTED_EXTENSIONS, load_documents
from cag.knowledge.compiler import compile_chunks
from cag.knowledge.document_map import list_document_profiles
from cag.knowledge.store import connect, initialize, list_knowledge_graph

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
INGEST_STEPS = [
    ("load", "Load files"),
    ("chunk", "Split into chunks"),
    ("compile", "Compile knowledge"),
    ("embed", "Index vectors"),
]


def _count_supported_files(data_dir: str | Path) -> int:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        return 0
    return sum(1 for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _build_ingest_steps(stage: str = "idle", failed_stage: str = "", skipped_stage: str = "") -> list[dict[str, str]]:
    order = [step_id for step_id, _label in INGEST_STEPS]
    active_index = order.index(stage) if stage in order else -1
    failed_index = order.index(failed_stage) if failed_stage in order else -1
    skipped_index = order.index(skipped_stage) if skipped_stage in order else -1
    steps: list[dict[str, str]] = []
    for index, (step_id, label) in enumerate(INGEST_STEPS):
        step_status = "pending"
        if failed_index == index:
            step_status = "failed"
        elif skipped_index == index:
            step_status = "skipped"
        elif stage == "ready" or (active_index >= 0 and index < active_index):
            step_status = "done"
        elif active_index == index:
            step_status = "running"
        steps.append({"id": step_id, "label": label, "status": step_status})
    return steps


def _empty_ingest_status() -> dict[str, Any]:
    return {
        "status": "idle",
        "stage": "idle",
        "message": "",
        "chunks_indexed": 0,
        "files_total": 0,
        "documents_loaded": 0,
        "chunks_created": 0,
        "claims_created": 0,
        "profiles_created": 0,
        "vectors_indexed": 0,
        "progress": 0.0,
        "active_file": "",
        "steps": _build_ingest_steps(),
    }


_ingest_status = _empty_ingest_status()


def _set_ingest_status(
    *,
    status: str,
    stage: str,
    message: str,
    chunks_indexed: int | None = None,
    files_total: int | None = None,
    documents_loaded: int | None = None,
    chunks_created: int | None = None,
    claims_created: int | None = None,
    profiles_created: int | None = None,
    vectors_indexed: int | None = None,
    progress: float | None = None,
    active_file: str | None = None,
    failed_stage: str = "",
    skipped_stage: str = "",
) -> None:
    update: dict[str, Any] = {
        "status": status,
        "stage": stage,
        "message": message,
        "steps": _build_ingest_steps(stage, failed_stage=failed_stage, skipped_stage=skipped_stage),
    }
    optional_values = {
        "chunks_indexed": chunks_indexed,
        "files_total": files_total,
        "documents_loaded": documents_loaded,
        "chunks_created": chunks_created,
        "claims_created": claims_created,
        "profiles_created": profiles_created,
        "vectors_indexed": vectors_indexed,
        "progress": max(0.0, min(1.0, progress)) if progress is not None else None,
        "active_file": active_file,
    }
    update.update({key: value for key, value in optional_values.items() if value is not None})
    _ingest_status.update(update)

_CONVERSATION_TRANSFORM_RE = re.compile(
    r"\b("
    r"traduc\w*|tradurre|translate|translation|"
    r"riassum\w*|sintetizz\w*|summari[sz]\w*|shorten|brief|"
    r"riformul\w*|riscriv\w*|rewrite|rephrase|"
    r"spieg\w*|explain|semplific\w*|simplif\w*|"
    r"bullet|punti|elenco|list|"
    r"approfond\w*|expand|more detail|"
    r"in italiano|italian|inglese|english|francese|french|spagnolo|spanish|"
    r"tedesco|german|portoghese|portuguese"
    r")\b",
    re.IGNORECASE,
)
_ANAPHORA_RE = re.compile(
    r"\b("
    r"\w+lo|\w+la|"
    r"lo|la|questo|questa|quello|quella|risposta|messaggio|sopra|precedente|"
    r"it|that|this|answer|message|previous|above"
    r")\b",
    re.IGNORECASE,
)
_SHORT_CONVERSATION_EDIT_RE = re.compile(
    r"^\s*(?:"
    r"spieg\w*(?:\s+meglio)?|explain(?:\s+better)?|"
    r"semplific\w*|simplif\w*|"
    r"riassum\w*|sintetizz\w*|summari[sz]\w*|"
    r"riscriv\w*|riformul\w*|rewrite|rephrase|"
    r"bullet|punti|elenco|list"
    r")(?:\s+(?:meglio|di\s+piu|di\s+piu'|more|please|per\s+favore|grazie))*\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_ELABORATION_FOLLOWUP_RE = re.compile(
    r"\b(approfond\w*|dettagli\w*|piu dettagli|piu' dettagli|piu info|more detail|tell me more|go deeper)\b",
    re.IGNORECASE,
)
_TARGET_LANGUAGES = [
    ("it", "Italian", re.compile(r"\b(italiano|italian)\b", re.IGNORECASE)),
    ("en", "English", re.compile(r"\b(inglese|english)\b", re.IGNORECASE)),
    ("fr", "French", re.compile(r"\b(francese|french)\b", re.IGNORECASE)),
    ("es", "Spanish", re.compile(r"\b(spagnolo|spanish)\b", re.IGNORECASE)),
    ("de", "German", re.compile(r"\b(tedesco|german)\b", re.IGNORECASE)),
    ("pt", "Portuguese", re.compile(r"\b(portoghese|portuguese)\b", re.IGNORECASE)),
]
_BAD_MODEL_TEXT_RE = re.compile(
    r"^\s*(connection error|timeout|timed out|error|model unavailable|service unavailable|none|null)\.?\s*$",
    re.IGNORECASE,
)

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


class UploadResponse(BaseModel):
    status: str
    saved: list[str]
    ingest_started: bool


class FileInfo(BaseModel):
    name: str
    size_bytes: int
    modified: float


class FilesResponse(BaseModel):
    files: list[FileInfo]
    total: int


class DocumentProfileInfo(BaseModel):
    profile_id: str
    source_version_id: str
    filename: str
    source: str
    version: int
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    covered_intents: list[str] = Field(default_factory=list)
    status: str = "active"
    generator: str = "local_fallback"
    created_at: str = ""
    chunk_count: int = 0


class DocumentProfilesResponse(BaseModel):
    profiles: list[DocumentProfileInfo]
    total: int


class KnowledgeGraphNode(BaseModel):
    id: str
    type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    confidence: float = 0.0
    evidence_chunk_id: str = ""


class KnowledgeGraphResponse(BaseModel):
    nodes: list[KnowledgeGraphNode]
    edges: list[KnowledgeGraphEdge]
    total_nodes: int
    total_edges: int


class DeleteFileResponse(BaseModel):
    status: str
    deleted: str
    reindex_started: bool


class IngestStepInfo(BaseModel):
    id: str
    label: str
    status: str


class IngestStatusResponse(BaseModel):
    status: str
    stage: str = "idle"
    message: str = ""
    chunks_indexed: int = 0
    files_total: int = 0
    documents_loaded: int = 0
    chunks_created: int = 0
    claims_created: int = 0
    profiles_created: int = 0
    vectors_indexed: int = 0
    progress: float = 0.0
    active_file: str = ""
    steps: list[IngestStepInfo] = Field(default_factory=list)


class DemoCorpusResetResponse(BaseModel):
    status: str
    copied: list[str]
    ingest_started: bool


class ResetAllResponse(BaseModel):
    status: str
    deleted_files: list[str]
    knowledge_deleted: bool
    vector_reset: bool


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    confidence: float = 0.0
    citations: list[Any] = Field(default_factory=list)
    query_type: str = "GENERAL"
    document_candidates: list[Any] = Field(default_factory=list)
    ranked_chunks: list[Any] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    hallucination_risk: float = 0.0
    should_escalate: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    original_query: str = ""
    modified_query: str = ""
    intent: dict[str, Any] = Field(default_factory=dict)
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    grounding_checks: list[Any] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    post_grounding_status: str = ""
    suggested_actions: list[Any] = Field(default_factory=list)
    node_trace: list[str] = Field(default_factory=list)


class RetrievalDiagnosticsResponse(BaseModel):
    query: str
    modified_query: str = ""
    intent: dict[str, Any] = Field(default_factory=dict)
    query_type: str
    retrieval_strategy: str
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    document_candidates: list[Any] = Field(default_factory=list)
    chunks: list[Any] = Field(default_factory=list)
    ranked_chunks: list[Any] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    node_trace: list[str] = Field(default_factory=list)


def _ensure_raw_dir() -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _frontend_dist_dir() -> Path:
    return PROJECT_ROOT / "frontend" / "dist"


def _benchmark_corpus_dir() -> Path:
    return PROJECT_ROOT / "data" / "benchmark_corpus"


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
    files_total = _count_supported_files(data_dir)
    _set_ingest_status(
        status="running",
        stage="load",
        message=f"Reading {files_total} supported file(s) from {data_dir}.",
        chunks_indexed=0,
        files_total=files_total,
        documents_loaded=0,
        chunks_created=0,
        claims_created=0,
        profiles_created=0,
        vectors_indexed=0,
        progress=0.12,
    )
    try:
        logger.info("Starting ingestion for %s", data_dir)
        documents = load_documents(data_dir)
        if not documents:
            logger.warning("No documents found in %s", data_dir)
            _set_ingest_status(
                status="failed",
                stage="load",
                message=f"No documents found in {data_dir}.",
                chunks_indexed=0,
                files_total=files_total,
                documents_loaded=0,
                progress=0.0,
                failed_stage="load",
            )
            return

        _set_ingest_status(
            status="running",
            stage="chunk",
            message=f"Loaded {len(documents)} document(s). Splitting content into semantic chunks.",
            files_total=files_total,
            documents_loaded=len(documents),
            progress=0.34,
        )
        chunks = chunk_documents(documents)
        _set_ingest_status(
            status="running",
            stage="compile",
            message=f"Created {len(chunks)} chunk(s). Extracting claims, topics, and document profiles.",
            files_total=files_total,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            progress=0.58,
        )
        compile_summary = compile_chunks(chunks, settings.knowledge_db_path)
        logger.info("Document Map compilation completed: %s", compile_summary)
        claims_created = int(compile_summary.get("claims", 0))
        profiles_created = int(compile_summary.get("document_profiles", 0))
        _set_ingest_status(
            status="running",
            stage="embed",
            message=f"Compiled {claims_created} claim(s) and {profiles_created} profile(s). Indexing vectors.",
            files_total=files_total,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            claims_created=claims_created,
            profiles_created=profiles_created,
            progress=0.78,
        )
        try:
            embeddings = get_embeddings()
            vector_store = get_vector_store(embeddings)
            upsert_chunks(chunks, vector_store)
            message = "Ingestion completed."
            vectors_indexed = len(chunks)
            skipped_stage = ""
        except Exception as exc:
            message = "Knowledge graph compiled; vector embeddings unavailable, using lexical fallback."
            vectors_indexed = 0
            skipped_stage = "embed"
            logger.warning("%s Error: %s", message, exc)
        _set_ingest_status(
            status="ready",
            stage="ready",
            message=message,
            chunks_indexed=len(chunks),
            files_total=files_total,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            claims_created=claims_created,
            profiles_created=profiles_created,
            vectors_indexed=vectors_indexed,
            progress=1.0,
            skipped_stage=skipped_stage,
        )
        logger.info("Ingestion completed: %s chunks processed", len(chunks))
    except Exception as exc:
        failed_stage = str(_ingest_status.get("stage") or "load")
        _set_ingest_status(
            status="failed",
            stage=failed_stage,
            message=str(exc),
            chunks_indexed=0,
            files_total=files_total,
            progress=float(_ingest_status.get("progress") or 0.0),
            failed_stage=failed_stage,
        )
        logger.exception("Ingestion failed: %s", exc)


def _reset_raw_to_demo_corpus(raw_dir: Path, demo_dir: Path) -> list[str]:
    if not demo_dir.exists() or not demo_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Demo corpus directory not found.")

    raw_root = raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in raw_dir.iterdir():
        target = path.resolve()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and raw_root in target.parents:
            path.unlink()

    copied: list[str] = []
    for source in sorted(demo_dir.iterdir()):
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
            destination = raw_dir / source.name
            shutil.copy2(source, destination)
            copied.append(source.name)

    if not copied:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Demo corpus has no supported files.")
    return copied


def _clear_raw_documents(raw_dir: Path) -> list[str]:
    raw_root = raw_dir.resolve()
    deleted: list[str] = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_dir.iterdir()):
        target = path.resolve()
        if path.is_file() and path.name != ".gitkeep" and raw_root in target.parents:
            deleted.append(path.name)
            path.unlink()
    return deleted


def _reset_vector_store() -> bool:
    try:
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)
        if hasattr(vector_store, "delete_collection"):
            vector_store.delete_collection()
            return True
    except Exception as exc:
        logger.warning("Vector store reset failed: %s", exc)
        return False
    return False


@app.post("/upload", response_model=UploadResponse)
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
        saved_files.append(filename)
        logger.info("Saved uploaded file: %s", destination)

    if ingest:
        _set_ingest_status(
            status="queued",
            stage="load",
            message=f"Queued {len(saved_files)} uploaded file(s) for ingestion.",
            chunks_indexed=0,
            files_total=_count_supported_files(raw_dir),
            documents_loaded=0,
            chunks_created=0,
            claims_created=0,
            profiles_created=0,
            vectors_indexed=0,
            progress=0.04,
        )
        background.add_task(_ingest_dir, str(raw_dir))

    return {"status": "ok", "saved": saved_files, "ingest_started": ingest}


@app.get("/ingest/status", response_model=IngestStatusResponse)
async def ingest_status(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Return the latest local ingestion status."""

    _require_api_key(x_api_key)
    return _ingest_status


@app.post("/demo/reset", response_model=DemoCorpusResetResponse)
async def reset_demo_corpus(
    background: BackgroundTasks,
    ingest: bool = True,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Replace data/raw/ with the bundled benchmark corpus and optionally trigger ingestion."""

    _require_api_key(x_api_key)
    raw_dir = _ensure_raw_dir()
    copied = _reset_raw_to_demo_corpus(raw_dir, _benchmark_corpus_dir())

    if ingest:
        _set_ingest_status(
            status="queued",
            stage="load",
            message=f"Queued demo corpus with {len(copied)} file(s) for ingestion.",
            chunks_indexed=0,
            files_total=_count_supported_files(raw_dir),
            documents_loaded=0,
            chunks_created=0,
            claims_created=0,
            profiles_created=0,
            vectors_indexed=0,
            progress=0.04,
        )
        background.add_task(_ingest_dir, str(raw_dir))

    return {"status": "ok", "copied": copied, "ingest_started": ingest}


@app.delete("/reset/all", response_model=ResetAllResponse)
async def reset_all(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Delete uploaded documents, compiled knowledge, and vector index data."""

    _require_api_key(x_api_key)
    raw_dir = _ensure_raw_dir()
    deleted_files = _clear_raw_documents(raw_dir)

    knowledge_path = Path(settings.knowledge_db_path)
    knowledge_deleted = False
    if knowledge_path.exists() and knowledge_path.is_file():
        knowledge_path.unlink()
        knowledge_deleted = True

    vector_reset = _reset_vector_store()
    _ingest_status.clear()
    _ingest_status.update(_empty_ingest_status())
    return {
        "status": "ok",
        "deleted_files": deleted_files,
        "knowledge_deleted": knowledge_deleted,
        "vector_reset": vector_reset,
    }


@app.get("/files", response_model=FilesResponse)
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


@app.get("/document-profiles", response_model=DocumentProfilesResponse)
async def document_profiles(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Return compiled document intelligence profiles."""

    _require_api_key(x_api_key)
    try:
        profiles = list_document_profiles(db_path=settings.knowledge_db_path)
    except Exception as exc:
        logger.warning("Document profiles unavailable; returning empty profile list: %s", exc)
        profiles = []
    return {"profiles": profiles, "total": len(profiles)}


@app.get("/knowledge-graph", response_model=KnowledgeGraphResponse)
async def knowledge_graph(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Return the compiled knowledge graph as nodes and typed edges."""

    _require_api_key(x_api_key)
    connection = connect(settings.knowledge_db_path)
    try:
        initialize(connection)
        graph = list_knowledge_graph(connection)
    finally:
        connection.close()
    return {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
    }


@app.delete("/files/{filename}", response_model=DeleteFileResponse)
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
    files_total = _count_supported_files(data_dir)
    _set_ingest_status(
        status="running",
        stage="load",
        message=f"Re-indexing {files_total} remaining file(s) after deletion.",
        chunks_indexed=0,
        files_total=files_total,
        documents_loaded=0,
        chunks_created=0,
        claims_created=0,
        profiles_created=0,
        vectors_indexed=0,
        progress=0.12,
    )
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
            _set_ingest_status(
                status="ready",
                stage="ready",
                message="No documents remaining; vector store cleared.",
                chunks_indexed=0,
                files_total=files_total,
                documents_loaded=0,
                chunks_created=0,
                claims_created=0,
                profiles_created=0,
                vectors_indexed=0,
                progress=1.0,
            )
            return

        _set_ingest_status(
            status="running",
            stage="chunk",
            message=f"Loaded {len(documents)} document(s). Rebuilding chunks.",
            files_total=files_total,
            documents_loaded=len(documents),
            progress=0.34,
        )
        chunks = chunk_documents(documents)
        _set_ingest_status(
            status="running",
            stage="compile",
            message=f"Created {len(chunks)} chunk(s). Recompiling knowledge.",
            files_total=files_total,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            progress=0.58,
        )
        compile_summary = compile_chunks(chunks, settings.knowledge_db_path)
        claims_created = int(compile_summary.get("claims", 0))
        profiles_created = int(compile_summary.get("document_profiles", 0))
        _set_ingest_status(
            status="running",
            stage="embed",
            message=f"Compiled {claims_created} claim(s). Re-indexing vectors.",
            files_total=files_total,
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            claims_created=claims_created,
            profiles_created=profiles_created,
            progress=0.78,
        )
        try:
            upsert_chunks(chunks, vector_store)
            _set_ingest_status(
                status="ready",
                stage="ready",
                message="Re-indexing completed.",
                chunks_indexed=len(chunks),
                files_total=files_total,
                documents_loaded=len(documents),
                chunks_created=len(chunks),
                claims_created=claims_created,
                profiles_created=profiles_created,
                vectors_indexed=len(chunks),
                progress=1.0,
            )
            logger.info("Re-indexing completed: %s chunks indexed.", len(chunks))
        except Exception as exc:
            logger.warning("Re-indexing compiled knowledge completed, but vector upsert failed: %s", exc)
            _set_ingest_status(
                status="ready",
                stage="ready",
                message="Knowledge recompiled; vector embeddings unavailable, using lexical fallback.",
                chunks_indexed=len(chunks),
                files_total=files_total,
                documents_loaded=len(documents),
                chunks_created=len(chunks),
                claims_created=claims_created,
                profiles_created=profiles_created,
                vectors_indexed=0,
                progress=1.0,
                skipped_stage="embed",
            )
    except Exception as exc:
        failed_stage = str(_ingest_status.get("stage") or "load")
        _set_ingest_status(
            status="failed",
            stage=failed_stage,
            message=str(exc),
            chunks_indexed=0,
            files_total=files_total,
            progress=float(_ingest_status.get("progress") or 0.0),
            failed_stage=failed_stage,
        )
        logger.exception("Re-indexing after deletion failed: %s", exc)



class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    conversation_history: list[Any] | None = None
    relevance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    access_filter: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        return query


class ConversationRoute(BaseModel):
    """Planner decision for a conversational turn before document retrieval."""

    action: str = Field(
        description="One of: answer_transform, rewrite_for_retrieval, direct_retrieval",
        pattern="^(answer_transform|rewrite_for_retrieval|direct_retrieval)$",
    )
    rewritten_query: str = Field(default="", description="Standalone query when action is rewrite_for_retrieval.")
    target_language: str = Field(default="", description="Optional target language name for answer transforms.")
    reason: str = Field(default="", description="Short reason for the route.")
    confidence: float = Field(default=0.0, description="Route confidence from 0 to 1.", ge=0.0, le=1.0)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    role = getattr(message, "role", "")
    if role:
        return str(role).lower()
    message_type = getattr(message, "type", "")
    if message_type:
        return "assistant" if message_type == "ai" else str(message_type).lower()
    return message.__class__.__name__.lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(getattr(message, "content", "")).strip()


def _latest_assistant_message(conversation_history: list[Any] | None) -> str:
    for message in reversed(conversation_history or []):
        role = _message_role(message)
        if role in {"assistant", "ai", "aimessage"}:
            content = _message_content(message)
            if content:
                return content
    return ""


def _latest_user_message(conversation_history: list[Any] | None) -> str:
    for message in reversed(conversation_history or []):
        role = _message_role(message)
        if role in {"user", "human", "humanmessage"}:
            content = _message_content(message)
            if content:
                return content
    return ""


def _recent_conversation_text(conversation_history: list[Any] | None, limit: int = 6) -> str:
    messages = list(conversation_history or [])[-limit:]
    lines: list[str] = []
    for message in messages:
        role = _message_role(message) or "message"
        content = _message_content(message)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _target_translation_language(query: str) -> tuple[str, str] | None:
    for code, name, pattern in _TARGET_LANGUAGES:
        if pattern.search(query):
            return code, name
    if re.search(r"\btraduc\w*\b", query, re.IGNORECASE):
        return "it", "Italian"
    return None


def _conversation_response_language(query: str, target_code: str = "") -> str:
    if target_code:
        return target_code
    normalized = " ".join(query.lower().split())
    if re.search(r"\b(spiegami|spiega|meglio|approfondisci|puoi|per favore|grazie|elenco|punti)\b", normalized):
        return "it"
    return "en"


def _is_conversation_transform_request(query: str, conversation_history: list[Any] | None) -> bool:
    if not _latest_assistant_message(conversation_history):
        return False
    if _ELABORATION_FOLLOWUP_RE.search(query):
        return False
    if not _CONVERSATION_TRANSFORM_RE.search(query):
        return False
    if _ANAPHORA_RE.search(query):
        return True
    return bool(_SHORT_CONVERSATION_EDIT_RE.search(query))


def _fallback_conversation_route(query: str, conversation_history: list[Any] | None) -> ConversationRoute:
    if _ELABORATION_FOLLOWUP_RE.search(query) and _latest_assistant_message(conversation_history):
        return ConversationRoute(
            action="rewrite_for_retrieval",
            rewritten_query=_fallback_rewrite_query(query, conversation_history),
            reason="Fallback detected a request to retrieve deeper supporting information.",
            confidence=0.70,
        )
    if _is_conversation_transform_request(query, conversation_history):
        target = _target_translation_language(query)
        return ConversationRoute(
            action="answer_transform",
            target_language=target[1] if target else "",
            reason="Fallback detected an answer transformation request.",
            confidence=0.55,
        )
    if _looks_like_contextual_followup(query, conversation_history):
        return ConversationRoute(
            action="rewrite_for_retrieval",
            rewritten_query=_fallback_rewrite_query(query, conversation_history),
            reason="Fallback detected a contextual follow-up.",
            confidence=0.65,
        )
    return ConversationRoute(action="direct_retrieval", reason="Fallback selected normal retrieval.", confidence=0.35)


def _route_conversation_turn(query: str, conversation_history: list[Any] | None) -> ConversationRoute:
    if not conversation_history:
        return ConversationRoute(action="direct_retrieval", reason="No conversation history.")

    deterministic_route = _fallback_conversation_route(query, conversation_history)
    if deterministic_route.action != "direct_retrieval" or not settings.enable_conversation_router_llm:
        return deterministic_route

    from agno.agent import Agent

    from cag.llm_factory import get_agno_model

    conversation = _recent_conversation_text(conversation_history)
    if not conversation:
        return ConversationRoute(action="direct_retrieval", reason="No usable conversation text.")

    agent = Agent(
        name="ConversationIntentRouter",
        model=get_agno_model(),
        role="Routes conversational turns before document retrieval.",
        instructions=[
            "Decide how to handle the latest user message in a document QA chat.",
            "Use answer_transform when the user asks to translate, summarize, rewrite, simplify, format, or otherwise restyle the previous assistant answer without needing new evidence.",
            "Use rewrite_for_retrieval when the user asks to go deeper, add substance, provide more details, or retrieve more useful information about the previous topic.",
            "Use rewrite_for_retrieval when the user asks a new factual follow-up that needs documents, but the message depends on recent context.",
            "Use direct_retrieval when the message is already a standalone document question or a new topic.",
            "For rewrite_for_retrieval, produce a standalone rewritten_query in the user's language that includes the topic from the conversation and asks for the useful missing details.",
            "For answer_transform, set target_language only when the user explicitly requests a language.",
            "Set confidence to how certain you are.",
            "Do not answer the user. Return structured output only.",
        ],
        structured_outputs=True,
        output_schema=ConversationRoute,
    )
    prompt = f"""RECENT CONVERSATION:
{conversation}

LATEST USER MESSAGE:
{query}

Return the route decision.
"""
    response = agent.run(prompt)
    route = response.content
    if isinstance(route, ConversationRoute):
        return route
    if isinstance(route, dict):
        return ConversationRoute(**route)
    return ConversationRoute.model_validate_json(str(route))


def _looks_like_contextual_followup(query: str, conversation_history: list[Any] | None) -> bool:
    if not conversation_history:
        return False
    normalized = " ".join(query.lower().split())
    if _ANAPHORA_RE.search(normalized):
        return True
    if len(normalized.split()) <= 8 and re.search(r"\b(e|and|also|anche|invece|why|perche|perchÃ©|come|how)\b", normalized):
        return True
    return False


def _transform_previous_answer(text: str, query: str, language_name: str | None = None) -> str:
    from agno.agent import Agent

    from cag.llm_factory import get_agno_model

    language_rule = (
        f"Use {language_name}."
        if language_name
        else "Use the language requested by the user; otherwise preserve the user's language."
    )
    agent = Agent(
        name="ConversationTransformAgent",
        model=get_agno_model(),
        instructions=[
            "Transform the previous assistant answer according to the user's request.",
            language_rule,
            "Preserve meaning, names, product names, and technical terms.",
            "Do not add new factual claims that are not present in the previous answer.",
            "Do not ask for document evidence and do not recommend escalation.",
            "Return only the transformed answer.",
        ],
    )
    prompt = f"""USER TRANSFORM REQUEST:
{query}

PREVIOUS ASSISTANT ANSWER:
{text}
"""
    response = agent.run(prompt)
    transformed = response.content if isinstance(response.content, str) else str(response.content)
    return transformed.strip()


def _rewrite_query_with_history(query: str, conversation_history: list[Any] | None) -> str:
    from agno.agent import Agent

    from cag.llm_factory import get_agno_model

    conversation = _recent_conversation_text(conversation_history)
    if not conversation:
        return query

    agent = Agent(
        name="ConversationRouterAgent",
        model=get_agno_model(),
        instructions=[
            "Rewrite the user's latest message as a standalone document QA query.",
            "Use the recent conversation only to resolve pronouns, ellipses, and missing subjects.",
            "Preserve the user's intent and language.",
            "Do not answer the question.",
            "If the latest message is already standalone, return it unchanged.",
            "Return only the rewritten query.",
        ],
    )
    prompt = f"""RECENT CONVERSATION:
{conversation}

LATEST USER MESSAGE:
{query}
"""
    response = agent.run(prompt)
    rewritten = response.content if isinstance(response.content, str) else str(response.content)
    rewritten = rewritten.strip()
    return rewritten or query


def _fallback_rewrite_query(query: str, conversation_history: list[Any] | None) -> str:
    latest_user = _latest_user_message(conversation_history)
    latest_assistant = _latest_assistant_message(conversation_history)
    if latest_user and latest_assistant:
        assistant_keywords = " ".join(re.findall(r"\b[\wÀ-ÿ]{4,}\b", latest_assistant)[:12])
        return (
            f"{latest_user} Approfondisci recuperando dettagli utili su servizi, prodotti, "
            f"soluzioni, competenze e contesto citati nella risposta precedente: {assistant_keywords}"
        ).strip()
    return query


def _usable_rewritten_query(value: str, original_query: str) -> str:
    rewritten = " ".join(str(value or "").split()).strip()
    if not rewritten:
        return ""
    if _BAD_MODEL_TEXT_RE.match(rewritten):
        return ""
    if len(rewritten.split()) <= 2 and len(original_query.split()) > 2:
        return ""
    return rewritten


def _conversation_transform_response(
    query: str,
    conversation_history: list[Any] | None,
    route: ConversationRoute | None = None,
) -> dict | None:
    target = _target_translation_language(query)
    source_text = _latest_assistant_message(conversation_history)
    if not source_text:
        return None

    target_code, target_name = target if target is not None else ("", None)
    if not target_name and route and route.target_language:
        target_name = route.target_language
    response_language = _conversation_response_language(query, target_code)
    try:
        answer = _transform_previous_answer(source_text, query, target_name)
        fallback_used = False
        fallback_reason = ""
        confidence = 0.99
    except Exception as exc:
        logger.warning("Conversation transform failed: %s", exc)
        answer = (
            "Non sono riuscito a trasformare automaticamente l'ultima risposta, "
            "ma questa e' una richiesta di trasformazione conversazionale e non richiede escalation sui documenti."
            if response_language == "it"
            else "I could not transform the previous answer automatically, but this conversational request does not require document escalation."
        )
        fallback_used = True
        fallback_reason = "conversation_transform_model_unavailable"
        confidence = 0.0

    return {
        "answer": answer,
        "confidence": confidence,
        "citations": [],
        "query_type": "CONVERSATION_TRANSFORM",
        "document_candidates": [],
        "ranked_chunks": [],
        "gaps": [],
        "hallucination_risk": 0.0 if not fallback_used else 1.0,
        "should_escalate": False,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "original_query": query,
        "modified_query": query,
        "intent": {
            "query_type": "CONVERSATION_TRANSFORM",
            "question_scope": "conversation",
            "route_reason": route.reason if route else "",
        },
        "retrieval_plan": {
            "strategy": "conversation_transform",
            "sources": ["conversation_history", "conversation_intent_router"],
        },
        "grounding_checks": [],
        "unsupported_claims": [],
        "post_grounding_status": "skipped",
        "suggested_actions": [
            {
                "id": "ask_followup",
                "label": "Continua" if response_language == "it" else "Continue",
                "type": "query",
                "reason": "La risposta e' stata trasformata dalla conversazione." if response_language == "it" else "The answer was transformed from conversation context.",
            }
        ],
        "node_trace": ["CONVERSATION_TRANSFORM"],
    }


def _prepare_query_for_cag(
    query: str,
    conversation_history: list[Any] | None,
    route: ConversationRoute | None = None,
) -> tuple[str, str]:
    if route and route.action == "direct_retrieval":
        return query, ""
    if route and route.action == "rewrite_for_retrieval" and route.rewritten_query.strip():
        return route.rewritten_query.strip(), "conversation_route_rewrite"
    should_rewrite = bool(route and route.action == "rewrite_for_retrieval")
    if not should_rewrite and not _looks_like_contextual_followup(query, conversation_history):
        return query, ""
    try:
        rewritten = _rewrite_query_with_history(query, conversation_history)
    except Exception as exc:
        logger.warning("Conversation query rewrite failed; using fallback rewrite: %s", exc)
        fallback_query = _fallback_rewrite_query(query, conversation_history)
        return fallback_query, "conversation_rewrite_failed_fallback"
    rewritten = _usable_rewritten_query(rewritten, query)
    if not rewritten:
        fallback_query = _fallback_rewrite_query(query, conversation_history)
        logger.warning("Conversation query rewrite returned unusable output; using fallback rewrite: %r", fallback_query)
        return fallback_query, "conversation_rewrite_unusable_fallback"
    if rewritten != query:
        logger.info("Conversation follow-up rewritten for retrieval: %r -> %r", query, rewritten)
        return rewritten, "conversation_rewrite"
    return query, ""


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Query the CAG pipeline from the frontend."""

    _require_api_key(x_api_key)
    try:
        try:
            conversation_route = _route_conversation_turn(payload.query, payload.conversation_history)
        except Exception as exc:
            logger.warning("Conversation route failed; using deterministic fallback: %s", exc)
            conversation_route = _fallback_conversation_route(payload.query, payload.conversation_history)

        if conversation_route.action == "answer_transform":
            transform_response = _conversation_transform_response(
                payload.query,
                payload.conversation_history,
                conversation_route,
            )
            if transform_response is not None:
                return transform_response

        effective_query, coordinator_reason = _prepare_query_for_cag(
            payload.query,
            payload.conversation_history,
            conversation_route,
        )
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
            query=effective_query,
            conversation_history=payload.conversation_history or [],
            runtime_config=runtime_config,
            access_filter=payload.access_filter or {},
            original_query=payload.query,
        )
    except Exception as exc:
        logger.exception("Query endpoint failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/diagnostics/retrieval", response_model=RetrievalDiagnosticsResponse)
async def retrieval_diagnostics_endpoint(
    payload: QueryRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Run ENTRY, RETRIEVE, and SELECT_CONTEXT without answer generation."""

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
        state = {
            "query": payload.query,
            "original_query": payload.query,
            "modified_query": payload.query,
            "question_scope": "domain",
            "retrieval_strategy": "semantic",
            "intent": {},
            "retrieval_plan": {},
            "access_filter": payload.access_filter or {},
            "chunks": [],
            "ranked_chunks": [],
            "document_candidates": [],
            "gaps": [],
            "relevance_score": 0.0,
            "answer": "",
            "confidence": 0.0,
            "citations": [],
            "hallucination_risk": 0.0,
            "query_type": "GENERAL",
            "response_language": "en",
            "should_escalate": False,
            "should_retry_reason": False,
            "should_retry_retrieval": False,
            "retrieval_retry_used": False,
            "reason_retries": 0,
            "error_message": "",
            "retry_guidance": "",
            "fallback_used": False,
            "fallback_reason": "",
            "grounding_checks": [],
            "unsupported_claims": [],
            "post_grounding_status": "pending",
            "suggested_actions": [],
            "node_trace": [],
            "conversation_history": payload.conversation_history or [],
            "relevance_threshold": runtime_config.relevance_threshold,
            "confidence_threshold": runtime_config.confidence_threshold,
            "hallucination_threshold": runtime_config.hallucination_threshold,
            "retrieval_top_k": runtime_config.retrieval_top_k,
            "search_fn": None,
        }
        state.update(entry_node(state))
        state.update(retrieve_node(state))
        state.update(select_context_node(state))
        return state
    except Exception as exc:
        logger.exception("Retrieval diagnostics endpoint failed")
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an empty favicon response to avoid noisy browser 404s in local preview."""

    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
