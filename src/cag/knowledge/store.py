"""SQLite store for DB-first compiled knowledge."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from cag.config import settings

SCHEMA_VERSION = 4

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_status TEXT NOT NULL DEFAULT 'indexed',
    UNIQUE(source_uri, sha256)
);

CREATE TABLE IF NOT EXISTS source_versions (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, version),
    UNIQUE(source_id, sha256)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES source_versions(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    domain_module TEXT NOT NULL DEFAULT 'general',
    start_offset INTEGER,
    end_offset INTEGER,
    vector_id TEXT,
    UNIQUE(source_version_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS document_profiles (
    id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES source_versions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    topics TEXT NOT NULL DEFAULT '[]',
    entities TEXT NOT NULL DEFAULT '[]',
    covered_intents TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    generator TEXT NOT NULL DEFAULT 'local_fallback',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_version_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL DEFAULT 'general',
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(claim_text)
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    support_type TEXT NOT NULL DEFAULT 'supports',
    evidence_quote TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (claim_id, chunk_id, support_type)
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    aliases TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topic_claims (
    topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (topic_id, claim_id)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    evidence_chunk_id TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_type, source_id, relation, target_type, target_id, evidence_chunk_id)
);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    left_claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    right_claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_log (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS saved_answers (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    claim_ids TEXT NOT NULL DEFAULT '[]',
    citation_chunk_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_term_stats (
    term TEXT PRIMARY KEY,
    doc_frequency INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrieval_corpus_stats (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


def _new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection for the compiled knowledge store."""

    path = Path(db_path or settings.knowledge_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations."""

    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
        (SCHEMA_VERSION,),
    )
    connection.commit()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_bytes(source_uri: str, fallback_content: str = "") -> bytes:
    path = Path(source_uri)
    if path.exists() and path.is_file():
        return path.read_bytes()
    return fallback_content.encode("utf-8")


def upsert_source_version(
    connection: sqlite3.Connection,
    *,
    filename: str,
    source_uri: str,
    content: bytes,
    mime_type: str = "text/plain",
) -> tuple[str, str]:
    """Store a source and version, returning `(source_id, source_version_id)`."""

    digest = sha256_bytes(content)
    existing_source = connection.execute(
        "SELECT id FROM sources WHERE source_uri = ? ORDER BY created_at DESC LIMIT 1",
        (source_uri,),
    ).fetchone()
    source_id = existing_source["id"] if existing_source else _new_id()

    connection.execute(
        """
        INSERT OR IGNORE INTO sources(id, filename, source_uri, mime_type, sha256, ingest_status)
        VALUES (?, ?, ?, ?, ?, 'indexed')
        """,
        (source_id, filename, source_uri, mime_type, digest),
    )

    existing_version = connection.execute(
        "SELECT id FROM source_versions WHERE source_id = ? AND sha256 = ?",
        (source_id, digest),
    ).fetchone()
    if existing_version:
        return source_id, existing_version["id"]

    latest = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM source_versions WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    version = int(latest["version"]) + 1
    source_version_id = _new_id()
    connection.execute(
        """
        INSERT INTO source_versions(id, source_id, version, sha256)
        VALUES (?, ?, ?, ?)
        """,
        (source_version_id, source_id, version, digest),
    )
    return source_id, source_version_id


def store_chunk(
    connection: sqlite3.Connection,
    *,
    source_version_id: str,
    chunk_index: int,
    content: str,
    domain_module: str = "general",
    start_offset: int | None = None,
    end_offset: int | None = None,
    vector_id: str | None = None,
) -> str:
    existing = connection.execute(
        "SELECT id FROM chunks WHERE source_version_id = ? AND chunk_index = ?",
        (source_version_id, chunk_index),
    ).fetchone()
    chunk_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR REPLACE INTO chunks(
            id, source_version_id, chunk_index, content, domain_module, start_offset, end_offset, vector_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, source_version_id, chunk_index, content, domain_module, start_offset, end_offset, vector_id),
    )
    return chunk_id


def store_document_profile(
    connection: sqlite3.Connection,
    *,
    source_version_id: str,
    summary: str,
    keywords: list[str],
    topics: list[str],
    entities: list[str],
    covered_intents: list[str],
    status: str = "active",
    generator: str = "local_fallback",
) -> str:
    existing = connection.execute(
        "SELECT id FROM document_profiles WHERE source_version_id = ?",
        (source_version_id,),
    ).fetchone()
    profile_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR REPLACE INTO document_profiles(
            id, source_version_id, summary, keywords, topics, entities, covered_intents, status, generator
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile_id,
            source_version_id,
            summary,
            json.dumps(keywords, ensure_ascii=False),
            json.dumps(topics, ensure_ascii=False),
            json.dumps(entities, ensure_ascii=False),
            json.dumps(covered_intents, ensure_ascii=False),
            status,
            generator,
        ),
    )
    return profile_id


def store_claim(
    connection: sqlite3.Connection,
    *,
    claim_text: str,
    chunk_id: str,
    claim_type: str = "general",
    confidence: float = 0.5,
    evidence_quote: str = "",
    support_type: str = "supports",
) -> str:
    existing = connection.execute(
        "SELECT id FROM claims WHERE claim_text = ?",
        (claim_text,),
    ).fetchone()
    claim_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR IGNORE INTO claims(id, claim_text, claim_type, confidence, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        (claim_id, claim_text, claim_type, confidence),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO claim_evidence(claim_id, chunk_id, support_type, evidence_quote)
        VALUES (?, ?, ?, ?)
        """,
        (claim_id, chunk_id, support_type, evidence_quote or claim_text[:240]),
    )
    return claim_id


def store_entity(
    connection: sqlite3.Connection,
    *,
    name: str,
    entity_type: str = "concept",
    aliases: list[str] | None = None,
) -> str:
    normalized_name = " ".join(name.strip().split())
    existing = connection.execute(
        "SELECT id, aliases FROM entities WHERE name = ?",
        (normalized_name,),
    ).fetchone()
    entity_id = existing["id"] if existing else _new_id()
    alias_values = list(aliases or [])
    if existing:
        try:
            alias_values.extend(json.loads(existing["aliases"] or "[]"))
        except json.JSONDecodeError:
            pass
    clean_aliases = []
    for alias in alias_values:
        clean_alias = " ".join(str(alias).strip().split())
        if clean_alias and clean_alias != normalized_name and clean_alias not in clean_aliases:
            clean_aliases.append(clean_alias)
    connection.execute(
        """
        INSERT OR REPLACE INTO entities(id, name, entity_type, aliases)
        VALUES (?, ?, ?, ?)
        """,
        (entity_id, normalized_name, entity_type, json.dumps(clean_aliases, ensure_ascii=False)),
    )
    return entity_id


def store_topic(
    connection: sqlite3.Connection,
    *,
    slug: str,
    title: str,
    summary: str = "",
    status: str = "active",
) -> str:
    existing = connection.execute(
        "SELECT id FROM topics WHERE slug = ?",
        (slug,),
    ).fetchone()
    topic_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR REPLACE INTO topics(id, slug, title, summary, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (topic_id, slug, title, summary, status),
    )
    return topic_id


def store_topic_claim(connection: sqlite3.Connection, *, topic_id: str, claim_id: str, rank: int = 0) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO topic_claims(topic_id, claim_id, rank)
        VALUES (?, ?, ?)
        """,
        (topic_id, claim_id, rank),
    )


def store_graph_edge(
    connection: sqlite3.Connection,
    *,
    source_type: str,
    source_id: str,
    relation: str,
    target_type: str,
    target_id: str,
    evidence_chunk_id: str | None = None,
    confidence: float = 0.5,
) -> str:
    existing = connection.execute(
        """
        SELECT id FROM graph_edges
        WHERE source_type = ?
          AND source_id = ?
          AND relation = ?
          AND target_type = ?
          AND target_id = ?
          AND COALESCE(evidence_chunk_id, '') = COALESCE(?, '')
        """,
        (source_type, source_id, relation, target_type, target_id, evidence_chunk_id),
    ).fetchone()
    edge_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR REPLACE INTO graph_edges(
            id, source_type, source_id, relation, target_type, target_id, evidence_chunk_id, confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            edge_id,
            source_type,
            source_id,
            relation,
            target_type,
            target_id,
            evidence_chunk_id,
            confidence,
        ),
    )
    return edge_id


def store_contradiction(
    connection: sqlite3.Connection,
    *,
    left_claim_id: str,
    right_claim_id: str,
    reason: str,
    status: str = "open",
) -> str:
    left, right = sorted([left_claim_id, right_claim_id])
    existing = connection.execute(
        """
        SELECT id FROM contradictions
        WHERE left_claim_id = ? AND right_claim_id = ? AND status = ?
        """,
        (left, right, status),
    ).fetchone()
    contradiction_id = existing["id"] if existing else _new_id()
    connection.execute(
        """
        INSERT OR IGNORE INTO contradictions(id, left_claim_id, right_claim_id, reason, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (contradiction_id, left, right, reason, status),
    )
    return contradiction_id


def log_event(connection: sqlite3.Connection, event_type: str, payload: dict) -> None:
    connection.execute(
        "INSERT INTO knowledge_log(id, event_type, payload) VALUES (?, ?, ?)",
        (_new_id(), event_type, json.dumps(payload, ensure_ascii=False)),
    )


def rows(connection: sqlite3.Connection, query: str, params: Iterable = ()) -> list[sqlite3.Row]:
    return list(connection.execute(query, tuple(params)).fetchall())


def replace_retrieval_term_stats(
    connection: sqlite3.Connection,
    *,
    doc_frequencies: dict[str, int],
    corpus_size: int,
) -> None:
    connection.execute("DELETE FROM retrieval_term_stats")
    connection.execute("DELETE FROM retrieval_corpus_stats")
    if doc_frequencies:
        connection.executemany(
            "INSERT INTO retrieval_term_stats(term, doc_frequency) VALUES (?, ?)",
            sorted((term, int(freq)) for term, freq in doc_frequencies.items() if term),
        )
    connection.execute(
        "INSERT INTO retrieval_corpus_stats(key, value) VALUES ('corpus_size', ?)",
        (max(0, int(corpus_size)),),
    )


def load_retrieval_term_stats(connection: sqlite3.Connection) -> tuple[dict[str, int], int]:
    term_rows = rows(connection, "SELECT term, doc_frequency FROM retrieval_term_stats")
    stats = {str(row["term"]): int(row["doc_frequency"]) for row in term_rows}
    corpus_row = connection.execute(
        "SELECT value FROM retrieval_corpus_stats WHERE key = 'corpus_size'"
    ).fetchone()
    corpus_size = int(corpus_row["value"]) if corpus_row else 0
    return stats, corpus_size


def list_knowledge_graph(
    connection: sqlite3.Connection,
    *,
    node_limit: int = 200,
    edge_limit: int = 500,
) -> dict[str, list[dict]]:
    """Return a compact graph view for UI/API consumers."""

    node_by_key: dict[tuple[str, str], dict] = {}

    def add_node(node_type: str, node_id: str, label: str, **properties) -> None:
        key = (node_type, node_id)
        if key in node_by_key:
            return
        if len(node_by_key) >= node_limit:
            return
        node_by_key[key] = {
            "id": f"{node_type}:{node_id}",
            "type": node_type,
            "label": label,
            "properties": properties,
        }

    for row in rows(
        connection,
        """
        SELECT source_versions.id AS id, sources.filename AS label, sources.source_uri AS source
        FROM source_versions
        JOIN sources ON sources.id = source_versions.source_id
        ORDER BY sources.filename ASC
        """,
    ):
        add_node("source", row["id"], row["label"], source=row["source"])

    for row in rows(connection, "SELECT id, claim_text, claim_type, confidence FROM claims WHERE status = 'active'"):
        add_node("claim", row["id"], row["claim_text"], claim_type=row["claim_type"], confidence=row["confidence"])

    for row in rows(connection, "SELECT id, slug, title, summary FROM topics WHERE status = 'active'"):
        add_node("topic", row["id"], row["title"], slug=row["slug"], summary=row["summary"])

    for row in rows(connection, "SELECT id, name, entity_type, aliases FROM entities"):
        add_node("entity", row["id"], row["name"], entity_type=row["entity_type"], aliases=_safe_json_list(row["aliases"]))

    graph_edges = []
    for row in rows(
        connection,
        """
        SELECT id, source_type, source_id, relation, target_type, target_id, evidence_chunk_id, confidence
        FROM graph_edges
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (edge_limit,),
    ):
        source_key = (row["source_type"], row["source_id"])
        target_key = (row["target_type"], row["target_id"])
        if source_key not in node_by_key or target_key not in node_by_key:
            continue
        graph_edges.append(
            {
                "id": row["id"],
                "source": f"{row['source_type']}:{row['source_id']}",
                "target": f"{row['target_type']}:{row['target_id']}",
                "relation": row["relation"],
                "confidence": row["confidence"],
                "evidence_chunk_id": row["evidence_chunk_id"] or "",
            }
        )

    return {"nodes": list(node_by_key.values()), "edges": graph_edges}


def _safe_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def claim_document(row: sqlite3.Row) -> Document:
    return Document(
        page_content=row["claim_text"],
        metadata={
            "filename": row["filename"],
            "source": row["source_uri"],
            "domain_module": row["domain_module"],
            "chunk_index": row["chunk_index"],
            "claim_id": row["claim_id"],
            "chunk_id": row["chunk_id"],
            "selection_category": row["claim_type"],
            "compiled_knowledge": True,
        },
    )
