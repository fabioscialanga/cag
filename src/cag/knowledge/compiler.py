"""Deterministic first slice of the DB-first Knowledge Compiler."""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from cag.knowledge.store import (
    claim_document,
    connect,
    initialize,
    log_event,
    replace_retrieval_term_stats,
    rows,
    source_bytes,
    store_chunk,
    store_claim,
    store_document_profile,
    store_entity,
    store_graph_edge,
    store_topic,
    store_topic_claim,
    upsert_source_version,
)
from cag.knowledge.document_map import build_document_profile, build_hypothetical_questions
from cag.retrieval.lexical import build_weighted_query_terms, document_terms

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b")


def _slugify(value: str) -> str:
    slug = "-".join(part for part in re.sub(r"[^a-zA-Z0-9]+", " ", value.lower()).split() if part)
    return slug[:80] or "general"


def _extract_entities(text: str, limit: int = 8) -> list[str]:
    entities: list[str] = []
    for match in _ENTITY_RE.findall(text):
        clean = " ".join(match.split())
        if clean and clean not in entities:
            entities.append(clean)
        if len(entities) >= limit:
            break
    return entities


def extract_claims(text: str, limit: int = 5) -> list[str]:
    """Extract deterministic claim candidates from chunk text."""

    claims: list[str] = []
    for part in _SENTENCE_RE.split(text.strip()):
        claim = " ".join(part.split())
        if len(claim) < 24 or claim in claims:
            continue
        claims.append(claim)
        if len(claims) >= limit:
            break
    return claims


def compile_chunks(chunks: list[Document], db_path: str | Path | None = None) -> dict[str, int]:
    """Store sources, chunks, deterministic claims, and provenance links."""

    connection = connect(db_path)
    try:
        initialize(connection)
        grouped: dict[str, list[Document]] = defaultdict(list)
        for chunk in chunks:
            source_uri = str(chunk.metadata.get("source") or chunk.metadata.get("filename") or "unknown")
            grouped[source_uri].append(chunk)

        source_count = 0
        chunk_count = 0
        claim_count = 0
        profile_count = 0
        retrieval_doc_frequencies: dict[str, int] = {}
        for source_uri, source_chunks in grouped.items():
            first = source_chunks[0]
            filename = str(first.metadata.get("filename") or Path(source_uri).name)
            fallback_content = "\n\n".join(chunk.page_content for chunk in source_chunks)
            _source_id, source_version_id = upsert_source_version(
                connection,
                filename=filename,
                source_uri=source_uri,
                content=source_bytes(source_uri, fallback_content),
            )
            source_count += 1
            source_claim_ids: list[str] = []
            for position, chunk in enumerate(source_chunks):
                chunk_index = position
                for term in set(document_terms(chunk)):
                    retrieval_doc_frequencies[term] = retrieval_doc_frequencies.get(term, 0) + 1
                chunk_id = store_chunk(
                    connection,
                    source_version_id=source_version_id,
                    chunk_index=chunk_index,
                    content=chunk.page_content,
                    domain_module=str(chunk.metadata.get("domain_module", "general")),
                )
                chunk_count += 1
                for claim in extract_claims(chunk.page_content):
                    claim_id = store_claim(
                        connection,
                        claim_text=claim,
                        chunk_id=chunk_id,
                        claim_type=str(chunk.metadata.get("selection_category", "general")),
                        confidence=0.6,
                        evidence_quote=claim[:240],
                    )
                    source_claim_ids.append(claim_id)
                    store_graph_edge(
                        connection,
                        source_type="source",
                        source_id=source_version_id,
                        relation="contains_claim",
                        target_type="claim",
                        target_id=claim_id,
                        evidence_chunk_id=chunk_id,
                        confidence=0.8,
                    )
                    claim_count += 1
            profile, generator = build_document_profile(filename, fallback_content)
            hypothetical_questions = build_hypothetical_questions(filename, profile)
            covered_intents = []
            for intent in [*profile.covered_intents, *hypothetical_questions]:
                clean_intent = " ".join(str(intent).strip().split())
                if clean_intent and clean_intent not in covered_intents:
                    covered_intents.append(clean_intent)
            store_document_profile(
                connection,
                source_version_id=source_version_id,
                summary=profile.summary,
                keywords=profile.keywords,
                topics=profile.topics,
                entities=profile.entities,
                covered_intents=covered_intents,
                status="active",
                generator=generator,
            )
            for chunk in source_chunks:
                chunk.metadata["document_summary"] = profile.summary
                chunk.metadata["document_topics"] = profile.topics
                chunk.metadata["document_keywords"] = profile.keywords
                chunk.metadata["hypothetical_questions"] = hypothetical_questions
            for rank, topic in enumerate(profile.topics):
                topic_id = store_topic(
                    connection,
                    slug=_slugify(f"{filename}-{topic}"),
                    title=topic,
                    summary=f"Topic extracted from {filename}.",
                )
                store_graph_edge(
                    connection,
                    source_type="source",
                    source_id=source_version_id,
                    relation="covers_topic",
                    target_type="topic",
                    target_id=topic_id,
                    confidence=0.7,
                )
                for claim_id in source_claim_ids:
                    store_topic_claim(connection, topic_id=topic_id, claim_id=claim_id, rank=rank)
                    store_graph_edge(
                        connection,
                        source_type="topic",
                        source_id=topic_id,
                        relation="supported_by",
                        target_type="claim",
                        target_id=claim_id,
                        confidence=0.55,
                    )

            profile_entities = [*profile.entities, *_extract_entities(fallback_content)]
            for entity in profile_entities[:14]:
                entity_id = store_entity(connection, name=entity, entity_type="concept")
                store_graph_edge(
                    connection,
                    source_type="source",
                    source_id=source_version_id,
                    relation="mentions",
                    target_type="entity",
                    target_id=entity_id,
                    confidence=0.65,
                )
                entity_lower = entity.lower()
                for claim_id in source_claim_ids:
                    claim_rows = rows(connection, "SELECT claim_text FROM claims WHERE id = ?", (claim_id,))
                    if claim_rows and entity_lower in claim_rows[0]["claim_text"].lower():
                        store_graph_edge(
                            connection,
                            source_type="claim",
                            source_id=claim_id,
                            relation="mentions",
                            target_type="entity",
                            target_id=entity_id,
                            confidence=0.75,
                        )
            profile_count += 1

        replace_retrieval_term_stats(
            connection,
            doc_frequencies=retrieval_doc_frequencies,
            corpus_size=chunk_count,
        )

        log_event(
            connection,
            "compile",
            {"sources": source_count, "chunks": chunk_count, "claims": claim_count, "document_profiles": profile_count},
        )
        connection.commit()
        return {"sources": source_count, "chunks": chunk_count, "claims": claim_count, "document_profiles": profile_count}
    finally:
        connection.close()


def compiled_search(query: str, k: int = 10, db_path: str | Path | None = None) -> list[Document]:
    """Search compiled claims and return LangChain-compatible documents."""

    weighted_terms = build_weighted_query_terms(query)
    if not weighted_terms:
        return []

    connection = connect(db_path)
    try:
        initialize(connection)
        candidates = rows(
            connection,
            """
            SELECT
                claims.id AS claim_id,
                claims.claim_text,
                claims.claim_type,
                chunks.id AS chunk_id,
                chunks.chunk_index,
                chunks.domain_module,
                sources.filename,
                sources.source_uri
            FROM claims
            JOIN claim_evidence ON claim_evidence.claim_id = claims.id
            JOIN chunks ON chunks.id = claim_evidence.chunk_id
            JOIN source_versions ON source_versions.id = chunks.source_version_id
            JOIN sources ON sources.id = source_versions.source_id
            WHERE claims.status = 'active'
            """,
        )
        scored = []
        doc_by_claim_id = {}
        for row in candidates:
            doc = claim_document(row)
            claim_id = str(row["claim_id"])
            doc_by_claim_id[claim_id] = doc
            terms = set(document_terms(doc))
            score = sum(weight for term, weight in weighted_terms.items() if term in terms)
            if score:
                scored.append((score, claim_id, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) < k and scored:
            seen_claim_ids = {claim_id for _score, claim_id, _doc in scored}
            for neighbor_id in _graph_neighbor_claim_ids(connection, seen_claim_ids):
                if neighbor_id in seen_claim_ids:
                    continue
                neighbor_doc = doc_by_claim_id.get(neighbor_id)
                if neighbor_doc is None:
                    continue
                scored.append((0.35, neighbor_id, neighbor_doc))
                seen_claim_ids.add(neighbor_id)
                if len(scored) >= k:
                    break

        return [doc for _score, _claim_id, doc in scored[:k]]
    finally:
        connection.close()


def _graph_neighbor_claim_ids(connection: sqlite3.Connection, seed_claim_ids: set[str]) -> list[str]:
    if not seed_claim_ids:
        return []

    edge_rows = rows(
        connection,
        """
        SELECT source_type, source_id, relation, target_type, target_id
        FROM graph_edges
        WHERE relation IN ('mentions', 'supported_by')
        """,
    )
    topic_to_claims: dict[str, list[str]] = defaultdict(list)
    entity_to_claims: dict[str, list[str]] = defaultdict(list)
    claim_to_entities: dict[str, list[str]] = defaultdict(list)

    for row in edge_rows:
        source_type = row["source_type"]
        target_type = row["target_type"]
        if source_type == "topic" and target_type == "claim" and row["relation"] == "supported_by":
            topic_to_claims[row["source_id"]].append(row["target_id"])
        elif source_type == "claim" and target_type == "entity" and row["relation"] == "mentions":
            claim_to_entities[row["source_id"]].append(row["target_id"])
            entity_to_claims[row["target_id"]].append(row["source_id"])

    neighbors: list[str] = []
    for topic_claims in topic_to_claims.values():
        if seed_claim_ids & set(topic_claims):
            for claim_id in topic_claims:
                if claim_id not in neighbors:
                    neighbors.append(claim_id)

    for seed_claim_id in seed_claim_ids:
        for entity_id in claim_to_entities.get(seed_claim_id, []):
            for claim_id in entity_to_claims.get(entity_id, []):
                if claim_id not in neighbors:
                    neighbors.append(claim_id)

    return neighbors
