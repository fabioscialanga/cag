"""Document-level semantic profiles and document-first retrieval."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from agno.agent import Agent
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from cag.knowledge.store import connect, initialize, rows
from cag.llm_factory import get_agno_model
from cag.retrieval.lexical import (
    build_weighted_query_terms,
    discriminative_document_score,
    document_terms,
    expand_query_concepts,
    extract_keywords,
)

logger = logging.getLogger(__name__)

DOCUMENT_PROFILE_LIMIT = 5
DOCUMENT_PROFILE_MIN_SCORE = 1.0


class DocumentProfileOutput(BaseModel):
    summary: str = Field(default="")
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    covered_intents: list[str] = Field(default_factory=list)


def build_hypothetical_questions(filename: str, profile: DocumentProfileOutput, limit: int = 8) -> list[str]:
    """Generate HyPE-lite retrieval prompts from profile metadata."""

    questions: list[str] = []

    def add(question: str) -> None:
        clean = " ".join(question.strip().split())
        if clean and clean not in questions:
            questions.append(clean)

    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename
    add(f"What is covered in {stem}?")
    add(f"Summarize {stem}.")

    for topic in profile.topics[:4]:
        add(f"What does {stem} say about {topic}?")
        add(f"How does {topic} work?")

    for entity in profile.entities[:3]:
        add(f"What does the document say about {entity}?")

    for keyword in profile.keywords[:4]:
        add(f"Questions about {keyword}")

    for intent in profile.covered_intents[:4]:
        add(intent)

    return questions[:limit]


def _json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _clean_list(values: list[str], limit: int = 12) -> list[str]:
    clean: list[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().split())
        if normalized and normalized not in clean:
            clean.append(normalized)
        if len(clean) >= limit:
            break
    return clean


def _fallback_summary(text: str) -> str:
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    overview_markers = {
        "about", "azienda", "aziendale", "company", "focused", "mission",
        "overview", "piattaforma", "profilo", "services", "servizi",
        "solutions", "soluzioni", "specializzata", "technology",
    }
    sentence_candidates = [
        " ".join(part.split())
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if part.strip()
    ]
    candidates = []
    for index, paragraph in enumerate([*sentence_candidates, *paragraphs]):
        lowered = paragraph.lower()
        if len(paragraph) < 60 or "table of contents" in lowered or not set(paragraph) - {"=", "-", " "}:
            continue
        if lowered.startswith(("indice ", "riepilogo del sito web")) and "questo documento contiene" not in lowered:
            continue
        if "tel." in lowered and "email" in lowered:
            contact_penalty = 5
        else:
            contact_penalty = 0
        nav_penalty = 3 if any(marker in lowered for marker in ("linkedin", "facebook", "cerca login", "menu chiudi")) else 0
        terms = set(extract_keywords(lowered))
        sentence_bonus = 2 if any(marker in lowered for marker in ("presenta l'azienda", "sito presenta", "siamo specializzati", "affianca le aziende")) else 0
        score = (len(terms & overview_markers) * 4) + sentence_bonus - contact_penalty - nav_penalty - min(index, 8) * 0.1
        candidates.append((score, paragraph))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[0][1]
        for marker in ("Il sito presenta", "Studio 81 affianca", "Siamo specializzati", "Mettiamo in pratica"):
            marker_index = selected.find(marker)
            if marker_index >= 0:
                selected = selected[marker_index:]
                break
        return _truncate_summary(selected)
    compact = " ".join(text.split())
    return _truncate_summary(compact)


def _truncate_summary(text: str, limit: int = 760) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return _clean_summary_tail(clean)
    window = clean[:limit].rstrip()
    sentence_end = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if sentence_end >= 180:
        return _clean_summary_tail(window[: sentence_end + 1])
    return _clean_summary_tail(window.rsplit(" ", 1)[0].rstrip(" ,;:") + ".")


def _clean_summary_tail(text: str) -> str:
    clean = re.sub(r"\s+([.,;:!?])", r"\1", text.strip())
    clean = re.sub(r"(?:\s+\d+\.){2,}\s*$", "", clean).strip()
    return clean


def build_local_document_profile(filename: str, content: str) -> DocumentProfileOutput:
    """Build a deterministic profile when the LLM profile compiler is unavailable."""

    terms = extract_keywords(f"{filename} {content}")
    counts = Counter(terms)
    keywords = [term for term, _count in counts.most_common(14)]
    lowered = f"{filename} {content}".lower()
    topic_markers = {
        "overview": ["overview", "mission", "company", "about", "azienda", "aziendale", "profilo"],
        "configuration": ["configuration", "settings", "setup", "install", "configurazione"],
        "api": ["api", "endpoint", "token", "authentication"],
        "incident": ["incident", "error", "severity", "runbook"],
        "policy": ["policy", "handbook", "benefits", "conduct", "policy"],
        "services": ["servizi", "services", "soluzioni", "solutions", "prodotti"],
    }
    topics = [
        topic
        for topic, markers in topic_markers.items()
        if any(marker in lowered for marker in markers)
    ]
    entities = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b", content)
    covered_intents = [
        f"Questions about {topic}" for topic in topics
    ] or ["General questions about this document"]
    return DocumentProfileOutput(
        summary=_fallback_summary(content),
        keywords=_clean_list(keywords, 14),
        topics=_clean_list(topics, 8),
        entities=_clean_list(entities, 10),
        covered_intents=_clean_list(covered_intents, 8),
    )


def build_llm_document_profile(filename: str, content: str) -> DocumentProfileOutput:
    """Build a semantic document profile with the configured LLM."""

    agent = Agent(
        name="DocumentProfileCompiler",
        model=get_agno_model(),
        role="Summarizes one source document into retrieval-oriented metadata.",
        structured_outputs=True,
        output_schema=DocumentProfileOutput,
        instructions=[
            "Create a retrieval-oriented profile for one document.",
            "Return concise metadata only from the provided document.",
            "keywords should include natural synonyms users might ask.",
            "topics should be broad labels such as overview, configuration, api, incident, policy.",
            "covered_intents should be natural user questions or short user-question intents this document can answer.",
            "Return valid structured output only.",
        ],
    )
    excerpt = content[:12000]
    response = agent.run(
        f"FILENAME: {filename}\n\nDOCUMENT:\n{excerpt}\n\nBuild the document profile."
    )
    if isinstance(response.content, DocumentProfileOutput):
        return response.content
    if isinstance(response.content, str):
        return DocumentProfileOutput(**json.loads(response.content))
    return DocumentProfileOutput(**json.loads(str(response.content)))


def build_document_profile(filename: str, content: str) -> tuple[DocumentProfileOutput, str]:
    """Build an LLM-first profile, falling back locally when the provider fails."""

    try:
        profile = build_llm_document_profile(filename, content)
        return profile, "llm"
    except Exception as exc:
        logger.warning("LLM document profile failed for %s; using local fallback: %s", filename, exc)
        return build_local_document_profile(filename, content), "local_fallback"


def _profile_search_text(profile: dict) -> str:
    return " ".join(
        [
            profile.get("filename", ""),
            profile.get("summary", ""),
            " ".join(profile.get("keywords", [])),
            " ".join(profile.get("topics", [])),
            " ".join(profile.get("entities", [])),
            " ".join(profile.get("covered_intents", [])),
        ]
    )


def _query_expansion(query: str) -> str:
    return expand_query_concepts(query)


def search_document_profiles(
    query: str,
    *,
    k: int = DOCUMENT_PROFILE_LIMIT,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Return document candidates ranked by profile metadata."""

    weighted_terms = build_weighted_query_terms(query, [_query_expansion(query)])
    if not weighted_terms:
        return []

    connection = connect(db_path)
    try:
        initialize(connection)
        profile_rows = rows(
            connection,
            """
            SELECT
                document_profiles.id AS profile_id,
                document_profiles.source_version_id,
                document_profiles.summary,
                document_profiles.keywords,
                document_profiles.topics,
                document_profiles.entities,
                document_profiles.covered_intents,
                document_profiles.status,
                document_profiles.generator,
                sources.filename,
                sources.source_uri
            FROM document_profiles
            JOIN source_versions ON source_versions.id = document_profiles.source_version_id
            JOIN sources ON sources.id = source_versions.source_id
            WHERE document_profiles.status = 'active'
            """,
        )
    finally:
        connection.close()

    candidates = []
    for row in profile_rows:
        profile = {
            "profile_id": row["profile_id"],
            "source_version_id": row["source_version_id"],
            "filename": row["filename"],
            "source": row["source_uri"],
            "summary": row["summary"],
            "keywords": _json_list(row["keywords"]),
            "topics": _json_list(row["topics"]),
            "entities": _json_list(row["entities"]),
            "covered_intents": _json_list(row["covered_intents"]),
            "status": row["status"],
            "generator": row["generator"],
        }
        doc = Document(page_content=_profile_search_text(profile), metadata={"filename": profile["filename"]})
        terms = set(document_terms(doc))
        score = sum(weight * (1.0 if term in terms else 0.0) for term, weight in weighted_terms.items())
        if score >= DOCUMENT_PROFILE_MIN_SCORE:
            profile["score"] = round(score, 3)
            profile["match_reason"] = _match_reason(weighted_terms, terms, profile)
            candidates.append(profile)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:k]


def list_document_profiles(*, db_path: str | Path | None = None) -> list[dict]:
    """Return the latest stored document profiles for dashboard display."""

    connection = connect(db_path)
    try:
        initialize(connection)
        profile_rows = rows(
            connection,
            """
            SELECT
                document_profiles.id AS profile_id,
                document_profiles.source_version_id,
                document_profiles.summary,
                document_profiles.keywords,
                document_profiles.topics,
                document_profiles.entities,
                document_profiles.covered_intents,
                document_profiles.status,
                document_profiles.generator,
                document_profiles.created_at,
                sources.filename,
                sources.source_uri,
                source_versions.version,
                COUNT(chunks.id) AS chunk_count
            FROM document_profiles
            JOIN source_versions ON source_versions.id = document_profiles.source_version_id
            JOIN sources ON sources.id = source_versions.source_id
            LEFT JOIN chunks ON chunks.source_version_id = source_versions.id
            GROUP BY document_profiles.id
            ORDER BY document_profiles.created_at DESC, sources.filename ASC
            """,
        )
    finally:
        connection.close()

    return [
        {
            "profile_id": row["profile_id"],
            "source_version_id": row["source_version_id"],
            "filename": row["filename"],
            "source": row["source_uri"],
            "version": row["version"],
            "summary": row["summary"],
            "keywords": _json_list(row["keywords"]),
            "topics": _json_list(row["topics"]),
            "entities": _json_list(row["entities"]),
            "covered_intents": _json_list(row["covered_intents"]),
            "status": row["status"],
            "generator": row["generator"],
            "created_at": row["created_at"],
            "chunk_count": int(row["chunk_count"] or 0),
        }
        for row in profile_rows
    ]


def _match_reason(weighted_terms: dict[str, float], terms: set[str], profile: dict) -> str:
    matched = [term for term in weighted_terms if term in terms][:6]
    if matched:
        return "Matched profile terms: " + ", ".join(matched)
    return f"Matched document profile for {profile.get('filename', 'document')}"


def search_chunks_for_document_candidates(
    query: str,
    candidates: list[dict],
    *,
    k: int = 10,
    include_neighbors: bool = False,
    neighbor_window: int = 1,
    db_path: str | Path | None = None,
) -> list[Document]:
    """Search raw chunks only inside selected document candidates."""

    if not candidates:
        return []

    candidate_by_version = {candidate["source_version_id"]: candidate for candidate in candidates}
    placeholders = ",".join("?" for _ in candidate_by_version)
    connection = connect(db_path)
    try:
        initialize(connection)
        chunk_rows = rows(
            connection,
            f"""
            SELECT
                chunks.content,
                chunks.chunk_index,
                chunks.domain_module,
                chunks.source_version_id,
                sources.filename,
                sources.source_uri
            FROM chunks
            JOIN source_versions ON source_versions.id = chunks.source_version_id
            JOIN sources ON sources.id = source_versions.source_id
            WHERE chunks.source_version_id IN ({placeholders})
            """,
            tuple(candidate_by_version),
        )
    finally:
        connection.close()

    docs = [
        Document(
            page_content=row["content"],
            metadata={
                "filename": row["filename"],
                "source": row["source_uri"],
                "domain_module": row["domain_module"],
                "chunk_index": row["chunk_index"],
                "source_version_id": row["source_version_id"],
                "document_profile_id": candidate_by_version[row["source_version_id"]]["profile_id"],
                "document_score": candidate_by_version[row["source_version_id"]]["score"],
                "document_match_reason": candidate_by_version[row["source_version_id"]]["match_reason"],
                "document_profile_generator": candidate_by_version[row["source_version_id"]]["generator"],
            },
        )
        for row in chunk_rows
    ]
    if not docs:
        return []

    weighted_terms = build_weighted_query_terms(query, [_query_expansion(query)])
    doc_frequencies: dict[str, int] = {}
    for doc in docs:
        for term in set(document_terms(doc)):
            doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

    scored_docs = [
        (
            (
                discriminative_document_score(doc, weighted_terms, doc_frequencies, len(docs))
                + float(doc.metadata.get("document_score", 0.0))
            ),
            doc,
        )
        for doc in docs
    ]
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    selected_docs = [doc for score, doc in scored_docs if score > 0.0][:k]
    if not include_neighbors or not selected_docs:
        return selected_docs

    docs_by_identity = {
        (
            str(doc.metadata.get("source_version_id", "")),
            int(doc.metadata.get("chunk_index", 0)),
        ): doc
        for doc in docs
    }
    expanded: list[Document] = []
    seen: set[tuple[str, int]] = set()
    for doc in selected_docs:
        version_id = str(doc.metadata.get("source_version_id", ""))
        chunk_index = int(doc.metadata.get("chunk_index", 0))
        for offset in range(-neighbor_window, neighbor_window + 1):
            key = (version_id, chunk_index + offset)
            neighbor = docs_by_identity.get(key)
            if not neighbor or key in seen:
                continue
            seen.add(key)
            expanded.append(
                neighbor.model_copy(
                    update={
                        "metadata": {
                            **neighbor.metadata,
                            "procedural_neighbor": offset != 0,
                            "procedural_anchor_index": chunk_index,
                        }
                    }
                )
            )

    neighbor_budget = len(selected_docs) * ((neighbor_window * 2) + 1)
    return expanded[: max(k, neighbor_budget)]
