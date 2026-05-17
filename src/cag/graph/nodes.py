"""
LangGraph nodes for the CAG pipeline.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langdetect import DetectorFactory
from langdetect import detect as _langdetect_detect

from cag.agents.reasoning_agent import run_reasoning_agent
from cag.agents.review_agent import run_review_agent
from cag.agents.retrieval_agent import run_retrieval_agent
from cag.config import settings
from cag.graph.state import CAGState
from cag.ingestion.embedder import lexical_file_search, similarity_search
from cag.knowledge.store import connect as _connect_knowledge_db
from cag.knowledge.store import initialize as _initialize_knowledge_db
from cag.knowledge.store import load_retrieval_term_stats as _load_retrieval_term_stats
from cag.knowledge.document_map import search_chunks_for_document_candidates, search_document_profiles
from cag.retrieval.lexical import dedupe_documents as _dedupe_documents
from cag.retrieval.lexical import expand_query_concepts as _expand_query_concepts
from cag.retrieval.lexical import extract_keywords as _extract_keywords

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

QUERY_REWRITES = {
    "configure": "configuration",
    "configured": "configuration",
    "setup": "configuration",
    "setting": "configuration",
    "settings": "configuration",
    "insert": "insertion",
    "inserted": "insertion",
    "calculate": "calculation",
    "close": "closure",
    "solve": "resolution",
    "resolve": "resolution",
    "rejected": "error",
    "svilupppate": "sviluppate",
    "sviluppate": "sviluppo",
    "app": "applicazioni",
}

INSUFFICIENT_ANSWER_SIGNALS = [
    "documentation is insufficient",
    "documentation available does not cover",
    "the documentation does not cover",
    "not documented",
    "not available",
    "not present",
    "cannot determine",
    "insufficient information",
    "human support required",
]

CONTEXTUAL_FOLLOWUP_MARKERS_RE = re.compile(
    r"\b("
    r"spiegami meglio|spiega meglio|approfondisci|piu dettagli|piu' dettagli|"
    r"come potete esserci di aiuto|come potete aiut|in che modo potete|"
    r"tell me more|explain more|more details|how can you help|how could you help"
    r")\b",
    re.IGNORECASE,
)

# Pre-compiled regex patterns for query classification
_DIAGNOSTIC_RE = re.compile(
    r"\b(error|not working|problem|diagnostic|fault|404|500|why|how do i fix|how can i fix|rejected)\b",
    re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"\b(procedure|step by step|how do i|how can i|how to)\b",
    re.IGNORECASE,
)
_CONFIGURATION_RE = re.compile(
    r"\b(configur\w*|setup|settings?|parameters?|parametr\w*|"
    r"which fields|which data|required fields|required data|"
    r"prerequisites?)\b",
    re.IGNORECASE,
)
_GENERAL_RE = re.compile(
    r"\b(what is|what does|which are|how does|timeline)\b",
    re.IGNORECASE,
)

_DOMAIN_MARKERS_RE = re.compile(
    r"\b(document(?:ation)?|workflow|report|error|issue|setup|settings?|"
    r"contract|scrum|policy|procedure|process|module|manual)\b",
    re.IGNORECASE,
)
_PERSONAL_MARKERS_RE = re.compile(
    r"\b(how are you|who are you|tell me a joke|tell me about yourself)\b",
    re.IGNORECASE,
)
_CONSULTATIVE_MARKERS_RE = re.compile(
    r"\b(best practice|recommend|recommended|should i|what should we choose)\b",
    re.IGNORECASE,
)

_STRIP_PRONOUNS_RE = re.compile(
    r"\b(how do i|how can i|which are|why\b|i want to|i need to)\b",
    re.IGNORECASE,
)

_ITALIAN_LANGUAGE_MARKERS = (
    "come posso",
    "come faccio",
    "risorse umane",
    "posso",
    "vorrei",
    "voglio",
    "devo",
    "gestire",
    "gestione",
    "risorse",
    "umane",
    "dove",
    "operate",
    "siete",
    "quali",
    "cosa",
    "perche",
    "perché",
    "azienda",
    "sede",
    "sedi",
    "documenti",
    "configuro",
    "configurare",
)
_FRENCH_LANGUAGE_MARKERS = (
    "comment puis",
    "puis-je",
    "je veux",
    "voudrais",
    "ressources humaines",
    "gerer",
    "gérer",
    "quelles",
    "quoi",
    "pourquoi",
    "entreprise",
)


def _document_profile_evidence(document_candidates: list[dict]) -> list[Document]:
    profile_docs: list[Document] = []
    for index, candidate in enumerate(document_candidates[:2]):
        summary = str(candidate.get("summary", "")).strip()
        if not summary:
            continue
        profile_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "filename": candidate.get("filename", candidate.get("source", "N/A")),
                    "source": candidate.get("source", candidate.get("filename", "N/A")),
                    "domain_module": "document_profile",
                    "chunk_index": -1 - index,
                    "compiled_knowledge": True,
                    "document_profile_id": candidate.get("profile_id", ""),
                    "document_score": candidate.get("score", 0.0),
                    "document_match_reason": candidate.get("match_reason", ""),
                    "document_profile_generator": candidate.get("generator", ""),
                    "selection_category": "overview",
                },
            )
        )
    return profile_docs


def _source_name(value: object) -> str:
    return Path(str(value or "")).name.lower()


def _passes_access_filter(metadata: dict, access_filter: dict | None) -> bool:
    if not access_filter:
        return True

    allowed_sources = {
        _source_name(source)
        for source in access_filter.get("allowed_sources", [])
        if str(source).strip()
    }
    if allowed_sources:
        source = _source_name(metadata.get("filename") or metadata.get("source"))
        if source not in allowed_sources:
            return False

    workspace_id = access_filter.get("workspace_id")
    if workspace_id and metadata.get("workspace_id") != workspace_id:
        return False

    visibility = access_filter.get("visibility")
    if visibility and metadata.get("visibility") != visibility:
        return False

    required_tags = set(access_filter.get("tags", []) or [])
    if required_tags:
        metadata_tags = set(metadata.get("tags", []) or [])
        if not required_tags.intersection(metadata_tags):
            return False

    return True


def _filter_document_candidates(candidates: list[dict], access_filter: dict | None) -> list[dict]:
    return [candidate for candidate in candidates if _passes_access_filter(candidate, access_filter)]


def _filter_documents(documents: list[Document], access_filter: dict | None) -> list[Document]:
    return [document for document in documents if _passes_access_filter(document.metadata, access_filter)]


def _display_content(doc: Document) -> str:
    original_content = doc.metadata.get("original_content")
    if isinstance(original_content, str) and original_content.strip():
        return original_content
    return doc.page_content


def _log_node(state: CAGState, node_name: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("[LOG] %s -- Graph node: %s -- Query: '%s'", timestamp, node_name, state.get("query", "")[:60])


def _language_marker_score(normalized_query: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in normalized_query)


def _infer_response_language(query: str) -> str:
    """Detect the query language, with deterministic guards for short Italian questions."""
    normalized_query = " ".join(query.lower().split())
    italian_score = _language_marker_score(normalized_query, _ITALIAN_LANGUAGE_MARKERS)
    french_score = _language_marker_score(normalized_query, _FRENCH_LANGUAGE_MARKERS)

    # langdetect often mistakes short Italian business questions for French.
    if italian_score >= 2 and italian_score > french_score:
        return "it"

    try:
        lang = _langdetect_detect(query)
        if isinstance(lang, str) and len(lang) >= 2:
            detected = lang[:2].lower()
            if detected == "fr" and italian_score >= 1 and italian_score >= french_score:
                return "it"
            return detected
    except Exception:
        pass
    return "en"


def _localized_message(language: str, italian: str, english: str) -> str:
    if language == "it":
        return italian
    return english


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    role = getattr(message, "role", "")
    if role:
        return str(role).lower()
    message_type = getattr(message, "type", "")
    if message_type:
        return "assistant" if message_type == "ai" else str(message_type).lower()
    return message.__class__.__name__.lower()


def _message_content(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(getattr(message, "content", "")).strip()


def _latest_assistant_message(conversation_history: list | None) -> str:
    for message in reversed(conversation_history or []):
        if _message_role(message) in {"assistant", "ai", "aimessage"}:
            content = _message_content(message)
            if content:
                return content
    return ""


def _looks_like_contextual_followup(query: str, conversation_history: list | None) -> bool:
    if not conversation_history or not _latest_assistant_message(conversation_history):
        return False
    normalized = " ".join(query.lower().split())
    if CONTEXTUAL_FOLLOWUP_MARKERS_RE.search(normalized):
        return True
    if len(normalized.split()) <= 8 and re.search(r"\b(anche|meglio|dettagli|come|e|and|also|more|how)\b", normalized):
        return True
    return False


def _history_anchor_terms(text: str, limit: int = 24) -> str:
    terms: list[str] = []
    for term in _extract_keywords(text):
        normalized = term.lower()
        if normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) >= limit:
            break
    return " ".join(terms)


def _contextualize_followup_query(query: str, conversation_history: list | None, response_language: str) -> str:
    if not _looks_like_contextual_followup(query, conversation_history):
        return query

    latest_assistant = _latest_assistant_message(conversation_history)
    anchor_terms = _history_anchor_terms(latest_assistant)
    if not anchor_terms:
        return query

    if response_language == "it":
        return (
            f"{query}. Approfondisci come l'azienda puo' aiutare usando il contesto "
            f"della risposta precedente e recuperando dettagli documentati su: {anchor_terms}."
        )
    return (
        f"{query}. Explain how the organization can help using the previous answer context "
        f"and retrieve documented details about: {anchor_terms}."
    )


def _classify_question_scope(query: str) -> str:
    if _PERSONAL_MARKERS_RE.search(query) and not _DOMAIN_MARKERS_RE.search(query):
        return "personal"
    if _CONSULTATIVE_MARKERS_RE.search(query):
        return "consultative"
    return "domain"


def _infer_query_type(query: str) -> str:
    if _DIAGNOSTIC_RE.search(query):
        return "DIAGNOSTIC"
    if _PROCEDURAL_RE.search(query):
        return "PROCEDURAL"
    if _CONFIGURATION_RE.search(query):
        return "CONFIGURATION"
    if _GENERAL_RE.search(query):
        return "GENERAL"
    return "GENERAL"


def _select_retrieval_strategy(query_type: str, question_scope: str) -> str:
    if question_scope == "consultative":
        return "multi_evidence"
    if query_type == "PROCEDURAL":
        return "hierarchical"
    if query_type == "DIAGNOSTIC":
        return "multi_evidence"
    return "semantic"


def _build_query_variants(query: str, query_type: str) -> list[str]:
    variants = [query.strip()]
    normalized = query.lower()
    for source, target in QUERY_REWRITES.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)

    keyword_focus = " ".join(_extract_keywords(normalized)[:8])
    if keyword_focus and keyword_focus not in variants:
        variants.append(keyword_focus)

    concept_variant = _expand_query_concepts(query)
    if concept_variant != query and concept_variant not in variants:
        variants.append(concept_variant)

    if query_type in {"PROCEDURAL", "DIAGNOSTIC", "CONFIGURATION"}:
        compact = _STRIP_PRONOUNS_RE.sub(" ", normalized)
        compact = " ".join(part for part in compact.split() if part)
        if compact and compact not in variants:
            variants.append(compact)

    deduped: list[str] = []
    for variant in variants:
        clean_variant = variant.strip()
        if clean_variant and clean_variant not in deduped:
            deduped.append(clean_variant)
    return deduped[:4]


def _build_retrieval_plan(
    query: str,
    query_type: str,
    question_scope: str,
    retrieval_strategy: str,
    retrieval_top_k: int,
    access_filter: dict | None,
) -> dict:
    variants = _build_query_variants(query, query_type)
    sources = ["document_profiles", "semantic_index", "lexical_fallback"]
    if access_filter:
        sources.append("access_filter")
    return {
        "strategy": retrieval_strategy,
        "query_variants": variants,
        "top_k": retrieval_top_k,
        "sources": sources,
        "rationale": (
            f"{query_type.lower()} {question_scope} request routed through "
            f"{retrieval_strategy} grounding."
        ),
        "access_filter_applied": bool(access_filter),
        "document_candidates": 0,
        "retrieved_chunks": 0,
    }


def _knowledge_corpus_term_stats() -> tuple[dict[str, int], int]:
    try:
        connection = _connect_knowledge_db(settings.knowledge_db_path)
        try:
            _initialize_knowledge_db(connection)
            return _load_retrieval_term_stats(connection)
        finally:
            connection.close()
    except Exception:
        return {}, 0


def _top_chunk_score(state: CAGState) -> float:
    ranked_chunks = state.get("ranked_chunks", [])
    if not ranked_chunks:
        return 0.0
    return max(float(chunk.get("relevance_score", 0.0)) for chunk in ranked_chunks)


def _moderately_supported_chunks(state: CAGState) -> int:
    return sum(
        1 for chunk in state.get("ranked_chunks", [])
        if float(chunk.get("relevance_score", 0.0)) >= settings.moderate_relevance_threshold
    )


def _state_relevance_threshold(state: CAGState) -> float:
    return float(state.get("relevance_threshold", settings.relevance_threshold))


def _state_confidence_threshold(state: CAGState) -> float:
    return float(state.get("confidence_threshold", settings.confidence_threshold))


def _state_hallucination_threshold(state: CAGState) -> float:
    return float(state.get("hallucination_threshold", settings.hallucination_threshold))


def _has_reasonable_evidence(state: CAGState) -> bool:
    top_score = _top_chunk_score(state)
    moderate_chunks = _moderately_supported_chunks(state)
    query_type = state.get("query_type", "GENERAL")
    relevance_threshold = _state_relevance_threshold(state)
    ranked_chunks = state.get("ranked_chunks", [])

    if top_score >= relevance_threshold:
        return True
    if query_type == "GENERAL" and any(
        chunk.get("domain_module") == "document_profile"
        and chunk.get("selection_category") in {"overview", "definitions", "general"}
        and float(chunk.get("document_score", 0.0) or 0.0) >= 1.0
        for chunk in ranked_chunks
    ):
        return True
    if query_type in {"PROCEDURAL", "DIAGNOSTIC"} and top_score >= settings.moderate_relevance_threshold:
        return True
    if moderate_chunks >= 2 and state.get("relevance_score", 0.0) >= settings.moderate_relevance_threshold:
        return True
    return False


def _build_retry_context(state: CAGState) -> list[dict]:
    ranked_chunks = list(state.get("ranked_chunks", []))
    if not ranked_chunks:
        return []

    query_type = state.get("query_type", "GENERAL")
    narrowed_limit = 4 if query_type in {"PROCEDURAL", "DIAGNOSTIC"} else 3
    strong_chunks = [
        chunk
        for chunk in ranked_chunks
        if float(chunk.get("relevance_score", 0.0)) >= settings.moderate_relevance_threshold
    ]
    narrowed = strong_chunks[:narrowed_limit] or ranked_chunks[:narrowed_limit]
    return narrowed


def _adaptive_retry_query(query: str, query_type: str) -> str:
    if query_type == "PROCEDURAL":
        return f"{query} procedure steps prerequisites navigation fields exact sequence"
    if query_type == "DIAGNOSTIC":
        return f"{query} symptoms causes checks resolution workaround error"
    if query_type == "CONFIGURATION":
        return f"{query} prerequisites permissions settings fields parameters options"
    return f"{query} definitions overview constraints details supporting evidence"


def _looks_like_insufficient_answer(answer: str) -> bool:
    normalized = answer.lower().strip()
    if not normalized:
        return True
    return any(signal in normalized for signal in INSUFFICIENT_ANSWER_SIGNALS)


def _chunk_document_score(chunks: list[dict], source: str, content: str) -> float:
    for chunk in chunks:
        if chunk.get("source") == source and chunk.get("content") == content:
            return float(chunk.get("document_score", 0.0) or 0.0)
    return 0.0


def _answer_sentences(answer: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]


def _evidence_terms(state: CAGState) -> set[str]:
    texts: list[str] = []
    for chunk in state.get("ranked_chunks", []):
        texts.append(str(chunk.get("content", "")))
    for citation in state.get("citations", []):
        texts.append(str(citation.get("text", "")))
    joined = " ".join(texts)
    return {term.lower() for term in _extract_keywords(joined) if len(term) >= 4}


def _is_generic_answer_sentence(sentence: str) -> bool:
    normalized = sentence.lower()
    generic_markers = (
        "secondo la documentazione",
        "based on the documentation",
        "non emerge",
        "not enough information",
        "la documentazione",
        "the documentation",
    )
    return any(marker in normalized for marker in generic_markers)


def _suggested_actions(state: CAGState) -> list[dict]:
    response_language = state.get("response_language", "en")
    italian = response_language == "it"
    actions: list[dict] = []

    if state.get("should_escalate"):
        actions.append({
            "id": "escalate_review",
            "label": "Revisione umana" if italian else "Human review",
            "type": "handoff",
            "reason": "Evidenze insufficienti o rischio alto." if italian else "Insufficient evidence or elevated risk.",
        })
        actions.append({
            "id": "add_sources",
            "label": "Aggiungi fonti" if italian else "Add sources",
            "type": "source",
            "reason": "Carica materiale piu' specifico." if italian else "Upload more specific support material.",
        })
    else:
        actions.append({
            "id": "inspect_evidence",
            "label": "Ispeziona evidenze" if italian else "Inspect evidence",
            "type": "inspect",
            "reason": "Verifica chunk, citazioni e lacune." if italian else "Review chunks, citations, and gaps.",
        })
        actions.append({
            "id": "ask_followup",
            "label": "Domanda di follow-up" if italian else "Ask follow-up",
            "type": "query",
            "prompt": "Puoi approfondire con i passaggi operativi?" if italian else "Can you expand with the operational steps?",
            "reason": "Continua dal contesto corrente." if italian else "Continue from the current context.",
        })

    if state.get("unsupported_claims"):
        actions.append({
            "id": "tighten_answer",
            "label": "Stringi ai fatti" if italian else "Tighten to facts",
            "type": "query",
            "prompt": "Rispondi usando solo i fatti direttamente supportati." if italian else "Answer using only directly supported facts.",
            "reason": "Il post-processing ha rilevato claim deboli." if italian else "Post-processing found weakly supported claims.",
        })

    return actions[:3]


def entry_node(state: CAGState) -> dict:
    _log_node(state, "ENTRY")
    query = state.get("query", "").strip()

    question_scope = _classify_question_scope(query)
    query_type = _infer_query_type(query)
    retrieval_strategy = _select_retrieval_strategy(query_type, question_scope)
    response_language = _infer_response_language(query)
    access_filter = state.get("access_filter") or {}
    retrieval_top_k = state.get("retrieval_top_k", settings.retrieval_top_k)
    retrieval_plan = _build_retrieval_plan(
        query,
        query_type,
        question_scope,
        retrieval_strategy,
        retrieval_top_k,
        access_filter,
    )

    history = state.get("conversation_history", [])
    new_history = history + [HumanMessage(content=query)]

    return {
        "query": query,
        "original_query": state.get("original_query") or query,
        "modified_query": query,
        "question_scope": question_scope,
        "query_type": query_type,
        "retrieval_strategy": retrieval_strategy,
        "intent": {
            "query_type": query_type,
            "question_scope": question_scope,
            "response_language": response_language,
        },
        "retrieval_plan": retrieval_plan,
        "access_filter": access_filter,
        "chunks": [],
        "ranked_chunks": [],
        "document_candidates": [],
        "gaps": [],
        "relevance_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "hallucination_risk": 0.0,
        "should_escalate": False,
        "should_retry_reason": False,
        "should_retry_retrieval": False,
        "retrieval_retry_used": bool(state.get("retrieval_retry_used", False)),
        "reason_retries": state.get("reason_retries", 0),
        "error_message": "",
        "retry_guidance": "",
        "response_language": response_language,
        "fallback_used": False,
        "fallback_reason": "",
        "grounding_checks": [],
        "unsupported_claims": [],
        "post_grounding_status": "pending",
        "suggested_actions": [],
        "node_trace": ["ENTRY"],
        "conversation_history": new_history,
        "relevance_threshold": state.get("relevance_threshold", settings.relevance_threshold),
        "confidence_threshold": state.get("confidence_threshold", settings.confidence_threshold),
        "hallucination_threshold": state.get("hallucination_threshold", settings.hallucination_threshold),
        "retrieval_top_k": state.get("retrieval_top_k", settings.retrieval_top_k),
        "search_fn": state.get("search_fn") or similarity_search,
    }


def contextualize_query_node(state: CAGState) -> dict:
    _log_node(state, "CONTEXTUALIZE_QUERY")
    query = state.get("query", "").strip()
    contextualized_query = _contextualize_followup_query(
        query,
        state.get("conversation_history", []),
        state.get("response_language", "en"),
    )
    trace = state.get("node_trace", []) + ["CONTEXTUALIZE_QUERY"]
    if contextualized_query == query:
        return {"node_trace": trace}

    retrieval_plan = {
        **(state.get("retrieval_plan") or {}),
        "conversation_contextualized": True,
        "original_turn_query": state.get("original_query") or query,
        "sources": [
            *(state.get("retrieval_plan") or {}).get("sources", []),
            "conversation_history",
        ],
    }
    intent = {
        **(state.get("intent") or {}),
        "conversation_contextualized": True,
    }
    logger.info("CONTEXTUALIZE_QUERY: %r -> %r", query, contextualized_query)
    return {
        "query": contextualized_query,
        "modified_query": contextualized_query,
        "intent": intent,
        "retrieval_plan": retrieval_plan,
        "node_trace": trace,
    }


def retrieve_node(state: CAGState) -> dict:
    _log_node(state, "RETRIEVE")
    query = state["query"]
    query_type = state.get("query_type", "GENERAL")
    strategy = state.get("retrieval_strategy", "semantic")
    query_variants = _build_query_variants(query, query_type)
    retrieval_top_k = int(state.get("retrieval_top_k", settings.retrieval_top_k))
    access_filter = state.get("access_filter") or {}
    per_query_k = retrieval_top_k
    if strategy == "hierarchical":
        per_query_k = max(retrieval_top_k, 12)
    elif strategy == "multi_evidence":
        per_query_k = max(retrieval_top_k, 14)
    corpus_doc_frequencies, corpus_size = _knowledge_corpus_term_stats()

    try:
        active_search_fn = state.get("search_fn") or similarity_search
        document_candidates = _filter_document_candidates(search_document_profiles(query, k=5), access_filter)
        if document_candidates:
            raw_results = search_chunks_for_document_candidates(
                query,
                document_candidates,
                k=max(per_query_k, retrieval_top_k),
                include_neighbors=query_type == "PROCEDURAL",
            )
            raw_results = _filter_documents(raw_results, access_filter)
            raw_results = _document_profile_evidence(document_candidates) + raw_results
            if not raw_results:
                logger.info("Document Map found candidates but no chunks; falling back to global retrieval.")
        else:
            raw_results = []

        if not raw_results:
            for index, variant in enumerate(query_variants):
                variant_k = per_query_k if index == 0 else max(4, per_query_k // 2)
                raw_results.extend(active_search_fn(variant, k=variant_k))
            raw_results = _filter_documents(raw_results, access_filter)

        if settings.hybrid_lexical_retrieval and not document_candidates:
            hybrid_query = query_variants[0] if query_variants else query
            lexical_results = lexical_file_search(
                hybrid_query,
                k=min(settings.hybrid_lexical_top_k, retrieval_top_k),
            )
            lexical_results = _filter_documents(lexical_results, access_filter)
            if lexical_results:
                raw_results.extend(lexical_results)
        results = _dedupe_documents(
            query,
            raw_results,
            query_variants,
            corpus_doc_frequencies=corpus_doc_frequencies,
            corpus_size=corpus_size,
        )
        max_results = retrieval_top_k if strategy == "semantic" else retrieval_top_k + 4
        results = results[:max_results]

        chunks = [
            {
                "content": _display_content(doc),
                "source": doc.metadata.get("filename", doc.metadata.get("source", "N/A")),
                "domain_module": doc.metadata.get("domain_module", "general"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "compiled_knowledge": bool(doc.metadata.get("compiled_knowledge", False)),
                "claim_id": doc.metadata.get("claim_id", ""),
                "knowledge_chunk_id": doc.metadata.get("chunk_id", ""),
                "document_profile_id": doc.metadata.get("document_profile_id", ""),
                "document_score": doc.metadata.get("document_score", 0.0),
                "document_match_reason": doc.metadata.get("document_match_reason", ""),
                "document_profile_generator": doc.metadata.get("document_profile_generator", ""),
                "procedural_neighbor": bool(doc.metadata.get("procedural_neighbor", False)),
                "procedural_anchor_index": doc.metadata.get("procedural_anchor_index", ""),
            }
            for doc in results
        ]
        logger.info(
            "RETRIEVE: %s chunks recovered | documents=%s | strategy=%s | variants=%s",
            len(chunks),
            len(document_candidates),
            strategy,
            " | ".join(query_variants),
        )
    except Exception as exc:
        logger.error("RETRIEVE error: %s", exc)
        chunks = []
        document_candidates = []

    trace = state.get("node_trace", []) + ["RETRIEVE"]
    retrieval_plan = {
        **(state.get("retrieval_plan") or {}),
        "query_variants": query_variants,
        "document_candidates": len(document_candidates),
        "retrieved_chunks": len(chunks),
        "access_filter_applied": bool(access_filter),
    }
    return {
        "chunks": chunks,
        "document_candidates": document_candidates,
        "retrieval_plan": retrieval_plan,
        "node_trace": trace,
    }


def select_context_node(state: CAGState) -> dict:
    _log_node(state, "SELECT_CONTEXT")
    output = run_retrieval_agent(
        query=state["query"],
        chunks=state["chunks"],
        query_type_hint=state.get("query_type", "GENERAL"),
        strategy_hint=state.get("retrieval_strategy", "semantic"),
    )

    logger.info("SELECT_CONTEXT: relevance_score=%.2f, gaps=%s", output.relevance_score, len(output.gaps))

    ranked = [
        {
            "content": chunk.content,
            "source": chunk.source,
            "domain_module": chunk.domain_module,
            "chunk_index": chunk.chunk_index,
            "cluster_id": chunk.cluster_id,
            "selection_category": chunk.selection_category,
            "relevance_score": chunk.relevance_score,
            "relevance_reason": chunk.relevance_reason,
            "document_score": _chunk_document_score(state.get("chunks", []), chunk.source, chunk.content),
        }
        for chunk in output.chunks_ranked
    ]

    trace = state.get("node_trace", []) + ["SELECT_CONTEXT"]
    return {
        "ranked_chunks": ranked,
        "gaps": output.gaps,
        "relevance_score": output.relevance_score,
        "fallback_used": bool(state.get("fallback_used", False) or output.fallback_used),
        "fallback_reason": str(state.get("fallback_reason", "") or output.fallback_reason or ""),
        "node_trace": trace,
    }


def reason_node(state: CAGState) -> dict:
    _log_node(state, "REASON")
    retries = state.get("reason_retries", 0)
    output = run_reasoning_agent(
        query=state["query"],
        ranked_chunks=state["ranked_chunks"],
        gaps=state["gaps"],
        query_type_hint=state.get("query_type", "GENERAL"),
        response_language=state.get("response_language", "en"),
        retry_guidance=state.get("retry_guidance", ""),
    )
    output = run_review_agent(
        query=state["query"],
        output=output,
        ranked_chunks=state["ranked_chunks"],
        gaps=state["gaps"],
        response_language=state.get("response_language", "en"),
    )

    logger.info(
        "REASON: confidence=%.2f, hallucination_risk=%.2f, type=%s",
        output.confidence,
        output.hallucination_risk,
        output.query_type,
    )

    citations = [
        {"text": citation.text, "source": citation.source, "domain_module": citation.domain_module}
        for citation in output.citations
    ]

    trace = state.get("node_trace", []) + [f"REASON(retry={retries})", "REVIEW"]
    return {
        "answer": output.answer,
        "query_type": state.get("query_type", output.query_type),
        "confidence": output.confidence,
        "citations": citations,
        "hallucination_risk": output.hallucination_risk,
        "reason_retries": retries + 1,
        "should_retry_reason": False,
        "retry_guidance": "",
        "fallback_used": bool(state.get("fallback_used", False) or output.fallback_used),
        "fallback_reason": str(state.get("fallback_reason", "") or output.fallback_reason or ""),
        "node_trace": trace,
    }


def post_grounding_node(state: CAGState) -> dict:
    _log_node(state, "POST_GROUNDING")
    answer = state.get("answer", "")
    evidence_terms = _evidence_terms(state)
    sentences = _answer_sentences(answer)

    grounding_checks: list[dict] = []
    unsupported_claims: list[str] = []
    if not answer:
        status = "skipped"
    elif not evidence_terms:
        status = "warn"
        unsupported_claims = sentences[:3]
    else:
        for sentence in sentences:
            sentence_terms = {term.lower() for term in _extract_keywords(sentence) if len(term) >= 4}
            overlap = sorted(sentence_terms.intersection(evidence_terms))
            supported = bool(overlap) or _is_generic_answer_sentence(sentence)
            grounding_checks.append({
                "claim": sentence,
                "supported": supported,
                "matched_terms": overlap[:6],
            })
            if not supported and len(sentence) > 32:
                unsupported_claims.append(sentence)
        status = "warn" if unsupported_claims else "passed"

    unsupported_ratio = len(unsupported_claims) / max(1, len(sentences))
    confidence = float(state.get("confidence", 0.0))
    hallucination_risk = float(state.get("hallucination_risk", 0.0))
    if unsupported_ratio >= 0.5 and not any(chunk.get("domain_module") == "document_profile" for chunk in state.get("ranked_chunks", [])):
        confidence = min(confidence, 0.45)
        hallucination_risk = max(hallucination_risk, 0.65)
    elif unsupported_claims:
        hallucination_risk = max(hallucination_risk, 0.35)

    trace = state.get("node_trace", []) + ["POST_GROUNDING"]
    return {
        "confidence": confidence,
        "hallucination_risk": hallucination_risk,
        "grounding_checks": grounding_checks,
        "unsupported_claims": unsupported_claims[:5],
        "post_grounding_status": status,
        "node_trace": trace,
    }


def validate_node(state: CAGState) -> dict:
    _log_node(state, "VALIDATE")

    hallucination_risk = state.get("hallucination_risk", 1.0)
    confidence = state.get("confidence", 0.0)
    reason_retries = state.get("reason_retries", 0)
    relevance_score = state.get("relevance_score", 0.0)
    answer = state.get("answer", "")
    has_reasonable_evidence = _has_reasonable_evidence(state)
    insufficient_answer = _looks_like_insufficient_answer(answer)
    confidence_threshold = _state_confidence_threshold(state)
    hallucination_threshold = _state_hallucination_threshold(state)
    relevance_threshold = _state_relevance_threshold(state)
    response_language = state.get("response_language", "en")

    should_escalate = False
    should_retry_reason = False
    should_retry_retrieval = False
    error_message = ""
    retry_guidance = ""
    retry_ranked_chunks: list[dict] | None = None
    retry_query = ""
    retry_strategy = ""
    retry_top_k = int(state.get("retrieval_top_k", settings.retrieval_top_k))

    if (
        settings.adaptive_retrieval_retry
        and not state.get("retrieval_retry_used", False)
        and reason_retries == 1
        and not insufficient_answer
        and has_reasonable_evidence
        and (confidence < confidence_threshold or hallucination_risk > hallucination_threshold)
    ):
        should_retry_retrieval = True
        query_type = state.get("query_type", "GENERAL")
        retry_query = _adaptive_retry_query(state.get("query", ""), query_type)
        retry_strategy = "hierarchical" if query_type in {"PROCEDURAL", "CONFIGURATION"} else "multi_evidence"
        retry_top_k = min(50, retry_top_k + settings.adaptive_retry_top_k_boost)
        retry_guidance = _localized_message(
            response_language,
            "Retry adattivo: recupera piu' evidenze, usa query espansa e cambia strategia prima di rigenerare.",
            "Adaptive retry: retrieve more evidence, use an expanded query, and change strategy before regenerating.",
        )
    elif not answer and not has_reasonable_evidence:
        should_escalate = True
        error_message = _localized_message(
            response_language,
            "La documentazione recuperata non copre questa richiesta in modo affidabile. "
            "Serve una revisione umana oppure altro materiale di supporto.",
            "The retrieved documentation does not cover this request reliably. "
            "Human review or additional source material is required.",
        )
    elif insufficient_answer and (
        not has_reasonable_evidence
        or confidence < confidence_threshold
        or hallucination_risk > hallucination_threshold
    ):
        should_escalate = True
        error_message = _localized_message(
            response_language,
            "La risposta generata indica che la documentazione disponibile non e' sufficiente "
            "per rispondere in sicurezza alla richiesta.",
            "The generated answer indicates that the available documentation is not sufficient "
            "to answer the request safely.",
        )
    elif relevance_score < relevance_threshold and not has_reasonable_evidence:
        should_escalate = True
        error_message = _localized_message(
            response_language,
            "Le evidenze recuperate sono troppo deboli o indirette per rispondere con affidabilita'.",
            "The recovered evidence is too weak or indirect to answer confidently.",
        )
    elif hallucination_risk > hallucination_threshold and reason_retries >= settings.max_reason_retries:
        should_escalate = True
        error_message = _localized_message(
            response_language,
            f"Il rischio di allucinazione resta alto ({hallucination_risk:.0%}) "
            f"dopo {reason_retries} tentativi.",
            f"Hallucination risk remains high ({hallucination_risk:.0%}) after {reason_retries} attempts.",
        )
    elif confidence < confidence_threshold and reason_retries >= settings.max_reason_retries:
        should_escalate = True
        error_message = _localized_message(
            response_language,
            f"La confidenza resta troppo bassa ({confidence:.0%}) per rispondere in sicurezza.",
            f"Confidence remains too low ({confidence:.0%}) to answer safely.",
        )
    elif hallucination_risk > hallucination_threshold and reason_retries < settings.max_reason_retries:
        should_retry_reason = True
        retry_guidance = _localized_message(
            response_language,
            "Retry piu' conservativo: usa solo i chunk piu' forti, evita inferenze laterali, "
            "preferisci una risposta parziale ma ben supportata ed evidenzia cio' che manca.",
            "Retry more conservatively: use only the strongest chunks, avoid lateral inference, "
            "prefer a partial but well-supported answer, and make missing support explicit.",
        )
        retry_ranked_chunks = _build_retry_context(state)
    elif confidence < confidence_threshold and has_reasonable_evidence and reason_retries < settings.max_reason_retries:
        should_retry_reason = True
        retry_guidance = _localized_message(
            response_language,
            "Retry con risposta piu' stretta: limita la risposta ai fatti supportati direttamente, "
            "riduci il contesto al nucleo piu' rilevante e dichiara eventuali lacune.",
            "Retry with a narrower answer: limit the answer to directly supported facts, "
            "shrink the context to the most relevant core, and state any remaining gaps.",
        )
        retry_ranked_chunks = _build_retry_context(state)

    logger.info(
        "VALIDATE: should_escalate=%s, should_retry_reason=%s, should_retry_retrieval=%s, hallucination_risk=%.2f, confidence=%.2f",
        should_escalate,
        should_retry_reason,
        should_retry_retrieval,
        hallucination_risk,
        confidence,
    )

    trace = state.get("node_trace", []) + ["VALIDATE"]
    update = {
        "should_escalate": should_escalate,
        "should_retry_reason": should_retry_reason,
        "should_retry_retrieval": should_retry_retrieval,
        "error_message": error_message,
        "retry_guidance": retry_guidance,
        "node_trace": trace,
    }
    if retry_ranked_chunks is not None:
        update["ranked_chunks"] = retry_ranked_chunks
    if should_retry_retrieval:
        update["query"] = retry_query
        update["modified_query"] = retry_query
        update["retrieval_strategy"] = retry_strategy
        update["retrieval_top_k"] = retry_top_k
        update["retrieval_retry_used"] = True
        update["chunks"] = []
        update["ranked_chunks"] = []
        update["gaps"] = []
        update["relevance_score"] = 0.0
        update["answer"] = ""
        update["citations"] = []
        update["retrieval_plan"] = {
            **(state.get("retrieval_plan") or {}),
            "adaptive_retry": True,
            "top_k": retry_top_k,
            "strategy": retry_strategy,
            "query_variants": _build_query_variants(retry_query, state.get("query_type", "GENERAL")),
        }
    return update


def exit_node(state: CAGState) -> dict:
    _log_node(state, "EXIT")

    if state.get("should_escalate"):
        response_language = state.get("response_language", "en")
        default_message = _localized_message(
            response_language,
            "Questa richiesta richiede l'intervento di uno specialista.",
            "This request requires a human specialist.",
        )
        error_message = state.get("error_message") or default_message
        answer = _localized_message(
            response_language,
            (
                "Escalation al supporto consigliata.\n\n"
                f"{error_message}\n\n"
                "Instrada questa richiesta a un revisore umano oppure aggiungi documenti di supporto."
            ),
            (
                "Support escalation recommended.\n\n"
                f"{error_message}\n\n"
                "Please route this question to a human reviewer or provide additional supporting documents."
            ),
        )
    else:
        answer = state.get("answer", "No answer was generated.")

    history = state.get("conversation_history", [])
    new_history = history + [AIMessage(content=answer)]

    trace = state.get("node_trace", []) + ["EXIT"]
    suggested_actions = _suggested_actions({**state, "answer": answer})
    logger.info("EXIT: delivered response (%s characters)", len(answer))

    return {
        "answer": answer,
        "conversation_history": new_history,
        "fallback_used": bool(state.get("fallback_used", False)),
        "fallback_reason": str(state.get("fallback_reason", "")),
        "suggested_actions": suggested_actions,
        "node_trace": trace,
    }


def route_after_select_context(state: CAGState) -> str:
    if _has_reasonable_evidence(state):
        return "reason"
    logger.info(
        "route_after_select_context -> VALIDATE (insufficient evidence: relevance=%.2f top_chunk=%.2f)",
        state.get("relevance_score", 0.0),
        _top_chunk_score(state),
    )
    return "validate"


def route_after_validate(state: CAGState) -> str:
    if state.get("should_escalate"):
        return "exit"

    if state.get("should_retry_retrieval"):
        logger.info(
            "route_after_validate -> RETRIEVE adaptive retry (query=%s)",
            str(state.get("query", ""))[:120],
        )
        return "retrieve"

    if state.get("should_retry_reason"):
        logger.info(
            "route_after_validate -> REASON retry (guidance=%s)",
            str(state.get("retry_guidance", ""))[:120],
        )
        return "reason"

    return "exit"
