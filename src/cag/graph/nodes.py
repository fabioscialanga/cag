"""
LangGraph nodes for the CAG pipeline.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langdetect import detect as _langdetect_detect

from cag.agents.reasoning_agent import run_reasoning_agent
from cag.agents.retrieval_agent import run_retrieval_agent
from cag.config import settings
from cag.graph.state import CAGState
from cag.ingestion.embedder import similarity_search

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "and", "do", "for", "how", "i", "in", "is", "of", "on",
    "the", "to", "what", "which",
}

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


def _log_node(state: CAGState, node_name: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("[LOG] %s -- Graph node: %s -- Query: '%s'", timestamp, node_name, state.get("query", "")[:60])


def _normalize_text(text: str) -> str:
    return "".join(character.lower() if character.isalnum() or character.isspace() else " " for character in text)


def _infer_response_language(query: str) -> str:
    """Detect the language of the query using langdetect, falling back to English."""
    try:
        lang = _langdetect_detect(query)
        if isinstance(lang, str) and len(lang) >= 2:
            return lang[:2].lower()
    except Exception:
        pass
    return "en"


def _localized_message(language: str, italian: str, english: str) -> str:
    if language == "it":
        return italian
    return english


def _extract_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    for token in _normalize_text(text).split():
        if len(token) <= 2 or token in STOPWORDS or token in keywords:
            continue
        keywords.append(token)
    return keywords


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
    return deduped[:3]


def _dedupe_documents(query: str, docs: list) -> list:
    query_terms = set(_extract_keywords(query))
    keyed: dict[tuple[str, int, str], object] = {}
    for doc in docs:
        key = (
            str(doc.metadata.get("filename", doc.metadata.get("source", "N/A"))),
            int(doc.metadata.get("chunk_index", 0)),
            doc.page_content[:160],
        )
        keyed.setdefault(key, doc)

    unique_docs = list(keyed.values())
    unique_docs.sort(
        key=lambda doc: len(
            query_terms
            & set(
                _extract_keywords(
                    f"{doc.page_content} {doc.metadata.get('filename', doc.metadata.get('source', 'N/A'))}"
                )
            )
        ),
        reverse=True,
    )
    return unique_docs


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

    if top_score >= relevance_threshold:
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


def _looks_like_insufficient_answer(answer: str) -> bool:
    normalized = answer.lower().strip()
    if not normalized:
        return True
    return any(signal in normalized for signal in INSUFFICIENT_ANSWER_SIGNALS)


def entry_node(state: CAGState) -> dict:
    _log_node(state, "ENTRY")
    query = state.get("query", "").strip()

    question_scope = _classify_question_scope(query)
    query_type = _infer_query_type(query)
    retrieval_strategy = _select_retrieval_strategy(query_type, question_scope)
    response_language = _infer_response_language(query)

    history = state.get("conversation_history", [])
    new_history = history + [HumanMessage(content=query)]

    return {
        "query": query,
        "question_scope": question_scope,
        "query_type": query_type,
        "retrieval_strategy": retrieval_strategy,
        "chunks": [],
        "ranked_chunks": [],
        "gaps": [],
        "relevance_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "hallucination_risk": 0.0,
        "should_escalate": False,
        "should_retry_reason": False,
        "reason_retries": state.get("reason_retries", 0),
        "error_message": "",
        "retry_guidance": "",
        "response_language": response_language,
        "fallback_used": False,
        "fallback_reason": "",
        "node_trace": ["ENTRY"],
        "conversation_history": new_history,
        "relevance_threshold": state.get("relevance_threshold", settings.relevance_threshold),
        "confidence_threshold": state.get("confidence_threshold", settings.confidence_threshold),
        "hallucination_threshold": state.get("hallucination_threshold", settings.hallucination_threshold),
        "search_fn": state.get("search_fn") or similarity_search,
    }


def retrieve_node(state: CAGState) -> dict:
    _log_node(state, "RETRIEVE")
    query = state["query"]
    query_type = state.get("query_type", "GENERAL")
    strategy = state.get("retrieval_strategy", "semantic")
    query_variants = _build_query_variants(query, query_type)
    per_query_k = settings.retrieval_top_k
    if strategy == "hierarchical":
        per_query_k = max(settings.retrieval_top_k, 12)
    elif strategy == "multi_evidence":
        per_query_k = max(settings.retrieval_top_k, 14)

    try:
        active_search_fn = state.get("search_fn") or similarity_search
        raw_results = []
        for index, variant in enumerate(query_variants):
            variant_k = per_query_k if index == 0 else max(4, per_query_k // 2)
            raw_results.extend(active_search_fn(variant, k=variant_k))
        results = _dedupe_documents(query, raw_results)
        max_results = settings.retrieval_top_k if strategy == "semantic" else settings.retrieval_top_k + 4
        results = results[:max_results]

        chunks = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("filename", doc.metadata.get("source", "N/A")),
                "domain_module": doc.metadata.get("domain_module", "general"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc in results
        ]
        logger.info(
            "RETRIEVE: %s chunks recovered | strategy=%s | variants=%s",
            len(chunks),
            strategy,
            " | ".join(query_variants),
        )
    except Exception as exc:
        logger.error("RETRIEVE error: %s", exc)
        chunks = []

    trace = state.get("node_trace", []) + ["RETRIEVE"]
    return {"chunks": chunks, "node_trace": trace}


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

    trace = state.get("node_trace", []) + [f"REASON(retry={retries})"]
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
    error_message = ""
    retry_guidance = ""
    retry_ranked_chunks: list[dict] | None = None

    if not answer and not has_reasonable_evidence:
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
        "VALIDATE: should_escalate=%s, should_retry_reason=%s, hallucination_risk=%.2f, confidence=%.2f",
        should_escalate,
        should_retry_reason,
        hallucination_risk,
        confidence,
    )

    trace = state.get("node_trace", []) + ["VALIDATE"]
    update = {
        "should_escalate": should_escalate,
        "should_retry_reason": should_retry_reason,
        "error_message": error_message,
        "retry_guidance": retry_guidance,
        "node_trace": trace,
    }
    if retry_ranked_chunks is not None:
        update["ranked_chunks"] = retry_ranked_chunks
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
    logger.info("EXIT: delivered response (%s characters)", len(answer))

    return {
        "answer": answer,
        "conversation_history": new_history,
        "fallback_used": bool(state.get("fallback_used", False)),
        "fallback_reason": str(state.get("fallback_reason", "")),
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

    if state.get("should_retry_reason"):
        logger.info(
            "route_after_validate -> REASON retry (guidance=%s)",
            str(state.get("retry_guidance", ""))[:120],
        )
        return "reason"

    return "exit"
