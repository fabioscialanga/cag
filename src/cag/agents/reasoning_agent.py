"""
Reasoning agent for grounded answer generation.
"""
from __future__ import annotations

import json
import logging
import re

from agno.agent import Agent

from cag.agents.models import Citation, ReasoningOutput
from cag.config import settings
from cag.llm_factory import get_agno_model
from cag.retrieval.lexical import extract_keywords

logger = logging.getLogger(__name__)
MIN_USEFUL_ANSWER_CHARS = 260
MAX_FALLBACK_FACTS = 6

_REASONING_INSTRUCTIONS = [
    "You are the ReasoningAgent of a CAG (Cognitive Augmented Generation) system.",
    "You receive a user question, ranked documentation chunks, and explicit information gaps.",
    "Your job is to generate a structured, grounded, cited answer.",
    "Answer in the same language as the user query.",
    "Detect the language of the user query and respond in that language.",
    "Preserve the user's language unless they explicitly request a different one.",
    "",
    "QUERY TYPES AND RESPONSE SHAPE:",
    "- DIAGNOSTIC: structure as Cause -> Check -> Resolution.",
    "- PROCEDURAL: write a short numbered sequence of steps.",
    "- CONFIGURATION: focus on prerequisites, fields, parameters, and options.",
    "- GENERAL: provide a concise factual explanation with relevant context.",
    "",
    "CORE RULES:",
    "1. Every claim must be supported by the provided chunks.",
    "2. Never invent steps, settings, values, paths, or decisions that are not in the evidence.",
    "3. If the chunks do not cover the request, say so explicitly.",
    "4. If the core concept is unsupported, do not answer by analogy.",
    "5. For insufficient coverage set confidence <= 0.35 and hallucination_risk >= 0.80.",
    "6. For partial coverage, answer only the supported portion and state what is missing.",
    "7. Always include citations.",
    "8. Answer the core request first and avoid unrelated prerequisites or side topics.",
    "9. Reuse field names, step names, and labels exactly as they appear in the evidence when possible.",
    "10. If the chunks already contain a numbered procedure or cause/solution wording, preserve that phrasing as much as possible.",
    "11. Avoid generic closing filler or meta commentary unless the question requires it.",
    "12. Synthesize across chunks instead of copying long excerpts verbatim.",
    "13. Start with the direct answer in the first sentence.",
    "14. Prefer compact bullets only when they make multiple facts, steps, or limits easier to scan.",
    "15. When document_profile evidence is present, use it to frame the answer, then support details from raw chunks.",
    "",
    "FEW-SHOT RESPONSE EXAMPLES:",
    "DIAGNOSTIC example: If evidence says '429 means rate limit exceeded' and 'honor Retry-After', answer with "
    "Cause: rate limit exceeded. Check: inspect Retry-After/request frequency. Resolution: reduce frequency and retry after the indicated delay.",
    "PROCEDURAL example: If evidence lists 'open Settings > Integrations, generate token, update services, revoke old token', "
    "answer as numbered steps in that exact order.",
    "INSUFFICIENT example: If evidence only mentions office hours and the query asks for software development services, "
    "say the retrieved documentation does not support the service claim and set low confidence/high hallucination risk.",
    "",
    "Return ONLY valid JSON with this structure:",
    '{"answer": "...", "query_type": "DIAGNOSTIC|PROCEDURAL|CONFIGURATION|GENERAL",'
    ' "confidence": 0.0, "citations": [...], "hallucination_risk": 0.0, "hallucination_reason": "..."}',
]

MODE_INSTRUCTIONS = {
    "DIAGNOSTIC": (
        "Structure the answer as Cause, Checks, and Resolution when evidence supports all three. "
        "Reuse the verbs and nouns from the source chunks when they already describe checks or corrective actions."
    ),
    "PROCEDURAL": (
        "Write short numbered steps with only one action per step. If the evidence already contains ordered "
        "steps, preserve the same sequence and wording as much as possible. Do not add related processes that "
        "were not requested."
    ),
    "CONFIGURATION": (
        "Highlight only the prerequisites, required values, fields, parameters, limits, and options required "
        "by the question. Do not recommend undocumented settings."
    ),
    "GENERAL": (
        "Answer briefly but with enough context to be useful. If the question asks 'what' or 'which', "
        "prefer a factual explanation or list over a step-by-step procedure."
    ),
}

RESPONSE_CATEGORY_ORDER = {
    "PROCEDURAL": ["overview", "definitions", "prerequisites", "permissions", "navigation", "ordered_steps", "fields", "settings", "constraints", "general"],
    "DIAGNOSTIC": ["overview", "symptoms", "error_causes", "checks", "resolution", "constraints", "general"],
    "CONFIGURATION": ["overview", "definitions", "prerequisites", "permissions", "settings", "fields", "options", "constraints", "general"],
    "GENERAL": ["definitions", "overview", "constraints", "timeline", "options", "general"],
}

LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
}


_reasoning_agent: Agent | None = None


def get_reasoning_agent() -> Agent:
    """Return the configured reasoning agent singleton."""

    global _reasoning_agent
    if _reasoning_agent is None:
        _reasoning_agent = Agent(
            name="ReasoningAgent",
            model=get_agno_model(),
            role="Builds grounded, structured answers with citations and confidence scores.",
            instructions=_REASONING_INSTRUCTIONS,
            structured_outputs=True,
            output_schema=ReasoningOutput,
        )
    return _reasoning_agent


def _extract_json(text: str) -> str:
    """Strip optional Markdown fences and return raw JSON text."""

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def _compact_evidence_text(text: str, query: str = "", max_chars: int = 420) -> str:
    """Return a short source excerpt without adding unsupported claims."""

    clean_text = " ".join(str(text).split())
    overview_query = _is_overview_query(query)
    if len(clean_text) <= max_chars:
        if overview_query and _is_bad_overview_excerpt(clean_text):
            return ""
        return clean_text

    query_terms = set(extract_keywords(query))
    normalized_query = query.lower()
    if overview_query:
        query_terms.update({
            "azienda", "aziendale", "company", "designed", "focused", "mission",
            "overview", "piattaforma", "platform", "prodotti", "profilo",
            "services", "servizi", "soluzioni", "specializzata",
        })
    raw_text = str(text)
    paragraph_candidates = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", raw_text)
        if part.strip()
    ]
    line_candidates = [
        " ".join(line.split())
        for line in raw_text.splitlines()
        if line.strip()
    ]
    candidates = [
        candidate
        for candidate in [*paragraph_candidates, *line_candidates]
        if 40 <= len(candidate) <= 900
        and "table of contents" not in candidate.lower()
        and not set(candidate) <= {"=", "-", " "}
    ]
    if query_terms and candidates:
        best_candidate = max(
            candidates,
            key=lambda candidate: (
                len(query_terms & set(extract_keywords(candidate))),
                min(len(candidate), 260),
                0 if overview_query and _is_bad_overview_excerpt(candidate) else 1,
                -len(candidate),
            ),
        )
        if len(query_terms & set(extract_keywords(best_candidate))) > 0 and not (
            overview_query and _is_bad_overview_excerpt(best_candidate)
        ):
            return best_candidate[:max_chars].strip()

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean_text) if part.strip()]
    if query_terms and sentences:
        best_sentence = max(
            sentences,
            key=lambda sentence: (
                len(query_terms & set(extract_keywords(sentence))),
                -len(sentence),
            ),
        )
        if len(query_terms & set(extract_keywords(best_sentence))) > 0:
            return best_sentence[:max_chars].strip()

    sentence_match = re.match(r"^(.{120,}?[.!?])\s", clean_text)
    if sentence_match:
        return sentence_match.group(1)[:max_chars].strip()
    return clean_text[: max_chars - 3].rstrip() + "..."


def _is_bad_overview_excerpt(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    if len(normalized) < 45:
        return True
    if normalized.count("tel.") >= 1 and "email" in normalized:
        return True
    trailing_words = {"con", "sulle", "sui", "sul", "delle", "dei", "di", "of", "with", "and"}
    return normalized.split()[-1].strip(".,;:") in trailing_words


def _is_overview_query(query: str) -> bool:
    normalized_query = query.lower()
    return any(marker in normalized_query for marker in ("what", "about", "overview", "cosa", "occupate", "fate", "chi siete"))


def _evidence_only_fallback(
    query: str,
    ranked_chunks: list[dict],
    query_type_hint: str,
    response_language: str,
    exc: Exception,
) -> ReasoningOutput:
    """Return only retrieved evidence when the reasoning model is unavailable."""

    query_type = query_type_hint if query_type_hint in MODE_INSTRUCTIONS else "GENERAL"
    evidence_chunks = _select_evidence_chunks(ranked_chunks)

    if not evidence_chunks:
        error_messages = {
            "it": "Il modello di generazione non e' raggiungibile e non sono disponibili evidenze recuperate.",
            "en": "The generation model is unavailable and no retrieved evidence is available.",
            "fr": "Le modele de generation n'est pas disponible et aucune preuve recuperee n'est disponible.",
            "de": "Das Generierungsmodell ist nicht verfuegbar und es liegen keine abgerufenen Belege vor.",
            "es": "El modelo de generacion no esta disponible y no hay evidencias recuperadas.",
            "pt": "O modelo de geracao nao esta disponivel e nao ha evidencias recuperadas.",
        }
        return ReasoningOutput(
            answer=error_messages.get(response_language, error_messages["en"]),
            query_type=query_type,
            confidence=0.0,
            citations=[],
            hallucination_risk=1.0,
            hallucination_reason=f"Reasoning model unavailable: {str(exc)[:200]}",
            fallback_used=True,
            fallback_reason="reasoning_model_unavailable_no_evidence",
        )

    citations = _citations_from_chunks(evidence_chunks, query=query)
    if not citations:
        citations = [
            Citation(
                text=_compact_evidence_text(evidence_chunks[0].get("content", ""), query="", max_chars=260),
                source=str(evidence_chunks[0].get("source", "N/A")),
                domain_module=str(evidence_chunks[0].get("domain_module", "general")),
            )
        ]
    answer = _format_evidence_answer(
        query=query,
        query_type=query_type,
        citations=citations,
        gaps=[],
        response_language=response_language,
    )
    hallucination_reason = (
        "Risposta di fallback sintetizzata solo da evidenze recuperate dalle fonti."
        if response_language == "it"
        else "Fallback response synthesized only from retrieved source evidence."
    )

    return ReasoningOutput(
        answer=answer,
        query_type=query_type,
        confidence=0.68,
        citations=citations,
        hallucination_risk=0.22,
        hallucination_reason=hallucination_reason,
        fallback_used=True,
        fallback_reason="reasoning_model_unavailable_evidence_only",
    )


def _select_evidence_chunks(ranked_chunks: list[dict], limit: int = 5) -> list[dict]:
    usable_chunks = [
        chunk for chunk in ranked_chunks
        if str(chunk.get("content", "")).strip()
    ]
    if not usable_chunks:
        return []

    top_relevance = max((float(chunk.get("relevance_score", 0.0)) for chunk in usable_chunks), default=0.0)
    relevance_floor = max(0.45, top_relevance - 0.25) if top_relevance else 0.0
    selected: list[dict] = []

    for chunk in usable_chunks:
        if str(chunk.get("domain_module", "")) == "document_profile":
            selected.append(chunk)
        elif float(chunk.get("relevance_score", 0.0)) >= relevance_floor:
            selected.append(chunk)
        if len(selected) >= limit:
            break

    return selected or usable_chunks[:limit]


def _citations_from_chunks(chunks: list[dict], query: str) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        excerpt_limit = 900 if str(chunk.get("domain_module", "")) == "document_profile" else 520
        excerpt = _compact_evidence_text(chunk.get("content", ""), query=query, max_chars=excerpt_limit)
        if not excerpt:
            continue
        citations.append(
            Citation(
                text=excerpt,
                source=str(chunk.get("source", "N/A")),
                domain_module=str(chunk.get("domain_module", "general")),
            )
        )
    return citations


def _format_evidence_answer(
    *,
    query: str,
    query_type: str,
    citations: list[Citation],
    gaps: list[str],
    response_language: str,
) -> str:
    """Build a readable fallback answer without adding facts beyond citations."""

    if not citations:
        return (
            "Non ho evidenze sufficienti nella documentazione recuperata per rispondere con affidabilita'."
            if response_language == "it"
            else "I do not have enough recovered documentation evidence to answer reliably."
        )

    facts = _dedupe_fact_sentences([citation.text for citation in citations], query=query, total_limit=MAX_FALLBACK_FACTS)
    if not facts:
        facts = [citation.text for citation in citations if citation.text][:3]

    if response_language == "it":
        if query_type == "PROCEDURAL":
            heading = "La documentazione recuperata supporta questi passaggi:"
        elif query_type == "DIAGNOSTIC":
            heading = "La documentazione recuperata indica questi elementi utili:"
        elif query_type == "CONFIGURATION":
            heading = "La documentazione recuperata supporta questi dettagli di configurazione:"
        else:
            heading = "La documentazione recuperata indica questo:"
        missing_label = "Limite della risposta"
    else:
        if query_type == "PROCEDURAL":
            heading = "From the recovered documentation, the supported steps are:"
        elif query_type == "DIAGNOSTIC":
            heading = "From the recovered documentation, the relevant finding is:"
        elif query_type == "CONFIGURATION":
            heading = "From the recovered documentation, the supported configuration detail is:"
        else:
            heading = "From the recovered documentation:"
        missing_label = "Answer limit"

    if query_type == "PROCEDURAL":
        body = "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts[:5], start=1))
    elif len(facts) == 1:
        body = facts[0]
    else:
        body = "\n".join(f"- {fact}" for fact in facts[:5])

    answer = f"{heading}\n{body}"
    if gaps:
        answer += f"\n\n{missing_label}: " + "; ".join(gaps[:3])
    return answer


def _dedupe_fact_sentences(texts: list[str], query: str = "", total_limit: int = 5) -> list[str]:
    query_terms = set(extract_keywords(query))
    seen: set[str] = set()
    facts: list[str] = []
    for text in texts:
        clean_text = " ".join(str(text).split())
        sentences = [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+|;\s+|\n+", clean_text) if part.strip()]
        if not sentences and clean_text:
            sentences = [clean_text]
        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: (
                len(query_terms & set(extract_keywords(sentence))) if query_terms else 0,
                min(len(sentence), 240),
            ),
            reverse=True,
        )
        picked_from_text = 0
        for sentence in ranked_sentences:
            normalized = sentence.lower().strip(" .")
            if len(sentence) < 20 or normalized in seen:
                continue
            seen.add(normalized)
            facts.append(sentence.rstrip(".") + ".")
            picked_from_text += 1
            if picked_from_text >= 2 or len(facts) >= total_limit:
                break
        if len(facts) >= total_limit:
            break
    return facts


def _postprocess_reasoning_output(
    output: ReasoningOutput,
    *,
    query: str,
    ranked_chunks: list[dict],
    gaps: list[str],
    response_language: str,
) -> ReasoningOutput:
    clean_answer = " ".join(output.answer.split())
    if (
        _is_informative_answer(clean_answer)
        or len(clean_answer) >= MIN_USEFUL_ANSWER_CHARS
        or output.hallucination_risk >= 0.8
    ):
        return output

    citations = output.citations or _citations_from_chunks(_select_evidence_chunks(ranked_chunks), query=query)
    facts = _dedupe_fact_sentences([citation.text for citation in citations], query=query, total_limit=4)
    facts = [
        fact for fact in facts
        if fact.lower().strip(" .") not in clean_answer.lower()
    ]
    if not facts:
        return output

    detail_label = "Dettagli supportati" if response_language == "it" else "Supported details"
    gap_label = "Limiti" if response_language == "it" else "Limits"
    enrichment = "\n\n" + detail_label + ":\n" + "\n".join(f"- {fact}" for fact in facts[:4])
    if gaps:
        enrichment += f"\n\n{gap_label}: " + "; ".join(gaps[:3])

    return output.model_copy(
        update={
            "answer": output.answer.rstrip() + enrichment,
            "citations": citations,
            "confidence": max(output.confidence, 0.62),
            "hallucination_reason": (
                output.hallucination_reason
                or "Answer enriched with additional facts directly extracted from selected evidence."
            ),
        }
    )


def _is_informative_answer(answer: str) -> bool:
    """Avoid bolting extra bullets onto concise answers that are already complete."""

    words = re.findall(r"\w+", answer)
    if len(words) >= 32:
        return True
    has_specific_values = bool(re.search(r"\d", answer)) or len(re.findall(r"\b[A-Z][a-zA-Z]+\b", answer)) >= 3
    if has_specific_values and len(words) >= 18:
        return True
    return False


def _context_limit(query_type_hint: str) -> int:
    if query_type_hint.upper() in {"PROCEDURAL", "DIAGNOSTIC", "CONFIGURATION"}:
        return settings.complex_context_selection_limit
    return settings.context_selection_limit


def _semantic_context_sort_key(query_type_hint: str, chunk: dict, index: int) -> tuple[int, float, int]:
    order = RESPONSE_CATEGORY_ORDER.get(query_type_hint.upper(), RESPONSE_CATEGORY_ORDER["GENERAL"])
    category = str(chunk.get("selection_category", "general"))
    category_rank = order.index(category) if category in order else len(order)
    relevance = float(chunk.get("relevance_score", 0.0) or 0.0)
    return (category_rank, -relevance, index)


def run_reasoning_agent(
    query: str,
    ranked_chunks: list[dict],
    gaps: list[str],
    query_type_hint: str = "GENERAL",
    response_language: str = "en",
    retry_guidance: str = "",
) -> ReasoningOutput:
    """Generate a grounded answer from ranked evidence."""

    ranked_serialized = []
    serialized_candidates = []
    for chunk in ranked_chunks:
        if hasattr(chunk, "model_dump"):
            try:
                serialized_candidates.append(chunk.model_dump())
            except Exception:
                serialized_candidates.append(chunk)
        else:
            serialized_candidates.append(chunk)

    ranked_serialized = [
        chunk
        for _category_rank, _relevance, _index, chunk in sorted(
            (
                (*_semantic_context_sort_key(query_type_hint, chunk, index), chunk)
                for index, chunk in enumerate(serialized_candidates)
            )
        )
    ][: _context_limit(query_type_hint)]

    context = "\n\n".join(
        (
            f"[SOURCE {index + 1}: {chunk.get('source', 'N/A')} | "
            f"Chunk: {chunk.get('chunk_index', 0)} | "
            f"Cluster: {chunk.get('cluster_id', 'cluster_1')} | "
            f"Category: {chunk.get('selection_category', 'general')} | "
            f"Relevance: {chunk.get('relevance_score', 0):.2f}]\n"
            f"{chunk.get('content', '')}"
        )
        for index, chunk in enumerate(ranked_serialized)
    )

    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "No explicit gaps identified."
    mode_instruction = MODE_INSTRUCTIONS.get(query_type_hint, MODE_INSTRUCTIONS["GENERAL"])
    retry_guidance_text = retry_guidance.strip() or "No retry guidance."
    language_name = LANGUAGE_NAMES.get(response_language, response_language)

    prompt = f"""USER QUERY: {query}
SUGGESTED QUERY TYPE: {query_type_hint}
RESPONSE LANGUAGE: {response_language} ({language_name})
LANGUAGE RULE: Answer only in {language_name}; do not switch to a neighboring language.
MODE INSTRUCTIONS: {mode_instruction}

DOCUMENT CONTEXT:
{context}

IDENTIFIED GAPS:
{gaps_text}

RETRY GUIDANCE:
{retry_guidance_text}

Build the structured answer and return JSON only.
Quality bar:
- First sentence answers the user's actual question.
- Then add only the evidence-backed details needed to make the answer useful.
- Merge duplicate chunks; do not repeat the same fact in different words.
- Use citations for the specific facts used in the answer.
- If evidence is partial, give the supported answer and name the missing part.
If the evidence already includes numbered steps or an explicit cause/solution wording, preserve it.
If the documentation does not cover the core of the request, say so clearly and answer conservatively."""

    try:
        agent = get_reasoning_agent()
        response = agent.run(prompt)
        if isinstance(response.content, ReasoningOutput):
            return _postprocess_reasoning_output(
                response.content,
                query=query,
                ranked_chunks=ranked_serialized,
                gaps=gaps,
                response_language=response_language,
            )

        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(_extract_json(content))
        return _postprocess_reasoning_output(
            ReasoningOutput(**data),
            query=query,
            ranked_chunks=ranked_serialized,
            gaps=gaps,
            response_language=response_language,
        )

    except Exception as exc:
        logger.error("ReasoningAgent error: %s", exc)
        return _evidence_only_fallback(query, ranked_serialized, query_type_hint, response_language, exc)
        error_messages = {
            "it": "Si e' verificato un errore interno durante la generazione della risposta. "
                  "Riprova oppure fornisci altro materiale di supporto.",
            "en": "An internal error occurred while generating the answer. "
                  "Please try again or provide additional source material.",
            "fr": "Une erreur interne s'est produite lors de la génération de la réponse. "
                  "Veuillez réessayer ou fournir du matériel source supplémentaire.",
            "de": "Beim Generieren der Antwort ist ein interner Fehler aufgetreten. "
                  "Bitte versuchen Sie es erneut oder stellen Sie zusätzliches Quellenmaterial zur Verfügung.",
            "es": "Se produjo un error interno al generar la respuesta. "
                  "Inténtelo de nuevo o proporcione material de origen adicional.",
            "pt": "Ocorreu um erro interno ao gerar a resposta. "
                  "Tente novamente ou forneça material de origem adicional.",
        }
        return ReasoningOutput(
            answer=error_messages.get(response_language, error_messages["en"]),
            query_type="GENERAL",
            confidence=0.0,
            citations=[],
            hallucination_risk=1.0,
            hallucination_reason=f"Internal reasoning agent error: {str(exc)[:200]}",
            fallback_used=True,
            fallback_reason="reasoning_agent_error",
        )
