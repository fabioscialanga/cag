"""
Retrieval agent for evidence ranking and gap detection.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from agno.agent import Agent

from cag.agents.models import RankedChunk, RetrievalOutput
from cag.config import settings
from cag.llm_factory import get_agno_model
from cag.retrieval.lexical import build_weighted_query_terms as _build_weighted_query_terms
from cag.retrieval.lexical import discriminative_document_score as _discriminative_document_score
from cag.retrieval.lexical import document_terms as _document_terms
from cag.retrieval.lexical import expand_query_concepts as _expand_query_concepts

logger = logging.getLogger(__name__)
SELECTION_CONTEXT_LIMIT = 8
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "which", "with",
}
CATEGORY_ALIASES = {
    "steps": "ordered_steps",
    "step": "ordered_steps",
    "procedure": "ordered_steps",
    "ordered": "ordered_steps",
    "navigation": "navigation",
    "menu": "navigation",
    "path": "navigation",
    "prerequisite": "prerequisites",
    "prerequisites": "prerequisites",
    "requirement": "prerequisites",
    "requirements": "prerequisites",
    "permission": "permissions",
    "permissions": "permissions",
    "role": "permissions",
    "roles": "permissions",
    "field": "fields",
    "fields": "fields",
    "parameter": "fields",
    "parameters": "fields",
    "setting": "settings",
    "settings": "settings",
    "option": "options",
    "options": "options",
    "symptom": "symptoms",
    "symptoms": "symptoms",
    "error": "error_causes",
    "cause": "error_causes",
    "causes": "error_causes",
    "check": "checks",
    "checks": "checks",
    "resolution": "resolution",
    "resolve": "resolution",
    "workaround": "resolution",
    "definition": "definitions",
    "definitions": "definitions",
    "overview": "overview",
    "timeline": "timeline",
    "constraint": "constraints",
    "constraints": "constraints",
}
QUERY_TYPE_CATEGORY_PRIORITIES = {
    "PROCEDURAL": ["ordered_steps", "navigation", "prerequisites", "fields", "settings"],
    "DIAGNOSTIC": ["symptoms", "error_causes", "checks", "resolution", "constraints"],
    "CONFIGURATION": ["prerequisites", "settings", "fields", "permissions", "options"],
    "GENERAL": ["definitions", "overview", "constraints", "timeline", "options"],
}

_RETRIEVAL_INSTRUCTIONS = [
    "You are the RetrievalAgent of a CAG (Cognitive Augmented Generation) system.",
    "You receive a user question and a list of retrieved documentation chunks.",
    "Your job is to score the relevance of each chunk, identify missing information, and organize context selection.",
    "",
    "PROCESS:",
    "1. Identify the core task or question.",
    "2. Use the automatic clusters and document index as structural hints, not as ground truth.",
    "3. Generate short semantic categories for the useful evidence (for example: prerequisites, permissions, error causes, ordered steps).",
    "4. Assign each ranked chunk a cluster_id and selection_category.",
    "5. Score each chunk with relevance_score (0.0-1.0).",
    "6. Record information gaps that are required to answer safely.",
    "7. Compute the overall relevance_score using the top evidence.",
    "8. If the core feature or answer is not documented, state that clearly in gaps and lower relevance.",
    "",
    "QUERY TYPE GUIDANCE:",
    "- PROCEDURAL: prioritize ordered steps, menu paths, fields, and operational sequences.",
    "- DIAGNOSTIC: prioritize symptoms, likely causes, checks, and corrective actions.",
    "- CONFIGURATION: prioritize prerequisites, settings, fields, and options.",
    "- GENERAL: prioritize definitions, constraints, timelines, and context.",
    "",
    "SCORING RUBRIC:",
    "- 0.90-1.00: directly answers the core question, with exact entity, step, field, error, or service.",
    "- 0.70-0.89: strongly useful supporting evidence, but may need another chunk for completeness.",
    "- 0.45-0.69: related background or partial evidence; keep only if it fills a different category.",
    "- 0.00-0.44: generic, duplicate, or off-topic evidence; include only when no better evidence exists.",
    "",
    "BORDERLINE RULES:",
    "- Do not over-score a chunk just because it shares broad words like service, document, system, or workflow.",
    "- For PROCEDURAL queries, preserve sequence: prerequisites/navigation before steps, steps before notes.",
    "- For DIAGNOSTIC queries, prefer chunks that pair a symptom/error with a check or resolution.",
    "- For CONFIGURATION queries, prefer exact parameters, fields, limits, and permissions over overview text.",
    "- Mark real gaps when required parts are missing instead of inflating weak chunks.",
    "",
    "FEW-SHOT EXAMPLES:",
    "Example A: Query='How do I rotate an API token?' Chunk='API tokens are rotated from Settings > Integrations. "
    "Generate a new token, update dependent services, then revoke the old token.' "
    "=> selection_category='ordered_steps', relevance_score=0.95.",
    "Example B: Query='Why do notifications return 429?' Chunk='HTTP 429 means the rate limit was exceeded; "
    "reduce request frequency and honor Retry-After.' => selection_category='error_causes', relevance_score=0.92.",
    "Example C: Query='What services do you provide?' Chunk='Office opening hours are Monday through Friday.' "
    "=> selection_category='general', relevance_score=0.15, gap='No service evidence in this chunk.'",
    "",
    "Return ONLY valid JSON with this structure:",
    '{"chunks_ranked": [{"content": "...", "source": "...", "domain_module": "...", '
    '"chunk_index": 0, "cluster_id": "cluster_1", "selection_category": "...", "relevance_score": 0.0, '
    '"relevance_reason": "..."}], "gaps": [...], "relevance_score": 0.0, "summary": "..."}',
    "Sort chunks_ranked from most relevant to least relevant.",
]


def _selection_limit(query_type_hint: str) -> int:
    if query_type_hint.upper() in {"PROCEDURAL", "DIAGNOSTIC", "CONFIGURATION"}:
        return settings.complex_context_selection_limit
    return settings.context_selection_limit


def create_retrieval_agent() -> Agent:
    """Create the configured retrieval agent."""

    return Agent(
        name="RetrievalAgent",
        model=get_agno_model(),
        role="Scores retrieved evidence and identifies information gaps.",
        instructions=_RETRIEVAL_INSTRUCTIONS,
        structured_outputs=True,
        output_schema=RetrievalOutput,
    )


_retrieval_agent: Agent | None = None


def get_retrieval_agent() -> Agent:
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = create_retrieval_agent()
    return _retrieval_agent


def _extract_json(text: str) -> str:
    """Strip optional Markdown fences and return raw JSON text."""

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def _normalize_text(text: str) -> str:
    return "".join(character.lower() if character.isalnum() or character.isspace() else " " for character in text)


def _extract_keywords(text: str, limit: int = 10) -> list[str]:
    keywords: list[str] = []
    for token in _normalize_text(text).split():
        if len(token) <= 2 or token in STOPWORDS or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _keyword_overlap_ratio(left: str, right: str) -> float:
    left_terms = set(_extract_keywords(left, limit=16))
    right_terms = set(_extract_keywords(right, limit=16))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _build_document_index(chunks: list[dict]) -> str:
    grouped: dict[str, dict[str, set]] = defaultdict(lambda: {"modules": set(), "indexes": set()})
    for chunk in chunks:
        source = str(chunk.get("source", "N/A"))
        grouped[source]["modules"].add(str(chunk.get("domain_module", "general")))
        chunk_index = chunk.get("chunk_index")
        if isinstance(chunk_index, int):
            grouped[source]["indexes"].add(chunk_index)

    if not grouped:
        return "No document index available."

    lines: list[str] = []
    for source in sorted(grouped):
        indexes = sorted(grouped[source]["indexes"])
        modules = ", ".join(sorted(grouped[source]["modules"]))
        if indexes:
            if len(indexes) == 1:
                index_text = str(indexes[0])
            else:
                index_text = f"{indexes[0]}-{indexes[-1]}"
            lines.append(f"- {source}: chunks {index_text}; modules={modules}")
        else:
            lines.append(f"- {source}: modules={modules}")
    return "\n".join(lines)


def _cluster_chunks(chunks: list[dict]) -> dict[int, str]:
    cluster_keyword_sets: list[set[str]] = []
    cluster_sources: list[set[str]] = []
    assignments: dict[int, str] = {}

    for index, chunk in enumerate(chunks):
        keywords = set(
            _extract_keywords(
                f"{chunk.get('source', '')} {chunk.get('domain_module', '')} {chunk.get('content', '')}",
                limit=12,
            )
        )
        source = str(chunk.get("source", "N/A"))
        assigned_cluster: int | None = None

        for cluster_index, cluster_keywords in enumerate(cluster_keyword_sets):
            keyword_overlap = len(keywords & cluster_keywords)
            same_source = source in cluster_sources[cluster_index]
            if keyword_overlap >= 2 or (same_source and keyword_overlap >= 1):
                assigned_cluster = cluster_index
                cluster_keyword_sets[cluster_index].update(keywords)
                cluster_sources[cluster_index].add(source)
                break

        if assigned_cluster is None:
            cluster_keyword_sets.append(set(keywords))
            cluster_sources.append({source})
            assigned_cluster = len(cluster_keyword_sets) - 1

        assignments[index] = f"cluster_{assigned_cluster + 1}"

    return assignments


def _normalize_category(category: str) -> str:
    normalized = _normalize_text(category)
    for token in normalized.split():
        alias = CATEGORY_ALIASES.get(token)
        if alias:
            return alias
    return "_".join(normalized.split()[:3]) if normalized.strip() else "general"


def _category_priority(query_type_hint: str, category: str) -> float:
    priorities = QUERY_TYPE_CATEGORY_PRIORITIES.get(query_type_hint.upper(), QUERY_TYPE_CATEGORY_PRIORITIES["GENERAL"])
    normalized = _normalize_category(category)
    if normalized in priorities:
        return float(len(priorities) - priorities.index(normalized))
    return 0.0


def _query_overlap(query: str, chunk: RankedChunk) -> int:
    query_terms = set(_extract_keywords(query, limit=12))
    chunk_terms = set(
        _extract_keywords(
            f"{chunk.source} {chunk.domain_module} {chunk.selection_category} {chunk.content}",
            limit=16,
        )
    )
    return len(query_terms & chunk_terms)


def _reorder_for_context_selection(
    query: str,
    query_type_hint: str,
    ranked_chunks: list[RankedChunk],
) -> list[RankedChunk]:
    if len(ranked_chunks) <= 1:
        return ranked_chunks

    selected: list[RankedChunk] = []
    remaining = list(ranked_chunks)

    while remaining:
        cluster_counts = defaultdict(int)
        category_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for chunk in selected:
            cluster_counts[chunk.cluster_id] += 1
            category_counts[_normalize_category(chunk.selection_category)] += 1
            source_counts[chunk.source] += 1

        best_index = 0
        best_score = float("-inf")
        for index, chunk in enumerate(remaining):
            normalized_category = _normalize_category(chunk.selection_category)
            category_seen = category_counts[normalized_category]
            cluster_seen = cluster_counts[chunk.cluster_id]
            source_seen = source_counts[chunk.source]
            query_type = query_type_hint.upper()
            category_priority_multiplier = 2.0 if query_type == "GENERAL" else 5.0
            overlap_bonus = 2.0 * _query_overlap(query, chunk)
            top_selected_relevance = max((selected_chunk.relevance_score for selected_chunk in selected), default=chunk.relevance_score)
            near_duplicate_penalty = 0.0
            if selected:
                strongest_similarity = max(
                    _keyword_overlap_ratio(
                        f"{chunk.source} {chunk.selection_category} {chunk.content}",
                        f"{selected_chunk.source} {selected_chunk.selection_category} {selected_chunk.content}",
                    )
                    for selected_chunk in selected
                )
                if strongest_similarity >= 0.72:
                    near_duplicate_penalty -= 5.0 * strongest_similarity

            if query_type == "GENERAL":
                diversity_bonus = 2.5 if cluster_seen == 0 and len(selected) < _selection_limit(query_type) else -4.0 * cluster_seen
                category_bonus = 2.0 if category_seen == 0 else -2.0 * category_seen
                source_penalty = -0.5 * source_seen
                weaker_new_source_penalty = 0.0
                if selected and source_seen == 0 and chunk.relevance_score < top_selected_relevance:
                    weaker_new_source_penalty -= (top_selected_relevance - chunk.relevance_score) * 50.0
                if selected and source_seen == 0 and chunk.relevance_score < top_selected_relevance - 0.03:
                    diversity_bonus = min(diversity_bonus, 0.0)
                    category_bonus = min(category_bonus, 0.0)
            else:
                diversity_bonus = 6.0 if cluster_seen == 0 and len(selected) < _selection_limit(query_type) else -4.0 * cluster_seen
                category_bonus = 4.0 if category_seen == 0 else -2.0 * category_seen
                source_penalty = -1.5 * source_seen
                weaker_new_source_penalty = 0.0
            procedural_bonus = 0.0
            if query_type == "PROCEDURAL":
                if _normalize_category(chunk.selection_category) in {"ordered_steps", "navigation"}:
                    procedural_bonus += 6.0
                procedural_bonus += max(0.0, 2.0 - (float(chunk.chunk_index) * 0.15))
            elif query_type == "DIAGNOSTIC" and _normalize_category(chunk.selection_category) in {"symptoms", "error_causes", "checks", "resolution"}:
                procedural_bonus += 4.0
            elif query_type == "CONFIGURATION" and _normalize_category(chunk.selection_category) in {"prerequisites", "settings", "fields", "permissions"}:
                procedural_bonus += 4.0

            score = (
                (chunk.relevance_score * 100.0)
                + (_category_priority(query_type, chunk.selection_category) * category_priority_multiplier)
                + diversity_bonus
                + category_bonus
                + source_penalty
                + overlap_bonus
                + procedural_bonus
                + weaker_new_source_penalty
                + near_duplicate_penalty
                - (index * 0.01)
            )

            if score > best_score:
                best_score = score
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected


def _postprocess_retrieval_output(
    query: str,
    query_type_hint: str,
    raw_chunks: list[dict],
    output: RetrievalOutput,
    automatic_clusters: dict[int, str],
) -> RetrievalOutput:
    raw_by_exact_identity = {
        (
            str(chunk.get("source", "N/A")),
            int(chunk.get("chunk_index", 0)),
            str(chunk.get("content", ""))[:160],
        ): chunk
        for chunk in raw_chunks
    }
    raw_by_source_and_content = {
        (
            str(chunk.get("source", "N/A")),
            str(chunk.get("content", ""))[:160],
        ): chunk
        for chunk in raw_chunks
    }

    enriched_chunks: list[RankedChunk] = []
    for index, chunk in enumerate(output.chunks_ranked):
        exact_key = (chunk.source, int(chunk.chunk_index), chunk.content[:160])
        fallback_key = (chunk.source, chunk.content[:160])
        raw_chunk = raw_by_exact_identity.get(exact_key) or raw_by_source_and_content.get(fallback_key, {})
        normalized_category = _normalize_category(chunk.selection_category)
        enriched_chunks.append(
            chunk.model_copy(
                update={
                    "chunk_index": int(raw_chunk.get("chunk_index", chunk.chunk_index or 0)),
                    "cluster_id": raw_chunk.get("cluster_id", chunk.cluster_id or automatic_clusters.get(index, "cluster_1")),
                    "selection_category": normalized_category,
                }
            )
        )

    return output.model_copy(update={"chunks_ranked": _reorder_for_context_selection(query, query_type_hint, enriched_chunks)})


def _infer_deterministic_category(query_type_hint: str, chunk: dict) -> str:
    if str(chunk.get("domain_module", "")) == "document_profile":
        return "overview"
    query_type = query_type_hint.upper()
    content = f"{chunk.get('domain_module', '')} {chunk.get('content', '')}".lower()
    if query_type == "PROCEDURAL":
        if any(marker in content for marker in ("step", "procedure", "workflow", "menu", "path", "passo", "procedura")):
            return "ordered_steps"
        return "navigation"
    if query_type == "DIAGNOSTIC":
        if any(marker in content for marker in ("error", "cause", "fault", "errore", "causa")):
            return "error_causes"
        if any(marker in content for marker in ("check", "verify", "resolution", "resolve", "verifica", "risol")):
            return "resolution"
        return "checks"
    if query_type == "CONFIGURATION":
        if any(marker in content for marker in ("required", "prerequisite", "must", "requisit")):
            return "prerequisites"
        if any(marker in content for marker in ("field", "parameter", "setting", "campo", "parametr")):
            return "fields"
        return "settings"
    if any(marker in content for marker in ("overview", "mission", "azienda", "company", "servizi", "services")):
        return "overview"
    return "general"


def _score_chunks(query: str, chunks: list[dict]) -> list[tuple[float, int, dict]]:
    expanded_query = _expand_query_concepts(query)
    weighted_terms = _build_weighted_query_terms(query, [expanded_query])
    doc_frequencies: dict[str, int] = {}
    for chunk in chunks:
        pseudo_doc = type("PseudoDoc", (), {"page_content": chunk.get("content", ""), "metadata": chunk})()
        for term in set(_document_terms(pseudo_doc)):
            doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

    corpus_size = max(1, len(chunks))
    scored: list[tuple[float, int, dict]] = []
    for index, chunk in enumerate(chunks):
        pseudo_doc = type("PseudoDoc", (), {"page_content": chunk.get("content", ""), "metadata": chunk})()
        lexical_score = _discriminative_document_score(pseudo_doc, weighted_terms, doc_frequencies, corpus_size)
        if str(chunk.get("domain_module", "")) == "document_profile":
            lexical_score += float(chunk.get("document_score", 0.0) or 0.0) * 0.35
        content = str(chunk.get("content", ""))
        if len(content.strip()) < 35:
            lexical_score *= 0.25
        scored.append((lexical_score, index, chunk))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored


def _rank_scored_chunks(
    query: str,
    query_type_hint: str,
    scored: list[tuple[float, int, dict]],
    automatic_clusters: dict[int, str],
) -> list[RankedChunk]:
    max_score = max((score for score, _index, _chunk in scored), default=0.0)
    if max_score <= 0.0:
        return []

    selected: list[RankedChunk] = []
    score_floor = max(0.12, max_score * 0.28)
    for score, index, chunk in scored:
        if score < score_floor:
            continue
        relevance = max(0.25, min(0.92, score / max_score))
        selected.append(
            RankedChunk(
                content=(
                    chunk.get("content", "")
                    if len(chunk.get("content", "")) <= 2500
                    else chunk.get("content", "")[:2500] + "...[TRUNCATED]"
                ),
                source=chunk.get("source", "N/A"),
                domain_module=chunk.get("domain_module", "general"),
                chunk_index=int(chunk.get("chunk_index", 0)),
                cluster_id=automatic_clusters.get(index, "cluster_1"),
                selection_category=_infer_deterministic_category(query_type_hint, chunk),
                relevance_score=relevance,
                relevance_reason=f"Deterministic lexical fallback score {score:.2f}.",
            )
        )
        if len(selected) >= _selection_limit(query_type_hint):
            break
    return _reorder_for_context_selection(query, query_type_hint, selected)


def _fallback_rank_chunks(query: str, chunks: list[dict], automatic_clusters: dict[int, str], query_type_hint: str = "GENERAL") -> list[RankedChunk]:
    return _rank_scored_chunks(query, query_type_hint, _score_chunks(query, chunks), automatic_clusters)


def _fast_context_output(
    query: str,
    chunks: list[dict],
    query_type_hint: str,
    automatic_clusters: dict[int, str],
) -> RetrievalOutput | None:
    if not settings.fast_context_selection:
        return None

    scored = _score_chunks(query, chunks)
    max_score = max((score for score, _index, _chunk in scored), default=0.0)
    if max_score < settings.fast_context_min_score:
        return None

    ranked_chunks = _rank_scored_chunks(query, query_type_hint, scored, automatic_clusters)
    if not ranked_chunks:
        return None

    return RetrievalOutput(
        chunks_ranked=ranked_chunks,
        gaps=[],
        relevance_score=max(chunk.relevance_score for chunk in ranked_chunks),
        summary="Fast deterministic context selection used high-confidence lexical evidence.",
        fallback_used=False,
        fallback_reason=None,
    )


def run_retrieval_agent(
    query: str,
    chunks: list[dict],
    query_type_hint: str = "GENERAL",
    strategy_hint: str = "semantic",
) -> RetrievalOutput:
    """Rank chunks and estimate evidence coverage for a query."""

    if not chunks:
        return RetrievalOutput(
            chunks_ranked=[],
            gaps=["No documentation chunks were retrieved for this query."],
            relevance_score=0.0,
            summary="No retrieved chunks available for evidence ranking.",
        )

    automatic_clusters = _cluster_chunks(chunks)
    fast_output = _fast_context_output(query, chunks, query_type_hint, automatic_clusters)
    if fast_output is not None:
        logger.info(
            "RetrievalAgent fast path: selected %s chunks for %s query.",
            len(fast_output.chunks_ranked),
            query_type_hint,
        )
        return fast_output

    agent = get_retrieval_agent()
    document_index = _build_document_index(chunks)
    chunks_text = "\n\n".join(
        (
            f"[CHUNK {index + 1} | AUTO_CLUSTER {automatic_clusters.get(index, 'cluster_1')}]\n"
            f"Source: {chunk.get('source', 'N/A')}\n"
            f"Domain module: {chunk.get('domain_module', 'general')}\n"
            f"Chunk index: {chunk.get('chunk_index', 'N/A')}\n"
            f"{chunk.get('content', '')}"
        )
        for index, chunk in enumerate(chunks)
    )

    prompt = f"""USER QUERY: {query}
QUERY TYPE: {query_type_hint}
RETRIEVAL STRATEGY: {strategy_hint}

DOCUMENT INDEX:
{document_index}

DOCUMENT CHUNKS:
{chunks_text}

RUNTIME SELECTION RULES:
- Use the scoring rubric from the system instructions.
- Select enough category diversity for the query type, not just the first similar chunks.
- For PROCEDURAL: order categories as prerequisites -> navigation -> ordered_steps -> fields/settings -> constraints.
- For DIAGNOSTIC: order categories as symptoms -> error_causes -> checks -> resolution -> constraints.
- For CONFIGURATION: order categories as prerequisites -> permissions -> settings -> fields -> options.
- For GENERAL: order categories as definitions/overview -> constraints -> timeline/options.
- If a chunk is only loosely related, score it below 0.45 and explain the gap.

Analyze the chunks and return the evaluation JSON."""

    try:
        response = agent.run(prompt)
        if isinstance(response.content, RetrievalOutput):
            return _postprocess_retrieval_output(query, query_type_hint, chunks, response.content, automatic_clusters)

        content = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(_extract_json(content))
        return _postprocess_retrieval_output(query, query_type_hint, chunks, RetrievalOutput(**data), automatic_clusters)

    except Exception as exc:
        logger.error("RetrievalAgent error: %s", exc)
        fallback_chunks = _fallback_rank_chunks(query, chunks, automatic_clusters, query_type_hint)
        fallback_score = max((chunk.relevance_score for chunk in fallback_chunks), default=0.0)
        return RetrievalOutput(
            chunks_ranked=fallback_chunks,
            gaps=[] if fallback_chunks else ["No sufficiently relevant documentation chunks were found."],
            relevance_score=fallback_score,
            summary="Deterministic lexical fallback retrieval output after agent failure.",
            fallback_used=True,
            fallback_reason="retrieval_agent_error",
        )
